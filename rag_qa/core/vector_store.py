# -*- coding:utf-8 -*-
"""
Milvus 向量库封装。

检索：BGE-M3 稠密+稀疏混合检索 → BGE-Reranker 重排 → 父块聚合。
"""
import os
import sys
import hashlib

_current = os.path.dirname(os.path.abspath(__file__))
_rag_qa_path = os.path.dirname(_current)
_project_root = os.path.dirname(_rag_qa_path)
for _p in (_rag_qa_path, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder
from base import Config, logger

conf = Config()


class VectorStore:
    """Milvus + BGE-M3 + BGE-Reranker 向量库。"""

    def __init__(
        self,
        collection_name=conf.MILVUS_COLLECTION_NAME,
        host=conf.MILVUS_HOST,
        port=conf.MILVUS_PORT,
        database=conf.MILVUS_DATABASE_NAME,
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"VectorStore 使用设备: {self.device}")

        # BGE-Reranker
        reranker_path = os.path.join(conf.MODELS_DIR, "bge-reranker-large")
        self.reranker = CrossEncoder(reranker_path, device=self.device)

        # BGE-M3 嵌入函数
        m3_path = os.path.join(conf.MODELS_DIR, "bge-m3")
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=m3_path,
            use_fp16=(self.device == "cuda"),
            device=self.device,
        )
        # 密集向量的维度
        self.dense_dim = self.embedding_function.dim["dense"]

        # Milvus 客户端
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        self._create_or_load_collection()

    def _create_or_load_collection(self):
        """创建或加载集合。schema 含章节溯源字段。"""
        if not self.client.has_collection(self.collection_name):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_filed=True)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
            schema.add_field("text", DataType.VARCHAR, max_length=65535)
            schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self.dense_dim)
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            # 父子分块
            schema.add_field("parent_id", DataType.VARCHAR, max_length=128)
            schema.add_field("parent_content", DataType.VARCHAR, max_length=65535)
            # 章节溯源（论文卖点）
            schema.add_field("chapter", DataType.VARCHAR, max_length=64)
            schema.add_field("chapter_title", DataType.VARCHAR, max_length=256)
            schema.add_field("clause", DataType.VARCHAR, max_length=64)
            schema.add_field("page", DataType.VARCHAR, max_length=32)
            schema.add_field("chunk_type", DataType.VARCHAR, max_length=32)
            # 通用元数据
            schema.add_field("source", DataType.VARCHAR, max_length=64)
            schema.add_field("timestamp", DataType.VARCHAR, max_length=64)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector", index_name="dense_index",
                index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128},
            )
            index_params.add_index(
                field_name="sparse_vector", index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            self.logger.info(f"创建集合: {self.collection_name}")
        else:
            self.logger.info(f"加载集合: {self.collection_name}")
        self.client.load_collection(collection_name=self.collection_name)

    def add_document(self, documents):
        """向集合插入子块文档。"""
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_function(texts)

        data = []
        for i, doc in enumerate(documents):
            text_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            # 稀疏向量转 dict
            sparse_vector = {}
            row = embeddings["sparse"].getrow(i)
            for idx, val in zip(row.indices, row.data):
                sparse_vector[int(idx)] = float(val)

            data.append(
                {
                    "id": text_hash,
                    "text": doc.page_content,
                    "dense_vector": embeddings["dense"][i].tolist(),
                    "sparse_vector": sparse_vector,
                    "parent_id": doc.metadata.get("parent_id", ""),
                    "parent_content": doc.metadata.get("parent_content", ""),
                    "chapter": str(doc.metadata.get("chapter", "")),
                    "chapter_title": doc.metadata.get("chapter_title", ""),
                    "clause": str(doc.metadata.get("clause", "")),
                    "page": str(doc.metadata.get("page", "")),
                    "chunk_type": doc.metadata.get("chunk_type", "text"),
                    "source": doc.metadata.get("source", "student_handbook"),
                    "timestamp": doc.metadata.get("timestamp", ""),
                }
            )
        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)
            self.logger.info(f"插入或更新了 {len(data)} 个子块")

    def hybird_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        """
        混合检索 + 重排。
        1. BGE-M3 稠密+稀疏混合检索 Top-K 子块
        2. 子块聚合为唯一父块（整章）
        3. BGE-Reranker 对父块重排
        返回 List[Document]，metadata 带章节溯源信息。
        """
        # 对输入的query 进行向量化
        query_embeddings = self.embedding_function([query])
        # 密集向量
        dense_query_vector = query_embeddings["dense"][0].tolist()
        # 稀疏向量
        row = query_embeddings["sparse"].getrow(0)
        sparse_query_vector = {}
        # 稀疏向量转 dict
        for idx, val in zip(row.indices, row.data):
            sparse_query_vector[int(idx)] = float(val)

        # 过滤条件
        filter_expr = f"source == '{source_filter.lower()}'" if source_filter else ""
        # 混合检索请求
        dense_request = AnnSearchRequest(
            data=[dense_query_vector], anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k, expr=filter_expr,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector], anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k, expr=filter_expr,
        )

        # 混合检索权重 (dense, sparse)
        ranker = WeightedRanker(1.0, 0.7)
        # 混合检索
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=[
                "text", "parent_id", "parent_content",
                "chapter", "chapter_title", "clause", "page", "chunk_type",
                "source", "timestamp",
            ],
        )[0]

        # 子块转 Document
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        # 聚合唯一父块
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        if len(parent_docs) < 2:
            return parent_docs[: conf.CANDIDATE_M]
        # Reranker 重排
        pairs = [[query, doc.page_content] for doc in parent_docs]
        scores = self.reranker.predict(pairs)
        ranked = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        return ranked[: conf.CANDIDATE_M]

    def _get_unique_parent_docs(self, sub_chunks):
        """子块按 parent_content 去重，聚合为父块（整章）。"""
        seen = set()
        unique_docs = []
        for chunk in sub_chunks:
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            if parent_content and parent_content not in seen:
                # 父块内容用整章，但保留首个子块的溯源信息
                unique_docs.append(
                    Document(page_content=parent_content, metadata=chunk.metadata)
                )
                seen.add(parent_content)
        return unique_docs

    def _doc_from_hit(self, hit):
        """从 Milvus 查询结果构造 Document。"""
        return Document(
            page_content=hit.get("text", ""),
            metadata={
                "parent_id": hit.get("parent_id", ""),
                "parent_content": hit.get("parent_content", ""),
                "chapter": hit.get("chapter", ""),
                "chapter_title": hit.get("chapter_title", ""),
                "clause": hit.get("clause", ""),
                "page": hit.get("page", ""),
                "chunk_type": hit.get("chunk_type", "text"),
                "source": hit.get("source", ""),
                "timestamp": hit.get("timestamp", ""),
            },
        )


if __name__ == "__main__":
    vs = VectorStore()
    results = vs.hybird_search_with_rerank("晚归怎么处分")
    for r in results:
        print(f"[{r.metadata.get('chapter_title')}/条{r.metadata.get('clause')}] {r.page_content[:60]}")
