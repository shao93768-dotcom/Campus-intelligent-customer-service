# -*- coding:utf-8 -*-
"""
RAG 系统：专业知识意图的核心。

职责：
- 四策略检索（直接/回溯/子查询/HyDE）
- 父块聚合 + Reranker 重排
- 构造带章节溯源的上下文
- 调用 Ollama 生成答案

注入式设计：llm 参数为 (prompt:str)->str 的函数（call_ollama），
与参考项目 RAGSystem(llm, vector_store) 一致，可无缝替换。

本模块专注专业知识 RAG，三意图路由由 main.py 的 CampusQASystem 统一调度。
"""
import os
import sys
import time

_current = os.path.dirname(os.path.abspath(__file__))
_rag_qa_path = os.path.dirname(_current)
_project_root = os.path.dirname(_rag_qa_path)
for _p in (_rag_qa_path, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.prompts import RAGPrompts
from core.query_classifier import QueryClassifier
from core.strategy_selector import StrategySelector
from core.vector_store import VectorStore
from base import logger, Config

conf = Config()


class RAGSystem:
    """专业知识 RAG 系统。"""

    def __init__(self, llm, vector_store, classifier=None, strategy_selector=None):
        """
        :param llm: (prompt:str)->str 函数，注入式调用大模型
        :param vector_store: VectorStore 实例
        :param classifier: 查询分类器（可选，三意图路由由 main 调度，这里仅备用）
        :param strategy_selector: 策略选择器
        """
        self.vector_store = vector_store
        self.llm = llm
        self.rag_prompt = RAGPrompts.rag_prompt()
        self.query_classifier = classifier
        self.strategy_selector = strategy_selector or StrategySelector(use_llm=False)

    # ---------- 四策略检索 ----------
    def _retrieve_with_backtracking(self, query):
        logger.info(f"回溯问题检索: {query}")
        prompt = RAGPrompts.backtracking_prompt().format(query=query)
        try:
            simplified = self.llm(prompt).strip()
            logger.info(f"简化后问题: {simplified}")
            return self.vector_store.hybird_search_with_rerank(simplified)
        except Exception as e:
            logger.error(f"回溯检索失败: {e}")
            return []

    def _retrieve_with_subqueries(self, query):
        logger.info(f"子查询检索: {query}")
        prompt = RAGPrompts.subquery_prompt().format(query=query)
        try:
            resp = self.llm(prompt).strip()
            sub_queries = [s.strip() for s in resp.split("\n") if s.strip()]
            logger.info(f"生成子查询: {sub_queries}")
            if not sub_queries:
                return []
            all_docs = []
            for sub in sub_queries:
                docs = self.vector_store.hybird_search_with_rerank(sub, k=conf.RETRIEVAL_K // 2)
                all_docs.extend(docs)
            # 去重
            unique = {doc.page_content: doc for doc in all_docs}
            logger.info(f"子查询共 {len(all_docs)}，去重后 {len(unique)}")
            return list(unique.values())
        except Exception as e:
            logger.error(f"子查询检索失败: {e}")
            return []

    def _retrieve_with_hyde(self, query):
        logger.info(f"HyDE 假设问题检索: {query}")
        prompt = RAGPrompts.hyde_prompt().format(query=query)
        try:
            hyde_answer = self.llm(prompt).strip()
            logger.info(f"假设答案: {hyde_answer}")
            return self.vector_store.hybird_search_with_rerank(hyde_answer)
        except Exception as e:
            logger.error(f"HyDE 检索失败: {e}")
            return []

    def retrieve_and_merge(self, query, strategy=None):
        """
        检索相关文档。
        :return: List[Document]，已去重、重排，取 Top-M
        """
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)
        logger.info(f"检索策略: {strategy}")

        if strategy == "回溯问题检索":
            docs = self._retrieve_with_backtracking(query)
        elif strategy == "子查询检索":
            docs = self._retrieve_with_subqueries(query)
        elif strategy == "假设问题检索":
            docs = self._retrieve_with_hyde(query)
        else:
            logger.info(f"直接检索: {query}")
            docs = self.vector_store.hybird_search_with_rerank(query)

        logger.info(f"检索到文档数: {len(docs)}")
        return docs[: conf.CANDIDATE_M]

    # ---------- 上下文构造 ----------
    def _build_context(self, docs):
        """
        把检索结果构造为带章节溯源的上下文。
        每个块前加"【第X章 第X条 | 页X】"标注，方便 LLM 引用。
        """
        if not docs:
            return ""
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            chapter = meta.get("chapter", "")
            chapter_title = meta.get("chapter_title", "")
            clause = meta.get("clause", "")
            page = meta.get("page", "")
            chunk_type = meta.get("chunk_type", "text")

            # 溯源标注
            source_tag = ""
            if chapter and clause:
                source_tag = f"【第{chapter}章《{chapter_title}》 第{clause}条 | 页{page}】"
            elif chapter:
                source_tag = f"【第{chapter}章《{chapter_title}》 | 页{page}】"
            else:
                source_tag = f"【页{page}】"
            if chunk_type == "table":
                source_tag += "[表格]"

            parts.append(f"[资料{i}] {source_tag}\n{doc.page_content}")
        return "\n\n".join(parts)

    # ---------- 生成 ----------
    def generate_answer(self, query, history=None):
        """
        非流式生成答案。
        :param history: 多轮历史列表 [{question, answer}, ...]
        :return: str 答案
        """
        start = time.time()
        logger.info(f"RAG 处理专业知识问题: {query}")

        # 检索
        strategy = self.strategy_selector.select_strategy(query)
        docs = self.retrieve_and_merge(query, strategy)
        context = self._build_context(docs)
        if not context:
            logger.warning("未检索到相关文档")
            return (
                "学生手册中未找到相关条款，建议咨询辅导员或拨打"
                f"{conf.CUSTOMER_SERVICE_PHONE}。"
            )

        # 构造 prompt
        prompt = self.rag_prompt.format(
            context=context, question=query, phone=conf.CUSTOMER_SERVICE_PHONE
        )
        try:
            answer = self.llm(prompt)
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            answer = f"抱歉，生成答案时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"

        elapsed = time.time() - start
        logger.info(f"RAG 完成，耗时 {elapsed:.2f}s，源块数 {len(docs)}")
        return answer

    def retrieve_for_stream(self, query):
        """
        仅供流式生成使用：先做检索，返回 (docs, context)。
        实际流式拼装与生成放在 main.py 统一调度。
        """
        strategy = self.strategy_selector.select_strategy(query)
        docs = self.retrieve_and_merge(query, strategy)
        context = self._build_context(docs)
        return docs, context, strategy


if __name__ == "__main__":
    # 自测
    from llm.ollama_client import call_ollama

    vs = VectorStore()
    rag = RAGSystem(call_ollama, vs)
    ans = rag.generate_answer("晚归怎么处分")
    print(f"答案:\n{ans}")
