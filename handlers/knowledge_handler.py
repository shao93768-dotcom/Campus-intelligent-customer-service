# -*- coding:utf-8 -*-
"""
专业知识 Handler。

封装 RAG 调用：
1. RAG 系统检索学生手册（带章节溯源）
2. 构造带来源标注的上下文
3. Ollama 流式生成答案

依赖注入：接受一个 RAGSystem 实例，避免重复初始化向量库。
"""
import os
import sys
import time

_current = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from base import logger, Config
from rag_qa.core.prompts import RAGPrompts
from llm.ollama_client import stream_ollama

conf = Config()


class KnowledgeHandler:
    """专业知识：RAG 检索 + 大模型 流式生成。"""

    def __init__(self, rag_system):
        """
        :param rag_system: RAGSystem 实例（已注入 llm 与 vector_store）
        """
        self.rag = rag_system
        self.rag_prompt = RAGPrompts.rag_prompt()

    def handle(self, query, history=None):
        """
        流式生成答案。
        :yield: 文本片段
        """
        start = time.time()
        # 1. 检索（带章节溯源）
        docs, context, strategy = self.rag.retrieve_for_stream(query)
        if not context:
            logger.warning(f"未检索到相关文档: {query}")
            yield (
                f"学生手册中未找到相关条款，建议咨询辅导员，"
                f"或拨打学生处总值班电话：{conf.CUSTOMER_SERVICE_PHONE}。"
            )
            return

        logger.info(f"检索完成，策略={strategy}，源块数={len(docs)}")

        # 2. 构造 prompt
        prompt = self.rag_prompt.format(
            context=context, question=query, phone=conf.CUSTOMER_SERVICE_PHONE
        )

        # 3. 流式生成
        try:
            for token in stream_ollama(prompt):
                yield token
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield f"抱歉，生成答案时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"

        elapsed = time.time() - start
        logger.info(f"专业知识回答完成，耗时 {elapsed:.2f}s")
        # 附注来源（可选，便于追溯）
        sources = []
        for d in docs:
            ch = d.metadata.get("chapter", "")
            ct = d.metadata.get("chapter_title", "")
            cl = d.metadata.get("clause", "")
            if ch and cl:
                sources.append(f"第{ch}章《{ct}》第{cl}条")
            elif ch:
                sources.append(f"第{ch}章《{ct}》")
        if sources:
            yield f"\n\n参考资料：{ '；'.join(sources[:3]) }"


if __name__ == "__main__":
    # 自测需先初始化 RAGSystem
    from rag_qa.core.rag_system import RAGSystem
    from rag_qa.core.vector_store import VectorStore
    from llm.ollama_client import call_ollama

    vs = VectorStore()
    rag = RAGSystem(call_ollama, vs)
    h = KnowledgeHandler(rag)
    for q in ["晚归怎么处分", "奖学金评定标准"]:
        print(f"\nquery: {q}")
        for piece in h.handle(q):
            print(piece, end="", flush=True)
        print()
