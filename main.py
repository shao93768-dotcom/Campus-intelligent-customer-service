# -*- coding:utf-8 -*-
"""
校园学生手册问答系统 —— 集成核心（三意图路由中枢）。

职责：
1. 初始化 BERT 分类器、向量库、RAG 系统、三个 Handler
2. 接收 query → BERT 三分类 → 分发到对应 Handler
3. 提供流式 / 非流式两种调用方式

三意图路由：
- 闲聊     → ChitchatHandler (Ollama 直接对话)
- 转人工   → HandoffHandler   (联系方式库匹配)
- 专业知识 → KnowledgeHandler (RAG 检索 + Ollama 生成)

当 BERT 模型未训练时，启用规则兜底分类器，保证系统可用。
"""
import os
import sys
import time

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from base import logger, Config
from llm.ollama_client import call_ollama
from rag_qa.core.query_classifier import QueryClassifier
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem
from handlers import ChitchatHandler, HandoffHandler, KnowledgeHandler

conf = Config()


# ---------- 规则兜底分类器（BERT 未训练时启用）----------
class RuleClassifier:
    """BERT 模型未训练时的规则兜底。"""

    HANDOFF_KW = [
        "联系", "电话", "找谁", "找老师", "找辅导员", "找班主任", "投诉", "举报",
        "申诉", "反映", "找领导", "预约", "报修", "报案", "报警", "丢失", "丢了",
        "被盗", "被骗", "心理咨询", "心情", "想找人", "人工", "客服", "值班",
    ]
    KNOWLEDGE_KW = [
        "规定", "制度", "标准", "条件", "流程", "怎么办", "怎么办理", "怎么申请",
        "怎么处分", "处罚", "处分", "几点", "时间", "要求", "条件是什么",
        "怎么评", "怎么管", "怎么办手续", "规定是什么", "有哪些",
    ]

    def predict(self, query):
        if any(kw in query for kw in self.HANDOFF_KW):
            return "转人工"
        if any(kw in query for kw in self.KNOWLEDGE_KW):
            return "专业知识"
        return "闲聊"


class CampusQASystem:
    """校园问答系统中枢。"""

    def __init__(self, use_rule_fallback=True):
        """
        :param use_rule_fallback: BERT 未训练时是否启用规则兜底
        """
        logger.info("=" * 60)
        logger.info("初始化校园学生手册问答系统...")
        logger.info("=" * 60)

        # 1. 加载训练后的 BERT 分类器
        self._bert_ready = self._check_bert_ready()
        if self._bert_ready:
            self.classifier = QueryClassifier()
            logger.info("BERT 三分类器已加载")
        elif use_rule_fallback:
            self.classifier = RuleClassifier()
            logger.warning("BERT 模型未训练，启用规则兜底分类器。请运行 rag_main.py train 训练。")
        else:
            self.classifier = QueryClassifier()  # 会用未训练模型（不建议）

        # 2. 向量库
        try:
            self.vector_store = VectorStore()
            logger.info("Milvus 向量库已连接")
        except Exception as e:
            logger.error(f"Milvus 连接失败: {e}，专业知识功能将不可用")
            self.vector_store = None

        # 3. RAG 系统
        if self.vector_store is not None:
            self.rag_system = RAGSystem(call_ollama, self.vector_store)
        else:
            self.rag_system = None

        # 4. 三个 Handler
        self.chitchat_handler = ChitchatHandler()
        self.handoff_handler = HandoffHandler()
        self.knowledge_handler = (
            KnowledgeHandler(self.rag_system) if self.rag_system else None
        )

        logger.info("系统初始化完成 ✅")

    def _check_bert_ready(self):
        """检查 BERT 模型是否已训练。"""
        return os.path.exists(
            os.path.join(conf.BERT_MODEL_PATH, "config.json")
        )

    def route(self, query):
        """意图路由：返回分类标签。"""
        if isinstance(self.classifier, RuleClassifier):
            return self.classifier.predict(query)
        return self.classifier.predict_category(query)

    def stream_answer(self, query, history=None):
        """
        流式回答（SSE 用）。
        :param history: 多轮历史 [{question, answer}, ...]
        :yield: 文本片段
        """
        start = time.time()
        intent = self.route(query)
        logger.info(f"意图识别: {intent} | query: {query}")

        # 先 yield 一个意图标记（前端可据此显示路由）
        yield f"[intent:{intent}]"

        try:
            if intent == "闲聊":
                yield from self.chitchat_handler.handle(query, history)
            elif intent == "转人工":
                yield from self.handoff_handler.handle(query, history)
            else:  # 专业知识
                if self.knowledge_handler is None:
                    yield (
                        "专业知识功能不可用（向量库未连接）。"
                        f"请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
                    )
                else:
                    yield from self.knowledge_handler.handle(query, history)
        except Exception as e:
            logger.error(f"处理失败: {e}")
            yield f"抱歉，处理您的问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"

        elapsed = time.time() - start
        logger.info(f"本次回答完成，意图={intent}，耗时 {elapsed:.2f}s")

    def answer(self, query, history=None):
        """非流式回答：拼接所有片段返回完整字符串。"""
        # 跳过意图标记
        parts = []
        for chunk in self.stream_answer(query, history):
            if chunk.startswith("[intent:"):
                continue
            parts.append(chunk)
        return "".join(parts)


# 单例缓存
_system_instance = None


def get_system():
    """获取系统单例（避免重复初始化）。"""
    global _system_instance
    if _system_instance is None:
        _system_instance = CampusQASystem()
    return _system_instance


if __name__ == "__main__":
    # 命令行自测
    system = CampusQASystem()
    test_queries = [
        "你好呀",            # 闲聊
        "怎么联系辅导员",     # 转人工
        "晚归怎么处分",      # 专业知识
    ]
    for q in test_queries:
        print(f"\n{'='*60}\n学生: {q}")
        print(f"校小通: ", end="", flush=True)
        for chunk in system.stream_answer(q):
            if chunk.startswith("[intent:"):
                print(f"[{chunk}]", end=" ", flush=True)
                continue
            print(chunk, end="", flush=True)
        print()
