# -*- coding:utf-8 -*-
"""
检索策略选择器。

四种检索策略：
- 直接检索          问题清晰、关键词明确
- 回溯问题检索      问题表述复杂，先简化
- 子查询检索        包含多个并列子问题
- 假设问题检索(HyDE) 问题抽象，先生成假设答案

"""
import os
import sys

_current = os.path.dirname(os.path.abspath(__file__))
_rag_qa_path = os.path.dirname(_current)
_project_root = os.path.dirname(_rag_qa_path)
for _p in (_rag_qa_path, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base import logger, Config
from core.prompts import RAGPrompts
from llm.ollama_client import call_ollama

conf = Config()

VALID_STRATEGIES = ["直接检索", "回溯问题检索", "子查询检索", "假设问题检索"]


class StrategySelector:
    """检索策略选择器。"""

    def __init__(self, use_llm=True):
        """
        :param use_llm: 是否用 LLM 选策略。False 时走规则匹配（更快、不耗 LLM）。
        """
        self.use_llm = use_llm
        self.call_llm = call_ollama  # 统一 LLM 调用入口，注入用

    def select_strategy(self, query):
        """选择检索策略。"""
        if not self.use_llm:
            return self._rule_based_select(query)
        try:
            prompt = RAGPrompts.strategy_select_prompt().format(query=query)
            resp = self.call_llm(prompt).strip()
            # 容错：匹配返回文本中的策略名
            for s in VALID_STRATEGIES:
                if s in resp:
                    logger.info(f"LLM 选择策略: {s} (query: {query})")
                    return s
            logger.warning(f"LLM 返回无法识别的策略: {resp}，降级为规则匹配")
            return self._rule_based_select(query)
        except Exception as e:
            logger.error(f"LLM 策略选择失败: {e}，降级为规则匹配")
            return self._rule_based_select(query)

    def _rule_based_select(self, query):
        """规则兜底的策略选择。"""
        # 含"和/与/以及/分别"等多问题标志 -> 子查询
        if any(kw in query for kw in ["和", "与", "以及", "分别", "还有", "另外"]):
            return "子查询检索"
        # 问题很长、带"关于...那些事"等模糊表述 -> 回溯
        if len(query) > 25 or any(kw in query for kw in ["那些事", "相关", "关于", "方面"]):
            return "回溯问题检索"
        # 默认直接检索
        return "直接检索"


if __name__ == "__main__":
    selector = StrategySelector(use_llm=False)
    for q in [
        "晚归怎么处分",
        "奖学金评定条件和晚归处分分别是什么",
        "关于学籍管理那些事儿",
        "学生管理的相关规定",
    ]:
        print(f"{q} -> {selector.select_strategy(q)}")
