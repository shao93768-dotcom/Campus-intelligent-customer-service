# -*- coding:utf-8 -*-
"""
闲聊 Handler。

调用 Ollama 本地模型直接生成，配置"校小通"校园助手人设。
含心理安全网：检测极端情绪词时主动引导心理咨询/转人工。
"""
import os
import sys

_current = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from base import logger, Config
from llm.ollama_client import build_ollama_llm
from rag_qa.core.prompts import RAGPrompts

conf = Config()

# 极端情绪关键词，触发心理安全网
CRISIS_KEYWORDS = [
    "自杀", "自残", "不想活", "想死", "轻生", "了断", "活不下去",
    "跳楼", "割腕", "吃安眠药", "结束生命", "没意思了",
    "人间不值得", "撑不下去了", "熬不过去",
    "看不到希望", "没有未来", "一切都没意义", "解脱", "彻底解脱",
    "消失", "永远消失", "睡过去不再醒来", "离开这个世界",
    "告别", "写遗书", "交代后事", "没意思",
    "上吊", "烧炭自杀", "喝农药", "敌敌畏",
    "百草枯", "安眠药", "割脉", "跳河", "跳江", "卧轨",
    "撞车", "立遗嘱", "签遗体捐献",
]



class ChitchatHandler:
    """闲聊：Ollama 直接对话 + 人设 + 心理安全网。"""

    def __init__(self):
        self.llm = build_ollama_llm(stream=True)
        self.chitchat_prompt = RAGPrompts.chitchat_prompt()

    def handle(self, query, history=None):
        """
        流式生成回复。
        :param query: 学生问题
        :param history: 多轮历史 [{question, answer}, ...]
        :yield: 文本片段
        """
        # 心理安全网：检测极端情绪，主动引导
        if self._detect_crisis(query):
            logger.warning(f"检测到极端情绪表达: {query}")
            yield (
                "我听到你现在的情绪好像很沉重，你愿意说的我都愿意听。"
                "如果你感到难以承受，建议你尽快联系学校心理咨询中心，"
                f"电话 10000000001（工作日 14:00-17:00 可预约），"
                f"或拨打学生处总值班电话 {conf.CUSTOMER_SERVICE_PHONE}。"
                "你不是一个人，请给身边的人一个机会帮你。"
            )
            return

        # 构造 prompt
        prompt = self.chitchat_prompt.format(question=query)
        logger.info(f"闲聊调用 模型，query: {query}")
        try:
            for chunk in self.llm.stream(prompt):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"闲聊 模型 调用失败: {e}")
            yield f"抱歉，我这边有点小状况。可直接联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"

    def _detect_crisis(self, query):
        """检测极端情绪关键词。"""
        return any(kw in query for kw in CRISIS_KEYWORDS)


if __name__ == "__main__":
    h = ChitchatHandler()
    for q in ["你好呀", "帮我写个请假条", "我今天好累"]:
        print(f"\nquery: {q}")
        for piece in h.handle(q):
            print(piece, end="", flush=True)
        print()

