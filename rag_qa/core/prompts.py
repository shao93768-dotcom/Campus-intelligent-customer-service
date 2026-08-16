# -*- coding:utf-8 -*-
"""
Prompt 模板集合。

三类意图各自一套 Prompt：
- chitchat_prompt   闲聊人设（校园助手"校小通"，带心理安全网）
- rag_prompt        专业知识 RAG（强制基于学生手册、引用溯源、不编造）
- handoff_prompt    转人工兜底（无检索无 LLM 时使用）

以及三种 RAG 检索策略的辅助 prompt（回溯/子查询/HyDE）。
"""
from langchain.prompts import PromptTemplate


class RAGPrompts:
    """集中管理所有 Prompt 模板。"""

    # ---------- 闲聊 ----------
    @staticmethod
    def chitchat_prompt():
        """
        闲聊人设：高校校园智能助手"校小通"。
        含心理安全网——检测极端情绪主动引导心理咨询。
        """
        return PromptTemplate(
            template=(
                "你是高校校园智能助手“校小通”，服务对象是在校大学生。"
                "风格友好、热情、接地气，像一个懂学校事务的学长/学姐。"
                "面对学生的日常闲聊、情绪倾诉、知识问答、写作辅助，给出贴心、简洁的回复，不要啰嗦。\n"
                "对于学生的问题, 请站在老师, 辅导员, 学长学姐的角度想想应该怎么回复。\n"
                "【特别规则】如果学生表达严重心理困扰（自伤、极端情绪、轻生念头），"
                "请温和地建议联系学校心理咨询中心，并提示可转人工获取联系方式。\n\n"
                "【学生问题】: {question}\n"
                "【你的回复】:"
            ),
            input_variables=["question"],
        )

    # ---------- 专业知识 RAG ----------
    @staticmethod
    def rag_prompt():
        """
        校规权威性约束 Prompt：
        1. 只基于学生手册资料作答，不编造
        2. 回答末尾注明来源章节条款
        3. 资料无法回答时引导转人工
        """
        return PromptTemplate(
            template=(
                "你是高校学生手册智能问答助手。请严格根据以下【学生手册】内容回答学生问题。\n"
                "回答规则：\n"
                "1. 只基于提供的【学生手册资料】作答，资料中没有的信息严禁编造。\n"
                "2. 回答末尾需注明条款来源，格式如“依据《学生手册》第X章第X条”。"
                "若资料带有页码，可附注页码。\n"
                "3. 若资料无法回答该问题，明确告知：“学生手册中未找到相关条款，建议咨询辅导员”，"
                "并可提示转人工获取联系方式（电话：{phone}）。\n"
                "4. 语言简洁、条理清晰，分点作答优先。\n\n"
                "【学生手册资料】：\n{context}\n\n"
                "【学生问题】：{question}\n"
                "【回答】："
            ),
            input_variables=["context", "question", "phone"],
        )

    # ---------- 转人工兜底 ----------
    @staticmethod
    def handoff_prompt():
        """转人工无匹配联系人时的兜底回复模板。"""
        return PromptTemplate(
            template=(
                "未匹配到精确联系人。已为您转接学校学生处总值班：{phone}。"
                "工作时间可电话咨询，或描述更具体的需求（如“找辅导员”“心理咨询”）再次提问。"
            ),
            input_variables=["phone"],
        )

    # ---------- RAG 检索策略辅助 prompt ----------
    @staticmethod
    def hyde_prompt():
        """HyDE：对抽象问题生成假设答案后检索。"""
        return PromptTemplate(
            template=(
                "假设你是高校学生，想了解以下问题，请基于学生手册常识生成一个简短的假设答案：\n"
                "问题: {query}\n"
                "假设答案:"
            ),
            input_variables=["query"],
        )

    @staticmethod
    def subquery_prompt():
        """子查询：将复杂问题分解为多个简单子查询。"""
        return PromptTemplate(
            template=(
                "将以下复杂查询分解为多个简单子查询，每行一个，最多生成两个子查询"
                "（只保留子查询问题，不要其他文本）：\n"
                "示例：\n"
                "原始输入: “奖学金评定条件和晚归处分分别是什么？”\n"
                "子查询:\n"
                "    “奖学金评定条件是什么？”\n"
                "    “晚归处分规定是什么？”\n\n"
                "查询: {query}\n"
                "子查询:"
            ),
            input_variables=["query"],
        )

    @staticmethod
    def backtracking_prompt():
        """回溯检索：简化复杂问题后检索。"""
        return PromptTemplate(
            template=(
                "将以下复杂问题简化为一个更简单、更核心的问题：\n"
                "查询: {query}\n"
                "简化问题:"
            ),
            input_variables=["query"],
        )

    # ---------- 策略选择 prompt ----------
    @staticmethod
    def strategy_select_prompt():
        """让 LLM 选择检索策略。"""
        return PromptTemplate(
            template=(
                "请根据学生问题，从以下四种检索策略中选择最合适的一种，"
                "只输出策略名称（直接检索/回溯问题检索/子查询检索/假设问题检索），不要其他文字：\n"
                "- 直接检索：问题清晰、关键词明确（如“晚归处分是什么”）\n"
                "- 回溯问题检索：问题表述复杂，需简化核心（如“关于学籍管理那些事儿”）\n"
                "- 子查询检索：问题包含多个并列子问题\n"
                "- 假设问题检索：问题抽象，需先生成假设答案\n\n"
                "学生问题: {query}\n"
                "策略:"
            ),
            input_variables=["query"],
        )


if __name__ == "__main__":
    p = RAGPrompts.rag_prompt()
    print(p.format(context="第三条 学生应当遵守……", question="晚归怎么处分", phone="0714-6512345"))
