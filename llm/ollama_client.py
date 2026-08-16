# -*- coding:utf-8 -*-
"""
Ollama 本地大模型封装。

提供三类调用方式，供不同场景使用：
- call_ollama(prompt)        -> str     非流式，用于策略选择器/子查询生成等中间步骤
- stream_ollama(prompt)      -> generator 流式，用于最终答案输出
- build_ollama_llm(stream)   -> ChatOllama 原始对象，用于需要消息列表的场景

"""
import sys
import os

# 把项目根目录加入 sys.path，保证能 import base
_current = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from base import logger, Config

conf = Config()


def build_ollama_llm(stream=False):
    """
    构建一个 ChatOllama 实例。
    :param stream: 是否启用流式输出
    :return: 大模型对象
    """

    # 本地模型
    # llm = ChatOllama(
    #     base_url=conf.OLLAMA_BASE_URL,
    #     model=conf.OLLAMA_MODEL,
    #     temperature=conf.OLLAMA_TEMPERATURE,
    #     num_ctx=conf.OLLAMA_NUM_CTX,
    #     stream=stream,
    # )

    # 调用外部模型
    llm = ChatOpenAI(
        api_key=conf.API_KEY,
        base_url=conf.BASE_URL,
        model=conf.MODEL_NAME,
        temperature=conf.temperature
    )

    return llm


def call_ollama(prompt: str) -> str:
    """
    非流式调用 Ollama，返回完整字符串。
    """
    try:
        llm = build_ollama_llm(stream=False)
        resp = llm.invoke(prompt)
        # print(f'resp__>{resp}')
        # ChatOllama.invoke 返回 AIMessage，取 content
        content = resp.content if hasattr(resp, "content") else str(resp)
        logger.info(f"模型 非流式调用完成，返回长度: {len(content)}")
        return content
    except Exception as e:
        logger.error(f"模型 调用失败: {e}")
        return f"抱歉，响应失败，请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"


def stream_ollama(prompt: str):
    """
    流式调用 模型，逐 token yield 返回。
    用于 SSE 接口的最终答案输出。
    """
    try:
        llm = build_ollama_llm(stream=True)
        for chunk in llm.stream(prompt):
            # print(f'chunk-->{chunk}')
            if chunk.content:
                yield chunk.content
    except Exception as e:
        logger.error(f"模型 流式调用失败: {e}")
        yield f"抱歉，模型调用失败，请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"


if __name__ == "__main__":
    # 自测： python -m llm.ollama_client
    logger.info(f"测试 Ollama，模型: {conf.OLLAMA_MODEL}，地址: {conf.OLLAMA_BASE_URL}")
    ans = call_ollama("你好，请用一句话介绍你自己。")
    print(f"非流式返回:\n{ans}")
    print("-" * 60)
    print("流式返回:")
    for token in stream_ollama("你好，请用一句话介绍你自己。"):
        print(token, end="", flush=True)
    print()
