# -*- coding:utf-8 -*-
"""llm 包：本地 Ollama 大模型封装。"""
from .ollama_client import (
    build_ollama_llm,
    call_ollama,
    stream_ollama,
)

__all__ = ["build_ollama_llm", "call_ollama", "stream_ollama"]
