# -*- coding:utf-8 -*-
"""
base 包：提供全局配置(Config)与日志(logger)。
沿用 Edu_qa_project 的引用方式： from base import logger, Config
"""
from .config import Config
from .logger import logger

__all__ = ["Config", "logger"]
