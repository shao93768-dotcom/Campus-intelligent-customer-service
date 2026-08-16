# -*- coding:utf-8 -*-
"""
全局日志器。
控制台输出 + 文件滚动输出，供全项目复用。
"""
import os
import logging
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)
os.makedirs(_LOG_DIR, exist_ok=True)

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(filename)s:%(lineno)d | %(message)s"

logger = logging.getLogger("campus_qa")
logger.setLevel(logging.INFO)
# 避免重复添加 handler（多次 import 时）
if not logger.handlers:
    _fmt = logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    # 控制台
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    logger.addHandler(_sh)
    # 文件（10MB 滚动，保留 3 份）
    _fh = RotatingFileHandler(
        os.path.join(_LOG_DIR, "campus_qa.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)

logger.propagate = False
