# -*- coding:utf-8 -*-
"""
edu_document_loaders 包：文档加载器。
- pdf_loader.PDPLoader     纯文本优先 + OCR 兜底的学生手册加载器
"""
from .pdf_loader import HandbookPDFLoader

__all__ = ["HandbookPDFLoader"]
