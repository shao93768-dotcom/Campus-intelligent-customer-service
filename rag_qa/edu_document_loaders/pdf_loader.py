# -*- coding:utf-8 -*-
"""
学生手册 PDF 加载器。

策略（自适应解析）：
1. 优先用 pdfplumber 提取纯文本 + 表格（转 Markdown）
2. 对提取字数过少的页（< MIN_CHARS，疑似扫描页），用 PyMuPDF 渲染成图片 + OCR 兜底
3. 每页产出一个 Document，metadata 带 page_number，供后续条款切分器溯源

输出：List[langchain.Document]
"""
import os
import sys

# 保证可 import base
_current = os.path.dirname(os.path.abspath(__file__))
# print(_current)
_project_root = os.path.dirname(os.path.dirname(_current))
# print(_project_root)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pdfplumber
from langchain.docstore.document import Document
from base import logger

# 一页文本字数低于该阈值视为疑似扫描页，触发 OCR
MIN_CHARS = 100


def _table_to_markdown(table):
    """将 pdfplumber 提取的二维表格转为 Markdown 表格字符串。"""
    if not table or len(table) == 0:
        return ""
    lines = []
    for i, row in enumerate(table):
        # row 中可能存在 None，统一转为空串
        cells = [str(c).replace("\n", " ").strip() if c is not None else "" for c in row]
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            # 第一行下方加分隔行
            lines.append("| " + " | ".join(["---"] * len(row)) + " |")
    return "\n".join(lines)


def _ocr_page_with_pymupdf(pdf_path, page_index):
    """用 PyMuPDF 把指定页渲染成图片，再调用 rapidocr 识别。失败时返回空串。"""
    try:
        import fitz  # PyMuPDF
        from rapidocr_onnxruntime import RapidOCR

        doc = fitz.open(pdf_path)
        page = doc[page_index]
        # 2x 缩放提升 OCR 精度
        matrix = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")
        doc.close()

        ocr = RapidOCR()
        result, _ = ocr(img_bytes)
        if not result:
            return ""
        # result 每项: [bbox, text, score]
        texts = [item[1] for item in result if item[1]]
        return "\n".join(texts)
    except Exception as e:
        logger.error(f"OCR 第 {page_index + 1} 页失败: {e}")
        return ""


class HandbookPDFLoader:
    """学生手册 PDF 自适应加载器。"""

    def __init__(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
        self.pdf_path = pdf_path

    def load(self):
        """加载 PDF，返回 List[Document]，每个 Document 对应一页。"""
        documents = []
        logger.info(f"开始加载学生手册 PDF: {self.pdf_path}")
        with pdfplumber.open(self.pdf_path) as pdf:
            total = len(pdf.pages)
            logger.info(f"PDF 共 {total} 页")
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                # 1. 提取纯文本
                text = page.extract_text() or ""
                text = text.strip()

                # 2. 提取表格并转 Markdown 追加到文本后
                tables = page.extract_tables() or []
                table_md_list = [_table_to_markdown(t) for t in tables if t]
                table_md = "\n\n".join([md for md in table_md_list if md])
                if table_md:
                    if text:
                        text = text + "\n\n" + table_md
                    else:
                        text = table_md

                # 3. 字数过少（疑似扫描页）触发 OCR 兜底
                if len(text) < MIN_CHARS:
                    logger.info(
                        f"第 {page_num} 页文本仅 {len(text)} 字，触发 OCR 兜底..."
                    )
                    ocr_text = _ocr_page_with_pymupdf(self.pdf_path, i)
                    if ocr_text:
                        # OCR 结果与原文本合并去重
                        if text:
                            text = text + "\n" + ocr_text
                        else:
                            text = ocr_text
                        logger.info(f"第 {page_num} 页 OCR 补充 {len(ocr_text)} 字")

                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "page": page_num,
                                "source": "student_handbook",
                                "file_path": self.pdf_path,
                            },
                        )
                    )
                else:
                    logger.warning(f"第 {page_num} 页无任何文本内容，跳过")

        logger.info(f"PDF 加载完成，共生成 {len(documents)} 页文档")
        return documents



