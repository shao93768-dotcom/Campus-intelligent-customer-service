# -*- coding:utf-8 -*-
"""
学生手册文档处理：加载 PDF → 按条款切分（父子分块）。

相比参考项目通用递归切分，这里用 ClauseSplitter 按"第X章/第X条"切分，
保证条款完整、支持章节溯源。
"""
import os
import sys
from datetime import datetime

_current = os.path.dirname(os.path.abspath(__file__))
_rag_qa_path = os.path.dirname(_current)
_project_root = os.path.dirname(_rag_qa_path)
for _p in (_rag_qa_path, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base import logger, Config
from edu_document_loaders import HandbookPDFLoader
from edu_text_spliter import ClauseSplitter

conf = Config()


def load_handbook(pdf_path):
    """
    加载学生手册 PDF（按页返回 Document）。
    :param pdf_path: PDF 文件路径
    :return: List[Document]，每页一个，metadata 带 page
    """
    loader = HandbookPDFLoader(pdf_path)
    return loader.load()


def process_handbook(pdf_path):
    """
    加载并按条款切分学生手册。
    :param pdf_path: PDF 文件路径
    :return: List[Document] 子块，metadata 带 chapter/clause/page/parent_id/parent_content
    """
    logger.info(f"开始处理学生手册: {pdf_path}")
    pages = load_handbook(pdf_path)
    logger.info(f"PDF 加载完成，共 {len(pages)} 页")

    splitter = ClauseSplitter(
        max_clause_len=conf.CHILD_CHUNK_SIZE * 2,
        fallback_chunk_size=conf.CHILD_CHUNK_SIZE,
        chunk_overlap=conf.CHUNK_OVERLAP,
    )
    child_chunks = splitter.split(pages)

    # 补充 timestamp
    ts = datetime.now().isoformat()
    for ch in child_chunks:
        ch.metadata["timestamp"] = ts

    logger.info(f"切分完成，共 {len(child_chunks)} 个子块")
    # 统计章节分布
    chapter_count = len(set(c.metadata.get("chapter") for c in child_chunks))
    table_count = sum(1 for c in child_chunks if c.metadata.get("chunk_type") == "table")
    logger.info(f"覆盖 {chapter_count} 章，含 {table_count} 个表格块")
    return child_chunks


if __name__ == "__main__":
    # 自测：处理 data 目录下的学生手册
    pdf = os.path.join(_project_root, "data", "湖北理工学院学生手册.pdf")
    if os.path.exists(pdf):
        chunks = process_handbook(pdf)
        print(f"切分子块数: {len(chunks)}")
        if chunks:
            c = chunks[0]
            print(f"首块 metadata: {c.metadata}")
            print(f"首块内容前100字: {c.page_content[:100]}")
    else:
        print(f"未找到 PDF: {pdf}")
