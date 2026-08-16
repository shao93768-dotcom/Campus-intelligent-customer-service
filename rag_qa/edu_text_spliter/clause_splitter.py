# -*- coding:utf-8 -*-
"""
学生手册条款切分器（ClauseSplitter）。

针对"第X章 → 第X条"法规式结构，采用：
- 父块 = 整章（生成时提供完整上下文）
- 子块 = 单条法规（检索时精度高、定位准）
- 表格单独成块（type=table）
- metadata 带 chapter/chapter_title/clause/page，支持"依据《手册》第X章第X条"溯源

这是相比通用递归切分（方案A）的关键改进：避免条款被腰斩、跨章混切，
同时为论文提供"可解释性 + 引用溯源"卖点。
"""
import os
import re
import sys

_current = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_current)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langchain.docstore.document import Document
from base import logger

# 中文数字 + 阿拉伯数字 通用匹配
_CHN_NUM = r"[一二三四五六七八九十百零〇0-9]+"
# 章标题：第三章 学生管理 / 第3章 学籍管理
CHAPTER_PATTERN = re.compile(rf"第{_CHN_NUM}章\s*[^\n]{{0,50}}")
# 条标题：第十五条 / 第3条 / 第十五条（
CLAUSE_PATTERN = re.compile(rf"第{_CHN_NUM}条")
# 从章标题中提取章号与章名
CHAPTER_INFO_PATTERN = re.compile(rf"第({_CHN_NUM})章\s*(.+)")


def _cn_to_int(cn):
    """把中文数字或阿拉伯数字字符串转 int。失败返回原字符串。"""
    cn_map = {
        "零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }
    if cn.isdigit():
        return int(cn)
    # 简单处理：十、二十、十五、十二、一百二十等常见形态
    try:
        if "十" in cn:
            parts = cn.split("十")
            left = parts[0]
            right = parts[1] if len(parts) > 1 else ""
            left_v = 0
            if left:
                left_v = cn_map.get(left, 0) if len(left) == 1 else sum(
                    cn_map.get(c, 0) * (10 ** (len(left) - i - 1)) for i, c in enumerate(left)
                )
                if left_v == 0:
                    left_v = 1  # "十" 单独出现 = 10
            right_v = 0
            if right:
                right_v = cn_map.get(right, 0) if len(right) == 1 else sum(
                    cn_map.get(c, 0) * (10 ** (len(right) - i - 1)) for i, c in enumerate(right)
                )
            return left_v * 10 + right_v
        # 纯汉字逐位
        return int("".join(str(cn_map.get(c, c)) for c in cn))
    except Exception:
        return cn  # 转换失败原样返回


class ClauseSplitter:
    """
    按条款切分 + 父子分块。

    输入：load_documents 返回的按页 Document 列表
    输出：子块 Document 列表，metadata 带：
        - chapter: 章号（int 或 str）
        - chapter_title: 章名
        - clause: 条号（int 或 str）
        - page: 页码
        - parent_id: 父块ID（"chapter_X"）
        - parent_content: 所在章完整文本
        - chunk_type: "text" / "table"
        - source / timestamp
    """

    def __init__(self, max_clause_len=600, fallback_chunk_size=300, chunk_overlap=50):
        """
        :param max_clause_len: 单条法规超过此长度时按 fallback 切分，避免过长
        :param fallback_chunk_size: 兜底切分块大小
        :param chunk_overlap: 兜底切分重叠
        """
        self.max_clause_len = max_clause_len
        self.fallback_chunk_size = fallback_chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents):
        """
        切分主入口。
        :param documents: 按页的 Document 列表（来自 HandbookPDFLoader.load()）
        :return: 子块 Document 列表
        """
        # 1. 先把所有页文本合并，按章切分
        chapters = self._split_into_chapters(documents)
        logger.info(f"按章切分完成，共 {len(chapters)} 章")

        # 2. 对每章按条切分，建立父子关系
        child_chunks = []
        for ch in chapters:
            clauses = self._split_chapter_into_clauses(ch)
            for cl in clauses:
                # 过长条款兜底切分
                if len(cl.page_content) > self.max_clause_len:
                    cl_chunks = self._fallback_split(cl)
                    child_chunks.extend(cl_chunks)
                else:
                    child_chunks.append(cl)
        logger.info(f"条款切分完成，共 {len(child_chunks)} 个子块")
        return child_chunks

    def _split_into_chapters(self, documents):
        """
        把按页文档合并，按"第X章"切分为章列表。
        跨页的章会被合并，每章记录其起止页与完整文本。
        """
        # 合并全文，保留每段的页码
        full_text_parts = []
        for doc in documents:
            full_text_parts.append((doc.page_content, doc.metadata.get("page", 1)))
        full_text = "\n".join(t for t, _ in full_text_parts)

        # 找所有章标题位置
        matches = list(CHAPTER_PATTERN.finditer(full_text))
        if not matches:
            # 没识别到章节结构，整篇当作一章
            logger.warning("未识别到章节结构，整篇作为一章处理")
            ch = {
                "chapter": 1,
                "chapter_title": "学生手册全文",
                "content": full_text,
                "start_page": documents[0].metadata.get("page", 1) if documents else 1,
            }
            return [ch]

        chapters = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            chapter_text = full_text[start:end].strip()
            # 解析章号、章名
            info = CHAPTER_INFO_PATTERN.search(m.group())
            if info:
                ch_no = _cn_to_int(info.group(1))
                ch_title = info.group(2).strip().split("\n")[0].strip()
            else:
                ch_no = i + 1
                ch_title = m.group().strip()

            # 估算起始页（在合并文本中的偏移对应的页）
            # 简化：用第一个匹配到该章文本的页
            start_page = self._estimate_page(full_text_parts, start)

            chapters.append(
                {
                    "chapter": ch_no,
                    "chapter_title": ch_title,
                    "content": chapter_text,
                    "start_page": start_page,
                }
            )
        return chapters

    def _estimate_page(self, full_text_parts, offset):
        """根据合并文本中的偏移估算所属页码。"""
        acc = 0
        for text, page in full_text_parts:
            if acc + len(text) + 1 > offset:  # +1 是 join 的换行
                return page
            acc += len(text) + 1
        return full_text_parts[-1][1] if full_text_parts else 1

    def _split_chapter_into_clauses(self, chapter):
        """
        把一章文本按"第X条"切分为子块。
        每个子块 metadata 带 chapter/chapter_title/clause/page/parent_id/parent_content。
        """
        content = chapter["content"]
        ch_no = chapter["chapter"]
        ch_title = chapter["chapter_title"]
        parent_id = f"chapter_{ch_no}"
        parent_content = content

        # 找所有"第X条"位置
        matches = list(CLAUSE_PATTERN.finditer(content))
        if not matches:
            # 本章无条款结构，整章作为一个子块
            return [
                self._make_chunk(
                    content, ch_no, ch_title, None, chapter["start_page"],
                    parent_id, parent_content, "text",
                )
            ]

        clauses = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            clause_text = content[start:end].strip()
            # 解析条号
            cn_match = re.search(rf"第({_CHN_NUM})条", m.group())
            cl_no = _cn_to_int(cn_match.group(1)) if cn_match else None

            # 简单判断是否含表格（Markdown 表格标记 | ）
            chunk_type = "table" if "|" in clause_text and "---" in clause_text else "text"

            clauses.append(
                self._make_chunk(
                    clause_text, ch_no, ch_title, cl_no, chapter["start_page"],
                    parent_id, parent_content, chunk_type,
                )
            )
        return clauses

    def _make_chunk(self, text, ch_no, ch_title, cl_no, page, parent_id, parent_content, chunk_type):
        """构造一个子块 Document。"""
        meta = {
            "chapter": ch_no,
            "chapter_title": ch_title,
            "clause": cl_no if cl_no is not None else "",
            "page": page,
            "parent_id": parent_id,
            "parent_content": parent_content,
            "chunk_type": chunk_type,
            "source": "student_handbook",
        }
        return Document(page_content=text, metadata=meta)

    def _fallback_split(self, doc):
        """过长条款用递归字符切分兜底，每块继承 metadata。"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.fallback_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        texts = splitter.split_text(doc.page_content)
        chunks = []
        for idx, t in enumerate(texts):
            meta = dict(doc.metadata)
            meta["sub_idx"] = idx
            chunks.append(Document(page_content=t, metadata=meta))
        return chunks


if __name__ == "__main__":
    # 自测：切分示例文本
    sample = [
        Document(
            page_content=(
                "第一章 总则\n"
                "第一条 为规范学生管理，制定本手册。\n"
                "第二条 本手册适用于全体在校生。\n\n"
                "第二章 学生学籍管理\n"
                "第三条 学生学籍自入学注册之日起取得。\n"
                "第四条 转专业需满足以下条件：成绩前30%，无违纪。"
            ),
            metadata={"page": 1, "source": "student_handbook"},
        )
    ]
    splitter = ClauseSplitter()
    chunks = splitter.split(sample)
    for c in chunks:
        print(f"[{c.metadata.get('chapter')}.{c.metadata.get('clause')}] {c.page_content[:40]}")
