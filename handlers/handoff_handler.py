# -*- coding:utf-8 -*-
"""
转人工 Handler。

匹配逻辑：
1. 关键词精确匹配（contacts.json 中 keywords 字段）
2. 关键词未命中时，模糊包含匹配
3. 仍未命中时，返回学生处总值班兜底

不调用 LLM，直接返回结构化卡片，节省算力。
"""
import os
import sys
import json

_current = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from base import logger, Config

conf = Config()


class HandoffHandler:
    """转人工：匹配联系方式并返回结构化卡片。"""

    def __init__(self):
        self.contacts = self._load_contacts()
        logger.info(f"加载联系方式库，共 {len(self.contacts)} 条联系人")

    def _load_contacts(self):
        """从 contacts.json 加载联系方式。"""
        path = os.path.join(_project_root, conf.CONTACTS_FILE)
        if not os.path.exists(path):
            logger.error(f"联系方式库不存在: {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def handle(self, query, history=None):
        """
        处理转人工请求，流式 yield 卡片文本。
        """
        matched = self._match(query)
        print(f'matched-->{matched}')
        if matched:
            logger.info(f"匹配到联系人: {matched.get('role')} - {matched.get('name')}")
            card = self._format_card(matched)
            # 心理类特别提示
            if "心理" in matched.get("keywords", ""):
                card = "已识别到您可能需要心理支持，" + card + "\n（心理咨询全程保密，请放心拨打）"
            yield card
        else:
            logger.info(f"未匹配到精确联系人，返回总值班兜底 (query: {query})")
            yield (
                f"未匹配到精确联系人，可拨打学校学生处总值班电话：{conf.CUSTOMER_SERVICE_PHONE}。"
                f"如需更精确的联系方式，可说明具体需求（如“找辅导员”“心理咨询”“宿管”等）再次提问。"
            )

    def _match(self, query):
        """
        关键词匹配联系人。
        :return: 匹配到的联系人 dict，无则 None
        """
        # 1. 精确关键词命中（任一关键词出现在 query 中即命中）
        for c in self.contacts:
            kws = [k.strip() for k in c.get("keywords", "").split(",") if k.strip()]
            print(f'kws-->{kws}')
            print('*'*80)
            for kw in kws:
                if kw and kw in query:
                    return c

        # 2. 模糊匹配：role/description 包含
        for c in self.contacts:
            if c.get("role", "") in query or c.get("department", "") in query:
                return c

        # 3. 投诉类匹配学生处
        complaint_kw = ["投诉", "举报", "申诉", "反映", "找领导"]
        if any(kw in query for kw in complaint_kw):
            for c in self.contacts:
                if "总值班" in c.get("role", ""):
                    return c

        return None

    def _format_card(self, c):
        """格式化联系人卡片。"""
        lines = [
            "已为您匹配到联系人：",
            f"  姓名/单位：{c.get('name', '')}",
            f"  职务：{c.get('role', '')}",
            f"  所属部门：{c.get('department', '')}",
            f"  联系电话：{c.get('phone', '')}",
            f"  办公地点：{c.get('office', '')}",
            f"  办公时间：{c.get('office_hours', '')}",
        ]
        if c.get("description"):
            lines.append(f"  服务说明：{c.get('description', '')}")
        lines.append("如需进一步帮助，可直接电话联系。")
        return "\n".join(lines)


if __name__ == "__main__":
    h = HandoffHandler()
    for q in ["怎么联系辅导员", "我心情好差想找人聊聊", "宿舍空调坏了", "我要投诉"]:
        print(f"\nquery: {q}")
        for piece in h.handle(q):
            print(piece)
