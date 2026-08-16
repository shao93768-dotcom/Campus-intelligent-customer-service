# -*- coding:utf-8 -*-
"""
离线工具脚本：建库 / 训练分类器 / 测试检索。

用法：
    # 1. 建立学生手册知识库（处理 PDF + 切分 + 入库 Milvus）
    python rag_main.py build

    # 2. 训练 BERT 三分类器
    python rag_main.py train

    # 3. 评估分类器（含困难集）
    python rag_main.py eval

    # 4. 测试向量检索
    python rag_main.py search "晚归怎么处分"

    # 5. 端到端测试（三意图）
    python rag_main.py test "你好呀"
"""
import os
import sys
import json
import argparse

# 保证项目根在 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from base import logger, Config

conf = Config()


def find_handbook_pdf():
    """在 data 目录下查找学生手册 PDF。"""
    data_dir = os.path.join(_project_root, "data")
    if not os.path.exists(data_dir):
        return None
    for f in os.listdir(data_dir):
        if f.lower().endswith(".pdf"):
            return os.path.join(data_dir, f)
    return None


def cmd_build():
    """建库：处理 PDF + 切分 + 入库。"""
    logger.info("=" * 60)
    logger.info("【建库】处理学生手册 PDF → 切分 → 入库 Milvus")
    logger.info("=" * 60)

    pdf_path = find_handbook_pdf()
    if not pdf_path:
        logger.error(f"data 目录下未找到 PDF 文件: {os.path.join(_project_root, 'data')}")
        return

    logger.info(f"PDF 路径: {pdf_path}")

    # 1. 切分
    from rag_qa.core.document_processor import process_handbook

    chunks = process_handbook(pdf_path)
    logger.info(f"切分完成，子块数: {len(chunks)}")

    # 2. 入库
    from rag_qa.core.vector_store import VectorStore

    vs = VectorStore()
    # 分批入库，避免单次过大
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vs.add_document(batch)
        logger.info(f"已入库 {min(i + batch_size, len(chunks))}/{len(chunks)}")

    logger.info("✅ 建库完成！")


def cmd_train():
    """训练 BERT 三分类器。"""
    logger.info("=" * 60)
    logger.info("【训练】BERT 三意图分类器")
    logger.info("=" * 60)

    from rag_qa.core.query_classifier import QueryClassifier

    clf = QueryClassifier()
    data_file = os.path.join(_project_root, "train_data", "intent_train.jsonl")
    clf.train_model(data_file=data_file, epochs=3, batch_size=8)
    logger.info("✅ 训练完成！模型已保存到 rag_qa/core/bert_query_classifier")


def cmd_eval():
    """评估分类器（训练集验证 + 困难集）。"""
    logger.info("=" * 60)
    logger.info("【评估】分类器性能")
    logger.info("=" * 60)

    from rag_qa.core.query_classifier import QueryClassifier

    clf = QueryClassifier()

    # 困难集评估
    hard_file = os.path.join(_project_root, "train_data", "intent_test_hard.jsonl")
    if os.path.exists(hard_file):
        with open(hard_file, "r", encoding="utf-8") as f:
            hard_data = [json.loads(line) for line in f if line.strip()]
        texts = [d["query"] for d in hard_data]
        labels = [d["label"] for d in hard_data]

        logger.info(f"困难集大小: {len(texts)}")
        correct = 0
        wrong_cases = []
        for q, true_label in zip(texts, labels):
            pred = clf.predict_category(q)
            if pred == true_label:
                correct += 1
            else:
                wrong_cases.append({"query": q, "true": true_label, "pred": pred})

        acc = correct / len(texts)
        logger.info(f"困难集准确率: {acc:.4f} ({correct}/{len(texts)})")
        if wrong_cases:
            logger.info("错误案例:")
            for c in wrong_cases:
                logger.info(f"  query={c['query']} | 真={c['true']} 预测={c['pred']}")
    else:
        logger.warning(f"困难集文件不存在: {hard_file}")


def cmd_search(query):
    """测试向量检索。"""
    logger.info("=" * 60)
    logger.info(f"【检索测试】query: {query}")
    logger.info("=" * 60)

    from rag_qa.core.vector_store import VectorStore

    vs = VectorStore()
    results = vs.hybird_search_with_rerank(query)
    print(f"\n检索到 {len(results)} 个文档块:\n")
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        print(f"--- 结果 {i} ---")
        print(f"章节: 第{meta.get('chapter', '?')}章《{meta.get('chapter_title', '?')}》")
        print(f"条号: 第{meta.get('clause', '?')}条 | 页码: {meta.get('page', '?')}")
        print(f"类型: {meta.get('chunk_type', 'text')}")
        print(f"内容: {doc.page_content[:200]}...")
        print()


def cmd_test(query):
    """端到端测试（三意图路由）。"""
    logger.info("=" * 60)
    logger.info(f"【端到端测试】query: {query}")
    logger.info("=" * 60)

    from main import CampusQASystem

    system = CampusQASystem()
    print(f"\n学生: {query}")
    print(f"校小通: ", end="", flush=True)
    for chunk in system.stream_answer(query):
        if chunk.startswith("[intent:"):
            intent = chunk.replace("[intent:", "").replace("]", "")
            print(f"\n[意图: {intent}]", end="\n回复: ", flush=True)
            continue
        print(chunk, end="", flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(description="校园学生手册 RAG 系统工具")
    parser.add_argument("command", choices=["build", "train", "eval", "search", "test"],
                        help="build=建库, train=训练分类器, eval=评估, search=检索测试, test=端到端测试")
    parser.add_argument("--query", "-q", default="晚归怎么处分", help="search/test 时的查询语句")
    args = parser.parse_args()

    if args.command == "build":
        cmd_build()
    elif args.command == "train":
        cmd_train()
    elif args.command == "eval":
        cmd_eval()
    elif args.command == "search":
        cmd_search(args.query)
    elif args.command == "test":
        cmd_test(args.query)


if __name__ == "__main__":
    main()
