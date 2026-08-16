# -*- coding:utf-8 -*-
"""
BERT 三意图分类器。

    闲聊(0) / 转人工(1) / 专业知识(2)

"""
import os
import sys
import json
import torch
import numpy as np

_current = os.path.dirname(os.path.abspath(__file__))       # core 目录
_rag_qa_path = os.path.dirname(_current)       # rag_qa 目录
_project_root = os.path.dirname(_rag_qa_path)       # 项目目录
for _p in (_rag_qa_path, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base import logger
# bert分词器, 模型, 训练器, 训练参数
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
# 数据集切分
from sklearn.model_selection import train_test_split
# 分类报告, 混淆矩阵, 准确率, F1-score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# 三分类标签映射
LABEL_MAP = {"闲聊": 0, "转人工": 1, "专业知识": 2}
IDX_LABEL_MAP = {0: "闲聊", 1: "转人工", 2: "专业知识"}
NUM_LABELS = 3
TARGET_NAMES = ["闲聊", "转人工", "专业知识"]


class QueryClassifier:
    """BERT 三意图分类器。"""

    def __init__(self, model_path=None):
        """
        :param model_path: 已训练模型保存路径；为空则用未训练的预训练模型
        """
        # 如果没有给路径从默认路径加载 模型保存路径默认 core/bert_query_classifier
        if model_path is None:
            model_path = os.path.join(_current, "bert_query_classifier")
        self.model_path = model_path

        # 加载 bert-base-chinese 分词器
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(_rag_qa_path, "models", "bert-base-chinese")
        )
        # 初始化模型, 设备, 标签映射, 加载模型函数
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"分类器使用设备: {self.device}")
        self.label_map = LABEL_MAP
        self.idx_label_map = IDX_LABEL_MAP
        self.load_model()

    def load_model(self):
        """加载已训练模型；不存在则初始化未训练的预训练模型。"""

        # 如果存在模型并且 config.json 也存在则加载模型
        if os.path.exists(self.model_path) and os.path.exists(
            os.path.join(self.model_path, "config.json")
        ):
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            logger.info(f"成功加载已训练分类模型: {self.model_path}")
        # 否则直接加载基础bert模型
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                os.path.join(_rag_qa_path, "models", "bert-base-chinese"),
                num_labels=NUM_LABELS,
            )
            self.model.to(self.device)
            logger.warning(
                f"未找到已训练模型，初始化预训练 bert-base-chinese (num_labels={NUM_LABELS})。"
                f"请先运行 train_model() 训练。"
            )

    # ---------- 训练模型函数 ----------
    def train_model(self, data_file=None, output_dir=None, epochs=3, batch_size=8):
        """
        训练三分类模型。
        :param data_file: 训练数据 JSONL 路径
        :param output_dir: 训练输出目录
        :param epochs: 训练轮数
        :param batch_size: 批次大小
        """
        # 如果数据文件和模型保存路径不存在, 则给定默认路径
        if data_file is None:
            data_file = os.path.join(_project_root, "train_data", "intent_train.jsonl")
        if output_dir is None:
            output_dir = os.path.join(_current, "bert_results")
        if not os.path.exists(data_file):
            logger.error(f"训练数据文件不存在: {data_file}")
            raise FileNotFoundError(f"训练数据文件不存在: {data_file}")

        # 读取 JSON
        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        # 获取文本和标签
        texts = [d["query"] for d in data]
        labels_str = [d["label"] for d in data]

        # 判断标签是否存在给定的标签映射中
        for lb in set(labels_str):
            if lb not in self.label_map:
                raise ValueError(f"数据中存在未知标签: {lb}，合法标签: {list(self.label_map.keys())}")

        logger.info(f"训练数据加载完成，共 {len(data)} 条，标签分布: " +
                    ", ".join([f"{k}:{labels_str.count(k)}" for k in self.label_map.keys()]))


        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels_str, test_size=0.2, random_state=100
        )

        # 预处理切分的训练集和测试集, 分词转为模型输入
        train_enc, train_lab = self.preprocess_data(train_texts, train_labels)
        val_enc, val_lab = self.preprocess_data(val_texts, val_labels)

        # 训练集和验证集 dataset
        train_ds = self.create_dataset(train_enc, train_lab)
        val_ds = self.create_dataset(val_enc, val_lab)

        # 使用 TrainingArguments 配置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=35,
            weight_decay=0.01,
            logging_dir=os.path.join(_current, "bert_logs"),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            fp16=(self.device == "cuda"),
        )

        # 模型训练
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=self.compute_metrics,
        )
        logger.info("开始训练三分类模型...")
        trainer.train()
        self.save_model()
        # 训练后评估
        self.evaluate_model(val_texts, val_labels)

    # ---------- 评估 ----------
    def evaluate_model(self, texts, labels_str):
        """评估模型，输出分类报告 + 混淆矩阵。"""
        encodings = self.tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=128, return_tensors="pt",
        )
        labels = torch.tensor([self.label_map[l] for l in labels_str])
        dataset = self.create_dataset(encodings, labels)
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        pred_labels = np.argmax(predictions.predictions, axis=-1)

        logger.info(f"\n=== 三分类评估报告 ===\n"
                    f"{classification_report(labels.tolist(), pred_labels.tolist(), target_names=TARGET_NAMES)}")
        logger.info(f"混淆矩阵:\n{confusion_matrix(labels.tolist(), pred_labels.tolist())}")
        acc = accuracy_score(labels.tolist(), pred_labels.tolist())
        macro_f1 = f1_score(labels.tolist(), pred_labels.tolist(), average="macro")
        logger.info(f"Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}")

    def save_model(self):
        """保存模型与分词器。"""
        os.makedirs(self.model_path, exist_ok=True)
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"分类模型保存成功: {self.model_path}")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": (predictions == labels).mean()}

    def preprocess_data(self, texts, labels_str):
        encodings = self.tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=128, return_tensors="pt",
        )
        labels = torch.tensor([self.label_map[l] for l in labels_str])
        return encodings, labels

    def create_dataset(self, encodings, labels):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                d = {k: v[idx] for k, v in self.encodings.items()}
                d["labels"] = self.labels[idx]
                return d

        return Dataset(encodings, labels)

    # ---------- 预测 ----------
    def predict_category(self, query):
        """预测单条 query 的意图，返回标签字符串。"""
        if self.model is None:
            logger.error("模型未加载")
            return "闲聊"  # 兜底
        encoding = self.tokenizer(
            query, truncation=True, padding="max_length",
            max_length=128, return_tensors="pt",
        )
        # print( encoding)
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            pred = torch.argmax(outputs.logits, dim=-1).item()
        return self.idx_label_map[pred]

    def predict_proba(self, query):
        """返回三类的概率分布（softmax）。"""
        import torch.nn.functional as F
        encoding = self.tokenizer(
            query, truncation=True, padding="max_length",
            max_length=128, return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = self.model(**encoding).logits
            probs = F.softmax(logits, dim=-1)[0]
        return {self.idx_label_map[i]: float(probs[i]) for i in range(NUM_LABELS)}


if __name__ == "__main__":
    clf = QueryClassifier()
    # 训练（首次运行）
    # clf.train_model()
    # 测试预测
    for q in ["你好呀", "辅导员电话多少", "晚归怎么处分"]:
        print(f"{q} -> {clf.predict_category(q)}")
