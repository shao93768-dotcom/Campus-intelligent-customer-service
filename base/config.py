# -*- coding:utf-8 -*-
"""
全局配置读取器。
读取项目根目录下的 config.ini，对外暴露统一配置项。
"""
import os
from configparser import ConfigParser

# 定位项目根目录（base 的上两层）
_current = os.path.dirname(os.path.abspath(__file__))       # 当前文件目录
PROJECT_ROOT = os.path.dirname(_current)                    # 项目根目录
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.ini")      # 配置文件路径


class Config:
    """读取 config.ini 的所有配置文件。"""

    def __init__(self):
        self.config = ConfigParser()
        # 读取配置文件，找不到时给空配置兜底，避免启动直接报错
        if os.path.exists(CONFIG_PATH):
            self.config.read(CONFIG_PATH, encoding="utf-8")
        else:
            raise FileNotFoundError(f"配置文件不存在: {CONFIG_PATH}")

        # ---------- MySQL ----------
        self.MYSQL_HOST = self.config.get("mysql", "host", fallback="localhost")
        self.MYSQL_USER = self.config.get("mysql", "user", fallback="")
        self.MYSQL_PASSWORD = self.config.get("mysql", "password", fallback="")
        self.MYSQL_DATABASE = self.config.get("mysql", "database", fallback="campus_qa")

        # ---------- Redis ----------
        self.REDIS_HOST = self.config.get("redis", "host", fallback="localhost")
        self.REDIS_PORT = self.config.getint("redis", "port", fallback=6379)
        self.REDIS_PASSWORD = self.config.get("redis", "password", fallback="")
        self.REDIS_DB = self.config.getint("redis", "db", fallback=0)

        # ---------- Milvus ----------
        self.MILVUS_HOST = self.config.get("milvus", "host", fallback="localhost")
        self.MILVUS_PORT = self.config.get("milvus", "port", fallback="19530")
        self.MILVUS_DATABASE_NAME = self.config.get(
            "milvus", "database_name", fallback="default"
        )
        self.MILVUS_COLLECTION_NAME = self.config.get(
            "milvus", "collection_name", fallback="student_handbook"
        )

        # ---------- Ollama 本地大模型 ----------
        self.OLLAMA_BASE_URL = self.config.get(
            "ollama", "base_url", fallback="http://localhost:11434"
        )
        self.OLLAMA_MODEL = self.config.get("ollama", "model", fallback="qwen2.5:7b")
        self.OLLAMA_TEMPERATURE = self.config.getfloat(
            "ollama", "temperature", fallback=0.3
        )
        self.OLLAMA_NUM_CTX = self.config.getint("ollama", "num_ctx", fallback=4096)

        # ---------- Deepseek 大模型 ----------
        self.BASE_URL = self.config.get("llm", "base_url")
        self.API_KEY = self.config.get("llm", "api_key")
        self.MODEL_NAME = self.config.get("llm", "model_name")
        self.temperature = self.config.get("llm", "temperature")

        # ---------- 检索参数 ----------
        self.PARENT_CHUNK_SIZE = self.config.getint(
            "retrieval", "parent_chunk_size", fallback=1200
        )
        self.CHILD_CHUNK_SIZE = self.config.getint(
            "retrieval", "child_chunk_size", fallback=300
        )
        self.CHUNK_OVERLAP = self.config.getint("retrieval", "chunk_overlap", fallback=50)
        self.RETRIEVAL_K = self.config.getint("retrieval", "retrieval_k", fallback=5)
        self.CANDIDATE_M = self.config.getint("retrieval", "candidate_m", fallback=2)

        # ---------- 应用配置 ----------
        self.SCHOOL_NAME = self.config.get(
            "app", "school_name", fallback=""
        )
        self.CUSTOMER_SERVICE_PHONE = self.config.get(
            "app", "customer_service_phone", fallback=""
        )
        self.CONTACTS_FILE = self.config.get(
            "app", "contacts_file", fallback="contacts/contacts.json"
        )

        # ---------- 关键路径 ----------
        # rag_qa 目录（core 的上两层）
        _core_dir = os.path.join(PROJECT_ROOT, "rag_qa", "core")
        self.RAG_QA_PATH = os.path.join(PROJECT_ROOT, "rag_qa")
        # 模型目录
        self.MODELS_DIR = os.path.join(self.RAG_QA_PATH, "models")
        # BERT 训练后保存路径
        self.BERT_MODEL_PATH = os.path.join(_core_dir, "bert_query_classifier")
        # 数据目录
        self.DATA_DIR = os.path.join(self.RAG_QA_PATH, "data")
