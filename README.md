# 校园学生手册智能问答系统

基于 **Ollama + BERT + RAG** 的校园三意图问答系统（毕业论文项目）。

针对高校学生手册咨询场景，实现 **闲聊 / 转人工 / 专业知识** 三意图路由：BERT 先判断意图，再分发到对应处理链路，兼顾响应速度与回答权威性。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| 🧭 三意图路由 | BERT 三分类（闲聊/转人工/专业知识），避免所有问题都走重 RAG |
| 🏠 本地化 | LLM 走 Ollama 本地部署，数据不出校园，零 API 费用 |
| 📑 条款切分 | 按"第X章/第X条"法规式结构切分 + 父子分块，避免条款腰斩 |
| 🔖 章节溯源 | metadata 带章号/条号/页码，回答可引用"依据《手册》第X章第X条" |
| 🛟 心理安全网 | 检测极端情绪自动引导心理咨询 |
| 🔄 四策略检索 | 直接/回溯/子查询/HyDE，LLM 智能选择 |

---

## 系统架构

```
                    ┌─────────────┐
                    │  学生提问    │
                    └──────┬──────┘
                           ▼
                ┌───────────────────────┐
                │  BERT 三意图分类器    │ ← 闲聊(0)/转人工(1)/专业知识(2)
                └───┬────────┬────────┬─┘
              闲聊  │  转人工 │  专业知识│
                    ▼        ▼         ▼
              ┌──────────┐ ┌────────┐ ┌──────────────┐
              │ Ollama   │ │联系方式 │ │ RAG 检索引擎  │
              │ 直接对话  │ │库匹配  │ │ Milvus+Rerank│
              │(校园人设) │ │(不调LLM)│ │ +条款溯源    │
              └──────────┘ └────────┘ └──────┬───────┘
                                          ▼
                                    ┌──────────┐
                                    │Ollama生成│
                                    └──────────┘
```

---

## 目录结构

```
Campus intelligent customer service/
├── config.ini                      # 全局配置（MySQL/Milvus/Ollama/检索参数）
├── requirements.txt                # Python 依赖
├── main.py                         # ★ 三意图路由中枢（CampusQASystem）
├── api.py                          # ★ FastAPI + SSE 服务入口
├── base/
│   ├── config.py                   # 配置读取器
│   └── logger.py                   # 全局日志
├── llm/
│   └── ollama_client.py            # ★ Ollama 本地封装（call_ollama/stream_ollama）
├── handlers/                       # ★ 三意图处理器
│   ├── chitchat_handler.py         #   闲聊：Ollama对话+人设+心理安全网
│   ├── handoff_handler.py          #   转人工：联系方式库匹配
│   └── knowledge_handler.py        #   专业知识：RAG检索+流式生成
├── contacts/
│   └── contacts.json               # 联系方式库（辅导员/心理/宿管/保卫/教务…）
├── train_data/
│   ├── intent_train.jsonl          # ★ BERT 三分类训练数据（210+条）
│   └── intent_test_hard.jsonl      # 困难集（50条边界模糊样本）
├── rag_qa/
│   ├── rag_main.py                 # ★ 离线工具：建库/训练/评估/测试
│   ├── core/
│   │   ├── query_classifier.py     # ★ BERT 三分类器
│   │   ├── vector_store.py         # Milvus+BGE-M3+Reranker（带章节溯源）
│   │   ├── document_processor.py   # PDF加载→条款切分
│   │   ├── rag_system.py           # RAG 系统（四策略检索+上下文构造）
│   │   ├── strategy_selector.py    # 检索策略选择（接Ollama）
│   │   └── prompts.py              # 全部 Prompt 模板
│   ├── edu_document_loaders/
│   │   └── pdf_loader.py           # ★ 学生手册 PDF 加载（pdfplumber+OCR兜底）
│   ├── edu_text_spliter/
│   │   └── clause_splitter.py      # ★ 按条款切分+父子分块
│   └── models/                     # 模型权重目录（需下载，见下文）
│       ├── bert-base-chinese/
│       ├── bge-m3/
│       └── bge-reranker-large/
└── data/
    └── 湖北理工学院学生手册.pdf     # 学生手册 PDF
```

> ★ 为核心改造/新建文件

---

## 快速开始

### 1. 环境准备

**Python 依赖**：
```bash
cd "Campus intelligent customer service"
pip install -r requirements.txt
```

**模型权重下载**（放到 `rag_qa/models/` 下）：

| 模型 | 用途 | 下载地址 |
|------|------|---------|
| bert-base-chinese | 意图分类 | HuggingFace `bert-base-chinese` |
| bge-m3 | 文本向量化 | HuggingFace `BAAI/bge-m3` |
| bge-reranker-large | 检索重排 | HuggingFace `BAAI/bge-reranker-large` |

目录结构：
```
rag_qa/models/
├── bert-base-chinese/      # 含 vocab.txt、config.json、pytorch_model.bin
├── bge-m3/                 # 含 model.safetensors 等
└── bge-reranker-large/
```

**外部服务**：
- **Ollama**：[官网](https://ollama.com)下载安装，拉取模型：
  ```bash
  ollama pull qwen2.5:7b    # 推荐，中文好，显存约 6GB
  # 显存不足时
  ollama pull qwen2.5:3b
  ```
- **Milvus 2.5**：Docker 一键部署
  ```bash
  docker run -d --name milvus -p 19530:19530 -v milvus_data:/var/lib/milvus \
    milvusdb/milvus:latest milvus run standalone
  ```

### 2. 修改配置

编辑 `config.ini`，确认 Milvus / Ollama / MySQL 地址端口正确：
```ini
[milvus]
host = localhost
port = 19530
collection_name = student_handbook

[ollama]
base_url = http://localhost:11434
model = qwen2.5:7b
```

### 3. 建立知识库

把学生手册 PDF 放到 `data/` 目录（已就位），然后：
```bash
python rag_qa/rag_main.py build
```
该命令会：加载 PDF → 按条款切分 → BGE-M3 向量化 → 入库 Milvus。

### 4. 训练分类器

```bash
python rag_qa/rag_main.py train
```
使用 `train_data/intent_train.jsonl` 训练 BERT 三分类，3 轮 epoch，完成后模型保存到 `rag_qa/core/bert_query_classifier/`。

评估（含困难集）：
```bash
python rag_qa/rag_main.py eval
```

### 5. 启动服务

```bash
python api.py
# 或
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 6. 测试

```bash
# 端到端测试（三意图）
python rag_qa/rag_main.py test "你好呀"
python rag_qa/rag_main.py test "怎么联系辅导员"
python rag_qa/rag_main.py test "晚归怎么处分"

# 单独测试向量检索
python rag_qa/rag_main.py search "奖学金评定标准"
```

---

## API 接口

### SSE 流式问答
```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"晚归怎么处分","session_id":"test"}'
```
返回 SSE 事件流：
```
data: {"type":"intent","intent":"专业知识"}
data: {"type":"token","content":"根据《学生手册》..."}
data: {"type":"done"}
```

### 非流式问答
```bash
curl -X POST http://localhost:8000/api/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"query":"你好","session_id":"test"}'
```
返回：
```json
{"query": "你好", "answer": "你好呀！我是校小通..."}
```

### 健康检查
```bash
curl http://localhost:8000/api/health
```

---

## 三意图设计（论文核心）

| 意图 | 标签 | 触发场景 | 处理方式 |
|------|------|---------|---------|
| 闲聊 | 0 | 问候、知识问答、情绪、写作 | Ollama 直接生成（带人设） |
| 转人工 | 1 | 联系老师、投诉、求助、心理支持 | 联系方式库匹配，不调 LLM |
| 专业知识 | 2 | 规章制度、奖惩、流程 | RAG 检索 + Ollama 生成 |

**边界模糊样本**（见 `train_data/intent_test_hard.jsonl`）：
- "帮我写个请假条"（闲聊）vs "请假流程是什么"（专业知识）
- "宿舍太吵怎么办"（转人工）vs "宿舍管理规定"（专业知识）

---

## 评估方案（论文章节）

### 1. 意图分类层
- Accuracy / Precision / Recall / F1（各类）
- Macro-F1 / 混淆矩阵
- **困难集准确率**（边界模糊样本）

### 2. RAG 检索层
- Recall@5 / MRR / Hit Rate

### 3. 端到端生成层
- Faithfulness（忠实度）/ Answer Relevancy（相关性）
- Context Precision / Context Recall

### 4. 对比实验（论文加分）
| 对比维度 | 实验 |
|---------|------|
| LLM 规模 | qwen2.5:3b vs 7b vs 14b |
| 切分策略 | 固定字数 vs 条款+父子分块 |
| 检索方式 | 纯稠密 vs 稠密+稀疏 vs +Rerank |
| 有无意图路由 | 三意图 vs 全走 RAG |

---

## 技术栈

| 层 | 技术 |
|----|------|
| 分类 | bert-base-chinese + Transformers |
| 嵌入 | BAAI/bge-m3（稠密+稀疏） |
| 重排 | BAAI/bge-reranker-large |
| 向量库 | Milvus 2.5 + pymilvus |
| 本地 LLM | Ollama + langchain-ollama |
| Web | FastAPI + SSE |
| PDF | pdfplumber + PyMuPDF + rapidocr（兜底） |

---

## 常见问题

**Q: BERT 模型还没训练，系统能用吗？**
A: 能。`main.py` 在检测到模型未训练时会自动启用规则兜底分类器（关键词匹配），系统可正常路由。训练后自动切换为 BERT。

**Q: Milvus/Ollama 没启动，专业知识功能怎么办？**
A: 系统会优雅降级——专业知识意图会返回"功能不可用，请联系客服"。其他两个意图不受影响。

**Q: 训练数据只有 210 条够吗？**
A: 作为种子集可用。建议用 LLM 扩展到每类 1500+ 条（共 5000+），效果更好。可用 DeepSeek/Qwen 批量生成同类变体。

**Q: 学生手册 PDF 是扫描件怎么办？**
A: `pdf_loader.py` 已实现自适应——提取字数过少的页会自动 OCR 兜底（PyMuPDF 渲染 + rapidocr 识别）。
