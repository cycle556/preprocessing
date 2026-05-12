# 保险文档 Agentic RAG 系统

基于 LangGraph 的保险文档智能问答系统，支持多保险公司文档的加载、索引、混合检索和带溯源的答案生成。

## 项目结构

```
.
├── main.py                 # 主入口，流程编排
├── agent.py                # Agent 编排器（LangGraph 工作流）
├── document_loader.py      # 文档加载器（PDF/TXT/MD）
├── text_splitter.py        # 文本分块器（语义+递归双策略）
├── vector_store.py         # 向量数据库（ChromaDB / 腾讯云）
├── query_enhancer.py       # 查询增强器（意图识别+实体提取+查询扩展）
├── retriever.py            # 混合检索引擎（语义+BM25+重排序）
├── answer_generator.py     # 答案生成器（LLM + 引用标注）
├── conversation_manager.py # 会话管理器（多轮对话+历史持久化）
├── logger.py               # 日志模块
├── config.yaml             # 全局配置文件
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板
├── 保司文件2.0/             # 保险文档目录（按公司分文件夹）
├── chroma_db/              # ChromaDB 持久化目录（自动生成）
└── logs/                   # 日志目录（自动生成）
```

## 架构概述

```
用户查询
  → query_enhancer   （意图识别、实体提取、查询扩展）
  → retriever         （语义检索 + BM25 关键词检索 → 重排序）
  → answer_generator  （LLM 生成答案，附来源引用）
```

LangGraph 工作流节点：`query_analysis → retrieval → answer_generation`

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入火山引擎 API Key：

```
VOLC_API_KEY=your_api_key_here
VOLC_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3
```

### 3. 放置文档

将保险文档（PDF/TXT/MD）按公司分文件夹放入 `保司文件2.0/` 目录：

```
保司文件2.0/
├── 友邦/
│   ├── 友邦产品手册/
│   └── 友邦投保指引/
├── 保诚/
├── 宏利/
├── 万通/
└── ...
```

公司名会自动从文件夹路径中识别。

### 4. 运行

```bash
# 首次运行：加载文档 → 建立索引 → 演示查询
python main.py

# 交互式多轮对话
python main.py --interactive

# 强制重建索引（文档有更新时）
python main.py --reload

# 跳过索引，直接问答（使用已有向量库）
python main.py --no-index

# 查看系统状态
python main.py --stats
```

### 5. 交互模式命令

| 输入 | 功能 |
|------|------|
| 你的问题 | 基于文档库回答 |
| `new` | 开始新会话 |
| `stats` | 查看系统状态 |
| `quit` / `exit` | 退出 |

## 配置文件说明

`config.yaml` 主要配置项：

| 配置段 | 关键参数 | 说明 |
|--------|---------|------|
| `document` | `source_dir` | 文档源目录 |
| `chunking` | `chunk_size`, `chunk_overlap` | 分块大小（800）和重叠（100） |
| `embedding` | `model` | 向量模型（doubao-embedding-vision） |
| `llm` | `model`, `temperature` | 大语言模型（doubao-seed-2.0-pro） |
| `vector_store` | `provider` | 向量库：`chroma`（本地）或 `tencent_cloud`（云端） |
| `retrieval` | `top_k`, `use_rerank` | 检索结果数和重排序开关 |
| `answer_generation` | `require_citation` | 是否要求答案带来源引用 |

## 核心特性

### 公司级文档隔离

查询时自动识别公司名，检索结果仅返回该公司文档内容，避免跨公司文档串扰。

- 支持中英文公司名识别：友邦/AIA、保诚/Prudential、宏利/Manulife、万通、中国人寿、安盛/AXA 等
- 不带公司名的查询（如"重疾险等待期多久"）跨公司检索

### 混合检索

- **语义检索**：基于向量相似度，捕捉语义相关文档
- **关键词检索**：BM25 算法，精确匹配关键术语
- **重排序**：结合关键词密度、章节标题匹配进行二次排序

### 答案溯源

所有生成的答案均附带来源引用，格式：

```
【来源：文件名.md 第X页 章节名】
```

### 多轮对话

支持上下文感知的多轮对话，自动维护会话历史。

## 向量库切换

支持 ChromaDB（本地）和腾讯云向量数据库两种后端，通过 `config.yaml` 切换：

```yaml
vector_store:
  provider: "chroma"        # 本地 ChromaDB
  # provider: "tencent_cloud"  # 腾讯云向量数据库
```

## 依赖

- Python 3.10+
- LangChain + LangGraph
- ChromaDB
- PyPDF2
- jieba（中文分词）
- rank-bm25（BM25 检索）
- 火山引擎 API（豆包大模型 + Embedding）
