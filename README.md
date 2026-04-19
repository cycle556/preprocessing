# 保险文档Agentic RAG系统 - Python工程化实现

## 代码结构与运行依赖

本代码严格对应框架8个核心模块实现，包含：文档预处理、混合检索、多智能体调度、数据提取、合规校验、会话管理等。依赖：Python 3.10+，OpenAI API，LangChain，ChromaDB，PyPDF2等。

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑.env文件，填入您的OpenAI API Key
```

### 3. 运行示例
```bash
python main.py
```

## 模块说明

| 文件名 | 对应框架模块 | 功能 |
|--------|-------------|------|
| document_processor.py | 保险文档预处理与知识图谱构建 | PDF/Excel解析、条款分块、表格提取 |
| retrieval_engine.py | 保险专属混合检索引擎 | 语义检索+关键词检索混合检索 |
| data_extractor.py | 保险数据精准提取与逻辑推理 | 等待期、保额、免责条款等提取 |
| compliance_checker.py | 全链路合规与事实性校验 | 原文一致性、来源溯源校验 |
| session_manager.py | 会话记忆与文档版本管理 | 对话历史、文档版本管理 |
| agent_orchestrator.py | 多智能体协同调度核心引擎 | LangGraph实现多智能体流程 |
| main.py | 端到端示例 | 完整演示查询流程 |

## 核心功能

- 按保险条款层级结构分块
- 语义+BM25混合检索
- 多智能体协同（意图分析→检索→提取→合规→输出）
- 100%原文提取，不做演绎
- 来源溯源标注
