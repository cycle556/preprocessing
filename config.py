# -*- coding: utf-8 -*-
"""全局配置 —— 所有参数集中管理，优先从 .env 读取。"""

import os

# --- 分块参数 ---
MAX_CHARS = 1800          # 单个内容块的最大字符数
MIN_CHARS = 80            # 过短块告警阈值
OVERLAP_CHARS = 80        # prose 子块间的重叠字符数

# --- 结构恢复 (Phase 2) ---
# 优先环境变量，回退到默认值
STRUCTURE_LLM_PROVIDER = os.environ.get(
    "STRUCTURE_PROVIDER",
    os.environ.get("LLM_PROVIDER", "openai")
)
STRUCTURE_LLM_MODEL = os.environ.get(
    "LLM_MODEL",
    "deepseek-v3.2"
)
STRUCTURE_SNIPPET_CHARS = 80  # 每个标题后取前 N 字符作为 LLM 上下文

# --- 语义切分 (Phase 3) ---
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    "BAAI/bge-m3"  # 默认从 HuggingFace/ModelScope 下载；本机用 .env 覆写
)
SEMANTIC_BREAK_THRESHOLD = 0.6

# --- 输出 ---
# 默认输出 xlsx + jsonl（xlsx 用于人工审阅切分质量）
OUTPUT_FORMATS = ["jsonl", "md"]

# --- 页码回标 ---
# 需要原始 PDF + PyMuPDF

# --- 递归切分分隔符（中英双语）---
RECURSIVE_SEPARATORS = [
    "\n\n", "\n",
    "。", "！", "？",
    ". ", "! ", "? ",
    "；", "; ",
    "，", ", ",
    " ", "",
]
