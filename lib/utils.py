# -*- coding: utf-8 -*-
"""通用工具：token 估算、环境变量加载、文件 I/O。"""

import os
import re
from pathlib import Path


# --- Token 估算 ---

def _make_token_counter():
    """尝试用 tiktoken，不可用时回退为启发式估计。"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s))
    except Exception:
        def heuristic(s: str) -> int:
            cjk = sum(1 for ch in s if '一' <= ch <= '鿿')
            other = len(s) - cjk
            return int(cjk * 1.5 + other / 4)
        return heuristic


count_tokens = _make_token_counter()


# --- 环境变量 ---

def load_dotenv(extra_dirs: list = None):
    """手动加载 .env 文件，不依赖 python-dotenv。

    优先级：lib/.env > 额外目录/.env > 系统环境变量（系统环境变量已存在的不会被覆盖）
    """
    candidates = [Path(__file__).resolve().parent / '.env']  # lib/.env 最高优
    if extra_dirs:
        for d in extra_dirs:
            p = Path(d) / '.env'
            if p.is_file():
                candidates.append(p)
    candidates.append(Path.cwd() / '.env')  # CWD/.env 最后

    for env_path in candidates:
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key, val = key.strip(), val.strip()
            val = val.strip('\'"')
            if key not in os.environ:
                os.environ[key] = val


# --- ID 处理 ---

def sanitize_for_id(name: str) -> str:
    """文件基名 → 分块标识符前缀。保留 CJK 字符，替换其他符号。"""
    s = re.sub(r'[^A-Za-z0-9一-鿿]+', '_', name)
    return s.strip('_').lower() or "chunk"


# --- 文件 I/O ---

def read_md(path: str | Path) -> str:
    return Path(path).read_text(encoding='utf-8')


def write_md(path: str | Path, content: str):
    Path(path).write_text(content, encoding='utf-8')
