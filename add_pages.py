# -*- coding: utf-8 -*-
"""
页码回标脚本 —— 将 chunk JSONL 中的每个 chunk 映射回原始 PDF 的页码。

描述（Agentic RAG）：
    页码不需要精确到行。Agent 只需要知道 chunk 大概在 PDF 的第几页，
    就能在需要引证时定位原文。因此本脚本追求高覆盖率（>90% chunk 有页码），
    而非精确的页码边界。

流程：
    1. 用 PyMuPDF 逐页提取 PDF 文本，构建 {页码: 页面文本} 映射
    2. 读取 chunk JSONL，对每个 chunk 提取"锚文本"（前 100 字符）
    3. 在页面文本中搜索锚文本，找到匹配页
    4. 向前后扩展，确定 chunk 覆盖的页码范围
    5. 输出带 page_start / page_end 的新 JSONL

用法：
    python add_pages.py chunks.jsonl 原始.pdf -o chunks_with_pages.jsonl

依赖：
    pip install pymupdf
"""

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional


# ============================================================
# 文本标准化
# ============================================================

def _normalize(text: str) -> str:
    """将文本标准化以便跨 PDF/Markdown 比较。"""
    # 统一空白字符
    text = re.sub(r'\s+', ' ', text)
    # 移除 PDF 常见的软连字符和零宽字符
    text = text.replace('\x00', '').replace('\xad', '')
    # 统一中文标点两侧的空格
    text = re.sub(r'\s*([，。！？；：、])\s*', r'\1', text)
    return text.strip()


# ============================================================
# PDF 页面文本提取
# ============================================================

def extract_pages(pdf_path: str) -> dict[int, str]:
    """从 PDF 逐页提取文本，返回 {页码(1-based): 页面文本}。"""
    import fitz  # PyMuPDF

    pages: dict[int, str] = {}
    doc = fitz.open(pdf_path)

    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages[i] = _normalize(text)

    doc.close()
    return pages


# ============================================================
# 锚文本提取
# ============================================================

def _extract_anchors(content: str, section_title: str = '') -> list[str]:
    """从 chunk 内容中提取多个候选锚文本（用于多轮匹配尝试）。

    返回至多 3 个锚文本，按优先级排列：
    1. 第一段散文（前 3 行合并）
    2. 中间段散文
    3. 标题 + 内容开头
    """
    lines = content.split('\n')
    prose_lines = [
        l.strip() for l in lines
        if l.strip()
        and not l.strip().startswith('|')
        and not l.strip().startswith('#')
        and not l.strip().startswith('- ')
    ]

    anchors = []

    # 锚 1：前 3 行散文
    if prose_lines:
        anchors.append(_normalize(' '.join(prose_lines[:3])))

    # 锚 2：中间部分（如果内容足够长）
    if len(prose_lines) > 6:
        mid = len(prose_lines) // 2
        anchors.append(_normalize(' '.join(prose_lines[mid:mid+3])))

    # 锚 3：标题 + 内容
    if section_title:
        combined = _normalize(section_title + ' ' + content)
        if combined not in anchors:
            anchors.append(combined)

    # 去重、限制长度
    seen = set()
    result = []
    for a in anchors:
        if a and a not in seen:
            seen.add(a)
            result.append(a)
    return result


# ============================================================
# 字符 n-gram 匹配（核心策略）
# ============================================================

def _char_ngrams(text: str, n: int = 5) -> set[str]:
    """将文本拆分为字符 n-gram 集合。

    中文字符之间没有天然的分词边界，n-gram 比基于词的匹配
    更鲁棒，能容忍 MinerU 和 PyMuPDF 之间的空格、标点差异。
    """
    # 先移除纯空白，保留单个空格作为分隔
    cleaned = re.sub(r'\s+', '', text)
    if len(cleaned) < n:
        return {cleaned}
    return {cleaned[i:i+n] for i in range(len(cleaned) - n + 1)}


def _find_page_by_ngrams(
    anchor: str,
    pages: dict[int, str],
    n: int = 5,
    min_score: float = 0.25,
) -> Optional[int]:
    """用字符 n-gram 重叠率找到最佳匹配页面。

    对每个页面计算：|anchor的n-gram ∩ 页面的n-gram| / |anchor的n-gram|
    取最高分页面，要求分数 ≥ min_score。
    """
    if not anchor or len(anchor) < 20:
        return None

    anchor_ngrams = _char_ngrams(anchor, n)
    if len(anchor_ngrams) < 4:
        return None

    best_page = None
    best_score = 0.0

    for page_num, page_text in pages.items():
        page_ngrams = _char_ngrams(page_text, n)
        overlap = len(anchor_ngrams & page_ngrams)
        score = overlap / len(anchor_ngrams)

        if score > best_score:
            best_score = score
            best_page = page_num

    if best_score >= min_score:
        return best_page
    return None


def _find_page_range(
    chunk_content: str,
    start_page: int,
    pages: dict[int, str],
    total_pages: int,
) -> tuple[int, Optional[int]]:
    """估算 chunk 的页码范围（start_page → page_end）。

    用 chunk 末尾的 n-gram 在后续 3 页中搜索。如果后续页面
    有显著重叠（≥0.20），扩展 page_end。
    """
    tail_text = _normalize(chunk_content)[-150:]
    if len(tail_text) < 40:
        return start_page, start_page

    tail_ngrams = _char_ngrams(tail_text, 5)
    if len(tail_ngrams) < 5:
        return start_page, start_page

    page_end = start_page
    for offset in range(1, min(4, total_pages - start_page + 1)):
        next_page = start_page + offset
        if next_page not in pages:
            break
        page_ngrams = _char_ngrams(pages[next_page], 5)
        overlap = len(tail_ngrams & page_ngrams) / len(tail_ngrams)
        if overlap >= 0.20:
            page_end = next_page
        else:
            break

    if page_end == start_page:
        return start_page, start_page
    return start_page, page_end


# ============================================================
# 主流程
# ============================================================

def add_pages(
    jsonl_path: str,
    pdf_path: str,
    out_path: str,
    anchor_len: int = 120,
) -> Path:
    """主入口：读入 chunk JSONL，回标页码，输出新 JSONL。

    Args:
        jsonl_path: chunk JSONL 文件路径
        pdf_path: 原始 PDF 路径
        out_path: 输出 JSONL 路径
        anchor_len: 锚文本长度（字符）
    """
    # 1. 提取 PDF 页面
    print(f"提取 PDF 页面文本: {pdf_path}")
    pages = extract_pages(pdf_path)
    print(f"  共 {len(pages)} 页有文本内容")

    # 2. 读入 chunk JSONL
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"  共 {len(chunks)} 个 chunk")

    # 3. 逐 chunk 匹配页码（多锚点 + n-gram 策略）
    matched = 0
    unmatched = 0
    total_pages = len(pages)

    for i, chunk in enumerate(chunks):
        content = chunk.get('content', '')
        section_title = chunk.get('section_title', '')
        anchors = _extract_anchors(content, section_title)

        page = None
        for anchor in anchors:
            page = _find_page_by_ngrams(anchor, pages)
            if page is not None:
                break

        if page is not None:
            matched += 1
            page_start, page_end = _find_page_range(
                content, page, pages, total_pages
            )
            chunk['page_start'] = page_start
            chunk['page_end'] = page_end
        else:
            unmatched += 1
            chunk['page_start'] = None
            chunk['page_end'] = None

        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(chunks)}  (已匹配 {matched})")

    # 4. 输出
    out = Path(out_path)
    with open(out, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    print(f"\n结果:")
    print(f"  匹配成功: {matched} ({matched * 100 // len(chunks)}%)")
    print(f"  未匹配:   {unmatched} ({unmatched * 100 // len(chunks)}%)")
    print(f"  输出: {out}")
    return out


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="页码回标 —— 将 chunk JSONL 映射回原始 PDF 页码",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  python add_pages.py chunks.jsonl 原始.pdf -o chunks_paged.jsonl
  python add_pages.py chunks.jsonl 原始.pdf -o chunks_paged.jsonl --anchor-len 80
        """,
    )
    parser.add_argument("jsonl", help="chunk JSONL 文件")
    parser.add_argument("pdf", help="原始 PDF 文件")
    parser.add_argument("-o", "--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--anchor-len", type=int, default=120,
                        help="锚文本长度，默认 120 字符")
    args = parser.parse_args()

    add_pages(args.jsonl, args.pdf, args.output, args.anchor_len)
