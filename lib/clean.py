# -*- coding: utf-8 -*-
"""Phase 1: 确定性清洗。

- 复用 agentic_clean.py 的 pre_clean / post_clean
- 新增：表格连续体拼接（检测"(續)"标记，合并拆分表）
- 新增：重复表格去重（embedding 相似度检测）
"""

import re
from lib.html2md import convert_html_tables
from lib.latex_fix import fix_latex_artifacts


# --- 正则模式 ---
_EMPTY_HEADING_RE = re.compile(r'^#\s*$', re.MULTILINE)
_IMAGE_RE = re.compile(r'!\[[^\]]*\]\([^)]+\)\s*')
_HTML_BR_RE = re.compile(r'<br\s*/?>', re.IGNORECASE)
_DETAILS_RE = re.compile(
    r'<details>\s*<summary>([^<]*)</summary>\s*(.*?)\s*</details>',
    re.DOTALL | re.IGNORECASE,
)


# --- <details> 解包，把折叠的内容强行展开，变成普通文字 ---

def _extract_details_block(m: re.Match) -> str:
    summary = m.group(1).strip()
    body = m.group(2).strip()
    if not body:
        return f'**{summary}**'
    return f'**{summary}**\n\n{body}'


def unwrap_details_blocks(content: str) -> str:
    return _DETAILS_RE.sub(_extract_details_block, content)


# --- 项目符号规范化 ---

def normalize_bullets(content: str) -> str:
    content = re.sub(r'^[ \t]*[·•]\s*', '- ', content, flags=re.MULTILINE)
    content = re.sub(r'^[ \t]*\.\s+(?=\S)', '- ', content, flags=re.MULTILINE)
    return content


# --- 表格连续体拼接 ---

# 匹配标题含"(續)"、"(cont'd)"、"(continued)" 的模式
_CONTINUED_HEADING_RE = re.compile(
    r'^(#{1,6})\s+(.+)\((續|cont\'?d|continued)\)\s*$',
    re.MULTILINE | re.IGNORECASE,
)


def _has_continued_heading_before(lines: list[str], html_idx: int) -> bool:
    """检查 HTML 表格前 5 行内是否有 '(續)' 标题。"""
    start = max(0, html_idx - 5)
    for i in range(start, html_idx):
        if _CONTINUED_HEADING_RE.search(lines[i]):
            return True
    return False


def stitch_continued_tables(content: str) -> str:
    """将被页面切开的 HTML 连续表格拼接为一个逻辑表格。

    检测模式：标题含"(續)" → 下一个 <table> 拼接到前一个同类表格后面。
    拼接前去掉各片段各自的 <table> 和 </table> 标签，用单个 <table> 包裹。
    """
    # 找到所有 <table>...</table> 块
    tables = list(re.finditer(r'<table[^>]*>.*?</table>', content, re.DOTALL | re.IGNORECASE))
    if len(tables) <= 1:
        return content

    lines = content.split('\n')

    # 找出哪些 table 是续表
    is_continuation = [False] * len(tables)
    for i in range(1, len(tables)):
        # 找到此 table 前的行号
        pre_text = content[:tables[i].start()]
        pre_lines = pre_text.split('\n')
        table_line = len(pre_lines) - 1
        if _has_continued_heading_before(lines, table_line):
            is_continuation[i] = True

    if not any(is_continuation):
        return content

    # 分组：连续的 False+True+True... → 拼接为一个 table
    groups = []
    i = 0
    while i < len(tables):
        group = [tables[i]]
        j = i + 1
        while j < len(tables) and is_continuation[j]:
            group.append(tables[j])
            j += 1
        groups.append(group)
        i = j

    # 拼接每组
    result = []
    last_end = 0
    for group in groups:
        if len(group) == 1:
            # 单表，保持原样
            result.append(content[last_end:group[0].end()])
        else:
            # 多表拼接
            result.append(content[last_end:group[0].start()])
            # 合并所有 table 的内容
            merged_inner = []
            for t in group:
                html = t.group(0)
                # 去掉外层 <table...> 和 </table>
                inner = re.sub(r'^<table[^>]*>', '', html, flags=re.IGNORECASE)
                inner = re.sub(r'</table>\s*$', '', inner, flags=re.IGNORECASE)
                merged_inner.append(inner)
            result.append('<table>\n' + '\n'.join(merged_inner) + '\n</table>')
        last_end = group[-1].end()

    result.append(content[last_end:])
    return ''.join(result)


# --- 重复表格去重 ---

def dedup_tables(md_content: str) -> str:
    """检测并移除非首发的重复 Markdown 表格。

    策略：提取所有 MD 表格，对表格的"结构指纹"（列数+前两行文本）
    做哈希比较。完全相同的表格只保留第一次出现。
    """
    tables = list(re.finditer(
        r'((?:^\|.+\|\n)+)',
        md_content,
        re.MULTILINE,
    ))

    if len(tables) <= 1:
        return md_content

    seen_fingerprints = set()
    to_remove = []

    for t in tables:
        text = t.group(0)
        lines = [l for l in text.split('\n') if l.strip() and '---' not in l]
        # 指纹：列数 + 前 2 行内容
        ncols = text.count('|') // max(1, len(lines))
        fingerprint = f'{ncols}|{"|".join(lines[:2])}'
        if fingerprint in seen_fingerprints:
            to_remove.append((t.start(), t.end()))
        else:
            seen_fingerprints.add(fingerprint)

    # 从后往前删除，避免偏移
    result = md_content
    for start, end in reversed(to_remove):
        result = result[:start] + result[end:]

    return result


# --- 主预清洗 ---

def pre_clean(content: str, stitch_tables: bool = True,
              dedup: bool = True) -> str:
    """阶段 1：确定性清洗。

    Args:
        content: MinerU 原始 MD 文本
        stitch_tables: 是否拼接被页面切开的连续表格
        dedup: 是否移除非首发的重复表格
    """
    # 0. 拼接连续表格（在 HTML→MD 转换前做）
    if stitch_tables:
        content = stitch_continued_tables(content)

    # 1. 移除图片
    content = _IMAGE_RE.sub('', content)

    # 2. 移除空标题
    content = _EMPTY_HEADING_RE.sub('', content)

    # 3. 修复 LaTeX 伪影
    content = fix_latex_artifacts(content)

    # 4. 将 HTML 表格转换为 Markdown 表格
    content = convert_html_tables(content)

    # 5. 解包 <details> 块
    content = unwrap_details_blocks(content)

    # 6. 移除残留的 <br> 标签
    content = _HTML_BR_RE.sub('', content)

    # 7. 规范化项目符号
    content = normalize_bullets(content)

    # 8. 移除游离行（1-3 个大写 ASCII 字符单独成行）
    content = re.sub(r'\n\s*[A-Z]{1,3}\s*\n', '\n', content)

    # 9. 去重表格
    if dedup:
        content = dedup_tables(content)

    # 10. 去除行尾空白
    content = '\n'.join(line.rstrip() for line in content.split('\n'))

    # 11. 压缩多余空行
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content.strip()


def post_clean(content: str) -> str:
    """阶段 3 确定性后处理。"""
    content = content.replace('\r\n', '\n').replace('\r', '\n')

    # 移除空表格行
    content = re.sub(r'^\|\s*\|\s*$', '', content, flags=re.MULTILINE)

    # 移除孤立的表格分隔行
    content = re.sub(
        r'\n\|\s*(?::?-{2,}:?\s*\|\s*)+(?::?-{2,}:?)\s*\|?\s*\n(?![|\-])',
        '\n',
        content,
    )

    content = re.sub(r'\n{3,}', '\n\n', content)
    content = '\n'.join(l.rstrip() for l in content.split('\n'))
    return content.strip() + '\n'


def clean_md(md_path: str, stitch_tables: bool = True,
             dedup: bool = True) -> str:
    """完整的 Phase 1 清洗。读取文件 → pre_clean → post_clean → 返回文本。"""
    from pathlib import Path
    content = Path(md_path).read_text(encoding='utf-8')
    print(f"  [clean] 输入: {len(content)} 字符")
    content = pre_clean(content, stitch_tables=stitch_tables, dedup=dedup)
    print(f"  [clean] pre_clean 后: {len(content)} 字符")
    content = post_clean(content)
    print(f"  [clean] post_clean 后: {len(content)} 字符")
    return content
