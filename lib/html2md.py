# -*- coding: utf-8 -*-
"""HTML 表格 → Markdown 表格转换。
从 agentic_clean.py 提取并改进：修复了空单元格被过滤的 bug。
支持 colspan/rowspan 展开、复合表头合并、脚注提取。
"""

import re
from bs4 import BeautifulSoup


_HTML_TABLE_RE = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)


# --- 单元格 / 行 / 网格工具 ---

def _dedup_adjacent(s: str) -> str:
    """合并相邻的重复词：'標準 標準' → '標準'。"""
    return re.sub(r'\b(\S+)\s+\1\b', r'\1', s)


def _strip_cell(text: str) -> str:
    """清洗单个表格单元格的文本内容。"""
    text = ' '.join(text.split())
    text = text.replace('|', '\\|')
    text = _dedup_adjacent(text)
    return text


def _row_is_empty(row: list[str]) -> bool:
    """判断表格行是否全为空。"""
    return all(cell.strip() == '' for cell in row)


def _normalize_grid_width(grid: list[list[str]]) -> list[list[str]]:
    """将所有行补齐到相同的列数。"""
    max_cols = max(len(row) for row in grid) if grid else 0
    for row in grid:
        row.extend([''] * (max_cols - len(row)))
    return grid


def _merge_two_row_header(grid: list[list[str]]) -> list[list[str]]:
    """当第 0 行和第 1 行构成一个逻辑表头时，将它们合并为复合表头。"""
    if len(grid) < 2:
        return grid
    r0, r1 = grid[0], grid[1]
    if len(r0) != len(r1):
        return grid

    has_colspan = any(
        r0[i].strip() and r0[i] == r0[i + 1]
        for i in range(len(r0) - 1)
    )
    if not has_colspan:
        return grid

    has_rowspan = any(
        a.strip() and a.strip() == b.strip()
        for a, b in zip(r0, r1)
    )
    if not has_rowspan:
        return grid

    merged = []
    for a, b in zip(r0, r1):
        a_s, b_s = a.strip(), b.strip()
        if a_s == b_s or not b_s:
            merged.append(a_s)
        elif not a_s:
            merged.append(b_s)
        else:
            merged.append(f'{a_s} {b_s}')
    return [merged] + grid[2:]


def _extract_colspan_footnote(grid: list[list[str]]):
    """弹出全宽 colspan 脚注行。"""
    if len(grid) < 2:
        return grid, None
    last = grid[-1]
    non_empty = [c.strip() for c in last if c.strip()]
    if non_empty and len(set(non_empty)) == 1 and len(non_empty) == len(last):
        return grid[:-1], non_empty[0]
    return grid, None


def _grid_to_md(grid: list[list[str]], footnote: str | None = None) -> str:
    """将二维网格渲染为 Markdown 表格。
    改进：保留空单元格 `| |`，不再过滤掉。
    """
    if not grid:
        return footnote or ''

    header = grid[0]
    body = grid[1:]

    lines = [
        '| ' + ' | '.join(header) + ' |',
        '| ' + ' | '.join(['---'] * len(header)) + ' |',
    ]
    for row in body:
        lines.append('| ' + ' | '.join(row) + ' |')

    if footnote:
        lines.append('')
        lines.append(footnote)
    return '\n'.join(lines)


# --- 主转换函数 ---

def html_table_to_md(html: str) -> str:
    """将单个 <table> 元素（含 rowspan/colspan）转换为 Markdown。

    算法：展开 colspan 为相邻同文本单元格，展开 rowspan 为后续行的
    相同位置单元格。然后合并复合表头、提取脚注、渲染为 MD 表格。
    """
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    if not table:
        return ''

    grid: list[list[str]] = []
    occupied: dict[tuple[int, int], str] = {}

    for r, tr in enumerate(table.find_all('tr')):
        row: list[str] = []
        c = 0

        for cell in tr.find_all(['th', 'td']):
            while (r, c) in occupied:
                row.append(occupied.pop((r, c)))
                c += 1

            text = _strip_cell(cell.get_text(' ', strip=True))

            try:
                rs = int(cell.get('rowspan', 1) or 1)
                cs = int(cell.get('colspan', 1) or 1)
            except (ValueError, TypeError):
                rs = cs = 1

            for col_offset in range(cs):
                row.append(text)
                for dr in range(1, rs):
                    occupied[(r + dr, c + col_offset)] = text
                c += 1

        while (r, c) in occupied:
            row.append(occupied.pop((r, c)))
            c += 1

        grid.append(row)

    if not grid:
        return ''

    grid = _normalize_grid_width(grid)
    grid = [row for row in grid if not _row_is_empty(row)]
    if not grid:
        return ''

    grid = _merge_two_row_header(grid)
    grid, footnote = _extract_colspan_footnote(grid)

    return _grid_to_md(grid, footnote)


def convert_html_tables(content: str) -> str:
    """查找并转换内容中所有的 <table> 块为 Markdown 表格。"""
    return _HTML_TABLE_RE.sub(
        lambda m: '\n\n' + html_table_to_md(m.group(0)) + '\n\n',
        content,
    )
