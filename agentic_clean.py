# -*- coding: utf-8 -*-
"""
Agentic Markdown Cleaner —— 基于 LLM 的 MinerU 提取 Markdown 清洗工具。

描述（agentic RAG）：
    不再为每种文档类型硬编码正则规则，而是让 LLM 充当"清洗 agent"，
    读取杂乱的 Markdown 并输出干净、结构化的 Markdown。
    LLM 能理解上下文,以正则做不到的方式区分信号与噪声。

    流水线采用混合架构：
      阶段 1（确定性）：快速的正则/BS4 修复，处理明显、安全的模式：
          - 移除图片、游离字符、空标题
          - 将 HTML 表格转换为 Markdown 表格（BeautifulSoup）
          - 修复常见的 LaTeX 伪影
          - 规范化项目符号与空白字符
      阶段 2（agentic / LLM）：分块 → LLM 智能清洗 → 返回干净的 MD。
          处理复杂部分：上下文感知的噪声移除、复杂表格重组、
          语义格式修复。
      阶段 3（确定性）：最终规范化（空白字符、验证）。

    LLM 步骤是可选的。当没有 API 密钥时，确定性路径仍能处理
    约 80% 的常见 MinerU 问题（图片、表格、LaTeX、项目符号）。

用法：
    python agentic_clean.py input.md [output.md]        # 默认使用 LLM
    python agentic_clean.py input.md --no-llm           # 仅确定性清洗

    

环境变量（示例）：
    OPENAI_API_KEY    — OpenAI 或兼容端点
    OPENAI_BASE_URL   — 可选，用于自定义端点（Ollama、vLLM 等）
    LLM_MODEL         — 覆盖默认模型

    也可以在脚本同目录下创建 .env 文件，脚本启动时会自动加载：
        # .env 示例
        OPENAI_API_KEY=sk-xxxxxxxx
        # OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama 等本地模型
        # LLM_MODEL=gpt-4o
"""

import re
import os
import sys
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup


# ============================================================
# 自动加载 .env 文件
# ============================================================

def _load_dotenv() -> None:
    """从脚本所在目录及当前工作目录加载 .env 文件（不依赖 python-dotenv）。

    仅设置尚未在系统环境变量中定义的键（系统环境变量优先级更高）。
    """
    candidates = [
        Path(__file__).resolve().parent / '.env',   # 脚本同目录
        Path.cwd() / '.env',                         # 当前工作目录
    ]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        try:
            for line in env_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, val = line.partition('=')
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and key not in os.environ:      # 不覆盖已有的系统环境变量
                    os.environ[key] = val
        except Exception:
            pass

_load_dotenv()


# ============================================================
# 阶段 1：确定性预清洗
# ============================================================

# --- 正则模式 ---
_EMPTY_HEADING_RE = re.compile(r'^#\s*$', re.MULTILINE)                            #清理空标题
_IMAGE_RE = re.compile(r'!\[[^\]]*\]\([^)]+\)\s*')                                 #移除图片
_HTML_BR_RE = re.compile(r'<br\s*/?>', re.IGNORECASE)                              #替换换行
_HTML_TABLE_RE = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE) #过滤 HTML 表格


def _dedup_adjacent(s: str) -> str:
    """合并相邻的重复词，例如'標準 標準' → '標準'。"""
    return re.sub(r'\b(\S+)\s+\1\b', r'\1', s)


def _strip_cell(text: str) -> str:
    """清洗单个表格单元格的文本内容。"""
    text = ' '.join(text.split())     # 压缩所有空白字符
    text = text.replace('|', '\\|')   # 转义管道符
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
    """当第 0 行和第 1 行构成一个逻辑表头时，将它们合并为复合表头。

    启发式规则：第 0 行存在连续的相同单元格（colspan 信号），
    且至少有一列中第 0 行与第 1 行的文本相同（rowspan 信号）。
    这可以防止错误合并真正的数据行。
    """
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
    """如果最后一行有 N 个相同且非空的单元格（全宽 colspan），
    则将其弹出并作为脚注文本返回。"""
    if len(grid) < 2:
        return grid, None
    last = grid[-1]
    non_empty = [c.strip() for c in last if c.strip()]
    if non_empty and len(set(non_empty)) == 1 and len(non_empty) == len(last):
        return grid[:-1], non_empty[0]
    return grid, None


def _grid_to_md(grid: list[list[str]], footnote: str | None = None) -> str:
    """将二维网格渲染为 Markdown 表格。"""
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


def html_table_to_md(html: str) -> str:
    """将单个 <table> 元素（含 rowspan/colspan）转换为 Markdown。

    将跨行/跨列单元格展开为平面网格，合并两行复合表头，
    并提取全宽脚注行。
    """
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    if not table:
        return ''

    grid: list[list[str]] = []
    occupied: dict[tuple[int, int], str] = {}  # (行, 列) → 由 rowspan 携带的文本

    for r, tr in enumerate(table.find_all('tr')):
        row: list[str] = []
        c = 0

        for cell in tr.find_all(['th', 'td']):
            # 填充由上一行的 rowspan 占用的位置
            while (r, c) in occupied:
                row.append(occupied.pop((r, c)))
                c += 1

            text = _strip_cell(cell.get_text(' ', strip=True))

            try:
                rs = int(cell.get('rowspan', 1) or 1)
                cs = int(cell.get('colspan', 1) or 1)
            except (ValueError, TypeError):
                rs = cs = 1

            # 水平展开 colspan，并为后续行注册 rowspan
            for col_offset in range(cs):
                row.append(text)
                for dr in range(1, rs):
                    occupied[(r + dr, c + col_offset)] = text
                c += 1

        # 填充本行中由 rowspan 占用的剩余位置
        while (r, c) in occupied:
            row.append(occupied.pop((r, c)))
            c += 1

        grid.append(row)

    if not grid:
        return ''

    grid = _normalize_grid_width(grid)

    # 移除完全为空的行
    grid = [row for row in grid if not _row_is_empty(row)]
    if not grid:
        return ''

    # 后处理：合并复合表头、提取脚注
    grid = _merge_two_row_header(grid)
    grid, footnote = _extract_colspan_footnote(grid)

    return _grid_to_md(grid, footnote)


def convert_html_tables(content: str) -> str:
    """查找并转换内容中所有的 <table> 块为 Markdown 表格。"""
    return _HTML_TABLE_RE.sub(
        lambda m: '\n\n' + html_table_to_md(m.group(0)) + '\n\n',
        content,
    )


# --- LaTeX 伪影正则模式，清理公式、提取数字 ---

_SUP_TAG_RE = re.compile(
    r'\$\s*\{\s*\}\s*\^\s*\{\s*([<>])\s*\}\s*([\d\s]+?)\s*\$'
)
_TT_TAG_RE = re.compile(
    r'\$\s*\{\s*\\tt\s*([<>])\s*\}\s*([\d\s]+?)'
    r'\s*(?:\{\s*\\tt\s*\)\s*\})?\s*\$'
)
# $1 0 0 \%$ 或 $100\%$
_DOLLAR_WRAPPED_RE = re.compile(
    r'\$\s*([\d\s.,]+?)\s*(\\%)?\s*\$'
)
# \textcircled { \textbf { 5 } } \% → 5%
_CIRCLED_RE = re.compile(
    r'\\textcircled\s*\{\s*\\textbf\s*\{\s*(\d+)\s*\}\s*\}\s*\\%'
)
# $^ 4 5 \%$ → 45%（上标百分号）
_SUP_PCT_RE = re.compile(
    r'\$\s*\^\s*([\d\s]+?)\s*\\%\s*\$'
)
# $ . 5 0 \%$ → 0.50%
_DOT_PCT_RE = re.compile(
    r'\$\s*\.\s*(\d[\d\s]*?)\s*\\?\s*%\s*\$'
)


def fix_latex_artifacts(content: str) -> str:
    """修复常见的 MinerU LaTeX 渲染伪影。"""

    # --- 具体的已知模式 ---

    # ${ }^{>}3 0$ → >30
    content = _SUP_TAG_RE.sub(
        lambda m: f'{m.group(1)}{re.sub(r"\s+", "", m.group(2))}',
        content,
    )
    # ${ \tt > } 3 5 { \tt ) }$ → >35)
    content = _TT_TAG_RE.sub(
        lambda m: (
            f'{m.group(1)}{re.sub(r"\s+", "", m.group(2))}'
            + (')' if '\\tt )' in m.group(0) or '\\tt  )' in m.group(0) else '')
        ),
        content,
    )

    # \textcircled { \textbf { 5 } } \% → 5%
    content = _CIRCLED_RE.sub(r'\1%', content)

    # $\left( { \mathfrak { g } } \right)$ → (g)
    content = re.sub(
        r'\$\s*\\left\s*\(\s*\{\s*\\mathfrak\s*\{\s*(\w+)\s*\}\s*\}\s*\\right\s*\)\s*\$',
        r'(\1)',
        content,
    )

    # $\ddagger$ → ‡
    content = re.sub(r'\$\s*\\ddagger\s*\$', '‡', content)

    # $^ +$ → ⁺
    content = re.sub(r'\$\s*\^\s*\+\s*\$', '⁺', content)

    # $^ \circ$ → 移除
    content = re.sub(r'\$\s*\^\s*\\circ\s*\$', '', content)

    # $数字%$ → 数字%（$...$ 中包裹的纯数字加百分号）
    content = re.sub(
        r'\$\s*(\d+(?:\.\d+)?)\s*%\s*\$',
        r'\1%',
        content,
    )

    # $^ 4 5 \%$ → 45%
    content = _SUP_PCT_RE.sub(
        lambda m: f'{re.sub(r"\\s+", "", m.group(1))}%',
        content,
    )
    # $ . 5 0 \%$ → 0.50%
    content = _DOT_PCT_RE.sub(
        lambda m: f'0.{re.sub(r"\\s+", "", m.group(1))}%',
        content,
    )

    # --- 通用的短数字片段 $...$ 展开 ---

    def _unwrap_dollar(m):
        inner = m.group(1)
        pct = m.group(2) or ''
        num = re.sub(r'\s+', '', inner)
        if re.fullmatch(r'[\d.,]+', num):
            return f'{num}{"%" if pct else ""}'
        if re.fullmatch(r'[\d\s.,]+', inner) and pct:
            return f'{num}%'
        return m.group(0)

    content = re.sub(
        r'\$\s*([\d\s.,]{1,20})\s*(\\%)?\s*\$',
        _unwrap_dollar,
        content,
    )

    return content


# --- 项目符号规范化 ---
#处理行首的三类列表符号，统一成标准 - ，不破坏正文标点
def normalize_bullets(content: str) -> str:
    """规范化项目符号标记：· • . → -"""
    # 行首的 · • → -
    content = re.sub(r'^[ \t]*[·•]\s*', '- ', content, flags=re.MULTILINE)
    # 行首的 . 后跟空格 + 文本 → -
    content = re.sub(r'^[ \t]*\.\s+(?=\S)', '- ', content, flags=re.MULTILINE)
    return content


# --- 主预清洗函数 ---

def pre_clean(content: str) -> str:
    """阶段 1：快速的确定性修复，无需 LLM 参与。"""

    # 1. 移除图片（MinerU 输出的图片几乎都是装饰性重复）
    content = _IMAGE_RE.sub('', content)

    # 2. 移除空标题
    content = _EMPTY_HEADING_RE.sub('', content)

    # 3. 修复 LaTeX 伪影
    content = fix_latex_artifacts(content)

    # 4. 将 HTML 表格转换为 Markdown 表格
    content = convert_html_tables(content)

    # 5. 移除残留的 <br> 标签
    content = _HTML_BR_RE.sub('', content)

    # 6. 规范化项目符号
    content = normalize_bullets(content)

    # 7. 移除真正的游离行（行内只有 1-3 个大写 ASCII 字符）
    #    必须被空行包围，以避免误删
    content = re.sub(
        r'\n\s*[A-Z]{1,3}\s*\n',
        '\n',
        content,
    )

    # 8. 去除行尾空白
    content = '\n'.join(line.rstrip() for line in content.split('\n'))

    # 9. 将 3 个及以上的连续空行压缩为 2 个
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content.strip()


# ============================================================
# 阶段 2：基于 LLM 的智能清洗（"agentic" 部分）
# ============================================================

CLEANING_SYSTEM_PROMPT = """\
You are a precise document cleaner for a RAG pipeline. You receive messy Markdown
auto-extracted from PDFs by MinerU. Output ONLY the cleaned Markdown.

## RULES

### 1. REMOVE REMAINING NOISE
- Remove stray characters / gibberish on otherwise empty lines.
- Remove OCR artifacts that are clearly not content (random symbols, broken fragments).
- BUT keep: section labels, table captions, clause numbers (e.g. "2.5", "2,8"),
  product names, short Chinese phrases, and all numeric data.

### 2. FIX FORMATTING ARTIFACTS
- `$...$` wrapping a plain number or percent → unwrap (e.g. `$5\\%$` → `5%`).
- Split numbers like `1 0 0` inside dollar signs → `100`.
- `\\textcircled` / `\\textbf` fragments → extract the number/letter inside.
- Remove empty LaTeX spans like `${}$` or `${ }$`.
- If LaTeX IS a genuine formula (fractions, sums, integrals), keep it as-is.

### 3. PRESERVE EXACTLY
- ALL Chinese text (traditional & simplified) — verbatim, no translation.
- Heading hierarchy — do not re-level, re-title, or merge headings.
- Numbered lists, bullet lists, paragraph structure.
- Superscript references (², ¹, ²·⁸) and footnote-style numbers in text.
- ALL table content — every cell's text must survive.

### 4. STRUCTURE
- Each heading preceded by a blank line.
- Paragraphs separated by exactly one blank line.
- Tables have blank lines before and after.
- No trailing whitespace on any line.

### 5. WHAT NOT TO DO
- Do NOT translate, rewrite, summarize, or rephrase content.
- Do NOT add commentary, analysis, or metadata.
- Do NOT invent new headings, merge sections, or restructure the document.
- Do NOT add preamble like "Here's the cleaned version:".
- Output raw Markdown only, starting from the first `#` or text line.\
"""


def _chunk_at_boundary(content: str, max_chars: int = 7000) -> list[str]:
    """在标题/空行边界处将内容切分为适合 LLM 处理的分块。
    不粗暴按字符硬切，优先在标题、空行处分割，保证语义完整性。"""
    if len(content) <= max_chars:
        return [content]

    chunks = []
    lines = content.split('\n')
    buf, buf_len = [], 0

    for line in lines:
        line_len = len(line) + 1
        if buf_len + line_len > max_chars and buf:
            # 回退寻找干净的分割点
            split_at = len(buf)
            for i in range(len(buf) - 1, max(0, len(buf) - 30), -1):
                if buf[i].strip() == '' or buf[i].startswith('#'):
                    split_at = i + 1
                    break
            chunks.append('\n'.join(buf[:split_at]).strip())
            buf = buf[split_at:]
            buf_len = sum(len(l) + 1 for l in buf)

        buf.append(line)
        buf_len += line_len

    if buf:
        chunks.append('\n'.join(buf).strip())

    return chunks


def _make_llm_caller():
    """返回 (调用函数, 模型名称)。"""
    import openai
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if not api_key:
        raise ValueError("未设置 OPENAI_API_KEY 环境变量")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)
    model = os.environ.get("LLM_MODEL", "gpt-4o")

    def call(chunk: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLEANING_SYSTEM_PROMPT},
                {"role": "user", "content": chunk},
            ],
            temperature=0.1,
            max_tokens=16000,
        )
        return resp.choices[0].message.content

    return call, model


def llm_clean(content: str) -> str:
    """通过 LLM 清洗内容，必要时进行分块。"""
    llm_call, model = _make_llm_caller()
    chunks = _chunk_at_boundary(content)

    print(f"  模型: {model}  |  分块数: {len(chunks)}")

    if len(chunks) == 1:
        return llm_call(chunks[0])

    cleaned = []
    for i, chunk in enumerate(chunks):
        print(f"  分块 {i+1}/{len(chunks)} ({len(chunk)} 字符) ...")
        result = llm_call(chunk)
        cleaned.append(result)

    return '\n\n'.join(cleaned)


# ============================================================
# 阶段 3：确定性后处理
# ============================================================

def post_clean(content: str) -> str:
    """最终的确定性规范化。"""

    content = content.replace('\r\n', '\n').replace('\r', '\n')

    # 移除空表格行（所有单元格均为空）
    content = re.sub(r'^\|\s*\|\s*$', '', content, flags=re.MULTILINE)

    # 移除孤立的表格分隔行（前后无文本）
    content = re.sub(
        r'\n\|\s*(?::?-{2,}:?\s*\|\s*)+(?::?-{2,}:?)\s*\|?\s*\n(?![|\-])',
        '\n',
        content,
    )

    # 将 3 个及以上的连续空行压缩为 2 个
    content = re.sub(r'\n{3,}', '\n\n', content)

    # 去除行尾空白
    content = '\n'.join(l.rstrip() for l in content.split('\n'))

    return content.strip() + '\n'


# ============================================================
# 流水线主函数
# ============================================================

def clean(
    md_path: str | Path,
    out_path: str | Path | None = None,
    use_llm: bool = True,
) -> Path:
    """清洗 MinerU 提取的 Markdown 文件。

    参数:
        md_path: 杂乱的 .md 文件路径。
        out_path: 输出路径（默认：与输入同目录，文件名为 <输入>_clean.md）。
        use_llm: 是否启用阶段 2 的 LLM 清洗。若未找到 API 密钥则回退到确定性模式。
    """
    md_path = Path(md_path)
    content = md_path.read_text(encoding='utf-8')
    print(f"输入: {md_path}  ({len(content)} 字符)")

    # --- 阶段 1 ---
    print("阶段 1: 预清洗（确定性）")
    content = pre_clean(content)
    print(f"  → 预清洗后 {len(content)} 字符")

    # --- 阶段 2 ---
    if use_llm:
        print("阶段 2: LLM 清洗（agentic）")
        try:
            content = llm_clean(content)
            print(f"  → LLM 清洗后 {len(content)} 字符")
        except ValueError as e:
            print(f"  [警告] {e}")
            print("  继续使用仅确定性模式的结果。")
        except Exception as e:
            print(f"  [错误] LLM 清洗失败: {e}")
            print("  继续使用仅确定性模式的结果。")

    # --- 阶段 3 ---
    print("阶段 3: 后清洗（确定性）")
    content = post_clean(content)
    print(f"  → 最终 {len(content)} 字符")

    # --- 写入输出 ---
    if out_path:
        out = Path(out_path)
    else:
        out = md_path.with_name(md_path.stem + '_clean.md')
    out.write_text(content, encoding='utf-8')
    print(f"输出: {out}")
    return out


# ============================================================
# 命令行接口
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Agentic Markdown Cleaner —— 基于 LLM 的 MinerU 输出清洗工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  python agentic_clean.py messy.md                    # 使用默认 LLM 清洗
  python agentic_clean.py messy.md --no-llm           # 仅确定性清洗
  python agentic_clean.py messy.md -o clean.md
        """,
    )
    parser.add_argument("input", help="MinerU 输出的 .md 文件")
    parser.add_argument("output", nargs="?", default=None, help="输出 .md 文件（位置参数）")
    parser.add_argument("-o", "--output", dest="output_opt", default=None,
                        help="输出 .md 文件（可选，同位置参数）")
    parser.add_argument("--no-llm", action="store_true",
                        help="仅确定性模式（跳过 LLM）")
    args = parser.parse_args()

    out = args.output_opt or args.output
    clean(args.input, out, use_llm=not args.no_llm)
