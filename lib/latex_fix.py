# -*- coding: utf-8 -*-
"""LaTeX 伪影修复 """

import re

# --- 正则模式 ---

_SUP_TAG_RE = re.compile(
    r'\$\s*\{\s*\}\s*\^\s*\{\s*([<>])\s*\}\s*([\d\s]+?)\s*\$'
)
_TT_TAG_RE = re.compile(
    r'\$\s*\{\s*\\tt\s*([<>])\s*\}\s*([\d\s]+?)'
    r'\s*(?:\{\s*\\tt\s*\)\s*\})?\s*\$'
)
_DOLLAR_WRAPPED_RE = re.compile(
    r'\$\s*([\d\s.,]+?)\s*(\\%)?\s*\$'
)
_CIRCLED_RE = re.compile(
    r'\\textcircled\s*\{\s*\\textbf\s*\{\s*(\d+)\s*\}\s*\}\s*\\%'
)
_SUP_PCT_RE = re.compile(
    r'\$\s*\^\s*([\d\s]+?)\s*\\%\s*\$'
)
_DOT_PCT_RE = re.compile(
    r'\$\s*\.\s*(\d[\d\s]*?)\s*\\?\s*%\s*\$'
)


def fix_latex_artifacts(content: str) -> str:
    """修复常见的 MinerU LaTeX 渲染伪影。

    处理顺序：先处理已知的具体模式，再用通用规则展开
    短数字串的 $...$ 包裹。
    """
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

    # $数字%$ → 数字%
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
