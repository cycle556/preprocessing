# -*- coding: utf-8 -*-
"""Phase 2: LLM 辅助的文档结构恢复。

MinerU 将所有标题输出为 H1（全 `#`），此模块用 LLM 推断真实层级，
将平铺标题重写为 H1→H2→H3→H4 的正确嵌套结构。

LLM 只收"目录+摘要"（3000 tokens 以内），成本较低。
无 API key 时回退为启发式规则推断。
"""

import json
import os
import re
from typing import Optional


# 标题匹配
HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*$')


def extract_headings(content: str, snippet_chars: int = 80) -> list[dict]:
    """从文档中提取所有标题及其上下文片段。

    Returns:
        [{"line": 行号(从1开始), "level": 当前层级(1-6),
          "text": 标题文字, "snippet": 标题后前N字符}, ...]
    """
    lines = content.split('\n')
    headings = []
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        text = m.group(2).strip()
        # 取标题后的文本片段（跳过后续标题和空行）
        snippet_parts = []
        for j in range(i + 1, min(len(lines), i + 5)):
            if HEADING_RE.match(lines[j]):
                break
            if lines[j].strip():
                snippet_parts.append(lines[j].strip())
            if sum(len(p) for p in snippet_parts) >= snippet_chars:
                break
        snippet = ' '.join(snippet_parts)[:snippet_chars]
        headings.append({
            "line": i + 1,
            "level": level,
            "text": text,
            "snippet": snippet,
        })
    return headings


def build_structure_prompt(headings: list[dict]) -> str:
    """构建 LLM 结构恢复的 prompt。

    输入：标题列表 + 上下文片段。
    输出格式：JSON 映射。
    """
    heading_list = []
    for h in headings:
        has_table = '表' in h['text'] or 'table' in h['text'].lower()
        has_cont = '續' in h['text'] or 'continued' in h['text'].lower()
        tags = []
        if has_table:
            tags.append('含表格')
        if has_cont:
            tags.append('续表')
        tag_str = f' [{", ".join(tags)}]' if tags else ''
        heading_list.append(
            f"L{h['line']}: {h['text']}{tag_str}\n"
            f"   上下文: {h['snippet'][:80]}"
        )

    return f"""你是一位保险文档结构化专家。以下是从 PDF 自动提取的文档标题列表。
原始提取工具将所有标题都标为 H1 级别，需要你根据语义推断真实的层级关系。

请分析每个标题在文档中的逻辑层级，输出 JSON 映射。

规则：
1. H1: 文档标题（通常只有 1 个，位于文档最开头）
2. H2: 大主题/大板块（如"产品概述"、"保障详情"、"法律条款"）
3. H3: 主题下的子话题（如"严重疾病保障"、"癌症保险赔偿"）
4. H4: 子话题下的细分（如具体的表格标题、备注、注）
5. 如果标题含"(續)"/"(cont'd)"，它是上一个同名标题的延续，层级与上一个相同
6. 标记为"含表格"的标题通常是某个章节下的参考数据

=== 标题列表 ===

{chr(10).join(heading_list)}

=== 输出格式 ===
只输出 JSON，不要其他文字。格式：
{{"L1": {{"level": 1, "parent": null}}, "L3": {{"level": 2, "parent": "L1"}}, ...}}
parent 是该标题所属的父标题行号（null 表示顶层 H1）。
每个标题行号都必须有映射。"""


def _repair_json(text: str) -> str:
    """修复 LLM 常见 JSON 格式错误。"""
    # 去掉 markdown 代码块标记
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    # 修复缺失的逗号：换行前的 } 或 " 后面缺逗号
    text = re.sub(r'([}\]])\s*\n\s*"', r'\1,\n"', text)
    # 修复对象末尾多余逗号："level": 2,} → "level": 2}
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    return text


def parse_structure_response(response: str, heading_count: int) -> dict[str, dict]:
    """解析 LLM 返回的 JSON 结构映射，带修复重试。"""
    json_match = re.search(r'\{[\s\S]*\}', response)
    if not json_match:
        raise ValueError(f"LLM 响应中未找到 JSON: {response[:200]}...")

    raw = json_match.group(0)
    for attempt in range(2):
        try:
            mapping = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            if attempt == 0:
                raw = _repair_json(raw)
            else:
                raise ValueError(f"JSON 解析失败(已尝试修复): {e}\n响应: {response[:500]}...")

    if len(mapping) < heading_count * 0.8:
        print(f"  [warn] LLM 只覆盖了 {len(mapping)}/{heading_count} 个标题")

    return mapping


def apply_structure(content: str, mapping: dict[str, dict]) -> str:
    """将层级映射应用到文档，重写标题的 # 数量。

    mapping: {"L<line_number>": {"level": 1-4, "parent": "L<line>" or null}, ...}
    """
    lines = content.split('\n')
    result = []
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if m:
            key = f'L{i + 1}'
            if key in mapping:
                new_level = int(mapping[key].get('level', 1))
                text = m.group(2).strip()
                result.append('#' * new_level + ' ' + text)
            else:
                result.append(line)
        else:
            result.append(line)
    return '\n'.join(result)


# --- 启发式回退（无需 LLM） ---

def _heuristic_level(text: str) -> int:
    """基于关键词推断标题层级（无 LLM 时的回退方案）。"""
    # H4: 脚注、备注行
    h4_patterns = [
        r'^[\(（]?\d+[\)）]',           # "(1) xxx"
        r'^備註$', r'^備註[：:]',        # "備註"
        r'^注[：:]', r'^Note\d*\b',     # "注："、"Note1"
    ]
    # H3: 子话题、表格标题（带冒号）
    h3_patterns = [
        r'^表[一二三四五六七八九十]',       # "表一：xxx"
        r'^說明例子', r'^例子\d*\b', r'^Example\b',
        r'資料一覽表',
        r'保障知多點',
        r'^適用於',
        r'^紅利',
        r'^不保事項',
    ]
    # H2: 大板块（更明确的主题词）
    h2_patterns = [
        r'^為.*而設',
        r'身故保險賠償',
        r'^終身財富', r'^財富',
        r'^多重賠償', r'^持續.*保險賠償',
        r'^全期保障', r'^全程守護',
        r'^索償', r'^開支', r'^投資', r'^資產',
        r'^保費', r'^終止', r'^冷靜期',
        r'^關於', r'^了解',
        r'^嚴重認知障礙', r'^嚴重疾病多重',
        r'^增值權益', r'^延長寬限期',
        r'^未雨網繆', r'^核保',
    ]

    for pat in h4_patterns:
        if re.search(pat, text):
            return 4
    for pat in h3_patterns:
        if re.search(pat, text):
            return 3
    for pat in h2_patterns:
        if re.search(pat, text):
            return 2
    return 2  # default H2


def heuristic_recover(content: str) -> str:
    """无 LLM 时的启发式层级恢复。
    第一个标题 → H1，其余用关键词规则推断 H2-H4。
    """
    lines = content.split('\n')
    result = []
    seen_first = False
    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            text = m.group(2).strip()
            if not seen_first:
                level = 1  # 第一个标题是文档标题
                seen_first = True
            else:
                level = _heuristic_level(text)
            result.append('#' * level + ' ' + text)
        else:
            result.append(line)
    return '\n'.join(result)


# --- LLM 调用 ---

def _call_llm(prompt: str, provider: str = "anthropic") -> str:
    """调用 LLM 进行结构恢复。"""
    if provider == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("未设置 ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)
        model = os.environ.get("LLM_MODEL", "claude-haiku-4-5")
        resp = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4096,
        )
        return resp.content[0].text
    else:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("未设置 OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = openai.OpenAI(**kwargs)
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4096,
        )
        return resp.choices[0].message.content


# --- 主函数 ---

def recover_structure(content: str, use_llm: bool = True,
                      provider: str = "anthropic",
                      snippet_chars: int = 80) -> str:
    """恢复文档的标题层级结构。

    Args:
        content: 清洗后的 MD（全 H1）
        use_llm: True=LLM 推断, False=启发式回退
        provider:  "openai"
        snippet_chars: 每个标题后取多少字符作为 LLM 上下文

    Returns:
        层级化后的 MD 文本
    """
    headings = extract_headings(content, snippet_chars)
    if not headings:
        print("  [recover] 未检测到标题，跳过")
        return content
    print(f"  [recover] 检测到 {len(headings)} 个标题")

    if use_llm:
        try:
            # 大文档分批处理（每批 ≤ 80 个标题），减少 LLM JSON 出错概率
            BATCH_SIZE = 80
            if len(headings) <= BATCH_SIZE:
                prompt = build_structure_prompt(headings)
                print(f"  [recover] prompt: {len(prompt)} 字符 → LLM...")
                response = _call_llm(prompt, provider)
                mapping = parse_structure_response(response, len(headings))
                print(f"  [recover] LLM 返回 {len(mapping)} 个层级映射")
            else:
                print(f"  [recover] {len(headings)} 个标题，分 { (len(headings) + BATCH_SIZE - 1) // BATCH_SIZE} 批处理...")
                mapping = {}
                for batch_idx in range(0, len(headings), BATCH_SIZE):
                    batch = headings[batch_idx:batch_idx + BATCH_SIZE]
                    prompt = build_structure_prompt(batch)
                    bn = batch_idx // BATCH_SIZE + 1
                    total_bn = (len(headings) + BATCH_SIZE - 1) // BATCH_SIZE
                    print(f"    批次 {bn}/{total_bn}: {len(batch)} 个标题, {len(prompt)} 字符...")
                    response = _call_llm(prompt, provider)
                    batch_map = parse_structure_response(response, len(batch))
                    mapping.update(batch_map)
                    print(f"    批次 {bn}: {len(batch_map)} 个映射")
                print(f"  [recover] LLM 总计返回 {len(mapping)} 个层级映射")

            return apply_structure(content, mapping)
        except Exception as e:
            print(f"  [recover] LLM 失败 ({e})，回退到启发式方法")

    print("  [recover] 使用启发式回退")
    return heuristic_recover(content)
