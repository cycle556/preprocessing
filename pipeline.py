# -*- coding: utf-8 -*-
"""主编排器 —— 一键运行完整预处理管线。

用法：
    python pipeline.py input.md --pdf original.pdf [--company 安盛] [--out-dir ./output]

管线：
    Phase 1: 确定性清洗 (clean)
    Phase 2: 结构恢复 (recover_structure, 默认 LLM)
    Phase 3: 多层分块 (chunker_v2)
    Phase 4: 页码回标 + 输出 JSONL + XLSX

输出：
    - 带页码的 _paged.jsonl（如果提供了 PDF）
    - 审阅用的 .chunks.xlsx
"""

import argparse
import os
import sys
from pathlib import Path

# 确保 lib 在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 必须在 import config 之前加载 .env（config 在 import 时读取环境变量）
from lib.utils import load_dotenv
load_dotenv()  # lib/.env 优先级最高

from config import (
    MAX_CHARS, OVERLAP_CHARS,
    STRUCTURE_LLM_PROVIDER,
    STRUCTURE_SNIPPET_CHARS,
    OUTPUT_FORMATS,
)
from lib.utils import sanitize_for_id
from lib.clean import clean_md
from lib.recover_structure import recover_structure
from lib.chunker_v2 import (
    chunk_markdown_v2, write_jsonl, write_md, print_summary,
)


def process_file(
    input_path: str,
    out_dir: str | None = None,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP_CHARS,
    use_llm: bool = True,
    llm_provider: str = "",
    source: str | None = None,
    id_prefix: str | None = None,
    pdf_path: str | None = None,
    company: str | None = None,
) -> str:
    """处理单个 MinerU MD → 输出 _paged.jsonl + .chunks.xlsx。"""
    input_path = Path(input_path)
    basename = input_path.stem
    derived = sanitize_for_id(basename)
    src = source or basename
    prefix = id_prefix or derived

    target_dir = Path(out_dir) if out_dir else input_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    provider = llm_provider or STRUCTURE_LLM_PROVIDER

    print(f"\n{'='*60}")
    print(f"处理: {input_path.name}")
    print(f"  company={company or '(未指定)'}  source={src}  prefix={prefix}")
    print(f"  LLM: {provider} / {os.environ.get('LLM_MODEL', 'default')}")
    print(f"{'='*60}")

    # ---- Phase 1: Clean ----
    print("\n[Phase 1] 确定性清洗")
    clean_text = clean_md(str(input_path), stitch_tables=True, dedup=True)

    # ---- Phase 2: Structure Recovery ----
    print("\n[Phase 2] 结构恢复")
    if use_llm:
        print(f"  使用 LLM ({provider}) 推断标题层级...")
    else:
        print("  使用启发式规则推断标题层级（--no-llm）")
    structured_text = recover_structure(
        clean_text, use_llm=use_llm, provider=provider,
        snippet_chars=STRUCTURE_SNIPPET_CHARS,
    )

    # ---- Phase 3: Multi-layer Chunking ----
    print("\n[Phase 3] 多层分块")
    # embedding 模型改为懒加载——只在真的需要语义切分时才加载
    chunks = chunk_markdown_v2(
        structured_text,
        source=src,
        max_chars=max_chars,
        overlap=overlap,
        id_prefix=prefix,
        embedding_model=None,  # 懒加载，按需
        company=company or "",
    )
    print_summary(chunks)

    # ---- 输出 JSONL（先写到临时文件，页码回标后再产出最终文件） ----
    tmp_jsonl = str(target_dir / f"{basename}.tmp.jsonl")
    write_jsonl(chunks, tmp_jsonl)

    # ---- 输出 MD 审阅文件 ----
    md_path = str(target_dir / f"{basename}.chunks.md")
    write_md(chunks, md_path)
    print(f"\n  [OK] MD → {md_path}")

    # ---- Phase 4: 页码回标 + 最终 JSONL ----
    final_path = str(target_dir / f"{basename}_paged.jsonl")
    if pdf_path and Path(pdf_path).exists():
        print("\n[Phase 4] 页码回标")
        try:
            from add_pages import add_pages as add_pages_to_jsonl
            add_pages_to_jsonl(tmp_jsonl, pdf_path, final_path)
            os.remove(tmp_jsonl)
            print(f"  [OK] PAGED → {final_path}")
        except ImportError:
            print("  [warn] add_pages.py 未找到，输出不带页码的 JSONL")
            os.rename(tmp_jsonl, final_path)
        except Exception as e:
            print(f"  [warn] 页码回标失败 ({e})，输出不带页码的 JSONL")
            os.rename(tmp_jsonl, final_path)
    else:
        if not pdf_path:
            print("\n[Phase 4] 未提供 PDF，跳过页码回标")
        else:
            print(f"\n[Phase 4] PDF 不存在: {pdf_path}，跳过页码回标")
        os.rename(tmp_jsonl, final_path)
        print(f"  [OK] JSONL → {final_path}")

    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG 预处理管线 —— MinerU MD → 清洗 → 结构恢复 → 多层分块 → 页码回标",
    )
    parser.add_argument("input", help="MinerU 输出的 .md 文件")
    parser.add_argument("--out-dir", default=None,
                        help="输出目录（默认：与输入同目录）")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS,
                        help=f"叶子块最大字符数（默认: {MAX_CHARS}）")
    parser.add_argument("--overlap", type=int, default=OVERLAP_CHARS,
                        help=f"prose 子块重叠字符数（默认: {OVERLAP_CHARS}）")
    parser.add_argument("--no-llm", action="store_true",
                        help="跳过 LLM 结构恢复，使用启发式回退")
    parser.add_argument("--provider", default="",
                        choices=["", "anthropic", "openai"],
                        help=f"LLM 提供商（默认: {STRUCTURE_LLM_PROVIDER}，来自 .env）")
    parser.add_argument("--source", default=None,
                        help="覆盖 source 字段")
    parser.add_argument("--id-prefix", default=None,
                        help="覆盖 chunk_id 前缀")
    parser.add_argument("--pdf", default=None,
                        help="原始 PDF 路径（触发页码回标）")
    parser.add_argument("--company", default=None,
                        help="所属保险公司名称（如 安盛、保诚；不填则从路径推导）")
    args = parser.parse_args()

    # 如果没填 company，尝试从路径推导（看父目录名）
    company = args.company
    if not company:
        input_parent = Path(args.input).resolve().parent.name
        # 排除一些明显不是公司名的目录
        skip_names = {"lib", "output", "newTry", "newtry", "src", "data", "tmp"}
        if input_parent.lower() not in skip_names:
            company = input_parent
            print(f"  自动推导 company: {company}")

    if not Path(args.input).is_file():
        print(f"[ERROR] 输入文件不存在: {args.input}")
        sys.exit(1)

    result = process_file(
        input_path=args.input,
        out_dir=args.out_dir,
        max_chars=args.max_chars,
        overlap=args.overlap,
        use_llm=not args.no_llm,
        llm_provider=args.provider,
        source=args.source,
        id_prefix=args.id_prefix,
        pdf_path=args.pdf,
        company=company,
    )
    print(f"\n完成 → {result}")


if __name__ == "__main__":
    main()
