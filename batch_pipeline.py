# -*- coding: utf-8 -*-
r"""批量处理脚本 —— 按公司目录批量运行预处理管线。

用法：
    python batch_pipeline.py <保司文件根目录> --out-dir ./output

    可选：
    --no-llm          跳过 LLM 结构恢复
    --max-chars 1800  叶子块最大字符数

工作流：
    保司文件/
    ├── 安盛/
    │   ├── product1.md   <- MinerU 输出的脏 MD
    │   └── product1.pdf  <- 原始 PDF（可选，用于页码回标）
    ├── 保诚/
    │   └── ...
    └── ...

    输出：
    output/
    ├── 安盛/
    │   ├── product1_paged.jsonl
    │   └── product1.chunks.xlsx
    ├── 保诚/
    │   └── ...
"""

import argparse
import os
import sys
from pathlib import Path

# 确保能找到 pipeline 和 lib
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import process_file


def find_md_files(base_dir: str) -> dict[str, list[tuple[str, str | None]]]:
    """递归扫描目录，返回 {公司名: [(md_path, pdf_path_or_None), ...]}。

    支持两种目录结构：
      a) 保司文件/安盛/product.md  （平铺）
      b) 保司文件/安盛/产品名/product.md  （嵌套）
    同名 .pdf 在相同子目录下自动关联。
    """
    base = Path(base_dir)
    company_files: dict[str, list[tuple[str, str | None]]] = {}

    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        company = d.name

        # 递归找所有 .md
        md_files = list(d.rglob('*.md'))
        if not md_files:
            continue

        tasks = []
        for md_path in md_files:
            # 同名 .pdf 在同一目录
            pdf_candidate = md_path.with_suffix('.pdf')
            pdf_path = str(pdf_candidate) if pdf_candidate.is_file() else None
            tasks.append((str(md_path), pdf_path))

        if tasks:
            company_files[company] = tasks

    return company_files


def main():
    parser = argparse.ArgumentParser(
        description="批量预处理 —— 按公司目录运行完整管线",
    )
    parser.add_argument("base_dir", help="保司文件根目录")
    parser.add_argument("--out-dir", default="./output",
                        help="输出根目录（默认: ./output）")
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--no-llm", action="store_true",
                        help="跳过 LLM 结构恢复")
    parser.add_argument("--provider", default="",
                        choices=["", "anthropic", "openai"],
                        help="LLM 提供商（默认：来自 .env 配置）")
    parser.add_argument("--no-pdf", action="store_true",
                        help="不关联 PDF（跳过页码回标）")
    parser.add_argument("--force", action="store_true",
                        help="强制重新处理所有文件（默认跳过已完成的）")
    args = parser.parse_args()

    company_files = find_md_files(args.base_dir)

    if not company_files:
        print(f"[ERROR] 在 {args.base_dir} 下未找到任何 .md 文件")
        sys.exit(1)

    total_tasks = sum(len(tasks) for tasks in company_files.values())
    print(f"找到 {len(company_files)} 家公司，共 {total_tasks} 个 .md 文件")

    # 过滤已完成的任务
    skipped = 0
    if not args.force:
        new_tasks = {}
        for company, tasks in company_files.items():
            company_out = Path(args.out_dir) / company
            remaining = []
            for md_path, pdf_path in tasks:
                stem = Path(md_path).stem
                output_file = company_out / f"{stem}_paged.jsonl"
                if output_file.is_file():
                    skipped += 1
                else:
                    remaining.append((md_path, pdf_path))
            if remaining:
                new_tasks[company] = remaining
        company_files = new_tasks

    if skipped:
        print(f"跳过 {skipped} 个已完成（用 --force 强制重跑）")
    if not company_files:
        print("全部已完成，无需处理")
        sys.exit(0)

    print(f"待处理: {sum(len(t) for t in company_files.values())} 个文件\n")

    succeeded = 0
    failed = []

    for company, tasks in sorted(company_files.items()):
        print(f"\n{'='*60}")
        print(f"公司: {company} ({len(tasks)} 个文件)")
        print(f"{'='*60}")

        company_out = Path(args.out_dir) / company

        for md_path, pdf_path in tasks:
            try:
                name = Path(md_path).stem
                print(f"\n  处理: {name}")

                result = process_file(
                    input_path=md_path,
                    out_dir=str(company_out),
                    max_chars=args.max_chars,
                    overlap=args.overlap,
                    use_llm=not args.no_llm,
                    llm_provider=args.provider,
                    company=company,
                    pdf_path=pdf_path if not args.no_pdf else None,
                )
                succeeded += 1
                print(f"  [OK] → {result}")
            except Exception as e:
                failed.append((company, name, str(e)))
                print(f"  [FAIL] {e}")

    print(f"\n{'='*60}")
    print(f"完成: {succeeded}/{total_tasks} 成功")
    if failed:
        print(f"失败 {len(failed)}:")
        for company, name, err in failed:
            print(f"  - {company}/{name}: {err}")


if __name__ == "__main__":
    main()
