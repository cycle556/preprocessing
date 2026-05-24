# -*- coding: utf-8 -*-
"""嵌入 + 上传到 PolarDB (pgvector)。

用法：
    python upload_polardb.py ./output              # 上传 output/ 下所有 _paged.jsonl
    python upload_polardb.py ./output --dry-run    # 只看不写
"""

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg2
from sentence_transformers import SentenceTransformer

# ---- 配置 ----
DB_CONFIG = {
    "host": os.environ.get("POLARDB_HOST", "rag-2026.rwlb.rds.aliyuncs.com"),
    "port": int(os.environ.get("POLARDB_PORT", "3306")),
    "user": os.environ.get("POLARDB_USER", "rag111"),
    "password": os.environ.get("POLARDB_PASSWORD", "Abc@122333"),
    "dbname": os.environ.get("POLARDB_DB", "postgres"),
    "connect_timeout": 30,
}

MODEL_PATH = r"C:\Users\CYL\Desktop\finnal\models\BAAI\bge-m3"

# 旧表需要的列 → 如果不存在则新增
NEW_COLUMNS = {
    "company":          "TEXT DEFAULT ''",
    "chunk_type":       "TEXT DEFAULT 'leaf'",
    "parent_id":        "TEXT DEFAULT ''",
    "children_ids":     "TEXT DEFAULT ''",
    "sibling_ids":      "TEXT DEFAULT ''",
    "content_type":     "TEXT DEFAULT ''",
    "est_token_count":  "INTEGER DEFAULT 0",
    "is_split_part":    "BOOLEAN DEFAULT FALSE",
    "split_part_index": "INTEGER",
    "split_part_total": "INTEGER",
}


def ensure_columns(cur) -> list[str]:
    """检查缺失的列并返回可 ADD 的列名列表。不自动执行，需用户确认。"""
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name='insurance_chunks'"
    )
    existing = {row[0] for row in cur.fetchall()}
    missing = {k: v for k, v in NEW_COLUMNS.items() if k not in existing}
    return missing


def migrate_table(cur, conn, missing: dict) -> bool:
    """添加缺失的列。返回是否成功。"""
    print(f"\n需要 ALTER TABLE 添加 {len(missing)} 个列:")
    for col, defn in missing.items():
        print(f"  ALTER TABLE public.insurance_chunks ADD COLUMN {col} {defn};")

    if "--yes" in sys.argv:
        for col, defn in missing.items():
            sql = f"ALTER TABLE public.insurance_chunks ADD COLUMN {col} {defn}"
            print(f"  执行: {sql}")
            cur.execute(sql)
        conn.commit()
        print("  列已添加。")
        return True

    print("\n  用 --yes 自动执行，或手动在 DMS 中运行上述 SQL，然后重新运行上传。")
    return False


def main():
    parser = argparse.ArgumentParser(description="bge-m3 嵌入 → PolarDB pgvector 上传")
    parser.add_argument("data_dir", help="存放 _paged.jsonl 的目录（递归搜索）")
    parser.add_argument("--dry-run", action="store_true", help="只显示不上传")
    parser.add_argument("--yes", action="store_true", help="自动执行 ALTER TABLE")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    files = sorted(data_dir.rglob("*_paged.jsonl"))
    if not files:
        print(f"[ERROR] 在 {data_dir} 下未找到 _paged.jsonl")
        sys.exit(1)
    print(f"找到 {len(files)} 个 JSONL 文件")

    # 加载本地 bge-m3
    print(f"加载 bge-m3 ({MODEL_PATH}) ...")
    model = SentenceTransformer(MODEL_PATH, local_files_only=True)
    print(f"就绪，维度: {model.get_sentence_embedding_dimension()}")

    # 连接 PolarDB
    print(f"\n连接 {DB_CONFIG['host']}:{DB_CONFIG['port']} ...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    print("连接成功。")

    # 检查列
    missing = ensure_columns(cur)
    if missing:
        if not migrate_table(cur, conn, missing):
            cur.close(); conn.close()
            sys.exit(1)

    # 计数
    cur.execute("SELECT COUNT(*) FROM public.insurance_chunks")
    existing = cur.fetchone()[0]
    if existing:
        print(f"\n表中已有 {existing} 条数据（新上传会跳过同 ID）。")
        print("如需清空旧数据，先运行: python clear_polardb.py")

    if args.dry_run:
        total = 0
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                n = sum(1 for l in f if l.strip())
            print(f"  [DRY] {fp.parent.name}/{fp.name}: {n} chunks")
            total += n
        print(f"\n[DRY] 总计 {total} chunks，未实际写入。")
        cur.close(); conn.close()
        return

    # 逐文件上传
    total = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]
        if not chunks:
            continue

        print(f"\n  {fp.parent.name}/{fp.name}: {len(chunks)} chunks", end=" ", flush=True)

        for c in chunks:
            text = c.get("content_with_context", c["content"])
            vec = model.encode(text, normalize_embeddings=True).tolist()
            bc = c.get("breadcrumb", [])

            # 新字段兼容旧表：用 get(..., default) 兜底
            cur.execute(
                """INSERT INTO public.insurance_chunks
                   (id, content, source, page_start, page_end, breadcrumb,
                    section_title, table_caption, has_table, heading_level,
                    char_count, est_token_count,
                    is_split_part, split_part_index, split_part_total,
                    embedding,
                    company, chunk_type, parent_id, children_ids, sibling_ids, content_type)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (id) DO NOTHING""",
                (
                    c["chunk_id"],
                    c.get("content", ""),
                    c.get("source", ""),
                    str(c["page_start"]) if c.get("page_start") is not None else "",
                    str(c["page_end"]) if c.get("page_end") is not None else "",
                    " > ".join(bc) if bc else "",
                    c.get("section_title", ""),
                    c.get("table_caption") or "",
                    c.get("has_table", False),
                    c.get("heading_level", 0),
                    c.get("char_count", 0),
                    c.get("est_token_count", 0),
                    c.get("is_split_part", False),
                    c.get("split_part_index"),
                    c.get("split_part_total"),
                    vec,
                    c.get("company", ""),
                    c.get("chunk_type", "leaf"),
                    c.get("parent_id") or "",
                    ",".join(c.get("children_ids", [])) or "",
                    ",".join(c.get("sibling_ids", [])) or "",
                    c.get("content_type", ""),
                ),
            )

        conn.commit()
        total += len(chunks)
        print("OK")

    cur.close()
    conn.close()
    print(f"\n上传完成: {total} chunks → PolarDB")


if __name__ == "__main__":
    main()
