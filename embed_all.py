# -*- coding: utf-8 -*-
"""批量 embedding —— 读入 _paged.jsonl → bge-m3 向量 → ChromaDB。

用法：
    python embed_all.py <jsonl目录> [--db ./chroma_db]

示例：
    python embed_all.py ./output              # 嵌入 output/ 下所有 _paged.jsonl
    python embed_all.py ./output --db ./insurance_db
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# 本地模型路径
MODEL_PATH = r"C:\Users\CYL\Desktop\finnal\models\BAAI\bge-m3"


def main():
    parser = argparse.ArgumentParser(description="批量 embedding → ChromaDB")
    parser.add_argument("data_dir", help="存放 _paged.jsonl 的目录（会递归搜索）")
    parser.add_argument("--db", default="./chroma_db", help="ChromaDB 持久化目录")
    parser.add_argument("--model", default=MODEL_PATH, help="bge-m3 模型路径")
    parser.add_argument("--collection", default="insurance_docs",
                        help="ChromaDB collection 名称")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    db_dir = Path(args.db).resolve()

    # 递归搜索所有 _paged.jsonl
    files = sorted(data_dir.rglob("*_paged.jsonl"))
    if not files:
        print(f"[ERROR] 在 {data_dir} 下未找到 _paged.jsonl 文件")
        sys.exit(1)

    print(f"找到 {len(files)} 个 JSONL 文件")

    # 加载模型（本地，不联网）
    print(f"加载 bge-m3 ({args.model}) ...")
    model = SentenceTransformer(args.model, local_files_only=True)
    print(f"模型就绪，维度: {model.get_sentence_embedding_dimension()}")

    # ChromaDB
    client = chromadb.PersistentClient(path=str(db_dir))
    # 如果 collection 已存在，先删再建（全量刷新）
    try:
        client.delete_collection(args.collection)
        print(f"已清空旧 collection: {args.collection}")
    except Exception:
        pass
    collection = client.create_collection(
        name=args.collection,
        metadata={"description": "保司文档分块，分层索引"},
    )

    total = 0
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]

        if not chunks:
            continue

        ids, embeddings, metadatas, documents = [], [], [], []
        for c in chunks:
            # 用 content_with_context 做 embedding（含面包屑上下文）
            text_to_embed = c.get("content_with_context", c["content"])
            emb = model.encode(text_to_embed).tolist()

            ids.append(c["chunk_id"])
            embeddings.append(emb)
            metadatas.append({
                "source": c.get("source", ""),
                "company": c.get("company", ""),
                "chunk_type": c.get("chunk_type", "leaf"),
                "heading_level": c.get("heading_level", 0),
                "breadcrumb": " > ".join(c.get("breadcrumb", [])),
                "section_title": c.get("section_title", ""),
                "has_table": c.get("has_table", False),
                "table_caption": c.get("table_caption") or "",
                "content_type": c.get("content_type", ""),
                "page_start": str(c["page_start"]) if c.get("page_start") is not None else "",
                "page_end": str(c["page_end"]) if c.get("page_end") is not None else "",
                "char_count": c.get("char_count", 0),
                "parent_id": c.get("parent_id") or "",
            })
            documents.append(c.get("content", ""))

        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        print(f"  {fpath.parent.name}/{fpath.name}: {len(chunks)} chunks")
        total += len(chunks)

    print(f"\n完成: {total} 个 chunk → {db_dir} ({collection.count()} 条)")


if __name__ == "__main__":
    main()
