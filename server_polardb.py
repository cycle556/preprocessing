# -*- coding: utf-8 -*-
"""检索服务 —— PolarDB + bge-m3。支持分层字段 + 公司过滤。

启动：python server_polardb.py
访问：http://本机IP:8765/search?q=癌症最多赔几次&n=5
过滤：http://本机IP:8765/search?q=癌症&company=安盛&n=5
"""

import json
import socket
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import psycopg2
from sentence_transformers import SentenceTransformer

# ====== PolarDB ======
DB_CONFIG = {
    "host": "rag-2026.rwlb.rds.aliyuncs.com",
    "port": 3306,
    "user": "rag111",
    "password": "Abc@122333",
    "dbname": "postgres",
    "connect_timeout": 10,
    "options": "-c search_path=public,cron,pg_catalog",
}

SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_PATH = SCRIPT_DIR.parent / "finnal" / "models" / "BAAI" / "bge-m3"

if _MODEL_PATH.exists():
    MODEL_DIR = str(_MODEL_PATH)
    print(f"本地模型: {MODEL_DIR}")
else:
    MODEL_DIR = "BAAI/bge-m3"
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print("下载模型...")

print("加载 bge-m3 ...")
model = SentenceTransformer(MODEL_DIR)
print("就绪\n")

# 验证连接
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM public.insurance_chunks")
    total = cur.fetchone()[0]
    cur.close()
    conn.close()
    print(f"PolarDB 连接成功，{total} 条\n")
except Exception as e:
    print(f"连接失败: {e}")
    exit(1)


class SearchHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/search":
            self.send_error(404)
            return

        params = parse_qs(parsed.query)
        query = params.get("q", [""])[0]
        n = int(params.get("n", ["5"])[0])
        company = params.get("company", [None])[0]

        if not query:
            self.send_error(400, "缺少 q 参数")
            return

        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()

            vec = model.encode(query, normalize_embeddings=True).tolist()
            vec_str = "[" + ",".join(f"{v:.8f}" for v in vec) + "]"

            sql = (
                "SELECT id, content, source, page_start, page_end, "
                "breadcrumb, section_title, table_caption, has_table, "
                "chunk_type, parent_id, children_ids, sibling_ids, "
                "company, heading_level, content_type, "
                "char_count, est_token_count, "
                "is_split_part, split_part_index, split_part_total, "
                "1 - (embedding <=> %s::vector) AS similarity "
                "FROM public.insurance_chunks "
            )
            if company:
                sql += "WHERE company = %s "
                sql += "ORDER BY embedding <=> %s::vector LIMIT %s"
                cur.execute(sql, (vec_str, company, vec_str, n))
            else:
                sql += "ORDER BY embedding <=> %s::vector LIMIT %s"
                cur.execute(sql, (vec_str, vec_str, n))

            rows = cur.fetchall()
            cur.close()
            conn.close()
        except Exception as e:
            self.send_error(500, str(e))
            return

        out = []
        for i, row in enumerate(rows):
            (cid, content, source, pg_start, pg_end,
             bc, sec, cap, has_tbl,
             ctype, parent_id, children_raw, siblings_raw,
             comp, hlevel, content_type,
             char_count, est_tokens,
             is_split, split_idx, split_total, sim) = row

            children = [x for x in (children_raw or "").split(",") if x]
            siblings = [x for x in (siblings_raw or "").split(",") if x]

            out.append({
                "rank": i + 1,
                "chunk_id": cid,
                "source": source or "",
                "company": comp or "",
                "page": f"{pg_start}~{pg_end}" if pg_start and pg_end else "",
                "breadcrumb": bc or "",
                "section_title": sec or "",
                "heading_level": hlevel or 0,
                "table_caption": cap or "",
                "has_table": has_tbl or False,
                "chunk_type": ctype or "leaf",
                "parent_id": parent_id or "",
                "children_ids": children,
                "sibling_ids": siblings,
                "content_type": content_type or "",
                "is_split_part": is_split or False,
                "split_part_index": split_idx,
                "split_part_total": split_total,
                "char_count": char_count or 0,
                "est_token_count": est_tokens or 0,
                "score": round(float(sim), 4),
                "text": content,
            })

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        print(f"  {args[0]}")


if __name__ == "__main__":
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    server = HTTPServer(("0.0.0.0", 8765), SearchHandler)
    print(f"本机: http://127.0.0.1:8765/search?q=关键词&n=5")
    print(f"过滤: http://127.0.0.1:8765/search?q=癌症&company=安盛&n=5")
    print(f"局域网: http://{local_ip}:8765/search?q=关键词&n=5")
    print("Ctrl+C 停止\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
