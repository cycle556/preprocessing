# -*- coding: utf-8 -*-
"""
轻量检索服务 —— 连接 PolarDB (pgvector)，在自己电脑上启动即可。

依赖：pip install psycopg2 sentence_transformers

启动：python server_polardb.py

访问例子：http://你的IP:8765/search?q=癌症最多赔几次&n=5

"""

import json
import socket
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import psycopg2
from sentence_transformers import SentenceTransformer

# ====== 连接 PolarDB ======
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
_MODEL_PATH = SCRIPT_DIR / "models" / "BAAI" / "bge-m3"

if _MODEL_PATH.exists():
    MODEL_DIR = str(_MODEL_PATH)
    print(f"使用本地模型: {MODEL_DIR}")
else:
    # 没有本地模型时自动从镜像下载
    MODEL_DIR = "BAAI/bge-m3"
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print("本地模型不存在，从 hf-mirror.com 镜像下载（首次约 2GB，需几分钟）...")

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
    print(f"PolarDB 连接成功，共 {total} 条数据\n")
except Exception as e:
    print(f"连接 PolarDB 失败: {e}")
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

        if not query:
            self.send_error(400, "缺少 q 参数")
            return

        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()

            # 向量化查询
            vec = model.encode(query, normalize_embeddings=True).tolist()
            vec_str = "[" + ",".join(f"{v:.8f}" for v in vec) + "]"
            cur.execute(
                "SELECT id, content, source, page_start, page_end, "
                "breadcrumb, section_title, table_caption, has_table, "
                "1 - (embedding <=> %s::vector) AS similarity "
                "FROM public.insurance_chunks "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (vec_str, vec_str, n),
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
        except Exception as e:
            self.send_error(500, str(e))
            return

        out = []
        for i, row in enumerate(rows):
            cid, content, source, pg_start, pg_end, bc, sec, cap, has_tbl, sim = row
            out.append({
                "rank": i + 1,
                "chunk_id": cid,
                "source": source or "",
                "page": f"{pg_start}~{pg_end}" if pg_start and pg_end else "",
                "breadcrumb": bc or "",
                "section_title": sec or "",
                "table_caption": cap or "",
                "has_table": has_tbl or False,
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
    print(f"本机访问:   http://127.0.0.1:8765/search?q=关键词&n=5")
    print(f"别人访问:   http://{local_ip}:8765/search?q=关键词&n=5")
    print("按 Ctrl+C 停止\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
