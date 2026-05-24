# -*- coding: utf-8 -*-
"""清空 PolarDB insurance_chunks 表中所有旧数据。

用法：
    python clear_polardb.py          # 先看有多少条，确认后删除
    python clear_polardb.py --force  # 不确认，直接删
"""

import sys
import psycopg2

DB_CONFIG = {
    "host": "rag-2026.rwlb.rds.aliyuncs.com",
    "port": 3306,
    "user": "rag111",
    "password": "Abc@122333",
    "dbname": "postgres",
    "connect_timeout": 10,
}

force = "--force" in sys.argv or "-y" in sys.argv

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # 看当前数据量
    cur.execute("SELECT COUNT(*) FROM public.insurance_chunks")
    total = cur.fetchone()[0]
    print(f"当前表中有 {total} 条数据")

    if total == 0:
        print("表已空，无需操作。")
        cur.close(); conn.close()
        sys.exit(0)

    if not force:
        print(f"\n确认删除 {total} 条？（y/n）", end=" ")
        answer = input().strip().lower()
        if answer not in ("y", "yes"):
            print("已取消。")
            cur.close(); conn.close()
            sys.exit(0)

    cur.execute("DELETE FROM public.insurance_chunks")
    conn.commit()
    print(f"已删除 {total} 条。")

    # VACUUM 必须在事务外执行
    conn.autocommit = True
    cur.execute("VACUUM public.insurance_chunks")
    conn.autocommit = False
    print("空间已回收。")

    cur.close()
    conn.close()
except Exception as e:
    print(f"操作失败: {e}")
    sys.exit(1)
