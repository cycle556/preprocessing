"""
保险文档 Agentic RAG 系统 - 主入口
功能：一键启动完整的保险文档知识库问答系统，
     支持文档加载→索引→问答全流程，交互式多轮对话。
用法：
    python main.py              # 单次演示查询
    python main.py --interactive  # 交互式多轮对话模式
    python main.py --no-index   # 跳过索引，直接使用已有向量库进行问答
    python main.py --reload     # 强制重建索引
    python main.py --stats      # 查看系统状态
"""
import os
import sys
import yaml
from pathlib import Path

from logger import SystemLogger
from agent import InsuranceRAGAgent


def load_config(config_path: str = "config.yaml") -> dict:
    """加载 YAML 配置文件"""
    default_config = {
        "system": {"name": "InsuranceRAG", "version": "2.0.0",
                    "log_level": "INFO", "log_dir": "./logs"},
        "document": {"source_dir": "./保司文件2.0"},
        "chunking": {"chunk_size": 800, "chunk_overlap": 100},
        "embedding": {"model": "doubao-embedding-vision", "batch_size": 4},
        "llm": {"model": "doubao-seed-2.0-pro", "temperature": 0.0},
        "vector_store": {"provider": "chroma", "persist_directory": "./chroma_db",
                          "collection_name": "insurance_knowledge"},
        "retrieval": {"top_k": 10, "use_rerank": True},
        "conversation": {"storage_dir": "./conversation_data"},
        "answer_generation": {"require_citation": True},
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
        except Exception as e:
            print(f"[WARNING] 配置文件加载失败，使用默认配置: {e}")

    return default_config


def run_demo(agent: InsuranceRAGAgent):
    """运行演示查询"""
    queries = [
        "友邦对内地居民投保有什么要求？",
        "保诚的缴费方式有哪些？",
        "宏利环球货币保障计划的保险责任是什么？",
    ]

    print("\n" + "=" * 60)
    print("  保险文档 Agentic RAG 系统 - 演示查询")
    print("=" * 60)

    conv_id = agent.conversation_manager.create_conversation()

    for query in queries:
        print(f"\n用户: {query}")
        print("-" * 60)
        response = agent.query(query, conv_id)
        print(f"系统: {response}")
        print()

    print("-" * 60)
    print("演示完成！使用 --interactive 进入交互模式。")


def run_interactive(agent: InsuranceRAGAgent):
    """运行交互式多轮对话模式"""
    print("\n" + "=" * 60)
    print("  保险文档 Agentic RAG 系统 - 交互模式")
    print("=" * 60)
    print("输入您的问题，系统将基于保险文档知识库回答。")
    print("输入 'quit' 或 'exit' 退出，输入 'new' 开始新会话。")
    print("=" * 60 + "\n")

    conv_id = agent.conversation_manager.create_conversation()

    while True:
        try:
            query = input("您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if query.lower() == "new":
            conv_id = agent.conversation_manager.create_conversation()
            print(f"[INFO] 新会话已创建: {conv_id}")
            continue

        if query.lower() == "stats":
            stats = agent.get_stats()
            print(f"[INFO] 向量库文档数: {stats['vector_store'].get('document_count', 0)}")
            print(f"[INFO] 活跃会话数: {stats.get('conversations', 0)}")
            continue

        print("\n系统思考中...")
        response = agent.query(query, conv_id)
        print(f"\n系统:\n{response}\n")
        print("-" * 60)


def main():
    """主函数"""
    config = load_config()

    sys_cfg = config.get("system", {})
    logger = SystemLogger().init(
        log_dir=sys_cfg.get("log_dir", "./logs"),
        log_level=sys_cfg.get("log_level", "INFO"),
    )

    logger.info("=" * 60)
    logger.info("保险文档 Agentic RAG 系统 v2.0.0 启动")
    logger.info("=" * 60)

    doc_cfg = config.get("document", {})
    source_dir = doc_cfg.get("source_dir", "./保司文件2.0")

    logger.info("初始化 Agent...")
    try:
        agent = InsuranceRAGAgent(config)
    except Exception as e:
        logger.error(f"Agent 初始化失败: {e}")
        print(f"\n[ERROR] 系统初始化失败: {e}")
        print("请检查:\n"
              "  1. .env 文件中是否配置了 VOLC_API_KEY\n"
              "  2. 网络连接是否正常\n"
              "  3. 依赖是否完整安装 (pip install -r requirements.txt)")
        return

    force_reload = "--reload" in sys.argv
    skip_index = "--no-index" in sys.argv
    if "--stats" in sys.argv:
        stats = agent.get_stats()
        print("\n系统状态:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if skip_index:
        logger.info("跳过索引阶段，直接使用已有向量库")
        print("\n[1/2] 跳过索引（--no-index），使用已有向量库...")
    else:
        logger.info("加载文档并建立索引...")
        print("\n[1/2] 加载文档并建立索引...")
        try:
            chunk_count = agent.load_and_index_documents(source_dir, force_reload=force_reload)
            if chunk_count == 0 and not force_reload:
                print("  文档已是最新，无需重新索引")
            else:
                print(f"  索引完成：{chunk_count} 个文档块")
        except Exception as e:
            logger.error(f"文档索引失败: {e}")
            print(f"  [WARNING] 文档索引失败: {e}")
            print("  将继续使用已有索引运行")

    print("\n[2/2] 开始问答服务")

    if "--interactive" in sys.argv or "-i" in sys.argv:
        run_interactive(agent)
    else:
        run_demo(agent)


if __name__ == "__main__":
    main()
