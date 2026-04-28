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
        "document": {"source_dir": "./保司文件"},
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


def create_sample_pdf(output_dir: str = "./保司文件"):
    """在目标目录创建示例保险条款 PDF（如果目录为空）"""
    os.makedirs(output_dir, exist_ok=True)

    existing = [f for f in os.listdir(output_dir)
                if f.endswith(('.pdf', '.txt'))]
    if existing:
        return

    sample_text = """重大疾病保险条款

第一章 总则
第一条 本保险合同由保险单或其他保险凭证、所附条款、投保单、与本合同有关的投保文件、合法有效的声明、批注、附贴批单及其他书面协议构成。

第二章 保险责任
第二条 在本合同保险期间内，本公司承担下列保险责任：
一、重大疾病保险金
被保险人于本合同生效（或最后一次复效，以较迟者为准）之日起90日内（含第90日），经专科医生确诊初次患有本合同所定义的重大疾病（无论一种或多种），本公司按投保人已交纳的本合同的保险费（不计利息）给付重大疾病保险金，本合同终止。
被保险人于本合同生效（或最后一次复效，以较迟者为准）之日起90日后，经专科医生确诊初次患有本合同所定义的重大疾病（无论一种或多种），本公司按基本保险金额给付重大疾病保险金，本合同终止。

第三章 责任免除
第三条 因下列情形之一，导致被保险人发生疾病、达到疾病状态或进行手术的，本公司不承担给付保险金的责任：
一、投保人对被保险人的故意杀害、故意伤害；
二、被保险人故意自伤、故意犯罪或抗拒依法采取的刑事强制措施；
三、被保险人服用、吸食或注射毒品；
四、被保险人酒后驾驶、无合法有效驾驶证驾驶，或驾驶无有效行驶证的机动车；
五、被保险人感染艾滋病病毒或患艾滋病；
六、战争、军事冲突、暴乱或武装叛乱；
七、核爆炸、核辐射或核污染；
八、遗传性疾病，先天性畸形、变形或染色体异常。

第四章 保险金额
第四条 本合同的基本保险金额由投保人与本公司约定并在保险单上载明。

第五章 保险期间
第五条 本合同的保险期间为终身，自本合同生效日起至被保险人身故时止。

第六章 保险费
第六条 本合同的保险费交付方式分为趸交、年交、半年交和月交，由投保人在投保时选择。分期交付保险费的，交费期间由投保人与本公司约定并在保险单上载明。
"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        output_path = os.path.join(output_dir, "重大疾病保险条款_示例.pdf")
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter

        try:
            pdfmetrics.registerFont(TTFont('SimSun', 'simsun.ttc'))
            font = 'SimSun'
        except Exception:
            font = 'Helvetica'

        c.setFont(font, 10)
        lines = sample_text.split('\n')
        y = height - 50

        for line in lines:
            if y < 50:
                c.showPage()
                c.setFont(font, 10)
                y = height - 50
            c.drawString(50, y, line[:80])
            y -= 15

        c.save()
        print(f"[INFO] 示例文档已生成: {output_path}")
    except ImportError:
        output_path = os.path.join(output_dir, "重大疾病保险条款_示例.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(sample_text)
        print(f"[INFO] reportlab 未安装，已生成文本版示例: {output_path}")


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
    source_dir = doc_cfg.get("source_dir", "./保司文件")

    create_sample_pdf(source_dir)

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
