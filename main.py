import os
from dotenv import load_dotenv
from document_processor import InsuranceDocumentProcessor
from retrieval_engine import InsuranceRetrievalEngine
from session_manager import SessionManager
from agent_orchestrator import InsuranceAgentOrchestrator


def create_sample_insurance_clause():
    sample_text = """
重大疾病保险条款

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
    
    with open("sample_insurance_clause.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        c = canvas.Canvas("sample_insurance_clause.pdf", pagesize=letter)
        width, height = letter
        
        try:
            pdfmetrics.registerFont(TTFont('SimSun', 'simsun.ttc'))
            font_name = 'SimSun'
        except:
            font_name = 'Helvetica'
        
        c.setFont(font_name, 10)
        
        lines = sample_text.split('\n')
        y = height - 50
        
        for line in lines:
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont(font_name, 10)
            
            c.drawString(50, y, line[:80])
            y -= 15
        
        c.save()
        print("示例PDF条款已生成: sample_insurance_clause.pdf")
    except:
        print("提示: 未安装reportlab，跳过PDF生成，使用文本文件处理")


def main():
    load_dotenv()
    
    openai_api_key = os.getenv("VOLC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("请在.env文件中设置VOLC_API_KEY或OPENAI_API_KEY")
        return
    
    volc_base_url = os.getenv("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
    volc_embedding_model = os.getenv("VOLC_EMBEDDING_MODEL", "doubao-embedding-vision")
    volc_llm_model = os.getenv("VOLC_LLM_MODEL", "doubao-seed-2.0-pro")
    
    print("=" * 60)
    print("保险文档Agentic RAG系统 - 端到端示例")
    print("=" * 60)
    
    print("\n[1/6] 创建示例保险条款文档...")
    create_sample_insurance_clause()
    
    print("\n[2/6] 初始化文档处理器...")
    processor = InsuranceDocumentProcessor(chunk_size=800, chunk_overlap=100)
    
    print("\n[3/6] 加载并处理文档...")
    pdf_path = "sample_insurance_clause.pdf"
    txt_path = "sample_insurance_clause.txt"
    
    if os.path.exists(pdf_path):
        result = processor.process_document(pdf_path)
    else:
        text, metadata = processor.load_pdf(txt_path) if txt_path.endswith('.pdf') else (open(txt_path, 'r', encoding='utf-8').read(), {"source": txt_path, "file_type": "txt"})
        result = {
            "chunks": processor.split_by_clause_structure(text, metadata),
            "tables": [],
            "metadata": metadata
        }
    
    print(f"   文档分块数: {len(result['chunks'])}")
    
    print("\n[4/6] 初始化检索引擎并建立索引...")
    retrieval_engine = InsuranceRetrievalEngine(
        persist_directory="./chroma_db",
        api_key=openai_api_key,
        base_url=volc_base_url,
        embedding_model=volc_embedding_model
    )
    retrieval_engine.create_collection("insurance_docs")
    retrieval_engine.add_documents(result['chunks'], result['tables'])
    print("   索引建立完成")
    
    print("\n[5/6] 初始化会话管理与智能体编排器...")
    session_manager = SessionManager(storage_dir="./session_data")
    agent = InsuranceAgentOrchestrator(
        openai_api_key=openai_api_key,
        retrieval_engine=retrieval_engine,
        session_manager=session_manager,
        model=volc_llm_model,
        base_url=volc_base_url
    )
    print("   智能体初始化完成")
    
    print("\n[6/6] 执行查询示例...")
    user_query = "这款重疾险的等待期是多少？哪些重疾相关情况在免责范围内？"
    print(f"\n用户查询: {user_query}")
    print("-" * 60)
    
    conversation_id = session_manager.create_conversation()
    response = agent.run(user_query, conversation_id)
    
    print("\n系统回复:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    
    print("\n示例运行完成！")
    print("\n提示: 您可以继续使用以下方式测试系统:")
    print("  - 修改main.py中的user_query进行不同查询")
    print("  - 替换sample_insurance_clause.pdf为真实的保险条款")
    print("  - 查看./chroma_db和./session_data目录了解数据存储")


if __name__ == "__main__":
    main()
