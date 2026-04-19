import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str


class SimpleInsuranceRAG:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
    
    def load_sample_document(self):
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
        return self._split_by_clause(sample_text)
    
    def _split_by_clause(self, text: str) -> List[DocumentChunk]:
        chunks = []
        clause_patterns = [
            "第一章", "第二章", "第三章", "第四章", "第五章", "第六章",
            "第一条", "第二条", "第三条", "第四条", "第五条", "第六条"
        ]
        
        lines = text.split('\n')
        current_chapter = ""
        current_section = ""
        current_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            is_clause_start = any(pattern in line for pattern in clause_patterns)
            
            if is_clause_start and current_content:
                chunk = DocumentChunk(
                    content='\n'.join(current_content),
                    metadata={
                        "source": "sample_insurance_clause",
                        "chapter": current_chapter,
                        "section": current_section,
                        "chunk_id": f"chunk_{len(chunks)}"
                    },
                    chunk_id=f"chunk_{len(chunks)}"
                )
                chunks.append(chunk)
                current_content = []
            
            if is_clause_start:
                if "章" in line:
                    current_chapter = line
                elif "条" in line:
                    current_section = line
            
            current_content.append(line)
        
        if current_content:
            chunk = DocumentChunk(
                content='\n'.join(current_content),
                metadata={
                    "source": "sample_insurance_clause",
                    "chapter": current_chapter,
                    "section": current_section,
                    "chunk_id": f"chunk_{len(chunks)}"
                },
                chunk_id=f"chunk_{len(chunks)}"
            )
            chunks.append(chunk)
        
        self.chunks = chunks
        return chunks
    
    def search(self, query: str) -> List[DocumentChunk]:
        keywords = ["等待期", "免责", "责任免除", "90日"]
        results = []
        
        for chunk in self.chunks:
            score = 0
            for keyword in keywords:
                if keyword in chunk.content:
                    score += 1
            if score > 0:
                results.append((chunk, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in results[:5]]
    
    def extract_waiting_period(self, chunks: List[DocumentChunk]) -> Optional[Dict[str, Any]]:
        for chunk in chunks:
            if "等待期" in chunk.content or "90日" in chunk.content:
                match = re.search(r'(\d+)日', chunk.content)
                if match:
                    return {
                        "value": f"{match.group(1)}日",
                        "source_text": chunk.content,
                        "source_location": f"{chunk.metadata.get('chapter', '')} {chunk.metadata.get('section', '')}"
                    }
        return None
    
    def extract_exclusions(self, chunks: List[DocumentChunk]) -> Optional[Dict[str, Any]]:
        for chunk in chunks:
            if "责任免除" in chunk.content or "免责" in chunk.content:
                exclusions = []
                lines = chunk.content.split('\n')
                for line in lines:
                    if line.strip().startswith(("一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、")):
                        exclusions.append(line.strip())
                
                return {
                    "value": "\n".join(exclusions),
                    "source_text": chunk.content,
                    "source_location": f"{chunk.metadata.get('chapter', '')} {chunk.metadata.get('section', '')}"
                }
        return None
    
    def query(self, user_query: str) -> str:
        print(f"用户查询: {user_query}")
        print("-" * 60)
        
        chunks = self.search(user_query)
        print(f"检索到 {len(chunks)} 个相关文档片段\n")
        
        response_parts = ["根据保险文档查询结果：\n"]
        
        if "等待期" in user_query:
            waiting_period = self.extract_waiting_period(chunks)
            if waiting_period:
                response_parts.append("【等待期】")
                response_parts.append(waiting_period["value"])
                response_parts.append(f"来源：{waiting_period['source_location']}")
                response_parts.append("")
        
        if "免责" in user_query or "责任免除" in user_query:
            exclusions = self.extract_exclusions(chunks)
            if exclusions:
                response_parts.append("【免责事项】")
                response_parts.append(exclusions["value"])
                response_parts.append(f"来源：{exclusions['source_location']}")
                response_parts.append("")
        
        if len(response_parts) == 1:
            return "未查询到相关保险信息，请尝试补充查询条件。"
        
        return "\n".join(response_parts)


def main():
    print("=" * 60)
    print("保险文档Agentic RAG系统 - 简化演示版")
    print("=" * 60)
    
    print("\n[1/3] 初始化并加载示例保险条款...")
    rag = SimpleInsuranceRAG()
    chunks = rag.load_sample_document()
    print(f"   文档分块数: {len(chunks)}")
    
    print("\n[2/3] 执行查询示例...")
    user_query = "这款重疾险的等待期是多少？哪些重疾相关情况在免责范围内？"
    
    print("\n[3/3] 生成回复...")
    response = rag.query(user_query)
    
    print("\n系统回复:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    
    print("\n演示完成！")
    print("\n说明: 这是一个简化版本，展示了保险文档处理的核心逻辑:")
    print("  - 按条款层级分块")
    print("  - 关键词检索")
    print("  - 规则提取保险数据")
    print("  - 来源溯源标注")


if __name__ == "__main__":
    main()
