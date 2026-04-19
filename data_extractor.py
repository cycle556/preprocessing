from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from retrieval_engine import RetrievalResult


@dataclass
class ExtractedField:
    field_name: str
    value: str
    source_text: str
    source_metadata: Dict[str, Any]
    confidence: float


INSURANCE_FIELDS = {
    "waiting_period": ["等待期", "观察期", "等待期限"],
    "sum_insured": ["保额", "保险金额", "基本保额"],
    "deductible": ["免赔额", "免赔"],
    "payment_ratio": ["赔付比例", "给付比例", "报销比例"],
    "coverage_period": ["保障期间", "保险期间", "保障期限"],
    "premium_payment_period": ["缴费期间", "缴费期限"],
    "exclusions": ["责任免除", "免责条款", "除外责任"],
    "coverage": ["保险责任", "保障责任"]
}


class InsuranceDataExtractor:
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model,
            api_key=openai_api_key,
            temperature=0
        )
    
    def extract_by_rules(self, results: List[RetrievalResult], 
                        field_name: str) -> List[ExtractedField]:
        keywords = INSURANCE_FIELDS.get(field_name, [])
        extracted_fields = []
        
        for result in results:
            for keyword in keywords:
                if keyword in result.content:
                    value = self._extract_value_around_keyword(result.content, keyword)
                    if value:
                        extracted_fields.append(ExtractedField(
                            field_name=field_name,
                            value=value,
                            source_text=result.content,
                            source_metadata=result.metadata,
                            confidence=0.7
                        ))
        
        return extracted_fields
    
    def _extract_value_around_keyword(self, text: str, keyword: str) -> Optional[str]:
        pattern = rf'{keyword}[：:]\s*([^。\n！？]{1,50})'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        
        pattern = rf'{keyword}\s*[为是是：:]\s*([^。\n！？]{1,50})'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        
        return None
    
    def extract_by_llm(self, results: List[RetrievalResult], 
                      query: str, target_fields: List[str]) -> Dict[str, ExtractedField]:
        context = self._format_retrieval_results(results)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名专业的保险数据提取专家。请从给定的保险文档内容中，精准提取用户需要的保险数据。

提取规则：
1. 必须100%从原文提取，不做任何演绎、扩写、简化
2. 提取结果后必须附上原文引用片段
3. 标注数据来源位置
4. 如果数据有前提条件，必须完整提取条件
5. 如果没有找到相关信息，明确说明"未找到"

输出格式：JSON格式
{{
  "字段名": {{
    "value": "提取值",
    "source_text": "原文引用片段",
    "source_location": "来源位置",
    "confidence": 0.0-1.0
  }}
}}
"""),
            ("user", "用户查询：{query}\n\n需要提取的字段：{target_fields}\n\n文档内容：\n{context}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "query": query,
            "target_fields": ", ".join(target_fields),
            "context": context
        })
        
        return self._parse_llm_response(response.content, results)
    
    def _format_retrieval_results(self, results: List[RetrievalResult]) -> str:
        formatted = []
        for i, result in enumerate(results):
            location = f"{result.metadata.get('source', 'unknown')}"
            if result.metadata.get('chapter'):
                location += f" - {result.metadata.get('chapter')}"
            if result.metadata.get('section'):
                location += f" - {result.metadata.get('section')}"
            
            formatted.append(f"[文档片段 {i+1}]\n位置：{location}\n内容：{result.content}\n")
        
        return "\n".join(formatted)
    
    def _parse_llm_response(self, response_text: str, 
                          results: List[RetrievalResult]) -> Dict[str, ExtractedField]:
        import json
        try:
            json_str = response_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str)
            
            extracted = {}
            for field_name, field_data in data.items():
                source_metadata = {}
                for result in results:
                    if field_data.get("source_text", "") in result.content:
                        source_metadata = result.metadata
                        break
                
                extracted[field_name] = ExtractedField(
                    field_name=field_name,
                    value=field_data.get("value", ""),
                    source_text=field_data.get("source_text", ""),
                    source_metadata=source_metadata,
                    confidence=field_data.get("confidence", 0.8)
                )
            
            return extracted
        except Exception as e:
            return {}
    
    def extract_waiting_period(self, results: List[RetrievalResult]) -> Optional[ExtractedField]:
        fields = self.extract_by_rules(results, "waiting_period")
        if fields:
            return fields[0]
        return None
    
    def extract_exclusions(self, results: List[RetrievalResult]) -> List[ExtractedField]:
        return self.extract_by_rules(results, "exclusions")
