from typing import Dict, Any, List, TypedDict, Annotated, Sequence
from dataclasses import dataclass
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from document_processor import DocumentChunk
from retrieval_engine import RetrievalResult, InsuranceRetrievalEngine
from data_extractor import InsuranceDataExtractor, ExtractedField
from compliance_checker import InsuranceComplianceChecker, ComplianceCheckResult
from session_manager import SessionManager


class AgentState(TypedDict):
    user_query: str
    conversation_id: str
    intent: Dict[str, Any]
    retrieval_results: List[RetrievalResult]
    extracted_fields: List[ExtractedField]
    compliance_results: List[ComplianceCheckResult]
    final_response: str
    error: str


class InsuranceAgentOrchestrator:
    def __init__(self, openai_api_key: str, retrieval_engine: InsuranceRetrievalEngine,
                 session_manager: SessionManager, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, api_key=openai_api_key, temperature=0)
        self.retrieval_engine = retrieval_engine
        self.data_extractor = InsuranceDataExtractor(openai_api_key, model)
        self.compliance_checker = InsuranceComplianceChecker()
        self.session_manager = session_manager
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("intent_analysis", self._intent_analysis_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("data_extraction", self._data_extraction_node)
        workflow.add_node("compliance_check", self._compliance_check_node)
        workflow.add_node("response_generation", self._response_generation_node)
        
        workflow.set_entry_point("intent_analysis")
        workflow.add_edge("intent_analysis", "retrieval")
        workflow.add_edge("retrieval", "data_extraction")
        workflow.add_edge("data_extraction", "compliance_check")
        workflow.add_edge("compliance_check", "response_generation")
        workflow.add_edge("response_generation", END)
        
        return workflow.compile()
    
    def _intent_analysis_node(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        
        prompt = SystemMessage(content="""你是一名专业的保险查询意图分析师。请分析用户的保险查询，输出JSON格式的意图分析结果。

意图分类：
- data_extraction: 数据提取（等待期、保额、免赔额、免责条款等）
- clause_inquiry: 条款咨询
- compliance_inquiry: 合规咨询

输出格式：
{
  "intent_type": "data_extraction/clause_inquiry/compliance_inquiry",
  "target_fields": ["waiting_period", "sum_insured", "deductible", "exclusions"],
  "entities": {"product_name": "", "insurance_type": ""},
  "clarification_needed": []
}

仅输出JSON，不要其他内容。
""")
        
        response = self.llm.invoke([prompt, HumanMessage(content=query)])
        
        try:
            intent = json.loads(response.content)
        except:
            intent = {
                "intent_type": "data_extraction",
                "target_fields": [],
                "entities": {},
                "clarification_needed": []
            }
        
        state["intent"] = intent
        return state
    
    def _retrieval_node(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        results = self.retrieval_engine.hybrid_search(query, top_k=5)
        state["retrieval_results"] = results
        return state
    
    def _data_extraction_node(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        intent = state["intent"]
        results = state["retrieval_results"]
        
        target_fields = intent.get("target_fields", [])
        
        if not target_fields:
            target_fields = ["waiting_period", "exclusions", "sum_insured", "deductible"]
        
        extracted_dict = self.data_extractor.extract_by_llm(results, query, target_fields)
        extracted_fields = list(extracted_dict.values())
        
        if not extracted_fields:
            for field_name in target_fields:
                fields = self.data_extractor.extract_by_rules(results, field_name)
                extracted_fields.extend(fields)
        
        state["extracted_fields"] = extracted_fields
        return state
    
    def _compliance_check_node(self, state: AgentState) -> AgentState:
        fields = state["extracted_fields"]
        check_results = self.compliance_checker.check_all_fields(fields)
        state["compliance_results"] = check_results
        return state
    
    def _response_generation_node(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        fields = state["extracted_fields"]
        check_results = state["compliance_results"]
        
        response_parts = ["根据保险文档查询结果：\n"]
        
        for i, field in enumerate(fields):
            if i < len(check_results):
                check_result = check_results[i]
                if not check_result.overall_pass:
                    continue
            
            source_info = self.compliance_checker.format_source_info(field.source_metadata)
            
            response_parts.append(f"【{self._format_field_name(field.field_name)}】")
            response_parts.append(f"{field.value}")
            response_parts.append(f"来源：{source_info}")
            response_parts.append("")
        
        if not any(part for part in response_parts if part.strip()):
            response_parts = ["未查询到相关保险信息，请尝试补充产品名称或查询条件。"]
        
        final_response = "\n".join(response_parts)
        state["final_response"] = final_response
        
        if state.get("conversation_id"):
            self.session_manager.add_turn(
                state["conversation_id"],
                query,
                final_response
            )
        
        return state
    
    def _format_field_name(self, field_name: str) -> str:
        name_map = {
            "waiting_period": "等待期",
            "sum_insured": "保额",
            "deductible": "免赔额",
            "payment_ratio": "赔付比例",
            "coverage_period": "保障期间",
            "premium_payment_period": "缴费期间",
            "exclusions": "免责事项",
            "coverage": "保险责任"
        }
        return name_map.get(field_name, field_name)
    
    def run(self, user_query: str, conversation_id: str = None) -> str:
        initial_state: AgentState = {
            "user_query": user_query,
            "conversation_id": conversation_id or self.session_manager.create_conversation(),
            "intent": {},
            "retrieval_results": [],
            "extracted_fields": [],
            "compliance_results": [],
            "final_response": "",
            "error": ""
        }
        
        result = self.workflow.invoke(initial_state)
        return result["final_response"]
