"""
智能体编排器模块
功能：使用LangGraph构建多步工作流，实现意图分析、检索、数据提取、合规检查和回复生成的完整流程
特点：模块化设计，可扩展的节点结构，支持复杂的保险查询处理
"""
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
    """智能体工作流状态类型定义"""
    user_query: str                  # 用户原始查询
    conversation_id: str             # 对话ID
    intent: Dict[str, Any]           # 意图分析结果
    retrieval_results: List[RetrievalResult]  # 检索结果
    extracted_fields: List[ExtractedField]    # 提取的字段数据
    compliance_results: List[ComplianceCheckResult]  # 合规检查结果
    final_response: str              # 最终生成的回复
    error: str                       # 错误信息


class InsuranceAgentOrchestrator:
    """保险智能体编排器，基于LangGraph构建多步处理工作流"""
    def __init__(self, openai_api_key: str, retrieval_engine: InsuranceRetrievalEngine,
                 session_manager: SessionManager, model: str = "doubao-seed-2.0-pro",
                 base_url: str = "https://ark.cn-beijing.volces.com/api/coding/v3"):
        """
        初始化智能体编排器
        :param openai_api_key: API密钥
        :param retrieval_engine: 检索引擎实例
        :param session_manager: 会话管理器实例
        :param model: LLM模型名称
        :param base_url: API端点地址
        """
        self.llm = ChatOpenAI(model=model, api_key=openai_api_key, base_url=base_url, temperature=0)
        self.retrieval_engine = retrieval_engine
        self.data_extractor = InsuranceDataExtractor(openai_api_key, model, base_url)
        self.compliance_checker = InsuranceComplianceChecker()
        self.session_manager = session_manager
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """
        构建LangGraph工作流
        工作流节点：意图分析 -> 检索 -> 数据提取 -> 合规检查 -> 回复生成
        :return: 编译后的工作流实例
        """
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
        """
        意图分析节点：分析用户查询的意图类型和目标字段
        :param state: 当前工作流状态
        :return: 更新后的状态
        """
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
        """
        检索节点：根据用户查询执行混合检索
        :param state: 当前工作流状态
        :return: 更新后的状态
        """
        query = state["user_query"]
        results = self.retrieval_engine.hybrid_search(query, top_k=10)
        state["retrieval_results"] = results
        return state
    
    def _data_extraction_node(self, state: AgentState) -> AgentState:
        """
        数据提取节点：从检索结果中提取用户需要的保险字段信息
        优先使用LLM提取，失败则使用规则提取
        :param state: 当前工作流状态
        :return: 更新后的状态
        """
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
        """
        合规检查节点：检查提取的字段是否符合合规要求（原文一致性、来源可追溯）
        :param state: 当前工作流状态
        :return: 更新后的状态
        """
        fields = state["extracted_fields"]
        check_results = self.compliance_checker.check_all_fields(fields)
        state["compliance_results"] = check_results
        return state
    
    def _response_generation_node(self, state: AgentState) -> AgentState:
        """
        回复生成节点：根据提取的字段和检查结果生成最终回复
        如果有提取到字段，直接结构化展示；否则使用LLM基于检索结果生成回复
        :param state: 当前工作流状态
        :return: 更新后的状态
        """
        query = state["user_query"]
        fields = state["extracted_fields"]
        check_results = state["compliance_results"]
        
        response_parts = ["根据保险文档查询结果：\n"]
        
        for i, field in enumerate(fields):
            source_info = ""
            if i < len(check_results):
                check_result = check_results[i]
                source_info = self.compliance_checker.format_source_info(field.source_metadata)
                if not check_result.source_traceable:
                    source_info = "来源信息不完整"
            else:
                source_info = self.compliance_checker.format_source_info(field.source_metadata)
            
            response_parts.append(f"【{self._format_field_name(field.field_name)}】")
            response_parts.append(f"{field.value}")
            response_parts.append(f"来源：{source_info}")
            response_parts.append("")
        
        if len(fields) == 0:
            context_parts = []
            for result in state.get("retrieval_results", []):
                context_parts.append(result.content)
            context = "\n\n".join(context_parts) if context_parts else "未找到相关文档内容"
            
            try:
                prompt = SystemMessage(content="""你是一名专业的保险条款咨询助手。请根据提供的保险文档内容，准确回答用户的问题。
要求：
1. 仅基于提供的文档内容回答，不做任何演绎
2. 引用原文关键语句
3. 如果文档中没有相关信息，明确告知""")
                response = self.llm.invoke([prompt, HumanMessage(content=f"用户问题：{query}\n\n文档内容：\n{context}")])
                state["final_response"] = response.content
            except Exception as e:
                state["final_response"] = f"未查询到相关保险信息。错误：{e}"
        else:
            final_response = "\n".join(response_parts)
            state["final_response"] = final_response
        
        if state.get("conversation_id"):
            self.session_manager.add_turn(
                state["conversation_id"],
                query,
                state["final_response"]
            )
        
        return state
    
    def _format_field_name(self, field_name: str) -> str:
        """
        将英文字段名转换为中文显示名称
        :param field_name: 英文字段名
        :return: 中文字段名
        """
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
        """
        执行完整的智能体工作流
        :param user_query: 用户查询
        :param conversation_id: 可选的对话ID，用于多轮对话
        :return: 最终回复文本
        """
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
