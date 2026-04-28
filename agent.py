"""
保险文档 Agentic RAG 系统 - 主 Agent 编排器
功能：使用 LangGraph 构建多节点 Agent 工作流，
     串联查询理解→检索→重排序→答案生成全流程，
     支持多轮对话、异常恢复、日志追踪。
"""
from typing import Dict, Any, List, TypedDict
import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from document_loader import DocumentLoader, LoadedDocument
from text_splitter import InsuranceTextSplitter, TextChunk
from vector_store import (
    BaseVectorStore, VectorDocument, EmbeddingProvider,
    create_vector_store, SearchResult,
)
from query_enhancer import QueryEnhancer, EnhancedQuery
from retriever import InsuranceRetriever
from answer_generator import AnswerGenerator, GeneratedAnswer
from conversation_manager import ConversationManager

from logger import get_logger

logger = get_logger()


class AgentState(TypedDict):
    """Agent 工作流状态"""
    user_query: str
    conversation_id: str
    enhanced_query: Dict[str, Any]
    retrieval_results: List[Dict[str, Any]]
    generated_answer: Dict[str, Any]
    final_response: str
    error: str


class InsuranceRAGAgent:
    """
    保险文档 RAG Agent 编排器
    使用 LangGraph 构建完整的问答工作流
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 全局配置字典
        """
        self.config = config

        # 从环境变量覆盖腾讯云配置（优先级: 环境变量 > config.yaml）
        if vs_cfg := config.get("vector_store", {}):
            tc_cfg = vs_cfg.get("tencent_cloud", {})
            tc_cfg["host"] = self._get_env("TENCENT_VDB_HOST", tc_cfg.get("host", ""))
            tc_cfg["key"] = self._get_env("TENCENT_VDB_KEY", tc_cfg.get("key", ""))
            tc_cfg["username"] = self._get_env("TENCENT_VDB_USERNAME", tc_cfg.get("username", "root"))
            tc_cfg["database"] = self._get_env("TENCENT_VDB_DATABASE", tc_cfg.get("database", "insurance_db"))
            vs_cfg["tencent_cloud"] = tc_cfg

        embedding_cfg = config.get("embedding", {})
        llm_cfg = config.get("llm", {})
        retrieval_cfg = config.get("retrieval", {})
        vs_cfg = config.get("vector_store", {})
        conversation_cfg = config.get("conversation", {})
        answer_cfg = config.get("answer_generation", {})
        chunk_cfg = config.get("chunking", {})
        doc_cfg = config.get("document", {})

        base_url = self._get_env("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
        api_key = self._get_env("VOLC_API_KEY", "") or self._get_env("OPENAI_API_KEY", "")
        embedding_model = self._get_env("VOLC_EMBEDDING_MODEL", embedding_cfg.get("model", "doubao-embedding-vision"))
        llm_model = self._get_env("VOLC_LLM_MODEL", llm_cfg.get("model", "doubao-seed-2.0-pro"))

        self.embedding_provider = EmbeddingProvider(
            api_key=api_key,
            base_url=base_url,
            model=embedding_model,
            batch_size=embedding_cfg.get("batch_size", 4),
        )

        self.vector_store = create_vector_store(config, self.embedding_provider)
        self.vector_store.create_collection(
            vs_cfg.get("collection_name", "insurance_knowledge")
        )

        self.retriever = InsuranceRetriever(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            top_k=retrieval_cfg.get("top_k", 10),
            semantic_weight=retrieval_cfg.get("semantic_weight", 0.6),
            keyword_weight=retrieval_cfg.get("keyword_weight", 0.4),
            similarity_threshold=retrieval_cfg.get("similarity_threshold", 0.3),
            use_rerank=retrieval_cfg.get("use_rerank", True),
            rerank_top_k=retrieval_cfg.get("rerank_top_k", 5),
        )

        self.query_enhancer = QueryEnhancer(
            expand_query=retrieval_cfg.get("expand_query", True),
            expansion_terms=retrieval_cfg.get("expansion_terms", 3),
        )

        self.answer_generator = AnswerGenerator(
            api_key=api_key,
            base_url=base_url,
            model=llm_model,
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 2048),
            require_citation=answer_cfg.get("require_citation", True),
            fallback_message=answer_cfg.get("fallback_message", ""),
        )

        self.conversation_manager = ConversationManager(
            storage_dir=conversation_cfg.get("storage_dir", "./conversation_data"),
            max_history_turns=conversation_cfg.get("max_history_turns", 10),
            context_window_size=conversation_cfg.get("context_window_size", 5),
        )

        self.document_loader = DocumentLoader(
            source_dir=doc_cfg.get("source_dir", "./保司文件"),
            supported_formats=doc_cfg.get("supported_formats", [".pdf", ".txt"]),
            max_file_size_mb=doc_cfg.get("max_file_size_mb", 50),
            encoding=doc_cfg.get("encoding", "utf-8"),
        )

        self.text_splitter = InsuranceTextSplitter(
            chunk_size=chunk_cfg.get("chunk_size", 800),
            chunk_overlap=chunk_cfg.get("chunk_overlap", 100),
            min_chunk_size=chunk_cfg.get("min_chunk_size", 50),
            separators=chunk_cfg.get("separators"),
            semantic_headings=chunk_cfg.get("semantic_headings"),
        )

        self.workflow = self._build_workflow()
        self._initialized = False

        logger.info("InsuranceRAGAgent 初始化完成")

    def _build_workflow(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        workflow = StateGraph(AgentState)

        workflow.add_node("query_analysis", self._query_analysis_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("answer_generation", self._answer_generation_node)
        workflow.add_node("error_handler", self._error_handler_node)

        workflow.set_entry_point("query_analysis")
        workflow.add_edge("query_analysis", "retrieval")
        workflow.add_edge("retrieval", "answer_generation")
        workflow.add_edge("answer_generation", END)

        return workflow.compile()

    def _query_analysis_node(self, state: AgentState) -> AgentState:
        """查询分析节点：意图识别 + 查询增强"""
        try:
            query = state["user_query"]
            enhanced = self.query_enhancer.enhance(query)
            state["enhanced_query"] = {
                "original_query": enhanced.original_query,
                "intent_type": enhanced.intent_type,
                "expanded_queries": enhanced.expanded_queries,
                "keywords": enhanced.keywords,
                "entities": enhanced.entities,
            }
            logger.info(f"查询分析: intent={enhanced.intent_type}")
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            state["error"] = str(e)
        return state

    def _retrieval_node(self, state: AgentState) -> AgentState:
        """检索节点：多路召回 + 重排序"""
        try:
            enhanced_data = state.get("enhanced_query", {})
            enhanced = EnhancedQuery(
                original_query=enhanced_data.get("original_query", state["user_query"]),
                intent_type=enhanced_data.get("intent_type", "综合咨询"),
                expanded_queries=enhanced_data.get("expanded_queries", [state["user_query"]]),
                keywords=enhanced_data.get("keywords", []),
                entities=enhanced_data.get("entities", {}),
                clarification_needed=[],
            )

            results = self.retriever.search(enhanced)
            state["retrieval_results"] = [
                {
                    "id": r.id,
                    "content": r.content,
                    "metadata": r.metadata,
                    "score": r.score,
                    "retrieval_type": r.retrieval_type,
                }
                for r in results
            ]
            logger.info(f"检索完成: {len(results)} 条结果")
        except Exception as e:
            logger.error(f"检索失败: {e}")
            state["error"] = str(e)
            state["retrieval_results"] = []
        return state

    def _answer_generation_node(self, state: AgentState) -> AgentState:
        """答案生成节点"""
        try:
            query = state["user_query"]
            conv_id = state.get("conversation_id", "")

            raw_results = state.get("retrieval_results", [])
            search_results = [
                SearchResult(
                    id=r.get("id", ""),
                    content=r.get("content", ""),
                    metadata=r.get("metadata", {}),
                    score=r.get("score", 0.0),
                    retrieval_type=r.get("retrieval_type", "semantic"),
                )
                for r in raw_results
            ]

            history = self.conversation_manager.get_context_for_query(conv_id)

            generated = self.answer_generator.generate(query, search_results, history)
            state["generated_answer"] = {
                "answer": generated.answer,
                "citations": generated.citations,
                "confidence": generated.confidence,
                "source_count": generated.source_count,
            }
            state["final_response"] = generated.answer

            if conv_id:
                self.conversation_manager.add_turn(
                    conv_id, query, generated.answer,
                    citations=generated.citations,
                    intent=state.get("enhanced_query", {}).get("intent_type", ""),
                )
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            state["error"] = str(e)
            state["final_response"] = "系统处理您的查询时出现内部错误，请稍后重试。"
        return state

    def _error_handler_node(self, state: AgentState) -> AgentState:
        """错误处理节点"""
        state["final_response"] = (
            f"处理查询时遇到错误：{state.get('error', '未知错误')}。"
            "请检查文档是否已正确加载，或尝试重新表述您的问题。"
        )
        return state

    def query(self, user_query: str, conversation_id: str = None) -> str:
        """
        执行一次完整的问答流程

        Args:
            user_query: 用户查询文本
            conversation_id: 会话 ID（可选，用于多轮对话）

        Returns:
            系统生成的回答文本
        """
        if not user_query or not user_query.strip():
            return "请输入您的问题，我将为您查询保险文档中的相关信息。"

        conv_id = conversation_id or self.conversation_manager.create_conversation()

        initial_state: AgentState = {
            "user_query": user_query,
            "conversation_id": conv_id,
            "enhanced_query": {},
            "retrieval_results": [],
            "generated_answer": {},
            "final_response": "",
            "error": "",
        }

        try:
            result = self.workflow.invoke(initial_state)
            return result.get("final_response", "系统未能生成回答，请稍后重试。")
        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            return f"系统内部错误：{str(e)[:200]}。请稍后重试或联系技术支持。"

    def load_and_index_documents(self, source_dir: str = None, force_reload: bool = False) -> int:
        """
        加载「保司文件」目录中的文档并建立索引

        Args:
            source_dir: 可选，覆盖配置中的目录路径
            force_reload: 是否强制重新加载所有文档

        Returns:
            索引的文档块数量
        """
        if source_dir:
            from pathlib import Path
            self.document_loader.source_dir = Path(source_dir)

        logger.info("=" * 60)
        logger.info("开始加载文档并建立索引...")
        logger.info("=" * 60)

        vs_provider = self.config.get("vector_store", {}).get("provider", "chroma")
        logger.info(f"向量库类型: {vs_provider}")

        if force_reload:
            stats = self.vector_store.get_collection_stats()
            logger.info(f"强制重建索引，当前文档数: {stats.get('document_count', 0)}")
            self.vector_store.delete_by_filter({})

        documents = self.document_loader.load_all()
        if not documents:
            logger.warning("未发现任何文档文件")
            return 0

        chunks = self.text_splitter.split_documents(documents)
        if not chunks:
            logger.warning("文档分块结果为空")
            return 0

        existing_ids = set()
        if not force_reload:
            existing_ids = self._get_existing_chunk_ids()

        vector_docs = []
        skip_count = 0
        for chunk in chunks:
            if chunk.id in existing_ids:
                skip_count += 1
                continue
            vector_docs.append(VectorDocument(
                id=chunk.id,
                content=chunk.content,
                metadata=chunk.metadata,
            ))

        if skip_count > 0:
            logger.info(f"跳过 {skip_count} 个已存在的文档块（增量更新）")

        if vector_docs:
            success = self.vector_store.add_documents(vector_docs)
            if not success:
                logger.error("向量索引写入失败")
                return 0

        all_docs = vector_docs if vector_docs else [
            VectorDocument(id=c.id, content=c.content, metadata=c.metadata)
            for c in chunks
        ]
        self.retriever.build_bm25_index(all_docs)

        stats = self.vector_store.get_collection_stats()
        logger.info(f"索引建立完成: 新增 {len(vector_docs)} 块, "
                     f"跳过 {skip_count} 块, 总计 {stats.get('document_count', 0)} 块")
        return len(vector_docs)

    def _get_existing_chunk_ids(self) -> set:
        """
        获取已索引的文档块 ID 集合
        支持 ChromaDB（从本地集合查询）和腾讯云（从追踪文件恢复）
        """
        vs_provider = self.config.get("vector_store", {}).get("provider", "chroma")

        if vs_provider == "chroma":
            try:
                if hasattr(self.vector_store, '_collection') and self.vector_store._collection:
                    result = self.vector_store._collection.get()
                    if result and result.get("ids"):
                        return set(result["ids"])
            except Exception:
                pass

        if vs_provider == "tencent_cloud":
            return self._load_cloud_tracking_ids()

        return set()

    def _load_cloud_tracking_ids(self) -> set:
        """从本地追踪文件恢复已索引的 ID 集合（腾讯云模式下使用）"""
        import json
        from pathlib import Path

        tracking_dir = Path("./index_tracking")
        tracking_dir.mkdir(exist_ok=True)
        tracking_file = tracking_dir / "cloud_indexed_ids.json"

        if tracking_file.exists():
            try:
                with open(tracking_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ids = set(data.get("ids", []))
                logger.debug(f"从追踪文件恢复 {len(ids)} 个已索引 ID")
                return ids
            except Exception as e:
                logger.warning(f"追踪文件读取失败: {e}")

        return set()

    def _save_cloud_tracking_ids(self, ids: set):
        """保存已索引的 ID 集合到本地追踪文件"""
        import json
        from pathlib import Path

        tracking_dir = Path("./index_tracking")
        tracking_dir.mkdir(exist_ok=True)
        tracking_file = tracking_dir / "cloud_indexed_ids.json"

        try:
            with open(tracking_file, "w", encoding="utf-8") as f:
                json.dump({"ids": sorted(ids), "updated_at": ""}, f, ensure_ascii=False)
            logger.debug(f"追踪文件已更新: {len(ids)} 个 ID")
        except Exception as e:
            logger.warning(f"追踪文件写入失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        vs_stats = self.vector_store.get_collection_stats()
        conversations = self.conversation_manager.list_conversations()

        return {
            "system": self.config.get("system", {}).get("name", "InsuranceRAG"),
            "version": self.config.get("system", {}).get("version", "2.0.0"),
            "vector_store": vs_stats,
            "conversations": len(conversations),
            "provider": self.config.get("vector_store", {}).get("provider", "chroma"),
        }

    @staticmethod
    def _get_env(key: str, default: str = "") -> str:
        """从环境变量获取配置，优先 .env 文件"""
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        return os.getenv(key, default)
