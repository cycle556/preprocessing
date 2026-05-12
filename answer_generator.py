"""
保险文档 Agentic RAG 系统 - 答案生成器
功能：基于召回片段生成精准答案，无幻觉、可溯源、格式友好，
     必须支持原文引用+来源标注（文件名+页码+段落）。
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from vector_store import SearchResult
from logger import get_logger

logger = get_logger()


@dataclass
class GeneratedAnswer:
    """生成的答案数据结构"""
    answer: str
    citations: List[Dict[str, str]]
    confidence: str
    source_count: int


class AnswerGenerator:
    """
    保险问答答案生成器
    严格基于检索结果生成答案，附带原文引用和来源标注
    """

    CITATION_TEMPLATE = "【来源：{company_name} | {file_name} 第{page}页 {chapter}{section}】"

    SYSTEM_PROMPT = """你是一名专业的保险条款咨询专家。你必须严格按照以下规则回答问题：

【核心规则】
1. 基于提供的文档内容回答问题，不要编造文档中不存在的信息
2. 文档内容可能与用户提问使用不同表述（如繁体/简体差异、同义词、上下位词），只要文档内容与问题主题相关，就应当总结并回答
3. 如果用户问题提到某公司（如友邦、保诚、宏利等），而文档内容涉及同类保险产品或投保流程，即使未直接出现该公司名称，也应基于文档内容提供相关信息
4. 如果文档内容确实与问题完全无关，才告知"根据现有文档，未找到相关信息"
5. 每条关键信息必须附带来源引用，格式为：【来源：公司名 | 文件名 第X页 章节名】
6. 引用时优先使用原文表述，必要时可将繁体转简体以便阅读

【回答格式要求】
- 先用简洁的语言直接回答问题
- 然后列出关键信息点，每条附带来源引用
- 如果有免责条款等敏感内容，提醒用户仔细阅读原文
- 使用清晰的分段和编号

【特殊场景处理】
- 等待期类：明确说明天数、起算时间、适用条件
- 免责条款：逐条列出，标注完整原文
- 理赔条件：说明必须满足的全部条件
- 对比问题：用表格或分条对比"""

    def __init__(self, api_key: str, base_url: str,
                 model: str = "doubao-seed-2.0-pro",
                 temperature: float = 0.0,
                 max_tokens: int = 2048,
                 require_citation: bool = True,
                 fallback_message: str = ""):
        """
        Args:
            api_key: API 密钥
            base_url: API 基础 URL
            model: 模型名称
            temperature: 生成温度（0=最确定）
            max_tokens: 最大生成 token 数
            require_citation: 是否强制要求引用
            fallback_message: 无结果时的回退消息
        """
        self.require_citation = require_citation
        self.fallback_message = fallback_message or \
            "抱歉，在现有保险文档中未找到相关信息。请尝试调整查询条件或联系客服获取更多帮助。"

        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.info(f"AnswerGenerator 初始化: model={model}, "
                     f"temperature={temperature}, citation={require_citation}")

    def generate(self, query: str, search_results: List[SearchResult],
                 conversation_history: Optional[List[Dict[str, str]]] = None) -> GeneratedAnswer:
        """
        基于检索结果生成答案

        Args:
            query: 用户查询
            search_results: 检索结果列表
            conversation_history: 历史对话记录

        Returns:
            生成的答案
        """
        if not search_results:
            return GeneratedAnswer(
                answer=self.fallback_message,
                citations=[],
                confidence="low",
                source_count=0,
            )

        context = self._build_context(search_results)
        citations = self._extract_citations(search_results)

        messages = [SystemMessage(content=self.SYSTEM_PROMPT)]

        if conversation_history:
            for turn in conversation_history[-3:]:
                messages.append(HumanMessage(content=turn.get("question", "")))
                messages.append(AIMessage(content=turn.get("answer", "")))
        user_prompt = f"""用户问题：{query}

请基于以下保险文档内容回答问题。注意：文档可能使用繁体中文，与问题的简体中文表述不同，但内容相关即可。

{context}

回答要求：
1. 如果文档内容与问题主题相关，请总结文档中的关键信息来回答
2. 即使文档中没有直接出现用户提到的公司名/产品名，但内容涉及同类事项（如投保流程、缴费方式、保险责任等），也应据实回答
3. 每条信息标注来源（文件名、页码、章节）
4. 不要编造文档中没有的信息"""

        messages.append(HumanMessage(content=user_prompt))

        try:
            response = self.llm.invoke(messages)
            answer_text = response.content
            if not answer_text or not answer_text.strip():
                logger.warning("LLM 返回空响应，使用 fallback")
                answer_text = self._build_fallback_answer(search_results)
        except Exception as e:
            logger.error(f"LLM 生成答案失败: {e}")
            answer_text = self._build_fallback_answer(search_results)

        confidence = "high" if len(search_results) >= 3 else "medium" if search_results else "low"

        logger.info(f"答案生成完成: confidence={confidence}, "
                     f"sources={len(search_results)}, length={len(answer_text)}")
        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            source_count=len(search_results),
        )

    def _build_context(self, results: List[SearchResult]) -> str:
        """构建 LLM 上下文"""
        parts = []
        for i, r in enumerate(results):
            file_name = r.metadata.get("file_name", "未知文档")
            source = r.metadata.get("source", "")
            company = r.metadata.get("company_name", "")
            page = r.metadata.get("page_number", "N/A")
            chapter = r.metadata.get("chapter", "")
            section = r.metadata.get("section", "")

            # 拼接可读的文档路径：公司名 / 相对路径
            display_path = file_name
            if company:
                display_path = f"{company}/{file_name}"
            elif source:
                m = re.search(r'保司文件2\.0[/\\](.+?)(?:[/\\][^/\\]+\.md)?$', source)
                if m:
                    display_path = m.group(1).replace('\\', '/')
                else:
                    display_path = source.replace('\\', '/')

            header = f"[文档片段 {i + 1}]"
            header += f" 文件: {display_path}"
            if page:
                header += f" | 第{page}页"
            if chapter:
                header += f" | {chapter}"
            if section:
                header += f" | {section}"

            parts.append(f"{header}\n```\n{r.content}\n```")

        return "\n\n".join(parts)

    def _extract_citations(self, results: List[SearchResult]) -> List[Dict[str, str]]:
        """提取来源引用信息"""
        citations = []
        for r in results:
            citations.append({
                "company_name": r.metadata.get("company_name", ""),
                "file_name": r.metadata.get("file_name", "未知文档"),
                "source": r.metadata.get("source", ""),
                "page": str(r.metadata.get("page_number", "N/A")),
                "chapter": r.metadata.get("chapter", ""),
                "section": r.metadata.get("section", ""),
                "snippet": r.content[:100],
                "score": f"{r.score:.2f}",
            })
        return citations

    def _build_fallback_answer(self, results: List[SearchResult]) -> str:
        """当 LLM 调用失败时，用检索结果直接构建答案"""
        if not results:
            return self.fallback_message

        parts = ["根据保险文档查询结果：\n"]
        for i, r in enumerate(results[:5]):
            file_name = r.metadata.get("file_name", "未知文档")
            company = r.metadata.get("company_name", "")
            page = r.metadata.get("page_number", "")
            chapter = r.metadata.get("chapter", "")

            parts.append(f"【相关内容 {i + 1}】")
            parts.append(r.content[:300])
            if page:
                parts.append(self.CITATION_TEMPLATE.format(
                    company_name=company, file_name=file_name,
                    page=page, chapter=chapter, section=""
                ))
            parts.append("")

        return "\n".join(parts)
