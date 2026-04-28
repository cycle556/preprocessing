"""
保险文档 Agentic RAG 系统 - 查询增强器
功能：自动识别用户问题类型（条款查询、理赔咨询、责任认定、对比提问等），
     对模糊问题自动扩写、改写、关键词提取，提升检索精度。
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import jieba
import jieba.analyse

from logger import get_logger

logger = get_logger()


@dataclass
class EnhancedQuery:
    """增强后的查询数据结构"""
    original_query: str
    intent_type: str
    expanded_queries: List[str]
    keywords: List[str]
    entities: Dict[str, str]
    clarification_needed: List[str]


class QueryEnhancer:
    """
    查询增强器
    综合意图识别、关键词提取、查询改写等功能
    """

    INTENT_PATTERNS = {
        "条款查询": [
            r"(条款|规定|约定|说明|定义)",
            r"(什么|哪些|多少|怎么|如何).*(保|赔|免|等)",
        ],
        "理赔咨询": [
            r"(理赔|索赔|报案|赔付|给付|赔偿)",
            r"(能不能赔|可以赔吗|怎么赔|赔多少)",
        ],
        "责任认定": [
            r"(保险责任|保障责任|覆盖|包含|包括)",
            r"(是否|是不是|属于).*(责任|范围|保障)",
        ],
        "免责查询": [
            r"(免责|除外|不赔|不予|不承担)",
            r"(责任免除|免责条款|除外责任)",
        ],
        "数据提取": [
            r"(等待期|保额|免赔额|保费|费率|比例)",
            r"(多少钱|多少天|多少年|多久)",
        ],
        "对比提问": [
            r"(对比|比较|区别|差异|不同)",
            r"(哪个好|哪个更|有什么区别)",
        ],
    }

    INSURANCE_ENTITIES = {
        "product_name": r"([\u4e00-\u9fff]+(?:保险|险|医疗险|重疾险|寿险|意外险|车险))",
        "insurance_type": r"(重疾|医疗|意外|寿险|年金|车险|财产)",
        "time_period": r"(\d+[天日月年]|[一二三]十年|[一二三]个月)",
        "money_amount": r"(\d+[万亿千百]?\s*(?:元|块))",
    }

    def __init__(self, expand_query: bool = True, expansion_terms: int = 3):
        """
        Args:
            expand_query: 是否启用查询扩写
            expansion_terms: 扩写时额外添加的关键词数量
        """
        self.expand_query = expand_query
        self.expansion_terms = expansion_terms

        jieba.setLogLevel(jieba.logging.INFO)
        logger.info(f"QueryEnhancer 初始化: expand={expand_query}, terms={expansion_terms}")

    def enhance(self, query: str) -> EnhancedQuery:
        """
        对用户查询进行全方位增强

        Args:
            query: 用户原始查询

        Returns:
            增强后的查询结构
        """
        if not query or not query.strip():
            return EnhancedQuery(
                original_query=query,
                intent_type="unknown",
                expanded_queries=[],
                keywords=[],
                entities={},
                clarification_needed=["查询内容为空，请输入您的问题"]
            )

        intent_type = self._detect_intent(query)
        keywords = self._extract_keywords(query)
        entities = self._extract_entities(query)
        clarification = self._check_clarification(query, intent_type, entities)
        expanded = self._expand_query(query, keywords) if self.expand_query else [query]

        enhanced = EnhancedQuery(
            original_query=query,
            intent_type=intent_type,
            expanded_queries=expanded,
            keywords=keywords,
            entities=entities,
            clarification_needed=clarification,
        )

        logger.debug(f"查询增强: intent={intent_type}, keywords={keywords}, "
                     f"expanded={len(expanded)}")
        return enhanced

    def _detect_intent(self, query: str) -> str:
        """识别用户问题意图类型"""
        scores = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, query)
                score += len(matches)
            scores[intent] = score

        if scores:
            best = max(scores, key=scores.get)
            if scores[best] > 0:
                return best
        return "综合咨询"

    def _extract_keywords(self, query: str) -> List[str]:
        """提取查询关键词"""
        stop_words = {"的", "了", "是", "在", "和", "吗", "呢", "啊", "吧",
                      "什么", "怎么", "如何", "哪些", "多少", "有没有",
                      "这个", "那个", "一个", "一下", "请问", "帮我", "我想"}

        keywords = jieba.analyse.extract_tags(
            query, topK=10, withWeight=False,
            allowPOS=('n', 'v', 'a', 'nr', 'ns', 'nt')
        )

        keywords = [kw for kw in keywords if kw not in stop_words and len(kw) > 1]

        insurance_terms = []
        for kw in keywords:
            if any(term in kw for term in ["保险", "赔", "免", "保", "费", "责", "等"]):
                insurance_terms.append(kw)

        final_keywords = insurance_terms + [kw for kw in keywords if kw not in insurance_terms]
        return final_keywords[:8]

    def _extract_entities(self, query: str) -> Dict[str, str]:
        """提取保险相关实体"""
        entities = {}
        for entity_name, pattern in self.INSURANCE_ENTITIES.items():
            matches = re.findall(pattern, query)
            if matches:
                entities[entity_name] = matches[0] if isinstance(matches[0], str) else matches[0][0]
        return entities

    def _check_clarification(self, query: str, intent: str,
                            entities: Dict[str, str]) -> List[str]:
        """检查是否需要向用户澄清问题"""
        clarifications = []

        if len(query) < 5:
            clarifications.append("问题过于简短，请提供更多细节")

        if intent in ["条款查询", "免责查询", "数据提取"] and "product_name" not in entities:
            pass

        return clarifications

    def _expand_query(self, query: str, keywords: List[str]) -> List[str]:
        """查询扩写：生成多个变体以提高检索召回率"""
        expanded = [query]

        if keywords:
            query_with_kw = f"{query} {' '.join(keywords[:self.expansion_terms])}"
            expanded.append(query_with_kw)

        insurance_prefixes = ["保险条款中关于", "请查找保险合同中与"]
        insurance_suffixes = ["的规定", "的相关条款", "的具体内容"]

        core = re.sub(r'^(请问|帮我查|帮我找|我想知道|我想了解)\s*', '', query)
        core = re.sub(r'[？?！!。.]$', '', core)

        if len(core) > 3:
            for prefix in insurance_prefixes[:1]:
                expanded.append(f"{prefix}{core}")
            for suffix in insurance_suffixes[:1]:
                expanded.append(f"{core}{suffix}")

        seen = set()
        unique_expanded = []
        for q in expanded:
            if q not in seen:
                seen.add(q)
                unique_expanded.append(q)

        return unique_expanded

    def generate_search_queries(self, enhanced: EnhancedQuery) -> List[str]:
        """生成用于检索的查询列表"""
        queries = [enhanced.original_query]
        queries.extend(enhanced.expanded_queries[1:])
        if enhanced.keywords:
            queries.append(" ".join(enhanced.keywords))
        return queries
