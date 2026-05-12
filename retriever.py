"""
保险文档 Agentic RAG 系统 - 检索引擎
功能：混合检索（语义+关键词）+ 重排序（Rerank），
     支持多查询融合、元数据过滤、相似度阈值控制。
"""
import re
from typing import List, Dict, Any, Optional

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from vector_store import BaseVectorStore, SearchResult, VectorDocument, EmbeddingProvider
from query_enhancer import EnhancedQuery
from logger import get_logger

logger = get_logger()


class InsuranceRetriever:
    """
    保险文档检索引擎
    组合语义检索和关键词检索，支持重排序和多查询融合
    """

    def __init__(self, vector_store: BaseVectorStore,
                 embedding_provider: EmbeddingProvider,
                 top_k: int = 10,
                 semantic_weight: float = 0.6,
                 keyword_weight: float = 0.4,
                 similarity_threshold: float = 0.3,
                 use_rerank: bool = True,
                 rerank_top_k: int = 5):
        """
        Args:
            vector_store: 向量数据库实例
            embedding_provider: 向量嵌入提供者
            top_k: 最终返回结果数
            semantic_weight: 语义检索权重
            keyword_weight: 关键词检索权重
            similarity_threshold: 相似度阈值，低于此值的结果将被过滤
            use_rerank: 是否启用重排序
            rerank_top_k: 重排序后保留的结果数
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.similarity_threshold = similarity_threshold
        self.use_rerank = use_rerank
        self.rerank_top_k = rerank_top_k

        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_documents: List[str] = []
        self._bm25_metadatas: List[Dict[str, Any]] = []
        self._bm25_ids: List[str] = []

        logger.info(f"InsuranceRetriever 初始化: top_k={top_k}, "
                     f"weights=({semantic_weight},{keyword_weight}), "
                     f"rerank={use_rerank}")

    def build_bm25_index(self, documents: List[VectorDocument]):
        """构建 BM25 关键词检索索引"""
        if not documents:
            return

        self._bm25_documents = [doc.content for doc in documents]
        self._bm25_metadatas = [doc.metadata for doc in documents]
        self._bm25_ids = [doc.id for doc in documents]

        tokenized = [jieba.lcut(doc) for doc in self._bm25_documents]
        self._bm25_index = BM25Okapi(tokenized)

        logger.info(f"BM25 索引构建完成: {len(documents)} 条文档")

    def search(self, enhanced_query: EnhancedQuery,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        执行混合检索

        Args:
            enhanced_query: 增强后的查询
            filter_metadata: 元数据过滤条件

        Returns:
            检索结果列表
        """
        all_results: Dict[str, SearchResult] = {}

        # 从查询中识别公司名
        company_name = enhanced_query.entities.get("company_name")
        if company_name:
            logger.info(f"检测到公司名过滤: {company_name}")

        search_queries = enhanced_query.expanded_queries[:2]

        for query in search_queries:
            query_embedding = self.embedding_provider.embed_single(query)
            # 不在 ChromaDB 层做公司过滤，改为在 Python 层后过滤（更可靠）
            semantic_results = self.vector_store.search(
                query_embedding, top_k=self.top_k * 2,
            )
            for r in semantic_results:
                # Python 层公司过滤
                if company_name and r.metadata.get("company_name") != company_name:
                    continue
                key = r.id or r.content[:50]
                if key not in all_results or r.score > all_results[key].score:
                    all_results[key] = r

            if self._bm25_index:
                keyword_results = self._keyword_search(query, top_k=self.top_k)
                for r in keyword_results:
                    if company_name and r.metadata.get("company_name") != company_name:
                        continue

                    key = r.id or r.content[:50]
                    if key in all_results:
                        all_results[key].score = (
                            all_results[key].score * self.semantic_weight +
                            r.score * self.keyword_weight
                        )
                        all_results[key].retrieval_type = "hybrid"
                    else:
                        r.score = r.score * self.keyword_weight
                        r.retrieval_type = "hybrid"
                        all_results[key] = r

        results = [r for r in all_results.values()
                    if r.score >= self.similarity_threshold]
        results.sort(key=lambda x: x.score, reverse=True)

        if self.use_rerank and len(results) > self.rerank_top_k:
            results = self._rerank(results, enhanced_query.original_query)

        results = results[:self.top_k]

        if not results:
            logger.info("检索无结果")
            return []

        logger.debug(f"检索完成: {len(results)} 条结果, "
                     f"最高分={results[0].score:.4f}")
        return results

    def search_simple(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """简化版检索（不需要 EnhancedQuery）"""
        query_embedding = self.embedding_provider.embed_single(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)

        if self._bm25_index:
            keyword_results = self._keyword_search(query, top_k=top_k)
            result_map = {r.id or r.content[:50]: r for r in results}
            for r in keyword_results:
                key = r.id or r.content[:50]
                if key in result_map:
                    result_map[key].score = max(result_map[key].score, r.score)
                    result_map[key].retrieval_type = "hybrid"
                else:
                    result_map[key] = r
            results = sorted(result_map.values(), key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def _keyword_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """BM25 关键词检索"""
        if not self._bm25_index:
            return []

        tokenized = jieba.lcut(query)
        scores = self._bm25_index.get_scores(tokenized)

        max_score = max(scores) if max(scores) > 0 else 1
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append(SearchResult(
                id=self._bm25_ids[idx] if idx < len(self._bm25_ids) else "",
                content=self._bm25_documents[idx],
                metadata=self._bm25_metadatas[idx] if idx < len(self._bm25_metadatas) else {},
                score=float(scores[idx] / max_score),
                retrieval_type="keyword"
            ))

        return results

    def _rerank(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        重排序：基于查询与文档的多维度相关性进行二次排序
        使用启发式规则：关键词匹配度 + 语义相似度混合打分
        """
        query_keywords = set(jieba.lcut(query))
        query_keywords.discard("的")

        for result in results:
            content_words = set(jieba.lcut(result.content))

            keyword_overlap = len(query_keywords & content_words)
            keyword_density = keyword_overlap / max(len(query_keywords), 1)

            title_boost = 0.0
            chapter = result.metadata.get("chapter", "")
            if any(kw in chapter for kw in query_keywords if len(kw) > 1):
                title_boost = 0.15

            citation_boost = 0.0
            if re.search(r'第[一二三四五六七八九十]+条', result.content):
                citation_boost = 0.05

            result.score = (
                result.score * 0.5 +
                keyword_density * 0.3 +
                title_boost +
                citation_boost
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:self.rerank_top_k]

    def format_context(self, results: List[SearchResult]) -> str:
        """格式化检索结果为 LLM 可用的上下文字符串"""
        if not results:
            return "暂无相关文档内容"

        parts = []
        for i, r in enumerate(results):
            source = r.metadata.get("file_name", "未知文档")
            page = r.metadata.get("page_number", "N/A")
            chapter = r.metadata.get("chapter", "")
            section = r.metadata.get("section", "")

            header = f"[片段 {i + 1}] 来源: {source}"
            if page:
                header += f" | 第{page}页"
            if chapter:
                header += f" | {chapter}"
            if section:
                header += f" | {section}"

            parts.append(f"{header}\n{r.content}")

        return "\n\n---\n\n".join(parts)
