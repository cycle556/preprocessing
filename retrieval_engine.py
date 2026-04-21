"""
检索引擎模块
功能：提供混合检索能力，结合语义检索和关键词检索，支持保险文档的精准检索
特点：使用火山引擎Embedding模型，结合BM25关键词检索，提供加权混合检索结果
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import chromadb
import os
from openai import OpenAI
from rank_bm25 import BM25Okapi
import jieba
import numpy as np
from document_processor import DocumentChunk, TableData


@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    content: str          # 检索到的文本内容
    metadata: Dict[str, Any]  # 元数据（来源、章节、条款等）
    score: float          # 检索相关度得分
    retrieval_type: str   # 检索类型：semantic/keyword/hybrid


class VolcengineEmbeddingFunction(chromadb.EmbeddingFunction):
    """火山引擎Embedding函数，适配ChromaDB接口"""
    def __init__(self, api_key: str, base_url: str = "https://ark.cn-beijing.volces.com/api/coding/v3", model: str = "doubao-embedding-vision"):
        """
        初始化Embedding函数
        :param api_key: API密钥
        :param base_url: API端点地址
        :param model: Embedding模型名称
        """
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def name(self) -> str:
        """返回Embedding函数名称"""
        return f"volcengine_{self._model}"

    def default_embedding_function(self):
        """返回默认Embedding函数实例"""
        return self

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        调用Embedding API生成向量
        :param input: 输入文本列表
        :return: 向量列表
        """
        embeddings = []
        batch_size = 4
        for i in range(0, len(input), batch_size):
            batch = input[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self._model,
                    input=batch
                )
                for item in response.data:
                    embeddings.append(item.embedding)
            except Exception as e:
                print(f"Embedding API调用失败: {e}")
                for _ in batch:
                    embeddings.append([0.0] * 1024)
        return embeddings


class InsuranceRetrievalEngine:
    """保险文档检索引擎，支持语义检索、关键词检索和混合检索"""
    def __init__(self, persist_directory: str = "./chroma_db",
                 api_key: str = None,
                 base_url: str = "https://ark.cn-beijing.volces.com/api/coding/v3",
                 embedding_model: str = "doubao-embedding-vision"):
        """
        初始化检索引擎
        :param persist_directory: 向量数据库持久化目录
        :param api_key: API密钥
        :param base_url: API端点地址
        :param embedding_model: Embedding模型名称
        """
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        api_key = api_key or os.getenv("VOLC_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        self.embedding_function = VolcengineEmbeddingFunction(
            api_key=api_key,
            base_url=base_url,
            model=embedding_model
        )
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_metadatas = []
        
    def create_collection(self, collection_name: str = "insurance_docs"):
        """
        创建或获取向量数据库集合
        :param collection_name: 集合名称，默认insurance_docs
        :return: 集合对象
        """
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "保险文档检索集合"}
            )
        except Exception:
            self.client.delete_collection(name=collection_name)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "保险文档检索集合"}
            )
        return self.collection
    
    def add_documents(self, chunks: List[DocumentChunk], tables: List[TableData] = None):
        """
        添加文档到向量数据库和BM25索引
        :param chunks: 文档分块列表
        :param tables: 表格数据列表
        """
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk.content)
            metadatas.append(chunk.metadata)
            ids.append(chunk.chunk_id)
        
        if tables:
            for table in tables:
                table_text = self._table_to_searchable_text(table)
                documents.append(table_text)
                metadatas.append(table.metadata)
                ids.append(table.table_id)
        
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            self._build_bm25_index(documents, metadatas)
    
    def _table_to_searchable_text(self, table: TableData) -> str:
        """
        将表格转换为可检索的文本格式
        :param table: 表格数据对象
        :return: 格式化的文本内容
        """
        text_parts = [f"表格: {table.metadata.get('sheet_name', 'unknown')}"]
        text_parts.append(table.df.to_string(index=False))
        return "\n".join(text_parts)
    
    def _build_bm25_index(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        构建BM25关键词检索索引
        :param documents: 文档列表
        :param metadatas: 元数据列表
        """
        tokenized_docs = [jieba.lcut(doc) for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.bm25_documents = documents
        self.bm25_metadatas = metadatas
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        语义检索，基于向量相似度匹配
        :param query: 用户查询
        :param top_k: 返回结果数量，默认5
        :return: 检索结果列表
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        retrieval_results = []
        for i in range(len(results['documents'][0])):
            retrieval_results.append(RetrievalResult(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                score=1 - results['distances'][0][i] if results['distances'] else 0.0,
                retrieval_type="semantic"
            ))
        
        return retrieval_results
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        关键词检索，基于BM25算法
        :param query: 用户查询
        :param top_k: 返回结果数量，默认5
        :return: 检索结果列表
        """
        if not self.bm25_index:
            return []
        
        tokenized_query = jieba.lcut(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        retrieval_results = []
        for idx in top_indices:
            if scores[idx] > 0:
                retrieval_results.append(RetrievalResult(
                    content=self.bm25_documents[idx],
                    metadata=self.bm25_metadatas[idx],
                    score=float(scores[idx]) / max(scores) if max(scores) > 0 else 0.0,
                    retrieval_type="keyword"
                ))
        
        return retrieval_results
    
    def hybrid_search(self, query: str, top_k: int = 5,
                     semantic_weight: float = 0.6,
                     keyword_weight: float = 0.4) -> List[RetrievalResult]:
        """
        混合检索，结合语义检索和关键词检索的结果
        :param query: 用户查询
        :param top_k: 返回结果数量，默认5
        :param semantic_weight: 语义检索权重，默认0.6
        :param keyword_weight: 关键词检索权重，默认0.4
        :return: 加权混合后的检索结果列表
        """
        semantic_results = self.semantic_search(query, top_k * 2)
        keyword_results = self.keyword_search(query, top_k * 2)
        
        combined_results = {}
        
        for result in semantic_results:
            doc_id = result.metadata.get('chunk_id') or result.metadata.get('table_id')
            if doc_id:
                combined_results[doc_id] = {
                    'result': result,
                    'semantic_score': result.score,
                    'keyword_score': 0.0
                }
        
        for result in keyword_results:
            doc_id = result.metadata.get('chunk_id') or result.metadata.get('table_id')
            if doc_id in combined_results:
                combined_results[doc_id]['keyword_score'] = result.score
            else:
                combined_results[doc_id] = {
                    'result': result,
                    'semantic_score': 0.0,
                    'keyword_score': result.score
                }
        
        final_results = []
        for doc_id, data in combined_results.items():
            hybrid_score = (data['semantic_score'] * semantic_weight + 
                          data['keyword_score'] * keyword_weight)
            result = data['result']
            result.score = hybrid_score
            result.retrieval_type = "hybrid"
            final_results.append(result)
        
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]
