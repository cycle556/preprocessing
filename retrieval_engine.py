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
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_type: str


class VolcengineEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key: str, base_url: str = "https://ark.cn-beijing.volces.com/api/coding/v3", model: str = "doubao-embedding-vision"):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def name(self) -> str:
        return f"volcengine_{self._model}"
    
    def default_embedding_function(self):
        return self
    
    def __call__(self, input: List[str]) -> List[List[float]]:
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
    def __init__(self, persist_directory: str = "./chroma_db", 
                 api_key: str = None,
                 base_url: str = "https://ark.cn-beijing.volces.com/api/coding/v3",
                 embedding_model: str = "doubao-embedding-vision"):
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
        text_parts = [f"表格: {table.metadata.get('sheet_name', 'unknown')}"]
        text_parts.append(table.df.to_string(index=False))
        return "\n".join(text_parts)
    
    def _build_bm25_index(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        tokenized_docs = [jieba.lcut(doc) for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.bm25_documents = documents
        self.bm25_metadatas = metadatas
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
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
