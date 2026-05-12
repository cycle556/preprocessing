"""
保险文档 Agentic RAG 系统 - 向量数据库模块
功能：提供向量数据库抽象层，支持 ChromaDB（本地）和腾讯云向量数据库（云端）无缝切换。
     支持批量写入、批量查询、元数据过滤检索、增量更新。
"""
import os
import hashlib
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from openai import OpenAI

from logger import get_logger

logger = get_logger()


@dataclass
class VectorDocument:
    """向量文档数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        clean_meta = {}
        for k, v in self.metadata.items():
            if v is None:
                clean_meta[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            elif isinstance(v, list):
                clean_meta[k] = str(v)
            elif isinstance(v, dict):
                clean_meta[k] = str(v)
            else:
                clean_meta[k] = str(v)
        self.metadata = clean_meta


@dataclass
class SearchResult:
    """检索结果数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_type: str = "semantic"


class BaseVectorStore(ABC):
    """向量数据库抽象基类，定义标准接口"""

    @abstractmethod
    def create_collection(self, collection_name: str, **kwargs) -> Any:
        """创建或获取集合"""
        pass

    @abstractmethod
    def add_documents(self, documents: List[VectorDocument], batch_size: int = 100) -> bool:
        """批量添加文档"""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 10,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """向量相似度搜索"""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str):
        """删除整个集合"""
        pass

    @abstractmethod
    def delete_by_filter(self, filter_metadata: Dict[str, Any]) -> int:
        """按元数据过滤删除"""
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        pass

    @abstractmethod
    def document_exists(self, doc_id: str) -> bool:
        """检查文档是否已存在"""
        pass


class EmbeddingProvider:
    """向量嵌入提供者，封装火山引擎 Embedding API"""

    def __init__(self, api_key: str, base_url: str,
                 model: str = "doubao-embedding-vision",
                 batch_size: int = 4):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        self._known_dimension = None
        self._max_retries = 3
        self._retry_delay = 3.0
        self._inter_batch_delay = 0.5
        logger.info(f"EmbeddingProvider 初始化: model={model}, batch_size={batch_size}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本向量

        Args:
            texts: 文本列表

        Returns:
            向量列表，与输入文本一一对应
        """
        if not texts:
            return []

        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch_with_retry(batch)
            embeddings.extend(batch_embeddings)

            if i + self.batch_size < len(texts):
                total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
                current_batch = i // self.batch_size + 1
                if current_batch % 10 == 0:
                    logger.info(f"Embedding 进度: {current_batch}/{total_batches} 批次")
                time.sleep(self._inter_batch_delay)

        return embeddings

    def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        """带重试的批次嵌入"""
        last_error = None

        for attempt in range(self._max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                results = []
                for item in response.data:
                    emb = item.embedding
                    if self._known_dimension is None:
                        self._known_dimension = len(emb)
                    results.append(emb)
                logger.debug(f"Embedding 批次完成: {len(batch)} 条文本")
                return results

            except Exception as e:
                last_error = e
                error_str = str(e)
                is_rate_limit = "429" in error_str or "RateLimit" in error_str or "TooManyRequests" in error_str
                is_server_error = "500" in error_str or "InternalServerError" in error_str

                if is_rate_limit or is_server_error:
                    wait_time = self._retry_delay * (2 ** attempt)
                    logger.warning(f"Embedding API {error_str[:80]} 重试 {attempt + 1}/{self._max_retries}，等待 {wait_time:.1f}s")
                    time.sleep(wait_time)
                else:
                    break

        logger.error(f"Embedding API 调用最终失败: {last_error}")
        raise RuntimeError(
            f"Embedding API 调用失败（已重试 {self._max_retries} 次）: {last_error}"
        ) from last_error

    def embed_single(self, text: str) -> List[float]:
        """生成单条文本向量"""
        results = self.embed([text])
        return results[0] if results else [0.0] * (self._known_dimension or 2048)


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB 向量数据库实现
    基于本地持久化存储，适用于开发和小规模部署
    """

    def __init__(self, persist_directory: str, embedding_provider: EmbeddingProvider):
        self.persist_directory = persist_directory
        self.embedding_provider = embedding_provider
        self._client = None
        self._collection = None
        self._collection_name = None

    def _ensure_client(self):
        """确保 ChromaDB 客户端已初始化"""
        if self._client is None:
            import chromadb
            from chromadb.api.types import EmbeddingFunction as ChromaEmbeddingFunction

            class ChromaEmbeddingWrapper(ChromaEmbeddingFunction):
                def __init__(self, provider: EmbeddingProvider):
                    self.provider = provider

                def name(self) -> str:
                    return f"volcengine_{self.provider.model}"

                def default_embedding_function(self):
                    return self

                def __call__(self, input: List[str]) -> List[List[float]]:
                    return self.provider.embed(input)

            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._embed_fn = ChromaEmbeddingWrapper(self.embedding_provider)
            logger.info(f"ChromaDB 客户端初始化: {self.persist_directory}")

    def create_collection(self, collection_name: str, **kwargs) -> Any:
        """创建或获取 ChromaDB 集合"""
        self._ensure_client()
        self._collection_name = collection_name

        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embed_fn,
                metadata={"description": "保险文档知识库", **kwargs}
            )
            logger.info(f"ChromaDB 集合已就绪: {collection_name}")
        except Exception as e:
            logger.warning(f"集合获取异常，尝试重建: {e}")
            try:
                self._client.delete_collection(name=collection_name)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embed_fn,
                metadata={"description": "保险文档知识库", **kwargs}
            )
            logger.info(f"ChromaDB 集合已重建: {collection_name}")

        return self._collection

    def add_documents(self, documents: List[VectorDocument], batch_size: int = 100) -> bool:
        """批量添加文档到 ChromaDB"""
        if not documents:
            logger.warning("add_documents: 文档列表为空")
            return False

        self._ensure_client()
        if self._collection is None:
            logger.error("add_documents: 集合未初始化，请先调用 create_collection")
            return False

        try:
            ids = []
            contents = []
            metadatas = []

            for doc in documents:
                ids.append(doc.id)
                contents.append(doc.content)
                metadatas.append(doc.metadata)

            for i in range(0, len(documents), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_contents = contents[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]

                self._collection.add(
                    documents=batch_contents,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                logger.debug(f"ChromaDB 写入批次: {len(batch_ids)} 条")

            logger.info(f"ChromaDB 写入完成: 共 {len(documents)} 条文档")
            return True
        except Exception as e:
            logger.error(f"ChromaDB 写入失败: {e}")
            return False

    def search(self, query_embedding: List[float], top_k: int = 10,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """向量相似度搜索"""
        self._ensure_client()
        if self._collection is None:
            logger.error("search: 集合未初始化")
            return []

        try:
            kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
            }
            if filter_metadata:
                kwargs["where"] = filter_metadata

            results = self._collection.query(**kwargs)

            search_results = []
            if results.get("documents") and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    distance = results["distances"][0][i] if results.get("distances") else [0.0] * len(results["documents"][0])
                    score = 1.0 / (1.0 + distance) if distance else 1.0

                    search_results.append(SearchResult(
                        id=results["ids"][0][i] if results.get("ids") else "",
                        content=doc,
                        metadata=metadata,
                        score=score,
                        retrieval_type="semantic"
                    ))

            logger.debug(f"ChromaDB 搜索完成: top_k={top_k}, 结果数={len(search_results)}")
            return search_results
        except Exception as e:
            logger.error(f"ChromaDB 搜索失败: {e}")
            return []

    def delete_collection(self, collection_name: str):
        """删除整个集合"""
        self._ensure_client()
        try:
            self._client.delete_collection(name=collection_name)
            self._collection = None
            logger.info(f"ChromaDB 集合已删除: {collection_name}")
        except Exception as e:
            logger.warning(f"ChromaDB 删除集合失败（可能不存在）: {e}")

    def delete_by_filter(self, filter_metadata: Dict[str, Any]) -> int:
        """按元数据过滤删除文档"""
        self._ensure_client()
        if self._collection is None:
            return 0

        if not filter_metadata:
            logger.warning("delete_by_filter: 空过滤条件，跳过。请使用 delete_collection 清空集合")
            return 0

        try:
            result = self._collection.get(where=filter_metadata)
            if result and result.get("ids"):
                self._collection.delete(ids=result["ids"])
                deleted = len(result["ids"])
                logger.info(f"ChromaDB 删除: {deleted} 条文档 (filter={filter_metadata})")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"ChromaDB 删除失败: {e}")
            return 0

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        self._ensure_client()
        if self._collection is None:
            return {"document_count": 0, "collection_name": ""}

        try:
            count = self._collection.count()
            return {
                "collection_name": self._collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "provider": "chroma"
            }
        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            return {"document_count": 0, "collection_name": self._collection_name, "error": str(e)}

    def document_exists(self, doc_id: str) -> bool:
        """检查文档是否已存在"""
        self._ensure_client()
        if self._collection is None:
            return False
        try:
            result = self._collection.get(ids=[doc_id])
            return bool(result and result.get("ids"))
        except Exception:
            return False


class TencentCloudVectorStore(BaseVectorStore):
    """
    腾讯云向量数据库实现
    使用官方 tcvectordb SDK 连接腾讯云向量数据库实例。

    前置条件:
        1. pip install tcvectordb
        2. 在腾讯云控制台创建向量数据库实例
        3. 在 config.yaml 的 vector_store.tencent_cloud 中填写连接信息

    切换方式:
        将 config.yaml 中 vector_store.provider 改为 "tencent_cloud" 即可
    """

    def __init__(self, config: Dict[str, Any], embedding_provider: EmbeddingProvider):
        """
        Args:
            config: tencent_cloud 配置节
            embedding_provider: 向量嵌入提供者
        """
        self.config = config
        self.embedding_provider = embedding_provider
        self._client = None
        self._db = None
        self._collection = None
        self._collection_name = None
        self._dimension = config.get("dimension", 1024)
        logger.info(f"TencentCloudVectorStore 初始化: host={config.get('host')}")

    def _ensure_client(self):
        """建立腾讯云向量数据库连接"""
        if self._client is not None:
            return

        host = self.config.get("host", "")
        if not host:
            raise ValueError(
                "腾讯云向量数据库 host 未配置。请在 config.yaml 的 "
                "vector_store.tencent_cloud.host 中填写实例地址，"
                "格式如 http://10.0.X.X:80"
            )

        try:
            from tcvectordb import VectorDBClient

            self._client = VectorDBClient(
                url=host,
                username=self.config.get("username", "root"),
                key=self.config.get("key", ""),
                timeout=self.config.get("timeout", 30),
            )

            db_name = self.config.get("database", "insurance_db")
            self._db = self._client.database(db_name)
            logger.info(f"腾讯云向量数据库连接成功: host={host}, db={db_name}")
        except ImportError:
            raise ImportError(
                "请安装腾讯云向量数据库 SDK: pip install tcvectordb"
            )
        except Exception as e:
            logger.error(f"腾讯云向量数据库连接失败: {e}")
            raise ConnectionError(f"无法连接腾讯云向量数据库: {e}") from e

    def create_collection(self, collection_name: str, **kwargs) -> Any:
        """
        创建或获取腾讯云向量数据库集合

        集合 Schema：
            id         (String)   - 主键，文档唯一ID
            content    (String)   - 文档内容文本
            source     (String)   - 来源文件路径
            file_name  (String)   - 文件名
            page       (Uint64)   - 页码
            chunk_index (Uint64)  - 分块序号
            chapter    (String)   - 章节名
            section    (String)   - 条款名
            vector     (Vector)   - 文本向量 (HNSW + Cosine)
        """
        self._ensure_client()
        self._collection_name = collection_name

        from tcvectordb.model.enum import FieldType, IndexType, MetricType
        from tcvectordb.model.index import Index, VectorIndex, FilterIndex, HNSWParams

        try:
            self._collection = self._db.collection(collection_name)
            count = self._collection.count()
            logger.info(f"腾讯云向量集合已就绪: {collection_name} (文档数: {count})")
            return self._collection
        except Exception:
            pass

        logger.info(f"集合 {collection_name} 不存在，正在创建...")
        try:
            index = Index()
            index.add(FilterIndex(
                name="id", field_type=FieldType.String,
                index_type=IndexType.PRIMARY_KEY
            ))
            index.add(FilterIndex(
                name="content", field_type=FieldType.String,
                index_type=IndexType.FILTER
            ))
            index.add(FilterIndex(
                name="source", field_type=FieldType.String,
                index_type=IndexType.FILTER
            ))
            index.add(FilterIndex(
                name="file_name", field_type=FieldType.String,
                index_type=IndexType.FILTER
            ))
            index.add(FilterIndex(
                name="page", field_type=FieldType.Uint64,
                index_type=IndexType.FILTER
            ))
            index.add(FilterIndex(
                name="chunk_index", field_type=FieldType.Uint64,
                index_type=IndexType.FILTER
            ))
            index.add(FilterIndex(
                name="chapter", field_type=FieldType.String,
                index_type=IndexType.FILTER
            ))
            index.add(FilterIndex(
                name="section", field_type=FieldType.String,
                index_type=IndexType.FILTER
            ))
            index.add(FilterIndex(
                name="company_name", field_type=FieldType.String,
                index_type=IndexType.FILTER
            ))
            index.add(VectorIndex(
                name="vector", field_type=FieldType.Vector,
                index_type=IndexType.HNSW,
                dimension=self._dimension,
                metric_type=MetricType.COSINE,
                params=HNSWParams(m=16, efconstruction=200),
            ))

            self._collection = self._db.create_collection(
                name=collection_name,
                shard=self.config.get("shard", 1),
                replicas=self.config.get("replicas", 1),
                description="保险文档知识库",
                index=index,
            )
            logger.info(f"腾讯云向量集合已创建: {collection_name} "
                       f"(dim={self._dimension}, metric=cosine)")
        except Exception as e:
            logger.error(f"腾讯云向量集合操作失败: {e}")
            raise RuntimeError(f"集合初始化失败: {e}") from e

        return self._collection

    def add_documents(self, documents: List[VectorDocument], batch_size: int = 100) -> bool:
        """批量添加文档到腾讯云向量数据库（含 embedding）"""
        if not documents:
            return False

        self._ensure_client()
        if self._collection is None:
            logger.error("add_documents: 集合未初始化，请先调用 create_collection")
            return False

        try:
            from tcvectordb.model.document import Document

            total = len(documents)
            for start in range(0, total, batch_size):
                batch = documents[start:start + batch_size]
                contents = [d.content for d in batch]
                embeddings = self.embedding_provider.embed(contents)

                tc_docs = []
                for doc, emb in zip(batch, embeddings):
                    tc_docs.append(Document(
                        id=doc.id,
                        vector=emb,
                        content=doc.content,
                        source=doc.metadata.get("source", ""),
                        file_name=doc.metadata.get("file_name", ""),
                        page=doc.metadata.get("page_number", 0),
                        chunk_index=doc.metadata.get("chunk_index", 0),
                        chapter=doc.metadata.get("chapter", ""),
                        section=doc.metadata.get("section", ""),
                        company_name=doc.metadata.get("company_name", ""),
                    ))

                result = self._collection.upsert(documents=tc_docs)
                logger.debug(f"腾讯云写入批次: {len(tc_docs)} 条, "
                            f"affected={result.get('affectedCount', 0)}")

            logger.info(f"腾讯云向量写入完成: 共 {total} 条文档")
            return True
        except Exception as e:
            logger.error(f"腾讯云向量写入失败: {e}")
            return False

    def search(self, query_embedding: List[float], top_k: int = 10,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """向量相似度搜索"""
        self._ensure_client()
        if self._collection is None:
            logger.error("search: 集合未初始化")
            return []

        try:
            from tcvectordb.model.document import SearchParams

            filter_expr = None
            if filter_metadata:
                conditions = []
                for k, v in filter_metadata.items():
                    if isinstance(v, str):
                        conditions.append(f'{k}="{v}"')
                    else:
                        conditions.append(f'{k}={v}')
                filter_expr = " and ".join(conditions)

            results = self._collection.search(
                vectors=[query_embedding],
                limit=top_k,
                filter=filter_expr,
                output_fields=[
                    "content", "source", "file_name",
                    "page", "chapter", "section", "chunk_index",
                    "company_name"
                ],
                params=SearchParams(ef=200),
            )

            search_results = []
            for result_list in results:
                for item in result_list:
                    search_results.append(SearchResult(
                        id=item.get("id", ""),
                        content=item.get("content", ""),
                        metadata={
                            "source": item.get("source", ""),
                            "file_name": item.get("file_name", ""),
                            "page_number": item.get("page", 0),
                            "chapter": item.get("chapter", ""),
                            "section": item.get("section", ""),
                            "chunk_index": item.get("chunk_index", 0),
                            "company_name": item.get("company_name", ""),
                        },
                        score=item.get("score", 0.0),
                        retrieval_type="semantic"
                    ))

            logger.debug(f"腾讯云搜索完成: top_k={top_k}, 结果={len(search_results)}")
            return search_results
        except Exception as e:
            logger.error(f"腾讯云向量搜索失败: {e}")
            return []

    def delete_collection(self, collection_name: str):
        """删除整个集合"""
        self._ensure_client()
        try:
            self._db.drop_collection(collection_name)
            self._collection = None
            logger.info(f"腾讯云向量集合已删除: {collection_name}")
        except Exception as e:
            logger.warning(f"腾讯云向量删除集合失败: {e}")

    def delete_by_filter(self, filter_metadata: Dict[str, Any]) -> int:
        """按元数据过滤删除文档"""
        self._ensure_client()
        if self._collection is None:
            return 0

        try:
            filter_expr = " and ".join(
                f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}'
                for k, v in filter_metadata.items()
            )
            result = self._collection.delete(filter=filter_expr)
            affected = result.get("affectedCount", 0)
            logger.info(f"腾讯云向量删除: {affected} 条 (filter={filter_expr})")
            return affected
        except Exception as e:
            logger.error(f"腾讯云向量删除失败: {e}")
            return 0

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if self._collection is None:
            return {"document_count": 0, "collection_name": self._collection_name,
                    "provider": "tencent_cloud", "status": "not_initialized"}

        try:
            count = self._collection.count()
            return {
                "collection_name": self._collection_name,
                "document_count": count,
                "provider": "tencent_cloud",
                "status": "connected",
                "host": self.config.get("host", ""),
            }
        except Exception as e:
            logger.warning(f"获取腾讯云集合统计失败: {e}")
            return {
                "collection_name": self._collection_name,
                "document_count": 0,
                "provider": "tencent_cloud",
                "status": f"error: {e}"
            }

    def document_exists(self, doc_id: str) -> bool:
        """检查文档是否已存在"""
        if self._collection is None:
            return False
        try:
            result = self._collection.searchById(doc_id)
            return bool(result)
        except Exception:
            return False


def create_vector_store(config: Dict[str, Any], embedding_provider: EmbeddingProvider) -> BaseVectorStore:
    """
    向量数据库工厂函数 — 一键切换本地/云端

    切换方式：
        config.yaml 中修改 vector_store.provider:
          "chroma"        → 本地 ChromaDB
          "tencent_cloud" → 腾讯云向量数据库

    Args:
        config: 完整配置字典
        embedding_provider: 向量嵌入提供者

    Returns:
        BaseVectorStore 实现实例
    """
    vs_config = config.get("vector_store", {})
    provider = vs_config.get("provider", "chroma")

    if provider == "chroma":
        chroma_cfg = vs_config.get("chroma", {})
        persist_dir = chroma_cfg.get("persist_directory", "./chroma_db")
        os.makedirs(persist_dir, exist_ok=True)
        logger.info(f"创建 ChromaVectorStore (本地): {persist_dir}")
        return ChromaVectorStore(persist_dir, embedding_provider)

    elif provider == "tencent_cloud":
        tc_config = vs_config.get("tencent_cloud", {})
        host = tc_config.get("host", "")
        if not host:
            logger.error(
                "腾讯云向量数据库 host 未配置。请在 config.yaml 中填写 "
                "vector_store.tencent_cloud.host 后重试。"
            )
            raise ValueError(
                "腾讯云向量数据库 host 未配置，请在 config.yaml 的 "
                "vector_store.tencent_cloud.host 中填写实例地址"
            )

        tc_config["dimension"] = config.get("embedding", {}).get("dimension", 2048)
        logger.info(f"创建 TencentCloudVectorStore (云端): {host}")
        return TencentCloudVectorStore(tc_config, embedding_provider)

    else:
        supported = ["chroma", "tencent_cloud"]
        logger.error(f"不支持的向量库 provider: {provider}，可选: {supported}")
        raise ValueError(f"不支持的 vector_store.provider '{provider}'，可选: {supported}")


def generate_doc_id(source: str, chunk_index: int) -> str:
    """生成唯一文档ID"""
    raw = f"{source}_{chunk_index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()
