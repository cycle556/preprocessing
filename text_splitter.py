"""
保险文档 Agentic RAG 系统 - 文本分块器
功能：提供语义分块 + 递归分块两种策略，针对保险长条款文本优化，
     保留条款结构完整性（章、条、款），生成带元数据摘要的文档块。
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import LoadedDocument
from logger import get_logger

logger = get_logger()


@dataclass
class TextChunk:
    """文本分块数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    chunk_index: int = 0

    def __post_init__(self):
        self.summary = self._generate_summary()
        self.metadata["chunk_summary"] = self.summary

    def _generate_summary(self, max_len: int = 80) -> str:
        """自动生成分块摘要"""
        clean = self.content.strip()
        summary = clean[:max_len].replace('\n', ' ')
        if len(clean) > max_len:
            summary += "..."
        return summary


class InsuranceTextSplitter:
    """
    保险文档专用分块器
    组合语义分块（按章节/条款结构）和递归分块（按字符数）两种策略
    """

    CLAUSE_PATTERNS = [
        r'^第[一二三四五六七八九十百千]+章',
        r'^第[一二三四五六七八九十百千]+节',
        r'^第[一二三四五六七八九十百千]+条',
        r'^[一二三四五六七八九十]+[、.]',
        r'^(保险责任|责任免除|保险金额|保险期间|保险费|投保人|被保险人)',
    ]

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100,
                 min_chunk_size: int = 50,
                 separators: List[str] = None,
                 semantic_headings: List[str] = None):
        """
        Args:
            chunk_size: 分块最大字符数
            chunk_overlap: 分块重叠字符数
            min_chunk_size: 最小分块字符数，小于此值会合并到前一个块
            separators: 递归分块分隔符
            semantic_headings: 语义分块识别关键词
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.semantic_headings = semantic_headings or ["章", "条", "节", "款"]

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n\n", "\n\n", "\n", "。", "！", "？", " ", ""],
            length_function=len,
        )

        logger.info(f"InsuranceTextSplitter 初始化: chunk_size={chunk_size}, "
                     f"overlap={chunk_overlap}, min={min_chunk_size}")

    def split_document(self, doc: LoadedDocument) -> List[TextChunk]:
        """
        对单个文档执行分块

        Args:
            doc: 已加载的文档

        Returns:
            分块列表
        """
        try:
            clause_chunks = self._semantic_split(doc.text, doc)
        except Exception as e:
            logger.warning(f"语义分块失败，回退到递归分块: {e}")
            clause_chunks = []

        if not clause_chunks:
            clause_chunks = self._recursive_split(doc.text, doc)

        chunks = self._merge_small_chunks(clause_chunks, doc)
        chunks = self._add_page_info(chunks, doc)
        self._assign_chunk_ids(chunks)

        logger.info(f"文档分块完成: {doc.file_name}, 共 {len(chunks)} 个块")
        return chunks

    def split_documents(self, documents: List[LoadedDocument]) -> List[TextChunk]:
        """
        批量分块

        Args:
            documents: 已加载文档列表

        Returns:
            所有分块列表
        """
        all_chunks = []
        for doc in documents:
            try:
                chunks = self.split_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"文档 {doc.file_name} 分块失败: {e}")
        logger.info(f"批量分块完成: {len(documents)} 个文档, 共 {len(all_chunks)} 个块")
        return all_chunks

    def _semantic_split(self, text: str, doc: LoadedDocument) -> List[TextChunk]:
        """
        语义分块：按保险条款章节结构拆分

        Args:
            text: 文档全文
            doc: 原始文档

        Returns:
            按语义分割的文本块列表
        """
        lines = text.split('\n')
        chunks = []
        current_chapter = ""
        current_section = ""
        current_lines = []
        chunk_index = 0

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if current_lines:
                    current_lines.append("")
                continue

            is_heading = any(re.match(pattern, line_stripped)
                            for pattern in self.CLAUSE_PATTERNS)

            if is_heading:
                if current_lines:
                    chunk_text = '\n'.join(current_lines).strip()
                    if len(chunk_text) >= self.min_chunk_size or self._has_meaningful_content(chunk_text):
                        chunks.append(TextChunk(
                            id="",
                            content=chunk_text,
                            metadata={
                                "source": doc.file_path,
                                "file_name": doc.file_name,
                                "file_type": doc.file_type,
                                "file_hash": doc.file_hash,
                                "chapter": current_chapter,
                                "section": current_section,
                            },
                            chunk_index=chunk_index,
                        ))
                        chunk_index += 1
                    current_lines = []

                if "章" in line_stripped:
                    current_chapter = line_stripped
                    current_section = ""
                elif any(kw in line_stripped for kw in self.semantic_headings):
                    current_section = line_stripped

            current_lines.append(line_stripped)

            if len('\n'.join(current_lines)) > self.chunk_size * 2:
                chunk_text = '\n'.join(current_lines[:-1]).strip()
                if chunk_text:
                    chunks.append(TextChunk(
                        id="",
                        content=chunk_text,
                        metadata={
                            "source": doc.file_path,
                            "file_name": doc.file_name,
                            "file_type": doc.file_type,
                            "file_hash": doc.file_hash,
                            "chapter": current_chapter,
                            "section": current_section,
                        },
                        chunk_index=chunk_index,
                    ))
                    chunk_index += 1
                current_lines = [current_lines[-1]]

        if current_lines:
            chunk_text = '\n'.join(current_lines).strip()
            if len(chunk_text) >= self.min_chunk_size or self._has_meaningful_content(chunk_text):
                chunks.append(TextChunk(
                    id="",
                    content=chunk_text,
                    metadata={
                        "source": doc.file_path,
                        "file_name": doc.file_name,
                        "file_type": doc.file_type,
                        "file_hash": doc.file_hash,
                        "chapter": current_chapter,
                        "section": current_section,
                    },
                    chunk_index=chunk_index,
                ))

        return chunks

    def _recursive_split(self, text: str, doc: LoadedDocument) -> List[TextChunk]:
        """
        递归分块：按字符数和分隔符递归切分

        Args:
            text: 文档全文
            doc: 原始文档

        Returns:
            递归分块结果
        """
        texts = self.recursive_splitter.split_text(text)
        chunks = []

        for i, chunk_text in enumerate(texts):
            chunks.append(TextChunk(
                id="",
                content=chunk_text,
                metadata={
                    "source": doc.file_path,
                    "file_name": doc.file_name,
                    "file_type": doc.file_type,
                    "file_hash": doc.file_hash,
                    "chapter": "",
                    "section": "",
                },
                chunk_index=i,
            ))

        return chunks

    def _merge_small_chunks(self, chunks: List[TextChunk],
                            doc: LoadedDocument) -> List[TextChunk]:
        """合并过小的分块"""
        if len(chunks) <= 1:
            return chunks

        merged = []
        buffer = []

        for chunk in chunks:
            buffer.append(chunk)
            combined_len = sum(len(c.content) for c in buffer)
            if combined_len >= self.min_chunk_size:
                if len(buffer) == 1:
                    merged.append(buffer[0])
                else:
                    merged_content = '\n'.join(c.content for c in buffer)
                    merged_chunk = TextChunk(
                        id="",
                        content=merged_content,
                        metadata=buffer[0].metadata,
                        chunk_index=buffer[0].chunk_index,
                    )
                    merged.append(merged_chunk)
                buffer = []

        if buffer:
            if merged:
                last = merged[-1]
                combined = last.content + '\n' + '\n'.join(c.content for c in buffer)
                merged[-1] = TextChunk(
                    id="",
                    content=combined,
                    metadata=last.metadata,
                    chunk_index=last.chunk_index,
                )
            else:
                merged.extend(buffer)

        return merged

    def _add_page_info(self, chunks: List[TextChunk],
                       doc: LoadedDocument) -> List[TextChunk]:
        """为分块添加页码信息"""
        if not hasattr(doc, '_pages'):
            for c in chunks:
                c.metadata["page_number"] = 1
            return chunks

        pages = doc._pages  # type: ignore
        if not pages:
            for c in chunks:
                c.metadata["page_number"] = 1
            return chunks

        for chunk in chunks:
            best_page = 1
            best_overlap = 0
            chunk_start = chunk.content[:30].strip()

            for page in pages:
                if chunk_start in page.text:
                    best_page = page.page_number
                    break

                overlap = self._text_overlap(chunk.content[:100], page.text[:200])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_page = page.page_number

            chunk.metadata["page_number"] = best_page

        return chunks

    def _assign_chunk_ids(self, chunks: List[TextChunk]):
        """为分块分配唯一ID"""
        from vector_store import generate_doc_id
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.id = generate_doc_id(chunk.metadata.get("source", ""), i)
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_id"] = chunk.id

    @staticmethod
    def _has_meaningful_content(text: str) -> bool:
        """检查文本是否有实质性内容"""
        clean = re.sub(r'\s+', '', text)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', clean)
        return len(clean) > 5 and len(chinese_chars) > 2

    @staticmethod
    def _text_overlap(text1: str, text2: str) -> int:
        """计算两段文本的字符重叠数"""
        if not text1 or not text2:
            return 0
        return sum(1 for c in text1 if c in text2)
