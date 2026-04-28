"""
保险文档 Agentic RAG 系统 - 文档加载器
功能：自动遍历「保司文件」文件夹，批量加载 PDF/TXT，提取元数据（文件名、路径、页码），
     支持异常处理、文件大小校验、增量更新检测。
"""
import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

import PyPDF2

from logger import get_logger

logger = get_logger()


@dataclass
class LoadedDocument:
    """加载后的文档数据结构"""
    file_path: str
    file_name: str
    file_type: str
    file_hash: str
    total_pages: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}


@dataclass
class PageContent:
    """单页文档内容"""
    page_number: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """
    保险文档加载器
    遍历指定目录，批量加载 PDF/TXT 文件，提取文本和元数据
    """

    def __init__(self, source_dir: str, supported_formats: List[str] = None,
                 max_file_size_mb: float = 50, encoding: str = "utf-8"):
        """
        Args:
            source_dir: 文档源目录路径（如 "./保司文件"）
            supported_formats: 支持的文件格式列表
            max_file_size_mb: 单文件最大大小（MB）
            encoding: 文本文件编码
        """
        self.source_dir = Path(source_dir)
        self.supported_formats = supported_formats or [".pdf", ".txt"]
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.encoding = encoding

        if not self.source_dir.exists():
            os.makedirs(self.source_dir, exist_ok=True)
            logger.warning(f"源目录不存在，已自动创建: {self.source_dir}")

        logger.info(f"DocumentLoader 初始化: source_dir={self.source_dir}, "
                     f"formats={self.supported_formats}, max_size={max_file_size_mb}MB")

    def discover_files(self) -> List[Path]:
        """
        递归发现目录中所有支持的文档文件

        Returns:
            文件路径列表
        """
        files = []
        try:
            for file_path in self.source_dir.rglob("*"):
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    if suffix in self.supported_formats:
                        files.append(file_path)
        except Exception as e:
            logger.error(f"文件发现失败: {e}")

        logger.info(f"发现 {len(files)} 个支持的文档文件")
        return sorted(files)

    def load_all(self) -> List[LoadedDocument]:
        """
        加载所有发现的文档

        Returns:
            已加载的文档列表
        """
        files = self.discover_files()
        if not files:
            logger.warning(f"目录 '{self.source_dir}' 中没有找到支持的文档（{self.supported_formats}）")
            return []

        documents = []
        for file_path in files:
            try:
                doc = self.load_file(file_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"加载文件失败 {file_path}: {e}")

        logger.info(f"文档加载完成: 成功 {len(documents)}/{len(files)} 个文件")
        return documents

    def load_file(self, file_path: Path) -> Optional[LoadedDocument]:
        """
        加载单个文件

        Args:
            file_path: 文件路径

        Returns:
            LoadedDocument 或 None（加载失败时）
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return None

        file_size = file_path.stat().st_size
        if file_size > self.max_file_size_bytes:
            logger.error(f"文件过大 ({file_size / 1024 / 1024:.1f}MB > {self.max_file_size_bytes / 1024 / 1024:.1f}MB): {file_path}")
            return None

        if file_size == 0:
            logger.warning(f"文件为空: {file_path}")
            return None

        suffix = file_path.suffix.lower()
        file_name = file_path.name

        try:
            if suffix == ".pdf":
                return self._load_pdf(file_path)
            elif suffix == ".txt":
                return self._load_txt(file_path)
            else:
                logger.warning(f"不支持的文件格式: {suffix}")
                return None
        except Exception as e:
            logger.error(f"文件加载异常 {file_path}: {e}")
            return None

    def _load_pdf(self, file_path: Path) -> Optional[LoadedDocument]:
        """
        加载 PDF 文件并提取文本

        Args:
            file_path: PDF 文件路径

        Returns:
            LoadedDocument 对象
        """
        logger.info(f"加载 PDF: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)

                if reader.is_encrypted:
                    logger.warning(f"PDF 已加密，跳过: {file_path}")
                    return None

                total_pages = len(reader.pages)
                full_text_parts = []
                pages_content = []

                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                        page_text = page_text.strip()
                        if page_text:
                            full_text_parts.append(page_text)
                            pages_content.append(PageContent(
                                page_number=page_num + 1,
                                text=page_text,
                                metadata={"file_name": file_path.name}
                            ))
                    except Exception as e:
                        logger.warning(f"PDF 第 {page_num + 1} 页提取失败: {e}")

                if not full_text_parts:
                    logger.warning(f"PDF 未提取到任何文本: {file_path}")
                    return None

                full_text = "\n\n".join(full_text_parts)

            file_hash = self._compute_file_hash(file_path)

            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": "pdf",
                "file_size": file_path.stat().st_size,
                "total_pages": total_pages,
                "extracted_pages": len(pages_content),
                "file_hash": file_hash,
            }

            doc = LoadedDocument(
                file_path=str(file_path),
                file_name=file_path.name,
                file_type="pdf",
                file_hash=file_hash,
                total_pages=total_pages,
                text=full_text,
                metadata=metadata,
            )
            doc._pages = pages_content  # type: ignore

            logger.info(f"PDF 加载成功: {file_path.name}, {total_pages} 页, "
                         f"{len(full_text)} 字符")
            return doc

        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"PDF 读取错误 {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"PDF 加载异常 {file_path}: {e}")
            return None

    def _load_txt(self, file_path: Path) -> Optional[LoadedDocument]:
        """
        加载 TXT 文本文件

        Args:
            file_path: TXT 文件路径

        Returns:
            LoadedDocument 对象
        """
        logger.info(f"加载 TXT: {file_path}")

        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                text = f.read()

            if not text.strip():
                logger.warning(f"TXT 文件内容为空: {file_path}")
                return None

            file_hash = self._compute_file_hash(file_path)

            line_count = text.count('\n') + 1
            estimated_pages = max(1, line_count // 40)

            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": "txt",
                "file_size": file_path.stat().st_size,
                "total_pages": estimated_pages,
                "extracted_pages": 1,
                "file_hash": file_hash,
            }

            doc = LoadedDocument(
                file_path=str(file_path),
                file_name=file_path.name,
                file_type="txt",
                file_hash=file_hash,
                total_pages=estimated_pages,
                text=text,
                metadata=metadata,
            )
            doc._pages = [PageContent(page_number=1, text=text,  # type: ignore
                                       metadata={"file_name": file_path.name})]

            logger.info(f"TXT 加载成功: {file_path.name}, {len(text)} 字符")
            return doc

        except UnicodeDecodeError:
            logger.warning(f"TXT 编码错误，尝试 GBK: {file_path}")
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    text = f.read()
                file_hash = self._compute_file_hash(file_path)
                metadata = {
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": "txt",
                    "file_size": file_path.stat().st_size,
                    "total_pages": 1,
                    "extracted_pages": 1,
                    "file_hash": file_hash,
                }
                return LoadedDocument(
                    file_path=str(file_path),
                    file_name=file_path.name,
                    file_type="txt",
                    file_hash=file_hash,
                    total_pages=1,
                    text=text,
                    metadata=metadata,
                )
            except Exception as e2:
                logger.error(f"TXT 加载失败 {file_path}: {e2}")
                return None
        except Exception as e:
            logger.error(f"TXT 加载异常 {file_path}: {e}")
            return None

    def get_page_text(self, doc: LoadedDocument, page_number: int) -> Optional[str]:
        """
        获取文档指定页的文本内容

        Args:
            doc: 已加载的文档
            page_number: 页码（从1开始）

        Returns:
            页面文本或 None
        """
        if hasattr(doc, '_pages'):
            for page in doc._pages:  # type: ignore
                if page.page_number == page_number:
                    return page.text
        return None

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """计算文件的 MD5 哈希值"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def detect_insurance_document_type(text: str) -> str:
        """
        自动检测保险文档类型

        Args:
            text: 文档全文

        Returns:
            文档类型标签
        """
        patterns = {
            "保险条款": r"(保险条款|保险合同|保险责任|责任免除)",
            "理赔规则": r"(理赔|索赔|赔付|给付)",
            "投保须知": r"(投保|被保险人|投保人|受益人)",
            "费率表": r"(费率|保费|缴费|趸交)",
            "健康告知": r"(健康告知|既往症|病史)",
        }

        scores = {}
        for doc_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            scores[doc_type] = len(matches)

        if scores:
            return max(scores, key=scores.get)
        return "其他文档"
