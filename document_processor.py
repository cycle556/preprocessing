"""
文档处理器模块
功能：负责加载、解析和分块保险文档，支持PDF和Excel格式
特点：针对保险条款的结构化特点进行分块优化，保留章节和条款信息
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import PyPDF2
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    """文档分块数据结构"""
    content: str          # 分块文本内容
    metadata: Dict[str, Any]  # 元数据（来源、章节、条款等）
    chunk_id: str         # 分块唯一标识


@dataclass
class TableData:
    """表格数据结构"""
    df: pd.DataFrame      # 表格数据框
    metadata: Dict[str, Any]  # 元数据（来源、工作表名等）
    table_id: str         # 表格唯一标识


class InsuranceDocumentProcessor:
    """保险文档处理器，专门针对保险条款的结构化特点进行优化"""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初始化文档处理器
        :param chunk_size: 分块大小，默认1000字符
        :param chunk_overlap: 分块重叠大小，默认200字符
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def load_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        加载PDF文档并提取文本
        :param file_path: PDF文件路径
        :return: 提取的文本内容和元数据
        """
        text = ""
        metadata = {"source": file_path, "file_type": "pdf"}
        
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            metadata["total_pages"] = len(reader.pages)
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n\n"
        
        return text, metadata

    def load_excel(self, file_path: str) -> List[TableData]:
        """
        加载Excel文档并提取表格数据
        :param file_path: Excel文件路径
        :return: 表格数据列表
        """
        tables = []
        xl = pd.ExcelFile(file_path)
        
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            table_id = f"{os.path.basename(file_path)}_{sheet_name}"
            tables.append(TableData(
                df=df,
                metadata={
                    "source": file_path,
                    "sheet_name": sheet_name,
                    "file_type": "excel",
                    "table_id": table_id
                },
                table_id=table_id
            ))
        
        return tables

    def split_by_clause_structure(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        按照保险条款的结构化特点进行分块
        根据章节、条款等关键词进行智能分块，保留结构信息
        :param text: 文档全文
        :param metadata: 文档元数据
        :return: 文档分块列表
        """
        chunks = []
        clause_patterns = [
            "第一章", "第二章", "第三章", "第四章", "第五章", "第六章", "第七章", "第八章", "第九章", "第十章",
            "第1章", "第2章", "第3章", "第4章", "第5章", "第6章", "第7章", "第8章", "第9章", "第10章",
            "第一条", "第二条", "第三条", "第四条", "第五条", "第六条", "第七条", "第八条", "第九条", "第十条",
            "第1条", "第2条", "第3条", "第4条", "第5条", "第6条", "第7条", "第8条", "第9条", "第10条",
            "总则", "保险责任", "责任免除", "保险金额", "保险期间", "保险费", "投保人义务", "被保险人义务",
            "理赔处理", "争议处理", "其他事项"
        ]
        
        lines = text.split('\n')
        current_chapter = ""
        current_section = ""
        current_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            is_clause_start = any(pattern in line for pattern in clause_patterns)
            
            if is_clause_start and current_content:
                chunk = self._create_chunk(
                    '\n'.join(current_content),
                    metadata,
                    current_chapter,
                    current_section,
                    len(chunks)
                )
                chunks.append(chunk)
                current_content = []
            
            if is_clause_start:
                if "章" in line:
                    current_chapter = line
                elif "条" in line or any(keyword in line for keyword in ["保险责任", "责任免除", "保险金额"]):
                    current_section = line
            
            current_content.append(line)
        
        if current_content:
            chunk = self._create_chunk(
                '\n'.join(current_content),
                metadata,
                current_chapter,
                current_section,
                len(chunks)
            )
            chunks.append(chunk)
        
        return chunks

    def _create_chunk(self, content: str, metadata: Dict[str, Any],
                     chapter: str, section: str, chunk_index: int) -> DocumentChunk:
        """
        创建文档分块，添加章节和条款信息到元数据
        :param content: 分块内容
        :param metadata: 原始元数据
        :param chapter: 所属章节
        :param section: 所属条款
        :param chunk_index: 分块索引
        :return: 文档分块对象
        """
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "chapter": chapter,
            "section": section,
            "chunk_index": chunk_index,
            "chunk_id": f"{metadata.get('source', 'unknown')}_{chunk_index}"
        })
        return DocumentChunk(
            content=content,
            metadata=chunk_metadata,
            chunk_id=chunk_metadata["chunk_id"]
        )

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        处理文档，根据文件类型自动选择处理方式
        :param file_path: 文档路径
        :return: 处理结果，包含分块和表格数据
        """
        result = {"chunks": [], "tables": [], "metadata": {}}
        
        if file_path.endswith('.pdf'):
            text, metadata = self.load_pdf(file_path)
            result["metadata"] = metadata
            result["chunks"] = self.split_by_clause_structure(text, metadata)
        
        elif file_path.endswith(('.xlsx', '.xls')):
            result["tables"] = self.load_excel(file_path)
        
        return result

    def table_to_text(self, table: TableData) -> str:
        """
        将表格数据转换为可检索的文本格式
        :param table: 表格数据对象
        :return: 格式化的文本内容
        """
        text_parts = [f"表格名称: {table.metadata.get('sheet_name', 'unknown')}"]
        text_parts.append(f"来源: {table.metadata.get('source', 'unknown')}")
        text_parts.append("\n表格内容:")
        text_parts.append(table.df.to_string(index=False))
        return "\n".join(text_parts)
