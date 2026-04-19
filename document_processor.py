import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import PyPDF2
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str


@dataclass
class TableData:
    df: pd.DataFrame
    metadata: Dict[str, Any]
    table_id: str


class InsuranceDocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def load_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
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
        result = {"chunks": [], "tables": [], "metadata": {}}
        
        if file_path.endswith('.pdf'):
            text, metadata = self.load_pdf(file_path)
            result["metadata"] = metadata
            result["chunks"] = self.split_by_clause_structure(text, metadata)
        
        elif file_path.endswith(('.xlsx', '.xls')):
            result["tables"] = self.load_excel(file_path)
        
        return result

    def table_to_text(self, table: TableData) -> str:
        text_parts = [f"表格名称: {table.metadata.get('sheet_name', 'unknown')}"]
        text_parts.append(f"来源: {table.metadata.get('source', 'unknown')}")
        text_parts.append("\n表格内容:")
        text_parts.append(table.df.to_string(index=False))
        return "\n".join(text_parts)
