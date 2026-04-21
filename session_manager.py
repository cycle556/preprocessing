"""
会话管理模块
功能：负责管理用户对话历史和文档版本信息，支持多轮对话上下文管理
特点：持久化存储对话记录，支持文档版本追踪，提供上下文格式化能力
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class ConversationTurn:
    """对话轮次数据结构"""
    turn_id: str               # 轮次ID
    timestamp: str             # 时间戳
    user_query: str            # 用户查询
    assistant_response: str    # 助手回复
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


@dataclass
class DocumentVersion:
    """文档版本数据结构"""
    version_id: str            # 版本ID
    file_path: str             # 文件路径
    version_number: str        # 版本号
    effective_date: str        # 生效日期
    upload_date: str           # 上传日期
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    changelog: str = ""        # 更新日志


class SessionManager:
    """会话管理器，负责对话历史和文档版本的管理"""
    def __init__(self, storage_dir: str = "./session_data"):
        """
        初始化会话管理器
        :param storage_dir: 数据存储目录，默认./session_data
        """
        self.storage_dir = storage_dir
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        self.document_versions: Dict[str, List[DocumentVersion]] = {}
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self):
        """确保存储目录存在，不存在则创建"""
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(os.path.join(self.storage_dir, "conversations"), exist_ok=True)
        os.makedirs(os.path.join(self.storage_dir, "documents"), exist_ok=True)
    
    def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        创建新的对话会话
        :param conversation_id: 可选的对话ID，未提供则自动生成
        :return: 对话ID
        """
        if not conversation_id:
            conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.conversations[conversation_id] = []
        return conversation_id
    
    def add_turn(self, conversation_id: str, user_query: str,
                 assistant_response: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        添加对话轮次到会话
        :param conversation_id: 对话ID
        :param user_query: 用户查询
        :param assistant_response: 助手回复
        :param metadata: 可选元数据
        :return: 轮次ID
        """
        if conversation_id not in self.conversations:
            self.create_conversation(conversation_id)
        
        turn_id = f"turn_{len(self.conversations[conversation_id]) + 1}"
        turn = ConversationTurn(
            turn_id=turn_id,
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            assistant_response=assistant_response,
            metadata=metadata or {}
        )
        
        self.conversations[conversation_id].append(turn)
        self._save_conversation(conversation_id)
        return turn_id
    
    def get_conversation_history(self, conversation_id: str,
                                  last_n: Optional[int] = None) -> List[ConversationTurn]:
        """
        获取对话历史
        :param conversation_id: 对话ID
        :param last_n: 可选，只返回最近n轮对话
        :return: 对话轮次列表
        """
        if conversation_id not in self.conversations:
            self._load_conversation(conversation_id)
        
        history = self.conversations.get(conversation_id, [])
        if last_n:
            return history[-last_n:]
        return history
    
    def _save_conversation(self, conversation_id: str):
        """
        持久化保存对话到文件
        :param conversation_id: 对话ID
        """
        file_path = os.path.join(self.storage_dir, "conversations", f"{conversation_id}.json")
        data = []
        for turn in self.conversations[conversation_id]:
            data.append({
                "turn_id": turn.turn_id,
                "timestamp": turn.timestamp,
                "user_query": turn.user_query,
                "assistant_response": turn.assistant_response,
                "metadata": turn.metadata
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_conversation(self, conversation_id: str):
        """
        从文件加载对话历史
        :param conversation_id: 对话ID
        """
        file_path = os.path.join(self.storage_dir, "conversations", f"{conversation_id}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.conversations[conversation_id] = []
            for item in data:
                self.conversations[conversation_id].append(ConversationTurn(**item))
    
    def add_document_version(self, file_path: str, version_number: str,
                             effective_date: str, changelog: str = "",
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        添加文档版本
        :param file_path: 文档路径
        :param version_number: 版本号
        :param effective_date: 生效日期
        :param changelog: 更新日志
        :param metadata: 可选元数据
        :return: 版本ID
        """
        doc_key = os.path.basename(file_path)
        if doc_key not in self.document_versions:
            self.document_versions[doc_key] = []
        
        version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version = DocumentVersion(
            version_id=version_id,
            file_path=file_path,
            version_number=version_number,
            effective_date=effective_date,
            upload_date=datetime.now().isoformat(),
            metadata=metadata or {},
            changelog=changelog
        )
        
        self.document_versions[doc_key].append(version)
        self._save_document_versions(doc_key)
        return version_id
    
    def get_document_versions(self, file_path: str) -> List[DocumentVersion]:
        """
        获取文档的所有版本
        :param file_path: 文档路径
        :return: 文档版本列表
        """
        doc_key = os.path.basename(file_path)
        if doc_key not in self.document_versions:
            self._load_document_versions(doc_key)
        return self.document_versions.get(doc_key, [])
    
    def get_latest_version(self, file_path: str) -> Optional[DocumentVersion]:
        """
        获取文档的最新版本
        :param file_path: 文档路径
        :return: 最新版本对象，没有则返回None
        """
        versions = self.get_document_versions(file_path)
        if versions:
            return sorted(versions, key=lambda v: v.upload_date, reverse=True)[0]
        return None
    
    def _save_document_versions(self, doc_key: str):
        """
        持久化保存文档版本信息
        :param doc_key: 文档标识（文件名）
        """
        file_path = os.path.join(self.storage_dir, "documents", f"{doc_key}_versions.json")
        data = []
        for version in self.document_versions[doc_key]:
            data.append({
                "version_id": version.version_id,
                "file_path": version.file_path,
                "version_number": version.version_number,
                "effective_date": version.effective_date,
                "upload_date": version.upload_date,
                "metadata": version.metadata,
                "changelog": version.changelog
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_document_versions(self, doc_key: str):
        """
        从文件加载文档版本信息
        :param doc_key: 文档标识（文件名）
        """
        file_path = os.path.join(self.storage_dir, "documents", f"{doc_key}_versions.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.document_versions[doc_key] = []
            for item in data:
                self.document_versions[doc_key].append(DocumentVersion(**item))
    
    def format_context_for_llm(self, conversation_id: str,
                                last_n: int = 5) -> str:
        """
        格式化对话历史为LLM可用的上下文格式
        :param conversation_id: 对话ID
        :param last_n: 最近n轮对话，默认5
        :return: 格式化的上下文文本
        """
        history = self.get_conversation_history(conversation_id, last_n)
        if not history:
            return ""
        
        context_parts = ["【对话历史】"]
        for turn in history:
            context_parts.append(f"用户: {turn.user_query}")
            context_parts.append(f"助手: {turn.assistant_response}")
        
        return "\n".join(context_parts)
