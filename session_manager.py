from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class ConversationTurn:
    turn_id: str
    timestamp: str
    user_query: str
    assistant_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentVersion:
    version_id: str
    file_path: str
    version_number: str
    effective_date: str
    upload_date: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    changelog: str = ""


class SessionManager:
    def __init__(self, storage_dir: str = "./session_data"):
        self.storage_dir = storage_dir
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        self.document_versions: Dict[str, List[DocumentVersion]] = {}
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(os.path.join(self.storage_dir, "conversations"), exist_ok=True)
        os.makedirs(os.path.join(self.storage_dir, "documents"), exist_ok=True)
    
    def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        if not conversation_id:
            conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.conversations[conversation_id] = []
        return conversation_id
    
    def add_turn(self, conversation_id: str, user_query: str, 
                 assistant_response: str, metadata: Optional[Dict[str, Any]] = None) -> str:
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
        if conversation_id not in self.conversations:
            self._load_conversation(conversation_id)
        
        history = self.conversations.get(conversation_id, [])
        if last_n:
            return history[-last_n:]
        return history
    
    def _save_conversation(self, conversation_id: str):
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
        doc_key = os.path.basename(file_path)
        if doc_key not in self.document_versions:
            self._load_document_versions(doc_key)
        return self.document_versions.get(doc_key, [])
    
    def get_latest_version(self, file_path: str) -> Optional[DocumentVersion]:
        versions = self.get_document_versions(file_path)
        if versions:
            return sorted(versions, key=lambda v: v.upload_date, reverse=True)[0]
        return None
    
    def _save_document_versions(self, doc_key: str):
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
        file_path = os.path.join(self.storage_dir, "documents", f"{doc_key}_versions.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.document_versions[doc_key] = []
            for item in data:
                self.document_versions[doc_key].append(DocumentVersion(**item))
    
    def format_context_for_llm(self, conversation_id: str, 
                                last_n: int = 5) -> str:
        history = self.get_conversation_history(conversation_id, last_n)
        if not history:
            return ""
        
        context_parts = ["【对话历史】"]
        for turn in history:
            context_parts.append(f"用户: {turn.user_query}")
            context_parts.append(f"助手: {turn.assistant_response}")
        
        return "\n".join(context_parts)
