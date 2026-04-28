"""
保险文档 Agentic RAG 系统 - 多轮对话管理器
功能：支持多轮对话上下文感知，保留历史上下文，避免重复提问，
     自动管理对话窗口大小和会话持久化。
"""
import os
import json
import time
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from logger import get_logger

logger = get_logger()


@dataclass
class ConversationTurn:
    """单轮对话记录"""
    question: str
    answer: str
    citations: List[Dict[str, str]] = field(default_factory=list)
    intent: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class Conversation:
    """完整对话会话"""
    id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def add_turn(self, question: str, answer: str,
                 citations: List[Dict[str, str]] = None,
                 intent: str = ""):
        """添加一轮对话"""
        turn = ConversationTurn(
            question=question,
            answer=answer,
            citations=citations or [],
            intent=intent,
        )
        self.turns.append(turn)
        self.updated_at = datetime.now().isoformat()

    def get_recent_history(self, n: int = 5) -> List[Dict[str, str]]:
        """获取最近的对话历史"""
        return [
            {"question": t.question, "answer": t.answer}
            for t in self.turns[-n:]
        ]

    def get_last_question(self) -> Optional[str]:
        """获取上一轮问题"""
        if len(self.turns) >= 2:
            return self.turns[-2].question
        return None

    def get_last_answer(self) -> Optional[str]:
        """获取上一轮回答"""
        if len(self.turns) >= 2:
            return self.turns[-2].answer
        return None


class ConversationManager:
    """
    对话管理器
    创建、管理对话会话，支持持久化存储和历史上下文回溯
    """

    def __init__(self, storage_dir: str = "./conversation_data",
                 max_history_turns: int = 10,
                 context_window_size: int = 5):
        """
        Args:
            storage_dir: 会话数据存储目录
            max_history_turns: 每个会话最多保存的对话轮数
            context_window_size: 上下文窗口大小（传递给 LLM 的历史轮数）
        """
        self.storage_dir = storage_dir
        self.max_history_turns = max_history_turns
        self.context_window_size = context_window_size
        self._conversations: Dict[str, Conversation] = {}

        os.makedirs(storage_dir, exist_ok=True)
        self._load_saved_conversations()

        logger.info(f"ConversationManager 初始化: storage={storage_dir}, "
                     f"max_turns={max_history_turns}, window={context_window_size}")

    def create_conversation(self) -> str:
        """创建新的对话会话，返回会话 ID"""
        conv_id = str(uuid.uuid4())[:8]
        conversation = Conversation(id=conv_id)
        self._conversations[conv_id] = conversation
        logger.info(f"创建会话: {conv_id}")
        return conv_id

    def add_turn(self, conv_id: str, question: str, answer: str,
                 citations: List[Dict[str, str]] = None,
                 intent: str = ""):
        """
        添加一轮对话

        Args:
            conv_id: 会话 ID
            question: 用户问题
            answer: 系统回答
            citations: 引用列表
            intent: 意图类别
        """
        if conv_id not in self._conversations:
            self._conversations[conv_id] = Conversation(id=conv_id)

        conversation = self._conversations[conv_id]
        conversation.add_turn(question, answer, citations, intent)

        if len(conversation.turns) > self.max_history_turns:
            conversation.turns = conversation.turns[-self.max_history_turns:]

        self._save_conversation(conv_id)

    def get_history(self, conv_id: str) -> List[Dict[str, str]]:
        """获取会话的对话历史"""
        if conv_id not in self._conversations:
            return []
        return self._conversations[conv_id].get_recent_history(self.context_window_size)

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """获取完整会话对象"""
        return self._conversations.get(conv_id)

    def delete_conversation(self, conv_id: str):
        """删除会话"""
        if conv_id in self._conversations:
            del self._conversations[conv_id]
            file_path = os.path.join(self.storage_dir, f"{conv_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.info(f"删除会话: {conv_id}")

    def list_conversations(self) -> List[Dict[str, Any]]:
        """列出所有会话摘要"""
        summaries = []
        for conv_id, conv in self._conversations.items():
            summaries.append({
                "id": conv_id,
                "turns": len(conv.turns),
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "last_question": conv.turns[-1].question if conv.turns else "",
            })
        return sorted(summaries, key=lambda x: x["updated_at"], reverse=True)

    def get_context_for_query(self, conv_id: str) -> Optional[List[Dict[str, str]]]:
        """获取用于当前查询的上下文历史"""
        if conv_id not in self._conversations:
            return None
        return self._conversations[conv_id].get_recent_history(self.context_window_size)

    def _save_conversation(self, conv_id: str):
        """持久化保存会话"""
        try:
            conversation = self._conversations.get(conv_id)
            if not conversation:
                return

            data = {
                "id": conversation.id,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "turns": [
                    {
                        "question": t.question,
                        "answer": t.answer,
                        "citations": t.citations,
                        "intent": t.intent,
                        "timestamp": t.timestamp,
                    }
                    for t in conversation.turns
                ]
            }

            file_path = os.path.join(self.storage_dir, f"{conv_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存会话失败 {conv_id}: {e}")

    def _load_saved_conversations(self):
        """加载已持久化的会话"""
        if not os.path.exists(self.storage_dir):
            return

        loaded = 0
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith(".json"):
                continue

            try:
                file_path = os.path.join(self.storage_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                conv = Conversation(
                    id=data["id"],
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                )

                for turn_data in data.get("turns", []):
                    conv.turns.append(ConversationTurn(
                        question=turn_data["question"],
                        answer=turn_data["answer"],
                        citations=turn_data.get("citations", []),
                        intent=turn_data.get("intent", ""),
                        timestamp=turn_data.get("timestamp", ""),
                    ))

                self._conversations[conv.id] = conv
                loaded += 1
            except Exception as e:
                logger.warning(f"加载会话失败 {filename}: {e}")

        if loaded > 0:
            logger.info(f"加载了 {loaded} 个历史会话")
