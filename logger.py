"""
保险文档 Agentic RAG 系统 - 日志模块
功能：提供分级日志（DEBUG/INFO/WARNING/ERROR），支持控制台+文件双输出，
     自动轮转，记录文件处理、检索、问答全流程。
"""
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path


class SystemLogger:
    """
    系统日志管理器
    提供单例模式的日志记录器，支持文件轮转和控制台双输出
    """

    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def init(self, log_dir: str = "./logs", log_level: str = "INFO",
             max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
        """
        初始化日志系统

        Args:
            log_dir: 日志文件存储目录
            log_level: 日志级别（DEBUG/INFO/WARNING/ERROR）
            max_bytes: 单个日志文件最大字节数，超出后自动轮转
            backup_count: 保留的历史日志文件数量
        """
        if self._logger is not None:
            return self._logger

        os.makedirs(log_dir, exist_ok=True)

        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        level = level_map.get(log_level.upper(), logging.INFO)

        self._logger = logging.getLogger("InsuranceRAG")
        self._logger.setLevel(level)
        self._logger.handlers.clear()

        log_format = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(log_format)
        self._logger.addHandler(console_handler)

        log_filename = os.path.join(
            log_dir,
            f"insurance_rag_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.handlers.RotatingFileHandler(
            log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(log_format)
        self._logger.addHandler(file_handler)

        self._logger.info("=" * 60)
        self._logger.info("保险文档 Agentic RAG 系统日志初始化完成")
        self._logger.info(f"日志级别: {log_level} | 日志目录: {log_dir}")
        self._logger.info("=" * 60)

        return self._logger

    def get_logger(self) -> logging.Logger:
        """获取日志记录器实例"""
        if self._logger is None:
            self.init()
        return self._logger


def get_logger() -> logging.Logger:
    """便捷函数：获取全局日志记录器"""
    return SystemLogger().get_logger()
