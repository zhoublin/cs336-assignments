
# project_logger.py

import logging
import os
import sys
import time
import json
import socket
import threading
import traceback

from pathlib import Path
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Any, Optional


class ProjectLogger:
    """
    通用项目日志类。

    功能：
      1. 控制台 + 文件双输出
      2. 日志按大小自动轮转
      3. 记录时间、级别、进程、线程、模块、文件、行号、函数
      4. 支持异常堆栈
      5. 支持自定义上下文信息
      6. 支持耗时统计
      7. 防止同一个 logger 重复添加 handler

    使用示例：
        log = ProjectLogger("inference", log_dir="logs").logger
        log.info("开始推理")
        log.exception("推理失败")
    """

    _configured_loggers = set()
    _lock = threading.RLock()

    def __init__(
        self,
        name: str = "app",
        log_dir: str = "logs",
        level: int = logging.DEBUG,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        max_bytes: int = 50 * 1024 * 1024,
        backup_count: int = 10,
        console: bool = True,
        file: bool = True,
        json_format: bool = False,
        encoding: str = "utf-8",
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.json_format = json_format

        # 使用命名 logger，不修改 root logger 的配置
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        with self._lock:
            # 同一个 logger 名称只配置一次 Handler
            if name not in self._configured_loggers:
                self._configure(
                    console=console,
                    file=file,
                    console_level=console_level,
                    file_level=file_level,
                    max_bytes=max_bytes,
                    backup_count=backup_count,
                    encoding=encoding,
                )
                self._configured_loggers.add(name)

    def _configure(
        self,
        console: bool,
        file: bool,
        console_level: int,
        file_level: int,
        max_bytes: int,
        backup_count: int,
        encoding: str,
    ):
        formatter = (
            JsonFormatter()
            if self.json_format
            else logging.Formatter(
                fmt=(
                    "%(asctime)s.%(msecs)03d | "
                    "%(levelname)-8s | "
                    "host=%(hostname)s | "
                    "pid=%(process)d | "
                    "tid=%(thread)d | "
                    "%(name)s | "
                    "%(filename)s:%(lineno)d | "
                    "func=%(funcName)s | "
                    "%(message)s"
                ),
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        if console:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if file:
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # 不同 logger 使用独立日志文件
            safe_name = self.name.replace("/", "_").replace("\\", "_")
            log_path = self.log_dir / f"{safe_name}.log"

            file_handler = RotatingFileHandler(
                filename=log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding=encoding,
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # 为每条日志补充主机名等字段
        for handler in self.logger.handlers:
            handler.addFilter(ContextFilter())

    @contextmanager
    def timed(self, operation: str, **context):
        """
        记录操作开始、结束、耗时和异常。

        用法：
            with log.timed("extract_features", sample_id="abc"):
                extract_features()
        """
        start = time.perf_counter()
        self.logger.info(
            "operation_started | operation=%s | context=%s",
            operation,
            json.dumps(context, ensure_ascii=False, default=str),
        )

        try:
            yield
        except Exception:
            elapsed = time.perf_counter() - start
            self.logger.exception(
                "operation_failed | operation=%s | elapsed_sec=%.6f | context=%s",
                operation,
                elapsed,
                json.dumps(context, ensure_ascii=False, default=str),
            )
            raise
        else:
            elapsed = time.perf_counter() - start
            self.logger.info(
                "operation_finished | operation=%s | elapsed_sec=%.6f | context=%s",
                operation,
                elapsed,
                json.dumps(context, ensure_ascii=False, default=str),
            )

    def log_exception(
        self,
        message: str,
        exc: Optional[BaseException] = None,
        **context: Any,
    ):
        """记录异常及堆栈，并附加上下文。"""
        self.logger.error(
            "%s | context=%s",
            message,
            json.dumps(context, ensure_ascii=False, default=str),
            exc_info=(
                (type(exc), exc, exc.__traceback__)
                if exc is not None
                else True
            ),
        )


class ContextFilter(logging.Filter):
    """补齐日志格式所需的字段，避免格式化时报缺失字段。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.hostname = socket.gethostname()
        return True


class JsonFormatter(logging.Formatter):
    """每条日志输出为一行 JSON，方便日志平台检索。"""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.fromtimestamp(record.created).astimezone().isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "hostname": getattr(record, "hostname", socket.gethostname()),
            "pid": record.process,
            "thread_id": record.thread,
            "thread_name": record.threadName,
            "module": record.module,
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        # logger.info("...", extra={"sample_id": "..."}) 等自定义字段
        standard_fields = set(logging.makeLogRecord({}).__dict__.keys())
        standard_fields.update({
            "message", "asctime", "hostname", "exc_text"
        })

        for key, value in record.__dict__.items():
            if key not in standard_fields and key not in data:
                data[key] = value

        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)

        return json.dumps(data, ensure_ascii=False, default=str)
