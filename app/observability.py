"""Observability: structured JSON logs, the request_logs record, and stats helpers.

Logging is best-effort by design: it must never be able to break an answer."""

from __future__ import annotations
import json
import logging
import sys
from datetime import datetime, timezone
from app.config import Settings


# --- structured logging ---

_stdout: logging.Logger | None = None


def _request_logger() -> logging.Logger:
    """Lazily configure the request logger (no side effects at import time)."""
    global _stdout
    if _stdout is None:
        _stdout = logging.getLogger("doctalk.requests")
        _stdout.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        _stdout.addHandler(handler)
        _stdout.propagate = False
    return _stdout


class RequestLogger:
    """Writes one JSON line to stdout and one document to request_logs.

    The Mongo write is best-effort: observability must never break an answer.
    """

    def __init__(self, logs, settings: Settings):
        self._logs = logs
        self._settings = settings

    def log(
        self,
        *,
        request_id: str,
        owner_key: str,
        conversation_id: str,
        status: str,
        total_ms: int,
        node_ms: dict,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        claims_kept: int,
        claims_dropped: int,
        dropped_reasons: list | None = None,
        judge_calls: int = 0,
        question: str | None = None,
        answer: str | None = None,
        rewritten_question: str | None = None,
        error: str | None = None,
    ) -> None:
        record = {
            "request_id": request_id,
            "owner_key": owner_key,
            "conversation_id": conversation_id,
            "status": status,
            "total_ms": total_ms,
            "node_ms": node_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost_usd, 6),
            "claims_kept": claims_kept,
            "claims_dropped": claims_dropped,
            "dropped_reasons": dropped_reasons or [],
            "judge_calls": judge_calls,
            "error": error,
            "created_at": datetime.now(timezone.utc),
        }
        if self._settings.log_text:
            record["question"] = question
            record["answer"] = answer
            record["rewritten_question"] = rewritten_question

        try:
            _request_logger().info(json.dumps(record, default=str))
        except Exception:
            pass
        try:
            self._logs.insert(record)
        except Exception:
            pass


# --- system logging to Mongo ---

class MongoLogHandler(logging.Handler):
    """Writes standard library log records into MongoDB's ``system_logs`` collection.

    Best-effort and recursion-safe by design: a failed write is swallowed and never
    re-logged, so observability can never break an answer or loop on a Mongo outage."""

    def __init__(self, db, settings: Settings):
        super().__init__(level=logging.INFO)
        self._c = db.system_logs
        self._settings = settings

    def emit(self, record: logging.LogRecord) -> None:
        try:
            doc = {
                "created_at": datetime.now(timezone.utc),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                doc["exc"] = self.format(record)
            self._c.insert_one(doc)
        except Exception:
            pass


def install_mongo_logging(db, settings: Settings) -> None:
    """Attach the Mongo handler to the shared 'doctalk' logger (idempotent)."""
    logger = logging.getLogger("doctalk")
    if any(isinstance(h, MongoLogHandler) for h in logger.handlers):
        return
    handler = MongoLogHandler(db, settings)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --- metrics ---

def compute_stats(log_repo) -> dict:
    """Thin wrapper so routers never touch collections directly."""
    return log_repo.stats()
