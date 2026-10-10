"""Persistence: Mongo client + indexes, owner-scoped repositories, and original files.

EVERY query filters on owner_key, and it is applied here and nowhere else, so a router
or graph node cannot accidentally read another user's data.
"""

from __future__ import annotations
import uuid
import os
import re
from datetime import datetime, timezone
from typing import Any
from pymongo import ASCENDING, MongoClient
from pymongo.database import Database
from app.config import Settings
from app.domain import Chunk, Sentence


# --- MongoDB connection ---

def get_client(settings: Settings) -> MongoClient:
    return MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000, tz_aware=True)


def get_db(client: MongoClient, settings: Settings) -> Database:
    return client[settings.mongo_db]


def ensure_indexes(db: Database, settings: Settings) -> None:
    """Idempotent. owner_key leads every compound index so owner-scoped queries are covered."""
    db.documents.create_index([("owner_key", ASCENDING)])
    db.documents.create_index([("owner_key", ASCENDING), ("created_at", ASCENDING)])
    db.documents.create_index([("owner_key", ASCENDING), ("content_hash", ASCENDING)])

    db.sentences.create_index([("owner_key", ASCENDING)])
    db.sentences.create_index([("owner_key", ASCENDING), ("document_id", ASCENDING), ("idx", ASCENDING)])

    db.chunks.create_index([("owner_key", ASCENDING)])
    db.chunks.create_index([("owner_key", ASCENDING), ("document_id", ASCENDING), ("chunk_idx", ASCENDING)])

    db.conversations.create_index([("owner_key", ASCENDING)])
    db.conversations.create_index([("owner_key", ASCENDING), ("created_at", ASCENDING)])

    db.messages.create_index([("owner_key", ASCENDING), ("conversation_id", ASCENDING), ("created_at", ASCENDING)])

    db.request_logs.create_index([("owner_key", ASCENDING)])
    db.request_logs.create_index(
        [("created_at", ASCENDING)], expireAfterSeconds=settings.log_ttl_days * 86400
    )


# --- repositories ---

def now() -> datetime:
    return datetime.now(timezone.utc)


def _clean(doc: dict | None) -> dict | None:
    if not doc:
        return None
    out = dict(doc)
    out["id"] = str(out.pop("_id"))
    if isinstance(out.get("created_at"), datetime):
        out["created_at"] = out["created_at"].isoformat()
    return out


class DocumentRepo:
    def __init__(self, db: Database):
        self._c = db.documents

    def count(self, owner_key: str) -> int:
        return self._c.count_documents({"owner_key": owner_key})

    def create(
        self,
        owner_key: str,
        name: str,
        type_: str,
        file_path: str,
        sentence_count: int = 0,
        document_id: str | None = None,
        content_hash: str | None = None,
    ) -> str:
        document_id = document_id or "d_" + uuid.uuid4().hex[:16]
        doc = {
            "_id": document_id,
            "owner_key": owner_key,
            "name": name,
            "type": type_,
            "file_path": file_path,
            "sentence_count": sentence_count,
            "status": "ready",
            "created_at": now(),
        }
        if content_hash:
            doc["content_hash"] = content_hash
        self._c.insert_one(doc)
        return document_id

    def list(self, owner_key: str) -> list[dict]:
        docs = self._c.find({"owner_key": owner_key}).sort("created_at", 1)
        return [d for d in (_clean(x) for x in docs) if d]

    def get(self, owner_key: str, document_id: str) -> dict | None:
        return _clean(self._c.find_one({"_id": document_id, "owner_key": owner_key}))

    def find_by_hash(self, owner_key: str, content_hash: str) -> dict | None:
        return _clean(self._c.find_one({"owner_key": owner_key, "content_hash": content_hash}))

    def set_sentence_count(self, owner_key: str, document_id: str, count: int) -> None:
        self._c.update_one(
            {"_id": document_id, "owner_key": owner_key}, {"$set": {"sentence_count": count}}
        )

    def delete(self, owner_key: str, document_id: str) -> bool:
        res = self._c.delete_one({"_id": document_id, "owner_key": owner_key})
        return res.deleted_count > 0


class SentenceRepo:
    def __init__(self, db: Database):
        self._c = db.sentences

    def bulk_insert(self, owner_key: str, sentences: list[Sentence]) -> None:
        if not sentences:
            return
        docs = [
            {
                "_id": s.sentence_id,
                "owner_key": owner_key,
                "document_id": s.document_id,
                "idx": s.idx,
                "text": s.text,
                "page": s.page,
                "heading": s.heading,
                "bbox": s.bbox,
                "char_start": s.char_start,
                "char_end": s.char_end,
                "is_table_row": s.is_table_row,
                "chunk_id": s.chunk_id,
            }
            for s in sentences
        ]
        self._c.insert_many(docs, ordered=False)

    def get(self, owner_key: str, sentence_id: str) -> dict | None:
        return _clean(self._c.find_one({"_id": sentence_id, "owner_key": owner_key}))

    def get_many(self, owner_key: str, sentence_ids: list[str]) -> list[dict]:
        if not sentence_ids:
            return []
        docs = self._c.find({"_id": {"$in": sentence_ids}, "owner_key": owner_key})
        by_id = {str(d["_id"]): _clean(d) for d in docs}
        return [by_id[sid] for sid in sentence_ids if sid in by_id]

    def neighbors(self, owner_key: str, document_id: str, idx: int, radius: int) -> list[dict]:
        if radius <= 0:
            return []
        docs = self._c.find(
            {
                "owner_key": owner_key,
                "document_id": document_id,
                "idx": {"$gte": idx - radius, "$lte": idx + radius},
        }).sort("idx", 1)
        return [d for d in (_clean(x) for x in docs) if d]

    def all_for_owner(self, owner_key: str) -> list[dict]:
        docs = self._c.find({"owner_key": owner_key}).sort([("document_id", 1), ("idx", 1)])
        return [d for d in (_clean(x) for x in docs) if d]

    def delete_by_document(self, owner_key: str, document_id: str) -> int:
        res = self._c.delete_many({"owner_key": owner_key, "document_id": document_id})
        return res.deleted_count


class ChunkRepo:
    """Parent chunks (passages) — the unit that gets embedded and retrieved."""

    def __init__(self, db: Database):
        self._c = db.chunks

    def bulk_insert(self, owner_key: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        docs = [
            {
                "_id": c.chunk_id,
                "owner_key": owner_key,
                "document_id": c.document_id,
                "chunk_idx": c.chunk_idx,
                "text": c.text,
                "sentence_ids": c.sentence_ids,
                "heading": c.heading,
                "page": c.page,
                "token_count": c.token_count,
                "headings": c.headings,
                "context_before": c.context_before,
                "is_table": c.is_table,
            }
            for c in chunks
        ]
        self._c.insert_many(docs, ordered=False)

    def get(self, owner_key: str, chunk_id: str) -> dict | None:
        return _clean(self._c.find_one({"_id": chunk_id, "owner_key": owner_key}))

    def get_many(self, owner_key: str, chunk_ids: list[str]) -> list[dict]:
        if not chunk_ids:
            return []
        docs = self._c.find({"_id": {"$in": chunk_ids}, "owner_key": owner_key})
        by_id = {str(d["_id"]): _clean(d) for d in docs}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def all_for_owner(self, owner_key: str) -> list[dict]:
        docs = self._c.find({"owner_key": owner_key}).sort([("document_id", 1), ("chunk_idx", 1)])
        return [d for d in (_clean(x) for x in docs) if d]

    def delete_by_document(self, owner_key: str, document_id: str) -> int:
        res = self._c.delete_many({"owner_key": owner_key, "document_id": document_id})
        return res.deleted_count

    def fingerprint(self, owner_key: str) -> tuple:
        """Cheap content fingerprint of an owner's chunk set (count + document ids).

        Lets every replica notice that another replica added/removed a document, so the
        in-process BM25 cache can never serve a stale index."""
        count = self._c.count_documents({"owner_key": owner_key})
        doc_ids = tuple(sorted(self._c.distinct("document_id", {"owner_key": owner_key})))
        return (count, doc_ids)


class ConversationRepo:
    def __init__(self, db: Database):
        self._convs = db.conversations
        self._msgs = db.messages

    def create(self, owner_key: str, title: str = "New conversation") -> str:
        conversation_id = "c_" + uuid.uuid4().hex[:16]
        self._convs.insert_one(
            {"_id": conversation_id, "owner_key": owner_key, "title": title, "created_at": now()}
        )
        return conversation_id

    def list(self, owner_key: str) -> list[dict]:
        docs = self._convs.find({"owner_key": owner_key}).sort("created_at", -1)
        return [d for d in (_clean(x) for x in docs) if d]

    def get(self, owner_key: str, conversation_id: str) -> dict | None:
        return _clean(self._convs.find_one({"_id": conversation_id, "owner_key": owner_key}))

    def delete(self, owner_key: str, conversation_id: str) -> bool:
        self._msgs.delete_many({"owner_key": owner_key, "conversation_id": conversation_id})
        res = self._convs.delete_one({"_id": conversation_id, "owner_key": owner_key})
        return res.deleted_count > 0

    def add_message(
        self,
        owner_key: str,
        conversation_id: str,
        role: str,
        text: str,
        segments: list[dict] | None = None,
        citations: list[dict] | None = None,
        status: str = "ok",
    ) -> str:
        message_id = "m_" + uuid.uuid4().hex[:16]
        self._msgs.insert_one(
            {
                "_id": message_id,
                "owner_key": owner_key,
                "conversation_id": conversation_id,
                "role": role,
                "text": text,
                "segments": segments or [],
                "citations": citations or [],
                "status": status,
                "created_at": now(),
            }
        )
        return message_id

    def mark_document_deleted(self, owner_key: str, document_id: str) -> int:
        """Flag every stored citation that points at a now-deleted document."""
        res = self._msgs.update_many(
            {"owner_key": owner_key, "citations.document_id": document_id},
            {"$set": {"citations.$[c].document_deleted": True}},
            array_filters=[{"c.document_id": document_id}],
        )
        return res.modified_count

    def list_messages(self, owner_key: str, conversation_id: str, limit: int = 50) -> list[dict]:
        docs = (
            self._msgs.find({"owner_key": owner_key, "conversation_id": conversation_id})
            .sort("created_at", 1)
            .limit(limit)
        )
        return [d for d in (_clean(x) for x in docs) if d]

    def recent_history(self, owner_key: str, conversation_id: str, limit: int = 6) -> list[dict]:
        docs = list(self.list_messages(owner_key, conversation_id, limit=limit))
        return [{"role": m["role"], "text": m.get("text", "")} for m in docs]


class LogRepo:
    def __init__(self, db: Database):
        self._c = db.request_logs

    def insert(self, record: dict[str, Any]) -> None:
        record.setdefault("created_at", now())
        record.setdefault("_id", "r_" + uuid.uuid4().hex[:16])
        self._c.insert_one(record)

    def stats(self) -> dict:
        """Aggregates only: latency percentiles, cost, refusal rate, judge-usage rate."""
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "count": {"$sum": 1},
                    "total_ms": {"$sum": "$total_ms"},
                    "cost_usd": {"$sum": "$cost_usd"},
                    "refused": {"$sum": {"$cond": [{"$eq": ["$status", "refused"]}, 1, 0]}},
                    "partial": {"$sum": {"$cond": [{"$eq": ["$status", "partial"]}, 1, 0]}},
                    "errors": {"$sum": {"$cond": [{"$eq": ["$status", "error"]}, 1, 0]}},
                    "judge_calls": {"$sum": {"$ifNull": ["$judge_calls", 0]}},
                    "latencies": {"$push": "$total_ms"},
                }
            }
        ]
        rows = list(self._c.aggregate(pipeline))
        if not rows:
            return {
                "requests": 0, "avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "cost_usd": 0.0,
                "refusal_rate": 0.0, "partial_rate": 0.0, "error_rate": 0.0, "judge_usage_rate": 0.0,
            }
        row = rows[0]
        latencies = sorted(x for x in row.get("latencies", []) if isinstance(x, (int, float)))
        n = row["count"] or 1
        return {
            "requests": row["count"],
            "avg_ms": round(row["total_ms"] / n, 1),
            "p50_ms": _percentile(latencies, 50),
            "p95_ms": _percentile(latencies, 95),
            "cost_usd": round(row["cost_usd"] or 0.0, 6),
            "refusal_rate": round(row["refused"] / n, 3),
            "partial_rate": round(row["partial"] / n, 3),
            "error_rate": round(row["errors"] / n, 3),
            "judge_usage_rate": round((row["judge_calls"] or 0) / n, 3),
        }


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 1)
    k = (len(values) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    frac = k - lo
    return round(values[lo] + (values[hi] - values[lo]) * frac, 1)


# --- file store ---

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(filename: str) -> str:
    base = os.path.basename(filename or "file")
    base = _UNSAFE.sub("_", base).strip("._") or "file"
    return base[:120]


class FileStore:
    """One directory per document: <files_dir>/<owner_hash>/<document_id>/<safename>."""

    def __init__(self, root: str, max_bytes: int):
        self.root = root
        self.max_bytes = max_bytes

    @staticmethod
    def _owner_dir(owner_key: str) -> str:
        return uuid.uuid5(uuid.NAMESPACE_URL, owner_key).hex

    def _doc_dir(self, owner_key: str, document_id: str) -> str:
        return os.path.join(self.root, self._owner_dir(owner_key), document_id)

    def save(self, owner_key: str, document_id: str, filename: str, content: bytes) -> str:
        if len(content) > self.max_bytes:
            raise ValueError("file too large")
        directory = self._doc_dir(owner_key, document_id)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, safe_name(filename))
        with open(path, "wb") as fh:
            fh.write(content)
        return path

    def path_for(self, owner_key: str, document_id: str, filename: str) -> str:
        return os.path.join(self._doc_dir(owner_key, document_id), safe_name(filename))

    def delete_document(self, owner_key: str, document_id: str) -> None:
        directory = self._doc_dir(owner_key, document_id)
        if not os.path.isdir(directory):
            return
        for name in os.listdir(directory):
            try:
                os.remove(os.path.join(directory, name))
            except OSError:
                pass
        try:
            os.rmdir(directory)
        except OSError:
            pass
