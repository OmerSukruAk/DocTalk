"""Persistence: Mongo client + indexes, owner-scoped repositories, and original files.

EVERY query filters on owner_key, and it is applied here and nowhere else, so a router
or graph node cannot accidentally read another user's data. Documents now belong to a
conversation; the only cross-tenant exception is the shared public pool (``is_public``).
Original uploads live in MongoDB GridFS, never on a local filesystem folder.
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
    db.documents.create_index([("owner_key", ASCENDING), ("conversation_id", ASCENDING)])
    db.documents.create_index(
        [("owner_key", ASCENDING), ("conversation_id", ASCENDING), ("content_hash", ASCENDING)]
    )
    db.documents.create_index([("is_public", ASCENDING)])

    db.sentences.create_index([("owner_key", ASCENDING)])
    db.sentences.create_index([("owner_key", ASCENDING), ("conversation_id", ASCENDING)])
    db.sentences.create_index([("owner_key", ASCENDING), ("document_id", ASCENDING), ("idx", ASCENDING)])
    db.sentences.create_index([("is_public", ASCENDING)])

    db.chunks.create_index([("owner_key", ASCENDING)])
    db.chunks.create_index([("owner_key", ASCENDING), ("conversation_id", ASCENDING)])
    db.chunks.create_index([("owner_key", ASCENDING), ("document_id", ASCENDING), ("chunk_idx", ASCENDING)])
    db.chunks.create_index([("is_public", ASCENDING)])

    db.conversations.create_index([("owner_key", ASCENDING)])
    db.conversations.create_index([("owner_key", ASCENDING), ("created_at", ASCENDING)])

    db.messages.create_index([("owner_key", ASCENDING), ("conversation_id", ASCENDING), ("created_at", ASCENDING)])

    db.request_logs.create_index([("owner_key", ASCENDING)])
    db.request_logs.create_index(
        [("created_at", ASCENDING)], expireAfterSeconds=settings.log_ttl_days * 86400
    )

    db.system_logs.create_index([("created_at", ASCENDING)], expireAfterSeconds=settings.log_ttl_days * 86400)


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


def _owner_or_public(owner_key: str, include_public: bool) -> dict:
    """Owner filter that, when ``include_public``, also admits the shared public pool."""
    if include_public:
        return {"$or": [{"owner_key": owner_key}, {"is_public": True}]}
    return {"owner_key": owner_key}


class DocumentRepo:
    def __init__(self, db: Database):
        self._c = db.documents

    def count(self, owner_key: str, conversation_id: str) -> int:
        return self._c.count_documents({"owner_key": owner_key, "conversation_id": conversation_id})

    def create(
        self,
        owner_key: str,
        name: str,
        type_: str,
        file_id: str,
        sentence_count: int = 0,
        document_id: str | None = None,
        content_hash: str | None = None,
        conversation_id: str | None = None,
        is_public: bool = False,
    ) -> str:
        document_id = document_id or "d_" + uuid.uuid4().hex[:16]
        doc = {
            "_id": document_id,
            "owner_key": owner_key,
            "name": name,
            "type": type_,
            "file_id": file_id,
            "conversation_id": conversation_id,
            "is_public": is_public,
            "sentence_count": sentence_count,
            "status": "ready",
            "created_at": now(),
        }
        if content_hash:
            doc["content_hash"] = content_hash
        self._c.insert_one(doc)
        return document_id

    def list(self, owner_key: str, conversation_id: str | None = None) -> list[dict]:
        q: dict = {"owner_key": owner_key}
        if conversation_id:
            q["conversation_id"] = conversation_id
        docs = self._c.find(q).sort("created_at", 1)
        return [d for d in (_clean(x) for x in docs) if d]

    def get(self, owner_key: str, document_id: str) -> dict | None:
        return _clean(self._c.find_one({"_id": document_id, "owner_key": owner_key}))

    def get_scoped(self, owner_key: str, document_id: str, include_public: bool) -> dict | None:
        q = {"_id": document_id}
        q.update(_owner_or_public(owner_key, include_public))
        return _clean(self._c.find_one(q))

    def find_by_hash(self, owner_key: str, conversation_id: str, content_hash: str) -> dict | None:
        return _clean(
            self._c.find_one(
                {"owner_key": owner_key, "conversation_id": conversation_id, "content_hash": content_hash}
            )
        )

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

    def bulk_insert(self, owner_key: str, sentences: list[Sentence], conversation_id: str | None = None, is_public: bool = False) -> None:
        if not sentences:
            return
        docs = [
            {
                "_id": s.sentence_id,
                "owner_key": owner_key,
                "document_id": s.document_id,
                "conversation_id": conversation_id,
                "is_public": is_public,
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

    def get_many(self, owner_key: str, sentence_ids: list[str], include_public: bool = False) -> list[dict]:
        if not sentence_ids:
            return []
        q: dict = {"_id": {"$in": sentence_ids}}
        q.update(_owner_or_public(owner_key, include_public))
        docs = self._c.find(q)
        by_id = {str(d["_id"]): _clean(d) for d in docs}
        return [by_id[sid] for sid in sentence_ids if sid in by_id]

    def neighbors(self, owner_key: str, document_id: str, idx: int, radius: int, include_public: bool = False) -> list[dict]:
        if radius <= 0:
            return []
        q: dict = {
            "document_id": document_id,
            "idx": {"$gte": idx - radius, "$lte": idx + radius},
        }
        q.update(_owner_or_public(owner_key, include_public))
        docs = self._c.find(q).sort("idx", 1)
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

    def bulk_insert(self, owner_key: str, chunks: list[Chunk], conversation_id: str | None = None, is_public: bool = False) -> None:
        if not chunks:
            return
        docs = [
            {
                "_id": c.chunk_id,
                "owner_key": owner_key,
                "conversation_id": conversation_id,
                "is_public": is_public,
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

    def get_many(self, owner_key: str, chunk_ids: list[str], include_public: bool = False) -> list[dict]:
        if not chunk_ids:
            return []
        q: dict = {"_id": {"$in": chunk_ids}}
        q.update(_owner_or_public(owner_key, include_public))
        docs = self._c.find(q)
        by_id = {str(d["_id"]): _clean(d) for d in docs}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def all_for_owner(self, owner_key: str) -> list[dict]:
        docs = self._c.find({"owner_key": owner_key}).sort([("document_id", 1), ("chunk_idx", 1)])
        return [d for d in (_clean(x) for x in docs) if d]

    def all_for_scope(self, owner_key: str, conversation_id: str, include_public: bool) -> list[dict]:
        """Chunks visible to a conversation: its own chunks plus (optionally) the public pool."""
        ors: list[dict] = [{"owner_key": owner_key, "conversation_id": conversation_id}]
        if include_public:
            ors.append({"is_public": True})
        docs = self._c.find({"$or": ors}).sort([("document_id", 1), ("chunk_idx", 1)])
        return [d for d in (_clean(x) for x in docs) if d]

    def delete_by_document(self, owner_key: str, document_id: str) -> int:
        res = self._c.delete_many({"owner_key": owner_key, "document_id": document_id})
        return res.deleted_count

    def fingerprint(self, owner_key: str, conversation_id: str, include_public: bool) -> tuple:
        """Cheap content fingerprint of a conversation's chunk set (count + document ids).

        Lets every replica notice that another replica added/removed a document, so the
        in-process BM25 cache can never serve a stale index."""
        ors: list[dict] = [{"owner_key": owner_key, "conversation_id": conversation_id}]
        if include_public:
            ors.append({"is_public": True})
        q = {"$or": ors}
        count = self._c.count_documents(q)
        doc_ids = tuple(sorted(self._c.distinct("document_id", q)))
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
        """The LAST ``limit`` messages, oldest first (list_messages returns the first ones)."""
        docs = (
            self._msgs.find({"owner_key": owner_key, "conversation_id": conversation_id})
            .sort("created_at", -1)
            .limit(limit)
        )
        return [{"role": m["role"], "text": m.get("text", "")} for m in reversed(list(docs))]


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
                    "stages": {"$push": "$node_ms"},
                }
            }
        ]
        rows = list(self._c.aggregate(pipeline))
        if not rows:
            return {
                "requests": 0, "avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "cost_usd": 0.0,
                "refusal_rate": 0.0, "partial_rate": 0.0, "error_rate": 0.0, "judge_usage_rate": 0.0,
                "stages": {},
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
            "stages": _stage_stats(row.get("stages") or []),
        }


def _stage_stats(node_ms_list: list) -> dict:
    """Per-stage latency over the requests in which that stage ran:
    {"retrieve": {"runs": 40, "avg_ms": 812.3, "p50_ms": 640.0, "p95_ms": 2100.5}, ...}"""
    by_stage: dict[str, list[float]] = {}
    for node_ms in node_ms_list:
        for stage, ms in (node_ms or {}).items():
            if isinstance(ms, (int, float)):
                by_stage.setdefault(stage, []).append(float(ms))
    out: dict[str, dict] = {}
    for stage, values in by_stage.items():
        values.sort()
        out[stage] = {
            "runs": len(values),
            "avg_ms": round(sum(values) / len(values), 1),
            "p50_ms": _percentile(values, 50),
            "p95_ms": _percentile(values, 95),
        }
    return out


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 1)
    k = (len(values) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    frac = k - lo
    return round(values[lo] + (values[hi] - values[lo]) * frac, 1)


# --- file store (GridFS) ---

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(filename: str) -> str:
    base = os.path.basename(filename or "file")
    base = _UNSAFE.sub("_", base).strip("._") or "file"
    return base[:120]


class FileStore:
    """Original uploads stored in MongoDB GridFS — never written to a local folder.

    Each file is tagged with ``owner_key`` + ``document_id`` so it can be read back and
    cascade-deleted with its document. GridFS (rather than a single BSON field) is used
    because uploads may exceed Mongo's 16 MB document limit.
    """

    def __init__(self, db: Database, max_bytes: int):
        from gridfs import GridFS

        self._fs = GridFS(db)
        self.max_bytes = max_bytes

    def save(self, owner_key: str, document_id: str, filename: str, content: bytes) -> str:
        if len(content) > self.max_bytes:
            raise ValueError("file too large")
        self.delete_document(owner_key, document_id)
        fid = self._fs.put(
            content,
            filename=safe_name(filename),
            owner_key=owner_key,
            document_id=document_id,
        )
        return str(fid)

    def read(self, owner_key: str, document_id: str) -> bytes:
        f = self._fs.find_one({"owner_key": owner_key, "document_id": document_id})
        if f is None:
            raise FileNotFoundError(document_id)
        return f.read()

    def delete_document(self, owner_key: str, document_id: str) -> None:
        for f in self._fs.find({"owner_key": owner_key, "document_id": document_id}):
            try:
                self._fs.delete(f._id)
            except Exception:
                pass
