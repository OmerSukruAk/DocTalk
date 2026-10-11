"""Test fixtures: the real application graph wired to in-memory fakes.

Mongo -> mongomock, Qdrant -> qdrant-client's in-memory mode, embeddings -> a deterministic
hashed bag-of-words embedder, LLM -> a scripted extractive fake. Nothing here touches the
network, so the whole suite runs offline in seconds and exercises the REAL retrieval, gate,
verification and workflow code."""
from __future__ import annotations

import hashlib
import math
import re
import sys
import tempfile
from pathlib import Path

import mongomock
import pytest
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.ingestion import IngestionService  # noqa: E402
from app.agent.llm import ChatResult  # noqa: E402
from app.agent.rerank import ReRanker  # noqa: E402
from app.agent.retrieval import STOPWORDS, BM25Index  # noqa: E402
from app.agent.workflow import build_graph  # noqa: E402
from app.config import Settings  # noqa: E402
from app.dependencies import Services  # noqa: E402
from app.observability import RequestLogger  # noqa: E402
from app.repositories import storage  # noqa: E402
from app.repositories.storage import (  # noqa: E402
    ChunkRepo, ConversationRepo, DocumentRepo, LogRepo, SentenceRepo,
)
from app.repositories.vector_store import VectorStore  # noqa: E402

DIM = 128
_W = re.compile(r"\w+", re.UNICODE)


def _terms(text: str) -> list[str]:
    return [t for t in _W.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class FakeEmbeddings:
    """Hashed bag-of-content-words, L2-normalised: related texts share words, so cosine is
    meaningfully high; unrelated texts are ~0 (like a real model on an off-topic query)."""

    configured = True
    dimensions = DIM
    model = "fake-bow"

    async def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * DIM
            for w in _terms(t):
                v[int(hashlib.md5(w[:5].encode()).hexdigest(), 16) % DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    async def embed_one(self, text):
        return (await self.embed([text]))[0]


class FakeLLM:
    """Extractive stand-in for the chat model. ``claims_for`` can be overridden per test."""

    configured = True

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.fail = False
        self.extra_claims: list[dict] = []  # injected 'hallucinated' claims

    async def chat(self, system, user, **kw):
        self.calls.append(("chat", user))
        if self.fail:
            raise RuntimeError("503 from provider")
        q = user.split("Latest question:")[1].split("\n")[0].strip()
        return ChatResult(f"standalone: {q}", 10, 5, 0.0)

    async def complete_json(self, system, user, **kw):
        self.calls.append(("json", user))
        if self.fail:
            raise RuntimeError("503 from provider")
        if "entailment judge" in system:
            return {"label": "supported", "reason": "ok"}, ChatResult("", 5, 5, 0.0)
        blocks = re.findall(r"<<<SOURCE id=(\S+) doc=.*?>>>\n(.*?)\n<<<END SOURCE>>>", user, re.S)
        question = user.split("Question:")[-1].split("JSON answer:")[0]
        q = set(_terms(question))
        scored = sorted(
            ((len(q & set(_terms(text))), sid, text) for sid, text in blocks), reverse=True
        )
        claims = [{"text": text, "citations": [sid]} for n, sid, text in scored[:1] if n >= 2 or (n == 1 and len(q) <= 2)]
        claims += self.extra_claims
        return {"claims": claims}, ChatResult("", 50, 20, 0.0)


class InMemoryFileStore:
    """GridFS stand-in: mongomock has no GridFS, and tests never touch the filesystem."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._data: dict[str, bytes] = {}

    def _key(self, owner_key: str, document_id: str) -> str:
        return f"{owner_key}:{document_id}"

    def save(self, owner_key: str, document_id: str, filename: str, content: bytes) -> str:
        if len(content) > self.max_bytes:
            raise ValueError("file too large")
        self._data[self._key(owner_key, document_id)] = content
        return "gridfs-id"

    def read(self, owner_key: str, document_id: str) -> bytes:
        try:
            return self._data[self._key(owner_key, document_id)]
        except KeyError:
            raise FileNotFoundError(document_id)

    def delete_document(self, owner_key: str, document_id: str) -> None:
        self._data.pop(self._key(owner_key, document_id), None)


class NoNLI:
    configured = False


@pytest.fixture()
def settings():
    return Settings(
        _env_file=None, llm_api_key="x", embed_api_key="x",
        embed_dimensions=DIM, embed_model="fake-bow", rate_limit_ask_per_min=0,
        rate_limit_upload_per_min=0, log_text=True,
    )


@pytest.fixture()
def services(settings):
    db = mongomock.MongoClient()["doctalk_test"]
    storage.ensure_indexes(db, settings)
    chunks = ChunkRepo(db)
    vectors = VectorStore(settings)
    vectors._client = QdrantClient(":memory:")
    vectors.ensure_collection(settings.embed_model, settings.embed_dimensions)
    bm25 = BM25Index(chunks)
    emb = FakeEmbeddings()
    files = InMemoryFileStore(settings.max_file_bytes)
    docs, sents, convs, logs = DocumentRepo(db), SentenceRepo(db), ConversationRepo(db), LogRepo(db)
    svc = Services(
        settings=settings, llm=FakeLLM(), embeddings=emb, nli=NoNLI(), reranker=ReRanker(settings),
        documents=docs, sentences=sents, chunks=chunks, conversations=convs, logs=logs,
        files=files, vectors=vectors, bm25=bm25,
        ingestion=IngestionService(
            settings=settings, documents=docs, sentences=sents, chunks=chunks, files=files,
            embeddings=emb, vectors=vectors, bm25=bm25,
        ),
        request_logger=RequestLogger(logs, settings),
    )
    svc.graph = build_graph(svc)
    return svc


HR_POLICY = b"""# HR Policy

## Annual Leave

Employees get annual leave of 25 days. Unused days may be carried over until 31 March.

## Sick Leave

Sick leave is 10 days per year. A medical certificate is required after 3 consecutive days.

## Remote Work

Remote work is allowed two days per week. Employees cannot work remotely from another country without approval.

## Expenses

| Category | Limit |
|---|---|
| Hotel | 120 EUR per night |
| Meals | 40 EUR per day |
"""


@pytest.fixture()
def hr_doc():
    return HR_POLICY
