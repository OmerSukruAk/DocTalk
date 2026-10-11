"""Composition root: builds every dependency and wires the application together.

This is the ONLY place object construction happens. ``main.py`` calls
``build_container()`` once at startup and stores the result on ``app.state``;
tests build their own fakes directly (see ``api/tests/conftest.py``) rather than
going through this module, so no two construction paths can drift apart.
"""

from __future__ import annotations

import logging

from pymongo import MongoClient

from app.config import Settings, get_settings
from app.dependencies import Services
from app.observability import RequestLogger, install_mongo_logging
from app.repositories import storage
from app.repositories.storage import (
    ChunkRepo,
    ConversationRepo,
    DocumentRepo,
    FileStore,
    LogRepo,
    SentenceRepo,
)
from app.repositories.vector_store import VectorStore
from app.agent.workflow import build_graph
from app.agent.ingestion import IngestionService
from app.agent.llm import ChatClient
from app.agent.rerank import ReRanker
from app.agent.retrieval import BM25Index, EmbeddingsClient
from app.agent.verification import NLIScorer

logger = logging.getLogger("doctalk")


def build_container(settings: Settings | None = None) -> tuple[Services, MongoClient]:
    """Construct the full application container and the Mongo client that owns it.

    Returns ``(services, client)``; the caller is responsible for closing the
    client on shutdown.
    """
    settings = settings or get_settings()

    # --- persistence ---
    client = storage.get_client(settings)
    db = storage.get_db(client, settings)
    storage.ensure_indexes(db, settings)
    install_mongo_logging(db, settings)

    files = FileStore(db, settings.max_file_bytes)
    documents = DocumentRepo(db)
    sentences = SentenceRepo(db)
    chunks = ChunkRepo(db)
    conversations = ConversationRepo(db)
    logs = LogRepo(db)
    vectors = VectorStore(settings)
    bm25 = BM25Index(chunks)

    # --- model clients ---
    llm = ChatClient(settings)
    embeddings = EmbeddingsClient(settings)
    nli = NLIScorer(settings)
    reranker = ReRanker(settings)

    # --- vector-store bootstrap (degrade to BM25-only on failure) ---
    if settings.embeddings_configured:
        try:
            vectors.ensure_collection(settings.embed_model, settings.embed_dimensions)
        except Exception as exc:  # Qdrant down / model mismatch -> BM25-only degraded mode
            logger.warning("Vector store unavailable, running BM25-only: %s", exc)
    else:
        logger.warning("EMBED_API_KEY not set: running BM25-only (no vector search)")

    # --- composite services ---
    ingestion = IngestionService(
        settings=settings,
        documents=documents,
        sentences=sentences,
        chunks=chunks,
        files=files,
        embeddings=embeddings,
        vectors=vectors,
        bm25=bm25,
    )

    services = Services(
        settings=settings,
        llm=llm,
        embeddings=embeddings,
        nli=nli,
        reranker=reranker,
        documents=documents,
        sentences=sentences,
        chunks=chunks,
        conversations=conversations,
        logs=logs,
        files=files,
        vectors=vectors,
        bm25=bm25,
        ingestion=ingestion,
        request_logger=RequestLogger(logs, settings),
    )
    services.graph = build_graph(services)
    return services, client
