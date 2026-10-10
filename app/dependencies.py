"""Application container: one object holding every shared dependency.

Built once at startup and stored on app.state; routers and graph nodes receive it
through FastAPI dependencies. Tests can construct a Services with fakes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import Request

from app.config import Settings

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
from app.observability import RequestLogger
from app.repositories.storage import (
    ChunkRepo,
    ConversationRepo,
    DocumentRepo,
    FileStore,
    LogRepo,
    SentenceRepo,
)
from app.repositories.vector_store import VectorStore
from app.agent.ingestion import IngestionService
from app.agent.llm import ChatClient
from app.agent.rerank import ReRanker
from app.agent.retrieval import BM25Index, EmbeddingsClient
from app.agent.verification import NLIScorer


@dataclass
class Services:
    """The application container handed to routers and graph nodes via ``Depends()``.

    Grouped by role for readability. The compiled LangGraph is attached after
    construction because it needs the fully-built container (see ``app/container.py``).
    """

    # configuration
    settings: Settings

    # model clients (chat / embeddings / local NLI / cross-encoder re-ranker)
    llm: ChatClient
    embeddings: EmbeddingsClient
    nli: NLIScorer
    reranker: ReRanker

    # repositories — the only code that talks to Mongo / Qdrant
    documents: DocumentRepo
    sentences: SentenceRepo
    chunks: ChunkRepo
    conversations: ConversationRepo
    logs: LogRepo
    files: FileStore
    vectors: VectorStore

    # in-process search index
    bm25: BM25Index

    # composite services
    ingestion: IngestionService
    request_logger: RequestLogger

    # compiled LangGraph workflow — set once by the composition root
    graph: "CompiledStateGraph | None" = field(init=False, default=None)


def get_services(request: Request) -> "Services":
    """FastAPI dependency: the container built in the app lifespan."""
    return request.app.state.services


def get_graph(request: Request):
    """FastAPI dependency: the compiled LangGraph."""
    return request.app.state.graph
