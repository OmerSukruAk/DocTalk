"""Shared domain types and pure value helpers.

This module is deliberately dependency-free (no FastAPI, no persistence, no LLM
imports) so that every layer — repositories, services, routers — can import from it
without creating a layering violation. It holds the data that crosses layer
boundaries: the Sentence write-model and the citation shape that both the answer
graph and the documents router render.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Sentence:
    """One segmentable unit of a document — the atom every citation points at.

    ``chunk_id`` links each sentence to its parent chunk (the passage it was
    grouped into for embedding/retrieval). Citations stay sentence-level; chunks
    are purely a retrieval concern.
    """

    sentence_id: str
    document_id: str
    idx: int
    text: str
    page: int | None
    heading: str | None
    bbox: list[float] | None
    char_start: int
    char_end: int
    is_table_row: bool
    chunk_id: str | None = None


def sentence_id(document_id: str, idx: int) -> str:
    """Stable, citation-friendly id: '<document_id>:s<idx>'."""
    return f"{document_id}:s{idx}"


def chunk_id(document_id: str, chunk_idx: int) -> str:
    """Stable parent-chunk id: '<document_id>:c<chunk_idx>'."""
    return f"{document_id}:c{chunk_idx}"


@dataclass
class Chunk:
    """A parent chunk: a passage of contiguous sentences embedded as one unit.

    Retrieval happens at this granularity (a chunk carries enough multi-sentence
    context for a meaningful embedding); citations still point at the child
    sentences it contains.
    """

    chunk_id: str
    document_id: str
    chunk_idx: int
    text: str
    sentence_ids: list[str]
    heading: str | None
    page: int | None
    token_count: int
    # Structure-aware chunking metadata (see agent/ingestion.py::chunk)
    headings: list[str] = field(default_factory=list)  # every section heading covered by the chunk
    context_before: str = ""  # overlap: tail of the previous chunk, used for embedding only
    is_table: bool = False


def citation_dict(
    *,
    n: int,
    sentence_id: str,
    document_id: str,
    document_name: str,
    text: str,
    page: int | None = None,
    heading: str | None = None,
    bbox: list[float] | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
) -> dict:
    """Render the shared citation shape (mirrors ``app.schemas.Citation``).

    The citation text is always the sentence as stored, never model output.
    """
    return {
        "n": n,
        "sentence_id": sentence_id,
        "document_id": document_id,
        "document_name": document_name,
        "document_deleted": False,
        "location": {
            "page": page,
            "heading": heading,
            "bbox": bbox,
            "char_start": char_start,
            "char_end": char_end,
        },
        "text": text,
    }
