"""API contract (request/response models). The Dash UI is built against these.

Keep in sync with docs/api_contract.md."""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


# --- API models ---

class Location(BaseModel):
    page: int | None = None            # PDF
    heading: str | None = None         # DOCX / MD
    bbox: list[float] | None = None    # PDF highlight [x0, y0, x1, y1]
    char_start: int | None = None
    char_end: int | None = None


class Citation(BaseModel):
    n: int                             # the [n] marker shown in the answer
    sentence_id: str
    document_id: str
    document_name: str
    document_deleted: bool = False
    location: Location
    text: str                          # always rendered from MongoDB, never from model output


class AnswerSegment(BaseModel):
    text: str
    citations: list[int]               # -> Citation.n, rendered as [1][2]


class AnswerMetrics(BaseModel):
    total_ms: int
    tokens_in: int
    tokens_out: int
    cost_usd: float


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    message_id: str
    status: Literal["ok", "partial", "refused", "error"]
    refusal_reason: str | None = None
    rewritten_question: str | None = None
    segments: list[AnswerSegment]
    citations: list[Citation]
    dropped_claims: int = 0
    metrics: AnswerMetrics


class DocumentInfo(BaseModel):
    id: str
    name: str
    type: str
    sentence_count: int
    created_at: str


class Passage(BaseModel):
    citation: Citation
    context_before: list[str] = []
    context_after: list[str] = []
