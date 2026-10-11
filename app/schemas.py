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
    kind: Literal["fact", "insight", "gap"] = "fact"  # insight = assistant's grounded commentary
    format: Literal["paragraph", "bullet", "text"] = "text"


class AnswerMetrics(BaseModel):
    total_ms: int
    tokens_in: int                     # all LLM calls: rewrite, translate, answer, verifier
    tokens_out: int
    cost_usd: float
    # Wall time per stage: rewrite, translate, retrieve (embed + search + re-rank),
    # generate (answer LLM), verify (rules + judge LLM), other (history, persistence, logging)
    timings_ms: dict[str, float] = {}


class RemovedStatement(BaseModel):
    """A statement the verifier removed: NOT supported by the documents, shown only as a warning."""
    text: str
    kind: Literal["fact", "insight", "gap"] = "fact"
    reasons: list[str] = []
    checked_by: str                    # rules | nli_fail | judge | judge_unavailable | gap


class AnswerDiagnostics(BaseModel):
    removed: list[RemovedStatement] = []
    translated_query: str | None = None  # the question translated into the documents' language
    attempts: int = 1                    # 2 = nothing verified at first, retried with a wider search
    judge_calls: int = 0                 # statements the LLM judge had to decide


class AskRequest(BaseModel):
    question: str
    style: str | None = None  # optional tone / format, e.g. "casual", "executive summary", "in Turkish"
    include_public: bool = False  # also search the shared public pool of documents


class AskResponse(BaseModel):
    message_id: str
    status: Literal["ok", "partial", "refused", "error"]
    refusal_reason: str | None = None
    rewritten_question: str | None = None
    answer: str = ""                   # markdown rendering of the segments with [n] markers
    segments: list[AnswerSegment]
    citations: list[Citation]
    dropped_claims: int = 0
    metrics: AnswerMetrics
    diagnostics: AnswerDiagnostics = AnswerDiagnostics()


class DocumentInfo(BaseModel):
    id: str
    name: str
    type: str
    sentence_count: int
    created_at: str
    conversation_id: str | None = None
    is_public: bool = False


class Passage(BaseModel):
    citation: Citation
    context_before: list[str] = []
    context_after: list[str] = []
