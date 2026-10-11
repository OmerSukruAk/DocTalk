"""The LangGraph workflow and its state.

rewrite -> retrieve_and_gate -(no)-> refuse
                            -(yes)-> generate -(LLM error)-> fail
                                              -> verify -(pass)-> finalize_answer
                                                        -(fail & attempt<MAX_RETRIES)-> retry_prepare
                                                        -(fail, retries used)-> finalize/refuse

Failure policy: a provider outage never becomes a 500 and never a guess. Rewrite falls back
to the user's own question, the judge fails closed, and generation errors end in an explicit
status="error" answer that is still persisted and logged.
answer_question() orchestrates one question end-to-end: load history, invoke the graph,
persist the messages and emit one request_logs record."""

from __future__ import annotations
import asyncio
import logging
import time
import uuid
from typing import TypedDict
from langgraph.graph import END, StateGraph
from app.domain import citation_dict
from app.agent.llm import (
    GENERATE_SYSTEM,
    GENERATE_USER,
    REWRITE_SYSTEM,
    REWRITE_USER,
    build_history_block,
    build_sources_block,
)
from app.agent.retrieval import Hit, RetrievalSignals, hybrid_search_ex, passes_gate
from app.agent.verification import Claim, ClaimVerdict, verify_claims

logger = logging.getLogger("doctalk")

ERROR_REASON = "The answer service is temporarily unavailable. Please try again in a moment."


# --- graph state ---

class GraphState(TypedDict, total=False):
    owner_key: str
    conversation_id: str
    question: str
    include_public: bool
    rewritten_question: str
    history: list[dict]

    attempt: int
    top_k: int
    hits: list[Hit]
    relevant: bool
    signals: RetrievalSignals
    error: str | None

    claims: list[Claim]
    verdicts: list[ClaimVerdict]

    # outputs
    status: str  # ok | partial | refused
    refusal_reason: str | None
    segments: list[dict]
    citations: list[dict]
    dropped_claims: int

    # accounting
    node_ms: dict
    tokens_in: int
    tokens_out: int
    cost_usd: float
    judge_calls: int


# --- graph nodes ---

def _acc(state: GraphState, name: str, start: float, t_in: int = 0, t_out: int = 0, cost: float = 0.0) -> dict:
    node_ms = dict(state.get("node_ms") or {})
    node_ms[name] = round((time.perf_counter() - start) * 1000, 1)
    return {
        "node_ms": node_ms,
        "tokens_in": (state.get("tokens_in") or 0) + t_in,
        "tokens_out": (state.get("tokens_out") or 0) + t_out,
        "cost_usd": (state.get("cost_usd") or 0.0) + cost,
    }


class Nodes:
    def __init__(self, services):
        self.s = services

    # ---------------------------------------------------------------- rewrite
    async def rewrite(self, state: GraphState) -> dict:
        start = time.perf_counter()
        question = state["question"]
        history = state.get("history") or []
        rewritten = question
        t_in = t_out = 0
        cost = 0.0
        # Skip the LLM entirely for the first turn (nothing to resolve -> saves tokens).
        if history and self.s.llm.configured:
            try:
                res = await self.s.llm.chat(
                    REWRITE_SYSTEM,
                    REWRITE_USER.format(history=build_history_block(history), question=question),
                    max_tokens=256,
                )
                rewritten = res.text.strip() or question
                t_in, t_out, cost = res.tokens_in, res.tokens_out, res.cost_usd
            except Exception as exc:  # rewriting is an optimisation: fall back to the raw question
                logger.warning("question rewrite failed, using the original: %s", exc)
        return {"rewritten_question": rewritten, **_acc(state, "rewrite", start, t_in, t_out, cost)}

    # -------------------------------------------------- retrieve_and_gate
    async def retrieve_and_gate(self, state: GraphState) -> dict:
        start = time.perf_counter()
        query = state.get("rewritten_question") or state["question"]
        query_vector = None
        if self.s.embeddings.configured:
            try:
                query_vector = await self.s.embeddings.embed_one(query)
            except Exception:
                query_vector = None  # fall back to BM25-only
        # Mongo, BM25 and the cross-encoder are synchronous: run them in a worker thread.
        hits, signals = await asyncio.to_thread(
            lambda: hybrid_search_ex(
                sentences=self.s.sentences,
                chunks=self.s.chunks,
                documents=self.s.documents,
                vectors=self.s.vectors,
                bm25=self.s.bm25,
                reranker=self.s.reranker,
                owner_key=state["owner_key"],
                conversation_id=state["conversation_id"],
                include_public=state.get("include_public", False),
                query=query,
                query_vector=query_vector,
                top_k=state.get("top_k") or self.s.settings.top_k,
                neighbours=self.s.settings.context_neighbours,
            )
        )
        relevant = passes_gate(hits, signals, self.s.settings)
        return {"hits": hits, "signals": signals, "relevant": relevant, **_acc(state, "retrieve", start)}

    # --------------------------------------------------------------- generate
    async def generate(self, state: GraphState) -> dict:
        start = time.perf_counter()
        hits = state.get("hits") or []
        sources = [
            {"sentence_id": h.sentence_id, "document_name": h.document_name, "text": h.text}
            for h in hits
        ]
        if not self.s.llm.configured:
            return {"claims": [], **_acc(state, "generate", start)}
        # Generate against the *resolved* question ("And remote work?" -> "What is the remote
        # work policy?"), the same one retrieval used, so follow-ups stay grounded.
        question = state.get("rewritten_question") or state["question"]
        try:
            data, res = await self.s.llm.complete_json(
                GENERATE_SYSTEM,
                GENERATE_USER.format(sources=build_sources_block(sources), question=question),
                max_tokens=1024,
            )
        except Exception as exc:
            logger.error("generation failed: %s", exc)
            return {"claims": [], "error": f"{type(exc).__name__}: {exc}", **_acc(state, "generate", start)}
        claims: list[Claim] = []
        for raw in data.get("claims", []) or []:
            text = str(raw.get("text", "")).strip()
            citations = [str(c) for c in (raw.get("citations") or []) if c]
            if text:
                claims.append(Claim(text=text, citations=citations))
        return {
            "claims": claims,
            **_acc(state, "generate", start, res.tokens_in, res.tokens_out, res.cost_usd),
        }

    # ----------------------------------------------------------------- verify
    async def verify(self, state: GraphState) -> dict:
        start = time.perf_counter()
        hits = state.get("hits") or []
        context_by_id = {
            h.sentence_id: {
                "sentence_id": h.sentence_id,
                "document_name": h.document_name,
                "text": h.text,
            }
            for h in hits
        }
        verdicts = await verify_claims(
            state.get("claims") or [],
            context_by_id=context_by_id,
            nli=self.s.nli,
            llm=self.s.llm,
            nli_pass=self.s.settings.nli_pass,
            nli_fail=self.s.settings.nli_fail,
        )
        judge_calls = sum(1 for v in verdicts if v.used_judge)
        return {
            "verdicts": verdicts,
            "judge_calls": (state.get("judge_calls") or 0) + judge_calls,
            **_acc(state, "verify", start),
        }

    # ---------------------------------------------------------- retry_prepare
    def retry_prepare(self, state: GraphState) -> dict:
        """Widen the net and re-retrieve, then regenerate (attempt is capped by routing)."""
        return {
            "attempt": (state.get("attempt") or 0) + 1,
            "top_k": (state.get("top_k") or self.s.settings.top_k) + 4,
            "claims": [],
            "verdicts": [],
        }

    # -------------------------------------------------------- finalize_answer
    def finalize_answer(self, state: GraphState) -> dict:
        verdicts = state.get("verdicts") or []
        kept = [v for v in verdicts if v.ok]
        dropped = [v for v in verdicts if not v.ok]
        hits = state.get("hits") or []
        hit_by_id = {h.sentence_id: h for h in hits}

        citations: list[dict] = []
        number: dict[str, int] = {}
        segments: list[dict] = []
        for verdict in kept:
            nums: list[int] = []
            for sid in verdict.claim.citations:
                hit = hit_by_id.get(sid)
                if hit is None:
                    continue
                if sid not in number:
                    number[sid] = len(number) + 1
                    citations.append(
                        citation_dict(
                            n=number[sid],
                            sentence_id=hit.sentence_id,
                            document_id=hit.document_id,
                            document_name=hit.document_name,
                            text=hit.text,
                            page=hit.page,
                            heading=hit.heading,
                            bbox=hit.bbox,
                            char_start=hit.char_start,
                            char_end=hit.char_end,
                        )
                    )
                nums.append(number[sid])
            nums = sorted(set(nums))
            if nums:
                segments.append({"text": verdict.claim.text, "citations": nums})

        if not segments:
            return {
                "status": "refused",
                "refusal_reason": (
                    "every candidate claim failed verification against the sources"
                    if verdicts
                    else "the documents do not contain enough information to answer this question"
                ),
                "segments": [],
                "citations": [],
                "dropped_claims": len(dropped),
            }
        return {
            "status": "ok" if not dropped else "partial",
            "refusal_reason": None,
            "segments": segments,
            "citations": citations,
            "dropped_claims": len(dropped),
        }

    # ----------------------------------------------------------------- refuse
    def refuse(self, state: GraphState) -> dict:
        return {
            "status": "refused",
            "refusal_reason": "no relevant content found in your documents",
            "segments": [],
            "citations": [],
            "dropped_claims": 0,
        }

    # ------------------------------------------------------------------- fail
    def fail(self, state: GraphState) -> dict:
        return {
            "status": "error",
            "refusal_reason": ERROR_REASON,
            "segments": [],
            "citations": [],
            "dropped_claims": 0,
        }

    # --------------------------------------------------------------- routing
    def route_after_generate(self, state: GraphState) -> str:
        return "fail" if state.get("error") else "verify"

    def route_after_gate(self, state: GraphState) -> str:
        return "generate" if state.get("relevant") else "refuse"

    def route_after_verify(self, state: GraphState) -> str:
        verdicts = state.get("verdicts") or []
        any_failed = any(not v.ok for v in verdicts)
        if any_failed and (state.get("attempt") or 0) < self.s.settings.max_retries:
            return "retry_prepare"
        return "finalize_answer"


# --- graph builder ---

def build_graph(services):
    nodes = Nodes(services)
    graph = StateGraph(GraphState)
    graph.add_node("rewrite", nodes.rewrite)
    graph.add_node("retrieve_and_gate", nodes.retrieve_and_gate)
    graph.add_node("generate", nodes.generate)
    graph.add_node("verify", nodes.verify)
    graph.add_node("retry_prepare", nodes.retry_prepare)
    graph.add_node("finalize_answer", nodes.finalize_answer)
    graph.add_node("refuse", nodes.refuse)
    graph.add_node("fail", nodes.fail)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve_and_gate")
    graph.add_conditional_edges(
        "retrieve_and_gate", nodes.route_after_gate, {"generate": "generate", "refuse": "refuse"}
    )
    graph.add_conditional_edges(
        "generate", nodes.route_after_generate, {"verify": "verify", "fail": "fail"}
    )
    graph.add_conditional_edges(
        "verify",
        nodes.route_after_verify,
        {"retry_prepare": "retry_prepare", "finalize_answer": "finalize_answer"},
    )
    graph.add_edge("retry_prepare", "retrieve_and_gate")
    graph.add_edge("finalize_answer", END)
    graph.add_edge("refuse", END)
    graph.add_edge("fail", END)
    return graph.compile()


def render_answer(segments: list[dict]) -> str:
    """'XXXX[1] YYYY[2]' — claims joined, each followed by its citation markers."""
    parts = []
    for seg in segments:
        markers = "".join(f"[{n}]" for n in seg.get("citations", []))
        parts.append(f"{seg['text']}{markers}")
    return " ".join(parts)


async def answer_question(services, graph, *, owner_key: str, conversation_id: str, question: str, include_public: bool = False) -> dict:
    request_id = uuid.uuid4().hex
    start = time.perf_counter()
    history = await asyncio.to_thread(services.conversations.recent_history, owner_key, conversation_id)
    await asyncio.to_thread(services.conversations.add_message, owner_key, conversation_id, "user", question)

    initial: GraphState = {
        "owner_key": owner_key,
        "conversation_id": conversation_id,
        "question": question,
        "include_public": include_public,
        "history": history,
        "attempt": 0,
        "top_k": services.settings.top_k,
        "node_ms": {},
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "judge_calls": 0,
    }
    error: str | None = None
    try:
        final = await graph.ainvoke(initial)
        error = final.get("error")
    except Exception as exc:  # unexpected failure anywhere in the graph: explicit error, never a 500
        logger.exception("answer graph crashed")
        error = f"{type(exc).__name__}: {exc}"
        final = {
            "status": "error",
            "refusal_reason": ERROR_REASON,
            "segments": [],
            "citations": [],
            "dropped_claims": 0,
        }
    total_ms = round((time.perf_counter() - start) * 1000)

    segments = final.get("segments", [])
    citations = final.get("citations", [])
    status = final.get("status", "refused")
    answer_text = render_answer(segments)

    message_id = await asyncio.to_thread(
        services.conversations.add_message,
        owner_key,
        conversation_id,
        "assistant",
        answer_text or final.get("refusal_reason") or "",
        segments=segments,
        citations=citations,
        status=status,
    )

    dropped_reasons: list[str] = []
    for verdict in final.get("verdicts", []) or []:
        if not verdict.ok:
            dropped_reasons.extend(verdict.reasons or [verdict.stage_decided])

    await asyncio.to_thread(
        services.request_logger.log,
        request_id=request_id,
        owner_key=owner_key,
        conversation_id=conversation_id,
        status=status,
        total_ms=total_ms,
        node_ms=final.get("node_ms", {}),
        tokens_in=final.get("tokens_in", 0),
        tokens_out=final.get("tokens_out", 0),
        cost_usd=final.get("cost_usd", 0.0),
        claims_kept=len(segments),
        claims_dropped=final.get("dropped_claims", 0),
        dropped_reasons=dropped_reasons,
        judge_calls=final.get("judge_calls", 0),
        question=question,
        answer=answer_text,
        rewritten_question=final.get("rewritten_question"),
        error=error,
    )

    return {
        "message_id": message_id,
        "status": status,
        "refusal_reason": final.get("refusal_reason"),
        "rewritten_question": final.get("rewritten_question"),
        "segments": segments,
        "citations": citations,
        "dropped_claims": final.get("dropped_claims", 0),
        "metrics": {
            "total_ms": total_ms,
            "tokens_in": final.get("tokens_in", 0),
            "tokens_out": final.get("tokens_out", 0),
            "cost_usd": round(final.get("cost_usd", 0.0), 6),
        },
    }
