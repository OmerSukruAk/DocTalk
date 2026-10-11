"""The LangGraph workflow and its state.

rewrite -> retrieve_and_gate -(no)-> refuse
                            -(no, foreign-language question)-> translate -> retrieve_and_gate (once)
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
import re
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
    TRANSLATE_SYSTEM,
    TRANSLATE_USER,
    build_history_block,
    build_sources_block,
)
from app.agent.retrieval import Hit, RetrievalSignals, hybrid_search_ex, looks_foreign, passes_gate
from app.agent.verification import Claim, ClaimVerdict, verify_claims

logger = logging.getLogger("doctalk")

ERROR_REASON = "The answer service is temporarily unavailable. Please try again in a moment."


# --- graph state ---

class GraphState(TypedDict, total=False):
    owner_key: str
    conversation_id: str
    question: str
    style: str | None
    include_public: bool
    rewritten_question: str
    search_query: str | None  # set by translate: the question in the documents' language
    translated: bool
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


_NO_CONTENT = ("gap", "framing")  # segments that carry no answer of their own


# --- graph nodes ---

def _acc(state: GraphState, name: str, start: float, t_in: int = 0, t_out: int = 0, cost: float = 0.0) -> dict:
    """Add this node's wall time and token use to the running totals. Times ADD UP: after a
    retry or a translation, retrieve/generate/verify run twice and both runs count."""
    node_ms = dict(state.get("node_ms") or {})
    node_ms[name] = round(node_ms.get(name, 0.0) + (time.perf_counter() - start) * 1000, 1)
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
                    max_tokens=1024,  # headroom for hidden reasoning tokens
                )
                rewritten = res.text.strip() or question
                t_in, t_out, cost = res.tokens_in, res.tokens_out, res.cost_usd
            except Exception as exc:  # rewriting is an optimisation: fall back to the raw question
                logger.warning("question rewrite failed, using the original: %s", exc)
        return {"rewritten_question": rewritten, **_acc(state, "rewrite", start, t_in, t_out, cost)}

    # -------------------------------------------------- retrieve_and_gate
    async def retrieve_and_gate(self, state: GraphState) -> dict:
        start = time.perf_counter()
        query = state.get("search_query") or state.get("rewritten_question") or state["question"]
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
                raw_query=state["question"],
                query_vector=query_vector,
                top_k=state.get("top_k") or self.s.settings.top_k,
                neighbours=self.s.settings.context_neighbours,
            )
        )
        relevant = passes_gate(hits, signals, self.s.settings)
        return {"hits": hits, "signals": signals, "relevant": relevant, **_acc(state, "retrieve", start)}

    # -------------------------------------------------------------- translate
    async def translate(self, state: GraphState) -> dict:
        """Cross-language fallback: a Turkish question about an English document shares no
        keywords with it, and an English cross-encoder cannot score it. Rewrite the question
        once into the documents' language (judged from their section titles) and search again."""
        start = time.perf_counter()
        query = state.get("rewritten_question") or state["question"]
        chunks = await asyncio.to_thread(
            self.s.bm25.all_documents, state["owner_key"], state["conversation_id"], state.get("include_public", False)
        )
        titles: list[str] = []
        for c in chunks:
            for h in c.get("headings") or ([c["heading"]] if c.get("heading") else []):
                title = h.split(" > ")[-1]
                if title not in titles:
                    titles.append(title)
        sample = "\n".join(titles[:15]) or (chunks[0]["text"][:300] if chunks else "")
        translated = query
        t_in = t_out = 0
        cost = 0.0
        if sample:
            try:
                res = await self.s.llm.chat(
                    TRANSLATE_SYSTEM, TRANSLATE_USER.format(sample=sample, question=query), max_tokens=1024
                )
                translated = res.text.strip() or query
                t_in, t_out, cost = res.tokens_in, res.tokens_out, res.cost_usd
            except Exception as exc:
                logger.warning("query translation failed: %s", exc)
        return {"search_query": translated, "translated": True, **_acc(state, "translate", start, t_in, t_out, cost)}

    # --------------------------------------------------------------- generate
    async def generate(self, state: GraphState) -> dict:
        start = time.perf_counter()
        hits = state.get("hits") or []
        sources = [
            {
                "sentence_id": h.sentence_id,
                "document_name": h.document_name,
                "text": h.text,
                "heading": h.heading,
                "page": h.page,
                "idx": h.idx,
            }
            for h in hits
        ]
        if not self.s.llm.configured:
            return {"claims": [], **_acc(state, "generate", start)}
        # The model sees the user's own words (tone / format instructions live there) plus the
        # resolved question that retrieval used ("And meals?" -> "What is the meal limit?").
        question = state["question"]
        rewritten = state.get("rewritten_question") or question
        resolved = f"\n(Standalone version: {rewritten})" if rewritten.strip() != question.strip() else ""
        style = f"\nRequested answer style: {state['style']}" if state.get("style") else ""
        try:
            data, res = await self.s.llm.complete_json(
                GENERATE_SYSTEM,
                GENERATE_USER.format(
                    history=build_history_block(state.get("history") or []),
                    sources=build_sources_block(sources),
                    question=question,
                    resolved=resolved,
                    style=style,
                ),
                max_tokens=4000,  # reasoning models spend part of this before answering
                temperature=0.3,
            )
        except Exception as exc:
            logger.error("generation failed: %s", exc)
            return {"claims": [], "error": f"{type(exc).__name__}: {exc}", **_acc(state, "generate", start)}
        claims = parse_segments(data)
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
        usage: dict = {}
        verdicts = await verify_claims(
            state.get("claims") or [],
            context_by_id=context_by_id,
            nli=self.s.nli,
            llm=self.s.llm,
            nli_pass=self.s.settings.nli_pass,
            nli_fail=self.s.settings.nli_fail,
            usage=usage,
        )
        judge_calls = sum(1 for v in verdicts if v.used_judge)
        return {
            "verdicts": verdicts,
            "judge_calls": (state.get("judge_calls") or 0) + judge_calls,
            **_acc(
                state, "verify", start,
                usage.get("tokens_in", 0), usage.get("tokens_out", 0), usage.get("cost_usd", 0.0),
            ),
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
        gaps: list[str] = []
        for verdict in kept:
            if verdict.claim.kind in ("gap", "framing"):
                if verdict.claim.kind == "gap":
                    gaps.append(verdict.claim.text)
                segments.append(_segment(verdict.claim, []))
                continue
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
                segments.append(_segment(verdict.claim, nums))

        if not any(s["kind"] not in _NO_CONTENT for s in segments):
            if gaps:
                reason = " ".join(gaps)
            elif dropped:
                reason = "every candidate claim failed verification against the sources"
            else:
                reason = "the documents do not contain enough information to answer this question"
            return {
                "status": "refused",
                "refusal_reason": reason,
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
        if state.get("relevant"):
            return "generate"
        signals = state.get("signals")
        query = state.get("rewritten_question") or state["question"]
        if (
            not state.get("translated")
            and self.s.llm.configured
            and (signals is None or signals.lexical_top < 0.34)
            and looks_foreign(query)
        ):
            return "translate"
        return "refuse"

    def route_after_translate(self, state: GraphState) -> str:
        before = (state.get("rewritten_question") or state["question"]).strip().lower()
        return "retrieve_and_gate" if (state.get("search_query") or "").strip().lower() != before else "refuse"

    def route_after_verify(self, state: GraphState) -> str:
        """Retry (wider retrieval, fresh generation) only when NOTHING substantive survived
        verification. A partly verified answer is shown as 'partial' instead: regenerating
        it would double cost and latency and could replace good content with worse."""
        verdicts = state.get("verdicts") or []
        any_failed = any(not v.ok for v in verdicts)
        any_kept = any(v.ok and v.claim.kind not in _NO_CONTENT for v in verdicts)
        if any_failed and not any_kept and (state.get("attempt") or 0) < self.s.settings.max_retries:
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
    graph.add_node("translate", nodes.translate)
    graph.add_conditional_edges(
        "retrieve_and_gate",
        nodes.route_after_gate,
        {"generate": "generate", "translate": "translate", "refuse": "refuse"},
    )
    graph.add_conditional_edges(
        "translate",
        nodes.route_after_translate,
        {"retrieve_and_gate": "retrieve_and_gate", "refuse": "refuse"},
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


_KINDS = {"fact", "insight", "gap", "framing"}
_FORMATS = {"paragraph", "bullet", "text"}


# Sentence ids the model sometimes writes into the text itself ("... trigger [d_ab12:s28]."):
# their digits would fail the numbers check, and they must never reach the user.
_INLINE_ID = re.compile(r"\s*[\[(]?\b(d_[0-9a-z]+:s\d+)\b[\])]?", re.IGNORECASE)


def parse_segments(data: dict) -> list[Claim]:
    """Model JSON -> claims. Accepts the current {"segments": [...]} shape and the older
    {"claims": [...]} one (facts only)."""
    raw_items = data.get("segments")
    if raw_items is None:
        raw_items = data.get("claims")
    claims: list[Claim] = []
    for raw in raw_items or []:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        kind = str(raw.get("kind", "fact")).strip().lower()
        fmt = str(raw.get("format", "text")).strip().lower()
        citations = [str(c).strip().strip("[]") for c in (raw.get("citations") or []) if c]
        inline = _INLINE_ID.findall(text)
        if inline:  # move them out of the text and into the citation list
            text = _INLINE_ID.sub("", text).strip()
            citations += [sid for sid in inline if sid not in citations]
            if not text:
                continue
        claims.append(
            Claim(
                text=text.lstrip("-•* ").strip() if fmt == "bullet" else text,
                citations=citations,
                kind=kind if kind in _KINDS else "fact",
                format=fmt if fmt in _FORMATS else "text",
            )
        )
    return claims


def _segment(claim: Claim, citations: list[int]) -> dict:
    return {"text": claim.text, "citations": citations, "kind": claim.kind, "format": claim.format}


def render_answer(segments: list[dict]) -> str:
    """Markdown answer: paragraphs and bullets as the model laid them out, every sentence
    followed by its citation markers ('... 25 days.[1] That is above ...[1][2]')."""
    out = ""
    for seg in segments:
        markers = "".join(f"[{n}]" for n in seg.get("citations", []))
        piece = f"{seg['text']}{markers}"
        fmt = seg.get("format", "text")
        if not out:
            out = f"- {piece}" if fmt == "bullet" else piece
        elif fmt == "bullet":
            out += f"\n- {piece}"
        elif fmt == "paragraph" or out.split("\n")[-1].startswith("- "):
            out += f"\n\n{piece}"
        else:
            out += f" {piece}"
    return out


async def answer_question(
    services, graph, *, owner_key: str, conversation_id: str, question: str,
    include_public: bool = False, style: str | None = None,
) -> dict:
    request_id = uuid.uuid4().hex
    start = time.perf_counter()
    history = await asyncio.to_thread(services.conversations.recent_history, owner_key, conversation_id)
    await asyncio.to_thread(services.conversations.add_message, owner_key, conversation_id, "user", question)

    initial: GraphState = {
        "owner_key": owner_key,
        "conversation_id": conversation_id,
        "question": question,
        "style": (style or "").strip()[:200] or None,
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

    dropped_reasons: list[str] = []
    removed: list[dict] = []
    for verdict in final.get("verdicts", []) or []:
        if not verdict.ok:
            reasons = verdict.reasons or [verdict.stage_decided]
            dropped_reasons.extend(reasons)
            removed.append({"text": verdict.claim.text, "kind": verdict.claim.kind, "reasons": reasons,
                            "checked_by": verdict.stage_decided})

    details = {
        "style": initial["style"],
        "rewritten_question": final.get("rewritten_question"),
        "dropped_claims": final.get("dropped_claims", 0),
        "metrics": {
            "total_ms": total_ms,
            "tokens_in": final.get("tokens_in", 0),
            "tokens_out": final.get("tokens_out", 0),
            "cost_usd": round(final.get("cost_usd", 0.0), 6),
            "timings_ms": stage_timings(final.get("node_ms") or {}, total_ms),
        },
        # Stored with the message so the details survive a reload; removed statements are
        # unverified text, shown only as warnings and never sent back to the model.
        "diagnostics": {
            "search": search_stats(final),
            "verification": verification_counts(final.get("verdicts") or []),
            "removed": removed,
            "translated_query": final.get("search_query") if final.get("translated") else None,
            "attempts": (final.get("attempt") or 0) + 1,
            "judge_calls": final.get("judge_calls", 0),
        },
    }

    message_id = await asyncio.to_thread(
        services.conversations.add_message,
        owner_key,
        conversation_id,
        "assistant",
        answer_text or final.get("refusal_reason") or "",
        segments=segments,
        citations=citations,
        status=status,
        details=details,
    )

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
        "answer": answer_text,
        "segments": segments,
        "citations": citations,
        **details,
    }


def search_stats(final: dict) -> dict | None:
    """The retrieval funnel of the last attempt, for the answer's diagnostics."""
    s = final.get("signals")
    if s is None:
        return None
    hits = final.get("hits") or []
    mode = "overview" if s.overview else ("hybrid" if s.vector_top is not None else "keyword")
    return {
        "mode": mode, "top_k": final.get("top_k") or 0,
        "keyword_found": s.keyword_found, "vector_found": s.vector_found, "merged": s.merged,
        "candidates": s.candidates, "reranked": s.reranked, "kept_chunks": s.kept_chunks,
        "sentences": sum(1 for h in hits if not h.is_context),
        "context_sentences": sum(1 for h in hits if h.is_context),
        "gate_passed": final.get("relevant"),
        "vector_top": s.vector_top, "rerank_top": s.rerank_top, "lexical_top": s.lexical_top,
    }


_DECIDED_BY = {"nli_fail": "nli"}  # NLI keeps ("nli") and removes ("nli_fail"): one check


def verification_counts(verdicts: list) -> dict:
    """Per check, how many statements of each kind it kept and removed (last attempt)."""
    out: dict[str, dict] = {}
    for v in verdicts:
        check = out.setdefault(_DECIDED_BY.get(v.stage_decided, v.stage_decided), {"kept": {}, "removed": {}})
        bucket = check["kept" if v.ok else "removed"]
        bucket[v.claim.kind] = bucket.get(v.claim.kind, 0) + 1
    return out


# Stages in pipeline order. 'other' = loading history, saving messages and logs, graph overhead.
STAGES = ("rewrite", "translate", "retrieve", "generate", "verify")


def stage_timings(node_ms: dict, total_ms: float) -> dict:
    """Per-stage wall time for the response: retrieve (embedding the question, keyword +
    vector search, re-ranking), generate (the answer LLM call), verify (rule checks + the
    judge LLM call). Stages that did not run are 0."""
    out = {stage: round(float(node_ms.get(stage, 0.0)), 1) for stage in STAGES}
    out["other"] = round(max(0.0, total_ms - sum(out.values())), 1)
    return out
