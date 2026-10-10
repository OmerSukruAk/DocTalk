"""Retrieval: keyword search, embeddings, rank fusion, re-ranking and the gate.

Parent-child retrieval: BM25 + vector search propose candidate *chunks* (parent
passages), fused with reciprocal rank fusion and optionally re-ranked by a
cross-encoder, then expanded back into sentence-level hits for citation. The gate is
deterministic and runs before any LLM call, so an unanswerable question costs zero
tokens."""

from __future__ import annotations
import re
import threading
from dataclasses import dataclass, field
from rank_bm25 import BM25Plus
from openai import AsyncAzureOpenAI, AsyncOpenAI
from app.config import Settings


# --- BM25 ---

_TOKEN = re.compile(r"\w+", re.UNICODE)

# Function words (EN + CS). Without this filter a question like "What is the capital of
# France?" matches every chunk through "what/is/the/of" and the gate can never say no.
STOPWORDS = frozenset("""
a an the and or but if then else of to in on at by for with from into over under about as is are was
were be been being am do does did done have has had having this that these those it its they them
their he she his her we our you your i me my what which who whom whose when where why how can could
should would will shall may might must not no yes than so such there here also any all each every
some more most other only own same very just per via s t
a i k o s u v z ve se si je jsou byl byla bylo byli být bude budou ale nebo že aby jak co kdo kde kdy
proč který která které kteří tento tato toto tyto ten ta to ti na do od po při pro před přes pod nad
za mezi bez podle jako také už jen pouze velmi než tedy tak jsem jsi jsme jste mám má máme máte mají
""".split())


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_terms(text: str) -> list[str]:
    """Query words that actually carry meaning (no stopwords, no 1-char tokens)."""
    return [t for t in _tokenize(text) if t not in STOPWORDS and len(t) > 1]


def _stem(token: str) -> str:
    """Crude, language-agnostic stem (prefix) so 'leave'/'leaves' and Czech inflections match."""
    return token[:5] if len(token) > 5 else token


def lexical_coverage(query_terms: list[str], chunk_text: str) -> float:
    """Share of the query's content terms present in the chunk (0..1)."""
    if not query_terms:
        return 0.0
    stems = {_stem(t) for t in _tokenize(chunk_text)}
    return sum(1 for q in query_terms if _stem(q) in stems) / len(query_terms)


class BM25Index:
    """Per-owner in-memory BM25 over parent chunks (BM25+ keeps scores non-negative even
    for one-document corpora, where classic BM25 produces negative IDF).

    Multi-replica safe: every lookup compares a cheap fingerprint of the owner's chunks in
    Mongo with the one the cached index was built from, so a document uploaded or deleted
    through another replica is picked up on the next question instead of being served from
    a stale index. ``invalidate`` is still called locally for an immediate refresh."""

    def __init__(self, chunks):
        self._chunks = chunks
        self._cache: dict[str, tuple[BM25Plus | None, list[dict], object]] = {}
        self._lock = threading.Lock()

    def invalidate(self, owner_key: str) -> None:
        with self._lock:
            self._cache.pop(owner_key, None)

    def _fingerprint(self, owner_key: str):
        fp = getattr(self._chunks, "fingerprint", None)
        return fp(owner_key) if fp else None

    def _get(self, owner_key: str) -> tuple[BM25Plus | None, list[dict]]:
        fp = self._fingerprint(owner_key)
        with self._lock:
            cached = self._cache.get(owner_key)
            if cached is not None and cached[2] == fp:
                return cached[0], cached[1]
        docs = self._chunks.all_for_owner(owner_key)  # heavy work outside the lock
        index = None
        if docs:
            # The heading is part of the searchable text: "leave policy" must find the
            # chunk under "## Leave Policy" even if its sentences never repeat the words.
            corpus = [_tokenize(f"{' '.join(d.get('headings') or [d.get('heading') or ''])} {d['text']}") for d in docs]
            index = BM25Plus(corpus) if any(corpus) else None
        with self._lock:
            self._cache[owner_key] = (index, docs, fp)
        return index, docs

    def search(self, owner_key: str, query: str, top_k: int) -> list[tuple[str, float]]:
        terms = content_terms(query)
        if not terms:  # nothing but stopwords -> no lexical evidence at all
            return []
        index, docs = self._get(owner_key)
        if not index or not docs:
            return []
        scores = index.get_scores(terms)
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        return [(d["_id"] if "_id" in d else d["id"], float(s)) for d, s in ranked[:top_k] if s > 0]

    def all_documents(self, owner_key: str) -> list[dict]:
        return self._get(owner_key)[1]


# --- embeddings ---

class EmbeddingsNotConfigured(RuntimeError):
    """Raised when embeddings are requested without an API key (offline/tests)."""


class EmbeddingsClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._batch = 64
        self._client = None
        if settings.embeddings_configured:
            if settings.embed_provider == "azure":
                self._client = AsyncAzureOpenAI(
                    api_key=settings.embed_api_key,
                    azure_endpoint=settings.embed_base_url,
                    api_version="2024-02-01",
                    max_retries=2,
                    timeout=60.0,
                )
            else:
                kwargs: dict = {"api_key": settings.embed_api_key, "max_retries": 2, "timeout": 60.0}
                if settings.embed_base_url:
                    kwargs["base_url"] = settings.embed_base_url
                self._client = AsyncOpenAI(**kwargs)

    @property
    def configured(self) -> bool:
        return self._client is not None

    @property
    def dimensions(self) -> int:
        return self._settings.embed_dimensions

    @property
    def model(self) -> str:
        return self._settings.embed_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            raise EmbeddingsNotConfigured("EMBED_API_KEY is not set")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            chunk = [t if t.strip() else " " for t in texts[start : start + self._batch]]
            resp = await self._client.embeddings.create(model=self._settings.embed_model, input=chunk)
            ordered = sorted(resp.data, key=lambda d: d.index)
            vectors.extend([list(d.embedding) for d in ordered])
        for vec in vectors:
            if len(vec) != self._settings.embed_dimensions:
                raise ValueError(
                    f"Embedding dim {len(vec)} != EMBED_DIMENSIONS {self._settings.embed_dimensions}"
                )
        return vectors

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


# --- hybrid search ---

RRF_K = 60


@dataclass
class Hit:
    sentence_id: str
    document_id: str
    document_name: str
    idx: int
    text: str
    page: int | None
    heading: str | None
    bbox: list[float] | None
    score: float
    is_table_row: bool = False
    is_context: bool = False
    char_start: int | None = None
    char_end: int | None = None
    meta: dict = field(default_factory=dict)


def _rrf(rankings: list[list[str]]) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return fused


@dataclass
class RetrievalSignals:
    """Raw relevance evidence, independent of RRF (whose scores say nothing about absolute
    relevance: the top-ranked chunk of an off-topic question still gets a 'good' RRF score)."""

    vector_top: float | None = None  # best cosine similarity of the query to any chunk
    rerank_top: float | None = None  # best cross-encoder probability, when a re-ranker is on
    lexical_top: float = 0.0  # best coverage of the query's content words by a candidate chunk
    fused_top: float = 0.0


def hybrid_search(**kwargs) -> list[Hit]:
    """Backwards-compatible wrapper: hits only."""
    return hybrid_search_ex(**kwargs)[0]


def hybrid_search_ex(
    *,
    sentences,
    chunks,
    documents,
    vectors,
    bm25,
    reranker,
    owner_key: str,
    query: str,
    query_vector: list[float] | None,
    top_k: int,
    neighbours: int,
) -> tuple[list[Hit], RetrievalSignals]:
    """Parent-child retrieval: propose chunks, re-rank, then expand to sentence hits.
    Returns ``(hits, signals)``; the signals feed the relevance gate.

    BM25 + vector search propose candidate *chunks* (a chunk carries multi-sentence
    context, so its embedding is meaningful), they are fused with RRF and optionally
    re-ranked by a cross-encoder, and the surviving chunks expand back into their
    child sentences for citation. Degrades gracefully: BM25-only when embeddings are
    unavailable, fused-order-only when the re-ranker is off.
    """
    candidate_k = reranker.candidates(top_k) if reranker.configured else top_k

    bm25_ids: list[str] = []
    try:
        bm25_ids = [cid for cid, _ in bm25.search(owner_key, query, candidate_k * 2)]
    except Exception:
        bm25_ids = []

    vector_ids: list[str] = []
    signals = RetrievalSignals()
    if query_vector:
        try:
            found = vectors.search(owner_key, query_vector, candidate_k * 2)
            vector_ids = [cid for cid, _ in found]
            if found:
                signals.vector_top = max(score for _, score in found)
        except Exception:
            vector_ids = []

    fused = _rrf([r for r in (bm25_ids, vector_ids) if r])
    if not fused:
        return [], signals
    signals.fused_top = max(fused.values())
    ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:candidate_k]

    chunk_docs = {c["id"]: c for c in chunks.get_many(owner_key, ranked_ids)}
    ordered = [chunk_docs[cid] for cid in ranked_ids if cid in chunk_docs]

    # Re-rank (cross-encoder scores (query, chunk)) or keep the fused order.
    if reranker.configured and ordered:
        ranked = reranker.rerank(query, ordered, top_k)
        chunk_scores: dict[str, float] = {c["id"]: s for c, s in ranked}
        ordered = [c for c, _ in ranked]
        signals.rerank_top = max(chunk_scores.values()) if chunk_scores else None
    else:
        chunk_scores = {c["id"]: fused[c["id"]] for c in ordered}
        ordered = ordered[:top_k]

    q_terms = content_terms(query)
    signals.lexical_top = max(
        (
            lexical_coverage(q_terms, f"{' '.join(c.get('headings') or [c.get('heading') or ''])} {c['text']}")
            for c in ordered
        ),
        default=0.0,
    )

    # Expand surviving chunks into their child sentences (primary hits).
    sentence_ids: list[str] = []
    for c in ordered:
        sentence_ids.extend(c.get("sentence_ids") or [])
    sdocs = {s["id"]: s for s in sentences.get_many(owner_key, sentence_ids)}
    name_cache = _document_names(documents, owner_key, sdocs)

    hits: list[Hit] = []
    seen: set[str] = set()
    for c in ordered:
        score = chunk_scores.get(c["id"], fused.get(c["id"], 0.0))
        for sid in c.get("sentence_ids") or []:
            sdoc = sdocs.get(sid)
            if not sdoc or sid in seen:
                continue
            seen.add(sid)
            hits.append(_to_hit(sdoc, name_cache, score, is_context=False))

    # Neighbour expansion around cited sentences (context only; lower score).
    for sid in list(sentence_ids):
        sdoc = sdocs.get(sid)
        if not sdoc:
            continue
        base_score = chunk_scores.get(sdoc.get("chunk_id", ""), 0.0)
        for nb in sentences.neighbors(owner_key, sdoc["document_id"], sdoc["idx"], neighbours):
            if nb["id"] in seen:
                continue
            seen.add(nb["id"])
            name_cache.setdefault(nb["document_id"], name_cache.get(nb["document_id"], ""))
            hits.append(_to_hit(nb, name_cache, base_score * 0.5, is_context=True))

    return hits, signals


def _document_names(documents, owner_key: str, docs: dict[str, dict]) -> dict[str, str]:
    names: dict[str, str] = {}
    for doc in docs.values():
        did = doc["document_id"]
        if did in names:
            continue
        info = documents.get(owner_key, did)
        names[did] = info["name"] if info else ""
    return names


def _to_hit(doc: dict, names: dict[str, str], score: float, *, is_context: bool) -> Hit:
    return Hit(
        sentence_id=doc["id"],
        document_id=doc["document_id"],
        document_name=names.get(doc["document_id"], ""),
        idx=doc["idx"],
        text=doc["text"],
        page=doc.get("page"),
        heading=doc.get("heading"),
        bbox=doc.get("bbox"),
        score=score,
        is_table_row=doc.get("is_table_row", False),
        is_context=is_context,
        char_start=doc.get("char_start"),
        char_end=doc.get("char_end"),
    )


# --- relevance gate ---

def is_relevant(hits: list[Hit], min_score: float) -> bool:
    """Cheap, deterministic gate on the top fused/re-ranked hit. Runs before any LLM
    call so an unanswerable question costs zero tokens."""
    primary = [h for h in hits if not h.is_context]
    if not primary:
        return False
    return max(h.score for h in primary) >= min_score


def passes_gate(hits: list[Hit], signals: RetrievalSignals, settings) -> bool:
    """Deterministic relevance gate, evaluated before any LLM call so an unanswerable
    question costs zero tokens.

    * cross-encoder available  -> its probability is authoritative (most precise signal);
    * else vector search ran   -> best cosine similarity, OR strong lexical coverage (an exact
      keyword/ID question can legitimately have a weak embedding match);
    * else (BM25 only)         -> lexical coverage of the query's content words.
    Thresholds live in Settings and should be calibrated with eval/run_eval.py."""
    if not any(not h.is_context for h in hits):
        return False
    if signals.fused_top < settings.gate_min_score:
        return False
    if signals.rerank_top is not None:
        return signals.rerank_top >= settings.gate_min_rerank_score
    if signals.vector_top is not None and signals.vector_top >= settings.gate_min_vector_score:
        return True
    return signals.lexical_top >= settings.gate_min_lexical_coverage
