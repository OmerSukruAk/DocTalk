"""Cross-encoder re-ranking of retrieved chunks.

The cheap bi-encoder hybrid search proposes candidate chunks, then a cross-encoder
scores each (query, chunk) pair for exact relevance and keeps the top_k. Loaded
lazily and only when RERANK_MODEL is set, mirroring the NLI scorer's opt-in
behaviour. Scores are sigmoid probabilities in [0, 1].
"""

from __future__ import annotations

from app.config import Settings


class ReRanker:
    """Optional cross-encoder that re-orders candidate chunks before generation.

    If ``configured`` is False the retrieval pipeline falls back to the fused
    (RRF) order, so this is a pure quality upgrade with no hard dependency.
    """

    def __init__(self, settings: Settings):
        self._model_name = settings.rerank_model
        self._candidates = settings.rerank_candidates
        self._model = None

    @property
    def configured(self) -> bool:
        return bool(self._model_name)

    def candidates(self, top_k: int) -> int:
        """How many chunks to propose before re-ranking (must be >= top_k)."""
        return max(top_k, self._candidates)

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[tuple[dict, float]]:
        """Re-score ``(query, chunk_text)`` pairs, return top_k as ``(chunk, score)``."""
        if not chunks:
            return []
        model = self._load()
        pairs = [(query, c["text"]) for c in chunks]
        scores = model.predict(pairs)
        scored = sorted(zip(chunks, scores), key=lambda t: float(t[1]), reverse=True)
        return [(c, float(s)) for c, s in scored[:top_k]]
