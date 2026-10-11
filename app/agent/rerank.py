"""Cross-encoder re-ranking of retrieved chunks.

The cheap bi-encoder hybrid search proposes candidate chunks, then a cross-encoder
scores each (query, chunk) pair for exact relevance and keeps the top_k. Loaded
lazily and only when RERANK_MODEL is set, mirroring the NLI scorer's opt-in
behaviour. Scores are sigmoid probabilities in [0, 1].
"""

from __future__ import annotations

import math
import threading

from app.config import Settings


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class ReRanker:
    """Optional cross-encoder that re-orders candidate chunks before generation.

    If ``configured`` is False the retrieval pipeline falls back to the fused
    (RRF) order, so this is a pure quality upgrade with no hard dependency.
    """

    def __init__(self, settings: Settings):
        self._model_name = settings.rerank_model
        self._candidates = settings.rerank_candidates
        self._model = None
        self._raw_logits = False
        self._lock = threading.Lock()
        self._load_failed = False

    @property
    def configured(self) -> bool:
        return bool(self._model_name) and not self._load_failed

    def candidates(self, top_k: int) -> int:
        """How many chunks to propose before re-ranking (must be >= top_k)."""
        return max(top_k, self._candidates)

    def warm_up(self) -> None:
        """Load the model now (called at startup) so the first question does not pay for it."""
        if self.configured:
            self._load()

    def _load(self):
        with self._lock:  # a question arriving during the startup warm-up waits for it
            if self._model is None:
                try:
                    from sentence_transformers import CrossEncoder

                    model = CrossEncoder(self._model_name)
                except Exception:
                    # e.g. no network or an unwritable model cache: search keeps the fused order
                    self._load_failed = True
                    raise
                # Many cross-encoders (e.g. ms-marco-MiniLM) ship with an Identity activation and
                # return raw logits (-11 .. +10). The gate threshold is a probability, so convert.
                act = getattr(model, "activation_fn", None) or getattr(
                    model, "default_activation_function", None
                )
                self._raw_logits = act is None or type(act).__name__ == "Identity"
                self._model = model
        return self._model

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[tuple[dict, float]]:
        """Re-score ``(query, section + chunk_text)`` pairs, return top_k as ``(chunk, prob)``."""
        if not chunks:
            return []
        model = self._load()
        pairs = [(query, _passage(c)) for c in chunks]
        scores = [float(s) for s in model.predict(pairs)]
        if self._raw_logits or any(s < 0 or s > 1 for s in scores):
            scores = [_sigmoid(s) for s in scores]
        scored = sorted(zip(chunks, scores), key=lambda t: t[1], reverse=True)
        return scored[:top_k]


def _passage(chunk: dict) -> str:
    """The section title is part of the passage: 'dividend policy' must match a table whose
    rows only say 'INTERIM DPS: €1.28'."""
    headings = chunk.get("headings") or ([chunk["heading"]] if chunk.get("heading") else [])
    title = " | ".join(h.split(" > ")[-1] for h in headings)
    return f"{title}. {chunk['text']}" if title else chunk["text"]
