"""Claim verification, layered cheapest-first.

hard rules (never overridable) -> local multilingual NLI -> LLM judge for borderline
claims only. Anything that fails is dropped before the user ever sees it."""

from __future__ import annotations
import asyncio
import logging
import re
import threading
from dataclasses import dataclass, field
from app.config import Settings
from app.agent.llm import ChatClient, JUDGE_SYSTEM, JUDGE_USER, build_sources_block

logger = logging.getLogger("doctalk")


# --- hard rules ---

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
# Negation markers (EN + CS). Any of them counts: 'cannot' in a claim is supported by 'may not'
# in the source. Only the *presence* of a negation is compared, never the exact word.
_NEGATIONS = {
    "not", "no", "never", "none", "cannot", "can't", "won't", "don't", "doesn't", "isn't",
    "aren't", "without", "neither", "nor", "prohibited", "forbidden",
    "ne", "není", "nejsou", "nikdy", "žádný", "žádná", "žádné", "bez", "nelze", "zakázáno",
}
# Absolutes the sources may have only implied. Paraphrase makes these common, so they never
# fail a claim on their own; they just forbid the cheap NLI auto-pass (see needs_scrutiny).
_ABSOLUTES = {"always", "never", "all", "every", "only", "vždy", "nikdy", "všechny", "pouze"}


@dataclass
class RuleResult:
    ok: bool
    reason: str | None = None


def normalize_number(token: str) -> str:
    """'3,5' -> '3.5'; '1.000' kept as-is (thousands ambiguity is a known limitation)."""
    if "," in token and "." not in token:
        return token.replace(",", ".")
    return token


def extract_numbers(text: str) -> set[str]:
    return {normalize_number(m) for m in _NUMBER.findall(text)}


def check_citations_exist(cited: list[str], allowed_ids: set[str]) -> RuleResult:
    missing = [c for c in cited if c not in allowed_ids]
    if missing:
        return RuleResult(False, f"citation id(s) not in this request's context: {', '.join(missing)}")
    if not cited:
        return RuleResult(False, "claim has no citations")
    return RuleResult(True)


def check_numbers(claim_text: str, source_texts: list[str]) -> RuleResult:
    claim_numbers = extract_numbers(claim_text)
    if not claim_numbers:
        return RuleResult(True)
    source_numbers: set[str] = set()
    for text in source_texts:
        source_numbers |= extract_numbers(text)
    invented = sorted(n for n in claim_numbers if n not in source_numbers)
    if invented:
        return RuleResult(False, f"number(s) not present in cited sources: {', '.join(invented)}")
    return RuleResult(True)


def _has_negation(text_low: str) -> bool:
    return any(_contains_word(text_low, w) for w in _NEGATIONS)


def check_negation(claim_text: str, source_texts: list[str]) -> RuleResult:
    """Hard fail only for a *fabricated* negation: the claim denies something while none of the
    cited sources contains any negation at all. Wording differences ('must' vs 'shall',
    'cannot' vs 'may not') are NOT failures - they are left to NLI / the judge."""
    if _has_negation(claim_text.lower()) and not _has_negation(" ".join(source_texts).lower()):
        return RuleResult(False, "claim asserts a negation that none of the cited sources contains")
    return RuleResult(True)


def needs_scrutiny(claim_text: str, source_texts: list[str]) -> list[str]:
    """Soft signals that a claim may have changed the meaning (dropped negation, added
    absolute). They do not fail the claim; they route it to the LLM judge even when NLI is
    confident, because NLI models are weakest exactly on negation and quantifiers."""
    claim_low, source_low = claim_text.lower(), " ".join(source_texts).lower()
    reasons: list[str] = []
    if _has_negation(source_low) and not _has_negation(claim_low):
        reasons.append("cited sources contain a negation the claim omits")
    for word in sorted(_ABSOLUTES):
        if _contains_word(claim_low, word) and not _contains_word(source_low, word):
            reasons.append(f"claim adds absolute '{word}'")
    return reasons


def _contains_word(haystack: str, word: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", haystack) is not None


def run_rules(cited: list[str], claim_text: str, source_texts: list[str], allowed_ids: set[str]) -> list[RuleResult]:
    return [
        check_citations_exist(cited, allowed_ids),
        check_numbers(claim_text, source_texts),
        check_negation(claim_text, source_texts),
    ]


# --- NLI ---

class NLIScorer:
    """Multilingual entailment scorer (e.g. MoritzLaurer/mDeBERTa-v3-base-mnli-xnli).

    Loaded lazily and only when NLI_MODEL is set. If the model is unavailable the
    scorer reports configured=False and the pipeline falls back to the LLM judge.
    """

    def __init__(self, settings: Settings):
        self._model_name = settings.nli_model
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._load_failed = False
        self._entail_idx = 2  # most MNLI checkpoints order [contradiction, neutral, entailment]

    @property
    def configured(self) -> bool:
        return bool(self._model_name) and not self._load_failed

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self._model_name or self._load_failed:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
                self._model.eval()
                labels = getattr(self._model.config, "id2label", {}) or {}
                for idx, label in labels.items():
                    if "entail" in str(label).lower():
                        self._entail_idx = int(idx)
                self._torch = torch
            except Exception:
                self._load_failed = True
                return False
        return True

    def entailment_score(self, premise: str, hypothesis: str) -> float | None:
        if not self._ensure_loaded():
            return None
        import torch

        inputs = self._tokenizer(
            premise, hypothesis, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        return float(probs[self._entail_idx])


# --- LLM judge ---

async def judge_claim(
    llm: ChatClient,
    claim_text: str,
    cited_sources: list[dict],
) -> tuple[bool, str]:
    """Return (supported, reason). Any non-'supported' label is treated as fail."""
    if not llm.configured:
        # No judge available -> be conservative and fail the claim.
        return False, "judge unavailable (LLM not configured)"
    sources = build_sources_block(cited_sources)
    try:
        data, _usage = await llm.complete_json(
            JUDGE_SYSTEM, JUDGE_USER.format(sources=sources, claim=claim_text), max_tokens=256
        )
    except Exception as exc:  # provider outage must not crash the request: fail closed
        logger.warning("LLM judge failed: %s", exc)
        return False, f"judge unavailable ({type(exc).__name__})"
    label = str(data.get("label", "")).strip().lower()
    reason = str(data.get("reason", ""))[:300]
    return label == "supported", (reason or label or "unsupported")


# --- pipeline ---

@dataclass
class Claim:
    text: str
    citations: list[str] = field(default_factory=list)


@dataclass
class ClaimVerdict:
    claim: Claim
    ok: bool
    stage_decided: str  # rules | nli | nli_fail | judge | judge_unavailable
    reasons: list[str] = field(default_factory=list)
    nli_score: float | None = None

    @property
    def used_judge(self) -> bool:
        return self.stage_decided.startswith("judge")


async def verify_claims(
    claims: list[Claim],
    *,
    context_by_id: dict[str, dict],
    nli: NLIScorer,
    llm: ChatClient,
    nli_pass: float,
    nli_fail: float,
) -> list[ClaimVerdict]:
    """Layered verification, cheapest first:
    hard rules (never overridable) -> local NLI -> LLM judge for borderline only."""
    verdicts: list[ClaimVerdict] = []
    allowed_ids = set(context_by_id)

    for claim in claims:
        cited_sources = [context_by_id[c] for c in claim.citations if c in context_by_id]
        source_texts = [s["text"] for s in cited_sources]

        rule_results = run_rules(claim.citations, claim.text, source_texts, allowed_ids)
        failed = [r for r in rule_results if not r.ok]
        if failed:
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    ok=False,
                    stage_decided="rules",
                    reasons=[r.reason or "rule failed" for r in failed],
                )
            )
            continue

        score = None
        if nli.configured and cited_sources:
            # torch inference is CPU-bound: keep it off the event loop
            score = await asyncio.to_thread(nli.entailment_score, " ".join(source_texts), claim.text)

        scrutiny = needs_scrutiny(claim.text, source_texts)
        if score is not None and score >= nli_pass and not scrutiny:
            verdicts.append(
                ClaimVerdict(claim=claim, ok=True, stage_decided="nli", nli_score=score)
            )
            continue
        if score is not None and score <= nli_fail:
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    ok=False,
                    stage_decided="nli_fail",
                    reasons=[f"NLI entailment {score:.2f} <= {nli_fail}"],
                    nli_score=score,
                )
            )
            continue

        # borderline (or NLI unavailable) -> LLM judge
        supported, reason = await judge_claim(llm, claim.text, cited_sources)
        verdicts.append(
            ClaimVerdict(
                claim=claim,
                ok=supported,
                stage_decided="judge" if llm.configured and not reason.startswith("judge unavailable") else "judge_unavailable",
                reasons=[] if supported else [reason],
                nli_score=score,
            )
        )

    return verdicts
