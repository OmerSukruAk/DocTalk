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


def sources_for_repeated_numbers(
    claim_text: str, source_texts: list[str], answer_sources: dict[str, dict]
) -> dict[str, dict]:
    """A statement may repeat a number another statement of the same answer cited ('Mercury
    averages far below Venus's 465 °C'). Returns those other sources when they hold every number
    missing from the statement's own sources; otherwise {} and the number rule decides."""
    own: set[str] = set()
    for text in source_texts:
        own |= extract_numbers(text)
    missing = extract_numbers(claim_text) - own
    if not missing:
        return {}
    found, covered = {}, set()
    for sid, source in answer_sources.items():
        hit = missing & extract_numbers(source["text"])
        if hit:
            found[sid], covered = source, covered | hit
    return found if covered == missing else {}


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

    def warm_up(self) -> None:
        """Load the model now (called at startup) so the first question does not pay for it."""
        self._ensure_loaded()

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

_VERDICT_RE = re.compile(r'"id"\s*:\s*"?(\d+)"?\s*,\s*"label"\s*:\s*"(supported|unsupported)"', re.IGNORECASE)

def _judge_block(n: int, claim: "Claim", cited_sources: list[dict]) -> str:
    return (
        f"### Segment {n} (kind: {claim.kind})\n{claim.text}\n"
        f"Cited sources:\n{build_sources_block(cited_sources)}"
    )


async def judge_claims(
    llm: ChatClient,
    items: list[tuple["Claim", list[dict]]],
    usage_out: dict | None = None,
) -> list[tuple[bool, str]]:
    """Judge every pending segment in ONE call. Returns (supported, reason) per item, in order.
    Anything without an explicit 'supported' verdict fails (fail closed). The judge call's
    tokens and cost are added to ``usage_out`` so the request's cost metrics include them."""
    if not items:
        return []
    if not llm.configured:
        return [(False, "judge unavailable (LLM not configured)")] * len(items)
    body = "\n\n".join(_judge_block(i + 1, claim, sources) for i, (claim, sources) in enumerate(items))
    try:
        # Generous budget: reasoning models (e.g. deepseek-flash) spend part of max_tokens on
        # hidden reasoning, and a truncated reply would fail every claim.
        data, usage = await llm.complete_json(
            JUDGE_SYSTEM, JUDGE_USER.format(segments=body), max_tokens=1500 + 120 * len(items)
        )
    except Exception as exc:  # provider outage must not crash the request: fail closed
        logger.warning("LLM judge failed: %s", exc)
        return [(False, f"judge unavailable ({type(exc).__name__})")] * len(items)
    if usage_out is not None:
        for key in ("tokens_in", "tokens_out", "cost_usd"):
            usage_out[key] = usage_out.get(key, 0) + (getattr(usage, key, 0) or 0)

    by_id: dict[int, tuple[str, str]] = {}
    if not data.get("verdicts") and getattr(usage, "text", ""):
        # Truncated JSON: keep every verdict that was written completely.
        for m in _VERDICT_RE.finditer(usage.text):
            by_id[int(m.group(1))] = (m.group(2).lower(), "")
    for raw in data.get("verdicts") or []:
        try:
            by_id[int(raw.get("id"))] = (str(raw.get("label", "")).strip().lower(), str(raw.get("reason", ""))[:300])
        except (TypeError, ValueError, AttributeError):
            continue
    if not by_id and "label" in data and len(items) == 1:  # single-verdict shape
        by_id[1] = (str(data.get("label", "")).strip().lower(), str(data.get("reason", ""))[:300])

    out: list[tuple[bool, str]] = []
    for i in range(len(items)):
        label, reason = by_id.get(i + 1, ("", "judge returned no verdict"))
        out.append((label == "supported", reason or label or "unsupported"))
    return out


async def judge_claim(llm: ChatClient, claim_text: str, cited_sources: list[dict]) -> tuple[bool, str]:
    """Single-claim convenience wrapper around ``judge_claims``."""
    return (await judge_claims(llm, [(Claim(text=claim_text), cited_sources)]))[0]


# --- pipeline ---

@dataclass
class Claim:
    text: str
    citations: list[str] = field(default_factory=list)
    kind: str = "fact"  # fact | insight | gap
    format: str = "text"  # paragraph | bullet | text


@dataclass
class ClaimVerdict:
    claim: Claim
    ok: bool
    stage_decided: str  # rules | nli | nli_fail | judge | judge_unavailable | gap
    reasons: list[str] = field(default_factory=list)
    nli_score: float | None = None

    @property
    def used_judge(self) -> bool:
        return self.stage_decided.startswith("judge")


_MAX_GAP_WORDS = 40
_MAX_FRAMING_WORDS = 15
_WORD = re.compile(r"[^\W\d_]{3,}")


def check_framing(text: str, context_texts: list[str]) -> list[str]:
    """Framing (greeting, sign-off, heading, lead-in) is kept only if it carries no
    information: short, no numbers, no negation, and it does not restate the documents
    (two or more of their content words), unless it is a lead-in ending in ':'."""
    from app.agent.retrieval import STOPWORDS

    reasons = []
    if len(text.split()) > _MAX_FRAMING_WORDS:
        reasons.append("framing too long")
    if extract_numbers(text):
        reasons.append("framing contains a number")
    if _has_negation(text.lower()):
        reasons.append("framing contains a negation")
    if not text.rstrip().endswith(":"):
        words = {w.lower() for w in _WORD.findall(text)} - STOPWORDS
        source_words = {w.lower() for t in context_texts for w in _WORD.findall(t)}
        shared = sorted(words & source_words)
        if len(shared) >= 2:
            reasons.append(f"framing restates document content ({', '.join(shared)})")
    return reasons


async def verify_claims(
    claims: list[Claim],
    *,
    context_by_id: dict[str, dict],
    nli: NLIScorer,
    llm: ChatClient,
    nli_pass: float,
    nli_fail: float,
    usage: dict | None = None,
) -> list[ClaimVerdict]:
    """Layered verification, cheapest first:
    hard rules (never overridable) -> local NLI (facts only) -> ONE batched LLM judge call
    for everything still undecided.

    * fact    - must be stated by its cited sources (paraphrase allowed); numbers and
                negations are checked deterministically. A number repeated from a source
                another statement of the answer cites is not invented: that source is added
                to the statement's citations and the judge decides.
    * insight - the model's own commentary; must be grounded in (and cite) the sources, but
                may contain derived numbers (a calculation), so only the judge decides.
    * gap     - 'the documents do not cover X': carries no facts and no citations; kept
                when short.
    * framing - greeting, sign-off, heading or lead-in a requested format needs; kept only
                if it carries no information (see ``check_framing``)."""
    verdicts: list[ClaimVerdict | None] = []
    pending: list[tuple[int, Claim, list[dict], float | None]] = []
    allowed_ids = set(context_by_id)

    context_texts = [c["text"] for c in context_by_id.values()]
    answer_sources = {
        cid: context_by_id[cid]
        for c in claims if c.kind in ("fact", "insight")
        for cid in c.citations if cid in context_by_id
    }
    for claim in claims:
        if claim.kind == "framing":
            claim.citations = []
            reasons = check_framing(claim.text, context_texts)
            verdicts.append(ClaimVerdict(claim=claim, ok=not reasons, stage_decided="framing", reasons=reasons))
            continue
        if claim.kind == "gap":
            short = len(claim.text.split()) <= _MAX_GAP_WORDS
            claim.citations = []
            verdicts.append(
                ClaimVerdict(claim=claim, ok=short, stage_decided="gap", reasons=[] if short else ["gap note too long"])
            )
            continue

        cited_sources = [context_by_id[c] for c in claim.citations if c in context_by_id]
        source_texts = [s["text"] for s in cited_sources]
        borrowed = {}
        if claim.kind == "fact" and cited_sources:
            borrowed = sources_for_repeated_numbers(claim.text, source_texts, answer_sources)
            claim.citations = [*claim.citations, *borrowed]
            cited_sources += borrowed.values()
            source_texts += [s["text"] for s in borrowed.values()]

        if claim.kind == "insight":
            rule_results = [check_citations_exist(claim.citations, allowed_ids)]
        else:
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
        if claim.kind == "fact" and nli.configured and cited_sources and not borrowed:  # a comparison: judge
            # torch inference is CPU-bound: keep it off the event loop
            score = await asyncio.to_thread(nli.entailment_score, " ".join(source_texts), claim.text)

        if score is not None:
            scrutiny = needs_scrutiny(claim.text, source_texts)
            if score >= nli_pass and not scrutiny:
                verdicts.append(ClaimVerdict(claim=claim, ok=True, stage_decided="nli", nli_score=score))
                continue
            if score <= nli_fail:
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

        # borderline, an insight, or NLI unavailable -> batched LLM judge
        pending.append((len(verdicts), claim, cited_sources, score))
        verdicts.append(None)

    results = await judge_claims(llm, [(claim, sources) for _, claim, sources, _ in pending], usage)
    for (slot, claim, _sources, score), (supported, reason) in zip(pending, results):
        unavailable = reason.startswith("judge unavailable")
        verdicts[slot] = ClaimVerdict(
            claim=claim,
            ok=supported,
            stage_decided="judge_unavailable" if unavailable else "judge",
            reasons=[] if supported else [reason],
            nli_score=score,
        )
    return [v for v in verdicts if v is not None]
