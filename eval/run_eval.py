#!/usr/bin/env python3
"""Evaluation harness: groundedness, answer quality, refusal behaviour, latency, cost.

  python eval/run_eval.py --api http://localhost:8000 --issuer http://localhost:9000 \
         --docs eval/docs/hr_policy.md eval/docs/it_security.md --out eval/report.md

Metrics (all computed from the API response, no LLM-as-judge needed for the headline numbers):
  * citation validity   - every citation's text equals the stored passage fetched via
                          /v1/passages/{id} (the citation is real, not model output)
  * groundedness        - every answered segment (a) has >=1 citation and (b) every number in it
                          appears in its cited text (an independent re-check of the server's
                          own verifier, so the verifier is not grading itself)
  * answer accuracy     - expected facts present (expect_all) / any-of present (expect_any)
  * refusal precision / recall - unanswerable & injection questions must be refused;
                          answerable ones must not be (tunes the GATE_* thresholds)
  * injection safety    - forbidden strings never appear in an answer
  * latency p50/p95, tokens, cost (from the response metrics)"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import httpx

NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _norm_num(x: str) -> str:
    return x.replace(",", ".")


def pct(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, round((p / 100) * (len(v) - 1)))]


def evaluate(client: httpx.Client, dataset: list[dict], conversation_id: str | None = None) -> dict:
    convs: dict[str, str] = {}
    rows = []
    for item in dataset:
        key = item.get("conversation") or item["id"]
        if conversation_id is not None:
            cid = conversation_id  # every question runs in the pre-seeded conversation
        else:
            if key not in convs:
                convs[key] = client.post("/v1/conversations").json()["id"]
            cid = convs[key]
        t0 = time.perf_counter()
        resp = client.post(f"/v1/conversations/{cid}/messages", json={"question": item["q"]}).json()
        wall_ms = round((time.perf_counter() - t0) * 1000)

        answered = resp["status"] in ("ok", "partial")
        text = " ".join(s["text"] for s in resp.get("segments", []))
        cites = {c["n"]: c for c in resp.get("citations", [])}

        # citation validity: stored passage text == cited text
        valid = []
        for c in cites.values():
            p = client.get(f"/v1/passages/{c['sentence_id']}", params={"context": 0})
            valid.append(p.status_code == 200 and p.json()["citation"]["text"] == c["text"])

        # groundedness: each segment cited + numbers appear in cited text
        seg_ok = []
        for seg in resp.get("segments", []):
            cited_text = " ".join(cites[n]["text"] for n in seg["citations"] if n in cites)
            nums = {_norm_num(n) for n in NUM.findall(seg["text"])}
            src = {_norm_num(n) for n in NUM.findall(cited_text)}
            seg_ok.append(bool(seg["citations"]) and nums <= src)

        low = text.lower()
        acc = None
        if item["answerable"]:
            checks = [e.lower() in low for e in item.get("expect_all", [])]
            if item.get("expect_any"):
                checks.append(any(e.lower() in low for e in item["expect_any"]))
            acc = answered and all(checks)
            if answered and item.get("cite_doc"):
                acc = acc and any(c["document_name"] == item["cite_doc"] for c in cites.values())
        leaked = any(f.lower() in (low + json.dumps(resp).lower()) for f in item.get("forbid", [])) if answered else False

        rows.append({
            "id": item["id"], "type": item["type"], "question": item["q"], "answerable": item["answerable"],
            "status": resp["status"], "answered": answered, "accurate": acc, "leaked": leaked,
            "citations": len(cites), "citation_valid": all(valid) if valid else None,
            "grounded": all(seg_ok) if seg_ok else None, "dropped_claims": resp.get("dropped_claims", 0),
            "latency_ms": resp["metrics"]["total_ms"], "wall_ms": wall_ms,
            "tokens_in": resp["metrics"]["tokens_in"], "tokens_out": resp["metrics"]["tokens_out"],
            "cost_usd": resp["metrics"]["cost_usd"], "answer": text[:300], "refusal_reason": resp.get("refusal_reason"),
        })
    return summarize(rows)


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if r["answered"]]
    tp_refuse = sum(1 for r in unans if not r["answered"])
    refused_all = [r for r in rows if not r["answered"]]
    lat = [r["latency_ms"] for r in rows]
    s = {
        "n": len(rows),
        "answer_accuracy": round(sum(1 for r in ans if r["accurate"]) / max(1, len(ans)), 3),
        "refusal_recall": round(tp_refuse / max(1, len(unans)), 3),  # unanswerable correctly refused
        "refusal_precision": round(tp_refuse / max(1, len(refused_all)), 3),  # refusals that were right
        "false_refusal_rate": round(sum(1 for r in ans if not r["answered"]) / max(1, len(ans)), 3),
        "citation_validity": round(sum(1 for r in answered if r["citation_valid"]) / max(1, len(answered)), 3),
        "groundedness": round(sum(1 for r in answered if r["grounded"]) / max(1, len(answered)), 3),
        "injection_leaks": sum(1 for r in rows if r["leaked"]),
        "latency_p50_ms": pct(lat, 50), "latency_p95_ms": pct(lat, 95),
        "avg_tokens_in": round(statistics.mean(r["tokens_in"] for r in rows), 1),
        "avg_tokens_out": round(statistics.mean(r["tokens_out"] for r in rows), 1),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
        "avg_cost_usd": round(sum(r["cost_usd"] for r in rows) / max(1, len(rows)), 6),
    }
    return {"summary": s, "rows": rows}


def to_markdown(result: dict, meta: str = "") -> str:
    s = result["summary"]
    out = ["# DocTalk evaluation report", "", meta, "", "## Summary", "", "| Metric | Value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in s.items()]
    out += ["", "## Per question", "", "| id | type | status | accurate | grounded | cites | ms | question |", "|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        out.append(f"| {r['id']} | {r['type']} | {r['status']} | {r['accurate']} | {r['grounded']} | {r['citations']} | {r['latency_ms']} | {r['question']} |")
    fails = [r for r in result["rows"] if (r["answerable"] and not r["accurate"]) or (not r["answerable"] and r["answered"]) or r["leaked"]]
    out += ["", "## Failures to inspect", ""]
    out += [f"- **{r['id']}** ({r['type']}): status={r['status']} - {r['answer'] or r['refusal_reason']}" for r in fails] or ["None."]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--issuer", default="http://localhost:9000")
    ap.add_argument("--token")
    ap.add_argument("--docs", nargs="+", default=["eval/docs/hr_policy.md", "eval/docs/it_security.md"])
    ap.add_argument("--dataset", default=str(Path(__file__).with_name("dataset.json")))
    ap.add_argument("--out", default="eval/report.md")
    a = ap.parse_args()
    token = a.token or httpx.post(f"{a.issuer}/token", params={"user": "eval"}).json()["access_token"]
    client = httpx.Client(base_url=a.api, headers={"Authorization": f"Bearer {token}"}, timeout=180)
    cid = client.post("/v1/conversations").json()["id"]
    files = [("files", (Path(d).name, open(d, "rb"))) for d in a.docs]
    client.post("/v1/documents", data={"conversation_id": cid}, files=files).raise_for_status()
    result = evaluate(client, json.loads(Path(a.dataset).read_text()), conversation_id=cid)
    Path(a.out).write_text(to_markdown(result, f"API: {a.api} - run: {time.strftime('%Y-%m-%d %H:%M')}"))
    Path(a.out).with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
