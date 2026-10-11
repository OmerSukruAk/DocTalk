#!/usr/bin/env python3
"""DocTalk command-line client.

  python scripts/doctalk_cli.py demo eval/docs/hr_policy.md eval/docs/it_security.md
  python scripts/doctalk_cli.py chat  eval/docs/hr_policy.md          # interactive Q&A

Gets a token from the local dev issuer (AUTH_MODE=dev), uploads 1-5 documents, asks questions
and prints each answer with its numbered citations (document, page/section, sentence id,
verbatim text). With a real Entra tenant pass --token instead."""
from __future__ import annotations

import argparse
import sys
import textwrap

import httpx

DEMO_QUESTIONS = [
    "How many days of sick leave do I get, and when do I need a medical certificate?",
    "What is the hotel limit per night?",
    "And meals?",                                    # follow-up: resolved from conversation history
    "Can I work remotely from another country?",     # negation-sensitive
    "Summarise the documents in three bullet points for a new employee.",  # overview + tone
    "What is the capital of France?",                # not in the documents -> must refuse
]


def _seg_line(seg: dict) -> str:
    marks = "".join(f"[{n}]" for n in seg["citations"])
    note = " (analysis)" if seg.get("kind") == "insight" else ""
    return f"{seg['text']}{marks}{note}"


def render(resp: dict) -> None:
    status = resp["status"]
    if status in ("refused", "error"):
        print(f"  [{status.upper()}] {resp.get('refusal_reason')}")
    else:
        line = ""
        for seg in resp["segments"]:
            fmt = seg.get("format", "text")
            if fmt == "bullet" or (fmt == "paragraph" and line):
                if line:
                    print(textwrap.fill(line, 100, initial_indent="  ", subsequent_indent="  "))
                    if fmt == "paragraph":
                        print()
                line = ""
            if fmt == "bullet":
                print(textwrap.fill(_seg_line(seg), 100, initial_indent="  - ", subsequent_indent="    "))
            else:
                line = f"{line} {_seg_line(seg)}".strip()
        if line:
            print(textwrap.fill(line, 100, initial_indent="  ", subsequent_indent="  "))
        if status == "partial":
            print(f"  (note: {resp['dropped_claims']} unsupported statement(s) were removed by the verifier)")
    for c in resp["citations"]:
        loc = c["location"]
        where = f"p.{loc['page']}" if loc.get("page") else (loc.get("heading") or "")
        print(f"    [{c['n']}] {c['document_name']} - {where} - {c['sentence_id']}")
        print(textwrap.fill(f"\"{c['text']}\"", 96, initial_indent="        ", subsequent_indent="        "))
    m = resp["metrics"]
    print(f"  ({m['total_ms']} ms, {m['tokens_in']}+{m['tokens_out']} tokens, ${m['cost_usd']})")
    t = m.get("timings_ms") or {}
    parts = [f"{stage} {t[stage] / 1000:.1f}s" for stage in t if t[stage] >= 1 or stage in ("retrieve", "generate", "verify")]
    if parts:
        print(f"  time: {' | '.join(parts)}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["demo", "chat"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--issuer", default="http://localhost:9000")
    ap.add_argument("--user", default="alice")
    ap.add_argument("--token")
    ap.add_argument("--style", help='answer tone/format, e.g. "casual", "executive summary", "in Turkish"')
    a = ap.parse_args()

    token = a.token or httpx.post(f"{a.issuer}/token", params={"user": a.user}).json()["access_token"]
    c = httpx.Client(base_url=a.api, headers={"Authorization": f"Bearer {token}"}, timeout=120)

    cid = c.post("/v1/conversations").json()["id"]

    print("== 1. Upload ==")
    handles = [open(f, "rb") for f in a.files]
    r = c.post(
        "/v1/documents",
        data={"conversation_id": cid},
        files=[("files", (f.name.split("/")[-1], f)) for f in handles],
    )
    r.raise_for_status()
    for d in r.json():
        print(f"  {d['name']}: {d['sentence_count']} citable sentences (id {d['id']})")

    style: str | None = a.style

    def ask(q: str) -> dict:
        r = c.post(f"/v1/conversations/{cid}/messages", json={"question": q, "style": style})
        r.raise_for_status()
        return r.json()

    print("\n== 2. Ask ==")
    if a.mode == "chat":
        print("  (type 'exit' to quit, '/style <tone or format>' to change the answer style, '/style' to reset)")
    questions = DEMO_QUESTIONS if a.mode == "demo" else iter(lambda: input("you> ").strip(), "exit")
    for q in questions:
        if not q:
            continue
        if q.startswith("/style"):
            style = q[len("/style"):].strip() or None
            print(f"  style: {style or 'default'}\n")
            continue
        print(f"Q: {q}")
        render(ask(q))
    return 0


if __name__ == "__main__":
    sys.exit(main())
