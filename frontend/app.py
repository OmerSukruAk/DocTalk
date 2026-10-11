"""DocTalk web UI (Dash + Mantine). Runs in its own container and talks to the API over HTTP only.

  python frontend/app.py                      # http://localhost:8050, API at $API_URL
  gunicorn app:server -b 0.0.0.0:8050         # production (see Dockerfile)
"""
from __future__ import annotations

import base64
import binascii
import itertools
import json
import os
import time
import uuid
from datetime import datetime

import dash_mantine_components as dmc
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
from dash_iconify import DashIconify

import api
import viewer
from theme import ACCENT, FONT_URL, THEME

SUGGESTIONS = [
    "Summarise the documents in three bullet points",
    "What are the most important rules I should know?",
    "Are there any deadlines or limits mentioned?",
]

app = Dash(
    __name__,
    title="DocTalk",
    update_title=None,
    external_stylesheets=[FONT_URL, dmc.styles.NOTIFICATIONS],
    suppress_callback_exceptions=True,
)
server = app.server


@server.route("/healthz")
def healthz():
    return {"status": "ok"}


def icon(name: str, size: int = 18, **kwargs) -> DashIconify:
    return DashIconify(icon=f"tabler:{name}", width=size, **kwargs)


def notify(message: str, title: str | None = None, color: str = "nn") -> list[dict]:
    return [{
        "action": "show", "id": uuid.uuid4().hex, "title": title, "message": message,
        "color": color, "autoClose": 6000, "radius": "lg",
    }]


def api_error(exc: api.ApiError) -> list[dict]:
    if exc.status == 401:
        return notify("Your session has expired. Please sign out and sign in again.", "Signed out")
    return notify(exc.detail, "Something went wrong")


def _stamp(iso: str | None) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%b %d · %H:%M")
    except (TypeError, ValueError):
        return "Conversation"


def _display_name(token: str) -> str:
    """Name for the avatar only (the API validates the token; this does not)."""
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError, binascii.Error):
        return "User"
    name = claims.get("name") or claims.get("preferred_username") or claims.get("oid") or "User"
    return str(name).removeprefix("user-")  # dev issuer object ids look like "user-alice"


# --- answer rendering ---------------------------------------------------------------------------

def _location(citation: dict) -> str:
    loc = citation.get("location") or {}
    if loc.get("page"):
        return f"Page {loc['page']}"
    return loc.get("heading") or ""


def source_card(citation: dict, open_id: dict | None = None) -> html.Div:
    """A citation: document, location, verbatim sentence; `open_id` adds an "Open in document" button."""
    location = _location(citation)
    deleted = citation.get("document_deleted")
    return html.Div([
        dmc.Group([
            html.Span(str(citation["n"]), className="cite cite-static"),
            dmc.Text(citation["document_name"], fw=600, size="sm", truncate="end", style={"flex": 1}),
            dmc.Badge("deleted", color="gray", variant="light", size="xs") if deleted else None,
            dmc.Button("Open in document", id=open_id, size="compact-xs", variant="subtle",
                       rightSection=icon("arrow-up-right", 12)) if open_id and not deleted else None,
        ], gap=8, wrap="nowrap"),
        html.Div([html.Span(location) if location else None,
                  html.Code(citation["sentence_id"], className="mono")], className="meta source-meta"),
        html.Blockquote(citation["text"], className="quote"),
    ], className="source")


def _cite_pills(numbers: list[int], by_n: dict[int, dict], m: int, seq: itertools.count) -> list:
    """Clickable [n] pills; hovering previews the source, clicking opens it in the document."""
    pills = []
    for n in numbers:
        if n not in by_n:
            pills.append(html.Span(str(n), className="cite"))
            continue
        pill = html.Span(str(n), className="cite cite-link", n_clicks=0,
                         id={"type": "cite", "m": m, "n": n, "k": next(seq)})
        pills.append(dmc.HoverCard([
            dmc.HoverCardTarget(pill, boxWrapperProps={"style": {"display": "inline"}}),
            dmc.HoverCardDropdown([
                source_card(by_n[n]),
                html.Div([icon("pointer", 13), "Click to open it in the document"], className="meta", style={"marginTop": 8})
                if not by_n[n].get("document_deleted") else None,
            ]),
        ], width=380, openDelay=150, withArrow=True, position="top", withinPortal=True))
    return pills


def _segment(seg: dict, pills) -> html.Span:
    kind = seg.get("kind", "fact")
    if kind == "gap":
        return html.Span(seg["text"], className="seg-gap")
    children = [seg["text"], *pills(seg.get("citations", []))]
    return html.Span(children, className="seg-insight" if kind == "insight" else None)


def answer_blocks(segments: list[dict], pills) -> list:
    """Same layout rules as the CLI: bullets become a list, `paragraph` starts a new block."""
    blocks, line, bullets = [], [], []

    def flush_line():
        if line:
            blocks.append(html.P(list(line), className="answer-p"))
            line.clear()

    def flush_bullets():
        if bullets:
            blocks.append(html.Ul([html.Li(b) for b in bullets], className="answer-ul"))
            bullets.clear()

    for seg in segments:
        fmt = seg.get("format", "text")
        if fmt == "bullet":
            flush_line()
            bullets.append(_segment(seg, pills))
            continue
        flush_bullets()
        if fmt == "paragraph":
            flush_line()
        if line:
            line.append(" ")
        line.append(_segment(seg, pills))
    flush_line()
    flush_bullets()
    return blocks


def _metrics(m: dict | None) -> html.Div | None:
    if not m:
        return None
    tokens = m.get("tokens_in", 0) + m.get("tokens_out", 0)
    return html.Div([
        icon("clock", 13), f"{m.get('total_ms', 0) / 1000:.1f}s",
        html.Span("·"), f"{tokens:,} tokens",
        html.Span("·"), f"${m.get('cost_usd', 0):.4f}",
    ], className="meta")


_STATUS = {
    "ok": ("teal", "circle-check", "Verified"),
    "partial": ("yellow", "alert-circle", "Partly verified"),
    "refused": ("gray", "circle-minus", "Refused"),
    "error": ("red", "alert-triangle", "Error"),
}
_CHECKED_BY = {"rules": "rule check", "nli_fail": "NLI model", "judge": "LLM judge",
               "judge_unavailable": "LLM judge unavailable", "gap": "gap check"}


def _secs(ms: float | None) -> str:
    if ms is None:
        return ""
    return f"{ms / 1000:.1f} s" if ms >= 100 else f"{ms:.0f} ms"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _warnings(msg: dict) -> dmc.Alert | None:
    """Anything the user should know about how this answer was produced."""
    diag = msg.get("diagnostics") or {}
    items = []
    for r in diag.get("removed", []):
        why = "; ".join(r.get("reasons") or []) or "not supported by the cited sources"
        items.append(html.Li([
            "Removed: ", html.Q(r["text"], className="removed-text"),
            html.Div(f"{why} · {_CHECKED_BY.get(r.get('checked_by'), r.get('checked_by', ''))}", className="warn-why"),
        ]))
    if msg.get("status") == "partial" and not diag.get("removed"):  # saved answers keep only the status
        items.append(html.Li("Some statements were removed because the documents did not support them."))
    if diag.get("attempts", 1) > 1:
        items.append(html.Li("Nothing could be verified on the first try, so DocTalk searched wider and "
                             f"tried again ({diag['attempts']} attempts)."))
    if diag.get("translated_query"):
        items.append(html.Li(["The question was translated before searching: ", html.Q(diag["translated_query"])]))
    deleted = [f"[{c['n']}]" for c in msg.get("citations") or [] if c.get("document_deleted")]
    if deleted:
        items.append(html.Li(f"Citation {', '.join(deleted)} points to a document that has since been deleted."))
    if not items:
        return None
    return dmc.Alert(html.Ul(items, className="warn-list"), title=_plural(len(items), "warning"),
                     color="yellow", variant="light", radius="lg", icon=icon("alert-triangle"))


def _step(number: str, title: str, ms: float | None, detail=None, skipped: bool = False) -> html.Div:
    return html.Div([
        html.Div(number, className="step-num"),
        html.Div([html.Div(title, className="step-title"),
                  html.Div(detail, className="step-detail") if detail else None], className="step-body"),
        html.Div("skipped" if skipped else _secs(ms), className="step-time"),
    ], className="step step-skipped" if skipped else "step")


def _lines(items: list) -> html.Ul:
    return html.Ul([html.Li(i) for i in items if i], className="step-lines")


def _pct(x: float | None) -> str:
    return f"{x:.0%}" if x is not None else "–"


def _search_lines(search: dict | None, attempts: int, cited: int, documents: int):
    """The retrieval funnel in plain words: found → merged → re-ranked → kept → sentences."""
    if not search:
        return None
    if search["mode"] == "overview":
        lines = [f"Overview question: an even sample of {_plural(search['kept_chunks'], 'chunk')} from every section "
                 "(no similarity search)"]
    else:
        found = f"Keyword search (BM25) found {_plural(search['keyword_found'], 'chunk')}"
        found += (f" · vector search found {search['vector_found']}" if search["mode"] == "hybrid"
                  else " · vector search off (no embeddings key)")
        lines = [found, f"Merged (RRF) into {_plural(search['merged'], 'unique chunk')}"]
        if search["reranked"]:
            lines.append(f"Re-ranker scored the top {search['candidates']} and kept the best {search['kept_chunks']}")
        else:
            lines.append(f"Kept the top {search['kept_chunks']} (top-k = {search['top_k']}; no re-ranker configured)")
        signals = [f"best similarity {search['vector_top']:.2f}" if search.get("vector_top") is not None else None,
                   f"re-ranker {_pct(search['rerank_top'])}" if search.get("rerank_top") is not None else None,
                   f"keyword match {_pct(search.get('lexical_top'))}"]
        scores = " · ".join(x for x in signals if x)
        lines.append(f"Relevance check passed: {scores}" if search.get("gate_passed") else
                     f"Relevance check failed: {scores}, so nothing relevant enough to write an answer from")
    if search.get("gate_passed") is not False:
        extra = f" (+{search['context_sentences']} neighbours for context)" if search["context_sentences"] else ""
        lines.append(f"{_plural(search['sentences'], 'sentence')}{extra} went to the writer; "
                     f"{cited} cited from {_plural(documents, 'document')}")
    if attempts > 1:
        lines.append(f"Final attempt of {attempts}: the retry widened top-k to {search['top_k']}")
    return _lines(lines)


_KINDS = (("fact", "fact", "facts"), ("insight", "analysis", "analysis"), ("gap", "gap", "gaps"),
          ("framing", "framing", "framing"))
_CHECKS = (
    ("rules", "Rule checks (citation ids, numbers, negations)"),
    ("gap", "Gap check"),
    ("framing", "Framing check (greetings, headings: no information allowed)"),
    ("nli", "NLI model"),
    ("judge", "Judge agent (LLM)"),
    ("judge_unavailable", "Judge agent unavailable"),
)


def _kinds(counts: dict) -> str:
    return " · ".join(f"{counts[k]} {one if counts[k] == 1 else many}" for k, one, many in _KINDS if counts.get(k))


def _check_lines(verification: dict | None):
    """Who decided each statement: e.g. "Judge agent (LLM) kept 3 facts · 1 analysis; removed 1 fact"."""
    if verification is None:
        return None
    lines = []
    for key, label in _CHECKS:
        if key == "rules" and key not in verification:
            lines.append(f"{label}: all statements passed")
            continue
        counts = verification.get(key)
        if not counts:
            continue
        parts = [f"kept {_kinds(counts['kept'])}" if counts.get("kept") else None,
                 f"removed {_kinds(counts['removed'])}" if counts.get("removed") else None]
        lines.append(f"{label} " + "; ".join(x for x in parts if x))
    return _lines(lines)


def _steps(msg: dict, question: str | None) -> list:
    """The pipeline, one row per step: what it did and how long it took."""
    t = (msg.get("metrics") or {}).get("timings_ms") or {}
    metrics = msg.get("metrics") or {}
    diag = msg.get("diagnostics") or {}
    segments = msg.get("segments") or []
    citations = msg.get("citations") or []
    kinds = {k: sum(1 for s in segments if s.get("kind", "fact") == k) for k in ("fact", "insight", "gap")}
    removed = len(diag.get("removed", [])) or msg.get("dropped_claims", 0)
    documents = len({c["document_id"] for c in citations})
    nothing_written = t.get("generate", 0) < 1  # refused by the relevance gate before any LLM call

    rewritten = (msg.get("rewritten_question") or "").strip()
    if rewritten and rewritten.lower() != (question or "").strip().lower():
        understood = [html.Q(question or ""), " → ", html.Q(rewritten, className="rewritten")]
    elif t.get("rewrite", 0) < 1:
        understood = "First question in the chat: nothing to resolve"
    else:
        understood = "Clear on its own, used as asked"
    rows = [_step("1", "Understanding the question", t.get("rewrite"), understood, skipped=t.get("rewrite", 0) < 1)]
    if diag.get("translated_query"):
        rows.append(_step("↳", "Translating the question", t.get("translate"),
                          ["Searched as ", html.Q(diag["translated_query"])]))

    found = _search_lines(diag.get("search"), diag.get("attempts", 1), len(citations), documents)
    rows.append(_step("2", "Searching the documents", t.get("retrieve"), found or (
        "Nothing relevant found" if nothing_written
        else f"{_plural(len(citations), 'sentence')} cited from {_plural(documents, 'document')}")))
    rows.append(_step("3", "Writing the answer", t.get("generate"),
                      None if nothing_written else f"{_plural(len(segments) + removed, 'statement')} drafted"
                      + (f" in the requested style: “{msg['style']}”" if msg.get("style") else ""),
                      skipped=nothing_written))
    checked = _check_lines(diag.get("verification")) or (
        f"Kept {_plural(kinds['fact'], 'fact')} · {kinds['insight']} analysis · "
        f"{_plural(kinds['gap'], 'gap')} · removed {removed}")
    rows.append(_step("4", "Fact-checking", t.get("verify"), None if nothing_written else checked,
                      skipped=nothing_written))
    if t.get("other", 0) >= 10:
        rows.append(_step("", "Loading history, saving, logging", t["other"]))

    tokens_in, tokens_out = metrics.get("tokens_in", 0), metrics.get("tokens_out", 0)
    rows.append(html.Div([
        html.Div("", className="step-num"),
        html.Div([html.Div("Total", className="step-title"),
                  html.Div(f"{tokens_in + tokens_out:,} tokens ({tokens_in:,} in / {tokens_out:,} out) · "
                           f"${metrics.get('cost_usd', 0):.5f}", className="step-detail")], className="step-body"),
        html.Div(_secs(metrics.get("total_ms", 0)), className="step-time"),
    ], className="step step-total"))
    return rows


def answer_details(msg: dict, question: str | None) -> list:
    """Status, warnings and the step-by-step pipeline for one answer."""
    color, status_icon, status_label = _STATUS.get(msg.get("status", "ok"), _STATUS["ok"])
    header = dmc.Group([
        dmc.Badge(status_label, color=color, variant="light", leftSection=icon(status_icon, 12)),
        html.Code(msg.get("message_id") or msg.get("id") or "", className="mono"),
    ], justify="space-between")
    if msg.get("metrics"):
        return [header, _warnings(msg), html.Div(_steps(msg, question), className="steps")]
    # answers saved before the details were stored keep their text, status and sources only
    segments = msg.get("segments") or []
    citations = msg.get("citations") or []
    kinds = {k: sum(1 for s in segments if s.get("kind", "fact") == k) for k in ("fact", "insight", "gap")}
    documents = len({c["document_id"] for c in citations})
    return [
        header, _warnings(msg),
        dmc.Text(f"{_plural(kinds['fact'], 'fact')} · {kinds['insight']} analysis · {_plural(kinds['gap'], 'gap')} · "
                 f"{_plural(len(citations), 'sentence')} cited from {_plural(documents, 'document')}", size="sm"),
        dmc.Text("Step times, tokens and warnings were not saved for this older answer.",
                 size="xs", c="dimmed"),
    ]


def assistant_body(msg: dict, m: int, question: str | None = None, show_details: bool = True) -> list:
    """`m` is the message's index in the chat; citation ids carry it so a click finds its source."""
    status = msg.get("status", "ok")
    if status in ("refused", "error"):
        refused = status == "refused"
        return [dmc.Alert(
            msg.get("refusal_reason") or msg.get("text") or "",
            title="Not covered by your documents" if refused else "Something went wrong",
            color="gray" if refused else "nn",
            variant="light", radius="lg",
            icon=icon("file-search" if refused else "alert-triangle"),
        ), _extras(msg, question, show_details, []), None if show_details else _metrics(msg.get("metrics"))]

    citations = msg.get("citations") or []
    by_n = {c["n"]: c for c in citations}
    seq = itertools.count()
    segments = msg.get("segments") or []
    body = (answer_blocks(segments, lambda numbers: _cite_pills(numbers, by_n, m, seq)) if segments
            else [html.P(msg.get("text", ""), className="answer-p")])

    notes = []
    if any(s.get("kind") == "insight" for s in segments):
        notes.append(html.Div([html.Span(className="swatch"), "Highlighted text is DocTalk's own analysis of the cited facts."],
                              className="meta"))
    if status == "partial":
        dropped = msg.get("dropped_claims")
        what = f"{dropped} statement{'s' if dropped != 1 else ''}" if dropped else "Some statements"
        them = "it" if dropped == 1 else "them"
        notes.append(html.Div([icon("shield-check", 13), f"{what} removed: the documents did not support {them}."],
                              className="meta"))
    cards = [source_card(c, {"type": "cite", "m": m, "n": c["n"], "k": "src"}) for c in citations]
    footer = None if show_details else _metrics(msg.get("metrics"))  # the details panel already has it
    return [*body, *notes, _extras(msg, question, show_details, cards), footer]


def _extras(msg: dict, question: str | None, show_details: bool, cards: list):
    """Collapsible "Sources" and "Answer details" under an answer."""
    items = []
    if cards:
        items.append(dmc.AccordionItem([
            dmc.AccordionControl(f"Sources · {len(cards)}", icon=icon("quote", 16)),
            dmc.AccordionPanel(dmc.Stack(cards, gap="md")),
        ], value="sources"))
    if show_details:
        items.append(dmc.AccordionItem([
            dmc.AccordionControl("Answer details", icon=icon("list-details", 16)),
            dmc.AccordionPanel(dmc.Stack(answer_details(msg, question), gap="md")),
        ], value="details"))
    if not items:
        return None
    return dmc.Accordion(items, multiple=True, value=[], variant="filled", radius="lg",
                         chevronPosition="right", className="sources")


def render_message(msg: dict, m: int, question: str | None = None, show_details: bool = True) -> html.Div:
    if msg.get("role") == "user":
        return html.Div(html.Div(msg.get("text", ""), className="bubble"), className="msg msg-user")
    mark = html.Div(icon("message-2", 16), className="mark")
    if msg.get("pending"):
        return html.Div([mark, html.Div([html.Span(), html.Span(), html.Span()], className="typing")],
                        className="msg msg-assistant")
    return html.Div([mark, html.Div(assistant_body(msg, m, question, show_details), className="answer")],
                    className="msg msg-assistant")


def empty_state() -> html.Div:
    return html.Div([
        html.Div(icon("sparkles", 30), className="hero-icon"),
        dmc.Title("Ask your documents", order=2, className="hero-title"),
        dmc.Text("Add up to five PDF, Word or Markdown files, then ask anything. "
                 "Every answer shows the exact sentences it comes from.", c="dimmed", ta="center", maw=460),
        dmc.Group([
            dmc.Button(q, id={"type": "suggest", "index": i}, variant="default", size="sm", className="suggestion")
            for i, q in enumerate(SUGGESTIONS)
        ], justify="center", gap=8, mt="sm"),
    ], className="empty")


# --- layout -------------------------------------------------------------------------------------

def signin_view() -> html.Div:
    """Sign-in needs an access token: Entra ID in production, the dev issuer locally (see README)."""
    return html.Div(dmc.Paper([
        html.Div(icon("message-2", 30), className="hero-icon", style={"margin": "0 auto"}),
        dmc.Title("Welcome to DocTalk", order=2, ta="center", mt="lg", className="hero-title"),
        dmc.Text("Discuss your documents. Every answer comes with its sources.",
                 c="dimmed", ta="center", size="sm", mb="xl"),
        dmc.Stack([
            dmc.PasswordInput(id="signin-token", label="Access token", placeholder="Paste your bearer token",
                              size="md", radius="md"),
            dmc.Button("Sign in", id="signin-btn", fullWidth=True, size="md"),
        ], gap="sm"),
        dmc.Text("Your identity comes from the token, which the API verifies.", c="dimmed", ta="center",
                 size="xs", mt="md"),
    ], className="signin-card", p=36, shadow="lg"), className="signin", id="signin-view")


header = dmc.AppShellHeader(dmc.Group([
    dmc.Group([
        dmc.Burger(id="burger", size="sm", hiddenFrom="sm", opened=False),
        html.Div(icon("message-2", 16), className="mark"),
        dmc.Text("DocTalk", fw=650, size="lg", className="wordmark"),
    ], gap=10),
    dmc.Group([
        dmc.SegmentedControl(
            id="scheme", value="auto", size="xs", radius="xl", persistence=True, persistence_type="local",
            data=[{"value": "light", "label": icon("sun", 14)},
                  {"value": "dark", "label": icon("moon", 14)},
                  {"value": "auto", "label": icon("device-desktop", 14)}],
            visibleFrom="xs",
        ),
        dmc.Menu([
            dmc.MenuTarget(dmc.Avatar(id="avatar", radius="xl", color="nn", size=32, className="avatar")),
            dmc.MenuDropdown([
                dmc.MenuLabel("", id="user-label"),
                dmc.MenuItem("Sign out", id="signout", leftSection=icon("logout", 16)),
            ]),
        ], position="bottom-end", width=200),
    ], gap=12),
], justify="space-between", h="100%", px="md"), className="glass")

navbar = dmc.AppShellNavbar([
    dmc.Button("New chat", id="new-chat", leftSection=icon("edit", 16), variant="default", fullWidth=True),
    dmc.Text("Conversations", className="section-label"),
    dmc.ScrollArea(html.Div(id="conv-list"), style={"flex": 1}, type="scroll"),
    dmc.Text("Answers come only from your documents, with a source for every claim.", className="sidebar-foot"),
], p="md", className="glass sidebar")

main = dmc.AppShellMain(html.Div([
    dmc.Group([
        dmc.Text(id="conv-title", fw=600, size="sm", c="dimmed"),
        dmc.Tooltip(dmc.ActionIcon(icon("trash", 16), id="delete-conv", variant="subtle", color="gray"),
                    label="Delete conversation", withArrow=True),
    ], justify="space-between", className="topbar"),
    html.Div([
        html.Div(id="doc-chips", className="chips"),
        dcc.Upload(
            html.Div([icon("cloud-upload", 18), html.Span("Drop files here or "),
                      html.Span("browse", className="link"), html.Span("PDF · DOCX · MD, up to 5", className="hint")],
                     className="dropzone-inner"),
            id="upload", multiple=True, accept=".pdf,.docx,.md,.markdown", className="dropzone",
        ),
    ], className="docs"),
    html.Div(id="chat", className="chat"),
    html.Div([
        html.Div(id="style-badge", className="style-badge-row"),
        html.Div([
            dmc.Popover([
                dmc.PopoverTarget(dmc.ActionIcon(icon("adjustments-horizontal", 18), variant="subtle",
                                                 color="gray", size="lg")),
                dmc.PopoverDropdown(dmc.Stack([
                    dmc.TextInput(id="style", label="Answer style",
                                  placeholder="e.g. casual, executive summary, in Turkish"),
                    dmc.Switch(id="include-public", label="Also search shared public documents", size="sm"),
                    dmc.Switch(id="show-details", label="Show answer details", size="sm", checked=True,
                               persistence=True, persistence_type="local"),
                ], gap="md", w=290)),
            ], position="top-start", offset=12),
            dmc.Textarea(id="question", placeholder="Ask about your documents…", autosize=True, minRows=1,
                         maxRows=6, variant="unstyled", className="composer-input"),
            dmc.ActionIcon(icon("arrow-up", 18), id="send", size=36, variant="filled", className="send"),
        ], className="composer"),
        html.Div("DocTalk answers only from your documents and cites every claim.", className="composer-hint"),
    ], className="composer-wrap"),
], className="column"))

source_viewer = dmc.Drawer([
    dmc.Group([
        dmc.ActionIcon(icon("chevron-left", 16), id="viewer-prev", variant="default", size="lg"),
        dmc.Text(id="viewer-page", size="sm", fw=500, miw=90, ta="center"),
        dmc.ActionIcon(icon("chevron-right", 16), id="viewer-next", variant="default", size="lg"),
        dmc.Button("Back to cited page", id="viewer-back", variant="light", size="compact-sm",
                   leftSection=icon("focus-2", 14)),
    ], id="viewer-toolbar", gap=8, className="viewer-toolbar"),
    dcc.Loading(html.Div(id="viewer-body"), type="circle", color=ACCENT, delay_show=250),
], id="viewer", title=html.Div(id="viewer-title"), opened=False, position="right", size="760px",
   padding="lg", overlayProps={"backgroundOpacity": 0.2, "blur": 4}, className="viewer")

app.layout = dmc.MantineProvider([
    dcc.Store(id="session", storage_type="session"),
    dcc.Store(id="active-conv", storage_type="session"),
    dcc.Store(id="messages", data=[]),
    dcc.Store(id="ask-request"),
    dcc.Store(id="conv-version"),
    dcc.Store(id="docs-version"),
    dcc.Store(id="viewer-state"),
    dcc.Store(id="viewer-pages"),
    html.Div(id="scroll-sink", hidden=True),
    dmc.NotificationContainer(id="notify", position="top-right"),
    signin_view(),
    source_viewer,
    html.Div(dmc.AppShell(
        [header, navbar, main],
        id="shell", header={"height": 60}, padding=0,
        navbar={"width": 280, "breakpoint": "sm", "collapsed": {"mobile": True}},
    ), id="app-view", style={"display": "none"}),
], id="provider", theme=THEME, defaultColorScheme="auto")


# --- session ------------------------------------------------------------------------------------

@app.callback(
    Output("session", "data"),
    Output("notify", "sendNotifications"),
    Input("signin-btn", "n_clicks"),
    Input("signin-token", "n_submit"),
    State("signin-token", "value"),
    prevent_initial_call=True,
    running=[(Output("signin-btn", "loading"), True, False)],
)
def sign_in(_btn, _enter, pasted):
    token = (pasted or "").strip()
    if not token:
        return no_update, notify("Paste an access token first.")
    try:
        api.list_conversations(token)  # the API validates signature, issuer, audience and expiry
    except api.ApiError as exc:
        return no_update, notify(exc.detail, "Could not sign in")
    return {"token": token, "user": _display_name(token)}, no_update


@app.callback(Output("session", "data", allow_duplicate=True), Input("signout", "n_clicks"),
              prevent_initial_call=True)
def sign_out(_):
    return None


@app.callback(
    Output("signin-view", "style"),
    Output("app-view", "style"),
    Output("avatar", "children"),
    Output("user-label", "children"),
    Input("session", "data"),
)
def show_view(session):
    if not session:
        return {}, {"display": "none"}, None, None
    user = session.get("user", "User")
    return {"display": "none"}, {}, user[:2].upper(), f"Signed in as {user}"


@app.callback(Output("shell", "navbar"), Input("burger", "opened"), State("shell", "navbar"))
def toggle_navbar(opened, navbar):
    navbar["collapsed"] = {"mobile": not opened}
    return navbar


app.clientside_callback(
    """(scheme) => scheme === 'auto'
        ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
        : scheme""",
    Output("provider", "forceColorScheme"),
    Input("scheme", "value"),
)


# --- conversations ------------------------------------------------------------------------------

@app.callback(
    Output("active-conv", "data"),
    Output("messages", "data"),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Output("burger", "opened"),
    Input("session", "data"),
    Input({"type": "conv", "id": ALL}, "n_clicks"),
    Input("new-chat", "n_clicks"),
    State("active-conv", "data"),
    prevent_initial_call="initial_duplicate",
)
def open_conversation(session, _conv_clicks, _new, active):
    """Load a conversation's history; also closes the phone sidebar after a pick."""
    trigger = ctx.triggered_id
    if not session or trigger == "new-chat":
        return None, [], no_update, False
    if isinstance(trigger, dict):
        if not ctx.triggered[0]["value"]:  # the list was re-rendered, nothing was clicked
            return no_update, no_update, no_update, no_update
        active = trigger["id"]
    if not active:
        return None, [], no_update, False
    try:
        return active, api.get_conversation(session["token"], active)["messages"], no_update, False
    except api.ApiError as exc:
        return None, [], (no_update if exc.status == 404 else api_error(exc)), False


@app.callback(
    Output("conv-list", "children"),
    Output("conv-title", "children"),
    Output("delete-conv", "style"),
    Input("session", "data"),
    Input("active-conv", "data"),
    Input("conv-version", "data"),
)
def list_conversations(session, active, _version):
    hidden = {"visibility": "hidden"}
    if not session:
        return [], "", hidden
    try:
        conversations = api.list_conversations(session["token"])
    except api.ApiError:
        return [], "", hidden
    links = [
        dmc.NavLink(
            label=_stamp(c.get("created_at")), id={"type": "conv", "id": c["id"]},
            leftSection=icon("message", 16), active=c["id"] == active, variant="light",
            className="conv-link",
        )
        for c in conversations
    ]
    if not links:
        links = [dmc.Text("No conversations yet", size="sm", c="dimmed", px="sm")]
    title = next((_stamp(c.get("created_at")) for c in conversations if c["id"] == active), "New chat")
    return links, title, ({} if active else hidden)


@app.callback(
    Output("active-conv", "data", allow_duplicate=True),
    Output("messages", "data", allow_duplicate=True),
    Output("conv-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("delete-conv", "n_clicks"),
    State("session", "data"),
    State("active-conv", "data"),
    prevent_initial_call=True,
)
def delete_conversation(_, session, active):
    if not session or not active:
        return no_update, no_update, no_update, no_update
    try:
        api.delete_conversation(session["token"], active)
    except api.ApiError as exc:
        return no_update, no_update, no_update, api_error(exc)
    return None, [], time.time(), notify("Conversation deleted.", color="gray")


# --- documents ----------------------------------------------------------------------------------

_FILE_ICONS = {".pdf": "file-type-pdf", ".docx": "file-type-docx", ".md": "markdown", ".markdown": "markdown"}


@app.callback(
    Output("doc-chips", "children"),
    Input("session", "data"),
    Input("active-conv", "data"),
    Input("docs-version", "data"),
)
def show_documents(session, active, _version):
    if not session or not active:
        return []
    try:
        docs = api.list_documents(session["token"], active)
    except api.ApiError:
        return []
    return [
        dmc.Tooltip(html.Div([
            icon(_FILE_ICONS.get(os.path.splitext(d["name"])[1].lower(), "file-text"), 16),
            html.Span(d["name"], className="chip-name"),
            html.Span(f"{d['sentence_count']} sentences", className="chip-meta"),
            dmc.ActionIcon(icon("x", 12), id={"type": "doc-del", "id": d["id"]}, variant="subtle",
                           color="gray", size="sm"),
        ], className="chip"), label=_doc_info(d), multiline=True, w=260, withArrow=True, openDelay=300)
        for d in docs
    ]


def _doc_info(doc: dict) -> html.Div:
    return html.Div([
        html.Div(doc["name"], style={"fontWeight": 600}),
        html.Div(f"{doc.get('type', '').upper()} · {doc['sentence_count']} citable sentences"),
        html.Div(f"Uploaded {_stamp(doc.get('created_at'))}"),
        html.Div(doc["id"], className="mono"),
    ])


def _ensure_conversation(token: str, active: str | None) -> tuple[str, bool]:
    if active:
        return active, False
    return api.create_conversation(token)["id"], True


@app.callback(
    Output("active-conv", "data", allow_duplicate=True),
    Output("docs-version", "data"),
    Output("conv-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Output("upload", "contents"),
    Input("upload", "contents"),
    State("upload", "filename"),
    State("session", "data"),
    State("active-conv", "data"),
    prevent_initial_call=True,
    running=[(Output("upload", "disabled"), True, False),
             (Output("upload", "className"), "dropzone busy", "dropzone")],
)
def upload(contents, names, session, active):
    if not contents or not session:
        return no_update, no_update, no_update, no_update, no_update
    files = [(name, base64.b64decode(data.split(",", 1)[1])) for name, data in zip(names, contents)]
    try:
        conversation_id, created = _ensure_conversation(session["token"], active)
        docs = api.upload_documents(session["token"], conversation_id, files)
    except api.ApiError as exc:
        return no_update, time.time(), no_update, api_error(exc), None
    sentences = sum(d["sentence_count"] for d in docs)
    message = f"{len(docs)} document{'s' if len(docs) != 1 else ''} ready · {sentences:,} citable sentences"
    return conversation_id, time.time(), (time.time() if created else no_update), notify(message, color="teal"), None


@app.callback(
    Output("docs-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input({"type": "doc-del", "id": ALL}, "n_clicks"),
    State("session", "data"),
    prevent_initial_call=True,
)
def delete_document(_clicks, session):
    if not ctx.triggered[0]["value"] or not session:
        return no_update, no_update
    try:
        api.delete_document(session["token"], ctx.triggered_id["id"])
    except api.ApiError as exc:
        return no_update, api_error(exc)
    return time.time(), notify("Document removed. Older answers keep their citations, marked as deleted.",
                               color="gray")


# --- asking -------------------------------------------------------------------------------------

@app.callback(
    Output("messages", "data", allow_duplicate=True),
    Output("question", "value"),
    Output("ask-request", "data"),
    Output("active-conv", "data", allow_duplicate=True),
    Output("conv-version", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("send", "n_clicks"),
    Input("question", "n_submit"),
    Input({"type": "suggest", "index": ALL}, "n_clicks"),
    State("question", "value"),
    State("messages", "data"),
    State("session", "data"),
    State("active-conv", "data"),
    prevent_initial_call=True,
)
def submit(_send, _enter, _suggest, question, messages, session, active):
    """Show the question at once with a typing indicator; `answer` then fetches the reply."""
    nothing = (no_update,) * 6
    if isinstance(ctx.triggered_id, dict):
        if not ctx.triggered[0]["value"]:
            return nothing
        question = SUGGESTIONS[ctx.triggered_id["index"]]
    question = (question or "").strip()
    messages = messages or []
    if not question or not session or any(m.get("pending") for m in messages):
        return nothing
    try:
        conversation_id, created = _ensure_conversation(session["token"], active)
    except api.ApiError as exc:
        return *nothing[:5], api_error(exc)
    messages = [*messages, {"role": "user", "text": question}, {"role": "assistant", "pending": True}]
    request = {"question": question, "conversation_id": conversation_id, "nonce": uuid.uuid4().hex}
    return messages, "", request, conversation_id, (time.time() if created else no_update), no_update


@app.callback(
    Output("messages", "data", allow_duplicate=True),
    Output("notify", "sendNotifications", allow_duplicate=True),
    Input("ask-request", "data"),
    State("session", "data"),
    State("style", "value"),
    State("include-public", "checked"),
    State("messages", "data"),
    prevent_initial_call=True,
    running=[(Output("send", "loading"), True, False)],
)
def answer(request, session, style, include_public, messages):
    messages = [m for m in (messages or []) if not m.get("pending")]
    if not request or not session:
        return messages, no_update
    try:
        reply = api.ask(session["token"], request["conversation_id"], request["question"], style,
                        bool(include_public))
    except api.ApiError as exc:
        return messages, api_error(exc)
    return [*messages, {"role": "assistant", **reply, "style": (style or "").strip() or None}], no_update


@app.callback(Output("chat", "children"), Input("messages", "data"), Input("show-details", "checked"))
def show_chat(messages, show_details):
    if not messages:
        return empty_state()

    def asked(m: int) -> str | None:  # the user question an answer replies to
        return messages[m - 1].get("text") if m and messages[m - 1].get("role") == "user" else None

    return [render_message(msg, m, asked(m), bool(show_details)) for m, msg in enumerate(messages)]


app.clientside_callback(
    """() => { setTimeout(() => window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'}), 60);
               return ''; }""",
    Output("scroll-sink", "children"),
    Input("chat", "children"),
)


@app.callback(Output("style-badge", "children"), Input("style", "value"))
def show_style(style):
    """While a style is set, show it above the message box, with an x to clear it."""
    style = (style or "").strip()
    if not style:
        return None
    return dmc.Badge(
        [icon("wand", 12), f" Answer style: {style}"], variant="light", radius="xl", size="lg", tt="none",
        rightSection=dmc.ActionIcon(icon("x", 12), id="style-clear", variant="transparent", size="xs", color="nn"),
        className="style-badge",
    )


@app.callback(Output("style", "value", allow_duplicate=True), Input("style-clear", "n_clicks"),
              prevent_initial_call=True)
def clear_style(clicks):
    return "" if clicks else no_update


# --- source viewer ------------------------------------------------------------------------------

@app.callback(
    Output("viewer", "opened"),
    Output("viewer-state", "data"),
    Input({"type": "cite", "m": ALL, "n": ALL, "k": ALL}, "n_clicks"),
    State("messages", "data"),
    prevent_initial_call=True,
)
def open_source(_clicks, messages):
    """A click on [n] (or "Open in document") opens that citation where it sits in its document."""
    if not ctx.triggered[0]["value"]:  # pills were re-rendered, nothing was clicked
        return no_update, no_update
    m, n = ctx.triggered_id["m"], ctx.triggered_id["n"]
    try:
        citation = next(c for c in messages[m]["citations"] if c["n"] == n)
    except (IndexError, KeyError, StopIteration):
        return no_update, no_update
    return True, {"citation": citation, "page": (citation.get("location") or {}).get("page")}


@app.callback(
    Output("viewer-state", "data", allow_duplicate=True),
    Input("viewer-prev", "n_clicks"),
    Input("viewer-next", "n_clicks"),
    Input("viewer-back", "n_clicks"),
    State("viewer-state", "data"),
    State("viewer-pages", "data"),
    prevent_initial_call=True,
)
def turn_page(_prev, _next, _back, state, pages):
    if not state or not state.get("page"):
        return no_update
    cited = state["citation"]["location"]["page"]
    page = {"viewer-prev": state["page"] - 1, "viewer-next": state["page"] + 1}.get(ctx.triggered_id, cited)
    return {**state, "page": max(1, min(page, pages or page))}


@app.callback(
    Output("viewer-title", "children"),
    Output("viewer-body", "children"),
    Output("viewer-page", "children"),
    Output("viewer-toolbar", "style"),
    Output("viewer-back", "style"),
    Output("viewer-pages", "data"),
    Input("viewer-state", "data"),
    State("session", "data"),
    prevent_initial_call=True,
)
def show_source(state, session):
    if not state or not session:
        return no_update, no_update, no_update, no_update, no_update, no_update
    citation, page = state["citation"], state.get("page")
    name = citation["document_name"]
    title = dmc.Group([
        icon(_FILE_ICONS.get(os.path.splitext(name)[1].lower(), "file-text"), 20),
        dmc.Text(name, fw=650, size="md", truncate="end", maw=440),
    ], gap=8, wrap="nowrap")
    hidden = {"display": "none"}
    try:
        if citation.get("document_deleted"):
            raise api.ApiError(404, "This document was deleted, so only the cited sentence is left.")
        if page:  # PDF: the rendered page with the sentence highlighted
            cited = citation["location"]["page"]
            highlight = (citation["text"], citation["location"].get("bbox")) if page == cited else None
            page, count, image, boxes = viewer.render_pdf_page(session["token"], citation["document_id"], page,
                                                               highlight)
            body = [
                source_card(citation),
                html.Div([html.Img(src=image, className="pdf-img", alt=f"{name}, page {page}"),
                          *[html.Div(className="hl", style=box) for box in boxes]], className="pdf-page"),
            ]
            return title, body, f"Page {page} of {count}", {}, ({} if page != cited else hidden), count
        # DOCX / Markdown: no pages, so show the sentence inside its passage
        passage = api.passage(session["token"], citation["sentence_id"])
    except api.ApiError as exc:
        alert = dmc.Alert(exc.detail, color="gray", variant="light", radius="lg", icon=icon("file-off"))
        return title, [alert, source_card(citation)], "", hidden, hidden, None
    heading = (citation.get("location") or {}).get("heading")
    body = html.Div([
        dmc.Text(heading, className="section-label", px=0) if heading else None,
        html.Div([
            *[html.P(t, className="ctx") for t in passage.get("context_before", [])],
            html.P(html.Mark(citation["text"]), className="passage-hit"),
            *[html.P(t, className="ctx") for t in passage.get("context_after", [])],
        ], className="passage"),
    ])
    return title, body, "", hidden, hidden, None


app.clientside_callback(
    """() => { setTimeout(() => {
        const hit = document.querySelector('.viewer .hl, .viewer .passage-hit');
        if (hit) hit.scrollIntoView({block: 'center', behavior: 'smooth'});
    }, 200); return ''; }""",
    Output("scroll-sink", "title"),
    Input("viewer-body", "children"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=os.getenv("DASH_DEBUG") == "1")
