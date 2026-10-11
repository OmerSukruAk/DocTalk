"""Regression tests for answer quality: PDF layout recovery, the relevance gate, overview
questions, commentary ('insight') segments and conversation history."""
import asyncio
import time

import mongomock

from app.agent.ingestion import parse
from app.agent.llm import ChatResult, build_sources_block
from app.agent.retrieval import RetrievalSignals, Hit, is_overview_question, overview_chunks, passes_gate
from app.agent.verification import Claim, verify_claims
from app.agent.workflow import parse_segments, render_answer
from app.config import Settings
from app.repositories.storage import ConversationRepo


def _striped_table_pdf() -> bytes:
    """A zebra-striped table (only header + every other row filled) under a section heading,
    plus KPI cards in a big font: the layout that used to lose half the rows."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "OPERATING RESULT", fontsize=8)
    # (base-14 Helvetica cannot encode the euro sign, so amounts are written as EUR here)
    page.insert_text((60, 80), "EUR 1.44B", fontsize=15)
    page.insert_text((300, 60), "INTERIM DIVIDEND", fontsize=8)
    page.insert_text((300, 80), "EUR 1.38", fontsize=15)
    page.insert_text((50, 130), "8. Dividend Distribution", fontsize=13)
    page.insert_text((50, 150), "The table below shows the dividend per share by year.", fontsize=10.5)
    cols = [50, 200, 350, 500]
    headers = ["FINANCIAL YEAR", "INTERIM DPS", "TOTAL DPS"]
    rows = [["2023 Actual", "EUR 1.12", "EUR 3.20"], ["2024 Actual", "EUR 1.28", "EUR 3.44"],
            ["2025 Reported", "EUR 1.38", "EUR 3.66"], ["2026 Target", "EUR 1.48", "EUR 3.88"]]
    y = 170
    for i, cells in enumerate([headers] + rows):
        if i % 2 == 0:  # header and every second data row are shaded, with cell borders
            for c0, c1 in zip(cols, cols[1:]):
                page.draw_rect(fitz.Rect(c0, y, c1, y + 24), color=(0.6, 0.6, 0.6), fill=(0.9, 0.9, 0.9))
        for c0, text in zip(cols, cells):
            page.insert_text((c0 + 4, y + 16), text, fontsize=9)
        y += 24
    page.insert_text((50, y + 40), "Strategic Insight: the dividend grows every year.", fontsize=10.5)
    data = doc.tobytes()
    doc.close()
    return data


def test_striped_table_keeps_every_row_with_its_headers():
    blocks = parse("report.pdf", _striped_table_pdf())
    rows = [b.text for b in blocks if b.is_table_row]
    for year in ("2023 Actual", "2024 Actual", "2025 Reported", "2026 Target"):
        assert any(year in r for r in rows), (year, rows)
    assert any("FINANCIAL YEAR: 2024 Actual" in r and "TOTAL DPS: EUR 3.44" in r for r in rows), rows
    # the paragraph under the table is prose, not a table row
    assert not any("Strategic Insight" in r for r in rows)
    assert any(b.text.startswith("Strategic Insight") and not b.is_table_row for b in blocks)


def test_kpi_numbers_are_not_headings_and_stay_with_their_label():
    blocks = parse("report.pdf", _striped_table_pdf())
    for b in blocks:
        assert "1.44B" not in (b.heading or "") and "1.38" not in (b.heading or "")
    assert any("OPERATING RESULT" in b.text and "EUR 1.44B" in b.text for b in blocks)
    table = [b for b in blocks if b.is_table_row]
    assert table and all(b.heading == "8. Dividend Distribution" for b in table)


def _hit():
    return [Hit("d:s1", "d", "doc", 1, "text", None, None, None, 1.0)]


def test_gate_passes_when_any_semantic_signal_is_strong():
    s = Settings(_env_file=None)
    # an English cross-encoder scoring a paraphrase low must not veto a good vector match
    assert passes_gate(_hit(), RetrievalSignals(vector_top=0.45, rerank_top=0.01), s)
    assert passes_gate(_hit(), RetrievalSignals(vector_top=0.10, rerank_top=0.40), s)
    assert not passes_gate(_hit(), RetrievalSignals(vector_top=0.10, rerank_top=0.01, lexical_top=0.5), s)
    assert passes_gate(_hit(), RetrievalSignals(overview=True), s)
    assert passes_gate(_hit(), RetrievalSignals(lexical_top=0.5), s)  # BM25-only mode


def test_overview_questions_are_detected_but_topic_summaries_are_not():
    assert is_overview_question("Summarise the document")
    assert is_overview_question("What is this report about?")
    assert is_overview_question("Give me the key takeaways in 3 bullets")
    assert is_overview_question("Summarize NN Group Framework", ["NN Group_Framework.pdf"])
    assert not is_overview_question("Summarise the dividend policy")
    assert not is_overview_question("How many days of sick leave do I get?")


def test_overview_sample_covers_every_section_of_every_document():
    chunks = [
        {"id": f"{d}:c{i}", "document_id": d, "chunk_idx": i, "heading": f"H{i // 2}"}
        for d in ("a", "b") for i in range(20)
    ]
    picked = overview_chunks(chunks, 12)
    assert len(picked) <= 12
    assert {c["document_id"] for c in picked} == {"a", "b"}
    assert len({(c["document_id"], c["heading"]) for c in picked}) == len(picked)


class _BatchJudge:
    configured = True

    def __init__(self, labels):
        self.labels, self.calls = labels, 0

    async def complete_json(self, system, user, **kw):
        self.calls += 1
        return {"verdicts": [{"id": i + 1, "label": l} for i, l in enumerate(self.labels)]}, ChatResult("", 1, 1, 0.0)


class _NoNLI:
    configured = False


CTX = {
    "s1": {"sentence_id": "s1", "document_name": "d", "text": "TOTAL DPS: €3.20 in 2023 and €3.88 in 2026."},
}


def test_insight_may_compute_numbers_but_facts_may_not():
    claims = [
        Claim("The total dividend rises from €3.20 to €3.88.", ["s1"], kind="fact"),
        Claim("That is an increase of about 21%.", ["s1"], kind="insight"),
        Claim("The dividend was €4.10 in 2027.", ["s1"], kind="fact"),
        Claim("The documents do not give a figure for 2027.", [], kind="gap"),
    ]
    judge = _BatchJudge(["supported", "supported"])
    v = asyncio.run(verify_claims(claims, context_by_id=CTX, nli=_NoNLI(), llm=judge, nli_pass=0.8, nli_fail=0.2))
    assert [x.ok for x in v] == [True, True, False, True]
    assert v[2].stage_decided == "rules"  # invented number: never reaches the judge
    assert judge.calls == 1  # every undecided segment is judged in ONE call


def test_insight_without_citation_is_dropped():
    v = asyncio.run(verify_claims(
        [Claim("This is a strong position.", [], kind="insight")],
        context_by_id=CTX, nli=_NoNLI(), llm=_BatchJudge(["supported"]), nli_pass=0.8, nli_fail=0.2,
    ))
    assert not v[0].ok


def test_segments_render_as_markdown_paragraphs_and_bullets():
    segs = parse_segments({"segments": [
        {"text": "Dividends grew steadily.", "kind": "insight", "format": "paragraph", "citations": ["s1"]},
        {"text": "- 2023: €3.20", "kind": "fact", "format": "bullet", "citations": ["s1"]},
        {"text": "2026: €3.88", "kind": "fact", "format": "bullet", "citations": ["s1"]},
        {"text": "Overall a positive trend.", "kind": "insight", "format": "paragraph", "citations": ["s1"]},
    ]})
    assert segs[1].text == "2023: €3.20" and segs[0].kind == "insight"
    out = render_answer([
        {"text": s.text, "citations": [1], "kind": s.kind, "format": s.format} for s in segs
    ])
    assert out == "Dividends grew steadily.[1]\n- 2023: €3.20[1]\n- 2026: €3.88[1]\n\nOverall a positive trend.[1]"


def test_sources_are_grouped_into_passages_with_section_titles():
    block = build_sources_block([
        {"sentence_id": "d:s2", "document_name": "r.pdf", "heading": "Dividends", "page": 4, "idx": 2, "text": "B"},
        {"sentence_id": "d:s1", "document_name": "r.pdf", "heading": "Dividends", "page": 4, "idx": 1, "text": "A"},
    ])
    assert block == '<<<SOURCE doc="r.pdf" section="Dividends" page=4>>>\n[d:s1] A\n[d:s2] B\n<<<END SOURCE>>>'


def test_recent_history_returns_the_latest_messages():
    repo = ConversationRepo(mongomock.MongoClient()["t"])
    cid = repo.create("o")
    for i in range(10):
        repo.add_message("o", cid, "user", f"m{i}")
        time.sleep(0.002)
    assert [m["text"] for m in repo.recent_history("o", cid, limit=4)] == ["m6", "m7", "m8", "m9"]
