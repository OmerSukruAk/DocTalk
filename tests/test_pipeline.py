"""End-to-end through the REAL graph: ingestion -> hybrid retrieval -> gate -> generation ->
verification -> citations, with fakes only at the network edge."""
import asyncio

import pytest

from app.agent.workflow import answer_question

OWNER = "tid:alice"


def ask(services, cid, q, owner=OWNER):
    return asyncio.run(
        answer_question(services, services.graph, owner_key=owner, conversation_id=cid, question=q)
    )


@pytest.fixture()
def ready(services, hr_doc):
    cid = services.conversations.create(OWNER)
    info = asyncio.run(services.ingestion.ingest(OWNER, cid, "hr.md", hr_doc))
    return services, cid, info


def test_grounded_answer_has_verifiable_citation(ready):
    svc, cid, info = ready
    r = ask(svc, cid, "How many sick leave days do I get?")
    assert r["status"] == "ok"
    assert "10 days" in r["segments"][0]["text"]
    cit = r["citations"][0]
    assert cit["document_name"] == "hr.md" and cit["location"]["heading"] == "HR Policy > Sick Leave"
    stored = svc.sentences.get(OWNER, cit["sentence_id"])
    assert stored["text"] == cit["text"]  # citation text is the stored sentence, never model output


def test_metrics_break_time_down_by_stage_and_count_the_verifier(ready):
    svc, cid, _ = ready
    r = ask(svc, cid, "How many sick leave days do I get?")
    m = r["metrics"]
    t = m["timings_ms"]
    assert set(t) == {"rewrite", "translate", "retrieve", "generate", "verify", "other"}
    assert t["retrieve"] > 0 and t["generate"] > 0 and t["verify"] > 0
    assert abs(sum(t.values()) - m["total_ms"]) <= 1  # the stages account for the whole request
    # answer call (50 in / 20 out in the fake) + verifier judge call (5 / 5) are both counted
    assert m["tokens_in"] == 55 and m["tokens_out"] == 25
    stats = svc.logs.stats()["stages"]
    assert stats["retrieve"]["runs"] == 1 and stats["verify"]["avg_ms"] > 0


def test_stage_time_adds_up_when_a_stage_runs_twice(ready):
    svc, cid, _ = ready
    q = "Yıllık hastalık izni kaç gün?"  # refused once, translated, retrieved again
    svc.llm.translations[q] = "How many sick leave days per year?"
    r = ask(svc, cid, q)
    log = svc.logs._c.find_one({"question": q})
    assert r["metrics"]["timings_ms"]["translate"] > 0
    assert log["node_ms"]["retrieve"] == r["metrics"]["timings_ms"]["retrieve"]


def test_table_row_is_answerable(ready):
    svc, cid, _ = ready
    r = ask(svc, cid, "What is the hotel limit per night?")
    assert r["status"] == "ok" and "120" in r["segments"][0]["text"]


def test_offtopic_question_is_stopped_by_the_gate_with_zero_llm_tokens(ready):
    svc, cid, _ = ready
    before = len(svc.llm.calls)
    r = ask(svc, cid, "What is the capital of France?")
    assert r["status"] == "refused" and "no relevant content" in r["refusal_reason"]
    assert r["metrics"]["tokens_in"] == 0 and len(svc.llm.calls) == before


def test_foreign_language_question_is_translated_once_then_answered(ready):
    svc, cid, _ = ready
    q = "Yıllık hastalık izni kaç gün?"  # Turkish: shares no keyword with the English policy
    svc.llm.translations[q] = "How many sick leave days per year?"
    r = ask(svc, cid, q)
    assert r["status"] == "ok" and any("Sick leave is 10 days" in c["text"] for c in r["citations"])


def test_foreign_offtopic_question_is_still_refused(ready):
    svc, cid, _ = ready
    r = ask(svc, cid, "Fransa'nın başkenti neresi?")  # translation (identity here) finds nothing
    assert r["status"] == "refused"


def test_overview_question_answers_from_the_whole_document(ready):
    svc, cid, _ = ready
    r = ask(svc, cid, "Summarise the document")
    assert r["status"] != "refused" or r["refusal_reason"] != "no relevant content found in your documents"
    generate = [u for kind, u in svc.llm.calls if kind == "json" and "User's message:" in u][-1]
    for section in ("Annual Leave", "Sick Leave", "Remote Work", "Expenses"):
        assert section in generate  # every section reached the model, not one similarity hit


def test_hallucinated_claim_is_dropped_and_answer_is_partial(ready):
    svc, cid, _ = ready
    sid = ask(svc, cid, "sick leave days")["citations"][0]["sentence_id"]
    svc.llm.extra_claims = [{"text": "Sick leave is 99 days per year.", "citations": [sid]}]
    r = ask(svc, cid, "How many sick leave days do I get?")
    assert r["status"] == "partial" and r["dropped_claims"] == 1
    assert all("99" not in s["text"] for s in r["segments"])
    # the removed statement and why it was removed are reported, but never as part of the answer
    removed = r["diagnostics"]["removed"]
    assert [d["text"] for d in removed] == ["Sick leave is 99 days per year."]
    assert removed[0]["reasons"] and removed[0]["checked_by"]
    assert r["diagnostics"]["attempts"] == 1


def test_invented_citation_id_is_rejected(ready):
    svc, cid, _ = ready
    svc.llm.extra_claims = [{"text": "Sick leave is 10 days.", "citations": ["d_fake:s999"]}]
    r = ask(svc, cid, "sick leave days")
    assert r["dropped_claims"] == 1


def test_followup_is_generated_against_the_rewritten_question(ready):
    svc, cid, _ = ready
    ask(svc, cid, "How many sick leave days do I get?")
    r = ask(svc, cid, "And remote work?")
    assert r["rewritten_question"] == "standalone: And remote work?"
    last_generate = [u for kind, u in svc.llm.calls if kind == "json" and "User's message:" in u][-1]
    # the model sees the user's own words (tone instructions live there) AND the resolved question
    assert "User's message: And remote work?" in last_generate
    assert "(Standalone version: standalone: And remote work?)" in last_generate
    # ...and the conversation so far, so 'make it shorter' / 'and X?' have context
    assert "How many sick leave days do I get?" in last_generate.split("Sources:")[0]


def test_llm_outage_returns_explicit_error_and_persists_both_messages(ready):
    svc, cid, _ = ready
    svc.llm.fail = True
    r = ask(svc, cid, "How many sick leave days do I get?")
    assert r["status"] == "error" and r["segments"] == []
    roles = [m["role"] for m in svc.conversations.list_messages(OWNER, cid)]
    assert roles == ["user", "assistant"]
    assert svc.logs.stats()["error_rate"] == 1.0


def test_users_are_isolated(ready):
    svc, _, _ = ready
    other = svc.conversations.create("tid:bob")
    r = ask(svc, other, "How many sick leave days do I get?", owner="tid:bob")
    assert r["status"] == "refused" and r["citations"] == []


def test_bm25_only_degraded_mode_still_answers_and_gates(settings, services, hr_doc):
    class Off:
        configured = False

    services.embeddings = Off()
    services.ingestion._embeddings = Off()
    cid = services.conversations.create("tid:carol")
    asyncio.run(services.ingestion.ingest("tid:carol", cid, "hr.md", hr_doc))
    ok = ask(services, cid, "sick leave days", owner="tid:carol")
    assert ok["status"] == "ok"
    generations = sum(1 for k, u in services.llm.calls if k == "json" and "Question:" in u)
    no = ask(services, cid, "What is the capital of France?", owner="tid:carol")
    assert no["status"] == "refused"
    # the gate stopped it before generation (only the cheap follow-up rewrite may have run)
    assert sum(1 for k, u in services.llm.calls if k == "json" and "Question:" in u) == generations


def test_reupload_is_idempotent_and_limit_is_enforced(services, hr_doc):
    from app.agent.ingestion import IngestionError

    cid = services.conversations.create(OWNER)
    a = asyncio.run(services.ingestion.ingest(OWNER, cid, "hr.md", hr_doc))
    b = asyncio.run(services.ingestion.ingest(OWNER, cid, "hr-copy.md", hr_doc))
    assert a["id"] == b["id"]
    for i in range(4):
        asyncio.run(services.ingestion.ingest(OWNER, cid, f"d{i}.md", f"# T{i}\n\nUnique text number {i}.".encode()))
    with pytest.raises(IngestionError, match="limit"):
        asyncio.run(services.ingestion.ingest(OWNER, cid, "d9.md", b"# X\n\nOne more document."))


def test_corrupt_pdf_is_a_clean_user_error(services):
    from app.agent.ingestion import IngestionError

    cid = services.conversations.create(OWNER)
    with pytest.raises(IngestionError, match="could not parse"):
        asyncio.run(services.ingestion.ingest(OWNER, cid, "bad.pdf", b"%PDF-1.4 not really"))
    assert services.documents.count(OWNER, cid) == 0  # rolled back


def test_bm25_picks_up_documents_added_by_another_replica(services, hr_doc):
    """Simulates replica B: the chunk set changes without THIS process calling invalidate()."""
    cid = services.conversations.create(OWNER)
    asyncio.run(services.ingestion.ingest(OWNER, cid, "hr.md", hr_doc))
    assert services.bm25.search(OWNER, cid, False, "sick leave", 3)
    assert not services.bm25.search(OWNER, cid, False, "onboarding buddy", 3)
    other_replica_bm25 = services.bm25
    services.bm25 = type(other_replica_bm25)(services.chunks)  # fresh cache, as on another pod
    asyncio.run(services.ingestion.ingest(OWNER, cid, "ob.md", b"# Onboarding\n\nEvery new hire gets an onboarding buddy."))
    # the ORIGINAL replica never saw invalidate(); the fingerprint must still refresh it
    assert other_replica_bm25.search(OWNER, cid, False, "onboarding buddy", 3)
