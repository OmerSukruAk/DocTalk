from app.agent.ingestion import (
    _parse_markdown_like, _split_sentences, approx_tokens, chunk, embed_text, parse, segment,
)

DOC = "d_test"


def _chunks_for(md: str, tmp_path, max_tokens=256, **kw):
    p = tmp_path / "x.md"
    p.write_text(md, encoding="utf-8")
    sents = segment(parse(str(p)), DOC)
    return sents, chunk(sents, DOC, max_tokens, **kw)


def test_sentence_splitter_keeps_closing_quote_with_its_sentence():
    assert _split_sentences('He said "Stop." Then he left.') == ['He said "Stop."', "Then he left."]


def test_sentence_splitter_abbreviations_decimals_initials():
    out = _split_sentences("Dr. Novak approved a fee of 3.5 EUR. See Fig. 2 for details. J. Smith agreed.")
    assert out == ["Dr. Novak approved a fee of 3.5 EUR.", "See Fig. 2 for details.", "J. Smith agreed."]


def test_sentence_splitter_czech_abbreviation_and_lowercase_continuation():
    out = _split_sentences("Viz např. článek 5. Další věta následuje.")
    assert out == ["Viz např. článek 5.", "Další věta následuje."]


def test_runon_text_is_split_but_never_mid_word():
    words = " ".join(f"word{i}" for i in range(400))
    parts = _split_sentences(words)
    assert len(parts) > 1 and all(len(p) <= 700 for p in parts)
    assert " ".join(parts) == words


def test_chunks_never_cross_section_boundaries(tmp_path):
    md = "# A\n\n" + " ".join(f"Alpha sentence number {i} is here." for i in range(40)) + \
         "\n\n# B\n\n" + " ".join(f"Beta sentence number {i} is here." for i in range(40))
    sents, chunks = _chunks_for(md, tmp_path, max_tokens=100)
    by_id = {s.sentence_id: s for s in sents}
    for c in chunks:
        assert len({by_id[sid].heading for sid in c.sentence_ids}) == 1, "chunk spans two sections"


def test_tables_are_separate_from_prose_and_rows_stay_atomic(tmp_path):
    md = "# Exp\n\nIntro sentence here.\n\n| Cat | Limit |\n|---|---|\n| Hotel | 120 |\n| Meals | 40 |\n"
    sents, chunks = _chunks_for(md, tmp_path)
    assert any(c.is_table for c in chunks) and any(not c.is_table for c in chunks)
    rows = [s for s in sents if s.is_table_row]
    assert [r.text for r in rows] == ["Cat: Hotel | Limit: 120", "Cat: Meals | Limit: 40"]


def test_every_sentence_in_exactly_one_chunk(tmp_path):
    md = "# A\n\n" + " ".join(f"Sentence {i} of the policy applies." for i in range(60))
    sents, chunks = _chunks_for(md, tmp_path, max_tokens=80)
    ids = [sid for c in chunks for sid in c.sentence_ids]
    assert sorted(ids) == sorted(s.sentence_id for s in sents) and len(ids) == len(set(ids))
    assert all(s.chunk_id for s in sents)


def test_balanced_packing_avoids_tiny_tail(tmp_path):
    md = "# A\n\n" + " ".join(f"This is policy sentence number {i} with some words." for i in range(23))
    _, chunks = _chunks_for(md, tmp_path, max_tokens=100)
    sizes = [c.token_count for c in chunks]
    assert len(chunks) > 1 and min(sizes) >= 0.4 * max(sizes), sizes
    assert max(sizes) <= 100 * 1.25


def test_tiny_sections_are_merged_and_all_headings_kept(tmp_path):
    md = "# Doc\n\n## One\n\nShort fact one.\n\n## Two\n\nShort fact two.\n\n## Three\n\nShort fact three.\n"
    _, chunks = _chunks_for(md, tmp_path)
    assert len(chunks) == 1
    assert chunks[0].headings == ["Doc > One", "Doc > Two", "Doc > Three"]


def test_overlap_is_embedding_only_and_stays_inside_the_section(tmp_path):
    md = "# A\n\n" + " ".join(f"Unique sentence {i} about leave." for i in range(30))
    sents, chunks = _chunks_for(md, tmp_path, max_tokens=60, overlap_sentences=1)
    assert chunks[0].context_before == ""
    prev_last = sents[[s.sentence_id for s in sents].index(chunks[0].sentence_ids[-1])].text
    assert chunks[1].context_before == prev_last
    assert prev_last not in chunks[1].text  # cited sentences stay unique


def test_embed_text_carries_document_and_section_context(tmp_path):
    _, chunks = _chunks_for("# Leave Policy\n\nEmployees get 25 days.\n", tmp_path)
    e = embed_text(chunks[0], "hr.md")
    assert "Document: hr.md" in e and "Section: Leave Policy" in e and "25 days" in e


def test_markdown_lists_become_separate_sentences_and_markup_is_stripped(tmp_path):
    p = tmp_path / "l.md"
    p.write_text("# T\n\n- **Bold** item one\n- item two with [link](http://x.y)\n", encoding="utf-8")
    texts = [b.text for b in _parse_markdown_like(str(p))]
    assert texts == ["Bold item one", "item two with link"]


def test_token_estimate_is_monotonic():
    assert approx_tokens("a b c d e f") <= approx_tokens("a b c d e f g h i j k l")
