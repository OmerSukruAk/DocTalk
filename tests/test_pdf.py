import fitz

from app.agent.ingestion import chunk, parse, segment


def _make_pdf(path):
    doc = fitz.open()
    for page_no in range(1, 4):
        page = doc.new_page()
        page.insert_text((72, 30), "ACME Internal - Confidential", fontsize=8)  # running header
        page.insert_text((72, 820), f"Page {page_no}", fontsize=8)  # page number
        y = 100
        if page_no == 1:
            page.insert_text((72, y), "Leave Policy", fontsize=20, fontname="hebo")
            y += 40
            page.insert_text((72, y), "Employees get 25 days of annual leave.", fontsize=11)
            y += 30
            page.insert_text((72, y), "Sick Leave", fontsize=15, fontname="hebo")
            y += 30
            page.insert_text((72, y), "Sick leave is 10 days per year.", fontsize=11)
        else:
            page.insert_text((72, y), f"Body text on page {page_no} continues here.", fontsize=11)
    doc.save(path)


def test_pdf_headings_running_headers_and_reading_order(tmp_path):
    p = tmp_path / "policy.pdf"
    _make_pdf(str(p))
    blocks = parse("policy.pdf", p.read_bytes())
    texts = [b.text for b in blocks]
    assert not any("Confidential" in t or t.startswith("Page ") for t in texts), texts
    sick = next(b for b in blocks if "Sick leave is 10" in b.text)
    assert sick.heading and sick.heading.endswith("Sick Leave")
    annual = next(b for b in blocks if "25 days" in b.text)
    assert annual.heading == "Leave Policy" and annual.page == 1 and annual.bbox
    sents = segment(blocks, "d_pdf")
    assert chunk(sents, "d_pdf", 256)
