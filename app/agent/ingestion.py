"""Ingestion: file -> ParsedBlock -> Sentence -> Chunk -> MongoDB + Qdrant, atomically.

PDF is parsed with PyMuPDF (page + bbox, reading order, heading detection from font
metrics, running headers/footers removed, table rows kept atomic), DOCX/Markdown with
MarkItDown (heading path, list items as their own blocks). Segmentation is EN/CS aware
and gives every sentence a stable id ("<document_id>:s<n>") that citations refer to.

Chunking (see ``chunk``) is structure-aware: chunks never cross a section boundary,
tables never mix with prose, tiny sections are merged, big sections are packed into
balanced chunks, and every chunk is embedded together with its document/section
context plus a one-sentence overlap with the previous chunk."""

from __future__ import annotations
import asyncio
import hashlib
import math
import os
import re
import statistics
import uuid
from dataclasses import dataclass, field
from app.config import Settings
from app.domain import Chunk, Sentence, chunk_id, sentence_id


# --- parsers ---

SUPPORTED_EXTS = {".pdf", ".docx", ".md", ".markdown"}


@dataclass
class ParsedBlock:
    text: str
    page: int | None
    heading: str | None
    bbox: list[float] | None
    is_table_row: bool = False
    meta: dict = field(default_factory=dict)


def parse(filename: str, content: bytes) -> list[ParsedBlock]:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return _parse_pdf(content)
    if ext in {".docx", ".md", ".markdown"}:
        return _parse_markdown_like(ext, content)
    raise ValueError(f"Unsupported file type: {ext}")


# --------------------------------------------------------------------------- PDF
_SENTENCE_PUNCT = (".", "!", "?", "…", ":", ";", ",")


def _parse_pdf(content: bytes) -> list[ParsedBlock]:
    import fitz  # PyMuPDF

    raw: list[dict] = []  # one entry per text block, in reading order, across all pages
    page_heights: dict[int, float] = {}
    with fitz.open(stream=content, filetype="pdf") as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            page_no = page_index + 1
            page_heights[page_no] = float(page.rect.height)
            tables = _extract_tables(page, page_no)
            table_rects = [rect for rect, _rows in tables]

            text_blocks: list[dict] = []
            for b in page.get_text("dict", sort=True).get("blocks", []):
                if b.get("type") != 0:  # skip images
                    continue
                lines = []
                sizes: list[tuple[float, int]] = []
                bold_chars = total_chars = 0
                for line in b.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = "".join(s.get("text", "") for s in spans)
                    if line_text.strip():
                        lines.append(line_text.strip())
                    for s in spans:
                        n = len(s.get("text", "").strip())
                        if n:
                            sizes.append((round(float(s.get("size", 0)), 1), n))
                            total_chars += n
                            if int(s.get("flags", 0)) & 16 or "bold" in str(s.get("font", "")).lower():
                                bold_chars += n
                text = _join_pdf_lines(lines)
                if not text:
                    continue
                x0, y0, x1, y1 = b["bbox"]
                rect = fitz.Rect(x0, y0, x1, y1)
                if _mostly_inside(rect, table_rects):
                    continue  # already captured as table rows
                size = max(sizes, key=lambda s: s[1])[0] if sizes else 0.0
                text_blocks.append(
                    {
                        "text": text,
                        "page": page_no,
                        "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                        "size": size,
                        "bold": total_chars > 0 and bold_chars / total_chars >= 0.8,
                        "n_lines": len(lines),
                        "chars": total_chars,
                        "table": False,
                    }
                )

            text_blocks = _reading_order(text_blocks)

            # Merge table rows back into reading order (by the table's top edge).
            merged: list[dict] = []
            pending = sorted(tables, key=lambda t: t[0].y0)
            for blk in text_blocks:
                while pending and pending[0][0].y0 <= blk["bbox"][1]:
                    merged.extend(_table_entries(pending.pop(0)[1]))
                merged.append(blk)
            for _rect, rows in pending:
                merged.extend(_table_entries(rows))
            raw.extend(merged)

    raw = _drop_running_headers(raw, page_heights)
    return _assign_pdf_headings(raw)


def _table_entries(rows: list[ParsedBlock]) -> list[dict]:
    return [
        {
            "text": r.text, "page": r.page, "bbox": r.bbox, "size": 0.0, "bold": False,
            "n_lines": 1, "chars": len(r.text), "table": True,
        }
        for r in rows
    ]


def _mostly_inside(rect, regions: list) -> bool:
    """True when at least half of ``rect`` lies inside one of ``regions`` (a paragraph that
    merely touches a table edge is kept; text that belongs to the table is dropped)."""
    area = rect.get_area() or 1.0
    return any((rect & region).get_area() >= 0.5 * area for region in regions if rect.intersects(region))


def _reading_order(blocks: list[dict]) -> list[dict]:
    """Fix reading order for side-by-side layouts (KPI cards, two-column boxes).

    PyMuPDF sorts blocks by y then x, which interleaves columns: 'LABEL A LABEL B value A
    value B'. Runs of narrow blocks are split into columns (by horizontal overlap) and read
    column by column; each block gets a ``col`` key so ``_assign_pdf_headings`` can glue the
    pieces of one card back into a single citable block. Single-column text is untouched."""
    if len(blocks) < 2:
        return blocks
    left = min(b["bbox"][0] for b in blocks)
    right = max(b["bbox"][2] for b in blocks)
    narrow_max = 0.5 * max(1.0, right - left)

    out: list[dict] = []
    run: list[dict] = []

    def flush(run_no: int) -> None:
        columns: list[list[dict]] = []
        for b in run:
            for col in columns:
                cx0 = min(c["bbox"][0] for c in col)
                cx1 = max(c["bbox"][2] for c in col)
                overlap = min(cx1, b["bbox"][2]) - max(cx0, b["bbox"][0])
                if overlap >= 0.5 * min(cx1 - cx0, b["bbox"][2] - b["bbox"][0]):
                    col.append(b)
                    break
            else:
                columns.append([b])
        if len(columns) < 2:
            out.extend(run)
            return
        columns.sort(key=lambda col: min(c["bbox"][0] for c in col))
        for ci, col in enumerate(columns):
            for b in sorted(col, key=lambda c: c["bbox"][1]):
                out.append({**b, "col": (b["page"], run_no, ci)})

    narrow = [b for b in blocks if b["bbox"][2] - b["bbox"][0] < narrow_max]

    def beside_another(b: dict) -> bool:
        """Another narrow block shares its vertical band without overlapping horizontally:
        that is what a column layout looks like (a lone short heading is not one)."""
        x0, y0, x1, y1 = b["bbox"]
        return any(
            o is not b and o["bbox"][1] < y1 - 1 and o["bbox"][3] > y0 + 1
            and (o["bbox"][2] <= x0 or o["bbox"][0] >= x1)
            for o in narrow
        )

    for i, b in enumerate(blocks):
        if b["bbox"][2] - b["bbox"][0] < narrow_max and beside_another(b):
            run.append(b)
            continue
        if run:
            flush(i)
            run = []
        out.append(b)
    if run:
        flush(len(blocks))
    return out


def _is_textual(text: str) -> bool:
    """Headings are words, not numbers: '€1.38' or '197%' in a big KPI font is not a section."""
    chars = [c for c in text if not c.isspace()]
    letters = sum(c.isalpha() for c in chars)
    return letters >= 3 and letters >= 0.5 * len(chars)


def _join_pdf_lines(lines: list[str]) -> str:
    """Re-flow wrapped PDF lines: undo end-of-line hyphenation, then join with spaces."""
    out = ""
    for line in lines:
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}" if out else line
    return re.sub(r"\s+", " ", out).strip()


def _norm_running(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _drop_running_headers(raw: list[dict], page_heights: dict[int, float]) -> list[dict]:
    """Remove page numbers and running headers/footers (same text in the top/bottom margin
    of at least half the pages) so they never become citations or pollute chunks."""
    n_pages = len(page_heights)
    margin_hits: dict[str, set[int]] = {}
    for blk in raw:
        if blk["table"]:
            continue
        h = page_heights.get(blk["page"], 0) or 1
        y0, y1 = blk["bbox"][1], blk["bbox"][3]
        if y1 <= 0.09 * h or y0 >= 0.91 * h:
            margin_hits.setdefault(_norm_running(blk["text"]), set()).add(blk["page"])
    running = {k for k, pages in margin_hits.items() if n_pages >= 3 and len(pages) >= max(2, n_pages // 2)}
    kept = []
    for blk in raw:
        if not blk["table"]:
            h = page_heights.get(blk["page"], 0) or 1
            in_margin = blk["bbox"][3] <= 0.09 * h or blk["bbox"][1] >= 0.91 * h
            norm = _norm_running(blk["text"])
            if in_margin and (norm in running or re.fullmatch(r"(page |strana |str\. )?#( ?(/|of|z) ?#)?", norm)):
                continue
        kept.append(blk)
    return kept


def _assign_pdf_headings(raw: list[dict]) -> list[ParsedBlock]:
    """Detect headings from font metrics (larger than the body size, or short bold lines)
    and attach the running heading path to every following block."""
    prose = [b for b in raw if not b["table"] and b["size"] > 0]
    if not prose:
        return [
            ParsedBlock(b["text"], b["page"], None, b["bbox"], is_table_row=b["table"]) for b in raw
        ]
    weight: dict[float, int] = {}
    for b in prose:
        weight[b["size"]] = weight.get(b["size"], 0) + b["chars"]
    body = max(weight, key=weight.get)

    def is_heading(b: dict) -> bool:
        if b["table"] or b.get("merged") or b["n_lines"] > 3 or len(b["text"]) > 120:
            return False
        if b["text"].endswith(_SENTENCE_PUNCT) or not _is_textual(b["text"]):
            return False
        return b["size"] >= body * 1.15 or (b["bold"] and b["size"] >= body * 0.98 and len(b["text"]) > 2)

    # Glue the pieces of one card / column box ('OPERATING RESULT' + '€1.44B' + '+8.6% Y-o-Y')
    # into one block, so the value is cited together with its label.
    glued: list[dict] = []
    for b in raw:
        prev = glued[-1] if glued else None
        if (
            prev is not None
            and b.get("col") is not None
            and prev.get("col") == b.get("col")
            and b["bbox"][1] - prev["bbox"][3] <= max(8.0, 1.2 * max(prev["size"], b["size"]))
            and not is_heading(prev)
            and not is_heading(b)
        ):
            glued[-1] = {
                **prev,
                "text": f"{prev['text']} {b['text']}",
                "bbox": [
                    min(prev["bbox"][0], b["bbox"][0]), prev["bbox"][1],
                    max(prev["bbox"][2], b["bbox"][2]), b["bbox"][3],
                ],
                "n_lines": prev["n_lines"] + b["n_lines"],
                "chars": prev["chars"] + b["chars"],
                "merged": True,
            }
            continue
        glued.append(b)
    raw = glued

    heading_sizes = sorted({b["size"] for b in prose if is_heading(b) and b["size"] >= body * 1.15}, reverse=True)

    def level(b: dict) -> int:
        return heading_sizes.index(b["size"]) + 1 if b["size"] in heading_sizes else len(heading_sizes) + 1

    stack: list[tuple[int, str]] = []
    blocks: list[ParsedBlock] = []
    for b in raw:
        if is_heading(b):
            lvl = level(b)
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, b["text"]))
            continue
        heading = " > ".join(h for _, h in stack) or None
        blocks.append(
            ParsedBlock(
                text=b["text"], page=b["page"], heading=heading, bbox=b["bbox"], is_table_row=b["table"]
            )
        )
    return blocks


def _extract_tables(page, page_no: int) -> list[tuple[object, list[ParsedBlock]]]:
    """Best-effort table extraction: every data row becomes one atomic block with its column
    headers attached ('FINANCIAL YEAR: 2024 Actual | TOTAL DPS: €3.44'). Returns
    ``[(table_region, row_blocks)]``.

    PyMuPDF's detector often sees a zebra-striped table as several one-row 'tables' (only the
    shaded rows have filled rectangles) and misses the unshaded rows between them. Pieces that
    share the same columns are therefore merged back into one table, and the rows between or
    right after them are rebuilt from word positions using the detected column boundaries."""
    import fitz

    try:
        found = list(getattr(page.find_tables(), "tables", []) or [])
    except Exception:
        return []
    words = page.get_text("words")  # (x0, y0, x1, y1, text, block, line, word_no)
    out: list[tuple[object, list[ParsedBlock]]] = []
    for group in _group_split_tables(found):
        columns = _table_columns(group)
        if len(columns) < 2:
            continue
        bands = _table_bands(group, columns, words)
        rows = [cells for cells in (_cells_in_band(band, columns, words) for band in bands) if any(cells)]
        if not rows:
            continue
        x0, x1 = columns[0][0], columns[-1][1]
        region = fitz.Rect(x0, bands[0][0], x1, bands[-1][1])
        header = rows[0] if len(rows) > 1 and _looks_like_header(rows[0]) else None
        body = rows[1:] if header else rows
        blocks: list[ParsedBlock] = []
        for cells, band in zip(body, bands[1:] if header else bands):
            parts = []
            for i, cell in enumerate(cells):
                if not cell:
                    continue
                label = header[i] if header and i < len(header) else ""
                parts.append(f"{label}: {cell}" if label else cell)
            if parts:
                blocks.append(
                    ParsedBlock(
                        text=" | ".join(parts),
                        page=page_no,
                        heading=None,
                        bbox=[round(x0, 1), round(band[0], 1), round(x1, 1), round(band[1], 1)],
                        is_table_row=True,
                    )
                )
        if blocks:
            out.append((region, blocks))
    return out


def _group_split_tables(tables: list) -> list[list]:
    """Merge vertically stacked table pieces that have the same columns."""
    groups: list[list] = []
    for t in sorted(tables, key=lambda t: t.bbox[1]):
        if groups:
            last = groups[-1][-1]
            heights = [r.bbox[3] - r.bbox[1] for g in groups[-1] for r in g.rows] or [20.0]
            if (
                t.col_count == last.col_count
                and abs(t.bbox[0] - last.bbox[0]) <= 4
                and abs(t.bbox[2] - last.bbox[2]) <= 4
                and t.bbox[1] - last.bbox[3] <= 2.5 * max(heights)
            ):
                groups[-1].append(t)
                continue
        groups.append([t])
    return groups


def _table_columns(group: list) -> list[tuple[float, float]]:
    """Column x-ranges, taken from the first row of the group that has every cell."""
    for t in group:
        for row in t.rows:
            cells = row.cells
            if cells and all(c is not None for c in cells):
                return [(c[0], c[2]) for c in cells]
    t = group[0]
    return [(c[0], c[2]) for c in (t.rows[0].cells if t.rows else []) if c is not None]


def _words_in(words, y0: float, y1: float, x0: float, x1: float) -> list:
    return [
        w for w in words
        if y0 <= (w[1] + w[3]) / 2 < y1 and x0 <= (w[0] + w[2]) / 2 <= x1
    ]


def _table_bands(group: list, columns: list[tuple[float, float]], words) -> list[tuple[float, float]]:
    """Row bands (y0, y1): the detected rows, the gaps between them that contain text in at
    least two columns (unshaded rows), and the same for the rows right after the last piece."""
    x0, x1 = columns[0][0], columns[-1][1]

    def column_of(x: float) -> int | None:
        return next((i for i, (c0, c1) in enumerate(columns) if c0 <= x <= c1), None)

    def is_row(y0: float, y1: float) -> bool:
        hit = _words_in(words, y0, y1, x0, x1)
        lines: dict[tuple, set] = {}
        for w in hit:
            lines.setdefault((w[5], w[6]), set()).add(column_of((w[0] + w[2]) / 2))
        if any(len(cols) > 1 for cols in lines.values()):
            return False  # a text line running across columns is prose below the table
        return len(set().union(*lines.values()) - {None}) >= 2 if lines else False

    detected = sorted({(r.bbox[1], r.bbox[3]) for t in group for r in t.rows})
    bands: list[tuple[float, float]] = []
    for y0, y1 in detected:
        if bands and y0 - bands[-1][1] > 3 and is_row(bands[-1][1], y0):
            bands.append((bands[-1][1], y0))
        bands.append((y0, y1))
    height = statistics.median(y1 - y0 for y0, y1 in detected) if detected else 20.0
    for _ in range(50):  # unshaded rows after the last detected piece
        y0 = bands[-1][1]
        if not is_row(y0, y0 + height * 1.25):
            break
        bands.append((y0, y0 + height * 1.25))
    return bands


def _cells_in_band(band: tuple[float, float], columns: list[tuple[float, float]], words) -> list[str]:
    cells: list[str] = []
    for c0, c1 in columns:
        hit = sorted(_words_in(words, band[0], band[1], c0, c1), key=lambda w: (round(w[1]), w[0]))
        text = ""
        for w in hit:
            token = w[4]
            if text.endswith("-") and token[:1].isalpha():
                text += token  # 'Non-' + 'Life' -> 'Non-Life'
            else:
                text = f"{text} {token}" if text else token
        cells.append(re.sub(r"([€$£¥])\s+(?=\d)", r"", text).strip())
    return cells


def _looks_like_header(cells: list[str]) -> bool:
    """A header row labels columns: mostly upper-case or bold-looking words, no money/percent
    values ('2028 TARGET' is a header, '€1.28' is data)."""
    filled = [c for c in cells if c]
    if len(filled) < 2:
        return False
    if any(re.search(r"[€$£¥%]", c) or re.fullmatch(r"[+-]?[\d.,]+\s*[A-Za-z]{0,3}", c) for c in filled):
        return False
    letters = [ch for c in filled for ch in c if ch.isalpha()]
    upper = sum(ch.isupper() for ch in letters)
    return bool(letters) and (upper / len(letters) >= 0.6 or all(len(c) <= 30 for c in filled))


# ------------------------------------------------------------------ DOCX / MD
_LIST_ITEM = re.compile(r"^(?:[-*+•]|\d{1,3}[.)])\s+(.*)$")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_EMPH = re.compile(r"(\*\*|__)(.+?)\1")
_MD_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")
_MD_CODE = re.compile(r"`([^`]+)`")


def _strip_inline_md(text: str) -> str:
    """Drop inline Markdown markup so citations show clean text, not '**bold**'."""
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_EMPH.sub(r"\2", text)
    text = _MD_ITALIC.sub(r"\1", text)
    return _MD_CODE.sub(r"\1", text)


def _parse_markdown_like(ext: str, content: bytes) -> list[ParsedBlock]:
    text = _to_markdown(ext, content)
    blocks: list[ParsedBlock] = []
    heading_stack: list[str] = []
    buffer: list[str] = []
    table_buffer: list[str] = []
    in_fence = False

    def current_heading() -> str | None:
        return " > ".join(heading_stack) if heading_stack else None

    def flush_paragraph() -> None:
        if buffer:
            para = _strip_inline_md(" ".join(line.strip() for line in buffer).strip())
            if para:
                blocks.append(ParsedBlock(text=para, page=None, heading=current_heading(), bbox=None))
            buffer.clear()

    def flush_table() -> None:
        nonlocal table_buffer
        if not table_buffer:
            return
        header_cells: list[str] = []
        for i, raw in enumerate(table_buffer):
            cells = [_strip_inline_md(c.strip()) for c in raw.strip().strip("|").split("|")]
            if i == 0:
                header_cells = cells
                continue
            if all(set(c) <= {"-", ":", " "} for c in cells):
                continue  # markdown separator row
            parts = [
                f"{header_cells[j]}: {cells[j]}" if j < len(header_cells) and header_cells[j] else cells[j]
                for j in range(len(cells))
                if cells[j]
            ]
            row_text = " | ".join(parts)
            if row_text:
                blocks.append(
                    ParsedBlock(
                        text=row_text,
                        page=None,
                        heading=current_heading(),
                        bbox=None,
                        is_table_row=True,
                    )
                )
        table_buffer = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            flush_table()
            in_fence = not in_fence
            continue
        if in_fence:
            continue  # code fences are not prose; skip rather than cite them as sentences
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            table_buffer.append(stripped)
            continue
        flush_table()
        if stripped.startswith("#"):
            flush_paragraph()
            level = len(stripped) - len(stripped.lstrip("#"))
            title = _strip_inline_md(stripped[level:].strip())
            while len(heading_stack) >= level:
                heading_stack.pop()
            if title:
                heading_stack.append(title)
            continue
        if not stripped:
            flush_paragraph()
            continue
        item = _LIST_ITEM.match(stripped)
        if item:  # every bullet is its own citable unit, not glued into a run-on paragraph
            flush_paragraph()
            buffer.append(item.group(1))
            continue
        buffer.append(stripped)
    flush_table()
    flush_paragraph()
    return blocks


def _to_markdown(ext: str, content: bytes) -> str:
    if ext in {".md", ".markdown"}:
        return content.decode("utf-8", errors="replace")
    import io

    from markitdown import MarkItDown, StreamInfo

    result = MarkItDown().convert(io.BytesIO(content), stream_info=StreamInfo(extension=ext))
    return getattr(result, "text_content", "") or ""


# --- segmenter ---

# Abbreviations that end with a period but should NOT end a sentence (EN + CS).
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "no", "fig", "al", "inc", "ltd", "co", "approx", "min", "max", "p", "pp", "cf",
    "např", "tj", "tzn", "tzv", "apod", "atd", "mj", "resp", "č", "čl", "str", "obr", "tab",
}
# A sentence boundary candidate: terminal punctuation, optional CLOSING quotes/brackets
# (which belong to the sentence that just ended), then whitespace. Whether the next
# sentence really starts here is decided in code (uppercase / digit check, abbreviations).
_SENT_END = re.compile(r"(?<=[.!?…])(?P<close>[\"'”’»)\]]*)\s+")
_OPENERS = "„“\"'«([¿¡"
MAX_SENTENCE_CHARS = 700  # hard cap so run-on text (no punctuation) stays citable


def segment(blocks: list[ParsedBlock], document_id: str) -> list[Sentence]:
    sentences: list[Sentence] = []
    idx = 0
    for block in blocks:
        norm = re.sub(r"\s+", " ", block.text).strip()
        if not norm:
            continue
        pieces = [norm] if block.is_table_row else _split_sentences(norm)
        cursor = 0
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            start = norm.find(piece, cursor)
            if start == -1:
                start = cursor
            end = start + len(piece)
            cursor = end
            sentences.append(
                Sentence(
                    sentence_id=sentence_id(document_id, idx),
                    document_id=document_id,
                    idx=idx,
                    text=piece,
                    page=block.page,
                    heading=block.heading,
                    bbox=block.bbox,
                    char_start=start,
                    char_end=end,
                    is_table_row=block.is_table_row,
                )
            )
            idx += 1
    return sentences


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts: list[str] = []
    last = 0
    for match in _SENT_END.finditer(text):
        punct_pos = match.start() - 1
        candidate = text[last : match.start() + len(match.group("close"))].strip()
        nxt = text[match.end():].lstrip(_OPENERS)[:1]
        if not nxt or not (nxt.isupper() or nxt.isdigit()):
            continue  # next word is lowercase -> not a sentence start
        if text[punct_pos] == ".":
            token = text[last : match.start()].rsplit(" ", 1)[-1].rstrip(".").strip(_OPENERS).lower()
            if token in _ABBREVIATIONS:
                continue
            if len(token) == 1 and token.isalpha():  # initials: "J. Smith"
                continue
            if token.isdigit() and len(token) <= 2 and len(candidate.split()) <= 1:
                continue  # list numbering: "1. Scope"
        if candidate:
            parts.append(candidate)
            last = match.end()
    tail = text[last:].strip()
    if tail:
        parts.append(tail)
    out: list[str] = []
    for part in parts or [text]:
        out.extend(_split_long(part))
    return out


def _split_long(piece: str, max_chars: int = MAX_SENTENCE_CHARS) -> list[str]:
    """Break a run-on 'sentence' at ; / : / , and finally at spaces, never mid-word."""
    if len(piece) <= max_chars:
        return [piece]
    units = [piece]
    for pattern in (r"(?<=[;:])\s+", r"(?<=,)\s+", r"\s+"):
        nxt: list[str] = []
        for u in units:
            nxt.extend(re.split(pattern, u) if len(u) > max_chars else [u])
        units = nxt
    packed: list[str] = []
    current = ""
    for u in units:
        if current and len(current) + 1 + len(u) > max_chars:
            packed.append(current)
            current = u
        else:
            current = f"{current} {u}".strip()
    if current:
        packed.append(current)
    return packed


# --- chunker (structure-aware parent passages) ---

def approx_tokens(text: str) -> int:
    """Cheap token estimate: the larger of chars/4 and words*1.3 (the latter is closer for
    Czech and for number-heavy text, where chars/4 under-counts)."""
    return max(1, int(max(len(text) / 4, len(text.split()) * 1.3)))


@dataclass
class _Segment:
    sentences: list[Sentence]
    is_table: bool
    tokens: int

    @property
    def headings(self) -> list[str]:
        seen: list[str] = []
        for s in self.sentences:
            if s.heading and s.heading not in seen:
                seen.append(s.heading)
        return seen


def _segments(sentences: list[Sentence]) -> list[_Segment]:
    """Split into runs that share the same section heading and the same kind (prose/table)."""
    segs: list[_Segment] = []
    for s in sentences:
        t = approx_tokens(s.text)
        if segs:
            prev = segs[-1].sentences[-1]
            if prev.heading == s.heading and prev.is_table_row == s.is_table_row:
                segs[-1].sentences.append(s)
                segs[-1].tokens += t
                continue
        segs.append(_Segment([s], s.is_table_row, t))
    return segs


def _parent(heading: str | None) -> str:
    """'Policy > Leave > Sick' -> 'Policy > Leave' (the section the heading lives in)."""
    return heading.rsplit(" > ", 1)[0] if heading and " > " in heading else ""


def _merge_small(segs: list[_Segment], min_tokens: int, max_tokens: int) -> list[_Segment]:
    """Very small prose sections (under half of ``min_tokens``, i.e. a sentence or two) give
    weak embeddings, so fold them into the adjacent prose section - but only a SIBLING under
    the same parent heading, and only while the result still fits one chunk. Unrelated
    sections are never blended (that would blur retrieval), and tables are never merged."""
    out: list[_Segment] = []
    tiny = max(1, min_tokens // 2)
    for seg in segs:
        if out:
            prev = out[-1]
            if (
                not prev.is_table
                and not seg.is_table
                and (prev.tokens < tiny or seg.tokens < tiny)
                and prev.tokens + seg.tokens <= max_tokens
                and _parent(prev.sentences[-1].heading) == _parent(seg.sentences[0].heading)
            ):
                prev.sentences.extend(seg.sentences)
                prev.tokens += seg.tokens
                continue
        out.append(_Segment(list(seg.sentences), seg.is_table, seg.tokens))
    return out


def _pack(seg: _Segment, max_tokens: int, min_tokens: int) -> list[list[Sentence]]:
    """Pack a segment into BALANCED chunks (no 500-token chunk followed by a 20-token tail)."""
    if seg.tokens <= max_tokens:
        return [seg.sentences]
    n = math.ceil(seg.tokens / max_tokens)
    limit = min(max_tokens, math.ceil(seg.tokens / n * 1.15))
    groups: list[list[Sentence]] = []
    current: list[Sentence] = []
    used = 0
    for s in seg.sentences:
        t = approx_tokens(s.text)
        if current and used + t > limit:
            groups.append(current)
            current, used = [], 0
        current.append(s)
        used += t
    if current:
        groups.append(current)
    if len(groups) > 1:  # fold an undersized tail back into its predecessor
        tail_t = sum(approx_tokens(s.text) for s in groups[-1])
        prev_t = sum(approx_tokens(s.text) for s in groups[-2])
        if tail_t < min_tokens // 2 and tail_t + prev_t <= max_tokens * 1.2:
            groups[-2].extend(groups.pop())
    return groups


def chunk(
    sentences: list[Sentence],
    document_id: str,
    max_tokens: int,
    *,
    overlap_sentences: int = 1,
    min_tokens: int | None = None,
) -> list[Chunk]:
    """Structure-aware parent chunking.

    1. Boundaries: a chunk never crosses a section heading and never mixes table rows with
       prose (a table row already carries its column headers, so rows stay self-contained).
    2. Tiny sibling sections (< ``min_tokens``/2) are merged when the result still fits.
    3. Large sections are packed into balanced chunks <= ``max_tokens``.
    4. Prose chunks inside one section carry ``overlap_sentences`` of the previous chunk as
       ``context_before`` - used for embedding only, so cited sentences stay unique.

    Every sentence belongs to exactly one chunk (``sentence.chunk_id``)."""
    min_tokens = max_tokens // 3 if min_tokens is None else min_tokens
    segs = _merge_small(_segments(sentences), min_tokens, max_tokens)

    chunks: list[Chunk] = []
    last_group: list[Sentence] | None = None
    for seg in segs:
        previous: list[Sentence] | None = None
        if last_group and last_group[-1].heading == seg.sentences[0].heading:
            # Same section, different kind (prose intro -> table): the intro ('The table below
            # shows the stress-test impact...') is what makes the table's embedding findable.
            previous = last_group
        for group in _pack(seg, max_tokens, min_tokens):
            cidx = len(chunks)
            cid = chunk_id(document_id, cidx)
            for s in group:
                s.chunk_id = cid
            text = " ".join(s.text for s in group)
            headings: list[str] = []
            for s in group:
                if s.heading and s.heading not in headings:
                    headings.append(s.heading)
            context = ""
            if previous and overlap_sentences > 0 and not previous[-1].is_table_row:
                context = " ".join(s.text for s in previous[-overlap_sentences:])
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    document_id=document_id,
                    chunk_idx=cidx,
                    text=text,
                    sentence_ids=[s.sentence_id for s in group],
                    heading=group[0].heading,
                    page=group[0].page,
                    token_count=approx_tokens(text),
                    headings=headings,
                    context_before=context,
                    is_table=seg.is_table,
                )
            )
            previous = group
        last_group = previous
    return chunks


# --- ingestion service ---

class IngestionError(ValueError):
    """User-facing validation error (maps to 413/422 at the API edge)."""


class IngestionService:
    def __init__(self, *, settings: Settings, documents, sentences, chunks, files, embeddings, vectors, bm25):
        self._settings = settings
        self._documents = documents
        self._sentences = sentences
        self._chunks = chunks
        self._files = files
        self._embeddings = embeddings
        self._vectors = vectors
        self._bm25 = bm25

    async def ingest(
        self,
        owner_key: str,
        conversation_id: str,
        filename: str,
        content: bytes,
        is_public: bool = False,
    ) -> dict:
        """Validate -> parse/segment/chunk -> persist -> embed. CPU- and IO-bound steps run in
        worker threads so one upload never blocks every other request on the event loop.

        Every uploaded file belongs to exactly one conversation; ``is_public`` is reserved
        for documents placed in the shared public pool (managed out-of-band)."""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTS:
            raise IngestionError(f"unsupported file type '{ext}' (allowed: PDF, DOCX, MD)")
        if len(content) > self._settings.max_file_bytes:
            raise IngestionError(f"file exceeds {self._settings.max_file_mb} MB")

        content_hash = hashlib.sha256(content).hexdigest()

        def precheck() -> dict | None:
            # Idempotent re-upload within a conversation: identical bytes return the existing doc.
            existing = self._documents.find_by_hash(owner_key, conversation_id, content_hash)
            if existing:
                return existing
            if self._documents.count(owner_key, conversation_id) >= self._settings.max_docs_per_conversation:
                raise IngestionError(
                    f"document limit reached ({self._settings.max_docs_per_conversation} per conversation)"
                )
            return None

        existing = await asyncio.to_thread(precheck)
        if existing:
            return existing

        document_id = "d_" + uuid.uuid4().hex[:16]
        name = os.path.basename(filename)
        file_id = ""
        mongo_written = False
        try:
            def process() -> tuple[str, list[Sentence], list[Chunk]]:
                file_id = self._files.save(owner_key, document_id, filename, content)
                sents = segment(parse(filename, content), document_id)
                if not sents:
                    raise IngestionError("no extractable text found in document")
                chs = chunk(
                    sents,
                    document_id,
                    self._settings.chunk_max_tokens,
                    overlap_sentences=self._settings.chunk_overlap_sentences,
                    min_tokens=self._settings.chunk_min_tokens,
                )
                return file_id, sents, chs

            try:
                file_id, sentences, chunks = await asyncio.to_thread(process)
            except IngestionError:
                raise
            except Exception as exc:  # corrupt / encrypted / unparsable file -> 422, not 500
                raise IngestionError(f"could not parse document: {exc}") from exc

            def persist() -> None:
                self._documents.create(
                    owner_key,
                    name=name,
                    type_=ext.lstrip("."),
                    file_id=file_id,
                    sentence_count=len(sentences),
                    document_id=document_id,
                    content_hash=content_hash,
                    conversation_id=conversation_id,
                    is_public=is_public,
                )
                self._sentences.bulk_insert(owner_key, sentences, conversation_id=conversation_id, is_public=is_public)
                self._chunks.bulk_insert(owner_key, chunks, conversation_id=conversation_id, is_public=is_public)

            mongo_written = True  # set before the write so a partial write is rolled back too
            await asyncio.to_thread(persist)

            await self._index_vectors(owner_key, conversation_id, is_public, document_id, name, chunks)
            self._bm25.invalidate(owner_key)

            info = await asyncio.to_thread(self._documents.get, owner_key, document_id)
            return info or {"id": document_id}
        except Exception:
            await asyncio.to_thread(self._rollback, owner_key, document_id, mongo_written)
            raise

    async def _index_vectors(
        self,
        owner_key: str,
        conversation_id: str,
        is_public: bool,
        document_id: str,
        doc_name: str,
        chunks: list[Chunk],
    ) -> None:
        if not self._embeddings.configured:
            return  # degraded mode: BM25-only retrieval still works
        texts = [embed_text(c, doc_name) for c in chunks]
        vectors = await self._embeddings.embed(texts)
        points = [
            {
                "chunk_id": c.chunk_id,
                "document_id": document_id,
                "document_name": doc_name,
                "chunk_idx": c.chunk_idx,
                "text": c.text,
                "heading": c.heading,
                "headings": c.headings,
                "token_count": c.token_count,
                "sentence_ids": c.sentence_ids,
                "page": c.page,
                "is_table": c.is_table,
                "vector": vec,
            }
            for c, vec in zip(chunks, vectors)
        ]
        await asyncio.to_thread(self._vectors.upsert, owner_key, conversation_id, is_public, points)

    def _rollback(self, owner_key: str, document_id: str, mongo_written: bool) -> None:
        try:
            if mongo_written:
                self._sentences.delete_by_document(owner_key, document_id)
                self._chunks.delete_by_document(owner_key, document_id)
                self._documents.delete(owner_key, document_id)
        except Exception:
            pass
        try:
            self._vectors.delete_document(owner_key, document_id)
        except Exception:
            pass
        try:
            self._files.delete_document(owner_key, document_id)
        except Exception:
            pass
        try:
            self._bm25.invalidate(owner_key)
        except Exception:
            pass


def embed_text(chunk: Chunk, doc_name: str = "") -> str:
    """What actually gets embedded: document + section context, the overlap with the previous
    chunk, then the chunk itself. Context in the vector is what lets 'What does the leave
    policy say?' find a chunk whose own sentences never repeat the word 'leave'."""
    lines: list[str] = []
    if doc_name:
        lines.append(f"Document: {doc_name}")
    headings = chunk.headings or ([chunk.heading] if chunk.heading else [])
    if headings:
        lines.append("Section: " + " | ".join(headings))
    if chunk.context_before:
        lines.append(f"(preceding context) {chunk.context_before}")
    lines.append(chunk.text)
    return "\n".join(lines)
