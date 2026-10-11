"""Source viewer: shows a cited sentence where it lives in the original document.

PDF: the cited page is rendered to an image and the sentence's own lines are highlighted. The API
stores the paragraph's box (`location.bbox`, PDF points, top-left origin); inside it we find the
sentence by matching its letters against the page's words, so hyphenation and line breaks don't
matter. If that fails (e.g. a table row, whose text has column headers added) the box itself is
highlighted. DOCX/MD have no pages, so the sentence is shown highlighted inside its passage."""
from __future__ import annotations

import base64
import re
from functools import lru_cache

import pymupdf

import api

ZOOM = 2.0  # render at 144 dpi so text stays crisp on retina screens


@lru_cache(maxsize=8)
def _file(token: str, document_id: str) -> bytes:
    """Recently viewed files, so paging through a PDF doesn't download it again."""
    return api.document_file(token, document_id)


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def _line_boxes(words: list[tuple]) -> list[pymupdf.Rect]:
    lines: dict[tuple[int, int], pymupdf.Rect] = {}
    for x0, y0, x1, y1, _text, block, line, _n in words:
        key = (block, line)
        rect = pymupdf.Rect(x0, y0, x1, y1)
        lines[key] = lines[key] | rect if key in lines else rect
    return list(lines.values())


def sentence_boxes(page: pymupdf.Page, text: str, bbox: list[float] | None) -> list[pymupdf.Rect]:
    """One box per text line the sentence covers; the paragraph box when the words can't be matched."""
    needle = _norm(text)
    clips = ([pymupdf.Rect(bbox) + (-3, -3, 3, 3)] if bbox else []) + [None]
    for clip in clips:
        words = page.get_text("words", clip=clip)
        owner: list[int] = []
        for i, word in enumerate(words):
            owner.extend([i] * len(_norm(word[4])))
        haystack = "".join(_norm(w[4]) for w in words)
        start = haystack.find(needle) if needle else -1
        if start >= 0:
            hit = sorted(set(owner[start:start + len(needle)]))
            return _line_boxes([words[i] for i in hit])
    return [pymupdf.Rect(bbox)] if bbox else []


def render_pdf_page(token: str, document_id: str, page_no: int, highlight: tuple[str, list | None] | None):
    """(page_no, page_count, png data URI, highlight boxes as CSS percentages of the page)."""
    with pymupdf.open(stream=_file(token, document_id), filetype="pdf") as doc:
        page_no = max(1, min(page_no, doc.page_count))
        page = doc[page_no - 1]
        boxes = sentence_boxes(page, *highlight) if highlight else []
        png = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM)).tobytes("png")
        width, height = page.rect.width, page.rect.height
        css = [
            {
                "left": f"{(r.x0 - 2) / width:.4%}", "top": f"{(r.y0 - 1) / height:.4%}",
                "width": f"{(r.width + 4) / width:.4%}", "height": f"{(r.height + 2) / height:.4%}",
            }
            for r in boxes
        ]
        return page_no, doc.page_count, "data:image/png;base64," + base64.b64encode(png).decode(), css
