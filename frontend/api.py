"""Thin HTTP client for the DocTalk API.

The frontend never imports backend code: it only knows the HTTP contract (app/schemas.py in the
API), so the two run, scale and deploy as separate containers. Configured by environment only."""
from __future__ import annotations

import os

import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_S = float(os.getenv("API_TIMEOUT_S", "180"))  # answers with a judge call can take a while


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return response.text or response.reason_phrase
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(str(d.get("msg", d)) for d in detail)
    return str(detail)


def _call(method: str, url: str, token: str | None = None, raw: bool = False, **kwargs):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = httpx.request(method, url, headers=headers, timeout=TIMEOUT_S, **kwargs)
    except httpx.HTTPError as exc:
        raise ApiError(0, f"Cannot reach {url.split('/v1')[0]}. Is it running?") from exc
    if response.status_code >= 400:
        raise ApiError(response.status_code, _detail(response))
    return response.content if raw else response.json()


def list_conversations(token: str) -> list[dict]:
    return _call("GET", f"{API_URL}/v1/conversations", token)


def create_conversation(token: str) -> dict:
    return _call("POST", f"{API_URL}/v1/conversations", token)


def get_conversation(token: str, conversation_id: str) -> dict:
    return _call("GET", f"{API_URL}/v1/conversations/{conversation_id}", token)


def delete_conversation(token: str, conversation_id: str) -> None:
    _call("DELETE", f"{API_URL}/v1/conversations/{conversation_id}", token)


def list_documents(token: str, conversation_id: str) -> list[dict]:
    docs = _call("GET", f"{API_URL}/v1/documents", token)
    return [d for d in docs if d.get("conversation_id") == conversation_id]


def upload_documents(token: str, conversation_id: str, files: list[tuple[str, bytes]]) -> list[dict]:
    return _call(
        "POST",
        f"{API_URL}/v1/documents",
        token,
        data={"conversation_id": conversation_id},
        files=[("files", (name, content)) for name, content in files],
    )


def delete_document(token: str, document_id: str) -> None:
    _call("DELETE", f"{API_URL}/v1/documents/{document_id}", token)


def document_file(token: str, document_id: str) -> bytes:
    return _call("GET", f"{API_URL}/v1/documents/{document_id}/file", token, raw=True)


def passage(token: str, sentence_id: str, context: int = 4) -> dict:
    """The cited sentence plus `context` sentences before and after it."""
    return _call("GET", f"{API_URL}/v1/passages/{sentence_id}", token, params={"context": context})


def ask(token: str, conversation_id: str, question: str, style: str | None, include_public: bool) -> dict:
    return _call(
        "POST",
        f"{API_URL}/v1/conversations/{conversation_id}/messages",
        token,
        json={"question": question, "style": style or None, "include_public": include_public},
    )
