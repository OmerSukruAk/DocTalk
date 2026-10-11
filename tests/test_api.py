import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth import Principal, current_principal
from app.ratelimit import SlidingWindowLimiter
from app.routers import admin, conversations, documents
from app.secrets import env_name


@pytest.fixture()
def client(services):
    app = FastAPI()
    for r in (admin.router, documents.router, conversations.router):
        app.include_router(r)
    app.state.services, app.state.graph = services, services.graph
    app.dependency_overrides[current_principal] = lambda: Principal("tid:alice", [])
    return TestClient(app)


def test_upload_ask_citations_over_http(client, hr_doc):
    cid = client.post("/v1/conversations").json()["id"]
    r = client.post(
        "/v1/documents",
        data={"conversation_id": cid},
        files=[("files", ("hr.md", io.BytesIO(hr_doc), "text/markdown"))],
    )
    assert r.status_code == 201 and r.json()[0]["sentence_count"] > 5
    a = client.post(f"/v1/conversations/{cid}/messages", json={"question": "How many sick leave days?"})
    body = a.json()
    assert a.status_code == 200 and body["status"] == "ok" and body["citations"]
    passage = client.get(f"/v1/passages/{body['citations'][0]['sentence_id']}").json()
    assert passage["citation"]["text"] == body["citations"][0]["text"]


def test_upload_validation(client):
    cid = client.post("/v1/conversations").json()["id"]
    assert client.post("/v1/documents", data={"conversation_id": cid}, files=[("files", ("x.exe", b"MZ", "application/octet-stream"))]).status_code == 422
    six = [("files", (f"{i}.md", b"# a\n\nb c d.", "text/markdown")) for i in range(6)]
    assert client.post("/v1/documents", data={"conversation_id": cid}, files=six).status_code == 422


def test_oversized_upload_is_rejected_without_reading_it_all(client, services):
    cid = client.post("/v1/conversations").json()["id"]
    services.settings.max_file_mb = 1
    big = b"# T\n\n" + b"x" * (2 * 1024 * 1024)
    assert client.post("/v1/documents", data={"conversation_id": cid}, files=[("files", ("big.md", big, "text/markdown"))]).status_code == 413


def test_admin_stats_requires_admin_role(client):
    assert client.get("/v1/admin/stats").status_code == 401 or client.get("/v1/admin/stats").status_code == 403


def test_sliding_window_limiter():
    lim = SlidingWindowLimiter(window_s=60)
    assert [lim.check("u", 2, now=t) for t in (0, 1)] == [0, 0]
    assert lim.check("u", 2, now=2) > 0
    assert lim.check("u", 2, now=61) == 0  # first hit expired
    assert lim.check("other", 2, now=2) == 0  # per-user
    assert lim.check("u", 0, now=3) == 0  # 0 disables


def test_rate_limit_returns_429_with_retry_after(client, services, hr_doc):
    services.settings.rate_limit_ask_per_min = 1
    cid = client.post("/v1/conversations").json()["id"]
    url = f"/v1/conversations/{cid}/messages"
    assert client.post(url, json={"question": "hello?"}).status_code == 200
    r = client.post(url, json={"question": "hello again?"})
    assert r.status_code == 429 and "retry-after" in r.headers


def test_key_vault_secret_name_mapping():
    assert env_name("LLM-API-KEY") == "LLM_API_KEY"
