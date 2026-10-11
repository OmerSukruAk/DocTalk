"""Runs the evaluation harness against the in-memory app (fake LLM) to prove the harness itself
works: metrics are computed, citation validity is checked against the stored passages, and the
injection / refusal expectations behave. The NUMBERS here are not the evaluation result: that
requires a real LLM (see eval/run_eval.py and README)."""
import io
import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth import Principal, current_principal
from app.routers import admin, conversations, documents

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import run_eval  # noqa: E402


def test_harness_runs_end_to_end(services):
    app = FastAPI()
    for r in (admin.router, documents.router, conversations.router):
        app.include_router(r)
    app.state.services, app.state.graph = services, services.graph
    app.dependency_overrides[current_principal] = lambda: Principal("tid:eval", [])
    client = TestClient(app)
    cid = client.post("/v1/conversations").json()["id"]
    files = [("files", (p.name, io.BytesIO(p.read_bytes()))) for p in sorted((ROOT / "eval/docs").glob("*.md"))]
    assert client.post("/v1/documents", data={"conversation_id": cid}, files=files).status_code == 201
    result = run_eval.evaluate(client, json.loads((ROOT / "eval/dataset.json").read_text()), conversation_id=cid)
    s = result["summary"]
    assert s["n"] == 17
    assert s["citation_validity"] == 1.0 and s["groundedness"] == 1.0
    by_id = {r["id"]: r for r in result["rows"]}
    assert by_id["q12"]["status"] == "refused"  # clearly off-topic: stopped by the gate
    assert s["injection_leaks"] == 0
    assert "| answer_accuracy |" in run_eval.to_markdown(result)
