# Governance checklist – DocTalk (draft for Collibra / data-governance review)

## Asset entry (draft)
| Field | Value |
|---|---|
| Asset name | DocTalk – document Q&A service |
| Type | Application / AI service (RAG, retrieval-grounded generation) |
| Business owner / Steward | *to be assigned* |
| Technical owner | *to be assigned* |
| Purpose | Let employees query documents they upload, with verifiable citations |
| Data subjects / categories | Employee-authored internal documents (may contain personal or confidential data), user identity (Entra `tid:oid`), questions & answers |
| Classification | Treat inputs as **Confidential** until the user classifies otherwise |
| Systems | API (Container Apps), MongoDB (Cosmos DB API), Qdrant (vectors), file storage, LLM + embeddings provider |
| Lineage | Upload → parse → sentences/chunks → embeddings (provider) → retrieval → LLM (provider) → verified answer |
| Retention | Documents/conversations until user deletes (cascade to vectors/files); request logs `LOG_TTL_DAYS` (90) |
| Residency | Depends on LLM/embedding region: **open item** (use Azure OpenAI in an EU region for EU data) |

## Controls implemented
- [x] Authentication (Entra JWT) and per-user isolation on every query; admin role for aggregates
- [x] Secrets in Key Vault via managed identity (no secrets in image/repo); non-root container
- [x] Answers only from uploaded content; unsupported claims removed; citations read from storage
- [x] Prompt-injection hardening (fenced untrusted text, code-side verification)
- [x] Question/answer text not logged by default; log TTL; user-initiated deletion cascades
- [x] Rate limits, upload size/type/count limits
- [x] Audit trail: request id, user, status, cost, dropped-claim reasons
- [x] Automated tests (41) + evaluation harness with groundedness / refusal metrics

## Risks and open items
| # | Risk | Status / mitigation |
|---|---|---|
| 1 | Document content sent to a third-party LLM/embedding provider | **Open:** DPA, region, no-training / zero-retention terms must be confirmed; prefer Azure OpenAI |
| 2 | Personal data inside uploaded documents (GDPR) | **Open:** DPIA; add PII detection/redaction option before embedding |
| 3 | Near-topic unanswerable questions reach the LLM | Mitigated by prompt + verifier; enable cross-encoder re-ranker; see README limitations |
| 4 | Supported-but-irrelevant claim | Planned: answer-relevance check |
| 5 | Malware in uploads | **Open:** AV scanning (Defender for Storage) before parse |
| 6 | Per-replica rate limiting | Planned: API Management / Redis |
| 7 | Model/provider change alters behaviour | Re-run `eval/run_eval.py` as a release gate; compare to the last report |
| 8 | Stored text in Mongo/vector payloads at rest | Use encryption at rest (Cosmos/Azure defaults); customer-managed keys if required |

## Review cadence
Re-run evaluation on every model, prompt or threshold change; review `error_rate`, `refusal_rate`
and `judge_usage_rate` weekly from `/v1/admin/stats`.
