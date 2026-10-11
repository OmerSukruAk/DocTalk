# Governance checklist – DocTalk

Draft entry for a data catalogue (e.g. Collibra) and the controls and open items a data-governance
review would ask about.

## Asset entry (draft)

| Field | Value |
|---|---|
| Asset name | DocTalk – document Q&A service |
| Type | Application / AI service (retrieval-grounded generation) |
| Business owner / steward | *to be assigned* |
| Technical owner | *to be assigned* |
| Purpose | Let employees ask questions about documents they upload and get answers with verifiable citations |
| Data categories | Employee-uploaded internal documents (may contain personal or confidential data); user identity (Entra `tid:oid`); questions and answers |
| Classification | Treat all inputs as **Confidential** unless the user classifies them otherwise |
| Systems | Web UI and API (two Azure Container Apps); MongoDB / Cosmos DB (text, original files in GridFS, conversations, logs); Qdrant (vectors); LLM and embeddings provider |
| Lineage | Upload → parse → sentences and chunks → embeddings (provider) → retrieval → LLM (provider) → verified answer |
| Retention | Documents and conversations until the user deletes them (deletion cascades to chunks, vectors and files); request logs for `LOG_TTL_DAYS` (default 90) |
| Data residency | Depends on the LLM and embeddings region: **open item** (use Azure OpenAI in an EU region for EU data) |

## Controls in place

- [x] Entra ID authentication on every API request; per-user data isolation on every query; admin
      role for aggregate statistics
- [x] Web UI holds no secrets and signs in only with a token the API verifies
- [x] Secrets in Key Vault through a managed identity, none in images or the repo; containers run as
      non-root
- [x] Answers only from uploaded content; unsupported statements removed before display; citation
      text read from storage, never from model output
- [x] Prompt-injection hardening: untrusted text fenced, output verified in code
- [x] Question and answer text not logged by default; log retention limit; user deletion cascades
- [x] Rate limits; upload size, type and count limits
- [x] Audit trail per request: request id, user, status, cost, reasons for removed statements
- [x] 62 automated tests and an evaluation harness (groundedness, citation validity, refusal metrics)

## Risks and open items

| # | Risk | Status / mitigation |
|---|---|---|
| 1 | Document content is sent to a third-party LLM and embeddings provider | **Open:** confirm the data processing agreement, region and no-training / zero-retention terms; prefer Azure OpenAI |
| 2 | Personal data inside uploaded documents (GDPR) | **Open:** data protection impact assessment; optional PII detection and redaction before embedding |
| 3 | Near-topic questions the documents can't answer reach the LLM | Mitigated by the prompt and the verifier; enable the cross-encoder re-ranker (see README limitations) |
| 4 | A supported but off-target statement in an answer | Planned: answer-relevance check in the judge |
| 5 | Malware in uploads | **Open:** antivirus scanning (e.g. Defender for Storage) before parsing |
| 6 | Rate limiting is per replica | Planned: API Management or Redis |
| 7 | A model or provider change alters behaviour | Run `eval/run_eval.py` as a release gate and compare with the last report |
| 8 | Stored text at rest in MongoDB and vector payloads | Encryption at rest (Cosmos DB / Azure defaults); customer-managed keys if required |
| 9 | The local dev token issuer signs tokens for anyone | Demo only; not part of the Azure deployment, where tokens come from Entra ID |
| 10 | The web UI keeps the pasted token in browser session storage | Planned: "Sign in with Microsoft" (MSAL) with the token kept server-side |

## Review cadence

Re-run the evaluation on every model, prompt or threshold change. Review the error, refusal and
judge-usage rates weekly from `GET /v1/admin/stats`.
