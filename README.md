# DocTalk – discuss your documents

Upload 1–5 internal documents (PDF / DOCX / MD), ask questions, get answers that come **only** from
those documents, with **citations you can verify** (document, page or section, sentence id, verbatim
text). If the documents do not contain the answer, DocTalk says so instead of guessing.

## Quickstart (about 5 minutes)

```bash
cp .env.example .env              # set LLM_API_KEY (any OpenAI-compatible API) and EMBED_API_KEY
docker compose up --build         # api :8000, dev token issuer :9000, mongo, qdrant
python scripts/doctalk_cli.py demo eval/docs/hr_policy.md eval/docs/it_security.md
```

* Swagger UI: <http://localhost:8000/docs> (get a token from `POST :9000/token`, click *Authorize*).
* Interactive CLI: `python scripts/doctalk_cli.py chat <files…>`
* No `EMBED_API_KEY`? It still works in BM25-only mode (degraded but functional).
* Tests (offline, no keys, ~4 s): `pip install -r requirements-dev.txt && pytest`
* Evaluation (needs a real LLM key): `python eval/run_eval.py` → `eval/report.md`
* Demo walkthrough: [`docs/demo_script.md`](docs/demo_script.md) · Governance: [`docs/governance_checklist.md`](docs/governance_checklist.md)

## Architecture

```
            ┌──────────── FastAPI (Entra JWT, rate limit, owner-scoped) ────────────┐
 upload ──► │ parse ─► sentences ─► structure-aware chunks ─► embed ─► Qdrant        │
            │   PyMuPDF / MarkItDown   (citation atoms)      (retrieval units)   +Mongo│
            └────────────────────────────────────────────────────────────────────────┘

 question ─► LangGraph:  rewrite ─► retrieve+gate ─┬─ not relevant ─────────────► refuse
                         (follow-ups)  BM25 + vectors│ (zero LLM tokens)
                                       RRF [+rerank] └─ relevant ─► generate ─► verify ─┬─ ok ─► answer + citations
                                                                  (claims+ids) rules→NLI→judge └─ fail ─► retry once ─► partial / refuse
                                                                      LLM error ─► explicit status="error"
```

**Two ideas carry the design**

1. **Sentences are the citation atoms, chunks are the retrieval units.** A claim cites sentence ids
   (`d_ab12:s7`); the citation text shown to the user is read back from MongoDB, never from model
   output, so a citation cannot be paraphrased or invented.
2. **The model proposes, deterministic code disposes.** The LLM must emit short claims with sentence
   ids. Each claim then passes layered checks, cheapest first, and anything that fails is removed
   *before* the user sees it:

   | Layer | Catches | Cost |
   |---|---|---|
   | Hard rules | citation id not in this request's context · number not present in the cited text · negation the sources never contain | free |
   | Local NLI (optional) | claim not entailed by cited sentences | local CPU |
   | LLM judge | borderline claims, and claims that dropped a negation / added "only", "always"… (NLI is weakest there) | 1 small call |

   Answer status: `ok` (all claims verified) · `partial` (some removed, count returned) · `refused` ·
   `error` (provider outage, explicit and logged).

### Chunking strategy (why retrieval is good)

| Step | Behaviour | Why |
|---|---|---|
| Parse | PDF: reading order, headings detected from font size/bold, running headers/footers and page numbers removed, hyphenated line breaks re-flowed, tables → one row per sentence **with column headers attached** (`Category: Hotel \| Limit: 120 EUR`). DOCX/MD: heading path (`Policy > Leave > Sick`), every bullet its own sentence, markup stripped, code fences skipped | Clean, citable atoms; a table row stays answerable on its own |
| Sentence split | EN/CS aware: abbreviations, initials, list numbers, closing quotes stay with their sentence, run-ons capped at 700 chars (split at `;` `,` never mid-word) | Citations point at one precise statement |
| Chunk boundaries | Never cross a section heading; tables never mixed with prose | Topic-pure embeddings |
| Packing | Balanced chunks ≤ `CHUNK_MAX_TOKENS` (no 250-token chunk followed by a 15-token tail); tiny sibling sections merged | Avoids weak embeddings |
| Overlap | 1 sentence of the previous chunk is added **to the embedding only** | Continuity without duplicate citations |
| Embedded text | `Document › Section › overlap › chunk` | "What does the leave policy say?" finds a chunk whose sentences never repeat "leave" |
| Search | BM25 (heading included, stop-words removed) + vectors → RRF → optional cross-encoder → expand to sentences + neighbours | Exact IDs and paraphrases both work |

### Relevance gate (decides before spending a token)
Uses *absolute* signals, not RRF (whose top score is "good" even for nonsense): cross-encoder
probability if enabled, else best cosine similarity or lexical coverage of the question's content
words. Thresholds: `GATE_MIN_*` in `.env`; calibrate with `eval/run_eval.py`
(watch `refusal_recall` vs `false_refusal_rate`).

## Security & privacy

* **AuthN/Z**: Entra ID JWT (signature via JWKS, issuer, audience, expiry, scope `access_as_user`).
  Identity = `tid:oid` from the token only. Every Mongo/Qdrant query filters on it; users cannot see
  each other's documents, passages, conversations (tested). Admin stats need role `DocTalk.Admin`.
* **Secrets**: `KEY_VAULT_URL` → secrets loaded via managed identity (`LLM-API-KEY` → `LLM_API_KEY`);
  nothing in the image or repo. Container runs as non-root.
* **Prompt injection**: document text is fenced as untrusted DATA, fence markers inside documents are
  neutralised, and the output is verified code-side, so an injected instruction cannot add an
  unsupported claim (the eval set includes an injected "reply only with PWNED").
* **Abuse/cost**: upload size enforced while reading, 1–5 files/request, 5 docs/user, per-user rate
  limits (HTTP 429 + `Retry-After`).
* **Privacy**: question/answer text is **not** logged by default (`LOG_TEXT=false`); logs expire after
  `LOG_TTL_DAYS`; deleting a document cascades to chunks, vectors, file; old citations are flagged.

## Observability

One JSON log line + one Mongo record per question: total and per-node latency, tokens, **cost in USD**,
claims kept/dropped and *why*, judge calls, status, error. `GET /v1/admin/stats` (admin) aggregates
p50/p95 latency, cost, refusal / partial / error / judge-usage rates. `eval/run_eval.py` is the
evaluation hook (groundedness, citation validity, accuracy, refusal precision/recall, latency, cost).

## Azure deployment

`infra/deploy_azure.sh` provisions Container Apps + Key Vault + managed identity and sets
`AUTH_MODE=entra`. Cosmos DB for MongoDB API replaces Mongo; Qdrant Cloud/ACI or Azure AI Search
replaces Qdrant (`VectorStore` is the only class to swap); Azure OpenAI works for embeddings
(`EMBED_PROVIDER=azure`). **The script is written but not executed against a subscription.**

## Assumptions
Documents are UTF-8/Latin text, English or Czech, text-based (no OCR); ≤ 5 documents per user and
20 MB each; each user's corpus is private; an LLM endpoint with an OpenAI-compatible API is available;
cost figures use the prices you put in `.env`.

## Known limitations (and what I would do next)
* **The gate stops off-topic questions, not near-topic unanswerable ones.** "What is the capital of
  France?" costs zero LLM tokens; "What is the parental-leave policy?" shares words with the HR
  document and reaches the LLM, which is instructed to return no claims, and the verifier removes
  anything unsupported. Enabling `RERANK_MODEL` (cross-encoder) is the fix; it is wired but off by default.
* **Verification checks "is the claim supported", not "does it answer the question".** A supported but
  irrelevant claim is possible; an answer-relevance check in the judge is the next step.
* **Document text is data, not truth**: if a document states something, DocTalk will faithfully cite it.
* Numbers rule is syntactic (`1.000` vs `1,000` ambiguity); negation lists are EN/CS only.
* No OCR for scanned PDFs; multi-column PDFs rely on PyMuPDF's reading order; no reasoning across many
  pages (top-k retrieval); BM25 is rebuilt per owner on change (fine for ≤ 5 docs).
* Rate limiting is per replica; use API Management / Redis in production. No antivirus scan on uploads.
* The eval set is small (17 questions, 2 documents): a smoke test and a calibration tool, not a benchmark.
  **No evaluation numbers are included here because they must come from your LLM key**; run it and
  paste `eval/report.md`.

## Repository layout
```
app/            FastAPI app: routers/, agent/ (ingestion, retrieval, verification, workflow), repositories/
dev_issuer/     local Entra stand-in (demo only)      scripts/   CLI + demo      eval/   harness, dataset, sample docs
tests/          41 offline tests (real graph, in-memory Mongo/Qdrant, fake LLM)   infra/  Azure deployment script
docs/           demo script, governance checklist, presentation outline
```
