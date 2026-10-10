# Presentation outline – for a non-technical Product Owner (≈ 15 min + demo)

**1. The problem (1 min)** – People waste time hunting through policies; generic chatbots invent answers.
"DocTalk answers from *your* documents and shows the sentence it used."

**2. Live demo (6 min)** – follow `docs/demo_script.md`: ask → citation → table → follow-up → refusal.
Say the quotable line: *"If it can't prove it from your document, it doesn't say it."*

**3. How it stays honest, in plain words (3 min)** – three steps with an everyday analogy:
*Librarian* finds the right pages → *Writer* drafts short claims citing sentences → *Fact-checker* removes any
claim the pages don't support. Numbers and "not/never" get extra scrutiny.

**4. Is it safe? (2 min)** – each user sees only their own documents; sign-in with company identity;
secrets in Key Vault; no question text stored by default; hostile text inside a document can't take over.

**5. What it costs and how we know it works (2 min)** – cost/latency per question shown in every answer and
in an admin view; an evaluation set measures correctness, citation validity and refusal behaviour (show `eval/report.md`).

**6. Honest limits and next steps (1 min)** – scanned PDFs, "almost on-topic" questions, privacy review
(DPIA / provider region) before real data. Next: re-ranker on, answer-relevance check, Azure rollout.

### Likely questions
* *Can it be wrong?* It can fail to find an answer (it says so). It is built so it cannot state something the document doesn't support; the remaining risk is a supported but off-target answer, which the next iteration addresses.
* *Where does our data go?* To the configured LLM/embedding provider; choose Azure OpenAI in the EU region. Open governance item #1.
* *How much per question?* The answer shows it; multiply by usage. Gate and NLI avoid paying for hopeless questions.
* *Why not just ChatGPT?* No citations you can verify, no per-user isolation, no cost/quality telemetry.

### Technical depth if asked (the "explain details" 10 pts)
Sentence-level citations vs chunk-level retrieval · hybrid BM25+vector with RRF · structure-aware chunking
(section/table boundaries, balanced packing, embedding-only overlap) · deterministic gate before LLM ·
layered verification rules→NLI→judge, fail closed · LangGraph state machine with retry and explicit error path.
