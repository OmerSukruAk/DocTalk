# Presentation outline – for a non-technical Product Owner (about 15 minutes, demo included)

**1. The problem (1 min).** People lose time searching through policies, and generic chatbots make
answers up. *"DocTalk answers from your documents and shows you the sentence it used."*

**2. Live demo (6 min).** Follow [`demo_script.md`](demo_script.md) in the web UI: upload → ask →
click a citation to see the highlighted source → table question → follow-up → refusal. The line to
remember: *"If it can't prove it from your document, it doesn't say it."*

**3. How it stays honest, in plain words (3 min).** Three roles:
- a *librarian* finds the relevant passages;
- a *writer* drafts the answer in plain language, citing a sentence for every statement;
- a *fact-checker* removes anything the passages don't support, looking extra hard at numbers and at
  words like "not" and "never".

The writer may add its own analysis (a comparison, a calculation), but it is marked as analysis and
must still cite the facts it builds on.

**4. Is it safe? (2 min).** Each person sees only their own documents; sign-in uses the company
identity (Entra ID); keys are kept in Azure Key Vault; questions are not stored by default; and
instructions hidden inside a document can't take over the assistant.

**5. Cost, and how we know it works (2 min).** Every answer shows its time and cost, and an admin view
adds them up. An evaluation set measures correctness, citation validity and refusals (show
`eval/report.md`).

**6. Honest limits and next steps (1 min).** Scanned PDFs aren't supported yet; "almost on-topic"
questions reach the AI (and get filtered by the fact-checker); a privacy review is needed before real
data. Next: Microsoft sign-in in the web UI, an answer-relevance check, and the Azure rollout.

## Likely questions

- **Can it be wrong?** It can fail to find an answer, and then it says so. It is built not to state
  anything the documents don't support; the remaining risk is a correct but off-target statement,
  which the next iteration addresses.
- **Where does our data go?** To the configured AI provider. Choosing Azure OpenAI in an EU region
  keeps it within the company's cloud (governance open item 1).
- **How much does a question cost?** Each answer shows it, typically fractions of a cent; multiply by
  expected usage. Off-topic questions are refused before the AI is called, so they cost nothing.
- **Why not just use ChatGPT?** No citations you can check, no separation between users' documents,
  and no cost or quality monitoring.

## Technical depth, if asked

Sentence-level citations with chunk-level retrieval · hybrid BM25 + vector search fused with RRF ·
structure-aware chunking (section and table boundaries, balanced packing, embedding-only overlap) ·
a relevance gate before any LLM call · layered verification (rules → NLI → LLM judge) that fails
closed · a LangGraph state machine with retry and an explicit error path · two independently
deployable containers (FastAPI API, Dash web UI).
