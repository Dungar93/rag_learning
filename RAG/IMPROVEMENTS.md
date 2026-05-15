# RAG Project Improvements - Detailed Implementation Plan

This document provides a complete, step-by-step plan to upgrade the current RAG system. It is written so you can implement items incrementally, track progress, and verify results at each phase. The plan assumes the existing structure:

- RAG/ragdemo.py (core pipeline)
- RAG/api.py (FastAPI endpoints)
- RAG/ui (Next.js UI)

If you want to follow only a subset, start with Phase 1 and Phase 2.

---

## Phase 0 - Baseline and Safety Checks

Goal: Ensure the system is stable before major changes.

1) Add a baseline dataset
	- Create a small test corpus (3-10 files) with known answers.
	- Include at least 1 PDF, 1 CSV, and 1 TXT file.

2) Define baseline test questions
	- 10-20 questions with expected answer points and expected sources.
	- Store them in a JSON or CSV file for repeatable evaluation.

3) Create a baseline run checklist
	- Record current retrieval results and answer quality for the baseline dataset.
	- Save the outputs as a reference for regression checks.

Deliverable: A small evaluation dataset and a repeatable baseline run.

---

## Phase 1 - Data Ingestion and Source Registry

Goal: Support more file types and keep track of sources.

### 1.1 Add additional loaders

Add loaders for DOCX, PPTX, HTML, and Markdown.

Implementation steps:
1) Add new loader imports in RAG/ragdemo.py.
2) Extend the file-extension map in detect_and_load().
3) For HTML, strip boilerplate tags before storing content.

Recommended libraries:
- python-docx for DOCX
- python-pptx for PPTX
- beautifulsoup4 for HTML
- markdown or mistune for Markdown parsing

Testing:
- Add one file per new type, ensure it loads, chunks, and indexes.

### 1.2 Add data source registry

Create a JSON registry that stores:
- source path or URL
- file count
- last indexed timestamp
- optional tags

Implementation steps:
1) Add a small registry module, e.g. RAG/source_registry.py.
2) Update build_vectorstore_route to update registry after indexing.
3) Add an API endpoint to list sources.

Testing:
- Build vector store twice and verify registry updates.

Deliverable: Extended ingestion and a source registry.

---

## Phase 2 - Chunking and Embedding Improvements

Goal: Improve chunk quality and reduce re-embedding costs.

### 2.1 Structural chunking

Use headings or paragraph boundaries when possible.

Implementation steps:
1) Create a chunking strategy that inspects document metadata.
2) Use different chunking rules for Markdown, HTML, and PDF text.
3) Preserve heading titles as metadata in each chunk.

Testing:
- Compare chunk previews to ensure headings align with content.

### 2.2 Token-based overlap

Switch from character overlap to token overlap for consistency.

Implementation steps:
1) Introduce a token counter (tiktoken or model-specific tokenizer).
2) Modify the chunking function to approximate by tokens.

Testing:
- Validate that chunk lengths are near configured token sizes.

### 2.3 Embedding cache

Skip re-embedding unchanged chunks.

Implementation steps:
1) Hash chunk content (e.g., SHA-256).
2) Store hash -> embedding mapping in a local cache file or lightweight DB.
3) When building vector store, reuse cached embeddings by hash.

Testing:
- Re-run indexing on the same corpus and confirm embedding calls decrease.

Deliverable: Better chunk quality and faster indexing.

---

## Phase 3 - Retrieval Quality

Goal: Improve recall and precision for real user queries.

### 3.1 Hybrid retrieval (BM25 + vector)

Implementation steps:
1) Add a BM25 index (e.g., rank_bm25) over the same chunks.
2) For each query, retrieve top N from vector and BM25.
3) Merge and re-rank with a weighted score.

Testing:
- Compare retrieval sets for baseline questions.

### 3.2 Query expansion

Implementation steps:
1) Add a query expansion step using synonyms or acronym expansion.
2) Merge expanded queries and rerun retrieval.
3) Keep original query in the prompt for answer clarity.

Testing:
- Use acronym-heavy questions and compare recall.

### 3.3 Re-ranking

Implementation steps:
1) Add a re-ranker model (cross-encoder or LLM-based).
2) Re-rank top 20 candidates to select best 5-10.
3) Keep sources aligned after re-ranking.

Testing:
- Evaluate answer quality changes and latency impact.

Deliverable: Higher retrieval quality with manageable latency.

---

## Phase 4 - Prompting, Citations, and Confidence

Goal: Improve answer reliability and transparency.

### 4.1 Strict citation formatting

Implementation steps:
1) Enforce a citation format like [1], [2] in the final response.
2) Add a post-processing check to ensure cited sources exist.

### 4.2 Confidence score

Implementation steps:
1) Compute a simple confidence score based on retrieval scores.
2) Display it in UI and include in response footer.

### 4.3 Refusal for low relevance

Implementation steps:
1) If max retrieval score is below a threshold, respond with a safe refusal.
2) Suggest the user to load relevant sources.

Testing:
- Ask out-of-scope questions and verify refusal behavior.

Deliverable: Stronger and more honest responses.

---

## Phase 5 - Memory and Conversation Management

Goal: Reduce token usage and improve multi-turn chat.

### 5.1 Summary memory

Implementation steps:
1) After each N turns, summarize older history into a short memory block.
2) Store the summary separately from raw messages.
3) Insert summary into the prompt instead of full history.

### 5.2 Session-based history

Implementation steps:
1) Add a session_id to the API request.
2) Store chat history per session_id.
3) Update UI to send a stable session ID.

Testing:
- Open two sessions and confirm independent histories.

Deliverable: Scalable memory with lower cost.

---

## Phase 6 - Accurate Token and Cost Tracking

Goal: Replace approximations with exact metrics.

Implementation steps:
1) Add a token counter using tiktoken or the model's tokenizer.
2) Track input tokens, output tokens, and total cost per request.
3) Store per-message usage in chat history JSON.
4) Update UI to show per-message cost.

Testing:
- Verify token counts against known prompt sizes.

Deliverable: Accurate cost visibility.

---

## Phase 7 - API and UI Enhancements

Goal: Make the system robust and user-friendly.

### 7.1 Health and status endpoints

Implementation steps:
1) Add /api/health to check API status.
2) Add /api/status to report vector store status, last indexed time.

### 7.2 Input validation and rate limiting

Implementation steps:
1) Validate message length and source input size.
2) Add a basic rate limiter per IP.

### 7.3 File upload in UI

Implementation steps:
1) Add upload endpoint in FastAPI using multipart form data.
2) Store uploaded files in a safe temp directory.
3) Update UI to allow drag-and-drop upload.

Testing:
- Upload a file and index it without specifying a local path.

Deliverable: Better UX and stability.

---

## Phase 8 - Evaluation and Observability

Goal: Make improvements measurable.

### 8.1 Evaluation harness

Implementation steps:
1) Build a script to run evaluation questions in batch.
2) Record retrieval and answer outputs in a report file.
3) Compute metrics such as recall@k and MRR.

### 8.2 Structured logging

Implementation steps:
1) Log retrieval timing, prompt size, and errors in JSON lines.
2) Add a request ID to each API call and propagate it.

Deliverable: A reproducible benchmark and operational visibility.

---

## Phase 9 - Security and Compliance

Goal: Prepare for real usage and sensitive data.

Implementation steps:
1) Restrict CORS to allowed domains in production.
2) Add API key auth to protect endpoints.
3) Add PII detection and redaction before indexing.
4) Encrypt chat history at rest if needed.

Testing:
- Confirm unauthorized requests fail.
- Confirm PII is redacted in indexed chunks.

Deliverable: Safer deployment posture.

---

## Phase 10 - Deployment and Scaling

Goal: Production-ready packaging and scale.

Implementation steps:
1) Add Dockerfiles for API and UI.
2) Use env-based configuration for model and DB settings.
3) Add background indexing tasks for large data sources.
4) Add multi-tenant or per-user vector collections if needed.

Deliverable: Deployable system with growth path.

---

## Detailed File-by-File Plan

This section maps improvements to specific files and changes.

### RAG/ragdemo.py

- Add new loaders and metadata fields in detect_and_load().
- Extend chunk_documents() to accept strategy and token-based overlap.
- Add embedding cache logic in build_vectorstore().
- Add hybrid retrieval and re-ranking steps in rag_query().
- Add confidence calculation and refusal logic in rag_query().
- Add exact token accounting for cost tracking.

### RAG/api.py

- Add endpoints: /api/health, /api/status, /api/sources, /api/upload.
- Add session_id support for chat history.
- Add request ID propagation and logging.
- Add rate limiting and input validation.

### RAG/ui/src/app/page.tsx

- Add file upload UI and status display.
- Add session ID initialization and per-session history.
- Add per-message cost display.
- Add source registry panel with filters.

### New Files to Add

- RAG/source_registry.py: track indexed sources and metadata.
- RAG/eval_runner.py: batch evaluation harness.
- RAG/token_utils.py: accurate token counting helpers.
- RAG/logging_utils.py: structured logging helpers.

---

## Milestone Checklist

Use this list to track progress:

1) Baseline dataset and evaluation questions
2) New loaders and source registry
3) Chunking improvements and embedding cache
4) Hybrid retrieval and re-ranking
5) Prompting, citations, confidence, and refusal
6) Summary memory and session-based history
7) Accurate token accounting and cost reporting
8) API status endpoints and file uploads
9) Evaluation harness and structured logs
10) Security and deployment tasks

---

## Notes on Scope and Priorities

If you want the highest impact quickly, prioritize:
- Hybrid retrieval + re-ranking
- File upload support
- Evaluation harness
- Accurate token accounting

If you want production readiness, prioritize:
- Security, auth, and CORS restrictions
- Structured logging and monitoring
- Docker deployment

---

If you want, I can convert this plan into a set of concrete tasks with file diffs and code changes for each phase.
