# RAG Project Architecture

This repository contains a retrieval-augmented generation (RAG) system with a Python backend and a Next.js UI. The backend can run as a CLI or as an API server, while the UI consumes the API with streaming responses.

## Architecture at a Glance

- Ingestion: load files or URLs, then split to chunks with metadata labels.
- Indexing: embed chunks and persist to Chroma DB.
- Retrieval: MMR-based retriever for diverse context.
- Generation: prompt with context + sources + recent history, then stream tokens.
- Persistence: chat history and cost metrics saved to JSON.

## Key Components

### 1) RAG Core Pipeline (Python)

The RAG pipeline is implemented as a reusable library + CLI entrypoint.

- Configuration and shared settings live in [RAG/ragdemo.py](RAG/ragdemo.py#L70).
- Document loading and format detection are in [RAG/ragdemo.py](RAG/ragdemo.py#L130-L185).
- Chunking and source labeling are in [RAG/ragdemo.py](RAG/ragdemo.py#L188-L206).
- Vector store build/load is in [RAG/ragdemo.py](RAG/ragdemo.py#L210-L235).
- MMR retriever setup is in [RAG/ragdemo.py](RAG/ragdemo.py#L238-L254).
- Query rewriting is in [RAG/ragdemo.py](RAG/ragdemo.py#L257-L273).
- The RAG prompt with history is in [RAG/ragdemo.py](RAG/ragdemo.py#L276-L299).
- Chat history persistence is in [RAG/ragdemo.py](RAG/ragdemo.py#L302-L363).
- Cost tracking is in [RAG/ragdemo.py](RAG/ragdemo.py#L365-L392).
- The full RAG execution flow is in [RAG/ragdemo.py](RAG/ragdemo.py#L452-L524).
- The CLI loop is in [RAG/ragdemo.py](RAG/ragdemo.py#L527-L621).

### 2) API Server (FastAPI)

The API server wraps the same RAG pipeline for use by the UI.

- API setup and CORS configuration are in [RAG/api.py](RAG/api.py#L1-L33).
- Vector store build endpoint is in [RAG/api.py](RAG/api.py#L45-L55).
- Chat history read/clear endpoints are in [RAG/api.py](RAG/api.py#L57-L68).
- Streaming chat endpoint with SSE is in [RAG/api.py](RAG/api.py#L70-L124).

### 3) Web UI (Next.js)

The UI provides a clean interface to load data, chat, and see cost stats.

- The main page, API calls, and streaming parser are in [RAG/ui/src/app/page.tsx](RAG/ui/src/app/page.tsx#L1-L265).
- Global layout and font setup are in [RAG/ui/src/app/layout.tsx](RAG/ui/src/app/layout.tsx#L1-L33).
- Tailwind setup and base styles are in [RAG/ui/src/app/globals.css](RAG/ui/src/app/globals.css#L1-L27).

## Data and Persistence

- Vector store files are persisted under RAG/chroma_db/.
- Chat history is stored in RAG/chat_history.json.

## Request Flow

1. UI calls the build endpoint with a data source path or URL.
2. Backend loads, chunks, embeds, and persists the vector store.
3. UI sends user questions to the chat endpoint.
4. Backend rewrites, retrieves, builds the prompt, and streams tokens.
5. UI renders streaming chunks and updates cost stats.

## How to Run (High Level)

- Start the API server (FastAPI) and the UI (Next.js) in separate terminals.
- Then load a data source in the UI and start chatting.

If you want, I can add exact run commands, environment setup, and a diagram.
