import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing backend logic from ragdemo
from .ragdemo import (
    load_vectorstore,
    build_retriever,
    ChatHistory,
    CONFIG,
    CostTracker,
    detect_and_load,
    chunk_documents,
    build_vectorstore as build_vs,
    rewrite_query,
    RAG_PROMPT,
    _estimate_tokens
)
from langchain_openai import ChatOpenAI

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class LoadRequest(BaseModel):
    source: str

history = ChatHistory(CONFIG["history_file"])
tracker = CostTracker()
vs = load_vectorstore()
retriever = build_retriever(vs) if vs else None

@app.post("/api/vectorstore/build")
async def build_vectorstore_route(req: LoadRequest):
    global retriever
    try:
        documents = detect_and_load(req.source)
        chunks = chunk_documents(documents)
        vectorstore = build_vs(chunks)
        retriever = build_retriever(vectorstore)
        return {"success": True, "message": f"Loaded {len(documents)} documents, split into {len(chunks)} chunks."}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/history")
async def get_history():
    return {
        "messages": history.messages,
        "cost_queries": tracker.total_queries,
        "cost_usd": tracker.cost_usd
    }

@app.delete("/api/history")
async def delete_history():
    history.clear()
    return {"success": True, "message": "Chat history cleared"}

@app.post("/api/chat")
async def chat(request: Request, req: ChatRequest):
    async def event_generator():
        global retriever
        if not retriever:
            yield f"data: {json.dumps({'error': 'Vector store not initialized. Please load a document source first.'})}\n\n"
            return
            
        try:
            llm = ChatOpenAI(model=CONFIG["model"], temperature=CONFIG["temperature"], streaming=True)
            rewritten = rewrite_query(req.message, llm)
            retrieved_docs = retriever.invoke(rewritten)
            
            context_parts, source_labels = [], []
            seen_labels = set()
            for i, doc in enumerate(retrieved_docs):
                label = doc.metadata.get("source_label", f"chunk_{i}")
                context_parts.append(f"[{i+1}] {doc.page_content}")
                source_labels.append(f"[{i+1}] {label}")
                if label not in seen_labels:
                    seen_labels.add(label)
                    # Send source back to UI first
                    yield f"data: {json.dumps({'source': label})}\n\n"

            context_text = "\n\n".join(context_parts)
            sources_text = "\n".join(source_labels)
            
            input_tokens = _estimate_tokens(context_text + sources_text + req.message)
            
            final_prompt = RAG_PROMPT.invoke({
                "context":  context_text,
                "sources":  sources_text,
                "history":  history.langchain_messages(CONFIG["memory_turns"]),
                "question": req.message,
            })
            
            full_response = ""
            for chunk in llm.stream(final_prompt):
                if await request.is_disconnected():
                    break
                full_response += chunk.content
                yield f"data: {json.dumps({'chunk': chunk.content})}\n\n"
            
            tracker.add(input_tokens, _estimate_tokens(full_response))
            history.add("human", req.message)
            history.add("assistant", full_response)
            
            # Send latest cost details
            yield f"data: {json.dumps({'cost_queries': tracker.total_queries, 'cost_usd': round(tracker.cost_usd, 5)})}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
