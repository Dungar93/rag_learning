# =========================================================
# ENHANCED RAG APPLICATION USING LANGCHAIN + CHROMADB
# =========================================================
#
# FEATURES:
#   1.  Multi-format document loading  (txt, pdf, csv, web URL)
#   2.  Recursive text chunking
#   3.  OpenAI Embeddings
#   4.  Chroma vector database with persistence
#   5.  MMR Retriever  (diversity-aware retrieval)
#   6.  Relevance-score display
#   7.  Query rewriting  (improves recall)
#   8.  Conversational memory  (last N turns kept in context)
#   9.  Streaming LLM responses
#  10.  Source citations in answers
#  11.  Colorized terminal UI
#  12.  Token + cost tracker
#  13.  Persistent chat history saved to JSON
#  14.  /commands  (help, clear, history, sources, cost, exit)
# =========================================================


# =========================
# IMPORTS
# =========================

import os
import sys
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv

# ── LangChain loaders ──────────────────────────────────
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader,
    DirectoryLoader,
)

# ── Splitter ───────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── OpenAI ─────────────────────────────────────────────
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ── Vector store ───────────────────────────────────────
from langchain_community.vectorstores import Chroma

# ── Prompts & parsers ──────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()


# =========================================================
# CONFIGURATION
# =========================================================

CONFIG = {
    "model":            "gpt-3.5-turbo",   # swap to gpt-4o etc.
    "temperature":      0,
    "chunk_size":       500,
    "chunk_overlap":    100,
    "retriever_k":      5,                  # docs fetched before MMR
    "retriever_fetch":  10,                 # candidate pool for MMR
    "memory_turns":     6,                  # how many past turns to keep
    "persist_dir":      "chroma_db",
    "history_file":     "chat_history.json",
    "streaming":        True,
    # Cost per 1 000 tokens (USD) – update when OpenAI reprices
    "cost_per_1k_input":  0.0005,
    "cost_per_1k_output": 0.0015,
}

# ── Rough tiktoken estimate (4 chars ≈ 1 token) ────────
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# =========================================================
# TERMINAL COLORS
# =========================================================

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    GREY    = "\033[90m"
    WHITE   = "\033[97m"

def banner(text: str, color=C.CYAN):
    width = 60
    print(f"\n{color}{C.BOLD}{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}{C.RESET}\n")

def info(text: str):
    print(f"{C.BLUE}ℹ  {text}{C.RESET}")

def success(text: str):
    print(f"{C.GREEN}✔  {text}{C.RESET}")

def warn(text: str):
    print(f"{C.YELLOW}⚠  {text}{C.RESET}")

def error(text: str):
    print(f"{C.RED}✖  {text}{C.RESET}")


# =========================================================
# STEP 1 : DOCUMENT LOADING  (multi-format)
# =========================================================

def detect_and_load(source: str) -> List[Document]:
    """
    Auto-detect source type and return a list of LangChain Documents.
    Supports:
      - Web URLs  (http / https)
      - Directories  (loads .txt, .pdf, .csv recursively)
      - Individual files  (.txt  .pdf  .csv)
    """
    source = source.strip()

    # ── Web URL ───────────────────────────────────────
    if source.startswith("http://") or source.startswith("https://"):
        info(f"Loading web page: {source}")
        loader = WebBaseLoader(source)
        return loader.load()

    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    # ── Directory ─────────────────────────────────────
    if path.is_dir():
        info(f"Loading all documents from directory: {source}")
        loaders = {
            "**/*.txt": TextLoader,
            "**/*.pdf": PyPDFLoader,
            "**/*.csv": CSVLoader,
        }
        docs: List[Document] = []
        for glob_pattern, loader_cls in loaders.items():
            try:
                loader = DirectoryLoader(source, glob=glob_pattern, loader_cls=loader_cls, silent_errors=True)
                docs.extend(loader.load())
            except Exception:
                pass
        return docs

    # ── Single file ───────────────────────────────────
    ext = path.suffix.lower()
    loaders_map = {
        ".txt":  lambda p: TextLoader(p),
        ".pdf":  lambda p: PyPDFLoader(p),
        ".csv":  lambda p: CSVLoader(p),
    }

    if ext not in loaders_map:
        raise ValueError(f"Unsupported file type: {ext}")

    info(f"Loading file: {source}")
    loader = loaders_map[ext](str(path))
    return loader.load()


# =========================================================
# STEP 2 : CHUNKING
# =========================================================

def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CONFIG["chunk_size"],
        chunk_overlap = CONFIG["chunk_overlap"],
        add_start_index = True,           # metadata: start char position
    )
    chunks = splitter.split_documents(documents)

    # Attach a short readable source label to every chunk
    for i, chunk in enumerate(chunks):
        src = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source_label"] = Path(src).name if "/" in src or "\\" in src else src

    return chunks


# =========================================================
# STEP 3–4 : EMBEDDINGS + VECTORSTORE
# =========================================================

def build_vectorstore(chunks: List[Document]) -> Chroma:
    info("Creating embeddings and building Chroma vector store …")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents     = chunks,
        embedding     = embeddings,
        persist_directory = CONFIG["persist_dir"],
    )
    vectorstore.persist()
    return vectorstore


def load_vectorstore() -> Optional[Chroma]:
    """Load an existing persisted Chroma store (if it exists)."""
    persist_dir = CONFIG["persist_dir"]
    if Path(persist_dir).exists():
        info("Loading existing Chroma vector store …")
        embeddings = OpenAIEmbeddings()
        return Chroma(
            persist_directory = persist_dir,
            embedding_function = embeddings,
        )
    return None


# =========================================================
# STEP 5 : RETRIEVER  (MMR for diversity)
# =========================================================

def build_retriever(vectorstore: Chroma):
    """
    MMR = Maximal Marginal Relevance.
    Fetches `fetch_k` candidates then selects `k` that balance
    relevance AND diversity (avoids near-duplicate chunks).
    """
    return vectorstore.as_retriever(
        search_type   = "mmr",
        search_kwargs = {
            "k":       CONFIG["retriever_k"],
            "fetch_k": CONFIG["retriever_fetch"],
        },
    )


# =========================================================
# STEP 6 : QUERY REWRITING
# =========================================================

_rewrite_prompt = ChatPromptTemplate.from_template(
    """You are a search query optimizer.
Rewrite the user's question into a clearer, more specific search query.
Return ONLY the rewritten query with no explanation.

Original question: {question}
Rewritten query:"""
)

def rewrite_query(question: str, llm: ChatOpenAI) -> str:
    chain  = _rewrite_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})
    return result.strip()


# =========================================================
# STEP 7 : PROMPTS
# =========================================================

# ── Main RAG prompt with conversation history ─────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a knowledgeable AI assistant.
Answer the user's question using the provided context.
- Synthesize and summarize information from the chunks, don't look for exact matches.
- If the context has PARTIAL information, use it to give the best possible answer.
- Only say "I could not find the answer" if the context has ZERO relevant information.
- Be helpful and extract key points from the retrieved chunks.

CONTEXT:
{context}

SOURCE REFERENCES:
{sources}"""
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


# =========================================================
# STEP 8 : CHAT HISTORY  (in-memory + JSON persistence)
# =========================================================

class ChatHistory:
    def __init__(self, filepath: str):
        self.filepath   = filepath
        self.messages: List[Dict] = []   # raw dicts for JSON
        self._load()

    def add(self, role: str, content: str):
        self.messages.append({
            "role":      role,
            "content":   content,
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def langchain_messages(self, last_n: int):
        """Return last_n turns as LangChain message objects."""
        recent = self.messages[-(last_n * 2):]      # 2 messages per turn
        lc_msgs = []
        for m in recent:
            if m["role"] == "human":
                lc_msgs.append(HumanMessage(content=m["content"]))
            else:
                lc_msgs.append(AIMessage(content=m["content"]))
        return lc_msgs

    def clear(self):
        self.messages = []
        self._save()
        success("Chat history cleared.")

    def display(self):
        if not self.messages:
            warn("No chat history yet.")
            return
        banner("CHAT HISTORY", C.MAGENTA)
        for m in self.messages:
            role  = m["role"].upper()
            ts    = m["timestamp"][:19].replace("T", " ")
            color = C.CYAN if role == "HUMAN" else C.GREEN
            print(f"{color}{C.BOLD}[{ts}] {role}:{C.RESET}")
            print(f"  {m['content']}\n")

    def _save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _load(self):
        try:
            if Path(self.filepath).exists():
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.messages = json.load(f)
                info(f"Loaded {len(self.messages)} messages from {self.filepath}")
        except Exception:
            self.messages = []


# =========================================================
# STEP 9 : COST TRACKER
# =========================================================

class CostTracker:
    def __init__(self):
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.total_queries       = 0

    def add(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens  += input_tokens
        self.total_output_tokens += output_tokens
        self.total_queries       += 1

    @property
    def cost_usd(self) -> float:
        return (
            self.total_input_tokens  / 1000 * CONFIG["cost_per_1k_input"] +
            self.total_output_tokens / 1000 * CONFIG["cost_per_1k_output"]
        )

    def display(self):
        banner("USAGE & COST", C.YELLOW)
        print(f"  Queries run      : {self.total_queries}")
        print(f"  Input  tokens    : {self.total_input_tokens:,}")
        print(f"  Output tokens    : {self.total_output_tokens:,}")
        print(f"  Estimated cost   : ${self.cost_usd:.5f} USD\n")


# =========================================================
# STEP 10 : COMMAND HANDLER
# =========================================================

COMMANDS = {
    "/help":     "Show this help message",
    "/history":  "Print the full chat history",
    "/clear":    "Clear the chat history",
    "/sources":  "Show documents loaded into the knowledge base",
    "/cost":     "Show token usage and estimated cost",
    "/exit":     "Exit the chatbot",
}

def show_help():
    banner("AVAILABLE COMMANDS", C.YELLOW)
    for cmd, desc in COMMANDS.items():
        print(f"  {C.BOLD}{C.YELLOW}{cmd:<12}{C.RESET}  {desc}")
    print()

def handle_command(
    cmd: str,
    history: ChatHistory,
    tracker: CostTracker,
    chunks: List[Document],
) -> bool:
    """Returns True if the chatbot should exit."""
    cmd = cmd.strip().lower()

    if cmd == "/help":
        show_help()
    elif cmd == "/history":
        history.display()
    elif cmd == "/clear":
        history.clear()
    elif cmd == "/sources":
        banner("LOADED SOURCES", C.MAGENTA)
        seen = set()
        for c in chunks:
            label = c.metadata.get("source_label", "unknown")
            if label not in seen:
                seen.add(label)
                print(f"  {C.CYAN}•{C.RESET} {label}")
        print(f"\n  Total chunks : {len(chunks)}\n")
    elif cmd == "/cost":
        tracker.display()
    elif cmd in ("/exit", "/quit"):
        return True
    else:
        warn(f"Unknown command: {cmd}  (type /help for a list)")

    return False


# =========================================================
# STEP 11 : CORE RAG QUERY
# =========================================================

def rag_query(
    question:  str,
    retriever,
    llm:       ChatOpenAI,
    history:   ChatHistory,
    tracker:   CostTracker,
) -> str:
    """Full RAG pipeline: rewrite → retrieve → augment → generate."""

    # ── 1. Query rewriting ────────────────────────────
    info("Rewriting query for better retrieval …")
    rewritten = rewrite_query(question, llm)
    if rewritten.lower() != question.lower():
        print(f"  {C.GREY}Rewritten: {rewritten}{C.RESET}")

    # ── 2. Retrieve ───────────────────────────────────
    retrieved_docs = retriever.invoke(rewritten)

    # ── 3. Build context + source labels ─────────────
    context_parts: List[str] = []
    source_labels: List[str] = []

    banner("RETRIEVED CHUNKS", C.BLUE)
    for i, doc in enumerate(retrieved_docs):
        label   = doc.metadata.get("source_label", f"chunk_{i}")
        chunk_id = doc.metadata.get("chunk_id", i)
        print(f"{C.BOLD}{C.BLUE}── Source [{i+1}] : {label}  (chunk #{chunk_id}){C.RESET}")
        preview = doc.page_content[:250].replace("\n", " ")
        print(f"   {C.GREY}{preview}…{C.RESET}\n")
        context_parts.append(f"[{i+1}] {doc.page_content}")
        source_labels.append(f"[{i+1}] {label}")

    context_text = "\n\n".join(context_parts)
    sources_text  = "\n".join(source_labels)

    # ── 4. Estimate input tokens ──────────────────────
    history_msgs  = history.langchain_messages(CONFIG["memory_turns"])
    prompt_text   = context_text + sources_text + question
    input_tokens  = _estimate_tokens(prompt_text)

    # ── 5. Build final prompt ─────────────────────────
    final_prompt = RAG_PROMPT.invoke({
        "context":  context_text,
        "sources":  sources_text,
        "history":  history_msgs,
        "question": question,
    })

    # ── 6. Generate (streaming or batch) ─────────────
    banner("ANSWER", C.GREEN)
    answer_chunks: List[str] = []

    if CONFIG["streaming"]:
        for chunk in llm.stream(final_prompt):
            token = chunk.content
            print(f"{C.WHITE}{token}{C.RESET}", end="", flush=True)
            answer_chunks.append(token)
        print("\n")
        final_answer = "".join(answer_chunks)
    else:
        response     = llm.invoke(final_prompt)
        final_answer = StrOutputParser().invoke(response)
        print(f"{C.WHITE}{final_answer}{C.RESET}\n")

    # ── 7. Track tokens / cost ────────────────────────
    output_tokens = _estimate_tokens(final_answer)
    tracker.add(input_tokens, output_tokens)

    # ── 8. Persist to history ─────────────────────────
    history.add("human", question)
    history.add("assistant", final_answer)

    return final_answer


# =========================================================
# MAIN  — STARTUP & CHAT LOOP
# =========================================================

def main():
    banner("ENHANCED RAG CHATBOT", C.CYAN)

    # ── Ask for document source ───────────────────────
    print(f"{C.BOLD}Provide the path to your documents (file, directory, or URL).{C.RESET}")
    print(f"{C.GREY}Supported types: .txt  .pdf  .csv  https://…{C.RESET}\n")

    source = input(f"{C.CYAN}Document source: {C.RESET}").strip()

    # ── Load documents ────────────────────────────────
    try:
        documents = detect_and_load(source)
    except (FileNotFoundError, ValueError) as e:
        error(str(e))
        sys.exit(1)

    success(f"Loaded {len(documents)} document(s).")

    # ── Chunk ─────────────────────────────────────────
    chunks = chunk_documents(documents)
    success(f"Split into {len(chunks)} chunks.")

    # ── Vector store ──────────────────────────────────
    rebuild_choice = "y"
    existing = load_vectorstore()
    if existing:
        rebuild_choice = input(
            f"{C.YELLOW}Existing vector store found. Rebuild? [y/N]: {C.RESET}"
        ).strip().lower()

    if rebuild_choice == "y" or existing is None:
        vectorstore = build_vectorstore(chunks)
        success("Vector store built and persisted.")
    else:
        vectorstore = existing
        success("Using existing vector store.")

    # ── Retriever & LLM ───────────────────────────────
    retriever = build_retriever(vectorstore)
    llm       = ChatOpenAI(
        model       = CONFIG["model"],
        temperature = CONFIG["temperature"],
        streaming   = CONFIG["streaming"],
    )
    success(f"LLM ready  ({CONFIG['model']})")

    # ── History & tracker ─────────────────────────────
    history = ChatHistory(CONFIG["history_file"])
    tracker = CostTracker()

    # ── Welcome ───────────────────────────────────────
    banner("CHATBOT READY — type /help for commands", C.GREEN)

    # ── Chat loop ─────────────────────────────────────
    while True:
        try:
            question = input(f"{C.BOLD}{C.CYAN}You: {C.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not question:
            continue

        # Commands start with /
        if question.startswith("/"):
            should_exit = handle_command(question, history, tracker, chunks)
            if should_exit:
                break
            continue

        # Conversational shortcut: bare "exit" or "quit"
        if question.lower() in ("exit", "quit"):
            break

        try:
            rag_query(question, retriever, llm, history, tracker)
        except Exception as e:
            error(f"Query failed: {e}")

    # ── Goodbye ───────────────────────────────────────
    tracker.display()
    banner("Goodbye! Chat saved to " + CONFIG["history_file"], C.MAGENTA)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()


# =========================================================
# END
# =========================================================