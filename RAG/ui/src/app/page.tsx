"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import {
  Send, Upload, Trash2, Database,
  DollarSign, Activity, FileText, ChevronRight, Zap
} from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────
type Message = { role: string; content: string; id: string; sources?: string[] };

// ─── Typing cursor for streaming ─────────────────────────────────────────────
function TypingCursor() {
  return (
    <span
      className="inline-block w-[2px] h-[1em] bg-amber-400 ml-0.5 align-middle"
      style={{ animation: "blink 0.9s step-end infinite" }}
    />
  );
}

// ─── Source chip ─────────────────────────────────────────────────────────────
function SourceChip({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-amber-400/10 border border-amber-400/20 text-amber-400 text-[10px] font-mono tracking-wider mr-1 mb-1">
      <FileText className="w-2.5 h-2.5 shrink-0" />
      {label}
    </span>
  );
}

// ─── Animated stat block ──────────────────────────────────────────────────────
function StatBlock({ icon: Icon, label, value }: { icon: any; label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1 p-4 rounded-xl bg-[#0a0a0f] border border-white/5">
      <span className="flex items-center gap-1.5 text-[9px] font-mono uppercase tracking-[0.15em] text-zinc-500">
        <Icon className="w-2.5 h-2.5" />
        {label}
      </span>
      <span
        className="text-2xl font-display font-light text-white tabular-nums"
        style={{ fontFamily: "var(--font-dm-mono)" }}
      >
        {value}
      </span>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function Home() {
  const [source, setSource] = useState("");
  const [loading, setLoading] = useState(false);
  const [indexSuccess, setIndexSuccess] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [chatting, setChatting] = useState(false);
  const [costQueries, setCostQueries] = useState(0);
  const [costUsd, setCostUsd] = useState(0);
  const [streamingId, setStreamingId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // ── Load history on mount ──
  useEffect(() => {
    fetch("http://localhost:8000/api/history")
      .then(r => r.json())
      .then(data => {
        if (data.messages) {
          setMessages(
            data.messages.map((m: any, i: number) => ({
              role: m.role,
              content: m.content,
              id: `hist-${i}`,
            }))
          );
        }
        setCostQueries(data.cost_queries || 0);
        setCostUsd(data.cost_usd || 0);
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Build vectorstore ──
  const handleLoadVectorStore = async () => {
    if (!source.trim()) return;
    setLoading(true);
    setIndexSuccess(false);
    try {
      const res = await fetch("http://localhost:8000/api/vectorstore/build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source }),
      });
      const data = await res.json();
      if (data.success) {
        setIndexSuccess(true);
        setTimeout(() => setIndexSuccess(false), 4000);
      } else {
        alert("Error: " + data.error);
      }
    } catch {
      alert("Failed to build vector store. Is the API running?");
    }
    setLoading(false);
  };

  // ── Clear history ──
  const handleClearHistory = async () => {
    try {
      await fetch("http://localhost:8000/api/history", { method: "DELETE" });
      setMessages([]);
    } catch (e) {
      console.error(e);
    }
  };

  // ── Send message ──
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || chatting) return;

    const userMsg = input.trim();
    setInput("");

    const newMsgId = `human-${Date.now()}`;
    const botMsgId = `bot-${Date.now()}`;

    setMessages(prev => [...prev, { role: "human", content: userMsg, id: newMsgId }]);
    setChatting(true);
    setStreamingId(botMsgId);

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      setMessages(prev => [
        ...prev,
        { role: "assistant", content: "", id: botMsgId, sources: [] },
      ]);

      let botContent = "";
      let collectedSources: string[] = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (!value) continue;

        const raw = decoder.decode(value, { stream: true });
        const blocks = raw.split("\n\n");

        for (const block of blocks) {
          if (!block.startsWith("data: ")) continue;
          const dataStr = block.replace("data: ", "").trim();
          if (!dataStr || dataStr === "[DONE]") continue;

          try {
            const data = JSON.parse(dataStr);
            if (data.error) {
              botContent += `\n**Error:** ${data.error}`;
            } else if (data.chunk) {
              botContent += data.chunk;
            } else if (data.source) {
              collectedSources = [...new Set([...collectedSources, data.source])];
            } else if (data.cost_queries !== undefined) {
              setCostQueries(data.cost_queries);
              setCostUsd(data.cost_usd);
            }

            setMessages(prev =>
              prev.map(m =>
                m.id === botMsgId
                  ? { ...m, content: botContent, sources: collectedSources }
                  : m
              )
            );
          } catch {
            // ignore parse errors
          }
        }
      }
    } catch {
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: "**Error generating response.** Make sure the API is running.",
          id: botMsgId,
        },
      ]);
    } finally {
      setChatting(false);
      setStreamingId(null);
      inputRef.current?.focus();
    }
  };

  // ─────────────────────────────────────────────────────────────────────────────
  return (
    <>
      {/* Global styles */}
      <style>{`
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes fadeSlideUp {
          from { opacity:0; transform:translateY(12px) }
          to   { opacity:1; transform:translateY(0) }
        }
        @keyframes shimmer {
          0%   { background-position: -400px 0 }
          100% { background-position:  400px 0 }
        }
        @keyframes gradientShift {
          0%,100% { background-position: 0% 50% }
          50%     { background-position: 100% 50% }
        }
        .msg-enter { animation: fadeSlideUp 0.3s ease both }
        .font-display { font-family: var(--font-syne) }
        .font-mono-dm { font-family: var(--font-dm-mono) }

        /* Scrollbar */
        .chat-scroll::-webkit-scrollbar { width: 4px }
        .chat-scroll::-webkit-scrollbar-track { background: transparent }
        .chat-scroll::-webkit-scrollbar-thumb { background: #ffffff0f; border-radius: 99px }
        .chat-scroll::-webkit-scrollbar-thumb:hover { background: #ffffff1a }

        /* Animated border on sidebar input focus */
        .glow-input:focus { box-shadow: 0 0 0 2px #f59e0b33, 0 0 20px #f59e0b0d }

        /* Prose overrides */
        .rag-prose p  { margin:0 0 0.6em }
        .rag-prose p:last-child { margin-bottom:0 }
        .rag-prose code { font-family: var(--font-dm-mono); font-size:0.82em }
        .rag-prose pre  { background:#0a0a0f; border:1px solid #ffffff0f; border-radius:10px; padding:1rem; overflow-x:auto }

        /* Gradient header text */
        .title-gradient {
          background: linear-gradient(135deg, #fff 30%, #f59e0b 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
      `}</style>

      <div className="flex h-screen overflow-hidden" style={{ fontFamily: "var(--font-syne)" }}>

        {/* ── Sidebar ─────────────────────────────────────────────────────── */}
        <aside className="w-72 shrink-0 flex flex-col bg-[#0f0f16] border-r border-white/[0.05]">

          {/* Logo */}
          <div className="px-6 pt-7 pb-6 border-b border-white/[0.04]">
            <div className="flex items-center gap-3">
              <div className="relative w-9 h-9 rounded-xl bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center shadow-lg shadow-amber-500/25 shrink-0">
                <Zap className="w-4.5 h-4.5 text-white" strokeWidth={2.5} />
              </div>
              <div>
                <h1 className="font-display font-bold text-[15px] text-white title-gradient leading-none">
                  RAG Engine
                </h1>
                <p className="text-[10px] font-mono-dm text-zinc-500 mt-0.5 tracking-widest uppercase">
                  v2.0 · Modern Pipeline
                </p>
              </div>
            </div>
          </div>

          {/* Data source */}
          <div className="px-5 py-6 border-b border-white/[0.04] space-y-3">
            <label className="flex items-center gap-2 text-[9px] font-mono-dm uppercase tracking-[0.18em] text-zinc-500 mb-3">
              <Upload className="w-3 h-3" />
              Data Source
            </label>

            <input
              type="text"
              placeholder="URL or /path/to/dir"
              className="glow-input w-full px-4 py-3 rounded-xl bg-[#0a0a0f] border border-white/[0.07] text-sm text-zinc-300 placeholder:text-zinc-600 focus:outline-none transition-all font-mono-dm"
              style={{ fontFamily: "var(--font-dm-mono)", fontSize: "12px" }}
              value={source}
              onChange={e => setSource(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleLoadVectorStore()}
            />

            <button
              onClick={handleLoadVectorStore}
              disabled={loading || !source.trim()}
              className="w-full relative overflow-hidden rounded-xl py-3 text-sm font-display font-semibold tracking-wide transition-all
                disabled:opacity-40 disabled:cursor-not-allowed
                bg-gradient-to-r from-amber-400 to-orange-400 text-[#0a0a0f]
                hover:from-amber-300 hover:to-orange-300 active:scale-[0.98]
                shadow-lg shadow-amber-500/20"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-[#0a0a0f]/30 border-t-[#0a0a0f] rounded-full animate-spin" />
                  Indexing…
                </span>
              ) : indexSuccess ? (
                <span className="flex items-center justify-center gap-2">✓ Indexed!</span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  <Database className="w-3.5 h-3.5" /> Load & Index
                </span>
              )}
            </button>
          </div>

          {/* Stats */}
          <div className="px-5 py-5 border-b border-white/[0.04]">
            <label className="flex items-center gap-2 text-[9px] font-mono-dm uppercase tracking-[0.18em] text-zinc-500 mb-3">
              <Activity className="w-3 h-3" />
              Session Stats
            </label>
            <div className="grid grid-cols-2 gap-2">
              <StatBlock icon={FileText} label="Queries" value={String(costQueries)} />
              <StatBlock icon={DollarSign} label="Cost" value={`$${costUsd.toFixed(4)}`} />
            </div>
          </div>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Clear history */}
          <div className="px-5 py-5">
            <button
              onClick={handleClearHistory}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-xl text-xs font-display font-semibold tracking-wide
                text-red-400 border border-red-500/15 bg-red-500/[0.04]
                hover:bg-red-500/10 hover:border-red-500/30 transition-all"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear History
            </button>
          </div>
        </aside>

        {/* ── Chat area ────────────────────────────────────────────────────── */}
        <main className="flex-1 flex flex-col overflow-hidden bg-[#0a0a0f]">

          {/* Top bar */}
          <header className="shrink-0 flex items-center justify-between px-8 py-4 border-b border-white/[0.04]">
            <div className="flex items-center gap-2.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-lg shadow-emerald-400/50" />
              <span className="text-xs font-mono-dm text-zinc-500 tracking-widest uppercase">
                {chatting ? "Processing…" : "Ready"}
              </span>
            </div>
            <span className="text-[10px] font-mono-dm text-zinc-700 tracking-wide">
              {messages.filter(m => m.role === "human").length} messages
            </span>
          </header>

          {/* Messages */}
          <div className="chat-scroll flex-1 overflow-y-auto px-6 md:px-16 lg:px-24 py-8 space-y-6">
            {messages.length === 0 ? (

              /* ── Empty state ── */
              <div className="h-full min-h-[60vh] flex flex-col items-center justify-center select-none">
                <div
                  className="w-16 h-16 rounded-2xl mb-6 flex items-center justify-center
                    bg-gradient-to-br from-amber-400/20 to-orange-500/10
                    border border-amber-400/20 shadow-2xl shadow-amber-500/10"
                >
                  <Zap className="w-7 h-7 text-amber-400" />
                </div>
                <h2 className="font-display font-bold text-2xl text-white/80 mb-2 tracking-tight">
                  Ask your documents
                </h2>
                <p className="text-sm text-zinc-500 text-center max-w-sm leading-relaxed">
                  Load a document source in the sidebar, then ask anything.
                  The RAG pipeline will retrieve and synthesise an answer.
                </p>
                <div className="mt-8 flex flex-wrap items-center justify-center gap-2">
                  {["What is this document about?", "Summarise key points", "Find relevant sections"].map(q => (
                    <button
                      key={q}
                      onClick={() => setInput(q)}
                      className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-white/[0.03] border border-white/[0.06]
                        text-xs text-zinc-400 hover:text-white hover:border-amber-400/30 hover:bg-amber-400/[0.05] transition-all"
                    >
                      <ChevronRight className="w-3 h-3 text-amber-400" />
                      {q}
                    </button>
                  ))}
                </div>
              </div>

            ) : messages.map((msg, idx) => (

              /* ── Message bubble ── */
              <div
                key={msg.id}
                className={`msg-enter flex w-full gap-3 ${msg.role === "human" ? "justify-end" : "justify-start"}`}
                style={{ animationDelay: `${idx < 3 ? 0 : 0}ms` }}
              >
                {/* Avatar — assistant */}
                {msg.role === "assistant" && (
                  <div className="shrink-0 w-7 h-7 mt-1 rounded-lg bg-gradient-to-br from-amber-400 to-orange-500
                    flex items-center justify-center shadow-md shadow-amber-500/20">
                    <Zap className="w-3.5 h-3.5 text-[#0a0a0f]" strokeWidth={2.5} />
                  </div>
                )}

                <div className={`flex flex-col gap-1.5 max-w-[78%]`}>
                  {/* Sources row */}
                  {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                    <div className="flex flex-wrap px-1">
                      {msg.sources.map(s => <SourceChip key={s} label={s} />)}
                    </div>
                  )}

                  {/* Bubble */}
                  <div
                    className={`rounded-2xl px-5 py-4 text-sm leading-relaxed
                      ${msg.role === "human"
                        ? "bg-white/[0.06] border border-white/[0.08] text-zinc-100 rounded-tr-sm"
                        : "bg-[#0f0f16] border border-white/[0.06] text-zinc-300 rounded-tl-sm"
                      }`}
                  >
                    <div className="rag-prose">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                      {streamingId === msg.id && msg.content !== "" && <TypingCursor />}
                    </div>

                    {/* Streaming skeleton when content is empty */}
                    {streamingId === msg.id && msg.content === "" && (
                      <div className="space-y-2 py-1">
                        {[80, 65, 40].map((w, i) => (
                          <div
                            key={i}
                            className="h-3 rounded-full bg-gradient-to-r from-white/[0.04] via-white/[0.09] to-white/[0.04]"
                            style={{
                              width: `${w}%`,
                              backgroundSize: "400px 100%",
                              animation: `shimmer 1.4s infinite linear`,
                              animationDelay: `${i * 0.15}s`,
                            }}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Avatar — human */}
                {msg.role === "human" && (
                  <div className="shrink-0 w-7 h-7 mt-1 rounded-lg bg-zinc-700/60 border border-white/[0.08]
                    flex items-center justify-center text-[10px] font-display font-bold text-zinc-300">
                    U
                  </div>
                )}
              </div>
            ))}

            <div ref={messagesEndRef} className="h-4" />
          </div>

          {/* ── Input bar ────────────────────────────────────────────────── */}
          <div className="shrink-0 px-6 md:px-16 lg:px-24 py-5 border-t border-white/[0.04]">
            <form onSubmit={handleSendMessage} className="relative max-w-3xl mx-auto">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder="Ask anything about your documents…"
                disabled={chatting}
                className="w-full rounded-xl pl-5 pr-14 py-4 text-sm
                  bg-[#0f0f16] border border-white/[0.08] text-zinc-100
                  placeholder:text-zinc-600 font-mono-dm
                  focus:outline-none focus:border-amber-400/40 focus:shadow-[0_0_0_3px_#f59e0b14]
                  disabled:opacity-50 transition-all"
                style={{ fontFamily: "var(--font-dm-mono)", fontSize: "13px" }}
              />
              <button
                type="submit"
                disabled={!input.trim() || chatting}
                className="absolute right-2.5 top-2.5 w-9 h-9 rounded-lg
                  bg-gradient-to-br from-amber-400 to-orange-400
                  flex items-center justify-center
                  disabled:opacity-30 disabled:cursor-not-allowed
                  hover:from-amber-300 hover:to-orange-300
                  active:scale-95 transition-all shadow-md shadow-amber-500/20"
              >
                {chatting
                  ? <span className="w-4 h-4 border-2 border-[#0a0a0f]/30 border-t-[#0a0a0f] rounded-full animate-spin" />
                  : <Send className="w-4 h-4 text-[#0a0a0f]" strokeWidth={2.5} />
                }
              </button>
            </form>
            <p className="text-center mt-3 text-[10px] font-mono-dm tracking-widest uppercase text-zinc-700">
              Responses are machine-generated · RAG Engine 2.0
            </p>
          </div>
        </main>
      </div>
    </>
  );
}