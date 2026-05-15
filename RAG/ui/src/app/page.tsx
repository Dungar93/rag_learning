"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { Send, Upload, Trash2, Database, DollarSign, Activity, FileText } from "lucide-react";

export default function Home() {
  const [source, setSource] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<{ role: string; content: string; id: string }[]>([]);
  const [input, setInput] = useState("");
  const [chatting, setChatting] = useState(false);
  
  const [costQueries, setCostQueries] = useState(0);
  const [costUsd, setCostUsd] = useState(0);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/history")
      .then(res => res.json())
      .then(data => {
        if (data.messages) {
          setMessages(data.messages.map((m: any, i: number) => ({ role: m.role, content: m.content, id: `hist-${i}` })));
        }
        setCostQueries(data.cost_queries || 0);
        setCostUsd(data.cost_usd || 0);
      })
      .catch(console.error);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => scrollToBottom(), [messages]);

  const handleLoadVectorStore = async () => {
    if (!source.trim()) return;
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/api/vectorstore/build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source })
      });
      const data = await response.json();
      if (data.success) {
        alert(data.message);
      } else {
        alert("Error: " + data.error);
      }
    } catch (e) {
      alert("Failed to build vector store. Check if API is running.");
      console.error(e);
    }
    setLoading(false);
  };

  const handleClearHistory = async () => {
    try {
      await fetch("http://localhost:8000/api/history", { method: "DELETE" });
      setMessages([]);
    } catch (e) {
      console.error("Failed to clear", e);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || chatting) return;

    const userMsg = input.trim();
    setInput("");
    
    // Add user message to UI immediately
    const newMsgId = `human-${Date.now()}`;
    const botMsgId = `bot-${Date.now()}`;
    setMessages(prev => [...prev, { role: "human", content: userMsg, id: newMsgId }]);
    
    setChatting(true);

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg })
      });

      if (!response.body) throw new Error("No response body");
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      setMessages(prev => [...prev, { role: "assistant", content: "", id: botMsgId }]);
      let botContent = "";
      
      let done = false;
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (!value) continue;

        const chunk = decoder.decode(value, { stream: true });
        // The SSE backend might send multiple 'data: ...\n\n' chunks in one read
        const blocks = chunk.split('\n\n');
        
        for (const block of blocks) {
          if (!block || !block.startsWith('data: ')) continue;
          
          const dataStr = block.replace('data: ', '').trim();
          if (!dataStr) continue;
          if (dataStr === '[DONE]') continue;

          try {
            const data = JSON.parse(dataStr);
            if (data.error) {
              botContent += `\n**Error:** ${data.error}`;
            } else if (data.chunk) {
              botContent += data.chunk;
            } else if (data.source) {
              botContent = `_[Source: ${data.source}]_\n\n` + botContent;
            } else if (data.cost_queries !== undefined) {
              setCostQueries(data.cost_queries);
              setCostUsd(data.cost_usd);
            }
            
            // Re-render
            setMessages(prev => prev.map(m => m.id === botMsgId ? { ...m, content: botContent } : m));
          } catch (err) {
             console.error("Parse error:", dataStr, err);
          }
        }
      }
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: "assistant", content: "**Error generating response.** Make sure API is running.", id: botMsgId }]);
    } finally {
      setChatting(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100 font-sans">
      {/* Sidebar Focus Area */}
      <div className="w-80 bg-gray-900 border-r border-gray-800 p-6 flex flex-col z-20 shadow-2xl">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2.5 bg-blue-600 rounded-xl shadow-lg shadow-blue-500/20">
            <Database className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-white">RAG Engine</h1>
        </div>

        <div className="space-y-6 flex-1">
          <div className="space-y-4">
            <h2 className="text-xs font-bold uppercase tracking-widest text-gray-500 flex items-center gap-2">
              <Upload className="w-4 h-4" /> Data Source
            </h2>
            <div className="group">
              <input
                type="text"
                placeholder="URL, or /path/to/dir..."
                className="w-full px-4 py-3.5 bg-gray-950 border border-gray-800 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-600 focus:border-transparent transition-all placeholder:text-gray-600 text-gray-300"
                value={source}
                onChange={(e) => setSource(e.target.value)}
              />
            </div>
            <button 
              onClick={handleLoadVectorStore}
              disabled={loading || !source}
              className="w-full bg-white hover:bg-gray-100 disabled:bg-gray-800 disabled:text-gray-500 text-gray-900 font-semibold py-3.5 rounded-xl transition-all shadow-md active:scale-[0.98] flex justify-center border border-gray-200 disabled:border-transparent"
            >
              {loading ? (
                <span className="w-5 h-5 border-2 border-gray-900/30 border-t-gray-900 rounded-full animate-spin" />
              ) : "Load & Index Data"}
            </button>
          </div>

          <div className="h-px w-full bg-gradient-to-r from-transparent via-gray-800 to-transparent my-8" />

          <div className="space-y-4 bg-gray-950/50 p-5 rounded-2xl border border-gray-800/80 shadow-inner">
            <h2 className="text-xs font-bold uppercase tracking-widest text-gray-500 flex items-center gap-2 mb-4">
              <Activity className="w-3 h-3" /> Usage Stats
            </h2>
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col">
                <span className="text-gray-500 text-xs font-semibold mb-1 flex items-center gap-1.5 uppercase tracking-widest"><FileText className="w-3 h-3"/> Queries</span>
                <span className="text-2xl font-light text-white">{costQueries}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-gray-500 text-xs font-semibold mb-1 flex items-center gap-1.5 uppercase tracking-widest"><DollarSign className="w-3 h-3"/> Cost</span>
                <span className="text-2xl font-light text-white">${costUsd.toFixed(4)}</span>
              </div>
            </div>
          </div>
        </div>

        <button 
          onClick={handleClearHistory}
          className="mt-auto flex items-center justify-center gap-2 w-full py-3.5 px-4 bg-red-500/10 hover:bg-red-500/20 text-red-400 font-semibold rounded-xl transition-colors border border-red-500/20"
        >
          <Trash2 className="w-4 h-4" /> Clear History
        </button>
      </div>

      {/* Main Chat View */}
      <div className="flex-1 flex flex-col bg-gray-950 relative overflow-hidden">
        
        {/* Chat History Flow */}
        <div className="flex-1 overflow-y-auto px-4 md:px-8 lg:px-24 py-12 space-y-6 scroll-smooth z-10 custom-scrollbar">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-500 space-y-6 animate-pulse">
              <div className="w-20 h-20 rounded-2xl bg-gray-900 border border-gray-800 flex items-center justify-center shadow-2xl mb-4 text-4xl shadow-blue-500/5 hover:scale-105 transition-transform duration-500">🤖</div>
              <h3 className="text-3xl font-light text-gray-400 tracking-tight">How can I help you today?</h3>
              <p className="text-sm font-medium tracking-wide">Enter a document source in the sidebar to begin RAG processing.</p>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className={`flex w-full ${msg.role === 'human' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] md:max-w-[75%] rounded-3xl px-7 py-5 shadow-sm text-[15px] leading-relaxed
                  ${msg.role === 'human' 
                    ? 'bg-blue-600 text-white rounded-br-sm' 
                    : 'bg-gray-900 text-gray-300 rounded-bl-sm border border-gray-800'
                  }`}>
                  <div className={`prose ${msg.role === 'human' ? 'prose-invert prose-p:text-white prose-a:text-blue-200' : 'prose-invert prose-p:text-gray-300 prose-strong:text-gray-100 prose-a:text-blue-400 prose-code:text-emerald-400 '} max-w-none break-words`}>
                    <ReactMarkdown>
                      {msg.content}
                    </ReactMarkdown>
                  </div>
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} className="h-8" />
        </div>

        {/* Input Text Box */}
        <div className="p-6 bg-gradient-to-t from-gray-950 via-gray-950 to-transparent z-20 w-full">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={handleSendMessage} className="relative group">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask your assistant anything..."
                className="w-full bg-gray-900 border border-gray-800 focus:border-blue-600 focus:bg-gray-900/80 rounded-2xl pl-6 pr-16 py-4 focus:outline-none focus:ring-4 focus:ring-blue-600/10 text-gray-100 placeholder-gray-500 transition-all shadow-xl"
                disabled={chatting}
              />
              <button
                type="submit"
                disabled={!input.trim() || chatting}
                className="absolute right-3 top-2.5 p-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 text-white rounded-xl transition-all active:scale-[0.97] disabled:scale-100 disabled:opacity-50"
              >
                {chatting ? <span className="w-5 h-5 block border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Send className="w-5 h-5 ml-0.5" />}
              </button>
            </form>
            <div className="text-center mt-4 text-[11px] font-medium tracking-wide text-gray-600 uppercase">
              RAG Engine uses locally executed chunking. Responses are machine generated.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
