import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   RAG SYSTEM / ENTERPRISE SEARCH — ML System Design Reference (Google L6)
   Pearl white theme · 17 sections (HLD + LLD + ML Deep Dive)
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",                icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",            icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",     icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",              icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",       icon: "🏗️", color: "#9333ea" },
  { id: "chunking",      label: "Chunking Strategies",     icon: "✂️", color: "#c026d3" },
  { id: "embedding",     label: "Embedding & Indexing",    icon: "🔢", color: "#dc2626" },
  { id: "retrieval",     label: "Retrieval Pipeline",      icon: "🔍", color: "#d97706" },
  { id: "reranking",     label: "Re-Ranking",              icon: "🏆", color: "#0f766e" },
  { id: "generation",    label: "Generation & Grounding",  icon: "🧠", color: "#ea580c" },
  { id: "data",          label: "Data Model",              icon: "🗄️", color: "#059669" },
  { id: "ingestion",     label: "Ingestion Pipeline",      icon: "🔄", color: "#7e22ce" },
  { id: "evaluation",    label: "Evaluation & Quality",    icon: "📊", color: "#0284c7" },
  { id: "scalability",   label: "Scalability",             icon: "📈", color: "#059669" },
  { id: "watchouts",     label: "Failure Modes",           icon: "⚠️", color: "#dc2626" },
  { id: "enhancements",  label: "Enhancements",            icon: "🚀", color: "#7c3aed" },
  { id: "followups",     label: "Follow-up Questions",     icon: "❓", color: "#6366f1" },
];

/* ——— Reusable Components ——— */
const Card = ({ children, className = "", accent }) => (
  <div className={`bg-white rounded-xl border border-stone-200 p-5 shadow-sm ${className}`}
    style={accent ? { borderTop: `3px solid ${accent}` } : {}}>
    {children}
  </div>
);
const Label = ({ color = "#6366f1", children }) => (
  <div className="text-[10px] font-bold uppercase tracking-[0.12em] mb-2.5" style={{ color }}>{children}</div>
);
const Pill = ({ bg = "#f3f4f6", color = "#374151", children }) => (
  <span className="text-[10px] font-bold px-2.5 py-0.5 rounded-full" style={{ background: bg, color }}>{children}</span>
);
const Point = ({ icon = "›", color = "#6366f1", children }) => (
  <li className="flex items-start gap-2.5 text-[13px] text-stone-600 leading-relaxed">
    <span className="mt-0.5 shrink-0 font-bold" style={{ color }}>{icon}</span>
    <span>{children}</span>
  </li>
);

function MathStep({ step, formula, result, note, final: isFinal }) {
  return (
    <div className={`flex items-start gap-3 py-2.5 ${isFinal ? "bg-violet-50 -mx-2 px-4 rounded-lg border border-violet-200" : "border-b border-stone-100"}`}>
      <span className={`text-[10px] font-bold w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${isFinal ? "bg-violet-600 text-white" : "bg-stone-200 text-stone-500"}`}>{step}</span>
      <div className="flex-1 min-w-0">
        <div className="font-mono text-[12px] text-stone-700">{formula}</div>
        {note && <div className="text-[11px] text-stone-400 mt-0.5">{note}</div>}
      </div>
      <div className={`font-mono text-[13px] font-bold shrink-0 ${isFinal ? "text-violet-700" : "text-stone-700"}`}>{result}</div>
    </div>
  );
}

function CodeBlock({ title, code }) {
  const lines = code.split("\n");
  return (
    <div className="bg-stone-50 border border-stone-200 rounded-lg p-3.5 overflow-x-auto">
      {title && <div className="text-[10px] font-bold text-stone-400 uppercase tracking-[0.1em] mb-2">{title}</div>}
      <pre className="font-mono text-[11.5px] leading-[1.75]" style={{ whiteSpace: "pre" }}>
        {lines.map((line, i) => (
          <div key={i} className={`px-2 rounded ${line.trim().startsWith("#") || line.trim().startsWith("//") || line.trim().startsWith("--") ? "text-stone-400" : "text-stone-700"}`}>
            <span className="inline-block w-5 text-right mr-3 text-stone-300 select-none">{line.trim() ? i + 1 : ""}</span>{line}
          </div>
        ))}
      </pre>
    </div>
  );
}


/* ═══════════════════════════════════════════════════════════
   SECTIONS
   ═══════════════════════════════════════════════════════════ */

function ConceptSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-7 space-y-5">
          <Card accent="#6366f1">
            <Label>What is RAG / Enterprise Search?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              Retrieval-Augmented Generation (RAG) combines information retrieval with large language models to answer questions grounded in a specific knowledge base. Enterprise search extends this to index and search across an organization's internal documents — emails, docs, wikis, Slack, code, databases — returning precise, authoritative answers instead of just blue links.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Google builds this as <strong>Google Cloud Search</strong> (enterprise), <strong>Vertex AI Search</strong> (developer platform), and internally for searching across Google's own massive knowledge base. The core challenge: retrieve the right information from millions of documents and generate an accurate, grounded answer — without hallucinating.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard? (Google L6 Depth)</Label>
            <ul className="space-y-2.5">
              <Point icon="🎯" color="#0891b2">Retrieval quality is everything — if the right document isn't retrieved, the LLM can't generate a correct answer. Garbage in, garbage out. Retrieval errors compound into hallucinated answers.</Point>
              <Point icon="📄" color="#0891b2">Heterogeneous data — enterprise data spans PDFs, spreadsheets, emails, Slack threads, Confluence pages, code repos, databases. Each format requires different parsing, chunking, and representation strategies.</Point>
              <Point icon="🔒" color="#0891b2">Access control — a user must only see documents they have permission to access. ACLs must be enforced at retrieval time, not just generation time. Security-first design at scale is hard.</Point>
              <Point icon="🔄" color="#0891b2">Freshness — enterprise docs change constantly. An employee policy updated yesterday must surface today's version, not last month's. Index must stay in sync with source systems.</Point>
              <Point icon="🤥" color="#0891b2">Hallucination — the LLM may generate plausible-sounding answers not supported by the retrieved documents. In enterprise settings (HR policies, legal documents, financial data), a wrong answer can be costly.</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Scale</Label>
            <div className="space-y-2.5">
              {[
                { co: "Google Vertex AI Search", scale: "Billions of docs, multi-modal", approach: "Dense + sparse retrieval, Gemini gen" },
                { co: "Microsoft Copilot", scale: "O365 corpus per tenant", approach: "Graph RAG + GPT-4" },
                { co: "Perplexity", scale: "Web-scale RAG", approach: "Multi-step retrieval + citation" },
                { co: "Glean", scale: "Enterprise search SaaS", approach: "Connector + embedding + LLM" },
                { co: "Elastic", scale: "Self-hosted search infra", approach: "BM25 + vector + RRF" },
              ].map((e,i) => (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-28 shrink-0">{e.co}</span>
                  <span className="text-stone-500 flex-1">{e.scale}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <Label color="#2563eb">The RAG Pipeline (Preview)</Label>
            <svg viewBox="0 0 360 190" className="w-full">
              <defs><marker id="ah-rg" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>

              <rect x={10} y={10} width={75} height={35} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
              <text x={47} y={25} textAnchor="middle" fill="#2563eb" fontSize="8" fontWeight="700" fontFamily="monospace">User Query</text>
              <text x={47} y={38} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">natural language</text>

              <rect x={105} y={10} width={75} height={35} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
              <text x={142} y={25} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="700" fontFamily="monospace">Retrieve</text>
              <text x={142} y={38} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">top-K chunks</text>

              <rect x={200} y={10} width={70} height={35} rx={6} fill="#0f766e10" stroke="#0f766e" strokeWidth={1.5}/>
              <text x={235} y={25} textAnchor="middle" fill="#0f766e" fontSize="8" fontWeight="700" fontFamily="monospace">Re-Rank</text>
              <text x={235} y={38} textAnchor="middle" fill="#0f766e80" fontSize="7" fontFamily="monospace">cross-encoder</text>

              <rect x={290} y={10} width={60} height={35} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
              <text x={320} y={25} textAnchor="middle" fill="#ea580c" fontSize="8" fontWeight="700" fontFamily="monospace">Generate</text>
              <text x={320} y={38} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">LLM answer</text>

              <line x1={85} y1={28} x2={105} y2={28} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-rg)"/>
              <line x1={180} y1={28} x2={200} y2={28} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-rg)"/>
              <line x1={270} y1={28} x2={290} y2={28} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-rg)"/>

              <rect x={55} y={65} width={80} height={30} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
              <text x={95} y={84} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Vector Index</text>

              <rect x={155} y={65} width={80} height={30} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
              <text x={195} y={84} textAnchor="middle" fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Keyword Index</text>

              <line x1={120} y1={45} x2={95} y2={65} stroke="#94a3b8" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-rg)"/>
              <line x1={155} y1={45} x2={195} y2={65} stroke="#94a3b8" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-rg)"/>

              <rect x={10} y={115} width={340} height={65} rx={6} fill="#faf9f7" stroke="#e7e5e4"/>
              <text x={20} y={132} fill="#d97706" fontSize="7" fontWeight="600" fontFamily="monospace">Retrieve: Hybrid search — dense vectors (semantic) + sparse BM25 (keyword). Reciprocal Rank Fusion.</text>
              <text x={20} y={147} fill="#0f766e" fontSize="7" fontWeight="600" fontFamily="monospace">Re-Rank: Cross-encoder scores query-document pairs. Expensive but accurate. Top-K → Top-N.</text>
              <text x={20} y={162} fill="#ea580c" fontSize="7" fontWeight="600" fontFamily="monospace">Generate: LLM synthesizes answer from retrieved passages. Must cite sources. Must not hallucinate.</text>
              <text x={20} y={172} fill="#78716c" fontSize="7" fontFamily="monospace">Access control enforced at retrieval — user only sees docs they have permission for.</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Google Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Core to Vertex AI Search & Google Cloud</div>
              </div>
              <span className="text-indigo-500 font-bold text-sm">★★★★☆</span>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function RequirementsSection() {
  return (
    <div className="space-y-5">
      <Card className="bg-sky-50/50 border-sky-200">
        <div className="flex items-start gap-3">
          <span className="text-lg">💡</span>
          <div>
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Scope Like an L6</div>
            <p className="text-[12px] text-stone-500 mt-0.5">"Design a RAG system" is broad. Scope proactively: "I'll design an enterprise search + Q&A system that ingests heterogeneous documents (PDFs, docs, wikis, emails), indexes them with hybrid retrieval (dense + sparse), and generates grounded answers via an LLM. I'll focus on the retrieval pipeline, chunking strategy, re-ranking, and grounded generation with citations. I'll treat the LLM itself as a black box and focus on the system around it." Signal depth by mentioning hybrid retrieval and grounded generation early.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Ingest documents from multiple sources (Drive, Confluence, Slack, email, databases, code repos)</Point>
            <Point icon="2." color="#059669">Support natural language queries returning both document links AND synthesized answers</Point>
            <Point icon="3." color="#059669">Ground every answer in retrieved passages with inline citations</Point>
            <Point icon="4." color="#059669">Enforce per-user access control (ACLs) at retrieval time</Point>
            <Point icon="5." color="#059669">Support multi-turn conversational queries (follow-up questions with context)</Point>
            <Point icon="6." color="#059669">Support filters: date range, source, author, document type</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Query latency: retrieval &lt;200ms, generation &lt;3s (streaming first token &lt;500ms)</Point>
            <Point icon="2." color="#dc2626">Ingestion: index new/updated documents within 15 minutes of change</Point>
            <Point icon="3." color="#dc2626">Corpus scale: 10M-100M documents per organization, billions across platform</Point>
            <Point icon="4." color="#dc2626">Answer faithfulness: &gt;95% of generated claims must be supported by retrieved passages</Point>
            <Point icon="5." color="#dc2626">Retrieval quality: relevant document in top-5 results &gt;90% of queries (Recall@5)</Point>
            <Point icon="6." color="#dc2626">Availability: 99.9% — enterprise users depend on search for daily work</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask (L6 Signal)</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Single-tenant (one company) or multi-tenant SaaS? ACL model changes dramatically.",
            "What document types? Text-heavy (docs, wiki) or also tables, images, code?",
            "Real-time conversational (chatbot) or search-page (retrieve + display)?",
            "Do we need structured data querying (SQL generation from NL) or just unstructured?",
            "What's the freshness requirement? Minutes, hours, or daily sync is acceptable?",
            "Are there compliance requirements (data residency, audit logging, PII handling)?",
            "Budget for LLM inference? Gemini Pro vs Flash vs self-hosted affects architecture.",
            "Multi-language support? English-only or 50+ languages?",
          ].map((q,i) => (
            <div key={i} className="flex items-start gap-2 text-[12px] text-stone-500">
              <span className="text-fuchsia-500 font-bold shrink-0">?</span>{q}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function CapacitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Corpus & Index Size</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Documents per enterprise tenant" result="~10M docs" note="Google Drive + Confluence + Slack + email for a 10K-person org." />
            <MathStep step="2" formula="Avg document length" result="~3,000 tokens" note="Mix of short emails (100 tokens) and long reports (10K tokens)." />
            <MathStep step="3" formula="Chunks per doc (500 token chunks)" result="~6 chunks/doc" note="Overlapping chunks. Short docs = 1 chunk, long docs = 20+." />
            <MathStep step="4" formula="Total chunks = 10M x 6" result="~60M chunks" note="Per tenant. This is the vector index size." final />
            <MathStep step="5" formula="Embedding dim = 768 (float32)" result="3 KB/chunk" note="768 dims × 4 bytes. Standard for text-embedding models." />
            <MathStep step="6" formula="Vector index size = 60M x 3KB" result="~180 GB" note="Per tenant. With HNSW overhead: ~250 GB. Needs IVFPQ for compression." final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Query Volume & Latency</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Queries per employee per day" result="~20 searches" note="Internal wiki lookups, policy questions, finding docs." />
            <MathStep step="2" formula="QPS = 10K employees x 20 / 86400" result="~2.3 QPS" note="Per tenant. Low but bursty (Monday morning peak: 10x)." />
            <MathStep step="3" formula="Platform-wide (1000 tenants)" result="~2,300 QPS" note="Multi-tenant SaaS. Each query must be ACL-filtered." final />
            <MathStep step="4" formula="Retrieval latency budget" result="<200ms" note="Vector search + BM25 + ACL filter + merge." />
            <MathStep step="5" formula="Re-ranking latency budget" result="<300ms" note="Cross-encoder scoring top-50 candidates." />
            <MathStep step="6" formula="LLM generation" result="<3s total" note="Streaming response. First token in <500ms. Full answer in 2-3s." final />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Ingestion Pipeline</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="New/updated docs per day per tenant" result="~50K" note="Emails, Slack messages, doc edits. Most changes are small." />
            <MathStep step="2" formula="Platform-wide ingestion rate" result="~50M docs/day" note="Across all tenants. Each needs parsing, chunking, embedding." />
            <MathStep step="3" formula="Embedding throughput needed" result="~600 docs/sec" note="50M / 86400. Each doc → multiple chunks to embed." />
            <MathStep step="4" formula="Chunks to embed/sec = 600 x 6" result="~3,600 chunks/sec" note="Each chunk: ~5ms on GPU for embedding. Need ~18 GPUs." final />
            <MathStep step="5" formula="Freshness SLA" result="<15 min" note="New doc searchable within 15 min of creation/update." />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — LLM Inference Cost</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Queries requiring LLM answer" result="~60%" note="Some queries are just document search, no synthesis needed." />
            <MathStep step="2" formula="LLM queries/sec = 2300 x 0.6" result="~1,400/sec" note="Each query sends ~5K tokens (context) + generates ~500 tokens." />
            <MathStep step="3" formula="Input tokens/sec = 1400 x 5000" result="~7M tokens/sec" note="Context window: query + top-5 chunks + system prompt." final />
            <MathStep step="4" formula="Output tokens/sec = 1400 x 500" result="~700K tokens/sec" note="Answer + citations. Streaming to user." />
            <MathStep step="5" formula="Cost estimate (Gemini Flash)" result="~$0.003/query" note="$0.075 per 1M input tokens. Dominant cost at scale." />
          </div>
        </Card>
      </div>
    </div>
  );
}

function ApiSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#2563eb">
          <Label color="#2563eb">Search & Answer API</Label>
          <CodeBlock code={`# POST /v1/search
# Unified search + answer endpoint
{
  "query": "What is our parental leave policy?",
  "user_id": "user_abc123",
  "conversation_id": "conv_xyz",       // for multi-turn
  "options": {
    "mode": "search_and_answer",        // search_only | answer_only | search_and_answer
    "max_results": 10,
    "answer_style": "detailed",         // brief | detailed | extractive
    "filters": {
      "source_types": ["confluence", "google_drive"],
      "date_range": {"after": "2024-01-01"},
      "authors": [],
    },
    "boost": {
      "recency": 0.3,                   // weight recent docs higher
      "source_authority": 0.2,          // official policy docs > Slack
    }
  }
}

# Response (streamed):
{
  "answer": {
    "text": "The company offers 16 weeks of paid parental leave...",
    "citations": [
      {"chunk_id": "c_123", "doc_title": "Employee Handbook 2024",
       "source": "confluence", "url": "https://...",
       "excerpt": "...16 weeks of fully paid parental leave...",
       "relevance_score": 0.94}
    ],
    "confidence": 0.91,
    "grounding_score": 0.96,
  },
  "search_results": [
    {"doc_id": "d_456", "title": "Employee Handbook 2024",
     "snippet": "...parental leave policy updated Jan 2024...",
     "source": "confluence", "score": 0.94, "url": "..."}
  ],
  "metadata": {
    "retrieval_latency_ms": 145,
    "generation_latency_ms": 1820,
    "chunks_retrieved": 50,
    "chunks_after_rerank": 5,
  }
}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Design Decisions (L6 Depth)</Label>
          <div className="space-y-3">
            {[
              { q: "Why unified search + answer?", a: "Users want both: the synthesized answer for quick consumption AND the source documents for verification. Providing only an answer without links is unverifiable. Providing only links without an answer is the old paradigm. The unified response serves both needs." },
              { q: "Why conversation_id?", a: "Multi-turn queries are essential for enterprise search. 'What's the parental leave policy?' → 'Does that apply to contractors too?' → 'What about adoption?' The conversation_id maintains context across turns. The system rewrites follow-up queries using conversation history before retrieval." },
              { q: "Why grounding_score?", a: "Grounding score measures what % of the generated answer is directly supported by the retrieved passages. Confidence measures the LLM's self-assessed certainty. A high-confidence but low-grounding answer means the LLM is making claims not in the documents — likely hallucination. We surface this to the UI." },
              { q: "Why source_authority boost?", a: "Not all documents are equally authoritative. The official Employee Handbook should rank higher than a Slack message discussing leave policy informally. Source authority is a metadata signal: official_policy > wiki > email > chat. Configurable per deployment." },
              { q: "Why stream the response?", a: "LLM generation takes 2-3 seconds. Without streaming, the user stares at a spinner. With streaming, the first token appears in <500ms. Search results can render immediately while the answer streams in parallel. Dramatically better perceived latency." },
            ].map((d,i) => (
              <div key={i}>
                <div className="text-[11px] font-bold text-stone-700">{d.q}</div>
                <div className="text-[11px] text-stone-500 mt-0.5">{d.a}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function DesignSection() {
  return (
    <div className="space-y-5">
      <Card accent="#9333ea">
        <Label color="#9333ea">Full System Architecture — RAG Pipeline</Label>
        <svg viewBox="0 0 720 370" className="w-full">
          <defs><marker id="ah-hld" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>

          {/* Query path */}
          <rect x={10} y={40} width={70} height={40} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={45} y={57} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Query</text>
          <text x={45} y={70} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">+ user ACL</text>

          <rect x={100} y={35} width={85} height={50} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
          <text x={142} y={53} textAnchor="middle" fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Query</text>
          <text x={142} y={65} textAnchor="middle" fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Processing</text>
          <text x={142} y={78} textAnchor="middle" fill="#7c3aed80" fontSize="7" fontFamily="monospace">rewrite, expand</text>

          <rect x={210} y={25} width={90} height={30} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={255} y={44} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Dense Retrieval</text>

          <rect x={210} y={63} width={90} height={30} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={255} y={82} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Sparse Retrieval</text>

          <rect x={325} y={35} width={75} height={50} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={362} y={53} textAnchor="middle" fill="#0891b2" fontSize="8" fontWeight="600" fontFamily="monospace">Merge +</text>
          <text x={362} y={65} textAnchor="middle" fill="#0891b2" fontSize="8" fontWeight="600" fontFamily="monospace">ACL Filter</text>
          <text x={362} y={78} textAnchor="middle" fill="#0891b280" fontSize="7" fontFamily="monospace">RRF</text>

          <rect x={425} y={35} width={80} height={50} rx={6} fill="#0f766e10" stroke="#0f766e" strokeWidth={1.5}/>
          <text x={465} y={55} textAnchor="middle" fill="#0f766e" fontSize="8" fontWeight="600" fontFamily="monospace">Re-Rank</text>
          <text x={465} y={68} textAnchor="middle" fill="#0f766e80" fontSize="7" fontFamily="monospace">cross-encoder</text>
          <text x={465} y={79} textAnchor="middle" fill="#0f766e80" fontSize="7" fontFamily="monospace">50 → 5</text>

          <rect x={530} y={30} width={90} height={60} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={575} y={50} textAnchor="middle" fill="#ea580c" fontSize="8" fontWeight="600" fontFamily="monospace">LLM Generation</text>
          <text x={575} y={63} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">grounded answer</text>
          <text x={575} y={75} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">+ citations</text>

          <rect x={645} y={40} width={60} height={40} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={675} y={57} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Answer</text>
          <text x={675} y={70} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">streamed</text>

          {/* Arrows */}
          <line x1={80} y1={60} x2={100} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={185} y1={50} x2={210} y2={40} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={185} y1={65} x2={210} y2={78} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={300} y1={40} x2={325} y2={50} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={300} y1={78} x2={325} y2={68} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={400} y1={60} x2={425} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={505} y1={60} x2={530} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={620} y1={60} x2={645} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>

          {/* Ingestion path */}
          <rect x={10} y={140} width={80} height={35} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={50} y={161} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Data Sources</text>

          <rect x={110} y={140} width={80} height={35} rx={6} fill="#7e22ce10" stroke="#7e22ce" strokeWidth={1.5}/>
          <text x={150} y={156} textAnchor="middle" fill="#7e22ce" fontSize="8" fontWeight="600" fontFamily="monospace">Connectors</text>
          <text x={150} y={168} textAnchor="middle" fill="#7e22ce80" fontSize="7" fontFamily="monospace">parse, chunk</text>

          <rect x={210} y={140} width={80} height={35} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={250} y={156} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Embed</text>
          <text x={250} y={168} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">bi-encoder</text>

          <rect x={325} y={140} width={90} height={35} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={370} y={156} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Vector Index</text>
          <text x={370} y={168} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">HNSW / ScaNN</text>

          <rect x={440} y={140} width={90} height={35} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={485} y={156} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Keyword Index</text>
          <text x={485} y={168} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">BM25</text>

          <rect x={560} y={140} width={80} height={35} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={600} y={156} textAnchor="middle" fill="#0891b2" fontSize="8" fontWeight="600" fontFamily="monospace">ACL Store</text>
          <text x={600} y={168} textAnchor="middle" fill="#0891b280" fontSize="7" fontFamily="monospace">permissions</text>

          <line x1={90} y1={158} x2={110} y2={158} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={190} y1={158} x2={210} y2={158} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={290} y1={158} x2={325} y2={158} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={290} y1={165} x2={440} y2={165} stroke="#94a3b880" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>

          {/* Legend */}
          <rect x={15} y={200} width={695} height={155} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={25} y={220} fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Query Processing (~20ms): Rewrite multi-turn queries, expand abbreviations, generate sub-queries for complex questions.</text>
          <text x={25} y={237} fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Dense Retrieval (~50ms): Embed query with bi-encoder, ANN search over vector index. Semantic matching — captures meaning.</text>
          <text x={25} y={254} fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Sparse Retrieval (~30ms): BM25 keyword search. Exact term matching. Catches proper nouns, IDs, code that embeddings miss.</text>
          <text x={25} y={271} fill="#0891b2" fontSize="8" fontWeight="600" fontFamily="monospace">Merge + ACL (~30ms): Reciprocal Rank Fusion combines dense + sparse. Filter by user's document permissions. 100 → 50 candidates.</text>
          <text x={25} y={288} fill="#0f766e" fontSize="8" fontWeight="600" fontFamily="monospace">Re-Rank (~200ms): Cross-encoder scores each (query, chunk) pair. Much more accurate than bi-encoder. 50 → 5 passages.</text>
          <text x={25} y={305} fill="#ea580c" fontSize="8" fontWeight="600" fontFamily="monospace">Generation (~2s): LLM synthesizes answer from top-5 passages. Prompt instructs: cite sources, say "I don't know" if not in context.</text>
          <text x={25} y={325} fill="#7e22ce" fontSize="8" fontWeight="600" fontFamily="monospace">Ingestion (offline): Connectors crawl data sources → parse documents → chunk → embed → index in vector + keyword stores.</text>
          <text x={25} y={342} fill="#78716c" fontSize="8" fontFamily="monospace">KEY INSIGHT: Hybrid retrieval (dense + sparse) consistently outperforms either alone. RRF is the simplest effective fusion.</text>
        </svg>
      </Card>
    </div>
  );
}

function ChunkingSection() {
  const [sel, setSel] = useState("fixed");
  const strategies = {
    fixed: { name: "Fixed-Size Chunks", color: "#c026d3",
      desc: "Split document into chunks of fixed token count (e.g., 500 tokens) with overlap (e.g., 100 tokens). Simplest approach, works well as a baseline.",
      code: `# Fixed-size chunking with overlap
def chunk_fixed(text, chunk_size=500, overlap=100):
    tokens = tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        chunks.append({
            "text": detokenize(chunk),
            "start_token": start,
            "end_token": end,
            "doc_id": doc.id,
        })
        start += chunk_size - overlap  # slide window
    return chunks

# Pros: Simple, predictable chunk sizes, easy to batch embed
# Cons: May split sentences mid-thought, loses structure
# Overlap mitigates: important context near boundaries
#        appears in both adjacent chunks
#
# TUNING chunk_size:
# Too small (100 tokens): fragments lose context
# Too large (2000 tokens): embedding dilutes — averaging
#   over too many topics makes the embedding generic
# Sweet spot: 256-512 tokens for most retrieval tasks` },
    semantic: { name: "Semantic Chunks ★", color: "#dc2626",
      desc: "Split at semantic boundaries: paragraphs, sections, topic changes. Each chunk is a coherent unit of meaning. Better retrieval quality but more complex.",
      code: `# Semantic chunking — split at natural boundaries
def chunk_semantic(document):
    # Level 1: Split by document structure
    sections = split_by_headings(document)  # H1, H2, H3

    chunks = []
    for section in sections:
        # Level 2: Split large sections by paragraph
        if token_count(section) > MAX_CHUNK_SIZE:
            paragraphs = split_by_paragraph(section)
            # Group consecutive paragraphs into chunks
            current_chunk = []
            current_size = 0
            for para in paragraphs:
                if current_size + token_count(para) > MAX_CHUNK_SIZE:
                    chunks.append(merge(current_chunk))
                    # Overlap: keep last paragraph
                    current_chunk = [current_chunk[-1]] if current_chunk else []
                    current_size = token_count(current_chunk[0]) if current_chunk else 0
                current_chunk.append(para)
                current_size += token_count(para)
            if current_chunk:
                chunks.append(merge(current_chunk))
        else:
            chunks.append(section)

    # Level 3: Enrich with context
    for chunk in chunks:
        chunk.metadata = {
            "section_title": chunk.heading,
            "doc_title": document.title,
            "breadcrumb": f"{document.title} > {section.heading}",
        }
    return chunks

# WHY SEMANTIC > FIXED:
# Fixed chunk might split: "Parental leave is 16 weeks. | This applies..."
# Semantic chunk keeps the full policy section together
# Result: embedding captures the full concept, retrieval is better` },
    hierarchical: { name: "Hierarchical (Parent-Child)", color: "#059669",
      desc: "Store chunks at multiple granularities: sentence, paragraph, section. Retrieve at fine-grained level but return parent context for generation. Best of both worlds.",
      code: `# Hierarchical chunking — multi-level retrieval
def chunk_hierarchical(document):
    # Level 0: Full document summary (for broad queries)
    doc_summary = summarize(document)

    # Level 1: Section-level chunks (for section queries)
    sections = split_by_headings(document)

    # Level 2: Paragraph-level chunks (for specific queries)
    paragraphs = []
    for section in sections:
        for para in split_by_paragraph(section):
            paragraphs.append({
                "text": para,
                "parent_section_id": section.id,
                "doc_id": document.id,
                "level": 2,
            })

    # At retrieval time:
    # 1. Search at paragraph level (Level 2) — most precise
    # 2. If a paragraph matches, expand to parent section (Level 1)
    # 3. Pass the SECTION to the LLM, not just the paragraph
    #
    # WHY: paragraph-level embeddings are precise for retrieval
    #       but a single paragraph may lack context for generation
    #       Parent section provides full context for LLM

    # This is what Google Vertex AI Search does internally
    # "Small-to-big" retrieval: search small, generate from big

    return {
        "doc_summary": doc_summary,      # L0
        "sections": sections,             # L1
        "paragraphs": paragraphs,         # L2 (indexed for search)
    }` },
    table: { name: "Table & Structured Data", color: "#d97706",
      desc: "Tables, spreadsheets, and structured data need special handling. Can't just embed the raw text — need to preserve row/column relationships and generate natural language descriptions.",
      code: `# Table-aware chunking
def chunk_table(table):
    chunks = []

    # Strategy 1: Row-level chunks with column headers
    for row in table.rows:
        text = " | ".join(
            f"{col}: {val}" for col, val in zip(table.headers, row)
        )
        chunks.append({
            "text": f"Table: {table.title}\\n{text}",
            "type": "table_row",
        })

    # Strategy 2: Table summary (LLM-generated)
    summary = llm.summarize(
        f"Summarize this table:\\n{table.to_markdown()}"
    )
    chunks.append({
        "text": summary,
        "type": "table_summary",
    })

    # Strategy 3: Column-aware embedding
    # Each column becomes a searchable dimension
    for col in table.columns:
        col_text = f"{table.title} - {col.name}: {', '.join(col.values[:20])}"
        chunks.append({
            "text": col_text,
            "type": "table_column",
        })

    # WHY: "What was Q3 revenue?" needs to find the right CELL
    # in a table, not just the document containing the table.
    # Row-level chunks make individual data points retrievable.
    return chunks` },
  };
  const s = strategies[sel];
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Chunking Strategy Comparison</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200 mb-4">
          <table className="w-full text-[12px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Strategy</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Retrieval Quality</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Implementation</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Best For</th>
            </tr></thead>
            <tbody>
              {[
                { n:"Fixed-size", rq:"Good baseline", impl:"Simple", bf:"Homogeneous text" },
                { n:"Semantic ★", rq:"High", impl:"Medium", bf:"Structured docs (wiki, policies)", hl:true },
                { n:"Hierarchical", rq:"Very High", impl:"Complex", bf:"Long docs, varied query types" },
                { n:"Table-aware", rq:"Essential for tables", impl:"Complex", bf:"Spreadsheets, databases" },
              ].map((r,i) => (
                <tr key={i} className={r.hl?"bg-fuchsia-50":""}>
                  <td className="px-3 py-2 font-bold text-stone-700">{r.n}</td>
                  <td className="text-center px-3 py-2 text-stone-600">{r.rq}</td>
                  <td className="text-center px-3 py-2 text-stone-600">{r.impl}</td>
                  <td className="text-center px-3 py-2 text-stone-600">{r.bf}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-3 space-y-2">
          {Object.entries(strategies).map(([k,v]) => (
            <button key={k} onClick={() => setSel(k)}
              className={`w-full text-left px-3.5 py-2.5 rounded-lg text-[12px] font-medium border transition-all ${k===sel?`text-white border-transparent`:"bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}
              style={k===sel?{background:v.color}:{}}>
              {v.name}
            </button>
          ))}
        </div>
        <div className="col-span-9 space-y-4">
          <Card>
            <p className="text-[12px] text-stone-500">{s.desc}</p>
          </Card>
          <CodeBlock title={s.name} code={s.code} />
        </div>
      </div>
    </div>
  );
}

function EmbeddingSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Embedding Models — Bi-Encoder for Retrieval</Label>
          <CodeBlock code={`# Bi-encoder: independent query and doc embeddings
# Fast: encode query once, compare with pre-computed doc embeddings
# Used for initial retrieval (ANN search)

class BiEncoder:
    def __init__(self):
        # Shared or separate encoders for query and document
        self.query_encoder = TransformerEncoder("query")
        self.doc_encoder = TransformerEncoder("doc")

    def encode_query(self, query):
        return self.query_encoder(query)  # [768]

    def encode_document(self, chunk_text):
        return self.doc_encoder(chunk_text)  # [768]

    def score(self, query_emb, doc_emb):
        return cosine_similarity(query_emb, doc_emb)

# MODEL CHOICES (2024-2025):
# Google: text-embedding-004 (768-dim, multilingual)
# OpenAI: text-embedding-3-large (3072-dim, english-focused)
# Open: E5-large-v2 (1024-dim), BGE-large (1024-dim)
# Cohere: embed-v3 (1024-dim, multilingual)
#
# KEY TRADEOFF: dimension vs quality vs speed
# 768-dim: 3KB/vector, fast ANN, good quality
# 3072-dim: 12KB/vector, slower ANN, marginally better
# For enterprise (60M chunks), 768-dim is the sweet spot

# FINE-TUNING for enterprise:
# Generic embedding models trained on web data
# Enterprise data has domain-specific vocabulary
# Fine-tune on (query, relevant_doc) pairs from search logs
# Even 1000 pairs significantly improves retrieval quality`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Vector Index — ANN Search</Label>
          <CodeBlock code={`# Approximate Nearest Neighbor (ANN) index
# Exact nearest neighbor on 60M vectors is too slow
# ANN trades <5% recall for 100x speedup

# OPTION 1: HNSW (Hierarchical Navigable Small World)
# Graph-based. Best recall-speed tradeoff. Memory-hungry.
index = HNSWIndex(
    dim=768,
    M=32,                  # connections per node
    ef_construction=200,   # build-time quality
    ef_search=128,         # query-time quality
)
# Memory: 60M × 768 × 4B = 180GB + graph ~50GB = 230GB
# Query: ~5ms for top-100 results
# Recall@100: ~98%

# OPTION 2: IVF-PQ (Inverted File + Product Quantization)
# Compressed. Less memory. Slightly lower recall.
index = IVFPQIndex(
    dim=768,
    n_lists=4096,         # Voronoi cells
    n_probes=32,          # cells searched at query time
    m_subquantizers=48,   # PQ compression
    n_bits=8,             # bits per subquantizer
)
# Memory: 60M × 48 bytes = ~3GB (60x compression!)
# Query: ~3ms for top-100 results
# Recall@100: ~92%

# RECOMMENDATION FOR ENTERPRISE:
# Small corpus (<10M chunks): HNSW — best quality, fits in memory
# Large corpus (>100M chunks): IVF-PQ or ScaNN — memory-efficient
# Google uses ScaNN (open-source) internally

# METADATA FILTERING:
# "Search only in Confluence docs from 2024"
# Pre-filter: filter BEFORE ANN search (fast but reduces index)
# Post-filter: ANN search then filter (wastes computation)
# Hybrid: partition index by common filters (source_type, year)
#          then search within partition`} />
        </Card>
      </div>
    </div>
  );
}

function RetrievalSection() {
  return (
    <div className="space-y-5">
      <Card accent="#d97706">
        <Label color="#d97706">Hybrid Retrieval — Dense + Sparse + Fusion</Label>
        <p className="text-[12px] text-stone-500 mb-4">Neither dense (vector) nor sparse (keyword) retrieval alone is sufficient. Dense captures semantic similarity ("car" matches "automobile"). Sparse captures exact terms (error codes, employee IDs, product names). Hybrid combines both for the best of both worlds.</p>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Hybrid Retrieval Pipeline" code={`# Hybrid retrieval with Reciprocal Rank Fusion
async def hybrid_retrieve(query, user_id, filters, k=50):
    # Step 1: Process query
    query_emb = bi_encoder.encode_query(query)
    processed_query = query_processor.expand(query)

    # Step 2: Parallel retrieval from both indexes
    dense_results = vector_index.search(
        query_emb, k=100, filters=filters
    )  # semantic match

    sparse_results = keyword_index.search(
        processed_query, k=100, filters=filters
    )  # exact keyword match

    # Step 3: ACL filtering
    user_acls = acl_store.get_user_permissions(user_id)
    dense_results = [r for r in dense_results
                     if has_access(user_acls, r.doc_id)]
    sparse_results = [r for r in sparse_results
                      if has_access(user_acls, r.doc_id)]

    # Step 4: Reciprocal Rank Fusion (RRF)
    # Combine rankings from both retrievers
    fused = reciprocal_rank_fusion(
        rankings=[dense_results, sparse_results],
        k=60,  # RRF constant (standard: 60)
    )

    return fused[:k]  # top-k candidates for re-ranking

def reciprocal_rank_fusion(rankings, k=60):
    """RRF: score(d) = sum(1 / (k + rank_i(d)))"""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, result in enumerate(ranking):
            scores[result.chunk_id] += 1.0 / (k + rank + 1)

    # Sort by fused score descending
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused

# WHY RRF OVER LEARNED FUSION:
# RRF is parameter-free — no training data needed
# Works well when the two retrievers have different strengths
# More complex fusion (learned weighting) requires training data
# RRF is the standard for enterprise RAG (used in Elastic, Weaviate)`} />
          <div className="space-y-4">
            <Card accent="#d97706">
              <Label color="#d97706">When Dense Wins vs Sparse Wins</Label>
              <div className="space-y-2">
                {[
                  { query:"'How to request PTO?'", winner:"Dense", why:"Semantic match. Document says 'time off request procedure' — no exact keyword overlap. Vector similarity captures the meaning." },
                  { query:"'ERR_SSL_PROTOCOL_ERROR'", winner:"Sparse", why:"Exact string match. Error codes, product IDs, and technical identifiers must match exactly. Embeddings may not preserve these." },
                  { query:"'John Smith's last performance review'", winner:"Sparse", why:"Proper noun matching. 'John Smith' must match exactly. Dense search might return any performance review." },
                  { query:"'What's our approach to sustainability?'", winner:"Dense", why:"Broad semantic concept. The answer may use words like 'environmental responsibility', 'carbon neutral', 'green initiatives' — all semantically related." },
                  { query:"'Q3 2024 revenue for APAC region'", winner:"Both", why:"Dense for 'revenue' concept. Sparse for exact 'Q3 2024' and 'APAC' terms. Hybrid is essential here." },
                ].map((e,i) => (
                  <div key={i} className="rounded-lg border border-stone-200 p-2.5">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-[10px] font-mono text-stone-600">{e.query}</span>
                      <Pill bg={e.winner==="Dense"?"#fff7ed":e.winner==="Sparse"?"#f3e8ff":"#f0fdf4"} color={e.winner==="Dense"?"#d97706":e.winner==="Sparse"?"#7c3aed":"#059669"}>{e.winner}</Pill>
                    </div>
                    <p className="text-[10px] text-stone-400">{e.why}</p>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </Card>
    </div>
  );
}

function RerankingSection() {
  return (
    <div className="space-y-5">
      <Card accent="#0f766e">
        <Label color="#0f766e">Re-Ranking — The Quality Multiplier</Label>
        <p className="text-[12px] text-stone-500 mb-4">Re-ranking is the most impactful improvement you can add to any RAG system. A cross-encoder scores each (query, document) pair jointly — far more accurate than bi-encoder similarity — but too expensive to run on the full corpus. So we retrieve top-50 with bi-encoder, then re-rank to top-5 with cross-encoder.</p>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Cross-Encoder Re-Ranker" code={`# Cross-encoder: jointly encodes query + document
# Much more accurate than bi-encoder but 100x slower
# (can't pre-compute: must score each pair at query time)

class CrossEncoderReranker:
    def __init__(self):
        # Model: fine-tuned BERT/DeBERTa for relevance scoring
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12")

    def rerank(self, query, candidates, top_n=5):
        # Score each (query, candidate) pair
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)  # batch inference

        # Sort by cross-encoder score
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: -x[1]
        )
        return ranked[:top_n]

# WHY CROSS-ENCODER > BI-ENCODER for ranking:
# Bi-encoder: encodes query and doc SEPARATELY
#   cos_sim(embed("PTO policy"), embed("time off request"))
#   Misses: the interaction between "PTO" and "time off"
#
# Cross-encoder: encodes [query + doc] TOGETHER
#   BERT("[CLS] PTO policy [SEP] time off request procedure [SEP]")
#   Self-attention layers can directly compare query and doc tokens
#   Learns: "PTO" in query attends to "time off" in document
#
# Typical improvement: +5-15% nDCG over bi-encoder alone
# Latency: ~5ms per pair × 50 candidates = ~250ms (batched: ~100ms)

# ALTERNATIVE: Cohere Rerank, Google Ranking API, open-source bge-reranker
# For production: distill from large to small cross-encoder for speed`} />
          <div className="space-y-4">
            <Card accent="#0f766e">
              <Label color="#0f766e">Re-Ranking Strategies</Label>
              <div className="space-y-2">
                {[
                  { strat: "Cross-Encoder ★", desc: "Joint query-doc scoring with BERT. Gold standard for accuracy. ~5ms per pair. Use on top-50 candidates.", when: "Default choice" },
                  { strat: "LLM-as-Reranker", desc: "Prompt the LLM: 'Rank these 10 passages by relevance to the query.' More expensive but leverages LLM reasoning for complex queries.", when: "Complex reasoning queries" },
                  { strat: "ColBERT (late interaction)", desc: "Token-level similarity between query and document tokens. Faster than cross-encoder, better than bi-encoder. Good middle ground.", when: "High throughput needed" },
                  { strat: "MMR (Maximal Marginal Relevance)", desc: "Balance relevance with diversity. After scoring relevance, penalize candidates similar to already-selected ones. Prevents redundant results.", when: "Avoid duplicate info" },
                  { strat: "Metadata Boosting", desc: "Boost scores based on document freshness, source authority, and user preferences. Recent docs and official sources get score multipliers.", when: "Always (as post-processing)" },
                ].map((s,i) => (
                  <div key={i} className="rounded-lg border border-stone-200 p-2.5">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-[11px] font-bold text-stone-800">{s.strat}</span>
                      <span className="text-[9px] text-stone-400">{s.when}</span>
                    </div>
                    <p className="text-[10px] text-stone-500">{s.desc}</p>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </Card>
    </div>
  );
}

function GenerationSection() {
  return (
    <div className="space-y-5">
      <Card accent="#ea580c">
        <Label color="#ea580c">Grounded Generation — Avoiding Hallucination</Label>
        <p className="text-[12px] text-stone-500 mb-4">The generation step is where RAG becomes powerful — and dangerous. The LLM must synthesize a coherent answer from retrieved passages while ONLY stating facts supported by those passages. Hallucination (generating plausible but unsupported claims) is the #1 failure mode.</p>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Generation Prompt Design" code={`# Grounding prompt — the most important engineering decision in RAG
SYSTEM_PROMPT = """
You are an enterprise search assistant. Answer the user's question
using ONLY the provided context passages. Follow these rules strictly:

1. ONLY use information from the provided passages.
2. If the answer is not in the passages, say "I couldn't find
   information about this in the available documents."
3. Cite your sources using [Source N] notation after each claim.
4. If passages contain conflicting information, note the conflict
   and cite both sources.
5. Do not add information from your general knowledge.
6. If the question is ambiguous, ask for clarification.
"""

def generate_answer(query, passages, conversation_history=[]):
    # Build context from top-K retrieved passages
    context = ""
    for i, passage in enumerate(passages):
        context += f"[Source {i+1}]: {passage.doc_title}\\n"
        context += f"{passage.text}\\n\\n"

    # Build multi-turn context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in conversation_history:
        messages.append(turn)

    messages.append({
        "role": "user",
        "content": f"Context:\\n{context}\\n\\nQuestion: {query}"
    })

    # Generate with streaming
    response = llm.generate(
        messages=messages,
        temperature=0.1,   # low temp for factual accuracy
        max_tokens=1024,
        stream=True,
    )

    # Post-process: extract citations, compute grounding score
    answer = collect_stream(response)
    citations = extract_citations(answer, passages)
    grounding = compute_grounding_score(answer, passages)

    return {
        "text": answer,
        "citations": citations,
        "grounding_score": grounding,
    }`} />
          <Card accent="#dc2626">
            <Label color="#dc2626">Grounding Verification</Label>
            <CodeBlock code={`# Grounding score: what % of claims are supported?
def compute_grounding_score(answer, passages):
    # Split answer into individual claims
    claims = extract_claims(answer)  # NLI-based or LLM-based

    supported = 0
    for claim in claims:
        # Check if any passage entails this claim
        # Using NLI (Natural Language Inference) model
        for passage in passages:
            entailment = nli_model.predict(
                premise=passage.text,
                hypothesis=claim,
            )
            if entailment.label == "ENTAILMENT" and entailment.score > 0.8:
                supported += 1
                break

    return supported / len(claims) if claims else 1.0

# Grounding score thresholds:
# > 0.95: High confidence — display answer prominently
# 0.80-0.95: Medium — display with "Based on available docs"
# < 0.80: Low — flag for review, maybe show docs-only
#
# ALTERNATIVE: LLM-as-judge
# Prompt a second LLM: "Does this answer only contain
# information from the provided passages? Rate 1-5."
# Cheaper than NLI model, surprisingly effective

# POST-HOC CITATION VERIFICATION:
# After generating the answer, verify each [Source N] tag
# actually corresponds to content in source N
# LLMs sometimes cite the wrong source number
def verify_citations(answer, passages):
    for citation in extract_citations(answer):
        claim_text = citation.claim
        cited_passage = passages[citation.source_idx]
        if not is_supported(claim_text, cited_passage.text):
            citation.verified = False  # Flag for UI`} />
          </Card>
        </div>
      </Card>
    </div>
  );
}

function DataModelSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Core Data Stores</Label>
          <CodeBlock code={`-- Document Store (Cloud Storage + metadata in Spanner)
doc_id -> {
  title, source_type, source_url, author,
  created_at, updated_at, last_indexed_at,
  content_hash: SHA256,     # detect changes
  tenant_id: string,        # multi-tenant isolation
  acl: {
    users: [user_id, ...],
    groups: [group_id, ...],
    visibility: "org" | "team" | "private",
  },
  raw_content_url: "gs://...",  # original file
}

-- Chunk Store (Bigtable)
chunk_id -> {
  doc_id, chunk_index,
  text: string,              # chunk content
  token_count: int,
  embedding: float[768],     # may be stored in vector index only
  metadata: {
    section_title, breadcrumb, chunk_type,
    parent_chunk_id,          # for hierarchical
  },
}

-- Vector Index (ScaNN / Vertex AI Matching Engine)
-- Separate index per tenant (ACL isolation)
tenant_id -> ANN_Index(chunk_id -> embedding[768])

-- Keyword Index (Elasticsearch / Cloud Search)
-- BM25 index with field-level boosting
tenant_id -> {
  title^3, body, section_title^2, tags^2, author
}

-- ACL Cache (Redis)
user_id -> {
  groups: [group_id, ...],
  accessible_doc_ids: bloom_filter,  # fast pre-filter
  last_synced: timestamp,
}

-- Conversation Store (Firestore)
conversation_id -> {
  user_id, tenant_id,
  turns: [{role, content, timestamp, retrieved_docs}, ...],
}`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Why These Design Choices?</Label>
          <div className="space-y-3">
            {[
              { q: "Why separate vector index per tenant?", a: "ACL isolation at the index level. If tenant A's documents are in a different index than tenant B's, there's zero risk of cross-tenant data leakage even with a bug. Also allows tenant-specific index tuning (HNSW parameters, embedding model)." },
              { q: "Why Bloom filter for ACL?", a: "Checking 'can user X access doc Y?' for every result is expensive. A Bloom filter for each user's accessible doc_ids provides a fast pre-filter (~100ns). False positives are fine (just check the real ACL). False negatives are impossible (user never loses access to permitted docs)." },
              { q: "Why store content_hash in document store?", a: "Change detection. When the ingestion pipeline re-crawls a source, it computes the hash of the new version. If hash matches, skip re-indexing (save embedding compute). If hash differs, re-chunk, re-embed, and update the index." },
              { q: "Why Elasticsearch for keyword index?", a: "BM25 is the gold standard for keyword retrieval. Elasticsearch provides: field-level boosting (title matches count more), fuzzy matching, phrase queries, and filtering. It's battle-tested at enterprise scale. Alternative: Cloud Search, Solr, or custom BM25 implementation." },
              { q: "Why store conversation history?", a: "Multi-turn queries require context. 'What's the leave policy?' → 'Does that apply to contractors?' The system must rewrite the second query as 'Does the parental leave policy apply to contractors?' using conversation history. History is also used for analytics and fine-tuning." },
            ].map((d,i) => (
              <div key={i}>
                <div className="text-[11px] font-bold text-stone-700">{d.q}</div>
                <div className="text-[11px] text-stone-500 mt-0.5">{d.a}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function IngestionSection() {
  return (
    <div className="space-y-5">
      <Card accent="#7e22ce">
        <Label color="#7e22ce">Ingestion Pipeline — From Source to Searchable</Label>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="End-to-End Ingestion" code={`# Ingestion pipeline: source → parse → chunk → embed → index
class IngestionPipeline:
    def ingest_document(self, source_event):
        # Step 1: Fetch and parse (connector-specific)
        raw = self.connector.fetch(source_event.url)
        parsed = self.parser.parse(raw)
        # Parser handles: PDF (pdfminer), DOCX (python-docx),
        # HTML (BeautifulSoup), Slides (python-pptx),
        # Spreadsheets (openpyxl), Images (OCR via Vision API)

        # Step 2: Change detection
        content_hash = sha256(parsed.text)
        existing = doc_store.get(source_event.doc_id)
        if existing and existing.content_hash == content_hash:
            return  # unchanged — skip re-indexing

        # Step 3: Chunk
        chunks = self.chunker.chunk_semantic(parsed)

        # Step 4: Embed all chunks (batched GPU inference)
        embeddings = self.embedder.encode_batch(
            [c.text for c in chunks]
        )

        # Step 5: Update vector index
        # Delete old chunks for this doc, insert new ones
        vector_index.delete_by_doc(source_event.doc_id)
        vector_index.insert_batch(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            metadata=[c.metadata for c in chunks],
        )

        # Step 6: Update keyword index
        keyword_index.delete_by_doc(source_event.doc_id)
        keyword_index.index_batch(chunks)

        # Step 7: Update ACL
        acl = source_event.permissions  # from source system
        doc_store.update_acl(source_event.doc_id, acl)

        # Step 8: Update document metadata
        doc_store.upsert({
            "doc_id": source_event.doc_id,
            "content_hash": content_hash,
            "last_indexed_at": now(),
        })`} />
          <div className="space-y-4">
            <Card accent="#7e22ce">
              <Label color="#7e22ce">Connector Architecture</Label>
              <div className="space-y-2">
                {[
                  { source: "Google Drive", sync: "Push (Drive API webhook)", freq: "Real-time", notes: "Permissions sync from Drive ACLs. Handles Docs, Sheets, Slides, PDF." },
                  { source: "Confluence", sync: "Pull (REST API polling)", freq: "Every 5 min", notes: "Space-level and page-level permissions. Handles macros, attachments." },
                  { source: "Slack", sync: "Push (Events API)", freq: "Real-time", notes: "Channel-level access control. Thread-aware chunking. DMs excluded." },
                  { source: "Email (Gmail)", sync: "Pull (Gmail API)", freq: "Every 15 min", notes: "Per-user permissions only. Attachment parsing. Thread grouping." },
                  { source: "Code repos (GitHub)", sync: "Push (webhook)", freq: "Real-time", notes: "File-level parsing. README, code comments, PR descriptions. Repo-level ACL." },
                  { source: "Databases (BigQuery)", sync: "Pull (scheduled)", freq: "Daily", notes: "Table descriptions, column metadata. Row-level data excluded (schema only)." },
                ].map((c,i) => (
                  <div key={i} className="flex items-start gap-2 text-[11px]">
                    <span className="font-mono font-bold text-stone-700 w-20 shrink-0">{c.source}</span>
                    <div className="flex-1">
                      <span className="text-stone-500">{c.sync} · {c.freq}</span>
                      <div className="text-stone-400 text-[10px]">{c.notes}</div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </Card>
    </div>
  );
}

function EvaluationSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#0284c7">
          <Label color="#0284c7">Retrieval Quality Metrics</Label>
          <div className="space-y-2.5">
            {[
              { metric: "Recall@K", formula: "relevant docs in top-K / total relevant", target: "R@5 > 0.90", desc: "Is the right document in the top 5? The most important retrieval metric for RAG — if it's not retrieved, it can't be in the answer." },
              { metric: "nDCG@K", formula: "normalized discounted cumulative gain", target: "> 0.80", desc: "Are the most relevant documents ranked highest? Penalizes relevant docs appearing at low positions." },
              { metric: "MRR (Mean Reciprocal Rank)", formula: "1/rank of first relevant doc", target: "> 0.85", desc: "How quickly does the first relevant result appear? Important for single-answer queries." },
              { metric: "Precision@K", formula: "relevant docs in top-K / K", target: "P@5 > 0.60", desc: "What fraction of retrieved docs are relevant? Less critical for RAG (LLM can ignore irrelevant ones) but reduces noise." },
            ].map((m,i) => (
              <div key={i} className="border-b border-stone-100 pb-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold text-stone-700">{m.metric}</span>
                  <span className="text-[10px] font-mono text-cyan-700">{m.target}</span>
                </div>
                <div className="text-[9px] font-mono text-stone-400">{m.formula}</div>
                <div className="text-[10px] text-stone-500 mt-0.5">{m.desc}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#ea580c">
          <Label color="#ea580c">Answer Quality Metrics</Label>
          <div className="space-y-2.5">
            {[
              { metric: "Faithfulness (Grounding)", target: "> 0.95", desc: "% of claims in the answer supported by retrieved passages. The #1 RAG quality metric. Measures hallucination rate." },
              { metric: "Answer Relevance", target: "> 0.90", desc: "Does the answer address the user's actual question? Measured via LLM-as-judge or human evaluation." },
              { metric: "Citation Accuracy", target: "> 0.90", desc: "Are the cited sources actually supporting the claim they're attached to? LLMs sometimes cite the wrong source." },
              { metric: "Completeness", target: "> 0.80", desc: "Does the answer cover all aspects of the question? Partial answers are frustrating." },
              { metric: "Abstain Rate", target: "5-15%", desc: "How often does the system say 'I don't know'? Too high = poor retrieval. Too low = probably hallucinating." },
            ].map((m,i) => (
              <div key={i} className="border-b border-stone-100 pb-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold text-stone-700">{m.metric}</span>
                  <span className="text-[10px] font-mono text-orange-600">{m.target}</span>
                </div>
                <div className="text-[10px] text-stone-500 mt-0.5">{m.desc}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
      <Card>
        <Label color="#7c3aed">Evaluation Framework</Label>
        <CodeBlock code={`# RAG Evaluation Pipeline (offline + online)
class RAGEvaluator:
    # OFFLINE: golden test set of (query, relevant_docs, ideal_answer)
    def evaluate_offline(self, test_set):
        for query, gold_docs, gold_answer in test_set:
            # Retrieval quality
            retrieved = retriever.retrieve(query, k=5)
            recall_5 = len(set(retrieved) & set(gold_docs)) / len(gold_docs)
            ndcg_5 = compute_ndcg(retrieved, gold_docs, k=5)

            # Answer quality
            answer = generator.generate(query, retrieved)
            faithfulness = self.grounding_score(answer, retrieved)
            relevance = self.llm_judge_relevance(query, answer)
            citation_acc = self.verify_citations(answer, retrieved)

        return aggregate_metrics()

    # ONLINE: implicit user feedback
    def evaluate_online(self):
        metrics = {
            "click_through_rate": ...,       # user clicked a source link
            "thumbs_up_rate": ...,           # explicit positive feedback
            "reformulation_rate": ...,       # user rephrased query (= bad result)
            "answer_copy_rate": ...,         # user copied answer text (= useful)
            "follow_up_rate": ...,           # user asked follow-up (could be good or bad)
            "abandon_rate": ...,             # user left without interaction (= bad)
        }
        return metrics`} />
      </Card>
    </div>
  );
}

function ScalabilitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Multi-Tenant Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Index isolation</strong> — each tenant gets a dedicated vector index partition. No cross-tenant data leakage. Smaller tenants share index servers; large tenants get dedicated resources.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Embedding compute pooling</strong> — shared GPU pool for embedding generation across tenants. Auto-scale based on ingestion backlog. Batch similar-sized documents for efficient GPU utilization.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">LLM inference pooling</strong> — shared LLM serving cluster with per-tenant rate limiting. Use smaller models (Gemini Flash) for simple queries, larger models (Gemini Pro) for complex ones. Route based on query complexity.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Tiered storage</strong> — hot tenants (active querying) keep index in memory. Warm tenants (occasional use) load on-demand. Cold tenants (inactive) archived to disk. Auto-promote based on usage patterns.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Retrieval Latency Optimization</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Pre-filtered ANN</strong> — partition vector index by common filters (source_type, date_year). Query searches only the relevant partition. 3-5x faster than post-filtering.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Cached popular queries</strong> — cache retrieval results for frequent queries (top 10% of queries handle 50% of traffic). TTL-based invalidation when source docs change.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Parallel retrieval</strong> — dense and sparse retrieval run simultaneously. Total latency = max(dense, sparse), not sum. Both complete in under 50ms.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Streaming generation</strong> — start LLM generation while re-ranking is finishing. Stream first tokens to user immediately. Overlap retrieval, re-ranking, and generation in a pipeline.</Point>
          </ul>
        </Card>
      </div>
    </div>
  );
}

function WatchoutsSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        {[
          { title: "Hallucination", sev: "CRITICAL", desc: "LLM generates plausible but unsupported claims. In enterprise contexts (HR policy, legal, financial), a hallucinated answer can cause real harm — employees acting on wrong policy information.", fix: "Grounding verification (NLI or LLM-as-judge). Low temperature (0.1). Strong system prompt ('ONLY use provided context'). Grounding score threshold — suppress answers below 0.80. Display source documents alongside answer so users can verify.", icon: "🔴" },
          { title: "Retrieval Failure (Wrong Docs)", sev: "CRITICAL", desc: "The right document exists but isn't retrieved — so the LLM either halluccinates or says 'I don't know'. Root cause: poor chunking (relevant info split across chunks), embedding mismatch (query uses different vocabulary than doc), or stale index.", fix: "Hybrid retrieval (dense + sparse covers more cases). Query expansion (rewrite query in multiple ways). Hierarchical chunking (multiple granularities). Monitor Recall@5 continuously. A/B test retrieval improvements.", icon: "🔴" },
          { title: "Stale Index", sev: "HIGH", desc: "Document was updated 2 hours ago but the index still has the old version. User gets outdated HR policy, wrong financial data, or incorrect product information.", fix: "Streaming ingestion with <15 min freshness SLA. Change detection via content hashing. Priority queue for frequently-accessed docs. Display 'Last indexed: X ago' in UI so users know freshness. Source-of-record links for verification.", icon: "🟡" },
          { title: "ACL Leakage", sev: "CRITICAL", desc: "User sees content from a document they shouldn't have access to. Can happen if ACLs are synced with delay, or if the LLM includes information from a filtered-out document in its training knowledge.", fix: "Pre-retrieval ACL filtering (not post-generation). ACL sync within 5 minutes of permission change. Separate vector index per tenant (defense in depth). LLM only sees ACL-filtered passages — never the full corpus. Regular ACL audit logs.", icon: "🔴" },
          { title: "Multi-Lingual Quality Gap", sev: "MEDIUM", desc: "Embedding model trained primarily on English. Retrieval quality drops 20-30% for other languages. Cross-lingual queries (English query, Japanese doc) may fail entirely.", fix: "Multilingual embedding models (mE5, multilingual-e5-large). Query translation as a bridge (translate query to doc language before retrieval). Language-specific fine-tuning. Monitor retrieval quality per language.", icon: "🟠" },
          { title: "Cost Explosion from LLM Calls", sev: "HIGH", desc: "Every query with generation costs $0.003-0.01 in LLM inference. At 2,300 QPS platform-wide, this is $600K-2M/month in LLM costs alone. High context window queries with many retrieved chunks are especially expensive.", fix: "Route simple queries to search-only (no LLM). Use smaller models (Flash vs Pro) for simple queries. Compress context: summarize long passages before feeding to LLM. Cache LLM responses for frequent identical queries. Limit retrieved chunk count (5 vs 20).", icon: "🟡" },
        ].map((w,i) => (
          <Card key={i} accent="#dc2626">
            <div className="flex items-center gap-2 mb-2">
              <span>{w.icon}</span>
              <span className="text-[12px] font-bold text-stone-800">{w.title}</span>
              <Pill bg={w.sev==="CRITICAL"?"#fef2f2":w.sev==="HIGH"?"#fffbeb":"#fff7ed"} color={w.sev==="CRITICAL"?"#dc2626":w.sev==="HIGH"?"#d97706":"#ea580c"}>{w.sev}</Pill>
            </div>
            <p className="text-[12px] text-stone-500 mb-2">{w.desc}</p>
            <div className="text-[11px] text-emerald-700 bg-emerald-50 rounded-lg p-2.5">
              <strong>Fix:</strong> {w.fix}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

function EnhancementsSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        {[
          { title: "Multi-Step / Agentic RAG", d: "For complex queries, decompose into sub-queries, retrieve for each, synthesize. 'Compare Q3 and Q4 revenue' → retrieve Q3 data, retrieve Q4 data, compare.", effort: "Hard", detail: "Requires a query planner (LLM-based). Each sub-query goes through the full retrieval pipeline. Must handle failure of individual sub-queries gracefully." },
          { title: "Query Rewriting & Expansion", d: "LLM rewrites the user's query for better retrieval: expand abbreviations, resolve pronouns from conversation history, generate multiple query variants.", effort: "Medium", detail: "HyDE: generate a hypothetical answer, then search for documents similar to that answer. Surprisingly effective for complex queries." },
          { title: "Knowledge Graph Integration", d: "Build entity-relationship graph from documents. For entity-centric queries ('Who manages the APAC team?'), traverse the graph instead of full retrieval.", effort: "Hard", detail: "Graph RAG: Microsoft Research approach. Extract entities and relationships at ingestion time. Query can traverse relationships the embedding can't capture." },
          { title: "Fine-Tuned Embedding Models", d: "Train embedding model on enterprise-specific (query, document) pairs from search logs. Dramatically improves retrieval for domain-specific vocabulary.", effort: "Medium", detail: "Even 1000-5000 labeled pairs significantly help. Contrastive learning on in-batch negatives. Can improve Recall@5 by 10-20% over generic embeddings." },
          { title: "Structured Data Querying (NL2SQL)", d: "For questions about data in databases or spreadsheets, generate SQL from natural language. 'What was total revenue last quarter?' → SELECT SUM(revenue) ...", effort: "Hard", detail: "Requires schema understanding, SQL generation, result interpretation. Different from document retrieval — complementary capability." },
          { title: "Personalized Ranking", d: "Rank results based on user's role, team, recent activity. An engineer asking about 'deployment' wants the deploy guide, not the marketing deployment strategy.", effort: "Medium", detail: "User embedding from click history + role features. Personalization signal in re-ranking. Privacy-conscious: no cross-user learning." },
        ].map((e,i) => (
          <Card key={i}>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="text-[12px] font-bold text-stone-800">{e.title}</span>
              <Pill bg={e.effort==="Medium"?"#fffbeb":"#fef2f2"} color={e.effort==="Medium"?"#d97706":"#dc2626"}>{e.effort}</Pill>
            </div>
            <p className="text-[12px] text-stone-600 mb-1.5">{e.d}</p>
            <p className="text-[11px] text-stone-400">{e.detail}</p>
          </Card>
        ))}
      </div>
    </div>
  );
}

function FollowupsSection() {
  const [exp, setExp] = useState(null);
  const qas = [
    { q:"Why hybrid retrieval (dense + sparse) instead of just dense?", a:"Dense retrieval captures semantic similarity — great for conceptual queries ('How do I request time off?'). But it has critical blind spots: (1) Exact terms: error codes, employee IDs, product names, acronyms. 'ERR_SSL_PROTOCOL_ERROR' as an embedding looks like other SSL-related text, but you need EXACT string match. (2) Rare terms: if a word appeared rarely in embedding training data, the embedding quality is poor. (3) Negation: embeddings struggle with 'NOT covered by insurance' vs 'covered by insurance' — they look similar in embedding space. Sparse/BM25 retrieval handles all of these through exact term matching. Reciprocal Rank Fusion (RRF) combines both rankings without any training — it's the simplest fusion method and works remarkably well. In benchmarks, hybrid consistently outperforms either method alone by 5-15% on Recall@5.", tags:["retrieval"] },
    { q:"How do you evaluate a RAG system end-to-end?", a:"RAG evaluation has three layers: (1) Retrieval evaluation: given a golden set of (query, relevant_docs), measure Recall@K and nDCG@K. This tells you if the right documents are being found. Most RAG failures are retrieval failures. (2) Answer evaluation: given retrieved docs, is the generated answer faithful, relevant, and complete? Use grounding score (NLI-based), LLM-as-judge for relevance, and human evaluation for quality. (3) Online evaluation: in production, measure click-through rate on cited sources (users clicking = useful), reformulation rate (users rephrasing = bad result), thumbs up/down, and answer copy rate. The key insight: optimize retrieval first. Even a perfect LLM can't answer from the wrong documents. A 10% improvement in Recall@5 matters more than switching to a better LLM.", tags:["evaluation"] },
    { q:"How do you handle access control at scale?", a:"ACL enforcement is the #1 security requirement for enterprise RAG. Approach: (1) Pre-retrieval filtering: before ANN search, filter the candidate set to only documents the user can access. This is faster than post-filtering (fewer vectors to search) and has zero information leakage. (2) Implementation: maintain a Bloom filter per user of accessible doc_ids. Updated when permissions change. Pre-filter the ANN index using the Bloom filter. (3) ACL sync pipeline: connectors extract permissions from source systems (Drive ACLs, Confluence space permissions, Slack channel membership). Sync every 5 minutes. Permission changes must propagate before the user's next query. (4) Defense in depth: even if the retrieval layer leaks, the LLM prompt only contains ACL-filtered passages. The generation layer never sees unauthorized content. (5) Audit logging: every query logs which documents were retrieved and whether they passed ACL check. Required for compliance.", tags:["security"] },
    { q:"Cross-encoder re-ranking adds 200ms latency. Is it worth it?", a:"Absolutely — re-ranking is the highest-ROI component in any RAG pipeline. Here's why: (1) Bi-encoder retrieval has ~85% accuracy on 'is the right doc in top-5?'. Cross-encoder re-ranking pushes this to ~95%. That 10% improvement means 10% fewer wrong answers. (2) The 200ms is well within budget: total allowed is 3+ seconds (LLM generation dominates). 200ms for re-ranking is a small price for dramatically better input to the LLM. (3) Optimization: batch the cross-encoder scoring (50 pairs at once = ~100ms, not 250ms). Distill large cross-encoder to smaller model (MiniLM-L6 is 3x faster than L-12 with ~2% quality loss). Pre-compute re-ranking for common queries and cache results. (4) Impact on generation: better retrieval → better context → fewer hallucinations → higher grounding scores. The re-ranker pays for itself in answer quality.", tags:["architecture"] },
    { q:"How do you handle multi-turn conversations?", a:"Multi-turn requires conversation-aware query understanding. When user asks 'Does that apply to contractors?', the system must resolve 'that' to 'parental leave policy' from the previous turn. Approach: (1) Query rewriting: use the LLM to rewrite the follow-up query as a standalone query incorporating conversation context. 'Does that apply to contractors?' → 'Does the parental leave policy apply to contractors?' (2) Context window: include the last 3-5 turns of conversation in the LLM's generation prompt. This gives the LLM context for nuanced follow-ups. (3) Conversation-aware retrieval: the rewritten query goes through normal retrieval. But also consider retrieving from the same documents as previous turns (continuity bonus). (4) Session caching: cache retrieved documents from previous turns. If the follow-up is about the same document, skip retrieval entirely — just re-generate with the new question.", tags:["ux"] },
    { q:"What's the chunking strategy for very long documents (100+ pages)?", a:"Long documents (legal contracts, technical manuals, annual reports) need special handling: (1) Hierarchical chunking is essential. Create chunks at section, subsection, and paragraph level. Search at paragraph level (precise), retrieve at section level (context). (2) Table of contents as navigation: extract the ToC as a separate searchable element. 'What does section 4.2 say about liability?' can match the ToC entry first, then drill into section 4.2. (3) Summarization at each level: generate summaries for each section. Store summaries as additional searchable chunks. For broad queries, the summary matches; for specific queries, the detailed paragraph matches. (4) Cross-references: long documents have internal references ('as described in section 3.1'). Resolve these during chunking so each chunk is self-contained. (5) Sliding window with large overlap (25%): for fixed-size chunking fallback, large overlap ensures context isn't lost at boundaries.", tags:["chunking"] },
    { q:"How would you reduce hallucination to near-zero for legal/medical use cases?", a:"High-stakes domains require maximum grounding: (1) Extractive answers only: instead of generating free-form text, highlight exact passages from retrieved documents. Zero hallucination risk because you're showing the original text. Less user-friendly but maximally safe. (2) Dual verification: generate an answer, then run a separate NLI model to verify every claim against the source passages. Suppress any unsupported claims before showing to user. (3) Conservative abstention: set a high threshold for the grounding score. If below 0.95, show 'I found relevant documents but cannot provide a definitive answer — please review the sources directly.' Link to the source documents. (4) Constrained decoding: use the LLM's logit bias to strongly prefer tokens that appear in the retrieved passages. Penalizes novel token sequences. (5) Human-in-the-loop: for legal/medical queries, route to a domain expert for verification before displaying the answer. ML provides the draft; human validates.", tags:["safety"] },
    { q:"How does this system differ from web search (Google Search)?", a:"Key architectural differences: (1) Corpus: enterprise search indexes 10M internal docs with rich metadata and ACLs. Web search indexes 100B+ public pages with PageRank. (2) Retrieval: enterprise emphasizes exact match and freshness (policy docs, project plans). Web emphasizes authority (PageRank) and diversity. (3) ACLs: enterprise search MUST enforce per-user access control. Web search has no ACLs (everything is public). This is the hardest part of enterprise search. (4) Answer generation: enterprise can generate confident answers because the corpus is authoritative (official company docs). Web search must be more cautious (diverse, potentially contradictory sources). (5) Evaluation: enterprise measures answer accuracy against a known ground truth (the actual policy). Web search measures user satisfaction with diverse intent. (6) Freshness: enterprise docs change hourly. Web pages are relatively stable. Enterprise needs near-real-time index updates.", tags:["design"] },
    { q:"How do you handle documents that contradict each other?", a:"Document conflicts are common in enterprise: old policy vs new policy, draft vs final, different teams with different standards. Approach: (1) Recency bias: when multiple documents discuss the same topic, prefer the most recently updated version. The re-ranker should include recency as a boosting signal. (2) Source authority hierarchy: official policy > wiki > email > chat. Configure per-document-type authority levels. The re-ranker uses this as a feature. (3) Conflict detection in generation: the system prompt instructs the LLM to flag conflicting information: 'Source 1 says X, but Source 2 says Y. The most recent source (Source 2, updated Jan 2024) states Y.' (4) Version management: when a new version of a document replaces an old one, mark the old version as 'superseded' in metadata. Optionally exclude superseded documents from retrieval entirely. (5) Explicit contradiction UI: when the system detects conflicting sources, display both side-by-side and let the user judge.", tags:["quality"] },
    { q:"What's the role of fine-tuning vs prompt engineering in RAG?", a:"For RAG, prompt engineering is more important than LLM fine-tuning. Here's why: (1) The LLM's job in RAG is narrow: read passages and synthesize an answer with citations. This is well within the base model's capability. A well-crafted system prompt (with examples of good answers) is sufficient. (2) Fine-tuning the LLM is risky for RAG: it may teach the model to rely on its parametric knowledge instead of the retrieved context, increasing hallucination. (3) Where fine-tuning DOES help: the embedding model. Fine-tuning the bi-encoder on (query, relevant_doc) pairs from your domain dramatically improves retrieval quality. This is the highest-leverage fine-tuning in the RAG stack. (4) The cross-encoder re-ranker also benefits from domain-specific fine-tuning. (5) Summary: fine-tune retrieval models (embedding + re-ranker), prompt-engineer the generation model. This gives maximum quality with minimum hallucination risk.", tags:["ml"] },
  ];
  return (
    <div className="space-y-2.5">
      <Card className="bg-indigo-50/50 border-indigo-200 mb-1">
        <p className="text-[12px] text-stone-500">Common follow-up questions Google L6 interviewers ask about RAG and enterprise search. Click to reveal a strong answer.</p>
      </Card>
      {qas.map((qa,i) => (
        <div key={i} className="bg-white border border-stone-200 rounded-xl overflow-hidden shadow-sm">
          <button onClick={() => setExp(exp===i?null:i)} className="w-full flex items-center gap-3 px-5 py-3.5 text-left hover:bg-stone-50 transition-colors">
            <span className={`text-stone-400 text-sm transition-transform duration-200 ${exp===i?"rotate-90":""}`}>▸</span>
            <span className="text-[13px] text-stone-700 font-medium flex-1">{qa.q}</span>
            <div className="flex gap-1">
              {qa.tags.map(t => <span key={t} className="text-[9px] px-2 py-0.5 rounded-full bg-stone-100 text-stone-400">{t}</span>)}
            </div>
          </button>
          {exp===i && (
            <div className="px-5 pb-4 pt-1 border-t border-stone-100">
              <p className="text-[12px] text-stone-500 leading-relaxed">{qa.a}</p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}


/* ═══════════════════════════════════════════════════════════ */
const SECTION_COMPONENTS = {
  concept: ConceptSection, requirements: RequirementsSection, capacity: CapacitySection,
  api: ApiSection, design: DesignSection, chunking: ChunkingSection,
  embedding: EmbeddingSection, retrieval: RetrievalSection, reranking: RerankingSection,
  generation: GenerationSection, data: DataModelSection, ingestion: IngestionSection,
  evaluation: EvaluationSection, scalability: ScalabilitySection,
  watchouts: WatchoutsSection, enhancements: EnhancementsSection,
  followups: FollowupsSection,
};

export default function RAGSD() {
  const [active, setActive] = useState("concept");
  const refs = useRef({});

  const scrollTo = (id) => { setActive(id); refs.current[id]?.scrollIntoView({ behavior:"smooth", block:"start" }); };

  useEffect(() => {
    const obs = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) setActive(e.target.dataset.section);
    }, { rootMargin: "-15% 0px -65% 0px" });
    Object.values(refs.current).forEach(el => el && obs.observe(el));
    return () => obs.disconnect();
  }, []);

  return (
    <div className="min-h-screen" style={{ background: "#faf9f7", fontFamily: "'DM Sans', 'Segoe UI', system-ui, sans-serif" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />

      {/* Sticky Nav */}
      <div className="sticky top-0 z-50 border-b border-stone-200" style={{ background: "rgba(250,249,247,0.92)", backdropFilter: "blur(12px)" }}>
        <div className="max-w-7xl mx-auto px-5 py-3">
          <div className="flex items-center gap-3 mb-2.5">
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">RAG System / Enterprise Search</h1>
            <Pill bg="#f3e8ff" color="#7c3aed">ML System Design</Pill>
            <Pill bg="#fef2f2" color="#dc2626">Google L6</Pill>
          </div>
          <div className="flex gap-1.5 overflow-x-auto pb-0.5 -mb-0.5">
            {SECTIONS.map(s => (
              <button key={s.id} onClick={() => scrollTo(s.id)}
                className={`px-3 py-1.5 rounded-lg text-[11px] font-medium whitespace-nowrap transition-all border ${
                  active===s.id ? "text-white border-transparent" : "bg-white text-stone-500 border-stone-200 hover:border-stone-300 hover:text-stone-700"
                }`}
                style={active===s.id ? { background: s.color, borderColor: s.color } : {}}>
                <span className="mr-1">{s.icon}</span>{s.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-5 py-6 space-y-10">
        {SECTIONS.map(s => {
          const Comp = SECTION_COMPONENTS[s.id];
          return (
            <section key={s.id} ref={el => refs.current[s.id]=el} data-section={s.id}>
              <div className="flex items-center gap-3 mb-5">
                <span className="text-lg">{s.icon}</span>
                <h2 className="text-lg font-bold text-stone-800">{s.label}</h2>
                <div className="flex-1 h-px bg-stone-200" />
              </div>
              <Comp />
            </section>
          );
        })}
      </div>
    </div>
  );
}