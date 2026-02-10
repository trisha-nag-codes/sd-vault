import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   SEARCH RANKING — ML System Design Reference (Google L6)
   Pearl white theme · 17 sections (HLD + LLD + ML Deep Dive)
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",              icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",          icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",   icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",            icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",     icon: "🏗️", color: "#9333ea" },
  { id: "query",         label: "Query Understanding",   icon: "🔍", color: "#c026d3" },
  { id: "retrieval",     label: "Retrieval & Indexing",  icon: "📑", color: "#059669" },
  { id: "ranking",       label: "Ranking Model",         icon: "🧠", color: "#dc2626" },
  { id: "features",      label: "Feature Engineering",   icon: "⚙️", color: "#d97706" },
  { id: "data",          label: "Data Model",            icon: "🗄️", color: "#0f766e" },
  { id: "training",      label: "Training Pipeline",     icon: "🔄", color: "#7e22ce" },
  { id: "scalability",   label: "Scalability",           icon: "📈", color: "#059669" },
  { id: "availability",  label: "Availability",          icon: "🛡️", color: "#d97706" },
  { id: "observability", label: "Observability",         icon: "📊", color: "#0284c7" },
  { id: "watchouts",     label: "Failure Modes",         icon: "⚠️", color: "#dc2626" },
  { id: "enhancements",  label: "Enhancements",          icon: "🚀", color: "#7c3aed" },
  { id: "followups",     label: "Follow-up Questions",   icon: "❓", color: "#6366f1" },
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

function DiagramBox({ x, y, w, h, label, color, sub }) {
  const lines = label.split("\n");
  return (
    <g>
      <rect x={x-w/2} y={y-h/2} width={w} height={h} rx={8} fill={color+"12"} stroke={color} strokeWidth={1.5}/>
      {lines.map((l, i) => (
        <text key={i} x={x} y={y+(i-(lines.length-1)/2)*13-(sub?4:0)} textAnchor="middle" dominantBaseline="central" fill={color} fontSize="10" fontWeight="600" fontFamily="monospace">{l}</text>
      ))}
      {sub && <text x={x} y={y+(lines.length-1)/2*13+10} textAnchor="middle" fill={color+"90"} fontSize="8" fontFamily="monospace">{sub}</text>}
    </g>
  );
}
function Arrow({ x1,y1,x2,y2,label,dashed,id }) {
  const mx=(x1+x2)/2, my=(y1+y2)/2-10;
  return (
    <g>
      <defs><marker id={`ah-${id}`} markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#94a3b8" strokeWidth={1.5} strokeDasharray={dashed?"5,3":"none"} markerEnd={`url(#ah-${id})`}/>
      {label && <text x={mx} y={my} textAnchor="middle" fill="#64748b" fontSize="8" fontFamily="monospace">{label}</text>}
    </g>
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
            <Label>What is a Search Ranking System?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A search ranking system takes a user's query, retrieves relevant documents from a corpus of billions of pages, and returns them ordered by relevance, quality, and freshness. It's the core product of Google, Bing, and every e-commerce search.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              The fundamental challenge: given a query like "best restaurants near me", find the 10 most relevant results out of <strong>100+ billion indexed documents</strong> in under 500ms. This requires a multi-stage funnel — retrieval (find candidates), then ranking (score them), then re-ranking (apply business logic, diversity, freshness).
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard? (Google L6 Depth)</Label>
            <ul className="space-y-2.5">
              <Point icon="📊" color="#0891b2">Vocabulary mismatch — user searches "cheap flights" but the best doc says "affordable airfare". Lexical matching fails; you need semantic understanding.</Point>
              <Point icon="⏱️" color="#0891b2">Latency at scale — scoring 100B docs with a neural model is impossible in 500ms. You need a funnel: BM25 retrieves 10K → L1 model scores 1K → L2 model reranks 100 → return top 10.</Point>
              <Point icon="🎯" color="#0891b2">Relevance is multi-dimensional — query-document match, document quality (PageRank, authority), freshness, location, personalization, intent satisfaction.</Point>
              <Point icon="🔄" color="#0891b2">Feedback loops — clicks ≠ satisfaction. Users click clickbait, then bounce. You need dwell time, reformulation rate, pogo-sticking detection as quality signals.</Point>
              <Point icon="⚔️" color="#0891b2">Adversarial — SEO spam actively games your ranking signals. Your system must be robust to manipulation while still using the signals that matter.</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Scale</Label>
            <div className="space-y-2.5">
              {[
                { co: "Google", scale: "8.5B queries/day, 100B+ docs indexed", approach: "Multi-stage: BM25 → BERT → deep reranker" },
                { co: "Bing", scale: "1B+ queries/day", approach: "BM25 + transformer reranking" },
                { co: "Amazon", scale: "Product search, 350M+ products", approach: "Multi-objective: relevance + purchase probability" },
                { co: "YouTube", scale: "Video search + recommendation hybrid", approach: "Two-tower retrieval → deep ranking" },
                { co: "Spotify", scale: "Audio search + discovery", approach: "Embedding retrieval + LTR reranking" },
              ].map((e,i) => (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-20 shrink-0">{e.co}</span>
                  <span className="text-stone-500 flex-1">{e.scale}</span>
                  <span className="text-stone-400 text-[10px]">{e.approach}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <Label color="#2563eb">The Multi-Stage Funnel (Preview)</Label>
            <svg viewBox="0 0 360 180" className="w-full">
              {/* Funnel stages */}
              <polygon points="40,10 320,10 280,55 80,55" fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
              <text x={180} y={28} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="700" fontFamily="monospace">Retrieval (BM25 + ANN)</text>
              <text x={180} y={42} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">100B docs → 10K candidates  |  ~50ms</text>

              <polygon points="80,60 280,60 250,105 110,105" fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
              <text x={180} y={78} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="700" fontFamily="monospace">L1 Ranking (Lightweight Model)</text>
              <text x={180} y={92} textAnchor="middle" fill="#9333ea80" fontSize="7" fontFamily="monospace">10K → 500 candidates  |  ~100ms</text>

              <polygon points="110,110 250,110 230,155 130,155" fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
              <text x={180} y={128} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="700" fontFamily="monospace">L2 Reranking (Heavy Model)</text>
              <text x={180} y={142} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">500 → 10 results  |  ~200ms</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Google Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">The #1 most relevant ML system design for Google</div>
              </div>
              <span className="text-indigo-500 font-bold text-sm">★★★★★</span>
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
            <p className="text-[12px] text-stone-500 mt-0.5">"Design search ranking" is intentionally vague at L6. Don't ask "what kind of search?" — instead, <strong>propose a scope and justify it</strong>: "I'll focus on web search ranking — the pipeline from query to ranked results. I'll cover query understanding, retrieval, multi-stage ranking, and the training feedback loop. I'll keep crawling/indexing as a black box." This shows you drive ambiguity.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Given a text query, return top-K (10) ranked results with title, snippet, URL</Point>
            <Point icon="2." color="#059669">Handle multiple query types: navigational ("facebook login"), informational ("how to bake bread"), transactional ("buy iPhone 16")</Point>
            <Point icon="3." color="#059669">Support query understanding: spell correction, entity recognition, intent classification</Point>
            <Point icon="4." color="#059669">Results should reflect relevance, quality, freshness, and personalization</Point>
            <Point icon="5." color="#059669">Support pagination and "did you mean" suggestions</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">End-to-end latency: p50 &lt;300ms, p99 &lt;800ms (including network)</Point>
            <Point icon="2." color="#dc2626">Throughput: 100K+ QPS (Google scale: ~100K queries/sec)</Point>
            <Point icon="3." color="#dc2626">Index size: 100B+ documents, updated continuously (freshness)</Point>
            <Point icon="4." color="#dc2626">Availability: 99.99% — search must never go down</Point>
            <Point icon="5." color="#dc2626">Ranking quality: measurable via NDCG@10, MRR, user satisfaction metrics</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask (L6 Signal: Shows Breadth)</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Web search, product search, or enterprise search? (changes ranking signals entirely)",
            "What's our index size and freshness requirement? (minutes vs. hours)",
            "Do we personalize results? (search history, location, language)",
            "Multi-language or single-language? (affects tokenization, embeddings)",
            "Do we need to handle structured queries? (filters, facets, price range)",
            "What's our serving budget per query? (GPU, CPU, latency constraints)",
            "Ads integration? (organic + paid results interleaving)",
            "Do we need to support multimodal? (image search, voice queries)",
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
      <Card className="bg-violet-50/50 border-violet-200">
        <div className="flex items-start gap-3">
          <span className="text-lg">💡</span>
          <div>
            <div className="text-[12px] font-bold text-violet-700">Interview Tip — Derive, Don't Memorize</div>
            <p className="text-[12px] text-stone-500 mt-0.5">Walk through each step aloud. For ML system design, focus on: QPS, index size, model serving cost (GPU vs CPU), feature store lookups per query, and training data volume. These drive architecture decisions.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Traffic & Query Volume</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Total queries/day = 8.5B (Google scale)" result="8.5B" note="Ask interviewer for scale. Google: 8.5B, Bing: ~1B, mid-tier: 100M." />
            <MathStep step="2" formula="QPS = 8.5B / 86,400 ≈ 8.5B / 100K" result="~100K QPS" note="Average QPS. This is the serving load." final />
            <MathStep step="3" formula="Peak QPS = 100K × 3" result="~300K QPS" note="3× peak multiplier (events, news, elections)" />
            <MathStep step="4" formula="Avg results per query = 10 (page 1)" result="10 docs scored" note="But we score 500+ candidates to pick top 10" />
            <MathStep step="5" formula="Total docs scored/sec = 100K × 500" result="50M scores/sec" note="This is the L1 ranking model serving load" final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Index & Document Scale</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Total indexed documents" result="100B+" note="Web: 100B+ pages. Product search: 100M-1B items." />
            <MathStep step="2" formula="Avg document size (indexed) = ~5 KB" result="5 KB" note="Tokenized content, metadata, quality signals — not raw HTML" />
            <MathStep step="3" formula="Index size = 100B × 5 KB" result="~500 TB" note="Inverted index + document store + forward index" final />
            <MathStep step="4" formula="Embedding index (dense vectors) = 100B × 768 floats × 4B" result="~300 TB" note="Dense retrieval vectors. Quantized: ~75 TB with PQ" />
            <MathStep step="5" formula="Freshness: new/updated docs per day" result="~1B/day" note="Web crawl + real-time indexing for news/social" />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — ML Model Serving Budget</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="L1 model: score 500 docs per query" result="500 inferences" note="Lightweight model: GBDT or small neural net. ~0.1ms per doc." />
            <MathStep step="2" formula="L1 latency budget: 500 × 0.1ms (batched)" result="~20ms" note="GPU batching: score 500 docs in one forward pass" />
            <MathStep step="3" formula="L2 reranker: score 50 docs per query" result="50 inferences" note="Heavy model: BERT/cross-encoder. ~5ms per doc." />
            <MathStep step="4" formula="L2 latency budget: 50 × 5ms (batched)" result="~50ms" note="Must fit within total 200ms ranking budget" final />
            <MathStep step="5" formula="GPU fleet for L2: 100K QPS × 50 inferences" result="~2K GPUs" note="A100 GPU handles ~2.5K inferences/sec for BERT-base" />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Training Data Scale</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Click logs per day = 8.5B queries × 3 clicks avg" result="~25B clicks/day" note="Primary training signal. Need to de-bias position effects." />
            <MathStep step="2" formula="Human relevance labels (raters)" result="~10M/year" note="Expensive but gold-standard. Used for eval + fine-tuning." />
            <MathStep step="3" formula="Training data retention" result="~90 days" note="Older data loses relevance. Web changes rapidly." final />
            <MathStep step="4" formula="Training pipeline frequency" result="Daily → Weekly" note="L1 model: retrained daily. L2 model: weekly (expensive)." />
          </div>
        </Card>
      </div>
      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Query QPS", val: "~100K", sub: "Peak: 300K" },
            { label: "Index Size", val: "~500 TB", sub: "100B+ documents" },
            { label: "Docs Scored/sec", val: "~50M", sub: "L1 ranking load" },
            { label: "Latency Budget", val: "<500ms", sub: "End-to-end p99" },
          ].map((s,i) => (
            <div key={i} className="text-center py-3 rounded-lg bg-stone-50 border border-stone-200">
              <div className="text-[18px] font-bold text-violet-700 font-mono">{s.val}</div>
              <div className="text-[11px] font-medium text-stone-600 mt-0.5">{s.label}</div>
              <div className="text-[10px] text-stone-400">{s.sub}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function ApiSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#2563eb">
          <Label color="#2563eb">Search API</Label>
          <CodeBlock code={`# GET /v1/search?q=<query>&offset=0&limit=10
# Returns: ranked search results for query
#
# Request parameters:
#   q          - query string (required)
#   offset     - pagination offset (default 0)
#   limit      - results per page (default 10, max 50)
#   lang       - language hint (auto-detected if absent)
#   location   - lat,lng for local results
#   safe       - safe search filter (on/off)
#   freshness  - time filter (day/week/month/year)
#
# Response:
{
  "query": "best pizza nyc",
  "corrected_query": null,
  "did_you_mean": null,
  "intent": "local_informational",
  "results": [
    {
      "doc_id": "d_abc123",
      "url": "https://example.com/best-pizza-nyc",
      "title": "The 15 Best Pizza Places in NYC (2024)",
      "snippet": "From classic NY-style slices to...",
      "display_url": "example.com › food › pizza",
      "freshness": "2024-12-01",
      "relevance_score": 0.94,   # internal, not exposed
      "features": {              # rich results
        "type": "listicle",
        "thumbnail": "https://cdn.../thumb.jpg"
      }
    }
  ],
  "total_results": 1_240_000,
  "latency_ms": 187,
  "next_offset": 10
}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Design Decisions (L6 Depth)</Label>
          <div className="space-y-3">
            {[
              { q: "Why offset pagination, not cursor?", a: "Search results change between requests (index updates, personalization drift). Cursors assume stable result sets. Offset is simpler and users rarely go past page 3. Google uses offset." },
              { q: "Why return snippet, not full content?", a: "Bandwidth. 10 results × full page content = megabytes. Snippets are pre-computed during indexing (extract best matching passage). BERT-based snippet generation for query-dependent highlighting." },
              { q: "How is corrected_query generated?", a: "Spell correction pipeline: (1) character-level edit distance, (2) language model probability P(correction | context), (3) query log frequency — 'did you mean' only if corrected query has significantly higher search volume." },
              { q: "Why include intent in the response?", a: "Intent drives result presentation. Navigational → single site link. Informational → knowledge panel + organic. Transactional → product cards. Local → map pack. Frontend uses intent to pick the right layout." },
              { q: "Safe search implementation?", a: "Document-level classifier (NSFW score) at index time. Query-level intent classifier at query time. Filter at retrieval stage (cheapest) rather than post-ranking. Allow override for authenticated adults." },
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
        <Label color="#9333ea">Full System Architecture — Multi-Stage Ranking Pipeline</Label>
        <svg viewBox="0 0 720 380" className="w-full">
          <defs><marker id="ah-hld" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>

          {/* User query */}
          <rect x={10} y={60} width={70} height={36} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={45} y={76} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">User</text>
          <text x={45} y={88} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">query</text>

          {/* Query Understanding */}
          <rect x={110} y={45} width={100} height={55} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={160} y={65} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">Query</text>
          <text x={160} y={77} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">Understanding</text>
          <text x={160} y={92} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">spell · intent · NER</text>

          {/* Retrieval */}
          <rect x={240} y={30} width={100} height={40} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={290} y={48} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">BM25 Retrieval</text>
          <text x={290} y={62} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">inverted index</text>

          <rect x={240} y={80} width={100} height={40} rx={6} fill="#059669" stroke="#059669" strokeWidth={1.5} fill="#05966910"/>
          <text x={290} y={98} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">ANN Retrieval</text>
          <text x={290} y={112} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">dense vectors</text>

          {/* Merge */}
          <rect x={370} y={55} width={70} height={40} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={405} y={72} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Merge</text>
          <text x={405} y={85} textAnchor="middle" fill="#78716c80" fontSize="7" fontFamily="monospace">dedup · union</text>

          {/* L1 Ranking */}
          <rect x={470} y={50} width={90} height={50} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={515} y={68} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">L1 Ranking</text>
          <text x={515} y={80} textAnchor="middle" fill="#9333ea80" fontSize="7" fontFamily="monospace">GBDT / light NN</text>
          <text x={515} y={92} textAnchor="middle" fill="#9333ea80" fontSize="7" fontFamily="monospace">10K → 500</text>

          {/* L2 Reranking */}
          <rect x={590} y={50} width={100} height={50} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={640} y={68} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">L2 Reranker</text>
          <text x={640} y={80} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">BERT cross-enc</text>
          <text x={640} y={92} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">500 → 10</text>

          {/* Data stores */}
          <rect x={240} y={160} width={90} height={35} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={285} y={180} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Inverted Index</text>

          <rect x={370} y={160} width={90} height={35} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={415} y={180} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Vector Index</text>

          <rect x={500} y={160} width={90} height={35} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={545} y={180} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Feature Store</text>

          <rect x={500} y={210} width={90} height={35} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={545} y={230} textAnchor="middle" fill="#0891b2" fontSize="9" fontWeight="600" fontFamily="monospace">Doc Store</text>

          {/* Arrows */}
          <line x1={80} y1={78} x2={110} y2={72} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={210} y1={60} x2={240} y2={50} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={210} y1={80} x2={240} y2={95} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={340} y1={48} x2={370} y2={65} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={340} y1={100} x2={370} y2={82} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={440} y1={75} x2={470} y2={75} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={560} y1={75} x2={590} y2={75} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>

          {/* Data connections */}
          <line x1={290} y1={70} x2={285} y2={160} stroke="#d9770640" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>
          <line x1={290} y1={120} x2={415} y2={160} stroke="#d9770640" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>
          <line x1={515} y1={100} x2={545} y2={160} stroke="#d9770640" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>
          <line x1={640} y1={100} x2={545} y2={210} stroke="#0891b240" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>

          {/* Legend */}
          <rect x={15} y={270} width={695} height={100} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={25} y={290} fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 1 — Query Understanding (~20ms): Spell correct → intent classify → NER → query expand → generate query embedding</text>
          <text x={25} y={307} fill="#2563eb" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 2 — Retrieval (~50ms): BM25 (lexical) + ANN (semantic) in parallel. Merge + dedup → 10K candidates</text>
          <text x={25} y={324} fill="#9333ea" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 3 — L1 Ranking (~100ms): Lightweight model (GBDT/small NN) scores 10K → keep top 500. Features: BM25, PageRank, freshness</text>
          <text x={25} y={341} fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 4 — L2 Reranking (~200ms): Heavy model (BERT cross-encoder) reranks 500 → top 10. Query-doc interaction features</text>
          <text x={25} y={358} fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Key insight: Each stage is 10-100× more expensive per doc but sees 10-100× fewer docs. Total latency budget: ~500ms.</text>
        </svg>
      </Card>

      <Card>
        <Label color="#d97706">Latency Breakdown (p50)</Label>
        <div className="flex gap-2 items-end mt-2" style={{ height: 120 }}>
          {[
            { stage: "Query\nUnderstanding", ms: 20, color: "#c026d3", pct: "4%" },
            { stage: "BM25\nRetrieval", ms: 50, color: "#2563eb", pct: "17%" },
            { stage: "ANN\nRetrieval", ms: 40, color: "#059669", pct: "13%" },
            { stage: "Merge +\nDedup", ms: 10, color: "#78716c", pct: "3%" },
            { stage: "L1 Ranking\n(GBDT)", ms: 80, color: "#9333ea", pct: "27%" },
            { stage: "L2 Rerank\n(BERT)", ms: 100, color: "#dc2626", pct: "33%" },
            { stage: "Snippet\nGen", ms: 10, color: "#d97706", pct: "3%" },
          ].map((s,i) => (
            <div key={i} className="flex-1 flex flex-col items-center gap-1">
              <div className="text-[9px] font-mono font-bold" style={{ color: s.color }}>{s.ms}ms</div>
              <div className="w-full rounded-t" style={{ height: `${s.ms * 0.9}px`, background: s.color + "20", border: `1px solid ${s.color}40` }}/>
              <div className="text-[8px] text-stone-500 text-center whitespace-pre-line leading-tight">{s.stage}</div>
            </div>
          ))}
        </div>
        <div className="mt-3 pt-2 border-t border-stone-100 text-[11px] text-stone-500">
          <strong className="text-stone-700">Total p50: ~310ms.</strong> L1 + L2 ranking dominates. This is why the funnel approach is critical — we can't afford to run BERT on 10K docs.
        </div>
      </Card>
    </div>
  );
}

function QuerySection() {
  const [sel, setSel] = useState("spell");
  const components = {
    spell: { name: "Spell Correction", desc: "Fix typos before retrieval. Critical — 10-15% of queries have typos.",
      code: `# Spell Correction Pipeline
spell_correct(query: str) -> (corrected: str, confidence: float):
    # 1. Check if query exists in query log (exact match)
    if query_log.frequency(query) > THRESHOLD:
        return (query, 1.0)  # Known query, no correction

    # 2. Generate candidates via edit distance
    candidates = []
    for token in query.split():
        edits = generate_edits(token, max_dist=2)
        for edit in edits:
            freq = unigram_freq[edit]
            candidates.append((edit, freq))

    # 3. Score candidates with language model
    best = None
    for candidate_query in reconstruct(candidates):
        # P(correction) = P(candidate) × P(query|candidate)
        # P(query|candidate) = noisy channel model
        score = lm_score(candidate_query) * error_model(query, candidate_query)
        if score > best:
            best = (candidate_query, score)

    # 4. Only correct if confidence is high
    if best.score / lm_score(query) > 2.0:  # 2× more likely
        return best
    return (query, 1.0)

# Key: use query logs as dictionary, not just Webster's.
# "iPhone" wouldn't be in a dictionary but has high query freq.` },
    intent: { name: "Intent Classification", desc: "Classify query intent to change retrieval strategy and result presentation.",
      code: `# Intent Classification
# Classes: navigational, informational, transactional, local
classify_intent(query: str, context: dict) -> Intent:
    features = {
        "query_text": query,
        "has_url_pattern": bool(re.match(r'\\w+\\.\\w+', query)),
        "has_question_word": any(w in query for w in ["how","what","why","when"]),
        "has_commercial_word": any(w in query for w in ["buy","price","deal","cheap"]),
        "has_location_signal": context.get("location") is not None,
        "query_length": len(query.split()),
        "entity_types": ner_model.predict(query),  # [BRAND, PRODUCT, LOCATION, ...]
    }

    # Multi-class classifier (fine-tuned BERT or GBDT on features)
    intent = intent_model.predict(features)

    # Intent determines:
    # - navigational → boost exact-match URL, show sitelinks
    # - informational → show knowledge panel, featured snippet
    # - transactional → show product cards, price comparison
    # - local → show map pack, reviews, hours
    return intent

# Training data: human-labeled query logs (~500K labeled examples)
# Refresh: monthly, as query patterns shift seasonally` },
    ner: { name: "Entity Recognition", desc: "Extract entities to enable structured retrieval and knowledge graph lookups.",
      code: `# Named Entity Recognition for Queries
# Short text NER is harder than document NER — less context
extract_entities(query: str) -> List[Entity]:
    # Fine-tuned BERT-NER on search queries
    # Labels: PERSON, ORG, PRODUCT, LOCATION, DATE, ATTRIBUTE
    tokens = tokenize(query)
    predictions = ner_model.predict(tokens)  # BIO tagging

    entities = []
    for span, label in group_bio(predictions):
        entity = Entity(
            text=span,
            type=label,
            # Link to knowledge graph
            kg_id=entity_linker.link(span, label),
            # Confidence
            confidence=predictions[span].score
        )
        entities.append(entity)

    return entities

# Example: "Apple MacBook Pro 16 inch price"
# → [Entity("Apple", ORG, kg:Q312), Entity("MacBook Pro 16", PRODUCT, kg:Q98765)]
# → Enables: product search, price comparison widget, specs panel

# Key challenge: "Apple" = company or fruit?
# Solution: entity linking with knowledge graph disambiguates` },
    expand: { name: "Query Expansion", desc: "Add related terms to improve recall. Especially important for tail queries.",
      code: `# Query Expansion — Improve Recall
expand_query(query: str, entities: List[Entity]) -> ExpandedQuery:
    expanded_terms = []

    # 1. Synonym expansion from knowledge graph
    for entity in entities:
        synonyms = kg.get_synonyms(entity.kg_id)
        expanded_terms.extend(synonyms[:3])
        # "NYC" → ["New York City", "New York"]

    # 2. Query reformulation from click logs
    #    If users who searched X often also searched Y
    related = query_coclick_graph.get_related(query, top_k=3)
    expanded_terms.extend(related)
    # "cheap flights" → ["affordable flights", "budget airlines"]

    # 3. Embedding-based expansion
    #    Find nearest queries in embedding space
    q_emb = query_encoder.encode(query)
    similar = ann_index.search(q_emb, top_k=5)
    expanded_terms.extend([s.text for s in similar if s.score > 0.8])

    # 4. Build expanded query with original terms boosted
    return ExpandedQuery(
        original=query,          # boost: 1.0
        expansions=expanded_terms, # boost: 0.3-0.5
        operator="OR"             # any expanded term can match
    )

# Careful: expansion can hurt precision. Monitor NDCG.
# A/B test every expansion strategy.` },
  };
  const c = components[sel];
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Query Understanding Pipeline</Label>
        <p className="text-[12px] text-stone-500 mb-3">Before any document retrieval, the raw query goes through a transformation pipeline. Each component improves recall and precision independently. At Google, this accounts for ~20ms of the latency budget.</p>
        <svg viewBox="0 0 680 70" className="w-full">
          <defs><marker id="ah-qu" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          {[
            { x: 60, label: "Raw Query", color: "#78716c" },
            { x: 190, label: "Spell Correct", color: "#c026d3" },
            { x: 320, label: "Intent Classify", color: "#9333ea" },
            { x: 450, label: "NER + Linking", color: "#2563eb" },
            { x: 580, label: "Query Expand", color: "#059669" },
          ].map((s,i) => (
            <g key={i}>
              <rect x={s.x-55} y={18} width={110} height={34} rx={6} fill={s.color+"10"} stroke={s.color} strokeWidth={1.5}/>
              <text x={s.x} y={39} textAnchor="middle" fill={s.color} fontSize="9" fontWeight="600" fontFamily="monospace">{s.label}</text>
              {i < 4 && <line x1={s.x+55} y1={35} x2={s.x+75} y2={35} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-qu)"/>}
            </g>
          ))}
        </svg>
      </Card>
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-3 space-y-2">
          {Object.entries(components).map(([k,v]) => (
            <button key={k} onClick={() => setSel(k)}
              className={`w-full text-left px-3.5 py-2.5 rounded-lg text-[12px] font-medium border transition-all ${k===sel?"bg-purple-600 text-white border-purple-600":"bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
              {v.name}
            </button>
          ))}
        </div>
        <div className="col-span-9 space-y-5">
          <Card>
            <div className="flex items-center gap-3 mb-2">
              <span className="text-[14px] font-bold text-stone-800">{c.name}</span>
            </div>
            <p className="text-[12px] text-stone-500 mb-3">{c.desc}</p>
          </Card>
          <CodeBlock title={`${c.name} — Pseudocode`} code={c.code} />
        </div>
      </div>
    </div>
  );
}

function RetrievalSection() {
  const [sel, setSel] = useState("bm25");
  const methods = {
    bm25: { name: "BM25 (Lexical)", cx: "Precision on exact matches",
      desc: "Term-frequency based scoring using inverted index. The workhorse of search for 30+ years. Fast, interpretable, excellent for exact keyword matching.",
      pros: ["Extremely fast — O(k) where k = matching docs per term", "No ML model needed — pure math on term stats", "Excellent for navigational queries (exact URL/brand match)", "Index is highly compressed (posting lists)"],
      cons: ["Vocabulary mismatch — 'cheap flights' won't match 'affordable airfare'", "No semantic understanding — treats query as bag of words", "Struggles with long, complex queries", "Can't leverage user context or personalization"],
      code: `# BM25 Scoring
score_bm25(query, doc):
    score = 0
    for term in query.terms:
        tf = term_freq(term, doc)      # how many times term appears in doc
        df = doc_freq(term)            # how many docs contain this term
        dl = doc_length(doc)           # number of tokens in doc
        avgdl = avg_doc_length()       # corpus average

        # IDF: rare terms score higher
        idf = log((N - df + 0.5) / (df + 0.5) + 1)

        # TF normalization: diminishing returns + length penalty
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

        score += idf * tf_norm
    return score

# Typical parameters: k1=1.2, b=0.75
# Served from inverted index:
#   term → [(doc_id, tf, positions), ...]
# Index sharded by doc_id range across N machines
# Each shard scans its posting lists in parallel` },
    ann: { name: "ANN Dense Retrieval", cx: "Recall on semantic matches",
      desc: "Encode query and documents into dense vectors, then find nearest neighbors. Catches semantic matches that BM25 misses.",
      pros: ["Semantic matching — 'cheap flights' finds 'affordable airfare'", "Works for zero-shot / unseen queries", "Handles multilingual (shared embedding space)", "Can encode user context into query vector"],
      cons: ["Approximate — may miss exact matches BM25 would find", "Expensive to build and maintain vector index", "Training data needed for good embeddings", "Harder to debug than BM25 (black box vectors)"],
      code: `# Dense Retrieval with Two-Tower Model
# Offline: encode all documents
for doc in corpus:
    doc_emb = doc_encoder(doc.title + doc.content)  # 768-dim
    vector_index.add(doc.id, doc_emb)

# Online: encode query and search
retrieve_dense(query: str, top_k: int = 1000) -> List[DocScore]:
    # Encode query (~5ms on GPU)
    q_emb = query_encoder(query)  # 768-dim

    # ANN search (~20ms for 100B vectors with HNSW)
    candidates = vector_index.search(
        q_emb,
        top_k=top_k,
        algo="HNSW",   # Hierarchical Navigable Small World
        ef_search=128  # beam width (recall vs latency tradeoff)
    )

    return [(doc_id, cosine_sim) for doc_id, cosine_sim in candidates]

# Two-Tower architecture:
#   Query tower: BERT-base (shared weights possible)
#   Doc tower: BERT-base
#   Trained on: (query, clicked_doc) pairs
#   Negative sampling: in-batch negatives + hard negatives
#
# Index: HNSW with PQ (product quantization)
#   - 768-dim → 96 sub-vectors × 8-bit codes = 96 bytes/doc
#   - 100B docs × 96B = ~10 TB (fits in distributed memory)` },
    hybrid: { name: "Hybrid (BM25 + ANN) ★", cx: "Best of both worlds",
      desc: "Run BM25 and ANN in parallel, merge results. This is what Google, Bing, and modern search engines actually do.",
      pros: ["Catches both exact matches (BM25) and semantic matches (ANN)", "More robust — if one fails, the other still works", "Parallel execution — no additional latency", "Can weight BM25 vs ANN per query type"],
      cons: ["Need to maintain two index systems", "Merge/dedup logic adds complexity", "Score normalization across different systems is tricky", "Double the indexing pipeline cost"],
      code: `# Hybrid Retrieval — Parallel BM25 + ANN
retrieve_hybrid(query: str, query_info: QueryInfo) -> List[DocScore]:
    # Run both retrievals in parallel
    bm25_future = async bm25_retrieve(query, top_k=5000)
    ann_future  = async ann_retrieve(query, top_k=5000)

    bm25_results = await bm25_future   # ~50ms
    ann_results  = await ann_future     # ~40ms

    # Normalize scores to [0, 1]
    bm25_norm = min_max_normalize(bm25_results)
    ann_norm  = min_max_normalize(ann_results)

    # Reciprocal Rank Fusion (RRF) — simple and effective
    combined = {}
    for rank, (doc_id, _) in enumerate(bm25_norm):
        combined[doc_id] = combined.get(doc_id, 0) + 1 / (60 + rank)
    for rank, (doc_id, _) in enumerate(ann_norm):
        combined[doc_id] = combined.get(doc_id, 0) + 1 / (60 + rank)

    # Alternative: weighted linear combination
    # alpha = 0.6 if navigational_query else 0.4  # boost BM25 for navigational
    # score = alpha * bm25_score + (1 - alpha) * ann_score

    # Sort by combined score, take top 10K for L1 ranking
    return sorted(combined.items(), key=lambda x: -x[1])[:10000]

# RRF constant 60 is standard (from Cormack et al.)
# Key advantage: no score calibration needed — uses rank only` },
  };
  const m = methods[sel];
  return (
    <div className="space-y-5">
      <Card accent="#059669">
        <Label color="#059669">Retrieval Strategy Comparison</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[12px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Method</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Latency</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Precision</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Recall</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Index Size</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Used By</th>
            </tr></thead>
            <tbody>
              {[
                { n:"BM25 (Lexical)", l:"~50ms", p:"★★★★★", r:"★★★☆☆", idx:"~100 TB", u:"All engines (base)", hl:false },
                { n:"ANN (Dense)", l:"~40ms", p:"★★★☆☆", r:"★★★★★", idx:"~10 TB (PQ)", u:"Google, Bing", hl:false },
                { n:"Hybrid ★★", l:"~50ms (parallel)", p:"★★★★★", r:"★★★★★", idx:"~110 TB", u:"Google, Bing, modern", hl:true },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-emerald-50" : ""}>
                  <td className={`px-3 py-2 font-mono ${r.hl?"text-emerald-700 font-bold":"text-stone-600"}`}>{r.n}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.l}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.p}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.r}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.idx}</td>
                  <td className="text-center px-3 py-2 text-stone-400">{r.u}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-3 space-y-2">
          {Object.entries(methods).map(([k,v]) => (
            <button key={k} onClick={() => setSel(k)}
              className={`w-full text-left px-3.5 py-2.5 rounded-lg text-[12px] font-medium border transition-all ${k===sel?"bg-emerald-600 text-white border-emerald-600":"bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
              {v.name}
            </button>
          ))}
        </div>
        <div className="col-span-9 space-y-5">
          <Card>
            <div className="flex items-center gap-3 mb-3">
              <span className="text-[14px] font-bold text-stone-800">{m.name}</span>
              <Pill bg="#ecfdf5" color="#059669">{m.cx}</Pill>
            </div>
            <p className="text-[12px] text-stone-500 mb-3">{m.desc}</p>
            <div className="grid grid-cols-2 gap-5">
              <div>
                <div className="text-[10px] font-bold text-emerald-600 uppercase tracking-wider mb-1.5">Pros</div>
                <ul className="space-y-1.5">{m.pros.map((p,i) => <Point key={i} icon="✓" color="#059669">{p}</Point>)}</ul>
              </div>
              <div>
                <div className="text-[10px] font-bold text-red-500 uppercase tracking-wider mb-1.5">Cons</div>
                <ul className="space-y-1.5">{m.cons.map((c,i) => <Point key={i} icon="✗" color="#dc2626">{c}</Point>)}</ul>
              </div>
            </div>
          </Card>
          <CodeBlock title={`${m.name} — Implementation`} code={m.code} />
        </div>
      </div>
    </div>
  );
}

function RankingSection() {
  return (
    <div className="space-y-5">
      <Card accent="#dc2626">
        <Label color="#dc2626">Multi-Stage Ranking Models</Label>
        <p className="text-[12px] text-stone-500 mb-4">The ranking stack is the core ML component. Each layer trades off cost vs quality. At L6, you're expected to justify why each layer exists and what happens if you remove one.</p>
        <div className="grid grid-cols-3 gap-4">
          {[
            { stage: "L1: Lightweight Ranker", model: "GBDT (LambdaMART) or Small NN", input: "Pre-computed features only", docs: "10K → 500", latency: "~0.1ms / doc", cost: "CPU only",
              detail: "Features: BM25 score, PageRank, doc freshness, URL depth, domain authority, query-doc TF-IDF. No query-doc interaction features (too expensive at 10K scale).", color: "#9333ea" },
            { stage: "L2: Deep Ranker", model: "BERT Cross-Encoder", input: "Query-document pairs", docs: "500 → 50", latency: "~5ms / doc (GPU batched)", cost: "GPU required",
              detail: "BERT takes [CLS] query [SEP] document [SEP] as input. Learns deep query-document interaction. This is where the real quality gain is — cross-attention between query and document tokens.", color: "#dc2626" },
            { stage: "L3: Business Logic", model: "Rule-based + lightweight ML", input: "L2 scores + business signals", docs: "50 → 10", latency: "<5ms", cost: "CPU",
              detail: "Diversity (don't show 5 results from same domain), freshness boost for time-sensitive queries, safe search filtering, ad slot insertion, demotion of low-quality domains.", color: "#d97706" },
          ].map((s,i) => (
            <div key={i} className="rounded-lg border p-4" style={{ borderColor: s.color+"40", borderTop: `3px solid ${s.color}` }}>
              <div className="text-[11px] font-bold mb-2" style={{ color: s.color }}>{s.stage}</div>
              <div className="space-y-1.5 text-[11px]">
                <div><span className="text-stone-400">Model:</span> <span className="text-stone-600 font-mono">{s.model}</span></div>
                <div><span className="text-stone-400">Input:</span> <span className="text-stone-600">{s.input}</span></div>
                <div><span className="text-stone-400">Scale:</span> <span className="text-stone-600 font-mono">{s.docs}</span></div>
                <div><span className="text-stone-400">Latency:</span> <span className="text-stone-600 font-mono">{s.latency}</span></div>
                <div><span className="text-stone-400">Infra:</span> <span className="text-stone-600">{s.cost}</span></div>
              </div>
              <p className="text-[10px] text-stone-400 mt-2 pt-2 border-t border-stone-100">{s.detail}</p>
            </div>
          ))}
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#9333ea">
          <Label color="#9333ea">L1: LambdaMART (GBDT)</Label>
          <CodeBlock code={`# LambdaMART — Learning to Rank with GBDT
# The industry standard L1 ranker

# Training objective: pairwise — "doc A should rank above doc B"
# Lambda gradients weight pairs by position (NDCG gain of swap)
train_lambdamart(training_data):
    # training_data: (query_id, doc_features, relevance_label)
    # Labels: 0=irrelevant, 1=fair, 2=good, 3=excellent, 4=perfect

    model = LightGBM(
        objective="lambdarank",
        ndcg_eval_at=[1, 3, 5, 10],
        num_leaves=255,
        learning_rate=0.05,
        num_trees=500,
        feature_fraction=0.8,  # bagging for robustness
    )
    model.fit(features, labels, group=query_groups)
    return model

# Serving: ~0.01ms per doc (500 trees × 255 leaves = fast)
# Features (all pre-computed, stored in feature store):
#   Query features: length, language, intent
#   Doc features: PageRank, freshness, domain_authority,
#                 content_length, num_images, spam_score
#   Query-Doc: BM25, TF-IDF, query_coverage, title_match
#   ↑ No BERT features here — too expensive at 10K docs`} />
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">L2: BERT Cross-Encoder</Label>
          <CodeBlock code={`# BERT Cross-Encoder — Deep Reranking
# Input: [CLS] query [SEP] document_text [SEP]
# Output: relevance score (0-1)

class CrossEncoderRanker:
    def __init__(self):
        self.model = BertForSequenceClassification(
            "bert-base-uncased",
            num_labels=1  # regression → relevance score
        )
        # Fine-tuned on human-labeled (query, doc, relevance) triples

    def rerank(self, query: str, docs: List[Doc]) -> List[DocScore]:
        # Batch all query-doc pairs
        inputs = []
        for doc in docs:
            text = doc.title + " " + doc.snippet[:256]  # truncate
            inputs.append(f"[CLS] {query} [SEP] {text} [SEP]")

        # Batch inference on GPU
        scores = self.model.predict(inputs)  # shape: [500, 1]

        # Combine with L1 score (ensemble)
        final = []
        for doc, l2_score in zip(docs, scores):
            # Weighted combination — L2 dominates
            combined = 0.3 * doc.l1_score + 0.7 * l2_score
            final.append((doc, combined))

        return sorted(final, key=lambda x: -x[1])[:50]

# Why cross-encoder > bi-encoder for reranking:
#   Cross-attention between query and doc tokens
#   "python" in "python programming" vs "monty python" — resolved by context
# Why not use for retrieval:
#   O(N) inference — must run on every (query, doc) pair. Infeasible for 100B docs.`} />
        </Card>
      </div>
    </div>
  );
}

function FeaturesSection() {
  return (
    <div className="space-y-5">
      <Card accent="#d97706">
        <Label color="#d97706">Feature Categories for Search Ranking</Label>
        <p className="text-[12px] text-stone-500 mb-4">Features are the lifeblood of the ranking model. At Google L6, you should know which features go in which stage, why, and the cost of each.</p>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[11px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Category</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Example Features</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Computed</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Stage</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Impact</th>
            </tr></thead>
            <tbody>
              {[
                { cat: "Query Features", ex: "query_length, language, intent, is_question, has_entity", comp: "Online (QU)", stage: "L1 + L2", impact: "Medium" },
                { cat: "Document Static", ex: "PageRank, domain_authority, spam_score, content_length, num_images", comp: "Offline (index)", stage: "L1", impact: "High" },
                { cat: "Document Freshness", ex: "last_crawled, content_change_rate, publication_date", comp: "Offline + Online", stage: "L1 + L2", impact: "High (news)" },
                { cat: "Query-Doc Lexical", ex: "BM25, TF-IDF, query_terms_in_title, query_coverage", comp: "Online (retrieval)", stage: "L1", impact: "Very High" },
                { cat: "Query-Doc Semantic", ex: "BERT cross-encoder score, embedding cosine similarity", comp: "Online (GPU)", stage: "L2 only", impact: "Very High" },
                { cat: "Click Signals", ex: "historical_CTR, avg_dwell_time, bounce_rate, skip_rate", comp: "Offline (batch)", stage: "L1 + L2", impact: "Very High" },
                { cat: "User/Context", ex: "user_language, location, search_history, device_type", comp: "Online (profile)", stage: "L1 + L2", impact: "Medium" },
                { cat: "URL Features", ex: "url_depth, has_https, is_homepage, url_length", comp: "Offline (index)", stage: "L1", impact: "Low-Medium" },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-bold text-stone-700">{r.cat}</td>
                  <td className="px-3 py-2 text-stone-500 font-mono text-[10px]">{r.ex}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.comp}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.stage}</td>
                  <td className="text-center px-3 py-2"><Pill bg={r.impact.includes("Very")?"#fef2f2":r.impact==="High"?"#fffbeb":"#f0fdf4"} color={r.impact.includes("Very")?"#dc2626":r.impact==="High"?"#d97706":"#059669"}>{r.impact}</Pill></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Click Signal De-biasing (Critical for L6)</Label>
          <CodeBlock code={`# Click signals are the most powerful features
# BUT they have severe biases you MUST address:

# 1. Position Bias — users click top results more
#    regardless of relevance
debias_position(click_data):
    # Inverse Propensity Weighting (IPW)
    for click in click_data:
        position = click.rank_position
        # P(examine | position) — estimated from randomized experiments
        propensity = examination_prob[position]
        click.weight = 1.0 / propensity
    # Position 1: propensity=0.9, weight=1.1
    # Position 5: propensity=0.3, weight=3.3
    # → Clicks at lower positions count MORE

# 2. Presentation Bias — attractive titles get more clicks
#    even if content is poor (clickbait)
debias_presentation(click_data):
    # Use dwell time, not just click
    for click in click_data:
        if click.dwell_time < 10_seconds:
            click.label = "skip"  # short click = bad
        elif click.dwell_time > 120_seconds:
            click.label = "satisfied"  # long dwell = good
        # Also: pogo-sticking = click, back, click next result = BAD

# 3. Selection Bias — can only observe clicks on shown results
#    Solution: use impression data, not just clicks
#    CTR = clicks / impressions (per doc per query)`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Feature Store Architecture</Label>
          <CodeBlock code={`# Feature serving for search ranking
# Two paths: batch (offline) and real-time (online)

# OFFLINE features (pre-computed, stored in Bigtable/SSTable)
# Updated: daily batch job
offline_features = {
    "doc_features": {
        # Key: doc_id → feature vector
        "PageRank": 6.2,
        "domain_authority": 0.87,
        "content_length": 2450,
        "spam_score": 0.02,
        "historical_ctr": 0.034,  # position-debiased
        "avg_dwell_time": 45.2,
    },
}

# ONLINE features (computed per query at serving time)
# Must be fast: <5ms total
online_features = {
    "query_doc_features": {
        "bm25_score": 12.4,       # from retrieval
        "query_coverage": 0.8,    # % of query terms in doc
        "title_match": True,      # exact title match
    },
    "context_features": {
        "user_language": "en",
        "user_location": "NYC",
        "device": "mobile",
        "time_of_day": "morning",
    },
}

# Feature consistency: CRITICAL
# Training features must match serving features exactly
# Log serving features alongside clicks for training data
# → Prevents training-serving skew (top 3 ML bug)`} />
        </Card>
      </div>
    </div>
  );
}

function DataModelSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#0f766e">
          <Label color="#0f766e">Core Data Stores</Label>
          <CodeBlock code={`-- Inverted Index (sharded by term hash)
-- The backbone of lexical retrieval
term → PostingList:
  [(doc_id, term_freq, [positions]), ...]
  # Compressed with variable-byte encoding
  # Posting lists sorted by doc_id for merge joins

-- Forward Index (sharded by doc_id)
-- Stores pre-computed doc features
doc_id → {
  url, title, content_hash,
  pagerank: float,
  freshness_score: float,
  domain_authority: float,
  spam_score: float,
  language: str,
  content_length: int,
  crawl_timestamp: timestamp,
  feature_vector: bytes  # pre-computed L1 features
}

-- Vector Index (HNSW, sharded by partition)
-- Dense embeddings for ANN retrieval
doc_id → embedding[768]  # quantized to int8 or PQ codes

-- Click Log (append-only, time-partitioned)
-- Primary training signal
(timestamp, query, user_id, results_shown[],
 clicks[], dwell_times[], query_reformulations[])

-- Document Store (sharded by doc_id)
-- Full content for snippet generation
doc_id → {raw_html, extracted_text, metadata}`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Why These Design Choices?</Label>
          <div className="space-y-3">
            {[
              { q: "Why shard inverted index by term, not doc?", a: "A single query term like 'the' matches billions of docs. The posting list must be co-located on one shard for efficient scanning. Trade-off: rare terms create very uneven shards. Solution: terms sharded across a small number of 'index servers' with replicas." },
              { q: "Why keep forward index separate?", a: "Inverted index is read during retrieval (find candidates). Forward index is read during ranking (look up features). Different access patterns — inverted is sequential scan, forward is random lookup by doc_id. Different storage engines." },
              { q: "Why not just store embeddings in the inverted index?", a: "Fundamentally different data structure. Inverted index maps term → docs. Vector index maps region-of-space → docs. HNSW/IVF has its own graph structure that doesn't fit posting lists. Served on different hardware (potentially GPU for ANN)." },
              { q: "Why time-partition click logs?", a: "Click logs are massive (25B/day) and used for training. You only want recent data (90 days). Time partitioning enables cheap deletion of old data. Also enables time-based train/validation splits (train on weeks 1-12, validate on week 13)." },
              { q: "Feature store: why not just compute features on the fly?", a: "PageRank requires full web graph computation (days). Domain authority needs click aggregation across all queries. These are batch-computed and stored. Only query-doc interaction features are computed online. Feature store enables consistency between training and serving." },
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

function TrainingSection() {
  return (
    <div className="space-y-5">
      <Card accent="#7e22ce">
        <Label color="#7e22ce">Training Pipeline — End to End</Label>
        <svg viewBox="0 0 700 160" className="w-full">
          <defs><marker id="ah-tp" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          {[
            { x: 55, y: 50, w: 90, label: "Click Logs\n+ Rater Labels", color: "#d97706" },
            { x: 175, y: 50, w: 80, label: "Feature\nJoining", color: "#0891b2" },
            { x: 285, y: 50, w: 80, label: "De-biasing\n+ Sampling", color: "#dc2626" },
            { x: 395, y: 50, w: 80, label: "Model\nTraining", color: "#7e22ce" },
            { x: 505, y: 50, w: 80, label: "Offline\nEval", color: "#059669" },
            { x: 615, y: 50, w: 80, label: "Online\nA/B Test", color: "#2563eb" },
          ].map((s,i) => (
            <g key={i}>
              <rect x={s.x-s.w/2} y={s.y-22} width={s.w} height={44} rx={6} fill={s.color+"10"} stroke={s.color} strokeWidth={1.5}/>
              {s.label.split("\n").map((l,j) => (
                <text key={j} x={s.x} y={s.y-5+j*14} textAnchor="middle" fill={s.color} fontSize="9" fontWeight="600" fontFamily="monospace">{l}</text>
              ))}
              {i < 5 && <line x1={s.x+s.w/2} y1={s.y} x2={s.x+s.w/2+30} y2={s.y} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-tp)"/>}
            </g>
          ))}
          {/* Feedback loop */}
          <path d="M 615 72 L 615 110 L 55 110 L 55 72" fill="none" stroke="#dc2626" strokeWidth={1} strokeDasharray="4,3" markerEnd="url(#ah-tp)"/>
          <text x={335} y={125} textAnchor="middle" fill="#dc2626" fontSize="8" fontFamily="monospace">Feedback loop: A/B test results → retrain with new click data</text>

          {/* Timing */}
          <text x={55} y={150} fill="#78716c" fontSize="7" fontFamily="monospace">Daily: L1 retrain</text>
          <text x={285} y={150} fill="#78716c" fontSize="7" fontFamily="monospace">Weekly: L2 retrain</text>
          <text x={505} y={150} fill="#78716c" fontSize="7" fontFamily="monospace">Continuous: eval metrics</text>
        </svg>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7e22ce">
          <Label color="#7e22ce">Training Data Construction</Label>
          <CodeBlock code={`# Training Data for Learning-to-Rank
# Source: click logs with de-biased labels

build_training_data(click_logs, rater_labels):
    examples = []
    for query_session in click_logs:
        query = query_session.query
        shown_results = query_session.results

        for result in shown_results:
            # Compute relevance label from clicks
            label = compute_label(result)
            # 0: not clicked (but was it even examined?)
            # 1: clicked, short dwell (<10s) — bad click
            # 2: clicked, medium dwell (10-60s) — decent
            # 3: clicked, long dwell (>60s) — good
            # 4: last click in session — satisfied

            # Join features (logged at serving time!)
            features = feature_store.get(query, result.doc_id)

            examples.append({
                "query_id": query_session.id,
                "doc_id": result.doc_id,
                "features": features,
                "label": label,
                "position": result.position,  # for debiasing
            })

    # Supplement with human rater labels (gold standard)
    for (query, doc, rater_label) in rater_labels:
        features = feature_store.get(query, doc)
        examples.append({
            "query_id": query,
            "doc_id": doc,
            "features": features,
            "label": rater_label,  # 0-4 scale
            "position": None,      # no position bias
            "weight": 10.0,        # upweight rater labels
        })

    return examples
# Train/val split: by time (last week = validation)
# NOT random — prevents data leakage from same sessions`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Evaluation Metrics</Label>
          <div className="space-y-3">
            {[
              { name: "NDCG@K", desc: "Normalized Discounted Cumulative Gain. The primary offline metric. Measures how well the ranking matches ideal ordering. Values 0-1, higher is better.", formula: "DCG@K = Σ (2^rel_i - 1) / log2(i+1)", useCase: "Overall ranking quality" },
              { name: "MRR", desc: "Mean Reciprocal Rank. Position of first relevant result. Good for navigational queries where there's one right answer.", formula: "MRR = 1/|Q| × Σ 1/rank_i", useCase: "Navigational query quality" },
              { name: "pFound@K", desc: "Probability of user finding relevant result in top K. Models user browsing behavior (cascading clicks model).", formula: "pFound = Σ P(examine_i) × P(relevant_i)", useCase: "User satisfaction proxy" },
              { name: "Abandonment Rate", desc: "% of queries where user reformulates or leaves without clicking. Measures query satisfaction. Online metric.", formula: "abandon_rate = abandoned_sessions / total_sessions", useCase: "Online A/B test guardrail" },
              { name: "Session Success", desc: "Did the user accomplish their goal? Measured by: no reformulation, long dwell on final click, task completion signals.", formula: "success = long_dwell AND no_reformulation", useCase: "Ultimate satisfaction metric" },
            ].map((m,i) => (
              <div key={i} className="rounded-lg border border-stone-200 p-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[11px] font-bold text-stone-800">{m.name}</span>
                  <Pill bg="#ecfdf5" color="#059669">{m.useCase}</Pill>
                </div>
                <p className="text-[11px] text-stone-500">{m.desc}</p>
                <div className="text-[10px] font-mono text-stone-400 mt-1 bg-stone-50 px-2 py-1 rounded">{m.formula}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function ScalabilitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Index Sharding Strategy</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Document-partitioned index ★</strong> — shard by doc_id range. Each shard holds a subset of all documents with their complete inverted index. Query broadcast to all shards, results merged. Google's approach.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Term-partitioned index</strong> — shard by term. Each shard holds posting lists for a subset of terms. Less broadcast but requires cross-shard joins for multi-term queries. Rarely used in practice.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Tiered indexing ★</strong> — hot tier (recent/popular docs, fast SSD) + cold tier (long-tail docs, HDD/blob). Query hits hot tier first; only goes to cold tier if not enough results.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Replica scaling</strong> — each index shard has N replicas. Load balance queries across replicas. Scale replicas independently for read throughput (100K+ QPS).</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Model Serving Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">L1 model on CPU</strong> — GBDT inference is trivially parallelizable. Score 500 docs in ~20ms on a single core. Scale horizontally with more machines.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">L2 model on GPU with batching</strong> — BERT cross-encoder: batch 50 (query, doc) pairs into one GPU forward pass. Amortizes GPU kernel launch overhead. Dynamic batching: accumulate requests for 5ms, then fire.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Model distillation</strong> — train a smaller model (6-layer BERT) to mimic the full model (12-layer). 2× faster at 95% of the quality. Critical for serving budget.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Quantization</strong> — FP32 → INT8 for 2-4× speedup. BERT-base: 110M params × 4B = 440 MB (FP32) → 110 MB (INT8). Negligible quality loss.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Architecture</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Index Replication ★", d:"Full index replicated to each region. Queries served locally. Index updates propagate from central pipeline with ~minutes delay.", pros:["Lowest query latency","Region-autonomous for reads","Standard approach (Google)"], cons:["Massive storage per region (500TB)","Index lag = slight freshness loss","Expensive to replicate globally"], pick:true },
            { t:"Query Routing", d:"Route query to nearest region with relevant index. Some regions specialize (e.g., Japanese content in Tokyo).", pros:["Reduces per-region storage","Natural language/content locality","Flexible capacity allocation"], cons:["Cross-region latency for misrouted queries","Complex routing logic","Uneven load distribution"], pick:false },
            { t:"Federated Search", d:"Split index by content region. Query fans out to relevant regions, results merged centrally.", pros:["No full replication needed","Natural for multi-language","Each region owns its index"], cons:["Cross-region fan-out adds latency","Complex merge logic","Single-region failure affects global results"], pick:false },
          ].map((o,i) => (
            <div key={i} className={`rounded-lg border p-4 ${o.pick ? "border-purple-300 bg-purple-50/50" : "border-stone-200 bg-stone-50/30"}`}>
              <div className={`text-[11px] font-bold mb-1.5 ${o.pick ? "text-purple-700" : "text-stone-600"}`}>{o.t}</div>
              <p className="text-[11px] text-stone-500 mb-2">{o.d}</p>
              <div className="space-y-0.5 text-[11px]">
                {o.pros.map((p,j) => <div key={j} className="text-emerald-600">✓ {p}</div>)}
                {o.cons.map((c,j) => <div key={j} className="text-red-500">✗ {c}</div>)}
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function AvailabilitySection() {
  return (
    <div className="space-y-5">
      <Card className="bg-amber-50/50 border-amber-200" accent="#d97706">
        <Label color="#d97706">Search Must Never Go Down</Label>
        <p className="text-[12px] text-stone-500 mb-4">Google Search downtime costs ~$1M/minute in ad revenue. The system must degrade gracefully — always return something, even if quality is reduced.</p>
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Tier 1: Full Pipeline</div>
            <p className="text-[11px] text-stone-500">All stages operational. BM25 + ANN retrieval, L1 + L2 ranking, personalization. Best quality.</p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-white p-4">
            <div className="text-[11px] font-bold text-amber-700 mb-1.5">Tier 2: Skip L2 Reranker</div>
            <p className="text-[11px] text-stone-500">GPU fleet overloaded? Skip BERT reranking, serve L1 results directly. 10-15% quality drop but 200ms faster.</p>
          </div>
          <div className="rounded-lg border border-red-200 bg-white p-4">
            <div className="text-[11px] font-bold text-red-600 mb-1.5">Tier 3: BM25 Only</div>
            <p className="text-[11px] text-stone-500">Everything failing? Fall back to pure BM25 retrieval with static PageRank ranking. No ML. Still gives reasonable results.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#2563eb">Index Availability</Label>
          <ul className="space-y-2.5">
            <Point icon="→" color="#2563eb">Each index shard has 3+ replicas. Any 2 can serve queries while 1 is updated.</Point>
            <Point icon="→" color="#2563eb">Index updates are atomic per shard — new index is built offline, then swapped in (blue-green deployment).</Point>
            <Point icon="→" color="#2563eb">If a shard is down, queries skip it — results may be incomplete but never empty.</Point>
            <Point icon="→" color="#2563eb">Stale index fallback: if fresh index unavailable, serve from last-known-good index (hours old).</Point>
          </ul>
        </Card>
        <Card>
          <Label color="#d97706">Model Rollback Strategy</Label>
          <ul className="space-y-2.5">
            <Point icon="→" color="#d97706">Every model version is stored and can be rolled back in minutes.</Point>
            <Point icon="→" color="#d97706">Canary deployment: new model serves 1% of traffic, monitor NDCG/CTR for 2 hours before expanding.</Point>
            <Point icon="→" color="#d97706">Automatic rollback trigger: if NDCG drops &gt;1% or latency p99 increases &gt;50ms, auto-revert.</Point>
            <Point icon="→" color="#d97706">Shadow mode: new model scores queries in parallel with production, log results for offline comparison without affecting users.</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#9333ea">Degradation Ladder</Label>
        <div className="flex gap-3 items-stretch mt-1">
          {[
            { label: "Full Pipeline", sub: "BM25+ANN+L1+L2", color: "#059669", status: "HEALTHY" },
            { label: "Skip L2", sub: "BM25+ANN+L1 only", color: "#d97706", status: "DEGRADED" },
            { label: "Skip ANN", sub: "BM25+L1 only", color: "#ea580c", status: "FALLBACK" },
            { label: "BM25 + PageRank", sub: "No ML ranking", color: "#dc2626", status: "EMERGENCY" },
          ].map((t,i) => (
            <div key={i} className="flex-1 flex items-center gap-2">
              <div className="flex-1 text-center py-3 px-2 rounded-lg border" style={{ borderColor: t.color+"40", background: t.color+"08" }}>
                <div className="text-[10px] font-mono font-bold" style={{ color: t.color }}>{t.status}</div>
                <div className="text-[11px] text-stone-600 mt-0.5">{t.label}</div>
                <div className="text-[9px] text-stone-400">{t.sub}</div>
              </div>
              {i<3 && <span className="text-stone-300 text-lg shrink-0">→</span>}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function ObservabilitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-5">
        <Card accent="#0284c7">
          <Label color="#0284c7">Key Metrics</Label>
          <div className="space-y-2.5">
            {[
              { metric: "Query Latency (p50/p99)", target: "300ms / 800ms", why: "User experience — every 100ms delay = 1% less engagement" },
              { metric: "NDCG@10", target: ">0.65", why: "Ranking quality — primary offline metric for model quality" },
              { metric: "Session Abandonment", target: "<15%", why: "User satisfaction — are users finding what they need?" },
              { metric: "Zero-Result Rate", target: "<2%", why: "Coverage — how often do we fail to find anything?" },
              { metric: "Index Freshness", target: "<10 min for news", why: "Breaking news must appear fast" },
            ].map((m,i) => (
              <div key={i} className="border-b border-stone-100 pb-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold text-stone-700">{m.metric}</span>
                  <span className="text-[10px] font-mono text-cyan-700">{m.target}</span>
                </div>
                <div className="text-[10px] text-stone-400 mt-0.5">{m.why}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Alerts & SLOs</Label>
          <div className="space-y-2.5">
            {[
              { alert: "Latency p99 > 1s for 5 min", sev: "P0", action: "Shed L2 reranking load" },
              { alert: "NDCG drop > 2%", sev: "P0", action: "Auto-rollback model, page oncall" },
              { alert: "Zero-result rate > 5%", sev: "P1", action: "Check index health, query understanding" },
              { alert: "Index lag > 30 min", sev: "P1", action: "Check indexing pipeline, freshness" },
              { alert: "GPU utilization > 90%", sev: "P2", action: "Scale GPU fleet or reduce L2 batch size" },
            ].map((a,i) => (
              <div key={i} className="border-b border-stone-100 pb-2">
                <div className="flex items-center gap-2">
                  <Pill bg={a.sev==="P0"?"#fef2f2":a.sev==="P1"?"#fffbeb":"#f0fdf4"} color={a.sev==="P0"?"#dc2626":a.sev==="P1"?"#d97706":"#059669"}>{a.sev}</Pill>
                  <span className="text-[11px] text-stone-700">{a.alert}</span>
                </div>
                <div className="text-[10px] text-stone-400 mt-0.5 ml-9">→ {a.action}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Search Quality Monitoring</Label>
          <div className="space-y-2.5">
            {[
              { what: "Side-by-Side Evals", how: "Show raters results from model A vs B. Track win rate. Run continuously for every model change." },
              { what: "Live NDCG Tracking", how: "Compare model predictions against next-day click data. Real-time dashboard showing quality trend." },
              { what: "Query Cohort Analysis", how: "Track quality separately for: head queries (top 1K), torso, tail. Tail degrades first — early warning." },
              { what: "Regression Detection", how: "Automated queries with known-good results. If known results drop in ranking, alert immediately." },
              { what: "Bias Monitoring", how: "Track fairness across demographic groups, languages, regions. Ensure no systematic quality disparity." },
            ].map((m,i) => (
              <div key={i} className="border-b border-stone-100 pb-2">
                <div className="text-[11px] font-bold text-stone-700">{m.what}</div>
                <div className="text-[10px] text-stone-400 mt-0.5">{m.how}</div>
              </div>
            ))}
          </div>
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
          { title: "Training-Serving Skew", sev: "CRITICAL", desc: "Features computed differently in training vs serving. Example: BM25 score computed on stale index during training but fresh index at serving time.", fix: "Log features AT SERVING TIME and use those exact features for training. Never re-compute features for training data.", icon: "🔴" },
          { title: "Position Bias in Click Data", sev: "CRITICAL", desc: "Position 1 gets 30% CTR, position 10 gets 2% — regardless of relevance. Training on raw clicks teaches the model to rank by position, not relevance.", fix: "Inverse propensity weighting. Occasional randomized experiments to measure true examination probability per position.", icon: "🔴" },
          { title: "Index Staleness", sev: "HIGH", desc: "Breaking news query returns results from hours ago. Index update pipeline is delayed or broken.", fix: "Real-time indexing pipeline for breaking content. Freshness signals in ranking model. Monitoring for index lag per content type.", icon: "🟡" },
          { title: "Adversarial SEO / Spam", sev: "HIGH", desc: "Content farms and SEO spam gaming your signals. Link farms inflating PageRank. Keyword stuffing exploiting BM25.", fix: "Spam classifier as ranking feature. Manual domain-level penalties. Diverse signal portfolio — harder to game all signals simultaneously. Periodic spam audits.", icon: "🟡" },
          { title: "Cold Start for New Docs", sev: "MEDIUM", desc: "New documents have no click history, no PageRank. They can't rank well, so they never get clicks — feedback loop.", fix: "Freshness boost for new documents. Content quality features (not dependent on clicks). Exploration budget: show some new docs in lower positions to collect feedback.", icon: "🟠" },
          { title: "Query Drift", sev: "MEDIUM", desc: "Query patterns shift (seasonal, events, trends). 'virus' meant differently in 2019 vs 2020. Model trained on old data underperforms.", fix: "Retrain L1 daily, L2 weekly. Monitor per-query-cohort NDCG. Use recency-weighted training data. Trigger retrain on significant drift detection.", icon: "🟠" },
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
          { title: "Knowledge Graph Integration", d: "Link entities in queries to a knowledge graph. Power knowledge panels, direct answers, entity disambiguation. 'Apple CEO' → knowledge panel with Tim Cook.", effort: "Hard", detail: "Requires entity linking model, knowledge graph construction/maintenance, and a serving layer for structured data alongside organic results." },
          { title: "LLM-Powered Snippets (AI Overview)", d: "Use an LLM to generate a synthesized answer from top results. Reduces need to click through. Google's AI Overviews.", effort: "Hard", detail: "Challenge: hallucination, attribution, latency (LLM inference adds 500ms+). Need citation/grounding to search results. Cost: $0.01+ per query." },
          { title: "Personalization", d: "Use search history, clicked domains, location, and interests to re-rank results. Navigational queries benefit most (your 'Amazon' vs theirs).", effort: "Medium", detail: "Privacy concerns. Personalization can create filter bubbles. Need clear user controls. Feature: past_domain_clicks, topic_affinity scores." },
          { title: "Multimodal Search", d: "Accept image, voice, and video queries. Google Lens, voice search. Unified embedding space across modalities.", effort: "Hard", detail: "Requires multi-modal encoders (CLIP-style), ASR for voice, and a unified retrieval system that can match across modalities." },
          { title: "Result Diversity", d: "Ensure results cover different intents and sources. 'Java' → programming + island + coffee. MMR (Maximal Marginal Relevance) reranking.", effort: "Medium", detail: "Detect ambiguous queries. Cluster results by sub-intent. Pick representatives from each cluster. Trade off relevance for coverage." },
          { title: "Federated / Vertical Search", d: "Route queries to specialized search engines (Images, News, Shopping, Maps) and blend results into a unified SERP.", effort: "Hard", detail: "Need intent classifier to decide which verticals to trigger. Blending: interleave vertical results at the right positions in organic results." },
        ].map((e,i) => (
          <Card key={i}>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="text-[12px] font-bold text-stone-800">{e.title}</span>
              <Pill bg={e.effort==="Easy"?"#ecfdf5":e.effort==="Medium"?"#fffbeb":"#fef2f2"} color={e.effort==="Easy"?"#059669":e.effort==="Medium"?"#d97706":"#dc2626"}>{e.effort}</Pill>
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
    { q:"Why not just use a single BERT model to rank everything?", a:"BERT cross-encoder is O(N) — it must process every (query, doc) pair. At 100B docs, that's 100B BERT inferences per query, which would take hours. Even at 10K candidates, that's ~50 seconds on GPU. The funnel approach is essential: cheap models filter first (BM25: ~50ms for 100B docs), then expensive models refine (BERT: ~100ms for 500 docs). The funnel reduces BERT's work by 200,000× while losing <5% quality.", tags:["architecture"] },
    { q:"How would you handle a brand new query that's never been seen?", a:"Three strategies: (1) Semantic retrieval (ANN) works on any query — it embeds the query and finds similar documents even if exact terms were never searched before. (2) Query expansion via embeddings — find similar known queries in embedding space and borrow their results. (3) BM25 still works for keyword matching. The hardest case is truly novel entities (new product launch). Solution: real-time indexing ensures the product page is indexed within minutes of publication.", tags:["retrieval"] },
    { q:"How do you prevent the model from learning to just rank by popularity?", a:"Popularity bias is real — models learn to always rank Wikipedia and Reddit highly. Mitigations: (1) Diversity features that penalize same-domain concentration. (2) Freshness features that boost new content. (3) Query-specific relevance signals (BM25, title match) prevent generic popular pages from dominating. (4) Exploration: occasionally show less-popular results to collect unbiased feedback. (5) Rater-labeled data weighted higher — raters evaluate relevance, not popularity.", tags:["ml"] },
    { q:"BM25 vs TF-IDF — what's the difference and which is better?", a:"Both use term frequency and inverse document frequency. BM25 improves on TF-IDF with: (1) Saturating TF — BM25 uses tf/(tf+k1) so a term appearing 100× isn't scored 10× higher than 10×. Diminishing returns. (2) Document length normalization — BM25 penalizes long documents (they naturally have higher TF). (3) Tunable parameters k1 and b. BM25 consistently outperforms raw TF-IDF in IR benchmarks. Use BM25.", tags:["retrieval"] },
    { q:"How would you measure if the L2 reranker is actually worth the GPU cost?", a:"Run an A/B test: control = L1 only, treatment = L1 + L2. Measure: (1) NDCG@10 improvement — typically 5-15% with BERT reranker. (2) Session abandonment rate — should decrease. (3) Revenue impact (ads) — better relevance → more satisfied users → more queries → more ad revenue. (4) Cost: ~2K GPUs for L2 at Google scale ≈ $5M/month. If revenue lift from NDCG improvement exceeds cost, it's worth it. At Google's $150B/year ad revenue, even 0.1% lift = $150M/year.", tags:["evaluation"] },
    { q:"How would you handle multi-language search?", a:"Three approaches: (1) Separate indexes per language — simple but can't do cross-language retrieval. (2) Multilingual embeddings (mBERT, XLM-R) — single model handles 100+ languages, queries in one language can retrieve docs in another. (3) Translation-based — translate query to target language(s), search each. Google uses approach 2+3: multilingual dense retrieval + selective translation for high-value language pairs.", tags:["scalability"] },
    { q:"What's the biggest difference between web search and product search ranking?", a:"Three key differences: (1) Objective — web search optimizes for information satisfaction (dwell time, no reformulation). Product search optimizes for purchase (conversion rate, revenue). (2) Structured data — products have attributes (price, rating, category) enabling faceted search and hard filters. Web docs are unstructured. (3) Inventory constraints — product search must consider availability, shipping time, seller reputation. A 'relevant' product that's out of stock shouldn't rank first. Also: product search has explicit user actions (add to cart, purchase) providing cleaner training signals than clicks.", tags:["design"] },
    { q:"How do you handle adversarial attacks / SEO spam at scale?", a:"Multi-layered defense: (1) Content quality classifier trained on known-spam features (keyword density, hidden text, link farm association). (2) Link graph analysis — detect unnatural link patterns (link farms, PBNs) using graph algorithms. (3) User behavior signals — if many users pogo-stick (click then immediately return), the result is likely low quality. (4) Manual actions for worst offenders — domain-level penalties. (5) Diverse feature portfolio — BM25, PageRank, click signals, content quality, freshness. Gaming one signal doesn't game them all. (6) Regular adversarial audits — dedicated team that tests ranking with known-spam queries.", tags:["safety"] },
    { q:"How do you do A/B testing for search ranking?", a:"Interleaving is superior to simple A/B splits for ranking experiments. In interleaving, each user sees results from both models mixed together — model A's #1, model B's #1, model A's #2, etc. Users' clicks on each model's results determine the winner. Advantages: (1) Much more sensitive — needs 10× fewer queries than traditional A/B. (2) Controls for query mix and time-of-day effects. (3) Each query acts as its own control. Standard at Google, Bing, and Yandex. For larger changes, follow up with full A/B test measuring downstream metrics (session success, revenue).", tags:["evaluation"] },
    { q:"What would you change for a real-time search use case (Twitter/news)?", a:"Key changes: (1) Index freshness becomes primary — need seconds-to-index, not hours. Use an in-memory real-time index that's merged with the main index periodically. (2) Recency becomes the dominant ranking signal — a 1-hour-old tweet about breaking news should rank above a 1-day-old article. (3) Velocity signals — content engagement rate (likes/retweets per minute) becomes a ranking feature. (4) Reduced reliance on historical click signals — new content has no history. (5) Index pruning is more aggressive — old content drops out faster.", tags:["design"] },
  ];
  return (
    <div className="space-y-2.5">
      <Card className="bg-indigo-50/50 border-indigo-200 mb-1">
        <p className="text-[12px] text-stone-500">Common follow-up questions Google L6 interviewers ask. Click to reveal a strong answer.</p>
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
  api: ApiSection, design: DesignSection, query: QuerySection, retrieval: RetrievalSection,
  ranking: RankingSection, features: FeaturesSection, data: DataModelSection,
  training: TrainingSection, scalability: ScalabilitySection, availability: AvailabilitySection,
  observability: ObservabilitySection, watchouts: WatchoutsSection,
  enhancements: EnhancementsSection, followups: FollowupsSection,
};

export default function SearchRankingSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Search Ranking</h1>
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