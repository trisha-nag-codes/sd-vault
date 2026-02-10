import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   SEARCH AUTOCOMPLETE — System Design Reference
   Pearl white theme · Reusable section structure
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",             icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",         icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",  icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",           icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",    icon: "🏗️", color: "#9333ea" },
  { id: "algorithm",     label: "Algorithm Deep Dive",  icon: "⚙️", color: "#c026d3" },
  { id: "data",          label: "Data Model",           icon: "🗄️", color: "#dc2626" },
  { id: "scalability",   label: "Scalability",          icon: "📈", color: "#059669" },
  { id: "availability",  label: "Availability",         icon: "🛡️", color: "#d97706" },
  { id: "observability", label: "Observability",        icon: "📊", color: "#0284c7" },
  { id: "watchouts",     label: "Failure Modes",        icon: "⚠️", color: "#dc2626" },
  { id: "services",      label: "Service Architecture", icon: "🧩", color: "#0f766e" },
  { id: "flows",         label: "Request Flows",        icon: "🔀", color: "#7e22ce" },
  { id: "deployment",    label: "Deploy & Security",    icon: "🔒", color: "#b45309" },
  { id: "ops",           label: "Ops Playbook",         icon: "🔧", color: "#be123c" },
  { id: "enhancements",  label: "Enhancements",         icon: "🚀", color: "#7c3aed" },
  { id: "followups",     label: "Follow-up Questions",  icon: "❓", color: "#6366f1" },
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

function CodeBlock({ title, code, highlight = [] }) {
  const lines = code.split("\n");
  return (
    <div className="bg-stone-50 border border-stone-200 rounded-lg p-3.5 overflow-x-auto">
      {title && <div className="text-[10px] font-bold text-stone-400 uppercase tracking-[0.1em] mb-2">{title}</div>}
      <pre className="font-mono text-[11.5px] leading-[1.75]" style={{ whiteSpace: "pre" }}>
        {lines.map((line, i) => (
          <div key={i} className={`px-2 rounded ${highlight.includes(i) ? "bg-indigo-50 text-indigo-700" : line.trim().startsWith("#") || line.trim().startsWith("--") ? "text-stone-400" : "text-stone-700"}`}>
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

/* ═══════════════════════════════════════════════════════════
   SECTIONS
   ═══════════════════════════════════════════════════════════ */

function ConceptSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-7 space-y-5">
          <Card accent="#6366f1">
            <Label>What is Search Autocomplete?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              Search autocomplete (typeahead) is a feature that predicts and displays search suggestions as the user types each character. It reduces keystrokes, guides users to popular queries, and dramatically improves search UX. When you type "how to" into Google and see "how to tie a tie" — that's autocomplete.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Think of it as real-time prediction: for every keystroke, the system must query a massive dataset of popular search terms and return the top-k matches within 100ms — all while the user is still typing. The key challenges are ultra-low latency (p99 &lt; 100ms), handling billions of queries, and keeping suggestions fresh and relevant.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard?</Label>
            <ul className="space-y-2.5">
              <Point icon="⚡" color="#0891b2">Ultra-low latency — must return results before the user types the next character (~100-200ms window)</Point>
              <Point icon="📊" color="#0891b2">Massive dataset — Google handles 8.5B searches/day; the suggestion corpus is hundreds of millions of unique queries</Point>
              <Point icon="🔄" color="#0891b2">Real-time freshness — trending topics (elections, breaking news) must appear in suggestions within minutes</Point>
              <Point icon="🌍" color="#0891b2">Personalization — suggestions vary by user, language, location, and search history</Point>
              <Point icon="🛡️" color="#0891b2">Content safety — must filter offensive, dangerous, or legally problematic suggestions in real-time</Point>
              <Point icon="📱" color="#0891b2">Client-side efficiency — minimize network calls with debouncing, caching, and prefix grouping</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Examples</Label>
            <div className="space-y-2.5">
              {[
                { co: "Google", rule: "8.5B searches/day", algo: "Trie + ML ranking" },
                { co: "YouTube", rule: "Video-specific suggestions", algo: "Personalized + trending" },
                { co: "Amazon", rule: "Product search completion", algo: "Category-aware trie" },
                { co: "Spotify", rule: "Songs, artists, playlists", algo: "Multi-entity search" },
                { co: "Twitter/X", rule: "Users + hashtags + topics", algo: "Recency-weighted" },
              ].map((e,i) => (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-20 shrink-0">{e.co}</span>
                  <span className="text-stone-500 flex-1">{e.rule}</span>
                  <span className="text-stone-400 text-[10px]">{e.algo}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <Label color="#2563eb">How It Works — High Level</Label>
            <svg viewBox="0 0 360 200" className="w-full">
              <DiagramBox x={60} y={40} w={90} h={38} label="User Types" color="#2563eb" sub={'"how t..."'}/>
              <DiagramBox x={200} y={40} w={80} h={38} label="API\nGateway" color="#6366f1"/>
              <DiagramBox x={340} y={40} w={80} h={42} label="Suggestion\nService" color="#9333ea"/>
              <DiagramBox x={340} y={120} w={80} h={42} label="Trie\nCluster" color="#dc2626"/>
              <DiagramBox x={200} y={160} w={90} h={38} label="Top-K\nResults" color="#059669"/>
              <Arrow x1={105} y1={40} x2={160} y2={40} id="ac1"/>
              <Arrow x1={240} y1={40} x2={300} y2={40} id="ac2"/>
              <Arrow x1={340} y1={61} x2={340} y2={99} label="lookup" id="ac3" dashed/>
              <Arrow x1={300} y1={120} x2={245} y2={148} label="rank" id="ac4"/>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Google, Meta, Amazon, Microsoft, Uber, LinkedIn, Twitter</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Separate Read Path from Write Path</div>
            <p className="text-[12px] text-stone-500 mt-0.5">Autocomplete has two distinct flows: <strong>the read path</strong> (returning suggestions in &lt;100ms) and <strong>the write path</strong> (aggregating search queries to update suggestion data). The read path is latency-critical; the write path can be asynchronous. This separation is the key architectural insight.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Given a prefix (e.g., "how t"), return top-k (5-10) most popular search suggestions</Point>
            <Point icon="2." color="#059669">Suggestions update as the user types each character (real-time prefix matching)</Point>
            <Point icon="3." color="#059669">Rank suggestions by popularity (search frequency), with optional personalization</Point>
            <Point icon="4." color="#059669">Support trending queries — new popular terms should surface within minutes</Point>
            <Point icon="5." color="#059669">Filter inappropriate or harmful suggestions before returning results</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Ultra-low latency — p99 &lt; 100ms end-to-end (user perceives it as instant)</Point>
            <Point icon="2." color="#dc2626">High availability — 99.99% uptime; degraded suggestions are better than no suggestions</Point>
            <Point icon="3." color="#dc2626">Scale — handle 100K+ suggestion requests per second at peak</Point>
            <Point icon="4." color="#dc2626">Eventually consistent — new trending queries can take a few minutes to appear; not real-time</Point>
            <Point icon="5." color="#dc2626">Multi-language — support Unicode, CJK characters, and locale-specific ranking</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask the Interviewer</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Is this for a search engine, e-commerce site, or general app?",
            "How many unique search terms are in the corpus? Millions? Billions?",
            "Do we need personalized suggestions per user?",
            "Should we support multi-language / CJK input?",
            "How fast must trending topics appear in suggestions?",
            "Should spelling correction be part of autocomplete?",
            "Match only from the beginning of the query, or anywhere?",
            "How many suggestions should we return? 5? 10?",
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
            <div className="text-[12px] font-bold text-violet-700">Interview Tip — Show Your Math</div>
            <p className="text-[12px] text-stone-500 mt-0.5">Walk through estimation out loud. State assumptions: <em>"Let me assume a major search engine with 10B searches/day..."</em> Then derive QPS, storage, and memory requirements step by step.</p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Query Volume Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Daily searches = 10 billion" result="10B" note='Assumption — Google-scale search engine' />
            <MathStep step="2" formula="Avg characters per query = 20" result="20 chars" note="Short queries like 'weather' (7) + long tails like 'best restaurants near me' (25)" />
            <MathStep step="3" formula="Autocomplete requests per search = 20" result="20" note="One API call per keystroke (before debouncing)" />
            <MathStep step="4" formula="With debouncing (~50% reduction)" result="10 req/search" note="Client waits 100-150ms between calls; reduces by half" />
            <MathStep step="5" formula="Daily autocomplete requests = 10B × 10" result="100B/day" note="100 billion suggestion requests per day" />
            <MathStep step="6" formula="QPS = 100B / 86,400" result="~1.16M QPS" note="Average QPS" final />
            <MathStep step="7" formula="Peak QPS (3× average)" result="~3.5M QPS" note="Peak during high-traffic hours" final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Trie Storage Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Unique search queries in corpus" result="~500M" note="After dedup and filtering; long tail distribution" />
            <MathStep step="2" formula="Avg query length = 20 characters" result="20 chars" note="Each character = 1 trie node (optimized: compressed)" />
            <MathStep step="3" formula="Trie node size = ~60 bytes" result="60 B" note="char + children pointers + top-k list + metadata" />
            <MathStep step="4" formula="Total trie nodes (with prefix sharing)" result="~5B nodes" note="Common prefixes are shared; ~10× unique queries" />
            <MathStep step="5" formula="Raw trie size = 5B × 60 bytes" result="~300 GB" note="Uncompressed trie in memory" />
            <MathStep step="6" formula="Compressed (prefix collapsing)" result="~50-80 GB" note="Radix trie compresses single-child chains" final />
            <MathStep step="7" formula="Per shard (10 shards)" result="~5-8 GB" note="Fits comfortably in memory per server" final />
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Network / Bandwidth</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Request size (prefix string) = ~50 bytes" result="50 B" note="Prefix + headers + user context" />
            <MathStep step="2" formula="Response size (5 suggestions) = ~500 bytes" result="500 B" note="5 suggestions × ~100 chars each, JSON encoded" />
            <MathStep step="3" formula="Inbound bandwidth = 1.16M × 50 B" result="~58 MB/s" note="~464 Mbps inbound" />
            <MathStep step="4" formula="Outbound bandwidth = 1.16M × 500 B" result="~580 MB/s" note="~4.6 Gbps outbound" final />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Infrastructure Cost</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Trie servers (r6g.2xlarge, 64GB)" result="~10 nodes" note="~5-8 GB trie per shard, 10 shards for throughput" />
            <MathStep step="2" formula="Replicas (3 per shard for availability)" result="30 nodes" note="$0.50/hr × 30 × 730 hrs" final />
            <MathStep step="3" formula="Monthly compute" result="~$11K/mo" note="30 trie servers with replicas" />
            <MathStep step="4" formula="Data pipeline (Kafka + aggregation)" result="~$3K/mo" note="Log collection, query aggregation, trie rebuilds" />
            <MathStep step="5" formula="CDN for static suggestions" result="~$2K/mo" note="Cache top-level prefixes at edge" />
            <MathStep step="6" formula="Total monthly" result="~$16K/mo" note="Google-scale typeahead. Extremely cost-effective for impact." final />
          </div>
          <div className="mt-4 pt-3 border-t border-stone-100">
            <div className="text-[11px] text-stone-500">
              <strong className="text-stone-700">Key point:</strong> Autocomplete has enormous read amplification (10-20× queries vs actual searches) but each request is tiny and served from memory. Cost is dominated by the number of replicas needed for throughput, not storage.
            </div>
          </div>
        </Card>
      </div>

      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Avg QPS", val: "~1.16M", sub: "Peak: ~3.5M" },
            { label: "Trie Size", val: "~50-80 GB", sub: "Compressed, 10 shards" },
            { label: "Outbound BW", val: "~4.6 Gbps", sub: "~580 MB/s responses" },
            { label: "Monthly Cost", val: "~$16K", sub: "30 trie nodes + infra" },
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
          <Label color="#2563eb">Suggestion API (Read Path)</Label>
          <CodeBlock code={`# GET /v1/suggestions?prefix=how+t&limit=5
# Called on every keystroke (after debounce)

# Request
GET /v1/suggestions
  ?prefix=how+t        # current typed prefix
  &limit=5             # max suggestions
  &lang=en             # language
  &region=us           # for geo-aware results
  &user_id=abc123      # optional, for personalization

# Response (< 100ms)
{
  "prefix": "how t",
  "suggestions": [
    { "text": "how to tie a tie",     "score": 982340 },
    { "text": "how to screenshot",    "score": 871520 },
    { "text": "how to lose weight",   "score": 760210 },
    { "text": "how to cook rice",     "score": 654890 },
    { "text": "how to train a dog",   "score": 543210 }
  ],
  "trending": true  // indicates trending boost applied
}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Query Logging API (Write Path)</Label>
          <CodeBlock code={`# POST /v1/queries — Log completed searches
# Called when user submits a search (NOT on each key)

# Request (fire-and-forget, async)
POST /v1/queries
{
  "query": "how to tie a tie",
  "user_id": "abc123",
  "timestamp": "2025-01-15T10:23:45Z",
  "region": "us",
  "lang": "en",
  "device": "mobile",
  "session_id": "sess_789"
}

# Response
{ "status": "accepted" }

# This feeds the data pipeline:
# Kafka → Aggregator → Trie Builder
# Aggregated hourly / daily, NOT real-time per query`} />
          <div className="mt-4 pt-4 border-t border-stone-100">
            <Label color="#d97706">Design Decisions</Label>
            <ul className="space-y-1.5">
              <Point icon="→" color="#d97706">GET for suggestions — cacheable at CDN/proxy, idempotent, small payload</Point>
              <Point icon="→" color="#d97706">Client debounces at 100-150ms — reduces calls by 50-70%</Point>
              <Point icon="→" color="#d97706">No auth required for basic suggestions — reduces latency (no JWT decode)</Point>
              <Point icon="→" color="#d97706">Write path is async (Kafka) — query logging never blocks the search experience</Point>
            </ul>
          </div>
        </Card>
      </div>
    </div>
  );
}

function DesignSection() {
  const [phase, setPhase] = useState(0);
  const phases = [
    { label: "Start Simple", desc: "Single server with an in-memory trie. On each keystroke, walk the trie to the prefix node, return precomputed top-k. Works for millions of queries. Bottleneck: one server = limited QPS and single point of failure." },
    { label: "Separate Read/Write", desc: "Split into two paths. Read: stateless servers query replicated trie from memory. Write: search logs flow to Kafka → aggregator computes new frequencies → trie is rebuilt periodically (hourly). The trie is treated as a precomputed read-only data structure." },
    { label: "Shard + Replicate", desc: "Shard the trie by prefix range (a-f, g-m, n-z). Each shard is replicated 3× for availability and throughput. Client-side or gateway routes to correct shard based on first character. Trie is rebuilt offline and deployed via blue-green swap." },
    { label: "Full Architecture", desc: "Complete: CDN caches top prefixes → API Gateway → Trie Cluster (sharded + replicated, in-memory) ← Trie Builder (offline, Hadoop/Spark) ← Aggregator (Kafka + Flink) ← Query Logger. Trending overlay for real-time boost. Content filter for safety." },
  ];
  const diagrams = [
    () => (
      <svg viewBox="0 0 460 160" className="w-full">
        <DiagramBox x={70} y={65} w={80} h={38} label="Client" color="#2563eb"/>
        <DiagramBox x={230} y={65} w={110} h={42} label="Server\n+ In-Memory Trie" color="#9333ea"/>
        <DiagramBox x={400} y={65} w={80} h={38} label="Query Log" color="#d97706"/>
        <Arrow x1={110} y1={65} x2={175} y2={65} label="prefix" id="s0a"/>
        <Arrow x1={285} y1={65} x2={360} y2={65} label="log" id="s0b" dashed/>
        <rect x={140} y={115} width={200} height={22} rx={6} fill="#dc262608" stroke="#dc262630"/>
        <text x={240} y={127} textAnchor="middle" fill="#dc2626" fontSize="9" fontFamily="monospace">❌ Single server, no HA, stale data</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 200" className="w-full">
        <DiagramBox x={55} y={55} w={68} h={36} label="Client" color="#2563eb"/>
        <DiagramBox x={175} y={55} w={80} h={40} label="Trie\nServer ×3" color="#9333ea"/>
        <DiagramBox x={320} y={55} w={80} h={36} label="Read-Only\nTrie" color="#dc2626"/>
        <DiagramBox x={175} y={145} w={80} h={36} label="Aggregator" color="#059669"/>
        <DiagramBox x={320} y={145} w={80} h={40} label="Trie\nBuilder" color="#d97706"/>
        <Arrow x1={89} y1={55} x2={135} y2={55} id="rw1"/>
        <Arrow x1={215} y1={55} x2={280} y2={55} label="lookup" id="rw2" dashed/>
        <Arrow x1={215} y1={145} x2={280} y2={145} label="rebuild" id="rw3"/>
        <Arrow x1={360} y1={127} x2={360} y2={73} label="deploy" id="rw4"/>
        <rect x={100} y={175} width={260} height={18} rx={4} fill="#05966908" stroke="#05966930"/>
        <text x={230} y={187} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">✓ Read/Write separation</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 200" className="w-full">
        <DiagramBox x={55} y={85} w={68} h={36} label="Client" color="#2563eb"/>
        <DiagramBox x={155} y={85} w={55} h={36} label="Router" color="#64748b"/>
        <DiagramBox x={280} y={35} w={90} h={30} label="Shard a-f ×3" color="#9333ea"/>
        <DiagramBox x={280} y={85} w={90} h={30} label="Shard g-m ×3" color="#9333ea"/>
        <DiagramBox x={280} y={135} w={90} h={30} label="Shard n-z ×3" color="#9333ea"/>
        <DiagramBox x={410} y={85} w={70} h={36} label="Trie\nBuilder" color="#d97706"/>
        <Arrow x1={89} y1={85} x2={128} y2={85} id="sh1"/>
        <Arrow x1={183} y1={75} x2={235} y2={42} id="sh2"/>
        <Arrow x1={183} y1={85} x2={235} y2={85} id="sh3"/>
        <Arrow x1={183} y1={95} x2={235} y2={128} id="sh4"/>
        <Arrow x1={410} y1={67} x2={325} y2={50} label="deploy" id="sh5" dashed/>
        <Arrow x1={410} y1={85} x2={325} y2={85} id="sh6" dashed/>
        <Arrow x1={410} y1={103} x2={325} y2={128} id="sh7" dashed/>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 560 260" className="w-full">
        <DiagramBox x={40} y={50} w={50} h={34} label="User" color="#2563eb"/>
        <DiagramBox x={115} y={50} w={55} h={34} label="CDN" color="#64748b"/>
        <DiagramBox x={200} y={50} w={70} h={38} label="API\nGateway" color="#6366f1"/>
        <DiagramBox x={310} y={50} w={80} h={42} label="Trie\nCluster" color="#9333ea"/>
        <DiagramBox x={420} y={50} w={78} h={38} label="Content\nFilter" color="#dc2626"/>
        <DiagramBox x={200} y={140} w={70} h={34} label="Query\nLogger" color="#0891b2"/>
        <DiagramBox x={310} y={140} w={70} h={38} label="Kafka" color="#059669"/>
        <DiagramBox x={420} y={140} w={78} h={42} label="Frequency\nAggregator" color="#d97706"/>
        <DiagramBox x={530} y={140} w={50} h={38} label="Trie\nBuilder" color="#c026d3"/>
        <DiagramBox x={310} y={220} w={80} h={34} label="Trending\nOverlay" color="#be123c"/>
        <Arrow x1={65} y1={50} x2={88} y2={50} id="f0"/>
        <Arrow x1={143} y1={50} x2={165} y2={50} id="f1"/>
        <Arrow x1={235} y1={50} x2={270} y2={50} id="f2"/>
        <Arrow x1={350} y1={50} x2={381} y2={50} label="filter" id="f3"/>
        <Arrow x1={200} y1={69} x2={200} y2={123} label="async log" id="f4" dashed/>
        <Arrow x1={235} y1={140} x2={275} y2={140} id="f5"/>
        <Arrow x1={345} y1={140} x2={381} y2={140} id="f6"/>
        <Arrow x1={459} y1={140} x2={505} y2={140} label="hourly" id="f7"/>
        <Arrow x1={530} y1={121} x2={350} y2={68} label="deploy" id="f8" dashed/>
        <Arrow x1={310} y1={161} x2={310} y2={203} label="real-time" id="f9" dashed/>
        <Arrow x1={350} y1={220} x2={350} y2={71} label="boost" id="f10" dashed/>
      </svg>
    ),
  ];
  return (
    <div className="space-y-5">
      <Card accent="#9333ea">
        <Label color="#9333ea">Architecture Evolution</Label>
        <div className="flex gap-2 mb-4">
          {phases.map((p,i) => (
            <button key={i} onClick={() => setPhase(i)}
              className={`px-3.5 py-1.5 rounded-lg text-[12px] font-medium transition-all border ${i===phase ? "bg-purple-600 text-white border-purple-600" : "bg-white text-stone-500 border-stone-200 hover:border-stone-300 hover:text-stone-700"}`}>
              {i+1}. {p.label}
            </button>
          ))}
        </div>
        <p className="text-[13px] text-stone-500 mb-4">{phases[phase].desc}</p>
        <div className="bg-stone-50 rounded-lg border border-stone-200 p-3" style={{ minHeight: 170 }}>
          {diagrams[phase]()}
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#059669">Read Path — Suggestion Lookup</Label>
          <div className="space-y-1.5">
            {[
              { s:"1", t:'User types "how t" → client debounces 150ms → sends GET /suggestions?prefix=how+t', c:"text-blue-600" },
              { s:"2", t:"CDN check: prefix too specific → cache MISS → forward to API Gateway", c:"text-stone-500" },
              { s:"3", t:"Gateway routes to correct trie shard based on first character 'h' → Shard g-m", c:"text-purple-600" },
              { s:"4", t:"Trie server walks trie to node 'h→o→w→ →t' → reads precomputed top-5 list", c:"text-purple-600" },
              { s:"5", t:"Content filter: checks suggestions against blocklist → all pass", c:"text-red-600" },
              { s:"6", t:"Trending overlay: merge with any trending queries matching prefix", c:"text-rose-600" },
              { s:"7", t:'Return: ["how to tie a tie", "how to screenshot", ...] in 35ms', c:"text-emerald-600" },
            ].map((s,i) => (
              <div key={i} className="flex items-start gap-2 text-[12px]">
                <span className="text-stone-300 font-mono w-4 shrink-0">{s.s}</span>
                <span className={s.c}>{s.t}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <Label color="#d97706">Write Path — Trie Update Pipeline</Label>
          <div className="space-y-1.5">
            {[
              { s:"1", t:'User submits search "how to tie a tie" → logged async to Kafka', c:"text-blue-600" },
              { s:"2", t:"Kafka topic: search_queries. Partitioned by hash(query) for ordering.", c:"text-emerald-600" },
              { s:"3", t:"Flink/Spark Streaming aggregates query frequencies per time window (1hr, 1d, 7d)", c:"text-amber-600" },
              { s:"4", t:"Aggregated counts written to frequency table: {query: count, last_7d: count, ...}", c:"text-amber-600" },
              { s:"5", t:"Trie Builder (offline, hourly): reads frequency table → builds new trie", c:"text-purple-600" },
              { s:"6", t:"Each trie node stores precomputed top-k suggestions for its prefix", c:"text-purple-600" },
              { s:"7", t:"New trie deployed to servers via blue-green swap (no downtime)", c:"text-emerald-600" },
            ].map((s,i) => (
              <div key={i} className="flex items-start gap-2 text-[12px]">
                <span className="text-stone-300 font-mono w-4 shrink-0">{s.s}</span>
                <span className={s.c}>{s.t}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function AlgorithmSection() {
  const [sel, setSel] = useState("trie");
  const algos = {
    trie: { name: "Trie with Top-K ★", cx: "O(p) / O(Σ)",
      pros: ["O(p) lookup where p = prefix length — extremely fast","Precomputed top-k at every node eliminates sorting at query time","Natural prefix matching — the data structure IS the algorithm","Production-proven at Google, Bing, Amazon"],
      cons: ["Memory-heavy (~60 bytes/node, billions of nodes)","Rebuilding the entire trie takes time (offline process)","Not great for fuzzy/typo matching"],
      when: "The Trie with precomputed top-k is the industry standard for autocomplete. Each node stores the top-k suggestions reachable from that prefix. Lookup is O(p) where p is prefix length — typically < 20 characters.",
      code: `# Trie with Precomputed Top-K Suggestions
class TrieNode:
    def __init__(self):
        self.children = {}          # char → TrieNode
        self.top_k = []             # precomputed [(query, score)]
        self.is_end = False

class AutocompleteTrie:
    def __init__(self, k=5):
        self.root = TrieNode()
        self.k = k

    def build(self, query_frequencies: dict):
        """Build trie from {query: frequency} dict."""
        for query, freq in query_frequencies.items():
            node = self.root
            for char in query:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                # Update top-k at every node along the path
                self._update_top_k(node, query, freq)
            node.is_end = True

    def _update_top_k(self, node, query, freq):
        node.top_k.append((query, freq))
        node.top_k.sort(key=lambda x: -x[1])
        node.top_k = node.top_k[:self.k]

    def search(self, prefix: str) -> list:
        """O(p) lookup — just walk to prefix node."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return node.top_k  # already sorted!` },
    radix: { name: "Radix Trie (Compressed)", cx: "O(p) / O(n)",
      pros: ["Collapses single-child chains → 3-5× less memory","Same O(p) lookup performance as standard trie","Much fewer nodes to traverse for long common prefixes","Better cache locality (fewer pointer chases)"],
      cons: ["More complex insertion and deletion logic","Node splitting on insert (when new branch diverges mid-edge)","Still not ideal for fuzzy matching"],
      when: "Use Radix Trie (Patricia Trie) when memory is a concern. If many queries share long prefixes (e.g., 'how to ...'), the standard trie wastes nodes on single-child chains. Radix trie collapses 'h-o-w- -t-o- ' into one edge.",
      code: `# Radix Trie (Compressed Trie / Patricia Trie)
class RadixNode:
    def __init__(self, edge_label=""):
        self.edge_label = edge_label    # compressed edge
        self.children = {}              # first_char → RadixNode
        self.top_k = []
        self.is_end = False

# Example: queries ["how to cook", "how to code", "howdy"]
#
# Standard Trie:  h→o→w→ →t→o→ →c→o→o→k
#                                     →d→e
#                         →d→y
# = 16 nodes
#
# Radix Trie:     "how" → " to co" → "ok"
#                                   → "de"
#                        → "dy"
# = 5 nodes (3× compression)
#
# Lookup: same O(p) — compare edge labels character
# by character instead of single characters.
# Trade-off: edge comparison vs pointer chasing.

class RadixTrie:
    def search(self, prefix):
        node = self.root
        remaining = prefix
        while remaining:
            first = remaining[0]
            if first not in node.children:
                return []
            child = node.children[first]
            edge = child.edge_label
            if remaining.startswith(edge):
                remaining = remaining[len(edge):]
                node = child
            elif edge.startswith(remaining):
                return child.top_k  # partial match
            else:
                return []  # diverges
        return node.top_k` },
    hash_prefix: { name: "Hash-Based Prefix Map", cx: "O(1) / O(n×p)",
      pros: ["O(1) lookup per prefix — fastest possible","Simple implementation (just a HashMap)","Great for fixed, bounded prefix lengths","Easy to shard and cache"],
      cons: ["Stores every prefix separately — massive memory duplication","No structural sharing between related prefixes","Not practical for long queries (20 chars = 20 entries per query)"],
      when: "Use a prefix hash map when the prefix space is bounded (e.g., only serving top 2-3 character prefixes at CDN edge). Pre-generate all prefix→suggestions mappings and store as key-value pairs. Great for caching the most common prefixes.",
      code: `# Hash-Based Prefix Map (simple but memory-heavy)
class PrefixMap:
    def __init__(self, k=5):
        self.map = {}   # prefix_string → [(query, score)]
        self.k = k

    def build(self, query_frequencies: dict):
        """Generate ALL prefix → top-k mappings."""
        prefix_scores = defaultdict(list)

        for query, freq in query_frequencies.items():
            for i in range(1, len(query) + 1):
                prefix = query[:i]
                prefix_scores[prefix].append((query, freq))

        for prefix, scores in prefix_scores.items():
            scores.sort(key=lambda x: -x[1])
            self.map[prefix] = scores[:self.k]

    def search(self, prefix: str) -> list:
        return self.map.get(prefix, [])

# Memory: 500M queries × 20 avg len = 10B entries
# vs Trie: 5B nodes with prefix sharing
# → 2× worse memory, but O(1) vs O(20) lookup
# Good for: CDN edge cache with top 2-3 char prefixes` },
    trending: { name: "Trending Overlay (Real-Time)", cx: "O(log n) / O(n)",
      pros: ["Surfaces trending topics within minutes (not hours)","Decoupled from the main trie rebuild cycle","Small data structure — only recent trending queries","Can boost or suppress specific terms dynamically"],
      cons: ["Needs separate real-time stream processing (Flink/Storm)","Merging with main trie results adds complexity","Risk of surfacing flash-in-the-pan queries"],
      when: "The trending overlay is a small, frequently-updated data structure that runs alongside the main trie. It captures queries with sudden frequency spikes (e.g., breaking news). During suggestion lookup, results from the main trie are merged with trending matches, with trending getting a score boost.",
      code: `# Trending Overlay — Real-Time Suggestion Boost
class TrendingOverlay:
    def __init__(self, window_minutes=60, min_spike=5.0):
        self.current_counts = Counter()   # last 5min
        self.baseline_counts = Counter()  # last 7d avg
        self.trending = {}                # query → boost_score
        self.min_spike = min_spike        # 5× baseline

    def ingest(self, query: str):
        """Called from real-time stream (Flink/Kafka)."""
        self.current_counts[query] += 1

    def compute_trending(self):
        """Run every 5 minutes."""
        for query, count in self.current_counts.items():
            baseline = self.baseline_counts.get(query, 1)
            ratio = count / baseline
            if ratio >= self.min_spike:
                self.trending[query] = ratio * count
        self.current_counts.clear()

    def merge_with_trie(self, prefix, trie_results):
        """Merge trending into trie results."""
        merged = list(trie_results)
        for query, boost in self.trending.items():
            if query.startswith(prefix):
                merged.append((query, boost * 10))
        merged.sort(key=lambda x: -x[1])
        return merged[:5]

# Example: "super bowl" spikes 50× during the game
# Baseline: 100/hr → Current: 5000/5min
# Spike ratio: 50× → trending! Boost into suggestions.` },
  };
  const a = algos[sel];
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Algorithm Comparison</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[12px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Algorithm</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Lookup</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Memory</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Freshness</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Best For</th>
            </tr></thead>
            <tbody>
              {[
                { n:"Trie + Top-K ★", m:"O(p)", a:"High (shared prefixes)", b:"Hourly rebuild", f:"Primary suggestion engine", hl:true },
                { n:"Radix Trie", m:"O(p)", a:"3-5× less than Trie", b:"Hourly rebuild", f:"Memory-constrained" },
                { n:"Prefix Hash Map", m:"O(1)", a:"Very high (no sharing)", b:"Easy to update", f:"CDN edge caching" },
                { n:"Trending Overlay", m:"O(log n)", a:"Tiny (recent only)", b:"Real-time (minutes)", f:"Breaking news / events" },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-purple-50" : i%2?"bg-stone-50/50":""}>
                  <td className={`px-3 py-2 font-mono ${r.hl?"text-purple-700 font-bold":"text-stone-600"}`}>{r.n}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.m}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.a}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.b}</td>
                  <td className="text-center px-3 py-2 text-stone-400">{r.f}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-3 space-y-2">
          {Object.entries(algos).map(([k,v]) => (
            <button key={k} onClick={() => setSel(k)}
              className={`w-full text-left px-3.5 py-2.5 rounded-lg text-[12px] font-medium border transition-all ${k===sel?"bg-purple-600 text-white border-purple-600":"bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
              {v.name}
            </button>
          ))}
        </div>
        <div className="col-span-9 space-y-5">
          <Card>
            <div className="flex items-center gap-3 mb-3">
              <span className="text-[14px] font-bold text-stone-800">{a.name}</span>
              <Pill bg="#f3e8ff" color="#7c3aed">{a.cx}</Pill>
            </div>
            <div className="grid grid-cols-2 gap-5">
              <div>
                <div className="text-[10px] font-bold text-emerald-600 uppercase tracking-wider mb-1.5">Pros</div>
                <ul className="space-y-1.5">{a.pros.map((p,i) => <Point key={i} icon="✓" color="#059669">{p}</Point>)}</ul>
              </div>
              <div>
                <div className="text-[10px] font-bold text-red-500 uppercase tracking-wider mb-1.5">Cons</div>
                <ul className="space-y-1.5">{a.cons.map((c,i) => <Point key={i} icon="✗" color="#dc2626">{c}</Point>)}</ul>
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-stone-100">
              <div className="text-[10px] font-bold text-amber-600 uppercase tracking-wider mb-1">When to Use</div>
              <p className="text-[12px] text-stone-500">{a.when}</p>
            </div>
          </Card>
          <CodeBlock title={`${a.name} — Python`} code={a.code} />
        </div>
      </div>
    </div>
  );
}

function DataModelSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Query Frequency Table (Source of Truth)</Label>
          <CodeBlock code={`-- Aggregated search query frequencies
-- Updated hourly by the data pipeline
CREATE TABLE query_frequencies (
    query_hash     BIGINT PRIMARY KEY,
    query          TEXT NOT NULL,
    count_1h       BIGINT DEFAULT 0,
    count_1d       BIGINT DEFAULT 0,
    count_7d       BIGINT DEFAULT 0,
    count_30d      BIGINT DEFAULT 0,
    count_all_time BIGINT DEFAULT 0,
    language       VARCHAR(5) DEFAULT 'en',
    region         VARCHAR(10),
    is_blocked     BOOLEAN DEFAULT FALSE,
    last_updated   TIMESTAMP,
    created_at     TIMESTAMP DEFAULT NOW()
);

-- Index for trie builder: fetch top queries
CREATE INDEX idx_query_freq_score
    ON query_frequencies(count_7d DESC)
    WHERE NOT is_blocked;

-- Partitioned by language for multi-locale
-- trie builds`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Trie Serialization Format</Label>
          <CodeBlock code={`# Trie is built offline and serialized to a
# compact binary format for deployment

# Binary Trie Format (per node):
# ┌──────────────────────────────────────┐
# │ edge_label_len  (1 byte)             │
# │ edge_label      (variable)           │
# │ num_children    (1 byte)             │
# │ children_offset (4 bytes each)       │
# │ top_k_count     (1 byte, max 10)     │
# │ top_k_entries:                        │
# │   query_id      (4 bytes)            │
# │   score         (4 bytes)            │
# └──────────────────────────────────────┘

# Deployment flow:
# 1. Build trie in memory (Spark/Hadoop)
# 2. Serialize to binary file (~5-8 GB/shard)
# 3. Upload to S3
# 4. Trie servers download + mmap into memory
# 5. Blue-green swap: old trie → new trie

# query_id maps to a separate string table
# to avoid storing full query strings in every node`} />
        </Card>
      </div>
      <Card accent="#9333ea">
        <Label color="#9333ea">Storage Architecture — Data Pipeline</Label>
        <p className="text-[12px] text-stone-500 mb-4">Autocomplete has a clear separation: the read path serves from an in-memory trie, while the write path flows through a batch/streaming pipeline that periodically rebuilds the trie.</p>
        <div className="grid grid-cols-4 gap-4">
          {[
            { t: "Search Query Log (Kafka)", d: "Raw stream of every completed search query. Partitioned by hash(query). Retained 7 days.", data: "~10B events/day", tech: "Kafka (50+ partitions)", size: "~2 TB/day" },
            { t: "Frequency Aggregation (Flink/Spark)", d: "Rolls up raw logs into per-query counts for multiple time windows (1h, 1d, 7d, 30d).", data: "500M unique queries", tech: "Apache Flink (streaming) + Spark (batch)", size: "~50 GB output" },
            { t: "Trie Binary (S3)", d: "Serialized trie files, one per shard. Built hourly, stored with versioning.", data: "10 shard files", tech: "S3 with versioning", size: "~5-8 GB per shard" },
            { t: "In-Memory Trie (Servers)", d: "Memory-mapped trie loaded on trie servers. Read-only, swapped on rebuild.", data: "Active serving copy", tech: "Custom C++/Go/Rust process", size: "~5-8 GB per shard (RAM)" },
          ].map((o,i) => (
            <div key={i} className="rounded-lg border border-stone-200 p-4">
              <div className="text-[11px] font-bold text-stone-800 mb-1.5">{o.t}</div>
              <p className="text-[11px] text-stone-500 mb-2">{o.d}</p>
              <div className="space-y-0.5 text-[10px]">
                <div><span className="font-bold text-stone-600">Data:</span> <span className="text-stone-400">{o.data}</span></div>
                <div><span className="font-bold text-stone-600">Tech:</span> <span className="text-stone-400">{o.tech}</span></div>
                <div><span className="font-bold text-stone-600">Size:</span> <span className="font-mono text-stone-400">{o.size}</span></div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 pt-3 border-t border-stone-100">
          <div className="text-[11px] text-stone-500">
            <strong className="text-stone-700">Key insight to mention in interview:</strong> The trie is NOT updated in real-time per query. It's rebuilt hourly from aggregated frequency data. This is a deliberate design choice: batch rebuilds produce a perfectly optimized, read-only data structure. Real-time freshness is handled by the separate trending overlay, which is tiny and fast to update.
          </div>
        </div>
      </Card>
    </div>
  );
}

function ScalabilitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Scaling the Read Path</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Shard by prefix range</strong> — split trie into N shards by first character (or first 2 chars). Each shard is independently replicated. Query routing is trivial: first char → shard.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Replicate for throughput</strong> — each shard has 3+ replicas. At 3.5M peak QPS across 10 shards = 350K QPS/shard. With 3 replicas = ~117K QPS/server. Easily achievable from memory.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">CDN caching for popular prefixes</strong> — top 1-2 character prefixes (e.g., "a", "th", "wh") account for most requests. Cache at CDN with 5-min TTL. Eliminates 30-50% of backend traffic.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Client-side caching</strong> — browser/app caches results locally. If user types "how " and has cached results for "how", show immediately without network call.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Scaling the Write Path</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Kafka partitioning</strong> — partition by hash(query) so all occurrences of the same query go to the same partition. Enables simple count aggregation per consumer.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Two-stage aggregation</strong> — Flink does real-time windowed counts (5-min tumbling windows). Spark does daily/weekly rollups. They feed different use cases (trending vs trie rebuild).</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Trie builder is embarrassingly parallel</strong> — each shard's trie can be built independently. Build all 10 shards in parallel on a Spark cluster. Takes ~15 minutes per rebuild.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Incremental updates (optional)</strong> — instead of full rebuild, diff new frequency data and patch the trie. More complex but reduces rebuild time to ~2 minutes.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Deployment</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Option A: Full Replication ★", d:"Complete trie replicated to every region. Trie builder runs centrally, pushes serialized trie to all regions.", pros:["Lowest latency — fully local serving","Simple routing (any region can serve)","No cross-region dependencies on read path"], cons:["Storage cost × N regions","Rebuild must propagate to all regions"], pick:true },
            { t:"Option B: Region-Specific Tries", d:"Each region has its own trie built from region-specific query data. US users see US-popular suggestions.", pros:["Suggestions naturally localized","Smaller per-region trie","Independent rebuild schedules"], cons:["Users traveling get different suggestions","More complex build pipeline"], pick:false },
            { t:"Option C: Hybrid (Google-style)", d:"Global trie for universal queries + regional overlay for locale-specific suggestions. Merge at query time.", pros:["Best of both worlds","Global consistency + local relevance","Trending can be region-specific"], cons:["Most complex architecture","Merge logic adds latency"], pick:false },
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
        <Label color="#d97706">Critical Decision: Stale Suggestions vs No Suggestions</Label>
        <p className="text-[12px] text-stone-500 mb-4">What happens when the trie cluster has issues? Unlike transactional systems, autocomplete can always degrade gracefully — stale suggestions are infinitely better than no suggestions.</p>
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Serve Stale Trie (Recommended)</div>
            <p className="text-[11px] text-stone-500 mb-2">If trie rebuild fails, keep serving the last good trie. Suggestions may be hours old, but still useful.</p>
            <ul className="space-y-1"><Point icon="✓" color="#059669">Users always get suggestions</Point><Point icon="✓" color="#059669">Last-known-good trie is already in memory</Point><Point icon="✓" color="#059669">Trending overlay still provides freshness</Point></ul>
          </div>
          <div className="rounded-lg border border-red-200 bg-white p-4">
            <div className="text-[11px] font-bold text-red-600 mb-1.5">Disable Autocomplete (Emergency Only)</div>
            <p className="text-[11px] text-stone-500 mb-2">If trie is corrupted or serving harmful suggestions, disable autocomplete entirely.</p>
            <ul className="space-y-1"><Point icon="⚠" color="#d97706">Kill switch at API Gateway level</Point><Point icon="⚠" color="#d97706">Return empty suggestions, not errors</Point><Point icon="⚠" color="#d97706">Search still works — just no autocomplete</Point></ul>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Blue-Green Trie Deployment</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Two trie slots per server</strong> — "active" and "standby". Active serves traffic. Standby loads the new trie in background.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Atomic swap</strong> — once standby finishes loading + health check passes, swap active↔standby. Zero downtime, zero warm-up.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Rollback</strong> — if new trie has issues (higher error rate, lower coverage), swap back to old trie in seconds. Old trie stays in the standby slot.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Canary deployment</strong> — deploy new trie to 1 replica first. Monitor for 10 minutes. If healthy, roll to remaining replicas.</Point>
          </ul>
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Failure Recovery Matrix</Label>
          <div className="overflow-hidden rounded-lg border border-stone-200">
            <table className="w-full text-[11px]">
              <thead><tr className="bg-stone-50">
                <th className="text-left px-3 py-2 font-semibold text-stone-500">Component</th>
                <th className="text-left px-3 py-2 font-semibold text-stone-500">Impact</th>
                <th className="text-left px-3 py-2 font-semibold text-stone-500">Recovery</th>
                <th className="text-left px-3 py-2 font-semibold text-stone-500">RTO</th>
              </tr></thead>
              <tbody>
                {[
                  { c: "Trie server", impact: "Reduced QPS capacity", recovery: "Replicas absorb traffic. Replace pod.", rto: "<30s" },
                  { c: "All replicas of 1 shard", impact: "One prefix range unavailable", recovery: "Return empty for that prefix. Fast restart.", rto: "1-2 min" },
                  { c: "Trie builder", impact: "Stale suggestions", recovery: "Serve last good trie. Rebuild is hourly.", rto: "0 (no user impact)" },
                  { c: "Kafka pipeline", impact: "Query logs not collected", recovery: "Kafka replay from offset. No user impact.", rto: "0 (async path)" },
                  { c: "CDN", impact: "More traffic to origin", recovery: "Direct to API Gateway. Higher latency.", rto: "Instant failover" },
                  { c: "Content filter", impact: "Unsafe suggestions shown", recovery: "Kill switch: disable autocomplete entirely.", rto: "<1 min" },
                ].map((r,i) => (
                  <tr key={i} className={i%2?"bg-stone-50/50":""}>
                    <td className="px-3 py-2 font-mono text-stone-700">{r.c}</td>
                    <td className="px-3 py-2 text-stone-500">{r.impact}</td>
                    <td className="px-3 py-2 text-stone-500">{r.recovery}</td>
                    <td className="px-3 py-2 font-mono text-stone-400">{r.rto}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}

function ObservabilitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-5">
        <Card accent="#0284c7">
          <Label color="#0284c7">Key Metrics to Track</Label>
          <div className="space-y-2.5">
            {[
              { m: "suggestion_latency_ms", t: "Histogram", d: "End-to-end latency (target: p99 < 100ms)" },
              { m: "suggestion_qps", t: "Gauge", d: "Requests per second to suggestion service" },
              { m: "empty_result_rate", t: "Gauge", d: "% of requests returning 0 suggestions" },
              { m: "cache_hit_rate", t: "Gauge", d: "CDN + client cache hit ratio" },
              { m: "trie_coverage", t: "Gauge", d: "% of prefixes with ≥1 suggestion" },
              { m: "trie_build_duration", t: "Histogram", d: "Time to rebuild trie (target: < 30 min)" },
              { m: "trending_lag_seconds", t: "Gauge", d: "Delay between trending event and suggestion" },
              { m: "blocked_suggestion_rate", t: "Counter", d: "Suggestions filtered by content safety" },
            ].map((m,i) => (
              <div key={i} className="flex items-start gap-2.5 text-[11px]">
                <code className="font-mono text-sky-700 bg-sky-50 px-1.5 py-0.5 rounded shrink-0">{m.m}</code>
                <div className="flex-1">
                  <span className="text-stone-500">{m.d}</span>
                  <span className="text-stone-300 ml-1">({m.t})</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Dashboards</Label>
          <div className="space-y-3">
            {[
              { name: "Suggestion Performance", panels: "p50/p99 latency, QPS, cache hit rate, empty result rate", tool: "Grafana" },
              { name: "Trie Health", panels: "Build success/fail, build duration, trie size, coverage %, deployment status", tool: "Grafana" },
              { name: "Data Pipeline", panels: "Kafka lag, aggregator throughput, query volume trend, trending detection rate", tool: "Grafana" },
              { name: "Content Safety", panels: "Blocked suggestions count, new blocklist entries, filter latency overhead", tool: "Internal" },
            ].map((d,i) => (
              <div key={i} className="border border-stone-100 rounded-lg p-3">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-bold text-stone-700">{d.name}</span>
                  <Pill bg="#ecfdf5" color="#059669">{d.tool}</Pill>
                </div>
                <div className="text-[10px] text-stone-400 mt-0.5">{d.panels}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Alerts</Label>
          <div className="space-y-3">
            {[
              { alert: "p99 latency > 100ms", severity: "P1", action: "Check trie server health, network, shard hot-spots" },
              { alert: "Empty result rate > 10%", severity: "P2", action: "Trie may be corrupted or missing shard. Verify trie build." },
              { alert: "Trie build failed 2× consecutively", severity: "P2", action: "Check Spark cluster, frequency table integrity" },
              { alert: "Kafka consumer lag > 10M", severity: "P2", action: "Scale aggregation consumers, check Flink job" },
              { alert: "Blocked suggestion served", severity: "P1", action: "Content filter bypass. Activate kill switch if needed." },
              { alert: "CDN cache hit < 20%", severity: "P3", action: "Check CDN config, TTLs, cache key normalization" },
            ].map((a,i) => (
              <div key={i} className="border border-stone-100 rounded-lg p-2.5">
                <div className="flex items-center gap-2">
                  <Pill bg={a.severity==="P1"?"#fef2f2":a.severity==="P2"?"#fffbeb":"#f0fdf4"} color={a.severity==="P1"?"#dc2626":a.severity==="P2"?"#d97706":"#059669"}>{a.severity}</Pill>
                  <span className="text-[11px] font-bold text-stone-700">{a.alert}</span>
                </div>
                <div className="text-[10px] text-stone-400 mt-0.5">{a.action}</div>
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
      <Card className="bg-red-50/50 border-red-200">
        <div className="flex items-start gap-3">
          <span className="text-lg">⚠️</span>
          <div>
            <div className="text-[12px] font-bold text-red-700">Interview Tip — Content Safety is Non-Negotiable</div>
            <p className="text-[12px] text-stone-500 mt-0.5">Always mention content safety when discussing autocomplete. Suggesting harmful, offensive, or defamatory queries is a PR disaster. Google, Bing, and Amazon all have dedicated safety pipelines for this. Bringing it up unprompted shows maturity.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        {[
          { mode: "Offensive / Harmful Suggestions", desc: "Autocomplete suggests racist, sexual, violent, or defamatory queries. Happens because users actually search for these terms.", impact: "PR crisis, legal liability, user harm. Google has been sued over autocomplete suggestions.", fix: "Multi-layer filter: blocklist of terms, ML classifier for novel offensive phrases, human review pipeline for edge cases. Filter BEFORE returning to client, never after.", severity: "Critical" },
          { mode: "Trie Corruption on Deploy", desc: "New trie build has a bug — missing data, corrupted binary, or ranking regression. Deployed to all servers.", impact: "Empty suggestions, wrong suggestions, or crash across all shards.", fix: "Blue-green deployment with automated canary testing: compare coverage, latency, and sample suggestions against the old trie. Auto-rollback if metrics regress.", severity: "Critical" },
          { mode: "Hot Prefix (Thundering Herd)", desc: "Breaking news causes millions of users to type the same prefix simultaneously (e.g., 'who won the election').", impact: "Single trie shard overwhelmed. CDN cache helps but first requests hammer origin.", fix: "CDN caching for short prefixes (1-3 chars). Request coalescing at API Gateway (multiple identical requests get one backend call). Over-provision for peak.", severity: "High" },
          { mode: "Trending Manipulation (SEO/Abuse)", desc: "Bots generate fake searches to push a term into trending suggestions (astroturfing).", impact: "Manipulated suggestions damage trust and promote spam/misinformation.", fix: "Rate-limit per user/IP on query logging. Require minimum number of unique users (not just count). Anomaly detection on sudden spikes from few IPs. Human review for promoted trending terms.", severity: "High" },
          { mode: "Stale Suggestions After Major Event", desc: "A person dies, company changes name, or product is recalled — but autocomplete still suggests outdated queries.", impact: "Embarrassing or harmful (suggesting a dead person 'is alive', promoting recalled products).", fix: "Emergency blocklist update (minutes, not hours). Trending overlay can suppress terms. Have a fast-path for manual overrides that bypass the hourly trie rebuild.", severity: "Medium" },
          { mode: "Privacy Leakage via Suggestions", desc: "Autocomplete reveals that other people searched for sensitive queries (medical conditions, legal issues).", impact: "Privacy concerns — 'why does Google suggest X when I type my name?'", fix: "Minimum frequency threshold before a query enters suggestions (e.g., ≥100 unique users). Never show suggestions based on a single user's history in the global trie. Personalized suggestions stay client-side.", severity: "High" },
        ].map((f,i) => (
          <Card key={i}>
            <div className="flex items-center justify-between mb-2">
              <div className="text-[12px] font-bold text-stone-800">{f.mode}</div>
              <Pill bg={f.severity==="Critical"?"#fef2f2":f.severity==="High"?"#fffbeb":"#f0fdf4"} color={f.severity==="Critical"?"#dc2626":f.severity==="High"?"#d97706":"#059669"}>{f.severity}</Pill>
            </div>
            <p className="text-[12px] text-stone-500 mb-2">{f.desc}</p>
            <div className="text-[11px] text-red-600 mb-1.5"><strong>Impact:</strong> {f.impact}</div>
            <div className="text-[11px] text-emerald-700"><strong>Mitigation:</strong> {f.fix}</div>
          </Card>
        ))}
      </div>
    </div>
  );
}

function ServicesSection() {
  return (
    <div className="space-y-5">
      <Card className="bg-teal-50/50 border-teal-200">
        <div className="flex items-start gap-3">
          <span className="text-lg">🧩</span>
          <div>
            <div className="text-[12px] font-bold text-teal-700">Why This Matters</div>
            <p className="text-[12px] text-stone-500 mt-0.5">In an HLD you say "autocomplete service" and draw one box. In production, that's a read-path serving cluster, a streaming pipeline, a batch rebuild system, a content filter, and a CDN — each independently deployed, scaled, and monitored.</p>
          </div>
        </div>
      </Card>

      <Card accent="#0f766e">
        <Label color="#0f766e">Service Decomposition — What We Actually Build</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { name: "Suggestion Service (Read)", owns: "Serves autocomplete requests. Loads trie into memory. Routes prefix to correct shard. Returns top-k suggestions with content filtering.", tech: "Go/C++ for low-latency serving. Memory-mapped trie. K8s Deployment.", api: "GET /v1/suggestions?prefix=X&limit=5", scale: "10 shards × 3 replicas = 30 pods. Each handles ~100K QPS.", stateful: true,
              modules: ["Prefix Router (first char → shard mapping)", "Trie Walker (traverse to prefix node, read top-k)", "Content Filter (check blocklist, ML classifier)", "Trending Merger (merge trie results with trending overlay)", "Blue-Green Loader (swap active/standby trie)", "Response Cache (in-process LRU for hot prefixes)"] },
            { name: "Data Pipeline (Write)", owns: "Ingests search query logs, aggregates frequencies, triggers trie rebuilds. Async, never on the critical path.", tech: "Kafka (ingestion) + Apache Flink (streaming aggregation) + Spark (batch trie build)", api: "Kafka topic: search_queries (input), trie_build_trigger (output)", scale: "Kafka: 50 partitions. Flink: 10 task managers. Spark: on-demand cluster.", stateful: false,
              modules: ["Query Logger (fire-and-forget Kafka producer)", "Stream Aggregator (Flink: 5-min tumbling window counts)", "Batch Aggregator (Spark: daily/weekly rollups)", "Trie Builder (reads frequencies → builds serialized trie)", "Trie Deployer (uploads to S3, notifies servers)", "Quality Validator (checks trie coverage, size, sample queries)"] },
            { name: "Content Safety Service", owns: "Filters autocomplete suggestions for harmful, offensive, or legally problematic content. Called synchronously on every suggestion response.", tech: "Python ML inference (ONNX) + Redis blocklist. Low-latency (<5ms).", api: "gRPC: FilterSuggestions(suggestions) → filtered_suggestions", scale: "Stateless. Scale with suggestion service traffic.", stateful: false,
              modules: ["Blocklist Checker (exact match against 500K+ blocked terms)", "ML Classifier (ONNX model for novel offensive content)", "Category Filter (adult, violence, drugs per region/user-age)", "Emergency Override (manual block within minutes)", "Audit Logger (log all filtered suggestions for review)", "False Positive Reviewer (human-in-loop for edge cases)"] },
          ].map((s,i) => (
            <div key={i} className={`rounded-lg border p-4 ${s.stateful ? "border-teal-200 bg-teal-50/30" : "border-stone-200"}`}>
              <div className="flex items-center gap-2 mb-2">
                <div className="text-[12px] font-bold text-stone-800">{s.name}</div>
                <Pill bg={s.stateful?"#f0fdfa":"#f3f4f6"} color={s.stateful?"#0f766e":"#6b7280"}>{s.stateful?"Stateful":"Stateless"}</Pill>
              </div>
              <div className="text-[11px] text-stone-500 mb-2">{s.owns}</div>
              <div className="space-y-1 text-[10px]">
                <div><span className="font-bold text-stone-600">Tech:</span> <span className="text-stone-400">{s.tech}</span></div>
                <div><span className="font-bold text-stone-600">API:</span> <span className="font-mono text-stone-400">{s.api}</span></div>
                <div><span className="font-bold text-stone-600">Scale:</span> <span className="text-stone-400">{s.scale}</span></div>
              </div>
              <div className="mt-3 pt-2 border-t border-stone-100">
                <div className="text-[9px] font-bold text-teal-600 uppercase tracking-wider mb-1">Internal Modules</div>
                {s.modules.map((m,j) => (
                  <div key={j} className="text-[10px] text-stone-500 flex items-center gap-1.5">
                    <span className="w-1 h-1 rounded-full bg-teal-400 shrink-0"/>
                    {m}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card accent="#9333ea">
        <Label color="#9333ea">Full Service Architecture Diagram</Label>
        <svg viewBox="0 0 720 370" className="w-full">
          <rect x={5} y={5} width={710} height={360} rx={10} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          {/* Zone labels */}
          <rect x={15} y={12} width={90} height={18} rx={4} fill="#2563eb08" stroke="#2563eb30" strokeWidth={1}/>
          <text x={60} y={24} textAnchor="middle" fill="#2563eb" fontSize="7" fontWeight="700" fontFamily="monospace">CLIENT</text>
          <rect x={115} y={12} width={90} height={18} rx={4} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
          <text x={160} y={24} textAnchor="middle" fill="#6366f1" fontSize="7" fontWeight="700" fontFamily="monospace">EDGE</text>
          <rect x={215} y={12} width={160} height={18} rx={4} fill="#9333ea08" stroke="#9333ea30" strokeWidth={1}/>
          <text x={295} y={24} textAnchor="middle" fill="#9333ea" fontSize="7" fontWeight="700" fontFamily="monospace">READ PATH (&lt; 100ms)</text>
          <rect x={385} y={12} width={200} height={18} rx={4} fill="#05966908" stroke="#05966930" strokeWidth={1}/>
          <text x={485} y={24} textAnchor="middle" fill="#059669" fontSize="7" fontWeight="700" fontFamily="monospace">WRITE PATH (ASYNC)</text>
          <rect x={595} y={12} width={110} height={18} rx={4} fill="#dc262608" stroke="#dc262630" strokeWidth={1}/>
          <text x={650} y={24} textAnchor="middle" fill="#dc2626" fontSize="7" fontWeight="700" fontFamily="monospace">STORAGE</text>

          {/* Client */}
          <rect x={20} y={55} width={80} height={45} rx={8} fill="#2563eb12" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={60} y={73} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Browser</text>
          <text x={60} y={88} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">debounce + cache</text>

          {/* CDN */}
          <rect x={125} y={55} width={70} height={40} rx={8} fill="#64748b12" stroke="#64748b" strokeWidth={1.5}/>
          <text x={160} y={73} textAnchor="middle" fill="#64748b" fontSize="9" fontWeight="600" fontFamily="monospace">CDN</text>
          <text x={160} y={86} textAnchor="middle" fill="#64748b80" fontSize="7" fontFamily="monospace">top prefixes</text>

          {/* API Gateway */}
          <rect x={220} y={50} width={80} height={50} rx={8} fill="#6366f112" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={260} y={70} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">API</text>
          <text x={260} y={83} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">Gateway</text>
          <text x={260} y={96} textAnchor="middle" fill="#6366f180" fontSize="7" fontFamily="monospace">route by prefix</text>

          {/* Trie Cluster */}
          <rect x={325} y={40} width={95} height={70} rx={8} fill="#9333ea12" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={372} y={60} textAnchor="middle" fill="#9333ea" fontSize="10" fontWeight="700" fontFamily="monospace">Trie</text>
          <text x={372} y={75} textAnchor="middle" fill="#9333ea" fontSize="10" fontWeight="700" fontFamily="monospace">Cluster</text>
          <text x={372} y={90} textAnchor="middle" fill="#9333ea80" fontSize="7" fontFamily="monospace">10 shards × 3 replicas</text>
          <text x={372} y={102} textAnchor="middle" fill="#9333ea80" fontSize="7" fontFamily="monospace">in-memory, read-only</text>

          {/* Content Filter */}
          <rect x={325} y={125} width={95} height={38} rx={6} fill="#dc262612" stroke="#dc2626" strokeWidth={1}/>
          <text x={372} y={142} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Content Filter</text>
          <text x={372} y={155} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">blocklist + ML</text>

          {/* Trending Overlay */}
          <rect x={325} y={178} width={95} height={38} rx={6} fill="#be123c12" stroke="#be123c" strokeWidth={1}/>
          <text x={372} y={195} textAnchor="middle" fill="#be123c" fontSize="9" fontWeight="600" fontFamily="monospace">Trending</text>
          <text x={372} y={208} textAnchor="middle" fill="#be123c80" fontSize="7" fontFamily="monospace">real-time overlay</text>

          {/* Query Logger */}
          <rect x={220} y={140} width={80} height={35} rx={6} fill="#0891b212" stroke="#0891b2" strokeWidth={1}/>
          <text x={260} y={162} textAnchor="middle" fill="#0891b2" fontSize="9" fontWeight="600" fontFamily="monospace">Query Logger</text>

          {/* Kafka */}
          <rect x={445} y={55} width={75} height={42} rx={8} fill="#05966912" stroke="#059669" strokeWidth={1.5}/>
          <text x={482} y={73} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Kafka</text>
          <text x={482} y={88} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">search_queries</text>

          {/* Flink Aggregator */}
          <rect x={445} y={115} width={75} height={42} rx={8} fill="#d9770612" stroke="#d97706" strokeWidth={1.5}/>
          <text x={482} y={132} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Flink</text>
          <text x={482} y={148} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">aggregator</text>

          {/* Spark Trie Builder */}
          <rect x={445} y={175} width={75} height={42} rx={8} fill="#c026d312" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={482} y={192} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">Trie Builder</text>
          <text x={482} y={207} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">Spark (hourly)</text>

          {/* S3 */}
          <rect x={600} y={55} width={100} height={40} rx={8} fill="#dc262612" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={650} y={72} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">S3</text>
          <text x={650} y={86} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">trie binaries</text>

          {/* Frequency DB */}
          <rect x={600} y={115} width={100} height={40} rx={8} fill="#d9770612" stroke="#d97706" strokeWidth={1.5}/>
          <text x={650} y={132} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Frequency DB</text>
          <text x={650} y={148} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">query → counts</text>

          {/* Monitoring */}
          <rect x={600} y={175} width={100} height={40} rx={8} fill="#0284c712" stroke="#0284c7" strokeWidth={1.5}/>
          <text x={650} y={192} textAnchor="middle" fill="#0284c7" fontSize="9" fontWeight="600" fontFamily="monospace">Monitoring</text>
          <text x={650} y={207} textAnchor="middle" fill="#0284c780" fontSize="7" fontFamily="monospace">Grafana + alerts</text>

          {/* Arrows */}
          <defs><marker id="sa" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker>
          <marker id="sp" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#c026d3"/></marker></defs>
          <line x1={100} y1={75} x2={125} y2={75} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#sa)"/>
          <line x1={195} y1={75} x2={220} y2={75} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#sa)"/>
          <line x1={300} y1={75} x2={325} y2={75} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#sa)"/>
          <line x1={372} y1={110} x2={372} y2={125} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#sa)"/>
          <line x1={372} y1={163} x2={372} y2={178} stroke="#94a3b8" strokeWidth={1} strokeDasharray="4,2" markerEnd="url(#sa)"/>
          <line x1={260} y1={100} x2={260} y2={140} stroke="#94a3b8" strokeWidth={1} strokeDasharray="4,2" markerEnd="url(#sa)"/>
          <text x={270} y={120} fill="#64748b" fontSize="7" fontFamily="monospace">async</text>
          <line x1={300} y1={157} x2={445} y2={76} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#sa)"/>
          <line x1={482} y1={97} x2={482} y2={115} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#sa)"/>
          <line x1={482} y1={157} x2={482} y2={175} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#sa)"/>
          <text x={497} y={168} fill="#64748b" fontSize="7" fontFamily="monospace">hourly</text>
          <line x1={520} y1={75} x2={600} y2={75} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#sa)"/>
          <line x1={520} y1={136} x2={600} y2={136} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#sa)"/>
          {/* Trie builder → trie cluster (deploy) */}
          <path d="M 482 175 Q 482 240 372 240 Q 325 240 325 110" fill="none" stroke="#c026d3" strokeWidth={1.5} strokeDasharray="5,3" markerEnd="url(#sp)"/>
          <text x={430} y={248} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">deploy new trie</text>
          {/* Flink → trending */}
          <line x1={445} y1={136} x2={420} y2={178} stroke="#94a3b8" strokeWidth={1} strokeDasharray="4,2" markerEnd="url(#sa)"/>
          <text x={440} y={158} fill="#64748b" fontSize="6" fontFamily="monospace">trending</text>

          {/* Legend */}
          <rect x={15} y={270} width={690} height={88} rx={8} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={30} y={288} fill="#78716c" fontSize="9" fontWeight="700" fontFamily="monospace">Data Flow Summary</text>
          <text x={30} y={305} fill="#78716c" fontSize="8" fontFamily="monospace">READ: User types → CDN (cache hit?) → API Gateway → Trie Cluster (O(p) lookup) → Content Filter → Response (35ms avg)</text>
          <text x={30} y={320} fill="#78716c" fontSize="8" fontFamily="monospace">WRITE: Search submitted → Query Logger → Kafka → Flink (5-min aggregation) → Spark (hourly trie build) → S3 → Deploy to Trie Cluster</text>
          <text x={30} y={335} fill="#78716c" fontSize="8" fontFamily="monospace">TRENDING: Flink detects frequency spikes → Trending Overlay updated every 5 min → Merged with Trie results at query time</text>
          <text x={30} y={350} fill="#78716c" fontSize="8" fontFamily="monospace">SAFETY: Every response passes through Content Filter (blocklist + ML) before reaching the user. Kill switch available.</text>
        </svg>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#2563eb">
          <Label color="#2563eb">Supporting Services</Label>
          <div className="space-y-3">
            {[
              { name: "Personalization Service", role: "Boosts suggestions based on user's search history, preferences, and context. Reads from user profile store.", tech: "Feature store (Redis) + lightweight re-ranking model", critical: false },
              { name: "A/B Testing Framework", role: "Tests new ranking algorithms, trie configurations, and suggestion counts. Splits traffic by user cohort.", tech: "Feature flags + experiment tracking (LaunchDarkly/internal)", critical: false },
              { name: "Blocklist Management UI", role: "Internal tool for trust & safety team to add/remove blocked terms. Changes propagate to content filter within minutes.", tech: "React admin UI → Redis blocklist → Content Filter sync", critical: true },
              { name: "Query Analytics Service", role: "Powers dashboards showing search trends, popular queries, regional differences, and suggestion click-through rates.", tech: "ClickHouse + Grafana, fed by Kafka consumer", critical: false },
            ].map((s,i) => (
              <div key={i} className="flex items-start gap-3 border border-stone-100 rounded-lg p-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-bold text-stone-700">{s.name}</span>
                    {s.critical && <Pill bg="#fef2f2" color="#dc2626">Critical Path</Pill>}
                  </div>
                  <div className="text-[10px] text-stone-500 mt-0.5">{s.role}</div>
                  <div className="text-[10px] text-stone-400 font-mono mt-0.5">{s.tech}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card accent="#9333ea">
          <Label color="#9333ea">Service-to-Service Contracts</Label>
          <div className="overflow-hidden rounded-lg border border-stone-200">
            <table className="w-full text-[11px]">
              <thead><tr className="bg-stone-50">
                <th className="text-left px-3 py-2 font-semibold text-stone-500">Caller → Callee</th>
                <th className="text-left px-3 py-2 font-semibold text-stone-500">Protocol</th>
                <th className="text-left px-3 py-2 font-semibold text-stone-500">Timeout</th>
                <th className="text-left px-3 py-2 font-semibold text-stone-500">On Failure</th>
              </tr></thead>
              <tbody>
                {[
                  { route: "Gateway → Trie Server", proto: "gRPC", timeout: "50ms", fail: "Return cached/empty result" },
                  { route: "Trie → Content Filter", proto: "gRPC (sidecar)", timeout: "5ms", fail: "Pass through (fail-open)" },
                  { route: "Trie → Trending Overlay", proto: "In-memory", timeout: "N/A", fail: "Skip trending merge" },
                  { route: "Gateway → Query Logger", proto: "Kafka (async)", timeout: "Fire-and-forget", fail: "Drop log event" },
                  { route: "Flink → Frequency DB", proto: "JDBC batch", timeout: "30s", fail: "Retry batch, alert" },
                  { route: "Trie Builder → S3", proto: "AWS SDK", timeout: "60s", fail: "Retry. Trie servers use last good." },
                ].map((r,i) => (
                  <tr key={i} className={i%2?"bg-stone-50/50":""}>
                    <td className="px-3 py-2 font-mono text-teal-700 font-medium">{r.route}</td>
                    <td className="px-3 py-2 text-stone-500">{r.proto}</td>
                    <td className="px-3 py-2 font-mono text-stone-400">{r.timeout}</td>
                    <td className="px-3 py-2 text-stone-400">{r.fail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}

function FlowsSection() {
  const [flow, setFlow] = useState("happy");
  const flows = {
    happy: {
      title: "Happy Path — Suggestion Served",
      steps: [
        { actor: "User", action: 'Types "how t" in search box. Client debounces 150ms. No more keystrokes → fire API call.', type: "request" },
        { actor: "Browser", action: 'Check local cache for "how t" → MISS. Send GET /v1/suggestions?prefix=how+t&limit=5', type: "process" },
        { actor: "CDN", action: 'Cache key = "how+t:en:us". MISS (4-char prefix, not cached). Forward to origin.', type: "process" },
        { actor: "API Gateway", action: 'Parse prefix "how t". First char = "h" → route to Shard 3 (g-m range).', type: "auth" },
        { actor: "Trie Server (Shard 3)", action: 'Walk trie: root→h→o→w→" "→t. Read precomputed top-5 from node. 0.8ms.', type: "success" },
        { actor: "Content Filter", action: 'Check 5 suggestions against blocklist (Redis, 0.3ms) + ML classifier (1.2ms). All pass.', type: "check" },
        { actor: "Trending Overlay", action: 'Check trending map for "how t" prefix matches. No active trending match.', type: "process" },
        { actor: "API Gateway", action: 'Return JSON response. Total latency: 35ms. Set Cache-Control: max-age=60.', type: "success" },
        { actor: "Browser", action: 'Render dropdown: ["how to tie a tie", "how to screenshot", ...]. Cache result locally.', type: "success" },
        { actor: "Query Logger", action: 'No log yet — user hasn\'t submitted search. Log only fires on Enter/click.', type: "process" },
      ]
    },
    trending: {
      title: "Trending — Breaking News Boost",
      steps: [
        { actor: "Event", action: 'Major earthquake hits California. Millions start searching "earthquake" simultaneously.', type: "request" },
        { actor: "Flink Aggregator", action: '"earthquake california" count in 5-min window: 50,000. Baseline (7d avg): 200/5min. Spike ratio: 250×.', type: "check" },
        { actor: "Trending Overlay", action: 'Spike ≥ 5× threshold → Mark as trending. Score = 250 × 50000 = 12.5M. Added to trending map.', type: "success" },
        { actor: "User", action: 'Types "ear" in search box. Trie returns: ["earbuds", "ear infection", "early voting", ...].', type: "request" },
        { actor: "Trie Server", action: 'Trie lookup returns standard top-5. Score for "earbuds" = 543,210.', type: "process" },
        { actor: "Trending Merger", action: 'Trending match: "earthquake california" (score 12.5M). Merges into results, ranks #1.', type: "success" },
        { actor: "Response", action: '["earthquake california", "earbuds", "ear infection", "early voting", "ear wax removal"]', type: "success" },
        { actor: "Decay", action: 'After 2 hours, earthquake searches decline. Trending score decays. Eventually falls out of overlay.', type: "process" },
      ]
    },
    blocked: {
      title: "Content Filter — Blocked Suggestion",
      steps: [
        { actor: "User", action: 'Types "how to k" in search box.', type: "request" },
        { actor: "Trie Server", action: 'Trie returns top-5 including a potentially harmful suggestion at position 3.', type: "process" },
        { actor: "Content Filter (Blocklist)", action: 'Check suggestion #3 against blocklist of 500K+ terms → MATCH. Blocked.', type: "error" },
        { actor: "Content Filter", action: 'Remove blocked suggestion. Pull next candidate from trie (position 6) to fill the slot.', type: "process" },
        { actor: "Response", action: 'Return 5 clean suggestions. User never sees the blocked term. Audit log entry created.', type: "success" },
        { actor: "Monitoring", action: 'Increment blocked_suggestion_total{reason="blocklist"}. Dashboard updated.', type: "process" },
      ]
    },
    stale: {
      title: "Trie Build Failure — Stale Serving",
      steps: [
        { actor: "Trie Builder (Spark)", action: 'Hourly trie build starts. Reads frequency table. Spark job fails: OOM on executor.', type: "error" },
        { actor: "Alert", action: 'trie_build_failed alert fires. P2 page to data engineering on-call.', type: "error" },
        { actor: "Trie Deployer", action: 'No new trie binary uploaded to S3. Trie servers not notified.', type: "process" },
        { actor: "Trie Servers", action: 'Continue serving the last successfully built trie (from 1 hour ago). No user impact.', type: "success" },
        { actor: "Trending Overlay", action: 'Still updating every 5 minutes independently. Provides freshness despite stale trie.', type: "success" },
        { actor: "On-Call Engineer", action: 'Investigates Spark OOM. Increases executor memory. Manually triggers rebuild. Succeeds.', type: "process" },
        { actor: "Trie Servers", action: 'New trie deployed via blue-green swap. Canary passes. All replicas updated. Total gap: ~2 hours of stale trie.', type: "success" },
      ]
    },
  };
  const f = flows[flow];
  const typeColors = { request:"bg-blue-50 text-blue-700 border-blue-200", success:"bg-emerald-50 text-emerald-700 border-emerald-200", error:"bg-red-50 text-red-700 border-red-200", process:"bg-stone-50 text-stone-600 border-stone-200", auth:"bg-violet-50 text-violet-700 border-violet-200", check:"bg-amber-50 text-amber-700 border-amber-200" };
  return (
    <div className="space-y-5">
      <Card accent="#7e22ce">
        <div className="flex gap-2 mb-4 flex-wrap">
          {Object.entries(flows).map(([k,v]) => (
            <button key={k} onClick={() => setFlow(k)}
              className={`px-3 py-1.5 rounded-lg text-[12px] font-medium border transition-all ${k===flow?"bg-purple-600 text-white border-purple-600":"bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
              {v.title.split("—")[0].trim()}
            </button>
          ))}
        </div>
        <Label color="#7e22ce">{f.title}</Label>
        <div className="space-y-1.5 mt-3">
          {f.steps.map((s,i) => (
            <div key={i} className={`flex items-start gap-3 px-4 py-2.5 rounded-lg border ${typeColors[s.type]}`}>
              <span className="text-[10px] font-mono font-bold w-5 shrink-0 mt-0.5">{i+1}</span>
              <span className="text-[11px] font-bold w-36 shrink-0">{s.actor}</span>
              <span className="text-[11px] flex-1">{s.action}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function DeploymentSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#b45309">
          <Label color="#b45309">Kubernetes Deployment — Trie Serving Cluster</Label>
          <CodeBlock title="Trie servers: stateful (mmap'd trie), but horizontally scalable" code={`apiVersion: apps/v1
kind: Deployment
metadata:
  name: trie-server-shard-0   # one per shard (0-9)
spec:
  replicas: 3                  # 3 replicas per shard
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0        # zero-downtime for blue-green
  template:
    spec:
      containers:
      - name: trie-server
        image: autocomplete/trie:2.1
        env:
        - name: SHARD_ID
          value: "0"
        - name: TRIE_S3_PATH
          value: "s3://tries/prod/shard-0/latest.bin"
        - name: CONTENT_FILTER_HOST
          value: "content-filter.svc:50051"
        resources:
          requests:
            cpu: "2"
            memory: "12Gi"     # trie (8GB) + overhead
          limits:
            cpu: "4"
            memory: "16Gi"
        readinessProbe:
          httpGet:
            path: /healthz
            port: 8080
          periodSeconds: 5
        startupProbe:
          httpGet:
            path: /ready       # wait for trie to load
            port: 8080
          failureThreshold: 30 # trie load takes ~60s
          periodSeconds: 5
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone`} />
        </Card>
        <Card accent="#0f766e">
          <Label color="#0f766e">Multi-AZ Deployment Layout</Label>
          <svg viewBox="0 0 380 310" className="w-full">
            <rect x={5} y={5} width={370} height={300} rx={10} fill="#0f766e05" stroke="#0f766e30" strokeWidth={1} strokeDasharray="4,3"/>
            <text x={190} y={22} textAnchor="middle" fill="#0f766e" fontSize="10" fontWeight="700" fontFamily="monospace">us-east-1 (Primary Region)</text>

            <rect x={110} y={32} width={160} height={22} rx={4} fill="#2563eb12" stroke="#2563eb" strokeWidth={1}/>
            <text x={190} y={47} textAnchor="middle" fill="#2563eb" fontSize="8" fontWeight="600" fontFamily="monospace">CloudFront CDN (Global Edge)</text>

            {/* AZ-a */}
            <rect x={15} y={65} width={110} height={135} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={70} y={80} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-a</text>
            <rect x={22} y={88} width={96} height={22} rx={4} fill="#6366f115" stroke="#6366f1" strokeWidth={1}/>
            <text x={70} y={103} textAnchor="middle" fill="#6366f1" fontSize="7" fontFamily="monospace">API Gateway ×2</text>
            <rect x={22} y={116} width={96} height={22} rx={4} fill="#9333ea15" stroke="#9333ea" strokeWidth={1}/>
            <text x={70} y={131} textAnchor="middle" fill="#9333ea" fontSize="7" fontFamily="monospace">Trie Shard 0-3</text>
            <rect x={22} y={144} width={96} height={22} rx={4} fill="#dc262615" stroke="#dc2626" strokeWidth={1}/>
            <text x={70} y={159} textAnchor="middle" fill="#dc2626" fontSize="7" fontFamily="monospace">Content Filter ×2</text>
            <rect x={22} y={172} width={96} height={22} rx={4} fill="#05966915" stroke="#059669" strokeWidth={1}/>
            <text x={70} y={187} textAnchor="middle" fill="#059669" fontSize="7" fontFamily="monospace">Kafka Broker ×1</text>

            {/* AZ-b */}
            <rect x={135} y={65} width={110} height={135} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={190} y={80} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-b</text>
            <rect x={142} y={88} width={96} height={22} rx={4} fill="#6366f115" stroke="#6366f1" strokeWidth={1}/>
            <text x={190} y={103} textAnchor="middle" fill="#6366f1" fontSize="7" fontFamily="monospace">API Gateway ×2</text>
            <rect x={142} y={116} width={96} height={22} rx={4} fill="#9333ea15" stroke="#9333ea" strokeWidth={1}/>
            <text x={190} y={131} textAnchor="middle" fill="#9333ea" fontSize="7" fontFamily="monospace">Trie Shard 4-6</text>
            <rect x={142} y={144} width={96} height={22} rx={4} fill="#dc262615" stroke="#dc2626" strokeWidth={1}/>
            <text x={190} y={159} textAnchor="middle" fill="#dc2626" fontSize="7" fontFamily="monospace">Content Filter ×2</text>
            <rect x={142} y={172} width={96} height={22} rx={4} fill="#05966915" stroke="#059669" strokeWidth={1}/>
            <text x={190} y={187} textAnchor="middle" fill="#059669" fontSize="7" fontFamily="monospace">Kafka Broker ×1</text>

            {/* AZ-c */}
            <rect x={255} y={65} width={110} height={135} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={310} y={80} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-c</text>
            <rect x={262} y={88} width={96} height={22} rx={4} fill="#6366f115" stroke="#6366f1" strokeWidth={1}/>
            <text x={310} y={103} textAnchor="middle" fill="#6366f1" fontSize="7" fontFamily="monospace">API Gateway ×2</text>
            <rect x={262} y={116} width={96} height={22} rx={4} fill="#9333ea15" stroke="#9333ea" strokeWidth={1}/>
            <text x={310} y={131} textAnchor="middle" fill="#9333ea" fontSize="7" fontFamily="monospace">Trie Shard 7-9</text>
            <rect x={262} y={144} width={96} height={22} rx={4} fill="#dc262615" stroke="#dc2626" strokeWidth={1}/>
            <text x={310} y={159} textAnchor="middle" fill="#dc2626" fontSize="7" fontFamily="monospace">Content Filter ×2</text>
            <rect x={262} y={172} width={96} height={22} rx={4} fill="#05966915" stroke="#059669" strokeWidth={1}/>
            <text x={310} y={187} textAnchor="middle" fill="#059669" fontSize="7" fontFamily="monospace">Kafka Broker ×1</text>

            {/* Legend */}
            <rect x={15} y={210} width={350} height={88} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
            <text x={190} y={226} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Deployment Rules</text>
            <text x={30} y={244} fill="#78716c" fontSize="8" fontFamily="monospace">• Trie shards spread across AZs. Each shard has 3 replicas.</text>
            <text x={30} y={259} fill="#78716c" fontSize="8" fontFamily="monospace">• Content filter co-located as sidecar (sub-ms latency).</text>
            <text x={30} y={274} fill="#78716c" fontSize="8" fontFamily="monospace">• Trie Builder runs on EMR Spark (on-demand, hourly).</text>
            <text x={30} y={289} fill="#78716c" fontSize="8" fontFamily="monospace">• AZ failure: other AZs have full shard coverage. No downtime.</text>
          </svg>
        </Card>
      </div>
      <Card accent="#dc2626">
        <Label color="#dc2626">Security Considerations</Label>
        <div className="grid grid-cols-2 gap-5">
          <div>
            <div className="text-[10px] font-bold text-stone-600 uppercase tracking-wider mb-1.5">API Security</div>
            <ul className="space-y-1.5">
              <Point icon="🛡️" color="#dc2626">Rate limit autocomplete API per user/IP — prevent scraping of suggestion corpus</Point>
              <Point icon="🛡️" color="#dc2626">Input sanitization — prevent injection via prefix (XSS in suggestions display)</Point>
              <Point icon="🛡️" color="#dc2626">No PII in suggestions — never surface other users' personal queries in global suggestions</Point>
              <Point icon="🛡️" color="#dc2626">HTTPS only — suggestions can reveal sensitive search intent (health, legal, financial)</Point>
            </ul>
          </div>
          <div>
            <div className="text-[10px] font-bold text-stone-600 uppercase tracking-wider mb-1.5">Data Privacy</div>
            <ul className="space-y-1.5">
              <Point icon="🔒" color="#b45309">Query logs anonymized before aggregation — strip user_id, hash IP</Point>
              <Point icon="🔒" color="#b45309">Minimum frequency threshold (100+ unique users) before entering suggestion corpus</Point>
              <Point icon="🔒" color="#b45309">GDPR right-to-erasure: user can request removal of their search history from logs</Point>
              <Point icon="🔒" color="#b45309">Personalized suggestions stored client-side only — server never stores per-user trie</Point>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}

function OpsSection() {
  return (
    <div className="space-y-5">
      <Card accent="#be123c">
        <Label color="#be123c">Runbook — Common Incidents</Label>
        <div className="space-y-3">
          {[
            { incident: "Suggestion latency spike (p99 > 200ms)", steps: "1. Check trie server CPU/memory (is it GC'ing?). 2. Check CDN hit rate (did it drop?). 3. Look for hot-shard (one prefix range overloaded). 4. Check content filter latency. 5. Scale up replicas for affected shard.", severity: "P1" },
            { incident: "Offensive suggestion reported by user / press", steps: "1. Add term to emergency blocklist IMMEDIATELY (< 5 min). 2. Verify content filter picks up new blocklist entry. 3. Clear CDN cache for affected prefix range. 4. Investigate why filter missed it (ML gap? New term?). 5. Post-mortem and retrain classifier.", severity: "P1" },
            { incident: "Trie build failing repeatedly", steps: "1. Check Spark job logs for OOM, bad data, or schema mismatch. 2. Verify frequency table has fresh data (Flink pipeline healthy?). 3. Try building a smaller test trie. 4. Trie servers continue serving last good trie — no immediate user impact. 5. Fix pipeline, manually trigger rebuild.", severity: "P2" },
            { incident: "Empty suggestions for common prefixes", steps: "1. Check if affected shard is healthy. 2. Verify trie binary integrity (checksum match). 3. Check if recent trie deploy had a regression — rollback. 4. Query trie directly to isolate: is data missing or is routing broken? 5. If trie is corrupt, redeploy from S3 backup.", severity: "P1" },
            { incident: "Trending suggestions not appearing", steps: "1. Check Flink job health (is it consuming from Kafka?). 2. Verify trending overlay is being updated (last update timestamp). 3. Check spike detection thresholds (is threshold too high?). 4. Manually inject trending term if urgent. 5. Fix pipeline, verify trending recovers.", severity: "P3" },
          ].map((r,i) => (
            <div key={i} className="border border-stone-200 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Pill bg={r.severity==="P1"?"#fef2f2":"#fffbeb"} color={r.severity==="P1"?"#dc2626":"#d97706"}>{r.severity}</Pill>
                <span className="text-[12px] font-bold text-stone-800">{r.incident}</span>
              </div>
              <div className="text-[11px] text-stone-500 leading-relaxed">{r.steps}</div>
            </div>
          ))}
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#be123c">Health Checks</Label>
          <div className="space-y-2">
            {[
              { check: "Trie server /healthz", interval: "5s", action: "Remove from LB if unhealthy" },
              { check: "Trie loaded and ready", interval: "On startup", action: "Block traffic until trie is mmap'd" },
              { check: "Content filter ping", interval: "10s", action: "Fail-open if filter unreachable" },
              { check: "CDN origin health", interval: "30s", action: "Route to backup origin" },
              { check: "Kafka consumer lag", interval: "1m", action: "Alert if lag > 5M events" },
              { check: "Trie freshness (age)", interval: "5m", action: "Alert if trie > 3 hours old" },
            ].map((h,i) => (
              <div key={i} className="flex items-center gap-3 text-[11px] border-b border-stone-100 pb-2">
                <span className="font-mono text-stone-700 w-44 shrink-0">{h.check}</span>
                <span className="text-stone-400 w-16 shrink-0">{h.interval}</span>
                <span className="text-stone-500 flex-1">{h.action}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <Label color="#be123c">Capacity Planning</Label>
          <ul className="space-y-2">
            <Point icon="📊" color="#be123c">QPS per shard = total QPS / num_shards / num_replicas. Add replicas before approaching 100K QPS/server.</Point>
            <Point icon="💾" color="#be123c">Trie size grows with query corpus. Monitor monthly. Resize shards when trie exceeds 80% of server RAM.</Point>
            <Point icon="🌐" color="#be123c">CDN cache hit rate should be &gt;30%. If below, extend TTL for top-level prefixes or add more edge caching rules.</Point>
            <Point icon="📈" color="#be123c">Kafka: monitor partition count. Each Flink consumer needs ≥1 partition. Scale partitions for parallelism.</Point>
            <Point icon="⏱️" color="#be123c">Trie build time should stay under 30 min. If growing, increase Spark executor count or optimize trie serialization.</Point>
          </ul>
        </Card>
      </div>
    </div>
  );
}

function EnhancementsSection() {
  return (
    <div className="grid grid-cols-2 gap-5">
      {[
        { t: "Spell Correction in Autocomplete", d: "User types 'pythn' → suggest 'python tutorial'. Catch misspellings before they reach the trie.", detail: "Use edit-distance (Levenshtein ≤ 2) or a BK-Tree alongside the trie. Alternatively, maintain a 'did you mean' mapping from common misspellings to correct terms. Adds 5-10ms latency.", effort: "Medium" },
        { t: "Personalized Suggestions", d: "Rank suggestions based on user's search history, click behavior, and context (time, device, location).", detail: "Store user's recent queries in a lightweight feature store (Redis). At query time, re-rank trie results by boosting terms the user has searched before. All personalization data stays ephemeral — no long-term per-user trie.", effort: "Medium" },
        { t: "Multi-Entity Autocomplete", d: "Like Spotify: suggest songs, artists, and playlists in one dropdown. Or Amazon: products, categories, and brands.", detail: "Run parallel lookups against entity-specific tries (product trie, category trie, brand trie). Merge and interleave results. Tag each suggestion with its entity type for UI rendering.", effort: "Hard" },
        { t: "Query Completion vs Query Suggestion", d: "Completion: finish what the user is typing ('how to t' → 'how to tie a tie'). Suggestion: related queries ('python' → 'python tutorial', 'python download').", detail: "Completion uses prefix matching (trie). Suggestions use semantic similarity (embeddings). Combine both in the dropdown with visual distinction. Suggestion requires ML model inference.", effort: "Hard" },
        { t: "Client-Side Prediction", d: "Cache trie subtrees on the client. After fetching results for 'ho', the client can predict results for 'how' locally without a network call.", detail: "Server returns extra data: top suggestions for the next 2-3 characters. Client stores in a local trie. Eliminates 30-50% of network calls. Trade-off: larger response payload (~2-3KB vs 500B).", effort: "Easy" },
        { t: "Voice Search Autocomplete", d: "As speech-to-text streams partial results, show autocomplete suggestions in real-time based on the partial transcript.", detail: "Integrate with speech-to-text streaming API. As partial transcript updates (every 200-300ms), query autocomplete. Challenge: partial transcripts are noisy — need fuzzy prefix matching or phonetic matching.", effort: "Hard" },
      ].map((e,i) => (
        <Card key={i}>
          <div className="flex items-center justify-between mb-2">
            <div className="text-[11px] font-bold text-stone-800">{e.t}</div>
            <Pill bg={e.effort==="Easy"?"#ecfdf5":e.effort==="Medium"?"#fffbeb":"#fef2f2"} color={e.effort==="Easy"?"#059669":e.effort==="Medium"?"#d97706":"#dc2626"}>{e.effort}</Pill>
          </div>
          <p className="text-[12px] text-stone-600 mb-1.5">{e.d}</p>
          <p className="text-[11px] text-stone-400">{e.detail}</p>
        </Card>
      ))}
    </div>
  );
}

function FollowupsSection() {
  const [exp, setExp] = useState(null);
  const qas = [
    { q:"How does the trie handle queries in different languages?", a:"Build separate tries per language. Route by Accept-Language header or explicit locale parameter. CJK languages (Chinese, Japanese, Korean) need special tokenization — character-level trie works, but word-level segmentation gives better suggestions. For mixed-language contexts (English terms in Japanese search), consider a hybrid trie with cross-language prefix matching.", tags:["design"] },
    { q:"Why not use Elasticsearch for autocomplete?", a:"ES has a completion suggester that uses an in-memory FST (Finite State Transducer). It works well for moderate scale (millions of terms). But at Google/Amazon scale (100M+ terms, millions QPS), a custom trie gives you: (1) full control over memory layout, (2) sub-ms lookup guaranteed, (3) exact top-k precomputation (ES does approximate scoring), (4) custom sharding strategy. For smaller systems, ES is perfectly fine.", tags:["design"] },
    { q:"How fresh do suggestions need to be?", a:"Three tiers: (1) Main trie: hourly rebuild is fine for stable popular queries. (2) Trending overlay: 5-minute update cycle for breaking news. (3) Emergency blocklist: minutes — push-based update via Redis pub/sub to all servers. Most users won't notice if a new slang term takes an hour to appear, but trending events need minutes.", tags:["freshness"] },
    { q:"How do you handle the 'celebrity problem' — one trending term floods all prefixes?", a:"When a trending term like 'super bowl' dominates, it appears in suggestions for 's', 'su', 'sup', etc. — potentially replacing useful diverse suggestions. Solution: cap trending boost so it can promote to top-1 but not monopolize all 5 slots. Also, diversify results: ensure at least 3 of 5 suggestions come from different categories.", tags:["algorithm"] },
    { q:"What if the user types very fast — do we make a call per keystroke?", a:"No — client-side optimizations: (1) Debounce: wait 100-150ms after last keystroke before calling API. (2) Cancel: if a new keystroke arrives, cancel the in-flight request. (3) Local cache: if we have results for 'ho', we can filter them client-side for 'how' without a network call. (4) Predictive prefetch: after getting results for 'ho', preload results for 'how', 'hot', 'hou'.", tags:["client"] },
    { q:"How do you A/B test changes to autocomplete?", a:"Split users into cohorts (not requests — same user should see consistent behavior). Metrics to compare: (1) Suggestion click-through rate (CTR), (2) Search success rate after using suggestion, (3) Average keystrokes saved, (4) Time to first search. Run for 1-2 weeks. Watch for novelty effects — users may click more on new suggestions just because they're different.", tags:["testing"] },
    { q:"What's the relationship between autocomplete and search ranking?", a:"They're independent systems. Autocomplete suggests popular QUERIES; search ranking orders RESULTS for a query. But they share data: search click logs feed both. A common pitfall: optimizing autocomplete for popular queries but not verifying that those queries produce good search results. Monitor the 'suggestion-to-result' quality pipeline.", tags:["design"] },
    { q:"Can autocomplete be used for abuse (phishing, manipulation)?", a:"Yes — attackers can try to: (1) manipulate trending by botting searches, (2) use autocomplete to spread misinformation ('Company X is scam'), (3) phish by making malicious URLs appear in suggestions. Defenses: unique-user thresholds (not just raw count), anomaly detection on search patterns, content filter with legal/defamation category, human review for promoted terms.", tags:["security"] },
    { q:"How would you measure the quality of autocomplete?", a:"Key metrics: (1) Suggestion acceptance rate: % of times user clicks a suggestion vs typing full query. (2) Position bias: which slot gets clicked most (measure if ranking is effective). (3) Time saved: keystrokes saved = query length - prefix length at suggestion click. (4) Null rate: % of prefixes with no suggestions. (5) Offensive suggestion incident rate. (6) User satisfaction surveys.", tags:["metrics"] },
    { q:"How does Google's autocomplete work specifically?", a:"Google uses a combination of: (1) Trie-based prefix matching for fast lookup, (2) Real-time trending from Google Trends, (3) Personalization from your search history, (4) ML-based ranking that considers freshness, popularity, language model probability, and user context, (5) Multi-layer content filtering (automated + human review), (6) Geo-localized suggestions (US vs India see different results for same prefix).", tags:["design"] },
  ];
  return (
    <div className="space-y-2.5">
      <Card className="bg-indigo-50/50 border-indigo-200 mb-1">
        <p className="text-[12px] text-stone-500">Common follow-up questions interviewers ask after the initial design. Click to reveal a strong answer.</p>
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
  api: ApiSection, design: DesignSection, algorithm: AlgorithmSection, data: DataModelSection,
  scalability: ScalabilitySection, availability: AvailabilitySection, observability: ObservabilitySection,
  watchouts: WatchoutsSection, services: ServicesSection, flows: FlowsSection,
  deployment: DeploymentSection, ops: OpsSection,
  enhancements: EnhancementsSection, followups: FollowupsSection,
};

export default function SearchAutocompleteSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Search Autocomplete</h1>
            <Pill bg="#f3e8ff" color="#7c3aed">System Design</Pill>
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