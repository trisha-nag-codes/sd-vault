import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   DISTRIBUTED CACHE — System Design Reference
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
  { id: "services",      label: "Service Architecture",  icon: "🧩", color: "#0f766e" },
  { id: "flows",         label: "Request Flows",         icon: "🔀", color: "#7e22ce" },
  { id: "deployment",    label: "Deploy & Security",     icon: "🔒", color: "#b45309" },
  { id: "ops",           label: "Ops Playbook",          icon: "🔧", color: "#be123c" },
  { id: "enhancements",  label: "Enhancements",         icon: "🚀", color: "#7c3aed" },
  { id: "followups",     label: "Follow-up Questions",  icon: "❓", color: "#6366f1" },
];

/* ─── Reusable Components ─── */
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

function CodeBlock({ title, code, highlight = [] }) {
  const lines = code.split("\n");
  return (
    <div className="bg-stone-50 border border-stone-200 rounded-lg p-3.5 overflow-x-auto">
      {title && <div className="text-[10px] font-bold text-stone-400 uppercase tracking-[0.1em] mb-2">{title}</div>}
      <pre className="font-mono text-[11.5px] leading-[1.75]" style={{ whiteSpace: "pre" }}>
        {lines.map((line, i) => (
          <div key={i} className={`px-2 rounded ${highlight.includes(i) ? "bg-indigo-50 text-indigo-700" : line.trim().startsWith("#") || line.trim().startsWith("--") || line.trim().startsWith("//") ? "text-stone-400" : "text-stone-700"}`}>
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
            <Label>What is a Distributed Cache?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A distributed cache is a system that stores frequently accessed data in memory across multiple nodes, sitting between your application servers and the database. It dramatically reduces read latency (from ~10ms DB query to ~1ms cache hit) and offloads traffic from the database.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Think of it like a library's reference desk: instead of walking to the stacks every time (database), you keep the most popular books on the desk (cache) for instant access. When a book isn't there (cache miss), you fetch it from the stacks and put a copy on the desk.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Do We Need It?</Label>
            <ul className="space-y-2.5">
              <Point icon="⚡" color="#0891b2">Latency reduction — memory access (~1ms) vs disk/DB (~10-100ms), 10-100× faster reads</Point>
              <Point icon="🛡️" color="#0891b2">Database protection — absorb read traffic so DB handles only cache misses and writes</Point>
              <Point icon="📈" color="#0891b2">Throughput scaling — scale reads horizontally by adding cache nodes, independent of DB</Point>
              <Point icon="💰" color="#0891b2">Cost efficiency — memory is expensive per GB but cheap per read; avoids DB scaling for read-heavy loads</Point>
              <Point icon="🔄" color="#0891b2">Computed result caching — cache expensive computations (aggregations, ML inference, rendered pages)</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Examples</Label>
            <div className="space-y-2.5">
              {[
                { co: "Memcached", rule: "Simple key-value, multi-threaded", algo: "LRU slab allocator" },
                { co: "Redis", rule: "Rich data types, persistence, Lua", algo: "LRU / LFU / TTL" },
                { co: "Facebook", rule: "Memcached at 1B+ QPS globally", algo: "Custom (mcrouter)" },
                { co: "Twitter", rule: "Redis for timeline, counts", algo: "Hybrid LRU + TTL" },
                { co: "CDN Edge", rule: "Varnish/CloudFront at L7", algo: "TTL-based" },
              ].map((e,i) => (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-24 shrink-0">{e.co}</span>
                  <span className="text-stone-500 flex-1">{e.rule}</span>
                  <span className="text-stone-400 text-[10px]">{e.algo}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <Label color="#2563eb">Where Does It Live?</Label>
            <svg viewBox="0 0 360 190" className="w-full">
              <DiagramBox x={50} y={55} w={70} h={38} label="Client" color="#2563eb"/>
              <DiagramBox x={170} y={55} w={80} h={38} label="App\nServer" color="#9333ea"/>
              <DiagramBox x={300} y={30} w={80} h={34} label="Cache" color="#d97706"/>
              <DiagramBox x={300} y={85} w={80} h={34} label="Database" color="#059669"/>
              <Arrow x1={85} y1={55} x2={130} y2={55} id="c1"/>
              <Arrow x1={210} y1={45} x2={260} y2={35} label="1. check" id="c2"/>
              <Arrow x1={210} y1={65} x2={260} y2={80} label="2. miss" id="c3" dashed/>
              <Arrow x1={300} y1={57} x2={300} y2={68} label="" id="c4" dashed/>
              <rect x={110} y={125} width={210} height={22} rx={4} fill="#d9770608" stroke="#d9770630"/>
              <text x={215} y={137} textAnchor="middle" fill="#d97706" fontSize="8" fontFamily="monospace">cache-aside (most common pattern)</text>
              <rect x={110} y={155} width={210} height={22} rx={4} fill="#05966908" stroke="#05966930"/>
              <text x={215} y={167} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">hit ratio 95%+ = DB gets only 5% of reads</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Google, Meta, Amazon, Databricks, Stripe, Netflix</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Scope It Fast</div>
            <p className="text-[12px] text-stone-500 mt-0.5">There are many types of caches (browser, CDN, application, database). Clarify immediately: are we designing the cache <em>service</em> itself (like Redis/Memcached), or the caching <em>layer</em> in a larger system? This changes everything — the former is infrastructure design, the latter is integration design.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">GET(key) → value or null — sub-millisecond reads from memory</Point>
            <Point icon="2." color="#059669">SET(key, value, TTL) — write with optional time-to-live expiration</Point>
            <Point icon="3." color="#059669">DELETE(key) — explicit invalidation for cache consistency</Point>
            <Point icon="4." color="#059669">Support variable-size values (1 byte to ~1MB per entry)</Point>
            <Point icon="5." color="#059669">Automatic eviction when memory is full (LRU, LFU, or TTL-based)</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Low latency — p99 read latency &lt;5ms (p50 &lt;1ms)</Point>
            <Point icon="2." color="#dc2626">High throughput — 100K+ ops/sec per node</Point>
            <Point icon="3." color="#dc2626">Horizontal scalability — add nodes to scale linearly</Point>
            <Point icon="4." color="#dc2626">High availability — no single point of failure, graceful degradation</Point>
            <Point icon="5." color="#dc2626">Eventual consistency is acceptable — stale reads OK for short duration</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask the Interviewer</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "What's the read-to-write ratio? (typically 80:20 to 99:1)",
            "What data are we caching? User profiles? Feed? Sessions?",
            "What's the acceptable staleness? Seconds? Minutes?",
            "Do we need persistence or is pure in-memory OK?",
            "Single data center or multi-region?",
            "Expected scale? QPS, data size, number of keys?",
            "Cache-aside or do we need write-through/write-back?",
            "Are there hot keys? (celebrity profiles, trending posts)",
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
            <p className="text-[12px] text-stone-500 mt-0.5">Walk through estimation out loud. Round aggressively — interviewers care about your process and order-of-magnitude reasoning, not exact numbers. State assumptions clearly: <em>"Let me assume we're caching user profiles for a social media platform..."</em></p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Traffic Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="DAU = 50M users" result="50M" note='Assumption — large social platform (ask interviewer)' />
            <MathStep step="2" formula="Reads per user per day = ~200" result="200" note="Feed loads, profile views, API calls. Read-heavy workload." />
            <MathStep step="3" formula="Total reads/day = 50M × 200" result="10 Billion" note="10 × 10⁹ reads per day" />
            <MathStep step="4" formula="Avg read QPS = 10B / 86,400 ≈ 10B / 100K" result="~115K QPS" note="Reads only. Writes are much lower." final />
            <MathStep step="5" formula="Peak read QPS = Avg × 3" result="~350K QPS" note="Peak multiplier 2-3×. Use 3× to be safe." final />
            <MathStep step="6" formula="Write QPS = Read QPS × 0.05 (5% write ratio)" result="~6K QPS" note="95:5 read-to-write ratio is typical for cache" />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Storage Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Average cached value size" result="~1 KB" note='User profile JSON, session data, serialized object' />
            <MathStep step="2" formula="Key size (with overhead)" result="~100 B" note="Key string + hash table pointer + metadata" />
            <MathStep step="3" formula="Total per entry = key + value + overhead" result="~1.2 KB" note="Redis/Memcached add ~200 bytes of internal overhead per key" />
            <MathStep step="4" formula="Unique cacheable entities = 50M (all DAU)" result="50M keys" note="One entry per active user. Could be 200M for all entities." />
            <MathStep step="5" formula="Total memory = 50M × 1.2 KB" result="~60 GB" note="5 × 10⁷ × 1.2 × 10³ = 6 × 10¹⁰ bytes" final />
            <MathStep step="6" formula="With 1.5× headroom (fragmentation)" result="~90 GB" note="Need multiple nodes. Typical Redis node: 25-64 GB usable." final />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Cluster Sizing</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Single node throughput" result="~100K ops/s" note="Redis benchmark for GET/SET operations" />
            <MathStep step="2" formula="Nodes for throughput = 350K / 100K" result="4 shards" note="For peak QPS. Each shard handles a key partition." final />
            <MathStep step="3" formula="Nodes for memory = 90 GB / 25 GB" result="4 shards" note="25 GB usable per node (r6g.xlarge). Coincidence!" final />
            <MathStep step="4" formula="With replicas (1 per shard) = 4 × 2" result="8 nodes" note="4 primaries + 4 replicas for HA" />
            <MathStep step="5" formula="Cache hit ratio target" result="95%+" note="At 95%, DB gets only 5% of reads = ~17K QPS to DB" />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Cost Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Node = r6g.xlarge (26 GB, 4 vCPU)" result="~$0.33/hr" note="AWS ElastiCache on-demand pricing" />
            <MathStep step="2" formula="8 nodes × $0.33/hr × 730 hrs/month" result="~$1,930/mo" note="On-demand pricing." final />
            <MathStep step="3" formula="Reserved pricing (1yr commitment)" result="~$1,150/mo" note="~40% savings with reserved instances" />
            <MathStep step="4" formula="Cost per 1M cache reads" result="~$0.004" note="Compare: RDS read at ~$0.05/1M — cache is 12× cheaper per read" />
          </div>
          <div className="mt-4 pt-3 border-t border-stone-100">
            <div className="text-[11px] text-stone-500">
              <strong className="text-stone-700">Key point:</strong> Without the cache, you'd need a much larger DB cluster to handle 350K reads/sec (~$15K+/mo). The cache pays for itself 10× over by offloading reads. This is the core economic argument for caching.
            </div>
          </div>
        </Card>
      </div>
      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Peak Read QPS", val: "~350K", sub: "Avg: ~115K" },
            { label: "Total Memory", val: "~90 GB", sub: "50M keys × 1.2KB" },
            { label: "Cluster Size", val: "4 shards", sub: "8 nodes with replicas" },
            { label: "Monthly Cost", val: "~$1,150", sub: "Saves ~$14K vs DB-only" },
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
          <Label color="#2563eb">Core Interface</Label>
          <CodeBlock code={`# Cache-Aside Pattern (most common)
def get_user(user_id):
    key = f"user:{user_id}"

    # 1. Check cache first
    cached = cache.get(key)
    if cached is not None:
        return deserialize(cached)  # cache HIT

    # 2. Cache miss → fetch from DB
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    if user is None:
        return None

    # 3. Populate cache for next time
    cache.set(key, serialize(user), ttl=300)  # 5 min TTL
    return user

def update_user(user_id, data):
    # 1. Update DB first (source of truth)
    db.update("UPDATE users SET ... WHERE id = ?", data, user_id)

    # 2. Invalidate cache (NOT update — avoids race conditions)
    cache.delete(f"user:{user_id}")`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Cache Operations</Label>
          <div className="space-y-3">
            {[
              { op: "GET(key)", desc: "Retrieve value by key. Returns null on miss.", perf: "O(1) — hash lookup" },
              { op: "SET(key, val, ttl)", desc: "Store key-value pair with optional TTL expiration.", perf: "O(1) — hash insert" },
              { op: "DELETE(key)", desc: "Explicit invalidation. Use on write to source-of-truth.", perf: "O(1) — hash remove" },
              { op: "MGET(keys[])", desc: "Batch get — fetch multiple keys in one round-trip.", perf: "O(n) — reduces network hops" },
              { op: "EXISTS(key)", desc: "Check existence without fetching value.", perf: "O(1) — useful for bloom filter" },
            ].map((h,i) => (
              <div key={i} className="flex items-start gap-3">
                <code className="text-[11px] font-mono font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded shrink-0">{h.op}</code>
                <div>
                  <div className="text-[12px] text-stone-600">{h.desc}</div>
                  <div className="text-[10px] text-stone-400 font-mono">{h.perf}</div>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-stone-100">
            <Label color="#d97706">Design Decisions</Label>
            <ul className="space-y-1.5">
              <Point icon="→" color="#d97706">Delete on write, not update — simpler and avoids race conditions</Point>
              <Point icon="→" color="#d97706">Always set a TTL — prevents stale data from living forever</Point>
              <Point icon="→" color="#d97706">Use MGET for batch operations — reduces N round-trips to 1</Point>
              <Point icon="→" color="#d97706">Serialize to JSON or protobuf — avoid language-specific formats</Point>
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
    { label: "Local Cache", desc: "In-process cache (HashMap + LRU). Zero network latency. But each server has its own copy — inconsistent data, wasted memory, cache can't survive restarts." },
    { label: "Centralized", desc: "Single Redis/Memcached node. All servers share state. Simple. But single point of failure, and one node limits both memory and throughput." },
    { label: "Distributed", desc: "Shard data across multiple nodes using consistent hashing. Horizontal scaling for both memory and throughput. Clients hash the key to find the right node." },
    { label: "Multi-Tier", desc: "L1: local in-process cache (small, fastest). L2: distributed Redis cluster (shared, larger). DB: source of truth. Check L1 → L2 → DB. Write invalidates both." },
  ];
  const diagrams = [
    () => (
      <svg viewBox="0 0 460 170" className="w-full">
        <DiagramBox x={80} y={55} w={90} h={42} label="Server A\n+ Local Cache" color="#9333ea"/>
        <DiagramBox x={230} y={55} w={90} h={42} label="Server B\n+ Local Cache" color="#9333ea"/>
        <DiagramBox x={380} y={55} w={80} h={38} label="Database" color="#059669"/>
        <Arrow x1={125} y1={55} x2={340} y2={55} label="miss → DB" id="l1" dashed/>
        <Arrow x1={275} y1={55} x2={340} y2={55} label="" id="l2" dashed/>
        <rect x={100} y={110} width={230} height={22} rx={6} fill="#dc262608" stroke="#dc262630"/>
        <text x={215} y={122} textAnchor="middle" fill="#dc2626" fontSize="9" fontFamily="monospace">❌ Each server caches independently — inconsistent</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 190" className="w-full">
        <DiagramBox x={55} y={75} w={80} h={36} label="Server A" color="#9333ea"/>
        <DiagramBox x={165} y={75} w={80} h={36} label="Server B" color="#9333ea"/>
        <DiagramBox x={300} y={75} w={80} h={42} label="Redis\n(1 node)" color="#d97706"/>
        <DiagramBox x={430} y={75} w={62} h={36} label="DB" color="#059669"/>
        <Arrow x1={95} y1={68} x2={260} y2={68} label="get/set" id="c1" />
        <Arrow x1={205} y1={82} x2={260} y2={82} label="" id="c2"/>
        <Arrow x1={340} y1={75} x2={399} y2={75} label="miss" id="c3" dashed/>
        <rect x={240} y={130} width={120} height={20} rx={4} fill="#d9770608" stroke="#d9770630"/>
        <text x={300} y={141} textAnchor="middle" fill="#d97706" fontSize="8" fontFamily="monospace">⚠ Single point of failure</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 200" className="w-full">
        <DiagramBox x={50} y={85} w={68} h={36} label="Client" color="#2563eb"/>
        <DiagramBox x={145} y={85} w={70} h={36} label="Server" color="#9333ea"/>
        <DiagramBox x={280} y={40} w={70} h={34} label="Shard 1" color="#d97706"/>
        <DiagramBox x={280} y={85} w={70} h={34} label="Shard 2" color="#d97706"/>
        <DiagramBox x={280} y={130} w={70} h={34} label="Shard 3" color="#d97706"/>
        <DiagramBox x={410} y={85} w={62} h={36} label="DB" color="#059669"/>
        <Arrow x1={84} y1={85} x2={110} y2={85} id="d1"/>
        <Arrow x1={180} y1={75} x2={245} y2={45} label="hash(key)" id="d2"/>
        <Arrow x1={180} y1={85} x2={245} y2={85} label="" id="d3"/>
        <Arrow x1={180} y1={95} x2={245} y2={125} label="" id="d4"/>
        <Arrow x1={315} y1={85} x2={379} y2={85} label="miss" id="d5" dashed/>
        <rect x={225} y={168} width={130} height={18} rx={4} fill="#05966908" stroke="#05966930"/>
        <text x={290} y={178} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">✓ Consistent hashing</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 210" className="w-full">
        <DiagramBox x={50} y={75} w={68} h={36} label="Client" color="#2563eb"/>
        <DiagramBox x={155} y={75} w={80} h={42} label="Server\n+ L1 Cache" color="#9333ea"/>
        <DiagramBox x={300} y={45} w={70} h={34} label="L2 Shard 1" color="#d97706"/>
        <DiagramBox x={300} y={105} w={70} h={34} label="L2 Shard 2" color="#d97706"/>
        <DiagramBox x={420} y={75} w={62} h={36} label="DB" color="#059669"/>
        <Arrow x1={84} y1={75} x2={115} y2={75} id="m1"/>
        <Arrow x1={195} y1={65} x2={265} y2={50} label="L1 miss" id="m2"/>
        <Arrow x1={195} y1={85} x2={265} y2={100} label="" id="m3"/>
        <Arrow x1={335} y1={75} x2={389} y2={75} label="L2 miss" id="m4" dashed/>
        <rect x={100} y={155} width={280} height={18} rx={4} fill="#6366f108" stroke="#6366f130"/>
        <text x={240} y={165} textAnchor="middle" fill="#6366f1" fontSize="8" fontFamily="monospace">L1: ~0.1ms local | L2: ~1ms network | DB: ~10ms disk</text>
        <rect x={100} y={180} width={280} height={18} rx={4} fill="#05966908" stroke="#05966930"/>
        <text x={240} y={190} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">✓ Best of both: consistency (L2) + speed (L1)</text>
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
      <Card>
        <Label color="#c026d3">Write Strategies — The Core Tradeoff</Label>
        <p className="text-[12px] text-stone-500 mb-4">How you handle writes determines your consistency guarantees. This is the most important design decision after eviction policy.</p>
        <div className="grid grid-cols-4 gap-4">
          {[
            { t: "Cache-Aside ★", d: "App manages cache explicitly. Read: check cache → miss → read DB → populate cache. Write: update DB → delete cache.", pros: ["Simple, widely used","App controls what's cached","Lazy loading — only caches what's read"], cons: ["Cache miss penalty on first read","Potential stale window between DB write and cache delete"], pick: true },
            { t: "Write-Through", d: "Every write goes to both cache and DB synchronously. Cache always has latest data.", pros: ["Strong consistency","No stale data","Simple reads (always in cache)"], cons: ["Higher write latency (double write)","Caches data that may never be read"], pick: false },
            { t: "Write-Behind", d: "Write to cache immediately, asynchronously flush to DB in batches. Cache absorbs write spikes.", pros: ["Lowest write latency","Batch DB writes = efficiency","Absorbs write spikes"], cons: ["Data loss risk if cache crashes","Complex — need write-ahead log","Eventually consistent"], pick: false },
            { t: "Read-Through", d: "Cache itself fetches from DB on miss. App only talks to cache. Cache-as-a-proxy.", pros: ["Simplifies app code","Cache handles all data fetching"], cons: ["Cache must know about DB schema","Less flexible","Cache library coupling"], pick: false },
          ].map((o,i) => (
            <div key={i} className={`rounded-lg border p-3.5 ${o.pick ? "border-purple-300 bg-purple-50/50" : "border-stone-200"}`}>
              <div className={`text-[11px] font-bold mb-1.5 ${o.pick ? "text-purple-700" : "text-stone-600"}`}>{o.t}</div>
              <p className="text-[10px] text-stone-500 mb-2">{o.d}</p>
              <div className="space-y-0.5 text-[10px]">
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

function AlgorithmSection() {
  const [sel, setSel] = useState("lru");
  const algos = {
    lru: { name: "LRU (Least Recently Used) ★", cx: "O(1) / O(n)",
      pros: ["Intuitive — evicts what hasn't been accessed longest","O(1) get/put with HashMap + Doubly Linked List","Default in Redis, Memcached, and most caches","Works well for most workloads"],
      cons: ["Scan pollution — one-time bulk scans evict hot items","Doesn't account for frequency (item used once recently beats item used 1000× before)"],
      when: "Best general-purpose choice. Use when access patterns have temporal locality (recently accessed = likely accessed again). This is the recommended default.",
      code: `# LRU Cache — HashMap + Doubly Linked List
# HashMap: key → Node (O(1) lookup)
# DLL: ordered by access time (O(1) move-to-head)

GET(key):
    node = hashmap[key]
    if node exists:
        move node to HEAD of DLL    # mark as recently used
        return node.value
    return null  (cache miss)

SET(key, value):
    if key in hashmap:
        update node.value
        move node to HEAD
    else:
        if cache is FULL:
            evict TAIL of DLL       # least recently used
            remove from hashmap
        insert new node at HEAD
        add to hashmap` },
    lfu: { name: "LFU (Least Frequently Used)", cx: "O(1) / O(n)",
      pros: ["Better for skewed distributions — keeps truly popular items","Resistant to scan pollution (one-time access doesn't boost rank)","Used by Redis when configured (allkeys-lfu)"],
      cons: ["Cold start problem — new items have frequency=1, easily evicted","Stale popular items never leave (need aging/decay mechanism)","More complex implementation"],
      when: "When you have clear hot items that should always stay cached (celebrity profiles, trending posts). Needs frequency decay to handle changing popularity.",
      code: `# LFU Cache — HashMap + Frequency Buckets
# HashMap: key → (value, frequency)
# FreqMap: frequency → DoublyLinkedList of keys

GET(key):
    if key in hashmap:
        increment frequency[key]
        move key to next frequency bucket
        return value
    return null

SET(key, value):
    if cache is FULL:
        evict from LOWEST frequency bucket
        (within same frequency, evict LRU)
    insert key with frequency = 1

# Redis uses approximated LFU with logarithmic counter
# + decay factor to avoid stale popular items` },
    ttl: { name: "TTL-Based (Time-To-Live)", cx: "O(1) / O(n)",
      pros: ["Guarantees bounded staleness — data expires after set time","Simple to reason about — no complex eviction logic","Good for data with known freshness requirements","Can combine with LRU/LFU as secondary policy"],
      cons: ["All items with same TTL expire simultaneously (thundering herd)","Doesn't adapt to access patterns","Choosing the right TTL is hard"],
      when: "When data has a known freshness requirement (session tokens: 30min, stock prices: 5sec, user profiles: 5min). Often used alongside LRU as the primary consistency mechanism.",
      code: `# TTL-Based Eviction
# Each key has an expiration timestamp

SET(key, value, ttl=300):
    store value
    set expiry = now() + ttl

GET(key):
    if key exists AND now() < expiry:
        return value      # still fresh
    else:
        delete key        # expired — lazy cleanup
        return null       # treat as cache miss

# Two cleanup strategies:
# 1. Lazy: check on access (Redis does this)
# 2. Active: background thread scans for expired keys
# Redis uses hybrid: lazy + periodic sampling` },
    consistent: { name: "Consistent Hashing ★", cx: "O(log n) / O(n)",
      pros: ["Adding/removing node moves only K/N keys (not all keys)","Virtual nodes ensure even distribution","Standard for distributed caches (Memcached, DynamoDB)","Minimizes cache invalidation during scaling"],
      cons: ["More complex than simple modulo hashing","Virtual nodes add memory overhead for the ring","Rebalancing still causes some cache misses"],
      when: "Essential when you need to add or remove cache nodes without invalidating the entire cache. This is how you shard data across nodes in any production distributed cache.",
      code: `# Consistent Hashing — Hash Ring
# Nodes are placed on a circular hash ring
# Each key hashes to a point → find next node clockwise

LOOKUP(key):
    position = hash(key) % RING_SIZE
    walk clockwise until first node
    return that node

ADD_NODE(new_node):
    place on ring at hash(new_node)
    only keys between prev_node and new_node move
    → ~K/N keys relocate (not all keys!)

# Problem: Uneven distribution with few nodes
# Solution: Virtual nodes — each physical node gets
#   100-200 positions on the ring
#   Vnodes spread load evenly` },
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
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Purpose</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Complexity</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Weakness</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Best For</th>
            </tr></thead>
            <tbody>
              {[
                { n:"LRU ★", p:"Eviction", c:"O(1)", w:"Scan pollution", f:"General purpose", hl:true },
                { n:"LFU", p:"Eviction", c:"O(1)", w:"Cold start / stale popular", f:"Skewed access" },
                { n:"TTL-Based", p:"Expiration", c:"O(1)", w:"Thundering herd", f:"Known freshness" },
                { n:"Consistent Hash ★", p:"Sharding", c:"O(log n)", w:"Rebalance misses", f:"Node scaling", hl:true },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-purple-50" : i%2?"bg-stone-50/50":""}>
                  <td className={`px-3 py-2 font-mono ${r.hl?"text-purple-700 font-bold":"text-stone-600"}`}>{r.n}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.p}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.c}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.w}</td>
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
          <CodeBlock title={`${a.name} — Pseudocode`} code={a.code} />
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
          <Label color="#dc2626">Key Design Patterns</Label>
          <CodeBlock code={`# Key naming convention (use colons as separators)
user:{user_id}              → user profile JSON
user:{user_id}:feed         → pre-computed feed
session:{session_id}        → session data
post:{post_id}:likes_count  → counter
leaderboard:daily           → sorted set

# Examples
redis> GET user:42
'{"name":"Alice","avatar":"...","tier":"premium"}'

redis> GET user:42:feed
'[{"post_id":100,...},{"post_id":99,...}]'

# TTL strategy by data type
user profile   → TTL 300s  (5 min, changes rarely)
session        → TTL 1800s (30 min, sliding window)
feed           → TTL 60s   (1 min, changes often)
count/counter  → TTL 30s   (30 sec, near-real-time)`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Serialization Choices</Label>
          <div className="space-y-3">
            {[
              { fmt: "JSON", size: "~1x (baseline)", speed: "Moderate", pros: "Human-readable, debuggable, language-agnostic", cons: "Largest size, slowest parse" },
              { fmt: "Protocol Buffers", size: "~0.3-0.5x", speed: "Fast", pros: "Compact, schema-enforced, very fast", cons: "Not human-readable, needs schema" },
              { fmt: "MessagePack", size: "~0.5-0.7x", speed: "Fast", pros: "JSON-compatible but binary, no schema needed", cons: "Less tooling than protobuf" },
            ].map((s,i) => (
              <div key={i} className="border border-stone-100 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[11px] font-bold text-stone-700">{s.fmt}</span>
                  <Pill bg="#f3e8ff" color="#7c3aed">{s.size}</Pill>
                  <Pill bg="#ecfdf5" color="#059669">{s.speed}</Pill>
                </div>
                <div className="text-[10px] text-emerald-600">✓ {s.pros}</div>
                <div className="text-[10px] text-red-500">✗ {s.cons}</div>
              </div>
            ))}
          </div>
          <div className="mt-3 pt-3 border-t border-stone-100">
            <div className="text-[11px] text-stone-500">
              <strong className="text-stone-700">Recommendation:</strong> Start with JSON for debugging ease. Switch to protobuf when you need to optimize size/speed. At Facebook scale, they use a custom Thrift serialization — but for interviews, mentioning the tradeoff is sufficient.
            </div>
          </div>
        </Card>
      </div>
      <Card accent="#9333ea">
        <Label color="#9333ea">Cache Invalidation — The Hard Problem</Label>
        <p className="text-[12px] text-stone-500 mb-4">"There are only two hard things in computer science: cache invalidation and naming things." Here are the strategies and when each works:</p>
        <div className="grid grid-cols-3 gap-4">
          {[
            { t: "TTL Expiration ★", d: "Set a TTL on every key. After expiry, next read goes to DB and repopulates. Bounded staleness.", pros: ["Simple and predictable","Self-healing — stale data disappears","No coordination needed"], cons: ["Stale for up to TTL duration","Choosing right TTL is tricky"], pick: true },
            { t: "Explicit Invalidation", d: "On every write to DB, delete the corresponding cache key. Next read repopulates from DB.", pros: ["Near-instant consistency","Precise — only invalidate what changed"], cons: ["Requires discipline on every write path","Race condition between write + delete"], pick: false },
            { t: "Event-Driven (Pub/Sub)", d: "DB change events (CDC / binlog) trigger cache invalidation via message queue. Decoupled and reliable.", pros: ["Decoupled from write path","Catches all changes (even direct DB writes)","Scalable"], cons: ["Added infrastructure (Kafka/CDC)","Slightly delayed invalidation"], pick: false },
          ].map((o,i) => (
            <div key={i} className={`rounded-lg border p-4 ${o.pick ? "border-purple-300 bg-purple-50/50" : "border-stone-200"}`}>
              <div className={`text-[11px] font-bold mb-1.5 ${o.pick ? "text-purple-700" : "text-stone-600"}`}>{o.t}</div>
              <p className="text-[11px] text-stone-500 mb-2">{o.d}</p>
              <div className="space-y-0.5 text-[11px]">
                {o.pros.map((p,j) => <div key={j} className="text-emerald-600">✓ {p}</div>)}
                {o.cons.map((c,j) => <div key={j} className="text-red-500">✗ {c}</div>)}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 pt-3 border-t border-stone-100">
          <div className="text-[11px] text-stone-500">
            <strong className="text-stone-700">Best practice in production:</strong> Use TTL as the safety net (always set one) + explicit invalidation on writes for near-instant consistency. The TTL catches anything that explicit invalidation misses (bugs, race conditions, direct DB edits). Belt and suspenders.
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
          <Label color="#059669">Sharding (Horizontal Partitioning)</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Consistent hashing</strong> — keys map to positions on a hash ring. Adding/removing a node only moves ~K/N keys instead of rehashing everything.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Virtual nodes</strong> — each physical node gets 100-200 positions on the ring. Ensures even distribution even with few physical nodes.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Client-side vs proxy sharding</strong> — client hashes key directly (Memcached), or a proxy routes (Redis Cluster, Twemproxy). Client-side is faster, proxy is simpler to manage.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Hash slots (Redis Cluster)</strong> — 16,384 hash slots distributed across nodes. Key → CRC16(key) % 16384 → slot → node. Rebalancing moves entire slots.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Replication</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Primary-replica</strong> — writes go to primary, replicated to 1-2 replicas asynchronously. Replicas serve reads for additional throughput.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Read replicas for throughput</strong> — if you need 500K read QPS but each node handles 100K, use 1 primary + 4 read replicas instead of 5 shards.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Async replication tradeoff</strong> — replica might be slightly behind primary (milliseconds). A read after write to different nodes may see stale data.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Failover</strong> — when primary fails, promote a replica. Redis Sentinel or Redis Cluster handles this automatically in seconds.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Strategy</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Option A: Local Cache per Region", d:"Each region has its own independent cache cluster. Cache misses hit a local DB replica.", pros:["Lowest latency","No cross-region dependency","Each region is self-contained"], cons:["Different regions may have inconsistent cache","More total memory needed"], pick:false },
            { t:"Option B: Write-Through Replication ★", d:"Write to primary region's cache, replicate to other regions asynchronously. All regions read locally.", pros:["Low read latency everywhere","Eventual consistency across regions","Single write path"], cons:["Cross-region replication lag","Complexity of replication layer"], pick:true },
            { t:"Option C: Global Cache Cluster", d:"All regions talk to one central cache. Consistent but high latency for remote regions.", pros:["Single source of truth","Simple consistency model"], cons:["Cross-region latency 50-200ms","Defeats purpose of caching","Single point of failure"], pick:false },
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
        <Label color="#d97706">Critical Decision: What Happens When Cache Is Down?</Label>
        <p className="text-[12px] text-stone-500 mb-4">Unlike a rate limiter (which can fail-open), a cache failure means all traffic hits the database directly. This can cascade into a full outage if the DB isn't sized for full load.</p>
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Graceful Degradation (Recommended)</div>
            <p className="text-[11px] text-stone-500 mb-2">Cache down → fall through to DB, but with circuit breakers and backpressure</p>
            <ul className="space-y-1"><Point icon="✓" color="#059669">Service stays functional</Point><Point icon="✓" color="#059669">DB handles reduced traffic with rate limiting</Point><Point icon="⚠" color="#d97706">Higher latency, lower throughput during outage</Point></ul>
          </div>
          <div className="rounded-lg border border-red-200 bg-white p-4">
            <div className="text-[11px] font-bold text-red-600 mb-1.5">Thundering Herd on Recovery</div>
            <p className="text-[11px] text-stone-500 mb-2">Cache comes back empty → every request is a miss → DB overwhelmed</p>
            <ul className="space-y-1"><Point icon="→" color="#d97706">Warm cache before directing traffic</Point><Point icon="→" color="#d97706">Use stale-while-revalidate pattern</Point><Point icon="→" color="#d97706">Gradual traffic shift with connection draining</Point></ul>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#2563eb">Redis High Availability</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#2563eb"><strong className="text-stone-700">Redis Sentinel</strong> — monitors primaries, auto-promotes replica on failure. Clients discover new primary via Sentinel. Failover in seconds.</Point>
            <Point icon="2." color="#2563eb"><strong className="text-stone-700">Redis Cluster</strong> — built-in sharding + replication. Each shard has replicas. Survives node failures automatically with gossip protocol.</Point>
            <Point icon="3." color="#2563eb"><strong className="text-stone-700">Persistence (RDB + AOF)</strong> — RDB: periodic snapshots. AOF: append-only file of every write. Both together = fast restart with minimal data loss.</Point>
            <Point icon="4." color="#2563eb"><strong className="text-stone-700">Connection pooling</strong> — reuse connections. A pool of 50-100 connections per server prevents connection storms on reconnect.</Point>
          </ul>
        </Card>
        <Card>
          <Label color="#0891b2">Cache Warming Strategies</Label>
          <ul className="space-y-2.5">
            <Point icon="🔥" color="#0891b2"><strong className="text-stone-700">Pre-warm on deploy</strong> — before routing traffic to new cache, bulk-load hot keys from DB. Use MSET for batch loading.</Point>
            <Point icon="📊" color="#0891b2"><strong className="text-stone-700">Shadow traffic</strong> — replay production read traffic against new cache before cutover. Builds up hot set organically.</Point>
            <Point icon="🔄" color="#0891b2"><strong className="text-stone-700">Stale-while-revalidate</strong> — serve stale cached value immediately, refresh asynchronously in background. User sees fast response, cache stays fresh.</Point>
            <Point icon="📈" color="#0891b2"><strong className="text-stone-700">Gradual cutover</strong> — route 1% → 10% → 50% → 100% of traffic to new cache. Monitor hit ratio at each step.</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#9333ea">Degradation Ladder</Label>
        <div className="flex gap-3 items-stretch mt-1">
          {[
            { label: "Full Cache Cluster", sub: "Normal operation", color: "#059669", status: "HEALTHY" },
            { label: "Partial Failure", sub: "Some shards down", color: "#d97706", status: "DEGRADED" },
            { label: "Local Cache Only", sub: "L1 in-process", color: "#ea580c", status: "FALLBACK" },
            { label: "DB Direct", sub: "Cache fully down", color: "#dc2626", status: "EMERGENCY" },
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
              { metric: "cache.hit_ratio", type: "Gauge", desc: "Hit / (Hit + Miss). Target: >95%. The single most important metric." },
              { metric: "cache.latency_ms", type: "Histogram", desc: "p50, p95, p99 of GET/SET operations. Alert if p99 > 5ms." },
              { metric: "cache.eviction_rate", type: "Counter", desc: "Keys evicted/sec due to memory pressure. Spike = need more memory." },
              { metric: "cache.memory_used", type: "Gauge", desc: "Current memory usage vs max. Alert if > 85% (approaching evictions)." },
              { metric: "cache.connection_count", type: "Gauge", desc: "Active connections. Pool exhaustion causes connection timeouts." },
              { metric: "db.qps_after_cache", type: "Gauge", desc: "Actual QPS hitting DB. Should be ~5% of total reads." },
            ].map((m,i) => (
              <div key={i} className="flex items-start gap-2">
                <code className="text-[10px] font-mono text-sky-700 bg-sky-50 px-1.5 py-0.5 rounded shrink-0">{m.type}</code>
                <div>
                  <div className="text-[11px] font-mono font-bold text-stone-700">{m.metric}</div>
                  <div className="text-[10px] text-stone-400">{m.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Alerting Rules</Label>
          <div className="space-y-3">
            {[
              { name: "Hit Ratio Drop", rule: "hit_ratio < 90% for 5min", sev: "P2", action: "Check: key space changed? TTL too short? Evictions spiking? New traffic pattern?" },
              { name: "Latency Spike", rule: "p99 > 10ms for 3min", sev: "P2", action: "Check: network issues? Big values? Slow commands (KEYS *)? Memory swapping?" },
              { name: "Memory Critical", rule: "memory_used > 90%", sev: "P1", action: "Scale up (add shards) or reduce TTLs. Evictions will spike and hit ratio drops." },
              { name: "Node Down", rule: "node unreachable for 30s", sev: "P1", action: "Check Sentinel failover triggered. Verify replicas promoted. Monitor DB load spike." },
            ].map((a,i) => (
              <div key={i} className="border border-stone-100 rounded-lg p-2.5">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="text-[11px] font-bold text-stone-700">{a.name}</span>
                  <Pill bg={a.sev==="P1"?"#fef2f2":"#fefce8"} color={a.sev==="P1"?"#dc2626":"#d97706"}>{a.sev}</Pill>
                </div>
                <div className="text-[10px] font-mono text-stone-500">{a.rule}</div>
                <div className="text-[10px] text-stone-400 mt-0.5">{a.action}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Debugging Playbook</Label>
          <div className="space-y-3">
            {[
              { q: "Hit ratio suddenly dropped", steps: "Check: new code deploying different key patterns? TTL changed? Memory full causing evictions? Bulk data migration polluting cache?" },
              { q: "Cache latency spike", steps: "Check: large values being stored? Using O(n) commands (KEYS, SMEMBERS on huge sets)? Network saturation? Memory swapping to disk?" },
              { q: "DB load increased despite cache", steps: "Check: cache miss rate. Is a hot key expired? Are writes invalidating cache faster than reads fill it? Thundering herd on popular key?" },
              { q: "Inconsistent data between cache and DB", steps: "Check: write path — are all write paths invalidating cache? Race condition between concurrent write + invalidation? CDC lag?" },
            ].map((d,i) => (
              <div key={i}>
                <div className="text-[11px] font-bold text-stone-700">{d.q}</div>
                <div className="text-[10px] text-stone-500 mt-0.5">{d.steps}</div>
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
          { title: "Cache Stampede (Thundering Herd)", sev: "Critical", sevColor: "#dc2626",
            desc: "A popular key expires → hundreds of concurrent requests all miss → all hit DB simultaneously → DB overwhelmed. The more popular the key, the worse the stampede.",
            fix: "Locking: first request acquires a lock, fetches from DB, populates cache. Others wait or get stale value. Alternatively, use probabilistic early expiration — refresh the key slightly before TTL expires based on a random probability.",
            code: `Stampede scenario:\n  Key "trending_feed" expires at T=60\n  → 500 requests at T=60.001 all miss\n  → 500 identical DB queries\n  → DB CPU spikes to 100%\n\nSolution 1: Distributed lock\n  → First request locks key, fetches DB\n  → Others wait or get stale value\n\nSolution 2: Early probabilistic refresh\n  → At T=55, random 5% chance of refresh\n  → Key renewed before mass expiry` },
          { title: "Hot Key Problem", sev: "Critical", sevColor: "#dc2626",
            desc: "One key (celebrity profile, viral post) gets 100K+ QPS — overwhelms the single shard that owns it. Other shards are idle. Consistent hashing doesn't help because one key = one shard.",
            fix: "Replicate hot keys to multiple shards with a random suffix (key:1, key:2, ...key:N). Reads pick a random replica. Or use a local L1 cache for hot keys with short TTL (1-5 seconds) to absorb the load before it reaches the distributed cache.",
            code: `Hot key "user:celebrity_123" = 100K QPS\n  → All traffic hits Shard 2\n  → Shard 2 overloaded, others idle\n\nSolution 1: Key replication\n  → Store as celebrity_123:r0 ... :r9\n  → Read from random replica shard\n  → Spreads load across 10 shards\n\nSolution 2: L1 local cache\n  → Cache hot keys in-process (1-5s TTL)\n  → 50 servers × local = load distributed` },
          { title: "Cache Penetration", sev: "Medium", sevColor: "#d97706",
            desc: "Queries for keys that will NEVER exist in DB (e.g., invalid user IDs). Every request misses cache → hits DB → DB returns nothing → nothing cached → next request repeats. Attackers can exploit this.",
            fix: "Cache null results with short TTL (e.g., 60s). Or use a Bloom filter in front of the cache — if the Bloom filter says 'definitely not in DB,' skip both cache and DB entirely.",
            code: `Attack: GET user:99999999 (doesn't exist)\n  → Cache miss → DB query → no result\n  → Nothing to cache → next request repeats\n  → 10K requests/sec all hitting DB\n\nSolution 1: Cache null values\n  → SET user:99999999 = NULL, TTL=60\n  → Next request gets cached null\n\nSolution 2: Bloom filter\n  → Check: "Could this key exist?"\n  → If NO → return null immediately\n  → If MAYBE → proceed to cache/DB` },
          { title: "Cache Avalanche", sev: "Critical", sevColor: "#dc2626",
            desc: "Many keys all expire at the same time (e.g., all set with TTL=300 at the same moment). Massive spike of cache misses all hit DB simultaneously. Similar to stampede but across many keys.",
            fix: "Add jitter to TTL values: instead of TTL=300, use TTL=300 + random(0,60). This spreads expirations over a 60-second window instead of all at once. Simple and effective.",
            code: `Avalanche scenario:\n  1000 keys all set at T=0 with TTL=300\n  → All expire at T=300\n  → 1000 DB queries simultaneously\n\nSolution: TTL jitter\n  → TTL = base_ttl + random(0, jitter)\n  → TTL = 300 + random(0, 60)\n  → Expirations spread over 60 seconds\n  → DB load stays smooth` },
        ].map((w,i) => (
          <Card key={i}>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[12px] font-bold text-stone-800">{w.title}</span>
              <Pill bg={w.sev==="Critical"?"#fef2f2":"#fffbeb"} color={w.sevColor}>{w.sev}</Pill>
            </div>
            <p className="text-[12px] text-stone-500 mb-2">{w.desc}</p>
            <div className="text-[10px] font-bold text-emerald-600 uppercase tracking-wider mb-1">Fix</div>
            <p className="text-[12px] text-stone-500 mb-3">{w.fix}</p>
            <CodeBlock code={w.code} />
          </Card>
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   LLD — Implementation Architecture Sections
   ═══════════════════════════════════════════════════════════ */

function ServicesSection() {
  return (
    <div className="space-y-5">
      <Card className="bg-teal-50/50 border-teal-200">
        <div className="flex items-start gap-3">
          <span className="text-lg">🧩</span>
          <div>
            <div className="text-[12px] font-bold text-teal-700">Why This Matters</div>
            <p className="text-[12px] text-stone-500 mt-0.5">In an HLD interview you say "cache layer" and draw a box. In a real build, that box is 5-6 services, each with its own deploy, on-call, and failure modes. This section is about what's <em>inside</em> the boxes.</p>
          </div>
        </div>
      </Card>

      <Card accent="#0f766e">
        <Label color="#0f766e">Service Decomposition — What We Actually Build</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { name: "Cache Node (Data Plane)", owns: "Key-value storage, eviction, replication", tech: "Redis / custom C++ engine", api: "GET, SET, DEL, MGET, EXPIRE", scale: "Horizontal — add shards", stateful: true,
              modules: ["Memory Allocator (slab/jemalloc)", "Hash Table (key → slot lookup)", "Eviction Engine (LRU/LFU/TTL)", "Replication Module (primary → replica sync)", "Persistence Engine (RDB snapshots + AOF)", "Network Handler (RESP protocol parser)"] },
            { name: "Cache Proxy (Routing)", owns: "Key-to-shard routing, connection pooling, request fan-out", tech: "mcrouter / Envoy / Twemproxy / custom Go", api: "Same as cache node (transparent proxy)", scale: "Horizontal — stateless", stateful: false,
              modules: ["Consistent Hash Ring (virtual nodes)", "Connection Pool Manager", "Health Checker (per-node)", "Circuit Breaker (per-shard)", "Request Pipeline (batch, retry)", "Config Watcher (ring membership changes)"] },
            { name: "Config Service", owns: "Cluster topology, shard map, ring membership", tech: "ZooKeeper / etcd / Consul", api: "GET /topology, WATCH /changes", scale: "3-5 node quorum", stateful: true,
              modules: ["Topology Store (shard → nodes map)", "Membership Manager (node join/leave)", "Health Aggregator", "Change Notifier (push to proxies)", "Admin API (manual overrides)", "Audit Logger"] },
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

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#2563eb">
          <Label color="#2563eb">Supporting Services</Label>
          <div className="space-y-3">
            {[
              { name: "Metrics Collector", role: "Scrapes hit ratio, latency, eviction rate from each node. Pushes to Prometheus/Datadog.", tech: "Prometheus exporter sidecar", critical: false },
              { name: "Admin Dashboard", role: "UI for ops: view cluster topology, trigger resharding, force failover, inspect hot keys.", tech: "React + internal API", critical: false },
              { name: "Cache Warmer", role: "Pre-loads hot keys into new/cold nodes before traffic cutover. Reads from DB or snapshot.", tech: "Batch job (Spark/script)", critical: true },
              { name: "Invalidation Bus", role: "Consumes DB CDC events (Debezium/binlog) and issues DELETE to cache. Decoupled from write path.", tech: "Kafka consumer → cache proxy", critical: true },
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
          <Label color="#9333ea">Cache Node Internals — Block Diagram</Label>
          <svg viewBox="0 0 380 320" className="w-full">
            {/* Network layer */}
            <rect x={10} y={10} width={360} height={45} rx={6} fill="#6366f108" stroke="#6366f1" strokeWidth={1}/>
            <text x={190} y={28} textAnchor="middle" fill="#6366f1" fontSize="10" fontWeight="600" fontFamily="monospace">Network Handler (RESP Protocol)</text>
            <text x={190} y={43} textAnchor="middle" fill="#6366f180" fontSize="8" fontFamily="monospace">TCP connections · epoll/io_uring · request parsing</text>

            {/* Command processor */}
            <rect x={10} y={65} width={360} height={40} rx={6} fill="#9333ea08" stroke="#9333ea" strokeWidth={1}/>
            <text x={190} y={83} textAnchor="middle" fill="#9333ea" fontSize="10" fontWeight="600" fontFamily="monospace">Command Processor</text>
            <text x={190} y={96} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">parse cmd → auth check → execute → serialize response</text>

            {/* Hash table + memory */}
            <rect x={10} y={115} width={175} height={70} rx={6} fill="#d9770608" stroke="#d97706" strokeWidth={1}/>
            <text x={97} y={138} textAnchor="middle" fill="#d97706" fontSize="10" fontWeight="600" fontFamily="monospace">Hash Table</text>
            <text x={97} y={153} textAnchor="middle" fill="#d9770680" fontSize="8" fontFamily="monospace">key → memory pointer</text>
            <text x={97} y={168} textAnchor="middle" fill="#d9770680" fontSize="8" fontFamily="monospace">progressive rehashing</text>

            <rect x={195} y={115} width={175} height={70} rx={6} fill="#05966908" stroke="#059669" strokeWidth={1}/>
            <text x={282} y={138} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="600" fontFamily="monospace">Memory Allocator</text>
            <text x={282} y={153} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">jemalloc / slab allocator</text>
            <text x={282} y={168} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">defrag · size classes</text>

            {/* Eviction + TTL */}
            <rect x={10} y={195} width={175} height={55} rx={6} fill="#dc262608" stroke="#dc2626" strokeWidth={1}/>
            <text x={97} y={215} textAnchor="middle" fill="#dc2626" fontSize="10" fontWeight="600" fontFamily="monospace">Eviction Engine</text>
            <text x={97} y={230} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">LRU / LFU / TTL sampling</text>
            <text x={97} y={242} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">memory pressure triggers</text>

            <rect x={195} y={195} width={175} height={55} rx={6} fill="#0891b208" stroke="#0891b2" strokeWidth={1}/>
            <text x={282} y={215} textAnchor="middle" fill="#0891b2" fontSize="10" fontWeight="600" fontFamily="monospace">Replication Module</text>
            <text x={282} y={230} textAnchor="middle" fill="#0891b280" fontSize="8" fontFamily="monospace">primary → replica stream</text>
            <text x={282} y={242} textAnchor="middle" fill="#0891b280" fontSize="8" fontFamily="monospace">partial resync · backlog</text>

            {/* Persistence */}
            <rect x={10} y={260} width={360} height={50} rx={6} fill="#78716c08" stroke="#78716c" strokeWidth={1}/>
            <text x={190} y={280} textAnchor="middle" fill="#78716c" fontSize="10" fontWeight="600" fontFamily="monospace">Persistence Layer</text>
            <text x={100} y={298} textAnchor="middle" fill="#78716c80" fontSize="8" fontFamily="monospace">RDB Snapshots (fork + COW)</text>
            <text x={280} y={298} textAnchor="middle" fill="#78716c80" fontSize="8" fontFamily="monospace">AOF (append-only file + rewrite)</text>

            {/* Arrows */}
            <line x1={190} y1={55} x2={190} y2={65} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-ni1)"/>
            <line x1={190} y1={105} x2={97} y2={115} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={190} y1={105} x2={282} y2={115} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={97} y1={185} x2={97} y2={195} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={282} y1={185} x2={282} y2={195} stroke="#94a3b8" strokeWidth={1}/>
            <defs><marker id="ah-ni1" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          </svg>
        </Card>
      </div>

      <Card>
        <Label color="#0f766e">Service-to-Service Contracts</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[11px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Caller → Callee</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Protocol</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Contract</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Timeout</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">On Failure</th>
            </tr></thead>
            <tbody>
              {[
                { route: "App Server → Cache Proxy", proto: "RESP (TCP)", contract: "GET/SET/DEL with key + TTL", timeout: "50ms", fail: "Skip cache, hit DB directly" },
                { route: "Cache Proxy → Cache Node", proto: "RESP (TCP)", contract: "Proxied command + shard routing", timeout: "30ms", fail: "Try replica, then circuit-break shard" },
                { route: "Cache Proxy → Config Service", proto: "gRPC + Watch", contract: "GetTopology / WatchChanges stream", timeout: "5s (initial), streaming", fail: "Use last-known topology from local cache" },
                { route: "Invalidation Bus → Cache Proxy", proto: "Kafka consumer", contract: "CDC event → DELETE key", timeout: "N/A (async)", fail: "Retry with backoff, DLQ after 3 tries" },
                { route: "Cache Node → Cache Node", proto: "RESP (replication)", contract: "Primary streams writes to replica", timeout: "Configurable backlog", fail: "Full resync if backlog exceeded" },
                { route: "Metrics Collector → Cache Node", proto: "HTTP /metrics", contract: "Prometheus scrape endpoint", timeout: "2s", fail: "Alert: node_scrape_failed" },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-mono text-teal-700 font-medium">{r.route}</td>
                  <td className="px-3 py-2 text-stone-500">{r.proto}</td>
                  <td className="px-3 py-2 text-stone-500">{r.contract}</td>
                  <td className="px-3 py-2 font-mono text-stone-400">{r.timeout}</td>
                  <td className="px-3 py-2 text-stone-400">{r.fail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function FlowsSection() {
  const [flow, setFlow] = useState("read");
  const flows = {
    read: {
      title: "Cache Read (Full Path)",
      steps: [
        { actor: "Client", action: "HTTP GET /api/user/42", type: "request" },
        { actor: "API Gateway", action: "Authenticate request (JWT verify, rate limit check)", type: "auth" },
        { actor: "App Server", action: "Build cache key: user:42", type: "process" },
        { actor: "App Server → Proxy", action: "RESP GET user:42 (timeout: 50ms)", type: "request" },
        { actor: "Cache Proxy", action: "hash(user:42) → Shard 2, pick healthy node", type: "process" },
        { actor: "Cache Proxy", action: "Check circuit breaker for Shard 2: CLOSED ✓", type: "check" },
        { actor: "Cache Node", action: "Hash table lookup → found → check TTL → not expired", type: "success" },
        { actor: "Cache Node", action: "Move key to LRU head, return serialized JSON", type: "process" },
        { actor: "App Server", action: "Deserialize, return to client. Latency: ~2ms total", type: "success" },
      ]
    },
    miss: {
      title: "Cache Miss → DB Fetch → Populate",
      steps: [
        { actor: "Cache Node", action: "Hash table lookup → NOT FOUND (cache miss)", type: "error" },
        { actor: "Cache Proxy", action: "Return nil to App Server", type: "process" },
        { actor: "App Server", action: "Increment metric: cache.miss_count", type: "process" },
        { actor: "App Server → DB", action: "SELECT * FROM users WHERE id = 42 (timeout: 200ms)", type: "request" },
        { actor: "Database", action: "Query execution → return row", type: "process" },
        { actor: "App Server", action: "Serialize user object to JSON", type: "process" },
        { actor: "App Server → Proxy", action: "SET user:42 {json} EX 300 (async, don't block response)", type: "request" },
        { actor: "App Server", action: "Return user data to client. Latency: ~50ms total", type: "success" },
        { actor: "Cache Node", action: "Store key, set TTL, update LRU. Background.", type: "process" },
      ]
    },
    write: {
      title: "Write (Invalidation Path)",
      steps: [
        { actor: "Client", action: "PUT /api/user/42 {name: 'New Name'}", type: "request" },
        { actor: "API Gateway", action: "Auth + rate limit + idempotency key check", type: "auth" },
        { actor: "App Server → DB", action: "BEGIN TRANSACTION → UPDATE users SET name='New Name' WHERE id=42 → COMMIT", type: "request" },
        { actor: "Database", action: "Write committed. Binlog event emitted.", type: "success" },
        { actor: "App Server → Proxy", action: "DEL user:42 (explicit invalidation)", type: "process" },
        { actor: "Cache Node", action: "Delete key from hash table, free memory", type: "process" },
        { actor: "App Server", action: "Return 200 OK to client", type: "success" },
        { actor: "Invalidation Bus", action: "(Async) CDC picks up binlog → issues DEL user:42 (belt-and-suspenders)", type: "process" },
        { actor: "Cache Node", action: "DEL is idempotent — no-op if already deleted", type: "check" },
      ]
    },
    failure: {
      title: "Failure Path — Cache Node Down",
      steps: [
        { actor: "App Server → Proxy", action: "RESP GET user:42 (timeout: 50ms)", type: "request" },
        { actor: "Cache Proxy", action: "Route to Shard 2 primary → connection refused", type: "error" },
        { actor: "Cache Proxy", action: "Try Shard 2 replica → connection timeout (30ms)", type: "error" },
        { actor: "Cache Proxy", action: "Circuit breaker OPEN for Shard 2 (half-open in 30s)", type: "error" },
        { actor: "Cache Proxy", action: "Return error to App Server", type: "error" },
        { actor: "App Server", action: "Cache unavailable → fall through to DB (graceful degradation)", type: "check" },
        { actor: "App Server → DB", action: "SELECT * FROM users WHERE id = 42", type: "request" },
        { actor: "App Server", action: "Return data. Latency: ~50ms. Emit metric: cache.fallback_active", type: "success" },
        { actor: "Config Service", action: "(Async) Detect node failure → promote replica → update topology → notify proxies", type: "process" },
      ]
    },
    stampede: {
      title: "Stampede Prevention — Locking Flow",
      steps: [
        { actor: "Request 1", action: "GET trending_feed → MISS", type: "error" },
        { actor: "Request 1", action: "SET lock:trending_feed NX EX 5 → acquired ✓", type: "success" },
        { actor: "Request 2-500", action: "GET trending_feed → MISS", type: "error" },
        { actor: "Request 2-500", action: "SET lock:trending_feed NX → FAILED (lock exists)", type: "check" },
        { actor: "Request 2-500", action: "Wait 50ms, retry GET trending_feed (poll loop, max 3 retries)", type: "process" },
        { actor: "Request 1 → DB", action: "SELECT trending_feed query (heavy, 200ms)", type: "request" },
        { actor: "Request 1 → Cache", action: "SET trending_feed {data} EX 60 → stored", type: "success" },
        { actor: "Request 1", action: "DEL lock:trending_feed → released", type: "process" },
        { actor: "Request 2-500", action: "Retry GET trending_feed → HIT ✓ All 499 served from cache", type: "success" },
      ]
    },
  };
  const f = flows[flow];
  const typeColors = { request:"#2563eb", auth:"#d97706", process:"#6b7280", success:"#059669", error:"#dc2626", check:"#9333ea" };
  return (
    <div className="space-y-5">
      <Card accent="#7e22ce">
        <Label color="#7e22ce">Detailed Request Flows — Every Step, Every Error Path</Label>
        <div className="flex gap-2 mb-4 flex-wrap">
          {Object.entries(flows).map(([k,v]) => (
            <button key={k} onClick={() => setFlow(k)}
              className={`px-3.5 py-1.5 rounded-lg text-[12px] font-medium transition-all border ${k===flow ? "bg-purple-700 text-white border-purple-700" : "bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
              {v.title.split("—")[0].trim()}
            </button>
          ))}
        </div>
        <div className="text-[13px] font-bold text-stone-800 mb-3">{f.title}</div>
        <div className="space-y-0">
          {f.steps.map((s,i) => (
            <div key={i} className="flex items-start gap-3 py-2.5 border-b border-stone-100 last:border-0">
              <span className="text-[10px] font-bold w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5 bg-stone-200 text-stone-500">{i+1}</span>
              <div className="w-2 h-2 rounded-full shrink-0 mt-1.5" style={{ background: typeColors[s.type] }}/>
              <div className="flex-1 min-w-0">
                <span className="text-[11px] font-bold text-stone-700">{s.actor}:</span>
                <span className="text-[12px] text-stone-600 ml-1.5">{s.action}</span>
              </div>
              <span className="text-[9px] font-mono px-2 py-0.5 rounded-full shrink-0" style={{ background: typeColors[s.type]+"12", color: typeColors[s.type] }}>{s.type}</span>
            </div>
          ))}
        </div>
      </Card>
      <Card>
        <Label color="#7e22ce">Circuit Breaker State Machine</Label>
        <div className="flex gap-4 items-center justify-center py-4">
          {[
            { state: "CLOSED", sub: "Normal operation", color: "#059669", desc: "All requests pass through. Track failure count." },
            { state: "OPEN", sub: "Shard bypassed", color: "#dc2626", desc: "All requests fast-fail. Timer starts (30s)." },
            { state: "HALF-OPEN", sub: "Testing recovery", color: "#d97706", desc: "Allow 1 probe request. Success → CLOSED. Fail → OPEN." },
          ].map((s,i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="text-center w-36">
                <div className="py-3 px-4 rounded-lg border-2" style={{ borderColor: s.color, background: s.color+"08" }}>
                  <div className="text-[12px] font-bold font-mono" style={{ color: s.color }}>{s.state}</div>
                  <div className="text-[9px] text-stone-400 mt-0.5">{s.sub}</div>
                </div>
                <div className="text-[10px] text-stone-500 mt-1.5">{s.desc}</div>
              </div>
              {i<2 && <div className="text-stone-300 text-lg">→</div>}
            </div>
          ))}
        </div>
        <div className="grid grid-cols-3 gap-3 mt-3 text-[10px]">
          <div className="bg-emerald-50 rounded-lg p-2 text-center text-emerald-700">CLOSED → OPEN: 5 failures in 10s</div>
          <div className="bg-red-50 rounded-lg p-2 text-center text-red-700">OPEN → HALF-OPEN: after 30s cooldown</div>
          <div className="bg-amber-50 rounded-lg p-2 text-center text-amber-700">HALF-OPEN → CLOSED: probe succeeds</div>
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
          <Label color="#b45309">Kubernetes Deployment Topology</Label>
          <CodeBlock title="Cache Node — StatefulSet (NOT Deployment)" code={`# Cache nodes are STATEFUL — use StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: cache-shard
spec:
  replicas: 4              # 4 shards
  serviceName: cache-shard
  podManagementPolicy: Parallel
  template:
    spec:
      containers:
      - name: redis
        image: redis:7.2-alpine
        resources:
          requests:
            memory: "26Gi"   # r6g.xlarge equivalent
            cpu: "3"
          limits:
            memory: "28Gi"   # small headroom for fork()
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: data
          mountPath: /data   # RDB + AOF persistence
      - name: exporter       # Sidecar for metrics
        image: redis-exporter:latest
        ports:
        - containerPort: 9121
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        # ^ Spread shards across AZs
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi      # For RDB snapshots`} />
          <div className="mt-3 space-y-1.5">
            <Point icon="⚠" color="#b45309">StatefulSet not Deployment — pods have stable network IDs (cache-shard-0, cache-shard-1)</Point>
            <Point icon="⚠" color="#b45309">PersistentVolumeClaim per pod — data survives pod restart</Point>
            <Point icon="⚠" color="#b45309">Memory limit slightly above request — fork() for RDB snapshot needs ~30% extra (copy-on-write)</Point>
          </div>
        </Card>
        <Card accent="#0f766e">
          <Label color="#0f766e">Multi-AZ / Multi-Region Layout</Label>
          <svg viewBox="0 0 380 330" className="w-full">
            {/* Region */}
            <rect x={5} y={5} width={370} height={320} rx={10} fill="#0f766e05" stroke="#0f766e30" strokeWidth={1} strokeDasharray="4,3"/>
            <text x={190} y={22} textAnchor="middle" fill="#0f766e" fontSize="10" fontWeight="700" fontFamily="monospace">us-east-1 (Primary Region)</text>

            {/* AZ-a */}
            <rect x={15} y={35} width={115} height={175} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={72} y={50} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-a</text>
            <rect x={25} y={60} width={95} height={28} rx={4} fill="#d9770615" stroke="#d97706" strokeWidth={1}/>
            <text x={72} y={78} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Shard 1 (P)</text>
            <rect x={25} y={95} width={95} height={28} rx={4} fill="#0891b215" stroke="#0891b2" strokeWidth={1}/>
            <text x={72} y={113} textAnchor="middle" fill="#0891b2" fontSize="8" fontWeight="600" fontFamily="monospace">Shard 3 (R)</text>
            <rect x={25} y={135} width={95} height={25} rx={4} fill="#9333ea10" stroke="#9333ea80" strokeWidth={1}/>
            <text x={72} y={151} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">Proxy Pod ×2</text>
            <rect x={25} y={168} width={95} height={25} rx={4} fill="#78716c10" stroke="#78716c80" strokeWidth={1}/>
            <text x={72} y={184} textAnchor="middle" fill="#78716c" fontSize="8" fontFamily="monospace">Config Node</text>

            {/* AZ-b */}
            <rect x={140} y={35} width={115} height={175} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={197} y={50} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-b</text>
            <rect x={150} y={60} width={95} height={28} rx={4} fill="#d9770615" stroke="#d97706" strokeWidth={1}/>
            <text x={197} y={78} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Shard 2 (P)</text>
            <rect x={150} y={95} width={95} height={28} rx={4} fill="#0891b215" stroke="#0891b2" strokeWidth={1}/>
            <text x={197} y={113} textAnchor="middle" fill="#0891b2" fontSize="8" fontWeight="600" fontFamily="monospace">Shard 1 (R)</text>
            <rect x={150} y={135} width={95} height={25} rx={4} fill="#9333ea10" stroke="#9333ea80" strokeWidth={1}/>
            <text x={197} y={151} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">Proxy Pod ×2</text>
            <rect x={150} y={168} width={95} height={25} rx={4} fill="#78716c10" stroke="#78716c80" strokeWidth={1}/>
            <text x={197} y={184} textAnchor="middle" fill="#78716c" fontSize="8" fontFamily="monospace">Config Node</text>

            {/* AZ-c */}
            <rect x={265} y={35} width={100} height={175} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={315} y={50} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-c</text>
            <rect x={275} y={60} width={80} height={28} rx={4} fill="#0891b215" stroke="#0891b2" strokeWidth={1}/>
            <text x={315} y={78} textAnchor="middle" fill="#0891b2" fontSize="8" fontWeight="600" fontFamily="monospace">Shard 2 (R)</text>
            <rect x={275} y={95} width={80} height={28} rx={4} fill="#d9770615" stroke="#d97706" strokeWidth={1}/>
            <text x={315} y={113} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Shard 4 (P)</text>
            <rect x={275} y={135} width={80} height={25} rx={4} fill="#9333ea10" stroke="#9333ea80" strokeWidth={1}/>
            <text x={315} y={151} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">Proxy Pod ×2</text>
            <rect x={275} y={168} width={80} height={25} rx={4} fill="#78716c10" stroke="#78716c80" strokeWidth={1}/>
            <text x={315} y={184} textAnchor="middle" fill="#78716c" fontSize="8" fontFamily="monospace">Config Node</text>

            {/* Legend */}
            <rect x={15} y={225} width={350} height={90} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
            <text x={190} y={242} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Deployment Rules</text>
            <text x={30} y={260} fill="#78716c" fontSize="8" fontFamily="monospace">• Primary (P) and its Replica (R) NEVER in same AZ</text>
            <text x={30} y={275} fill="#78716c" fontSize="8" fontFamily="monospace">• Config service: 3 nodes across 3 AZs (quorum survives 1 AZ loss)</text>
            <text x={30} y={290} fill="#78716c" fontSize="8" fontFamily="monospace">• Proxy pods: ≥2 per AZ, stateless, behind NLB, auto-scaled on CPU</text>
            <text x={30} y={305} fill="#78716c" fontSize="8" fontFamily="monospace">• An AZ failure loses 0 data (replicas in other AZs), brief failover</text>
          </svg>
        </Card>
      </div>

      <Card accent="#dc2626">
        <Label color="#dc2626">Security & Authentication</Label>
        <div className="grid grid-cols-3 gap-4">
          {[
            { layer: "Client → API Gateway", what: "JWT / OAuth 2.0 tokens. API Gateway validates before request reaches app server. Per-user rate limiting happens here.",
              details: ["TLS termination at gateway (HTTPS)", "JWT expiry + refresh token rotation", "API key for service-to-service calls", "IP allowlisting for admin endpoints"] },
            { layer: "App Server → Cache (Internal)", what: "mTLS between services. Redis AUTH password or ACL-based auth. No plaintext over the wire.",
              details: ["mTLS with auto-rotated certs (SPIFFE/cert-manager)", "Redis AUTH requirepass (shared secret)", "Redis 6+ ACLs: per-user permissions (read-only replicas)", "Encrypt in-transit (TLS to Redis, stunnel for older versions)"] },
            { layer: "Data at Rest", what: "Cache data is in memory (lost on restart). Persistence files (RDB/AOF) should be encrypted on disk.",
              details: ["EBS encryption for PersistentVolumes", "No PII in cache keys (use opaque IDs)", "TTL ensures sensitive data auto-expires", "Audit log for admin operations (FLUSHALL, CONFIG SET)"] },
          ].map((s,i) => (
            <div key={i} className="rounded-lg border border-stone-200 p-4">
              <div className="text-[11px] font-bold text-stone-800 mb-1">{s.layer}</div>
              <p className="text-[11px] text-stone-500 mb-2">{s.what}</p>
              <div className="space-y-1">
                {s.details.map((d,j) => (
                  <div key={j} className="text-[10px] text-stone-500 flex items-center gap-1.5">
                    <span className="w-1 h-1 rounded-full bg-red-400 shrink-0"/>{d}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card accent="#b45309">
        <Label color="#b45309">Rolling Update Strategy for Stateful Cache</Label>
        <p className="text-[12px] text-stone-500 mb-3">You can't just kill-and-restart a cache node like a stateless API server. Here's the safe sequence:</p>
        <div className="space-y-0">
          {[
            { step: 1, action: "Start new replica for the target shard", detail: "New pod joins as replica, begins full sync from primary. Wait until replication lag = 0." },
            { step: 2, action: "Drain connections from old node", detail: "Update proxy config to stop sending new requests. Existing requests complete (connection draining, 30s timeout)." },
            { step: 3, action: "Promote new replica to primary (if updating primary)", detail: "Redis FAILOVER command. Atomic promotion. Old primary becomes replica." },
            { step: 4, action: "Verify cluster health", detail: "Check: all shards have primary + replica, hit ratio stable, no spike in errors. Wait 60s." },
            { step: 5, action: "Terminate old pod", detail: "StatefulSet PodManagementPolicy: one pod at a time. kubectl rollout proceeds to next shard." },
            { step: 6, action: "Repeat for each shard", detail: "Total rollout time: ~5-10 min per shard. Zero downtime, zero data loss." },
          ].map((s,i) => (
            <div key={i} className="flex items-start gap-3 py-2.5 border-b border-stone-100 last:border-0">
              <span className={`text-[10px] font-bold w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${s.step===6?"bg-emerald-600 text-white":"bg-amber-100 text-amber-700"}`}>{s.step}</span>
              <div>
                <div className="text-[12px] font-bold text-stone-700">{s.action}</div>
                <div className="text-[11px] text-stone-400">{s.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function OpsSection() {
  return (
    <div className="space-y-5">
      <Card accent="#be123c">
        <Label color="#be123c">Resharding — Adding a Shard Without Downtime</Label>
        <p className="text-[12px] text-stone-500 mb-3">This is what interviewers mean when they say "how do you scale?" Not just "add a shard" but the actual step-by-step with all the gotchas.</p>
        <div className="grid grid-cols-2 gap-5">
          <div>
            <div className="text-[10px] font-bold text-stone-600 uppercase tracking-wider mb-2">Step-by-Step Resharding Flow</div>
            <div className="space-y-0">
              {[
                { step: 1, title: "Provision new node(s)", detail: "Spin up new StatefulSet pods with empty Redis. Allocate PersistentVolume. Join cluster." },
                { step: 2, title: "Calculate slot migration plan", detail: "Redis Cluster has 16,384 hash slots. Decide which slots move to new node. Goal: ~equal slots per shard." },
                { step: 3, title: "Set slot to MIGRATING on source", detail: "Source shard marks slots as MIGRATING. Existing keys still served from source. New writes for migrated keys go to destination." },
                { step: 4, title: "Set slot to IMPORTING on destination", detail: "Destination shard marks slots as IMPORTING. Ready to receive keys." },
                { step: 5, title: "Migrate keys one by one", detail: "MIGRATE command moves individual keys. Atomic per key. Source deletes after destination confirms. Can take minutes for large slots." },
                { step: 6, title: "Update slot ownership", detail: "Once all keys migrated, flip slot ownership in cluster config. Proxies get notified via CLUSTER SLOTS." },
                { step: 7, title: "Verify and monitor", detail: "Check: all slots assigned, no MIGRATING/IMPORTING state left, hit ratio recovering, latency normal." },
              ].map((s,i) => (
                <div key={i} className="flex items-start gap-2.5 py-2 border-b border-stone-100 last:border-0">
                  <span className="text-[10px] font-bold w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5 bg-rose-100 text-rose-700">{s.step}</span>
                  <div>
                    <div className="text-[11px] font-bold text-stone-700">{s.title}</div>
                    <div className="text-[10px] text-stone-400">{s.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="text-[10px] font-bold text-stone-600 uppercase tracking-wider mb-2">What Can Go Wrong During Resharding</div>
            <div className="space-y-2.5">
              {[
                { issue: "Client requests hit wrong shard during migration", fix: "Redis returns -ASK redirect. Smart clients follow redirect to new shard. Dumb clients need proxy to handle." },
                { issue: "Migration takes too long, blocks the shard", fix: "MIGRATE has a timeout. Large keys (>1MB) can block. Solution: break large values into chunks before migration." },
                { issue: "Source node crashes mid-migration", fix: "Keys still on source are lost for those slots. Need replica failover + restart migration for incomplete slots." },
                { issue: "Hit ratio drops during/after resharding", fix: "Expected — new shard is cold. Pre-warm with cache warmer service before flipping traffic. Or accept temporary miss rate." },
                { issue: "Unbalanced slot distribution", fix: "Use redis-cli --cluster rebalance to evenly distribute slots. Monitor per-shard memory and QPS." },
              ].map((g,i) => (
                <div key={i} className="border border-stone-100 rounded-lg p-3">
                  <div className="text-[11px] font-bold text-red-600 mb-0.5">⚠ {g.issue}</div>
                  <div className="text-[10px] text-stone-500">→ {g.fix}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      <Card accent="#b45309">
        <Label color="#b45309">Auto-Scaling Triggers & Thresholds</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[11px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Trigger</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Threshold</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Action</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Cooldown</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Pitfall</th>
            </tr></thead>
            <tbody>
              {[
                { trigger: "Memory Usage", thresh: "> 80% for 10min", action: "Add shard (resharding)", cool: "30 min", pitfall: "Don't wait until 95% — evictions spike nonlinearly" },
                { trigger: "CPU Usage", thresh: "> 70% for 5min", action: "Scale proxy pods (HPA)", cool: "3 min", pitfall: "Redis is single-threaded — CPU spike on node = not fixable by more CPU" },
                { trigger: "QPS per Shard", thresh: "> 80K ops/sec", action: "Add shard or read replicas", cool: "30 min", pitfall: "Hot key won't be fixed by adding shards — same key, same shard" },
                { trigger: "Eviction Rate", thresh: "> 1000/sec sustained", action: "Add memory (vertical) or shards", cool: "30 min", pitfall: "May indicate key space grew, not just traffic" },
                { trigger: "Connection Count", thresh: "> 80% of max", action: "Scale proxy pods, increase pool size", cool: "5 min", pitfall: "Connection storm on reconnect can cascade" },
                { trigger: "Replication Lag", thresh: "> 10s for 5min", action: "Investigate network, consider read replica scale-down", cool: "15 min", pitfall: "High lag = reads from replica return stale data" },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-mono text-stone-700 font-medium">{r.trigger}</td>
                  <td className="px-3 py-2 font-mono text-amber-700">{r.thresh}</td>
                  <td className="px-3 py-2 text-stone-500">{r.action}</td>
                  <td className="px-3 py-2 text-stone-400">{r.cool}</td>
                  <td className="px-3 py-2 text-red-500 text-[10px]">{r.pitfall}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card accent="#be123c">
        <Label color="#be123c">Operational Pitfalls — Production War Stories</Label>
        <p className="text-[12px] text-stone-500 mb-3">These are the things you learn the hard way. Mentioning any of these in an interview signals real operational experience.</p>
        <div className="grid grid-cols-2 gap-4">
          {[
            { title: "Memory Fragmentation", symptom: "Redis reports 10 GB used, but OS shows 16 GB RSS. 6 GB lost to fragmentation.",
              cause: "Frequent SET/DEL of variable-size values creates memory holes that allocator can't reuse.",
              fix: "Enable activedefrag in Redis 4+. Use jemalloc (default). Monitor INFO memory → mem_fragmentation_ratio. If > 1.5, restart node (during maintenance window).",
              quote: "Fragmentation silently eats your capacity. You think you have 60% headroom but you actually have 20%." },
            { title: "Fork Bomb on RDB Snapshot", symptom: "Redis latency spikes every 60 seconds. Memory usage briefly doubles.",
              cause: "RDB snapshot uses fork(). Child process gets copy-on-write pages. Heavy writes during snapshot = massive COW overhead.",
              fix: "Schedule RDB saves during low-traffic windows. Reduce save frequency. Use AOF-only if latency matters more than snapshot speed. Set memory limit to 50% of available RAM to leave room for fork.",
              quote: "Had a P1 at 2am because RDB fork caused OOM killer to terminate Redis. We had memory limit at 90% — no room for fork." },
            { title: "Connection Pool Exhaustion", symptom: "App servers throwing 'unable to get connection from pool' errors. Cache appears down but Redis is healthy.",
              cause: "Pool size too small for traffic spike. Or slow commands blocking connections. Each checked-out connection that's waiting on a slow KEYS * blocks the pool.",
              fix: "Size pool for peak QPS / commands_per_connection. Set max-wait timeout (50ms). Ban O(n) commands in production (rename KEYS to empty). Use connection pool metrics as an alert.",
              quote: "Redis wasn't down. Our pool of 50 connections was exhausted because someone left a KEYS * in a debug endpoint that got called in prod." },
            { title: "Split-Brain After Network Partition", symptom: "Both old primary and promoted replica accept writes. Two sources of truth. Data divergence.",
              cause: "Network partition between primary and Sentinel. Sentinel promotes replica. Partition heals — now two primaries.",
              fix: "Redis config: min-replicas-to-write 1. Primary refuses writes if it can't reach any replica (detects it's isolated). Also set min-replicas-max-lag 10. Accept brief write unavailability to prevent split-brain.",
              quote: "We had 47 seconds of split-brain. 12,000 writes went to the wrong primary and were silently lost when it demoted itself." },
            { title: "Thundering Herd on Cold Start", symptom: "New cluster deployed → 100% miss rate → DB overwhelmed → cascading failure → whole system down.",
              cause: "New cache is empty. All traffic hits DB simultaneously. DB wasn't sized for full production load (it was sized for 5% of reads).",
              fix: "NEVER deploy empty cache to production traffic. Cache warmer pre-loads top 20% keys from DB. Gradual traffic shift: 1% → 10% → 50% → 100%. Feature flag to control traffic percentage.",
              quote: "We replaced our Redis cluster and 'forgot' to warm it. The DB went from 5K QPS to 120K QPS in 30 seconds. It didn't survive." },
            { title: "Serialization Version Mismatch", symptom: "App v2 writes protobuf v2 to cache. App v1 (still draining) reads it and crashes — can't deserialize.",
              cause: "Rolling deployment means old and new app versions run simultaneously. If cache value format changes, old code breaks.",
              fix: "Always use backward-compatible serialization (protobuf is good at this). Or: dual-write during migration (write both formats, read with fallback). Or: key versioning (user:42:v2) during transition.",
              quote: "Protobuf saved us — forward compatible by default. JSON with strict schema parsing... not so much." },
          ].map((p,i) => (
            <div key={i} className="rounded-lg border border-stone-200 p-4">
              <div className="text-[12px] font-bold text-stone-800 mb-1">{p.title}</div>
              <div className="space-y-1.5 text-[11px]">
                <div><span className="font-bold text-red-600">Symptom:</span> <span className="text-stone-500">{p.symptom}</span></div>
                <div><span className="font-bold text-amber-600">Cause:</span> <span className="text-stone-500">{p.cause}</span></div>
                <div><span className="font-bold text-emerald-600">Fix:</span> <span className="text-stone-500">{p.fix}</span></div>
              </div>
              <div className="mt-2 pt-2 border-t border-stone-100 text-[10px] italic text-stone-400">"{p.quote}"</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function EnhancementsSection() {
  return (
    <div className="grid grid-cols-3 gap-5">
      {[
        { t: "Multi-Tier Caching (L1 + L2)", d: "L1: in-process (Guava, Caffeine) with 1-5s TTL. L2: distributed Redis. L1 absorbs hot-key load, L2 provides shared consistency.", detail: "Check L1 → L2 → DB. Write invalidates both. L1 is tiny (100MB) but ultra-fast (no network).", effort: "Medium" },
        { t: "Write-Behind with WAL", d: "Buffer writes in cache, flush to DB asynchronously in batches. Use a Write-Ahead Log for durability.", detail: "Reduces DB write load by 10-50×. Risk: data loss if cache crashes before flush. WAL mitigates.", effort: "Hard" },
        { t: "Cache Compression", d: "Compress values before storing in cache. Trades CPU for memory. 2-5× compression on JSON.", detail: "LZ4 for speed (fast compress/decompress), Zstd for ratio. Worth it when memory is the bottleneck.", effort: "Easy" },
        { t: "Bloom Filter Gateway", d: "Bloom filter in front of cache to prevent cache penetration. O(1) 'definitely not in DB' check.", detail: "1% false positive rate uses ~10 bits per element. 50M keys = ~60MB. Eliminates null-key attacks.", effort: "Medium" },
        { t: "Cache Analytics Dashboard", d: "Real-time dashboard showing hit ratio, key distribution, hot keys, memory usage per shard.", detail: "Helps identify: keys that should be cached but aren't, keys cached but never read, hot key imbalances.", effort: "Easy" },
        { t: "Auto-Scaling", d: "Monitor memory usage and QPS. Auto-add shards when approaching limits. Scale down during off-peak.", detail: "Redis Cluster supports online resharding. Move hash slots between nodes without downtime.", effort: "Hard" },
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
    { q:"Redis vs Memcached — when would you pick each?", a:"Memcached: pure key-value, multi-threaded (better raw throughput per node), simpler. Redis: rich data types (lists, sets, sorted sets, hashes), single-threaded but supports persistence, Lua scripting, pub/sub, and clustering. Pick Memcached for pure caching with simple values. Pick Redis when you need data structures, persistence, or pub/sub. In practice, Redis is the default choice for most new systems.", tags:["design"] },
    { q:"How do you handle cache consistency with the database?", a:"Cache-aside pattern: write to DB first, then delete cache key (not update — avoids race conditions). TTL as safety net. For strong consistency needs: write-through (update both atomically). For eventual consistency at scale: CDC (change data capture) from DB binlog triggers cache invalidation via Kafka. The TTL + explicit delete combo handles 99% of use cases.", tags:["consistency"] },
    { q:"What happens during a cache node failure?", a:"With Redis Cluster or Sentinel: automatic failover to replica in seconds. During failover: those keys are temporarily unavailable (~1-5s). Clients retry and connect to new primary. Without replicas: those keys are lost, all requests hit DB. Solution: always have at least 1 replica per shard. Use connection pooling with retry logic.", tags:["availability"] },
    { q:"How do you prevent the thundering herd / cache stampede?", a:"Three approaches: (1) Distributed lock — first request acquires lock, fetches DB, populates cache; others wait. (2) Probabilistic early refresh — randomly refresh before TTL expires. (3) Stale-while-revalidate — serve stale value immediately, refresh async. Option 1 is most common. Option 3 gives best latency.", tags:["failure"] },
    { q:"How would you cache data for a multi-region system?", a:"Option A: Independent caches per region (simple but inconsistent). Option B: Write to primary region's cache, async replicate to others. Option C: Use a global cache (consistent but high latency). Recommendation: Option B for most cases. Write to primary region, replicate via message queue to secondary regions. Each region reads from local cache for low latency.", tags:["scalability"] },
    { q:"How do you decide what to cache?", a:"Cache data that is: (1) read-heavy (>10:1 read/write ratio), (2) expensive to compute or fetch, (3) tolerant of slight staleness. Don't cache: frequently changing data, large blobs that eat memory, data that must be real-time consistent. Start by caching the hottest 20% of keys — they typically serve 80% of traffic (Pareto principle).", tags:["design"] },
    { q:"How do you handle large values in cache?", a:"Problem: 1MB+ values cause latency spikes and memory fragmentation. Solutions: (1) Compress before caching (LZ4/Zstd). (2) Break into chunks with a manifest key. (3) Store in a separate blob store and cache only the reference. (4) Set max-value-size limit and reject oversized values. Memcached has a hard 1MB limit. Redis is more flexible but large values still hurt.", tags:["design"] },
    { q:"What's the difference between cache-aside and read-through?", a:"Cache-aside: application manages cache explicitly (check cache → miss → read DB → populate cache). Read-through: cache itself fetches from DB on miss — app only talks to cache. Cache-aside is more flexible and widely used. Read-through simplifies app code but couples cache to DB schema. Most production systems use cache-aside.", tags:["design"] },
    { q:"How would you monitor cache effectiveness?", a:"The #1 metric is hit ratio — target 95%+. Also track: p99 latency, eviction rate, memory usage, connection count, DB QPS (should be low). Dashboard showing hit ratio over time, by key prefix, by shard. Alert on: hit ratio < 90%, memory > 85%, eviction spike, latency spike.", tags:["observability"] },
    { q:"How does Facebook/Meta cache at scale?", a:"Meta runs the world's largest Memcached deployment (~1B QPS). Key innovations: (1) mcrouter — a proxy that handles consistent hashing, replication, failover. (2) Lease-based invalidation to prevent thundering herd. (3) Regional cache pools with cross-region invalidation via message queues. (4) Gutter pools — spare servers that absorb load when primary fails. Their paper 'Scaling Memcache at Facebook' is the definitive reference.", tags:["scalability"] },
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

export default function DistributedCacheSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Distributed Cache</h1>
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