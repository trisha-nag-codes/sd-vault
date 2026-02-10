import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   CONTENT DELIVERY NETWORK — System Design Reference
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
            <Label>What is a Content Delivery Network?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A CDN is a geographically distributed network of proxy servers and data centers that caches and delivers content to users from the closest edge location. It sits between users and the origin server, reducing latency from ~200ms (cross-continent round-trip) to ~20ms (nearby edge) and offloading 80-95% of traffic from origin.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Think of it like a franchise restaurant chain: instead of every customer flying to the headquarters kitchen (origin), you open locations in every city (edge PoPs). Each location stocks the most popular menu items (cached content). If they don't have something, they call the central kitchen — but 90%+ of orders are served locally.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Do We Need It?</Label>
            <ul className="space-y-2.5">
              <Point icon="⚡" color="#0891b2">Latency reduction — serve content from the nearest edge node (~20ms vs ~200ms from origin)</Point>
              <Point icon="🛡️" color="#0891b2">Origin offload — edge absorbs 80-95% of requests; origin handles only cache misses and dynamic content</Point>
              <Point icon="📈" color="#0891b2">Throughput scaling — global capacity of Tbps; no single origin could handle this traffic</Point>
              <Point icon="🌍" color="#0891b2">Global reach — serve users in 100+ countries without deploying your own infrastructure worldwide</Point>
              <Point icon="🔒" color="#0891b2">DDoS protection — distributed edge absorbs volumetric attacks before they reach origin</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Examples</Label>
            <div className="space-y-2.5">
              {[
                { co: "CloudFront", rule: "450+ PoPs globally, AWS-native", algo: "TTL + Regional Edge" },
                { co: "Cloudflare", rule: "310+ cities, reverse proxy model", algo: "Tiered caching" },
                { co: "Akamai", rule: "Oldest CDN, 4,100+ PoPs", algo: "Consistent hash + TTL" },
                { co: "Netflix OCA", rule: "Custom CDN for video streaming", algo: "Open Connect" },
                { co: "Fastly", rule: "Varnish-based, instant purge", algo: "VCL + Edge compute" },
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
              <DiagramBox x={50} y={40} w={70} h={34} label="User" color="#2563eb"/>
              <DiagramBox x={170} y={40} w={80} h={38} label="Edge\nPoP" color="#d97706"/>
              <DiagramBox x={300} y={20} w={80} h={30} label="Origin" color="#059669"/>
              <DiagramBox x={300} y={65} w={80} h={30} label="Object\nStore" color="#9333ea"/>
              <Arrow x1={85} y1={40} x2={130} y2={40} id="c1"/>
              <Arrow x1={210} y1={30} x2={260} y2={22} label="miss" id="c2" dashed/>
              <Arrow x1={210} y1={50} x2={260} y2={62} label="miss" id="c3" dashed/>
              <rect x={100} y={110} width={220} height={22} rx={4} fill="#d9770608" stroke="#d9770630"/>
              <text x={210} y={122} textAnchor="middle" fill="#d97706" fontSize="8" fontFamily="monospace">cache hit ratio 90-98% = origin gets 2-10% of requests</text>
              <rect x={100} y={140} width={220} height={22} rx={4} fill="#05966908" stroke="#05966930"/>
              <text x={210} y={152} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">200+ global PoPs for sub-50ms latency worldwide</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Google, Meta, Amazon, Netflix, Cloudflare, Akamai</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Scope Immediately</div>
            <p className="text-[12px] text-stone-500 mt-0.5">"Design a CDN" is very broad. Clarify: are we designing the CDN infrastructure itself (like Cloudflare/Akamai), or the CDN integration layer for an application? Also clarify content types: static assets only, or video streaming too? For a 45-min interview, focus on <strong>static content delivery with caching + cache invalidation</strong>. Video streaming and edge compute are follow-ups.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Serve static content (images, JS, CSS, videos) from nearest edge location</Point>
            <Point icon="2." color="#059669">Cache content at edge with configurable TTL per content type</Point>
            <Point icon="3." color="#059669">Cache invalidation — purge specific URLs or wildcard patterns within seconds</Point>
            <Point icon="4." color="#059669">Origin pull — on cache miss, fetch from origin server and cache the response</Point>
            <Point icon="5." color="#059669">Support custom domains and TLS termination at the edge</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Low latency — p99 cache hit response &lt;50ms globally</Point>
            <Point icon="2." color="#dc2626">High throughput — handle millions of requests/sec across all PoPs</Point>
            <Point icon="3." color="#dc2626">High availability — 99.99% uptime; edge failures must be transparent to users</Point>
            <Point icon="4." color="#dc2626">Cache hit ratio — target 90-98% for static content</Point>
            <Point icon="5." color="#dc2626">Fast purge — cache invalidation propagates globally within 5 seconds</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask the Interviewer</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "What content types? Static assets, video streams, API responses, or all?",
            "Expected geographic distribution of users? Global or regional?",
            "Do we need real-time purge/invalidation or is TTL-based OK?",
            "What's the origin infrastructure? Single server, cloud, or multi-region?",
            "Do we need edge compute (like running custom logic at edge)?",
            "Expected traffic volume? Peak requests/sec, total bandwidth?",
            "Do we need to support live streaming or just on-demand?",
            "Security requirements? DDoS protection, WAF, signed URLs?",
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
            <p className="text-[12px] text-stone-500 mt-0.5">Capacity estimation for a CDN is about <strong>bandwidth and storage per PoP</strong>, not just QPS. Think about: total content size, hit ratio, origin bandwidth saved, and per-PoP cache size. Start with user-facing numbers and work backward to infrastructure.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Traffic Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Total daily active users" result="100M" note="Global service like a mid-large platform" />
            <MathStep step="2" formula="Avg requests per user per day" result="50" note="Page loads × assets per page (images, JS, CSS)" />
            <MathStep step="3" formula="Total daily requests = 100M × 50" result="5B/day" note="Five billion content requests per day" />
            <MathStep step="4" formula="Avg QPS = 5B / 86,400" result="~58K QPS" note="Average across the day" />
            <MathStep step="5" formula="Peak QPS = 3× average" result="~175K QPS" note="Peak hours traffic spike" final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Bandwidth Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Average object size" result="~100 KB" note="Mix of images (~200KB), JS/CSS (~30KB), fonts (~50KB)" />
            <MathStep step="2" formula="Peak bandwidth = 175K × 100 KB" result="~17.5 GB/s" note="That's ~140 Gbps at peak" />
            <MathStep step="3" formula="Cache hit ratio target" result="95%" note="Edge serves 95%, origin only 5%" />
            <MathStep step="4" formula="Origin bandwidth = 140 Gbps × 5%" result="~7 Gbps" note="Origin only needs to handle cache misses" />
            <MathStep step="5" formula="Daily data transferred = 5B × 100 KB" result="~500 TB/day" note="Massive — this is why you need a CDN" final />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Storage per PoP</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Total unique content objects" result="~10M" note="Images, scripts, stylesheets, videos" />
            <MathStep step="2" formula="Avg object size" result="~100 KB" note="Weighted average across types" />
            <MathStep step="3" formula="Total content corpus = 10M × 100 KB" result="~1 TB" note="Total unique content" />
            <MathStep step="4" formula="Hot content (Pareto: top 20%)" result="~200 GB" note="20% of objects serve 80% of traffic" />
            <MathStep step="5" formula="Per-PoP cache size (SSD + RAM)" result="~500 GB" note="Hot content + warm tail. SSD-backed." final />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — PoP & Cost Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Number of PoPs globally" result="200+" note="Major cities, IXPs, cloud regions" />
            <MathStep step="2" formula="Servers per PoP (avg)" result="10-50" note="Small PoP: 10, Large PoP: 100+" />
            <MathStep step="3" formula="Total edge servers = 200 × 30 avg" result="~6,000" note="Each handling local traffic for its region" />
            <MathStep step="4" formula="CDN cost (commercial)" result="~$0.02-0.08/GB" note="CloudFront/Cloudflare pricing at scale" />
            <MathStep step="5" formula="Monthly cost = 500TB/day × 30 × $0.03" result="~$450K/mo" note="At scale, negotiate to $0.01/GB → ~$150K/mo" final />
          </div>
          <div className="mt-4 pt-3 border-t border-stone-100">
            <div className="text-[11px] text-stone-500">
              <strong className="text-stone-700">Key point:</strong> Without a CDN, you'd need origin servers in every region to achieve similar latency, plus 20× the bandwidth capacity. CDN costs are a fraction of the alternative. Netflix's Open Connect Appliance program is even cheaper — ISPs host Netflix servers for free because it reduces their transit costs.
            </div>
          </div>
        </Card>
      </div>
      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Peak QPS", val: "~175K", sub: "Avg: ~58K" },
            { label: "Peak Bandwidth", val: "~140 Gbps", sub: "Origin: ~7 Gbps" },
            { label: "Per-PoP Cache", val: "~500 GB", sub: "SSD + memory" },
            { label: "Monthly Cost", val: "~$150-450K", sub: "500 TB/day egress" },
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
          <Label color="#2563eb">Content Request Flow</Label>
          <CodeBlock code={`# CDN Request — HTTP-based (standard HTTP/2)
# Client → Edge PoP → (miss) → Origin

# 1. Client requests an asset
GET /static/images/hero.webp HTTP/2
Host: cdn.example.com
Accept-Encoding: gzip, br
If-None-Match: "etag_abc123"

# 2. Edge PoP — Cache HIT
HTTP/2 200 OK
Content-Type: image/webp
Cache-Control: public, max-age=86400
X-Cache: HIT from edge-us-east-1
X-Edge-Location: IAD53
Age: 3421
ETag: "etag_abc123"
Content-Length: 145892

# 3. Edge PoP — Cache MISS (fetch from origin)
# Edge sends conditional request to origin
GET /static/images/hero.webp HTTP/1.1
Host: origin.example.com
If-None-Match: "etag_abc123"
X-CDN-Request-ID: req_xyz789`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Management APIs</Label>
          <div className="space-y-3">
            {[
              { op: "POST /purge", desc: "Invalidate cached content by URL, path pattern, or surrogate key.", perf: "Global propagation < 5s" },
              { op: "POST /distributions", desc: "Create a CDN distribution — origin config, domain, SSL cert, cache behavior rules.", perf: "Provision in ~15 min" },
              { op: "PUT /cache-policy", desc: "Set cache rules: TTL by content type, query string handling, header forwarding.", perf: "Propagates in ~60s" },
              { op: "GET /analytics", desc: "Real-time traffic, hit ratio, bandwidth, error rates per PoP.", perf: "~30s data freshness" },
              { op: "POST /signed-url", desc: "Generate time-limited, signed URL for premium/protected content.", perf: "O(1) — local signing" },
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
              <Point icon="→" color="#d97706">Use Cache-Control headers — standard HTTP caching semantics, not proprietary</Point>
              <Point icon="→" color="#d97706">Surrogate keys for cache tagging — purge all assets tagged "product-42" in one call</Point>
              <Point icon="→" color="#d97706">Vary header for content negotiation — cache different versions per Accept-Encoding, device type</Point>
              <Point icon="→" color="#d97706">Signed URLs for protected content — time-limited access without exposing credentials</Point>
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
    { label: "Single Origin", desc: "All users fetch directly from one origin server. Simple but slow for distant users (200ms+ cross-continent), and origin is a bottleneck and SPOF." },
    { label: "Pull-Based CDN", desc: "Edge PoPs cache content on first request (lazy population). Cache miss → fetch from origin → cache at edge. Simple to operate. This is how most CDNs work for static content." },
    { label: "Push-Based CDN", desc: "Origin proactively pushes content to edge PoPs before any user requests it. Good for content you know will be popular (e.g., new movie release). Higher origin upload cost but zero first-request latency." },
    { label: "Tiered Architecture", desc: "Two-tier caching: Edge PoPs (L1) → Regional Mid-Tier (L2) → Origin. Mid-tier shields the origin — cache misses from edge hit regional cache first. This is how CloudFront, Cloudflare, and Akamai actually work." },
  ];
  const diagrams = [
    () => (
      <svg viewBox="0 0 460 150" className="w-full">
        <DiagramBox x={70} y={50} w={80} h={36} label="User (US)" color="#2563eb"/>
        <DiagramBox x={230} y={50} w={80} h={36} label="User (EU)" color="#2563eb"/>
        <DiagramBox x={390} y={50} w={80} h={42} label="Origin\n(US-East)" color="#059669"/>
        <Arrow x1={110} y1={50} x2={350} y2={50} label="~20ms" id="l1"/>
        <Arrow x1={270} y1={50} x2={350} y2={50} label="~200ms" id="l2"/>
        <rect x={130} y={100} width={230} height={22} rx={6} fill="#dc262608" stroke="#dc262630"/>
        <text x={245} y={112} textAnchor="middle" fill="#dc2626" fontSize="9" fontFamily="monospace">✗ EU users suffer high latency, origin is bottleneck</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 170" className="w-full">
        <DiagramBox x={55} y={60} w={70} h={36} label="User" color="#2563eb"/>
        <DiagramBox x={185} y={60} w={80} h={42} label="Edge PoP\n(nearby)" color="#d97706"/>
        <DiagramBox x={350} y={60} w={80} h={42} label="Origin\nServer" color="#059669"/>
        <Arrow x1={90} y1={60} x2={145} y2={60} label="~20ms" id="p1"/>
        <Arrow x1={225} y1={60} x2={310} y2={60} label="miss → fetch" id="p2" dashed/>
        <rect x={120} y={115} width={220} height={20} rx={4} fill="#d9770608" stroke="#d9770630"/>
        <text x={230} y={126} textAnchor="middle" fill="#d97706" fontSize="8" fontFamily="monospace">✓ First request: slow (origin fetch). After: fast (cached)</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 170" className="w-full">
        <DiagramBox x={55} y={60} w={70} h={36} label="User" color="#2563eb"/>
        <DiagramBox x={185} y={60} w={80} h={42} label="Edge PoP\n(pre-loaded)" color="#d97706"/>
        <DiagramBox x={350} y={60} w={80} h={42} label="Origin\nServer" color="#059669"/>
        <Arrow x1={90} y1={60} x2={145} y2={60} label="~20ms" id="pu1"/>
        <Arrow x1={310} y1={50} x2={225} y2={50} label="push content" id="pu2"/>
        <rect x={120} y={115} width={220} height={20} rx={4} fill="#05966908" stroke="#05966930"/>
        <text x={230} y={126} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">✓ Zero first-request latency, higher origin egress</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 200" className="w-full">
        <DiagramBox x={50} y={75} w={65} h={34} label="User" color="#2563eb"/>
        <DiagramBox x={155} y={75} w={75} h={38} label="Edge PoP\n(L1)" color="#d97706"/>
        <DiagramBox x={295} y={75} w={85} h={38} label="Regional\nMid-Tier (L2)" color="#9333ea"/>
        <DiagramBox x={425} y={75} w={65} h={34} label="Origin" color="#059669"/>
        <Arrow x1={82} y1={75} x2={117} y2={75} id="t1"/>
        <Arrow x1={192} y1={75} x2={252} y2={75} label="L1 miss" id="t2" dashed/>
        <Arrow x1={337} y1={75} x2={392} y2={75} label="L2 miss" id="t3" dashed/>
        <rect x={90} y={130} width={300} height={18} rx={4} fill="#6366f108" stroke="#6366f130"/>
        <text x={240} y={140} textAnchor="middle" fill="#6366f1" fontSize="8" fontFamily="monospace">Edge: ~20ms | Mid-Tier: ~50ms | Origin: ~200ms</text>
        <rect x={90} y={155} width={300} height={18} rx={4} fill="#05966908" stroke="#05966930"/>
        <text x={240} y={165} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">✓ Mid-tier shields origin — reduces origin QPS by 10×</text>
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
        <Label color="#c026d3">Caching Strategies — The Core Tradeoff</Label>
        <p className="text-[12px] text-stone-500 mb-4">How you populate and manage cache at edge determines freshness, hit ratio, and origin load. This is the most important design decision in a CDN.</p>
        <div className="grid grid-cols-4 gap-4">
          {[
            { t: "Pull / Origin-Pull ★", d: "Edge fetches from origin on cache miss. Lazy population. Content cached on first request.", pros: ["Simple — no push infrastructure needed","Only caches content that's actually requested","Self-healing — expired content refreshed automatically"], cons: ["Cold start latency on first request","Origin thundering herd on popular content expiry"], pick: true },
            { t: "Push / Pre-Warm", d: "Origin proactively pushes content to edge PoPs before users request it. Eager population.", pros: ["Zero first-request latency","Predictable performance","Good for scheduled content releases"], cons: ["Origin must know which PoPs need content","Wasted storage for unpopular content","More complex orchestration"], pick: false },
            { t: "Tiered Caching ★", d: "L1 edge cache → L2 regional cache → Origin. Mid-tier absorbs misses from multiple edge PoPs.", pros: ["Shields origin from cache storms","Higher effective hit ratio","Reduces cross-region traffic"], cons: ["Adds ~20-40ms on L1 miss","More infrastructure to manage","Potential consistency delay"], pick: true },
            { t: "Stale-While-Revalidate", d: "Serve stale cached content immediately, refresh asynchronously in background. User never waits.", pros: ["Zero-latency responses always","No thundering herd on TTL expiry","Best user experience"], cons: ["Serves stale content briefly","Needs background refresh workers","More complex invalidation"], pick: false },
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
  const [sel, setSel] = useState("consistent");
  const algos = {
    consistent: { name: "Consistent Hashing (Request Routing) ★", cx: "O(log n)",
      pros: ["Minimal key redistribution when nodes join/leave — only K/N keys move","Even distribution with virtual nodes (100-200 per physical node)","Standard for CDN request routing to determine which edge server in a PoP handles a URL"],
      cons: ["Virtual nodes add memory overhead","Uneven load possible without enough virtual nodes","Doesn't account for node capacity differences without weighted vnodes"],
      when: "Default choice for routing requests to cache servers within a PoP. Used by Akamai, Varnish, and most CDN internals. Essential when nodes are added/removed frequently.",
      code: `# Consistent Hashing — CDN Request Routing
# Map URLs to edge servers within a PoP

RING_SIZE = 2^32  # Standard hash ring

class ConsistentHashRing:
    def __init__(self, nodes, vnodes=150):
        self.ring = SortedDict()
        for node in nodes:
            for i in range(vnodes):
                hash_val = md5(f"{node}:{i}") % RING_SIZE
                self.ring[hash_val] = node

    def get_node(self, url):
        hash_val = md5(url) % RING_SIZE
        # Find first node clockwise on the ring
        idx = self.ring.bisect_right(hash_val)
        if idx >= len(self.ring):
            idx = 0  # wrap around
        return self.ring.values()[idx]

# Usage in CDN edge:
ring = ConsistentHashRing(["edge-1","edge-2","edge-3"])
server = ring.get_node("/images/hero.webp")
# → "edge-2"  (deterministic for this URL)` },
    lru: { name: "LRU with TTL (Cache Eviction)", cx: "O(1) / O(n)",
      pros: ["Intuitive — evicts least recently accessed content first","O(1) operations with HashMap + Doubly Linked List","Respects TTL — expired content evicted regardless of recency","Works well for CDN workloads with temporal locality"],
      cons: ["Scan pollution — one-time crawlers evict hot content","Doesn't account for content size (large video evicts many small images)","TTL-only eviction can waste space on expired-but-not-yet-evicted entries"],
      when: "Default eviction policy for CDN edge caches. Combined with TTL — content expires after TTL, LRU handles the overflow when cache is full.",
      code: `# LRU + TTL — CDN Edge Cache Eviction
# Two-level eviction: TTL-based expiry + LRU on memory pressure

class CDNEdgeCache:
    def __init__(self, max_size_bytes):
        self.max_size = max_size_bytes
        self.current_size = 0
        self.hashmap = {}      # url → Node
        self.dll = DoublyLinkedList()  # LRU ordering

    def get(self, url):
        node = self.hashmap.get(url)
        if node is None:
            return None  # MISS

        # Check TTL
        if node.expires_at < now():
            self._evict(node)  # Lazy TTL expiry
            return None  # MISS (expired)

        self.dll.move_to_head(node)  # Mark as recently used
        return node.content  # HIT

    def put(self, url, content, ttl):
        # Evict until we have space
        while self.current_size + len(content) > self.max_size:
            self._evict(self.dll.tail)  # Evict LRU

        node = Node(url, content, expires_at=now()+ttl)
        self.dll.add_to_head(node)
        self.hashmap[url] = node
        self.current_size += len(content)` },
    geolb: { name: "GeoDNS + Anycast (Global Routing)", cx: "DNS-based",
      pros: ["Routes users to nearest PoP automatically via DNS resolution","Anycast: same IP announced from all PoPs — BGP routes to closest","Failover: if PoP goes down, BGP re-routes to next nearest within seconds"],
      cons: ["DNS TTL causes stickiness — can't react instantly to load changes","Anycast routing isn't perfectly geographic — follows BGP best path","Client DNS resolver location may differ from actual user location (VPNs, public DNS)"],
      when: "How users are routed to the correct PoP. Every CDN uses some combination of GeoDNS and Anycast. Cloudflare uses pure Anycast; AWS CloudFront uses GeoDNS with latency-based routing.",
      code: `# GeoDNS — Route to nearest PoP by user location
# DNS server returns different IPs based on client IP geolocation

# Example DNS resolution for cdn.example.com:
# User in New York → resolves to 198.51.100.1 (US-East PoP)
# User in London   → resolves to 203.0.113.1  (EU-West PoP)
# User in Tokyo    → resolves to 192.0.2.1    (AP-NE PoP)

# Anycast — Same IP, closest PoP via BGP
# All PoPs announce the same IP prefix (e.g., 198.51.100.0/24)
# Internet routing (BGP) naturally sends packets to closest PoP

# Latency-based routing (CloudFront model):
def resolve_cdn(client_ip):
    client_region = geoip_lookup(client_ip)
    candidate_pops = get_healthy_pops(client_region)

    # Measure latency from client region to each PoP
    # (pre-computed via periodic probes)
    best_pop = min(candidate_pops,
                   key=lambda p: latency_map[client_region][p])

    # Health check: if best PoP is degraded, try next
    if best_pop.health < 0.9:
        best_pop = candidate_pops[1]

    return best_pop.ip_address` },
    purge: { name: "Cache Invalidation (Purge Propagation)", cx: "Fanout O(P)",
      pros: ["Instant purge — content removed from all PoPs within seconds","Surrogate keys enable bulk invalidation (purge all assets for a product)","Soft purge (stale-while-revalidate) avoids thundering herd on mass invalidation"],
      cons: ["Global fanout is expensive — message to every PoP for every purge","Wildcard purge is slower (must scan local cache index)","Can trigger thundering herd if many clients request purged content simultaneously"],
      when: "When you need content freshness guarantees beyond TTL. Fastly is known for sub-second purge. Critical for e-commerce (price changes), news (corrections), and security (removing compromised content).",
      code: `# Cache Purge — Global Invalidation Flow
# 1. API call → Control Plane → fanout to all PoPs

# Purge by exact URL
POST /purge
{ "urls": ["https://cdn.example.com/images/hero.webp"] }

# Purge by surrogate key (cache tag)
POST /purge
{ "surrogate_keys": ["product-42", "category-shoes"] }

# Purge by wildcard pattern
POST /purge
{ "pattern": "/images/products/*" }

# Internal fanout mechanism:
def handle_purge(request):
    keys = resolve_purge_target(request)

    # Option A: Pub/sub fanout (Kafka / Redis Pub/Sub)
    for pop in all_pops:
        publish("purge_topic", {
            pop: pop.id,
            keys: keys,
            type: "hard"  # or "soft" (stale-while-revalidate)
        })

    # Each PoP consumer:
    def on_purge_message(msg):
        for key in msg.keys:
            local_cache.delete(key)
            metrics.increment("purge.executed")` },
  };
  const a = algos[sel];
  return (
    <div className="space-y-5">
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
          <Label color="#dc2626">Cache Key Design</Label>
          <CodeBlock code={`# Cache key structure for CDN edge
# Key = normalized URL + vary dimensions

# Basic: URL path as key
/images/hero.webp                → cached object

# With query string normalization
/api/products?sort=price&page=1  → normalize, sort params
/api/products?page=1&sort=price  → same key after normalization

# With Vary header dimensions
/images/hero.webp|br             → Brotli-compressed version
/images/hero.webp|gzip           → Gzip-compressed version
/images/hero.webp|mobile         → Device-targeted version

# Surrogate key mapping (for bulk purge)
surrogate:product-42 → [
  /images/products/42/thumb.webp,
  /images/products/42/hero.webp,
  /api/products/42
]

# TTL by content type
images/*        → TTL 86400s  (24 hours)
scripts/*.js    → TTL 31536000s (1 year, versioned filenames)
api/products/*  → TTL 60s     (1 minute, dynamic)`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Edge Cache Storage</Label>
          <div className="space-y-3">
            {[
              { fmt: "Memory (RAM)", size: "~50 GB per server", speed: "Fastest (~0.1ms)", pros: "Ultra-low latency for hot objects", cons: "Expensive per GB, limited capacity" },
              { fmt: "SSD (NVMe)", size: "~2-8 TB per server", speed: "Fast (~0.5ms)", pros: "High capacity, good throughput, cost-effective", cons: "Slower than RAM for random reads" },
              { fmt: "HDD (archival)", size: "~20+ TB per server", speed: "Slow (~5-10ms)", pros: "Very cheap per GB, massive capacity", cons: "Only for cold/long-tail content" },
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
              <strong className="text-stone-700">Recommendation:</strong> Multi-tier: RAM for top-1% hottest objects, NVMe SSD for everything else. Netflix and Akamai use this model. RAM acts as L1 within the edge server, SSD as L2. At Cloudflare scale, NVMe SSDs serve the vast majority of requests.
            </div>
          </div>
        </Card>
      </div>
      <Card accent="#9333ea">
        <Label color="#9333ea">Cache-Control Headers — The Standard Interface</Label>
        <p className="text-[12px] text-stone-500 mb-4">HTTP Cache-Control is the universal language between origin, CDN, and browser. Mastering these headers is critical for a CDN design interview.</p>
        <div className="grid grid-cols-3 gap-4">
          {[
            { t: "Cache-Control: public, max-age=86400 ★", d: "Content is cacheable by CDN and browsers. Fresh for 24 hours. Most common for static assets.", pros: ["Simple and widely supported","Both CDN and browser cache","Reduces origin load significantly"], cons: ["Can't update content before TTL expires without purge","Stale content served until expiry"], pick: true },
            { t: "s-maxage=3600, max-age=60", d: "CDN caches for 1 hour, browsers only 1 minute. Gives CDN more cache time while keeping browser fresh.", pros: ["Fine-grained control per layer","CDN absorbs more traffic","Faster browser freshness"], cons: ["More complex header logic","Must understand s-maxage semantics"], pick: false },
            { t: "stale-while-revalidate=60", d: "Serve stale content for up to 60s while refreshing in background. User never waits for origin.", pros: ["Zero-latency on TTL boundary","No thundering herd","Best user experience"], cons: ["Brief stale window possible","Origin still gets refresh requests","Not all CDNs support it fully"], pick: false },
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
            <strong className="text-stone-700">Best practice:</strong> Use long max-age + content-addressable URLs for immutable assets (app.a1b2c3.js — filename includes hash). For mutable content, use short TTL + stale-while-revalidate + purge API as backup. This gives you both performance and freshness.
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
          <Label color="#059669">Horizontal Scaling — Adding PoPs</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Geographic expansion</strong> — add PoPs in underserved regions. Reduces latency for users in those areas. Each PoP operates independently with its own cache.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">PoP capacity scaling</strong> — add more servers within a PoP when traffic exceeds local capacity. Consistent hashing redistributes only 1/N of requests to the new server.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Anycast for automatic distribution</strong> — announce the same IP from new PoPs. BGP automatically routes nearby users to them with zero DNS changes.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">IXP peering</strong> — colocate at Internet Exchange Points for direct peering with ISPs. Reduces hops and transit costs. Major CDNs are present at 1000+ IXPs.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Tiered Caching Architecture</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">L1 — Edge PoP</strong> — closest to users (~200 PoPs). Small cache per PoP. Serves 90%+ of requests. Low latency (~20ms).</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">L2 — Regional Shield</strong> — mid-tier regional caches (~10-20 locations). Aggregates misses from multiple edge PoPs. Much larger cache. Serves 95%+ when combined with L1.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Origin Shield</strong> — single cache layer in front of origin. Collapses duplicate origin requests from all regional caches into one. Protects origin from thundering herd.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Request collapsing</strong> — when multiple edge servers request the same URL from mid-tier simultaneously, collapse into one origin request. All waiters get the same response.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Strategy</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Anycast Global Distribution ★", d:"Same IP prefix announced from all PoPs worldwide. Internet routing (BGP) naturally directs users to the closest PoP. No DNS changes needed to add/remove PoPs.", pros:["Automatic failover via BGP re-convergence","Simple — no DNS management needed","Works seamlessly with new PoP deployments"], cons:["BGP convergence can take 30-90 seconds","Routing is by BGP best path, not strictly geographic","Can't do weighted routing easily"], pick:true },
            { t:"GeoDNS with Latency-Based Routing", d:"DNS server resolves to different PoP IPs based on client IP geolocation. More control than Anycast over routing decisions.", pros:["Precise geographic routing","Can weight traffic to specific PoPs","Can avoid sending traffic to overloaded PoPs"], cons:["DNS TTL limits agility (minutes to shift)","Client DNS resolver location ≠ user location","More complex DNS infrastructure needed"], pick:false },
            { t:"Hybrid (Anycast + GeoDNS)", d:"GeoDNS routes to regional Anycast groups. Each region uses Anycast within its PoPs. Best of both — geographic control with automatic failover.", pros:["Regional control + PoP-level automatic failover","Cloudflare and Fastly use this model","Most resilient to failures"], cons:["Most complex to operate","Requires both DNS and BGP infrastructure","Harder to debug routing issues"], pick:false },
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
        <Label color="#d97706">Critical Decision: What Happens When a PoP Goes Down?</Label>
        <p className="text-[12px] text-stone-500 mb-4">Unlike a centralized service, a CDN is designed to survive PoP failures transparently. The key question is how fast traffic re-routes and whether users notice any disruption.</p>
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Anycast Auto-Failover (Recommended)</div>
            <p className="text-[11px] text-stone-500 mb-2">PoP stops announcing BGP route → traffic automatically shifts to next-closest PoP. No DNS changes needed.</p>
            <ul className="space-y-1"><Point icon="✓" color="#059669">Transparent to users — BGP re-converges in 30-90s</Point><Point icon="✓" color="#059669">No central coordinator needed</Point><Point icon="⚠" color="#d97706">Cold cache at failover PoP — temporary miss spike</Point></ul>
          </div>
          <div className="rounded-lg border border-red-200 bg-white p-4">
            <div className="text-[11px] font-bold text-red-600 mb-1.5">Origin Overload on Mass PoP Failure</div>
            <p className="text-[11px] text-stone-500 mb-2">Multiple PoPs fail simultaneously → all traffic hits origin → origin overloaded → cascading failure</p>
            <ul className="space-y-1"><Point icon="→" color="#d97706">Origin shield absorbs burst, rate-limits origin requests</Point><Point icon="→" color="#d97706">Stale-if-error: serve cached (stale) content if origin is down</Point><Point icon="→" color="#d97706">Custom error pages at edge — never show raw errors</Point></ul>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#2563eb">Edge Health & Failover</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#2563eb"><strong className="text-stone-700">Active health checks</strong> — control plane pings each PoP every 10-30s. If a PoP fails 3 consecutive checks, withdraw its BGP route and update DNS.</Point>
            <Point icon="2." color="#2563eb"><strong className="text-stone-700">Real User Monitoring (RUM)</strong> — JavaScript beacons from actual users report latency and errors. Detects issues health checks miss (partial failures, slow responses).</Point>
            <Point icon="3." color="#2563eb"><strong className="text-stone-700">Graceful degradation</strong> — if a PoP is degraded but not dead, reduce its weight in routing rather than fully withdrawing. Serve from it at reduced capacity.</Point>
            <Point icon="4." color="#2563eb"><strong className="text-stone-700">Connection draining</strong> — when taking a server out of rotation, let existing connections finish (30s drain) before removing it. No abrupt disconnects.</Point>
          </ul>
        </Card>
        <Card>
          <Label color="#0891b2">Stale Content Strategies</Label>
          <ul className="space-y-2.5">
            <Point icon="🛡️" color="#0891b2"><strong className="text-stone-700">stale-if-error</strong> — if origin returns 5xx, serve stale cached content with a header indicating staleness. Better to show slightly old content than an error page.</Point>
            <Point icon="📊" color="#0891b2"><strong className="text-stone-700">stale-while-revalidate</strong> — serve stale content immediately, refresh asynchronously in background. Zero user-facing latency on TTL boundary.</Point>
            <Point icon="🔄" color="#0891b2"><strong className="text-stone-700">Origin failover</strong> — configure backup origin. If primary origin is down, edge fetches from secondary origin (possibly in a different region).</Point>
            <Point icon="📈" color="#0891b2"><strong className="text-stone-700">Edge-side error pages</strong> — store custom error pages at edge. If origin is unreachable, return branded error page instead of generic 502.</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#9333ea">Degradation Ladder</Label>
        <div className="flex gap-3 items-stretch mt-1">
          {[
            { label: "All PoPs Healthy", sub: "Normal operation", color: "#059669", status: "HEALTHY" },
            { label: "PoP Failure", sub: "BGP re-routes traffic", color: "#d97706", status: "DEGRADED" },
            { label: "Origin Degraded", sub: "Serve stale content", color: "#ea580c", status: "FALLBACK" },
            { label: "Origin Down", sub: "Stale + error pages", color: "#dc2626", status: "EMERGENCY" },
          ].map((t,i) => (
            <div key={i} className="flex-1 flex items-center gap-2">
              {i > 0 && <span className="text-stone-300 text-lg shrink-0">→</span>}
              <div className="flex-1 rounded-lg border p-3 text-center" style={{ borderColor: t.color+"40", background: t.color+"08" }}>
                <Pill bg={t.color+"20"} color={t.color}>{t.status}</Pill>
                <div className="text-[11px] font-bold text-stone-700 mt-2">{t.label}</div>
                <div className="text-[10px] text-stone-400">{t.sub}</div>
              </div>
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
          <Label color="#0284c7">Key Metrics — The CDN Dashboard</Label>
          <ul className="space-y-2.5">
            <Point icon="📊" color="#0284c7"><strong className="text-stone-700">Cache Hit Ratio</strong> — #1 metric. Target: 90-98%. Below 85% = investigate TTLs, cache key design, or content mix.</Point>
            <Point icon="⏱️" color="#0284c7"><strong className="text-stone-700">TTFB (Time to First Byte)</strong> — p50: &lt;20ms, p99: &lt;100ms for cache hits. High TTFB on hits = server overload or I/O issue.</Point>
            <Point icon="📈" color="#0284c7"><strong className="text-stone-700">Origin Request Rate</strong> — should be 2-10% of total QPS. Spike = cache miss storm or purge event.</Point>
            <Point icon="🌍" color="#0284c7"><strong className="text-stone-700">Bandwidth per PoP</strong> — track egress per PoP to detect regional traffic shifts and capacity issues.</Point>
            <Point icon="❌" color="#0284c7"><strong className="text-stone-700">Error Rate (4xx/5xx)</strong> — split by edge vs origin. Edge 5xx = CDN issue. Origin 5xx = backend issue.</Point>
          </ul>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Logging & Tracing</Label>
          <ul className="space-y-2.5">
            <Point icon="📝" color="#059669"><strong className="text-stone-700">Edge access logs</strong> — every request logged: URL, status, cache status (HIT/MISS/STALE), TTFB, PoP ID, client IP. Shipped to central SIEM in near-real-time.</Point>
            <Point icon="🔍" color="#059669"><strong className="text-stone-700">Request tracing</strong> — X-CDN-Request-ID header propagated from edge → mid-tier → origin. End-to-end latency breakdown per hop.</Point>
            <Point icon="📊" color="#059669"><strong className="text-stone-700">Real User Monitoring</strong> — JavaScript SDK on client reports actual load times, TTFB, errors. Detects issues invisible to server-side metrics.</Point>
            <Point icon="🔔" color="#059669"><strong className="text-stone-700">Alerting</strong> — alert on: hit ratio &lt;90%, origin request spike &gt;2×, error rate &gt;1%, TTFB p99 &gt;200ms, PoP health check failure.</Point>
          </ul>
        </Card>
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Cache Analytics</Label>
          <ul className="space-y-2.5">
            <Point icon="🔥" color="#7c3aed"><strong className="text-stone-700">Hot object tracking</strong> — top-N most requested URLs per PoP. Identifies candidates for pre-warming or special handling.</Point>
            <Point icon="📉" color="#7c3aed"><strong className="text-stone-700">Eviction metrics</strong> — track eviction rate, reasons (TTL vs LRU vs purge). High LRU evictions = cache too small.</Point>
            <Point icon="💾" color="#7c3aed"><strong className="text-stone-700">Storage utilization</strong> — RAM and SSD usage per PoP. Alert at 80% to trigger capacity planning.</Point>
            <Point icon="🗺️" color="#7c3aed"><strong className="text-stone-700">Geographic heat map</strong> — visualize traffic and performance by region. Identifies underserved areas needing new PoPs.</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#0284c7">Sample X-Cache Headers (Debugging)</Label>
        <CodeBlock code={`# Cache HIT — served from edge
X-Cache: HIT from edge-us-east-1
X-Cache-Hits: 847
Age: 14523
X-Edge-Location: IAD53-P2

# Cache MISS — fetched from origin
X-Cache: MISS from edge-eu-west-1
X-Origin-Latency: 142ms
X-Cache-Key: /images/hero.webp|br|desktop

# Cache STALE — served stale, refreshing async
X-Cache: STALE from edge-ap-ne-1
X-Stale-Reason: stale-while-revalidate
X-Revalidation-Status: pending

# Cache BYPASS — not cacheable
X-Cache: BYPASS from edge-us-west-2
X-Bypass-Reason: Cache-Control: no-store`} />
      </Card>
    </div>
  );
}

function WatchoutsSection() {
  return (
    <div className="space-y-5">
      <Card accent="#dc2626">
        <Label color="#dc2626">Critical Failure Modes</Label>
        <div className="grid grid-cols-2 gap-4">
          {[
            { mode: "Thundering Herd on Purge", impact: "CRITICAL", desc: "Mass purge of popular content → all PoPs request origin simultaneously → origin overloaded → cascading failure.",
              mitigation: "Use soft purge (stale-while-revalidate). Request collapsing at mid-tier. Rate-limit origin requests. Pre-warm after purge.",
              example: "E-commerce site purges all product images before sale → 200 PoPs simultaneously request from origin → origin crashes." },
            { mode: "Origin Single Point of Failure", impact: "HIGH", desc: "Origin server goes down → all cache misses fail → new content can't be served → stale content gradually expires.",
              mitigation: "Multi-origin failover. Origin shield to reduce origin load. stale-if-error to serve stale content. Custom error pages at edge.",
              example: "AWS us-east-1 outage takes down origin → CDN serves stale content for TTL duration → new requests fail." },
            { mode: "Cache Poisoning", impact: "CRITICAL", desc: "Attacker tricks CDN into caching malicious content (via manipulated Host header, query params, or Vary header abuse).",
              mitigation: "Strict cache key normalization. Whitelist query params. Validate Host header. Pin cache keys to origin-defined rules only.",
              example: "Attacker sends request with Host: evil.com → CDN caches response keyed by Host → legitimate users see attacker's content." },
            { mode: "DNS Propagation Delay", impact: "MEDIUM", desc: "GeoDNS update takes minutes to propagate due to DNS TTL. During this window, some users route to old/failed PoP.",
              mitigation: "Use Anycast (instant re-routing via BGP). Keep DNS TTL low (60s). Use health-check-triggered DNS updates.",
              example: "PoP goes down, DNS updated, but clients cached old DNS for 300s → 5 minutes of failed requests." },
            { mode: "Cache Stampede on Cold Start", impact: "HIGH", desc: "New PoP deployed with empty cache → 100% miss rate → all requests hit origin → origin overloaded.",
              mitigation: "Pre-warm cache from mid-tier before routing traffic. Gradual traffic shift (1% → 10% → 100%). Request collapsing.",
              example: "New PoP in São Paulo goes live → 50K concurrent users get cache misses → origin swamped with 50K identical requests." },
            { mode: "TLS Certificate Expiry", impact: "HIGH", desc: "Edge TLS certificate expires → browsers show security warnings → all HTTPS traffic blocked.",
              mitigation: "Automated cert management (Let's Encrypt / ACM). Alert 30 days before expiry. Certificate pinning only with rotation plan.",
              example: "Custom domain cert expires at 3am Saturday → site shows 'connection not secure' → revenue loss until on-call rotates cert." },
          ].map((f,i) => (
            <div key={i} className="rounded-lg border border-stone-200 p-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-[12px] font-bold text-stone-800">{f.mode}</span>
                <Pill bg={f.impact==="CRITICAL"?"#fef2f2":"#fffbeb"} color={f.impact==="CRITICAL"?"#dc2626":"#d97706"}>{f.impact}</Pill>
              </div>
              <p className="text-[11px] text-stone-500 mb-2">{f.desc}</p>
              <div className="text-[10px] text-emerald-600 mb-1"><strong>Mitigation:</strong> {f.mitigation}</div>
              <div className="text-[10px] italic text-stone-400">Example: {f.example}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function ServicesSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#0f766e">
          <Label color="#0f766e">Service Breakdown</Label>
          <div className="space-y-3">
            {[
              { name: "Edge Proxy (Nginx/Envoy)", role: "Handles incoming requests: TLS termination, cache lookup, origin fetch on miss, response delivery. The core data path.", tech: "Nginx + Lua / Varnish / Envoy", critical: true },
              { name: "Cache Manager", role: "Manages local cache storage: eviction (LRU/TTL), disk I/O, memory allocation, cache index maintenance.", tech: "Custom daemon per edge server", critical: true },
              { name: "Origin Fetch Service", role: "Handles cache miss → origin requests. Connection pooling, retries, circuit breaker, request collapsing.", tech: "HTTP client with connection pool", critical: true },
              { name: "Purge Controller", role: "Receives purge commands from control plane, executes local cache deletion, reports completion.", tech: "gRPC listener + local cache API", critical: true },
              { name: "Control Plane", role: "Central management: routing config, PoP health, purge fanout, TLS cert distribution, traffic shifting.", tech: "gRPC + etcd/Consul", critical: false },
              { name: "Health Checker", role: "Probes each PoP and edge server. Reports health to control plane for routing decisions and BGP withdrawal.", tech: "Distributed probe agents", critical: false },
              { name: "Analytics Pipeline", role: "Collects edge logs, aggregates metrics, feeds dashboards and alerting. Near-real-time processing.", tech: "Kafka → Flink → ClickHouse", critical: false },
              { name: "Certificate Manager", role: "Provisions, renews, and distributes TLS certificates to all edge PoPs. Handles custom domains.", tech: "Let's Encrypt / ACM + push to edges", critical: true },
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
          <Label color="#9333ea">Edge Server Internals — Block Diagram</Label>
          <svg viewBox="0 0 380 320" className="w-full">
            {/* TLS + HTTP */}
            <rect x={10} y={10} width={360} height={45} rx={6} fill="#6366f108" stroke="#6366f1" strokeWidth={1}/>
            <text x={190} y={28} textAnchor="middle" fill="#6366f1" fontSize="10" fontWeight="600" fontFamily="monospace">TLS Termination + HTTP/2 Handler</text>
            <text x={190} y={43} textAnchor="middle" fill="#6366f180" fontSize="8" fontFamily="monospace">SNI routing · cert selection · request parsing · compression</text>

            {/* Cache Decision */}
            <rect x={10} y={65} width={360} height={40} rx={6} fill="#9333ea08" stroke="#9333ea" strokeWidth={1}/>
            <text x={190} y={83} textAnchor="middle" fill="#9333ea" fontSize="10" fontWeight="600" fontFamily="monospace">Cache Decision Engine</text>
            <text x={190} y={96} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">cache key build → lookup → HIT/MISS/STALE → route decision</text>

            {/* Cache Store + Origin Fetch */}
            <rect x={10} y={115} width={175} height={70} rx={6} fill="#d9770608" stroke="#d97706" strokeWidth={1}/>
            <text x={97} y={138} textAnchor="middle" fill="#d97706" fontSize="10" fontWeight="600" fontFamily="monospace">Cache Store</text>
            <text x={97} y={153} textAnchor="middle" fill="#d9770680" fontSize="8" fontFamily="monospace">RAM L1 + SSD L2</text>
            <text x={97} y={168} textAnchor="middle" fill="#d9770680" fontSize="8" fontFamily="monospace">LRU eviction · TTL expiry</text>

            <rect x={195} y={115} width={175} height={70} rx={6} fill="#05966908" stroke="#059669" strokeWidth={1}/>
            <text x={282} y={138} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="600" fontFamily="monospace">Origin Fetch</text>
            <text x={282} y={153} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">connection pool · retry</text>
            <text x={282} y={168} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">request collapsing</text>

            {/* Purge + Metrics */}
            <rect x={10} y={195} width={175} height={55} rx={6} fill="#dc262608" stroke="#dc2626" strokeWidth={1}/>
            <text x={97} y={215} textAnchor="middle" fill="#dc2626" fontSize="10" fontWeight="600" fontFamily="monospace">Purge Handler</text>
            <text x={97} y={230} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">subscribe purge topic</text>
            <text x={97} y={242} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">delete from local cache</text>

            <rect x={195} y={195} width={175} height={55} rx={6} fill="#0891b208" stroke="#0891b2" strokeWidth={1}/>
            <text x={282} y={215} textAnchor="middle" fill="#0891b2" fontSize="10" fontWeight="600" fontFamily="monospace">Metrics & Logging</text>
            <text x={282} y={230} textAnchor="middle" fill="#0891b280" fontSize="8" fontFamily="monospace">access logs → Kafka</text>
            <text x={282} y={242} textAnchor="middle" fill="#0891b280" fontSize="8" fontFamily="monospace">Prometheus exporter</text>

            {/* Health + Config */}
            <rect x={10} y={260} width={360} height={50} rx={6} fill="#78716c08" stroke="#78716c" strokeWidth={1}/>
            <text x={190} y={280} textAnchor="middle" fill="#78716c" fontSize="10" fontWeight="600" fontFamily="monospace">Config & Health</text>
            <text x={100} y={298} textAnchor="middle" fill="#78716c80" fontSize="8" fontFamily="monospace">Config sync from control plane</text>
            <text x={280} y={298} textAnchor="middle" fill="#78716c80" fontSize="8" fontFamily="monospace">Health check endpoint (/healthz)</text>

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
                { route: "Client → Edge PoP", proto: "HTTPS/2", contract: "Standard HTTP request with cache headers", timeout: "30s (client)", fail: "DNS failover to next-closest PoP" },
                { route: "Edge → Mid-Tier", proto: "HTTP/2 (internal)", contract: "Origin-pull with X-CDN-Request-ID", timeout: "5s", fail: "Fetch directly from origin (skip mid-tier)" },
                { route: "Edge → Origin", proto: "HTTPS/1.1 or HTTP/2", contract: "Conditional request (If-None-Match)", timeout: "10s", fail: "Serve stale if available, else 502" },
                { route: "Control Plane → Edge", proto: "gRPC + Push", contract: "Config update, purge command, cert push", timeout: "5s (per-PoP)", fail: "Retry with backoff, alert if 3 consecutive failures" },
                { route: "Edge → Analytics", proto: "Kafka producer", contract: "Access log events (async)", timeout: "N/A (fire-and-forget)", fail: "Buffer locally, retry. Drop after buffer full." },
                { route: "Health Checker → Edge", proto: "HTTP GET /healthz", contract: "200 OK if healthy", timeout: "3s", fail: "Mark unhealthy after 3 consecutive failures" },
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
  const [flow, setFlow] = useState("hit");
  const flows = {
    hit: {
      title: "Cache HIT (Happy Path)",
      steps: [
        { actor: "Client", action: "DNS resolution: cdn.example.com → Anycast IP (nearest PoP)", type: "request" },
        { actor: "Edge PoP", action: "TLS handshake (TLS 1.3 — 1-RTT). Reuse session ticket if available.", type: "auth" },
        { actor: "Edge Proxy", action: "Parse HTTP/2 request: GET /images/hero.webp", type: "process" },
        { actor: "Edge Proxy", action: "Build cache key: /images/hero.webp|br (Brotli-compressed)", type: "process" },
        { actor: "Cache Store", action: "Key lookup: found in RAM L1 cache. Check TTL: not expired ✓", type: "success" },
        { actor: "Edge Proxy", action: "Set response headers: X-Cache: HIT, Age: 3421, ETag: abc123", type: "process" },
        { actor: "Edge Proxy", action: "Stream response to client. Total latency: ~15ms", type: "success" },
        { actor: "Metrics", action: "Increment: cache.hit_count, edge.requests_total, edge.bytes_out", type: "process" },
      ]
    },
    miss: {
      title: "Cache MISS → Origin Fetch → Cache Store",
      steps: [
        { actor: "Cache Store", action: "Key lookup: NOT FOUND (cache MISS)", type: "error" },
        { actor: "Edge Proxy", action: "Check request collapsing: is another request for same URL in-flight?", type: "check" },
        { actor: "Edge Proxy", action: "No in-flight request. Acquire fetch lock for this URL.", type: "process" },
        { actor: "Edge → Mid-Tier", action: "HTTP GET /images/hero.webp (X-CDN-Request-ID: req_xyz, timeout: 5s)", type: "request" },
        { actor: "Mid-Tier Cache", action: "Lookup: FOUND in mid-tier cache. Return 200 with content.", type: "success" },
        { actor: "Edge Proxy", action: "Store in local cache with TTL from Cache-Control header", type: "process" },
        { actor: "Edge Proxy", action: "Stream response to client. Total latency: ~50ms (edge miss, mid-tier hit)", type: "success" },
        { actor: "Edge Proxy", action: "Release fetch lock. If other waiters, serve them from now-cached content.", type: "process" },
      ]
    },
    purge: {
      title: "Cache Purge (Global Invalidation)",
      steps: [
        { actor: "Admin", action: "POST /purge {urls: ['/images/hero.webp']} to Control Plane API", type: "request" },
        { actor: "Control Plane", action: "Validate request, authenticate caller, log purge event", type: "auth" },
        { actor: "Control Plane", action: "Publish purge message to Kafka topic: cdn.purge.commands", type: "process" },
        { actor: "All PoPs", action: "Each PoP's purge controller consumes message from Kafka", type: "process" },
        { actor: "Purge Controller", action: "Delete /images/hero.webp from local cache (all compression variants)", type: "process" },
        { actor: "Purge Controller", action: "ACK message, report completion to control plane", type: "success" },
        { actor: "Control Plane", action: "Aggregate ACKs. All 200 PoPs confirmed. Total propagation: ~3 seconds.", type: "success" },
        { actor: "Next Request", action: "Client requests /images/hero.webp → MISS → fetches fresh from origin", type: "process" },
      ]
    },
    failure: {
      title: "Failure Path — Origin Down",
      steps: [
        { actor: "Cache Store", action: "Key lookup: expired (TTL exceeded 300s ago)", type: "error" },
        { actor: "Edge → Origin", action: "HTTP GET /images/hero.webp → connection timeout (10s)", type: "error" },
        { actor: "Edge Proxy", action: "Retry once → second timeout", type: "error" },
        { actor: "Edge Proxy", action: "Circuit breaker OPEN for origin. Half-open retry in 30s.", type: "error" },
        { actor: "Edge Proxy", action: "Check stale-if-error policy: stale content available ✓", type: "check" },
        { actor: "Edge Proxy", action: "Serve stale content with headers: X-Cache: STALE, X-Stale-Reason: origin-error", type: "success" },
        { actor: "Edge Proxy", action: "Return to client. User sees content (slightly stale). Latency: ~20ms.", type: "success" },
        { actor: "Health Checker", action: "Detect origin failure → alert OPS team → switch to backup origin if configured", type: "process" },
      ]
    },
  };
  const colors = { request:"#2563eb", auth:"#7c3aed", process:"#64748b", success:"#059669", error:"#dc2626", check:"#d97706" };
  const f = flows[flow];
  return (
    <div className="space-y-5">
      <div className="flex gap-2">
        {Object.entries(flows).map(([k,v]) => (
          <button key={k} onClick={() => setFlow(k)}
            className={`px-3.5 py-1.5 rounded-lg text-[12px] font-medium transition-all border ${k===flow?"bg-purple-600 text-white border-purple-600":"bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
            {v.title}
          </button>
        ))}
      </div>
      <Card accent="#7e22ce">
        <Label color="#7e22ce">{f.title}</Label>
        <div className="space-y-0">
          {f.steps.map((s,i) => (
            <div key={i} className="flex items-start gap-3 py-2.5 border-b border-stone-100 last:border-0">
              <span className="text-[10px] font-bold w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5" style={{ background: colors[s.type]+"20", color: colors[s.type] }}>{i+1}</span>
              <span className="text-[11px] font-mono font-bold shrink-0 w-28" style={{ color: colors[s.type] }}>{s.actor}</span>
              <span className="text-[12px] text-stone-600">{s.action}</span>
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
          <Label color="#b45309">Edge Deployment Strategy</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#b45309"><strong className="text-stone-700">Canary rollouts</strong> — deploy new edge config to 1 PoP first. Monitor for 30 min. Then 5% → 25% → 100% of PoPs over 2 hours.</Point>
            <Point icon="2." color="#b45309"><strong className="text-stone-700">Blue-green per PoP</strong> — each PoP has standby servers. Deploy to standby, health check, then swap traffic. Instant rollback by swapping back.</Point>
            <Point icon="3." color="#b45309"><strong className="text-stone-700">Config-as-code</strong> — cache rules, routing config, and TLS certs managed in version control. Changes trigger automated deployment pipeline.</Point>
            <Point icon="4." color="#b45309"><strong className="text-stone-700">Immutable edge images</strong> — edge servers boot from immutable image with all software baked in. Config is the only variable. Rebuild, don't patch.</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Security at the Edge</Label>
          <ul className="space-y-2.5">
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">TLS everywhere</strong> — terminate TLS at edge (TLS 1.3, ECDSA certs). HSTS header with max-age=31536000. OCSP stapling for fast cert validation.</Point>
            <Point icon="🛡️" color="#dc2626"><strong className="text-stone-700">DDoS mitigation</strong> — edge absorbs volumetric attacks (Tbps capacity). Rate limiting per IP. SYN flood protection via SYN cookies. Challenge pages for suspicious traffic.</Point>
            <Point icon="🔑" color="#dc2626"><strong className="text-stone-700">Signed URLs / tokens</strong> — protect premium content with time-limited signed URLs. HMAC signature includes expiry timestamp and client IP binding.</Point>
            <Point icon="🧱" color="#dc2626"><strong className="text-stone-700">WAF at edge</strong> — Web Application Firewall rules at edge block SQLi, XSS, bot traffic before it reaches origin. Managed rulesets + custom rules.</Point>
            <Point icon="🌐" color="#dc2626"><strong className="text-stone-700">Origin shield</strong> — origin only accepts requests from CDN IPs (whitelist). Mutual TLS between edge and origin for additional authentication.</Point>
          </ul>
        </Card>
      </div>
    </div>
  );
}

function OpsSection() {
  return (
    <div className="space-y-5">
      <Card accent="#be123c">
        <Label color="#be123c">Auto-Scaling Triggers</Label>
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
                { trigger: "PoP Bandwidth", thresh: "> 80% capacity for 10min", action: "Add edge servers to PoP (horizontal scale)", cool: "15 min", pitfall: "New server has cold cache — pre-warm before routing traffic" },
                { trigger: "Cache Hit Ratio", thresh: "< 85% for 15min", action: "Investigate TTLs, increase cache size, check for cache-busting query params", cool: "30 min", pitfall: "Low hit ratio may be normal for long-tail content — check by content type" },
                { trigger: "Origin Request Rate", thresh: "> 3× baseline for 5min", action: "Check for mass purge, TTL expiry storm, or new uncacheable content", cool: "10 min", pitfall: "Could be legitimate traffic spike — don't auto-block, investigate first" },
                { trigger: "Error Rate (5xx)", thresh: "> 1% for 5min", action: "Check origin health, enable stale-if-error, activate backup origin", cool: "5 min", pitfall: "May be origin issue, not CDN — check edge vs origin error split" },
                { trigger: "TTFB p99", thresh: "> 200ms for 10min", action: "Check disk I/O, memory pressure, connection pool exhaustion", cool: "10 min", pitfall: "High TTFB on MISSes is expected — only alert on HITs" },
                { trigger: "TLS Cert Expiry", thresh: "< 30 days remaining", action: "Trigger automated renewal, alert if auto-renewal fails", cool: "24 hr", pitfall: "Custom domain certs may not auto-renew — verify domain validation" },
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
            { title: "Cache Key Explosion", symptom: "Hit ratio drops from 95% to 30%. Cache evictions spike. Origin overwhelmed.",
              cause: "Marketing added unique tracking params (?utm_source=..., ?fbclid=...) to every URL. Each unique query string = new cache key = cache miss.",
              fix: "Strip non-essential query params from cache key. Whitelist only params that affect content. Configure CDN to ignore marketing/tracking params.",
              quote: "One ?fbclid parameter turned our 95% cache hit ratio into 30% overnight. Every Facebook click was a unique cache key." },
            { title: "Negative Caching Mistake", symptom: "Origin returns 404 for new content. CDN caches the 404. Content exists but users see 404 for hours.",
              cause: "CDN caches error responses (404, 500) with same TTL as success responses. New content deployed to origin but CDN serves cached 404.",
              fix: "Set short TTL for error responses (Cache-Control: max-age=0 on 404/500). Or don't cache errors at all. Use surrogate key to purge on deploy.",
              quote: "We launched a product page and it showed 404 for 2 hours because CDN cached the 404 from the pre-launch check. $200K in lost revenue." },
            { title: "Hot Object Thundering Herd", symptom: "Celebrity posts viral tweet with image link. Single origin image receives 500K requests in 10 seconds.",
              cause: "Image TTL expires at the same time across all PoPs. All 200 PoPs request same image from origin simultaneously.",
              fix: "Add jitter to TTL (base_ttl + random(0, 60s)). Request collapsing at mid-tier. Stale-while-revalidate. Pre-detect viral content via traffic spike detection.",
              quote: "A K-pop star tweeted a selfie. Our origin got 500K req/s for one JPEG. Request collapsing saved us — only 200 actual origin requests (one per PoP)." },
            { title: "Vary Header Misconfiguration", symptom: "Different users see wrong language/compression. Cache serves German page to English users.",
              cause: "Origin sends Vary: Accept-Language but CDN doesn't include Accept-Language in cache key. All languages cached under same key.",
              fix: "Ensure Vary header dimensions are included in cache key. Or normalize at edge: map Accept-Language to a small set of variants (en, de, fr, etc.).",
              quote: "We Vary'd on User-Agent. 10,000 unique User-Agents = 10,000 cache variants per URL = 0.01% hit ratio. Normalize, don't Vary on the raw header." },
            { title: "Origin IP Leaked", symptom: "Attacker bypasses CDN, hits origin directly. DDoS takes down origin despite CDN protection.",
              cause: "Origin IP exposed via DNS history, email headers (MX records), or error pages showing origin IP address.",
              fix: "Whitelist only CDN IPs on origin firewall. Use private connectivity (AWS PrivateLink) between CDN and origin. Never expose origin IP in any public record.",
              quote: "Attacker found our origin IP from a 2-year-old DNS record. Bypassed CDN entirely and DDoS'd origin at 40 Gbps. CDN can't protect what it can't see." },
            { title: "Cache Warming Failure on New PoP", symptom: "New PoP launched in Mumbai. 100K users routed to it. 100% cache miss. Origin in US-East overwhelmed.",
              cause: "New PoP has empty cache. All traffic = cache miss. Origin wasn't provisioned for an extra 100K QPS of cache misses.",
              fix: "Pre-warm new PoP from mid-tier cache before routing traffic. Use gradual traffic shifting: 1% → 10% → 50% → 100%. Monitor hit ratio at each step.",
              quote: "We opened a new PoP in South America. 100% miss rate for 30 minutes. Origin's connection pool maxed out. Lesson: always warm before you route." },
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
        { t: "Edge Compute (Workers)", d: "Run custom JavaScript/Wasm at edge. Personalize responses, A/B test, transform content, route dynamically — all at edge latency.", detail: "Cloudflare Workers, AWS Lambda@Edge, Fastly Compute. Enables dynamic content at edge without origin round-trip.", effort: "Medium" },
        { t: "Video Streaming (HLS/DASH)", d: "Chunk videos into segments. Cache each segment independently. Adaptive bitrate: client requests quality based on bandwidth.", detail: "Live and VOD. Edge-side manifest manipulation for ad insertion. Netflix OCAs cache 80%+ of their catalog at ISP PoPs.", effort: "Hard" },
        { t: "Image Optimization at Edge", d: "Resize, compress, and convert images on-the-fly at edge. Serve WebP to Chrome, AVIF to supported browsers, JPEG to legacy.", detail: "Single origin image → multiple edge variants. Reduces storage and bandwidth by 40-60%. Cloudflare Polish, CloudFront Image Optimization.", effort: "Medium" },
        { t: "Prefetching & Predictive Warming", d: "Analyze traffic patterns to predict what content will be popular. Pre-warm caches before viral events or scheduled launches.", detail: "ML model predicts content popularity from social signals, trending topics. Pre-fetch to edge before demand spike.", effort: "Hard" },
        { t: "Multi-CDN Strategy", d: "Use multiple CDN providers simultaneously. Route by performance, cost, or availability. Avoid vendor lock-in.", detail: "DNS-level load balancing between CDNs. RUM data drives routing decisions. Fallback to secondary CDN on primary failure.", effort: "Medium" },
        { t: "HTTP/3 (QUIC) Support", d: "UDP-based transport with built-in encryption. Eliminates head-of-line blocking. 0-RTT connection resumption.", detail: "30% faster page loads on lossy networks (mobile). All major CDNs now support H3. Migration is transparent to origin.", effort: "Easy" },
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
    { q:"How does a CDN handle dynamic content vs static content?", a:"Static content (images, JS, CSS) is straightforward — cache with long TTLs. Dynamic content (API responses, personalized pages) requires more nuance: use short TTLs (1-60s), cache by API endpoint + relevant query params, or use Edge Compute to assemble personalized responses from cached fragments (ESI — Edge Side Includes). Some CDNs support caching POST responses. The key principle: cache what you can at edge, and only go to origin for what you must.", tags:["design"] },
    { q:"What's the difference between a CDN and a reverse proxy?", a:"A reverse proxy sits in front of one origin and does load balancing, caching, SSL termination. A CDN is essentially a globally distributed network of reverse proxies with DNS-based routing. The CDN adds: geographic distribution, Anycast/GeoDNS routing, global purge infrastructure, and massive scale. Varnish is a reverse proxy cache. Cloudflare is that + 300 cities + DDoS protection + edge compute.", tags:["design"] },
    { q:"How does cache invalidation work at global scale?", a:"Three approaches: (1) TTL-based — content expires after set time. Simple but bounded staleness. (2) Purge API — explicit delete from all PoPs via pub/sub fanout (Kafka). Fastly does this in <150ms globally. (3) Surrogate keys / cache tags — tag cached objects (e.g., 'product-42') and purge all objects with that tag in one call. In practice, combine TTL as safety net + explicit purge on content update. This is the same belt-and-suspenders approach as distributed caching.", tags:["consistency"] },
    { q:"How does Netflix's Open Connect work?", a:"Netflix built their own CDN called Open Connect. They place custom servers (OCAs — Open Connect Appliances) directly inside ISPs' data centers. ISPs host them for free because it reduces transit traffic. OCAs pre-fill with content during off-peak hours based on predicted demand. Result: 95%+ of Netflix traffic is served from within the ISP's own network. The control plane is in AWS, but the data plane is fully distributed at the edge — the ultimate CDN.", tags:["scalability"] },
    { q:"How do you handle cache consistency across PoPs?", a:"You don't — and that's by design. Each PoP caches independently. Two users in different cities may see different cached versions for the duration of TTL. This is acceptable for most content. For content that MUST be consistent globally (e.g., a product price), use very short TTL (5-10s) + instant purge. For truly real-time consistency, don't use CDN — serve from origin directly. The tradeoff: consistency vs latency. CDNs choose latency.", tags:["consistency"] },
    { q:"How do you protect against cache poisoning?", a:"Cache poisoning occurs when an attacker tricks the CDN into caching malicious content. Defenses: (1) Strict cache key normalization — only include whitelisted query params and headers. (2) Validate Host header — reject requests with unexpected Host values. (3) Don't cache responses with Set-Cookie headers. (4) Use Vary header correctly — don't Vary on headers the attacker can manipulate. (5) Web cache deception: prevent caching of /account/settings.css (which might return account page). Validate Content-Type matches URL extension.", tags:["security"] },
    { q:"How would you design a multi-CDN setup?", a:"Use DNS-level load balancing (like NS1 or Cloudflare Load Balancing) to route between CDN providers. RUM (Real User Monitoring) beacons report actual performance from each CDN. Route 70/30 between primary and secondary. If primary degrades, shift traffic. Benefits: avoid single-provider outages, negotiate better pricing, use each CDN's strengths (e.g., Fastly for instant purge, CloudFront for AWS integration). Cost: more operational complexity, cache fragmentation across providers.", tags:["scalability"] },
    { q:"What's edge computing and when would you use it?", a:"Edge compute lets you run code at CDN edge PoPs (Cloudflare Workers, Lambda@Edge, Fastly Compute). Use cases: (1) A/B testing — assign variant at edge without origin round-trip. (2) Auth token validation — verify JWT at edge, reject unauthorized before hitting origin. (3) Image resizing — transform images on-the-fly at edge. (4) Personalization — assemble personalized page from cached fragments. (5) API gateway — rate limiting, request routing at edge. Tradeoff: limited compute (CPU time caps), cold starts, debugging is harder.", tags:["design"] },
    { q:"How does HTTP/3 (QUIC) improve CDN performance?", a:"HTTP/3 uses QUIC (UDP-based) instead of TCP. Key improvements: (1) 0-RTT connection resumption — returning users connect instantly. (2) No head-of-line blocking — lost packet doesn't block other streams. (3) Built-in encryption — no separate TLS handshake. (4) Connection migration — switching WiFi to cellular doesn't break the connection. Impact: 10-30% faster page loads, especially on mobile/lossy networks. All major CDNs support it. Migration is transparent — edge handles protocol negotiation.", tags:["design"] },
    { q:"How do CDNs handle video streaming at scale?", a:"Video is chunked into segments (2-10 seconds each) and encoded at multiple bitrates. Client downloads a manifest file (HLS .m3u8 or DASH .mpd) listing available segments/qualities. CDN caches each segment independently. Adaptive bitrate: client measures download speed and requests appropriate quality. Live streaming: edge receives segments from encoder in real-time, caches and serves to viewers. CDN is perfect for video because: (1) segments are static once encoded, (2) the same segment serves thousands of viewers, (3) geographic proximity reduces buffering.", tags:["scalability"] },
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

export default function ContentDeliveryNetworkSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Content Delivery Network</h1>
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