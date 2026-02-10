import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   RATE LIMITER — System Design Reference
   Pearl white theme · Reusable section structure
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",            icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",        icon: "📋", color: "#0891b2" },
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

/* ═══════════════════════════════════════════════════════════
   SECTIONS
   ═══════════════════════════════════════════════════════════ */

function ConceptSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-7 space-y-5">
          <Card accent="#6366f1">
            <Label>What is a Rate Limiter?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A rate limiter controls how many requests a client can send to a server within a given time window. It sits between clients and your API as a gatekeeper — protecting backend services from being overwhelmed by traffic spikes, buggy clients, or malicious attacks.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Think of it like a bouncer at a club: there's a max capacity. Once it's full, new people have to wait or leave. The limiter can reject (HTTP 429), queue, or throttle excess traffic.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Do We Need It?</Label>
            <ul className="space-y-2.5">
              <Point icon="🛡️" color="#0891b2">Prevent DoS / DDoS — stop malicious actors from flooding your service</Point>
              <Point icon="💰" color="#0891b2">Cost control — prevent a single user from consuming disproportionate resources</Point>
              <Point icon="⚖️" color="#0891b2">Fair usage — ensure all users get equitable access</Point>
              <Point icon="🔒" color="#0891b2">Protect downstream — prevent cascading failures to databases and third-party APIs</Point>
              <Point icon="📊" color="#0891b2">Revenue enforcement — API tiers (free: 100/min, premium: 1000/min)</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Examples</Label>
            <div className="space-y-2.5">
              {[
                { co: "Stripe", rule: "100 req/sec per API key", algo: "Sliding window" },
                { co: "GitHub", rule: "5,000 req/hr authenticated", algo: "Token bucket" },
                { co: "Twitter/X", rule: "15 req/15min (search)", algo: "Fixed window" },
                { co: "AWS API GW", rule: "10K req/sec per region", algo: "Token bucket" },
                { co: "Discord", rule: "Varies per endpoint", algo: "Per-route" },
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
            <Label color="#2563eb">Where Does It Live?</Label>
            <svg viewBox="0 0 360 180" className="w-full">
              <DiagramBox x={50} y={50} w={70} h={38} label="Client" color="#2563eb"/>
              <DiagramBox x={170} y={50} w={80} h={42} label="API\nGateway" color="#9333ea"/>
              <DiagramBox x={300} y={50} w={75} h={38} label="Service" color="#059669"/>
              <DiagramBox x={170} y={130} w={90} h={38} label="Rate Limiter" color="#dc2626"/>
              <Arrow x1={85} y1={50} x2={130} y2={50} id="c1"/>
              <Arrow x1={210} y1={50} x2={263} y2={50} label="allow" id="c2"/>
              <Arrow x1={170} y1={71} x2={170} y2={111} label="check" id="c3" dashed/>
              <rect x={108} y={152} width={124} height={18} rx={4} fill="#dc262608" stroke="#dc262630"/>
              <text x={170} y={162} textAnchor="middle" fill="#dc2626" fontSize="8" fontFamily="monospace">middleware or sidecar</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Stripe, Meta, Uber, Amazon, Atlassian, Cloudflare, Google</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Always Start Here</div>
            <p className="text-[12px] text-stone-500 mt-0.5">Spend 3-5 minutes clarifying requirements before designing. Ask about scope, scale, constraints. This shows structured thinking and prevents wasted effort designing the wrong system.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Accurately limit requests per user per time window (e.g., 100 req/min)</Point>
            <Point icon="2." color="#059669">Return HTTP 429 with Retry-After header when limit exceeded</Point>
            <Point icon="3." color="#059669">Support tiered limits — different rates for free, pro, enterprise</Point>
            <Point icon="4." color="#059669">Support multi-dimension limiting: per user, per IP, per endpoint</Point>
            <Point icon="5." color="#059669">Provide rate-limit response headers (X-RateLimit-Remaining, Reset)</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Low latency — rate check must add &lt;5ms overhead per request</Point>
            <Point icon="2." color="#dc2626">Highly available — limiter down should not cause full outage (fail-open)</Point>
            <Point icon="3." color="#dc2626">Distributed — accurate across multiple API servers (shared counters)</Point>
            <Point icon="4." color="#dc2626">Eventually consistent is acceptable — slight over/under count OK</Point>
            <Point icon="5." color="#dc2626">Minimal memory — must scale to millions of users</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask the Interviewer</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "What entity are we limiting? User ID? IP? API key?",
            "What's the window? Per second? Per minute? Per day?",
            "Should we allow bursts above the sustained rate?",
            "What happens on limit? Hard reject (429)? Queue? Throttle?",
            "Do different users have different limits?",
            "Single data center or multi-region?",
            "What's the expected scale? Users, QPS?",
            "Is this for internal APIs, external APIs, or both?",
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

function CapacitySection() {
  return (
    <div className="space-y-5">
      <Card className="bg-violet-50/50 border-violet-200">
        <div className="flex items-start gap-3">
          <span className="text-lg">💡</span>
          <div>
            <div className="text-[12px] font-bold text-violet-700">Interview Tip — Show Your Math</div>
            <p className="text-[12px] text-stone-500 mt-0.5">Walk through estimation out loud. Round aggressively — interviewers care about your process and order-of-magnitude reasoning, not exact numbers. State assumptions clearly: <em>"Let me assume 10M DAU..."</em></p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        {/* Traffic Estimation */}
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Traffic Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="DAU = 10M users" result="10M" note='Assumption — ask interviewer or state: "Let me assume a large-scale API"' />
            <MathStep step="2" formula="Avg requests per user per day = ~500" result="500" note="Browsing + API calls. Could be 50 for simple apps, 5000 for heavy ones." />
            <MathStep step="3" formula="Total requests/day = 10M × 500" result="5 Billion" note="5 × 10⁹ requests per day" />
            <MathStep step="4" formula="Seconds in a day = 24 × 60 × 60" result="86,400" note="Round to ~100K for easier math (slightly conservative)" />
            <MathStep step="5" formula="Avg QPS = 5B / 86,400 ≈ 5B / 100K" result="~58K QPS" note="Using 100K makes mental math easier: 5B/100K = 50K" final />
            <MathStep step="6" formula="Peak QPS = Avg × 3 (peak multiplier)" result="~175K QPS" note="Industry standard: peak = 2-3× average. Use 3× to be safe." final />
          </div>
        </Card>

        {/* Storage Estimation */}
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Storage Estimation (Redis)</Label>
          <div className="space-y-0">
            <MathStep step="1" formula='Token bucket state per user:' result="~64 B" note='Redis HASH: key (~30B) + "tokens" float (8B) + "last_refill" float (8B) + overhead (~18B)' />
            <MathStep step="2" formula="Active users who need a key = DAU" result="10M" note="Each active user has one Redis key. Inactive users auto-expire via TTL." />
            <MathStep step="3" formula="Total memory = 10M × 64 bytes" result="640 MB" note="10⁷ × 64 = 6.4 × 10⁸ bytes" final />
            <MathStep step="4" formula="With 2× headroom (fragmentation, overhead)" result="~1.3 GB" note="Fits comfortably in a single Redis node (typical: 8-64 GB)" final />
            <MathStep step="5" formula="Key TTL = 2 × rate window" result="120 sec" note="For a 60s window. Auto-cleans users who stop sending requests." />
            <MathStep step="6" formula="Key churn = DAU / TTL" result="~83K/sec" note="Keys created/expired per second. Redis handles this easily." />
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-2 gap-5">
        {/* Redis Throughput */}
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Redis Throughput</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Single Redis node throughput" result="~100K ops/s" note="Benchmark for simple commands (GET, SET, INCR)" />
            <MathStep step="2" formula="Lua script = ~4 Redis commands bundled" result="~1.5× cost" note="HGET + HGET + HMSET + EXPIRE per call. Slightly heavier than single op." />
            <MathStep step="3" formula="Effective rate-limit QPS per node = 100K / 1.5" result="~65K QPS" note="Each rate-limit check = 1 Lua invocation" final />
            <MathStep step="4" formula="Nodes needed = Peak QPS / per-node = 175K / 65K" result="3 shards" note="Redis Cluster with 3 primary shards" final />
            <MathStep step="5" formula="With replicas (1 per shard) = 3 × 2" result="6 nodes" note="3 primaries + 3 replicas for failover" />
          </div>
        </Card>

        {/* Cost */}
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Cost Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Redis node = r6g.large (13GB, 2 vCPU)" result="~$0.17/hr" note="AWS ElastiCache on-demand pricing" />
            <MathStep step="2" formula="6 nodes × $0.17/hr × 730 hrs/month" result="~$750/mo" note="On-demand. Reserved instances ~40% cheaper." final />
            <MathStep step="3" formula="Reserved pricing (1yr commitment)" result="~$450/mo" note="~40% savings with reserved instances" />
            <MathStep step="4" formula="Cost per 1M rate-limit checks" result="~$0.003" note="$450 / (175K × 86400 / 1M) ≈ $0.003 per million" />
          </div>
          <div className="mt-4 pt-3 border-t border-stone-100">
            <div className="text-[11px] text-stone-500">
              <strong className="text-stone-700">Key point:</strong> The entire rate-limiting infrastructure costs less than a single backend engineer-hour per month. Compare this to the cost of a DDoS attack or a single user consuming $10K+ of compute. The ROI is enormous.
            </div>
          </div>
        </Card>
      </div>

      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Avg QPS", val: "~58K", sub: "Peak: ~175K" },
            { label: "Redis Memory", val: "~640 MB", sub: "1 node sufficient" },
            { label: "Redis Cluster", val: "3 shards", sub: "6 nodes with replicas" },
            { label: "Monthly Cost", val: "~$450", sub: "Reserved instances" },
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
          <CodeBlock code={`# Middleware signature
def rate_limit_middleware(request) -> Response:
    user_id = extract_user(request)
    endpoint = request.path
    result = rate_limiter.check(user_id, endpoint)

    if not result.allowed:
        return Response(
            status=429,
            headers={
                "Retry-After": str(result.retry_after),
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(result.reset_at),
            },
            body={"error": "Rate limit exceeded"}
        )

    response = forward_to_api(request)
    response.headers["X-RateLimit-Remaining"] = str(result.remaining)
    response.headers["X-RateLimit-Limit"] = str(result.limit)
    return response`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Response Headers (Standard)</Label>
          <div className="space-y-3">
            {[
              { header: "X-RateLimit-Limit", desc: "Maximum requests allowed in the window", ex: "100" },
              { header: "X-RateLimit-Remaining", desc: "Requests remaining in current window", ex: "42" },
              { header: "X-RateLimit-Reset", desc: "Unix timestamp when window resets", ex: "1706889660" },
              { header: "Retry-After", desc: "Seconds to wait before retrying (on 429 only)", ex: "30" },
            ].map((h,i) => (
              <div key={i} className="flex items-start gap-3">
                <code className="text-[11px] font-mono font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded shrink-0">{h.header}</code>
                <div>
                  <div className="text-[12px] text-stone-600">{h.desc}</div>
                  <div className="text-[10px] text-stone-400 font-mono">Example: {h.ex}</div>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-stone-100">
            <Label color="#d97706">Design Decisions</Label>
            <ul className="space-y-1.5">
              <Point icon="→" color="#d97706">Place limiter at API Gateway layer — single enforcement point</Point>
              <Point icon="→" color="#d97706">Use middleware pattern — transparent to business logic</Point>
              <Point icon="→" color="#d97706">Always return rate headers on success too — enables client self-throttling</Point>
              <Point icon="→" color="#d97706">Consider gRPC interceptor if using gRPC instead of REST</Point>
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
    { label: "Start Simple", desc: "Single server, in-memory dictionary. Works for prototyping. Breaks with multiple servers — each has its own counter, so a user hitting different servers gets N× their limit." },
    { label: "Centralize State", desc: "Move counters to Redis so all servers share state. Redis is fast (sub-ms), supports TTL for auto-cleanup, and is single-threaded which helps with atomicity." },
    { label: "Add Rules", desc: "Different users need different limits. A rules config service stores tier-based rules. Rate limiter fetches and caches them locally with a 60s TTL to avoid per-request DB lookups." },
    { label: "Full Architecture", desc: "Complete: LB → API Gateway (with rate limiter middleware) → Redis Cluster (Lua atomic check) → API. Rules from config service, rejection metrics to monitoring, graceful degradation on Redis failure." },
  ];
  const diagrams = [
    () => (
      <svg viewBox="0 0 460 170" className="w-full">
        <DiagramBox x={60} y={65} w={75} h={38} label="Client" color="#2563eb"/>
        <DiagramBox x={210} y={65} w={95} h={42} label="Server\n+ Limiter" color="#9333ea"/>
        <DiagramBox x={370} y={65} w={85} h={42} label="In-Memory\nDict" color="#d97706"/>
        <Arrow x1={97} y1={65} x2={163} y2={65} id="s0a"/>
        <Arrow x1={258} y1={65} x2={328} y2={65} label="check" id="s0b" dashed/>
        <rect x={135} y={120} width={220} height={22} rx={6} fill="#dc262608" stroke="#dc262630"/>
        <text x={245} y={132} textAnchor="middle" fill="#dc2626" fontSize="9" fontFamily="monospace">❌ Not shared across servers</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 190" className="w-full">
        <DiagramBox x={55} y={75} w={68} h={36} label="Client" color="#2563eb"/>
        <DiagramBox x={155} y={75} w={46} h={36} label="LB" color="#64748b"/>
        <DiagramBox x={270} y={40} w={80} h={36} label="Server 1" color="#9333ea"/>
        <DiagramBox x={270} y={110} w={80} h={36} label="Server 2" color="#9333ea"/>
        <DiagramBox x={400} y={75} w={70} h={42} label="Redis" color="#dc2626"/>
        <Arrow x1={89} y1={75} x2={132} y2={75} id="r0"/>
        <Arrow x1={178} y1={67} x2={230} y2={48} id="r1"/>
        <Arrow x1={178} y1={83} x2={230} y2={103} id="r2"/>
        <Arrow x1={310} y1={48} x2={365} y2={68} label="check" id="r3" dashed/>
        <Arrow x1={310} y1={103} x2={365} y2={83} label="check" id="r4" dashed/>
        <rect x={350} y={148} width={100} height={20} rx={4} fill="#05966908" stroke="#05966930"/>
        <text x={400} y={159} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">✓ Shared state</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 200" className="w-full">
        <DiagramBox x={55} y={85} w={68} h={36} label="Client" color="#2563eb"/>
        <DiagramBox x={155} y={85} w={46} h={36} label="LB" color="#64748b"/>
        <DiagramBox x={280} y={85} w={92} h={42} label="Gateway\n+ Limiter" color="#9333ea"/>
        <DiagramBox x={415} y={85} w={62} h={36} label="API" color="#059669"/>
        <DiagramBox x={280} y={172} w={70} h={36} label="Redis" color="#dc2626"/>
        <DiagramBox x={130} y={172} w={78} h={36} label="Rules\nConfig" color="#d97706"/>
        <Arrow x1={89} y1={85} x2={132} y2={85} id="ru0"/>
        <Arrow x1={178} y1={85} x2={234} y2={85} id="ru1"/>
        <Arrow x1={326} y1={85} x2={384} y2={85} label="allow" id="ru2"/>
        <Arrow x1={280} y1={106} x2={280} y2={154} label="Lua" id="ru3" dashed/>
        <Arrow x1={169} y1={168} x2={245} y2={168} label="rules" id="ru4" dashed/>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 235" className="w-full">
        <DiagramBox x={42} y={65} w={56} h={34} label="Client" color="#2563eb"/>
        <DiagramBox x={118} y={65} w={46} h={34} label="LB" color="#64748b"/>
        <DiagramBox x={218} y={65} w={88} h={42} label="Gateway\n+ Limiter" color="#9333ea"/>
        <DiagramBox x={340} y={38} w={58} h={30} label="API 1" color="#059669"/>
        <DiagramBox x={340} y={92} w={58} h={30} label="API 2" color="#059669"/>
        <DiagramBox x={218} y={170} w={78} h={40} label="Redis\nCluster" color="#dc2626"/>
        <DiagramBox x={88} y={170} w={68} h={34} label="Rules" color="#d97706"/>
        <DiagramBox x={375} y={170} w={72} h={34} label="Monitor" color="#0284c7"/>
        <Arrow x1={70} y1={65} x2={95} y2={65} id="f0"/>
        <Arrow x1={141} y1={65} x2={174} y2={65} id="f1"/>
        <Arrow x1={262} y1={55} x2={311} y2={43} id="f2"/>
        <Arrow x1={262} y1={75} x2={311} y2={87} id="f3"/>
        <Arrow x1={218} y1={86} x2={218} y2={150} label="Lua" id="f4" dashed/>
        <Arrow x1={122} y1={170} x2={179} y2={170} label="rules" id="f5" dashed/>
        <Arrow x1={257} y1={170} x2={339} y2={170} label="metrics" id="f6" dashed/>
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
          <Label color="#059669">Request Flow — Allowed</Label>
          <div className="space-y-1.5">
            {[
              { s:"1", t:"Client sends GET /api/feed with auth token", c:"text-blue-600" },
              { s:"2", t:"Load balancer routes to API Gateway instance", c:"text-stone-500" },
              { s:"3", t:"Gateway extracts user_id from JWT token", c:"text-purple-600" },
              { s:"4", t:"Calls Redis Lua script: atomic token_bucket_check(user_id)", c:"text-red-600" },
              { s:"5", t:"Lua: refill tokens based on elapsed time, consume 1", c:"text-red-600" },
              { s:"6", t:"Redis returns {allowed: true, remaining: 57}", c:"text-emerald-600" },
              { s:"7", t:"Gateway sets X-RateLimit headers, forwards to API", c:"text-emerald-600" },
            ].map((s,i) => (
              <div key={i} className="flex items-start gap-2 text-[12px]">
                <span className="text-stone-300 font-mono w-4 shrink-0">{s.s}</span>
                <span className={s.c}>{s.t}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <Label color="#dc2626">Request Flow — Rejected (429)</Label>
          <div className="space-y-1.5">
            {[
              { s:"1", t:"Same flow through Gateway → Redis Lua script", c:"text-stone-400" },
              { s:"2", t:"Lua: tokens = 0 after refill, cannot consume → denied", c:"text-red-600" },
              { s:"3", t:"Redis returns {allowed: false, retry_after: 12.5}", c:"text-red-600" },
              { s:"4", t:"Gateway returns HTTP 429 Too Many Requests", c:"text-red-600" },
              { s:"5", t:"Headers: Retry-After: 13, X-RateLimit-Remaining: 0", c:"text-amber-600" },
              { s:"6", t:"Client backs off and retries after Retry-After seconds", c:"text-blue-600" },
              { s:"7", t:"Gateway emits rate_limit.rejected metric asynchronously", c:"text-sky-600" },
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
  const [sel, setSel] = useState("token_bucket");
  const algos = {
    fixed_window: { name: "Fixed Window", cx: "O(1) / O(1)",
      pros: ["Dead simple to implement","Very low memory","Fast — just INCR a counter"],
      cons: ["Boundary spike: 2× burst at window edges","User sends 100 at 0:59 and 100 at 1:01 = 200 in 2s"],
      when: "Acceptable when approximate limiting is fine and simplicity is valued over precision.",
      code: `# Fixed Window Counter
def is_allowed(user_id, limit=100, window=60):
    key = f"rate:{user_id}:{int(time.time()//window)}"
    count = redis.incr(key)
    if count == 1:
        redis.expire(key, window)
    return count <= limit` },
    sliding_log: { name: "Sliding Window Log", cx: "O(n) / O(n)",
      pros: ["Perfectly accurate — no boundary spikes","Exact count in any rolling window"],
      cons: ["O(n) memory per user — stores every timestamp","Expensive cleanup of old entries","Doesn't scale for high-rate users"],
      when: "When you need exact precision and users have relatively low request rates.",
      code: `# Sliding Window Log (Sorted Set)
def is_allowed(user_id, limit=100, window=60):
    now = time.time()
    key = f"rate:{user_id}"
    redis.zremrangebyscore(key, 0, now - window)
    count = redis.zcard(key)
    if count < limit:
        redis.zadd(key, {str(now): now})
        redis.expire(key, window)
        return True
    return False` },
    sliding_counter: { name: "Sliding Window Counter", cx: "O(1) / O(1)",
      pros: ["Smooths boundary spikes","O(1) memory — just 2 counters","Good accuracy/efficiency balance"],
      cons: ["Approximate — weighted average","Slightly complex to reason about"],
      when: "When you want better accuracy than fixed window without the memory cost of sliding log.",
      code: `# Sliding Window Counter (Hybrid)
def is_allowed(user_id, limit=100, window=60):
    now = time.time()
    curr_window = int(now // window)
    prev_window = curr_window - 1
    elapsed_pct = (now % window) / window

    prev_count = get_count(user_id, prev_window)
    curr_count = get_count(user_id, curr_window)

    weighted = prev_count * (1 - elapsed_pct) + curr_count
    return weighted < limit` },
    token_bucket: { name: "Token Bucket ★", cx: "O(1) / O(1)",
      pros: ["Allows controlled bursts up to capacity","O(1) memory — just 2 values per user","Production standard (Stripe, AWS, nginx)","Easy to explain in interviews"],
      cons: ["Needs atomic read-modify-write (Lua script in Redis)","Capacity + refill_rate parameters need tuning"],
      when: "Best general-purpose choice. Use when you want burst tolerance with a sustained rate limit. This is the recommended default for most API rate limiting.",
      code: `# Token Bucket (Production Standard)
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity        # max burst
        self.refill_rate = refill_rate   # tokens/sec
        self.tokens = capacity
        self.last_refill = time.time()

    def allow(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False` },
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
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Memory</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Accuracy</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Burst Handling</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Best For</th>
            </tr></thead>
            <tbody>
              {[
                { n:"Fixed Window", m:"O(1)", a:"Poor (edges)", b:"2× spike", f:"Simple APIs" },
                { n:"Sliding Log", m:"O(n) ⚠", a:"Perfect", b:"None", f:"Low-rate users" },
                { n:"Sliding Counter", m:"O(1)", a:"Good (approx)", b:"Smooth", f:"Balanced needs" },
                { n:"Token Bucket ★", m:"O(1)", a:"Good", b:"Controlled bursts", f:"Production APIs", hl:true },
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
          <Label color="#dc2626">Redis Data Model (Token Bucket)</Label>
          <CodeBlock code={`# Key pattern
"rate_limit:{user_id}" -> HASH

# Fields
{
  "tokens":      "73.5",          # current tokens
  "last_refill": "1706889600.123" # unix timestamp
}

# TTL = 2x window (auto-cleanup inactive users)
redis> HGETALL rate_limit:user_42
1) "tokens"       "73.5"
2) "last_refill"  "1706889600.123"

redis> TTL rate_limit:user_42
(integer) 87`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Rules Configuration</Label>
          <CodeBlock code={`# Rules: stored in config DB, cached locally (60s TTL)
rules = {
    "free": {
        "capacity": 100,
        "refill_rate": 1.67,     # tokens/sec = 100/min
        "window": 60,
    },
    "premium": {
        "capacity": 500,
        "refill_rate": 16.67,    # 1000/min
        "window": 60,
    },
    "enterprise": {
        "capacity": 2000,
        "refill_rate": 83.33,    # 5000/min
        "window": 60,
    },
}`} />
        </Card>
      </div>
      <Card accent="#9333ea">
        <Label color="#9333ea">Atomicity — The Core Challenge</Label>
        <p className="text-[12px] text-stone-500 mb-4">Every rate-limit check involves a read → compute → write cycle. Without atomicity, concurrent requests cause race conditions (two servers both read tokens=1, both allow). Here are the options:</p>
        <div className="grid grid-cols-3 gap-4">
          {[
            { t: "Redis Lua Script", d: "Entire read-refill-consume-write runs as a single atomic operation inside Redis. No distributed locks needed.", pros: ["Truly atomic — single-threaded execution","No extra round-trips","Production standard"], cons: ["Coupled to Redis"], pick: true },
            { t: "Redis Transactions (MULTI/EXEC)", d: "Bundle commands together. But no conditional logic — can't do 'read then decide' inside a transaction.", pros: ["Simple for basic counters"], cons: ["Can't branch on read values","Not suitable for token bucket"], pick: false },
            { t: "Distributed Lock (Redlock)", d: "Acquire lock → read → write → release. Heavy-weight approach.", pros: ["Works with any storage backend"], cons: ["Lock contention at high QPS","Added latency per request","Complexity"], pick: false },
          ].map((o,i) => (
            <div key={i} className={`rounded-lg border p-4 ${o.pick ? "border-purple-300 bg-purple-50/50" : "border-stone-200"}`}>
              <div className={`text-[11px] font-bold mb-1.5 ${o.pick ? "text-purple-700" : "text-stone-600"}`}>{o.t} {o.pick && "★"}</div>
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
            <strong className="text-stone-700">Key insight to mention in interview:</strong> Redis is single-threaded, so a Lua script executes atomically without locks. The script reads current tokens, refills based on elapsed time, attempts to consume one token, and writes back — all in one round-trip. This eliminates the TOCTOU (time-of-check-time-of-use) race condition entirely.
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
          <Label color="#059669">Scaling Redis</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Redis Cluster</strong> — shard by user_id hash. Linear horizontal scaling. Each shard handles a subset of users independently.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Memory is tiny</strong> — Token bucket: ~64 bytes/user. 10M users = ~640MB. Fits in one Redis node. Sharding is for throughput, not memory.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Connection pooling</strong> — reuse Redis connections across requests. Never create per-request connections.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Pipeline batch checks</strong> — if processing batch requests, pipeline multiple Lua calls in one round-trip.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Scaling the Service Layer</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Stateless middleware</strong> — all state lives in Redis. Add/remove gateway instances freely behind a load balancer.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Local rule cache</strong> — cache user tier → rule mapping locally (60s TTL). Eliminates per-request DB lookup for rules.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Async metrics</strong> — log rate-limit events to Kafka/pub-sub asynchronously. Never block the request path for telemetry.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Sidecar pattern</strong> — alternatively, deploy rate limiter as a sidecar (Envoy, Istio) rather than in-process middleware.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Strategy</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Option A: Local per Region", d:"Each region has its own Redis. User gets separate quota per region.", pros:["Lowest latency","No cross-region dependency"], cons:["User gets limit × N regions"], pick:false },
            { t:"Option B: Central Redis", d:"Single Redis cluster in one region. All regions check against it.", pros:["Globally accurate count"], cons:["Cross-region latency 50-200ms","Single point of failure"], pick:false },
            { t:"Option C: Split Quota ★", d:"Divide limit across regions proportionally. US: 60/min, EU: 40/min. Sync periodically.", pros:["Low latency (local Redis)","Reasonably accurate globally"], cons:["Slight over-count during sync gaps"], pick:true },
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
        <Label color="#d97706">Critical Decision: Fail-Open vs Fail-Closed</Label>
        <p className="text-[12px] text-stone-500 mb-4">What happens when Redis is unreachable? This is the most important availability decision and a very common follow-up question.</p>
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Fail-Open (Recommended for most APIs)</div>
            <p className="text-[11px] text-stone-500 mb-2">Redis down → allow all requests through</p>
            <ul className="space-y-1"><Point icon="✓" color="#059669">Service stays available</Point><Point icon="✓" color="#059669">Users don't experience outage</Point><Point icon="⚠" color="#d97706">Temporarily unprotected — acceptable tradeoff</Point></ul>
          </div>
          <div className="rounded-lg border border-red-200 bg-white p-4">
            <div className="text-[11px] font-bold text-red-600 mb-1.5">Fail-Closed (Security/Financial APIs)</div>
            <p className="text-[11px] text-stone-500 mb-2">Redis down → reject all requests (503)</p>
            <ul className="space-y-1"><Point icon="✓" color="#059669">Maximum protection</Point><Point icon="✗" color="#dc2626">Complete service outage if Redis fails</Point><Point icon="→" color="#d97706">Only for payments, auth, billing APIs</Point></ul>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#2563eb">Redis High Availability</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#2563eb"><strong className="text-stone-700">Redis Sentinel</strong> — automatic failover. Primary dies → Sentinel promotes replica in seconds. Clients reconnect automatically.</Point>
            <Point icon="2." color="#2563eb"><strong className="text-stone-700">Redis Cluster</strong> — built-in sharding + replication. Each shard has replicas. Survives node failures automatically.</Point>
            <Point icon="3." color="#2563eb"><strong className="text-stone-700">Local fallback</strong> — if Redis unreachable, fall back to local in-memory token bucket. Imprecise but functional.</Point>
            <Point icon="4." color="#2563eb"><strong className="text-stone-700">Circuit breaker</strong> — if Redis latency spikes >50ms, trip breaker → bypass rate limiting rather than adding latency.</Point>
          </ul>
        </Card>
        <Card>
          <Label color="#0891b2">Resiliency Patterns</Label>
          <ul className="space-y-2.5">
            <Point icon="🔄" color="#0891b2"><strong className="text-stone-700">Retry with backoff</strong> — Redis ops: 10ms → 20ms → 40ms → give up → fail-open.</Point>
            <Point icon="⏱" color="#0891b2"><strong className="text-stone-700">Aggressive timeout</strong> — 50ms Redis timeout. No response? Fail-open immediately.</Point>
            <Point icon="📊" color="#0891b2"><strong className="text-stone-700">Health checks</strong> — ping Redis every 5s. Alert on p99 > 5ms or connection errors.</Point>
            <Point icon="🔌" color="#0891b2"><strong className="text-stone-700">Graceful degradation</strong> — Layer 1: Redis Cluster. Layer 2: Local in-memory. Layer 3: No limiting.</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#9333ea">Degradation Ladder</Label>
        <div className="flex gap-3 items-stretch mt-1">
          {[
            { label: "Redis Cluster", sub: "Full accuracy", color: "#059669", status: "HEALTHY" },
            { label: "Redis Sentinel", sub: "Failover mode", color: "#d97706", status: "DEGRADED" },
            { label: "Local In-Memory", sub: "Per-server only", color: "#ea580c", status: "FALLBACK" },
            { label: "No Limiting", sub: "Fail-open bypass", color: "#dc2626", status: "BYPASS" },
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
              { metric: "rate_limit.allowed", type: "Counter", desc: "Total allowed requests (by user_tier, endpoint)" },
              { metric: "rate_limit.rejected", type: "Counter", desc: "Total 429 responses (by user_tier, endpoint)" },
              { metric: "rate_limit.rejection_rate", type: "Gauge", desc: "% of requests rejected — alert if > 5%" },
              { metric: "redis.latency_ms", type: "Histogram", desc: "p50, p95, p99 of Lua script execution time" },
              { metric: "redis.connection_errors", type: "Counter", desc: "Failed Redis connections — triggers fallback" },
              { metric: "rate_limit.fallback_active", type: "Gauge", desc: "1 if local fallback is in use (Redis down)" },
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
              { name: "High Rejection Rate", rule: "rejection_rate > 5% for 5min", sev: "P2", action: "Check if legitimate spike or misconfigured limit" },
              { name: "Redis Latency Spike", rule: "p99 > 10ms for 3min", sev: "P2", action: "Check Redis CPU/memory, network, Lua script performance" },
              { name: "Redis Down", rule: "connection_errors > 0 for 30s", sev: "P1", action: "Verify fallback activated, check Redis cluster health" },
              { name: "Single User Abuse", rule: "user rejection count > 1000/min", sev: "P3", action: "Investigate — possible bot or compromised key" },
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
              { q: "User getting 429 but shouldn't be", steps: "Check: correct tier applied? Rule cache stale? Clock drift? Check Redis key state with HGETALL." },
              { q: "Rejection rate spike across all users", steps: "Check: Redis latency spike? Deployment changed rules? Capacity estimation wrong? Check QPS vs limits." },
              { q: "Rate limiter adding >5ms latency", steps: "Check: Redis network latency? Lua script doing too much? Connection pool exhausted? Consider local caching of recently-rejected users." },
              { q: "Limits not enforced accurately", steps: "Check: Race condition (using Lua or not)? Multiple Redis keys for same user? Clock drift between servers?" },
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
          { title: "Race Conditions (TOCTOU)", sev: "Critical", sevColor: "#dc2626",
            desc: "Two servers read tokens=1 simultaneously, both think they can allow → 2 requests pass through, exceeding the limit.",
            fix: "Use Redis Lua scripts for atomic read-check-update. Redis is single-threaded, so the entire operation executes without interleaving. No distributed locks needed. This is the standard production solution.",
            code: `Problem: Server A reads tokens=1\n         Server B reads tokens=1\n         Both allow → limit exceeded\n\nSolution: Redis Lua script\n→ Read + refill + consume + write\n→ All in one atomic operation\n→ No lock needed (single-threaded)` },
          { title: "Clock Drift Between Servers", sev: "Medium", sevColor: "#d97706",
            desc: "Different servers have slightly different clocks. When token refill depends on elapsed time, drift causes inconsistent rate limiting across servers.",
            fix: "Use Redis server time instead of local server clocks. Redis TIME command provides consistent timestamps. Alternatively, enforce NTP with tight sync (<10ms drift). Mention this proactively — it shows distributed systems awareness.",
            code: `Problem: Server A clock = 12:00:01\n         Server B clock = 12:00:03\n         → Different refill amounts\n\nSolution: Use Redis TIME command\n→ All servers use same clock source\n→ Consistent refill calculations` },
          { title: "Hot Keys / Celebrity Problem", sev: "Medium", sevColor: "#d97706",
            desc: "A viral user generates millions of rate-limit checks/sec on one Redis key, overloading a single shard.",
            fix: "Two approaches: (1) Shard the user's key across N sub-keys, sum when checking. (2) Cache recent rejections locally — if a user was just rejected, skip Redis for the next few seconds. This is also how you handle thundering herd on popular endpoints.",
            code: `Approach 1: Key sharding\n  rate:user_42:shard_0\n  rate:user_42:shard_1\n  → Spread load across Redis nodes\n\nApproach 2: Local rejection cache\n  → Recently rejected? Skip Redis\n  → Reduces hot-key load by ~90%` },
          { title: "Cascading Failure", sev: "Critical", sevColor: "#dc2626",
            desc: "Redis becomes slow → every request waits → request queues back up → gateway thread pool exhausted → entire service becomes unresponsive. The rate limiter meant to protect you now causes the outage.",
            fix: "Circuit breaker pattern: set aggressive Redis timeout (50ms). After N consecutive failures, trip the breaker → bypass rate limiting entirely (fail-open). This is counter-intuitive but correct: temporarily unprotected is better than completely down.",
            code: `Failure cascade:\n  Redis slow → requests queue\n  → Thread pool full → 503s\n  → All services affected\n\nCircuit breaker:\n  Redis timeout > 50ms → fail count++\n  fail count > 5 → OPEN breaker\n  → Bypass rate limiter (fail-open)\n  → Service stays healthy` },
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
            <p className="text-[12px] text-stone-500 mt-0.5">In an HLD you say "rate limiter middleware" and draw one box. In production, that box is a gateway plugin, a rules service, a Redis cluster, a config pipeline, and an analytics system — each separately deployed and owned.</p>
          </div>
        </div>
      </Card>

      <Card accent="#0f766e">
        <Label color="#0f766e">Service Decomposition — What We Actually Build</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { name: "Rate Limiter Middleware", owns: "Intercepts requests, checks quota, returns 429 or passes through", tech: "API Gateway plugin (Kong/Envoy filter) or Express/Go middleware", api: "Embedded — no external API", scale: "Scales with gateway / app server pods", stateful: false,
              modules: ["Request Key Extractor (user ID, IP, API key, endpoint)", "Algorithm Engine (token bucket, sliding window)", "Redis Client (connection pool, pipeline, circuit breaker)", "Response Decorator (X-RateLimit-* headers)", "Fallback Handler (fail-open when Redis down)", "Metrics Emitter (allowed/rejected counters)"] },
            { name: "Redis Rate Store", owns: "Stores per-key token counts, TTLs, atomic check-and-decrement", tech: "Redis Cluster (3+ shards, RESP protocol)", api: "EVALSHA (Lua script), GET/SET/DECR", scale: "Horizontal — shard by key hash", stateful: true,
              modules: ["Token Bucket State (hash: tokens + last_refill per key)", "Lua Script Engine (atomic check-decrement-refill)", "Key Expiry (active + lazy TTL cleanup)", "Replication (async primary → replica)", "Memory Manager (eviction when full — LRU on rate keys)", "Cluster Coordinator (gossip, slot management)"] },
            { name: "Rules Config Service", owns: "Stores and serves rate limit rules (per-tier, per-endpoint, per-user overrides)", tech: "etcd / Consul / PostgreSQL + config API", api: "GET /rules/:endpoint, PUT /rules (admin)", scale: "3-node quorum (config doesn't need high QPS)", stateful: true,
              modules: ["Rule Store (tier → endpoint → limits mapping)", "Override Manager (per-user custom limits)", "Change Publisher (push updates to gateway on rule change)", "Validation Engine (prevent invalid rules: limit=0)", "Audit Logger (who changed what rule, when)", "Version Manager (rollback to previous rule set)"] },
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
              { name: "Rate Limit Analytics", role: "Aggregates allowed/rejected events per user, endpoint, time window. Powers dashboards and abuse detection.", tech: "Kafka → Flink/Spark → ClickHouse", critical: false },
              { name: "Abuse Detection Worker", role: "Consumes rejection events. Flags users with sustained high rejection rates. Can auto-escalate to IP ban.", tech: "Kafka consumer → ML scoring → Rules Config API", critical: false },
              { name: "User Quota Dashboard", role: "Self-service UI for API consumers to view their usage, remaining quota, rate limit tier.", tech: "React + GET /rate-limit/status API", critical: false },
              { name: "Alert Manager", role: "Monitors Redis health, rejection rate spikes, per-user abuse patterns. PagerDuty integration.", tech: "Prometheus + Alertmanager → PagerDuty/Slack", critical: true },
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
          <Label color="#9333ea">Rate Limiter Middleware Internals</Label>
          <svg viewBox="0 0 380 310" className="w-full">
            {/* Incoming request */}
            <rect x={10} y={10} width={360} height={40} rx={6} fill="#2563eb08" stroke="#2563eb" strokeWidth={1}/>
            <text x={190} y={28} textAnchor="middle" fill="#2563eb" fontSize="10" fontWeight="600" fontFamily="monospace">Incoming HTTP Request</text>
            <text x={190} y={42} textAnchor="middle" fill="#2563eb80" fontSize="8" fontFamily="monospace">from API Gateway / Load Balancer</text>

            {/* Key extractor */}
            <rect x={10} y={60} width={360} height={38} rx={6} fill="#6366f108" stroke="#6366f1" strokeWidth={1}/>
            <text x={190} y={76} textAnchor="middle" fill="#6366f1" fontSize="10" fontWeight="600" fontFamily="monospace">Key Extractor</text>
            <text x={190} y={90} textAnchor="middle" fill="#6366f180" fontSize="8" fontFamily="monospace">user_id from JWT | IP from X-Forwarded-For | API key from header</text>

            {/* Rules lookup */}
            <rect x={10} y={108} width={175} height={42} rx={6} fill="#d9770608" stroke="#d97706" strokeWidth={1}/>
            <text x={97} y={126} textAnchor="middle" fill="#d97706" fontSize="10" fontWeight="600" fontFamily="monospace">Rules Cache</text>
            <text x={97} y={140} textAnchor="middle" fill="#d9770680" fontSize="8" fontFamily="monospace">local cache, 60s TTL</text>

            <rect x={195} y={108} width={175} height={42} rx={6} fill="#c026d308" stroke="#c026d3" strokeWidth={1}/>
            <text x={282} y={126} textAnchor="middle" fill="#c026d3" fontSize="10" fontWeight="600" fontFamily="monospace">Algorithm Engine</text>
            <text x={282} y={140} textAnchor="middle" fill="#c026d380" fontSize="8" fontFamily="monospace">token bucket logic</text>

            {/* Redis call */}
            <rect x={10} y={160} width={360} height={40} rx={6} fill="#dc262608" stroke="#dc2626" strokeWidth={1}/>
            <text x={190} y={177} textAnchor="middle" fill="#dc2626" fontSize="10" fontWeight="600" fontFamily="monospace">Redis Client (Lua Script Call)</text>
            <text x={190} y={192} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">EVALSHA token_bucket.lua → {`{allowed: true, remaining: 47, reset: 1707500000}`}</text>

            {/* Decision */}
            <rect x={10} y={210} width={175} height={38} rx={6} fill="#05966908" stroke="#059669" strokeWidth={1}/>
            <text x={97} y={226} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="600" fontFamily="monospace">✓ ALLOWED</text>
            <text x={97} y={240} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">add headers, pass through</text>

            <rect x={195} y={210} width={175} height={38} rx={6} fill="#dc262608" stroke="#dc2626" strokeWidth={1}/>
            <text x={282} y={226} textAnchor="middle" fill="#dc2626" fontSize="10" fontWeight="600" fontFamily="monospace">✗ REJECTED</text>
            <text x={282} y={240} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">429 + Retry-After header</text>

            {/* Metrics */}
            <rect x={10} y={258} width={360} height={38} rx={6} fill="#0284c708" stroke="#0284c7" strokeWidth={1}/>
            <text x={190} y={274} textAnchor="middle" fill="#0284c7" fontSize="10" fontWeight="600" fontFamily="monospace">Metrics Emitter</text>
            <text x={190} y={288} textAnchor="middle" fill="#0284c780" fontSize="8" fontFamily="monospace">rate_limit.allowed++ | rate_limit.rejected++ | redis.latency_ms</text>

            {/* Arrows */}
            <line x1={190} y1={50} x2={190} y2={60} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={190} y1={98} x2={97} y2={108} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={190} y1={98} x2={282} y2={108} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={190} y1={150} x2={190} y2={160} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={97} y1={200} x2={97} y2={210} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={282} y1={200} x2={282} y2={210} stroke="#94a3b8" strokeWidth={1}/>
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
                { route: "Gateway → Rate Limiter MW", proto: "In-process (plugin)", contract: "check(key, endpoint) → allowed|rejected", timeout: "N/A (same process)", fail: "Fail-open: allow request" },
                { route: "Rate Limiter MW → Redis", proto: "RESP (TCP)", contract: "EVALSHA lua_script key [tokens, capacity, refill_rate, now]", timeout: "50ms hard cutoff", fail: "Circuit breaker OPEN → fail-open" },
                { route: "Rate Limiter MW → Rules Cache", proto: "In-memory lookup", contract: "getRules(endpoint, tier) → {limit, window, burst}", timeout: "N/A (local)", fail: "Use hardcoded defaults" },
                { route: "Rules Config → Gateway (push)", proto: "gRPC stream / Webhook", contract: "onRuleChange(endpoint, newLimits)", timeout: "5s connect, streaming", fail: "Gateway keeps last-known rules" },
                { route: "Rate Limiter MW → Kafka", proto: "Kafka producer (async)", contract: "emit(key, endpoint, allowed, timestamp)", timeout: "Fire-and-forget", fail: "Drop event (analytics loss, not critical)" },
                { route: "Abuse Worker → Rules Config", proto: "REST (internal)", contract: "PUT /rules/override/:user_id {limit: 0}", timeout: "2s", fail: "Alert, manual intervention" },
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
  const [flow, setFlow] = useState("happy");
  const flows = {
    happy: {
      title: "Happy Path — Request Allowed",
      steps: [
        { actor: "Client", action: "GET /api/search?q=hotels (with Authorization: Bearer <JWT>)", type: "request" },
        { actor: "Load Balancer", action: "Route to healthy gateway pod (round-robin)", type: "process" },
        { actor: "API Gateway", action: "Extract JWT → decode user_id=42, tier=premium", type: "auth" },
        { actor: "Rate Limiter MW", action: "Build key: rate_limit:premium:/api/search:user:42", type: "process" },
        { actor: "Rate Limiter MW", action: "Local rules cache lookup → premium:/api/search = {limit:200, window:60s, burst:50}", type: "process" },
        { actor: "Rate Limiter MW → Redis", action: "EVALSHA token_bucket.lua 1 rate_limit:premium:... 200 60 50 1707500042.123", type: "request" },
        { actor: "Redis (Lua script)", action: "Check tokens: 147 remaining. Decrement → 146. Return {allowed:true, remaining:146, reset:1707500060}", type: "success" },
        { actor: "Rate Limiter MW", action: "Set headers: X-RateLimit-Limit:200, X-RateLimit-Remaining:146, X-RateLimit-Reset:1707500060", type: "process" },
        { actor: "API Gateway", action: "Forward request to backend /api/search service", type: "success" },
        { actor: "Rate Limiter MW", action: "Async emit: Kafka event {user:42, endpoint:/api/search, allowed:true, ts:...}", type: "process" },
      ]
    },
    rejected: {
      title: "Rejected — 429 Too Many Requests",
      steps: [
        { actor: "Client", action: "GET /api/search?q=hotels (50th request in 10 seconds)", type: "request" },
        { actor: "API Gateway", action: "Extract user_id=42, tier=free", type: "auth" },
        { actor: "Rate Limiter MW", action: "Build key: rate_limit:free:/api/search:user:42", type: "process" },
        { actor: "Rate Limiter MW → Redis", action: "EVALSHA token_bucket.lua ... → tokens=0, refill in 23s", type: "request" },
        { actor: "Redis (Lua script)", action: "Check tokens: 0 remaining. No refill yet. Return {allowed:false, remaining:0, reset:1707500083, retry_after:23}", type: "error" },
        { actor: "Rate Limiter MW", action: "Build 429 response with Retry-After: 23", type: "process" },
        { actor: "API Gateway", action: "Return HTTP 429 Too Many Requests. Body: {error: 'rate_limit_exceeded', retry_after: 23}", type: "error" },
        { actor: "Rate Limiter MW", action: "Async emit: Kafka event {user:42, rejected:true, remaining:0}", type: "process" },
        { actor: "Client", action: "Receives 429. Well-behaved client backs off for 23 seconds. Badly-behaved client retries immediately.", type: "check" },
      ]
    },
    redis_down: {
      title: "Redis Failure — Fail-Open Path",
      steps: [
        { actor: "Client", action: "GET /api/search?q=hotels", type: "request" },
        { actor: "Rate Limiter MW", action: "Build key, attempt Redis EVALSHA", type: "process" },
        { actor: "Rate Limiter MW → Redis", action: "Connection attempt → timeout after 50ms", type: "error" },
        { actor: "Circuit Breaker", action: "Failure count = 6 (threshold = 5). State: CLOSED → OPEN", type: "error" },
        { actor: "Rate Limiter MW", action: "Circuit breaker OPEN → skip Redis entirely for next 30s", type: "check" },
        { actor: "Rate Limiter MW", action: "Fallback: local in-memory token bucket (per-server, imprecise but functional)", type: "process" },
        { actor: "Rate Limiter MW", action: "Local bucket: user:42 has tokens → ALLOW. Set headers with X-RateLimit-Source: local", type: "success" },
        { actor: "API Gateway", action: "Forward to backend. Emit metric: rate_limiter.fallback_active=1", type: "success" },
        { actor: "Alert Manager", action: "(Async) redis_down alert fires → P1 page to on-call", type: "error" },
        { actor: "Circuit Breaker", action: "After 30s → HALF-OPEN → send 1 probe to Redis. If success → CLOSED. If fail → OPEN again.", type: "check" },
      ]
    },
    multi_dim: {
      title: "Multi-Dimension — User + Endpoint + IP",
      steps: [
        { actor: "Client (IP: 203.0.113.5)", action: "GET /api/search (user_id=42, tier=premium)", type: "request" },
        { actor: "Rate Limiter MW", action: "Build 3 keys: user:42 (global), user:42:/api/search (per-endpoint), ip:203.0.113.5 (per-IP)", type: "process" },
        { actor: "Rate Limiter MW → Redis", action: "Pipeline 3 EVALSHA calls in single round-trip (Redis pipeline, not 3 separate calls)", type: "request" },
        { actor: "Redis", action: "Check 1: user:42 global → 847/1000 remaining ✓", type: "success" },
        { actor: "Redis", action: "Check 2: user:42:/api/search → 146/200 remaining ✓", type: "success" },
        { actor: "Redis", action: "Check 3: ip:203.0.113.5 → 4,891/10,000 remaining ✓", type: "success" },
        { actor: "Rate Limiter MW", action: "ALL three checks passed → ALLOW. Return lowest remaining (146) in headers.", type: "success" },
        { actor: "Note", action: "If ANY check fails → REJECT with 429. Response indicates WHICH limit was hit.", type: "check" },
      ]
    },
    rule_update: {
      title: "Live Rule Update — No Restart",
      steps: [
        { actor: "Admin", action: "PUT /admin/rules/free:/api/search {limit: 20, window: 60} (was 50/min, now 20/min)", type: "request" },
        { actor: "Rules Config Service", action: "Validate rule (limit > 0, window > 0). Write to etcd. Log audit event.", type: "process" },
        { actor: "Rules Config Service", action: "Publish change event to all gateway pods via gRPC stream", type: "process" },
        { actor: "Gateway Pod 1", action: "Receive rule change notification. Update local rules cache immediately.", type: "success" },
        { actor: "Gateway Pod 2", action: "Receive rule change notification. Update local rules cache immediately.", type: "success" },
        { actor: "Gateway Pod N", action: "(If push fails) Local cache TTL=60s. Next cache refresh picks up new rule from Config Service.", type: "check" },
        { actor: "Rate Limiter MW", action: "Next request uses new rule: limit=20/min for free tier on /api/search", type: "success" },
        { actor: "Note", action: "Existing Redis keys with old limits: TTL expires naturally. Or force-delete: SCAN for matching keys.", type: "process" },
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
        <Label color="#7e22ce">Circuit Breaker State Machine (Redis Protection)</Label>
        <div className="flex gap-4 items-center justify-center py-4">
          {[
            { state: "CLOSED", sub: "Redis healthy", color: "#059669", desc: "All rate checks go to Redis. Track consecutive failures." },
            { state: "OPEN", sub: "Redis bypassed", color: "#dc2626", desc: "Skip Redis entirely. Use local fallback. Timer: 30s." },
            { state: "HALF-OPEN", sub: "Testing Redis", color: "#d97706", desc: "Send 1 probe to Redis. Success → CLOSED. Fail → OPEN." },
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
          <CodeBlock title="Rate Limiter — Embedded in Gateway (NOT a separate service)" code={`# The rate limiter is NOT a standalone microservice.
# It's a plugin/middleware inside the API Gateway.
# Deploying it separately adds a network hop on EVERY request.

# Option A: Kong Gateway with Rate Limiting Plugin
apiVersion: apps/v1
kind: Deployment            # Stateless — Deployment, not StatefulSet
metadata:
  name: api-gateway
spec:
  replicas: 6               # 2 per AZ (3 AZs)
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0     # Zero downtime deploys
  template:
    spec:
      containers:
      - name: kong
        image: kong:3.5
        env:
        - name: KONG_PLUGINS
          value: "rate-limiting-advanced,jwt,cors"
        - name: REDIS_HOST
          value: "redis-rate-store.cache.svc"
        - name: REDIS_PORT
          value: "6379"
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2"
            memory: "1Gi"
        readinessProbe:
          httpGet:
            path: /status
            port: 8001
          periodSeconds: 5
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone`} />
          <div className="mt-3 space-y-1.5">
            <Point icon="⚠" color="#b45309">Deployment not StatefulSet — gateway is stateless</Point>
            <Point icon="⚠" color="#b45309">maxUnavailable: 0 — always maintain full capacity during rollout</Point>
            <Point icon="⚠" color="#b45309">Rate limiter is a plugin, NOT a separate hop — adding 1ms per request at 100K QPS = matters</Point>
          </div>
        </Card>
        <Card accent="#0f766e">
          <Label color="#0f766e">Multi-AZ Deployment Layout</Label>
          <svg viewBox="0 0 380 320" className="w-full">
            <rect x={5} y={5} width={370} height={310} rx={10} fill="#0f766e05" stroke="#0f766e30" strokeWidth={1} strokeDasharray="4,3"/>
            <text x={190} y={22} textAnchor="middle" fill="#0f766e" fontSize="10" fontWeight="700" fontFamily="monospace">us-east-1</text>

            {/* NLB */}
            <rect x={130} y={35} width={120} height={24} rx={4} fill="#2563eb12" stroke="#2563eb" strokeWidth={1}/>
            <text x={190} y={51} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Network Load Balancer</text>

            {/* AZ-a */}
            <rect x={15} y={72} width={110} height={140} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={70} y={87} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-a</text>
            <rect x={22} y={96} width={96} height={24} rx={4} fill="#9333ea15" stroke="#9333ea" strokeWidth={1}/>
            <text x={70} y={112} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">Gateway ×2</text>
            <rect x={22} y={128} width={96} height={24} rx={4} fill="#d9770615" stroke="#d97706" strokeWidth={1}/>
            <text x={70} y={144} textAnchor="middle" fill="#d97706" fontSize="8" fontFamily="monospace">Redis (Primary)</text>
            <rect x={22} y={160} width={96} height={24} rx={4} fill="#05966915" stroke="#059669" strokeWidth={1}/>
            <text x={70} y={176} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">Config Node</text>

            {/* AZ-b */}
            <rect x={135} y={72} width={110} height={140} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={190} y={87} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-b</text>
            <rect x={142} y={96} width={96} height={24} rx={4} fill="#9333ea15" stroke="#9333ea" strokeWidth={1}/>
            <text x={190} y={112} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">Gateway ×2</text>
            <rect x={142} y={128} width={96} height={24} rx={4} fill="#0891b215" stroke="#0891b2" strokeWidth={1}/>
            <text x={190} y={144} textAnchor="middle" fill="#0891b2" fontSize="8" fontFamily="monospace">Redis (Replica)</text>
            <rect x={142} y={160} width={96} height={24} rx={4} fill="#05966915" stroke="#059669" strokeWidth={1}/>
            <text x={190} y={176} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">Config Node</text>

            {/* AZ-c */}
            <rect x={255} y={72} width={110} height={140} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={310} y={87} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-c</text>
            <rect x={262} y={96} width={96} height={24} rx={4} fill="#9333ea15" stroke="#9333ea" strokeWidth={1}/>
            <text x={310} y={112} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">Gateway ×2</text>
            <rect x={262} y={128} width={96} height={24} rx={4} fill="#0891b215" stroke="#0891b2" strokeWidth={1}/>
            <text x={310} y={144} textAnchor="middle" fill="#0891b2" fontSize="8" fontFamily="monospace">Redis (Replica)</text>
            <rect x={262} y={160} width={96} height={24} rx={4} fill="#05966915" stroke="#059669" strokeWidth={1}/>
            <text x={310} y={176} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">Config Node</text>

            {/* Legend */}
            <rect x={15} y={225} width={350} height={82} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
            <text x={190} y={242} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Deployment Rules</text>
            <text x={30} y={260} fill="#78716c" fontSize="8" fontFamily="monospace">• Gateway: stateless, 2+ per AZ, auto-scaled on CPU/QPS via HPA</text>
            <text x={30} y={275} fill="#78716c" fontSize="8" fontFamily="monospace">• Redis: 1 primary + 2 replicas across AZs. Sentinel for failover.</text>
            <text x={30} y={290} fill="#78716c" fontSize="8" fontFamily="monospace">• Config: 3-node etcd quorum. Survives 1 AZ loss.</text>
            <text x={30} y={305} fill="#78716c" fontSize="8" fontFamily="monospace">• AZ failure: gateway auto-routes away. Redis failover in &lt;5s.</text>
          </svg>
        </Card>
      </div>

      <Card accent="#dc2626">
        <Label color="#dc2626">Security & Authentication</Label>
        <div className="grid grid-cols-3 gap-4">
          {[
            { layer: "Client → Gateway", what: "JWT/OAuth2 for user identity. API keys for service-to-service. Gateway validates before rate limit check — no point rate-limiting invalid tokens.",
              details: ["TLS termination at NLB or gateway", "JWT expiry validation + signature check", "Extract user_id + tier from JWT claims", "Reject expired/invalid tokens BEFORE rate limit check (saves Redis calls)"] },
            { layer: "Gateway → Redis (Internal)", what: "mTLS between gateway pods and Redis. Redis AUTH password. Network policy restricts Redis access to gateway namespace only.",
              details: ["Redis AUTH requirepass or ACL (Redis 6+)", "K8s NetworkPolicy: only gateway pods can reach Redis port 6379", "mTLS with auto-rotated certs (SPIFFE/Istio mTLS)", "No TLS to Redis if same VPC + perf-critical (controversial — discuss tradeoff)"] },
            { layer: "Admin API Protection", what: "Rules config API is admin-only. Separate auth from user-facing API. Rate limit the rate limit config API (meta!).",
              details: ["Admin JWT with role=admin claim or separate auth system", "IP allowlisting for admin endpoints", "Audit log every rule change (who, when, what, previous value)", "Rate limit the admin API itself: 10 req/min (prevent accidental loops)"] },
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
        <Label color="#b45309">Rolling Update — Gateway + Rules</Label>
        <p className="text-[12px] text-stone-500 mb-3">Gateway is stateless (easy). But rules changes need careful coordination to avoid inconsistent limiting during rollout.</p>
        <div className="grid grid-cols-2 gap-5">
          <div>
            <div className="text-[10px] font-bold text-stone-600 uppercase tracking-wider mb-2">Gateway Pod Update</div>
            <div className="space-y-0">
              {[
                { step: 1, action: "New pod starts, loads rules from Config Service", detail: "readinessProbe passes only after rules are loaded and Redis connection pool is warm." },
                { step: 2, action: "NLB adds new pod to target group", detail: "Health check passes → traffic starts flowing to new pod." },
                { step: 3, action: "Old pod drained", detail: "K8s sends SIGTERM → pod stops accepting new connections → existing connections drain (30s grace)." },
                { step: 4, action: "Repeat pod by pod", detail: "maxSurge:1 + maxUnavailable:0 = always at full capacity. Zero dropped requests." },
              ].map((s,i) => (
                <div key={i} className="flex items-start gap-2.5 py-2 border-b border-stone-100 last:border-0">
                  <span className="text-[10px] font-bold w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5 bg-amber-100 text-amber-700">{s.step}</span>
                  <div>
                    <div className="text-[11px] font-bold text-stone-700">{s.action}</div>
                    <div className="text-[10px] text-stone-400">{s.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="text-[10px] font-bold text-stone-600 uppercase tracking-wider mb-2">Rule Change Rollout</div>
            <div className="space-y-0">
              {[
                { step: 1, action: "Admin pushes new rule via Config API", detail: "Validation: limit > 0, window > 0, tier exists. Stored in etcd with version number." },
                { step: 2, action: "Config Service pushes to all gateways", detail: "gRPC stream notification. All pods get update within 1-2 seconds." },
                { step: 3, action: "Brief inconsistency window (1-2s)", detail: "Some pods have new rule, some have old. Acceptable — rate limiting is best-effort anyway." },
                { step: 4, action: "Redis keys with old limits expire naturally", detail: "TTL on existing buckets = window duration. After 1 window cycle, all keys use new limits." },
              ].map((s,i) => (
                <div key={i} className="flex items-start gap-2.5 py-2 border-b border-stone-100 last:border-0">
                  <span className="text-[10px] font-bold w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5 bg-amber-100 text-amber-700">{s.step}</span>
                  <div>
                    <div className="text-[11px] font-bold text-stone-700">{s.action}</div>
                    <div className="text-[10px] text-stone-400">{s.detail}</div>
                  </div>
                </div>
              ))}
            </div>
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
        <Label color="#be123c">Scaling Playbook — Not Just "Add More Redis"</Label>
        <p className="text-[12px] text-stone-500 mb-3">When QPS grows, which component hits the wall first? And how do you fix each one without downtime?</p>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[11px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Bottleneck</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Symptom</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Fix</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Impact During Fix</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Pitfall</th>
            </tr></thead>
            <tbody>
              {[
                { bottleneck: "Gateway CPU", symptom: "p99 latency up, 503s", fix: "HPA scales gateway pods (target CPU 70%)", impact: "None — new pods online in 30s", pitfall: "Check: is it JWT validation or rate limit logic? Profile before scaling." },
                { bottleneck: "Redis single-thread", symptom: "Redis CPU 90%, ops/s flat", fix: "Shard: add Redis nodes, redistribute keys", impact: "~1% miss rate during resharding", pitfall: "Single-threaded — vertical scaling doesn't help. Must shard." },
                { bottleneck: "Redis memory", symptom: "Evictions spike, keys lost", fix: "Increase maxmemory or add shards", impact: "Brief eviction burst during migration", pitfall: "Rate limit keys are small (~64B). If memory full, something else is in Redis." },
                { bottleneck: "Redis connections", symptom: "Connection pool exhausted", fix: "Increase pool size or add proxy (Twemproxy)", impact: "None if pool resize. Brief reconnects if proxy added.", pitfall: "Each gateway pod × pool_size = total connections. 50 pods × 20 = 1000 connections." },
                { bottleneck: "Network bandwidth", symptom: "Packet drops between gateway and Redis", fix: "Collocate in same AZ, use UNIX socket if same host", impact: "Minimal", pitfall: "Rate limit payloads are tiny. If network is the issue, something else is wrong." },
                { bottleneck: "Hot key (one user)", symptom: "One Redis shard overwhelmed", fix: "Replicate hot key or use local cache for known hot keys", impact: "None", pitfall: "Attackers deliberately create hot keys. IP-based limit catches this before user-based." },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-mono text-stone-700 font-medium">{r.bottleneck}</td>
                  <td className="px-3 py-2 text-red-500">{r.symptom}</td>
                  <td className="px-3 py-2 text-stone-500">{r.fix}</td>
                  <td className="px-3 py-2 text-stone-400">{r.impact}</td>
                  <td className="px-3 py-2 text-amber-600 text-[10px]">{r.pitfall}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card accent="#be123c">
        <Label color="#be123c">Operational Pitfalls — Production War Stories</Label>
        <p className="text-[12px] text-stone-500 mb-3">Real failure modes that distinguish "I've read the book" from "I've been paged at 3am for this."</p>
        <div className="grid grid-cols-2 gap-4">
          {[
            { title: "Rate Limiter Adds Latency to Every Request", symptom: "p50 latency increased by 3ms across all APIs after enabling rate limiting. Product team complains.",
              cause: "Rate limiter makes a network call to Redis on every request. Even cache hits add 1-3ms. At 100K QPS, that's 100K extra Redis calls/sec.",
              fix: "Pipeline Redis calls with other middleware calls. Or use local in-memory rate limiter for non-critical endpoints (accept imprecision). Or embed Redis as a sidecar (localhost = ~0.1ms instead of 1-3ms network).",
              quote: "PM asked 'why did p50 go up 3ms?' We said 'because we added rate limiting.' PM said 'can we not?' We said 'sure, enjoy the next DDoS.'" },
            { title: "Clock Drift Between Servers", symptom: "User gets 429 on server A but succeeds on server B in same second. Rate limits appear inconsistent.",
              cause: "Token bucket refill uses timestamp. Different servers have different clocks (even with NTP, drift of 10-100ms is normal). Server A thinks it's 12:00:00.100, server B thinks 12:00:00.050.",
              fix: "Use Redis server time (TIME command) inside Lua script instead of application server time. Redis is the single source of truth for timestamps. All servers agree.",
              quote: "Users filed a bug: 'rate limiting is broken, I get 429 randomly.' Turned out NTP was 200ms off on 3 servers after a restart." },
            { title: "Lua Script Timeout Blocks All Redis Operations", symptom: "All rate limit checks hang. Redis appears frozen. Every API call times out.",
              cause: "Lua script has a bug (infinite loop) or processes too many keys. Redis is single-threaded — while Lua runs, nothing else executes. lua-time-limit (5s default) only raises warnings, doesn't kill the script.",
              fix: "Keep Lua scripts minimal (< 10 operations). Set lua-time-limit to 500ms. Test scripts with SLOWLOG. Have circuit breaker that skips Redis if latency > 50ms. Never iterate over unbounded key sets in Lua.",
              quote: "Deployed a 'small Lua optimization' that had an O(n) loop. Redis was frozen for 8 seconds. Every API in the company returned 503." },
            { title: "Rules Config Push Fails Silently", symptom: "Admin lowers rate limit to 10/min for abusive user. User continues making 100/min for 60+ seconds.",
              cause: "Config push via gRPC stream disconnected. Gateway pods have stale rules. Local cache TTL hasn't expired yet (60s). No monitoring on config staleness.",
              fix: "Add metric: config_last_updated_timestamp per gateway pod. Alert if any pod's config age > 2× TTL. Add /admin/config-status endpoint showing each pod's rule version. Dual-path: push + pull (TTL refresh).",
              quote: "We thought the rate limit was applied. Abuse team said 'they're still going.' Config push had silently died 4 hours ago." },
            { title: "Rate Limiting Breaks Retry Logic", symptom: "Downstream service retries on 500 errors. Rate limiter counts retries as new requests. Legitimate retry traffic gets 429'd. Cascading failure.",
              cause: "Service A calls Service B. B returns 500. A retries 3 times (standard retry with backoff). Rate limiter counts all 4 attempts. If user is near limit, retries push them over.",
              fix: "Don't rate-limit server-to-server retries (identify via X-Retry-Count header or internal service identity). Or use separate, higher limits for internal traffic. Or exclude 5xx-retries from rate counting entirely.",
              quote: "Payment service had a blip. Retry storms from checkout service hit rate limits. Users couldn't complete purchases for 10 minutes." },
            { title: "Forgot to Rate-Limit the /health Endpoint", symptom: "Monitoring system hits /health every 5s from 200 instances = 40 QPS. Health checks counted against rate limit. Real user traffic gets throttled.",
              cause: "Rate limiter applied globally to all endpoints. Health checks, readiness probes, and internal monitoring all consume rate limit quota.",
              fix: "Allowlist internal IPs and monitoring endpoints from rate limiting. Or apply rate limiting AFTER the /health route. Or use a separate listener port for health checks (common K8s pattern: port 8080 for traffic, 8081 for health).",
              quote: "Spent 2 hours debugging why users were getting 429d. It was Kubernetes liveness probes eating their rate limit quota." },
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
        { t: "Multi-Dimension Limiting", d: "Limit per user + per endpoint + per IP simultaneously. Request must pass ALL.", detail: "User 42: 100/min globally AND /search: 10/min per user AND IP 1.2.3.4: 1000/min total.", effort: "Medium" },
        { t: "Adaptive Rate Limiting", d: "Dynamically tighten limits when server CPU > 80%, relax when healthy.", detail: "Feedback loop: limiter protects the system from itself. Requires health signal integration.", effort: "Medium" },
        { t: "Request Queuing (Leaky Bucket)", d: "Instead of 429, queue the request and process when capacity available.", detail: "Returns 202 Accepted. Client polls for result. Better UX for background/async tasks.", effort: "Medium" },
        { t: "Distributed Gossip Protocol", d: "Each server tracks local counts, gossips with peers to sync.", detail: "No Redis dependency. Eventually consistent. Good for extreme throughput scenarios.", effort: "Hard" },
        { t: "User Dashboards & Alerts", d: "Proactively warn users approaching limits. Dashboard showing usage vs quota.", detail: "GET /rate-limit/status endpoint. Email/webhook at 80% usage. Self-service tier upgrades.", effort: "Easy" },
        { t: "IP Reputation & Allowlisting", d: "Bypass limits for trusted IPs (monitoring, internal). Stricter for flagged IPs.", detail: "Allowlist/denylist in Redis checked before token bucket. Internal services skip limiting.", effort: "Easy" },
      ].map((e,i) => (
        <Card key={i}>
          <div className="flex items-center justify-between mb-2">
            <div className="text-[11px] font-bold text-stone-800">{e.t}</div>
            <Pill bg={e.effort==="Easy"?"#ecfdf5":"#fffbeb"} color={e.effort==="Easy"?"#059669":"#d97706"}>{e.effort}</Pill>
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
    { q:"What if Redis goes down?", a:"Fail-open: allow all requests. Circuit breaker with 50ms timeout. Fall back to local in-memory token bucket (per-server, imprecise). For security-critical APIs (payments), consider fail-closed with cached allowlist.", tags:["availability"] },
    { q:"How do you rate-limit unauthenticated users?", a:"Use IP address as identifier. Beware NAT/proxies (corporate networks share IPs). Combine IP + User-Agent + fingerprinting. Set higher limits for shared IPs. For truly anonymous traffic, IP-based is the standard.", tags:["design"] },
    { q:"How would you handle rate limiting across regions?", a:"Three options: (1) Local Redis per region — low latency, separate quotas. (2) Central Redis — accurate, high latency. (3) Split quota — divide limit proportionally (US:60, EU:40), sync periodically. Option 3 is the best balance.", tags:["scalability"] },
    { q:"Token Bucket vs Leaky Bucket?", a:"Token Bucket allows bursts (process immediately if tokens available). Leaky Bucket enforces smooth constant output (requests queued at fixed rate). Token Bucket for APIs (allow bursts). Leaky Bucket for traffic shaping (smooth output).", tags:["algorithm"] },
    { q:"How is this different from DDoS protection?", a:"Rate limiting is L7 (per-user/per-key). DDoS protection is L3/L4 (drops traffic by IP/geo/packet pattern before reaching your servers). Use both: Cloudflare/AWS Shield for DDoS, your rate limiter for app-level abuse.", tags:["design"] },
    { q:"How do you test a rate limiter?", a:"Unit: verify refill math, boundary conditions. Integration: Redis + N+1 requests, verify 429. Load: 10× limit from multiple threads. Chaos: kill Redis mid-test, verify fail-open. Accuracy: measure actual vs expected allowed count.", tags:["testing"] },
    { q:"Rate limiting WebSocket connections?", a:"Two dimensions: (1) Connection rate — limit new WS connections/user/minute. (2) Message rate — per-connection messages/second. Messages use in-memory token bucket per connection (no Redis needed — per-connection state on single server).", tags:["design"] },
    { q:"Can you do this without Redis?", a:"Single-server: in-memory dict + threading.Lock. Distributed: (1) Sticky sessions (same user → same server). (2) Gossip protocol. (3) Client-side limiting with API keys. Redis is standard because it's simple and fast.", tags:["design"] },
    { q:"How would you monitor the rate limiter?", a:"Metrics: rejection rate (alert >5%), Redis p99 latency, per-user rejection counts, memory usage. Dashboards: Grafana + Redis metrics. Alerts: PagerDuty on Redis down, high rejection rate, latency spike.", tags:["observability"] },
    { q:"What about cost at scale?", a:"Redis is cheap: 3-node r6g.large cluster ~$150-300/month handles 175K QPS. Compare to the cost of your backend being DDoS'd or a single user consuming $10K of compute. Rate limiting ROI is extremely high.", tags:["cost"] },
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

export default function RateLimiterSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Rate Limiter</h1>
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