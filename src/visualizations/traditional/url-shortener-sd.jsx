import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   URL SHORTENER — System Design Reference
   Pearl white theme · Reusable section structure
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",             icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",         icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",  icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",           icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",    icon: "🏗️", color: "#9333ea" },
  { id: "algorithm",     label: "Key Generation",       icon: "⚙️", color: "#c026d3" },
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
            <Label>What is a URL Shortener?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A URL shortener takes a long URL and generates a compact, unique alias that redirects to the original destination. When a user visits the short link, the service looks up the original URL and issues an HTTP redirect (301 or 302).
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Think of it like a library call number: instead of memorizing the full shelf location "Building 3, Floor 2, Aisle 7, Shelf 4, Position 12," you just use "QA76.73" — a short code that maps to the exact location. The shortener maintains this mapping and does the translation.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Do We Need It?</Label>
            <ul className="space-y-2.5">
              <Point icon="📱" color="#0891b2">Character limits — Twitter/SMS have strict character limits; short URLs save space</Point>
              <Point icon="📊" color="#0891b2">Analytics — track click-through rates, geographic distribution, referral sources</Point>
              <Point icon="🎨" color="#0891b2">Aesthetics — clean, branded links are more shareable and trustworthy</Point>
              <Point icon="🔗" color="#0891b2">Link management — update destinations without changing the shared link</Point>
              <Point icon="🛡️" color="#0891b2">Obfuscation — hide complex query parameters and affiliate tracking codes</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Examples</Label>
            <div className="space-y-2.5">
              {[
                { co: "Bitly", rule: "bit.ly/abc123 → original URL", algo: "Base62 + custom aliases" },
                { co: "TinyURL", rule: "tinyurl.com/y7abc → long URL", algo: "Auto-increment + Base62" },
                { co: "t.co", rule: "Twitter's built-in shortener", algo: "Per-tweet, automatic" },
                { co: "goo.gl", rule: "Google (deprecated)", algo: "Base62 hash" },
                { co: "Rebrandly", rule: "Branded domain short links", algo: "Custom domain + alias" },
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
            <Label color="#2563eb">How It Works</Label>
            <svg viewBox="0 0 360 180" className="w-full">
              <DiagramBox x={55} y={50} w={80} h={38} label="Client" color="#2563eb"/>
              <DiagramBox x={185} y={50} w={90} h={42} label="URL\nShortener" color="#9333ea"/>
              <DiagramBox x={320} y={50} w={80} h={38} label="Database" color="#059669"/>
              <DiagramBox x={185} y={135} w={90} h={38} label="301/302" color="#dc2626"/>
              <Arrow x1={95} y1={50} x2={140} y2={50} id="c1"/>
              <Arrow x1={230} y1={50} x2={280} y2={50} label="lookup" id="c2"/>
              <Arrow x1={185} y1={71} x2={185} y2={116} label="redirect" id="c3" dashed/>
              <rect x={118} y={157} width={134} height={18} rx={4} fill="#dc262608" stroke="#dc262630"/>
              <text x={185} y={167} textAnchor="middle" fill="#dc2626" fontSize="8" fontFamily="monospace">read-heavy: 100:1 ratio</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Google, Amazon, Meta, Microsoft, Uber, Bloomberg</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Always Start Here</div>
            <p className="text-[12px] text-stone-500 mt-0.5">Spend 3-5 minutes clarifying requirements before designing. Ask about read:write ratio, URL length, custom aliases, analytics needs, and expiration policy. This shows structured thinking and prevents wasted effort designing the wrong system.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Given a long URL, generate a short, unique alias (e.g., short.ly/abc123)</Point>
            <Point icon="2." color="#059669">When user accesses short URL, redirect to the original long URL (301/302)</Point>
            <Point icon="3." color="#059669">Support custom aliases — users can choose their own short key</Point>
            <Point icon="4." color="#059669">Links should expire after a configurable TTL (default: never)</Point>
            <Point icon="5." color="#059669">Track click analytics — total clicks, unique visitors, geographic data</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Low latency — redirect must complete in &lt;50ms (p99)</Point>
            <Point icon="2." color="#dc2626">Highly available — redirect service cannot go down (write path can degrade)</Point>
            <Point icon="3." color="#dc2626">Read-heavy — 100:1 read-to-write ratio is typical</Point>
            <Point icon="4." color="#dc2626">Uniqueness — no two long URLs map to the same short key (collision-free)</Point>
            <Point icon="5." color="#dc2626">Non-guessable — short URLs should not be easily predictable/enumerable</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask the Interviewer</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "What is the expected traffic volume? Reads/sec? Writes/sec?",
            "How long should the short URL be? 6-8 characters?",
            "Should the same long URL always map to the same short URL?",
            "Do we need custom aliases? (e.g., short.ly/my-brand)",
            "Should URLs expire? After how long?",
            "Do we need click analytics? Real-time or batched?",
            "301 (permanent) or 302 (temporary) redirect?",
            "Do we need to support URL deletion or editing?",
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
            <p className="text-[12px] text-stone-500 mt-0.5">Walk through estimation out loud. Round aggressively — interviewers care about your process and order-of-magnitude reasoning, not exact numbers. State assumptions clearly: <em>"Let me assume 100M new URLs/month..."</em></p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Traffic Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="New URLs created per month = 100M" result="100M" note='Assumption — ask interviewer or state: "Let me assume a Bitly-scale service"' />
            <MathStep step="2" formula="Writes per second = 100M / (30 × 86,400)" result="~40 writes/s" note="100M / 2.6M seconds per month ≈ 40 URL creations per second" />
            <MathStep step="3" formula="Read:Write ratio = 100:1" result="100:1" note="Short URLs are clicked far more often than created" />
            <MathStep step="4" formula="Reads per second = 40 × 100" result="~4,000 QPS" note="4K redirect lookups per second average" final />
            <MathStep step="5" formula="Peak QPS = Avg × 3 (peak multiplier)" result="~12K QPS" note="Industry standard: peak = 2-3× average. Use 3× to be safe." final />
          </div>
        </Card>

        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Storage Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Storage per URL entry:" result="~500 B" note="short_key (7B) + long_url (avg 200B) + created_at (8B) + expires_at (8B) + user_id (8B) + metadata (~270B)" />
            <MathStep step="2" formula="New URLs per month = 100M" result="100M" note="Each URL stored once, immutable after creation" />
            <MathStep step="3" formula="Storage per month = 100M × 500 bytes" result="50 GB/mo" note="10⁸ × 500 = 5 × 10¹⁰ bytes" />
            <MathStep step="4" formula="5-year storage = 50 GB × 60 months" result="3 TB" note="Total storage over 5 years — easily fits on modern SSDs" final />
            <MathStep step="5" formula="Total URLs in 5 years = 100M × 60" result="6 Billion" note="6 × 10⁹ URLs. Key for determining short key length." final />
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Key Space Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Characters: [a-zA-Z0-9] = 62 chars" result="Base62" note="62 alphanumeric characters for URL-safe encoding" />
            <MathStep step="2" formula="6-char key: 62⁶ combinations" result="56.8 Billion" note="62⁶ = 56,800,235,584. Enough for 6B URLs with 9× headroom." />
            <MathStep step="3" formula="7-char key: 62⁷ combinations" result="3.5 Trillion" note="62⁷ = 3,521,614,606,208. Massive headroom — preferred." final />
            <MathStep step="4" formula="At 100M/month, 7-char lasts" result="~2,930 years" note="3.5T / 100M per month / 12 months. Never runs out." final />
          </div>
        </Card>

        <Card accent="#059669">
          <Label color="#059669">Step 4 — Cache & Bandwidth</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Cache hot URLs: 80/20 rule" result="20%" note="20% of URLs generate 80% of traffic — cache these" />
            <MathStep step="2" formula="Daily reads = 4,000 QPS × 86,400" result="~346M/day" note="Total redirect requests per day" />
            <MathStep step="3" formula="Cache 20% of daily URLs = 346M × 0.2" result="~70M entries" note="70M entries × 500 bytes each" />
            <MathStep step="4" formula="Cache memory = 70M × 500 bytes" result="~35 GB" note="Fits in a single Redis node (common: 64-256 GB)" final />
            <MathStep step="5" formula="Bandwidth: 4K QPS × ~500B response" result="~2 MB/s" note="Minimal — URL metadata is tiny. Not a bottleneck." />
          </div>
          <div className="mt-4 pt-3 border-t border-stone-100">
            <div className="text-[11px] text-stone-500">
              <strong className="text-stone-700">Key point:</strong> URL shorteners are a perfect caching use case. URLs are immutable after creation, making cache invalidation trivial (it almost never happens). A single Redis node can cache all hot URLs.
            </div>
          </div>
        </Card>
      </div>

      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Write QPS", val: "~40/s", sub: "Peak: ~120/s" },
            { label: "Read QPS", val: "~4K/s", sub: "Peak: ~12K/s" },
            { label: "5yr Storage", val: "~3 TB", sub: "6B URLs total" },
            { label: "Cache Memory", val: "~35 GB", sub: "Hot 20% of URLs" },
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
          <Label color="#2563eb">Create Short URL</Label>
          <CodeBlock code={`# POST /api/v1/urls
# Create a new short URL
{
  "long_url": "https://example.com/very/long/path?q=...",
  "custom_alias": "my-brand",     # optional
  "expires_at": "2026-12-31T00:00:00Z",  # optional
}

# Response — 201 Created
{
  "short_url": "https://short.ly/abc123",
  "short_key": "abc123",
  "long_url": "https://example.com/very/long/path?q=...",
  "created_at": "2026-02-09T10:30:00Z",
  "expires_at": "2026-12-31T00:00:00Z",
}

# Error — 409 Conflict (custom alias taken)
{
  "error": "custom_alias_taken",
  "message": "Alias 'my-brand' is already in use"
}

# Error — 400 Bad Request
{
  "error": "invalid_url",
  "message": "The provided URL is not valid"
}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Redirect (Core Read Path)</Label>
          <CodeBlock code={`# GET /:short_key
# e.g., GET /abc123
#
# Response — 301 Moved Permanently
# (or 302 Found for analytics tracking)
HTTP/1.1 301 Moved Permanently
Location: https://example.com/very/long/path?q=...
Cache-Control: max-age=86400

# 301 vs 302 — The Key Decision:
# 301: Browser caches redirect permanently.
#      Faster for users (no server hit after first visit).
#      BAD for analytics (subsequent clicks invisible).
#
# 302: Browser does NOT cache redirect.
#      Every click hits the server.
#      GOOD for analytics tracking.
#
# Recommendation: Use 302 if analytics matter,
#                 301 if performance is top priority.

# Error — 404 Not Found
{
  "error": "url_not_found",
  "message": "Short URL does not exist"
}

# Error — 410 Gone (expired)
{
  "error": "url_expired",
  "message": "This short URL has expired"
}`} />
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#d97706">
          <Label color="#d97706">Analytics & Management APIs</Label>
          <CodeBlock code={`# GET /api/v1/urls/:short_key/stats
# Get analytics for a short URL
{
  "short_key": "abc123",
  "total_clicks": 14523,
  "unique_visitors": 8901,
  "created_at": "2026-01-15T10:30:00Z",
  "last_clicked": "2026-02-09T08:45:00Z",
  "clicks_by_country": {
    "US": 5420, "UK": 2103, "DE": 1890
  },
  "clicks_by_day": [
    {"date": "2026-02-08", "clicks": 342},
    {"date": "2026-02-09", "clicks": 128}
  ]
}

# DELETE /api/v1/urls/:short_key
# Soft-delete a short URL (returns 410 on future access)

# GET /api/v1/urls?user_id=42&cursor=...&limit=20
# List user's short URLs (paginated)`} />
        </Card>
        <Card accent="#9333ea">
          <Label color="#9333ea">Design Decisions</Label>
          <div className="space-y-3">
            {[
              { header: "301 vs 302 Redirect", desc: "301 = permanent (browser caches), 302 = temporary (server sees every click)", ex: "Use 302 if you need analytics" },
              { header: "API Key Authentication", desc: "Rate limit and attribute URL creation to users via API keys", ex: "X-API-Key: sk_live_abc123" },
              { header: "URL Validation", desc: "Verify the long URL is well-formed and reachable before creating", ex: "HEAD request to check URL exists" },
              { header: "Idempotency", desc: "Same long URL from same user returns existing short URL (optional)", ex: "Deduplication via hash index" },
            ].map((h,i) => (
              <div key={i} className="flex items-start gap-3">
                <code className="text-[11px] font-mono font-bold text-purple-700 bg-purple-50 px-2 py-0.5 rounded shrink-0">{h.header}</code>
                <div>
                  <div className="text-[12px] text-stone-600">{h.desc}</div>
                  <div className="text-[10px] text-stone-400 font-mono">Example: {h.ex}</div>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-stone-100">
            <Label color="#d97706">Rate Limiting the Write API</Label>
            <ul className="space-y-1.5">
              <Point icon="→" color="#d97706">Free tier: 100 URLs/day — prevents spam/abuse</Point>
              <Point icon="→" color="#d97706">Authenticated users: 10,000 URLs/day — reasonable for businesses</Point>
              <Point icon="→" color="#d97706">Redirect path: NO rate limiting — must always be fast and available</Point>
              <Point icon="→" color="#d97706">Consider CAPTCHA for unauthenticated URL creation</Point>
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
    { label: "Start Simple", desc: "Single server with a database. Client sends long URL, server generates key, stores mapping, returns short URL. Redirect looks up the key in the DB. Simple but doesn't scale — DB becomes the bottleneck for reads." },
    { label: "Add Cache", desc: "Put a cache (Redis) between the app server and DB. On redirect, check cache first — cache hit skips the DB entirely. 80/20 rule: caching 20% of URLs handles 80% of redirects. Massive latency improvement." },
    { label: "Scale Horizontally", desc: "Add multiple app servers behind a load balancer. Separate read and write paths. Reads go through cache → DB. Writes go through a key generation service to avoid collisions. DB read replicas for read scaling." },
    { label: "Full Architecture", desc: "Complete: LB → App Server → Cache (Redis Cluster) for reads. Key Generation Service (pre-generated keys from KGS) for writes. DB with sharding by key hash. Analytics pipeline (async) via Kafka. CDN for further read optimization." },
  ];
  const diagrams = [
    () => (
      <svg viewBox="0 0 460 170" className="w-full">
        <DiagramBox x={60} y={65} w={75} h={38} label="Client" color="#2563eb"/>
        <DiagramBox x={210} y={65} w={95} h={42} label="App\nServer" color="#9333ea"/>
        <DiagramBox x={370} y={65} w={85} h={42} label="Database" color="#059669"/>
        <Arrow x1={97} y1={65} x2={163} y2={65} id="s0a"/>
        <Arrow x1={258} y1={65} x2={328} y2={65} label="read/write" id="s0b"/>
        <rect x={135} y={120} width={220} height={22} rx={6} fill="#dc262608" stroke="#dc262630"/>
        <text x={245} y={132} textAnchor="middle" fill="#dc2626" fontSize="9" fontFamily="monospace">⌀ DB hit on every redirect</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 190" className="w-full">
        <DiagramBox x={55} y={75} w={68} h={36} label="Client" color="#2563eb"/>
        <DiagramBox x={160} y={75} w={80} h={36} label="App Server" color="#9333ea"/>
        <DiagramBox x={300} y={45} w={70} h={36} label="Cache" color="#dc2626"/>
        <DiagramBox x={300} y={110} w={70} h={36} label="Database" color="#059669"/>
        <Arrow x1={89} y1={75} x2={120} y2={75} id="r0"/>
        <Arrow x1={200} y1={65} x2={265} y2={50} label="check" id="r1"/>
        <Arrow x1={200} y1={85} x2={265} y2={105} label="miss" id="r2" dashed/>
        <rect x={240} y={155} width={120} height={20} rx={4} fill="#05966908" stroke="#05966930"/>
        <text x={300} y={166} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">✓ Cache absorbs 80% reads</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 200" className="w-full">
        <DiagramBox x={45} y={85} w={60} h={36} label="Client" color="#2563eb"/>
        <DiagramBox x={125} y={85} w={46} h={36} label="LB" color="#64748b"/>
        <DiagramBox x={235} y={50} w={80} h={36} label="App Srv 1" color="#9333ea"/>
        <DiagramBox x={235} y={120} w={80} h={36} label="App Srv 2" color="#9333ea"/>
        <DiagramBox x={370} y={50} w={60} h={36} label="Cache" color="#dc2626"/>
        <DiagramBox x={370} y={120} w={60} h={36} label="DB" color="#059669"/>
        <Arrow x1={75} y1={85} x2={102} y2={85} id="ru0"/>
        <Arrow x1={148} y1={75} x2={195} y2={55} id="ru1"/>
        <Arrow x1={148} y1={95} x2={195} y2={115} id="ru2"/>
        <Arrow x1={275} y1={50} x2={340} y2={50} label="read" id="ru3"/>
        <Arrow x1={275} y1={120} x2={340} y2={120} label="write" id="ru4"/>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 235" className="w-full">
        <DiagramBox x={42} y={55} w={56} h={34} label="Client" color="#2563eb"/>
        <DiagramBox x={118} y={55} w={46} h={34} label="LB" color="#64748b"/>
        <DiagramBox x={218} y={55} w={75} h={42} label="App\nServer" color="#9333ea"/>
        <DiagramBox x={340} y={30} w={65} h={30} label="Cache" color="#dc2626"/>
        <DiagramBox x={340} y={80} w={65} h={30} label="DB" color="#059669"/>
        <DiagramBox x={218} y={155} w={70} h={36} label="KGS" color="#c026d3"/>
        <DiagramBox x={100} y={155} w={65} h={34} label="Analytics" color="#0284c7"/>
        <DiagramBox x={340} y={155} w={65} h={34} label="Kafka" color="#d97706"/>
        <Arrow x1={70} y1={55} x2={95} y2={55} id="f0"/>
        <Arrow x1={141} y1={55} x2={181} y2={55} id="f1"/>
        <Arrow x1={256} y1={45} x2={308} y2={35} id="f2"/>
        <Arrow x1={256} y1={65} x2={308} y2={75} id="f3"/>
        <Arrow x1={218} y1={76} x2={218} y2={137} label="keys" id="f4" dashed/>
        <Arrow x1={256} y1={155} x2={308} y2={155} label="clicks" id="f5" dashed/>
        <Arrow x1={133} y1={155} x2={183} y2={155} id="f6" dashed/>
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
          <Label color="#059669">Request Flow — Create Short URL (Write)</Label>
          <div className="space-y-1.5">
            {[
              { s:"1", t:"Client sends POST /api/v1/urls with long_url in body", c:"text-blue-600" },
              { s:"2", t:"App server validates URL format and reachability (HEAD check)", c:"text-stone-500" },
              { s:"3", t:"If custom alias: check DB if alias is taken (fail 409 if so)", c:"text-purple-600" },
              { s:"4", t:"If auto-generated: fetch pre-generated key from KGS", c:"text-fuchsia-600" },
              { s:"5", t:"Store mapping: short_key → long_url in database", c:"text-emerald-600" },
              { s:"6", t:"Write-through: populate cache with new mapping", c:"text-red-600" },
              { s:"7", t:"Return 201 with short URL to client", c:"text-emerald-600" },
            ].map((s,i) => (
              <div key={i} className="flex items-start gap-2 text-[12px]">
                <span className="text-stone-300 font-mono w-4 shrink-0">{s.s}</span>
                <span className={s.c}>{s.t}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <Label color="#dc2626">Request Flow — Redirect (Read)</Label>
          <div className="space-y-1.5">
            {[
              { s:"1", t:"Client sends GET /abc123 (follows short URL)", c:"text-blue-600" },
              { s:"2", t:"App server extracts short_key from URL path", c:"text-stone-500" },
              { s:"3", t:"Check cache (Redis) for key → long_url mapping", c:"text-red-600" },
              { s:"4", t:"Cache HIT → return 302 redirect immediately", c:"text-emerald-600" },
              { s:"5", t:"Cache MISS → query database for mapping", c:"text-amber-600" },
              { s:"6", t:"Populate cache with result for future requests", c:"text-red-600" },
              { s:"7", t:"Async: log click event to Kafka for analytics", c:"text-sky-600" },
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
  const [sel, setSel] = useState("base62_counter");
  const algos = {
    md5_hash: { name: "MD5/SHA Hash + Truncate", cx: "O(1) / O(1)",
      pros: ["Simple — hash the long URL, take first 7 chars","Deterministic — same URL always gives same key","No coordination needed between servers"],
      cons: ["Collisions — truncating a hash means collisions are likely","Collision resolution adds complexity and extra DB lookups","Not truly random — attackers can predict hashes"],
      when: "Avoid in production. The collision probability is too high at scale — with 6B URLs and 7-char truncation, you'll see frequent collisions requiring multiple retries.",
      code: `# MD5 Hash + Truncate (NOT Recommended)
import hashlib, base64

def generate_key(long_url):
    md5 = hashlib.md5(long_url.encode()).digest()
    b62 = base64.b64encode(md5)[:7].decode()
    # Problem: collisions are likely!
    # Must check DB and retry with salt:
    while db.exists(b62):
        long_url += str(random.randint(0,9))  # salt
        md5 = hashlib.md5(long_url.encode()).digest()
        b62 = base64.b64encode(md5)[:7].decode()
    return b62` },
    base62_counter: { name: "Counter + Base62 ★", cx: "O(1) / O(1)",
      pros: ["Zero collisions — each counter value is unique","Simple to implement and reason about","Short keys — starts at 1 char, grows as needed","Production standard (Bitly, TinyURL)"],
      cons: ["Sequential — keys are predictable/enumerable","Requires centralized counter (single point of failure)","Distributed counter adds complexity"],
      when: "Best general-purpose choice. Use a centralized auto-increment counter and convert to Base62. Combine with key generation service (KGS) for distributed systems.",
      code: `# Counter + Base62 (Production Standard)
CHARS = "0123456789abcdefghijklmnop" \\
        "qrstuvwxyzABCDEFGHIJKLMNOP" \\
        "QRSTUVWXYZ"  # 62 chars

def to_base62(num):
    if num == 0:
        return CHARS[0]
    result = []
    while num > 0:
        result.append(CHARS[num % 62])
        num //= 62
    return ''.join(reversed(result))

# Counter from DB auto-increment or Snowflake
# counter = 1 → "1"
# counter = 1000 → "g8"
# counter = 1000000 → "4c92"
# counter = 56800235584 → "zzzzzzz" (7 chars max)` },
    kgs: { name: "Key Generation Service (KGS)", cx: "O(1) / O(1)",
      pros: ["Pre-generated — no computation at request time","Zero collisions — keys are unique by construction","Decoupled — app servers just fetch keys, no coordination","Fast — O(1) key fetch from pre-computed pool"],
      cons: ["Extra service to maintain","If KGS dies, new URLs can't be created (writes fail)","Pre-generated keys consume storage","Must handle key exhaustion gracefully"],
      when: "Best for high-scale production. Generate keys offline in batches, store in a DB table. App servers fetch a batch (e.g., 1000 keys) at startup and dispense locally. When batch runs low, fetch more.",
      code: `# Key Generation Service (KGS)
# Pre-generate keys and store in DB:
#
# keys_table:
#   key_value  | is_used
#   "abc123"   | false
#   "def456"   | false
#   ...
#
# App server on startup:
def get_key_batch(batch_size=1000):
    # Atomic: SELECT + UPDATE in transaction
    keys = db.execute("""
        UPDATE keys_table
        SET is_used = true
        WHERE key_value IN (
            SELECT key_value FROM keys_table
            WHERE is_used = false
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        )
        RETURNING key_value
    """, batch_size)
    return keys

# Server keeps keys in memory, dispenses one per request
# When batch < 20% remaining, fetch new batch async` },
    snowflake: { name: "Snowflake ID + Base62", cx: "O(1) / O(1)",
      pros: ["Globally unique — no coordination between servers","Roughly time-ordered — useful for analytics","Proven at scale (Twitter, Discord, Instagram)","No central counter needed"],
      cons: ["64-bit IDs → 11 chars in Base62 (longer than 7)","Requires clock synchronization across servers","More complex than simple counter"],
      when: "When you need distributed uniqueness without any central coordination. The trade-off is slightly longer keys (11 chars vs 7). Good if you're already using Snowflake IDs elsewhere.",
      code: `# Snowflake ID → Base62
# 64-bit ID structure:
# [1 bit unused][41 bits timestamp][10 bits machine][12 bits sequence]
#
# 41 bits = ~69 years of milliseconds
# 10 bits = 1024 machines
# 12 bits = 4096 IDs per ms per machine

class SnowflakeGenerator:
    EPOCH = 1704067200000  # Jan 1, 2024

    def __init__(self, machine_id):
        self.machine_id = machine_id & 0x3FF
        self.sequence = 0
        self.last_ts = -1

    def next_id(self):
        ts = int(time.time() * 1000) - self.EPOCH
        if ts == self.last_ts:
            self.sequence = (self.sequence + 1) & 0xFFF
        else:
            self.sequence = 0
        self.last_ts = ts
        return (ts << 22) | (self.machine_id << 12) | self.sequence

# to_base62(next_id()) → "1a2B3c4D5eF" (11 chars)` },
  };
  const a = algos[sel];
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Key Generation — Algorithm Comparison</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[12px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Algorithm</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Collisions</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Key Length</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Coordination</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Best For</th>
            </tr></thead>
            <tbody>
              {[
                { n:"MD5 Hash + Truncate", c:"Likely ⚠", l:"7 chars", co:"None", f:"Avoid", hl:false },
                { n:"Counter + Base62 ★", c:"Zero ✓", l:"1-7 chars", co:"Central counter", f:"Most systems", hl:true },
                { n:"KGS (Pre-generated)", c:"Zero ✓", l:"7 chars", co:"KGS service", f:"High-scale prod", hl:false },
                { n:"Snowflake + Base62", c:"Zero ✓", l:"11 chars", co:"Clock sync only", f:"Distributed", hl:false },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-purple-50" : i%2?"bg-stone-50/50":""}>
                  <td className={`px-3 py-2 font-mono ${r.hl?"text-purple-700 font-bold":"text-stone-600"}`}>{r.n}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.c}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.l}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.co}</td>
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
          <Label color="#dc2626">Primary Table — URL Mappings</Label>
          <CodeBlock code={`# urls table (PostgreSQL / MySQL)
CREATE TABLE urls (
  short_key   VARCHAR(7) PRIMARY KEY,   -- Base62 key
  long_url    TEXT NOT NULL,             -- original URL (up to 2048 chars)
  user_id     BIGINT,                   -- creator (nullable for anon)
  created_at  TIMESTAMP DEFAULT NOW(),
  expires_at  TIMESTAMP,                -- NULL = never expires
  is_active   BOOLEAN DEFAULT TRUE,     -- soft delete flag
  click_count BIGINT DEFAULT 0          -- denormalized for quick access
);

-- Index for reverse lookup (optional: same URL → existing key)
CREATE INDEX idx_urls_long_url_hash
  ON urls USING HASH (long_url);

-- Index for user's URLs listing
CREATE INDEX idx_urls_user_id
  ON urls (user_id, created_at DESC);

-- Index for expiration cleanup job
CREATE INDEX idx_urls_expires_at
  ON urls (expires_at)
  WHERE expires_at IS NOT NULL AND is_active = TRUE;`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Analytics Table — Click Events</Label>
          <CodeBlock code={`# click_events table (ClickHouse / Cassandra)
# Write-optimized, append-only, time-series data
CREATE TABLE click_events (
  event_id    UUID DEFAULT gen_random_uuid(),
  short_key   VARCHAR(7) NOT NULL,
  clicked_at  TIMESTAMP DEFAULT NOW(),
  ip_address  INET,
  user_agent  TEXT,
  referrer    TEXT,
  country     VARCHAR(2),            -- GeoIP lookup
  device_type VARCHAR(10),           -- mobile/desktop/tablet
  browser     VARCHAR(20)
);

-- Partitioned by month for efficient cleanup
-- In ClickHouse: ORDER BY (short_key, clicked_at)
-- In Cassandra: PARTITION KEY (short_key, month)

# KGS table (if using Key Generation Service)
CREATE TABLE pre_generated_keys (
  key_value  VARCHAR(7) PRIMARY KEY,
  is_used    BOOLEAN DEFAULT FALSE
);
-- Pre-populate with millions of random Base62 keys`} />
        </Card>
      </div>
      <Card accent="#9333ea">
        <Label color="#9333ea">Database Choice — The Core Decision</Label>
        <p className="text-[12px] text-stone-500 mb-4">URL shorteners have a simple data model but extreme read volume. The database choice depends on scale and consistency requirements.</p>
        <div className="grid grid-cols-3 gap-4">
          {[
            { t: "PostgreSQL / MySQL ★", d: "Relational DB with strong consistency. Short key as primary key gives O(1) lookups. Perfect for moderate scale.", pros: ["ACID guarantees","Rich indexing and queries","Mature ecosystem"], cons: ["Vertical scaling limits","Sharding is manual and complex"], pick: true },
            { t: "DynamoDB / Cassandra", d: "NoSQL key-value store. short_key → long_url is a perfect KV pattern. Horizontally scalable by design.", pros: ["Auto-sharding","Single-digit ms latency at any scale","No sharding complexity"], cons: ["Limited query flexibility","Eventually consistent reads"], pick: false },
            { t: "Redis (Cache Layer)", d: "In-memory cache in front of primary DB. Cache all hot URLs. 35 GB handles 70M entries — most redirects never hit the DB.", pros: ["Sub-ms reads","Perfect for immutable data (URLs don't change)","Simple invalidation (almost never needed)"], cons: ["Not durable alone — needs backing store","Memory cost at extreme scale"], pick: false },
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
            <strong className="text-stone-700">Key insight to mention in interview:</strong> URL shortener data is append-mostly (URLs rarely updated/deleted) and the access pattern is a simple key-value lookup. This means both SQL and NoSQL work well. Start with PostgreSQL for simplicity. If you grow beyond a single DB's capacity, consider DynamoDB/Cassandra for automatic sharding. Redis cache in front of either handles the read-heavy workload.
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
          <Label color="#059669">Caching Strategy</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Redis Cluster</strong> — cache hot URLs. 80/20 rule: 20% of URLs = 80% of redirects. ~35 GB of cache handles most traffic.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Cache-aside pattern</strong> — on redirect: check cache → miss → query DB → populate cache. TTL = 24h (URLs rarely change).</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Write-through on create</strong> — when a new URL is created, immediately write to cache. First click is always a cache hit.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">No invalidation needed</strong> — URLs are immutable. Cache invalidation (the hardest problem in CS) is essentially a non-issue here.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Database Sharding</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Shard by short_key hash</strong> — consistent hashing on the short key distributes URLs evenly across DB shards.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Range-based won't work</strong> — Base62 keys don't distribute evenly by range. Hash-based sharding is required.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Read replicas</strong> — each shard has 1+ read replicas. Redirect reads go to replicas, writes to primary.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Cross-shard queries</strong> — listing "all URLs by user_id" requires scatter-gather across shards. Keep a separate user_urls index table.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Scaling at Each Layer</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Application Servers", d:"Stateless — scale horizontally behind a load balancer. Each server fetches a batch of pre-generated keys from KGS on startup. No shared state between servers.", pros:["Add/remove freely","Auto-scale on CPU/QPS"], cons:["KGS is a dependency for writes"], pick:false },
            { t:"Cache Layer (Redis) ★", d:"Redis Cluster with 3+ shards. Each shard caches a subset of keys. At 4K QPS, a single Redis node handles it easily. Scale only when cache size exceeds single node memory.", pros:["Sub-ms reads","Immutable data = perfect caching","35 GB handles 70M hot URLs"], cons:["Memory cost scales linearly"], pick:true },
            { t:"Database Layer", d:"Start with single PostgreSQL. At ~10K QPS reads, add read replicas. At ~50K QPS, shard by short_key hash. DynamoDB auto-scales if you want to avoid manual sharding.", pros:["Simple key-value access pattern","Shard key = primary key"], cons:["Cross-shard queries for user lists"], pick:false },
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
        <Label color="#d97706">Critical Decision: Read vs Write Availability</Label>
        <p className="text-[12px] text-stone-500 mb-4">The redirect path (read) MUST be highly available — it's the core product. The create path (write) can tolerate brief outages. Design accordingly.</p>
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Read Path (Redirect) — 99.99% Uptime</div>
            <p className="text-[11px] text-stone-500 mb-2">Redirects must ALWAYS work. This is what users see.</p>
            <ul className="space-y-1"><Point icon="✓" color="#059669">Multi-layer: CDN → Cache → DB replica</Point><Point icon="✓" color="#059669">Cache can serve reads even if DB is down</Point><Point icon="✓" color="#059669">DNS-level failover across regions</Point></ul>
          </div>
          <div className="rounded-lg border border-amber-200 bg-white p-4">
            <div className="text-[11px] font-bold text-amber-700 mb-1.5">Write Path (Create) — 99.9% Uptime</div>
            <p className="text-[11px] text-stone-500 mb-2">URL creation can tolerate brief outages.</p>
            <ul className="space-y-1"><Point icon="✓" color="#059669">KGS failure = can't create URLs (acceptable)</Point><Point icon="✓" color="#059669">DB primary failure = promote replica</Point><Point icon="⚠" color="#d97706">Writes degrade gracefully, reads unaffected</Point></ul>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#2563eb">Database High Availability</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#2563eb"><strong className="text-stone-700">Primary-Replica</strong> — async replication. Primary handles writes. 2+ replicas handle reads. Promote replica on primary failure.</Point>
            <Point icon="2." color="#2563eb"><strong className="text-stone-700">Multi-AZ deployment</strong> — primary and replicas in different availability zones. Survives AZ failures.</Point>
            <Point icon="3." color="#2563eb"><strong className="text-stone-700">Automated failover</strong> — use managed DB (RDS Multi-AZ) or Patroni for automated primary promotion in &lt;30 seconds.</Point>
            <Point icon="4." color="#2563eb"><strong className="text-stone-700">Backup strategy</strong> — daily full backups + continuous WAL archiving. Point-in-time recovery to any second.</Point>
          </ul>
        </Card>
        <Card>
          <Label color="#0891b2">Resiliency Patterns</Label>
          <ul className="space-y-2.5">
            <Point icon="🔄" color="#0891b2"><strong className="text-stone-700">Cache as fallback</strong> — if DB is completely down, Redis can still serve redirects for all cached URLs. Covers 80%+ of traffic.</Point>
            <Point icon="⏱" color="#0891b2"><strong className="text-stone-700">Aggressive timeouts</strong> — DB query timeout: 100ms. If no response, serve from cache or return 503.</Point>
            <Point icon="📊" color="#0891b2"><strong className="text-stone-700">Health checks</strong> — ping DB every 5s. If 3 consecutive failures, switch reads to replica immediately.</Point>
            <Point icon="🔌" color="#0891b2"><strong className="text-stone-700">KGS redundancy</strong> — run 2+ KGS instances. Each fetches key batches from different key ranges. No single point of failure for writes.</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#9333ea">Degradation Ladder</Label>
        <div className="flex gap-3 items-stretch mt-1">
          {[
            { label: "Full System", sub: "All services healthy", color: "#059669", status: "HEALTHY" },
            { label: "DB Failover", sub: "Replica promoted", color: "#d97706", status: "DEGRADED" },
            { label: "Cache-Only Reads", sub: "DB down, cache serves", color: "#ea580c", status: "FALLBACK" },
            { label: "Static 302", sub: "CDN cached redirects", color: "#dc2626", status: "EMERGENCY" },
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
              { metric: "redirect.latency_ms", type: "Histogram", desc: "p50, p95, p99 of redirect response time" },
              { metric: "redirect.cache_hit_rate", type: "Gauge", desc: "% of redirects served from cache — target >90%" },
              { metric: "redirect.not_found", type: "Counter", desc: "404s on redirect — possible enumeration attack" },
              { metric: "url.created", type: "Counter", desc: "New URLs created (by user_tier, auth_status)" },
              { metric: "url.expired", type: "Counter", desc: "URLs expired by cleanup job" },
              { metric: "kgs.keys_remaining", type: "Gauge", desc: "Pre-generated keys available — alert if <10K" },
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
              { name: "Redirect Latency Spike", rule: "p99 > 100ms for 3min", sev: "P1", action: "Check cache hit rate, DB latency, Redis health" },
              { name: "Cache Hit Rate Drop", rule: "cache_hit_rate < 80% for 5min", sev: "P2", action: "Redis evictions? Memory full? Cache warming issue?" },
              { name: "KGS Keys Low", rule: "keys_remaining < 10,000", sev: "P2", action: "Run key generation job immediately" },
              { name: "High 404 Rate", rule: "not_found rate > 5% for 5min", sev: "P3", action: "Possible enumeration attack — check IPs" },
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
              { q: "Redirect latency spike", steps: "Check: cache hit rate drop? Redis latency? DB connection pool exhausted? Recent deployment? Check slow query log." },
              { q: "URLs not resolving (404)", steps: "Check: replication lag? URL expired? Key in wrong shard? Cache and DB both miss? Verify with direct DB query." },
              { q: "Duplicate short keys created", steps: "Check: KGS returned same key twice? Race condition in counter? Check KGS transaction isolation level." },
              { q: "Click counts not updating", steps: "Check: Kafka consumer lag? Analytics pipeline down? ClickHouse ingestion errors? Check consumer group offsets." },
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
          { title: "Hash Collisions (MD5 Approach)", sev: "Critical", sevColor: "#dc2626",
            desc: "If using MD5 hash + truncation, two different long URLs can produce the same short key. At 6B URLs with 7-char keys, collision probability is significant.",
            fix: "Use counter-based or KGS approach — guaranteed unique keys by construction. If you must hash, check DB for collision and retry with salt. But this adds latency and complexity. Counter + Base62 is strictly better.",
            code: `Problem: MD5("url_a")[:7] == MD5("url_b")[:7]\n  → Both URLs map to same short key\n  → Second write overwrites first!\n\nSolution: Counter + Base62\n→ counter = 1000 → to_base62(1000) = "g8"\n→ counter = 1001 → to_base62(1001) = "g9"\n→ Every key is unique by construction\n→ Zero collision probability` },
          { title: "Hot Key Problem", sev: "Medium", sevColor: "#d97706",
            desc: "A viral short URL gets millions of clicks/sec. Single cache entry becomes a hot key, overwhelming one Redis shard or DB partition.",
            fix: "CDN absorbs most traffic (cache 302 redirects at edge). Redis read replicas distribute hot key reads. For extreme cases, replicate hot key across multiple cache shards manually. Monitor per-key QPS and auto-promote to CDN cache.",
            code: `Problem: short.ly/viral → 1M clicks/sec\n  → Single Redis shard overwhelmed\n  → DB shard overwhelmed\n\nSolution: Multi-layer caching\n→ CDN edge cache (Cloudflare/CloudFront)\n→ Cache-Control: max-age=300 on 302\n→ 99% of clicks never reach origin\n→ Monitor: per-key QPS > 10K → CDN promote` },
          { title: "URL Enumeration Attack", sev: "Medium", sevColor: "#d97706",
            desc: "Sequential counter keys (1, 2, 3...) allow attackers to enumerate all URLs by incrementing the key. Exposes private/sensitive URLs.",
            fix: "Don't use sequential Base62 directly. Either: (1) use KGS with randomly generated keys, (2) add a random offset to counter before Base62 encoding, (3) XOR counter with a secret before encoding. The key should look random even if the underlying counter is sequential.",
            code: `Problem: to_base62(1) = "1"\n         to_base62(2) = "2"\n         → Attacker iterates all URLs!\n\nSolution: Randomize appearance\n→ XOR counter with secret before Base62\n→ or use KGS with random keys\n→ "abc123" → "xK9mQ2" → "pL4nR7"\n→ Looks random, still unique` },
          { title: "Cache Stampede on Cold Start", sev: "Critical", sevColor: "#dc2626",
            desc: "After a cache restart or eviction, thousands of requests for the same popular URL all miss cache simultaneously and hit the DB, overwhelming it.",
            fix: "Use cache locking (singleflight pattern): first request that misses cache acquires a lock, queries DB, populates cache. All other requests for same key wait for the lock holder to finish. Only 1 DB query per key, no matter how many concurrent requests.",
            code: `Problem: Cache restart\n  → 10K requests for "viral_key"\n  → All miss cache simultaneously\n  → 10K identical DB queries!\n\nSolution: Singleflight / cache lock\n→ First request acquires lock for key\n→ Queries DB, populates cache\n→ 9,999 other requests WAIT for lock\n→ Lock released → all serve from cache\n→ 1 DB query instead of 10,000` },
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
            <p className="text-[12px] text-stone-500 mt-0.5">In an HLD you say "URL shortener service" and draw one box. In production, that box is a redirect service, a URL creation service, a key generation service, a cache layer, an analytics pipeline, and a cleanup job — each separately deployed and owned.</p>
          </div>
        </div>
      </Card>

      <Card accent="#0f766e">
        <Label color="#0f766e">Service Decomposition — What We Actually Build</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { name: "Redirect Service", owns: "Core read path: receives GET /:key, looks up cache/DB, returns 302 redirect. Must be extremely fast and highly available.", tech: "Go/Rust for minimal latency, stateless behind ALB", api: "GET /:short_key → 302 redirect", scale: "Auto-scale on QPS (target: 4K avg, 12K peak)", stateful: false,
              modules: ["Key Parser (extract short_key from URL path)", "Cache Client (Redis lookup with circuit breaker)", "DB Client (fallback on cache miss, connection pool)", "Redirect Builder (301/302 response with Location header)", "Click Logger (async Kafka producer, fire-and-forget)", "Health Check (/health endpoint for ALB)"] },
            { name: "URL Creation Service", owns: "Write path: validates URL, fetches key from KGS, stores mapping, populates cache", tech: "Python/Node.js behind API Gateway, authenticated", api: "POST /api/v1/urls → 201 Created", scale: "Much lower traffic (40 QPS avg), 2-3 pods sufficient", stateful: false,
              modules: ["URL Validator (format check, reachability HEAD request)", "Custom Alias Checker (uniqueness check in DB)", "KGS Client (fetch pre-generated keys in batches)", "DB Writer (insert URL mapping, write-through cache)", "Rate Limiter (per-user creation limits)", "Response Builder (return short URL to client)"] },
            { name: "Key Generation Service (KGS)", owns: "Pre-generates unique Base62 keys offline, stores in DB. App servers fetch batches on demand.", tech: "Background worker + PostgreSQL table of unused keys", api: "Internal gRPC: GetKeyBatch(size) → [keys]", scale: "2 instances for redundancy, low QPS", stateful: true,
              modules: ["Key Generator (random Base62 strings, uniqueness check)", "Batch Allocator (atomic SELECT FOR UPDATE SKIP LOCKED)", "Key Pool Monitor (alert when available keys < threshold)", "Refill Worker (background job to generate more keys)", "Deallocation Handler (return unused keys on server shutdown)", "Anti-collision Guard (verify uniqueness before insertion)"] },
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
                    <span className="w-1 h-1 rounded-full bg-teal-400 shrink-0"/>{m}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card accent="#9333ea">
        <Label color="#9333ea">System Architecture — Write Path + Read Path</Label>
        <svg viewBox="0 0 720 380" className="w-full">
          {/* Title areas */}
          <rect x={5} y={5} width={710} height={175} rx={8} fill="#05966904" stroke="#05966920" strokeWidth={1} strokeDasharray="4,3"/>
          <text x={15} y={22} fill="#059669" fontSize="10" fontWeight="700" fontFamily="monospace">WRITE PATH (Create Short URL)</text>

          <rect x={5} y={190} width={710} height={180} rx={8} fill="#2563eb04" stroke="#2563eb20" strokeWidth={1} strokeDasharray="4,3"/>
          <text x={15} y={207} fill="#2563eb" fontSize="10" fontWeight="700" fontFamily="monospace">READ PATH (Redirect)</text>

          {/* Write path boxes */}
          <rect x={20} y={40} width={80} height={40} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={60} y={55} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Client</text>
          <text x={60} y={68} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">POST /api/v1/urls</text>

          <rect x={130} y={40} width={85} height={40} rx={6} fill="#6366f110" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={172} y={55} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">API Gateway</text>
          <text x={172} y={68} textAnchor="middle" fill="#6366f180" fontSize="7" fontFamily="monospace">auth + rate limit</text>

          <rect x={245} y={40} width={95} height={40} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={292} y={55} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Create Service</text>
          <text x={292} y={68} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">validate + store</text>

          <rect x={370} y={30} width={80} height={35} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={410} y={44} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">KGS</text>
          <text x={410} y={57} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">pre-gen keys</text>

          <rect x={370} y={80} width={80} height={35} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={410} y={94} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">DB Primary</text>
          <text x={410} y={107} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">INSERT url</text>

          <rect x={480} y={80} width={80} height={35} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={520} y={94} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Redis Cache</text>
          <text x={520} y={107} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">write-through</text>

          <rect x={590} y={40} width={90} height={40} rx={6} fill="#0284c710" stroke="#0284c7" strokeWidth={1.5}/>
          <text x={635} y={55} textAnchor="middle" fill="#0284c7" fontSize="9" fontWeight="600" fontFamily="monospace">Abuse Check</text>
          <text x={635} y={68} textAnchor="middle" fill="#0284c780" fontSize="7" fontFamily="monospace">Safe Browsing API</text>

          {/* Write arrows */}
          <defs><marker id="ah-arch" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          <line x1={100} y1={60} x2={130} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>
          <line x1={215} y1={60} x2={245} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>
          <line x1={340} y1={50} x2={370} y2={47} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>
          <line x1={340} y1={70} x2={370} y2={90} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>
          <line x1={450} y1={97} x2={480} y2={97} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>
          <line x1={292} y1={40} x2={590} y2={55} stroke="#94a3b850" strokeWidth={1} markerEnd="url(#ah-arch)" strokeDasharray="4,3"/>

          {/* Write labels */}
          <text x={350} y={40} fill="#c026d390" fontSize="7" fontFamily="monospace">fetch key</text>
          <text x={350} y={85} fill="#dc262690" fontSize="7" fontFamily="monospace">store</text>
          <text x={458} y={90} fill="#d9770690" fontSize="7" fontFamily="monospace">cache</text>

          {/* Write path summary */}
          <rect x={20} y={135} width={690} height={30} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={30} y={152} fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Write: Client → Gateway (auth) → Create Svc → KGS (fetch key) → DB Primary (INSERT) → Redis (write-through) → 201 Created</text>

          {/* Read path boxes */}
          <rect x={20} y={220} width={80} height={40} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={60} y={235} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Client</text>
          <text x={60} y={248} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">GET /abc123</text>

          <rect x={130} y={220} width={75} height={40} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={167} y={235} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">CDN</text>
          <text x={167} y={248} textAnchor="middle" fill="#78716c80" fontSize="7" fontFamily="monospace">edge cache</text>

          <rect x={235} y={220} width={50} height={40} rx={6} fill="#64748b10" stroke="#64748b" strokeWidth={1.5}/>
          <text x={260} y={238} textAnchor="middle" fill="#64748b" fontSize="9" fontWeight="600" fontFamily="monospace">ALB</text>

          <rect x={315} y={220} width={100} height={40} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={365} y={235} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Redirect Svc</text>
          <text x={365} y={248} textAnchor="middle" fill="#9333ea80" fontSize="7" fontFamily="monospace">lookup + 302</text>

          <rect x={445} y={210} width={80} height={35} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={485} y={224} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Redis Cache</text>
          <text x={485} y={237} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">GET url:key</text>

          <rect x={445} y={255} width={80} height={35} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={485} y={269} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">DB Replica</text>
          <text x={485} y={282} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">cache miss</text>

          <rect x={560} y={220} width={80} height={40} rx={6} fill="#0284c710" stroke="#0284c7" strokeWidth={1.5}/>
          <text x={600} y={235} textAnchor="middle" fill="#0284c7" fontSize="9" fontWeight="600" fontFamily="monospace">Kafka</text>
          <text x={600} y={248} textAnchor="middle" fill="#0284c780" fontSize="7" fontFamily="monospace">click events</text>

          <rect x={660} y={220} width={50} height={40} rx={6} fill="#be123c10" stroke="#be123c" strokeWidth={1.5}/>
          <text x={685} y={235} textAnchor="middle" fill="#be123c" fontSize="9" fontWeight="600" fontFamily="monospace">Click</text>
          <text x={685} y={248} textAnchor="middle" fill="#be123c80" fontSize="7" fontFamily="monospace">Analytics</text>

          {/* Read arrows */}
          <line x1={100} y1={240} x2={130} y2={240} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>
          <line x1={205} y1={240} x2={235} y2={240} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>
          <line x1={285} y1={240} x2={315} y2={240} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>
          <line x1={415} y1={230} x2={445} y2={225} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>
          <line x1={415} y1={250} x2={445} y2={265} stroke="#94a3b850" strokeWidth={1} markerEnd="url(#ah-arch)" strokeDasharray="4,3"/>
          <line x1={415} y1={240} x2={560} y2={240} stroke="#0284c750" strokeWidth={1} markerEnd="url(#ah-arch)" strokeDasharray="4,3"/>
          <line x1={640} y1={240} x2={660} y2={240} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>

          {/* Read labels */}
          <text x={420} y={218} fill="#d9770690" fontSize="7" fontFamily="monospace">check cache</text>
          <text x={420} y={260} fill="#05966990" fontSize="7" fontFamily="monospace">fallback</text>
          <text x={478} y={250} fill="#0284c790" fontSize="7" fontFamily="monospace">async click log</text>

          {/* Read path summary */}
          <rect x={20} y={310} width={690} height={50} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={30} y={327} fill="#2563eb" fontSize="8" fontWeight="600" fontFamily="monospace">Read: Client → CDN (edge cache) → ALB → Redirect Svc → Redis (cache HIT → 302) or DB Replica (MISS → cache fill → 302)</text>
          <text x={30} y={342} fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Key insight: reads never touch DB primary. Cache absorbs 90%+ of reads. Writes and reads are fully separated.</text>
          <text x={30} y={357} fill="#78716c" fontSize="8" fontWeight="600" fontFamily="monospace">Analytics: click events fire-and-forget to Kafka (async). Zero impact on redirect latency. Processed by Flink → ClickHouse.</text>
        </svg>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#2563eb">
          <Label color="#2563eb">Supporting Services</Label>
          <div className="space-y-3">
            {[
              { name: "Analytics Pipeline", role: "Consumes click events from Kafka. Aggregates by short_key, country, device, time. Powers the analytics API.", tech: "Kafka → Flink/Spark → ClickHouse", critical: false },
              { name: "Expiration Worker", role: "Background job that soft-deletes expired URLs. Runs every hour. Scans expires_at index.", tech: "Cron job / K8s CronJob → PostgreSQL", critical: false },
              { name: "Abuse Detection", role: "Scans new URLs for spam, phishing, malware links. Blocks or flags suspicious URLs.", tech: "ML model + Google Safe Browsing API", critical: true },
              { name: "CDN / Edge Cache", role: "Caches popular 302 redirects at edge locations worldwide. Reduces origin load for viral URLs.", tech: "CloudFront / Cloudflare with 5-min cache TTL", critical: false },
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
                  { route: "Redirect → Cache", proto: "RESP (Redis)", timeout: "10ms", fail: "Fall through to DB" },
                  { route: "Redirect → DB", proto: "PostgreSQL", timeout: "100ms", fail: "Return 503" },
                  { route: "Redirect → Kafka", proto: "Kafka (async)", timeout: "Fire & forget", fail: "Drop click event" },
                  { route: "Create → KGS", proto: "gRPC", timeout: "200ms", fail: "Return 503 to client" },
                  { route: "Create → DB", proto: "PostgreSQL", timeout: "500ms", fail: "Return 503 to client" },
                  { route: "Create → Cache", proto: "RESP (Redis)", timeout: "50ms", fail: "Skip (cache-aside)" },
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
  const [flow, setFlow] = useState("redirect_hit");
  const flows = {
    redirect_hit: {
      title: "Redirect — Cache Hit (Happy Path)",
      steps: [
        { actor: "Client", action: "GET https://short.ly/abc123 (browser follows short URL)", type: "request" },
        { actor: "CDN Edge", action: "Cache MISS (first request or TTL expired) → forward to origin", type: "process" },
        { actor: "Load Balancer", action: "Route to healthy Redirect Service pod (round-robin)", type: "process" },
        { actor: "Redirect Service", action: "Extract short_key='abc123' from URL path", type: "process" },
        { actor: "Redirect → Redis", action: "GET url:abc123 → cache HIT → returns long_url", type: "success" },
        { actor: "Redirect Service", action: "Build HTTP 302 response: Location: https://example.com/long/path", type: "success" },
        { actor: "Redirect Service", action: "Async: produce click event to Kafka {key:abc123, ip, user_agent, ts}", type: "process" },
        { actor: "Client", action: "Browser follows 302 redirect to original URL. Total latency: <10ms.", type: "success" },
      ]
    },
    redirect_miss: {
      title: "Redirect — Cache Miss → DB Lookup",
      steps: [
        { actor: "Client", action: "GET https://short.ly/xyz789", type: "request" },
        { actor: "Redirect Service", action: "Extract short_key='xyz789', check Redis → MISS", type: "process" },
        { actor: "Redirect → Redis", action: "GET url:xyz789 → null (not in cache)", type: "check" },
        { actor: "Redirect → DB", action: "SELECT long_url FROM urls WHERE short_key='xyz789' AND is_active=TRUE", type: "request" },
        { actor: "Database", action: "B-tree index lookup on primary key → found: long_url='https://example.com/...'", type: "success" },
        { actor: "Redirect Service", action: "Populate cache: SET url:xyz789 long_url EX 86400 (24h TTL)", type: "process" },
        { actor: "Redirect Service", action: "Return HTTP 302 with Location header", type: "success" },
        { actor: "Note", action: "Next request for same key will be a cache HIT. Latency: ~15ms (vs <5ms cache hit).", type: "check" },
      ]
    },
    create: {
      title: "Create Short URL — Full Flow",
      steps: [
        { actor: "Client", action: "POST /api/v1/urls {long_url: 'https://example.com/...'} with API key", type: "request" },
        { actor: "API Gateway", action: "Validate API key, check rate limit (100/day for free tier)", type: "auth" },
        { actor: "URL Creation Svc", action: "Validate URL format (regex) and reachability (HEAD request, 2s timeout)", type: "process" },
        { actor: "URL Creation Svc", action: "No custom alias → fetch key from local KGS batch (in-memory pool)", type: "process" },
        { actor: "KGS (if batch empty)", action: "gRPC GetKeyBatch(1000) → atomic DB: mark 1000 keys as used, return them", type: "request" },
        { actor: "URL Creation Svc → DB", action: "INSERT INTO urls (short_key, long_url, user_id, ...) VALUES ('abc123', ...)", type: "success" },
        { actor: "URL Creation Svc → Redis", action: "SET url:abc123 long_url EX 86400 (write-through cache)", type: "process" },
        { actor: "URL Creation Svc", action: "Return 201 Created: {short_url: 'https://short.ly/abc123', ...}", type: "success" },
      ]
    },
    not_found: {
      title: "Redirect — URL Not Found (404)",
      steps: [
        { actor: "Client", action: "GET https://short.ly/nonexist", type: "request" },
        { actor: "Redirect Service", action: "Extract short_key='nonexist', check Redis → MISS", type: "process" },
        { actor: "Redirect → DB", action: "SELECT long_url FROM urls WHERE short_key='nonexist' → 0 rows", type: "error" },
        { actor: "Redirect Service", action: "Cache negative result: SET url:nonexist __NULL__ EX 300 (5min negative cache)", type: "process" },
        { actor: "Redirect Service", action: "Return HTTP 404 Not Found", type: "error" },
        { actor: "Redirect Service", action: "Increment not_found counter metric. If rate > threshold, alert on possible enumeration.", type: "check" },
        { actor: "Note", action: "Negative caching prevents DB floods from scanners probing random keys.", type: "check" },
      ]
    },
    expired: {
      title: "Redirect — Expired URL (410 Gone)",
      steps: [
        { actor: "Client", action: "GET https://short.ly/old123 (URL with expires_at in the past)", type: "request" },
        { actor: "Redirect Service", action: "Check Redis → HIT, but includes expires_at metadata", type: "process" },
        { actor: "Redirect Service", action: "Check: expires_at < now → URL has expired", type: "check" },
        { actor: "Redirect Service", action: "Delete from cache: DEL url:old123", type: "process" },
        { actor: "Redirect Service", action: "Return HTTP 410 Gone: {error: 'url_expired'}", type: "error" },
        { actor: "Note", action: "Background expiration worker also cleans up DB. But redirect-time check is the real-time guard.", type: "check" },
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
    </div>
  );
}

function DeploymentSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#b45309">
          <Label color="#b45309">Kubernetes Deployment Topology</Label>
          <CodeBlock title="Redirect Service — The Critical Path" code={`# Redirect Service is the #1 priority for uptime
# and latency. Everything else can degrade.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redirect-service
spec:
  replicas: 6               # 2 per AZ (3 AZs)
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0     # Zero downtime deploys
  template:
    spec:
      containers:
      - name: redirect
        image: redirect-svc:latest
        env:
        - name: REDIS_CLUSTER
          value: "redis-cluster.cache.svc:6379"
        - name: DB_READ_REPLICA
          value: "pg-replica.db.svc:5432"
        resources:
          requests:
            cpu: "250m"
            memory: "256Mi"
          limits:
            cpu: "1"
            memory: "512Mi"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          periodSeconds: 5
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone`} />
          <div className="mt-3 space-y-1.5">
            <Point icon="⚠" color="#b45309">Separate Redirect and Create into different Deployments — different scaling needs</Point>
            <Point icon="⚠" color="#b45309">Redirect pods read from DB replicas, never primary — protect write path</Point>
            <Point icon="⚠" color="#b45309">HPA on Redirect: target 70% CPU, min 6, max 30 pods</Point>
          </div>
        </Card>
        <Card accent="#0f766e">
          <Label color="#0f766e">Multi-AZ Deployment Layout</Label>
          <svg viewBox="0 0 380 320" className="w-full">
            <rect x={5} y={5} width={370} height={310} rx={10} fill="#0f766e05" stroke="#0f766e30" strokeWidth={1} strokeDasharray="4,3"/>
            <text x={190} y={22} textAnchor="middle" fill="#0f766e" fontSize="10" fontWeight="700" fontFamily="monospace">us-east-1</text>

            {/* NLB */}
            <rect x={130} y={35} width={120} height={24} rx={4} fill="#2563eb12" stroke="#2563eb" strokeWidth={1}/>
            <text x={190} y={51} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Application Load Balancer</text>

            {/* AZ-a */}
            <rect x={15} y={72} width={110} height={145} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={70} y={87} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-a</text>
            <rect x={22} y={96} width={96} height={24} rx={4} fill="#9333ea15" stroke="#9333ea" strokeWidth={1}/>
            <text x={70} y={112} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">Redirect ×2</text>
            <rect x={22} y={128} width={96} height={24} rx={4} fill="#05966915" stroke="#059669" strokeWidth={1}/>
            <text x={70} y={144} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">Create ×1</text>
            <rect x={22} y={160} width={96} height={24} rx={4} fill="#dc262615" stroke="#dc2626" strokeWidth={1}/>
            <text x={70} y={176} textAnchor="middle" fill="#dc2626" fontSize="8" fontFamily="monospace">Redis (Primary)</text>
            <rect x={22} y={190} width={96} height={20} rx={4} fill="#c026d315" stroke="#c026d3" strokeWidth={1}/>
            <text x={70} y={203} textAnchor="middle" fill="#c026d3" fontSize="7" fontFamily="monospace">DB Primary</text>

            {/* AZ-b */}
            <rect x={135} y={72} width={110} height={145} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={190} y={87} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-b</text>
            <rect x={142} y={96} width={96} height={24} rx={4} fill="#9333ea15" stroke="#9333ea" strokeWidth={1}/>
            <text x={190} y={112} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">Redirect ×2</text>
            <rect x={142} y={128} width={96} height={24} rx={4} fill="#05966915" stroke="#059669" strokeWidth={1}/>
            <text x={190} y={144} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">Create ×1</text>
            <rect x={142} y={160} width={96} height={24} rx={4} fill="#0891b215" stroke="#0891b2" strokeWidth={1}/>
            <text x={190} y={176} textAnchor="middle" fill="#0891b2" fontSize="8" fontFamily="monospace">Redis (Replica)</text>
            <rect x={142} y={190} width={96} height={20} rx={4} fill="#c026d315" stroke="#c026d3" strokeWidth={1}/>
            <text x={190} y={203} textAnchor="middle" fill="#c026d3" fontSize="7" fontFamily="monospace">DB Replica</text>

            {/* AZ-c */}
            <rect x={255} y={72} width={110} height={145} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={310} y={87} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-c</text>
            <rect x={262} y={96} width={96} height={24} rx={4} fill="#9333ea15" stroke="#9333ea" strokeWidth={1}/>
            <text x={310} y={112} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">Redirect ×2</text>
            <rect x={262} y={128} width={96} height={24} rx={4} fill="#05966915" stroke="#059669" strokeWidth={1}/>
            <text x={310} y={144} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">KGS ×1</text>
            <rect x={262} y={160} width={96} height={24} rx={4} fill="#0891b215" stroke="#0891b2" strokeWidth={1}/>
            <text x={310} y={176} textAnchor="middle" fill="#0891b2" fontSize="8" fontFamily="monospace">Redis (Replica)</text>
            <rect x={262} y={190} width={96} height={20} rx={4} fill="#c026d315" stroke="#c026d3" strokeWidth={1}/>
            <text x={310} y={203} textAnchor="middle" fill="#c026d3" fontSize="7" fontFamily="monospace">DB Replica</text>

            {/* Legend */}
            <rect x={15} y={225} width={350} height={82} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
            <text x={190} y={242} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Deployment Rules</text>
            <text x={30} y={260} fill="#78716c" fontSize="8" fontFamily="monospace">• Redirect: stateless, 2+ per AZ, HPA on CPU/QPS</text>
            <text x={30} y={275} fill="#78716c" fontSize="8" fontFamily="monospace">• Redis: 1 primary + 2 replicas. Sentinel for failover.</text>
            <text x={30} y={290} fill="#78716c" fontSize="8" fontFamily="monospace">• DB: 1 primary (writes) + 2 replicas (reads). RDS Multi-AZ.</text>
            <text x={30} y={305} fill="#78716c" fontSize="8" fontFamily="monospace">• AZ failure: ALB routes away. Redis/DB auto-failover.</text>
          </svg>
        </Card>
      </div>

      <Card accent="#dc2626">
        <Label color="#dc2626">Security & Abuse Prevention</Label>
        <div className="grid grid-cols-3 gap-4">
          {[
            { layer: "URL Validation & Safety", what: "Before creating a short URL, validate the destination isn't malicious. Block phishing, malware, and spam URLs.",
              details: ["Google Safe Browsing API check on every new URL", "Block known malicious domains (denylist)", "Rate limit unauthenticated URL creation (10/hour)", "CAPTCHA for anonymous users to prevent bot spam"] },
            { layer: "Short URL Abuse", what: "Prevent attackers from using your service to distribute malicious content or enumerate private URLs.",
              details: ["Non-sequential keys prevent enumeration attacks", "Report/flag mechanism for malicious short URLs", "Automatic scanning of destinations (periodic re-check)", "Honey-pot short keys to detect scrapers"] },
            { layer: "Infrastructure Security", what: "Protect internal services and data from unauthorized access.",
              details: ["TLS everywhere (ALB terminates, internal mTLS optional)", "API key auth for URL creation, no auth for redirects", "K8s NetworkPolicy: restrict DB access to app pods only", "WAF rules: block suspicious patterns (SQL injection in URLs)"] },
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
    </div>
  );
}

function OpsSection() {
  return (
    <div className="space-y-5">
      <Card accent="#be123c">
        <Label color="#be123c">Scaling Playbook — Component-by-Component</Label>
        <p className="text-[12px] text-stone-500 mb-3">When traffic grows, which component hits the wall first? And how do you fix each one without downtime?</p>
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
                { bottleneck: "Redirect latency", symptom: "p99 > 50ms", fix: "Check cache hit rate first. If low: increase Redis memory or warm cache. If high: profile app server code.", impact: "None — transparent to users", pitfall: "Don't scale app servers if the problem is cache misses." },
                { bottleneck: "DB read load", symptom: "Replica lag > 1s, connection pool full", fix: "Add read replicas. Or increase cache TTL to reduce DB fallthrough.", impact: "None — add replicas online", pitfall: "More replicas = more replication lag. Monitor lag closely." },
                { bottleneck: "Redis memory full", symptom: "Evictions spike, cache hit rate drops", fix: "Add Redis shards (Cluster mode). Or increase maxmemory. Or reduce TTL to evict stale entries.", impact: "Brief miss spike during resharding", pitfall: "LRU eviction removes potentially hot keys. Prefer explicit TTL." },
                { bottleneck: "KGS exhaustion", symptom: "Key pool < 1000, write latency spikes", fix: "Run emergency key generation job. Increase batch size. Add second KGS instance.", impact: "Writes may slow briefly", pitfall: "If KGS fully exhausted, NO new URLs can be created. Alert early." },
                { bottleneck: "DB write throughput", symptom: "INSERT latency > 100ms, lock contention", fix: "Batch writes. Or move to DynamoDB for auto-scaling writes. Or shard primary DB.", impact: "Brief during migration", pitfall: "URL creation is low-QPS (40/s). If writes are slow, something else is wrong." },
                { bottleneck: "Kafka consumer lag", symptom: "Analytics are hours behind", fix: "Add consumer instances. Increase partitions. Check for slow consumers.", impact: "Analytics delayed, not user-facing", pitfall: "Analytics is async — lag doesn't affect redirect latency." },
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
            { title: "Viral URL Melts Single Cache Shard", symptom: "One Redis shard at 100% CPU. Other shards idle. Redirect latency spikes for all URLs on that shard.",
              cause: "Celebrity tweeted a short URL. 500K clicks/sec to a single cache key. Hash-based sharding puts it on one shard.",
              fix: "CDN should absorb this. Set Cache-Control headers on 302 responses so CDN caches the redirect. For Redis: replicate hot key across multiple shards. Or use client-side random shard selection for known hot keys.",
              quote: "A K-pop star shared our link. Redis shard 3 caught fire. We added CloudFront in 20 minutes and it was fine." },
            { title: "Expired URL Still Redirecting", symptom: "Users report that an expired short URL still works. Expires_at was yesterday but redirects succeed.",
              cause: "Cache entry doesn't have expires_at metadata. Or cache TTL is longer than URL TTL. Cache serves stale redirect without checking expiration.",
              fix: "Store expires_at in cache alongside long_url. Check expiration at redirect time, not just in background worker. Or set cache TTL = min(cache_ttl, time_until_expiry).",
              quote: "Legal asked us to expire a URL immediately. It kept working for 24 hours because Redis TTL was 24h and we didn't store expires_at in cache." },
            { title: "KGS Ran Out of Keys on Black Friday", symptom: "URL creation returns 503. KGS pool empty. Key generation job takes 20 minutes to refill.",
              cause: "Marketing campaign created 10M URLs in 2 hours. Normal rate is 100M/month. KGS pool had 1M keys pre-generated — exhausted in 12 minutes.",
              fix: "Monitor KGS pool size aggressively. Alert at 20% remaining. Auto-trigger generation job at 50% remaining. Keep 10× peak daily usage in pool. Pre-generate before known campaigns.",
              quote: "Product launched a QR code campaign. 5M URLs in 1 hour. KGS pool was designed for 'normal' traffic. We added an emergency 'generate 10M keys' button after this." },
            { title: "301 Redirect Cached Permanently by Browsers", symptom: "Admin updates the destination URL but users still go to the old destination. No amount of cache clearing helps.",
              cause: "We used 301 (permanent redirect). Browsers cache 301s indefinitely with no way to clear them. Even clearing browser cache doesn't help on all browsers.",
              fix: "Use 302 (temporary redirect) unless you're absolutely sure destinations will never change. 302 gives you the ability to update destinations. Trade-off: slightly more origin hits, but full control.",
              quote: "Client said 'the link goes to the wrong page.' We changed the DB. Still wrong. 301 was cached in their browser forever. Had to create a new short URL." },
            { title: "Database Replication Lag Causes 404s", symptom: "User creates a URL, shares it immediately, recipient gets 404. URL exists in primary but hasn't replicated to replica yet.",
              cause: "Create writes to primary. Redirect reads from replica. Async replication lag = 50-200ms normally. Under load, can be seconds.",
              fix: "Write-through cache on create: SET cache immediately after DB insert. Redirects check cache first (no replication lag). Cache is written synchronously on create, so the URL is available instantly.",
              quote: "Users in Slack: 'I just shared a link and my teammate says it's broken.' Replication lag was 800ms. We added write-through caching and never saw this again." },
            { title: "Spam/Phishing URLs Created Faster Than Detection", symptom: "Abuse reports spike. Thousands of phishing URLs created via API. Block one, ten more appear.",
              cause: "Bots creating URLs faster than abuse detection pipeline can process. Free API with no phone verification.",
              fix: "Pre-creation check: Google Safe Browsing API (adds ~100ms to create, acceptable). Rate limit by IP + fingerprint. Phone verification for new accounts. Require API key for all creation. Retroactive scan with auto-disable.",
              quote: "Someone created 50K phishing URLs on a Saturday night. By Monday we had 200 abuse reports. We added Safe Browsing API check and it caught 99% before creation." },
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
        { t: "Real-Time Analytics Dashboard", d: "Live click counts, geographic heatmap, referrer breakdown, device stats.", detail: "Kafka → Flink real-time aggregation → WebSocket push to dashboard. ClickHouse for historical queries.", effort: "Medium" },
        { t: "Custom Branded Domains", d: "Users bring their own domain: brand.co/sale instead of short.ly/abc123.", detail: "DNS CNAME setup, TLS certificate provisioning (Let's Encrypt), domain → account mapping in DB.", effort: "Hard" },
        { t: "QR Code Generation", d: "Auto-generate QR codes for each short URL. Common for marketing campaigns.", detail: "QR library generates PNG/SVG on-the-fly. Cache generated QR codes in CDN. Embed short URL in QR.", effort: "Easy" },
        { t: "Link Preview / Open Graph", d: "When sharing a short URL on social media, show a preview of the destination page.", detail: "Fetch OG tags from destination URL at creation time. Cache title, description, image. Serve as meta tags on a preview page.", effort: "Medium" },
        { t: "A/B Testing (Split URLs)", d: "Single short URL redirects to different destinations based on percentage split.", detail: "URL maps to N destinations with weights. Consistent hashing on visitor fingerprint for sticky assignment.", effort: "Medium" },
        { t: "Geolocation-Based Redirect", d: "Same short URL redirects to different destinations based on user's country.", detail: "GeoIP lookup at redirect time. Country → URL mapping per short key. Useful for localized landing pages.", effort: "Easy" },
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
    { q:"How do you prevent hash collisions?", a:"Don't use hashing. Use a counter-based approach (auto-increment ID → Base62) or a Key Generation Service (pre-generate random unique keys). Both guarantee zero collisions by construction. Hashing + truncation has inherent collision risk and requires retry logic.", tags:["algorithm"] },
    { q:"301 or 302 redirect? When would you choose each?", a:"302 (temporary) is better for most use cases — server sees every click, enabling analytics and allowing destination updates. 301 (permanent) is better for SEO juice transfer and reduced server load, but browsers cache it indefinitely and you lose control. Default to 302 unless you have a strong reason for 301.", tags:["design"] },
    { q:"How would you handle the same long URL being submitted multiple times?", a:"Two approaches: (1) Deduplication — hash the long_url, check if it exists, return existing short URL. Saves storage but requires an index on long_url hash. (2) Always create new — each submission gets a unique key. Simpler, allows per-user tracking. Discuss trade-offs with interviewer.", tags:["design"] },
    { q:"How would you scale to 1 billion URLs?", a:"Cache handles read scaling (Redis Cluster). DB sharding by short_key hash handles storage. KGS pre-generates keys to avoid counter bottleneck. CDN absorbs viral URL traffic. At 1B URLs × 500B = 500 GB — still fits in a modestly sized DB cluster.", tags:["scalability"] },
    { q:"What happens if the KGS goes down?", a:"App servers keep a local buffer of pre-fetched keys (1000-10000 keys each). KGS outage = no new keys from KGS, but servers can create URLs from their local buffer for hours. Reads (redirects) are completely unaffected. Run 2+ KGS instances for redundancy.", tags:["availability"] },
    { q:"How do you prevent URL enumeration attacks?", a:"(1) Use random keys from KGS instead of sequential counter. (2) If using counter, XOR with a secret before Base62 encoding. (3) Rate limit redirect 404s per IP. (4) Monitor 404 patterns — sequential probing is a red flag. (5) CAPTCHA for unauthenticated URL creation.", tags:["security"] },
    { q:"How would you implement URL expiration?", a:"Store expires_at in both DB and cache. At redirect time, check if expired (fast — just a timestamp comparison). Background worker runs hourly: UPDATE urls SET is_active=FALSE WHERE expires_at < NOW() AND is_active=TRUE. Expired URLs return 410 Gone.", tags:["design"] },
    { q:"How would you implement analytics without slowing down redirects?", a:"Async pipeline: redirect service fires a Kafka event (fire-and-forget, <1ms overhead). Kafka → Flink/Spark aggregates clicks. Results stored in ClickHouse (OLAP-optimized). Analytics API reads from ClickHouse. Zero impact on redirect latency.", tags:["scalability"] },
    { q:"SQL or NoSQL for the URL store?", a:"Both work. PostgreSQL: strong consistency, rich queries, simple to start. DynamoDB/Cassandra: auto-sharding, single-digit ms at any scale. URL access is a simple KV lookup (short_key → long_url), which is ideal for NoSQL. Start with PostgreSQL, migrate to DynamoDB if you outgrow it.", tags:["data"] },
    { q:"How would you handle malicious URLs?", a:"Multi-layer: (1) Google Safe Browsing API check at creation time. (2) Domain denylist. (3) ML-based phishing detection. (4) User reporting mechanism. (5) Periodic re-scanning of existing URLs. (6) Instant disable capability for flagged URLs. (7) Rate limit creation per IP.", tags:["security"] },
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

export default function UrlShortenerSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">URL Shortener</h1>
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