import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   NEWS FEED — System Design Reference
   Pearl white theme · 17 sections (HLD + LLD)
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",             icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",         icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",  icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",           icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",    icon: "🏗️", color: "#9333ea" },
  { id: "algorithm",     label: "Fan-Out Deep Dive",    icon: "⚙️", color: "#c026d3" },
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
            <Label>What is a News Feed?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A news feed is a continuously updating stream of content (posts, photos, videos, links) aggregated from people and pages a user follows. It's the core product of every social platform — the first thing users see when they open the app.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              The fundamental challenge: User A posts something. It needs to appear in the feeds of all their followers — which could be 10 people or 50 million people. How you deliver that content is THE core design decision (fan-out-on-write vs fan-out-on-read).
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard?</Label>
            <ul className="space-y-2.5">
              <Point icon="📊" color="#0891b2">Scale asymmetry — one post from a celebrity creates 50M feed updates. One post from a normal user creates 200.</Point>
              <Point icon="⏱️" color="#0891b2">Latency expectations — users expect feed to load in &lt;500ms with fresh content, not minutes-old stale data</Point>
              <Point icon="🔀" color="#0891b2">Ranking complexity — chronological is simple but engagement-based ranking requires ML models scoring thousands of candidates</Point>
              <Point icon="💾" color="#0891b2">Storage explosion — pre-computing feeds for 1B users × 500 posts each = petabytes of duplicated data</Point>
              <Point icon="🔄" color="#0891b2">Consistency vs freshness — user posts, then checks their own feed and doesn't see it (read-after-write consistency)</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Scale</Label>
            <div className="space-y-2.5">
              {[
                { co: "Facebook", scale: "2B+ DAU, 350M photos/day", approach: "Hybrid fan-out + ranking" },
                { co: "Twitter/X", scale: "500M+ DAU, 500M tweets/day", approach: "Fan-out-on-write (timeline)" },
                { co: "Instagram", scale: "2B+ MAU, ranked feed", approach: "Fan-out-on-write + ML ranking" },
                { co: "LinkedIn", scale: "310M MAU, professional feed", approach: "Fan-out-on-write + relevance" },
                { co: "TikTok", scale: "1B+ MAU, For You page", approach: "Fan-out-on-read (recommendation)" },
              ].map((e,i) => (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-24 shrink-0">{e.co}</span>
                  <span className="text-stone-500 flex-1">{e.scale}</span>
                  <span className="text-stone-400 text-[10px]">{e.approach}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <Label color="#2563eb">The Core Tradeoff (Preview)</Label>
            <svg viewBox="0 0 360 140" className="w-full">
              <rect x={10} y={10} width={160} height={55} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
              <text x={90} y={30} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="700" fontFamily="monospace">Fan-Out on Write</text>
              <text x={90} y={46} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">Pre-compute feeds on post</text>
              <text x={90} y={58} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">Fast reads, slow writes</text>

              <rect x={190} y={10} width={160} height={55} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
              <text x={270} y={30} textAnchor="middle" fill="#2563eb" fontSize="10" fontWeight="700" fontFamily="monospace">Fan-Out on Read</text>
              <text x={270} y={46} textAnchor="middle" fill="#2563eb80" fontSize="8" fontFamily="monospace">Build feed at read time</text>
              <text x={270} y={58} textAnchor="middle" fill="#2563eb80" fontSize="8" fontFamily="monospace">Fast writes, slow reads</text>

              <rect x={60} y={80} width={240} height={50} rx={6} fill="#9333ea08" stroke="#9333ea" strokeWidth={1.5}/>
              <text x={180} y={98} textAnchor="middle" fill="#9333ea" fontSize="10" fontWeight="700" fontFamily="monospace">Hybrid Approach ★</text>
              <text x={180} y={114} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">Fan-out-on-write for normal users</text>
              <text x={180} y={124} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">Fan-out-on-read for celebrities (&gt;10K followers)</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">The #1 most asked system design question</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Scope Immediately</div>
            <p className="text-[12px] text-stone-500 mt-0.5">"News feed" can mean many things. Clarify: Are we building the feed generation pipeline, the ranking system, or the full social graph? For a 45-min interview, focus on <strong>feed generation + delivery</strong>. Ranking can be a follow-up. Don't try to design the entire social network.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Users can create posts (text, images, video links)</Point>
            <Point icon="2." color="#059669">Users can follow/unfollow other users</Point>
            <Point icon="3." color="#059669">GET /feed returns a ranked/chronological list of posts from followed users</Point>
            <Point icon="4." color="#059669">Feed supports pagination (infinite scroll, cursor-based)</Point>
            <Point icon="5." color="#059669">New posts appear in followers' feeds within seconds (near-real-time)</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Feed load latency: p99 &lt;500ms (p50 &lt;200ms)</Point>
            <Point icon="2." color="#dc2626">Post propagation: &lt;5s to appear in followers' feeds</Point>
            <Point icon="3." color="#dc2626">Highly available — feed should always load (degrade gracefully)</Point>
            <Point icon="4." color="#dc2626">Scale to 500M+ DAU, 10B+ feed reads/day</Point>
            <Point icon="5." color="#dc2626">Eventual consistency OK — brief staleness acceptable (except own posts)</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Chronological or ranked feed? (changes everything)",
            "What content types? Text only, or images/videos too?",
            "Average followers per user? Celebrity distribution?",
            "Do users see posts from friends-of-friends or only direct follows?",
            "Do we need real-time push or pull-to-refresh is OK?",
            "What's the feed length? Top 50? Infinite scroll?",
            "Do we need to support ads/sponsored posts in the feed?",
            "Single region or multi-region deployment?",
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
            <p className="text-[12px] text-stone-500 mt-0.5">Walk through each step aloud. The interviewer cares about your reasoning process, not the exact number. Round aggressively: 86,400 ≈ 100K, 500M × 200 ≈ 100B.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Traffic Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="DAU = 500M users" result="500M" note="Large social platform (Facebook-scale). Ask interviewer." />
            <MathStep step="2" formula="Feed reads per user/day = 10 (opens app × scrolls)" result="10" note="Each app open loads feed, user scrolls multiple times" />
            <MathStep step="3" formula="Total feed reads/day = 500M × 10" result="5 Billion" note="This is the READ load — the dominant dimension" />
            <MathStep step="4" formula="Feed read QPS = 5B / 86,400 ≈ 5B / 100K" result="~58K QPS" note="Average. Reads are the bottleneck, not writes." final />
            <MathStep step="5" formula="Peak feed read QPS = 58K × 3" result="~175K QPS" note="3× peak multiplier (morning, lunch, evening spikes)" final />
            <MathStep step="6" formula="Posts created per day = 500M × 0.1 (10% post)" result="50M posts/day" note="10% of users post daily. Most users only read." />
            <MathStep step="7" formula="Post write QPS = 50M / 100K" result="~580 QPS" note="Writes are tiny compared to reads. 300:1 read:write ratio." />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Fan-Out Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Average followers per user" result="~200" note="Most users have 100-500 followers. Power law distribution." />
            <MathStep step="2" formula="Fan-out writes per post = 200 (avg followers)" result="200" note="Each post → 200 feed entries to write (fan-out-on-write)" />
            <MathStep step="3" formula="Total fan-out writes/day = 50M posts × 200" result="10 Billion" note="10B write operations to pre-compute feeds" final />
            <MathStep step="4" formula="Fan-out write QPS = 10B / 100K" result="~115K QPS" note="This is the fan-out worker write load to Redis/cache" />
            <MathStep step="5" formula="Celebrity problem: user with 50M followers" result="50M writes" note="One celeb post = 50M fan-out writes. Takes minutes, not ms." />
            <MathStep step="6" formula="Solution: celebrities use fan-out-on-read" result="Hybrid ★" note="Threshold: >10K followers → skip pre-computation" final />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Storage Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Feed entry = post_id (8B) + user_id (8B) + timestamp (8B) + score (4B)" result="~32 bytes" note="Minimal feed entry — just IDs, fetch full post separately" />
            <MathStep step="2" formula="Feed length per user = 500 posts (cached)" result="500" note="Keep last 500 posts in pre-computed feed per user" />
            <MathStep step="3" formula="Per-user feed size = 500 × 32B" result="~16 KB" note="Very compact — IDs only, not full post content" />
            <MathStep step="4" formula="Total feed cache = 500M users × 16 KB" result="~8 TB" note="All pre-computed feeds in memory/Redis" final />
            <MathStep step="5" formula="Post storage = 50M posts/day × 1 KB avg" result="~50 GB/day" note="Post content stored separately in DB. Media in blob store." />
            <MathStep step="6" formula="Posts retained = 5 years × 365 × 50 GB" result="~90 TB" note="Post database grows over time. Old posts archived." />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Infrastructure Sizing</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Feed cache (Redis): 8 TB / 64 GB per node" result="125 shards" note="r6g.2xlarge nodes. With replicas: ~250 nodes." final />
            <MathStep step="2" formula="Fan-out workers: 115K writes/sec" result="20-30 workers" note="Each worker processes ~5K fan-out writes/sec" />
            <MathStep step="3" formula="Post DB: 90 TB, read-heavy" result="Sharded MySQL/Postgres" note="Shard by user_id. Read replicas for feed assembly." />
            <MathStep step="4" formula="Social graph DB: 500M users × 200 avg follows" result="100B edges" note="Graph DB or adjacency list in sharded DB" />
            <MathStep step="5" formula="Monthly cost (feed cache alone)" result="~$45K/mo" note="250 Redis nodes × $0.25/hr × 730 hrs. Reserved: ~$27K/mo." />
          </div>
          <div className="mt-4 pt-3 border-t border-stone-100 text-[11px] text-stone-500">
            <strong className="text-stone-700">Key insight:</strong> The feed cache is the most expensive component. Reducing feed length from 500 → 200 posts saves 60% memory. Trade off cache size vs cache miss rate.
          </div>
        </Card>
      </div>
      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Peak Read QPS", val: "~175K", sub: "Feed loads (dominant)" },
            { label: "Fan-Out Writes", val: "~115K QPS", sub: "10B/day to feed cache" },
            { label: "Feed Cache Size", val: "~8 TB", sub: "125 shards + replicas" },
            { label: "Read:Write Ratio", val: "300:1", sub: "Extremely read-heavy" },
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
          <Label color="#2563eb">Feed APIs</Label>
          <CodeBlock code={`# GET /v1/feed?cursor=<timestamp>&limit=20
# Returns: paginated feed for authenticated user
#
# Response:
{
  "posts": [
    {
      "post_id": "abc123",
      "author_id": "user_789",
      "author_name": "Alice",
      "content": "Hello world!",
      "media_url": "https://cdn.../photo.jpg",
      "created_at": "2024-02-09T10:30:00Z",
      "likes_count": 142,
      "comments_count": 23,
      "has_liked": false
    }
  ],
  "next_cursor": "1707470000.000",  # timestamp-based cursor
  "has_more": true
}

# POST /v1/posts
# Create a new post (triggers fan-out)
{
  "content": "Hello world!",
  "media_ids": ["img_abc"],        # pre-uploaded media
  "visibility": "followers"         # public | followers | private
}
# Returns: { "post_id": "abc123", "created_at": "..." }

# POST /v1/users/:id/follow
# POST /v1/users/:id/unfollow
# Modifies social graph + triggers feed backfill`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Design Decisions</Label>
          <div className="space-y-3">
            {[
              { q: "Cursor vs Offset pagination?", a: "Cursor (timestamp-based). Offset breaks when new posts are inserted — user sees duplicates or misses posts. Cursor is stable: 'give me posts older than this timestamp.'" },
              { q: "Return full posts or just IDs?", a: "Full posts with hydrated data (author name, counts, media URLs). Client shouldn't make N+1 calls. Server-side join is faster than client-side." },
              { q: "How does the client know there's new content?", a: "Two options: (1) Poll — client pings /feed/new-count every 30s. (2) Push — WebSocket/SSE sends 'new posts available' notification. Most apps use poll + pull-to-refresh." },
              { q: "What about read-after-write consistency?", a: "User posts, then refreshes feed. Must see their own post. Solution: always merge user's own recent posts into feed response, even if fan-out hasn't completed." },
              { q: "Rate limiting on feed reads?", a: "Yes — feed reads are expensive. Limit to ~100 reads/min per user. Aggressive scrapers can overwhelm the system. Return 429 with Retry-After." },
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
  const [phase, setPhase] = useState(0);
  const phases = [
    { label: "Pull Model", desc: "On feed request, query all followed users' posts, merge, sort, return. Simple but slow — if you follow 500 users, that's 500 DB queries per feed load. O(following × posts) per read." },
    { label: "Push Model (Fan-Out on Write) ★", desc: "When a user posts, immediately write to every follower's pre-computed feed (in cache/Redis). Feed reads become a single cache lookup — O(1). The dominant approach for most social networks." },
    { label: "Hybrid ★★", desc: "Fan-out-on-write for normal users (< 10K followers). Fan-out-on-read for celebrities. On feed read: merge pre-computed feed + live query from followed celebrities. Best of both worlds." },
  ];
  const diagrams = [
    () => (
      <svg viewBox="0 0 460 170" className="w-full">
        <DiagramBox x={60} y={50} w={70} h={36} label="User A" color="#2563eb"/>
        <DiagramBox x={200} y={50} w={90} h={40} label="Feed\nService" color="#9333ea"/>
        <DiagramBox x={340} y={25} w={80} h={30} label="User B posts" color="#059669"/>
        <DiagramBox x={340} y={60} w={80} h={30} label="User C posts" color="#059669"/>
        <DiagramBox x={340} y={95} w={80} h={30} label="User D posts" color="#059669"/>
        <Arrow x1={95} y1={50} x2={155} y2={50} label="GET /feed" id="p1"/>
        <Arrow x1={245} y1={35} x2={300} y2={28} label="query" id="p2"/>
        <Arrow x1={245} y1={50} x2={300} y2={60} id="p3"/>
        <Arrow x1={245} y1={65} x2={300} y2={92} id="p4"/>
        <rect x={90} y={130} width={260} height={20} rx={4} fill="#dc262608" stroke="#dc262630"/>
        <text x={220} y={142} textAnchor="middle" fill="#dc2626" fontSize="8" fontFamily="monospace">❌ 500 queries per feed load — too slow at scale</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 190" className="w-full">
        <DiagramBox x={50} y={40} w={70} h={36} label="User B" color="#059669" sub="posts"/>
        <DiagramBox x={170} y={40} w={80} h={40} label="Fan-Out\nWorker" color="#c026d3"/>
        <DiagramBox x={310} y={20} w={80} h={28} label="A's feed" color="#d97706"/>
        <DiagramBox x={310} y={55} w={80} h={28} label="C's feed" color="#d97706"/>
        <DiagramBox x={310} y={90} w={80} h={28} label="D's feed" color="#d97706"/>
        <DiagramBox x={420} y={55} w={50} h={28} label="Redis" color="#dc2626"/>
        <Arrow x1={85} y1={40} x2={130} y2={40} label="new post" id="fo1"/>
        <Arrow x1={210} y1={30} x2={270} y2={23} label="write" id="fo2"/>
        <Arrow x1={210} y1={40} x2={270} y2={55} id="fo3"/>
        <Arrow x1={210} y1={50} x2={270} y2={87} id="fo4"/>
        <rect x={80} y={135} width={260} height={20} rx={4} fill="#05966908" stroke="#05966930"/>
        <text x={210} y={147} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">✓ Write once → 200 feed entries. Reads are O(1) cache lookup</text>
        <rect x={80} y={162} width={260} height={20} rx={4} fill="#d9770608" stroke="#d9770630"/>
        <text x={210} y={174} textAnchor="middle" fill="#d97706" fontSize="8" fontFamily="monospace">⚠ Celebrity with 50M followers = 50M writes per post</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 200" className="w-full">
        <DiagramBox x={55} y={40} w={70} h={36} label="User A" color="#2563eb" sub="reads feed"/>
        <DiagramBox x={190} y={40} w={90} h={40} label="Feed\nService" color="#9333ea"/>
        <DiagramBox x={340} y={20} w={100} h={30} label="Pre-computed" color="#d97706" sub="(normal users)"/>
        <DiagramBox x={340} y={65} w={100} h={30} label="Live Query" color="#2563eb" sub="(celebrities)"/>
        <DiagramBox x={190} y={120} w={90} h={30} label="Merge + Rank" color="#c026d3"/>
        <Arrow x1={90} y1={40} x2={145} y2={40} label="GET /feed" id="h1"/>
        <Arrow x1={235} y1={30} x2={290} y2={25} label="cache read" id="h2"/>
        <Arrow x1={235} y1={50} x2={290} y2={72} label="query" id="h3"/>
        <Arrow x1={190} y1={60} x2={190} y2={105} label="both results" id="h4" dashed/>
        <rect x={80} y={165} width={280} height={22} rx={4} fill="#9333ea08" stroke="#9333ea30"/>
        <text x={220} y={178} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">✓ Best of both: fast reads + handles celebrities + ranked</text>
      </svg>
    ),
  ];
  return (
    <div className="space-y-5">
      <Card accent="#9333ea">
        <Label color="#9333ea">Architecture Evolution — The Fan-Out Decision</Label>
        <div className="flex gap-2 mb-4">
          {phases.map((p,i) => (
            <button key={i} onClick={() => setPhase(i)}
              className={`px-3.5 py-1.5 rounded-lg text-[12px] font-medium transition-all border ${i===phase ? "bg-purple-600 text-white border-purple-600" : "bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
              {i+1}. {p.label}
            </button>
          ))}
        </div>
        <p className="text-[13px] text-stone-500 mb-4">{phases[phase].desc}</p>
        <div className="bg-stone-50 rounded-lg border border-stone-200 p-3" style={{ minHeight: 160 }}>
          {diagrams[phase]()}
        </div>
      </Card>
      <Card>
        <Label color="#c026d3">Full System Architecture (Hybrid)</Label>
        <svg viewBox="0 0 680 240" className="w-full">
          <DiagramBox x={50} y={45} w={65} h={34} label="Client" color="#2563eb"/>
          <DiagramBox x={150} y={45} w={70} h={34} label="Gateway" color="#6366f1"/>
          {/* Write path */}
          <DiagramBox x={280} y={25} w={80} h={30} label="Post Service" color="#059669"/>
          <DiagramBox x={420} y={25} w={80} h={30} label="Fan-Out\nWorkers" color="#c026d3"/>
          <DiagramBox x={560} y={25} w={70} h={30} label="Feed\nCache" color="#d97706"/>
          {/* Read path */}
          <DiagramBox x={280} y={70} w={80} h={30} label="Feed Service" color="#9333ea"/>
          <DiagramBox x={420} y={70} w={80} h={30} label="Ranking\nService" color="#0891b2"/>
          {/* Data */}
          <DiagramBox x={280} y={130} w={70} h={30} label="Post DB" color="#059669"/>
          <DiagramBox x={420} y={130} w={75} h={30} label="Social\nGraph DB" color="#2563eb"/>
          <DiagramBox x={560} y={70} w={70} h={30} label="User\nCache" color="#d97706"/>
          <DiagramBox x={560} y={130} w={70} h={30} label="Media\nCDN" color="#78716c"/>

          <Arrow x1={82} y1={45} x2={115} y2={45} id="a1"/>
          <Arrow x1={185} y1={35} x2={240} y2={27} label="write" id="a2"/>
          <Arrow x1={185} y1={55} x2={240} y2={72} label="read" id="a3"/>
          <Arrow x1={320} y1={25} x2={380} y2={25} id="a4"/>
          <Arrow x1={460} y1={25} x2={525} y2={25} label="push" id="a5"/>
          <Arrow x1={320} y1={70} x2={380} y2={70} id="a6"/>
          <Arrow x1={280} y1={40} x2={280} y2={115} id="a7" dashed/>
          <Arrow x1={420} y1={40} x2={420} y2={115} id="a8" dashed/>
          <Arrow x1={460} y1={70} x2={525} y2={70} id="a9"/>

          <rect x={190} y={180} width={340} height={38} rx={6} fill="#faf9f7" stroke="#e7e5e4"/>
          <text x={210} y={195} fill="#059669" fontSize="8" fontFamily="monospace">Write path: Post → Post DB → Fan-Out Workers → Feed Cache</text>
          <text x={210} y={210} fill="#9333ea" fontSize="8" fontFamily="monospace">Read path: Feed Cache + Celebrity Query → Rank → Return</text>
        </svg>
      </Card>
    </div>
  );
}

function AlgorithmSection() {
  const [sel, setSel] = useState("fanout_write");
  const algos = {
    fanout_write: { name: "Fan-Out on Write (Push) ★", cx: "Write: O(followers) / Read: O(1)",
      pros: ["Feed reads are instant — single cache lookup","Feed is pre-computed and ready","Simple read path, easy to cache","Works perfectly for 99% of users (< 10K followers)"],
      cons: ["Celebrity problem: 1 post → 50M writes","Wasted work: many followers are inactive (won't read the feed)","Write amplification: high fan-out consumes massive I/O","Feed update delay proportional to follower count"],
      when: "Default choice for most social networks. Use for all users with < 10K followers. Twitter's home timeline, Instagram, LinkedIn all use this as the primary approach.",
      code: `# Fan-Out on Write — Post Creation Flow
on_new_post(author_id, post):
    # 1. Save post to Post DB
    post_id = post_db.save(post)

    # 2. Get all followers
    followers = social_graph.get_followers(author_id)

    # 3. Check: is this a celebrity?
    if len(followers) > CELEBRITY_THRESHOLD:  # e.g., 10,000
        return  # Skip fan-out — use fan-out-on-read

    # 4. Fan-out: write to each follower's feed cache
    for follower_id in followers:
        feed_cache.push(
            key=f"feed:{follower_id}",
            entry={post_id, author_id, timestamp, score},
            max_length=500   # trim old entries
        )
    # Done in background workers, NOT in the request path` },
    fanout_read: { name: "Fan-Out on Read (Pull)", cx: "Write: O(1) / Read: O(following)",
      pros: ["Zero write amplification — post is written once","No wasted work for inactive followers","Always shows freshest content (real-time)","Works for celebrities without any special handling"],
      cons: ["Slow reads: must query all followed users' posts","Read latency grows with number of followed users","Hard to rank/sort across many sources in real-time","Puts heavy load on post DB for every feed request"],
      when: "Use for celebrity accounts in the hybrid model. Also works for recommendation-based feeds (TikTok's For You page) where you're not limited to following relationships.",
      code: `# Fan-Out on Read — Feed Request Flow
get_feed(user_id, cursor, limit=20):
    # 1. Get list of users this person follows
    following = social_graph.get_following(user_id)

    # 2. For each followed user, get recent posts
    all_posts = []
    for followed_id in following:
        posts = post_db.get_recent(followed_id, since=cursor, limit=50)
        all_posts.extend(posts)

    # 3. Sort by timestamp (or rank by ML model)
    all_posts.sort(key=lambda p: p.timestamp, reverse=True)

    # 4. Return top N
    return all_posts[:limit]

# Problem: if user follows 500 people, that's 500 DB queries
# Solution: cache recent posts per user, batch queries` },
    hybrid: { name: "Hybrid Approach ★★ (Recommended)", cx: "Write: O(non-celeb followers) / Read: O(celeb following)",
      pros: ["Best of both worlds: fast reads + handles celebrities","99% of posts fan-out instantly (< 10K followers)","Celebrity posts are fresh (always live-queried)","Used by Facebook, Instagram, Twitter in production"],
      cons: ["More complex: two code paths (push + pull)","Feed merge logic on read (combine cached + live)","Need to define and maintain celebrity threshold","Slightly slower reads than pure push (merge step)"],
      when: "The recommended answer for interviews. Show you understand the tradeoff and can combine both approaches. Threshold: 10K followers is a common cutoff. Some systems use graduated thresholds (10K → partial fan-out).",
      code: `# Hybrid — Recommended Production Approach
CELEB_THRESHOLD = 10_000

on_new_post(author_id, post):
    post_id = post_db.save(post)
    followers = social_graph.get_followers(author_id)

    if len(followers) <= CELEB_THRESHOLD:
        # Normal user: fan-out-on-write
        fan_out_workers.enqueue(post_id, followers)
    else:
        # Celebrity: skip fan-out, will be pulled on read
        celebrity_posts_cache.push(author_id, post_id)

get_feed(user_id, cursor, limit=20):
    # 1. Get pre-computed feed (from fan-out)
    cached_feed = feed_cache.get(f"feed:{user_id}", cursor, limit=100)

    # 2. Get celebrity posts (fan-out-on-read)
    celeb_ids = social_graph.get_following_celebrities(user_id)
    celeb_posts = []
    for cid in celeb_ids:
        celeb_posts.extend(celebrity_posts_cache.get_recent(cid))

    # 3. Merge + rank + return top N
    merged = cached_feed + celeb_posts
    ranked = ranking_service.rank(user_id, merged)
    return ranked[:limit]` },
    ranking: { name: "Feed Ranking (ML-Based)", cx: "O(candidates × features)",
      pros: ["Dramatically increases engagement vs chronological","Personalizes feed per user","Can optimize for multiple objectives (engagement, diversity, quality)","Industry standard for all major platforms since ~2016"],
      cons: ["Complex ML pipeline: training, serving, monitoring","Cold start: new users have no signal","Filter bubbles / echo chambers","Latency: scoring 500 candidates in <50ms is hard"],
      when: "Follow-up discussion after the core feed generation design. Interviewers often ask: 'how would you rank the feed?' Have a high-level answer ready: feature extraction → ML model → scoring → diversification.",
      code: `# Feed Ranking — Simplified ML Pipeline
rank_feed(user_id, candidate_posts):
    scored = []
    for post in candidate_posts:
        features = extract_features(user_id, post)
        # Features include:
        #   - affinity: how often user interacts with author
        #   - recency: time since post created
        #   - engagement: likes, comments, shares so far
        #   - content_type: photo, video, text, link
        #   - user_history: what types user engages with

        score = ml_model.predict(features)
        #   → P(like) × w1 + P(comment) × w2 + P(share) × w3
        scored.append((post, score))

    # Sort by score, then diversify
    scored.sort(key=lambda x: x[1], reverse=True)

    # Diversify: don't show 5 posts from same author in a row
    return diversify(scored, max_consecutive_same_author=2)` },
  };
  const a = algos[sel];
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Fan-Out Strategy Comparison</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[12px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Strategy</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Write Cost</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Read Cost</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Latency</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Celebrity</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Used By</th>
            </tr></thead>
            <tbody>
              {[
                { n:"Push (FoW) ★", w:"O(followers)", r:"O(1)", l:"Read: <10ms", c:"❌ Breaks", u:"Twitter, Instagram", hl:true },
                { n:"Pull (FoR)", w:"O(1)", r:"O(following)", l:"Read: 100-500ms", c:"✓ Fine", u:"TikTok (rec-based)" },
                { n:"Hybrid ★★", w:"O(non-celeb)", r:"O(celeb follows)", l:"Read: 10-50ms", c:"✓ Handled", u:"Facebook, LinkedIn", hl:true },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-purple-50" : ""}>
                  <td className={`px-3 py-2 font-mono ${r.hl?"text-purple-700 font-bold":"text-stone-600"}`}>{r.n}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.w}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.r}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.l}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.c}</td>
                  <td className="text-center px-3 py-2 text-stone-400">{r.u}</td>
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
          <Label color="#dc2626">Database Schema</Label>
          <CodeBlock code={`-- Posts table (sharded by author_id)
CREATE TABLE posts (
  post_id     BIGINT PRIMARY KEY,  -- Snowflake ID (time-sortable)
  author_id   BIGINT NOT NULL,
  content     TEXT,
  media_type  ENUM('none','image','video','link'),
  media_url   VARCHAR(512),
  visibility  ENUM('public','followers','private'),
  created_at  TIMESTAMP,
  INDEX idx_author_time (author_id, created_at DESC)
);

-- Social graph (sharded by follower_id)
CREATE TABLE follows (
  follower_id  BIGINT NOT NULL,
  followee_id  BIGINT NOT NULL,
  created_at   TIMESTAMP,
  PRIMARY KEY (follower_id, followee_id),
  INDEX idx_followee (followee_id)  -- "who follows me?"
);

-- Feed cache (Redis sorted set per user)
-- Key: feed:{user_id}
-- Score: timestamp (or ranking score)
-- Value: post_id
-- ZRANGEBYSCORE feed:42 -inf +inf LIMIT 0 20`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Why These Design Choices?</Label>
          <div className="space-y-3">
            {[
              { q: "Snowflake IDs for post_id?", a: "Time-sortable + globally unique + no coordination needed. Encodes: timestamp (41 bits) + machine ID (10 bits) + sequence (12 bits). Can sort by post_id instead of created_at — faster index scan." },
              { q: "Shard posts by author_id?", a: "All posts by one user on same shard. GET /users/:id/posts is a single-shard query. Trade-off: celebrity authors create hot shards. Mitigate with read replicas for hot users." },
              { q: "Shard follows by follower_id?", a: "GET /users/:id/following (who do I follow) is the hot path for fan-out-on-read. Needs to be single-shard. Trade-off: GET /users/:id/followers (who follows me) requires scatter-gather." },
              { q: "Redis sorted set for feed?", a: "ZRANGEBYSCORE gives paginated feed in O(log n + k). Score = timestamp for chronological, or ML ranking score. ZADD automatically deduplicates by post_id. ZREMRANGEBYRANK trims to max length." },
              { q: "Why not store full posts in Redis feed?", a: "Feed cache stores only post IDs (32 bytes per entry). Full posts live in Post DB with a separate read-through cache. Saves 100× memory in the feed cache. Trade-off: extra lookup to hydrate posts." },
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

function ScalabilitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Read Path Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Feed cache (Redis Cluster)</strong> — 125+ shards holding pre-computed feeds. Shard by user_id. Each feed read is a single-shard ZRANGEBYSCORE — O(log n + k).</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Post content cache</strong> — separate cache for post bodies. Feed returns IDs → batch-fetch post content from this cache. MGET for efficiency.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">User profile cache</strong> — author names, avatars, verification badges cached. Avoids N lookups per feed page. Short TTL (5 min) for profile changes.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">CDN for media</strong> — images/videos served from edge CDN. Feed response includes CDN URLs. Client fetches media directly from CDN, not through API.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Write Path Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Fan-out workers (async)</strong> — post creation returns immediately. Fan-out happens in background via message queue (Kafka). Scale workers independently based on queue depth.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Batch fan-out</strong> — don't write to each follower's feed individually. Batch writes: get 1000 follower IDs → pipeline 1000 ZADD commands in one Redis round-trip.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Priority fan-out</strong> — fan-out to active users first (logged in last 7 days). Inactive users get their feed built on-demand when they return.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Social graph partitioning</strong> — shard follower lists so each worker handles a partition. Parallelizes fan-out across workers.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Strategy</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Option A: Per-Region Feed Cache ★", d:"Each region has its own feed cache. Fan-out workers in each region populate local cache from replicated post DB.", pros:["Lowest read latency","Region is self-contained","No cross-region dependency for reads"], cons:["Fan-out work duplicated per region","Slight delay for cross-region posts"], pick:true },
            { t:"Option B: Global Feed Cache", d:"Single feed cache, replicated globally. All fan-out writes go to primary, async replicate to other regions.", pros:["Single source of truth","No duplicate fan-out work"], cons:["Cross-region write latency","Single point of failure","Replication lag = stale feeds"], pick:false },
            { t:"Option C: User-Homed Regions", d:"Each user is 'homed' to the nearest region. Their feed cache lives there. Cross-region follows require remote fan-out.", pros:["User always reads from local cache","Balanced load across regions"], cons:["Complex: cross-region fan-out for international follows","User relocation is hard"], pick:false },
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
        <Label color="#d97706">Critical Decision: What If Feed Cache Is Down?</Label>
        <p className="text-[12px] text-stone-500 mb-4">The feed MUST always load. An empty feed = users leave. This is the highest-availability component in any social network.</p>
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Tier 1: Serve Stale Cache</div>
            <p className="text-[11px] text-stone-500">Feed cache returns last-known feed even if slightly stale. User sees content from 5 min ago — better than nothing.</p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-white p-4">
            <div className="text-[11px] font-bold text-amber-700 mb-1.5">Tier 2: Fan-Out on Read Fallback</div>
            <p className="text-[11px] text-stone-500">If pre-computed feed unavailable, dynamically build feed from Post DB. Slower (500ms+) but functional.</p>
          </div>
          <div className="rounded-lg border border-red-200 bg-white p-4">
            <div className="text-[11px] font-bold text-red-600 mb-1.5">Tier 3: Generic Popular Feed</div>
            <p className="text-[11px] text-stone-500">If everything is down, serve a pre-computed "trending" feed. Not personalized, but the app isn't blank.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#2563eb">Read-After-Write Consistency</Label>
          <p className="text-[12px] text-stone-500 mb-3">User posts, then immediately checks their feed. Their post must be there — even if fan-out hasn't completed.</p>
          <ul className="space-y-2.5">
            <Point icon="→" color="#2563eb">On feed read: always merge user's own recent posts (from Post DB) into feed response</Point>
            <Point icon="→" color="#2563eb">Client-side: optimistically insert new post into local feed state immediately</Point>
            <Point icon="→" color="#2563eb">Route user's own feed read to the same region where they posted (sticky routing by user_id)</Point>
          </ul>
        </Card>
        <Card>
          <Label color="#d97706">Fan-Out Worker Failure</Label>
          <p className="text-[12px] text-stone-500 mb-3">If a fan-out worker crashes mid-way through delivering to 10,000 followers, some feeds are updated and some aren't.</p>
          <ul className="space-y-2.5">
            <Point icon="→" color="#d97706">Kafka consumer with at-least-once delivery — message replayed on failure</Point>
            <Point icon="→" color="#d97706">ZADD is idempotent — re-delivering same post_id to a sorted set is a no-op</Point>
            <Point icon="→" color="#d97706">Checkpoint progress: save last-processed follower offset. Resume from there on restart.</Point>
            <Point icon="→" color="#d97706">Dead letter queue for persistently failing fan-outs. Alert after 3 retries.</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#9333ea">Degradation Ladder</Label>
        <div className="flex gap-3 items-stretch mt-1">
          {[
            { label: "Full System", sub: "Pre-computed + ranked", color: "#059669", status: "HEALTHY" },
            { label: "Stale Feed", sub: "Cache, no ranking", color: "#d97706", status: "DEGRADED" },
            { label: "Pull-Based", sub: "Live query from DB", color: "#ea580c", status: "FALLBACK" },
            { label: "Trending Feed", sub: "Generic, non-personalized", color: "#dc2626", status: "EMERGENCY" },
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
              { metric: "feed.load_latency_ms", type: "Histogram", desc: "p50/p95/p99 of GET /feed. Target: p99 < 500ms." },
              { metric: "feed.fanout_lag_sec", type: "Gauge", desc: "Time from post creation to appearing in last follower's feed. Target: < 5s for normal users." },
              { metric: "feed.cache_hit_ratio", type: "Gauge", desc: "% of feed reads served from cache. Target: > 95%." },
              { metric: "fanout.queue_depth", type: "Gauge", desc: "Kafka consumer lag for fan-out workers. Alert if growing (workers can't keep up)." },
              { metric: "feed.empty_rate", type: "Gauge", desc: "% of feed requests returning 0 posts. Should be near 0 for active users. Spike = system broken." },
              { metric: "post.create_latency_ms", type: "Histogram", desc: "Post creation response time. Should be < 200ms (fan-out is async)." },
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
              { name: "Feed Latency Spike", rule: "feed.load p99 > 1s for 3min", sev: "P2", action: "Check: cache hit ratio dropped? Ranking service slow? Celebrity fan-out-on-read queries exploding?" },
              { name: "Fan-Out Lag", rule: "fanout_lag > 30s for 5min", sev: "P2", action: "Check: worker pods healthy? Kafka partition assignment? Celebrity post causing fan-out storm?" },
              { name: "Empty Feeds", rule: "empty_rate > 1% for 5min", sev: "P1", action: "Users seeing blank feeds. Check: feed cache down? Fan-out completely broken? DB connection issues?" },
              { name: "Cache Miss Spike", rule: "cache_hit_ratio < 80%", sev: "P1", action: "Cache evictions or restart? Memory pressure? New deployment purged cache? Warm immediately." },
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
              { q: "User says 'my post isn't showing in my friend's feed'", steps: "Check: fan-out worker processed the post? Kafka lag? Friend's feed cache contains post_id? If missing, check: are they still following? Post visibility set to 'followers'?" },
              { q: "Feed loads but content seems stale", steps: "Check: fan-out lag metric. Cache serving old data? Feed cache TTL set too long? Ranking model deprioritizing recent posts? Celebrity post path broken?" },
              { q: "Feed latency spiked to 2-3 seconds", steps: "Check: ranking service latency (ML inference slow?). Cache miss rate (cold cache?). Celebrity fan-out-on-read queries (too many celeb follows?). Post content hydration slow (post cache miss?)." },
              { q: "Some users see empty feed after following new people", steps: "Check: follow event triggering feed backfill? When user follows someone new, need to inject their recent posts into the feed. Backfill worker running?" },
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
          { title: "Celebrity / Hot User Problem", sev: "Critical", sevColor: "#dc2626",
            desc: "User with 50M followers posts. Fan-out-on-write attempts 50M cache writes. Takes minutes, overloads Redis shards, delays all other fan-outs in the queue.",
            fix: "Hybrid approach: users above threshold (10K followers) skip fan-out. Their posts are pulled live at feed read time. Or: tiered fan-out — immediately fan-out to active users, delay for inactive.",
            code: `Celebrity posts:\n  Justin Bieber posts → 50M followers\n  Fan-out at 100K writes/sec = 500 seconds = 8+ minutes\n  Meanwhile, ALL other fan-outs are queued behind\n\nSolution: Hybrid threshold\n  if followers > 10K: skip fan-out (pull on read)\n  if followers ≤ 10K: normal fan-out (push)\n  Result: 99% of posts fan-out instantly\n  Celebrity posts: 0 fan-out cost on write` },
          { title: "Feed Cache Cold Start / Warming", sev: "Critical", sevColor: "#dc2626",
            desc: "New cache cluster deployed or cache eviction clears feeds. All feed reads become misses. Fallback to DB crushes the database.",
            fix: "Pre-warm feed cache before routing traffic. Replay recent fan-out events from Kafka (retain 7 days). Gradual traffic shift with canary. For new users: compute feed on first login from post DB, then cache it.",
            code: `Cold cache scenario:\n  Feed cache restart → all 500M feeds gone\n  175K feed reads/sec → all miss → hit DB\n  DB capacity: 10K QPS → instant overload\n\nSolution: Never deploy empty cache\n  1. Replay Kafka fan-out events (last 24h)\n  2. Pre-warm top 20% active users' feeds\n  3. Canary: 1% → 10% → 50% → 100% traffic\n  4. Monitor hit ratio at each step` },
          { title: "Feed Duplication / Missing Posts", sev: "Medium", sevColor: "#d97706",
            desc: "User sees same post twice in feed, or scrolls past a post that was never shown. Caused by concurrent fan-out writes + cursor pagination race condition.",
            fix: "Use Snowflake IDs as cursor (globally unique, time-ordered). Redis ZADD with post_id as member deduplicates automatically. Client-side dedup by post_id as safety net.",
            code: `Duplication scenario:\n  Cursor = timestamp. Two posts at same timestamp.\n  Page 1 shows post A. Page 2 shows post A again.\n\nMissing post scenario:\n  New post inserted between cursor reads.\n  Post falls in the 'gap' between pages.\n\nSolution: Snowflake ID cursor\n  → Globally unique, time-ordered\n  → cursor = last_post_snowflake_id\n  → No duplicates, no gaps\n  → Redis ZADD deduplicates by member` },
          { title: "Fan-Out Storm After System Recovery", sev: "Critical", sevColor: "#dc2626",
            desc: "System was down for 2 hours. When it recovers, there are 2 hours of posts queued in Kafka. Fan-out workers try to process all at once — overwhelm Redis.",
            fix: "Rate-limit fan-out workers on recovery. Process queue at steady-state throughput, not burst. Skip fan-out for posts older than 1 hour (users will get them via fan-out-on-read fallback). Backpressure mechanism.",
            code: `Recovery storm:\n  2 hours down = ~7M posts queued\n  7M × 200 followers = 1.4B fan-out writes\n  Workers process at max speed → Redis overloaded\n\nSolution: Throttled recovery\n  1. Rate-limit workers to 80% of steady-state\n  2. Skip fan-out for posts > 1 hour old\n  3. Those posts served via fan-out-on-read\n  4. Prioritize: recent posts first, then backfill` },
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
      <Card accent="#0f766e">
        <Label color="#0f766e">Service Decomposition — What We Actually Build</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { name: "Post Service", owns: "Create, read, update, delete posts. Store post content. Media upload coordination.", tech: "Go/Java microservice + sharded MySQL", api: "POST /posts, GET /posts/:id, GET /users/:id/posts", scale: "Horizontal — shard by author_id", stateful: false,
              modules: ["Post Creator (validate, save, emit event)", "Media Coordinator (pre-signed URL → S3 → CDN)", "Post Cache (read-through, invalidate on edit/delete)", "Content Validator (text length, profanity filter)", "Snowflake ID Generator (time-sortable unique IDs)", "Event Publisher (Kafka: post.created, post.deleted)"] },
            { name: "Feed Service", owns: "Assemble and return personalized feed for a user. Merge cached + live + own posts.", tech: "Go/Java microservice + Redis Cluster", api: "GET /feed?cursor=&limit=20", scale: "Horizontal — stateless (state in Redis)", stateful: false,
              modules: ["Feed Assembler (merge cached feed + celeb posts + own posts)", "Post Hydrator (batch-fetch post content from Post Cache)", "Cursor Manager (Snowflake-based pagination)", "Celebrity Detector (check if followed user is celeb)", "Read-After-Write Merger (inject user's own recent posts)", "Fallback Handler (pull-based feed if cache miss)"] },
            { name: "Fan-Out Service", owns: "Consume post events. Distribute to followers' feed caches. Handle priority + batching.", tech: "Go workers consuming from Kafka", api: "Internal — Kafka consumer (no external API)", scale: "Horizontal — scale by Kafka partitions", stateful: false,
              modules: ["Kafka Consumer (post.created events)", "Follower Fetcher (get follower list from Social Graph)", "Celebrity Filter (skip if followers > threshold)", "Priority Sorter (active users first, inactive later)", "Batch Writer (pipeline 1000 ZADD per Redis call)", "Progress Tracker (checkpoint per post + partition)"] },
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
          <text x={15} y={22} fill="#059669" fontSize="10" fontWeight="700" fontFamily="monospace">WRITE PATH (Post Creation → Fan-Out)</text>

          <rect x={5} y={190} width={710} height={180} rx={8} fill="#2563eb04" stroke="#2563eb20" strokeWidth={1} strokeDasharray="4,3"/>
          <text x={15} y={207} fill="#2563eb" fontSize="10" fontWeight="700" fontFamily="monospace">READ PATH (Feed Assembly)</text>

          {/* Write path */}
          <rect x={20} y={40} width={80} height={40} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={60} y={58} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Client</text>
          <text x={60} y={70} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">POST /posts</text>

          <rect x={130} y={40} width={85} height={40} rx={6} fill="#6366f110" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={172} y={58} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">API Gateway</text>
          <text x={172} y={70} textAnchor="middle" fill="#6366f180" fontSize="7" fontFamily="monospace">auth + rate limit</text>

          <rect x={245} y={40} width={85} height={40} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={287} y={58} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Post Service</text>
          <text x={287} y={70} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">validate + save</text>

          <rect x={360} y={40} width={70} height={40} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={395} y={58} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Post DB</text>
          <text x={395} y={70} textAnchor="middle" fill="#78716c80" fontSize="7" fontFamily="monospace">sharded MySQL</text>

          <rect x={245} y={100} width={85} height={36} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={287} y={116} textAnchor="middle" fill="#ea580c" fontSize="9" fontWeight="600" fontFamily="monospace">Kafka</text>
          <text x={287} y={128} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">post.created</text>

          <rect x={370} y={100} width={90} height={36} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={415} y={116} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">Fan-Out Workers</text>
          <text x={415} y={128} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">batch ZADD</text>

          <rect x={495} y={85} width={80} height={30} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={535} y={104} textAnchor="middle" fill="#0891b2" fontSize="9" fontWeight="600" fontFamily="monospace">Social Graph</text>

          <rect x={495} y={122} width={80} height={30} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={535} y={141} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Feed Cache</text>

          <rect x={605} y={100} width={100} height={52} rx={6} fill="#d9770608" stroke="#d97706" strokeWidth={1}/>
          <text x={655} y={117} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Redis Cluster</text>
          <text x={655} y={130} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">feed:{'{user_id}'}</text>
          <text x={655} y={142} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">Sorted Set (ZADD)</text>

          {/* Write arrows */}
          <line x1={100} y1={60} x2={130} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={215} y1={60} x2={245} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={330} y1={60} x2={360} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={287} y1={80} x2={287} y2={100} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={330} y1={118} x2={370} y2={118} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={460} y1={108} x2={495} y2={100} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={460} y1={126} x2={495} y2={137} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={575} y1={137} x2={605} y2={130} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>

          {/* Read path */}
          <rect x={20} y={220} width={80} height={40} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={60} y={238} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Client</text>
          <text x={60} y={250} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">GET /feed</text>

          <rect x={130} y={220} width={85} height={40} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={172} y={238} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Feed Service</text>
          <text x={172} y={250} textAnchor="middle" fill="#9333ea80" fontSize="7" fontFamily="monospace">assemble feed</text>

          <rect x={255} y={205} width={90} height={30} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={300} y={224} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Feed Cache ①</text>

          <rect x={255} y={245} width={90} height={30} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={300} y={264} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Celeb Posts ②</text>

          <rect x={255} y={285} width={90} height={30} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={300} y={304} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Own Posts ③</text>

          <rect x={385} y={240} width={85} height={40} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={427} y={256} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">Merge + Rank</text>
          <text x={427} y={270} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">ML scoring</text>

          <rect x={505} y={220} width={85} height={40} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={547} y={238} textAnchor="middle" fill="#0891b2" fontSize="9" fontWeight="600" fontFamily="monospace">Post Hydrator</text>
          <text x={547} y={250} textAnchor="middle" fill="#0891b280" fontSize="7" fontFamily="monospace">MGET content</text>

          <rect x={625} y={220} width={80} height={40} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={665} y={238} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Post Cache</text>
          <text x={665} y={250} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">content + media</text>

          {/* Read arrows */}
          <line x1={100} y1={240} x2={130} y2={240} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={215} y1={230} x2={255} y2={220} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={215} y1={240} x2={255} y2={257} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={215} y1={250} x2={255} y2={297} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={345} y1={220} x2={385} y2={250} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={345} y1={260} x2={385} y2={260} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={345} y1={300} x2={385} y2={270} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={470} y1={255} x2={505} y2={242} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>
          <line x1={590} y1={240} x2={625} y2={240} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-sd1)"/>

          {/* Labels */}
          <text x={245} y={215} fill="#d9770690" fontSize="7" fontFamily="monospace">pre-computed</text>
          <text x={245} y={255} fill="#05966990" fontSize="7" fontFamily="monospace">live query</text>
          <text x={245} y={295} fill="#2563eb90" fontSize="7" fontFamily="monospace">read-after-write</text>

          {/* Legend */}
          <rect x={385} y={295} width={310} height={65} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={395} y={310} fill="#78716c" fontSize="8" fontWeight="600" fontFamily="monospace">Key Design Decisions</text>
          <text x={395} y={325} fill="#78716c" fontSize="7" fontFamily="monospace">① Pre-computed feed from fan-out (normal users, &lt;10K followers)</text>
          <text x={395} y={338} fill="#78716c" fontSize="7" fontFamily="monospace">② Celebrity posts pulled live at read time (hybrid approach)</text>
          <text x={395} y={351} fill="#78716c" fontSize="7" fontFamily="monospace">③ User's own recent posts always merged (read-after-write consistency)</text>

          <defs><marker id="ah-sd1" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
        </svg>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#2563eb">
          <Label color="#2563eb">Supporting Services</Label>
          <div className="space-y-3">
            {[
              { name: "Social Graph Service", role: "Stores follow relationships. get_followers(user_id), get_following(user_id), is_celebrity(user_id).", tech: "Sharded MySQL or TAO-like graph store", critical: true },
              { name: "Ranking Service", role: "ML model inference. Scores candidate posts by P(engagement). Returns ranked list.", tech: "Python/TF Serving behind gRPC. Feature store (Redis).", critical: false },
              { name: "Notification Service", role: "Push notifications for new posts from close friends, mentions, likes on your posts.", tech: "Kafka consumer → APNs/FCM", critical: false },
              { name: "Content Moderation", role: "Async ML pipeline. Scans new posts for policy violations. Can retroactively hide posts from feeds.", tech: "Kafka consumer → ML classifiers → action queue", critical: true },
              { name: "Feed Backfill Worker", role: "When user follows someone new, inject their recent posts into user's feed cache.", tech: "Triggered by follow events. Reads recent posts, ZADD to feed.", critical: false },
              { name: "Analytics Pipeline", role: "Track impressions, engagement per post. Feed back into ranking model training data.", tech: "Kafka → Flink → data warehouse (BigQuery/Hive)", critical: false },
            ].map((s,i) => (
              <div key={i} className="flex items-start gap-3 border border-stone-100 rounded-lg p-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-bold text-stone-700">{s.name}</span>
                    {s.critical && <Pill bg="#fef2f2" color="#dc2626">Critical</Pill>}
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
            <table className="w-full text-[10px]">
              <thead><tr className="bg-stone-50">
                <th className="text-left px-2 py-1.5 font-semibold text-stone-500">Caller → Callee</th>
                <th className="text-left px-2 py-1.5 font-semibold text-stone-500">Proto</th>
                <th className="text-left px-2 py-1.5 font-semibold text-stone-500">Timeout</th>
                <th className="text-left px-2 py-1.5 font-semibold text-stone-500">On Failure</th>
              </tr></thead>
              <tbody>
                {[
                  { route: "Feed Svc → Feed Cache", proto: "RESP", timeout: "50ms", fail: "Pull-based fallback from DB" },
                  { route: "Feed Svc → Post Cache", proto: "RESP", timeout: "50ms", fail: "Query Post DB directly" },
                  { route: "Feed Svc → Ranking Svc", proto: "gRPC", timeout: "100ms", fail: "Return chronological order" },
                  { route: "Post Svc → Kafka", proto: "Kafka", timeout: "Async", fail: "Retry 3x, then DLQ" },
                  { route: "Fan-Out → Social Graph", proto: "gRPC", timeout: "200ms", fail: "Retry from checkpoint" },
                  { route: "Fan-Out → Feed Cache", proto: "RESP", timeout: "30ms", fail: "Skip user, retry later" },
                  { route: "Client → Feed Svc", proto: "HTTP/2", timeout: "3s (client)", fail: "Show cached feed or spinner" },
                ].map((r,i) => (
                  <tr key={i} className={i%2?"bg-stone-50/50":""}>
                    <td className="px-2 py-1.5 font-mono text-teal-700">{r.route}</td>
                    <td className="px-2 py-1.5 text-stone-500">{r.proto}</td>
                    <td className="px-2 py-1.5 font-mono text-stone-400">{r.timeout}</td>
                    <td className="px-2 py-1.5 text-stone-400">{r.fail}</td>
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
  const [flow, setFlow] = useState("post");
  const flows = {
    post: { title: "Post Creation → Fan-Out", steps: [
      { actor: "Client", action: "POST /v1/posts {content: 'Hello!', media_ids: ['img_1']}", type: "request" },
      { actor: "API Gateway", action: "Auth (JWT), rate limit check, route to Post Service", type: "auth" },
      { actor: "Post Service", action: "Validate content (length, profanity). Generate Snowflake ID.", type: "process" },
      { actor: "Post Service → DB", action: "INSERT INTO posts (post_id, author_id, content, ...) — sharded by author_id", type: "request" },
      { actor: "Post Service → Kafka", action: "Emit {event: 'post.created', post_id, author_id, timestamp} to 'posts' topic", type: "process" },
      { actor: "Post Service", action: "Return 201 {post_id, created_at} to client. Post is live. Fan-out is async.", type: "success" },
      { actor: "Fan-Out Worker", action: "Consume post.created event from Kafka partition", type: "process" },
      { actor: "Fan-Out → Social Graph", action: "get_followers(author_id) → [follower_1, ..., follower_N]", type: "request" },
      { actor: "Fan-Out Worker", action: "Check: N > 10K? → Celebrity, skip fan-out. N ≤ 10K? → continue.", type: "check" },
      { actor: "Fan-Out → Redis", action: "Pipeline: ZADD feed:{follower_1} score post_id, ... ZADD feed:{follower_N} score post_id (batches of 1000)", type: "request" },
      { actor: "Fan-Out → Redis", action: "ZREMRANGEBYRANK feed:{follower_i} 0 -(MAX_FEED_LENGTH+1) — trim old entries", type: "process" },
      { actor: "Fan-Out Worker", action: "Checkpoint: mark post_id as fully fanned-out. Commit Kafka offset.", type: "success" },
    ]},
    read: { title: "Feed Read (Hybrid Path)", steps: [
      { actor: "Client", action: "GET /v1/feed?cursor=1707500000.000&limit=20", type: "request" },
      { actor: "API Gateway", action: "Auth, rate limit (100/min per user)", type: "auth" },
      { actor: "Feed Service", action: "Get user's followed celebrity list from Social Graph (cached locally, 5min TTL)", type: "process" },
      { actor: "Feed Svc → Redis", action: "ZREVRANGEBYSCORE feed:{user_id} {cursor} -inf LIMIT 0 100 — pre-computed posts", type: "request" },
      { actor: "Feed Svc → Post DB", action: "For each followed celebrity: get recent posts since cursor (batched query)", type: "request" },
      { actor: "Feed Service", action: "Merge pre-computed feed + celebrity posts + user's own recent posts (read-after-write)", type: "process" },
      { actor: "Feed Svc → Ranking", action: "gRPC: rank(user_id, candidate_posts[]) → scored + ordered list (timeout: 100ms)", type: "request" },
      { actor: "Feed Service", action: "If ranking timeout → fallback to chronological order", type: "check" },
      { actor: "Feed Svc → Post Cache", action: "MGET post content for top 20 results (author name, media URLs, counts)", type: "request" },
      { actor: "Feed Service", action: "Return paginated feed with next_cursor = last post's Snowflake ID", type: "success" },
    ]},
    follow: { title: "Follow User → Feed Backfill", steps: [
      { actor: "Client", action: "POST /v1/users/789/follow (user 42 follows user 789)", type: "request" },
      { actor: "Social Graph", action: "INSERT INTO follows (42, 789). Update cached follower/following counts.", type: "process" },
      { actor: "Social Graph → Kafka", action: "Emit {event: 'follow.created', follower: 42, followee: 789}", type: "process" },
      { actor: "API", action: "Return 200 OK immediately. Backfill is async.", type: "success" },
      { actor: "Backfill Worker", action: "Consume follow.created event", type: "process" },
      { actor: "Backfill → Post DB", action: "Get 789's last 50 posts (recent enough to be relevant)", type: "request" },
      { actor: "Backfill → Redis", action: "ZADD feed:42 [score] [post_id] for each of 789's posts. Feed now includes their content.", type: "request" },
      { actor: "Note", action: "Next time user 42 loads feed, they'll see user 789's recent posts mixed in.", type: "success" },
    ]},
    failure: { title: "Feed Cache Down → Degradation", steps: [
      { actor: "Client", action: "GET /v1/feed", type: "request" },
      { actor: "Feed Svc → Redis", action: "ZREVRANGEBYSCORE → connection timeout after 50ms", type: "error" },
      { actor: "Feed Service", action: "Circuit breaker trips for feed cache shard. Metric: cache.fallback_active=1", type: "error" },
      { actor: "Feed Service", action: "Fallback: fan-out-on-read. Get following list → batch query Post DB for recent posts.", type: "check" },
      { actor: "Feed Svc → Post DB", action: "SELECT * FROM posts WHERE author_id IN (...following) ORDER BY created_at DESC LIMIT 100", type: "request" },
      { actor: "Feed Service", action: "Skip ranking (reduce latency in degraded mode). Return chronological.", type: "process" },
      { actor: "Feed Service", action: "Return feed to client. Latency: ~800ms (vs normal ~100ms). But functional.", type: "success" },
      { actor: "Alert", action: "P1 alert: feed cache down. On-call investigates. Cache warming begins for recovery.", type: "error" },
    ]},
  };
  const f = flows[flow];
  const typeColors = { request:"#2563eb", auth:"#d97706", process:"#6b7280", success:"#059669", error:"#dc2626", check:"#9333ea" };
  return (
    <div className="space-y-5">
      <Card accent="#7e22ce">
        <Label color="#7e22ce">Detailed Request Flows</Label>
        <div className="flex gap-2 mb-4 flex-wrap">
          {Object.entries(flows).map(([k,v]) => (
            <button key={k} onClick={() => setFlow(k)}
              className={`px-3.5 py-1.5 rounded-lg text-[12px] font-medium transition-all border ${k===flow ? "bg-purple-700 text-white border-purple-700" : "bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
              {v.title.split("→")[0].trim()}
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
          <Label color="#b45309">K8s Deployment Strategy</Label>
          <CodeBlock title="Fan-Out Workers — Kafka Consumer Group" code={`apiVersion: apps/v1
kind: Deployment
metadata:
  name: fanout-workers
spec:
  replicas: 30               # Match Kafka partitions
  template:
    spec:
      containers:
      - name: fanout
        image: feed/fanout-worker:v2.3
        env:
        - name: KAFKA_TOPIC
          value: "posts"
        - name: KAFKA_GROUP_ID
          value: "fanout-workers"
        - name: REDIS_CLUSTER
          value: "feed-cache.redis.svc"
        - name: CELEBRITY_THRESHOLD
          value: "10000"
        - name: BATCH_SIZE
          value: "1000"     # ZADD pipeline size
        resources:
          requests:
            cpu: "1"
            memory: "512Mi"
      # Key: 1 worker per Kafka partition for ordering
      # Scale workers = scale Kafka partitions
      # NOT auto-scaled by HPA — Kafka rebalance is expensive
      # Instead: manual scaling during known traffic events`} />
          <div className="mt-3 space-y-1.5">
            <Point icon="⚠" color="#b45309">Workers = Kafka partitions. Adding workers beyond partition count = idle workers.</Point>
            <Point icon="⚠" color="#b45309">Kafka rebalance on pod restart takes 30-60s. Use cooperative-sticky assignor to minimize disruption.</Point>
            <Point icon="⚠" color="#b45309">Feed Service is stateless → HPA auto-scales on CPU. Fan-Out workers scale manually.</Point>
          </div>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Security & Content Safety</Label>
          <div className="space-y-3">
            {[
              { layer: "Post Creation", details: ["Content length limits (text: 2000 chars, media: 50MB)", "Profanity filter (sync, blocks post creation)", "Rate limit: 50 posts/day per user", "Spam detection ML model (async, can retroactively hide)"] },
              { layer: "Feed Serving", details: ["Only show posts from followed users (no data leaks)", "Respect post visibility (public/followers/private)", "Block: hide posts from blocked users in feed", "Mute: exclude muted users from feed without unfollowing"] },
              { layer: "Content Moderation", details: ["Async ML pipeline classifies images/text after creation", "Violating posts hidden from all feeds retroactively", "Appeal workflow: human review queue for edge cases", "CSAM detection: immediate takedown + report (legal requirement)"] },
              { layer: "Data Privacy", details: ["GDPR: user deletion removes all posts + feed entries", "Feed cache uses user_id (not PII) as key", "Analytics pipeline anonymizes after 30 days", "Followers list: privacy settings control visibility"] },
            ].map((s,i) => (
              <div key={i} className="rounded-lg border border-stone-200 p-3">
                <div className="text-[11px] font-bold text-stone-800 mb-1">{s.layer}</div>
                <div className="space-y-0.5">
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
    </div>
  );
}

function OpsSection() {
  return (
    <div className="space-y-5">
      <Card accent="#be123c">
        <Label color="#be123c">Scaling Bottleneck Map</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[11px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Bottleneck</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Symptom</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Fix</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Pitfall</th>
            </tr></thead>
            <tbody>
              {[
                { b: "Feed Cache Memory", s: "Evictions spike → hit ratio drops", f: "Add shards. Or reduce feed length (500→200 posts). Or evict inactive users.", p: "Don't just add memory — check if feed length or value size grew unexpectedly." },
                { b: "Fan-Out Throughput", s: "Kafka consumer lag growing", f: "Add Kafka partitions + workers. Or raise celebrity threshold (10K→5K).", p: "Adding workers beyond partitions = idle. Must add partitions first." },
                { b: "Social Graph Latency", s: "get_followers() calls slow → fan-out stalls", f: "Cache follower lists in Redis (TTL 5min). Denormalize hot users.", p: "Follower list cache must invalidate on follow/unfollow events." },
                { b: "Ranking Service", s: "ML inference > 100ms → feed latency spike", f: "Pre-compute scores offline. Or simplify model. Or skip ranking (chronological fallback).", p: "ML model complexity grows over time. Set a latency budget and enforce it." },
                { b: "Post DB Hot Shard", s: "Celebrity's shard overwhelmed by reads", f: "Read replicas for hot users. Or cache celeb posts aggressively (TTL 30s).", p: "Sharding by author_id means celebs are inherently hot. Accept and mitigate." },
                { b: "Kafka Backlog", s: "Events processed hours behind real-time", f: "Scale consumers. Compact topic. Drop old events beyond 1h (served via pull).", p: "Don't just scale consumers — find why they're slow (Redis writes? Graph lookups?)." },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-mono text-stone-700 font-medium">{r.b}</td>
                  <td className="px-3 py-2 text-red-500">{r.s}</td>
                  <td className="px-3 py-2 text-stone-500">{r.f}</td>
                  <td className="px-3 py-2 text-amber-600 text-[10px]">{r.p}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card accent="#be123c">
        <Label color="#be123c">Production War Stories</Label>
        <div className="grid grid-cols-2 gap-4">
          {[
            { title: "Celebrity Tweets Breaking the Fan-Out Queue", symptom: "One celebrity posts → 50M fan-out writes queued → all other users' posts delayed by 10+ minutes.",
              cause: "Fan-out is FIFO. Celebrity post with 50M followers blocks the queue. Normal users' posts (200 followers each) stuck behind it.",
              fix: "Separate Kafka topics/queues by priority. Celebrity posts → skip fan-out entirely (hybrid model). Or: dedicated fan-out partition for users > 10K followers with rate-limited processing.",
              quote: "Taylor Swift tweeted and our fan-out queue backed up to 40 minutes. 500M users saw stale feeds. We shipped the hybrid model the next sprint." },
            { title: "Feed Ranking A/B Test Tanks Engagement", symptom: "New ranking model deployed to 10% of users. Engagement drops 15% in treatment group. But model scored higher on offline metrics.",
              cause: "Offline metrics (AUC, NDCG) don't always correlate with online metrics (clicks, time-spent). Model optimized for predicted clicks but showed too many clickbait posts — users engaged less overall.",
              fix: "Always A/B test ranking changes online. Use guardrail metrics (session length, return rate) alongside primary metrics. Ramp slowly: 1% → 5% → 10%. Kill switch with instant rollback.",
              quote: "Our ML team was so proud of the new model. We had to roll it back in 4 hours. Offline eval is necessary but never sufficient." },
            { title: "Feed Backfill on Mass Follow Event", symptom: "Influencer tells followers to follow a specific account. 500K follows in 1 minute → 500K backfill jobs → Social Graph DB overwhelmed.",
              cause: "Each follow triggers a backfill: read recent posts → write to follower's feed. 500K simultaneous backfills = 500K × 50 post reads = 25M DB reads in 1 minute.",
              fix: "Rate-limit backfill workers. Debounce: batch follows and backfill once per user per minute. Lazy backfill: don't backfill immediately, let the next fan-out naturally populate the feed.",
              quote: "Mr. Beast said 'everyone follow @charity.' Our social graph service returned 503 for 8 minutes." },
            { title: "GDPR Deletion Cascading Through Feed Cache", symptom: "User requests account deletion. Their posts must be removed from ALL followers' feeds. With 500K followers, that's 500K Redis operations.",
              cause: "GDPR requires complete deletion within 30 days. Posts stored as IDs in feed cache. Must scan and remove from every follower's sorted set.",
              fix: "Async deletion worker: scan follower list, ZREM post_ids from each feed. Low priority (run overnight). Also: when feed is loaded, filter out deleted posts at read time (tombstone check). Belt and suspenders.",
              quote: "We got a GDPR audit. 'Show us that deleted user's posts are gone from all feeds.' Took us 3 days to prove it because the deletion worker had a bug." },
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
        { t: "ML-Based Ranking", d: "Replace chronological with engagement-optimized ranking. Feature extraction → model inference → re-rank candidates.", detail: "Two-pass: lightweight model filters 500 → 50 candidates, then heavy model ranks top 50. Latency budget: 50ms.", effort: "Hard" },
        { t: "Real-Time Feed Push (SSE/WS)", d: "Push new posts to connected clients instantly via Server-Sent Events or WebSocket, instead of poll-to-refresh.", detail: "Maintain connection per active user. When fan-out completes, notify connected client. Reduces 'pull to refresh' delay to zero.", effort: "Medium" },
        { t: "Stories / Ephemeral Content", d: "24-hour expiring content shown in a separate carousel above the feed. Separate storage + TTL.", detail: "Separate data path: stories stored with 24h TTL. No fan-out — pull on read (users check stories from followed users). CDN-heavy (media).", effort: "Medium" },
        { t: "Interest-Based Feed (Discover/Explore)", d: "Feed of content from users you DON'T follow, based on collaborative filtering and content similarity.", detail: "TikTok's For You page model: fully recommendation-based. No social graph dependency. Content embedding + user embedding similarity.", effort: "Hard" },
        { t: "Feed Diversity / Anti-Echo Chamber", d: "Ensure feed shows diverse topics and viewpoints, not just most-engaging content.", detail: "Post-ranking diversification: max 2 consecutive posts from same author, mix content types (photos, text, links), inject serendipity.", effort: "Medium" },
        { t: "Ads Integration", d: "Insert sponsored posts at predefined positions in the feed (every 5th post). Separate ad ranking pipeline.", detail: "Ad server returns ranked ads. Feed service interleaves: organic post 1-4, ad, organic 5-8, ad, etc. Track ad impressions separately.", effort: "Medium" },
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
    { q:"Why not just use fan-out-on-read for everyone?", a:"At 175K feed reads/sec, each requiring queries to 200+ followed users' posts, you'd generate 35M DB queries/sec. No database survives that. Fan-out-on-write pre-computes the answer so reads are O(1) cache lookups. The 300:1 read:write ratio means optimizing reads is the right call.", tags:["design"] },
    { q:"How do you handle the celebrity/hot user problem?", a:"Hybrid approach: set a threshold (e.g., 10K followers). Below threshold: normal fan-out-on-write. Above: skip fan-out, pull their posts at read time. On feed read: merge pre-computed feed + live celebrity query. Twitter uses this exact approach. The threshold is tunable — some systems use graduated tiers.", tags:["scalability"] },
    { q:"How do you ensure read-after-write consistency?", a:"Three layers: (1) Client-side: optimistically insert own post into local feed state. (2) Server-side: feed service always merges user's own recent posts (from Post DB) into feed response. (3) Routing: route user's feed reads to the same region where they posted.", tags:["consistency"] },
    { q:"Chronological vs ranked feed — which would you build?", a:"Start chronological (simpler, good enough for MVP). Then add ranking as an enhancement. For the interview: propose chronological first, then offer ranking as a follow-up. Ranking requires: feature extraction, ML model, A/B testing infrastructure, monitoring for engagement metrics.", tags:["design"] },
    { q:"How would you handle post deletion?", a:"Delete from Post DB. Emit post.deleted event on Kafka. Fan-out delete: remove post_id from all followers' feed cache (ZREM). Also filter at read time (check post exists before returning). Double-delete handles eventual consistency gaps. GDPR requires full deletion proof.", tags:["consistency"] },
    { q:"How would you add ads to the feed?", a:"Separate ad ranking service. Feed service calls it in parallel with organic feed assembly. Interleave: insert ad at every Nth position (configurable). Track ad impressions/clicks separately. Don't count ads toward feed pagination cursor. Users can 'hide ad' — feedback loop to ad targeting.", tags:["enhancement"] },
    { q:"What if Kafka goes down?", a:"Fan-out stops but feed reads still work (serving cached feeds). New posts are stored in Post DB but not fanned out. When Kafka recovers, replay from last committed offset. Posts created during outage get fanned out late — users see them appear retroactively. If extended: fall back to pull model.", tags:["availability"] },
    { q:"How would you handle a 'viral' post?", a:"Viral post = exponentially growing engagement (likes, comments, reshares). The post itself isn't the problem — it's already fanned out. The problem is engagement count updates: millions of like/comment events per second for one post. Solution: approximate counts (HyperLogLog), batch counter updates, separate counter cache.", tags:["scalability"] },
    { q:"How does this differ from TikTok's For You page?", a:"TikTok is pure recommendation — no social graph dependency. Content is ranked by ML model trained on watch time, likes, shares. It's fundamentally fan-out-on-read (or rather, recommendation-on-read). Every feed request runs an ML model over millions of candidate videos. Very different architecture from a follow-based feed.", tags:["design"] },
    { q:"How would you test this system?", a:"Unit: fan-out logic, cursor pagination, merge algorithm. Integration: post creation → verify appears in follower's feed within 5s. Load: simulate 175K feed reads/sec + celebrity posts. Chaos: kill fan-out workers, verify feeds still serve (stale). Shadow: replay production traffic against new version, compare feed contents.", tags:["testing"] },
  ];
  return (
    <div className="space-y-2.5">
      <Card className="bg-indigo-50/50 border-indigo-200 mb-1">
        <p className="text-[12px] text-stone-500">Common follow-up questions interviewers ask. Click to reveal a strong answer.</p>
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

export default function NewsFeedSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">News Feed</h1>
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