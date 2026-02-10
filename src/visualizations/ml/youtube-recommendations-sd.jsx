import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   YOUTUBE RECOMMENDATIONS — ML System Design Reference (Google L6)
   Pearl white theme · 17 sections (HLD + LLD + ML Deep Dive)
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",                icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",            icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",     icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",              icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",       icon: "🏗️", color: "#9333ea" },
  { id: "candidate",     label: "Candidate Generation",    icon: "🎯", color: "#c026d3" },
  { id: "ranking",       label: "Ranking Model",           icon: "🧠", color: "#dc2626" },
  { id: "features",      label: "Feature Engineering",     icon: "⚙️", color: "#d97706" },
  { id: "objectives",    label: "Multi-Objective",         icon: "⚖️", color: "#0f766e" },
  { id: "data",          label: "Data Model",              icon: "🗄️", color: "#059669" },
  { id: "training",      label: "Training Pipeline",       icon: "🔄", color: "#7e22ce" },
  { id: "coldstart",     label: "Cold Start & Explore",    icon: "🧊", color: "#0284c7" },
  { id: "scalability",   label: "Scalability",             icon: "📈", color: "#059669" },
  { id: "watchouts",     label: "Failure Modes",           icon: "⚠️", color: "#dc2626" },
  { id: "observability", label: "Observability",           icon: "📊", color: "#0284c7" },
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
            <Label>What is a Video Recommendation System?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A video recommendation system selects and ranks videos from a corpus of hundreds of millions to show each user a personalized feed. It powers the YouTube Homepage, "Up Next" sidebar, and Shorts feed — surfaces that drive <strong>over 70% of all watch time</strong> on YouTube.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              The fundamental challenge: given a user who just opened the app, select ~50 videos out of 800M+ that this specific user is most likely to watch and enjoy — in under 200ms. Unlike search (explicit intent), recommendations must <em>infer</em> intent from behavior. This is the hardest personalization problem at scale.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard? (Google L6 Depth)</Label>
            <ul className="space-y-2.5">
              <Point icon="📊" color="#0891b2">Massive corpus, sparse signals — 800M+ videos, 2B+ users, but each user has watched only 0.0001% of all videos. Extreme sparsity.</Point>
              <Point icon="⏱️" color="#0891b2">Multi-objective tension — maximize watch time? Users binge-watch clickbait. Maximize satisfaction? Hard to measure. Balance engagement, satisfaction, diversity, and responsibility.</Point>
              <Point icon="🔄" color="#0891b2">Distribution shift — new videos uploaded every second (500hrs/min). User interests drift. Yesterday's model doesn't reflect today's trending content.</Point>
              <Point icon="🎯" color="#0891b2">Implicit feedback only — users don't rate videos 1-5. You have: watch time, like/dislike, share, subscribe, scroll-past, "not interested". Each signal is noisy and biased.</Point>
              <Point icon="⚔️" color="#0891b2">Creator ecosystem — recommendations shape what creators make. Bad recommendations create perverse incentives (clickbait, misinformation). System impacts society.</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Scale</Label>
            <div className="space-y-2.5">
              {[
                { co: "YouTube", scale: "2B+ MAU, 1B hrs watched/day", approach: "Two-tower retrieval + deep ranking" },
                { co: "TikTok", scale: "1B+ MAU, pure rec-driven", approach: "Interest-based retrieval + ranking" },
                { co: "Netflix", scale: "260M subs, 80% rec-driven", approach: "Collaborative filtering + deep models" },
                { co: "Spotify", scale: "600M MAU, Discover Weekly", approach: "Collaborative + content + graph" },
                { co: "Instagram", scale: "2B+ MAU, Explore + Reels", approach: "Two-tower + multi-objective ranking" },
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
              <polygon points="30,10 330,10 290,55 70,55" fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
              <text x={180} y={28} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="700" fontFamily="monospace">Candidate Generation</text>
              <text x={180} y={42} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">800M videos → ~1000 candidates  |  ~50ms</text>

              <polygon points="70,60 290,60 260,105 100,105" fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
              <text x={180} y={78} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="700" fontFamily="monospace">Ranking (Deep Neural Network)</text>
              <text x={180} y={92} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">1000 → ~100 scored  |  ~80ms</text>

              <polygon points="100,110 260,110 240,155 120,155" fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
              <text x={180} y={128} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="700" fontFamily="monospace">Re-ranking + Business Logic</text>
              <text x={180} y={142} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">100 → 30 displayed  |  ~20ms</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Google Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">#2 most relevant ML system design for Google</div>
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
            <p className="text-[12px] text-stone-500 mt-0.5">"Design YouTube recommendations" is intentionally vague. <strong>Drive the scoping yourself</strong>: "I'll focus on the Homepage feed recommendation pipeline — candidate generation through final ranking. I'll design for multi-objective optimization (engagement + satisfaction) and discuss the cold-start problem. I'll treat video ingestion and search as black boxes." This signals L6 ownership.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Generate a personalized homepage feed of ~30-50 videos per user request</Point>
            <Point icon="2." color="#059669">Support multiple surfaces: Homepage, "Up Next" (watch page), Shorts feed</Point>
            <Point icon="3." color="#059669">Handle new users (cold start) with reasonable recommendations</Point>
            <Point icon="4." color="#059669">Support real-time user feedback: don't recommend what user already watched, disliked, or marked "not interested"</Point>
            <Point icon="5." color="#059669">Blend multiple content types: long-form, Shorts, Live, Premieres</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">End-to-end latency: p50 &lt;150ms, p99 &lt;400ms</Point>
            <Point icon="2." color="#dc2626">Throughput: ~500K homepage loads/sec at peak (2B MAU)</Point>
            <Point icon="3." color="#dc2626">Corpus: 800M+ videos, growing 500 hrs of video uploaded per minute</Point>
            <Point icon="4." color="#dc2626">Availability: 99.99% — homepage must always show something</Point>
            <Point icon="5." color="#dc2626">Quality: measurable via engagement metrics + user satisfaction surveys</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask (L6 Signal)</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Which surface: Homepage, Up Next, or Shorts? (architectures differ)",
            "What's the optimization objective? Pure watch time or multi-objective?",
            "Do we personalize for logged-out users? (~30% of traffic)",
            "How aggressive should explore/exploit be? (new creators vs. safe bets)",
            "Are there content policy constraints? (demote borderline content)",
            "Multi-language? (impacts embedding models, candidate pool)",
            "Do we need real-time adaptation? (in-session interest shift)",
            "How do we handle viral/trending content injection?",
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
            <p className="text-[12px] text-stone-500 mt-0.5">For recommendation systems, the key numbers to derive are: QPS for feed requests, candidate pool size, model inference budget per request, and embedding table sizes. These drive whether you need GPUs, how many retrieval shards, and your feature store sizing.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Traffic & Request Volume</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="MAU = 2B users" result="2B" note="YouTube scale. DAU ~ 800M (40% daily active ratio)" />
            <MathStep step="2" formula="Homepage loads/user/day = 5" result="5" note="Open app + multiple revisits + refresh" />
            <MathStep step="3" formula="Total feed requests/day = 800M x 5" result="4B/day" note="Homepage feed generation requests" />
            <MathStep step="4" formula="QPS = 4B / 86,400 ~ 4B / 100K" result="~46K QPS" note="Average. This is the rec system serving load." final />
            <MathStep step="5" formula="Peak QPS = 46K x 3" result="~140K QPS" note="Evening peak in each timezone" />
            <MathStep step="6" formula="Up Next requests/day (autoplay)" result="~10B/day" note="Each video watch triggers next rec. Higher volume than homepage." final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Corpus & Embedding Scale</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Total indexed videos" result="800M+" note="Not all videos are recommendation-eligible (policy, quality)" />
            <MathStep step="2" formula="Eligible for recs (active, quality-filtered)" result="~200M" note="Many videos are private, deleted, low-quality, or policy-violating" />
            <MathStep step="3" formula="Video embedding dim = 256 floats x 4B" result="1 KB/video" note="Two-tower model video embedding" />
            <MathStep step="4" formula="Video embedding index = 200M x 1KB" result="~200 GB" note="Fits in distributed memory for ANN retrieval" final />
            <MathStep step="5" formula="User embedding table = 2B x 256 floats x 4B" result="~2 TB" note="Must be sharded. Hot users cached." />
            <MathStep step="6" formula="New videos/day = 500 hrs/min x 1440 min" result="~720K hrs/day" note="Millions of new videos. Must index within hours." />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Model Serving Budget</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Candidate gen: retrieve 1000 videos" result="~30ms" note="ANN search + multiple source retrieval in parallel" />
            <MathStep step="2" formula="Ranking model: score 1000 candidates" result="~80ms" note="Deep NN on GPU. Batched inference: 1000 items in one pass." final />
            <MathStep step="3" formula="Feature lookup: 1000 candidates x 50 features" result="~20ms" note="Feature store batch read (Redis/Bigtable)" />
            <MathStep step="4" formula="Re-ranking + business logic" result="~15ms" note="Diversity, freshness, policy filtering — lightweight" />
            <MathStep step="5" formula="Total serving latency budget" result="<200ms" note="All stages pipelined. Critical path: candidate gen → features → rank." final />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Training Data Scale</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Watch events/day = 1B hrs / avg 7 min" result="~8.5B events/day" note="Each video watch (or impression) is a training example" />
            <MathStep step="2" formula="Engagement events (like, share, sub)" result="~500M/day" note="Sparse but high-signal labels" />
            <MathStep step="3" formula="Negative signals (skip, not interested)" result="~2B/day" note="Implicit negatives from impressions without clicks" />
            <MathStep step="4" formula="Training data retention" result="~30 days" note="Interests shift fast. Older data loses predictive power." final />
          </div>
        </Card>
      </div>
      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Feed QPS", val: "~46K", sub: "Homepage + Up Next: ~160K" },
            { label: "Eligible Videos", val: "~200M", sub: "Embedding index: ~200 GB" },
            { label: "Items Scored/req", val: "~1000", sub: "Per feed generation" },
            { label: "Latency Budget", val: "<200ms", sub: "End-to-end p50" },
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
          <Label color="#2563eb">Recommendation API</Label>
          <CodeBlock code={`# GET /v1/recommendations/homepage
# Returns: personalized video feed for user
#
# Headers:
#   Authorization: Bearer <token>
#   X-Device-Type: mobile|desktop|tv
#   X-Client-Language: en
#
# Query params:
#   page_token  - cursor for next page (opaque)
#   count       - videos per page (default 30)
#   surface     - homepage|watch_next|shorts
#
# Response:
{
  "videos": [
    {
      "video_id": "dQw4w9WgXcQ",
      "title": "How to Build a Recommendation System",
      "channel": { "id": "UCx123", "name": "ML Engineer" },
      "thumbnail_url": "https://img.../thumb.jpg",
      "duration_seconds": 847,
      "view_count": 1_240_000,
      "published_at": "2024-11-15T08:00:00Z",
      "reason": "Based on your interest in ML",
      "content_type": "long_form",
      "predicted_watch_pct": 0.72
    }
  ],
  "page_token": "eyJjdXJzb3IiOi4uLn0=",
  "request_id": "req_abc123"  // for logging
}

# POST /v1/feedback
# Real-time negative signals
{
  "video_id": "xyz789",
  "action": "not_interested",  // or "dont_recommend_channel"
  "request_id": "req_abc123"   // links to impression
}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Design Decisions (L6 Depth)</Label>
          <div className="space-y-3">
            {[
              { q: "Why return predicted_watch_pct?", a: "Transparency for debugging and for future client-side re-ranking. If the client needs to re-sort for layout changes (portrait vs landscape), it can use this score. Also powers the 'why is this recommended' explanation." },
              { q: "Why include request_id in response?", a: "Critical for closing the feedback loop. When user clicks, skips, or watches, the client sends request_id back. This links the impression → action, enabling training data construction. Without this, you can't attribute user behavior to a specific recommendation." },
              { q: "Why opaque page_token instead of offset?", a: "The recommendation set changes between requests (new videos uploaded, user watched something). Opaque tokens encode model state, cursor position, and diversification state. Offset pagination would cause duplicates or gaps." },
              { q: "Why separate surfaces (homepage vs watch_next)?", a: "Completely different models. Homepage: broad interest exploration, user just opened app, multiple content types. Watch_next: narrow continuation, user is mid-session, topically related. Different candidate generators, different ranking objectives, different latency budgets." },
              { q: "How does 'not_interested' propagate?", a: "Immediately: added to a real-time negative filter (Redis set). Within minutes: user embedding updated via streaming pipeline. Within hours: retraining data includes this negative signal. Three layers ensure the feedback is reflected at different timescales." },
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
        <Label color="#9333ea">Full System Architecture — Multi-Stage Recommendation Pipeline</Label>
        <svg viewBox="0 0 720 400" className="w-full">
          <defs><marker id="ah-hld" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>

          {/* User */}
          <rect x={10} y={70} width={70} height={36} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={45} y={86} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">User</text>
          <text x={45} y={98} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">opens app</text>

          {/* Candidate Generators */}
          <rect x={115} y={15} width={110} height={30} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={170} y={34} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Collaborative Filter</text>
          <rect x={115} y={52} width={110} height={30} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={170} y={71} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Two-Tower ANN</text>
          <rect x={115} y={89} width={110} height={30} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={170} y={108} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Content-Based</text>
          <rect x={115} y={126} width={110} height={30} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={170} y={145} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Trending / Fresh</text>
          <text x={170} y={170} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">~1000 candidates total</text>

          {/* Merge + Dedup */}
          <rect x={260} y={60} width={80} height={45} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={300} y={78} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Merge</text>
          <text x={300} y={92} textAnchor="middle" fill="#78716c80" fontSize="7" fontFamily="monospace">dedup + filter</text>

          {/* Feature Store Lookup */}
          <rect x={370} y={40} width={90} height={45} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={415} y={58} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Feature</text>
          <text x={415} y={72} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Enrichment</text>

          {/* Ranking */}
          <rect x={490} y={40} width={90} height={50} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={535} y={58} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Deep Ranking</text>
          <text x={535} y={72} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">multi-objective</text>
          <text x={535} y={83} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">1000 → 100</text>

          {/* Re-ranking */}
          <rect x={610} y={45} width={95} height={45} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={657} y={62} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Re-Rank</text>
          <text x={657} y={76} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">diversity + policy</text>

          {/* Data stores */}
          <rect x={260} y={190} width={90} height={35} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={305} y={211} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">User Profile Store</text>
          <rect x={380} y={190} width={90} height={35} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={425} y={211} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Feature Store</text>
          <rect x={500} y={190} width={90} height={35} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={545} y={211} textAnchor="middle" fill="#0891b2" fontSize="8" fontWeight="600" fontFamily="monospace">Video Metadata</text>
          <rect x={130} y={190} width={100} height={35} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={180} y={211} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Embedding Index</text>

          {/* Arrows */}
          <line x1={80} y1={85} x2={115} y2={68} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={225} y1={30} x2={260} y2={70} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={225} y1={67} x2={260} y2={75} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={225} y1={104} x2={260} y2={85} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={225} y1={141} x2={260} y2={95} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={340} y1={80} x2={370} y2={65} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={460} y1={62} x2={490} y2={62} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={580} y1={65} x2={610} y2={65} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>

          {/* Data connections */}
          <line x1={170} y1={156} x2={180} y2={190} stroke="#c026d340" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>
          <line x1={300} y1={105} x2={305} y2={190} stroke="#d9770640" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>
          <line x1={415} y1={85} x2={425} y2={190} stroke="#d9770640" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>
          <line x1={535} y1={90} x2={545} y2={190} stroke="#0891b240" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>

          {/* Legend */}
          <rect x={15} y={260} width={695} height={125} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={25} y={280} fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 1 — Candidate Generation (~30ms): Multiple generators in parallel. Each returns ~300 videos. Merged + deduped = ~1000.</text>
          <text x={25} y={297} fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 2 — Feature Enrichment (~20ms): Batch-fetch user features, video features, cross features from feature store.</text>
          <text x={25} y={314} fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 3 — Deep Ranking (~80ms): Score each candidate with multi-task NN. Predicts P(click), E[watch_time], P(like), P(dismiss).</text>
          <text x={25} y={331} fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 4 — Re-ranking (~15ms): Diversity (don't show 5 cooking videos in a row), freshness boost, policy demotion, ad slots.</text>
          <text x={25} y={348} fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Key: Candidate gen is RECALL-focused (don't miss good videos). Ranking is PRECISION-focused (order them correctly).</text>
          <text x={25} y={365} fill="#78716c" fontSize="8" fontWeight="600" fontFamily="monospace">YouTube's real system has ~10 candidate generators running in parallel, not 4. Each specializes in a different signal.</text>
        </svg>
      </Card>

      <Card>
        <Label color="#d97706">Latency Breakdown (p50)</Label>
        <div className="flex gap-2 items-end mt-2" style={{ height: 120 }}>
          {[
            { stage: "User\nProfile", ms: 10, color: "#d97706", pct: "6%" },
            { stage: "Collab\nFilter", ms: 25, color: "#c026d3", pct: "15%" },
            { stage: "Two-Tower\nANN", ms: 30, color: "#c026d3", pct: "18%" },
            { stage: "Trending\nFetch", ms: 15, color: "#c026d3", pct: "9%" },
            { stage: "Merge +\nDedup", ms: 5, color: "#78716c", pct: "3%" },
            { stage: "Feature\nEnrich", ms: 20, color: "#d97706", pct: "12%" },
            { stage: "Deep\nRanking", ms: 80, color: "#dc2626", pct: "47%" },
            { stage: "Re-Rank\n+ Policy", ms: 15, color: "#059669", pct: "9%" },
          ].map((s,i) => (
            <div key={i} className="flex-1 flex flex-col items-center gap-1">
              <div className="text-[9px] font-mono font-bold" style={{ color: s.color }}>{s.ms}ms</div>
              <div className="w-full rounded-t" style={{ height: `${s.ms * 1.1}px`, background: s.color + "20", border: `1px solid ${s.color}40` }}/>
              <div className="text-[8px] text-stone-500 text-center whitespace-pre-line leading-tight">{s.stage}</div>
            </div>
          ))}
        </div>
        <div className="mt-3 pt-2 border-t border-stone-100 text-[11px] text-stone-500">
          <strong className="text-stone-700">Total p50: ~160ms.</strong> Candidate generators run in parallel (max of 30ms, not sum). Deep ranking dominates — this is where ML budget goes. Candidate gen latency is hidden by parallelism.
        </div>
      </Card>
    </div>
  );
}

function CandidateSection() {
  const [sel, setSel] = useState("twotower");
  const generators = {
    twotower: { name: "Two-Tower (ANN) ★", cx: "Primary generator",
      desc: "Learn separate user and video embeddings. At serving time, encode user and find nearest video embeddings via ANN search. This is the core retrieval model at YouTube, Instagram, and TikTok.",
      code: `# Two-Tower Retrieval Model
# Offline: train two separate encoder towers

class TwoTowerModel:
    def __init__(self):
        self.user_tower = UserEncoder(
            # Input: user_id, watch_history, demographics
            # Output: 256-dim user embedding
            layers=[512, 256]
        )
        self.video_tower = VideoEncoder(
            # Input: video_id, title, category, channel
            # Output: 256-dim video embedding
            layers=[512, 256]
        )

    def train(self, user, positive_video, negative_videos):
        u_emb = self.user_tower(user)
        v_pos = self.video_tower(positive_video)
        v_negs = [self.video_tower(v) for v in negative_videos]

        # Sampled softmax loss (in-batch negatives)
        pos_score = dot(u_emb, v_pos)
        neg_scores = [dot(u_emb, v) for v in v_negs]
        loss = -log(exp(pos_score) / sum(exp(neg_scores)))
        return loss

# Offline: encode ALL eligible videos
video_index = ANN_Index(algo="ScaNN")  # Google's ANN library
for video in eligible_videos:  # 200M videos
    emb = video_tower.encode(video)
    video_index.add(video.id, emb)

# Online: retrieve candidates for user (~30ms)
def retrieve(user_context):
    u_emb = user_tower.encode(user_context)  # ~5ms
    candidates = video_index.search(u_emb, top_k=300)  # ~25ms
    return candidates` },
    collab: { name: "Collaborative Filtering", cx: "Co-watch patterns",
      desc: "Users who watched A also watched B. Simple, interpretable, and catches patterns that content-based models miss. Implemented as item-item similarity lookup.",
      code: `# Item-Item Collaborative Filtering
# Pre-computed offline, served from a lookup table

# Offline: build co-watch graph
build_cowatch_matrix(watch_logs):
    # For each pair of videos (A, B):
    #   cowatch[A][B] = number of users who watched both
    for user_session in watch_logs:
        videos = user_session.videos_watched
        for i, v_i in enumerate(videos):
            for j, v_j in enumerate(videos):
                if i != j:
                    cowatch[v_i][v_j] += 1

    # Normalize: cosine similarity
    for v_i in cowatch:
        for v_j in cowatch[v_i]:
            sim = cowatch[v_i][v_j] / sqrt(
                total_watches[v_i] * total_watches[v_j]
            )
            if sim > THRESHOLD:
                item_sim[v_i].append((v_j, sim))

    # Keep top 100 similar videos per video
    for v in item_sim:
        item_sim[v] = sorted(item_sim[v], key=lambda x: -x[1])[:100]

# Online: given user's recent watches, find similar videos
def retrieve_collab(user_watch_history, top_k=300):
    scores = Counter()
    for watched_video in user_watch_history[-50:]:
        for similar_video, sim in item_sim[watched_video]:
            if similar_video not in already_watched:
                scores[similar_video] += sim
    return scores.most_common(top_k)` },
    content: { name: "Content-Based", cx: "Topic/category matching",
      desc: "Recommend videos similar in content to what user watches. Uses video embeddings from title, description, tags, visual features. Helps with cold-start (new videos with no watch history).",
      code: `# Content-Based Retrieval
# Uses video content features, not user behavior

# Offline: build content embeddings for all videos
def build_content_embeddings():
    for video in all_videos:
        # Multi-modal content features
        text_emb = bert_encode(video.title + " " + video.description)
        visual_emb = resnet_encode(video.thumbnail)
        audio_emb = audio_classify(video.audio_sample)

        # Concatenate and project to 256-dim
        content_emb = projection_layer(
            concat(text_emb, visual_emb, audio_emb)
        )
        content_index.add(video.id, content_emb)

# Online: find videos similar to user's topic interests
def retrieve_content(user_topic_profile, top_k=200):
    # User topic profile = weighted average of content embeddings
    # of recently watched videos (exponential decay)
    profile_emb = compute_user_topic_profile(user_watch_history)
    candidates = content_index.search(profile_emb, top_k=top_k)
    return candidates

# Strengths: works for new users (based on few watches)
#            surfaces new videos (no watch history needed)
# Weakness: can create filter bubbles (only shows similar content)
#           misses serendipitous discoveries` },
    trending: { name: "Trending / Fresh", cx: "Recency + virality",
      desc: "Inject trending and recently uploaded videos. Without this, the system only shows well-established content and new creators can never break through.",
      code: `# Trending & Freshness Candidate Generator
# Non-personalized but essential for content freshness

def retrieve_trending(user_context, top_k=200):
    candidates = []

    # 1. Globally trending (viral videos)
    #    Score = engagement_velocity * quality_score
    trending = trending_service.get_global(
        country=user_context.country,
        language=user_context.language,
        top_k=50
    )
    candidates.extend(trending)

    # 2. Topic-trending (trending within user's interest)
    for topic in user_context.top_topics[:5]:
        topic_trending = trending_service.get_by_topic(
            topic=topic,
            recency="24h",
            top_k=20
        )
        candidates.extend(topic_trending)

    # 3. Fresh uploads from subscribed channels
    subscribed_new = subscription_service.get_recent(
        user_id=user_context.user_id,
        since="48h",
        top_k=50
    )
    candidates.extend(subscribed_new)

    # 4. Explore: random sample of quality new videos
    #    Gives new creators a chance to be discovered
    explore = quality_pool.sample(
        quality_threshold=0.7,
        recency="7d",
        top_k=30
    )
    candidates.extend(explore)

    return candidates[:top_k]` },
  };
  const g = generators[sel];
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Candidate Generator Comparison</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[12px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Generator</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Signal</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Latency</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Cold Start</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Diversity</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Candidates</th>
            </tr></thead>
            <tbody>
              {[
                { n:"Two-Tower ANN ★", sig:"Learned embeddings", l:"30ms", cs:"Needs history", d:"Medium", c:"300", hl:true },
                { n:"Collaborative Filter", sig:"Co-watch graph", l:"15ms", cs:"Needs history", d:"Low", c:"300", hl:false },
                { n:"Content-Based", sig:"Content similarity", l:"20ms", cs:"Works (few watches)", d:"Low", c:"200", hl:false },
                { n:"Trending/Fresh", sig:"Velocity + recency", l:"10ms", cs:"Works (no history)", d:"High", c:"200", hl:false },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-fuchsia-50" : ""}>
                  <td className={`px-3 py-2 font-mono ${r.hl?"text-fuchsia-700 font-bold":"text-stone-600"}`}>{r.n}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.sig}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.l}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.cs}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.d}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.c}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-3 space-y-2">
          {Object.entries(generators).map(([k,v]) => (
            <button key={k} onClick={() => setSel(k)}
              className={`w-full text-left px-3.5 py-2.5 rounded-lg text-[12px] font-medium border transition-all ${k===sel?"bg-fuchsia-600 text-white border-fuchsia-600":"bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
              {v.name}
            </button>
          ))}
        </div>
        <div className="col-span-9 space-y-5">
          <Card>
            <div className="flex items-center gap-3 mb-2">
              <span className="text-[14px] font-bold text-stone-800">{g.name}</span>
              <Pill bg="#fdf4ff" color="#c026d3">{g.cx}</Pill>
            </div>
            <p className="text-[12px] text-stone-500">{g.desc}</p>
          </Card>
          <CodeBlock title={`${g.name} — Implementation`} code={g.code} />
        </div>
      </div>
    </div>
  );
}

function RankingSection() {
  return (
    <div className="space-y-5">
      <Card accent="#dc2626">
        <Label color="#dc2626">Deep Ranking Model Architecture</Label>
        <p className="text-[12px] text-stone-500 mb-4">The ranking model is a multi-task deep neural network that predicts multiple user actions simultaneously. This is the core ML component — it determines what users see. Based on YouTube's published Deep Neural Networks for YouTube Recommendations paper + subsequent work.</p>
        <div className="grid grid-cols-2 gap-5">
          <div>
            <CodeBlock title="Ranking Model — Multi-Task Architecture" code={`# Multi-Task Ranking Model
class YouTubeRanker(nn.Module):
    def __init__(self):
        # Shared bottom layers
        self.shared = nn.Sequential(
            nn.Linear(FEATURE_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Task-specific towers (Multi-gate MoE)
        self.click_tower = TaskTower(256, 1)
        self.watch_tower = TaskTower(256, 1)
        self.like_tower = TaskTower(256, 1)
        self.dismiss_tower = TaskTower(256, 1)

    def forward(self, features):
        shared_repr = self.shared(features)
        return {
            "p_click": sigmoid(self.click_tower(shared_repr)),
            "e_watch_time": relu(self.watch_tower(shared_repr)),
            "p_like": sigmoid(self.like_tower(shared_repr)),
            "p_dismiss": sigmoid(self.dismiss_tower(shared_repr)),
        }

# Combined scoring function
def compute_rank_score(preds):
    score = (
        preds["p_click"]
        * preds["e_watch_time"]     # Expected watch time
        * (1 + w_like * preds["p_like"])
        * (1 - w_dismiss * preds["p_dismiss"])
    )
    return score`} />
          </div>
          <div className="space-y-4">
            <Card accent="#9333ea">
              <Label color="#9333ea">Why Multi-Task?</Label>
              <div className="space-y-2">
                {[
                  { q: "Why not just predict P(click)?", a: "Clicks alone optimizes for clickbait. A video with a sensational thumbnail gets clicks but users watch 5 seconds and leave. P(click) * E[watch_time] captures actual engagement." },
                  { q: "Why predict P(dismiss)?", a: "A penalty signal. If users frequently hit 'not interested' on a type of video, demote it. Without this, the model keeps recommending things users don't want but accidentally click." },
                  { q: "Why shared bottom layers?", a: "Transfer learning across tasks. Click, watch, like all share common relevance signals. Shared layers learn general user-video affinity; task towers specialize for each action." },
                  { q: "Why not a single combined label?", a: "Different tasks have different data volumes. Clicks: billions/day. Likes: millions/day. Single label would be dominated by click signal. Multi-task lets rare signals (likes) have their own gradient path." },
                ].map((d,i) => (
                  <div key={i}>
                    <div className="text-[11px] font-bold text-stone-700">{d.q}</div>
                    <div className="text-[10px] text-stone-500 mt-0.5">{d.a}</div>
                  </div>
                ))}
              </div>
            </Card>
            <Card>
              <Label color="#d97706">Scoring Formula Deep Dive</Label>
              <div className="bg-stone-50 rounded-lg p-3 font-mono text-[11px] text-stone-700 leading-loose">
                <div>score = P(click) × E[watch_time]</div>
                <div className="ml-8">× (1 + 0.1 × P(like))</div>
                <div className="ml-8">× (1 - 0.3 × P(dismiss))</div>
              </div>
              <div className="mt-2 space-y-1 text-[10px] text-stone-500">
                <div>• <strong>P(click) × E[watch_time]</strong> = expected watch time if shown. The primary objective.</div>
                <div>• <strong>P(like) boost</strong> = small bonus for satisfying content. Weight 0.1 keeps it as a tiebreaker.</div>
                <div>• <strong>P(dismiss) penalty</strong> = strong demotion. Weight 0.3 because false dismissals are costly to user experience.</div>
                <div>• Weights are tuned via A/B tests on user satisfaction surveys.</div>
              </div>
            </Card>
          </div>
        </div>
      </Card>
    </div>
  );
}

function FeaturesSection() {
  return (
    <div className="space-y-5">
      <Card accent="#d97706">
        <Label color="#d97706">Feature Categories for Video Recommendation</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[11px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Category</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Example Features</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Computed</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Impact</th>
            </tr></thead>
            <tbody>
              {[
                { cat: "User Watch History", ex: "last_50_watched_video_ids, watch_time_per_video, topics_watched_7d", comp: "Online (profile)", impact: "Very High" },
                { cat: "User Demographics", ex: "country, language, age_bucket, device_type, time_of_day", comp: "Online (request)", impact: "Medium" },
                { cat: "User Engagement", ex: "avg_session_length, daily_active_days_30d, subscription_count", comp: "Batch (daily)", impact: "High" },
                { cat: "Video Static", ex: "category, channel_id, duration, upload_date, language, title_embedding", comp: "Batch (index)", impact: "High" },
                { cat: "Video Engagement", ex: "total_views, CTR_7d, avg_watch_pct, like_ratio, avg_dwell_time", comp: "Near-real-time", impact: "Very High" },
                { cat: "Cross Features", ex: "user_channel_affinity, user_topic_overlap, user_watched_channel_before", comp: "Online (join)", impact: "Very High" },
                { cat: "Context", ex: "day_of_week, hour_of_day, is_weekend, device_screen_size, network_speed", comp: "Online (request)", impact: "Medium" },
                { cat: "Freshness", ex: "hours_since_upload, is_trending, engagement_velocity", comp: "Near-real-time", impact: "High" },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-bold text-stone-700">{r.cat}</td>
                  <td className="px-3 py-2 text-stone-500 font-mono text-[10px]">{r.ex}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.comp}</td>
                  <td className="text-center px-3 py-2"><Pill bg={r.impact.includes("Very")?"#fef2f2":r.impact==="High"?"#fffbeb":"#f0fdf4"} color={r.impact.includes("Very")?"#dc2626":r.impact==="High"?"#d97706":"#059669"}>{r.impact}</Pill></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Critical Feature: Watch History Encoding</Label>
          <CodeBlock code={`# How to encode variable-length watch history
# This is THE most impactful feature for recommendations

# Option 1: Average pooling (simple, baseline)
def encode_history_avg(watch_history):
    embs = [video_embedding[v.id] for v in watch_history[-50:]]
    return mean(embs)  # 256-dim

# Option 2: Weighted average (better — recency matters)
def encode_history_weighted(watch_history):
    embs = []
    for i, v in enumerate(watch_history[-50:]):
        weight = exp(-0.1 * (len(watch_history) - i))  # exp decay
        embs.append(weight * video_embedding[v.id])
    return sum(embs) / sum(weights)

# Option 3: Attention-based (best — learns what matters)
def encode_history_attention(watch_history, target_video=None):
    # Self-attention over watch history
    # If target_video provided: cross-attention
    # "Which past watches are relevant to THIS candidate?"
    history_embs = stack([video_emb[v.id] for v in watch_history])
    attended = multi_head_attention(
        query=target_video_emb if target_video else learnable_query,
        key=history_embs,
        value=history_embs,
    )
    return attended  # 256-dim

# YouTube uses Option 3 for ranking (expensive but accurate)
# and Option 2 for candidate gen (must be fast)`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Feature Store Architecture</Label>
          <CodeBlock code={`# Feature serving for recommendations
# Three latency tiers with different update frequencies

# TIER 1: Real-time features (<1 sec freshness)
# Stored in Redis, updated by streaming pipeline
realtime_features = {
    "user_session": {
        "videos_watched_this_session": ["v1", "v2"],
        "current_topic_interest": "cooking",
        "session_start_time": "2024-02-10T14:00:00Z",
    },
    "video_realtime": {
        "views_last_hour": 45000,
        "engagement_velocity": 2.3,  # vs baseline
    }
}

# TIER 2: Near-real-time features (minutes freshness)
# Updated by mini-batch pipeline (Flink/Dataflow)
nearrt_features = {
    "video_engagement": {
        "ctr_24h": 0.045,
        "avg_watch_pct_24h": 0.62,
        "like_ratio_7d": 0.04,
    }
}

# TIER 3: Batch features (daily freshness)
# Updated by daily batch pipeline (Spark/BigQuery)
batch_features = {
    "user_profile": {
        "topic_affinities": {"cooking": 0.8, "tech": 0.6},
        "avg_session_length": 25.5,  # minutes
        "active_days_30d": 22,
    },
    "video_static": {
        "category": "education",
        "channel_subscriber_count": 1_200_000,
        "content_quality_score": 0.85,
    }
}

# At serving time: parallel batch-read all three tiers
# Total feature lookup: ~20ms for 1000 candidates`} />
        </Card>
      </div>
    </div>
  );
}

function ObjectivesSection() {
  return (
    <div className="space-y-5">
      <Card accent="#0f766e">
        <Label color="#0f766e">Multi-Objective Optimization — The Core L6 Challenge</Label>
        <p className="text-[12px] text-stone-500 mb-4">This is where L6 candidates differentiate themselves. Optimizing for a single metric (e.g., watch time) is straightforward but harmful. The real challenge is balancing multiple competing objectives.</p>
        <div className="grid grid-cols-3 gap-4">
          {[
            { obj: "Engagement", metrics: "Watch time, clicks, session length", tradeoff: "Maximizing alone → clickbait, autoplay rabbit holes, addictive loops", color: "#dc2626" },
            { obj: "Satisfaction", metrics: "Likes, shares, survey ratings, not-interested rate", tradeoff: "Maximizing alone → only safe/popular content, no discovery", color: "#059669" },
            { obj: "Responsibility", metrics: "Misinformation exposure, borderline content views, minor safety", tradeoff: "Maximizing alone → over-censorship, reduced utility", color: "#2563eb" },
          ].map((o,i) => (
            <div key={i} className="rounded-lg border p-4" style={{ borderTop: `3px solid ${o.color}` }}>
              <div className="text-[12px] font-bold mb-2" style={{ color: o.color }}>{o.obj}</div>
              <div className="text-[10px] text-stone-500 mb-2"><strong>Metrics:</strong> {o.metrics}</div>
              <div className="text-[10px] text-red-500"><strong>Risk if maximized alone:</strong> {o.tradeoff}</div>
            </div>
          ))}
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7e22ce">
          <Label color="#7e22ce">Combining Objectives — Approaches</Label>
          <div className="space-y-3">
            {[
              { name: "Scalarization (Weighted Sum) ★", desc: "score = w1*engagement + w2*satisfaction - w3*responsibility_risk. Simple, interpretable, tunable. Weights set by leadership + A/B tested.", pros: "Easy to implement, easy to tune", cons: "Linear combination may miss Pareto-optimal solutions" },
              { name: "Constrained Optimization", desc: "Maximize engagement SUBJECT TO satisfaction > threshold AND responsibility_risk < limit. Treats some objectives as constraints, not variables.", pros: "Hard guarantees on safety/satisfaction", cons: "Feasibility issues — may have no solution that satisfies all constraints" },
              { name: "Multi-Task Learning (MMoE)", desc: "Train one model with multiple heads — each head predicts a different objective. Shared bottom layers learn universal representations. Combine scores at serving time.", pros: "Efficient (one forward pass), transfer learning across tasks", cons: "Task conflicts — improving one head may hurt another" },
              { name: "Pareto-Optimal Re-ranking", desc: "Rank by engagement first, then re-rank to push diverse/satisfying content up without dropping engagement more than X%. Post-hoc adjustment.", pros: "Preserves most engagement, adds diversity", cons: "Greedy — may not find globally optimal mix" },
            ].map((a,i) => (
              <div key={i} className="rounded-lg border border-stone-200 p-3">
                <div className="text-[11px] font-bold text-stone-800">{a.name}</div>
                <p className="text-[10px] text-stone-500 mt-0.5">{a.desc}</p>
                <div className="flex gap-4 mt-1 text-[10px]">
                  <span className="text-emerald-600">✓ {a.pros}</span>
                  <span className="text-red-500">✗ {a.cons}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#0f766e">
          <Label color="#0f766e">Diversity & Freshness Rules (Re-ranking)</Label>
          <CodeBlock code={`# Re-ranking for diversity and business logic
# Applied AFTER deep ranking, BEFORE serving

def rerank(ranked_videos, user_context):
    final_feed = []
    seen_channels = set()
    seen_topics = Counter()
    last_content_type = None

    for video in ranked_videos:
        # Rule 1: Channel diversity
        # Max 2 videos per channel in top 30
        if seen_channels.count(video.channel) >= 2:
            continue

        # Rule 2: Topic diversity
        # Max 5 videos per topic in top 30
        if seen_topics[video.topic] >= 5:
            continue

        # Rule 3: Content type interleaving
        # Don't show 3 Shorts in a row
        if video.type == last_content_type == "short":
            continue

        # Rule 4: Freshness injection
        # At least 20% of feed should be < 24hrs old
        fresh_count = sum(1 for v in final_feed
                         if v.age < hours(24))
        if len(final_feed) > 10 and fresh_count / len(final_feed) < 0.2:
            # Promote next fresh video
            boost_fresh(video)

        # Rule 5: Policy demotion
        # Borderline content demoted (not removed)
        if video.borderline_score > 0.7:
            video.score *= 0.3  # heavy demotion

        final_feed.append(video)
        seen_channels.add(video.channel)
        seen_topics[video.topic] += 1
        last_content_type = video.type

        if len(final_feed) >= 30:
            break

    return final_feed`} />
        </Card>
      </div>
    </div>
  );
}

function DataModelSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Core Data Stores</Label>
          <CodeBlock code={`-- User Profile Store (Bigtable, sharded by user_id)
user_id -> {
  demographics: {country, language, age_bucket},
  watch_history: [(video_id, watch_pct, ts), ...], # last 1000
  topic_affinities: {topic: score, ...},
  subscriptions: [channel_id, ...],
  negative_feedback: [(video_id, action, ts), ...],
  user_embedding: float[256],  # updated hourly
}

-- Video Metadata (Bigtable, sharded by video_id)
video_id -> {
  title, description, category, channel_id,
  duration, upload_ts, language,
  view_count, like_count, comment_count,
  content_quality_score: float,  # ML-scored
  safety_score: float,           # content policy
  title_embedding: float[256],
  visual_embedding: float[256],
}

-- Video Engagement Stats (Redis, near-real-time)
video_id -> {
  views_1h, views_24h,
  ctr_7d, avg_watch_pct_7d,
  engagement_velocity: float,
  trending_score: float,
}

-- Embedding Index (ScaNN/FAISS, in-memory)
video_id -> embedding[256]  # for ANN retrieval
# 200M videos x 256 x 4B = ~200 GB, sharded

-- Interaction Logs (BigQuery, append-only)
(ts, user_id, video_id, action, watch_time_sec,
 surface, request_id, position, device)`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Why These Design Choices?</Label>
          <div className="space-y-3">
            {[
              { q: "Why Bigtable for user profiles, not MySQL?", a: "User profiles are wide (1000-element watch history), accessed by single key (user_id), and updated very frequently (every watch event). Bigtable excels at this: wide-column, single-row reads, high write throughput. MySQL would need complex schema + sharding for 2B users." },
              { q: "Why Redis for engagement stats?", a: "These change every minute (views, CTR). Batch pipeline can't keep up. Redis pub/sub + Flink streaming pipeline updates counters in real-time. TTL-based expiration for time-windowed stats (views_1h auto-expires)." },
              { q: "Why separate embedding index from metadata?", a: "Different access patterns. Embedding index is memory-resident for ANN search (ScaNN). Metadata is disk-based for feature lookup. Embedding index is rebuilt daily; metadata is updated per-event. Different scaling characteristics." },
              { q: "Why store watch_history in user profile, not a separate table?", a: "Watch history is ALWAYS read with user profile (every recommendation request). Co-locating avoids a second network hop. Bigtable row can hold 100MB+ — 1000 watch records is tiny. Trade-off: updating history requires row-level mutation, but Bigtable handles this well." },
              { q: "Why BigQuery for interaction logs?", a: "Append-only, massive volume (8.5B/day), used for batch training data construction. BigQuery handles petabyte-scale analytics with SQL. Training pipeline reads 30 days of logs per run. Cost-effective storage + compute separation." },
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
        <svg viewBox="0 0 700 150" className="w-full">
          <defs><marker id="ah-tp" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          {[
            { x: 55, y: 50, w: 90, label: "Watch Logs\n+ Actions", color: "#d97706" },
            { x: 175, y: 50, w: 80, label: "Label\nConstruction", color: "#dc2626" },
            { x: 285, y: 50, w: 80, label: "Feature\nJoining", color: "#0891b2" },
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
          <path d="M 615 72 L 615 105 L 55 105 L 55 72" fill="none" stroke="#dc2626" strokeWidth={1} strokeDasharray="4,3" markerEnd="url(#ah-tp)"/>
          <text x={335} y={120} textAnchor="middle" fill="#dc2626" fontSize="8" fontFamily="monospace">Feedback loop: engagement data from A/B test feeds into next training cycle</text>
          <text x={175} y={140} fill="#78716c" fontSize="7" fontFamily="monospace">Candidate gen model: retrained daily</text>
          <text x={450} y={140} fill="#78716c" fontSize="7" fontFamily="monospace">Ranking model: retrained daily, fine-tuned continuously</text>
        </svg>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Label Construction (Critical)</Label>
          <CodeBlock code={`# Constructing training labels from implicit feedback
# This is where most rec systems get it wrong

def construct_labels(interaction_logs):
    examples = []
    for impression in interaction_logs:
        video = impression.video_id
        user = impression.user_id

        # POSITIVE: user engaged meaningfully
        if impression.clicked:
            watch_pct = impression.watch_time / video.duration
            labels = {
                "click": 1,
                "watch_time": impression.watch_time,
                "watch_pct": watch_pct,
                "liked": impression.liked,
                "shared": impression.shared,
                "subscribed_after": impression.subscribed_after,
                # Satisfaction proxy: did they complete the video?
                "satisfied": watch_pct > 0.7 and not impression.disliked,
            }
        else:
            # NEGATIVE: shown but not clicked
            labels = {
                "click": 0,
                "watch_time": 0,
                "watch_pct": 0,
                "liked": False,
                "satisfied": False,
            }

        # IMPORTANT: log features AT IMPRESSION TIME
        # Not re-computed — prevents training-serving skew
        features = impression.logged_features

        # Position de-biasing
        labels["position"] = impression.position
        labels["propensity"] = position_propensity[impression.position]

        examples.append((user, video, features, labels))

    return examples

# Key: use LOGGED features, not re-computed features
# Key: weight by inverse propensity for position bias`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Negative Sampling Strategy</Label>
          <div className="space-y-3">
            {[
              { name: "In-Batch Negatives", desc: "Other videos in the same training batch serve as negatives. Simple and effective. But biased toward popular videos (appear in more batches).", used: "Two-tower candidate gen model" },
              { name: "Impressed-But-Not-Clicked", desc: "Videos shown to user but not clicked. True negatives — user saw and rejected them. Strongest signal but has position bias (user may not have scrolled that far).", used: "Ranking model (primary negative)" },
              { name: "Random Negatives", desc: "Random videos from the corpus. Easy to generate. But most random videos are so irrelevant they're trivial negatives — model doesn't learn much.", used: "Candidate gen (supplementary)" },
              { name: "Hard Negatives", desc: "Videos that are similar to positives but not engaged with. Found by: mining near-miss clicks, same-channel-different-video. Most informative for model learning.", used: "Ranking model (curriculum learning)" },
            ].map((n,i) => (
              <div key={i} className="rounded-lg border border-stone-200 p-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[11px] font-bold text-stone-800">{n.name}</span>
                  <Pill bg="#ecfdf5" color="#059669">{n.used}</Pill>
                </div>
                <p className="text-[10px] text-stone-500">{n.desc}</p>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function ColdStartSection() {
  return (
    <div className="space-y-5">
      <Card accent="#0284c7">
        <Label color="#0284c7">Cold Start Problem — Users and Videos</Label>
        <p className="text-[12px] text-stone-500 mb-4">Cold start is a critical interview topic. There are two sides: new users (no watch history) and new videos (no engagement data). Each requires different strategies.</p>
        <div className="grid grid-cols-2 gap-5">
          <div className="rounded-lg border border-stone-200 p-4" style={{ borderTop: "3px solid #0284c7" }}>
            <div className="text-[12px] font-bold text-cyan-700 mb-3">New User Cold Start</div>
            <div className="space-y-2">
              {[
                { stage: "No history at all", strategy: "Serve popularity-based recs. Country + language + device type → pre-computed popular feeds. Not personalized but functional.", icon: "1" },
                { stage: "1-3 watches", strategy: "Content-based: find videos similar to what they watched. Trending in same category. Start building topic affinity profile.", icon: "2" },
                { stage: "10-50 watches", strategy: "Collaborative filtering kicks in. Enough history to find similar users. Two-tower model starts producing personalized embeddings.", icon: "3" },
                { stage: "100+ watches", strategy: "Full personalization. All candidate generators active. Watch history attention works well. User is no longer cold.", icon: "4" },
              ].map((s,i) => (
                <div key={i} className="flex items-start gap-2">
                  <span className="text-[10px] font-bold w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5 bg-cyan-100 text-cyan-700">{s.icon}</span>
                  <div>
                    <div className="text-[11px] font-bold text-stone-700">{s.stage}</div>
                    <div className="text-[10px] text-stone-500">{s.strategy}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="rounded-lg border border-stone-200 p-4" style={{ borderTop: "3px solid #7e22ce" }}>
            <div className="text-[12px] font-bold text-purple-700 mb-3">New Video Cold Start</div>
            <div className="space-y-2">
              {[
                { stage: "Just uploaded", strategy: "Content features only: title, description, thumbnail, channel history. Channel's past performance is a strong prior. Classify into topic/quality tier.", icon: "1" },
                { stage: "0-1K views", strategy: "Show to subscribers of the channel first (guaranteed interested audience). Measure early engagement signals: CTR, watch %, like ratio.", icon: "2" },
                { stage: "1K-100K views", strategy: "Early engagement signals enable the ranking model. If CTR and watch % are high, the system starts recommending to broader audience (explore).", icon: "3" },
                { stage: "100K+ views", strategy: "Rich engagement data. Collaborative filtering signals emerge (co-watch patterns). Video is now fully in the recommendation ecosystem.", icon: "4" },
              ].map((s,i) => (
                <div key={i} className="flex items-start gap-2">
                  <span className="text-[10px] font-bold w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5 bg-purple-100 text-purple-700">{s.icon}</span>
                  <div>
                    <div className="text-[11px] font-bold text-stone-700">{s.stage}</div>
                    <div className="text-[10px] text-stone-500">{s.strategy}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>
      <Card accent="#d97706">
        <Label color="#d97706">Exploration vs Exploitation (Contextual Bandits)</Label>
        <CodeBlock code={`# Explore/Exploit for new content discovery
# Without exploration, only popular content gets shown → rich get richer

class EpsilonGreedyExplorer:
    def inject_exploration(self, ranked_feed, explore_pool, epsilon=0.05):
        """Replace epsilon fraction of feed with explore candidates"""
        final_feed = []
        for i, video in enumerate(ranked_feed):
            if random() < epsilon:
                # Replace with exploration candidate
                explore_video = explore_pool.sample(
                    quality_threshold=0.6,  # don't show garbage
                    age_max_days=7,         # prioritize new content
                )
                explore_video.is_explore = True  # track for analysis
                final_feed.append(explore_video)
            else:
                final_feed.append(video)
        return final_feed

# More sophisticated: Thompson Sampling
# Maintain posterior distribution of each video's engagement rate
# Sample from posterior → naturally explores uncertain videos
class ThompsonSamplingExplorer:
    def score_with_exploration(self, video):
        # Prior: Beta(alpha, beta) based on content features
        alpha = video.clicks + prior_alpha
        beta = video.impressions - video.clicks + prior_beta
        # Sample from posterior
        sampled_ctr = np.random.beta(alpha, beta)
        return sampled_ctr * video.predicted_watch_time

# Thompson Sampling naturally balances:
# - High-confidence good videos → narrow distribution, high mean
# - Uncertain new videos → wide distribution, occasionally sampled high
# - High-confidence bad videos → narrow distribution, low mean`} />
      </Card>
    </div>
  );
}

function ScalabilitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Candidate Generation Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">ANN index sharding</strong> — 200M video embeddings sharded across ~50 machines. Each shard holds ~4M vectors in memory. Query broadcast to all shards, top-K results merged. Total: ~200GB.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">ScaNN quantization</strong> — Google's ANN library. Product quantization reduces 256 floats (1KB) to 32 bytes per vector. 8x memory savings. Negligible recall loss with reranking.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Pre-computed user embeddings</strong> — for non-real-time features, pre-compute user embeddings in batch (hourly). Cache in Redis. Saves GPU inference at serving time. Only update in real-time for session features.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Candidate cache</strong> — for repeat visits within short time window, serve cached candidates with re-ranking. Avoid regenerating full candidate set if user revisits within 5 minutes.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Ranking Model Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Batched GPU inference</strong> — 1000 candidates in one forward pass. Dynamic batching: accumulate requests for 3ms, then fire. GPU utilization: 60-80% with dynamic batching vs 20% without.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Model distillation</strong> — teacher model (deep, slow) trains a student model (shallow, fast). Student: 3-layer MLP instead of 6-layer. 2x faster, 97% of teacher quality. Used for serving.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Feature pre-computation</strong> — expensive cross-features (user-video interactions) pre-computed and cached. Reduces online compute from O(users × videos) to O(1) lookups.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Two-pass ranking</strong> — L1: lightweight model scores 1000 → 200 (CPU). L2: heavy model scores 200 → 30 (GPU). Reduces GPU cost by 5x while keeping top-30 quality.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Architecture</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Per-Region Serving ★", d:"Full recommendation stack replicated per region. User embeddings homed to nearest region. Video index globally replicated (same corpus). Models replicated.", pros:["Lowest latency (no cross-region)","Region-autonomous","Natural data locality"], cons:["Model sync across regions","User travel = cold start in new region","Expensive replication"], pick:true },
            { t:"Central Ranking", d:"Candidate gen local, ranking done centrally. Candidates sent to central region for scoring.", pros:["Single model to manage","Consistent quality globally","Simpler deployment"], cons:["Cross-region latency (+50-100ms)","Central bottleneck","Single point of failure"], pick:false },
            { t:"Federated Learning", d:"Train regional models on local data, aggregate centrally. Each region has specialized model tuned to local preferences.", pros:["Data locality (privacy)","Region-specific tuning","No raw data transfer"], cons:["Complex training infra","Model divergence risk","Slower convergence"], pick:false },
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

function WatchoutsSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        {[
          { title: "Popularity Bias (Rich Get Richer)", sev: "CRITICAL", desc: "Popular videos have more engagement data → model learns to recommend them → they get even more views → feedback loop. New creators and niche content can never break through.", fix: "Exploration budget (5% of impressions). Inverse propensity weighting on popularity. Freshness features. Creator equity analysis dashboards.", icon: "🔴" },
          { title: "Filter Bubbles", sev: "CRITICAL", desc: "User watches cooking videos → system only recommends cooking → user never discovers music, science, or other interests. Limits user growth and platform stickiness.", fix: "Topic diversity rules in re-ranking. Serendipity injection (random quality content from outside user's bubble). Monitor topic entropy per user over time.", icon: "🔴" },
          { title: "Training-Serving Skew", sev: "HIGH", desc: "Features computed differently in training (batch) vs serving (online). Example: video view count at training time vs serving time are days apart. Model sees different feature distributions.", fix: "Log features AT SERVING TIME and join with outcomes for training. Never re-compute features for training data. Feature monitoring: alert on distribution drift between logged vs batch features.", icon: "🟡" },
          { title: "Position Bias in Clicks", sev: "HIGH", desc: "Top-of-feed videos get more clicks regardless of quality. If you train on raw clicks, model learns to predict position, not relevance. Self-reinforcing: model puts video at top → gets clicks → trains to put it at top.", fix: "Inverse propensity scoring. Randomized position experiments (shuffle top-20 for 1% of traffic). Use watch time (post-click) rather than click alone.", icon: "🟡" },
          { title: "Stale User Embeddings", sev: "MEDIUM", desc: "User's interests shift within a session (opened for cooking, now browsing music). Pre-computed user embedding reflects yesterday's interests, not current session context.", fix: "Real-time session features: last 5 videos watched in this session. Two-level profile: batch (long-term) + real-time (session). Session-aware candidate generator.", icon: "🟠" },
          { title: "Engagement Trap (Autoplay)", sev: "MEDIUM", desc: "Autoplay artificially inflates watch time. User falls asleep with autoplay on → model thinks they love the content. Optimizing for watch time rewards passive viewing over active choice.", fix: "Distinguish active vs passive engagement. Weight active actions (click, like) higher than autoplay. Track 'regret' signals: unsubscribe after binge, low satisfaction surveys.", icon: "🟠" },
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

function ObservabilitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-5">
        <Card accent="#0284c7">
          <Label color="#0284c7">Engagement Metrics</Label>
          <div className="space-y-2.5">
            {[
              { metric: "Watch Time per DAU", target: ">60 min/day", why: "Primary business metric. Total platform engagement." },
              { metric: "Homepage CTR", target: ">4%", why: "Are we showing relevant thumbnails? Low CTR = bad candidates." },
              { metric: "Avg Watch Percentage", target: ">55%", why: "Are users finishing videos? Low = clickbait / bad ranking." },
              { metric: "Session Length", target: ">25 min", why: "Are users staying on platform? Multi-video engagement." },
              { metric: "Videos per Session", target: ">8", why: "Feed quality — users should want to watch more." },
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
        <Card accent="#059669">
          <Label color="#059669">Satisfaction Metrics</Label>
          <div className="space-y-2.5">
            {[
              { metric: "Like / Dislike Ratio", target: ">20:1", why: "Direct quality signal. Spikes in dislikes = bad recs." },
              { metric: "Not Interested Rate", target: "<2%", why: "Users actively dismissing recs. High = model failure." },
              { metric: "Survey Satisfaction Score", target: ">4.0/5", why: "Sampled user surveys. Ground truth satisfaction." },
              { metric: "Reformulation Rate", target: "<10%", why: "User changes from homepage to search = rec failure." },
              { metric: "DAU 7-day Retention", target: ">75%", why: "Long-term health. Bad recs → users leave platform." },
            ].map((m,i) => (
              <div key={i} className="border-b border-stone-100 pb-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold text-stone-700">{m.metric}</span>
                  <span className="text-[10px] font-mono text-emerald-700">{m.target}</span>
                </div>
                <div className="text-[10px] text-stone-400 mt-0.5">{m.why}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">System Health & Alerts</Label>
          <div className="space-y-2.5">
            {[
              { alert: "Serving latency p99 > 500ms", sev: "P0", action: "Shed ranking model load, serve L1-only" },
              { alert: "CTR drops > 10% day-over-day", sev: "P0", action: "Auto-rollback model, investigate" },
              { alert: "Candidate gen returns < 100 videos", sev: "P1", action: "Check embedding index health" },
              { alert: "Feature store latency > 50ms", sev: "P1", action: "Scale feature store, check cache hit rate" },
              { alert: "New video indexing lag > 6 hours", sev: "P2", action: "Check embedding pipeline backlog" },
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
      </div>
    </div>
  );
}

function EnhancementsSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        {[
          { title: "LLM-Powered Understanding", d: "Use LLMs to understand video content (transcript analysis), generate better titles/descriptions for embedding, and power conversational recommendations ('show me something like X but shorter').", effort: "Hard", detail: "Cost: LLM inference per video at upload time. Benefit: richer content features, especially for videos with poor metadata." },
          { title: "Cross-Surface Optimization", d: "Jointly optimize Homepage, Watch Next, Shorts, and Notifications. A video shown on homepage shouldn't also be pushed as a notification. Unified frequency capping and holistic engagement modeling.", effort: "Hard", detail: "Challenge: each surface has its own latency budget, candidate pool, and ranking model. Need a central coordination layer." },
          { title: "Creator-Side Optimization", d: "Recommend upload times, topics, thumbnails to creators based on audience patterns. Healthy creator ecosystem = healthy content supply = better recommendations.", effort: "Medium", detail: "YouTube Studio analytics already does this partially. ML-powered suggestions could significantly improve creator success rates." },
          { title: "Multi-Modal Embeddings", d: "Combine audio, visual, transcript, and metadata into a unified video embedding. Enables: find videos that 'feel' similar (mood, energy, style), not just topically similar.", effort: "Hard", detail: "Requires pre-processing all videos with audio/visual models. Storage and compute intensive but dramatically improves content understanding." },
          { title: "Causal Impact Estimation", d: "Measure the causal effect of recommendations, not just correlation. Did showing video X cause user to watch more, or would they have watched anyway?", effort: "Hard", detail: "Requires counterfactual estimation, instrumental variables, or A/B testing at scale. Prevents the system from taking credit for organic behavior." },
          { title: "Real-Time Personalization", d: "Adapt recommendations within a single session based on what the user just watched. If user switches from cooking to music, next refresh should reflect this immediately.", effort: "Medium", detail: "Requires streaming feature pipeline + session-aware ranking model. Most impactful for long sessions (30+ min)." },
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
    { q:"Why does YouTube use a two-stage system instead of one model?", a:"Math. With 200M eligible videos and 140K QPS, scoring every video with a deep model would require 200M x 140K = 28 trillion inferences per second. Impossible. The two-stage approach reduces this: candidate gen (cheap ANN) filters to ~1000, then the deep model scores only 1000. This is 200,000x less work. The candidate gen model is optimized for recall (don't miss good videos), while the ranking model is optimized for precision (correct ordering). Different objectives, different model architectures, different compute budgets.", tags:["architecture"] },
    { q:"How is this different from Netflix recommendations?", a:"Three key differences: (1) Content type — YouTube has 800M+ user-generated videos of wildly varying quality. Netflix has ~15K professionally produced titles. YouTube needs quality scoring; Netflix doesn't. (2) Real-time signals — YouTube videos go viral in hours. Netflix content popularity changes over weeks. YouTube needs near-real-time feature updates. (3) Objective — YouTube maximizes session engagement (watch time across many videos). Netflix maximizes per-title satisfaction (did you enjoy this movie?). YouTube is more sequential/session-oriented.", tags:["design"] },
    { q:"How would you handle a creator with 100M subscribers uploading a new video?", a:"This is the 'celebrity post' problem for recommendations. When a mega-creator uploads: (1) Immediately push to subscription feed of all subscribers. (2) In recommendation system: the video starts with a strong prior (channel's historical performance). (3) Early engagement signals come in within minutes from subscriber views. (4) If engagement metrics are strong, the recommendation system rapidly expands the audience through the two-tower retrieval. (5) Trending generator picks it up within an hour. The key insight: subscriber notification is a separate system from recommendations. Recs discover the video independently based on engagement signals, not just subscriber count.", tags:["scalability"] },
    { q:"How do you prevent the system from recommending harmful content?", a:"Multi-layered defense: (1) At upload: content moderation classifiers flag videos for review. Violating content is removed. (2) Borderline content (not violating but problematic): assigned a 'borderline score' by a separate classifier. (3) In ranking: borderline_score is a feature with a strong negative weight — demoted but not removed. (4) In re-ranking: hard rules prevent showing borderline content to minors. (5) External raters evaluate recommendation quality weekly. (6) Audit pipeline: sample recommendations and check for policy violations. YouTube calls this 'raising the bar' — reducing borderline content recommendations by 70% was a major initiative.", tags:["responsibility"] },
    { q:"Why not use reinforcement learning instead of supervised learning?", a:"YouTube has experimented with RL (REINFORCE, SlateQ) but it's hard to deploy at scale because: (1) Credit assignment — did the user leave because of video #5 or because they had to go to work? Long-horizon rewards are noisy. (2) Off-policy evaluation is unreliable — you can't trust simulated rewards. (3) RL can find adversarial policies that exploit users (maximizing session length through addictive patterns). (4) Training instability — RL is much harder to debug than supervised learning when things go wrong. In practice, YouTube uses contextual bandits for exploration (simpler form of RL) and supervised multi-task learning for ranking. Full RL is more of a research direction than production reality.", tags:["ml"] },
    { q:"How do you evaluate recommendation quality offline?", a:"Multiple offline metrics: (1) Recall@K — of videos the user will watch in the next 24h, how many appear in our top-K candidates? Measures candidate gen quality. (2) NDCG — are the videos in the right order? Measures ranking quality. (3) Coverage — what fraction of eligible videos ever get recommended? Low coverage = popularity bias. (4) Novelty — how many recommended videos are NOT in the user's typical consumption pattern? Measures serendipity. (5) Calibration — if model predicts 40% watch probability, do ~40% of users actually watch? Important for multi-objective combining. All measured on held-out future data (time-split, NOT random split — critical to avoid leakage).", tags:["evaluation"] },
    { q:"What happens when the embedding index is out of date?", a:"The embedding index is rebuilt daily with the latest video embeddings. Between rebuilds: (1) New videos are added to a small 'real-time index' (in-memory, updated every few minutes). At retrieval time, query both the main index AND the real-time index, merge results. (2) Deleted/taken-down videos are kept in a 'blocklist' and filtered post-retrieval. Cheaper than rebuilding the entire index. (3) If the index is significantly stale (rebuild failed): serve from last-known-good index. Freshness features in the ranking model compensate — older videos get demoted. (4) Monitoring: track 'index staleness' metric. Alert if rebuild is >6 hours late.", tags:["infrastructure"] },
    { q:"How would you A/B test a new candidate generator?", a:"Recommendation A/B tests are tricky because of network effects and interference. Approach: (1) Interleaving: mix candidates from old and new generators, let the ranking model score all of them. Track which generator's candidates end up in the final top-30 (win rate). More sensitive than traditional A/B. (2) Full A/B: route 5% of users to new generator, 95% to old. Measure: watch time per session, CTR, satisfaction survey scores, not-interested rate. Run for 2+ weeks to capture weekly patterns. (3) Guardrails: auto-stop if any metric degrades by >2%. (4) Holdback experiment: after launching, keep 1% on the old version indefinitely to measure long-term impact (some effects only show up after weeks).", tags:["evaluation"] },
    { q:"How does YouTube's system compare to TikTok's?", a:"Fundamental architectural difference: (1) YouTube is subscription-based with recommendation overlay. Users follow channels; recs augment this. TikTok is pure recommendation — no social graph needed. (2) YouTube has long-form content (10-60 min) where watch time matters. TikTok is short-form (15-60 sec) where completion rate and replay are key signals. (3) YouTube's candidate gen uses subscription graph + collaborative filtering heavily. TikTok relies almost entirely on content-based + behavioral signals (interest tags). (4) YouTube optimizes for session-level watch time. TikTok optimizes for swipe-level engagement (each swipe is a mini-recommendation). (5) TikTok's feedback loop is tighter — 15 seconds to get a signal vs 10+ minutes for YouTube.", tags:["design"] },
    { q:"How do you handle seasonal and trending content?", a:"Three mechanisms: (1) Trending candidate generator — separate pipeline that tracks engagement velocity (views/hour vs. baseline for that video's age and category). Videos with 3x+ velocity are injected into candidate pool. Updated every 5 minutes. (2) Freshness features in ranking — hours_since_upload, engagement_velocity, is_trending are features the ranking model uses. During events (World Cup, elections), these features automatically boost relevant fresh content. (3) Manual editorial overrides — for major global events, editorial team can pin content to trending feeds. Used sparingly (maybe 5 times/year for safety-critical events like natural disasters).", tags:["design"] },
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
  api: ApiSection, design: DesignSection, candidate: CandidateSection,
  ranking: RankingSection, features: FeaturesSection, objectives: ObjectivesSection,
  data: DataModelSection, training: TrainingSection, coldstart: ColdStartSection,
  scalability: ScalabilitySection, watchouts: WatchoutsSection,
  observability: ObservabilitySection,
  enhancements: EnhancementsSection, followups: FollowupsSection,
};

export default function YouTubeRecsSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">YouTube Recommendations</h1>
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