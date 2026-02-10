import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   NOTIFICATION / FEED RANKING — ML System Design Reference (Google L6)
   Pearl white theme · 17 sections (HLD + LLD + ML Deep Dive)
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",                icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",            icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",     icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",              icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",       icon: "🏗️", color: "#9333ea" },
  { id: "candidates",    label: "Candidate Generation",    icon: "🎯", color: "#c026d3" },
  { id: "ranking",       label: "Ranking Models",          icon: "🏆", color: "#dc2626" },
  { id: "features",      label: "Feature Engineering",     icon: "⚙️", color: "#d97706" },
  { id: "diversity",     label: "Diversity & Freshness",   icon: "🌈", color: "#0f766e" },
  { id: "notifications", label: "Notification Policy",     icon: "🔔", color: "#ea580c" },
  { id: "data",          label: "Data Model",              icon: "🗄️", color: "#059669" },
  { id: "training",      label: "Training Pipeline",       icon: "🔄", color: "#7e22ce" },
  { id: "objectives",    label: "Multi-Objective Ranking",  icon: "⚖️", color: "#0284c7" },
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
            <Label>What is Notification / Feed Ranking?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              Feed ranking decides <strong>what content to show, in what order, and when</strong> every time a user opens a feed-based product — Google Discover, Gmail Inbox, YouTube Home, Google News, or social feeds. Notification ranking extends this: deciding <strong>whether to push-notify</strong> a user about a piece of content, balancing engagement against annoyance.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              The core challenge is multi-objective: maximize engagement (clicks, dwell time) while preserving content diversity, freshness, and user satisfaction. A feed that only shows clickbait maximizes short-term CTR but destroys long-term retention. An L6 must articulate this tension and design a system that balances competing objectives.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard? (Google L6 Depth)</Label>
            <ul className="space-y-2.5">
              <Point icon="⚖️" color="#0891b2">Multi-objective optimization — you're simultaneously optimizing for click probability, dwell time, satisfaction, diversity, freshness, and long-term retention. These objectives conflict: high-CTR clickbait destroys long-term satisfaction.</Point>
              <Point icon="🕐" color="#0891b2">Temporal dynamics — a breaking news story is highly relevant for 2 hours, then stale. An email from your boss is urgent for 10 minutes, then less so. The ranking function must model time-dependent utility.</Point>
              <Point icon="🔔" color="#0891b2">Notification is a scarce resource — every push notification costs user attention and goodwill. Send too many → user disables notifications (permanent loss). Send too few → user misses important content. The optimal notification rate varies per user.</Point>
              <Point icon="🌊" color="#0891b2">Cold-start on new content — content is constantly arriving. You have zero interaction data on a new article or email. Must rank it alongside items with rich engagement history. Explore-exploit tradeoff.</Point>
              <Point icon="📏" color="#0891b2">Position bias — users click item 1 more than item 10 regardless of quality. Naive training on click data learns position bias, not true relevance. Must debias training data or the model perpetuates the existing ranking.</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Google Products with Feed Ranking</Label>
            <div className="space-y-2.5">
              {[
                { prod: "Google Discover", desc: "Interest-based content feed on mobile home", key: "Topic modeling + CTR + dwell" },
                { prod: "Gmail Inbox", desc: "Email priority and categorization", key: "Importance model + notification" },
                { prod: "YouTube Home", desc: "Video recommendations on homepage", key: "Engagement + satisfaction" },
                { prod: "Google News", desc: "Personalized news feed", key: "Freshness + diversity + authority" },
                { prod: "Google Photos", desc: "Memories and photo suggestions", key: "Emotional value + recency" },
              ].map((e,i) => (
                <div key={i} className="flex items-start gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-28 shrink-0">{e.prod}</span>
                  <div>
                    <span className="text-stone-500">{e.desc}</span>
                    <div className="text-stone-400 text-[10px]">{e.key}</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <Label color="#2563eb">The Feed Pipeline (Preview)</Label>
            <svg viewBox="0 0 360 170" className="w-full">
              <defs><marker id="ah-fr" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>

              <rect x={10} y={10} width={70} height={35} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
              <text x={45} y={25} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="700" fontFamily="monospace">Candidate</text>
              <text x={45} y={37} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">10K items</text>

              <rect x={100} y={10} width={70} height={35} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
              <text x={135} y={25} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="700" fontFamily="monospace">Rank</text>
              <text x={135} y={37} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">score all</text>

              <rect x={190} y={10} width={70} height={35} rx={6} fill="#0f766e10" stroke="#0f766e" strokeWidth={1.5}/>
              <text x={225} y={25} textAnchor="middle" fill="#0f766e" fontSize="8" fontWeight="700" fontFamily="monospace">Re-Rank</text>
              <text x={225} y={37} textAnchor="middle" fill="#0f766e80" fontSize="7" fontFamily="monospace">diversity</text>

              <rect x={280} y={10} width={70} height={35} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
              <text x={315} y={25} textAnchor="middle" fill="#ea580c" fontSize="8" fontWeight="700" fontFamily="monospace">Notify?</text>
              <text x={315} y={37} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">push policy</text>

              <line x1={80} y1={28} x2={100} y2={28} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-fr)"/>
              <line x1={170} y1={28} x2={190} y2={28} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-fr)"/>
              <line x1={260} y1={28} x2={280} y2={28} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-fr)"/>

              <rect x={10} y={65} width={340} height={95} rx={6} fill="#faf9f7" stroke="#e7e5e4"/>
              <text x={20} y={82} fill="#c026d3" fontSize="7" fontWeight="600" fontFamily="monospace">Candidate Gen: Retrieve 10K potential items from multiple sources (followed topics, trending,</text>
              <text x={20} y={94} fill="#c026d3" fontSize="7" fontFamily="monospace">  collaborative filtering, fresh content). Lightweight models — recall-optimized.</text>
              <text x={20} y={110} fill="#dc2626" fontSize="7" fontWeight="600" fontFamily="monospace">Ranking: Score all candidates with a deep model. P(click), P(dwell>30s), P(satisfied).</text>
              <text x={20} y={122} fill="#dc2626" fontSize="7" fontFamily="monospace">  Multi-objective: combine engagement + satisfaction + freshness scores.</text>
              <text x={20} y={138} fill="#0f766e" fontSize="7" fontWeight="600" fontFamily="monospace">Re-Rank: Apply diversity constraints — no >2 items from same source, mix topics, inject fresh content.</text>
              <text x={20} y={152} fill="#ea580c" fontSize="7" fontWeight="600" fontFamily="monospace">Notification: Decide if the top item warrants a push notification. Per-user notification budget.</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Google Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Core to Discover, Gmail, YouTube Home, News</div>
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
            <p className="text-[12px] text-stone-500 mt-0.5">"Design a feed ranking system" — scope: "I'll design a personalized content feed with push notifications — think Google Discover or a news home feed. I'll cover the full funnel: candidate generation (10K items), multi-stage ranking (engagement + satisfaction), diversity re-ranking, and a notification policy layer that decides when to push-notify. I'll focus on the multi-objective ranking formulation and the notification volume optimization." Calling out multi-objective and notification volume early distinguishes L6 from L5.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Generate a personalized feed of 50-200 items for each user on every app open</Point>
            <Point icon="2." color="#059669">Rank items by predicted engagement (click, dwell) AND predicted satisfaction (like, share, not-interested)</Point>
            <Point icon="3." color="#059669">Enforce diversity: mix content types, topics, sources; avoid repetitive feeds</Point>
            <Point icon="4." color="#059669">Support push notifications with per-user volume optimization (not too many, not too few)</Point>
            <Point icon="5." color="#059669">Handle cold-start: rank brand-new content with no interaction history</Point>
            <Point icon="6." color="#059669">Support explicit user controls: follow/unfollow topics, "not interested", mute sources</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Feed generation latency: &lt;200ms end-to-end (user opens app → feed appears)</Point>
            <Point icon="2." color="#dc2626">Candidate pool: score 10K+ candidates per request</Point>
            <Point icon="3." color="#dc2626">Scale: 1B+ daily active users, each opening the feed 5-10 times/day</Point>
            <Point icon="4." color="#dc2626">Notification delivery: &lt;30 seconds from event to push notification</Point>
            <Point icon="5." color="#dc2626">Model freshness: retrain ranking models daily, update embeddings hourly</Point>
            <Point icon="6." color="#dc2626">Availability: 99.99% — the feed IS the product; if ranking is down, the product is empty</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask (L6 Signal)</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "What type of feed? News articles, social posts, emails, videos, or heterogeneous?",
            "Push notifications: are they for all top items or only 'breaking' content?",
            "Is there a social graph (friends/follows) or interest-based only?",
            "What's the content lifecycle? Minutes (news) vs days (articles) vs permanent (emails)?",
            "Do users have explicit preferences (followed topics) or is everything implicit?",
            "Business model: engagement-optimized (ad-supported) or subscription (satisfaction-optimized)?",
            "Regulatory: any content diversity requirements (e.g., EU media pluralism)?",
            "Multi-device: does the feed sync across phone, tablet, desktop?",
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
          <Label color="#7c3aed">Step 1 — Feed Request Volume</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Daily active users (DAU)" result="~1B users" note="Google Discover / YouTube Home scale." />
            <MathStep step="2" formula="Feed opens per user per day" result="~8" note="Every app open triggers a fresh feed." />
            <MathStep step="3" formula="Feed requests/day = 1B × 8" result="~8B/day" note="Each requires candidate gen + ranking." />
            <MathStep step="4" formula="Feed QPS = 8B / 86400" result="~93K QPS" note="Average. Peak: 3x at morning commute." final />
            <MathStep step="5" formula="Items scored per request" result="~10K" note="Candidate gen retrieves 10K, all scored by ranker." />
            <MathStep step="6" formula="Model inferences/sec = 93K × 10K" result="~930M/sec" note="Per-item scoring across all users." final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Content Pool & Freshness</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="New content items per day" result="~50M" note="Articles, videos, posts, emails entering the system." />
            <MathStep step="2" formula="Active content pool" result="~500M items" note="Content eligible for ranking (past 7-30 days)." />
            <MathStep step="3" formula="Content embeddings" result="768-dim each" note="Each item embedded for retrieval. 500M × 3KB = 1.5TB." final />
            <MathStep step="4" formula="Item feature updates" result="Continuous" note="Engagement signals (clicks, shares) update item features in real-time." />
            <MathStep step="5" formula="Content half-life (news)" result="~4 hours" note="News relevance decays exponentially. Emails decay differently." />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Notification Volume</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Notification-eligible events/day" result="~500M" note="Breaking news, trending topics, important emails." />
            <MathStep step="2" formula="Target notifications per user per day" result="~3-5" note="Too many → user disables. Too few → missed engagement." />
            <MathStep step="3" formula="Total notifications sent/day = 1B × 4" result="~4B/day" note="Each notification scored by the notification policy model." final />
            <MathStep step="4" formula="Notification CTR (good)" result=">8%" note="Below 5% → notifications are annoying. Above 10% → under-sending." />
            <MathStep step="5" formula="Notification disable rate" result="<0.1%/month" note="Critical health metric. Irreversible — user never comes back." />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Latency Budget</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Total feed generation budget" result="<200ms" note="User opens app → feed visible. Must feel instant." final />
            <MathStep step="2" formula="Candidate generation" result="~30ms" note="Multi-source retrieval in parallel." />
            <MathStep step="3" formula="Feature assembly" result="~20ms" note="Batch fetch user + item features from feature store." />
            <MathStep step="4" formula="Ranking model inference (10K items)" result="~50ms" note="Light two-tower for pre-rank, deep model for top-500." />
            <MathStep step="5" formula="Diversity re-ranking" result="~10ms" note="Greedy MMR over top-200 scored items." />
            <MathStep step="6" formula="Network + serialization overhead" result="~30ms" note="gRPC between services, response formatting." />
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
          <Label color="#2563eb">Feed & Notification APIs</Label>
          <CodeBlock code={`# 1. FEED API — called on every app open
# GET /v1/feed
{
  "user_id": "u_abc123",
  "context": {
    "device_type": "mobile",
    "time_of_day": "08:30",
    "timezone": "America/New_York",
    "connection": "wifi",
    "locale": "en-US",
    "last_feed_ts": "2024-02-10T07:15:00Z",
  },
  "pagination": {
    "page_size": 20,
    "cursor": null,         // null for first page
  },
  "filters": {
    "content_types": ["article", "video", "web_story"],
    "exclude_seen": true,
  }
}

# Response:
{
  "items": [
    {
      "item_id": "itm_xyz",
      "content_type": "article",
      "title": "SpaceX Starship...",
      "source": "Reuters",
      "published_at": "2024-02-10T07:45:00Z",
      "scores": {
        "relevance": 0.92,
        "engagement": 0.78,
        "satisfaction": 0.85,
      },
      "explanation": "Based on your interest in Space",
      "notification_sent": false,
    },
    // ... 19 more items
  ],
  "cursor": "cursor_abc",
  "metadata": {
    "candidates_considered": 8432,
    "generation_latency_ms": 142,
  }
}

# 2. NOTIFICATION DECISION API (internal)
# Called when high-value content arrives
{
  "user_id": "u_abc123",
  "item_id": "itm_xyz",
  "urgency": "high",        // low | medium | high | breaking
  "decision": "send",       // send | suppress | defer
  "reason": "Matches followed topic + high-quality source",
  "daily_budget_remaining": 3,
}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Design Decisions (L6 Depth)</Label>
          <div className="space-y-3">
            {[
              { q: "Why return scores in the feed response?", a: "Transparency and debugging. Client can show 'Why this?' explanations. ML team can debug ranking issues from logged responses. Also enables client-side re-ranking for layout optimization (e.g., if a video card needs more space, pick highest-scoring video from available candidates)." },
              { q: "Why context signals (time, device, connection)?", a: "Context dramatically affects ranking. At 8am on mobile with cellular = user wants quick headlines. At 8pm on WiFi = user wants long-form video. On desktop = user wants detailed articles. The ranking model takes context as input features to personalize the ordering." },
              { q: "Why exclude_seen?", a: "Showing already-seen items is the #1 user complaint about feeds. Track impression history per user. The candidate generator excludes seen items, but the filter is also enforced at the API layer as defense-in-depth. Challenge: what about items seen on a different device?" },
              { q: "Why notification daily_budget_remaining?", a: "Each user has a daily notification budget (3-5 per day typically). The notification policy model decides which items are important enough to 'spend' from this budget. Budget prevents over-notification even when many interesting items arrive simultaneously (e.g., breaking news day)." },
              { q: "Why defer as a notification decision?", a: "'Defer' means: the item is notification-worthy but not right now. Examples: it's 2am (user is sleeping), user is in a meeting (calendar signal), user just received 2 notifications in the last hour (rate limit). Deferred items are re-evaluated at a better time." },
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
        <Label color="#9333ea">Full System Architecture — Feed + Notification Pipeline</Label>
        <svg viewBox="0 0 720 360" className="w-full">
          <defs><marker id="ah-hld" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>

          {/* Feed Request Path */}
          <text x={15} y={18} fill="#9333ea" fontSize="9" fontWeight="700" fontFamily="monospace">FEED RANKING PATH (on every app open)</text>
          <rect x={15} y={28} width={70} height={40} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={50} y={45} textAnchor="middle" fill="#2563eb" fontSize="8" fontWeight="600" fontFamily="monospace">User</text>
          <text x={50} y={58} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">opens app</text>

          <rect x={105} y={28} width={85} height={40} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={147} y={45} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Candidate Gen</text>
          <text x={147} y={58} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">10K items (~30ms)</text>

          <rect x={210} y={28} width={80} height={40} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={250} y={45} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Pre-Rank</text>
          <text x={250} y={58} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">10K→500 (~20ms)</text>

          <rect x={310} y={28} width={85} height={40} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={352} y={45} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Full Rank</text>
          <text x={352} y={58} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">500→200 (~40ms)</text>

          <rect x={415} y={28} width={80} height={40} rx={6} fill="#0f766e10" stroke="#0f766e" strokeWidth={1.5}/>
          <text x={455} y={45} textAnchor="middle" fill="#0f766e" fontSize="8" fontWeight="600" fontFamily="monospace">Re-Rank</text>
          <text x={455} y={58} textAnchor="middle" fill="#0f766e80" fontSize="7" fontFamily="monospace">diversity (~10ms)</text>

          <rect x={515} y={28} width={60} height={40} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={545} y={45} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Feed</text>
          <text x={545} y={58} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">top 50</text>

          {/* Feed arrows */}
          <line x1={85} y1={48} x2={105} y2={48} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={190} y1={48} x2={210} y2={48} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={290} y1={48} x2={310} y2={48} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={395} y1={48} x2={415} y2={48} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={495} y1={48} x2={515} y2={48} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>

          {/* Notification Path */}
          <text x={15} y={98} fill="#ea580c" fontSize="9" fontWeight="700" fontFamily="monospace">NOTIFICATION PATH (event-driven)</text>
          <rect x={15} y={108} width={80} height={40} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={55} y={125} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">New Content</text>
          <text x={55} y={138} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">event arrives</text>

          <rect x={115} y={108} width={85} height={40} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={157} y={125} textAnchor="middle" fill="#ea580c" fontSize="8" fontWeight="600" fontFamily="monospace">Notify Model</text>
          <text x={157} y={138} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">P(click|notify)</text>

          <rect x={220} y={108} width={90} height={40} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
          <text x={265} y={125} textAnchor="middle" fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Volume Policy</text>
          <text x={265} y={138} textAnchor="middle" fill="#7c3aed80" fontSize="7" fontFamily="monospace">budget + timing</text>

          <rect x={330} y={108} width={70} height={40} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={365} y={125} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Deliver</text>
          <text x={365} y={138} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">FCM / APNs</text>

          {/* Notification arrows */}
          <line x1={95} y1={128} x2={115} y2={128} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={200} y1={128} x2={220} y2={128} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={310} y1={128} x2={330} y2={128} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>

          {/* Data stores */}
          <rect x={600} y={28} width={80} height={30} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={640} y={47} textAnchor="middle" fill="#d97706" fontSize="7" fontWeight="600" fontFamily="monospace">Feature Store</text>
          <rect x={600} y={66} width={80} height={30} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={640} y={85} textAnchor="middle" fill="#dc2626" fontSize="7" fontWeight="600" fontFamily="monospace">Content Index</text>
          <rect x={600} y={104} width={80} height={30} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={640} y={123} textAnchor="middle" fill="#0891b2" fontSize="7" fontWeight="600" fontFamily="monospace">User Profiles</text>

          {/* Legend */}
          <rect x={15} y={170} width={700} height={180} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={25} y={190} fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Candidate Gen (~30ms): Multi-source retrieval. Followed topics, collaborative filtering, trending, fresh content.</text>
          <text x={25} y={205} fill="#c026d3" fontSize="8" fontFamily="monospace">  Each source nominates candidates. Union = ~10K items. Lightweight models (two-tower ANN retrieval).</text>
          <text x={25} y={222} fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Pre-Rank (~20ms): Lightweight model scores 10K items. Dot product of user embedding × item embedding.</text>
          <text x={25} y={237} fill="#dc2626" fontSize="8" fontFamily="monospace">  Reduces 10K → 500 candidates. Fast: no cross-features, just embedding similarity.</text>
          <text x={25} y={254} fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Full Rank (~40ms): Deep model with cross-features. P(click), P(dwell>30s), P(satisfy). 500 items scored.</text>
          <text x={25} y={269} fill="#dc2626" fontSize="8" fontFamily="monospace">  Multi-task architecture with shared bottom and task-specific towers. 200+ features.</text>
          <text x={25} y={286} fill="#0f766e" fontSize="8" fontWeight="600" fontFamily="monospace">Re-Rank (~10ms): Diversity injection. MMR to ensure topic variety. Freshness boost. Source deduplication.</text>
          <text x={25} y={303} fill="#ea580c" fontSize="8" fontWeight="600" fontFamily="monospace">Notification: Event-driven. When important content arrives, score P(click|push) and check volume budget.</text>
          <text x={25} y={318} fill="#ea580c" fontSize="8" fontFamily="monospace">  Only send if: P(engage) > threshold AND daily_budget_remaining > 0 AND timing is appropriate.</text>
          <text x={25} y={338} fill="#6366f1" fontSize="8" fontWeight="600" fontFamily="monospace">L6 KEY INSIGHT: Feed ranking is a FUNNEL. Each stage trades off speed vs accuracy. Candidate gen = fast recall.</text>
          <text x={25} y={348} fill="#6366f1" fontSize="8" fontFamily="monospace">Pre-rank = cheap scoring. Full rank = expensive but accurate. Only the final 50 items reach the user.</text>
        </svg>
      </Card>
    </div>
  );
}

function CandidatesSection() {
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Candidate Generation — Multi-Source Retrieval</Label>
        <p className="text-[12px] text-stone-500 mb-4">The candidate generator must retrieve ~10K potentially relevant items from a pool of 500M+ in under 30ms. No single retrieval method is sufficient — use multiple sources in parallel, each capturing different relevance signals.</p>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Multi-Source Candidate Retrieval" code={`# Candidate generation: parallel multi-source retrieval
async def generate_candidates(user, context, k=10000):
    # Launch all sources in parallel
    sources = await asyncio.gather(
        # Source 1: Interest-based (ANN retrieval)
        interest_retrieval(user.embedding, k=3000),
        # "Items similar to what this user likes"
        # Two-tower model: user_emb ⋅ item_emb via ANN index

        # Source 2: Followed topics
        topic_retrieval(user.followed_topics, k=2000),
        # Latest content tagged with user's followed topics

        # Source 3: Collaborative filtering
        cf_retrieval(user.similar_users, k=2000),
        # "Users like you engaged with these items"
        # item2item ANN: seed items → similar items

        # Source 4: Trending / popular
        trending_retrieval(user.geo, user.locale, k=1500),
        # Globally/locally trending content
        # Provides diversity and serendipity

        # Source 5: Fresh content (explore)
        fresh_retrieval(context.time, k=1500),
        # Content published in last 2 hours
        # No interaction data yet — cold start exploration

        # Source 6: Re-engagement
        reengagement_retrieval(user.id, k=500),
        # Items user saw but didn't engage with last session
        # Give a second chance at a different position
    )

    # Merge, deduplicate, exclude already-seen
    candidates = deduplicate(flatten(sources))
    candidates = exclude_seen(candidates, user.impression_history)

    return candidates[:k]

# WHY MULTIPLE SOURCES:
# Interest-based alone → filter bubble (only things you already like)
# Trending alone → no personalization
# CF alone → popularity bias (popular items dominate)
# Fresh alone → no quality signal
# TOGETHER → personalized, diverse, fresh, high-quality`} />
          <div className="space-y-4">
            <Card accent="#c026d3">
              <Label color="#c026d3">Source Contribution Analysis</Label>
              <div className="space-y-2">
                {[
                  { source: "Interest-based (ANN)", pct: "30%", role: "Personalization backbone. Items matching user's interest embedding. Highest precision but risks filter bubble.", color: "#c026d3" },
                  { source: "Followed topics", pct: "20%", role: "Explicit user intent. Latest content from topics user subscribed to. Users expect to see these items.", color: "#7c3aed" },
                  { source: "Collaborative filtering", pct: "20%", role: "Discovery. 'Users like you also liked...' Surfaces content outside user's explicit interests.", color: "#2563eb" },
                  { source: "Trending / popular", pct: "15%", role: "Serendipity + social proof. Everyone is talking about this. Keeps feed feeling current and relevant.", color: "#059669" },
                  { source: "Fresh content (explore)", pct: "10%", role: "Cold-start exploration. New content with no engagement data. Essential for content freshness.", color: "#d97706" },
                  { source: "Re-engagement", pct: "5%", role: "Second chance. Items user scrolled past — maybe wrong position/time. Low volume but high conversion.", color: "#78716c" },
                ].map((s,i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded shrink-0" style={{ background: s.color+"15", color: s.color }}>{s.pct}</span>
                    <div>
                      <div className="text-[11px] font-bold text-stone-700">{s.source}</div>
                      <div className="text-[10px] text-stone-500">{s.role}</div>
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

function RankingSection() {
  return (
    <div className="space-y-5">
      <Card accent="#dc2626">
        <Label color="#dc2626">Ranking Models — Multi-Stage Scoring</Label>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Pre-Ranker (10K → 500)" code={`# Pre-ranker: lightweight scoring for 10K candidates
# Must be FAST — sub-5ms for 10K items
class PreRanker:
    def __init__(self):
        # Two-tower model (pre-computed embeddings)
        self.user_tower = UserTower(dim=128)  # shared with retrieval
        self.item_tower = ItemTower(dim=128)

    def score(self, user_emb, item_embs):
        # Dot product scoring — O(n) for n items
        # user_emb: [128], item_embs: [10000, 128]
        scores = item_embs @ user_emb  # matmul, ~2ms

        # Add lightweight features (no cross-features)
        freshness_boost = compute_freshness(item_timestamps)
        source_trust = item_source_scores
        scores += 0.1 * freshness_boost + 0.05 * source_trust

        # Top-500 by score
        top_indices = topk(scores, k=500)
        return top_indices

# WHY TWO STAGES (pre-rank + full rank)?
# Full ranker uses 200+ cross-features → ~0.5ms per item
# 10K × 0.5ms = 5 seconds — TOO SLOW
# Pre-ranker: dot product = 0.0002ms per item → 2ms for 10K
# Pre-rank eliminates 9,500 clearly-irrelevant items
# Full ranker only scores 500 promising candidates`} />
          <CodeBlock title="Full Ranker (500 → 200)" code={`# Full ranker: deep multi-task model
class FeedRanker(nn.Module):
    def __init__(self):
        # Shared bottom: learns common representations
        self.shared = nn.Sequential(
            nn.Linear(FEATURE_DIM, 512),
            nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        # Task-specific towers (Multi-gate Mixture of Experts)
        self.click_tower = TaskTower(256, 1)   # P(click)
        self.dwell_tower = TaskTower(256, 1)    # P(dwell > 30s)
        self.satisfy_tower = TaskTower(256, 1)  # P(satisfaction)
        self.share_tower = TaskTower(256, 1)    # P(share)

    def forward(self, features):
        shared_repr = self.shared(features)
        return {
            "p_click": sigmoid(self.click_tower(shared_repr)),
            "p_dwell": sigmoid(self.dwell_tower(shared_repr)),
            "p_satisfy": sigmoid(self.satisfy_tower(shared_repr)),
            "p_share": sigmoid(self.share_tower(shared_repr)),
        }

# FINAL SCORE = weighted combination of task predictions
# score = w1*P(click) + w2*P(dwell>30s) + w3*P(satisfy) + w4*P(share)
# Weights tuned via online A/B tests to optimize long-term metrics
# w1=0.3, w2=0.3, w3=0.3, w4=0.1 (example)
#
# WHY MULTI-TASK:
# Optimizing click alone → clickbait rises
# Adding dwell → clickbait drops (users leave quickly)
# Adding satisfaction → high-quality content rises
# Adding share → viral/informative content rises`} />
        </div>
      </Card>
      <Card>
        <Label color="#d97706">Position Bias — The Hidden Problem</Label>
        <div className="grid grid-cols-2 gap-5">
          <div className="text-[12px] text-stone-500 space-y-2">
            <p><strong>The problem:</strong> Users click position 1 more than position 10 regardless of content quality. If you train on click data naively, the model learns: "items that were shown at position 1 are good" — a self-fulfilling prophecy. The model just learns to replicate the current ranking.</p>
            <p><strong>Why it matters at L6:</strong> Position bias is a subtle feedback loop that most engineers miss. The model appears to have high AUC (because it predicts clicks well) but isn't actually learning relevance — it's learning position.</p>
          </div>
          <CodeBlock title="Position Bias Correction" code={`# Approach 1: Position as feature (with dropout at serving)
# During training: include position as an input feature
# During serving: set position = 0 (or expected value)
# Model learns to separate position effect from content quality
features["position"] = item.displayed_position  # training
features["position"] = 0                         # serving

# Approach 2: Inverse Propensity Weighting (IPW)
# Weight examples by 1/P(click | position)
# Items at position 10 get higher weight (clicks are more meaningful)
propensity = position_click_rates[item.position]
sample_weight = 1.0 / propensity

# Approach 3: Pairwise learning
# Instead of predicting absolute clicks, learn:
# "Item A is better than Item B"
# Pairs sampled from different positions to cancel out position effect`} />
        </div>
      </Card>
    </div>
  );
}

function FeaturesSection() {
  return (
    <div className="space-y-5">
      <Card accent="#d97706">
        <Label color="#d97706">Feature Categories for Feed Ranking</Label>
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
                { cat: "User Interest", ex: "user_embedding, followed_topics, long_term_interests, short_term_interests (last 24h)", comp: "Batch + stream", impact: "Very High" },
                { cat: "Item Content", ex: "item_embedding, topic_tags, content_quality_score, source_authority, language", comp: "On publish", impact: "Very High" },
                { cat: "User × Item Cross", ex: "user_topic_affinity, user_source_history, embedding_similarity, past_engagement_with_source", comp: "Online", impact: "Very High" },
                { cat: "Freshness / Time", ex: "item_age_minutes, time_decay_score, is_breaking_news, content_half_life", comp: "Online (computed)", impact: "High" },
                { cat: "Engagement Signals", ex: "item_ctr, item_avg_dwell_time, item_share_rate, item_not_interested_rate", comp: "Stream (real-time)", impact: "Very High" },
                { cat: "Context", ex: "time_of_day, day_of_week, device_type, connection_type, user_active_minutes_today", comp: "Online (request)", impact: "High" },
                { cat: "Social Proof", ex: "friends_who_engaged, trending_score, comment_count, reaction_count", comp: "Stream", impact: "Medium" },
                { cat: "Notification State", ex: "was_push_notified, time_since_notification, daily_notif_count, user_notif_ctr", comp: "Online", impact: "High (for notif)" },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-bold text-stone-700">{r.cat}</td>
                  <td className="px-3 py-2 text-stone-500 font-mono text-[10px]">{r.ex}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.comp}</td>
                  <td className="text-center px-3 py-2"><Pill bg={r.impact.includes("Very")?"#fef2f2":"#fffbeb"} color={r.impact.includes("Very")?"#dc2626":"#d97706"}>{r.impact}</Pill></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <Card>
        <Label color="#dc2626">The Most Important Features (L6 Insight)</Label>
        <div className="grid grid-cols-3 gap-4">
          {[
            { f: "User × Item embedding similarity", why: "Core personalization signal. 'How well does this item match this user's interests?' Dot product of user/item embeddings from two-tower model. Single most predictive feature.", rank: "#1" },
            { f: "Item engagement rate (real-time)", why: "Social proof + quality signal. Items with high engagement from similar users are likely good for this user too. Updated in real-time via streaming pipeline.", rank: "#2" },
            { f: "Content freshness (age in minutes)", why: "Time-decayed relevance. A news article is 10x more relevant at 1 hour old than 24 hours old. The decay function shape varies by content type (news decays fast, evergreen content decays slowly).", rank: "#3" },
          ].map((f,i) => (
            <Card key={i} className="border-red-200">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-red-100 text-red-700">{f.rank}</span>
                <span className="text-[11px] font-bold text-stone-800">{f.f}</span>
              </div>
              <p className="text-[10px] text-stone-500">{f.why}</p>
            </Card>
          ))}
        </div>
      </Card>
    </div>
  );
}

function DiversitySection() {
  return (
    <div className="space-y-5">
      <Card accent="#0f766e">
        <Label color="#0f766e">Diversity Re-Ranking — Beyond Pure Relevance</Label>
        <p className="text-[12px] text-stone-500 mb-4">A feed ranked purely by predicted engagement is boring and repetitive: 10 articles about the same trending topic, all from the same source. Diversity re-ranking ensures the feed feels varied, fresh, and covers the user's breadth of interests — not just their strongest one.</p>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Maximal Marginal Relevance (MMR)" code={`# MMR: balance relevance with diversity
# Greedily select items that are BOTH relevant AND different
# from already-selected items

def mmr_rerank(candidates, scores, lambda_=0.6, k=50):
    """
    lambda_: balance between relevance and diversity
    lambda_=1.0: pure relevance (no diversity)
    lambda_=0.0: pure diversity (ignore relevance)
    lambda_=0.6: good balance for most feeds
    """
    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(k):
        best_idx = None
        best_mmr = -float("inf")

        for idx in remaining:
            relevance = scores[idx]

            # Max similarity to any already-selected item
            if selected:
                max_sim = max(
                    cosine_similarity(
                        candidates[idx].embedding,
                        candidates[s].embedding
                    )
                    for s in selected
                )
            else:
                max_sim = 0

            # MMR score: high relevance + low redundancy
            mmr = lambda_ * relevance - (1 - lambda_) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]

# ADDITIONAL DIVERSITY CONSTRAINTS:
# - Max 2 items from same source in top 10
# - Max 3 items on same topic in top 20
# - At least 1 video, 1 article, 1 web story in top 5
# - Fresh content boost: at least 20% of feed < 2 hours old
# Applied as hard constraints AFTER MMR soft diversity`} />
          <div className="space-y-4">
            <Card accent="#0f766e">
              <Label color="#0f766e">Diversity Dimensions</Label>
              <div className="space-y-2">
                {[
                  { dim: "Topic Diversity", desc: "Mix topics in the feed. Don't show 10 sports articles in a row even if the user loves sports. Rotate: sports → tech → world news → sports.", impl: "Topic classifier + max per-topic quota in top-K" },
                  { dim: "Source Diversity", desc: "Mix publishers/creators. Don't let one source dominate. Important for news (media pluralism) and creator platforms (fairness).", impl: "Source dedup: max 2 from same source in top-10" },
                  { dim: "Format Diversity", desc: "Mix content types: articles, videos, web stories, short clips. Different formats serve different moods and contexts.", impl: "Format-aware slot allocation in the feed layout" },
                  { dim: "Temporal Diversity", desc: "Mix fresh content (< 2 hours) with proven content (engagement-validated). Ensure the feed always feels 'current' with breaking items.", impl: "Freshness boost + minimum fresh item quota" },
                  { dim: "Exploration vs Exploitation", desc: "Show some items outside the user's known interests. Prevents filter bubble. Allows the system to discover new interests.", impl: "10-15% explore slots with epsilon-greedy or Thompson sampling" },
                ].map((d,i) => (
                  <div key={i} className="rounded-lg border border-stone-200 p-2.5">
                    <div className="text-[11px] font-bold text-stone-700">{d.dim}</div>
                    <p className="text-[10px] text-stone-500 mt-0.5">{d.desc}</p>
                    <p className="text-[9px] text-teal-600 mt-0.5"><strong>Impl:</strong> {d.impl}</p>
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

function NotificationsSection() {
  return (
    <div className="space-y-5">
      <Card accent="#ea580c">
        <Label color="#ea580c">Notification Policy — The Hardest Ranking Problem</Label>
        <p className="text-[12px] text-stone-500 mb-4">Notifications are fundamentally different from feed ranking. A feed item that's ranked 50th is still shown — the user scrolls to it. A notification is all-or-nothing: you either interrupt the user or you don't. The cost of a bad notification is much higher than a bad feed item.</p>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Notification Decision Model" code={`# Notification policy: should we push-notify this user about this item?
class NotificationPolicy:
    def __init__(self):
        # Model 1: relevance model (same as feed ranker)
        self.relevance_model = FeedRanker()

        # Model 2: notification-specific model
        # P(engage | notification) — not just P(engage | shown in feed)
        self.notify_model = NotifyRanker()

        # Volume controller: per-user budget management
        self.budget_controller = NotifBudgetController()

    def should_notify(self, user, item, context):
        # Step 1: Is this item notification-worthy?
        scores = self.relevance_model.predict(user, item, context)
        notify_score = self.notify_model.predict(user, item, context)

        if notify_score < NOTIFY_THRESHOLD:
            return Decision("suppress", "Below notification threshold")

        # Step 2: Does the user have budget remaining?
        budget = self.budget_controller.get_budget(user.id)
        if budget.daily_remaining <= 0:
            return Decision("suppress", "Daily budget exhausted")

        # Step 3: Is the timing appropriate?
        if not self.is_good_time(user, context):
            return Decision("defer", "Bad timing — will re-evaluate later")

        # Step 4: Is this better than items we might want to notify later?
        # Don't spend the last notification slot on a 0.7 item
        # if a 0.95 item might arrive in the next hour
        if budget.daily_remaining <= 1 and notify_score < HIGH_THRESHOLD:
            return Decision("defer", "Saving last slot for higher-priority")

        # Step 5: Send it
        self.budget_controller.decrement(user.id)
        return Decision("send", f"Score {notify_score:.2f}")

    def is_good_time(self, user, context):
        hour = context.local_hour
        # User's active hours (learned from engagement patterns)
        if hour < user.typical_wake_hour or hour > user.typical_sleep_hour:
            return False
        # Don't stack notifications
        last_notif = self.budget_controller.last_notification_time(user.id)
        if (context.time - last_notif).minutes < 30:
            return False  # minimum 30 min between notifications
        return True`} />
          <div className="space-y-4">
            <Card accent="#ea580c">
              <Label color="#ea580c">Notification Budget Optimization</Label>
              <div className="space-y-2">
                {[
                  { rule: "Per-user daily budget", desc: "Each user gets 3-5 notifications/day. Computed from historical engagement: users who click 80%+ of notifications get higher budget (6-8). Users who click <20% get lower budget (1-2).", color: "#ea580c" },
                  { rule: "Diminishing returns", desc: "1st notification: 12% CTR. 2nd: 10%. 3rd: 7%. 5th: 3%. 10th: 0.5% (and user disables notifications). Model the marginal value of each additional notification.", color: "#dc2626" },
                  { rule: "Category budgets", desc: "Within the daily budget, allocate across categories: max 2 news, max 1 social, max 1 promotional. Prevents one category from dominating all notification slots.", color: "#d97706" },
                  { rule: "Time-of-day optimization", desc: "Send notifications when the user is most likely to engage. Learn per-user active hours from app open patterns. Respect quiet hours (sleep, meetings).", color: "#7c3aed" },
                  { rule: "Competitive items", desc: "If 5 notification-worthy items arrive in 1 hour, pick the best 1-2, not all 5. Compare items against each other, not just against a fixed threshold.", color: "#0f766e" },
                ].map((r,i) => (
                  <div key={i} className="rounded-lg border p-2.5" style={{ borderLeft: `3px solid ${r.color}` }}>
                    <div className="text-[11px] font-bold text-stone-700">{r.rule}</div>
                    <p className="text-[10px] text-stone-500 mt-0.5">{r.desc}</p>
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

function DataModelSection() {
  return (
    <div className="space-y-5">
      <Card accent="#059669">
        <Label color="#059669">Core Data Stores</Label>
        <CodeBlock code={`-- User Profile Store (Bigtable)
user_id -> {
  embedding: float[128],                    # updated daily (batch)
  short_term_interests: [topic_embedding],  # updated hourly (stream)
  followed_topics: [topic_id, ...],
  engagement_history: {
    clicked_items_24h: [item_id],
    dwell_times: {item_id: seconds},
    not_interested: [item_id],
  },
  notification_state: {
    daily_budget: int,
    budget_remaining: int,
    last_notif_ts: timestamp,
    notif_ctr_30d: float,            # personal notification click-through rate
    active_hours: [8, 9, ..., 22],   # learned from app open times
    disabled_categories: [cat_id],
  },
  impression_history: bloom_filter,  # seen items (compact, ~10KB per user)
}

-- Content / Item Store (Bigtable + ANN Index)
item_id -> {
  embedding: float[128],
  title, url, source_id, content_type,
  published_at, indexed_at,
  topics: [topic_id: weight],
  quality_score: float,           # computed at ingest
  engagement: {                   # updated real-time via streaming
    impressions: int,
    clicks: int,
    ctr: float,
    avg_dwell_sec: float,
    share_count: int,
    not_interested_count: int,
    not_interested_rate: float,
  },
  freshness_half_life: float,    # content-type dependent
}

-- Impression / Event Log (Kafka → BigQuery)
(timestamp, user_id, item_id, position, action,
 context: {device, time, connection},
 ranking_score, model_version, features_hash)
# Actions: impression, click, dwell_start, dwell_end,
#          share, not_interested, notification_click, notification_dismiss

-- Notification Log (Spanner)
(user_id, item_id, timestamp, decision,
 score, budget_at_decision, was_clicked, dismiss_reason)`} />
      </Card>
    </div>
  );
}

function TrainingSection() {
  return (
    <div className="space-y-5">
      <Card accent="#7e22ce">
        <Label color="#7e22ce">Training Pipeline — Handling Engagement Data Biases</Label>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Training Data Construction" code={`# Training data from impression logs
# Each (user, item, position, context) → engagement labels

def construct_training_data(logs, date_range):
    examples = []
    for log in logs.where(date=date_range):
        features = {
            # User features (point-in-time)
            **feature_store.get_user_features(log.user_id, log.timestamp),
            # Item features (point-in-time)
            **feature_store.get_item_features(log.item_id, log.timestamp),
            # Cross features
            "user_item_sim": cosine_sim(log.user_emb, log.item_emb),
            "user_topic_affinity": topic_affinity(log.user_id, log.item.topics),
            # Context features
            "hour_of_day": log.context.hour,
            "day_of_week": log.context.day,
            "device_type": log.context.device,
            # Position (for debiasing)
            "position": log.position,
        }

        labels = {
            "clicked": log.clicked,
            "dwell_30s": log.dwell_sec >= 30,
            "satisfied": log.clicked and log.dwell_sec >= 30 and not log.bounced,
            "shared": log.shared,
            "not_interested": log.not_interested,
        }

        # Position bias correction: inverse propensity weighting
        position_propensity = POSITION_CTR_CURVE[log.position]
        weight = 1.0 / position_propensity if log.clicked else 1.0

        examples.append((features, labels, weight))

    return examples

# MULTI-TASK TRAINING:
# loss = w1*BCE(p_click, y_click) + w2*BCE(p_dwell, y_dwell)
#      + w3*BCE(p_satisfy, y_satisfy) - w4*BCE(p_not_interested, y_ni)
# Note: not_interested is a NEGATIVE objective (minimize it)
# w1=1.0, w2=1.0, w3=2.0, w4=0.5 (satisfaction weighted highest)`} />
          <div className="space-y-4">
            <Card accent="#7e22ce">
              <Label color="#7e22ce">Training Challenges & Solutions</Label>
              <div className="space-y-2">
                {[
                  { challenge: "Selection Bias", desc: "We only have labels for items we showed. Items ranked low were rarely seen → no label. The model can't learn about items it never showed.", fix: "Exploration: show 10-15% random items to get unbiased labels. Log and train on exploration traffic separately. Use counterfactual learning (IPS weighting)." },
                  { challenge: "Position Bias", desc: "Items at position 1 get clicked more regardless of quality. Training on raw clicks learns position, not relevance.", fix: "Include position as a training feature, set to 0 at serving. Or use IPW: weight clicks by 1/P(click|position). Pairwise learning cancels position effect." },
                  { challenge: "Popularity Bias", desc: "Popular items get more impressions → more clicks → higher CTR → shown more. Rich-get-richer feedback loop.", fix: "Calibrate CTR by impression count (items with few impressions get wider confidence intervals). Bayesian smoothing. Add exploration slots for low-impression items." },
                  { challenge: "Delayed Feedback", desc: "Satisfaction signals (long dwell, share) arrive minutes to hours after impression. Training on click-only data skews toward clickbait.", fix: "Train on a 24-hour delayed window: join impressions with all engagement events within 24 hours. Accept the data is 24h stale but labels are more complete." },
                ].map((c,i) => (
                  <div key={i} className="rounded-lg border border-stone-200 p-2.5">
                    <div className="text-[11px] font-bold text-stone-700">{c.challenge}</div>
                    <p className="text-[10px] text-stone-500 mt-0.5">{c.desc}</p>
                    <p className="text-[9px] text-violet-600 mt-0.5"><strong>Fix:</strong> {c.fix}</p>
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

function ObjectivesSection() {
  return (
    <div className="space-y-5">
      <Card accent="#0284c7">
        <Label color="#0284c7">Multi-Objective Ranking — The Core L6 Challenge</Label>
        <p className="text-[12px] text-stone-500 mb-4">Feed ranking optimizes multiple competing objectives simultaneously. The art is in how you combine them. This is where L6 candidates differentiate themselves from L5.</p>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Objective Formulation" code={`# Multi-objective scoring function
def compute_final_score(predictions, item, context, weights):
    """
    Combine multiple objectives into a single ranking score.
    This is THE most important design decision in feed ranking.
    """
    # Engagement objectives (short-term)
    engagement = (
        weights["click"] * predictions["p_click"]
        + weights["dwell"] * predictions["p_dwell_30s"]
        + weights["share"] * predictions["p_share"]
    )

    # Satisfaction objectives (long-term proxy)
    satisfaction = (
        weights["satisfy"] * predictions["p_satisfied"]
        - weights["not_interested"] * predictions["p_not_interested"]
    )

    # Content quality (editorial signal)
    quality = (
        weights["source_trust"] * item.source_trust_score
        + weights["content_quality"] * item.quality_score
    )

    # Freshness (time-decayed)
    age_hours = (context.time - item.published_at).hours
    half_life = item.freshness_half_life  # news=4h, article=48h
    freshness = math.exp(-0.693 * age_hours / half_life)

    # Combined score
    score = (
        0.35 * engagement
        + 0.35 * satisfaction
        + 0.15 * quality
        + 0.15 * freshness * weights["freshness_boost"]
    )

    return score

# WEIGHT TUNING:
# These weights are NOT trained — they're TUNED via online A/B tests.
# Example experiments:
# Increase satisfaction weight → user retention improves
# Increase engagement weight → clicks go up, retention goes down
# Increase freshness weight → feed feels more "current"
# Decrease quality weight → clickbait increases
# The optimal weights balance short-term engagement with
# long-term retention and user satisfaction.`} />
          <div className="space-y-4">
            <Card accent="#0284c7">
              <Label color="#0284c7">Objective Conflicts & Resolutions</Label>
              <div className="space-y-2">
                {[
                  { obj1: "Click ↑", obj2: "Satisfaction ↑", conflict: "Clickbait has high CTR but low satisfaction. Sensational headlines get clicks but users feel misled.", resolution: "Combine click with dwell time. Clicks with <10s dwell are 'bad clicks'. Train satisfaction as a separate objective and weight it equally." },
                  { obj1: "Engagement ↑", obj2: "Diversity ↑", conflict: "The most engaging feed shows 20 items on the user's top interest. But that's boring and creates filter bubbles.", resolution: "Hard diversity constraints in re-ranking (max 3 per topic in top 20). Accept ~5% engagement loss for better diversity. Long-term retention improves." },
                  { obj1: "Freshness ↑", obj2: "Quality ↑", conflict: "Fresh content has no engagement data (cold start). Proven content has high engagement signals. Over-indexing on freshness shows unvetted content.", resolution: "Exploration slots: 10-15% of feed is fresh content. Use content quality score (pre-engagement) to filter. Bayesian smoothing for new items." },
                  { obj1: "Notif volume ↑", obj2: "User retention ↑", conflict: "More notifications = more immediate engagement. Too many = user disables notifications = permanent loss.", resolution: "Per-user volume optimization. Model the marginal value of each notification. The Nth notification has diminishing returns. Stop when marginal value < marginal annoyance." },
                ].map((c,i) => (
                  <div key={i} className="rounded-lg border border-stone-200 p-2.5">
                    <div className="flex items-center gap-1.5 mb-1">
                      <Pill bg="#dbeafe" color="#2563eb">{c.obj1}</Pill>
                      <span className="text-stone-400 text-[10px]">vs</span>
                      <Pill bg="#dcfce7" color="#059669">{c.obj2}</Pill>
                    </div>
                    <p className="text-[10px] text-stone-500">{c.conflict}</p>
                    <p className="text-[10px] text-cyan-700 mt-1"><strong>Resolution:</strong> {c.resolution}</p>
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

function ScalabilitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Ranking Infrastructure Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Pre-rank eliminates 95% of candidates</strong> — two-tower dot product scores 10K items in ~2ms. Only 500 reach the expensive full ranker. This funnel architecture is how you score 930M items/sec platform-wide.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Embedding pre-computation</strong> — item embeddings computed once at publish/update time, not at query time. User embeddings updated in batch (daily) + streaming (hourly short-term interests). ANN index updated incrementally.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Model distillation for pre-ranker</strong> — the full ranker is a 100M-param deep model. Distill to a 1M-param model for pre-ranking. 100x faster with ~3% quality loss. The pre-ranker doesn't need to be perfect — it just needs to not miss good items.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Regional serving with CDN caching</strong> — for trending/popular candidates, cache the candidate list at the CDN level (TTL: 5 minutes). Only personalized components need per-request computation.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Feature Store at Feed Scale</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Batch user features (daily)</strong> — user embeddings, long-term interests, engagement statistics. Computed via Spark/Beam jobs. Materialized to Bigtable. Stable features that don't need real-time updates.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Streaming item features (real-time)</strong> — item CTR, dwell time, share count. Updated per-event via Flink/Dataflow streaming pipeline. Available within seconds of new engagement. Critical for cold-start items.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Impression deduplication</strong> — bloom filter per user (10KB each) tracking seen items. Checked at candidate generation to avoid re-showing. 1B users × 10KB = 10TB bloom filter store. Sharded by user_id.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Notification state tracking</strong> — per-user notification budget, last notification time, CTR history. Stored in Redis for sub-ms reads. Updated synchronously on every notification send/dismiss.</Point>
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
          { title: "Filter Bubble / Echo Chamber", sev: "HIGH", desc: "The system learns user preferences and shows more of the same. User's worldview narrows. For news feeds, this has societal implications (political polarization). The feedback loop is self-reinforcing: user clicks sports → model shows more sports → user has no opportunity to see tech → model 'learns' user doesn't like tech.", fix: "Exploration slots (10-15% of feed). Topic diversity constraints in re-ranking. Periodically probe user with diverse content. Monitor interest breadth over time — alert if a user's topic distribution narrows. Explicit 'Discover new topics' feature.", icon: "🟡" },
          { title: "Notification Fatigue → Disable", sev: "CRITICAL", desc: "Over-notification causes users to disable push notifications entirely. This is IRREVERSIBLE — most users never re-enable. Even 1% notification-disable rate means losing 10M users permanently. The lost engagement is never recovered.", fix: "Per-user notification budget (3-5/day). Model notification marginal value — each additional notification has diminishing returns. Hard guardrail: if user's notification CTR drops below 3% over 7 days, REDUCE budget automatically. Monitor disable rate as P0 metric. Recovery: re-permission prompt after 30 days (but success rate is <5%).", icon: "🔴" },
          { title: "Clickbait Optimization", sev: "HIGH", desc: "Optimizing for clicks alone leads to sensationalized content rising. Users click but feel manipulated. Short-term metrics improve, long-term retention drops. Classic Goodhart's law: 'When a measure becomes a target, it ceases to be a good measure.'", fix: "Multi-objective: P(satisfied) includes dwell time + not-interested rate. Clicks with <10s dwell are 'bad clicks' — penalize in training. Content quality signals from editorial review. Source authority scoring. A/B test with retention as the primary metric, not CTR.", icon: "🟡" },
          { title: "Cold-Start for New Content", sev: "MEDIUM", desc: "New content has zero engagement signals. The model has no click/dwell data. If the model only trusts engagement signals, new content never gets shown → never gets engagement data → permanent cold start.", fix: "Content-based features (quality score, source authority, topic relevance) that work without engagement. Exploration slots guarantee new content gets impressions. Bayesian smoothing: treat new items as having a prior CTR (e.g., category average). Thompson sampling for explore/exploit.", icon: "🟠" },
          { title: "Position Bias Feedback Loop", sev: "MEDIUM", desc: "Model trained on click data where position 1 is clicked most. Model learns to predict 'items that were at position 1' rather than 'items users actually prefer'. Self-reinforcing: items ranked high get more clicks → model ranks them higher → more clicks.", fix: "Position debiasing in training (IPW or position as feature with dropout). Randomized experiments: occasionally shuffle positions to get unbiased data. Pairwise learning: learn relative preferences, not absolute click probabilities.", icon: "🟠" },
          { title: "Stale User Profiles", sev: "MEDIUM", desc: "User interests change over time. A user who loved cooking last month may have shifted to gardening. If the user embedding updates only daily, the feed lags behind interest shifts. User sees irrelevant content for hours after an interest change.", fix: "Dual-time-scale interests: long-term embedding (updated daily) + short-term embedding (updated hourly from recent clicks). The ranking model takes both as input. Recent interactions weight more heavily. Explicit 'Not interested' signal immediately adjusts short-term embedding.", icon: "🟠" },
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
          { title: "Sequence-Aware Ranking", d: "Model the user's session as a sequence: what did they engage with in the last 5 minutes? Use transformer to model session context for next-item prediction.", effort: "Hard", detail: "The user's in-session behavior is the strongest short-term signal. Clicking a sports article → the next sports article scores higher. Transformers capture these sequential dependencies naturally." },
          { title: "Contextual Bandits for Exploration", d: "Replace epsilon-greedy exploration with contextual bandits (LinUCB, Thompson Sampling). Learn WHICH users benefit from WHICH types of exploration.", effort: "Medium", detail: "Not all users want exploration equally. Power users with strong preferences want less exploration. New users benefit from more. Contextual bandits personalize the exploration rate." },
          { title: "LLM-Powered Content Understanding", d: "Use LLMs to generate rich content features: topic classification, sentiment, reading level, content quality assessment, abstractive summaries.", effort: "Medium", detail: "LLM features at indexing time (offline) provide much richer content understanding than simple NLP. Helps especially for cold-start items where engagement data is missing." },
          { title: "Causal Inference for Long-Term Impact", d: "Measure the causal effect of feed decisions on long-term retention, not just immediate engagement. Did showing more diverse content actually cause higher 30-day retention?", effort: "Hard", detail: "Standard A/B tests measure short-term metrics. Causal inference methods (double ML, instrumental variables) estimate long-term effects from observational data." },
          { title: "User Controllability", d: "Let users directly influence their feed: 'Show more like this', 'Show less from this source', topic sliders, content type preferences.", effort: "Medium", detail: "Users who feel in control have higher satisfaction and retention. User preferences become explicit features in the ranking model. Challenge: most users never touch controls — defaults must be good." },
          { title: "Cross-Surface Ranking", d: "Unified ranking across surfaces: what you see in the feed affects what notification you receive and vice versa. If the item was already shown in the feed, don't notify.", effort: "Hard", detail: "Requires a cross-surface impression log and coordination between the feed ranker and notification system. Prevents redundancy and optimizes total user attention across surfaces." },
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
    { q:"How do you balance short-term engagement vs long-term retention?", a:"This is THE fundamental tension in feed ranking. Short-term: optimize clicks and dwell time today. Long-term: optimize 30-day retention and user satisfaction. They conflict because clickbait maximizes short-term clicks but erodes trust and causes churn. Approach: (1) Multi-objective training: explicitly model P(satisfied) alongside P(click). Satisfaction is measured by long dwell + no 'not interested' signal + return visits. (2) Weight tuning via long-term A/B tests: run experiments for 4+ weeks and measure retention, not just clicks. A model that increases clicks by 5% but decreases 30-day retention by 1% is a NET NEGATIVE. (3) Guardrail metrics: set hard floors on satisfaction and 'not interested' rate. Any model that violates these guardrails is rolled back, even if clicks increase. (4) Content quality signals: editorially-assessed quality scores that don't depend on engagement. Boost quality content even if predicted CTR is lower.", tags:["objectives"] },
    { q:"How do you handle notification timing across time zones?", a:"Notification timing is crucial — a 3am notification is annoying regardless of content quality. Approach: (1) Learn per-user active hours from app open patterns. Most users have predictable windows (e.g., 7-9am, 12-1pm, 6-10pm). Store as a distribution. (2) Calendar integration (optional): if the user is in a meeting, defer notifications. (3) Do Not Disturb respect: honor OS-level DND settings. (4) Time-zone-aware scheduling: convert all timing decisions to local time. A 'morning news digest' notification should arrive at 7:30am LOCAL time, not UTC. (5) Activity signals: if the user's device reports screen-off for 2+ hours, they're likely sleeping. Defer until screen-on event. (6) Batching: instead of 3 individual notifications over 2 hours, batch into a single 'digest' notification with 3 items. Less interruptive, same engagement.", tags:["notifications"] },
    { q:"How do you measure feed quality beyond clicks?", a:"Clicks are a weak signal — users click clickbait too. Better metrics: (1) Satisfied clicks: click + dwell > 30 seconds + no bounce-back. This measures genuinely useful engagement. (2) Not-interested rate: how often users explicitly signal 'not interested'. Lower = better feed. (3) Session depth: how many items does the user consume per session? Deeper sessions = more satisfying feed. (4) Return rate: does the user come back tomorrow? The ultimate satisfaction signal. (5) Notification click-through rate: are notifications driving genuine engagement? (6) Interest breadth: is the user discovering new topics or stuck in a filter bubble? (7) User-reported satisfaction: periodic in-app surveys ('How would you rate your feed today? 1-5'). Expensive to collect but the most direct signal. (8) Long-term retention: 7-day and 30-day active rate. The north star metric.", tags:["evaluation"] },
    { q:"How does the feed handle breaking news vs personalization?", a:"Breaking news creates a tension: it's relevant to EVERYONE (or a large segment), overriding personal interests. Approach: (1) Breaking news detection: content with rapidly accelerating engagement (clicks/minute > 10x baseline for that topic) is flagged as 'breaking'. (2) Override personalization for truly breaking news: if a major event occurs, inject it into ALL feeds regardless of personal interests. This is editorial-level importance, not ML-predicted relevance. (3) Tiered breaking: 'global breaking' (earthquake, election result) goes to everyone. 'Topic breaking' (major sports trade) goes to users with that topic interest. (4) Freshness decay: breaking news has a very short half-life (1-2 hours). After the initial surge, it decays rapidly to make room for new content. (5) Notification: breaking news may override normal notification budget — it's one of the few cases where sending an 'extra' notification is justified.", tags:["content"] },
    { q:"How do you prevent the model from overfitting to power users?", a:"Power users (top 10%) generate 50%+ of engagement data. Training on raw logs makes the model optimize for power user behavior, which may not generalize. Solutions: (1) Sample balancing: downsample power user events in training data so each user contributes roughly equally. (2) User-level stratification: ensure evaluation metrics are computed per-user then averaged, not per-event. Per-event metrics are dominated by power users. (3) Separate models: different ranking weights for new users (more exploration, less personalization) vs power users (deep personalization). (4) Cold-start handling: new users get a 'popular + diverse' feed until enough engagement data accumulates for personalization. (5) Engagement normalization: normalize features like 'user click count' by user activity level. A casual user clicking 3 items is equivalent to a power user clicking 30.", tags:["ml"] },
    { q:"What's the difference between feed ranking and search ranking?", a:"Key differences: (1) Intent: search has an explicit query ('best pizza nearby'). Feed has no query — the system must INFER what the user wants right now. Feed ranking = proactive intent prediction. (2) Candidate source: search retrieves from a corpus matching the query. Feed retrieves from ALL content, filtered by relevance + user interests. Much larger candidate space. (3) Ordering signal: search can use query-document relevance (BM25, semantic match). Feed uses user-item affinity (collaborative filtering, interest matching). (4) Freshness: search results can be stable (same query → similar results). Feeds must feel fresh every time — same items shouldn't reappear. (5) Diversity importance: search diversity matters (show different restaurants). Feed diversity is CRITICAL (topic, source, format variety). (6) Position: search users scan top 3-5 results. Feed users scroll 20-50 items. Feed must maintain quality deeper.", tags:["design"] },
    { q:"How do you A/B test notification volume changes?", a:"Notification experiments are uniquely risky because the cost of over-notification is permanent (user disables). Approach: (1) Small treatment groups: start with 1% of users in the treatment (e.g., increase budget from 4 to 6 notifications/day). Monitor disable rate daily. (2) Stratified sampling: ensure treatment/control have similar user profiles (activity level, current notification CTR, tenure). (3) Holdout period: run for 2+ weeks. Notification effects are delayed — a user might tolerate extra notifications for a few days before disabling. (4) Guardrail metrics: auto-stop if disable rate in treatment exceeds control by >0.05%. This is the hardest guardrail — once a user disables, they're gone. (5) Asymmetric risk: over-notification is much worse than under-notification. When uncertain, err on the side of fewer notifications. (6) Measure total engagement, not just notification CTR. More notifications might increase notification clicks but decrease organic app opens (users wait for notifications instead of checking the app).", tags:["experimentation"] },
    { q:"Should the notification model be separate from the feed ranking model?", a:"Yes — separate models for separate decisions. The feed ranker predicts: P(engage | item shown in feed). The notification model predicts: P(engage | item pushed as notification). These are fundamentally different: (1) Different baselines: feed items have ~5% CTR (user is already in the app, browsing). Notifications have ~8-12% CTR (user is interrupted, higher bar for quality). (2) Different costs: a bad feed item at position 30 is invisible. A bad notification is an interruption — much higher cost. (3) Different features: notification model needs timing features (time of day, last notification time, daily budget). Feed ranker doesn't. (4) Different training data: notification model trains on notification impressions + clicks. Feed ranker trains on feed impressions + clicks. (5) Shared components: both can share the underlying content quality model and user interest model. The notification model adds a notification-specific head. This is a multi-task architecture with shared bottom + task-specific towers.", tags:["architecture"] },
  ];
  return (
    <div className="space-y-2.5">
      <Card className="bg-indigo-50/50 border-indigo-200 mb-1">
        <p className="text-[12px] text-stone-500">Common follow-up questions Google L6 interviewers ask about feed ranking and notifications. Click to reveal a strong answer.</p>
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
  api: ApiSection, design: DesignSection, candidates: CandidatesSection,
  ranking: RankingSection, features: FeaturesSection, diversity: DiversitySection,
  notifications: NotificationsSection, data: DataModelSection, training: TrainingSection,
  objectives: ObjectivesSection, scalability: ScalabilitySection,
  watchouts: WatchoutsSection, enhancements: EnhancementsSection,
  followups: FollowupsSection,
};

export default function FeedRankingSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Notification / Feed Ranking</h1>
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