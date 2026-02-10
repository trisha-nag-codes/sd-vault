import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   ADS CLICK PREDICTION (CTR) — ML System Design Reference (Google L6)
   Pearl white theme · 17 sections (HLD + LLD + ML Deep Dive)
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",              icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",          icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",   icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",            icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",     icon: "🏗️", color: "#9333ea" },
  { id: "auction",       label: "Ad Auction Deep Dive",  icon: "🏷️", color: "#c026d3" },
  { id: "ranking",       label: "CTR Model",             icon: "🧠", color: "#dc2626" },
  { id: "features",      label: "Feature Engineering",   icon: "⚙️", color: "#d97706" },
  { id: "calibration",   label: "Calibration & Bidding", icon: "🎯", color: "#0f766e" },
  { id: "data",          label: "Data Model",            icon: "🗄️", color: "#059669" },
  { id: "training",      label: "Training Pipeline",     icon: "🔄", color: "#7e22ce" },
  { id: "serving",       label: "Real-Time Serving",     icon: "⚡", color: "#ea580c" },
  { id: "scalability",   label: "Scalability",           icon: "📈", color: "#059669" },
  { id: "watchouts",     label: "Failure Modes",         icon: "⚠️", color: "#dc2626" },
  { id: "observability", label: "Observability",         icon: "📊", color: "#0284c7" },
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


/* ═══════════════════════════════════════════════════════════
   SECTIONS
   ═══════════════════════════════════════════════════════════ */

function ConceptSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-7 space-y-5">
          <Card accent="#6366f1">
            <Label>What is Ads Click Prediction?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              Ads click prediction estimates the probability a user will click on a specific ad in a specific context (query, page, time). This predicted CTR (pCTR) is the linchpin of the ad auction — it determines which ads are shown, in what order, and how much advertisers pay. It directly drives <strong>Google's $240B+ annual ad revenue</strong>.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              The fundamental challenge: for every search query or page view, evaluate hundreds of candidate ads and predict click probability for each — in under <strong>10ms</strong>. The prediction must be <em>well-calibrated</em> (if you predict 2% CTR, exactly 2% of impressions should result in clicks) because the auction charges advertisers based on this prediction.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard? (Google L6 Depth)</Label>
            <ul className="space-y-2.5">
              <Point icon="💰" color="#0891b2">Calibration is existential — over-predicting CTR overcharges advertisers (they leave). Under-predicting loses revenue. A 1% calibration error at Google's scale = billions of dollars.</Point>
              <Point icon="⚡" color="#0891b2">Latency is brutal — the entire ad selection pipeline (retrieve candidates, predict CTR, run auction, select winner) must complete in &lt;10ms. Every extra ms = measurable revenue loss.</Point>
              <Point icon="📊" color="#0891b2">Extreme sparsity — most ads are rarely shown. Most user-ad pairs are never observed. Feature space is enormous (billions of ad IDs × millions of query patterns × user features).</Point>
              <Point icon="🔄" color="#0891b2">Non-stationarity — CTR changes with time of day, season, world events, ad fatigue, competitive landscape. Models trained on yesterday may be wrong today.</Point>
              <Point icon="⚔️" color="#0891b2">Adversarial — click fraud (bots clicking competitor ads), impression fraud, ad content manipulation. System must be robust to deliberate gaming.</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Scale</Label>
            <div className="space-y-2.5">
              {[
                { co: "Google Ads", scale: "$240B/yr revenue, 8.5B queries/day", approach: "Deep & cross networks + online learning" },
                { co: "Meta Ads", scale: "$135B/yr, 2B+ DAU", approach: "DLRM (embedding + MLP)" },
                { co: "Amazon Ads", scale: "$47B/yr, product search ads", approach: "Multi-task deep learning" },
                { co: "TikTok Ads", scale: "$20B/yr, in-feed native ads", approach: "MMoE multi-objective" },
                { co: "LinkedIn Ads", scale: "$6B/yr, B2B targeting", approach: "GBDT + deep hybrid" },
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
            <Label color="#2563eb">The Ad Serving Pipeline (Preview)</Label>
            <svg viewBox="0 0 360 170" className="w-full">
              <rect x={10} y={10} width={80} height={35} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
              <text x={50} y={30} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="700" fontFamily="monospace">User Query</text>

              <rect x={110} y={10} width={80} height={35} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
              <text x={150} y={25} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="700" fontFamily="monospace">Ad Retrieval</text>
              <text x={150} y={38} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">100K → 1000</text>

              <rect x={210} y={10} width={80} height={35} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
              <text x={250} y={25} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="700" fontFamily="monospace">CTR Prediction</text>
              <text x={250} y={38} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">score 1000 ads</text>

              <rect x={120} y={60} width={100} height={35} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
              <text x={170} y={75} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="700" fontFamily="monospace">Auction (GSP/VCG)</text>
              <text x={170} y={88} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">rank = pCTR × bid</text>

              <rect x={240} y={60} width={80} height={35} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
              <text x={280} y={75} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="700" fontFamily="monospace">Ad Selection</text>
              <text x={280} y={88} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">top 3-4 ads</text>

              <line x1={90} y1={28} x2={110} y2={28} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-cv)"/>
              <line x1={190} y1={28} x2={210} y2={28} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-cv)"/>
              <line x1={250} y1={45} x2={200} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-cv)"/>
              <line x1={220} y1={78} x2={240} y2={78} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-cv)"/>
              <defs><marker id="ah-cv" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>

              <rect x={20} y={115} width={310} height={45} rx={6} fill="#faf9f7" stroke="#e7e5e4"/>
              <text x={30} y={132} fill="#dc2626" fontSize="7" fontWeight="600" fontFamily="monospace">pCTR must be CALIBRATED — auction correctness depends on it</text>
              <text x={30} y={145} fill="#059669" fontSize="7" fontWeight="600" fontFamily="monospace">Revenue = sum(pCTR_next × bid_next) for each click — Vickrey pricing</text>
              <text x={30} y={155} fill="#78716c" fontSize="7" fontFamily="monospace">Total pipeline latency budget: &lt;10ms including network</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Google Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Google's revenue engine — extremely likely at L6</div>
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
            <p className="text-[12px] text-stone-500 mt-0.5">"Design an ads CTR prediction system" should be scoped proactively: "I'll design the CTR prediction pipeline for search ads — from query to predicted click probability. I'll cover the model architecture, feature engineering, real-time serving, and calibration. I'll treat ad retrieval/targeting as a black box and focus on the prediction + auction integration." Mention calibration early — it signals you understand what makes ads different from general ranking.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Given (user, query, ad, context), predict P(click) — a well-calibrated probability</Point>
            <Point icon="2." color="#059669">Score 100-1000 candidate ads per request in real-time</Point>
            <Point icon="3." color="#059669">Support multiple ad formats: search ads, display ads, video pre-roll</Point>
            <Point icon="4." color="#059669">Integrate with auction: ranking score = pCTR × bid (ad rank formula)</Point>
            <Point icon="5." color="#059669">Support advertiser-defined targeting constraints (geo, device, demographics, keywords)</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Prediction latency: p50 &lt;5ms, p99 &lt;10ms (this is inside the ad serving SLA)</Point>
            <Point icon="2." color="#dc2626">Throughput: 100K+ ad requests/sec (each scoring 100-1000 ads)</Point>
            <Point icon="3." color="#dc2626">Calibration error: &lt;2% (predicted CTR must match observed CTR)</Point>
            <Point icon="4." color="#dc2626">Availability: 99.99% — if ads don't show, revenue stops immediately</Point>
            <Point icon="5." color="#dc2626">Model freshness: update within hours, ideally real-time online learning</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask (L6 Signal)</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Search ads (query context) or display ads (page context)? Architecture differs significantly.",
            "What auction mechanism? GSP (Google) or VCG? Affects calibration requirements.",
            "Do we predict P(click) only, or also P(conversion)? Multi-stage funnel?",
            "What's our feature budget? Can we use real-time user signals or batch-only?",
            "Single model or per-vertical models? (Shopping, Travel, Local have different CTR patterns)",
            "Online learning requirement? How fast must the model adapt to new ads?",
            "Click fraud rate? Need adversarial robustness built in?",
            "Privacy constraints? Can we use user-level features or only aggregate?",
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
            <div className="text-[12px] font-bold text-violet-700">Interview Tip — Revenue-Driven Capacity</div>
            <p className="text-[12px] text-stone-500 mt-0.5">For ads systems, capacity planning is revenue-driven. Every millisecond of latency and every percent of downtime has a dollar cost. Frame your estimates in terms of revenue impact, not just QPS.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Traffic Volume</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Search queries/day = 8.5B" result="8.5B" note="Google Search. ~60% have ads eligible." />
            <MathStep step="2" formula="Ad-eligible queries = 8.5B x 0.6" result="5.1B/day" note="Some queries are too short, navigational, or non-commercial." />
            <MathStep step="3" formula="Ad requests/sec = 5.1B / 86,400" result="~60K QPS" note="Each request scores 100-1000 candidate ads." final />
            <MathStep step="4" formula="Peak QPS = 60K x 3" result="~180K QPS" note="Holiday shopping, major events, seasonal peaks." />
            <MathStep step="5" formula="Ads scored/sec = 60K x 500 avg" result="~30M scores/sec" note="This is the model inference throughput requirement." final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Ad Corpus & Feature Scale</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Active advertisers" result="~10M" note="Google Ads has millions of active advertisers." />
            <MathStep step="2" formula="Active ad creatives" result="~1B" note="Each advertiser has multiple campaigns/ad groups/creatives." />
            <MathStep step="3" formula="Keyword targets" result="~10B" note="Keyword × match type combinations." />
            <MathStep step="4" formula="Feature embedding table (ad IDs)" result="~100 GB" note="1B ad IDs × 64-dim × 4B per float. Sharded across servers." final />
            <MathStep step="5" formula="User feature table" result="~500 GB" note="2B users × ~250 bytes of features. In feature store." />
            <MathStep step="6" formula="Click logs for training" result="~50B events/day" note="Impressions + clicks + conversions. ~30 day retention." />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Latency Budget Breakdown</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Total ad serving budget" result="<10ms" note="From query arrival to ad selection. The tightest latency in all of ML." final />
            <MathStep step="2" formula="Ad retrieval (inverted index)" result="~2ms" note="Find candidate ads matching query keywords." />
            <MathStep step="3" formula="Feature lookup (feature store)" result="~2ms" note="Batch-fetch user + ad + context features." />
            <MathStep step="4" formula="CTR model inference" result="~3ms" note="Score 500 ads in one batched forward pass." />
            <MathStep step="5" formula="Auction + pricing + selection" result="~1ms" note="Sort by ad_rank, compute Vickrey prices, apply budget." />
            <MathStep step="6" formula="Serialization + response" result="~1ms" note="Build ad response with creative URLs, tracking pixels." />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Revenue Impact Analysis</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Google ads revenue/year" result="$240B" note="~$660M per day, ~$7,600 per second." />
            <MathStep step="2" formula="Revenue per ad request" result="~$0.013" note="$240B / (5.1B x 365). Not every query generates a click." />
            <MathStep step="3" formula="Revenue loss per 1ms latency" result="~$76/sec" note="1% drop per 100ms latency → 0.01% per 1ms × $7,600/sec." final />
            <MathStep step="4" formula="Revenue loss per 1% downtime" result="~$2.4B/yr" note="99% uptime loses $2.4B. 99.99% = $24M. Still a lot." />
            <MathStep step="5" formula="Revenue impact of 1% calibration error" result="~$2.4B" note="Systematic over/under-pricing of ads. Advertisers adjust or leave." final />
          </div>
        </Card>
      </div>
      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Ad Request QPS", val: "~60K", sub: "Peak: 180K" },
            { label: "Ads Scored/sec", val: "~30M", sub: "500 ads × 60K req" },
            { label: "Latency Budget", val: "<10ms", sub: "Tightest in all of ML" },
            { label: "Revenue/sec", val: "$7,600", sub: "Every ms counts" },
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
          <Label color="#2563eb">Ad Serving API (Internal)</Label>
          <CodeBlock code={`# Internal API — called by the search serving stack
# NOT a public API. Called per search query that's ad-eligible.
#
# RPC: AdService.GetAds(AdRequest) -> AdResponse
#
# AdRequest:
{
  "query": "cheap flights to hawaii",
  "user_context": {
    "user_id": "u_abc123",      // hashed, may be anonymous
    "country": "US",
    "language": "en",
    "device": "mobile",
    "browser": "chrome",
    "geo": {"lat": 37.7, "lng": -122.4}
  },
  "page_context": {
    "surface": "search",         // search | youtube | display
    "position_count": 4,         // max ad slots available
    "organic_results_count": 10
  },
  "request_id": "req_xyz789",
  "timestamp": "2024-02-10T14:30:00Z"
}

# AdResponse:
{
  "ads": [
    {
      "ad_id": "ad_12345",
      "creative": {
        "headline": "Flights to Hawaii from $199",
        "description": "Book now. Best prices...",
        "display_url": "flightsite.com/hawaii",
        "click_url": "https://track.../click?ad=12345"
      },
      "position": 1,
      "predicted_ctr": 0.034,    // internal — NOT exposed
      "ad_rank_score": 1.87,     // pCTR * bid * quality
      "price_per_click": 2.15,   // what advertiser pays if clicked
      "advertiser_id": "adv_789",
      "campaign_id": "camp_456"
    }
  ],
  "request_id": "req_xyz789",
  "latency_ms": 7.2
}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Design Decisions (L6 Depth)</Label>
          <div className="space-y-3">
            {[
              { q: "Why is pCTR not exposed to advertisers?", a: "Exposing pCTR would let advertisers game the system. If they know their predicted CTR, they can manipulate ad copy to inflate it (clickbait) without improving landing page quality. Also, pCTR is part of the auction's competitive advantage." },
              { q: "Why include price_per_click in response?", a: "The ad server computes Vickrey pricing (you pay the minimum needed to maintain your position). This is computed during the auction, not by the CTR model. The price is sent to the click tracker so it can charge the advertiser on click." },
              { q: "Why RPC instead of REST?", a: "This is a latency-critical internal service. gRPC with protobuf is 2-5x faster than REST/JSON for serialization. Every microsecond matters when the total budget is 10ms. Binary protocol, connection pooling, streaming all help." },
              { q: "How are anonymous users handled?", a: "~30% of traffic is logged-out. Use contextual features only: query, device, geo, time. No personalization. A separate model variant may be used for anonymous users. CTR is lower for anonymous users (less relevant ads), so floor prices may differ." },
              { q: "Why pass organic_results_count?", a: "The number of organic results affects ad click probability. If organic results are highly relevant (navigational query), ad CTR drops. The model uses this as a feature. Also affects ad slot allocation (fewer ads on highly-satisfied queries)." },
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
        <Label color="#9333ea">Full System Architecture — Ad Serving Pipeline</Label>
        <svg viewBox="0 0 720 340" className="w-full">
          <defs><marker id="ah-hld" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>

          {/* Query */}
          <rect x={10} y={55} width={65} height={36} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={42} y={71} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Query</text>
          <text x={42} y={83} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">user ctx</text>

          {/* Ad Retrieval */}
          <rect x={100} y={45} width={90} height={50} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={145} y={65} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">Ad Retrieval</text>
          <text x={145} y={78} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">keyword match</text>
          <text x={145} y={88} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">targeting filter</text>

          {/* Feature Assembly */}
          <rect x={215} y={45} width={90} height={50} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={260} y={65} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Feature</text>
          <text x={260} y={78} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Assembly</text>
          <text x={260} y={90} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">~2ms</text>

          {/* CTR Model */}
          <rect x={330} y={40} width={100} height={55} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={380} y={60} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">CTR Model</text>
          <text x={380} y={73} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">deep & cross net</text>
          <text x={380} y={86} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">~3ms (batched)</text>

          {/* Calibration */}
          <rect x={455} y={45} width={80} height={50} rx={6} fill="#0f766e10" stroke="#0f766e" strokeWidth={1.5}/>
          <text x={495} y={65} textAnchor="middle" fill="#0f766e" fontSize="9" fontWeight="600" fontFamily="monospace">Calibrate</text>
          <text x={495} y={78} textAnchor="middle" fill="#0f766e80" fontSize="7" fontFamily="monospace">isotonic reg</text>
          <text x={495} y={88} textAnchor="middle" fill="#0f766e80" fontSize="7" fontFamily="monospace">Platt scaling</text>

          {/* Auction */}
          <rect x={560} y={45} width={80} height={50} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={600} y={65} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Auction</text>
          <text x={600} y={78} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">GSP/VCG</text>
          <text x={600} y={88} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">pricing</text>

          {/* Result */}
          <rect x={665} y={55} width={45} height={36} rx={6} fill="#6366f110" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={687} y={71} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">Ads</text>
          <text x={687} y={83} textAnchor="middle" fill="#6366f180" fontSize="7" fontFamily="monospace">top K</text>

          {/* Data stores */}
          <rect x={100} y={150} width={80} height={35} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={140} y={171} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Ad Index</text>

          <rect x={215} y={150} width={80} height={35} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={255} y={171} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Feature Store</text>

          <rect x={330} y={150} width={80} height={35} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={370} y={171} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Model Server</text>

          <rect x={455} y={150} width={85} height={35} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={497} y={171} textAnchor="middle" fill="#0891b2" fontSize="8" fontWeight="600" fontFamily="monospace">Budget Manager</text>

          {/* Arrows */}
          <line x1={75} y1={72} x2={100} y2={70} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={190} y1={70} x2={215} y2={70} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={305} y1={70} x2={330} y2={68} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={430} y1={68} x2={455} y2={68} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={535} y1={70} x2={560} y2={70} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={640} y1={72} x2={665} y2={72} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>

          {/* Data connections */}
          <line x1={140} y1={95} x2={140} y2={150} stroke="#c026d340" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>
          <line x1={260} y1={95} x2={255} y2={150} stroke="#d9770640" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>
          <line x1={380} y1={95} x2={370} y2={150} stroke="#dc262640" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>
          <line x1={600} y1={95} x2={497} y2={150} stroke="#0891b240" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>

          {/* Legend */}
          <rect x={15} y={210} width={695} height={115} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={25} y={230} fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 1 — Ad Retrieval (~2ms): Match query to advertiser keywords. Apply targeting filters (geo, device, language, budget).</text>
          <text x={25} y={247} fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 2 — Feature Assembly (~2ms): Batch-fetch user features, ad features, query features, cross features from feature store.</text>
          <text x={25} y={264} fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 3 — CTR Prediction (~3ms): Deep model scores all candidates in batched inference. Outputs raw logit per (user, query, ad).</text>
          <text x={25} y={281} fill="#0f766e" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 4 — Calibration (~0.5ms): Transform raw scores to calibrated probabilities. Critical for auction correctness.</text>
          <text x={25} y={298} fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 5 — Auction (~1ms): Rank by ad_rank = pCTR x bid x quality_score. Vickrey pricing. Budget check. Select top K.</text>
          <text x={25} y={315} fill="#78716c" fontSize="8" fontWeight="600" fontFamily="monospace">Key: Unlike search ranking, calibration is mandatory. The auction CHARGES based on pCTR. Wrong pCTR = wrong prices.</text>
        </svg>
      </Card>
    </div>
  );
}

function AuctionSection() {
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Ad Auction Mechanics — Why Calibration Matters</Label>
        <p className="text-[12px] text-stone-500 mb-4">The ad auction is a Generalized Second-Price (GSP) auction. Advertisers bid, but they pay the price of the next-highest bidder. The ranking is based on <strong>ad_rank = pCTR × bid × quality_score</strong>, not just bid alone. This is why calibrated pCTR is critical.</p>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Ad Rank Calculation" code={`# Ad Rank Formula (Google Ads)
def compute_ad_rank(ad, predicted_ctr, quality_score):
    ad_rank = (
        predicted_ctr           # calibrated P(click)
        * ad.max_bid            # advertiser's max CPC bid
        * quality_score         # ad quality (landing page, relevance)
        + ad_format_bonus       # sitelinks, call extensions, etc.
    )
    return ad_rank

# Generalized Second-Price (GSP) Auction
def run_auction(candidates, max_slots=4):
    # Sort by ad_rank descending
    ranked = sorted(candidates, key=lambda a: -a.ad_rank)

    results = []
    for i in range(min(max_slots, len(ranked))):
        ad = ranked[i]
        # Price = minimum bid to maintain position
        # (Vickrey: you pay next bidder's ad_rank / your pCTR)
        if i + 1 < len(ranked):
            next_ad_rank = ranked[i + 1].ad_rank
            price = next_ad_rank / ad.predicted_ctr + 0.01
        else:
            price = ad.reserve_price  # floor price

        # Never exceed advertiser's max bid
        price = min(price, ad.max_bid)

        results.append({
            "ad": ad,
            "position": i + 1,
            "price_per_click": round(price, 2)
        })

    return results

# WHY CALIBRATION MATTERS:
# If pCTR is 2x too high:
#   ad_rank doubles → ad wins auctions it shouldn't
#   price = next_ad_rank / (2x pCTR) → pays HALF
#   Advertiser gets cheap clicks, Google loses revenue
#
# If pCTR is 2x too low:
#   ad_rank halves → ad loses auctions it should win
#   Advertiser gets fewer impressions, complains, leaves`} />
          <div className="space-y-4">
            <Card accent="#0f766e">
              <Label color="#0f766e">Why Not First-Price Auction?</Label>
              <div className="space-y-2">
                {[
                  { q: "GSP vs First-Price", a: "First-price: you pay what you bid. Encourages bid shading (bid less than your true value). Leads to unstable, game-theoretic equilibria. GSP: you pay the next bidder's price. Encourages truthful bidding. More stable, easier for advertisers." },
                  { q: "Why include quality_score?", a: "Without quality_score, the highest bidder always wins. This creates bad user experience (irrelevant high-bid ads). Quality score rewards relevant, high-quality ads with lower effective prices. Keeps the ad ecosystem healthy." },
                  { q: "Reserve price (floor price)?", a: "Minimum price per click. Prevents race-to-bottom pricing in uncompetitive auctions. Set per-query based on estimated commercial value. Without it, some clicks would cost $0.01 — not sustainable." },
                  { q: "Budget pacing", a: "Advertisers set daily budgets. If an ad wins but the campaign has exhausted its daily budget, it's removed from the auction. Budget manager tracks spend in near-real-time and throttles campaigns approaching their limit." },
                ].map((d,i) => (
                  <div key={i}>
                    <div className="text-[11px] font-bold text-stone-700">{d.q}</div>
                    <div className="text-[10px] text-stone-500 mt-0.5">{d.a}</div>
                  </div>
                ))}
              </div>
            </Card>
            <Card className="bg-red-50/50 border-red-200">
              <Label color="#dc2626">Calibration Failure Scenarios</Label>
              <div className="space-y-2 text-[11px]">
                <div className="flex items-start gap-2"><span className="text-red-500 font-bold">↑</span><span className="text-stone-600"><strong>Over-predict CTR:</strong> Ad wins more auctions but pays less per click (price = next_rank / inflated_pCTR). Advertiser gets great deal, Google loses revenue. At scale: billions lost.</span></div>
                <div className="flex items-start gap-2"><span className="text-red-500 font-bold">↓</span><span className="text-stone-600"><strong>Under-predict CTR:</strong> Ad loses auctions it should win. Advertiser gets fewer impressions, blames the platform, reduces spend. Revenue loss + advertiser churn.</span></div>
                <div className="flex items-start gap-2"><span className="text-red-500 font-bold">~</span><span className="text-stone-600"><strong>Group mis-calibration:</strong> CTR well-calibrated overall but 2x too high for mobile. Mobile advertisers get undercharged. Desktop advertisers get overcharged. Both are problems.</span></div>
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
        <Label color="#dc2626">CTR Model Architecture Evolution</Label>
        <p className="text-[12px] text-stone-500 mb-4">Ads CTR models have evolved from logistic regression to deep networks. At Google L6, you should know the evolution and why each step happened.</p>
        <div className="grid grid-cols-3 gap-4">
          {[
            { era: "2010s: LR + Feature Engineering", model: "Logistic Regression", desc: "Billions of sparse features (query-ad pairs). Feature crosses done manually. FTRL-Proximal optimizer for online learning. Still used as baseline.", pros: "Fast inference, interpretable, online learning trivial", cons: "Manual feature engineering, can't learn complex interactions", color: "#78716c" },
            { era: "2016+: Wide & Deep", model: "Wide & Deep Network", desc: "Google's paper. Wide (LR) path memorizes specific patterns. Deep path generalizes via embeddings. Combined: best of both.", pros: "Memorization + generalization, handles sparse + dense", cons: "Two paths to maintain, embedding table management", color: "#9333ea" },
            { era: "2020+: Deep & Cross (DCN) ★", model: "Deep & Cross Network v2", desc: "Cross network automatically learns feature interactions up to arbitrary order. No manual feature crosses. Current state of the art at Google.", pros: "Automatic interaction learning, fewer manual features", cons: "Larger model, needs more training data, slower inference", color: "#dc2626" },
          ].map((m,i) => (
            <div key={i} className="rounded-lg border p-4" style={{ borderTop: `3px solid ${m.color}` }}>
              <div className="text-[10px] font-bold mb-1" style={{ color: m.color }}>{m.era}</div>
              <div className="text-[12px] font-bold text-stone-800 mb-1">{m.model}</div>
              <p className="text-[10px] text-stone-500 mb-2">{m.desc}</p>
              <div className="text-[10px] text-emerald-600 mb-0.5">✓ {m.pros}</div>
              <div className="text-[10px] text-red-500">✗ {m.cons}</div>
            </div>
          ))}
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">DCNv2 Architecture (Current SOTA)</Label>
          <CodeBlock code={`# Deep & Cross Network v2 — Google Ads CTR Model
class DCNv2(nn.Module):
    def __init__(self, sparse_dims, dense_dim):
        # Embedding layers for sparse features
        self.embeddings = {
            "ad_id": Embedding(1B, 64),
            "advertiser_id": Embedding(10M, 32),
            "query_tokens": Embedding(500K, 32),
            "user_id": Embedding(2B, 64),
            "campaign_id": Embedding(100M, 32),
            # ... more sparse features
        }

        # Cross network — learns feature interactions
        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(3)
        ])

        # Deep network — learns complex non-linear patterns
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.output = nn.Linear(input_dim + 256, 1)

    def forward(self, sparse_features, dense_features):
        # Embed all sparse features
        embs = [self.embeddings[f](sparse_features[f])
                for f in sparse_features]
        x = concat(embs + [dense_features])

        # Cross network path
        x_cross = x
        for cross_layer in self.cross_layers:
            x_cross = cross_layer(x_cross, x)  # x0 * xT * w + b

        # Deep network path
        x_deep = self.deep(x)

        # Combine
        combined = concat([x_cross, x_deep])
        logit = self.output(combined)
        return logit  # raw logit — calibrated post-hoc`} />
        </Card>
        <Card accent="#9333ea">
          <Label color="#9333ea">Cross Layer — The Key Innovation</Label>
          <CodeBlock code={`# Cross Layer — Automatic Feature Interaction
# Learns explicit feature crosses without manual engineering
# Each layer adds one order of interaction

class CrossLayer(nn.Module):
    def __init__(self, dim):
        self.W = nn.Parameter(torch.randn(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x_l, x_0):
        # x_l: current layer output
        # x_0: original input (always)
        # Output: x_0 * (W * x_l + b) + x_l
        #
        # Layer 1: learns 2nd-order crosses (a*b)
        # Layer 2: learns 3rd-order crosses (a*b*c)
        # Layer 3: learns 4th-order crosses (a*b*c*d)
        cross = x_0 * (self.W @ x_l + self.b)
        return cross + x_l  # residual connection

# Example: query="flights", ad_category="travel"
# LR approach: manually create feature "query_flights_AND_cat_travel"
# Cross network: AUTOMATICALLY learns this interaction
# AND higher-order: "flights" AND "travel" AND "mobile" AND "evening"
#
# Why this matters for CTR:
#   CTR for "flights" + "travel ad" + "mobile" + "evening"
#   is NOT the sum of individual effects
#   It's a specific combination that has uniquely high CTR
#   Cross network captures this without manual features`} />
        </Card>
      </div>
    </div>
  );
}

function FeaturesSection() {
  return (
    <div className="space-y-5">
      <Card accent="#d97706">
        <Label color="#d97706">Feature Categories for Ads CTR Prediction</Label>
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
                { cat: "Query Features", ex: "query_tokens, query_length, query_intent, is_commercial, query_embedding", comp: "Online", impact: "Very High" },
                { cat: "Ad Features", ex: "ad_id, headline_tokens, ad_quality_score, landing_page_score, ad_format, ad_age", comp: "Batch + index", impact: "Very High" },
                { cat: "Advertiser Features", ex: "advertiser_id, historical_CTR, account_age, vertical, avg_bid, budget_remaining_pct", comp: "Near-real-time", impact: "High" },
                { cat: "User Features", ex: "user_id, ad_click_history, search_history_topics, demographics_bucket", comp: "Batch (daily)", impact: "High" },
                { cat: "Query-Ad Match", ex: "keyword_match_type (exact/phrase/broad), query_ad_cosine_sim, title_overlap_ratio", comp: "Online", impact: "Very High" },
                { cat: "Context", ex: "device_type, time_of_day, day_of_week, geo_region, ad_position (if known)", comp: "Online (request)", impact: "Medium-High" },
                { cat: "Historical CTR", ex: "ad_CTR_7d, ad_CTR_30d, keyword_CTR_7d, advertiser_CTR_30d, query_CTR_bucket", comp: "Batch (daily)", impact: "Very High" },
                { cat: "Competition", ex: "num_competing_ads, avg_competitor_bid, auction_density_bucket", comp: "Online", impact: "Medium" },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-bold text-stone-700">{r.cat}</td>
                  <td className="px-3 py-2 text-stone-500 font-mono text-[10px]">{r.ex}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.comp}</td>
                  <td className="text-center px-3 py-2"><Pill bg={r.impact.includes("Very")?"#fef2f2":r.impact.includes("High")?"#fffbeb":"#f0fdf4"} color={r.impact.includes("Very")?"#dc2626":r.impact.includes("High")?"#d97706":"#059669"}>{r.impact}</Pill></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">The Power of Historical CTR (and Its Trap)</Label>
          <CodeBlock code={`# Historical CTR is the single most predictive feature
# But it has severe bias and cold-start problems

# Feature: smoothed_historical_CTR
def compute_smoothed_ctr(ad_id, min_impressions=100):
    clicks = click_count[ad_id]
    impressions = impression_count[ad_id]

    if impressions < min_impressions:
        # Cold start: not enough data
        # Use hierarchical smoothing (Bayesian prior)
        # Prior = campaign-level CTR or advertiser-level CTR
        prior_ctr = get_parent_ctr(ad_id)
        prior_weight = min_impressions
        smoothed = (clicks + prior_ctr * prior_weight) / (
            impressions + prior_weight
        )
        return smoothed
    else:
        return clicks / impressions

# Hierarchical smoothing:
# Level 1: ad creative CTR (most specific, least data)
# Level 2: ad group CTR
# Level 3: campaign CTR
# Level 4: advertiser CTR
# Level 5: vertical CTR (e.g., "travel ads" avg CTR)
# Level 6: global CTR (~2% for search ads)
#
# New ad with 0 impressions → use campaign-level CTR
# After 100 impressions → blend toward actual CTR
# After 1000 impressions → actual CTR dominates
#
# TRAP: historical CTR is position-biased
# Position 1 has ~5% CTR, position 4 has ~1.5%
# Must normalize: CTR / expected_CTR_for_position`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Feature Interaction Importance</Label>
          <div className="space-y-3">
            {[
              { cross: "query × ad_headline", why: "The most important single interaction. Does the ad text match what the user searched for? 'flights hawaii' matching 'Hawaii Flights from $199' vs 'Car Rentals in LA'.", impact: "★★★★★" },
              { cross: "user_search_history × ad_vertical", why: "Was the user recently researching this topic? A user who searched 'best credit cards' yesterday is more likely to click a credit card ad today.", impact: "★★★★☆" },
              { cross: "device × ad_format", why: "Mobile users click call extensions more. Desktop users click sitelinks more. Video ads work better on tablet. Format must match device.", impact: "★★★★☆" },
              { cross: "time_of_day × ad_vertical", why: "Restaurant ads have higher CTR at lunch. Travel ads peak on Tuesday evenings. B2B ads work better during business hours.", impact: "★★★☆☆" },
              { cross: "geo × ad_local_flag", why: "Local business ads (plumber, restaurant) have much higher CTR when user is geographically close. 5 miles vs 50 miles is 3x CTR difference.", impact: "★★★★☆" },
            ].map((c,i) => (
              <div key={i} className="rounded-lg border border-stone-200 p-3">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[11px] font-bold text-stone-800 font-mono">{c.cross}</span>
                  <span className="text-[10px] text-amber-600">{c.impact}</span>
                </div>
                <p className="text-[10px] text-stone-500">{c.why}</p>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function CalibrationSection() {
  return (
    <div className="space-y-5">
      <Card accent="#0f766e">
        <Label color="#0f766e">Calibration — The Make-or-Break for Ads</Label>
        <p className="text-[12px] text-stone-500 mb-4">Calibration is what makes ads CTR fundamentally different from search/rec ranking. In ranking, only the ordering matters. In ads, the actual predicted probability is used for pricing. A well-ranked but poorly-calibrated model will bankrupt the platform.</p>
        <div className="grid grid-cols-2 gap-5">
          <div>
            <CodeBlock title="Post-Hoc Calibration Methods" code={`# Method 1: Platt Scaling (simple, effective)
# Fit sigmoid on validation set: P(y=1|s) = 1/(1+exp(-a*s-b))
class PlattCalibrator:
    def calibrate(self, raw_score, a, b):
        return 1.0 / (1.0 + exp(-a * raw_score - b))
    # a, b learned on held-out validation data
    # Assumes monotonic relationship between score and prob

# Method 2: Isotonic Regression (non-parametric, more flexible)
class IsotonicCalibrator:
    def fit(self, raw_scores, actual_labels):
        # Fit piecewise-constant monotonic function
        # Bins scores into buckets, computes actual CTR per bucket
        self.calibrator = IsotonicRegression(
            out_of_bounds="clip"
        )
        self.calibrator.fit(raw_scores, actual_labels)

    def calibrate(self, raw_score):
        return self.calibrator.predict([raw_score])[0]
    # More flexible than Platt — handles non-sigmoid patterns
    # Needs more data (overfits on small validation sets)

# Method 3: Field-aware calibration (segment-level)
# Calibrate separately for each segment that matters
def calibrate_by_segment(raw_scores, labels, segments):
    calibrators = {}
    for segment in ["mobile", "desktop", "tablet"]:
        mask = segments == segment
        calibrators[segment] = IsotonicCalibrator()
        calibrators[segment].fit(
            raw_scores[mask], labels[mask]
        )
    return calibrators

# CRITICAL: recalibrate frequently (hourly or daily)
# CTR distribution shifts with time of day, seasonality
# A calibrator from Monday is stale by Wednesday`} />
          </div>
          <div className="space-y-4">
            <Card accent="#dc2626">
              <Label color="#dc2626">Measuring Calibration Quality</Label>
              <div className="space-y-2">
                {[
                  { metric: "Expected Calibration Error (ECE)", formula: "Sum |avg(predicted) - avg(actual)| per bin", target: "<1%", desc: "Bin predictions into 10-20 buckets. For each bucket, compare mean predicted CTR vs actual CTR. Lower is better." },
                  { metric: "Log Loss (Cross-Entropy)", formula: "-avg(y*log(p) + (1-y)*log(1-p))", target: "Minimize", desc: "Directly penalizes miscalibrated probabilities. A miscalibrated prediction of 0.9 for a non-click is extremely costly." },
                  { metric: "Reliability Diagram", formula: "Plot predicted vs actual CTR per bucket", target: "Diagonal line", desc: "Visual check. Perfect calibration = points on the diagonal. Above diagonal = under-predicting. Below = over-predicting." },
                  { metric: "Segment-Level Calibration", formula: "ECE computed per device, geo, vertical", target: "<2% per segment", desc: "Overall calibration can hide segment-level errors. A model well-calibrated globally but 2x off on mobile is dangerous." },
                ].map((m,i) => (
                  <div key={i} className="rounded-lg border border-stone-200 p-2.5">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-[10px] font-bold text-stone-800">{m.metric}</span>
                      <Pill bg="#f0fdf4" color="#059669">{m.target}</Pill>
                    </div>
                    <div className="text-[9px] font-mono text-stone-400 mb-0.5">{m.formula}</div>
                    <p className="text-[10px] text-stone-500">{m.desc}</p>
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
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Core Data Stores</Label>
          <CodeBlock code={`-- Ad Index (inverted, sharded by keyword hash)
-- Maps keywords to eligible ads
keyword -> [(ad_id, match_type, bid, targeting), ...]
# Updated near-real-time as advertisers change bids/targeting
# Match types: exact, phrase, broad, broad_match_modifier

-- Ad Metadata (sharded by ad_id)
ad_id -> {
  advertiser_id, campaign_id, ad_group_id,
  headline, description, display_url, click_url,
  ad_format: enum(text, shopping, call, video),
  quality_score: float,     # ML-scored ad quality
  landing_page_score: float,
  approval_status: enum(approved, disapproved, limited),
  targeting: {geo, device, language, schedule, audience},
}

-- Feature Store (Redis + Bigtable, multi-tier)
-- Real-time: Redis (ad budget pacing, last-hour CTR)
-- Near-RT: Bigtable (daily CTR, smoothed metrics)
-- Batch: BigQuery (historical aggregations)
ad_id -> {ctr_7d, ctr_30d, impressions_7d, quality_score}
user_id -> {search_topics, ad_click_history, demographics}
query_hash -> {avg_ctr, commercial_intent_score}

-- Click/Impression Log (Kafka + BigQuery)
(ts, request_id, user_id, query, ad_id, position,
 predicted_ctr, actual_click, bid, price_charged,
 dwell_time_on_landing_page, conversion)

-- Budget Ledger (strongly consistent, sharded by campaign)
campaign_id -> {
  daily_budget, spent_today, remaining,
  last_updated: timestamp,
  pace_multiplier: float  # throttle if over-spending
}`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Why These Design Choices?</Label>
          <div className="space-y-3">
            {[
              { q: "Why inverted index for ad retrieval (not ANN)?", a: "Ads have explicit keyword targeting set by advertisers. It's an exact/phrase match problem, not a semantic similarity problem. BM25/inverted index is the right tool. ANN would find semantically similar ads but advertisers didn't target those queries — billing would be wrong." },
              { q: "Why Redis for budget tracking?", a: "Budget checks happen on EVERY ad request (60K QPS). Must be sub-millisecond. Budget is a strong consistency requirement — can't overspend. Redis with Lua scripts for atomic decrement. Sharded by campaign_id for parallelism." },
              { q: "Why log predicted_ctr alongside clicks?", a: "Training data must use logged features to avoid training-serving skew. Also enables offline calibration analysis: compare predicted vs actual across time, segments, positions. The logged pCTR IS the training target's complement." },
              { q: "Why separate click log and conversion log?", a: "Clicks happen immediately. Conversions happen minutes to days later (user lands on page, browses, maybe purchases later). Different time scales. Conversion data is joined to click data by request_id in a batch pipeline." },
              { q: "Why pace_multiplier in budget ledger?", a: "Budget pacing ensures the daily budget is spent evenly across the day. Without pacing, all budget would be spent in the morning. pace_multiplier < 1.0 randomly drops the ad from some auctions. Adjusted every 15 minutes based on remaining budget vs remaining time in day." },
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
        <Label color="#7e22ce">Training Pipeline — Batch + Online Hybrid</Label>
        <p className="text-[12px] text-stone-500 mb-4">Unlike search/rec systems that retrain daily, ads CTR models need near-real-time updates. New ads launch constantly, and CTR patterns shift within hours. The solution: batch training for the base model + online learning for real-time adaptation.</p>
        <div className="grid grid-cols-2 gap-5">
          <Card accent="#7e22ce">
            <Label color="#7e22ce">Batch Training (Base Model)</Label>
            <CodeBlock code={`# Batch Training — retrained daily
# Uses 7-30 days of click logs

batch_train_pipeline():
    # 1. Data construction
    data = join(
        click_logs.last_30_days(),   # impressions + clicks
        logged_features,              # features AT SERVING TIME
    )

    # 2. Label: binary (click / no-click)
    # Weighted by inverse propensity (position de-biasing)
    data["weight"] = 1.0 / position_propensity[data["position"]]

    # 3. Negative downsampling
    # Only 2% of impressions are clicks → massive class imbalance
    # Downsample negatives 10x, then correct in calibration
    negatives = data[data.click == 0].sample(frac=0.1)
    positives = data[data.click == 1]  # keep all clicks
    train_data = concat([positives, negatives])

    # 4. Train DCNv2 model
    model = DCNv2(sparse_dims, dense_dims)
    model.train(
        train_data,
        loss="binary_cross_entropy",
        optimizer="Adam",
        epochs=3,
        batch_size=4096,
    )

    # 5. Calibrate for negative downsampling
    # If we downsampled negatives by 10x, raw predictions
    # are inflated. Correct:
    # p_corrected = p / (p + (1-p) * sampling_rate)
    calibrator = fit_calibrator(model, validation_data)

    return model, calibrator`} />
          </Card>
          <Card accent="#ea580c">
            <Label color="#ea580c">Online Learning (Real-Time Adaptation)</Label>
            <CodeBlock code={`# Online Learning — continuous updates
# FTRL-Proximal (Follow The Regularized Leader)
# Updates model weights with each new click/impression

class OnlineLearner:
    def __init__(self, base_model):
        self.model = base_model
        self.optimizer = FTRLProximal(
            alpha=0.1,       # learning rate
            beta=1.0,        # smoothing
            l1=1.0,          # L1 regularization (sparsity)
            l2=1.0,          # L2 regularization
        )

    def update(self, impression_event):
        """Called for each new click/impression event"""
        features = impression_event.logged_features
        label = impression_event.clicked  # 0 or 1

        # Forward pass
        prediction = self.model.predict(features)

        # Compute gradient
        gradient = prediction - label  # BCE gradient

        # Update only the linear (wide) layer
        # Deep layers are too expensive to update online
        self.optimizer.update(
            self.model.wide_weights,
            gradient,
            features
        )

    # WHY FTRL over SGD:
    # - L1 regularization drives unused features to exactly 0
    #   (important with billions of sparse features)
    # - Per-coordinate learning rates (rare features learn faster)
    # - Proven at Google scale (paper: McMahan et al. 2013)

    # WHAT gets updated online:
    # - Wide (linear) layer: embedding weights for new ad IDs,
    #   new keyword patterns, bid changes
    # - NOT the deep layers: too expensive, too unstable
    #
    # New ad launched 5 minutes ago?
    # Online learning has already started adapting its CTR`} />
          </Card>
        </div>
      </Card>
    </div>
  );
}

function ServingSection() {
  return (
    <div className="space-y-5">
      <Card accent="#ea580c">
        <Label color="#ea580c">Real-Time Serving — Every Microsecond Counts</Label>
        <p className="text-[12px] text-stone-500 mb-4">Ads CTR prediction has the tightest latency requirements in all of ML. The entire pipeline — feature lookup, model inference, calibration, auction — must complete in under 10ms. Here's how.</p>
        <div className="grid grid-cols-2 gap-5">
          <div className="space-y-4">
            <Card accent="#ea580c">
              <Label color="#ea580c">Latency Optimization Techniques</Label>
              <div className="space-y-2">
                {[
                  { tech: "Feature Pre-fetching", desc: "Start fetching user features while ad retrieval is running. Overlap I/O with compute. Saves ~2ms of sequential latency.", saves: "2ms" },
                  { tech: "Batched Inference", desc: "Score all 500 candidate ads in a single GPU forward pass. Amortizes GPU kernel launch overhead. 500 individual calls = 500ms. Batched = 3ms.", saves: "Massive" },
                  { tech: "Embedding Caching", desc: "Cache hot ad embeddings in L2 cache on the model server. Top 1% of ads handle 50% of traffic. Avoid embedding table lookup for cached ads.", saves: "0.5ms" },
                  { tech: "Model Quantization", desc: "INT8 quantization for inference. 2-4x speedup vs FP32 with < 0.1% accuracy loss. Critical for meeting 3ms inference budget.", saves: "1.5ms" },
                  { tech: "Two-Phase Scoring", desc: "Phase 1: lightweight model (LR) scores 1000 ads in 1ms, keeps top 200. Phase 2: heavy model (DCN) scores 200 ads in 2ms. Saves 60% GPU cost.", saves: "GPU cost" },
                  { tech: "Async Logging", desc: "Don't log synchronously in the serving path. Fire-and-forget to Kafka. Logging adds 0 latency to the response.", saves: "0.5ms" },
                ].map((t,i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-orange-100 text-orange-700 shrink-0">{t.saves}</span>
                    <div>
                      <div className="text-[11px] font-bold text-stone-700">{t.tech}</div>
                      <div className="text-[10px] text-stone-500">{t.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
          <Card accent="#2563eb">
            <Label color="#2563eb">Serving Architecture</Label>
            <CodeBlock code={`# Ad Serving Request Flow — Optimized for <10ms

async def serve_ads(request: AdRequest) -> AdResponse:
    t0 = time.now()

    # PARALLEL: fetch features + retrieve ads simultaneously
    user_features_future = feature_store.get_async(
        request.user_id
    )
    candidate_ads = ad_index.retrieve(
        query=request.query,
        targeting=request.user_context,
        max_candidates=1000,
    )
    # ~2ms for both (parallel)

    user_features = await user_features_future

    # Phase 1: lightweight pre-filter (optional)
    if len(candidate_ads) > 500:
        quick_scores = lr_model.score_batch(candidate_ads)
        candidate_ads = top_k(candidate_ads, quick_scores, k=500)
    # ~1ms

    # Assemble feature vectors for all candidates
    feature_matrix = assemble_features(
        user=user_features,
        ads=candidate_ads,
        query=request.query,
        context=request.context,
    )
    # ~1ms

    # Phase 2: deep model scoring (batched GPU)
    raw_scores = dcn_model.predict_batch(feature_matrix)
    # ~3ms (500 ads in one forward pass)

    # Calibrate raw scores to probabilities
    calibrated_ctrs = calibrator.transform(raw_scores)
    # ~0.1ms

    # Run auction
    winners = auction.run(
        ads=candidate_ads,
        predicted_ctrs=calibrated_ctrs,
        max_slots=request.position_count,
    )
    # ~1ms (sort + pricing + budget check)

    # Async: log impression for training
    logger.log_async(request, candidate_ads, raw_scores, winners)

    elapsed = time.now() - t0  # target: <10ms
    return AdResponse(ads=winners, latency_ms=elapsed)`} />
          </Card>
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
          <Label color="#059669">Model Serving Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">GPU fleet sizing</strong> — 30M scores/sec at 3ms/batch of 500 = 60K batches/sec. Each GPU handles ~2K batches/sec. Need ~30 GPUs. With 3x redundancy and peak: ~300 GPUs globally.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Embedding table sharding</strong> — 100GB embedding table doesn't fit on one GPU. Shard across multiple parameter servers. Each GPU has its own copy of hot embeddings (top 1% by frequency). Cold embeddings fetched from parameter server.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Model replication</strong> — same model replicated across all serving regions. Model updates pushed via global model distribution service. Blue-green deployment: new model serves 1% traffic first.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Feature store scaling</strong> — Redis cluster with ~50 shards for real-time features. Bigtable for batch features with row caching. Hot keys (popular queries, frequent advertisers) cached in local memory.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Ad Index Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Inverted index sharding</strong> — shard by keyword hash. Each shard handles a subset of the vocabulary. Query is routed to relevant shard(s). Multi-word queries may hit multiple shards.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Real-time index updates</strong> — advertisers change bids, budgets, and targeting in real-time. Changes propagate to ad index within seconds via streaming pipeline. No full index rebuild needed.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Budget pacing at scale</strong> — 10M+ campaigns, each with a daily budget. Budget ledger is sharded by campaign. Atomic decrement on every impression. Pacing multiplier recomputed every 15 min per campaign.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Multi-region serving</strong> — ad index replicated to every region. Auction runs locally (no cross-region calls). Model and features replicated. Budget is a global constraint — requires cross-region coordination for shared campaigns.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Ad Serving</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Full Replication ★", d:"Entire ad index, feature store, and model replicated per region. Auction runs locally. Budget coordination via async eventual consistency.", pros:["Lowest latency (no cross-region)","Region-autonomous","Standard approach"], cons:["Budget over-spend risk (eventual consistency)","Index sync lag","Expensive replication"], pick:true },
            { t:"Centralized Auction", d:"Ad retrieval local, but auction runs in a central region. Ensures perfect budget enforcement and globally optimal allocation.", pros:["Perfect budget enforcement","Global optimization","Simpler consistency"], cons:["Cross-region latency (+30ms)","Central bottleneck","Higher failure blast radius"], pick:false },
            { t:"Hybrid Budget", d:"Local auction with global budget ledger. Each region has a budget allocation. Central coordinator rebalances allocations every minute.", pros:["Low latency","Controlled over-spend","Flexible allocation"], cons:["Complex coordination","Budget fragmentation","Rebalancing delay"], pick:false },
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
          { title: "Click Fraud", sev: "CRITICAL", desc: "Competitors clicking your ads to drain your budget. Bot networks generating fake clicks. Click farms in low-cost regions. At scale: 10-30% of all ad clicks may be fraudulent.", fix: "Multi-layer defense: IP/device fingerprinting, click pattern anomaly detection (too fast, too regular), mouse movement analysis, post-click conversion correlation (fraudulent clicks rarely convert). Refund advertisers for detected fraud.", icon: "🔴" },
          { title: "Calibration Drift", sev: "CRITICAL", desc: "Model predictions drift out of calibration over hours/days. Time-of-day effects, seasonal changes, new advertisers joining. 2% predicted CTR is actually 3% → auction undercharges → revenue loss.", fix: "Hourly recalibration on recent data. Segment-level calibration monitoring dashboard. Auto-rollback if ECE > 3% on any major segment. Online learning helps adapt faster than batch-only.", icon: "🔴" },
          { title: "Cold Start for New Ads", sev: "HIGH", desc: "New ad with zero impressions — model has no idea what CTR should be. If CTR prediction is too low, ad never gets shown. If too high, advertiser gets overcharged initially.", fix: "Hierarchical smoothing (use campaign/advertiser/vertical CTR as prior). Exploration budget: guarantee new ads get minimum impressions. Thompson Sampling for new ad exploration. Separate cold-start model variant.", icon: "🟡" },
          { title: "Position Bias in Training", sev: "HIGH", desc: "Ads in position 1 get higher CTR due to visibility, not relevance. Model trained on raw clicks learns to predict position, not intrinsic relevance. Self-reinforcing: model puts ad at top → gets clicks → reinforced.", fix: "Inverse propensity weighting (IPW) in training. Occasional position randomization experiments (1% of traffic). Position as an explicit feature (allows model to factor it out). Train on position-normalized CTR.", icon: "🟡" },
          { title: "Embedding Table Explosion", sev: "MEDIUM", desc: "1B ad IDs × 64 dims = 256 GB. New ads created constantly. Table grows without bound. Memory pressure on model servers. Stale embeddings for deleted ads waste space.", fix: "Frequency-based eviction: remove embeddings for ads with 0 impressions in last 30 days. Hash trick for rare ads (share embedding slots). Quantize embeddings from FP32 to FP16. Separate hot/cold embedding tiers.", icon: "🟠" },
          { title: "Feature Store Staleness", sev: "MEDIUM", desc: "Batch features (historical CTR) updated daily. An ad that went viral 2 hours ago still has yesterday's CTR features. Model under-predicts for trending ads, over-predicts for exhausted ads.", fix: "Three-tier feature architecture: real-time (seconds), near-real-time (minutes), batch (daily). Streaming pipeline updates engagement features continuously. Freshness indicator as a feature so model knows which features are stale.", icon: "🟠" },
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
          <Label color="#0284c7">Revenue Metrics</Label>
          <div className="space-y-2.5">
            {[
              { metric: "Revenue per Query (RPQ)", target: "$0.013 avg", why: "Primary business metric. Revenue generated per ad-eligible query." },
              { metric: "RPM (Revenue per 1000 impressions)", target: "Varies by vertical", why: "Monetization efficiency. Low RPM = bad ad quality or low demand." },
              { metric: "Fill Rate", target: ">85%", why: "% of ad-eligible queries that show at least one ad. Low = retrieval or budget issue." },
              { metric: "CPC (Cost per Click)", target: "$1-3 avg", why: "What advertisers pay. Trending down = calibration issue or low competition." },
              { metric: "Advertiser ROI", target: ">3x", why: "If advertisers don't get returns, they reduce spend. Lagging indicator." },
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
        <Card accent="#dc2626">
          <Label color="#dc2626">Model Quality Metrics</Label>
          <div className="space-y-2.5">
            {[
              { metric: "AUC-ROC", target: ">0.80", why: "Ranking quality — are clicks scored higher than non-clicks?" },
              { metric: "Log Loss", target: "Minimize", why: "Calibration-sensitive metric. Penalizes confident wrong predictions." },
              { metric: "ECE (Calibration Error)", target: "<2%", why: "Are predicted CTRs accurate? Critical for auction correctness." },
              { metric: "Normalized Entropy (NE)", target: "<0.95", why: "Log loss normalized by entropy of labels. Better model = lower NE." },
              { metric: "CTR Distribution Shift", target: "KL < 0.01", why: "Is today's CTR distribution similar to training data?" },
            ].map((m,i) => (
              <div key={i} className="border-b border-stone-100 pb-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold text-stone-700">{m.metric}</span>
                  <span className="text-[10px] font-mono text-red-600">{m.target}</span>
                </div>
                <div className="text-[10px] text-stone-400 mt-0.5">{m.why}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">System Health & Alerts</Label>
          <div className="space-y-2.5">
            {[
              { alert: "Serving latency p99 > 15ms", sev: "P0", action: "Shed model load, serve LR-only fallback" },
              { alert: "Revenue drops > 5% hour-over-hour", sev: "P0", action: "Check model, calibration, budget pacing" },
              { alert: "ECE > 3% on any segment", sev: "P0", action: "Recalibrate immediately, consider rollback" },
              { alert: "Click fraud rate > 20%", sev: "P1", action: "Escalate to fraud team, tighten filters" },
              { alert: "Feature store latency > 5ms", sev: "P1", action: "Scale feature store, check cache hit rates" },
            ].map((a,i) => (
              <div key={i} className="border-b border-stone-100 pb-2">
                <div className="flex items-center gap-2">
                  <Pill bg={a.sev==="P0"?"#fef2f2":"#fffbeb"} color={a.sev==="P0"?"#dc2626":"#d97706"}>{a.sev}</Pill>
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
          { title: "Conversion Prediction (pCVR)", d: "Predict P(conversion | click) alongside P(click). Rank by expected_value = pCTR × pCVR × conversion_value. Enables value-based bidding for advertisers.", effort: "Hard", detail: "Challenge: conversion data is extremely sparse (1-5% of clicks convert) and delayed (hours to days). Needs multi-task learning with CVR as auxiliary objective. Counterfactual estimation for unclicked ads." },
          { title: "Auto-Bidding (Smart Bidding)", d: "Automatically set bids to maximize advertiser's objective (target CPA, target ROAS). Replaces manual bidding with ML-optimized bidding in real-time.", effort: "Hard", detail: "Requires per-advertiser model or contextual bandit. Must respect budget constraints. Google's Smart Bidding handles >70% of search ad spend." },
          { title: "Creative Optimization", d: "ML-generated or ML-selected ad copy. Given multiple headlines and descriptions, predict which combination has highest CTR for each query. Responsive Search Ads.", effort: "Medium", detail: "Combinatorial explosion: 15 headlines × 4 descriptions = 32,760 combinations. Thompson Sampling to explore. LLM-generated ad copy is emerging." },
          { title: "Privacy-Preserving Features", d: "As cookie deprecation and privacy regulations increase, move from user-level features to cohort-level or contextual features. Federated learning for on-device signals.", effort: "Hard", detail: "Google's Privacy Sandbox, Topics API. Challenge: maintaining CTR prediction accuracy without cross-site tracking. Contextual targeting becomes more important." },
          { title: "Cross-Surface Optimization", d: "Jointly optimize ads across Search, YouTube, Display, Gmail. Frequency capping across surfaces. Unified attribution.", effort: "Hard", detail: "Challenge: different surfaces have different CTR distributions, ad formats, and user intents. Needs cross-surface user identity and a unified auction coordinator." },
          { title: "Auction Mechanism Innovation", d: "Move from GSP to VCG for better incentive compatibility. Or ML-optimized reserve prices. Dynamic floor pricing based on query value.", effort: "Medium", detail: "VCG is theoretically optimal but computationally expensive (need to run N sub-auctions). ML-optimized reserve prices can increase revenue 5-10%." },
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
    { q:"Why does Google use pCTR × bid instead of just the highest bid?", a:"If the highest bidder always wins, low-quality irrelevant ads would dominate (anyone can bid high). Users would stop clicking ads → long-term revenue destruction. pCTR × bid means a relevant ad with $1 bid and 5% CTR (rank = 0.05) beats an irrelevant ad with $2 bid and 1% CTR (rank = 0.02). This aligns Google's incentive (maximize clicks × price) with user experience (show relevant ads). Additionally, expected revenue per impression = pCTR × CPC. The formula directly maximizes expected revenue per impression.", tags:["auction"] },
    { q:"How do you handle the massive class imbalance (2% CTR = 98% negatives)?", a:"Three strategies: (1) Negative downsampling — randomly sample 10% of non-clicks, keep all clicks. This creates a 17% click rate in training. Faster training, less memory. BUT you must correct predictions post-training: p_corrected = p / (p + (1-p)/sampling_rate). (2) Focal loss — down-weights easy negatives (obviously non-clickable ads). Focuses model capacity on hard negatives (ads the model was uncertain about). (3) Don't use accuracy as a metric — a model predicting 'no click' for everything gets 98% accuracy. Use AUC-ROC and log loss instead.", tags:["ml"] },
    { q:"How is CTR prediction different from recommendation ranking?", a:"Three critical differences: (1) Calibration — rec ranking only needs correct ordering. CTR prediction needs accurate probabilities because they're used for pricing. A rec model off by 2x doesn't matter. An ads model off by 2x means billions in mispriced auctions. (2) Latency — rec systems get 150-200ms. Ads get <10ms (it's one component inside the search serving pipeline). (3) Adversarial — advertisers actively try to game CTR (clickbait ad copy, manipulation). Rec systems deal with organic content. (4) Online learning — ads must adapt to new ads within minutes. Rec systems retrain daily.", tags:["design"] },
    { q:"What happens if the CTR model goes down?", a:"Graceful degradation ladder: (1) Serve from a secondary model replica. (2) If all GPU model servers are down, fall back to a lightweight LR model running on CPU — always available, lower quality but functional. (3) If even LR is down, use pre-cached ad_rank scores (updated hourly). Quality degrades but ads still serve. (4) If everything is down, serve ads based on bid alone (first-price auction, no ML). Never show zero ads — every empty ad slot is lost revenue. Each tier has automatic failover monitored by an ad serving health check.", tags:["availability"] },
    { q:"How do you A/B test a new CTR model?", a:"Ads A/B tests are special because they affect advertiser revenue, not just user experience. Process: (1) Offline evaluation: AUC, log loss, calibration on held-out data. If regression, stop here. (2) Shadow scoring: new model scores all requests in parallel with production, but results aren't used. Compare predictions statistically. (3) Live experiment (1% traffic): monitor revenue per query, advertiser CPC changes, calibration, user CTR. Run for 2+ weeks. (4) Ramp: 1% → 5% → 20% → 100% with monitoring at each stage. (5) Guardrails: auto-stop if revenue drops >0.5% or calibration error increases >1%. Key metric: revenue per query, NOT just model accuracy.", tags:["evaluation"] },
    { q:"Why not use a transformer (GPT-style) for CTR prediction?", a:"Transformers are increasingly used but face challenges: (1) Latency — self-attention is O(n²) where n = number of tokens/features. With 500 candidates × 50 features, this is prohibitive for <10ms serving. (2) Sparse features — CTR models have billions of sparse IDs (ad_id, keyword_id). Transformers handle dense sequences well but struggle with huge sparse vocabularies. The embedding table dominates memory, not the transformer layers. (3) What works: use transformer for encoding sequences (user click history, query tokens) as a pre-processing step. Feed the transformer output AS A FEATURE into the DCN model. This gets you sequential understanding without the serving cost. Google's recent work uses small transformers inside the ranking model for history encoding.", tags:["ml"] },
    { q:"How do you prevent advertisers from gaming the quality score?", a:"Quality score is computed from multiple signals that are hard to simultaneously game: (1) Historical CTR (position-normalized) — you can't fake real clicks at scale without spending money. (2) Landing page quality — page load speed, mobile-friendliness, content relevance. Scored by a separate crawler. (3) Ad relevance — NLP model compares ad text to query. Keyword stuffing hurts readability and reduces CTR. (4) User experience signals — bounce rate after click, time on landing page. (5) Quality score is relative to competitors — even if you improve, competitors can too. The key insight: quality score improvements genuinely improve the user experience, so gaming it requires actually making better ads — which is the desired outcome.", tags:["auction"] },
    { q:"How does budget pacing work at scale?", a:"Budget pacing ensures a $1000/day budget is spent evenly across the day, not all by 9am. Algorithm: (1) Each campaign has a 'pace multiplier' (0 to 1). When pace_multiplier = 0.5, the ad is randomly excluded from 50% of eligible auctions. (2) Every 15 minutes, recompute pace_multiplier: remaining_budget / expected_remaining_cost. If 60% of budget remains with 40% of day left, pace_multiplier increases (spend faster). (3) Implementation: the budget ledger is a sharded counter (by campaign_id) in Redis. Each impression atomically decrements the budget. If budget hits zero, campaign is paused. (4) Challenge: cross-region coordination. A campaign targeting the US spends in both US-East and US-West regions. Each region holds a budget allocation, with rebalancing every minute.", tags:["infrastructure"] },
    { q:"What's the difference between click-through rate and click-through conversion rate?", a:"CTR = P(click | ad shown). CTCVR = P(conversion | ad shown) = P(click) × P(conversion | click). Two-stage funnel: (1) CTR model predicts if user will click. (2) CVR model predicts if clicking user will convert (purchase, sign up). They're often trained as multi-task models with shared layers. CTR has abundant data (billions of impressions). CVR has extremely sparse data (1-5% of clicks convert, days of delay). The selection bias problem: CVR model only sees data for clicked ads, not all shown ads. If you train naively, you learn P(conversion | clicked), but need P(conversion | shown). Solution: ESMM (Entire Space Multi-Task Model) trains on impression space, not just click space.", tags:["ml"] },
    { q:"How would you handle a sudden spike in a trending query?", a:"Example: Super Bowl → sudden spike in 'Super Bowl ads' queries. Challenges: (1) No historical CTR for this specific query-ad pair combination (cold start). (2) Massive traffic spike overwhelms model serving. (3) Budget pacing breaks — advertisers with related keywords exhaust budget in minutes. Solutions: (1) Query embedding similarity — even if 'Super Bowl ads 2024' is new, embedding is close to 'Super Bowl commercials' which has history. (2) Auto-scaling: model serving fleet scales based on QPS, not time. Pre-warming for predicted events. (3) Budget surge protection — temporarily increase pacing multiplier for event-related campaigns. Advertisers can opt into 'event budgets'. (4) Pre-compute trending query CTRs from early signals (first 100 impressions → initial CTR estimate).", tags:["scalability"] },
  ];
  return (
    <div className="space-y-2.5">
      <Card className="bg-indigo-50/50 border-indigo-200 mb-1">
        <p className="text-[12px] text-stone-500">Common follow-up questions Google L6 interviewers ask about ads systems. Click to reveal a strong answer.</p>
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
  api: ApiSection, design: DesignSection, auction: AuctionSection,
  ranking: RankingSection, features: FeaturesSection, calibration: CalibrationSection,
  data: DataModelSection, training: TrainingSection, serving: ServingSection,
  scalability: ScalabilitySection, watchouts: WatchoutsSection,
  observability: ObservabilitySection,
  enhancements: EnhancementsSection, followups: FollowupsSection,
};

export default function AdsCTRSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Ads Click Prediction (CTR)</h1>
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