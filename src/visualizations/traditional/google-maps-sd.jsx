import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   GOOGLE MAPS (Collaborative) — System Design Reference
   Pearl white theme · 17 sections (HLD + LLD)
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",             icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",         icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",  icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",           icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",    icon: "🏗️", color: "#9333ea" },
  { id: "algorithm",     label: "Routing Deep Dive",    icon: "⚙️", color: "#c026d3" },
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
            <Label>What is Google Maps (Collaborative)?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A mapping platform that renders interactive maps, provides turn-by-turn navigation, search for places, and enables <strong>collaborative features</strong> — shared trip planning, live location sharing, collaborative lists/reviews, and real-time traffic powered by crowd-sourced data from millions of users.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              The fundamental challenges: (1) Rendering the entire planet as interactive map tiles across 20+ zoom levels. (2) Computing optimal routes in real-time over a road graph with billions of edges. (3) Enabling millions of users to collaborate — share locations live, co-edit trip plans, contribute reviews — all with sub-second latency and strong consistency.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard?</Label>
            <ul className="space-y-2.5">
              <Point icon="🗺️" color="#0891b2">Map tile rendering — the entire Earth across 20+ zoom levels = trillions of tiles. Pre-render vs on-demand? Vector vs raster?</Point>
              <Point icon="🧭" color="#0891b2">Routing at scale — find shortest path on a graph with 1B+ nodes and 2B+ edges in &lt;200ms. Dijkstra is too slow. Need contraction hierarchies or A*.</Point>
              <Point icon="🚗" color="#0891b2">Real-time traffic — millions of GPS pings/second from phones. Aggregate, process, and update route ETAs in near real-time. Stale data = wrong ETAs.</Point>
              <Point icon="👥" color="#0891b2">Collaborative features — live location sharing to N friends with sub-second updates. Shared lists with concurrent edits. Reviews with spam detection at scale.</Point>
              <Point icon="🌍" color="#0891b2">Global consistency — map data updated constantly (new roads, closures, business hours). Must propagate worldwide without downtime.</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Scale</Label>
            <div className="space-y-2.5">
              {[
                { co: "Google Maps", scale: "1B+ MAU, 10B+ map loads/day", approach: "Vector tiles, CH routing" },
                { co: "Apple Maps", scale: "500M+ MAU, tight OS integration", approach: "Vector tiles, on-device ML" },
                { co: "Waze", scale: "150M+ MAU, crowd-sourced traffic", approach: "Community-driven data" },
                { co: "Mapbox", scale: "700M+ MAU (B2B), developer SDK", approach: "Open data, custom styling" },
                { co: "HERE Maps", scale: "OEM platform, autonomous driving", approach: "HD maps, sensor fusion" },
              ].map((e,i) => (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-28 shrink-0">{e.co}</span>
                  <span className="text-stone-500 flex-1">{e.scale}</span>
                  <span className="text-stone-400 text-[10px]">{e.approach}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <Label color="#2563eb">The Core Architecture Layers</Label>
            <svg viewBox="0 0 360 150" className="w-full">
              <rect x={10} y={5} width={340} height={32} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
              <text x={180} y={18} textAnchor="middle" fill="#2563eb" fontSize="10" fontWeight="700" fontFamily="monospace">Map Rendering (Tile Server + CDN)</text>
              <text x={180} y={30} textAnchor="middle" fill="#2563eb80" fontSize="8" fontFamily="monospace">Vector tiles, 20+ zoom levels, pre-rendered + dynamic</text>

              <rect x={10} y={45} width={165} height={32} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
              <text x={92} y={58} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="700" fontFamily="monospace">Routing Engine</text>
              <text x={92} y={70} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">CH + real-time traffic</text>

              <rect x={185} y={45} width={165} height={32} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
              <text x={267} y={58} textAnchor="middle" fill="#d97706" fontSize="10" fontWeight="700" fontFamily="monospace">Search / Geocoding</text>
              <text x={267} y={70} textAnchor="middle" fill="#d9770680" fontSize="8" fontFamily="monospace">Places, addresses, POIs</text>

              <rect x={10} y={85} width={340} height={32} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
              <text x={180} y={98} textAnchor="middle" fill="#c026d3" fontSize="10" fontWeight="700" fontFamily="monospace">Collaboration Layer ★</text>
              <text x={180} y={110} textAnchor="middle" fill="#c026d380" fontSize="8" fontFamily="monospace">Live sharing, shared lists, reviews, real-time sync</text>

              <rect x={10} y={125} width={340} height={22} rx={6} fill="#9333ea08" stroke="#9333ea" strokeWidth={1.5}/>
              <text x={180} y={139} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Traffic Data Pipeline (crowd-sourced GPS → ETA updates)</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Top 5 most asked system design question</div>
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
            <p className="text-[12px] text-stone-500 mt-0.5">"Google Maps" is enormous. Clarify: Are we building the tile rendering pipeline, the routing engine, or the collaborative features? For a 45-min interview, pick <strong>navigation + collaborative trip planning</strong> as the core, mention tile serving as a known solved problem (CDN). Don't try to design Street View, satellite imagery, and indoor maps all at once.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Render interactive maps with pan, zoom, rotate (vector tiles, 20+ zoom levels)</Point>
            <Point icon="2." color="#059669">Search for places, addresses, businesses (geocoding + reverse geocoding)</Point>
            <Point icon="3." color="#059669">Compute driving/walking/transit routes with real-time ETA (turn-by-turn navigation)</Point>
            <Point icon="4." color="#059669">Live location sharing — share real-time position with N friends for a configurable duration</Point>
            <Point icon="5." color="#059669">Collaborative trip planning — create shared lists of places, co-edit itineraries, vote on destinations</Point>
            <Point icon="6." color="#059669">User reviews and ratings for places — submit, read, upvote, with photo uploads</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Map tile load: p99 &lt;200ms (first tile visible). Tiles must be CDN-cacheable.</Point>
            <Point icon="2." color="#dc2626">Route computation: p99 &lt;500ms for city-scale, &lt;2s for cross-country routes</Point>
            <Point icon="3." color="#dc2626">Live location updates: &lt;2s end-to-end latency from sender GPS to receiver's map</Point>
            <Point icon="4." color="#dc2626">Scale to 1B+ MAU, 10B+ tile loads/day, 500M+ route queries/day</Point>
            <Point icon="5." color="#dc2626">High availability — maps and navigation must work even during partial outages. Offline fallback for cached tiles.</Point>
            <Point icon="6." color="#dc2626">Eventual consistency OK for reviews/traffic. Strong consistency for collaborative list edits.</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Vector tiles or raster tiles? (vector is modern, smaller, interactive)",
            "Scope of routing — driving only, or also walking/transit/cycling?",
            "Real-time traffic integration or static road weights?",
            "How many concurrent users in a live location sharing session?",
            "Collaborative lists — real-time co-editing or async?",
            "Do we need offline support (cached maps + routing)?",
            "Multi-modal routing (drive + park + walk)?",
            "Single region or global deployment?",
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
            <p className="text-[12px] text-stone-500 mt-0.5">Start from 1B MAU, derive DAU (~40%), then multiply by actions per user. Maps is extremely read-heavy: most users view tiles and get routes. Writes come from GPS telemetry (traffic), reviews, and collaborative edits. The read:write ratio for tiles is 10,000:1. For navigation, it's ~100:1.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Traffic Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="1B MAU × 40% = 400M DAU" result="400M DAU" note="~40% daily active ratio for navigation apps" />
            <MathStep step="2" formula="400M DAU × 25 tile loads/session = 10B tiles/day" result="10B tiles/day" note="Average map session loads ~25 tiles (pan + zoom)" />
            <MathStep step="3" formula="10B / 86400 = ~115K tile requests/sec" result="~115K QPS" note="Peak: 3× average = ~350K tile QPS" />
            <MathStep step="4" formula="400M DAU × 1.5 routes/day = 600M routes/day" result="600M routes/day" note="Many users check routes without navigating" />
            <MathStep step="5" formula="600M / 86400 = ~7K route queries/sec" result="~7K QPS" note="Peak: ~20K QPS during rush hour" final />
          </div>
        </Card>
        <Card accent="#c026d3">
          <Label color="#c026d3">Step 2 — Collaboration & Traffic Data</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="GPS telemetry: 100M active navigators × 1 ping/sec" result="100M GPS/sec" note="Phone sends location every 1-4 seconds during navigation" />
            <MathStep step="2" formula="Aggregate to road segments: 100M pings → ~5M segment updates/sec" result="5M seg/sec" note="Multiple pings mapped to same road segment, averaged" final />
            <MathStep step="3" formula="Live location sharing: 50M active sessions × 1 update/3s" result="~17M updates/sec" note="Each session shares with 1-8 friends" />
            <MathStep step="4" formula="Collaborative lists: 10M active editors" result="~500 edits/sec" note="Low write volume — most users read, few edit" />
            <MathStep step="5" formula="Reviews: 5M new reviews/day = ~60 writes/sec" result="~60 writes/sec" note="But 500M+ reads/day on review content" />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Storage Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Vector tile set (all zoom levels, global) = ~1-2 TB compressed" result="~2 TB" note="Vector tiles are compact. Raster would be 50-100 TB." />
            <MathStep step="2" formula="Road graph: 1B nodes × 64B + 2B edges × 32B" result="~130 GB" note="Fits in memory on a cluster. Contraction hierarchy adds ~2×." />
            <MathStep step="3" formula="Place/POI database: 250M places × 2 KB avg" result="~500 GB" note="Business info, hours, photos metadata, coordinates" />
            <MathStep step="4" formula="Reviews: 2B reviews × 500B avg" result="~1 TB" note="Text + metadata. Photos in object storage separately." final />
            <MathStep step="5" formula="Traffic data (current speeds per segment): 50M segments × 16B" result="~800 MB" note="Fits in Redis. Updated every few seconds per segment." />
            <MathStep step="6" formula="User data + lists + sharing: 1B users × 1 KB" result="~1 TB" note="Profiles, saved places, collaborative lists" />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Infrastructure Sizing</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Tile CDN: 99%+ cache hit rate → ~350K QPS at CDN edge" result="CDN-dominated" note="Origin servers handle <1% = ~3.5K QPS" final />
            <MathStep step="2" formula="Routing cluster: graph in memory, 20K peak QPS" result="50-100 servers" note="Each server: 256GB RAM, handles ~200-400 routes/sec" />
            <MathStep step="3" formula="Traffic pipeline: 100M GPS pings/sec → Kafka + Flink" result="200+ Kafka partitions" note="Stream processing cluster for real-time aggregation" />
            <MathStep step="4" formula="Location sharing: 17M updates/sec, Redis Pub/Sub" result="50+ Redis nodes" note="Pub/sub for real-time delivery to connected friends" />
            <MathStep step="5" formula="Monthly cost estimate (core services)" result="~$2-3M/mo" note="CDN bandwidth is the biggest cost. Routing cluster second." />
          </div>
          <div className="mt-4 pt-3 border-t border-stone-100 text-[11px] text-stone-500">
            <strong className="text-stone-700">Key insight:</strong> Maps is CDN-dominated. 99%+ of tile traffic is served from edge caches. The routing engine is CPU-bound (graph algorithms). Traffic pipeline is throughput-bound (GPS ingestion).
          </div>
        </Card>
      </div>
      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Peak Tile QPS", val: "~350K", sub: "99% CDN cache hit" },
            { label: "Route Queries", val: "~20K QPS", sub: "Peak rush hour" },
            { label: "GPS Telemetry", val: "100M/sec", sub: "Real-time traffic" },
            { label: "Road Graph", val: "~130 GB", sub: "Fits in memory" },
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
          <Label color="#2563eb">Map & Navigation APIs</Label>
          <CodeBlock code={`# GET /v1/tiles/{z}/{x}/{y}.mvt
# Returns: vector tile in Mapbox Vector Tile format
# Cached at CDN edge. Z=zoom, X/Y=tile coordinates.
# Headers: Cache-Control: public, max-age=86400

# GET /v1/directions?origin=lat,lng&dest=lat,lng
#   &mode=driving&departure_time=now
# Returns:
{
  "routes": [{
    "distance_m": 12450,
    "duration_s": 1080,
    "eta": "2024-02-09T11:18:00Z",
    "polyline": "encoded_polyline_string",
    "steps": [
      {
        "instruction": "Turn right onto Main St",
        "distance_m": 350,
        "duration_s": 45,
        "maneuver": "turn-right",
        "polyline": "step_polyline"
      }
    ],
    "traffic_model": "best_guess"
  }],
  "alternatives": 2
}

# GET /v1/places/search?q=coffee&lat=37.7&lng=-122.4
#   &radius=2000&limit=20
# Returns: ranked list of matching places

# GET /v1/geocode?address=1600+Amphitheatre+Parkway
# GET /v1/reverse-geocode?lat=37.422&lng=-122.084`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Collaboration APIs</Label>
          <CodeBlock code={`# POST /v1/location-sharing/sessions
# Start sharing live location with friends
{
  "shared_with": ["user_456", "user_789"],
  "duration_minutes": 60,
  "update_interval_s": 3
}
# Returns: { "session_id": "sess_abc", "ws_url": "wss://..." }

# WebSocket /v1/location-sharing/ws?session=sess_abc
# Bidirectional: send location, receive friend locations
# → { "user_id":"u42", "lat":37.77, "lng":-122.41, "ts":... }

# POST /v1/lists
# Create a collaborative list (trip plan)
{
  "name": "Tokyo Trip 2025",
  "collaborators": ["user_456", "user_789"],
  "type": "trip_plan"
}

# PUT /v1/lists/:id/items/:item_id
# Add/edit place in shared list (with conflict resolution)
{
  "place_id": "place_abc",
  "note": "Best ramen here!",
  "vote": "up",
  "version": 14     # optimistic concurrency control
}

# POST /v1/places/:id/reviews
{
  "rating": 4,
  "text": "Great coffee, slow wifi",
  "photos": ["photo_url_1"]
}`} />
        </Card>
      </div>
      <Card>
        <Label color="#d97706">Design Decisions</Label>
        <div className="space-y-3">
          {[
            { q: "Vector tiles vs raster tiles?", a: "Vector. Client renders tiles locally using the GPU. 10× smaller than raster PNGs. Supports rotation, smooth zoom, dynamic styling (night mode), and offline caching. All modern map apps use vector tiles." },
            { q: "WebSocket for live location sharing?", a: "Yes. HTTP polling at 1 update/3s × millions of sessions = massive overhead. WebSocket gives sub-second delivery and bidirectional updates. Fall back to SSE if WS is blocked." },
            { q: "Optimistic concurrency for shared lists?", a: "Each edit includes a version number. Server rejects stale writes (409 Conflict). Client fetches latest, merges locally, retries. For real-time co-editing: use CRDTs or OT (operational transforms)." },
            { q: "Encoded polyline for routes?", a: "Google's encoded polyline algorithm compresses lat/lng sequences by ~90%. A 100-step route = ~2KB instead of ~20KB. Client decodes and draws on the map. Bandwidth matters at mobile scale." },
            { q: "Tile URL scheme: z/x/y?", a: "Standard slippy map convention. Z = zoom level (0-22), X/Y = tile grid coordinates. At Z=0, the whole world is 1 tile. At Z=22, each tile covers ~5m². CDN caches entire URL paths." },
          ].map((d,i) => (
            <div key={i}>
              <div className="text-[11px] font-bold text-stone-700">{d.q}</div>
              <div className="text-[11px] text-stone-500 mt-0.5">{d.a}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function DesignSection() {
  const [phase, setPhase] = useState(0);
  const phases = [
    { label: "Tile Serving", desc: "Map tiles are the most traffic-intensive component. Pre-render vector tiles from OSM/proprietary map data, store in object storage, serve via multi-layer CDN. Client GPU renders the final image. At 350K QPS peak, CDN cache hit rate must be >99%." },
    { label: "Routing Engine ★", desc: "The core algorithmic challenge. Build a road graph from map data. Pre-process with contraction hierarchies to reduce query time from seconds to milliseconds. Layer real-time traffic speeds on top of static weights. Return ranked routes with alternatives." },
    { label: "Collaboration Layer ★★", desc: "The differentiating feature. Live location sharing via WebSocket pub/sub (Redis). Collaborative lists with CRDT-based conflict resolution. Reviews with spam detection + ranking. All backed by event-driven architecture for real-time sync across devices." },
  ];
  const diagrams = [
    () => (
      <svg viewBox="0 0 460 170" className="w-full">
        <DiagramBox x={55} y={50} w={70} h={36} label="Client" color="#2563eb"/>
        <DiagramBox x={170} y={50} w={70} h={36} label="CDN\nEdge" color="#d97706"/>
        <DiagramBox x={290} y={50} w={80} h={36} label="Tile\nServer" color="#9333ea"/>
        <DiagramBox x={410} y={30} w={70} h={28} label="Tile DB" color="#059669"/>
        <DiagramBox x={410} y={75} w={70} h={28} label="Map Data" color="#059669"/>
        <Arrow x1={90} y1={50} x2={135} y2={50} label="GET tile" id="t1"/>
        <Arrow x1={205} y1={50} x2={250} y2={50} label="cache miss" id="t2" dashed/>
        <Arrow x1={330} y1={38} x2={375} y2={33} id="t3"/>
        <Arrow x1={330} y1={62} x2={375} y2={72} id="t4" dashed/>
        <rect x={80} y={120} width={280} height={20} rx={4} fill="#05966908" stroke="#05966930"/>
        <text x={220} y={132} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">✓ 99%+ served from CDN. Origin only on cold tiles.</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 190" className="w-full">
        <DiagramBox x={55} y={50} w={70} h={36} label="Client" color="#2563eb"/>
        <DiagramBox x={175} y={50} w={80} h={40} label="Routing\nService" color="#9333ea"/>
        <DiagramBox x={310} y={25} w={90} h={28} label="Road Graph" color="#059669" sub="in-memory"/>
        <DiagramBox x={310} y={65} w={90} h={28} label="Traffic Data" color="#d97706" sub="Redis"/>
        <DiagramBox x={310} y={105} w={90} h={28} label="CH Index" color="#c026d3" sub="pre-computed"/>
        <Arrow x1={90} y1={50} x2={135} y2={50} label="directions" id="r1"/>
        <Arrow x1={215} y1={35} x2={265} y2={28} id="r2"/>
        <Arrow x1={215} y1={55} x2={265} y2={68} id="r3"/>
        <Arrow x1={215} y1={70} x2={265} y2={102} id="r4" dashed/>
        <rect x={70} y={150} width={300} height={20} rx={4} fill="#9333ea08" stroke="#9333ea30"/>
        <text x={220} y={162} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">✓ CH reduces billion-node queries to ~1000 node visits</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 200" className="w-full">
        <DiagramBox x={55} y={40} w={70} h={36} label="User A" color="#2563eb" sub="shares location"/>
        <DiagramBox x={55} y={110} w={70} h={36} label="User B" color="#2563eb" sub="views friend"/>
        <DiagramBox x={190} y={75} w={90} h={40} label="Location\nHub" color="#c026d3"/>
        <DiagramBox x={330} y={40} w={80} h={30} label="Redis\nPub/Sub" color="#dc2626"/>
        <DiagramBox x={330} y={90} w={80} h={30} label="Session\nStore" color="#d97706"/>
        <DiagramBox x={330} y={135} w={80} h={30} label="List CRDT\nService" color="#059669"/>
        <Arrow x1={90} y1={50} x2={145} y2={65} label="WS update" id="c1"/>
        <Arrow x1={145} y1={100} x2={90} y2={110} label="WS push" id="c2"/>
        <Arrow x1={235} y1={65} x2={290} y2={48} label="publish" id="c3"/>
        <Arrow x1={235} y1={80} x2={290} y2={100} id="c4"/>
        <rect x={70} y={175} width={310} height={20} rx={4} fill="#c026d308" stroke="#c026d330"/>
        <text x={225} y={187} textAnchor="middle" fill="#c026d3" fontSize="8" fontFamily="monospace">✓ Sub-second location sharing via WS + Redis Pub/Sub</text>
      </svg>
    ),
  ];
  return (
    <div className="space-y-5">
      <Card accent="#9333ea">
        <Label color="#9333ea">Architecture Layers — The Three Pillars</Label>
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
        <Label color="#c026d3">Full System Architecture</Label>
        <svg viewBox="0 0 680 260" className="w-full">
          <DiagramBox x={50} y={45} w={65} h={34} label="Client" color="#2563eb"/>
          <DiagramBox x={150} y={45} w={70} h={34} label="Gateway" color="#6366f1"/>
          {/* Map path */}
          <DiagramBox x={280} y={20} w={80} h={28} label="Tile Server" color="#059669"/>
          <DiagramBox x={420} y={20} w={70} h={28} label="CDN" color="#d97706"/>
          {/* Routing path */}
          <DiagramBox x={280} y={60} w={80} h={30} label="Routing\nEngine" color="#9333ea"/>
          <DiagramBox x={420} y={60} w={80} h={28} label="Traffic\nAggregator" color="#c026d3"/>
          {/* Collab path */}
          <DiagramBox x={280} y={105} w={80} h={30} label="Collab\nService" color="#0891b2"/>
          <DiagramBox x={420} y={105} w={80} h={28} label="Location\nHub" color="#dc2626"/>
          {/* Data stores */}
          <DiagramBox x={560} y={20} w={75} h={28} label="Tile\nStore" color="#059669"/>
          <DiagramBox x={560} y={60} w={75} h={28} label="Road\nGraph" color="#9333ea"/>
          <DiagramBox x={560} y={105} w={75} h={28} label="Redis\nPub/Sub" color="#dc2626"/>
          {/* Bottom data */}
          <DiagramBox x={280} y={165} w={75} h={28} label="Place DB" color="#d97706"/>
          <DiagramBox x={420} y={165} w={80} h={28} label="Review DB" color="#b45309"/>
          <DiagramBox x={560} y={165} w={75} h={28} label="User\nStore" color="#78716c"/>

          <Arrow x1={82} y1={45} x2={115} y2={45} id="a1"/>
          <Arrow x1={185} y1={33} x2={240} y2={23} label="tiles" id="a2"/>
          <Arrow x1={185} y1={45} x2={240} y2={63} label="route" id="a3"/>
          <Arrow x1={185} y1={55} x2={240} y2={102} label="collab" id="a4"/>
          <Arrow x1={320} y1={20} x2={385} y2={20} id="a5"/>
          <Arrow x1={455} y1={20} x2={523} y2={20} id="a5b"/>
          <Arrow x1={320} y1={60} x2={380} y2={60} id="a6"/>
          <Arrow x1={460} y1={60} x2={523} y2={60} id="a6b"/>
          <Arrow x1={320} y1={105} x2={380} y2={105} id="a7"/>
          <Arrow x1={460} y1={105} x2={523} y2={105} id="a7b"/>

          <rect x={140} y={215} width={400} height={38} rx={6} fill="#faf9f7" stroke="#e7e5e4"/>
          <text x={160} y={230} fill="#059669" fontSize="8" fontFamily="monospace">Tile path: Client → CDN edge → Tile Server → Object Storage</text>
          <text x={160} y={244} fill="#9333ea" fontSize="8" fontFamily="monospace">Route path: Client → Routing Engine (in-memory graph + traffic)</text>
        </svg>
      </Card>
    </div>
  );
}

function AlgorithmSection() {
  const [sel, setSel] = useState("contraction");
  const algos = {
    contraction: { name: "Contraction Hierarchies ★", cx: "Preprocess: O(n·log(n)) / Query: O(√n)",
      pros: ["Query time: ~1ms for continental routes (from seconds with Dijkstra)","Pre-computed shortcuts bypass low-importance nodes","Bidirectional search meets in the middle at high-importance nodes","Industry standard: used by Google Maps, OSRM, Valhalla"],
      cons: ["Pre-processing takes hours for a continent-scale graph","Static weights: must rebuild when road network changes","Adding real-time traffic requires weight overlay on top of CH","Memory overhead: 2-3× the original graph size"],
      when: "Default choice for production routing engines. Pre-compute the CH once (nightly), then use overlay for real-time traffic. Query is a modified bidirectional Dijkstra that only relaxes 'upward' edges in the hierarchy.",
      code: `# Contraction Hierarchies — Preprocessing
# Input: road graph G = (nodes, edges) with weights
#
# Idea: Rank nodes by "importance" (traffic, degree)
# Contract least important nodes first, add shortcuts

preprocess(graph):
    # 1. Assign importance to each node
    for node in graph.nodes:
        node.importance = compute_importance(node)
        # importance = edge_difference + node_level

    # 2. Contract nodes in order of importance
    for node in sorted(graph.nodes, key=importance):
        # For each pair of neighbors (u, v)
        for u, v in neighbor_pairs(node):
            # If shortest u→v goes through this node
            if shortest_path(u, v) goes through node:
                add_shortcut(u, v, weight=w(u,node)+w(node,v))
        mark_contracted(node)

    # Result: hierarchy with shortcuts
    # Top nodes = highways. Bottom = residential streets.

# Query: bidirectional Dijkstra, only go "upward"
query(source, target):
    forward = dijkstra_upward(source)   # upward only
    backward = dijkstra_upward(target)  # upward only
    # Meet at highest-importance node on shortest path
    return min(forward[v] + backward[v] for v in visited)` },
    astar: { name: "A* with Landmarks (ALT)", cx: "Query: O(k·log(k)) where k << n",
      pros: ["Heuristic guides search toward target — visits far fewer nodes","Landmarks provide tight lower bounds for A* heuristic","No pre-processing needed for basic A* (but landmarks help)","Handles dynamic weights (traffic) naturally"],
      cons: ["Slower than CH for long-distance routes (still O(thousands) of nodes)","Landmark selection affects quality — poor landmarks = bad heuristic","Memory for landmark distances: O(nodes × landmarks)","Not as fast as CH for production continental routing"],
      when: "Good for short-distance routes or when road network changes frequently. Use as fallback when CH is being rebuilt. Also used for walking/cycling where road graph is smaller and changes more often.",
      code: `# A* with Landmarks (ALT Algorithm)
# Landmarks: pre-selected reference points (e.g., major cities)
# For each landmark L, pre-compute dist(v, L) for all v

astar_alt(source, target, landmarks):
    # Heuristic: use triangle inequality with landmarks
    def h(node):
        return max(
            abs(dist(node, L) - dist(target, L))
            for L in landmarks
        )
        # This is a valid lower bound on dist(node, target)

    # Standard A* with this heuristic
    open_set = PriorityQueue()
    open_set.push(source, g=0, f=h(source))
    g_score = {source: 0}

    while not open_set.empty():
        current = open_set.pop()
        if current == target:
            return reconstruct_path()

        for neighbor, weight in graph.edges(current):
            tentative_g = g_score[current] + weight
            if tentative_g < g_score.get(neighbor, INF):
                g_score[neighbor] = tentative_g
                f = tentative_g + h(neighbor)
                open_set.push(neighbor, f)

# Typical: 16-32 landmarks placed at graph periphery
# Reduces nodes visited by 10-100× vs plain Dijkstra` },
    traffic: { name: "Real-Time Traffic Overlay", cx: "Ingest: O(GPS pings) / Update: O(segments)",
      pros: ["Live ETAs based on actual current road speeds","Crowd-sourced: more users = better data (network effect)","Can detect incidents, closures, congestion in near real-time","Enables predictive routing: 'leave at 5pm → this route is 20min faster'"],
      cons: ["GPS data is noisy — must aggregate and smooth per road segment","Privacy: tracking millions of users' locations continuously","Stale data on low-traffic roads — insufficient sample size","Cold start: new roads or areas with few users have no data"],
      when: "Essential for any production routing. Layer traffic speeds on top of CH/A* base weights. Google uses billions of GPS points daily from Android phones to build the most accurate traffic model. This is the competitive moat.",
      code: `# Real-Time Traffic Pipeline
# Input: GPS pings from millions of phones
# Output: speed per road segment, updated every 30-60 seconds

ingest_gps(ping):
    # 1. Map-match: snap GPS to nearest road segment
    segment_id = map_match(ping.lat, ping.lng, ping.heading)

    # 2. Compute speed on this segment
    speed = ping.distance_since_last / ping.time_since_last

    # 3. Push to Kafka (partitioned by region)
    kafka.produce("gps-speeds", key=segment_id, value=speed)

aggregate_traffic():
    # Flink/Spark Streaming job — runs continuously
    # Window: 60-second tumbling window per segment

    for segment_id, speeds in window:
        # Filter outliers (stopped cars, GPS drift)
        filtered = remove_outliers(speeds, method="IQR")

        # Weighted average (recent pings weighted more)
        avg_speed = weighted_average(filtered)

        # Compute congestion ratio
        free_flow = segment.speed_limit
        congestion = avg_speed / free_flow  # 0.0-1.0

        # Update traffic store (Redis)
        redis.set(f"traffic:{segment_id}", {
            speed: avg_speed,
            congestion: congestion,
            sample_size: len(filtered),
            updated_at: now()
        })

# Routing uses traffic overlay:
# edge_weight = segment_length / traffic_speed(segment_id)` },
  };
  const a = algos[sel];
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Routing Algorithm Comparison</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[12px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Algorithm</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Preprocess</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Query Time</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Dynamic?</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Use Case</th>
            </tr></thead>
            <tbody>
              {[
                { n:"Dijkstra", pre:"None", q:"Seconds", dyn:"✓ Yes", use:"Textbook only", hl:false },
                { n:"A* + Landmarks", pre:"Minutes", q:"10-100ms", dyn:"✓ Yes", use:"Short routes, fallback", hl:false },
                { n:"Contraction Hierarchies ★", pre:"Hours", q:"<1ms", dyn:"✗ Static", use:"Production routing", hl:true },
                { n:"CH + Traffic Overlay ★★", pre:"Hours + RT", q:"1-5ms", dyn:"✓ Overlay", use:"Google Maps, Waze", hl:true },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-purple-50" : ""}>
                  <td className={`px-3 py-2 font-mono ${r.hl?"text-purple-700 font-bold":"text-stone-600"}`}>{r.n}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.pre}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.q}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.dyn}</td>
                  <td className="text-center px-3 py-2 text-stone-400">{r.use}</td>
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
          <CodeBlock code={`-- Road segments (the graph edges)
CREATE TABLE road_segments (
  segment_id    BIGINT PRIMARY KEY,
  start_node_id BIGINT NOT NULL,
  end_node_id   BIGINT NOT NULL,
  distance_m    INT NOT NULL,
  speed_limit   SMALLINT,        -- km/h
  road_class    ENUM('highway','trunk','primary','secondary','residential'),
  one_way       BOOLEAN DEFAULT FALSE,
  polyline      GEOMETRY,        -- actual road shape
  INDEX idx_start (start_node_id),
  INDEX idx_end (end_node_id)
);
-- In-memory representation: adjacency list for routing

-- Places / POIs (sharded by geohash prefix)
CREATE TABLE places (
  place_id      BIGINT PRIMARY KEY,
  name          VARCHAR(255) NOT NULL,
  category      VARCHAR(64),     -- restaurant, gas_station, hotel
  lat           DECIMAL(9,6),
  lng           DECIMAL(9,6),
  geohash       CHAR(12),        -- for spatial queries
  address       TEXT,
  rating        DECIMAL(2,1),    -- 1.0-5.0
  review_count  INT DEFAULT 0,
  business_hours JSON,
  INDEX idx_geohash (geohash),
  INDEX idx_category_geo (category, geohash)
);

-- Collaborative lists (sharded by list_id)
CREATE TABLE collab_lists (
  list_id       BIGINT PRIMARY KEY,
  owner_id      BIGINT NOT NULL,
  name          VARCHAR(255),
  type          ENUM('saved','trip_plan','favorites'),
  version       INT DEFAULT 0,   -- optimistic concurrency
  created_at    TIMESTAMP,
  updated_at    TIMESTAMP
);

-- List items
CREATE TABLE list_items (
  item_id       BIGINT PRIMARY KEY,
  list_id       BIGINT NOT NULL,
  place_id      BIGINT NOT NULL,
  added_by      BIGINT NOT NULL,
  note          TEXT,
  position      INT,             -- ordering within list
  votes_up      INT DEFAULT 0,
  INDEX idx_list (list_id, position)
);

-- Reviews (sharded by place_id)
CREATE TABLE reviews (
  review_id     BIGINT PRIMARY KEY,  -- Snowflake ID
  place_id      BIGINT NOT NULL,
  user_id       BIGINT NOT NULL,
  rating        TINYINT NOT NULL,    -- 1-5
  text          TEXT,
  photo_urls    JSON,
  helpful_count INT DEFAULT 0,
  created_at    TIMESTAMP,
  INDEX idx_place_time (place_id, created_at DESC)
);`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Why These Design Choices?</Label>
          <div className="space-y-3">
            {[
              { q: "Road graph in memory, not DB?", a: "Routing requires traversing millions of edges per query. Disk-based DB is 1000× too slow. Load the entire graph into RAM across a cluster. 130 GB fits on a few large-memory servers. Pre-processed CH index also in memory." },
              { q: "Geohash for spatial queries?", a: "Geohash converts 2D coordinates into a 1D string that preserves locality. Prefix search on geohash = bounding box query. 'Find restaurants near me' = SELECT WHERE geohash LIKE '9q8yy%'. Works with standard B-tree indexes." },
              { q: "Shard places by geohash prefix?", a: "Geographically co-located data on the same shard. 'Nearby search' is single-shard in most cases. Trade-off: hot shards in dense cities (Manhattan). Mitigate with finer-grained sharding in hot zones." },
              { q: "Optimistic concurrency for lists?", a: "version column incremented on each write. UPDATE ... WHERE version = expected_version. If 0 rows affected → conflict. Simpler than pessimistic locking. Good enough for collaborative lists where conflicts are rare." },
              { q: "Snowflake IDs for reviews?", a: "Time-sortable, globally unique. Can sort reviews by review_id instead of created_at. No coordination needed across shards. Encodes timestamp for time-based queries." },
              { q: "Why not a graph database for roads?", a: "Graph DBs (Neo4j) are great for traversals but too slow for production routing at scale. The road graph is static enough to pre-process into CH format. Custom in-memory representation is 100× faster than any graph DB for shortest-path queries." },
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
          <Label color="#059669">Tile Serving Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Multi-layer CDN</strong> — edge PoPs worldwide cache tiles. L1 edge → L2 regional → origin. 99.5%+ cache hit rate at edge. Only cold or freshly-updated tiles hit origin.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Tile pre-rendering pipeline</strong> — nightly batch job renders all tiles for populated areas (zoom 0-16). On-demand rendering only for zoom 17+ (high detail, rarely requested).</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Vector tile compression</strong> — protobuf-encoded MVT format. ~20 KB per tile at mid-zoom. Gzip further reduces to ~5 KB. Client-side GPU rendering from vector data.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Tile versioning</strong> — tiles tagged with map data version. CDN invalidation on update. Stale tiles are acceptable for hours (roads don't change that fast).</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Routing Engine Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Partitioned graph</strong> — split the world into regions (continents, countries). Each routing server holds 1-2 regions in memory. Cross-region routes: stitch at border nodes.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Read replicas</strong> — the graph is read-only during queries. Replicate across N servers per region. Load-balance route requests. Each server handles ~200-400 QPS.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Traffic overlay (not rebuild)</strong> — don't rebuild CH when traffic changes. Overlay real-time speeds on top of pre-computed CH edges. Adjusts weights without full re-preprocessing.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Pre-computed popular routes</strong> — cache the top 10K most-requested origin-destination pairs. Home→work commute routes are highly repetitive. Cache hit saves full route computation.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Strategy</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Option A: Region-Local Everything ★", d:"Each region has its own tile cache, routing servers with local road graph, and collaboration services. Traffic data stays regional. Cross-region routes stitched at borders.", pros:["Lowest latency — all local","No cross-region dependency","Traffic data locality"], cons:["Cross-border routes need stitching","Map updates must propagate globally","Duplicate storage per region"], pick:true },
            { t:"Option B: CDN + Centralized Routing", d:"Tiles served from global CDN. Routing queries routed to nearest region with the relevant graph partition. Collaboration is globally replicated.", pros:["Simpler routing architecture","Single source of truth for graph","CDN handles tile distribution"], cons:["Cross-region routing latency","Single point of failure for graph","Collab latency for remote users"], pick:false },
            { t:"Option C: User-Homed + Global CDN", d:"Users homed to nearest region for collaboration features. Tiles global via CDN. Routing federated — query routed to the region containing the origin.", pros:["Best collab latency","CDN optimized for tiles","Routing scales per region"], cons:["Complex user-to-region mapping","Cross-region collab syncing","Routing handoff at borders"], pick:false },
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
        <Label color="#d97706">Critical Decision: What If Routing Goes Down?</Label>
        <p className="text-[12px] text-stone-500 mb-4">Navigation must always work. Users depend on it while driving. A routing outage at rush hour = millions of people lost. This is a safety-critical system.</p>
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Tier 1: Serve Cached Routes</div>
            <p className="text-[11px] text-stone-500">Popular routes are pre-cached. Serve from cache even if routing engine is down. Stale traffic data is better than no route. Cache top 100K routes per region.</p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-white p-4">
            <div className="text-[11px] font-bold text-amber-700 mb-1.5">Tier 2: Fallback to Static Routing</div>
            <p className="text-[11px] text-stone-500">Disable real-time traffic overlay. Route on static road weights (speed limits). Less accurate ETAs but still correct paths. Pre-computed CH still works without traffic.</p>
          </div>
          <div className="rounded-lg border border-red-200 bg-white p-4">
            <div className="text-[11px] font-bold text-red-700 mb-1.5">Tier 3: Client-Side Offline Routing</div>
            <p className="text-[11px] text-stone-500">Client has downloaded offline map data. Run simplified A* routing on-device. Limited to pre-downloaded region. Google Maps supports this with downloaded areas.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Availability Strategies</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Tile CDN is inherently HA</strong> — CDN PoPs are globally distributed. Origin failure → serve stale tiles from edge cache. Tiles have long TTL (hours to days). This is the easiest layer to keep up.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Routing N+2 redundancy</strong> — for each region, run N routing servers + 2 hot standbys. Health checks every 5s. Failed server replaced in seconds. Graph data loaded from shared storage on startup.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Traffic pipeline degradation</strong> — if traffic data pipeline fails, routing falls back to historical traffic patterns (same day/time last week). Still better than static speed limits.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Collaboration graceful degradation</strong> — if collaboration service is down: live location sharing pauses (last known location shown). Lists become read-only. Reviews queue locally and sync later.</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Consistency Guarantees</Label>
          <div className="space-y-3">
            {[
              { component: "Map Tiles", consistency: "Eventual", details: "Tiles update asynchronously. Users may see slightly outdated maps for hours. Acceptable — roads don't change often. CDN invalidation on major updates." },
              { component: "Traffic Data", consistency: "Eventual (30-60s)", details: "Traffic speeds are aggregated in 60-second windows. Routes computed with slightly stale traffic are still useful. Stale data → slightly off ETAs, not wrong routes." },
              { component: "Navigation (Active)", consistency: "Strong per-session", details: "Once a route is computed, the client follows it. Route recalculation triggered by significant deviation or user request. Session-local consistency." },
              { component: "Collaborative Lists", consistency: "Strong (versioned)", details: "Optimistic concurrency control. Reads always see latest committed version. Writes with stale version are rejected (409). CRDT for real-time co-editing." },
              { component: "Live Location", consistency: "Best-effort real-time", details: "Location updates are fire-and-forget. Missed update = show last known position. 2-3 second staleness is acceptable. No persistence needed." },
              { component: "Reviews", consistency: "Eventual", details: "New review may take seconds to appear. Rating recalculation is async. Spam detection runs asynchronously. User sees their own review immediately (read-after-write)." },
            ].map((c,i) => (
              <div key={i} className="flex items-start gap-2">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-bold text-stone-700">{c.component}</span>
                    <Pill bg={c.consistency.includes("Strong")?"#ecfdf5":"#fffbeb"} color={c.consistency.includes("Strong")?"#059669":"#d97706"}>{c.consistency}</Pill>
                  </div>
                  <div className="text-[10px] text-stone-400 mt-0.5">{c.details}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function ObservabilitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#0284c7">
          <Label color="#0284c7">Key Metrics</Label>
          <div className="space-y-3">
            {[
              { metric: "tile.latency_p99", type: "Histogram", desc: "Tile load time at p99. CDN edge should be <50ms. Origin miss <200ms. Track per zoom level." },
              { metric: "route.latency_p99", type: "Histogram", desc: "Route computation latency. Target: <500ms for city, <2s for cross-country. Track per distance bucket." },
              { metric: "route.eta_accuracy", type: "Gauge", desc: "Actual travel time vs predicted ETA. Accuracy > 90% within ±10%. Key quality metric." },
              { metric: "traffic.pipeline_lag", type: "Gauge", desc: "Kafka consumer lag in traffic pipeline. Growing lag = stale traffic data = bad ETAs." },
              { metric: "location.delivery_p99", type: "Histogram", desc: "Live location update end-to-end latency. Target: <2s. Track per WebSocket connection." },
              { metric: "cdn.cache_hit_rate", type: "Gauge", desc: "CDN cache hit ratio for tiles. Target: >99%. Drop below 98% = alert (eviction issue or update storm)." },
              { metric: "collab.conflict_rate", type: "Counter", desc: "Rate of optimistic concurrency conflicts on collaborative lists. High rate = UX problem." },
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
              { name: "Route Computation Timeout", rule: "route.latency_p99 > 5s for 2min", sev: "P1", action: "Routing server overloaded? Graph corruption? Memory pressure? Check: CPU utilization, GC pauses, graph version mismatch." },
              { name: "Traffic Pipeline Stale", rule: "traffic.pipeline_lag > 5min", sev: "P1", action: "Kafka consumer down? Flink job crashed? GPS ingestion broken? ETAs will be wrong. Fall back to historical patterns." },
              { name: "CDN Cache Hit Drop", rule: "cdn.cache_hit_rate < 98% for 10min", sev: "P2", action: "Mass tile invalidation? New map data version deployed? CDN eviction policy changed? Origin servers may be overwhelmed." },
              { name: "Location Sharing Latency", rule: "location.delivery_p99 > 5s for 3min", sev: "P2", action: "Redis Pub/Sub overloaded? WebSocket servers saturated? Network between location hub and WS servers?" },
            ].map((a,i) => (
              <div key={i} className="border border-stone-100 rounded-lg p-2.5">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="text-[11px] font-bold text-stone-700">{a.name}</span>
                  <Pill bg={a.sev==="P0"?"#fef2f2":a.sev==="P1"?"#fef2f2":"#fefce8"} color={a.sev==="P0"||a.sev==="P1"?"#dc2626":"#d97706"}>{a.sev}</Pill>
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
              { q: "User reports 'wrong ETA — said 20 min, took 45'", steps: "Check: was traffic data stale? (pipeline lag). Was the route through a known incident zone? Compare ETA at departure vs actual segment speeds. Check if route was recalculated mid-trip." },
              { q: "Map tiles not loading in a specific area", steps: "Check: CDN health in that region. Tile pre-rendering pipeline for that area. Was map data recently updated for that region? Try direct-to-origin request to bypass CDN." },
              { q: "Live location sharing stopped updating", steps: "Check: WebSocket connection status for both users. Redis Pub/Sub health. Session registry — is the session expired? GPS permissions on sender's device?" },
              { q: "Collaborative list shows stale data", steps: "Check: version mismatch? Concurrent edits creating conflicts? Replication lag on read replica? Client-side cache not invalidated?" },
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
          { title: "Stale Traffic → Wrong ETAs → User Distrust", sev: "Critical", sevColor: "#dc2626",
            desc: "Traffic pipeline lag causes routing engine to use stale speeds. Routes through heavy congestion are reported as 'clear'. Users get 30-minute ETAs that take 60 minutes. Erodes trust in the product.",
            fix: "Monitor pipeline lag with hard SLA (max 2 min stale). If lag exceeds threshold: fall back to historical patterns for that day/time (better than stale live data). Tag routes with 'traffic data freshness' — client shows warning if data is >5 min old.",
            code: `Traffic pipeline lag scenario:\n  T=0: Accident on highway, speed drops to 5 mph\n  T=0: Pipeline lag = 3 minutes (Kafka backlog)\n  T=0: User requests route through accident zone\n  T=0: Routing uses 3-min-old data: highway = 65 mph\n  T=0: ETA says 15 min. Actual: 55 min.\n\nSolution: Freshness-aware routing\n  if traffic_age(segment) > 2 min:\n      use historical_speed(segment, day, hour)\n  Route response includes:\n      "traffic_confidence": "low"  # client shows warning` },
          { title: "Contraction Hierarchy Rebuild During Peak", sev: "Critical", sevColor: "#dc2626",
            desc: "CH rebuild triggered during rush hour (road closure detected). Rebuild takes 2-4 hours. During rebuild, routing uses stale graph that doesn't know about the closure. Users routed through closed roads.",
            fix: "Never do full CH rebuild during peak. Instead: maintain a small 'patch graph' for recent road changes. Route = CH result + patch overlay. Full rebuild runs overnight. For closures: mark segments as blocked in traffic overlay (infinite weight) — no CH rebuild needed.",
            code: `Road closure handling:\n  Option A (bad): Rebuild entire CH\n    Time: 2-4 hours for continent\n    During rebuild: stale graph in production\n\n  Option B (good): Traffic overlay block\n    Set traffic_speed(closed_segment) = 0\n    Routing engine treats as impassable\n    No CH rebuild needed — immediate effect\n    Limitation: won't find shortcuts that\n    don't exist in original CH\n\n  Option C (best): CH + patch graph\n    Maintain small overlay graph for changes\n    Route = CH(main) + corrections(patch)\n    Full rebuild nightly to incorporate patches` },
          { title: "Hot Tile — Popular Area Overwhelms Origin", sev: "High", sevColor: "#d97706",
            desc: "Major event (concert, sports game) causes millions of users to view the same map area simultaneously. CDN cache works for existing tiles, but if tiles were just updated, all requests cascade to origin.",
            fix: "Stale-while-revalidate: CDN serves stale tile while fetching new one in background. Origin request coalescing: 1000 requests for same tile = 1 origin fetch. Pre-warm tiles for known events. Rate-limit origin requests per tile.",
            code: `Super Bowl in New Orleans:\n  10M users zoom into New Orleans simultaneously\n  CDN has tiles cached ✓\n  But: map data updated 1 hour ago for event roads\n  CDN TTL expired → all 10M requests hit origin\n\nSolution: Request coalescing at CDN\n  First request: fetches from origin\n  Next 999,999 requests: wait for first to complete\n  All 1M served from single origin fetch\n  CDN supports this natively (Cloudflare, Fastly)` },
          { title: "WebSocket Storm on Location Sharing", sev: "High", sevColor: "#d97706",
            desc: "Celebrity shares their live location publicly (like during a marathon). 500K users subscribe to their location feed. Each GPS update fans out to 500K WebSocket connections. Redis Pub/Sub and WS servers overwhelmed.",
            fix: "Tiered fan-out for popular location shares: >1K subscribers → switch to polling model (client polls every 5s instead of WS push). CDN the location update as a cacheable endpoint (5s TTL). Rate-limit subscriptions to any single location session.",
            code: `Celebrity location sharing:\n  1 GPS update/sec → 500K WS pushes\n  = 500K messages/sec for one session\n  Redis Pub/Sub: single channel bottleneck\n\nSolution: Tiered delivery\n  < 100 subscribers: direct WS push (real-time)\n  100-1K: batched WS push every 2s\n  > 1K: switch to HTTP polling endpoint\n    GET /v1/location/sess_abc/latest\n    Response: cached at CDN for 3 seconds\n    500K polls × 1/3s = 167K QPS (CDN handles it)` },
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
            { name: "Tile Service", owns: "Render, store, and serve vector map tiles. Handle tile versioning and CDN invalidation.", tech: "Go/Rust + Object Storage (S3) + CDN", api: "GET /tiles/{z}/{x}/{y}.mvt", scale: "CDN-dominated — origin servers handle <1% of traffic", stateful: false,
              modules: ["Tile Renderer (map data → protobuf MVT format)", "Tile Cache Manager (version tracking, CDN invalidation)", "On-Demand Renderer (high-zoom tiles rendered on request)", "Map Data Ingester (consume OSM/proprietary updates, trigger re-render)", "Tile Metadata Service (version, bounds, last-modified)", "CDN Origin Handler (serve tiles, set cache headers)"] },
            { name: "Routing Service", owns: "Compute driving/walking/transit routes with real-time traffic. Return turn-by-turn instructions.", tech: "C++/Rust (OSRM/Valhalla) + in-memory graph", api: "GET /directions?origin=...&dest=...", scale: "Horizontal — partition graph by region, replicate per region", stateful: true,
              modules: ["Graph Loader (load CH-preprocessed graph into memory)", "CH Query Engine (bidirectional upward Dijkstra)", "Traffic Overlay (merge live speeds onto CH edge weights)", "ETA Calculator (distance / speed per segment, sum)", "Alternative Route Finder (penalty-based: penalize edges of route 1, re-query)", "Turn Instruction Generator (graph edges → human-readable maneuvers)"] },
            { name: "Collaboration Service", owns: "Live location sharing, shared lists, voting, real-time sync between collaborators.", tech: "Go/Node + Redis Pub/Sub + PostgreSQL", api: "WS /location-sharing, REST /lists, REST /reviews", scale: "Horizontal — shard by session_id / list_id", stateful: false,
              modules: ["Location Hub (WebSocket manager, GPS ingestion, fan-out to subscribers)", "Session Manager (create/expire sharing sessions, manage participants)", "List CRDT Engine (conflict-free replicated data type for concurrent edits)", "Vote Aggregator (upvote/downvote on list items, compute rankings)", "Notification Dispatcher (push notifications for list changes)", "Review Pipeline (spam detection, sentiment analysis, rating aggregation)"] },
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
        <Label color="#9333ea">System Architecture — Full Pipeline</Label>
        <svg viewBox="0 0 720 370" className="w-full">
          {/* Map Layer */}
          <rect x={5} y={5} width={350} height={110} rx={8} fill="#05966904" stroke="#05966920" strokeWidth={1} strokeDasharray="4,3"/>
          <text x={15} y={22} fill="#059669" fontSize="10" fontWeight="700" fontFamily="monospace">MAP RENDERING LAYER</text>

          {/* Routing Layer */}
          <rect x={365} y={5} width={350} height={110} rx={8} fill="#9333ea04" stroke="#9333ea20" strokeWidth={1} strokeDasharray="4,3"/>
          <text x={375} y={22} fill="#9333ea" fontSize="10" fontWeight="700" fontFamily="monospace">ROUTING + TRAFFIC LAYER</text>

          {/* Collaboration Layer */}
          <rect x={5} y={125} width={710} height={105} rx={8} fill="#c026d304" stroke="#c026d320" strokeWidth={1} strokeDasharray="4,3"/>
          <text x={15} y={142} fill="#c026d3" fontSize="10" fontWeight="700" fontFamily="monospace">COLLABORATION LAYER</text>

          {/* Data Layer */}
          <rect x={5} y={240} width={710} height={80} rx={8} fill="#d9770604" stroke="#d9770620" strokeWidth={1} strokeDasharray="4,3"/>
          <text x={15} y={257} fill="#d97706" fontSize="10" fontWeight="700" fontFamily="monospace">DATA STORES</text>

          {/* Map components */}
          <rect x={20} y={35} width={70} height={34} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={55} y={52} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Client</text>
          <text x={55} y={64} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">mobile/web</text>

          <rect x={120} y={35} width={60} height={34} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={150} y={52} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">CDN</text>
          <text x={150} y={64} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">tile cache</text>

          <rect x={210} y={35} width={75} height={34} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={247} y={52} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Tile Svc</text>
          <text x={247} y={64} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">render + serve</text>

          <rect x={305} y={35} width={40} height={34} rx={6} fill="#05966908" stroke="#05966950" strokeWidth={1} strokeDasharray="3,2"/>
          <text x={325} y={55} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">S3</text>

          {/* Map path arrows */}
          <defs><marker id="ah-arch" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          <line x1={85} y1={47} x2={117} y2={47} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>
          <line x1={178} y1={52} x2={210} y2={52} stroke="#94a3b8" strokeWidth={1.2} strokeDasharray="4,2" markerEnd="url(#ah-arch)"/>
          <line x1={282} y1={52} x2={305} y2={52} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>

          {/* Routing components */}
          <rect x={380} y={35} width={80} height={34} rx={6} fill="#6366f110" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={420} y={52} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">Gateway</text>

          <rect x={480} y={30} width={85} height={34} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={522} y={44} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Routing Svc</text>
          <text x={522} y={57} textAnchor="middle" fill="#9333ea80" fontSize="7" fontFamily="monospace">CH + traffic</text>

          <rect x={480} y={72} width={85} height={30} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={522} y={89} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">Traffic Agg</text>

          <rect x={590} y={35} width={70} height={34} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={625} y={49} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Kafka</text>
          <text x={625} y={61} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">GPS stream</text>

          <line x1={457} y1={52} x2={478} y2={47} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>
          <line x1={457} y1={52} x2={478} y2={82} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>
          <line x1={563} y1={47} x2={588} y2={47} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>
          <line x1={563} y1={87} x2={588} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-arch)"/>

          {/* Collaboration components */}
          <rect x={30} y={155} width={90} height={34} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={75} y={169} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">Location Hub</text>
          <text x={75} y={181} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">WS + pub/sub</text>

          <rect x={150} y={155} width={90} height={34} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={195} y={169} textAnchor="middle" fill="#0891b2" fontSize="9" fontWeight="600" fontFamily="monospace">List Service</text>
          <text x={195} y={181} textAnchor="middle" fill="#0891b280" fontSize="7" fontFamily="monospace">CRDT sync</text>

          <rect x={270} y={155} width={90} height={34} rx={6} fill="#b4530910" stroke="#b45309" strokeWidth={1.5}/>
          <text x={315} y={169} textAnchor="middle" fill="#b45309" fontSize="9" fontWeight="600" fontFamily="monospace">Review Svc</text>
          <text x={315} y={181} textAnchor="middle" fill="#b4530980" fontSize="7" fontFamily="monospace">spam + rank</text>

          <rect x={395} y={155} width={85} height={34} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={437} y={169} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Redis P/S</text>
          <text x={437} y={181} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">real-time</text>

          <rect x={510} y={155} width={85} height={34} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={552} y={169} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Search Svc</text>
          <text x={552} y={181} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">geocoding</text>

          <rect x={625} y={155} width={75} height={34} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
          <text x={662} y={169} textAnchor="middle" fill="#7c3aed" fontSize="9" fontWeight="600" fontFamily="monospace">Push Svc</text>
          <text x={662} y={181} textAnchor="middle" fill="#7c3aed80" fontSize="7" fontFamily="monospace">APNs/FCM</text>

          {/* Data stores */}
          <rect x={20} y={265} width={75} height={34} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={57} y={285} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Road Graph</text>

          <rect x={115} y={265} width={70} height={34} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={150} y={285} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Place DB</text>

          <rect x={205} y={265} width={75} height={34} rx={6} fill="#b4530910" stroke="#b45309" strokeWidth={1.5}/>
          <text x={242} y={285} textAnchor="middle" fill="#b45309" fontSize="8" fontWeight="600" fontFamily="monospace">Review DB</text>

          <rect x={300} y={265} width={80} height={34} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={340} y={285} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Traffic Redis</text>

          <rect x={400} y={265} width={75} height={34} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={437} y={285} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">List DB</text>

          <rect x={495} y={265} width={75} height={34} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={532} y={285} textAnchor="middle" fill="#78716c" fontSize="8" fontWeight="600" fontFamily="monospace">User Store</text>

          <rect x={590} y={265} width={85} height={34} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
          <text x={632} y={285} textAnchor="middle" fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Search Index</text>

          {/* Legend */}
          <rect x={100} y={310} width={520} height={52} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={110} y={327} fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Tile path: Client → CDN → Tile Svc → S3 (99% CDN-cached)</text>
          <text x={110} y={342} fill="#9333ea" fontSize="8" fontWeight="600" fontFamily="monospace">Route path: Client → Gateway → Routing Svc (in-memory CH + Traffic Redis)</text>
          <text x={110} y={357} fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Collab path: Client → WS → Location Hub → Redis Pub/Sub → friend's WS</text>
        </svg>
      </Card>
    </div>
  );
}

function FlowsSection() {
  const [flow, setFlow] = useState("route_query");
  const flows = {
    route_query: { title: "Route Computation — City Drive", steps: [
      { actor: "Client", action: "User taps 'Directions' from A (home) to B (office). Client sends: origin, destination, mode=driving, departure_time=now", type: "request" },
      { actor: "Client → Gateway", action: "HTTPS: GET /v1/directions?origin=37.77,-122.41&dest=37.79,-122.40&mode=driving", type: "request" },
      { actor: "Gateway", action: "Auth check (API key or user token). Rate limit. Route to nearest Routing Service instance for this region.", type: "auth" },
      { actor: "Gateway → Routing Service", action: "gRPC: ComputeRoute(origin, dest, mode, departure_time). Routing server has US West graph in memory.", type: "request" },
      { actor: "Routing Service", action: "Snap origin/dest to nearest road nodes (map matching). source_node=N1, dest_node=N2.", type: "process" },
      { actor: "Routing Service", action: "Run CH bidirectional query: forward search from N1 (upward), backward from N2 (upward). Meet at high-importance node.", type: "process" },
      { actor: "Routing Service → Traffic Redis", action: "For each edge in shortest path: GET traffic:{segment_id}. Overlay live speeds on static weights.", type: "request" },
      { actor: "Routing Service", action: "Compute ETA: sum(segment_length / traffic_speed) for each edge. Apply penalties for turns, traffic lights.", type: "process" },
      { actor: "Routing Service", action: "Find 2 alternative routes: penalize edges of route 1, re-run CH. Return top 3 ranked by ETA.", type: "process" },
      { actor: "Routing Service → Client", action: "Response: {routes: [primary + 2 alternatives], each with encoded_polyline, steps, ETA, distance}", type: "success" },
      { actor: "Client", action: "Render routes on map (decode polyline → lat/lng points → draw on vector tiles). Show ETA comparison.", type: "success" },
    ]},
    live_location: { title: "Live Location Sharing — Friends", steps: [
      { actor: "User A", action: "Taps 'Share live location' with User B for 60 minutes. Client creates sharing session.", type: "request" },
      { actor: "Client A → Collab Service", action: "POST /v1/location-sharing/sessions {shared_with: ['user_B'], duration: 60min}", type: "request" },
      { actor: "Collab Service", action: "Create session in DB: session_id, participants, expiry. Notify User B via push notification.", type: "process" },
      { actor: "Collab Service → Client A", action: "Response: {session_id: 'sess_abc', ws_url: 'wss://location-hub.example.com/ws?session=sess_abc'}", type: "success" },
      { actor: "Client A", action: "Opens WebSocket to Location Hub. Sends GPS updates every 3 seconds.", type: "request" },
      { actor: "Client B", action: "Receives push notification. Opens app. Connects WebSocket to same Location Hub for sess_abc.", type: "process" },
      { actor: "Client A → Location Hub", action: "WS: {type: 'location_update', lat: 37.7749, lng: -122.4194, speed: 45, heading: 90, ts: ...}", type: "request" },
      { actor: "Location Hub", action: "Validate session (not expired). Publish to Redis: PUBLISH location:sess_abc {user_A's location}", type: "process" },
      { actor: "Location Hub → Client B", action: "Redis subscription triggers WS push to User B: {user_id: 'A', lat: 37.7749, lng: -122.4194, ts: ...}", type: "success" },
      { actor: "Client B", action: "Renders User A's live position on map. Updates marker every 3 seconds. Shows ETA to User A's location.", type: "success" },
      { actor: "Location Hub", action: "After 60 minutes: close session, disconnect WebSockets, clean up Redis subscription.", type: "process" },
    ]},
    collab_list: { title: "Collaborative Trip Planning", steps: [
      { actor: "User A", action: "Creates 'Tokyo Trip 2025' list and invites User B and User C as collaborators.", type: "request" },
      { actor: "Client A → List Service", action: "POST /v1/lists {name: 'Tokyo Trip', collaborators: ['B', 'C'], type: 'trip_plan'}", type: "request" },
      { actor: "List Service", action: "Create list in DB (version=0). Send push notifications to B and C.", type: "process" },
      { actor: "User A", action: "Adds 'Tsukiji Fish Market' to the list with a note 'Must go early morning!'", type: "request" },
      { actor: "Client A → List Service", action: "PUT /v1/lists/list_123/items {place_id: 'tsukiji', note: 'Must go early!', version: 0}", type: "request" },
      { actor: "List Service", action: "Validate version=0 matches DB. Insert item. Increment version to 1. Emit event on Kafka.", type: "process" },
      { actor: "User B (concurrent)", action: "Simultaneously adds 'Senso-ji Temple' with version: 0 (stale!)", type: "request" },
      { actor: "List Service", action: "Version mismatch: expected 0, actual 1. Return 409 Conflict with latest version.", type: "check" },
      { actor: "Client B", action: "Receives 409. Fetches latest list (version=1). Retries with version: 1. Success.", type: "process" },
      { actor: "Push Service", action: "All collaborators receive real-time update via push/WebSocket: 'B added Senso-ji Temple'", type: "success" },
    ]},
    tile_load: { title: "Map Tile Loading — Pan & Zoom", steps: [
      { actor: "Client", action: "User opens app. Map viewport calculated: center lat/lng, zoom level 14. Need tiles: z=14, x=2623-2625, y=6332-6334 (9 tiles).", type: "process" },
      { actor: "Client → CDN", action: "9 parallel requests: GET /tiles/14/2623/6332.mvt, GET /tiles/14/2624/6332.mvt, ...", type: "request" },
      { actor: "CDN Edge PoP", action: "8 of 9 tiles are in edge cache (cache hit). 1 tile was recently updated (cache miss).", type: "process" },
      { actor: "CDN → Tile Service", action: "Cache miss: forward request to Tile Service origin for tile 14/2625/6334.mvt", type: "request" },
      { actor: "Tile Service", action: "Check local tile cache (Redis). Miss → read from S3: tiles/v42/14/2625/6334.mvt", type: "process" },
      { actor: "Tile Service → CDN → Client", action: "Return tile with Cache-Control: public, max-age=86400. CDN caches for 24h.", type: "success" },
      { actor: "Client (GPU)", action: "All 9 vector tiles received. GPU renders: roads, buildings, labels, POI icons in <50ms.", type: "success" },
      { actor: "User pans east", action: "Need 3 new tiles on the right edge. Repeat: check CDN cache, fetch misses. Smooth panning.", type: "process" },
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
          <Label color="#b45309">K8s Deployment — Routing Service (Stateful!)</Label>
          <CodeBlock title="Routing Servers — StatefulSet with large memory" code={`apiVersion: apps/v1
kind: StatefulSet            # Stateful — in-memory graph
metadata:
  name: routing-service
spec:
  replicas: 50               # 10 per region × 5 regions
  podManagementPolicy: Parallel
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1      # Graph load takes ~5 min
  template:
    spec:
      terminationGracePeriodSeconds: 30
      containers:
      - name: routing
        image: maps/routing:v7.2
        ports:
        - containerPort: 8080  # gRPC
        resources:
          requests:
            memory: "256Gi"   # Full CH graph in memory
            cpu: "32"         # CPU-intensive route computation
          limits:
            memory: "280Gi"
        env:
        - name: GRAPH_REGION
          value: "us-west"
        - name: GRAPH_VERSION
          value: "v2024.02.09"
        - name: TRAFFIC_REDIS_URL
          value: "redis-cluster:6379"
        readinessProbe:
          exec:
            command: ["/health", "--graph-loaded"]
          initialDelaySeconds: 300  # 5 min to load graph
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080`} />
          <div className="mt-3 space-y-1.5">
            <Point icon="⚠" color="#b45309">readinessProbe waits for graph to be fully loaded in memory (~5 min startup)</Point>
            <Point icon="⚠" color="#b45309">256 GB RAM per pod — requires dedicated high-memory node pool (r6i.8xlarge)</Point>
            <Point icon="⚠" color="#b45309">Rolling update: one at a time. Don't drain too fast — in-flight route queries must complete.</Point>
          </div>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Security — Location Privacy + API</Label>
          <div className="space-y-3">
            {[
              { layer: "Location Data Privacy", details: ["GPS telemetry anonymized before aggregation (no user ID attached to traffic data)", "Location sharing encrypted in transit (TLS 1.3) and at rest", "Sharing sessions auto-expire — no indefinite tracking", "User can revoke sharing instantly — server closes WS and deletes session", "GDPR: delete all location history on account deletion within 30 days"] },
              { layer: "API Security", details: ["API key required for all tile and directions requests", "Rate limiting: 10K tiles/min, 100 routes/min per API key", "OAuth 2.0 for user-facing APIs (lists, reviews, sharing)", "CORS policy: restrict tile access to authorized domains", "DDoS protection: CDN-level rate limiting and WAF"] },
              { layer: "Data Integrity", details: ["Review spam detection: ML model flags suspicious reviews (bulk posting, paid reviews)", "Map data validation: automated + human review for road edits", "Traffic data outlier detection: filter GPS pings from planes, GPS drift, spoofed locations", "Collaborative list access control: owner manages permissions (view/edit/admin)"] },
              { layer: "Audit & Compliance", details: ["Access logs for all location data queries", "SOC 2 Type II compliance for enterprise customers", "Data residency: configurable per region (EU data stays in EU)", "Encryption at rest: AES-256 for all user data stores"] },
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
                { b: "CDN cache miss storm", s: "Origin tile servers overwhelmed, p99 >2s", f: "Stale-while-revalidate, request coalescing, pre-warm event areas", p: "Don't invalidate all tiles at once on map update. Stagger by region." },
                { b: "Routing server memory", s: "OOM kills on graph load, pods restarting", f: "Right-size pods (256 GB+), dedicated node pool, monitor RSS", p: "CH preprocessing can temporarily need 3× memory. Don't run in-place." },
                { b: "Traffic pipeline lag", s: "ETAs increasingly wrong during rush hour", f: "Scale Kafka partitions + Flink parallelism, auto-scale on lag metric", p: "Don't auto-scale routing servers — they won't help if traffic data is stale." },
                { b: "Redis Pub/Sub fan-out", s: "Location sharing latency >5s for popular sessions", f: "Tier fan-out: >1K subscribers → polling model. Dedicated Redis for pub/sub.", p: "Don't put location pub/sub on same Redis as traffic data cache." },
                { b: "Search index lag", s: "New businesses not appearing in search for hours", f: "Near-real-time indexing pipeline (Kafka → Elasticsearch). Incremental updates.", p: "Full reindex is expensive. Use incremental updates + nightly full rebuild." },
                { b: "Review write spikes", s: "Review submission latency >2s after major events", f: "Write buffer (Kafka queue), async processing. Show optimistic UI.", p: "Don't block review submission on spam detection — do it async." },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-mono text-rose-700">{r.b}</td>
                  <td className="px-3 py-2 text-stone-500">{r.s}</td>
                  <td className="px-3 py-2 text-stone-500">{r.f}</td>
                  <td className="px-3 py-2 text-stone-400">{r.p}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card accent="#d97706">
        <Label color="#d97706">Production War Stories</Label>
        <div className="grid grid-cols-2 gap-4">
          {[
            { title: "Graph version mismatch across routing servers", symptom: "Some routes were 10 min longer than others for the same query. Users getting inconsistent ETAs on refresh.", cause: "Rolling graph update: half the servers had the new graph (with new highway), half had the old graph. Load balancer sent requests to both.",
              fix: "Canary graph deployment: update 1 server, verify routes match baseline, then roll to all. Blue-green deployment for graph versions — atomic switchover.",
              quote: "User tweeted: 'Google Maps keeps changing my ETA by 10 minutes every time I refresh.' Went viral. Root cause: graph version mismatch across pods." },
            { title: "CDN invalidation cascade took down origin", symptom: "Tile load times went from 50ms to 30s globally. Map was blank for 15 minutes during peak hours.", cause: "Map data team shipped a global road network update. CDN invalidation for ALL tiles at once. 10B requests cascaded to origin in 2 minutes.",
              fix: "Never invalidate all tiles simultaneously. Stagger by region (US first, then EU, then APAC) over 4 hours. Pre-warm popular tiles before invalidation. Request coalescing at CDN.",
              quote: "Postmortem: 'We invalidated 4.2 trillion tile cache entries in one API call. The CDN provider called us.'" },
            { title: "Live location sharing drained phone batteries", symptom: "1-star reviews: 'This app killed my battery in 2 hours.' Support tickets spiking.", cause: "Location sharing defaulted to GPS update every 1 second with high-accuracy mode. Fine for driving, terrible for walking (which was 60% of use).",
              fix: "Adaptive update frequency: driving → 1s interval (GPS). Walking → 5s (network-based). Stationary → 30s. Client-side activity detection adjusts automatically.",
              quote: "Battery team found our app was responsible for 18% of system battery usage during a sharing session. We shipped the fix in 48 hours." },
            { title: "CRDT merge conflict in collaborative list", symptom: "User A added item at position 3. User B added different item at position 3. Both saw different orderings.", cause: "Concurrent inserts at the same position in a CRDT-based list. The merge algorithm produced different results depending on the order of operations received.",
              fix: "Switch to Fractional Indexing for position ordering. Each item gets a fractional position (e.g., 1.0, 1.5, 2.0). Concurrent inserts at 'between 1 and 2' get 1.25 and 1.75 — deterministic ordering regardless of arrival order.",
              quote: "Two friends planned a trip. One saw 'Ramen → Sushi → Temple.' The other saw 'Ramen → Temple → Sushi.' Neither could figure out why." },
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
        { t: "AR Navigation (Live View)", d: "Camera-based AR overlay showing turn arrows and street names on live camera feed. Replaces confusing 2D instructions.", detail: "On-device ML: detect road, sidewalk, buildings. Overlay 3D turn arrows. Requires ARCore/ARKit + real-time pose estimation. Battery-intensive.", effort: "Hard" },
        { t: "EV Routing (Battery-Aware)", d: "Route planning that considers EV battery level, charging station locations, and charging time.", detail: "Edge weights include elevation (uphill = more battery). Insert charging stops optimally. Minimize total travel time (drive + charge). Tesla does this well.", effort: "Hard" },
        { t: "Predictive ETA (ML)", d: "Use historical patterns + current traffic to predict future conditions along the route. 'Leave at 5pm → this route will be 20 min faster.'", detail: "Time-series model trained on historical GPS data per road segment per time-of-day per day-of-week. Integrate weather, events, holidays.", effort: "Hard" },
        { t: "Offline Maps + Routing", d: "Download map region for offline use. Route computation on-device without internet.", detail: "Download vector tiles + simplified road graph for a region. On-device A* routing (can't use server-side CH). Limited to pre-downloaded area.", effort: "Medium" },
        { t: "Indoor Maps (Malls, Airports)", d: "Floor-by-floor navigation inside large buildings. 'Navigate to Gate B23' in an airport.", detail: "Separate indoor map data (floor plans). WiFi/BLE positioning instead of GPS (no satellite indoors). Floor transitions (escalators, elevators).", effort: "Medium" },
        { t: "Social Reviews Feed", d: "See friends' reviews and photos for nearby places. 'Your friend Alice rated this 5 stars.'", detail: "Merge social graph with place data. Privacy-aware: only show reviews from mutual connections. Rank friend reviews higher.", effort: "Easy" },
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
    { q:"Why contraction hierarchies and not just Dijkstra or A*?", a:"Dijkstra visits every node reachable within the shortest path distance — for a continental route, that's millions of nodes (seconds of compute). A* with a good heuristic reduces this to thousands, but still ~100ms. Contraction Hierarchies pre-process the graph by 'contracting' unimportant nodes and adding shortcut edges. Query time drops to <1ms because the bidirectional search only visits ~1000 nodes by traversing shortcuts through the hierarchy. The trade-off: pre-processing takes hours, and the graph is static.", tags:["algorithm"] },
    { q:"How do you handle real-time traffic with pre-computed CH?", a:"You don't rebuild CH when traffic changes. Instead, use a traffic overlay: each CH edge has a base weight (distance/speed_limit) and a traffic multiplier (current_speed/speed_limit). On route query: effective_weight = base_weight × traffic_multiplier. The multiplier is fetched from Redis (updated every 30-60s by the traffic pipeline). This preserves CH's fast query time while incorporating live traffic.", tags:["design"] },
    { q:"Vector tiles vs raster tiles — why does it matter?", a:"Raster tiles are pre-rendered PNG images. Fixed style, fixed resolution. At 20 zoom levels, covering the whole Earth = ~50 TB of images. Vector tiles store geometry (roads, buildings, labels) as protobuf data (~2 TB total). Client GPU renders them. Benefits: 10× smaller, support rotation/tilt, dynamic styling (dark mode), smooth zoom between levels, and offline capability. All modern map apps (Google, Apple, Mapbox) use vector tiles.", tags:["design"] },
    { q:"How do you handle map-matching for GPS data?", a:"GPS accuracy is ±5-15m. A car on a highway might appear to be on a parallel side street. Map-matching: take the raw GPS trace and snap each point to the most likely road segment. Use Hidden Markov Model: states = road segments, transitions = road network connectivity, emissions = GPS distance to road. Viterbi algorithm finds the most likely sequence of road segments. Critical for accurate traffic data.", tags:["algorithm"] },
    { q:"How does live location sharing scale to millions of sessions?", a:"Each session has 2-8 participants. Updates are 1 GPS ping every 3 seconds. Use Redis Pub/Sub: one channel per session. Location Hub subscribes to channels for connected users. Total: ~17M updates/sec across 50M sessions. Shard Redis by session_id. For popular sessions (>1K subscribers): switch to CDN-cached polling endpoint to avoid pub/sub fan-out bottleneck.", tags:["scalability"] },
    { q:"How do you compute alternative routes?", a:"Three approaches: (1) Penalty method: compute route 1, then penalize its edges (3× weight), re-run CH. Route 2 avoids route 1's roads. Repeat for route 3. (2) Plateau method: during CH query, detect edges where forward and backward searches 'plateau' — these are decision points. Branch at plateaus. (3) Via-node method: force the route through a different intermediate node. Google uses a combination for fast alternatives.", tags:["algorithm"] },
    { q:"How do collaborative lists handle concurrent edits?", a:"Two options: (1) Optimistic Concurrency Control (OCC): each edit includes a version number. Server rejects stale versions (409 Conflict). Client retries. Simple, works for low-contention scenarios (most trip planning). (2) CRDTs (Conflict-free Replicated Data Types): for real-time co-editing. Each operation (add/remove/reorder) is designed to be commutative — order of operations doesn't matter. No conflicts by design. More complex to implement but better UX.", tags:["design"] },
    { q:"Why geohash for spatial queries instead of R-tree?", a:"Geohash is a 1D encoding of 2D coordinates that preserves locality. Key advantage: works with standard B-tree indexes in any database. 'Find places near me' = prefix scan on geohash column. R-trees are theoretically better for spatial queries but require specialized index support (PostGIS, spatial extensions). Geohash is good enough for most use cases and works everywhere — including Redis (GEOADD/GEOSEARCH uses geohash internally).", tags:["data"] },
    { q:"How would you handle cross-country routes spanning multiple graph partitions?", a:"Partition the road graph by region (e.g., US West, US East, Europe). Each partition is loaded on separate routing servers. For cross-partition routes: identify border nodes (highway crossings between regions). Pre-compute distances between all border node pairs. Route = local CH (origin to border) + pre-computed border-to-border + local CH (border to destination). This is called 'transit node routing' — an extension of CH.", tags:["scalability"] },
    { q:"How do you prevent review spam and fake ratings?", a:"Multi-layer defense: (1) Rate limiting — max 5 reviews/day per user. (2) ML classifier — trained on known spam patterns (bulk posting, identical text, suspicious timing). (3) Behavioral signals — user has no other activity, account age <7 days, reviews only one business category. (4) Post-hoc analysis — detect review rings (groups of accounts reviewing same places). (5) Human review — ML-flagged reviews queued for manual inspection. Google combines all of these.", tags:["security"] },
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

export default function GoogleMapsSD() {
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
      <div className="sticky top-0 z-50 border-b border-stone-200" style={{ background: "rgba(250,249,247,0.92)", backdropFilter: "blur(12px)" }}>
        <div className="max-w-7xl mx-auto px-5 py-3">
          <div className="flex items-center gap-3 mb-2.5">
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Google Maps (Collaborative)</h1>
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