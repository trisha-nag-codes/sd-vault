import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   INSTAGRAM — System Design Reference
   Pearl white theme · 17 sections (HLD + LLD)
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",             icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",         icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",  icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",           icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",    icon: "🏗️", color: "#9333ea" },
  { id: "algorithm",     label: "Feed & Fan-Out Deep Dive", icon: "⚙️", color: "#c026d3" },
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
          <div key={i} className={`px-2 rounded ${line.trim().startsWith("#") || line.trim().startsWith("//") ? "text-stone-400" : "text-stone-700"}`}>
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
            <Label>What is Instagram?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A photo and video sharing social network where users post content, follow other users, and browse a personalised feed. Think Instagram, Pinterest, or Flickr. The core challenges: ingesting and serving billions of images at scale, generating a ranked feed from millions of followed accounts, and supporting ephemeral content (Stories) alongside permanent posts.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Instagram is a read-heavy system: the feed-read to post-write ratio is roughly 100:1. The media pipeline (upload, process, store, serve via CDN) is just as critical as the social graph and feed-generation logic. Every photo is resized into 4+ variants, served from the nearest CDN edge, and the feed must render in under 500ms.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard?</Label>
            <ul className="space-y-2.5">
              <Point icon="📸" color="#0891b2">Media pipeline — every uploaded photo is resized into multiple resolutions (thumbnail, small, medium, original), stripped of EXIF data, and pushed to CDN. At 100M+ photos/day this is a massive data pipeline.</Point>
              <Point icon="📰" color="#0891b2">Feed generation — a user follows 500 accounts. Building their feed means fetching, ranking, and merging posts from all 500. At 500M DAU this is trillions of candidate evaluations per day.</Point>
              <Point icon="👥" color="#0891b2">Celebrity problem — a user with 500M followers posts a photo. Fan-out-on-write would create 500M feed writes instantly. Must use fan-out-on-read for celebrities (hybrid approach).</Point>
              <Point icon="⏳" color="#0891b2">Stories — ephemeral content that disappears after 24 hours. Separate storage, separate feed, separate delivery path. 500M+ daily Story viewers.</Point>
              <Point icon="🔍" color="#0891b2">Explore / discovery — recommend content from accounts a user doesn't follow. Requires ML ranking, embedding similarity, and real-time signals (trending, engagement velocity).</Point>
              <Point icon="🌐" color="#0891b2">Global scale — 2B+ monthly users across every continent. Content must be served from the nearest CDN edge with sub-200ms latency regardless of geography.</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Scale</Label>
            <div className="space-y-2.5">
              {[
                { co: "Instagram", scale: "2B+ MAU, 500M+ DAU", detail: "Meta. 100M+ photos/day" },
                { co: "Pinterest", scale: "450M+ MAU, visual search", detail: "Pin-based, recommendation-heavy" },
                { co: "Snapchat", scale: "750M+ MAU, ephemeral-first", detail: "Stories originated here" },
                { co: "TikTok", scale: "1.5B+ MAU, video-first", detail: "Algo-driven, not follow-based" },
                { co: "Flickr", scale: "100M+ users, photo archive", detail: "Pro photography, high-res" },
              ].map((e,i) => (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-24 shrink-0">{e.co}</span>
                  <span className="text-stone-500 flex-1">{e.scale}</span>
                  <span className="text-stone-400 text-[10px]">{e.detail}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <Label color="#2563eb">The Core Architecture Decision</Label>
            <svg viewBox="0 0 360 135" className="w-full">
              <rect x={10} y={10} width={160} height={45} rx={6} fill="#05966908" stroke="#059669" strokeWidth={1.5}/>
              <text x={90} y={28} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="700" fontFamily="monospace">Fan-Out on Write</text>
              <text x={90} y={44} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">✔ Pre-compute feed, fast reads</text>

              <rect x={190} y={10} width={160} height={45} rx={6} fill="#2563eb08" stroke="#2563eb" strokeWidth={1.5}/>
              <text x={270} y={28} textAnchor="middle" fill="#2563eb" fontSize="10" fontWeight="700" fontFamily="monospace">Fan-Out on Read</text>
              <text x={270} y={44} textAnchor="middle" fill="#2563eb80" fontSize="8" fontFamily="monospace">✔ No write amplification</text>

              <rect x={40} y={68} width={280} height={55} rx={6} fill="#9333ea08" stroke="#9333ea" strokeWidth={1.5}/>
              <text x={180} y={87} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Hybrid ★ (Instagram's actual approach)</text>
              <text x={180} y={101} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">Fan-out-on-write for normal users (&lt;10K followers)</text>
              <text x={180} y={115} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">Fan-out-on-read for celebrities (&gt;10K followers)</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">The #1 most-asked system design question at top companies</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Scope to the Core Loop</div>
            <p className="text-[12px] text-stone-500 mt-0.5">"Design Instagram" is enormous — feed, stories, reels, DMs, explore, ads, shopping, live video. Clarify scope immediately. For a 45-min interview, focus on <strong>photo upload + feed generation (fan-out) + media serving via CDN</strong>. Stories, Explore, Reels, and DMs are follow-ups. The core loop is: upload → fan-out → feed read.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Upload photos/videos — with caption, location tag, and user tags</Point>
            <Point icon="2." color="#059669">Follow / unfollow users — asymmetric social graph (follow ≠ friend)</Point>
            <Point icon="3." color="#059669">News feed — personalised, ranked feed of posts from followed accounts</Point>
            <Point icon="4." color="#059669">Like & comment on posts — real-time counters and comment threads</Point>
            <Point icon="5." color="#059669">Stories — ephemeral posts that disappear after 24 hours, shown at top of feed</Point>
            <Point icon="6." color="#059669">User profile — grid view of user's posts, follower/following counts, bio</Point>
            <Point icon="7." color="#059669">Search — find users, hashtags, and locations</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Feed load latency: p99 &lt;500ms (p50 &lt;200ms)</Point>
            <Point icon="2." color="#dc2626">Photo upload to visible in followers' feeds: &lt;5 seconds</Point>
            <Point icon="3." color="#dc2626">Scale to 2B MAU, 500M DAU, 100M+ uploads/day</Point>
            <Point icon="4." color="#dc2626">High availability — 99.99% uptime (feed is the core product)</Point>
            <Point icon="5." color="#dc2626">Media delivery: sub-100ms from CDN edge worldwide</Point>
            <Point icon="6." color="#dc2626">Eventual consistency acceptable for feeds (few-second delay OK), strong consistency for likes/follows</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Photos only, or photos + videos + Reels?",
            "Do we need Stories (ephemeral 24h posts)?",
            "Feed ranking (ML) or chronological?",
            "DMs / messaging? (usually separate system)",
            "Explore / recommendation page?",
            "What scale? (DAU, uploads/day, avg followers)",
            "Photo/video size limits? (max resolution?)",
            "Ads in feed? (monetization changes architecture)",
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
          <Label color="#7c3aed">Step 1 — Traffic Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="DAU = 500M users" result="500M" note="Instagram-scale. ~2B MAU, ~25% daily." />
            <MathStep step="2" formula="Photos uploaded/day = 100M" result="100M" note="~20% of DAU post. Some post multiple per day." />
            <MathStep step="3" formula="Feed reads/day = 500M × 10 opens" result="5B" note="Average user opens app 10×/day, each loads feed." />
            <MathStep step="4" formula="Feed reads/sec = 5B / 86,400" result="~58K req/s" note="Average. Peak is 3-5× during evenings." final />
            <MathStep step="5" formula="Photo uploads/sec = 100M / 86,400" result="~1,160/s" note="Write rate. Read:Write ratio ≈ 50:1." final />
            <MathStep step="6" formula="Likes/day = 500M DAU × 10 likes" result="5B likes/day" note="~58K likes/sec. High-volume counter updates." final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Storage Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Average photo size (original) = 2 MB" result="2 MB" note="After JPEG compression. Raw uploads may be larger." />
            <MathStep step="2" formula="Resized variants per photo = 4" result="4" note="Thumbnail 150px, small 320px, medium 640px, original." />
            <MathStep step="3" formula="Total storage per photo = ~3.5 MB" result="3.5 MB" note="Original (2MB) + 3 resized variants (~1.5MB total)." />
            <MathStep step="4" formula="Daily photo storage = 100M × 3.5MB" result="~350 TB/day" note="Photos dominate storage. This is the #1 cost driver." final />
            <MathStep step="5" formula="Yearly photo storage = 350TB × 365" result="~128 PB/year" note="Petabytes. Object storage (S3) is the only option." final />
            <MathStep step="6" formula="Metadata per post = 500 bytes" result="500 B" note="Caption, user_id, timestamp, location, tags. Tiny vs media." />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Step 3 — Feed & Fan-Out</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Avg followers per user = 200" result="200" note="Median. Power law: most have <100, celebrities have millions." />
            <MathStep step="2" formula="Fan-out writes per post (normal user)" result="200" note="Post → write to 200 followers' feed caches." />
            <MathStep step="3" formula="Total fan-out writes/sec = 1,160 × 200" result="~230K/s" note="For normal-user posts. Celebrity posts excluded (read path)." final />
            <MathStep step="4" formula="Celebrity post (1M followers)" result="0 fan-out" note="Not pre-computed. Merged at read time (fan-out-on-read)." />
            <MathStep step="5" formula="Feed cache size per user = 500 post IDs" result="~4 KB" note="500 post IDs × 8 bytes = 4KB. Fits in Redis." />
            <MathStep step="6" formula="Total feed cache = 500M × 4KB" result="~2 TB" note="All users' feed caches. Fits in a Redis cluster." final />
          </div>
        </Card>
        <Card>
          <Label color="#dc2626">Key Numbers to Remember</Label>
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "Feed Reads/s", val: "~58K", sub: "Avg (peak: 200K+)" },
              { label: "Photo Uploads/s", val: "~1,160", sub: "100M/day" },
              { label: "Daily Photo Storage", val: "350 TB", sub: "All resolutions" },
              { label: "Fan-Out Writes/s", val: "~230K", sub: "Normal users only" },
              { label: "Feed Cache (total)", val: "~2 TB", sub: "500M users × 4KB" },
              { label: "Read:Write Ratio", val: "~50:1", sub: "Extremely read-heavy" },
            ].map((s,i) => (
              <div key={i} className="text-center py-2.5 rounded-lg bg-stone-50 border border-stone-200">
                <div className="text-[16px] font-bold text-violet-700 font-mono">{s.val}</div>
                <div className="text-[10px] font-medium text-stone-600 mt-0.5">{s.label}</div>
                <div className="text-[9px] text-stone-400">{s.sub}</div>
              </div>
            ))}
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
          <Label color="#2563eb">Core Feed & Post APIs</Label>
          <CodeBlock code={`# GET /v1/feed?cursor=&limit=20
# Load personalised feed (ranked)
# Returns: {posts: [{post_id, author, media_urls,
#           caption, like_count, comment_count,
#           liked_by_viewer, created_at}],
#           next_cursor, has_more}
# Cursor = last post's ranking score (not timestamp)

# POST /v1/posts
# Create a new post
{
  "media_ids": ["media_abc"],     # Pre-uploaded via media API
  "caption": "Sunset in Bali 🌅",
  "location": {"lat": -8.65, "lng": 115.21,
               "name": "Uluwatu, Bali"},
  "tagged_users": ["user_789"],
  "hashtags": ["sunset", "bali", "travel"]
}
# Returns: {post_id, media_urls, created_at}

# POST /v1/media/upload
# Upload photo/video (pre-signed URL or direct)
# Content-Type: multipart/form-data
# Returns: {media_id, upload_status: "processing",
#           processing_eta_sec: 3}

# GET /v1/posts/:id
# Single post detail
# Returns: {post, comments: [...], likes_preview: [...]}

# POST /v1/posts/:id/like
# Like a post (idempotent)
# Returns: {liked: true, like_count: 4521}

# POST /v1/posts/:id/comments
# Add a comment
{ "text": "Amazing view! 😍", "reply_to": null }
# Returns: {comment_id, created_at}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Social Graph & Profile APIs</Label>
          <CodeBlock code={`# POST /v1/users/:id/follow
# Follow a user (asymmetric)
# Returns: {following: true, status: "following"}
# Status: following | requested (if private account)

# DELETE /v1/users/:id/follow
# Unfollow a user
# Returns: {following: false}

# GET /v1/users/:id/followers?cursor=&limit=50
# List followers (paginated)
# Returns: {users: [{user_id, username, avatar_url,
#           is_following_you}], next_cursor}

# GET /v1/users/:id/following?cursor=&limit=50
# List accounts this user follows

# GET /v1/users/:id
# User profile
# Returns: {user_id, username, bio, avatar_url,
#           post_count, follower_count, following_count,
#           is_private, posts: [...]}

# GET /v1/stories
# Load stories tray (followed users with active stories)
# Returns: {trays: [{user, stories: [{story_id,
#           media_url, created_at, expires_at, seen}]}]}

# POST /v1/stories
# Post a story (24h ephemeral)
{ "media_id": "media_xyz", "stickers": [...] }
# Returns: {story_id, expires_at}

# GET /v1/explore?cursor=
# Explore / discover page (recommended content)
# Returns: {posts: [...], topics: [...]}`} />
          <div className="mt-3 space-y-2">
            {[
              { q: "Why pre-upload media before creating the post?", a: "Photo processing (resize, compress, CDN distribution) takes 2-5 seconds. Uploading first lets this happen in the background. When the user hits 'Share', the media is already processed. The post creation is instant." },
              { q: "Why cursor-based pagination, not offset?", a: "Offset pagination breaks when new posts are inserted (skipped/duplicate items). Cursor (last post's ranking score or timestamp) is stable even as the feed changes. Essential for infinite scroll." },
              { q: "Why is the like endpoint idempotent?", a: "Users can double-tap rapidly. Network retries can duplicate requests. Idempotent likes (SET semantics, not INCREMENT) prevent inflated counts. DB: UPSERT into likes(user_id, post_id)." },
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
        <Label color="#9333ea">Full System Architecture</Label>
        <svg viewBox="0 0 720 350" className="w-full">
          {/* Clients */}
          <rect x={10} y={110} width={65} height={34} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={42} y={130} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Mobile</text>
          <rect x={10} y={160} width={65} height={34} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={42} y={180} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Web</text>

          {/* CDN */}
          <rect x={95} y={55} width={60} height={34} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={125} y={76} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">CDN</text>

          {/* API Gateway / LB */}
          <rect x={95} y={130} width={65} height={46} rx={6} fill="#6366f110" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={127} y={150} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">API</text>
          <text x={127} y={162} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">Gateway</text>

          {/* Services */}
          <rect x={195} y={40} width={85} height={30} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={237} y={59} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Post Service</text>

          <rect x={195} y={80} width={85} height={30} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={237} y={99} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">Feed Service</text>

          <rect x={195} y={120} width={85} height={30} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={237} y={139} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Media Svc</text>

          <rect x={195} y={160} width={85} height={30} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={237} y={179} textAnchor="middle" fill="#0891b2" fontSize="9" fontWeight="600" fontFamily="monospace">Social Graph</text>

          <rect x={195} y={200} width={85} height={30} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={237} y={219} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Story Service</text>

          {/* Workers */}
          <rect x={330} y={40} width={90} height={30} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={375} y={59} textAnchor="middle" fill="#ea580c" fontSize="9" fontWeight="600" fontFamily="monospace">Fan-Out Wkrs</text>

          <rect x={330} y={80} width={90} height={30} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={375} y={99} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Media Proc.</text>

          <rect x={330} y={120} width={90} height={30} rx={6} fill="#0284c710" stroke="#0284c7" strokeWidth={1.5}/>
          <text x={375} y={139} textAnchor="middle" fill="#0284c7" fontSize="9" fontWeight="600" fontFamily="monospace">Ranking Svc</text>

          <rect x={330} y={160} width={90} height={30} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
          <text x={375} y={179} textAnchor="middle" fill="#7c3aed" fontSize="9" fontWeight="600" fontFamily="monospace">Notification</text>

          {/* Data Stores */}
          <rect x={475} y={40} width={75} height={30} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={512} y={59} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Post DB</text>

          <rect x={475} y={80} width={75} height={30} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={512} y={99} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Feed Cache</text>
          <text x={512} y={108} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">Redis</text>

          <rect x={475} y={120} width={75} height={30} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={512} y={139} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Blob Store</text>
          <text x={512} y={148} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">S3 / photos</text>

          <rect x={475} y={160} width={75} height={30} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={512} y={179} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Graph DB</text>
          <text x={512} y={188} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">follows</text>

          <rect x={475} y={200} width={75} height={30} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={512} y={215} textAnchor="middle" fill="#ea580c" fontSize="9" fontWeight="600" fontFamily="monospace">Kafka</text>

          {/* External */}
          <rect x={600} y={40} width={90} height={30} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={645} y={59} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Search (ES)</text>

          <rect x={600} y={80} width={90} height={30} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={645} y={99} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">ML Ranking</text>

          <rect x={600} y={120} width={90} height={30} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={645} y={139} textAnchor="middle" fill="#0891b2" fontSize="9" fontWeight="600" fontFamily="monospace">Counter Svc</text>
          <text x={645} y={148} textAnchor="middle" fill="#0891b280" fontSize="7" fontFamily="monospace">likes, views</text>

          {/* Arrows */}
          <defs><marker id="ah-hld" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          <line x1={75} y1={127} x2={95} y2={72} stroke="#78716c80" strokeWidth={1} markerEnd="url(#ah-hld)" strokeDasharray="3,2"/>
          <line x1={75} y1={140} x2={95} y2={148} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={75} y1={177} x2={95} y2={160} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={160} y1={145} x2={195} y2={55} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-hld)"/>
          <line x1={160} y1={148} x2={195} y2={95} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-hld)"/>
          <line x1={160} y1={153} x2={195} y2={135} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-hld)"/>
          <line x1={160} y1={157} x2={195} y2={175} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-hld)"/>
          <line x1={160} y1={162} x2={195} y2={215} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-hld)"/>
          <line x1={280} y1={55} x2={330} y2={55} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={280} y1={135} x2={330} y2={95} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-hld)"/>
          <line x1={280} y1={95} x2={330} y2={135} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-hld)"/>
          <line x1={420} y1={55} x2={475} y2={55} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-hld)"/>
          <line x1={420} y1={55} x2={475} y2={95} stroke="#d97706" strokeWidth={1} markerEnd="url(#ah-hld)"/>
          <line x1={420} y1={95} x2={475} y2={135} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-hld)"/>
          <line x1={280} y1={175} x2={475} y2={175} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-hld)"/>
          <line x1={280} y1={55} x2={475} y2={215} stroke="#ea580c50" strokeWidth={1} markerEnd="url(#ah-hld)" strokeDasharray="3,2"/>
          <line x1={550} y1={55} x2={600} y2={55} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-hld)"/>
          <line x1={550} y1={95} x2={600} y2={95} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-hld)"/>
          <line x1={550} y1={55} x2={600} y2={135} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-hld)"/>

          {/* Flow labels */}
          <rect x={10} y={260} width={700} height={80} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={20} y={278} fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Upload: Client → API GW → Media Svc (pre-signed URL → S3) → Media Processor (resize 4×) → CDN → Post Svc (create post) → Kafka</text>
          <text x={20} y={293} fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">{"Fan-out: Kafka → Fan-Out Workers → for each follower: LPUSH feed:{user_id} post_id → Feed Cache (Redis)"}</text>
          <text x={20} y={308} fill="#2563eb" fontSize="8" fontWeight="600" fontFamily="monospace">{"Feed read: Client → Feed Svc → LRANGE feed:{user_id} 0 20 (from Redis) → hydrate post data → Ranking Svc → return ranked feed"}</text>
          <text x={20} y={323} fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Hybrid: Celebrity posts NOT fan-out'd. Feed Svc merges pre-computed feed + real-time query of celebrity posts at read time.</text>
        </svg>
      </Card>
      <Card>
        <Label color="#c026d3">Key Architecture Decisions</Label>
        <div className="grid grid-cols-3 gap-4">
          {[
            { q: "Why a hybrid fan-out approach?", a: "Pure fan-out-on-write: a celebrity with 100M followers → 100M Redis writes per post. Unacceptable. Pure fan-out-on-read: every feed load queries all 500 followed accounts → too slow. Hybrid: fan-out-on-write for normal users (<10K followers), fan-out-on-read for celebrities. Best of both worlds." },
            { q: "Why Kafka between Post Service and Fan-Out?", a: "Decouples the user-facing upload (fast, sub-second response) from the fan-out work (can take seconds for popular users). Kafka provides durability (if fan-out workers crash, messages are replayed), ordering (posts from same user arrive in order), and backpressure (fan-out workers consume at their own pace)." },
            { q: "Why Redis for feed cache instead of DB?", a: "Feed reads are the #1 hottest path (58K/sec). Redis LRANGE on a list of post IDs is sub-millisecond. DB would need index scans across hundreds of followed accounts. The feed cache is a materialised view — pre-computed so the read path is trivially fast. Trade-off: 2TB of Redis (expensive but worth it)." },
          ].map((d,i) => (
            <div key={i} className="rounded-lg border border-stone-200 p-3">
              <div className="text-[11px] font-bold text-stone-700">{d.q}</div>
              <div className="text-[11px] text-stone-500 mt-1">{d.a}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function AlgorithmSection() {
  const [sel, setSel] = useState("hybrid");
  const topics = {
    hybrid: { name: "Hybrid Fan-Out Strategy", cx: "Instagram's Core Feed Algorithm",
      desc: "The most important algorithmic decision: how does a new post reach all followers' feeds? Instagram uses a hybrid of fan-out-on-write (for normal users) and fan-out-on-read (for celebrities), with ML ranking on top.",
      code: `# On new post creation → Kafka event → fan-out workers

def fan_out_post(post):
    author = get_user(post.author_id)
    followers = get_followers(post.author_id)

    if author.follower_count > CELEBRITY_THRESHOLD:  # 10K
        # Celebrity: DON'T fan-out. Will be merged at read.
        mark_as_celebrity_post(post)
        return

    # Normal user: fan-out-on-write to all followers
    for batch in chunk(followers, 1000):
        pipeline = redis.pipeline()
        for follower_id in batch:
            key = f"feed:{follower_id}"
            pipeline.lpush(key, post.post_id)
            pipeline.ltrim(key, 0, 499)  # Keep last 500
        pipeline.execute()  # Batch Redis writes

# On feed read:
def get_feed(user_id, cursor, limit=20):
    # Step 1: Get pre-computed feed (fan-out-on-write posts)
    post_ids = redis.lrange(f"feed:{user_id}", 0, 499)

    # Step 2: Merge celebrity posts (fan-out-on-read)
    celeb_follows = get_celebrity_follows(user_id)
    for celeb_id in celeb_follows:
        recent = get_recent_posts(celeb_id, since=24h)
        post_ids.extend([p.id for p in recent])

    # Step 3: Hydrate posts (batch fetch from Post DB)
    posts = batch_get_posts(post_ids)

    # Step 4: Rank with ML model
    ranked = ranking_service.rank(user_id, posts)

    # Step 5: Paginate and return
    return paginate(ranked, cursor, limit)`,
      points: [
        "Fan-out-on-write for normal users (<10K followers): pre-compute feeds in Redis. A new post immediately appears in all followers' feed caches. Feed reads are instant (Redis LRANGE).",
        "Fan-out-on-read for celebrities (>10K followers): don't pre-compute. At feed load time, query recent posts from each celebrity the user follows and merge with the pre-computed feed.",
        "The threshold (10K followers) is tunable. Instagram's actual threshold is undisclosed but the principle is the same: avoid massive write amplification for popular accounts.",
        "Feed cache is a Redis list of post IDs (not full posts). LPUSH to prepend new post, LTRIM to keep max 500. Feed read: LRANGE to get IDs, then batch-hydrate from Post DB cache.",
        "ML ranking happens AFTER merging. The merged candidate set (500+ posts) is scored by a ranking model (engagement prediction, recency, relationship strength) and the top 20 are returned.",
      ],
    },
    media: { name: "Media Upload Pipeline", cx: "Photo Processing at Scale",
      desc: "Every photo uploaded to Instagram goes through a multi-step pipeline: validate, resize to multiple resolutions, strip metadata, compress, replicate to CDN, and generate a URL. At 100M photos/day, this pipeline is one of the largest data-processing systems in the world.",
      code: `# Media upload pipeline (async, triggered after upload)

# Step 1: Client uploads to pre-signed S3 URL
def get_upload_url(user_id, content_type):
    media_id = generate_id()
    url = s3.generate_presigned_url(
        bucket="uploads-raw",
        key=f"{media_id}/original",
        content_type=content_type,
        expires_in=3600
    )
    return {media_id, upload_url: url}

# Step 2: S3 event triggers media processor
def process_media(media_id):
    original = s3.get(f"uploads-raw/{media_id}/original")

    # Validate
    if not is_valid_image(original):
        mark_failed(media_id, "invalid_format")
        return

    # Strip EXIF (privacy: remove GPS, camera info)
    cleaned = strip_exif(original)

    # Resize to multiple variants
    variants = {
        "thumb":  resize(cleaned, 150, 150),   # Square thumb
        "small":  resize(cleaned, 320),          # List view
        "medium": resize(cleaned, 640),          # Feed view
        "large":  resize(cleaned, 1080),         # Full screen
    }

    # Compress (JPEG quality 85, WebP for supported clients)
    for name, img in variants.items():
        jpeg = compress(img, format="jpeg", quality=85)
        webp = compress(img, format="webp", quality=80)
        s3.put(f"media/{media_id}/{name}.jpg", jpeg)
        s3.put(f"media/{media_id}/{name}.webp", webp)

    # CDN invalidation / preload (optional)
    cdn.prefetch([f"media/{media_id}/medium.jpg"])

    # Update status
    db.update_media(media_id, status="ready",
                    urls=generate_cdn_urls(media_id))

# Step 3: Client polls for processing completion
# GET /v1/media/{media_id}/status
# Returns: {status: "ready", urls: {thumb, small, medium}}`,
      points: [
        "Pre-signed URL upload: client uploads directly to S3, bypassing the application server. Saves bandwidth and CPU on your servers. S3 event triggers the processing pipeline.",
        "4 resize variants cover all use cases: thumbnail (150px) for grid view, small (320px) for list/story, medium (640px) for feed, large (1080px) for full-screen. Serve the smallest that fits the viewport.",
        "EXIF stripping is critical for privacy. Raw photos contain GPS coordinates, camera serial number, timestamps. Must be removed before serving. Legal requirement in many jurisdictions.",
        "Dual format (JPEG + WebP): WebP is 25-35% smaller than JPEG at equivalent quality. Serve WebP to supported clients (Chrome, Android), JPEG as fallback (older Safari). Saves petabytes of CDN bandwidth.",
        "At 100M photos/day × 4 variants × 2 formats = 800M files/day generated. This pipeline is typically implemented as a Kafka consumer fleet — each worker processes one photo end-to-end in ~2-3 seconds.",
      ],
    },
    ranking: { name: "Feed Ranking (ML)", cx: "From Chronological to Personalised",
      desc: "Instagram switched from chronological to ranked feeds in 2016. The ranking model predicts which posts a user will engage with (like, comment, save, share) and orders the feed to maximise engagement. This single change dramatically increased time-spent on the platform.",
      code: `# Feed ranking pipeline (simplified)
# Runs at feed read time on the merged candidate set

def rank_feed(user_id, candidate_posts):
    user = get_user_features(user_id)
    # user features: interests, past engagement, time of day,
    #   following list, activity level, device type

    scored_posts = []
    for post in candidate_posts:
        author = get_user_features(post.author_id)
        post_feats = get_post_features(post.post_id)

        # Feature vector
        features = {
            # Relationship signals
            "interaction_freq": get_interaction_freq(
                user_id, post.author_id),  # How often user
                                           # engages with author
            "is_close_friend": is_close(user_id, post.author_id),

            # Post signals
            "post_age_hours": hours_since(post.created_at),
            "engagement_velocity": post_feats.likes_per_hour,
            "media_type": post.media_type,  # photo vs video
            "has_carousel": post.carousel_count > 1,

            # User signals
            "user_active_hours": user.typical_active_hours,
            "session_depth": get_session_depth(user_id),

            # Content signals
            "topic_affinity": cosine_sim(
                user.interest_embedding,
                post_feats.content_embedding),
        }

        # ML model predicts P(engage)
        score = ml_model.predict(features)
        # score = weighted sum of P(like), P(comment),
        #         P(save), P(share), with different weights

        scored_posts.append((post, score))

    # Sort by score descending
    ranked = sorted(scored_posts, key=lambda x: -x[1])

    # Diversity injection: avoid 5 posts from same author
    ranked = inject_diversity(ranked)

    return [post for post, _ in ranked[:50]]  # Top 50`,
      points: [
        "The ranking model predicts P(engagement) for each candidate post. Engagement = like, comment, save, share, or even 'time spent viewing'. Each action has a different weight.",
        "Relationship signal is the strongest feature: how often user X interacts with author Y. If you always like Alice's photos, Alice's posts rank higher. This creates the 'close friends bubble' effect.",
        "Post age is a decay factor — newer posts score higher. But a 6-hour-old post from a close friend beats a 5-minute-old post from a distant acquaintance. Age interacts with relationship strength.",
        "Content embedding similarity: the model learns user interest vectors and post content vectors (from image recognition + caption NLP). Similar content to what you've engaged with before scores higher.",
        "Diversity injection prevents the feed from being 10 posts from the same person. After ranking, insert diversity constraints: max 2 posts from same author in any 10-post window.",
      ],
    },
    stories: { name: "Stories Architecture", cx: "Ephemeral 24h Content",
      desc: "Stories are fundamentally different from posts: they expire after 24 hours, are viewed in a full-screen swipe format, and have their own 'tray' (horizontal scroll at the top of the feed). This requires separate storage, separate fan-out, and a TTL-based cleanup system.",
      code: `# Stories have unique characteristics:
# 1. TTL: auto-delete after 24 hours
# 2. View tracking: who has seen each story
# 3. Tray ordering: which users' stories to show first
# 4. Sequential: user swipes through stories in order

# Story storage (Redis + DB)
def post_story(user_id, media_id):
    story_id = generate_id()
    expires_at = now() + 24 * 3600

    # Store in DB (permanent for analytics, highlights)
    db.insert_story(story_id, user_id, media_id, expires_at)

    # Store in Redis with TTL (hot path for reads)
    key = f"stories:{user_id}"
    redis.zadd(key, {story_id: now()})  # Sorted by time
    redis.expire(key, 24 * 3600)

    # Fan-out: notify followers they have a new story
    # Lighter than post fan-out — just update tray position
    followers = get_followers(user_id)
    for follower_batch in chunk(followers, 1000):
        pipeline = redis.pipeline()
        for f_id in follower_batch:
            tray_key = f"story_tray:{f_id}"
            pipeline.zadd(tray_key, {user_id: now()})
        pipeline.execute()

    return {story_id, expires_at}

# Load stories tray
def get_story_tray(user_id):
    tray_key = f"story_tray:{user_id}"
    # Users with stories, sorted by most recent
    authors = redis.zrevrange(tray_key, 0, 50)

    trays = []
    for author_id in authors:
        stories_key = f"stories:{author_id}"
        story_ids = redis.zrange(stories_key, 0, -1)
        if not story_ids:  # All expired
            redis.zrem(tray_key, author_id)
            continue
        seen = get_seen_stories(user_id, story_ids)
        trays.append({author_id, story_ids, seen})

    # Sort: unseen stories first, then by recency
    return sort_trays(trays)

# View tracking (lightweight)
def mark_story_seen(user_id, story_id):
    redis.sadd(f"seen:{user_id}:{story_id[:10]}", story_id)
    redis.expire(f"seen:{user_id}:{story_id[:10]}", 48*3600)`,
      points: [
        "Redis sorted sets for stories: ZADD with timestamp as score. ZRANGEBYSCORE to get active stories. Redis TTL handles expiry automatically — no cleanup job needed.",
        "Story tray is a separate sorted set per user: which followed users have active stories. Updated on story post (fan-out), cleaned up lazily when a user's stories all expire.",
        "View tracking is append-only: SADD to a set of seen story IDs. Lightweight — no need to track exact view time. Expires 48h after creation (24h past story expiry).",
        "Stories fan-out is lighter than post fan-out: you're updating a tray position (ZADD), not inserting a full post into a feed list. Much cheaper per-follower operation.",
        "Stories media is stored in the same blob storage as posts, but with lifecycle rules: S3 lifecycle policy deletes raw media after 30 days (24h for visibility + 7 days for Highlights + buffer). DB record kept for analytics.",
      ],
    },
  };
  const t = topics[sel];
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Fan-Out Strategy Comparison</Label>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] border-collapse">
            <thead><tr className="border-b-2 border-stone-200">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Strategy</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">How It Works</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Write Cost</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Read Cost</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Verdict</th>
            </tr></thead>
            <tbody>
              {[
                { n:"Fan-Out on Write (Push)", how:"On post: write to ALL followers' caches", w:"O(followers) per post — 100M for celeb", r:"O(1) — just read from cache", v:"⚠️ Write amplification" },
                { n:"Fan-Out on Read (Pull)", how:"On feed read: query all followed users' posts", w:"O(1) — just store the post", r:"O(following) — query 500 accounts, merge", v:"⚠️ Slow reads" },
                { n:"Hybrid ★", how:"Push for normal, pull for celebrities", w:"O(followers) only for <10K follower users", r:"O(1) + O(celebrity_follows) — fast", v:"✔ Best of both", hl:true },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-purple-50" : i%2 ? "bg-stone-50/50" : ""}>
                  <td className={`px-3 py-2.5 font-mono ${r.hl?"text-purple-700 font-bold":"text-stone-600"}`}>{r.n}</td>
                  <td className="text-center px-3 py-2.5 text-stone-500">{r.how}</td>
                  <td className="text-center px-3 py-2.5 text-stone-500">{r.w}</td>
                  <td className="text-center px-3 py-2.5 text-stone-500">{r.r}</td>
                  <td className="text-center px-3 py-2.5 text-[10px]">{r.v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-3 space-y-2">
          {Object.entries(topics).map(([k,v]) => (
            <button key={k} onClick={() => setSel(k)}
              className={`w-full text-left px-3.5 py-2.5 rounded-lg text-[12px] font-medium border transition-all ${k===sel?"bg-purple-600 text-white border-purple-600":"bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
              {v.name}
            </button>
          ))}
        </div>
        <div className="col-span-9 space-y-5">
          <Card>
            <div className="flex items-center gap-3 mb-2">
              <span className="text-[14px] font-bold text-stone-800">{t.name}</span>
              <Pill bg="#f3e8ff" color="#7c3aed">{t.cx}</Pill>
            </div>
            <p className="text-[12px] text-stone-500 mb-3">{t.desc}</p>
            <div className="space-y-1.5">
              {t.points.map((p,i) => <Point key={i} icon="→" color="#9333ea">{p}</Point>)}
            </div>
          </Card>
          <CodeBlock title={`${t.name} — Pseudocode`} code={t.code} />
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
          <CodeBlock code={`-- Posts (sharded by user_id for profile queries)
CREATE TABLE posts (
  post_id         BIGINT PRIMARY KEY,     -- Snowflake ID
  author_id       BIGINT NOT NULL,
  caption         TEXT,
  location        JSONB,                  -- {lat, lng, name}
  media_type      ENUM('photo','video','carousel'),
  media_urls      JSONB,                  -- {thumb, small, medium, large}
  hashtags        TEXT[],
  tagged_users    BIGINT[],
  like_count      INT DEFAULT 0,          -- Denormalized counter
  comment_count   INT DEFAULT 0,
  created_at      TIMESTAMP NOT NULL,
  INDEX idx_author (author_id, created_at DESC)
);

-- Follows (social graph — sharded by follower_id)
CREATE TABLE follows (
  follower_id     BIGINT NOT NULL,
  followee_id     BIGINT NOT NULL,
  created_at      TIMESTAMP,
  PRIMARY KEY (follower_id, followee_id),
  INDEX idx_followee (followee_id)
);
-- Shard by follower_id: "who do I follow?" is single-shard
-- "Who follows me?" needs scatter-gather (or reverse index)

-- Likes (sharded by post_id for count queries)
CREATE TABLE likes (
  user_id         BIGINT NOT NULL,
  post_id         BIGINT NOT NULL,
  created_at      TIMESTAMP,
  PRIMARY KEY (post_id, user_id)
);
-- "Has user X liked post Y?" = point lookup
-- "Who liked post Y?" = range scan on post_id

-- Comments (sharded by post_id)
CREATE TABLE comments (
  comment_id      BIGINT PRIMARY KEY,
  post_id         BIGINT NOT NULL,
  author_id       BIGINT NOT NULL,
  text            TEXT NOT NULL,
  reply_to        BIGINT,                 -- Parent comment ID
  created_at      TIMESTAMP,
  INDEX idx_post (post_id, created_at)
);

-- Feed Cache (Redis)
-- Key: feed:{user_id}
-- Type: LIST of post_ids (sorted by insertion order)
-- Max length: 500 (LTRIM after LPUSH)

-- Stories (Redis + DB)
-- Redis: stories:{user_id} → Sorted Set {story_id: timestamp}
-- Redis: story_tray:{user_id} → Sorted Set {author_id: ts}
-- DB: stories table for persistence and analytics`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Why These Design Choices?</Label>
          <div className="space-y-3">
            {[
              { q: "Shard posts by user_id (not post_id)?", a: "Profile page ('show all posts by user X') is single-shard range scan on (author_id, created_at DESC). This is a hot query. Feed reads hydrate posts by post_id — uses a secondary index or separate post_id→shard lookup." },
              { q: "Shard follows by follower_id?", a: "'Who do I follow?' is the feed-build query — must be fast (single shard). 'Who follows me?' (follower list on profile) is less frequent and can use scatter-gather or a reverse index table." },
              { q: "Denormalized like_count on posts?", a: "Counting likes by scanning the likes table is O(n). A denormalized counter gives O(1) reads. Updated via atomic increment. May drift under extreme concurrency — reconciled periodically." },
              { q: "JSONB for media_urls?", a: "Each post has multiple media URLs (thumb, small, medium, large) in two formats (JPEG, WebP). JSONB is flexible and avoids a separate media_variants table with JOINs on the hot path." },
              { q: "Redis LIST for feed cache?", a: "LPUSH (prepend new post) and LRANGE (read page) are both O(1) for the operations we need. LTRIM keeps the list bounded at 500. Much simpler than a sorted set since we only need insertion-order within the pre-computed portion." },
              { q: "Why not Cassandra for everything?", a: "Cassandra is great for the feed cache / stories (write-heavy, TTL support). PostgreSQL for posts/users/follows (need ACID for follow/unfollow, complex queries for profiles). Polyglot persistence: right tool for each job." },
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
          <Label color="#059669">Read Path Scaling (Feed)</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Feed cache in Redis Cluster</strong> — 2TB across ~200 Redis nodes. Each user's feed is a list of 500 post IDs. LRANGE is sub-millisecond. The feed read hot path never touches the database.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Post hydration cache</strong> — after getting post IDs from feed cache, hydrate full post data from a second Redis cache (or Memcached). Cache-aside pattern: check cache → on miss, query Post DB → fill cache. 99%+ hit rate.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">CDN for all media</strong> — photos served from CDN edge (CloudFront/Akamai). Origin is S3. Once an image is cached at the edge (within seconds of first request), all subsequent reads are sub-50ms globally.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Read replicas for profile/search</strong> — profile pages, user search, explore page read from DB replicas. Only writes (post creation, follows, likes) go to primary. Read:write ratio of 50:1 means replicas handle most traffic.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Write Path Scaling (Upload & Fan-Out)</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Media upload direct to S3</strong> — client uploads via pre-signed URL. Application server never handles the binary data. S3 scales to unlimited concurrent uploads.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Kafka for async fan-out</strong> — post creation → Kafka topic → fan-out workers consume. Workers are horizontally scalable: add more workers to handle more posts. Each worker processes one post's fan-out independently.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Batched Redis writes</strong> — fan-out worker uses Redis pipeline: batch 1000 LPUSH commands into one round trip. Reduces network overhead 1000×. A 200-follower fan-out completes in a single pipeline call.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Counter service (separate)</strong> — likes, comments, view counts are high-frequency writes. Dedicated counter service with Redis HyperLogLog for unique counts and batched DB sync. Decoupled from the main post/feed path.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Strategy</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Option A: User-Homed Regions ★", d:"Each user is assigned to the nearest region. Their data (posts, feed cache, followers) lives in that region. Cross-region follows: remote query or async replication.", pros:["Low latency for local operations","Data locality for compliance (GDPR)","Each region is self-contained for local users"], cons:["Cross-region follows add latency","Celebrity content needs global replication","User relocation (travel) needs session migration"], pick:true },
            { t:"Option B: Global Feed Cache + Regional Writes", d:"Write path (uploads, fan-out) runs in user's home region. Feed cache replicated globally for fast reads anywhere.", pros:["Fast feed reads from any region","Write consistency in home region","CDN + global cache = excellent read latency"], cons:["Feed cache replication lag (1-5s)","Storage cost: 2TB × N regions","Cross-region fan-out for global followers"], pick:false },
            { t:"Option C: Follow-the-Sun Active-Active", d:"All regions accept reads and writes. Conflict resolution for concurrent operations.", pros:["Lowest latency for all operations","Survives full regional outage","No 'home region' concept"], cons:["Conflict resolution for likes/follows is complex","Eventual consistency everywhere","Counter accuracy suffers (double-counting risk)"], pick:false },
          ].map((o,i) => (
            <div key={i} className={`rounded-lg border p-4 ${o.pick ? "border-purple-300 bg-purple-50/50" : "border-stone-200 bg-stone-50/30"}`}>
              <div className={`text-[11px] font-bold mb-1.5 ${o.pick ? "text-purple-700" : "text-stone-600"}`}>{o.t}</div>
              <p className="text-[11px] text-stone-500 mb-2">{o.d}</p>
              <div className="space-y-0.5 text-[11px]">
                {o.pros.map((p,j) => <div key={j} className="text-emerald-600">✔ {p}</div>)}
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
      <Card accent="#d97706">
        <Label color="#d97706">Critical: Feed Must Always Load</Label>
        <p className="text-[12px] text-stone-500 mb-4">The feed is Instagram's entire product. If the feed doesn't load, the app is 'down' — even if uploads, profiles, and DMs still work. Every design decision prioritises feed availability above all else.</p>
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Feed Cache is the Source of Truth</div>
            <p className="text-[11px] text-stone-500">If the database is down, feed still loads from Redis cache. The cache IS the feed. DB is only needed for cache misses (cold-start users) and hydrating post details (which have a separate cache layer).</p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-white p-4">
            <div className="text-[11px] font-bold text-amber-700 mb-1.5">Graceful Degradation</div>
            <p className="text-[11px] text-stone-500">If fan-out workers are slow: feed still loads (slightly stale — missing very recent posts). If ranking service is down: serve unranked (chronological) feed. If media CDN is slow: show placeholders with text.</p>
          </div>
          <div className="rounded-lg border border-blue-200 bg-white p-4">
            <div className="text-[11px] font-bold text-blue-700 mb-1.5">Multi-Layer Caching</div>
            <p className="text-[11px] text-stone-500">Feed cache (Redis) → Post cache (Redis/Memcached) → DB read replicas → DB primary. Each layer absorbs failures of the layer below. Four layers of redundancy before a user sees an error.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Fan-Out Failure Handling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Kafka durability</strong> — if fan-out workers crash, messages stay in Kafka. Workers restart and resume from the last committed offset. No fan-out work is lost. RPO = 0 for the fan-out queue.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Partial fan-out is OK</strong> — if a fan-out reaches 180/200 followers and crashes, the remaining 20 miss the post in their pre-computed feed. They'll still see it via celebrity-path (fan-out-on-read) or on next app open.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Feed cache cold-start</strong> — new user or cache eviction → no pre-computed feed. Fall back to full fan-out-on-read: query recent posts from all followed accounts, rank, and return. Slower (~500ms) but works.</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Data Durability Guarantees</Label>
          <ul className="space-y-2.5">
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">Posts are sacred</strong> — every post written to primary DB with synchronous replication to at least 1 replica. S3 media has 99.999999999% durability. A user's post must NEVER be lost.</Point>
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">Feed cache is expendable</strong> — if Redis loses a user's feed cache, it's rebuilt on next access (fan-out-on-read fallback). The cache is a performance optimisation, not the source of truth for posts.</Point>
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">Likes/follows are eventually consistent</strong> — counter may lag by a few seconds. Follow graph replicated async. Acceptable: user sees "4,520 likes" when the real number is 4,523. Not acceptable: user sees their own like not reflected.</Point>
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">Read-your-own-writes</strong> — after a user posts, likes, or follows, their own view must immediately reflect the action. Use session-sticky routing or read-from-primary for the current user's data.</Point>
          </ul>
        </Card>
      </div>
    </div>
  );
}

function ObservabilitySection() {
  return (
    <div className="space-y-5">
      <Card accent="#0284c7">
        <Label color="#0284c7">Key Metrics Dashboard</Label>
        <div className="grid grid-cols-4 gap-3">
          {[
            { name: "Feed Load Latency", target: "p99 < 500ms", desc: "Time from app open to feed rendered", alarm: "> 1s" },
            { name: "Fan-Out Lag", target: "< 5s", desc: "Time from post creation to appearing in all followers' feeds", alarm: "> 30s" },
            { name: "Media Processing Time", target: "p99 < 10s", desc: "Upload to all variants ready on CDN", alarm: "> 30s" },
            { name: "CDN Cache Hit Rate", target: "> 95%", desc: "% of media requests served from CDN edge", alarm: "< 90%" },
            { name: "Feed Cache Hit Rate", target: "> 99%", desc: "% of feed reads served from Redis (no DB)", alarm: "< 95%" },
            { name: "Upload Success Rate", target: "> 99.5%", desc: "% of photo uploads that complete successfully", alarm: "< 98%" },
            { name: "Kafka Consumer Lag", target: "< 10K msgs", desc: "Fan-out worker backlog in Kafka", alarm: "> 100K" },
            { name: "Like Write Latency", target: "p99 < 50ms", desc: "Time from tap to like counter increment", alarm: "> 200ms" },
          ].map((m,i) => (
            <div key={i} className="rounded-lg border border-stone-200 p-3 bg-stone-50/30">
              <div className="text-[11px] font-bold text-stone-800">{m.name}</div>
              <div className="text-[12px] font-mono font-bold text-sky-700 mt-1">{m.target}</div>
              <div className="text-[10px] text-stone-400 mt-0.5">{m.desc}</div>
              <div className="text-[10px] text-red-500 mt-1">🚨 Alert: {m.alarm}</div>
            </div>
          ))}
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Distributed Tracing</Label>
          <CodeBlock code={`# Trace: User opens app → feed loads
trace_id: "feed-load-abc-123"
spans:
  ├─ [client] app_open              0ms
  ├─ [cdn] load_js_bundle          50ms   # Cached at edge
  ├─ [gateway] auth + route        55ms
  ├─ [feed_svc] get_feed_cache     57ms   # Redis LRANGE
  │   └─ [redis] lrange feed:u42   58ms   # 0.8ms
  ├─ [feed_svc] get_celeb_posts    60ms   # Fan-out-on-read
  │   ├─ [post_db] query celeb_1   62ms
  │   └─ [post_db] query celeb_2   63ms
  ├─ [feed_svc] merge_candidates   65ms
  ├─ [ranking_svc] rank_posts      75ms   # ML scoring
  ├─ [feed_svc] hydrate_posts      80ms   # Batch post cache
  │   └─ [redis] mget posts:*      82ms   # Multi-get
  ├─ [feed_svc] build_response     85ms
  └─ [client] render_feed         130ms   # Images from CDN
# Total: app open → feed visible ≈ 180ms`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Logging & Alerting Strategy</Label>
          <div className="space-y-3">
            {[
              { level: "P0 — Page Immediately", items: ["Feed cache (Redis cluster) unreachable", "S3 / media storage unavailable", "Kafka cluster down (fan-out stops)", "Post DB primary unreachable (no new posts)"] },
              { level: "P1 — Alert On-Call", items: ["Feed load latency p99 > 1s", "Fan-out lag > 30 seconds", "Media processing failure rate > 5%", "CDN cache hit rate < 90%"] },
              { level: "P2 — Dashboard Monitor", items: ["Like counter drift > 1% (reconciliation needed)", "Story expiry cleanup lagging", "Explore ranking model staleness > 1 hour", "Follower count async replication lag > 30s"] },
            ].map((l,i) => (
              <div key={i} className="rounded-lg border border-stone-200 p-3">
                <div className={`text-[10px] font-bold mb-1 ${i===0?"text-red-600":i===1?"text-amber-600":"text-sky-600"}`}>{l.level}</div>
                <div className="space-y-0.5">
                  {l.items.map((item,j) => (
                    <div key={j} className="text-[10px] text-stone-500 flex items-center gap-1.5">
                      <span className={`w-1 h-1 rounded-full shrink-0 ${i===0?"bg-red-400":i===1?"bg-amber-400":"bg-sky-400"}`}/>{item}
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

function WatchoutsSection() {
  return (
    <div className="space-y-5">
      <Card accent="#dc2626">
        <Label color="#dc2626">Critical Failure Scenarios</Label>
        <div className="grid grid-cols-2 gap-4">
          {[
            { title: "Celebrity Post → Fan-Out Storm", desc: "A normal user gains sudden fame (viral moment). They had 500 followers (fan-out-on-write) and post. Meanwhile, they gain 5M followers. Next post: system tries to fan-out to 5M feeds — but they're still classified as a 'normal' user.", impact: "HIGH — 5M Redis writes clog the fan-out workers. All other fan-outs are delayed. Feed freshness degrades globally.", mitigation: "Re-classify users on follow count change. If followers cross 10K threshold during fan-out, abort and switch to celebrity path. Rate-limit fan-out per post: max 100K writes. Beyond that, fall back to read-path merge." },
            { title: "Redis Feed Cache Cold-Start Stampede", desc: "Redis cluster restarts (upgrade or failure). All 500M users' feed caches are empty. First feed load for each user triggers full fan-out-on-read (query all followed accounts). 500M concurrent cold-start reads.", impact: "HIGH — Post DB crushed by 500M fan-out-on-read queries simultaneously. DB connection pools exhausted. Cascading failure.", mitigation: "Never cold-start the full cluster at once. Rolling restart: one shard at a time. Pre-warm caches from a Redis backup (RDB snapshot). Stagger user reconnections with client-side jitter. Feed cold-start fallback: serve a 'popular posts' placeholder while rebuilding the user's feed async." },
            { title: "Media Pipeline Backlog → Upload Failures", desc: "Media processor workers can't keep up. 100M photos/day with burst peaks. Processing queue grows. Users upload but their photo never appears ('processing' forever).", impact: "MEDIUM — posts are created but without processed media. Users see broken images. Upload success rate plummets. Users blame the app.", mitigation: "Auto-scale media processor fleet based on queue depth. Priority queue: process images for the post's immediate followers first (who will see it soonest). If queue > 1M: serve original (unresized) image as fallback while processing catches up." },
            { title: "Like Counter Inconsistency", desc: "High-frequency likes on a viral post: 50K likes/minute. Concurrent counter increments lead to lost updates (read-modify-write race). Like count shows 4.2M but actual likes in the likes table are 4.35M.", impact: "LOW-MEDIUM — visible to users as inaccurate counts. Not catastrophic but erodes trust. Celebrities notice when their counts don't match.", mitigation: "Use Redis INCR (atomic) for real-time counter. Periodically reconcile with SELECT COUNT(*) from likes table. Accept ±1% drift in real-time; nightly reconciliation corrects to exact count." },
          ].map((f,i) => (
            <div key={i} className="rounded-lg border border-stone-200 p-4">
              <div className="text-[12px] font-bold text-stone-800 mb-1.5">{f.title}</div>
              <p className="text-[11px] text-stone-500 mb-2">{f.desc}</p>
              <div className="text-[11px]"><span className="font-bold text-red-600">Impact:</span> <span className="text-stone-500">{f.impact}</span></div>
              <div className="text-[11px] mt-1"><span className="font-bold text-emerald-600">Mitigation:</span> <span className="text-stone-500">{f.mitigation}</span></div>
            </div>
          ))}
        </div>
      </Card>
      <Card>
        <Label color="#d97706">Race Conditions & Edge Cases</Label>
        <div className="space-y-3">
          {[
            { case: "User unfollows immediately after seeing a post in feed", detail: "Feed was cached before the unfollow. User sees posts from someone they just unfollowed. Acceptable: the feed snapshot was valid when generated. On next refresh, unfollowed user's posts are excluded. Don't retroactively remove posts from cached feeds." },
            { case: "Post deleted while fan-out is in progress", detail: "Author deletes a post. Fan-out workers have already pushed it to 50% of followers' feeds. Solution: deletion event also published to Kafka. Separate cleanup worker removes post_id from all feed caches. In the interim, hydration step filters out deleted posts (tombstone check)." },
            { case: "User blocks someone who has already liked/commented on their post", detail: "Block should hide the blocker's content from the blocked user AND hide the blocked user's interactions from the blocker. Lazy cleanup: filter blocked users' content at read time. Async background job removes likes/comments from blocked relationships." },
            { case: "Story viewed counter exceeds actual viewers (bot views)", detail: "Bots inflate story view counts. Use device fingerprinting + rate limiting for view tracking. Deduplicate views by user_id (not session). Accept that public stories are harder to protect than private ones." },
          ].map((c,i) => (
            <div key={i} className="flex gap-3 items-start">
              <span className="text-amber-500 text-sm mt-0.5">⚡</span>
              <div>
                <div className="text-[11px] font-bold text-stone-700">{c.case}</div>
                <div className="text-[11px] text-stone-500 mt-0.5">{c.detail}</div>
              </div>
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
      <Card accent="#0f766e">
        <Label color="#0f766e">Service Architecture Diagram</Label>
        <svg viewBox="0 0 720 310" className="w-full">
          <defs>
            <marker id="sa-arr" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker>
            <marker id="sa-kafka" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#ea580c"/></marker>
          </defs>

          {/* Row labels */}
          <text x={10} y={35} fill="#a8a29e" fontSize="8" fontWeight="600" fontFamily="monospace" textAnchor="start">API LAYER</text>
          <text x={10} y={115} fill="#a8a29e" fontSize="8" fontWeight="600" fontFamily="monospace" textAnchor="start">CORE SERVICES</text>
          <text x={10} y={195} fill="#a8a29e" fontSize="8" fontWeight="600" fontFamily="monospace" textAnchor="start">ASYNC WORKERS</text>
          <text x={10} y={268} fill="#a8a29e" fontSize="8" fontWeight="600" fontFamily="monospace" textAnchor="start">DATA STORES</text>

          {/* Divider lines */}
          <line x1={10} y1={42} x2={710} y2={42} stroke="#e7e5e4" strokeWidth={0.5} strokeDasharray="4,3"/>
          <line x1={10} y1={122} x2={710} y2={122} stroke="#e7e5e4" strokeWidth={0.5} strokeDasharray="4,3"/>
          <line x1={10} y1={202} x2={710} y2={202} stroke="#e7e5e4" strokeWidth={0.5} strokeDasharray="4,3"/>

          {/* API Gateway */}
          <rect x={280} y={52} width={160} height={32} rx={6} fill="#6366f110" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={360} y={72} textAnchor="middle" fill="#6366f1" fontSize="10" fontWeight="700" fontFamily="monospace">API Gateway / LB</text>

          {/* Core Services */}
          <rect x={60} y={132} width={90} height={34} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={105} y={153} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Post Service</text>

          <rect x={170} y={132} width={90} height={34} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={215} y={153} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">Feed Service</text>

          <rect x={280} y={132} width={90} height={34} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={325} y={153} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Media Svc</text>

          <rect x={390} y={132} width={90} height={34} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={435} y={153} textAnchor="middle" fill="#0891b2" fontSize="9" fontWeight="600" fontFamily="monospace">Social Graph</text>

          <rect x={500} y={132} width={90} height={34} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={545} y={153} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Story Svc</text>

          <rect x={610} y={132} width={90} height={34} rx={6} fill="#0284c710" stroke="#0284c7" strokeWidth={1.5}/>
          <text x={655} y={153} textAnchor="middle" fill="#0284c7" fontSize="9" fontWeight="600" fontFamily="monospace">Search Svc</text>

          {/* Async Workers */}
          <rect x={80} y={210} width={100} height={30} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={130} y={229} textAnchor="middle" fill="#ea580c" fontSize="9" fontWeight="600" fontFamily="monospace">Fan-Out Wkrs</text>

          <rect x={210} y={210} width={110} height={30} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={265} y={229} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Media Processor</text>

          <rect x={345} y={210} width={100} height={30} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
          <text x={395} y={229} textAnchor="middle" fill="#7c3aed" fontSize="9" fontWeight="600" fontFamily="monospace">ML Ranking</text>

          <rect x={470} y={210} width={100} height={30} rx={6} fill="#be123c10" stroke="#be123c" strokeWidth={1.5}/>
          <text x={520} y={229} textAnchor="middle" fill="#be123c" fontSize="9" fontWeight="600" fontFamily="monospace">Notification</text>

          <rect x={595} y={210} width={105} height={30} rx={6} fill="#0f766e10" stroke="#0f766e" strokeWidth={1.5}/>
          <text x={647} y={229} textAnchor="middle" fill="#0f766e" fontSize="9" fontWeight="600" fontFamily="monospace">Counter Svc</text>

          {/* Data Stores */}
          <rect x={60} y={275} width={75} height={28} rx={5} fill="#78716c10" stroke="#78716c" strokeWidth={1.2}/>
          <text x={97} y={293} textAnchor="middle" fill="#78716c" fontSize="8" fontWeight="600" fontFamily="monospace">Post DB (PG)</text>

          <rect x={155} y={275} width={75} height={28} rx={5} fill="#d9770610" stroke="#d97706" strokeWidth={1.2}/>
          <text x={192} y={289} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Feed Cache</text>
          <text x={192} y={298} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">Redis</text>

          <rect x={250} y={275} width={75} height={28} rx={5} fill="#dc262610" stroke="#dc2626" strokeWidth={1.2}/>
          <text x={287} y={289} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Blob Store</text>
          <text x={287} y={298} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">S3</text>

          <rect x={345} y={275} width={75} height={28} rx={5} fill="#05966910" stroke="#059669" strokeWidth={1.2}/>
          <text x={382} y={289} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Graph DB</text>
          <text x={382} y={298} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">follows</text>

          <rect x={440} y={275} width={75} height={28} rx={5} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.2}/>
          <text x={477} y={293} textAnchor="middle" fill="#ea580c" fontSize="8" fontWeight="600" fontFamily="monospace">Kafka</text>

          <rect x={535} y={275} width={75} height={28} rx={5} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.2}/>
          <text x={572} y={293} textAnchor="middle" fill="#9333ea" fontSize="8" fontWeight="600" fontFamily="monospace">Elasticsearch</text>

          <rect x={630} y={275} width={75} height={28} rx={5} fill="#78716c10" stroke="#78716c" strokeWidth={1.2}/>
          <text x={667} y={289} textAnchor="middle" fill="#78716c" fontSize="8" fontWeight="600" fontFamily="monospace">CDN</text>
          <text x={667} y={298} textAnchor="middle" fill="#78716c80" fontSize="7" fontFamily="monospace">CloudFront</text>

          {/* Arrows: Gateway → Services */}
          <line x1={300} y1={84} x2={105} y2={132} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#sa-arr)"/>
          <line x1={330} y1={84} x2={215} y2={132} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#sa-arr)"/>
          <line x1={360} y1={84} x2={325} y2={132} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#sa-arr)"/>
          <line x1={390} y1={84} x2={435} y2={132} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#sa-arr)"/>
          <line x1={410} y1={84} x2={545} y2={132} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#sa-arr)"/>
          <line x1={430} y1={84} x2={655} y2={132} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#sa-arr)"/>

          {/* Arrows: Post Svc → Kafka (event) */}
          <line x1={105} y1={166} x2={477} y2={275} stroke="#ea580c" strokeWidth={1.2} markerEnd="url(#sa-kafka)" strokeDasharray="4,2"/>
          {/* Kafka → Fan-Out Workers */}
          <line x1={477} y1={275} x2={130} y2={240} stroke="#ea580c" strokeWidth={1.2} markerEnd="url(#sa-kafka)" strokeDasharray="4,2"/>
          {/* Fan-Out → Feed Cache (Redis) */}
          <line x1={130} y1={240} x2={192} y2={275} stroke="#d97706" strokeWidth={1} markerEnd="url(#sa-arr)"/>
          {/* Feed Svc → Feed Cache */}
          <line x1={215} y1={166} x2={192} y2={275} stroke="#d97706" strokeWidth={1} markerEnd="url(#sa-arr)"/>
          {/* Feed Svc → ML Ranking */}
          <line x1={240} y1={166} x2={395} y2={210} stroke="#7c3aed80" strokeWidth={1} markerEnd="url(#sa-arr)"/>
          {/* Media Svc → S3 */}
          <line x1={325} y1={166} x2={287} y2={275} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#sa-arr)"/>
          {/* Media Svc → Media Processor */}
          <line x1={325} y1={166} x2={265} y2={210} stroke="#d97706" strokeWidth={1} markerEnd="url(#sa-arr)"/>
          {/* Media Processor → CDN */}
          <line x1={290} y1={240} x2={667} y2={275} stroke="#78716c80" strokeWidth={1} markerEnd="url(#sa-arr)" strokeDasharray="3,2"/>
          {/* Social Graph → Graph DB */}
          <line x1={435} y1={166} x2={382} y2={275} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#sa-arr)"/>
          {/* Post Svc → Post DB */}
          <line x1={105} y1={166} x2={97} y2={275} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#sa-arr)"/>
          {/* Search Svc → Elasticsearch */}
          <line x1={655} y1={166} x2={572} y2={275} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#sa-arr)"/>
          {/* Notification → Kafka */}
          <line x1={520} y1={240} x2={477} y2={275} stroke="#ea580c80" strokeWidth={1} markerEnd="url(#sa-kafka)" strokeDasharray="4,2"/>
        </svg>
      </Card>
      <Card accent="#0f766e">
        <Label color="#0f766e">Service Decomposition</Label>
        <div className="grid grid-cols-3 gap-4">
          {[
            { name: "Post Service", owns: "Post CRUD, caption processing, hashtag extraction, post metadata", tech: "Go + PostgreSQL (sharded by user_id) + Kafka", api: "gRPC: CreatePost, GetPost, DeletePost, GetUserPosts", scale: "Horizontal — stateless, shards by user_id", stateful: false,
              modules: ["Post Writer (validate, persist, publish to Kafka)", "Caption Processor (extract hashtags, mentions, URLs)", "Post Cache (Redis — hot posts for feed hydration)", "Deletion Handler (soft delete + Kafka event for fan-out cleanup)", "Carousel Manager (multi-image post ordering)", "Location Tagger (reverse geocode lat/lng to place name)"] },
            { name: "Feed Service", owns: "Feed generation, feed cache management, celebrity post merging, ranking orchestration", tech: "Go + Redis Cluster (feed cache) + ML Ranking (gRPC)", api: "gRPC: GetFeed, InvalidateFeed, RebuildFeed", scale: "Horizontal — stateless, reads from Redis cache", stateful: false,
              modules: ["Feed Reader (LRANGE from Redis + celebrity merge)", "Feed Writer (fan-out workers: LPUSH to follower feeds)", "Celebrity Detector (threshold-based classification)", "Ranking Orchestrator (call ML service, apply diversity rules)", "Cache Manager (TTL, eviction, cold-start fallback)", "Feed Debugger (explain why a post appeared/didn't appear)"] },
            { name: "Media Service", owns: "Photo/video upload, processing pipeline, CDN management, media URLs", tech: "Go + S3 (storage) + Kafka (processing queue) + CDN", api: "gRPC: GetUploadURL, GetMediaStatus, GetMediaURLs", scale: "Horizontal — processing workers scale independently", stateful: false,
              modules: ["Upload Manager (pre-signed URLs, multipart handling)", "Image Processor (resize, compress, EXIF strip, WebP convert)", "Video Processor (transcode to H.264/H.265, generate thumbnail)", "CDN Manager (invalidation, pre-fetch, URL signing)", "Abuse Scanner (nudity detection, copyright check — ML)", "Storage Lifecycle (archive old media, glacier for deleted posts)"] },
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

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#2563eb">
          <Label color="#2563eb">Supporting Services</Label>
          <div className="space-y-3">
            {[
              { name: "Social Graph Service", role: "Follow/unfollow, follower/following lists, mutual follows, block/mute. Backed by a graph-optimised store (TAO at Meta) or sharded PostgreSQL. Critical path for fan-out (who to send posts to)." },
              { name: "Story Service", role: "Story CRUD, 24h TTL management, story tray generation, view tracking. Redis-backed for hot path. Stories have separate fan-out (lighter — tray position only, not full post data)." },
              { name: "Notification Service", role: "Push notifications (new follower, like, comment, mention). Batching: 'Alice and 5 others liked your post' instead of 6 separate pushes. Async via Kafka. Respects user notification preferences." },
              { name: "Search & Explore Service", role: "User/hashtag/location search (Elasticsearch). Explore page: recommendation engine using content embeddings, engagement signals, and collaborative filtering. Separate infrastructure from the feed." },
            ].map((s,i) => (
              <div key={i} className="rounded-lg border border-stone-200 p-3">
                <div className="text-[11px] font-bold text-stone-800 mb-0.5">{s.name}</div>
                <div className="text-[10px] text-stone-500">{s.role}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Technology Choices</Label>
          <div className="space-y-3">
            {[
              { comp: "Feed Cache", choice: "Redis Cluster", why: "Sub-ms list operations (LPUSH, LRANGE). 2TB across 200 nodes. The entire feed experience depends on Redis being fast and available." },
              { comp: "Post / User DB", choice: "PostgreSQL (sharded)", why: "ACID for post creation, follow/unfollow. Sharded by user_id. Strong consistency for user-facing writes. Read replicas for profiles." },
              { comp: "Media Storage", choice: "S3 + CloudFront CDN", why: "Unlimited storage for photos/videos. 11 nines durability. CDN edge caching for global sub-100ms delivery. Lifecycle policies for archival." },
              { comp: "Message Queue", choice: "Kafka", why: "Fan-out event streaming. Millions of events/sec. Replay on failure. Partitioned by author_id for ordering. Consumer groups for parallel processing." },
              { comp: "Social Graph", choice: "TAO / Graph DB or Sharded PostgreSQL", why: "Optimised for adjacency queries: 'who does X follow?' and 'who follows X?' Must handle billions of edges (follows) with sub-ms lookups." },
              { comp: "Search", choice: "Elasticsearch", why: "Full-text search for usernames, hashtags, locations. Inverted index for fast prefix search. Updated async via CDC from primary DB." },
            ].map((t,i) => (
              <div key={i}>
                <div className="text-[11px]"><span className="font-bold text-stone-700">{t.comp}:</span> <span className="font-mono text-amber-700">{t.choice}</span></div>
                <div className="text-[10px] text-stone-400 mt-0.5">{t.why}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function FlowsSection() {
  const [sel, setSel] = useState("upload");
  const flows = {
    upload: { name: "Photo Upload & Fan-Out Flow", steps: [
      { actor: "Client", action: "User takes photo, adds caption and filters", detail: "Client requests pre-signed S3 upload URL from Media Service. Uploads photo directly to S3 (bypassing app server)." },
      { actor: "Media Service", action: "S3 upload triggers processing pipeline", detail: "S3 event → Kafka → Media Processor worker picks up. Resizes to 4 variants (150px, 320px, 640px, 1080px). Strips EXIF. Generates JPEG + WebP." },
      { actor: "Media Service", action: "Push variants to CDN, update media status", detail: "Upload all variants to S3 production bucket. CDN auto-caches on first request. Update DB: media status = 'ready', store URLs." },
      { actor: "Client", action: "User hits 'Share' — post creation", detail: "POST /v1/posts with media_ids, caption, tags. Media already processed. Post Service validates, persists to Post DB, publishes event to Kafka." },
      { actor: "Fan-Out Workers", action: "Consume post event from Kafka", detail: "Get follower list from Social Graph Service. If author has <10K followers: LPUSH post_id to each follower's feed cache in Redis (batched pipeline)." },
      { actor: "Fan-Out Workers", action: "Trim feed caches + send notifications", detail: "LTRIM each feed list to 500. Enqueue push notifications for close friends / users with notifications enabled." },
      { actor: "Follower's Client", action: "User opens app → feed loads with new post", detail: "Feed Service reads from Redis cache → hydrates posts → ranks → returns. New post appears in feed within ~5 seconds of upload." },
    ]},
    feed: { name: "Feed Read Flow", steps: [
      { actor: "Client", action: "User opens Instagram app (pull-to-refresh)", detail: "GET /v1/feed?cursor=null&limit=20. First page load — no cursor." },
      { actor: "API Gateway", action: "Authenticate + route to Feed Service", detail: "Verify JWT. Rate limit. Route to nearest Feed Service instance." },
      { actor: "Feed Service", action: "Read pre-computed feed from Redis", detail: "LRANGE feed:{user_id} 0 499 → get 500 post IDs. These are from fan-out-on-write (normal users' posts)." },
      { actor: "Feed Service", action: "Merge celebrity posts (fan-out-on-read)", detail: "Get list of celebrities this user follows. For each: query Post DB (or post cache) for their last 24h of posts. Merge into candidate set." },
      { actor: "Ranking Service", action: "Score and rank candidate posts", detail: "ML model scores each candidate: P(like), P(comment), P(save). Features: relationship strength, post age, content affinity. Return ranked list." },
      { actor: "Feed Service", action: "Hydrate top 20 posts with full data", detail: "Batch MGET from post cache (Redis): media URLs, caption, author info, like count, comment preview. Cache miss → query Post DB." },
      { actor: "Client", action: "Render feed with images from CDN", detail: "Display ranked posts. Images loaded from CDN edge (sub-100ms). Lazy-load below-the-fold images. Prefetch next page cursor." },
    ]},
    like: { name: "Like Flow", steps: [
      { actor: "Client", action: "User double-taps photo to like", detail: "POST /v1/posts/:id/like. Optimistic UI: heart animation plays immediately, before server response." },
      { actor: "API Gateway", action: "Route to Post Service", detail: "Idempotent endpoint: liking an already-liked post is a no-op (returns success)." },
      { actor: "Post Service", action: "Write like to DB + increment counter", detail: "UPSERT INTO likes(user_id, post_id). Redis INCR post:likes:{post_id}. Both operations succeed or the like is not counted." },
      { actor: "Notification Service", action: "Notify post author", detail: "Kafka event → Notification Service. If author has notifications enabled: push notification 'user_42 liked your photo'. Batching: 'user_42 and 5 others liked your photo'." },
      { actor: "Counter Service", action: "Async reconciliation", detail: "Background: periodically compare Redis counter with SELECT COUNT(*) FROM likes WHERE post_id=X. Fix any drift." },
    ]},
  };
  const f = flows[sel];
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-3 space-y-2">
          {Object.entries(flows).map(([k,v]) => (
            <button key={k} onClick={() => setSel(k)}
              className={`w-full text-left px-3.5 py-2.5 rounded-lg text-[12px] font-medium border transition-all ${k===sel?"bg-purple-600 text-white border-purple-600":"bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
              {v.name}
            </button>
          ))}
        </div>
        <div className="col-span-9">
          <Card accent="#7e22ce">
            <Label color="#7e22ce">{f.name}</Label>
            <div className="space-y-0">
              {f.steps.map((s,i) => (
                <div key={i} className={`flex gap-4 py-3 ${i < f.steps.length-1 ? "border-b border-stone-100" : ""}`}>
                  <div className="flex flex-col items-center">
                    <span className="w-7 h-7 rounded-full bg-purple-100 text-purple-700 text-[11px] font-bold flex items-center justify-center">{i+1}</span>
                    {i < f.steps.length-1 && <div className="w-px flex-1 bg-purple-200 mt-1"/>}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono font-bold text-purple-600 bg-purple-50 px-2 py-0.5 rounded">{s.actor}</span>
                      <span className="text-[12px] font-medium text-stone-700">{s.action}</span>
                    </div>
                    <div className="text-[11px] text-stone-400 mt-0.5">{s.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function DeploymentSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#b45309">
          <Label color="#b45309">Kubernetes Deployment — Feed Service</Label>
          <CodeBlock code={`apiVersion: apps/v1
kind: Deployment
metadata:
  name: feed-service
spec:
  replicas: 100                 # Scale for 58K feed reads/sec
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 0
  template:
    spec:
      containers:
      - name: feed
        image: feed-service:v7.3
        ports:
        - containerPort: 8080   # gRPC
        resources:
          requests:
            memory: "1Gi"
            cpu: "2"
          limits:
            memory: "2Gi"
        env:
        - name: REDIS_CLUSTER
          value: "redis-feed.internal:6379"
        - name: RANKING_SERVICE
          value: "ranking-svc.internal:8081"
        - name: CELEBRITY_THRESHOLD
          value: "10000"
        - name: FEED_CACHE_SIZE
          value: "500"
        - name: FEED_TTL_HOURS
          value: "48"
        readinessProbe:
          grpc:
            port: 8080
          periodSeconds: 5

# Fan-Out Workers (separate deployment)
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fanout-workers
spec:
  replicas: 50                  # Scale with Kafka lag
  template:
    spec:
      containers:
      - name: fanout
        image: fanout-worker:v7.3
        env:
        - name: KAFKA_BROKERS
          value: "kafka.internal:9092"
        - name: CONSUMER_GROUP
          value: "fanout-cg"
        - name: REDIS_PIPELINE_BATCH
          value: "1000"`} />
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Security & Content Safety</Label>
          <div className="space-y-3">
            {[
              { layer: "Content Moderation (Critical)", details: ["ML-based image classification (nudity, violence, self-harm)", "Text analysis for hate speech in captions/comments", "Copyright detection (hash-matching against known content)", "Human review queue for borderline cases (24/7 trust & safety team)", "Automated removal of child exploitation content (CSAM) with NCMEC reporting", "Appeal process for false positives (user can contest removal)"] },
              { layer: "Authentication & Privacy", details: ["OAuth 2.0 for third-party app integrations", "Session management: JWT with 1h access token + refresh token", "Private accounts: followers must be approved", "Close Friends: restricted story audience", "Block/mute: hidden at read time, async cleanup"] },
              { layer: "Data Protection", details: ["TLS 1.3 for all connections (API, CDN, internal services)", "Encryption at rest for all user data (AES-256)", "GDPR: data export, right to deletion, consent management", "EXIF stripping from uploaded photos (GPS, device info)", "Rate limiting: API, uploads, follows, likes (anti-spam)"] },
              { layer: "Anti-Abuse", details: ["Bot detection: device fingerprint, behavioral patterns", "Spam filter for comments and DMs (ML classifier)", "Follow/like rate limiting (max 200 follows/day, 350 likes/hour)", "IP-based throttling for brute-force login attempts", "Two-factor authentication (SMS, TOTP, security keys)"] },
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
        <Label color="#be123c">Bottleneck Analysis</Label>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] border-collapse">
            <thead><tr className="border-b-2 border-stone-200">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Bottleneck</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Symptom</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Fix</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Pitfall</th>
            </tr></thead>
            <tbody>
              {[
                { b: "Redis Feed Cache Memory", s: "Redis cluster at 90% memory. Evictions start. Feed cache miss rate spikes.", f: "Add Redis nodes. Reduce feed cache size from 500 to 300 post IDs. Evict inactive users' feeds (not opened app in 7 days). Compress post IDs (varint encoding).", p: "Adding nodes requires resharding — migration takes hours. Plan capacity 3 months ahead. Never let Redis exceed 75% memory." },
                { b: "Kafka Consumer Lag (Fan-Out)", s: "Fan-out lag > 30s. Users don't see new posts from friends in their feed for minutes.", f: "Add more fan-out worker replicas. Increase partition count (requires topic recreation or auto-partition). Optimize Redis pipeline batch size.", p: "More partitions = more open file handles on Kafka brokers. Balance partition count with broker capacity. Also check: is one partition hot? (celebrity post on one partition)" },
                { b: "Media Processing Throughput", s: "Upload-to-ready time > 30s. Users see 'processing' spinner. Upload abandonment increases.", f: "Scale media processor fleet. Use GPU instances for video transcoding. Prioritize images over videos (90% of uploads, 10% of processing time). Serve original as fallback while processing.", p: "GPU instances are expensive. Cost per upload matters at 100M/day. Use spot instances for non-time-sensitive reprocessing. Reserve on-demand for real-time pipeline." },
                { b: "Ranking Service Latency", s: "Feed load p99 > 500ms. Ranking model inference taking 200ms+ per request.", f: "Reduce candidate set (rank 200 posts, not 500). Pre-compute features (don't compute interaction frequency at request time). Batch inference. Cache ranking results for 5 minutes.", p: "Caching ranked feeds means users see stale ranking. 5-minute TTL is acceptable for most users. Invalidate on explicit refresh (pull-to-refresh)." },
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
            { title: "World Cup Final Spike Crashes Fan-Out", symptom: "Messi scores the winning goal. 50M posts in 10 minutes. Fan-out workers can't keep up. Kafka lag reaches 500M messages. Feed freshness degrades to 15 minutes globally.",
              cause: "Normal load: ~1,160 posts/sec. World Cup spike: ~83K posts/sec (70× normal). Fan-out workers sized for 2× normal, not 70×. Kafka partitions saturated.",
              fix: "Pre-scale for known events: 10× fan-out workers before major events. Dynamic fan-out: during spikes, temporarily raise celebrity threshold from 10K to 1K followers (more users get read-path fan-out = less write pressure). Shed non-essential fan-out: delay notification fan-out, prioritize feed fan-out.",
              quote: "We knew the World Cup final would be big. We 3×'d capacity. We needed 70×. Now we have a 'global events' playbook that pre-scales to 100× and has a kill switch for non-essential fan-out." },
            { title: "CDN Cache Purge Stampede", symptom: "CDN vendor performs emergency cache purge (security incident). All cached images evicted. Every image request hits S3 origin. S3 rate limit (5,500 PUT/GET per prefix) exceeded. Images fail to load globally.",
              cause: "CDN cache held ~80% of active images. Cache purge = 80% of image requests suddenly hit origin. S3 prefix rate limiting kicked in. S3 returned 503 (SlowDown) for 40% of requests.",
              fix: "Multiple S3 prefix strategy: distribute images across 1000+ prefixes (use media_id hash as prefix). Each prefix gets its own 5,500 req/sec limit. Total: 5.5M req/sec origin capacity. Also: multi-CDN strategy — don't depend on a single CDN vendor. Secondary CDN warms from primary CDN, not from origin.",
              quote: "One CDN cache purge event taught us that our 'infinite scale' S3 storage has very finite request rate limits. Prefix randomization was a one-line fix that prevented the next incident." },
            { title: "Like Counter Drift Reaches 15% on Viral Post", symptom: "A viral post shows 2.1M likes in the app but the likes table has 2.42M rows. 15% undercount. Celebrity notices and tweets about it. PR nightmare.",
              cause: "Redis INCR works for normal load, but this post received 50K likes/minute. Redis counter was incremented correctly, but a bug in the periodic reconciliation job was capped at 10M likes per run — this post exceeded that and was never fully reconciled.",
              fix: "Remove reconciliation cap. Run reconciliation more frequently for viral posts (detect via engagement velocity). Add a separate 'exact count' query path that reads from DB on demand (for profile pages of viral posts). Alert when Redis counter and DB count diverge by more than 5%.",
              quote: "The reconciliation job had a LIMIT 10000000 that nobody questioned. One viral post exceeded 10M likes and the counter was never corrected. We now alert on any post where Redis and DB diverge by more than 5%." },
            { title: "Story Expiry Cleanup Fills Redis Memory", symptom: "Redis memory at 95%. Story keys for expired stories not being cleaned up. Redis starts evicting feed cache entries (LRU) to make room for dead story data.",
              cause: "Story fan-out updates story_tray:{user_id} with ZADD. When stories expire, the tray entry remains (author_id still in sorted set). 500M users × 50 tray entries = 25B stale entries over time.",
              fix: "Lazy cleanup: when reading story tray, check if each author has active stories. If not, ZREM the author from the tray (clean on read). Also: background sweeper job that removes stale tray entries. Added proper TTL to tray keys (48 hours — stories last 24h + buffer).",
              quote: "We set TTL on the story content but not on the story tray pointers. Over 3 months, 25 billion stale tray entries accumulated silently until Redis started evicting actual feed data." },
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
        { t: "Reels (Short Video)", d: "TikTok-style short-form video feed. Algorithm-driven discovery (not follow-based). Separate recommendation engine.", detail: "Video transcoding pipeline (H.264/H.265). Content embeddings for recommendation. Engagement-optimized ranking (completion rate > likes). Separate feed infrastructure from main feed.", effort: "Hard" },
        { t: "Explore / Discovery Page", d: "Recommend content from accounts the user doesn't follow. Based on interests, engagement history, and trending topics.", detail: "Content embeddings (image recognition + NLP). Collaborative filtering (users similar to you liked X). Engagement velocity signals (trending detection). Real-time feature store for ranking.", effort: "Hard" },
        { t: "Direct Messages", d: "1:1 and group messaging between users. Separate from the feed system. Requires WebSocket for real-time delivery.", detail: "Essentially a separate chat system (see WhatsApp design). Shared inbox model. End-to-end encryption (optional). Photo/video sharing via existing media pipeline.", effort: "Hard" },
        { t: "Story Highlights", d: "Allow users to pin expired stories to their profile permanently. Organized in named collections.", detail: "Simple: reference existing story media (prevent deletion after 24h if highlighted). Highlight = ordered list of story_ids. Renders using same story viewer UI but without TTL.", effort: "Easy" },
        { t: "Hashtag & Location Pages", d: "Aggregated feeds for hashtags (#sunset) and locations (Bali, Indonesia). Top posts + recent posts.", detail: "Hashtag index: inverted index of hashtag → post_ids. Location index: geospatial index (PostGIS or Elasticsearch geo queries). Ranked by engagement (top) or recency (recent).", effort: "Medium" },
        { t: "Activity Feed (Notifications)", d: "Real-time feed of interactions: likes, comments, follows, mentions. Aggregated intelligently.", detail: "Separate fan-out from main feed. Notification aggregation: 'Alice, Bob, and 3 others liked your photo'. Push notifications with batching (max 1 push per 5 minutes). In-app badge counter.", effort: "Medium" },
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
    { q:"Why hybrid fan-out instead of pure fan-out-on-write?", a:"A celebrity with 100M followers posting once would trigger 100M Redis writes. At 1,160 posts/sec, if even 1% are celebrities, that's 1.16B fan-out writes/sec — impossible. The hybrid approach: fan-out-on-write for users with <10K followers (99% of users, cheap per-post), fan-out-on-read for celebrities (1% of users, their posts are queried at feed-read time and merged). This keeps write amplification manageable while keeping feed reads fast.", tags:["design"] },
    { q:"How do you rank the feed?", a:"ML model that predicts P(engagement) for each candidate post. Key features: (1) Relationship strength — how often user interacts with author. (2) Post age — recency decay. (3) Content affinity — similarity between user's interest embedding and post's content embedding. (4) Engagement velocity — how fast the post is getting likes. Model outputs: P(like), P(comment), P(save), P(share), weighted and combined into a single score. Top 20 posts returned. Diversity injection prevents same-author clustering.", tags:["algorithm"] },
    { q:"How does the media upload pipeline work?", a:"Three-phase: (1) Client requests pre-signed S3 URL → uploads directly to S3 (bypasses our servers). (2) S3 event → Kafka → media processor worker resizes to 4 variants (150px, 320px, 640px, 1080px), strips EXIF, generates JPEG + WebP. (3) Variants stored in S3, auto-cached by CDN on first request. Total processing time: 2-5 seconds. The post creation API is called AFTER media is ready — so 'Share' is instant.", tags:["design"] },
    { q:"How do Stories differ from posts architecturally?", a:"Stories are ephemeral (24h TTL), viewed full-screen in sequence, and have separate fan-out. Storage: Redis sorted set with TTL (auto-expiry). Fan-out is lighter: only update the story tray position (which users have stories), not inject into the full feed. View tracking is append-only (SADD, not a counter). Separate from the main feed cache. After 24h: Redis key expires, media lifecycle policy archives, DB record kept for analytics/Highlights.", tags:["design"] },
    { q:"How do you handle the 'cold start' problem for new users?", a:"New user has no followers and follows no one. Their feed cache is empty. Three approaches: (1) Onboarding: suggest popular/relevant accounts to follow immediately. (2) Explore-based feed: show algorithmically recommended content from the Explore engine until the user builds a follow graph. (3) Bootstrap from contacts: if the user shares their phone contacts, show posts from people they already know. Goal: user sees interesting content within 30 seconds of signup.", tags:["design"] },
    { q:"What database would you use for the social graph?", a:"At Instagram/Meta scale: TAO — a custom graph store built on MySQL. It's optimized for two queries: 'who does X follow?' and 'who follows X?' with sub-ms latency on billions of edges. For an interview: sharded PostgreSQL with two tables — follows(follower_id, followee_id) sharded by follower_id (for 'who do I follow?' — feed build) and a reverse-index table sharded by followee_id (for 'who follows me?' — profile page). Not a general-purpose graph DB like Neo4j — those don't scale to billions of edges.", tags:["data"] },
    { q:"How do you handle deleting a post after fan-out?", a:"Post deletion publishes a deletion event to Kafka. A separate cleanup worker removes the post_id from all followers' feed caches (LREM feed:{user_id} 0 post_id for each follower). This is an expensive operation (same cost as fan-out). In the interim: the feed hydration step checks if each post_id still exists — deleted posts are filtered out at read time (tombstone check). Lazy cleanup + eager filtering = user never sees a deleted post.", tags:["reliability"] },
    { q:"How do you serve images so fast globally?", a:"CDN (CloudFront/Akamai) with edge caching. Origin is S3. First request for an image hits S3 → CDN caches it at the nearest edge. Subsequent requests served from edge in <50ms. Popular images are cached at every edge worldwide within minutes. Four image variants (150px to 1080px): client requests the smallest that fills its viewport. WebP for supported clients (25-35% smaller than JPEG). Result: <100ms image load for 95%+ of requests.", tags:["scalability"] },
    { q:"How do you count likes accurately at this scale?", a:"Two-layer approach: (1) Redis INCR for real-time counter (atomic, sub-ms). This is what users see. (2) DB likes table as source of truth (UPSERT per like). Periodic reconciliation: compare Redis counter with SELECT COUNT(*) from likes. Fix drift. Accept ±1% real-time accuracy. For viral posts (>1M likes), reconcile every 5 minutes instead of hourly. Display approximate counts to reduce precision pressure: '4.5M likes' not '4,521,339 likes'.", tags:["data"] },
    { q:"What if Redis (feed cache) goes down?", a:"Full Redis outage: fall back to fan-out-on-read for ALL users. Every feed load queries recent posts from all followed accounts, merges, ranks, and returns. This is ~10× slower (500ms vs 50ms) but functional. The system degrades — it doesn't fail. To prevent full outage: Redis Cluster with 3 replicas per shard. Individual node failure → Sentinel promotes replica in <10 seconds. Full cluster loss → restore from RDB snapshot + rebuild from Post DB. Rolling restarts for upgrades (never restart the whole cluster).", tags:["availability"] },
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

export default function InstagramSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Instagram</h1>
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