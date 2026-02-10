import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   NOTIFICATION SYSTEM — System Design Reference
   Pearl white theme · Reusable section structure
   Scale: Billions of notifications per day
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",             icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",         icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",  icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",           icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",    icon: "🏗️", color: "#9333ea" },
  { id: "algorithm",     label: "Routing & Priority",   icon: "⚙️", color: "#c026d3" },
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
            <Label>What is a Notification System?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A notification system is a centralized platform that sends messages to users across multiple channels — push notifications (iOS/Android), SMS, email, and in-app. It receives notification requests from upstream services, applies user preferences and rate limits, selects the right channel, renders templates, dispatches to third-party providers, and tracks delivery status.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Think of it like a postal service for your app: different services hand over "letters" (notification events). The system decides how to deliver each one — by phone (push), mailbox (email), text (SMS), or in-person (in-app) — based on urgency, user preferences, and what's most likely to get read.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Do We Need It?</Label>
            <ul className="space-y-2.5">
              <Point icon="🔔" color="#0891b2">User engagement — bring users back with timely, relevant notifications</Point>
              <Point icon="🔀" color="#0891b2">Multi-channel — one event can fan out to push + email + SMS simultaneously</Point>
              <Point icon="⚖️" color="#0891b2">Centralized control — rate limiting, dedup, and preferences in one place, not scattered across services</Point>
              <Point icon="📈" color="#0891b2">Analytics — unified tracking of open rates, CTR, and delivery across all channels</Point>
              <Point icon="🛡️" color="#0891b2">Compliance — GDPR opt-out, CAN-SPAM, quiet hours, and per-user preferences enforced consistently</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Scale</Label>
            <div className="space-y-2.5">
              {[
                { co: "Facebook", rule: "~10B push/day to 2B+ devices", algo: "Multi-channel, ML ranking" },
                { co: "Amazon", rule: "Billions of order/promo notifications", algo: "SES + SNS + Pinpoint" },
                { co: "Uber", rule: "Millions of real-time trip updates", algo: "Priority-based, sub-second" },
                { co: "Twitter/X", rule: "Millions of mentions/likes per minute", algo: "Batching + digest" },
                { co: "WhatsApp", rule: "100B+ messages/day (incl. notifs)", algo: "XMPP push, end-to-end" },
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
            <Label color="#2563eb">Channels Overview</Label>
            <svg viewBox="0 0 360 200" className="w-full">
              <DiagramBox x={180} y={35} w={110} h={38} label="Notification\nService" color="#9333ea"/>
              <DiagramBox x={55} y={120} w={75} h={38} label="Push\n(APNs/FCM)" color="#2563eb"/>
              <DiagramBox x={165} y={120} w={60} h={38} label="Email\n(SES)" color="#059669"/>
              <DiagramBox x={260} y={120} w={55} h={38} label="SMS\n(Twilio)" color="#d97706"/>
              <DiagramBox x={345} y={120} w={55} h={38} label="In-App" color="#c026d3"/>
              <Arrow x1={145} y1={54} x2={75} y2={101} id="ch1"/>
              <Arrow x1={172} y1={54} x2={160} y2={101} label="" id="ch2"/>
              <Arrow x1={195} y1={54} x2={255} y2={101} label="" id="ch3"/>
              <Arrow x1={215} y1={54} x2={335} y2={101} label="" id="ch4"/>
              <rect x={25} y={165} width={330} height={22} rx={4} fill="#6366f108" stroke="#6366f130"/>
              <text x={190} y={178} textAnchor="middle" fill="#6366f1" fontSize="8" fontFamily="monospace">each channel: separate provider, retry logic, rate limits, delivery tracking</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Meta, Amazon, Google, Uber, LinkedIn, Twitter, Apple</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Scope the Channels</div>
            <p className="text-[12px] text-stone-500 mt-0.5">Immediately clarify which channels to support. "Notification system" can mean just push, or push + email + SMS + in-app + webhooks. For a 45-min interview, focus on <strong>push + email + SMS</strong> with the core pipeline. In-app and webhooks can be follow-ups. Always ask about scale — "millions" vs "billions" radically changes the design.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Send notifications via push (iOS/Android), email, SMS, and in-app</Point>
            <Point icon="2." color="#059669">Support both transactional (order confirmed) and bulk/marketing (weekly digest) notifications</Point>
            <Point icon="3." color="#059669">Users can set per-channel and per-category notification preferences (opt-in/opt-out)</Point>
            <Point icon="4." color="#059669">Support templated messages with dynamic variable substitution (user name, order ID, etc.)</Point>
            <Point icon="5." color="#059669">Track delivery status: sent → delivered → opened → clicked</Point>
            <Point icon="6." color="#059669">Rate limit notifications per user (no more than N per hour across all channels)</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Massive scale — handle 1B+ notifications per day (11.5K/sec avg, 50K/sec peak)</Point>
            <Point icon="2." color="#dc2626">Low latency for transactional — order confirmations delivered in &lt;5 seconds end-to-end</Point>
            <Point icon="3." color="#dc2626">At-least-once delivery — never silently drop a notification (duplicates are OK, loss is not)</Point>
            <Point icon="4." color="#dc2626">Highly available — 99.99% uptime for notification ingestion; channel delivery depends on providers</Point>
            <Point icon="5." color="#dc2626">Exactly-once semantics where possible — deduplicate retries to avoid spamming users</Point>
            <Point icon="6." color="#dc2626">Extensible — adding a new channel (e.g., WhatsApp, Slack webhook) should be a plugin, not a rewrite</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask the Interviewer</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Which channels? Push, email, SMS, in-app, webhooks?",
            "What's the daily volume? Millions or billions?",
            "Transactional only, or also marketing/bulk?",
            "Do we need real-time analytics or batched?",
            "How important is ordering? Strict per-user order?",
            "Do we need quiet hours / do-not-disturb?",
            "Multi-region or single region?",
            "Do we own the providers or use third-party (APNs, SES, Twilio)?",
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
            <div className="text-[12px] font-bold text-violet-700">Interview Tip — Billions Is Different</div>
            <p className="text-[12px] text-stone-500 mt-0.5">At billions/day, the bottleneck is not your API — it's the third-party providers (APNs, FCM, SES). You must design for provider rate limits, batch APIs, connection pooling, and graceful degradation when providers throttle you. State this upfront to show you understand the real constraints.</p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Traffic Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Users = 500M registered, 200M DAU" result="500M / 200M" note='Assumption — ask interviewer: "Let me assume a large social/e-commerce platform"' />
            <MathStep step="2" formula="Notifications per user per day = ~5 avg" result="5/user/day" note="Mix: 2 push, 1 email, 1 in-app, 0.5 SMS (transactional)" />
            <MathStep step="3" formula="Total notifications/day = 200M × 5" result="1 Billion/day" note="1 × 10⁹ notifications per day across all channels" final />
            <MathStep step="4" formula="Avg notifications/sec = 1B / 86,400" result="~11.5K/sec" note="Sustained throughput. Consistent load for pipeline sizing." />
            <MathStep step="5" formula="Peak = Avg × 5 (flash sales, breaking news)" result="~58K/sec" note="Black Friday, election night, major sporting events." final />
          </div>
        </Card>

        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Per-Channel Breakdown</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Push (iOS + Android) = 40% of 1B" result="400M/day" note="~4,600/sec. APNs: HTTP/2 multiplexed. FCM: batch API (500/req)." />
            <MathStep step="2" formula="Email = 25% of 1B" result="250M/day" note="~2,900/sec. SES limit: 50K/sec (adjustable). Cheapest channel." />
            <MathStep step="3" formula="In-App = 25% of 1B" result="250M/day" note="~2,900/sec. Stored in DB, pulled on app open. No external provider." />
            <MathStep step="4" formula="SMS = 10% of 1B" result="100M/day" note="~1,150/sec. Most expensive ($0.01-0.05/msg). Only critical notifs." final />
            <MathStep step="5" formula="Total provider API calls/sec (peak)" result="~58K/sec" note="Sum of all channels at peak. Must handle provider throttling." final />
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Storage Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Notification record size (avg)" result="~500 B" note="id (16B) + user_id (8B) + channel (4B) + template_id (8B) + payload (200B) + status + timestamps + metadata" />
            <MathStep step="2" formula="Daily storage = 1B × 500 bytes" result="500 GB/day" note="5 × 10¹¹ bytes per day for notification records" />
            <MathStep step="3" formula="30-day retention (hot storage)" result="15 TB" note="Keep recent notifications queryable for user inbox/history" final />
            <MathStep step="4" formula="1-year archive (cold storage)" result="~180 TB" note="Archive to S3/Glacier for compliance. Query via Athena." final />
            <MathStep step="5" formula="Template storage (all templates)" result="~10 MB" note="~10K templates × 1KB each. Trivial — fits in memory cache." />
          </div>
        </Card>

        <Card accent="#059669">
          <Label color="#059669">Step 4 — Queue & Cost Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Kafka throughput = 58K msgs/sec peak" result="58K/sec" note="Single Kafka cluster handles 500K+ msgs/sec. Headroom is large." />
            <MathStep step="2" formula="Kafka partitions = 58K / 2K per partition" result="~30 partitions" note="2K msgs/sec per partition (conservative). 30 partitions across topics." />
            <MathStep step="3" formula="Push cost = 400M × $0 (APNs/FCM free)" result="~$0/day" note="Apple and Google don't charge for push. Infra cost only." />
            <MathStep step="4" formula="Email cost = 250M × $0.10/1000 (SES)" result="~$25K/day" note="$750K/month. Largest cost center." final />
            <MathStep step="5" formula="SMS cost = 100M × $0.01 (Twilio avg)" result="~$1M/day" note="$30M/month. SMS is BY FAR the most expensive. Minimize usage." final />
          </div>
          <div className="mt-4 pt-3 border-t border-stone-100">
            <div className="text-[11px] text-stone-500">
              <strong className="text-stone-700">Key point:</strong> At billion-scale, SMS cost dominates everything. This is why smart routing is critical — send SMS only when push/email fails or for critical alerts (2FA, fraud). Every 1% shift from SMS to push saves $300K/month.
            </div>
          </div>
        </Card>
      </div>

      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Avg Throughput", val: "~11.5K/s", sub: "Peak: ~58K/s" },
            { label: "Daily Volume", val: "1 Billion", sub: "Across 4 channels" },
            { label: "Hot Storage (30d)", val: "~15 TB", sub: "Notification records" },
            { label: "Monthly Cost", val: "~$31.5M", sub: "SMS: $30M, Email: $750K" },
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
          <Label color="#2563eb">Send Notification (Core API)</Label>
          <CodeBlock code={`# POST /api/v1/notifications
# Internal API — called by upstream services
{
  "event_type": "order_shipped",
  "user_id": "user_42",
  "priority": "high",              # critical | high | medium | low
  "channels": ["push", "email"],   # or ["auto"] for smart routing
  "idempotency_key": "order_789_shipped",
  "template_id": "tmpl_order_shipped",
  "template_vars": {
    "user_name": "Alice",
    "order_id": "ORD-789",
    "tracking_url": "https://track.example.com/789"
  },
  "metadata": {
    "source_service": "order-service",
    "correlation_id": "corr_abc123"
  }
}

# Response — 202 Accepted (async processing)
{
  "notification_id": "notif_xyz456",
  "status": "queued",
  "accepted_channels": ["push", "email"],
  "estimated_delivery": "< 5 seconds"
}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Batch / Bulk Send</Label>
          <CodeBlock code={`# POST /api/v1/notifications/batch
# For marketing campaigns, digests, announcements
{
  "campaign_id": "camp_weekly_digest_feb09",
  "template_id": "tmpl_weekly_digest",
  "channel": "email",
  "priority": "low",
  "targeting": {
    "segment_id": "seg_active_users_30d",
    # or explicit list:
    # "user_ids": ["user_1", "user_2", ...]
    "estimated_recipients": 50000000
  },
  "schedule": {
    "send_at": "2026-02-10T09:00:00Z",  # scheduled
    "timezone_aware": true,  # send at 9am local time
    "throttle_rate": 100000  # max sends per minute
  }
}

# Response — 202 Accepted
{
  "batch_id": "batch_abc789",
  "status": "scheduled",
  "estimated_recipients": 50000000,
  "estimated_completion": "2026-02-10T10:30:00Z"
}`} />
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#d97706">
          <Label color="#d97706">User Preferences & History</Label>
          <CodeBlock code={`# GET /api/v1/users/:user_id/preferences
{
  "user_id": "user_42",
  "global_enabled": true,
  "quiet_hours": {"start": "22:00", "end": "08:00", "tz": "US/Pacific"},
  "channels": {
    "push": {"enabled": true, "device_tokens": ["tok_ios_...", "tok_android_..."]},
    "email": {"enabled": true, "address": "alice@example.com"},
    "sms": {"enabled": true, "phone": "+14155551234"},
    "in_app": {"enabled": true}
  },
  "categories": {
    "order_updates": {"push": true, "email": true, "sms": false},
    "marketing": {"push": false, "email": true, "sms": false},
    "security": {"push": true, "email": true, "sms": true}
  }
}

# PUT /api/v1/users/:user_id/preferences
# Update preferences (partial update with merge)

# GET /api/v1/users/:user_id/notifications?cursor=...&limit=50
# Notification inbox / history (paginated)`} />
        </Card>
        <Card accent="#9333ea">
          <Label color="#9333ea">Design Decisions</Label>
          <div className="space-y-3">
            {[
              { header: "202 Accepted (Async)", desc: "Notifications are enqueued, not sent synchronously. Decouples API latency from provider latency.", ex: "Return immediately, process via Kafka" },
              { header: "Idempotency Keys", desc: "Prevent duplicate sends on retry. Same key → same notification, not a new one.", ex: '"order_789_shipped" → deduplicated' },
              { header: "Smart Channel Routing", desc: 'channels: ["auto"] lets the system pick the best channel based on user prefs, reachability, cost.', ex: "Has push token? → push. Otherwise → email." },
              { header: "Priority Queues", desc: "Critical (2FA codes) processed before marketing emails. Separate Kafka topics per priority.", ex: "critical: <1s, high: <5s, low: minutes" },
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
            <Label color="#d97706">Rate Limiting Per User</Label>
            <ul className="space-y-1.5">
              <Point icon="→" color="#d97706">Max 10 push notifications per hour per user (prevent spam)</Point>
              <Point icon="→" color="#d97706">Max 3 SMS per day per user (cost + user experience)</Point>
              <Point icon="→" color="#d97706">Critical/security bypasses rate limits (2FA codes, fraud alerts)</Point>
              <Point icon="→" color="#d97706">Rate limits enforced at the notification service, NOT at the caller</Point>
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
    { label: "Naive Approach", desc: "Upstream service directly calls APNs/FCM/SES for each notification. Every service implements its own retry logic, template rendering, and preference checks. Breaks: no centralized preferences, no dedup, N services × M channels = N×M integrations to maintain." },
    { label: "Centralized Service", desc: "Single notification service receives all requests. Checks preferences, renders templates, dispatches to providers. Better but monolithic — a slow email provider blocks push delivery. Single queue means high-priority 2FA waits behind marketing blasts." },
    { label: "Priority Queues", desc: "Separate Kafka topics for critical/high/medium/low priority. Each priority has its own consumer group with different scaling. Critical has dedicated fast-path workers. Marketing is throttled to avoid provider rate limits. Template rendering is extracted to a service." },
    { label: "Full Architecture", desc: "Complete: API → Validation & Dedup → Priority Router → Per-Channel Queues → Channel Workers (Push/Email/SMS/In-App) → Provider Adapters → Delivery Tracker → Analytics. User preferences cached in Redis. Templates cached in memory. Provider circuit breakers. Dead-letter queues for failed deliveries." },
  ];
  const diagrams = [
    () => (
      <svg viewBox="0 0 460 170" className="w-full">
        <DiagramBox x={55} y={45} w={75} h={36} label="Order Svc" color="#059669"/>
        <DiagramBox x={55} y={110} w={75} h={36} label="Auth Svc" color="#0891b2"/>
        <DiagramBox x={230} y={45} w={70} h={36} label="APNs" color="#2563eb"/>
        <DiagramBox x={340} y={45} w={70} h={36} label="FCM" color="#d97706"/>
        <DiagramBox x={230} y={110} w={70} h={36} label="SES" color="#059669"/>
        <DiagramBox x={340} y={110} w={70} h={36} label="Twilio" color="#dc2626"/>
        <Arrow x1={92} y1={40} x2={195} y2={40} id="n1"/>
        <Arrow x1={92} y1={50} x2={305} y2={50} id="n2"/>
        <Arrow x1={92} y1={105} x2={195} y2={105} id="n3"/>
        <Arrow x1={92} y1={115} x2={305} y2={115} id="n4"/>
        <rect x={125} y={145} width={230} height={20} rx={4} fill="#dc262608" stroke="#dc262630"/>
        <text x={240} y={157} textAnchor="middle" fill="#dc2626" fontSize="8" fontFamily="monospace">⌀ N services × M channels = N×M integrations</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 180" className="w-full">
        <DiagramBox x={55} y={45} w={72} h={32} label="Order Svc" color="#059669"/>
        <DiagramBox x={55} y={100} w={72} h={32} label="Auth Svc" color="#0891b2"/>
        <DiagramBox x={55} y={150} w={72} h={32} label="Promo Svc" color="#c026d3"/>
        <DiagramBox x={210} y={90} w={90} h={42} label="Notification\nService" color="#9333ea"/>
        <DiagramBox x={360} y={45} w={65} h={32} label="Push" color="#2563eb"/>
        <DiagramBox x={360} y={95} w={65} h={32} label="Email" color="#059669"/>
        <DiagramBox x={360} y={145} w={65} h={32} label="SMS" color="#dc2626"/>
        <Arrow x1={91} y1={50} x2={165} y2={85} id="cs1"/>
        <Arrow x1={91} y1={100} x2={165} y2={100} id="cs2"/>
        <Arrow x1={91} y1={145} x2={165} y2={110} id="cs3"/>
        <Arrow x1={255} y1={80} x2={328} y2={50} id="cs4"/>
        <Arrow x1={255} y1={95} x2={328} y2={95} id="cs5"/>
        <Arrow x1={255} y1={110} x2={328} y2={140} id="cs6"/>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 200" className="w-full">
        <DiagramBox x={45} y={90} w={55} h={32} label="Services" color="#64748b"/>
        <DiagramBox x={130} y={90} w={72} h={38} label="Notif\nAPI" color="#9333ea"/>
        <DiagramBox x={240} y={35} w={65} h={28} label="Critical Q" color="#dc2626"/>
        <DiagramBox x={240} y={75} w={65} h={28} label="High Q" color="#d97706"/>
        <DiagramBox x={240} y={115} w={65} h={28} label="Medium Q" color="#0891b2"/>
        <DiagramBox x={240} y={155} w={65} h={28} label="Low Q" color="#64748b"/>
        <DiagramBox x={360} y={35} w={72} h={28} label="Fast Workers" color="#dc2626"/>
        <DiagramBox x={360} y={75} w={72} h={28} label="Workers" color="#d97706"/>
        <DiagramBox x={360} y={115} w={72} h={28} label="Workers" color="#0891b2"/>
        <DiagramBox x={360} y={155} w={72} h={28} label="Throttled" color="#64748b"/>
        <Arrow x1={72} y1={90} x2={94} y2={90} id="pq0"/>
        <Arrow x1={166} y1={78} x2={208} y2={40} id="pq1"/>
        <Arrow x1={166} y1={85} x2={208} y2={78} id="pq2"/>
        <Arrow x1={166} y1={95} x2={208} y2={115} id="pq3"/>
        <Arrow x1={166} y1={102} x2={208} y2={152} id="pq4"/>
        <Arrow x1={272} y1={35} x2={324} y2={35} id="pq5"/>
        <Arrow x1={272} y1={75} x2={324} y2={75} id="pq6"/>
        <Arrow x1={272} y1={115} x2={324} y2={115} id="pq7"/>
        <Arrow x1={272} y1={155} x2={324} y2={155} id="pq8"/>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 235" className="w-full">
        <DiagramBox x={38} y={55} w={50} h={30} label="Callers" color="#64748b"/>
        <DiagramBox x={115} y={55} w={68} h={34} label="Notif API\n+ Dedup" color="#9333ea"/>
        <DiagramBox x={208} y={55} w={65} h={34} label="Priority\nRouter" color="#c026d3"/>
        <DiagramBox x={300} y={30} w={60} h={26} label="Push W" color="#2563eb"/>
        <DiagramBox x={300} y={62} w={60} h={26} label="Email W" color="#059669"/>
        <DiagramBox x={300} y={94} w={60} h={26} label="SMS W" color="#d97706"/>
        <DiagramBox x={395} y={30} w={55} h={26} label="APNs" color="#2563eb"/>
        <DiagramBox x={395} y={62} w={55} h={26} label="SES" color="#059669"/>
        <DiagramBox x={395} y={94} w={55} h={26} label="Twilio" color="#d97706"/>
        <DiagramBox x={115} y={150} w={60} h={30} label="Prefs\nCache" color="#dc2626"/>
        <DiagramBox x={208} y={150} w={60} h={30} label="Template\nSvc" color="#7c3aed"/>
        <DiagramBox x={300} y={150} w={68} h={30} label="Delivery\nTracker" color="#0284c7"/>
        <DiagramBox x={395} y={150} w={55} h={30} label="DLQ" color="#be123c"/>
        <Arrow x1={63} y1={55} x2={81} y2={55} id="fa0"/>
        <Arrow x1={149} y1={55} x2={176} y2={55} id="fa1"/>
        <Arrow x1={240} y1={45} x2={270} y2={33} id="fa2"/>
        <Arrow x1={240} y1={55} x2={270} y2={62} id="fa3"/>
        <Arrow x1={240} y1={65} x2={270} y2={90} id="fa4"/>
        <Arrow x1={330} y1={30} x2={368} y2={30} id="fa5"/>
        <Arrow x1={330} y1={62} x2={368} y2={62} id="fa6"/>
        <Arrow x1={330} y1={94} x2={368} y2={94} id="fa7"/>
        <Arrow x1={115} y1={72} x2={115} y2={135} label="check" id="fa8" dashed/>
        <Arrow x1={208} y1={72} x2={208} y2={135} label="render" id="fa9" dashed/>
        <Arrow x1={300} y1={107} x2={300} y2={135} label="status" id="fa10" dashed/>
        <Arrow x1={340} y1={150} x2={368} y2={150} label="failed" id="fa11" dashed/>
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
          <Label color="#059669">Notification Pipeline — Happy Path</Label>
          <div className="space-y-1.5">
            {[
              { s:"1", t:"Order Service calls POST /notifications {event: order_shipped, user_id: 42}", c:"text-blue-600" },
              { s:"2", t:"API validates payload, checks idempotency_key for dedup (Redis SET NX)", c:"text-purple-600" },
              { s:"3", t:"Fetch user preferences from Redis cache (push: ✓, email: ✓, sms: ✗)", c:"text-red-600" },
              { s:"4", t:"Render template: 'Hi {user_name}, your order {order_id} has shipped!'", c:"text-violet-600" },
              { s:"5", t:"Priority router publishes to Kafka: topic=notif.high, partition by user_id", c:"text-fuchsia-600" },
              { s:"6", t:"Push worker consumes → calls APNs HTTP/2 with device_token → 200 OK", c:"text-blue-600" },
              { s:"7", t:"Email worker consumes → calls SES SendEmail → 200 OK", c:"text-emerald-600" },
              { s:"8", t:"Delivery tracker updates status: sent → (wait for callback) → delivered", c:"text-sky-600" },
            ].map((s,i) => (
              <div key={i} className="flex items-start gap-2 text-[12px]">
                <span className="text-stone-300 font-mono w-4 shrink-0">{s.s}</span>
                <span className={s.c}>{s.t}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <Label color="#dc2626">What Makes This Hard at Billions Scale</Label>
          <div className="space-y-1.5">
            {[
              { s:"1", t:"Provider rate limits: APNs throttles at sustained high QPS; FCM batch API limited to 500/request", c:"text-red-600" },
              { s:"2", t:"Token invalidation: ~5% of device tokens are stale daily = 20M invalid tokens to handle", c:"text-red-600" },
              { s:"3", t:"Email deliverability: ISPs throttle by sender reputation. Warm up IPs over weeks.", c:"text-amber-600" },
              { s:"4", t:"SMS cost: at $0.01/msg, 100M SMS/day = $1M/day. Every unnecessary SMS is waste.", c:"text-amber-600" },
              { s:"5", t:"User fatigue: too many notifications → users disable all notifications permanently", c:"text-purple-600" },
              { s:"6", t:"Ordering: user gets 'order delivered' before 'order shipped' due to consumer lag", c:"text-purple-600" },
              { s:"7", t:"Thundering herd: 50M-user campaign launch at exactly 9am → provider meltdown", c:"text-red-600" },
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
  const [sel, setSel] = useState("smart_routing");
  const algos = {
    smart_routing: { name: "Smart Channel Routing ★", cx: "O(1) per notif",
      pros: ["Picks the cheapest effective channel automatically","Reduces SMS costs by 60-80% (push first, SMS fallback)","Adapts to user reachability (push token expired? → email)","Maximizes delivery rate across all channels"],
      cons: ["Requires rich user profile data (device tokens, email, phone)","Delayed delivery if cascading through fallback channels","Complex logic — many edge cases to handle"],
      when: "Always use smart routing for non-critical notifications. Let the system pick the best channel based on user preferences, device availability, and cost. For critical notifications (2FA), blast all enabled channels simultaneously.",
      code: `# Smart Channel Routing
def route_notification(user, notification):
    prefs = get_user_preferences(user.id)
    category = notification.event_type

    # Critical → blast all enabled channels
    if notification.priority == "critical":
        return [ch for ch in ["push","email","sms"]
                if prefs.is_enabled(ch, category)]

    # Smart cascade: cheapest first, fallback on failure
    if prefs.has_push_token() and prefs.is_enabled("push", category):
        return ["push"]  # cheapest, fastest
    elif prefs.is_enabled("email", category):
        return ["email"]  # cheap, reliable
    elif prefs.is_enabled("sms", category):
        return ["sms"]    # expensive, last resort
    else:
        return ["in_app"]  # always available` },
    priority_queues: { name: "Priority Queue System", cx: "O(1) enqueue/dequeue",
      pros: ["Critical notifications (2FA) never wait behind marketing blasts","Each priority level scales independently","Low-priority can be throttled without affecting high-priority","Simple to reason about and operate"],
      cons: ["Starvation: low-priority may be delayed indefinitely during spikes","More Kafka topics/consumers to manage","Priority assignment is a policy decision (who decides what's 'high'?)"],
      when: "Essential at billions scale. Without priority queues, a 50M-user marketing campaign blocks real-time 2FA codes. Critical should have dedicated fast-path workers with guaranteed capacity.",
      code: `# Priority Queue Configuration
PRIORITY_TOPICS = {
    "critical": {  # 2FA, fraud alerts, password reset
        "topic": "notif.critical",
        "partitions": 12,
        "consumers": 20,        # over-provisioned
        "max_latency_ms": 1000, # SLA: <1 second
        "rate_limit": None,     # no throttling
    },
    "high": {      # order updates, payment confirmations
        "topic": "notif.high",
        "partitions": 24,
        "consumers": 40,
        "max_latency_ms": 5000, # SLA: <5 seconds
        "rate_limit": None,
    },
    "medium": {    # social notifications, comments
        "topic": "notif.medium",
        "partitions": 30,
        "consumers": 50,
        "max_latency_ms": 30000, # SLA: <30 seconds
        "rate_limit": 20000,     # 20K/sec max to providers
    },
    "low": {       # marketing, digests, recommendations
        "topic": "notif.low",
        "partitions": 20,
        "consumers": 30,
        "max_latency_ms": 300000, # SLA: <5 minutes
        "rate_limit": 5000,       # 5K/sec, heavily throttled
    },
}` },
    dedup_and_batching: { name: "Dedup + Digest Batching", cx: "O(1) per check",
      pros: ["Prevents duplicate notifications on retry (at-least-once → effectively-once)","Digest batching: '3 new likes' instead of 3 separate notifications","Dramatically reduces user notification fatigue","Saves provider costs (fewer API calls)"],
      cons: ["Digest adds latency (wait for batch window to close)","Dedup requires centralized state (Redis) with TTL","Tricky to get right: what's 'duplicate' vs 'legitimate repeat'?"],
      when: "Always implement dedup for transactional notifications (idempotency_key). Digest batching for high-frequency social events (likes, comments, followers). Don't batch transactional notifications.",
      code: `# Deduplication with Redis
def is_duplicate(idempotency_key, ttl=3600):
    # SET NX: returns True only if key is new
    is_new = redis.set(
        f"dedup:{idempotency_key}",
        "1",
        nx=True,      # only set if not exists
        ex=ttl         # expire after 1 hour
    )
    return not is_new  # True if duplicate

# Digest Batching
def maybe_batch(user_id, event_type, event):
    key = f"digest:{user_id}:{event_type}"
    redis.rpush(key, json.dumps(event))
    count = redis.llen(key)

    if count == 1:
        # First event: schedule digest flush in 5 min
        schedule_flush(user_id, event_type, delay=300)
    # When flush fires:
    # "Alice, Bob, and 3 others liked your post"` },
    rate_limiting: { name: "Per-User Rate Limiting", cx: "O(1) per check",
      pros: ["Prevents notification spam — protects user experience","Per-channel limits (10 push/hr, 3 SMS/day)","Critical notifications bypass limits","Reduces costs by preventing runaway notifications"],
      cons: ["Dropped notifications need to be handled (queue for later? drop silently?)","Rate limits must account for batch notifications","Edge case: user has rate limit hit, then critical 2FA arrives — must bypass"],
      when: "Essential for user experience. Without rate limiting, a buggy upstream service can send 1000 notifications to a user in a minute. Always have per-user, per-channel, and per-category rate limits.",
      code: `# Per-User Rate Limiting (Sliding Window)
RATE_LIMITS = {
    "push":  {"window": 3600, "max": 10},  # 10/hour
    "email": {"window": 86400, "max": 5},   # 5/day
    "sms":   {"window": 86400, "max": 3},   # 3/day
}

def check_rate_limit(user_id, channel, priority):
    # Critical bypasses rate limits
    if priority == "critical":
        return True

    limit = RATE_LIMITS[channel]
    key = f"rate:{user_id}:{channel}"
    now = time.time()

    # Sliding window counter in Redis
    redis.zremrangebyscore(key, 0, now - limit["window"])
    count = redis.zcard(key)

    if count >= limit["max"]:
        return False  # rate limited

    redis.zadd(key, {str(now): now})
    redis.expire(key, limit["window"])
    return True` },
  };
  const a = algos[sel];
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Routing & Priority — Algorithm Comparison</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[12px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Algorithm</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Purpose</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Latency Impact</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Cost Impact</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Best For</th>
            </tr></thead>
            <tbody>
              {[
                { n:"Smart Routing ★", p:"Channel selection", l:"< 1ms", c:"Saves 60-80% SMS", f:"All notifications", hl:true },
                { n:"Priority Queues", p:"Processing order", l:"Prevents delays", c:"N/A", f:"Mixed workloads", hl:false },
                { n:"Dedup + Batching", p:"Spam prevention", l:"+5min (digests)", c:"Fewer API calls", f:"Social/high-freq", hl:false },
                { n:"Rate Limiting", p:"User protection", l:"< 1ms", c:"Prevents waste", f:"Always", hl:false },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-purple-50" : i%2?"bg-stone-50/50":""}>
                  <td className={`px-3 py-2 font-mono ${r.hl?"text-purple-700 font-bold":"text-stone-600"}`}>{r.n}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.p}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.l}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.c}</td>
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
          <Label color="#dc2626">Notifications Table</Label>
          <CodeBlock code={`# notifications (Cassandra — write-optimized, time-series)
CREATE TABLE notifications (
  user_id       BIGINT,
  notification_id TIMEUUID,       -- time-ordered UUID
  event_type    TEXT,              -- "order_shipped", "new_follower"
  channel       TEXT,              -- "push", "email", "sms", "in_app"
  priority      TEXT,              -- "critical", "high", "medium", "low"
  title         TEXT,
  body          TEXT,              -- rendered content
  status        TEXT,              -- queued|sent|delivered|opened|failed
  provider_id   TEXT,              -- APNs message_id, SES message_id
  error_code    TEXT,              -- provider error on failure
  created_at    TIMESTAMP,
  sent_at       TIMESTAMP,
  delivered_at  TIMESTAMP,
  opened_at     TIMESTAMP,
  metadata      MAP<TEXT, TEXT>,   -- correlation_id, source_service
  PRIMARY KEY ((user_id), notification_id)
) WITH CLUSTERING ORDER BY (notification_id DESC)
  AND default_time_to_live = 2592000;  -- 30-day TTL

-- Partition by user_id: all user's notifs on same node
-- Clustered by time: latest notifications first
-- TTL auto-expires old notifications`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">User Preferences & Device Tokens</Label>
          <CodeBlock code={`# user_preferences (PostgreSQL — read-heavy, small)
CREATE TABLE user_preferences (
  user_id       BIGINT PRIMARY KEY,
  global_enabled BOOLEAN DEFAULT TRUE,
  quiet_start   TIME,               -- "22:00"
  quiet_end     TIME,               -- "08:00"
  timezone      TEXT DEFAULT 'UTC',
  push_enabled  BOOLEAN DEFAULT TRUE,
  email_enabled BOOLEAN DEFAULT TRUE,
  sms_enabled   BOOLEAN DEFAULT TRUE,
  category_prefs JSONB,             -- per-category overrides
  updated_at    TIMESTAMP DEFAULT NOW()
);

# device_tokens (PostgreSQL — 1:many per user)
CREATE TABLE device_tokens (
  token_id      BIGSERIAL PRIMARY KEY,
  user_id       BIGINT NOT NULL,
  platform      TEXT NOT NULL,       -- "ios", "android", "web"
  token         TEXT NOT NULL UNIQUE, -- APNs/FCM token
  is_active     BOOLEAN DEFAULT TRUE,
  last_used_at  TIMESTAMP,
  created_at    TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_tokens_user ON device_tokens(user_id)
  WHERE is_active = TRUE;

# templates (PostgreSQL — small, cached in memory)
CREATE TABLE templates (
  template_id   TEXT PRIMARY KEY,    -- "tmpl_order_shipped"
  channel       TEXT NOT NULL,
  subject       TEXT,                -- email subject line
  body_template TEXT NOT NULL,       -- "Hi {{user_name}}, ..."
  version       INT DEFAULT 1,
  updated_at    TIMESTAMP
);`} />
        </Card>
      </div>
      <Card accent="#9333ea">
        <Label color="#9333ea">Storage Architecture — Why Multiple Databases</Label>
        <p className="text-[12px] text-stone-500 mb-4">Different data has different access patterns. Using one database for everything is a common mistake at this scale.</p>
        <div className="grid grid-cols-3 gap-4">
          {[
            { t: "Cassandra — Notification History", d: "1B writes/day, append-only, time-series. Queried by user_id (partition key) + time range. 30-day TTL auto-cleans. No updates after write.", pros: ["Handles 1B writes/day easily","Time-ordered per user","Auto-expiry with TTL","Linear horizontal scaling"], cons: ["No ad-hoc queries","No joins"], pick: true },
            { t: "PostgreSQL — User Preferences", d: "500M rows, read-heavy, rarely updated. Cached in Redis with 5-min TTL. Updated via user settings page (low write QPS).", pros: ["Rich queries for admin","Strong consistency","Small dataset (~50 GB)"], cons: ["Can't handle notification write volume","Single-region limits"], pick: false },
            { t: "Redis — Hot State Cache", d: "User prefs cache (5-min TTL), dedup keys (1-hr TTL), rate limit counters (sliding window), device token cache.", pros: ["Sub-ms reads","Perfect for per-request lookups","Handles 58K lookups/sec easily"], cons: ["Not durable","Memory-bound (~100 GB for 500M user caches)"], pick: false },
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
      </Card>
    </div>
  );
}

function ScalabilitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Scaling the Pipeline (Kafka)</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Partition by user_id</strong> — ensures per-user ordering. All notifications for user_42 go to the same partition → processed in order.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Separate topics per priority</strong> — critical topic with 12 partitions + 20 consumers. Low topic with 20 partitions but throttled consumers. Independent scaling.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Consumer auto-scaling</strong> — Kubernetes HPA based on Kafka consumer lag. Lag {'>'} 10K messages → scale up. Lag {'<'} 100 → scale down.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Backpressure handling</strong> — if providers throttle, slow consumer commit offsets. Messages stay in Kafka (days of retention). No data loss.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Scaling Provider Connections</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">APNs: HTTP/2 multiplexing</strong> — single connection handles thousands of concurrent pushes. Pool of 20 connections per worker. Token-based auth (no cert per connection).</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">FCM: batch API</strong> — send up to 500 messages per HTTP request. Batch worker accumulates messages for 50ms, then fires batch. 10× fewer API calls.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">SES: connection pooling</strong> — SES SMTP or API with connection reuse. Warm up sending IPs gradually (1K/day → 10K → 100K → 1M). ISP reputation management.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">SMS: multi-provider</strong> — distribute across Twilio, Vonage, Plivo for redundancy and cost optimization. Route by country for best rates.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Handling Campaign Blasts (50M users at once)</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Throttled Ingestion", d:"Campaign controller feeds users into Kafka at controlled rate (100K/min). Prevents thundering herd. Uses cursor-based iteration over user segments.", pros:["Protects providers from spike","Predictable load on pipeline","Can pause/resume campaigns"], cons:["50M users takes ~8 hours at 100K/min","Requires scheduling system"], pick:false },
            { t:"Timezone-Aware Scheduling ★", d:"Send at 9am local time for each user. Segment users by timezone. Process each timezone cohort at the right time. Spreads load naturally over 24 hours.", pros:["Better open rates (users awake)","Naturally spreads load across hours","Reduces peak by 4-6×"], cons:["Complex scheduling logic","Users in rare timezones may be delayed"], pick:true },
            { t:"Provider Sharding", d:"Distribute sends across multiple provider accounts/regions. SES: 5 sending domains, 20 IPs. APNs: multiple topic bundles. Avoids single-provider throttle.", pros:["Higher aggregate throughput","Provider failure isolation","Per-provider rate limits don't stack"], cons:["More accounts to manage","Split analytics across providers"], pick:false },
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
        <Label color="#d97706">Critical Decision: What Happens When a Provider Goes Down?</Label>
        <p className="text-[12px] text-stone-500 mb-4">APNs, SES, and Twilio are external dependencies you don't control. Provider outages WILL happen. Your design must handle this gracefully.</p>
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Ingestion Must Never Go Down</div>
            <p className="text-[11px] text-stone-500 mb-2">Always accept and queue notifications, even if providers are down.</p>
            <ul className="space-y-1"><Point icon="✓" color="#059669">Kafka buffers messages for hours/days</Point><Point icon="✓" color="#059669">API always returns 202 Accepted</Point><Point icon="✓" color="#059669">Provider failures don't cascade to callers</Point></ul>
          </div>
          <div className="rounded-lg border border-amber-200 bg-white p-4">
            <div className="text-[11px] font-bold text-amber-700 mb-1.5">Channel Failover</div>
            <p className="text-[11px] text-stone-500 mb-2">If push provider is down, fall back to email. If email is down, fall back to SMS.</p>
            <ul className="space-y-1"><Point icon="✓" color="#059669">Circuit breaker per provider (trip after 5 failures)</Point><Point icon="✓" color="#059669">Automatic channel escalation on failure</Point><Point icon="⚠" color="#d97706">SMS fallback is expensive — alert on-call when triggered</Point></ul>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#2563eb">Provider Circuit Breakers</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#2563eb"><strong className="text-stone-700">Per-provider breaker</strong> — APNs, FCM, SES, Twilio each have independent circuit breakers. One provider failing doesn't affect others.</Point>
            <Point icon="2." color="#2563eb"><strong className="text-stone-700">Threshold</strong> — 5 consecutive failures or error rate > 10% in 30s → OPEN. Stop sending to that provider.</Point>
            <Point icon="3." color="#2563eb"><strong className="text-stone-700">Half-open probe</strong> — after 60s cooldown, send 1 test notification. Success → CLOSED. Failure → OPEN again.</Point>
            <Point icon="4." color="#2563eb"><strong className="text-stone-700">Failback to queue</strong> — when breaker opens, messages go to a retry queue (Kafka DLQ). Replayed when provider recovers.</Point>
          </ul>
        </Card>
        <Card>
          <Label color="#0891b2">Retry Strategy</Label>
          <ul className="space-y-2.5">
            <Point icon="🔄" color="#0891b2"><strong className="text-stone-700">Exponential backoff</strong> — 1s → 2s → 4s → 8s → 16s → give up. Max 5 retries per notification per channel.</Point>
            <Point icon="⏱" color="#0891b2"><strong className="text-stone-700">Jitter</strong> — add random 0-500ms to each retry. Prevents thundering herd when provider recovers and all workers retry simultaneously.</Point>
            <Point icon="📊" color="#0891b2"><strong className="text-stone-700">Dead letter queue</strong> — after max retries, move to DLQ. Alert on DLQ depth. Manual review or bulk retry.</Point>
            <Point icon="🔌" color="#0891b2"><strong className="text-stone-700">Channel escalation</strong> — push failed 3 times? Try email. Email bounced? Try SMS (if critical). Log escalation chain for analytics.</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#9333ea">Degradation Ladder</Label>
        <div className="flex gap-3 items-stretch mt-1">
          {[
            { label: "All Providers Up", sub: "Full multi-channel", color: "#059669", status: "HEALTHY" },
            { label: "One Provider Down", sub: "Channel failover active", color: "#d97706", status: "DEGRADED" },
            { label: "Multiple Providers", sub: "DLQ filling, alerts firing", color: "#ea580c", status: "IMPAIRED" },
            { label: "Queue-Only Mode", sub: "Accept & queue, no delivery", color: "#dc2626", status: "EMERGENCY" },
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
              { metric: "notif.ingested", type: "Counter", desc: "Notifications accepted by API (by priority, channel)" },
              { metric: "notif.sent", type: "Counter", desc: "Successfully sent to provider (by channel, provider)" },
              { metric: "notif.delivered", type: "Counter", desc: "Confirmed delivered (push: APNs ack, email: not bounced)" },
              { metric: "notif.failed", type: "Counter", desc: "Failed after max retries (by channel, error_code)" },
              { metric: "notif.e2e_latency_ms", type: "Histogram", desc: "Ingestion → sent to provider. p50, p95, p99." },
              { metric: "provider.error_rate", type: "Gauge", desc: "Per-provider error %. Alert if > 5%. Trip circuit breaker at 10%." },
              { metric: "kafka.consumer_lag", type: "Gauge", desc: "Messages behind per topic. Alert if critical lag > 100." },
              { metric: "dlq.depth", type: "Gauge", desc: "Dead letter queue size. Should be near 0. Alert if > 1000." },
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
              { name: "Critical Queue Lag", rule: "consumer_lag(critical) > 100 for 30s", sev: "P1", action: "2FA codes delayed. Scale critical consumers immediately." },
              { name: "Provider Circuit Open", rule: "circuit_breaker(any) == OPEN", sev: "P1", action: "Check provider status page. Verify failover active. Monitor DLQ." },
              { name: "High E2E Latency", rule: "e2e_p99 > 30s for 5min", sev: "P2", action: "Check Kafka lag, provider latency, worker pod health." },
              { name: "DLQ Growing", rule: "dlq.depth > 10,000", sev: "P2", action: "Messages permanently failing. Check error codes, provider status." },
              { name: "SMS Cost Spike", rule: "sms.sent > 2× daily avg", sev: "P3", action: "Check: unexpected campaign? Smart routing bypassed? Fallback storm?" },
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
              { q: "User says 'I didn't get my notification'", steps: "Trace by user_id: was it ingested? Check prefs (opted out?). Check rate limit (hit?). Check Kafka offset. Check provider response. Check device token validity." },
              { q: "Push delivery rate drops below 80%", steps: "Check: stale device tokens? APNs returning 410 (unregistered)? Token refresh pipeline broken? Users uninstalled app?" },
              { q: "Email going to spam folder", steps: "Check: SPF/DKIM/DMARC configured? IP reputation on Sender Score? Content triggering spam filters? Warm-up new IPs gradually." },
              { q: "Campaign taking 10× longer than expected", steps: "Check: provider throttling? Consumer lag? Throttle rate too low? Workers OOMing? Kafka partition imbalance?" },
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
          { title: "Thundering Herd on Provider Recovery", sev: "Critical", sevColor: "#dc2626",
            desc: "APNs goes down for 10 minutes. 3M push notifications queue up in Kafka. APNs comes back. All workers resume simultaneously → 3M requests in seconds → APNs throttles or crashes again.",
            fix: "Ramp-up on recovery: when circuit breaker transitions from OPEN → HALF-OPEN → CLOSED, start at 10% throughput and linearly increase over 5 minutes. Use token bucket rate limiter on provider connections. Jitter on retry timestamps prevents synchronized bursts.",
            code: `Provider down 10 min → 3M queued\nProvider recovers → all workers resume\n→ 3M requests in 5 seconds\n→ Provider crashes again!\n\nFix: Ramp-up on recovery\n→ CLOSED state: start at 10% throughput\n→ Increase 10% every 30 seconds\n→ Full throughput after 5 minutes\n→ Token bucket on provider connection pool` },
          { title: "Stale Device Tokens (Silent Failure)", sev: "Critical", sevColor: "#dc2626",
            desc: "Users reinstall app → new device token. Old token becomes invalid. APNs returns 410 (Unregistered). If we keep sending to stale tokens, Apple throttles us. At 500M users, ~5% tokens go stale daily = 25M invalid tokens.",
            fix: "Process APNs feedback service responses. On 410: mark token as inactive in DB immediately. Run nightly cleanup job. Track token freshness metric. Alert if invalid token rate > 10%. Force token refresh on app open.",
            code: `Daily token churn: 500M × 5% = 25M stale\nAPNs 410 response: "this token is dead"\n\nIf we ignore: Apple throttles our account\n→ ALL push delivery degrades\n\nFix: Token lifecycle management\n→ APNs 410 → DELETE token immediately\n→ App open → re-register token\n→ Nightly: purge tokens not seen in 30 days\n→ Metric: invalid_token_rate < 5%` },
          { title: "Out-of-Order Delivery", sev: "Medium", sevColor: "#d97706",
            desc: "User gets 'Order Delivered' notification before 'Order Shipped'. Kafka partition assignment changes during rebalance. Or push arrives before email due to channel latency differences.",
            fix: "Partition Kafka by user_id — guarantees per-user ordering within a partition. For cross-channel ordering: embed sequence_num in notification. Client-side: sort by sequence_num before displaying. Accept that push and email have different latencies — ordering across channels is impossible to guarantee.",
            code: `Problem: "delivered" arrives before "shipped"\n\nCause 1: Kafka consumer rebalance\n→ Fix: partition by user_id (sticky)\n\nCause 2: push is faster than email\n→ Fix: can't solve cross-channel ordering\n→ Mitigation: sequence_num in payload\n→ Client sorts by sequence_num` },
          { title: "SMS Cost Explosion from Fallback Storm", sev: "Critical", sevColor: "#dc2626",
            desc: "APNs goes down. Smart routing falls back to SMS for 400M push notifications. At $0.01/SMS, that's $4M in a few hours. Your Twilio bill bankrupts the company.",
            fix: "SMS fallback must have its own rate limit and budget cap. Set daily SMS budget: $50K/day. When budget exhausted, fall back to email instead. Never auto-escalate to SMS for non-critical notifications. Alert immediately when SMS fallback activates at scale.",
            code: `APNs down → 400M push fail\n→ Smart routing: "try SMS next"\n→ 400M × $0.01 = $4,000,000\n\nFix: SMS budget cap\n→ daily_sms_budget = $50,000\n→ sms_spent > budget → STOP SMS fallback\n→ Fall back to email instead\n→ P1 alert: "SMS budget exhausted"\n→ Only critical (2FA) bypasses budget cap` },
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
            <p className="text-[12px] text-stone-500 mt-0.5">In an HLD you say "notification service" and draw one box. In production, that box is an API gateway, a preference service, a template engine, per-channel worker pools, provider adapters with circuit breakers, a delivery tracker, and an analytics pipeline — each independently scaled and deployed.</p>
          </div>
        </div>
      </Card>

      <Card accent="#0f766e">
        <Label color="#0f766e">Service Decomposition — What We Actually Build</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { name: "Notification API", owns: "Ingestion: validate, dedup, check prefs, route to priority queue. The front door for all notifications.", tech: "Go/Java behind ALB, horizontally scaled", api: "POST /api/v1/notifications → 202", scale: "Auto-scale on QPS. 20+ pods for 58K/sec peak.", stateful: false,
              modules: ["Payload Validator (schema, required fields)", "Idempotency Guard (Redis SET NX, 1hr TTL)", "Preference Checker (Redis-cached user prefs)", "Rate Limiter (per-user, per-channel sliding window)", "Priority Router (classify → Kafka topic)", "Template Renderer (Mustache/Handlebars, cached)"] },
            { name: "Channel Workers", owns: "Consume from Kafka, call provider APIs, handle retries and errors. One worker pool per channel.", tech: "Go workers, auto-scaled by Kafka lag", api: "Internal Kafka consumer → provider API", scale: "Push: 50 pods. Email: 30 pods. SMS: 10 pods.", stateful: false,
              modules: ["Kafka Consumer (manual offset commit, at-least-once)", "Provider Adapter (APNs HTTP/2, SES API, Twilio REST)", "Connection Pool Manager (HTTP/2 multiplex, keep-alive)", "Circuit Breaker (per-provider, 5 failures → OPEN)", "Retry Scheduler (exponential backoff + jitter)", "Delivery Status Reporter (update Cassandra, emit metric)"] },
            { name: "Delivery Tracker", owns: "Tracks notification lifecycle: queued → sent → delivered → opened → clicked. Processes provider callbacks.", tech: "Webhook receiver + Kafka consumer → Cassandra", api: "POST /webhooks/apns, /webhooks/ses, ...", scale: "Scales with callback volume. 10 pods.", stateful: false,
              modules: ["Webhook Receiver (APNs feedback, SES events, Twilio status)", "Status Updater (Cassandra write: notification_id → new status)", "Open/Click Tracker (pixel tracking for email, deep link for push)", "Bounce Processor (email hard bounce → deactivate address)", "Unsubscribe Handler (one-click unsubscribe → update prefs)", "Analytics Emitter (delivery funnel metrics to ClickHouse)"] },
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
          <rect x={5} y={5} width={710} height={165} rx={8} fill="#9333ea04" stroke="#9333ea20" strokeWidth={1} strokeDasharray="4,3"/>
          <text x={15} y={22} fill="#9333ea" fontSize="10" fontWeight="700" fontFamily="monospace">NOTIFICATION PIPELINE</text>

          <rect x={5} y={180} width={710} height={80} rx={8} fill="#0284c704" stroke="#0284c720" strokeWidth={1} strokeDasharray="4,3"/>
          <text x={15} y={197} fill="#0284c7" fontSize="10" fontWeight="700" fontFamily="monospace">DELIVERY TRACKING & ANALYTICS</text>

          <rect x={5} y={270} width={710} height={90} rx={8} fill="#d9770604" stroke="#d9770620" strokeWidth={1} strokeDasharray="4,3"/>
          <text x={15} y={287} fill="#d97706" fontSize="10" fontWeight="700" fontFamily="monospace">DATA STORES</text>

          {/* Pipeline */}
          <rect x={15} y={35} width={70} height={40} rx={6} fill="#64748b10" stroke="#64748b" strokeWidth={1.5}/>
          <text x={50} y={50} textAnchor="middle" fill="#64748b" fontSize="8" fontWeight="600" fontFamily="monospace">Upstream</text>
          <text x={50} y={63} textAnchor="middle" fill="#64748b80" fontSize="7" fontFamily="monospace">Services</text>

          <rect x={100} y={35} width={80} height={40} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={140} y={50} textAnchor="middle" fill="#9333ea" fontSize="8" fontWeight="600" fontFamily="monospace">Notif API</text>
          <text x={140} y={63} textAnchor="middle" fill="#9333ea80" fontSize="7" fontFamily="monospace">validate+dedup</text>

          <rect x={200} y={30} width={75} height={50} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={237} y={48} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Priority</text>
          <text x={237} y={60} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Router</text>
          <text x={237} y={73} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">Kafka topics</text>

          <rect x={295} y={30} width={55} height={22} rx={4} fill="#2563eb10" stroke="#2563eb" strokeWidth={1}/>
          <text x={322} y={44} textAnchor="middle" fill="#2563eb" fontSize="7" fontWeight="600" fontFamily="monospace">Push W</text>
          <rect x={295} y={56} width={55} height={22} rx={4} fill="#05966910" stroke="#059669" strokeWidth={1}/>
          <text x={322} y={70} textAnchor="middle" fill="#059669" fontSize="7" fontWeight="600" fontFamily="monospace">Email W</text>
          <rect x={295} y={82} width={55} height={22} rx={4} fill="#d9770610" stroke="#d97706" strokeWidth={1}/>
          <text x={322} y={96} textAnchor="middle" fill="#d97706" fontSize="7" fontWeight="600" fontFamily="monospace">SMS W</text>
          <rect x={295} y={108} width={55} height={22} rx={4} fill="#c026d310" stroke="#c026d3" strokeWidth={1}/>
          <text x={322} y={122} textAnchor="middle" fill="#c026d3" fontSize="7" fontWeight="600" fontFamily="monospace">InApp W</text>

          <rect x={375} y={30} width={60} height={22} rx={4} fill="#2563eb10" stroke="#2563eb" strokeWidth={1}/>
          <text x={405} y={44} textAnchor="middle" fill="#2563eb" fontSize="7" fontWeight="600" fontFamily="monospace">APNs/FCM</text>
          <rect x={375} y={56} width={60} height={22} rx={4} fill="#05966910" stroke="#059669" strokeWidth={1}/>
          <text x={405} y={70} textAnchor="middle" fill="#059669" fontSize="7" fontWeight="600" fontFamily="monospace">SES/SMTP</text>
          <rect x={375} y={82} width={60} height={22} rx={4} fill="#d9770610" stroke="#d97706" strokeWidth={1}/>
          <text x={405} y={96} textAnchor="middle" fill="#d97706" fontSize="7" fontWeight="600" fontFamily="monospace">Twilio</text>
          <rect x={375} y={108} width={60} height={22} rx={4} fill="#c026d310" stroke="#c026d3" strokeWidth={1}/>
          <text x={405} y={122} textAnchor="middle" fill="#c026d3" fontSize="7" fontWeight="600" fontFamily="monospace">WebSocket</text>

          <rect x={460} y={55} width={70} height={40} rx={6} fill="#be123c10" stroke="#be123c" strokeWidth={1.5}/>
          <text x={495} y={70} textAnchor="middle" fill="#be123c" fontSize="8" fontWeight="600" fontFamily="monospace">DLQ</text>
          <text x={495} y={83} textAnchor="middle" fill="#be123c80" fontSize="7" fontFamily="monospace">failed notifs</text>

          <rect x={555} y={35} width={75} height={40} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
          <text x={592} y={50} textAnchor="middle" fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Campaign</text>
          <text x={592} y={63} textAnchor="middle" fill="#7c3aed80" fontSize="7" fontFamily="monospace">Scheduler</text>

          <rect x={645} y={35} width={65} height={40} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
          <text x={677} y={50} textAnchor="middle" fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Template</text>
          <text x={677} y={63} textAnchor="middle" fill="#7c3aed80" fontSize="7" fontFamily="monospace">Service</text>

          {/* Tracking */}
          <rect x={100} y={195} width={80} height={36} rx={6} fill="#0284c710" stroke="#0284c7" strokeWidth={1.5}/>
          <text x={140} y={210} textAnchor="middle" fill="#0284c7" fontSize="8" fontWeight="600" fontFamily="monospace">Delivery</text>
          <text x={140} y={223} textAnchor="middle" fill="#0284c780" fontSize="7" fontFamily="monospace">Tracker</text>

          <rect x={210} y={195} width={80} height={36} rx={6} fill="#0284c710" stroke="#0284c7" strokeWidth={1.5}/>
          <text x={250} y={210} textAnchor="middle" fill="#0284c7" fontSize="8" fontWeight="600" fontFamily="monospace">Webhook</text>
          <text x={250} y={223} textAnchor="middle" fill="#0284c780" fontSize="7" fontFamily="monospace">Receiver</text>

          <rect x={320} y={195} width={90} height={36} rx={6} fill="#0284c710" stroke="#0284c7" strokeWidth={1.5}/>
          <text x={365} y={210} textAnchor="middle" fill="#0284c7" fontSize="8" fontWeight="600" fontFamily="monospace">Analytics</text>
          <text x={365} y={223} textAnchor="middle" fill="#0284c780" fontSize="7" fontFamily="monospace">ClickHouse</text>

          {/* Data Stores */}
          <rect x={100} y={295} width={80} height={36} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={140} y={310} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Redis</text>
          <text x={140} y={323} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">prefs + dedup</text>

          <rect x={210} y={295} width={80} height={36} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={250} y={310} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">PostgreSQL</text>
          <text x={250} y={323} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">prefs + tokens</text>

          <rect x={320} y={295} width={80} height={36} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={360} y={310} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Cassandra</text>
          <text x={360} y={323} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">notif history</text>

          <rect x={430} y={295} width={80} height={36} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={470} y={310} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Kafka</text>
          <text x={470} y={323} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">priority queues</text>

          {/* Arrows */}
          <defs><marker id="ah-svc2" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          <line x1={85} y1={55} x2={100} y2={55} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc2)"/>
          <line x1={180} y1={55} x2={200} y2={55} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc2)"/>
          <line x1={275} y1={41} x2={295} y2={41} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc2)"/>
          <line x1={275} y1={67} x2={295} y2={67} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc2)"/>
          <line x1={275} y1={80} x2={295} y2={93} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc2)"/>
          <line x1={275} y1={80} x2={295} y2={119} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc2)"/>
          <line x1={350} y1={41} x2={375} y2={41} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc2)"/>
          <line x1={350} y1={67} x2={375} y2={67} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc2)"/>
          <line x1={350} y1={93} x2={375} y2={93} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc2)"/>
          <line x1={350} y1={119} x2={375} y2={119} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc2)"/>
          <line x1={435} y1={75} x2={460} y2={75} stroke="#be123c50" strokeWidth={1} markerEnd="url(#ah-svc2)" strokeDasharray="3,2"/>

          {/* Down arrows to tracking & stores */}
          <line x1={140} y1={75} x2={140} y2={195} stroke="#0284c730" strokeWidth={1} markerEnd="url(#ah-svc2)" strokeDasharray="3,2"/>
          <line x1={140} y1={75} x2={140} y2={295} stroke="#dc262630" strokeWidth={1} strokeDasharray="3,2"/>

          {/* Legend */}
          <rect x={540} y={290} width={170} height={65} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={550} y={305} fill="#78716c" fontSize="8" fontWeight="600" fontFamily="monospace">Key Design Points</text>
          <text x={550} y={318} fill="#78716c" fontSize="7" fontFamily="monospace">• Priority → separate Kafka topics</text>
          <text x={550} y={330} fill="#78716c" fontSize="7" fontFamily="monospace">• Per-channel worker pools</text>
          <text x={550} y={342} fill="#78716c" fontSize="7" fontFamily="monospace">• Circuit breaker per provider</text>
          <text x={550} y={354} fill="#78716c" fontSize="7" fontFamily="monospace">• DLQ for permanently failed</text>
        </svg>
      </Card>

      <Card>
        <Label color="#0f766e">Service-to-Service Contracts</Label>
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
                { route: "Upstream → Notif API", proto: "REST / gRPC", timeout: "500ms", fail: "Retry with backoff (caller's responsibility)" },
                { route: "Notif API → Redis (dedup)", proto: "RESP", timeout: "10ms", fail: "Allow through (risk dup, not loss)" },
                { route: "Notif API → Kafka", proto: "Kafka producer", timeout: "100ms", fail: "Return 503 to caller (critical)" },
                { route: "Worker → APNs", proto: "HTTP/2 (TLS)", timeout: "5s", fail: "Retry 3× → DLQ → channel failover" },
                { route: "Worker → SES", proto: "HTTPS (AWS SDK)", timeout: "5s", fail: "Retry 3× → DLQ" },
                { route: "Worker → Twilio", proto: "REST (HTTPS)", timeout: "5s", fail: "Retry 3× → DLQ → alert (cost!)" },
                { route: "Worker → Cassandra", proto: "CQL", timeout: "200ms", fail: "Fire & forget (status update not critical path)" },
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
  );
}

function FlowsSection() {
  const [flow, setFlow] = useState("transactional");
  const flows = {
    transactional: { title: "Transactional — Order Shipped (High Priority)", steps: [
      { actor: "Order Service", action: "POST /notifications {event: order_shipped, user: 42, priority: high, channels: [push, email]}", type: "request" },
      { actor: "Notif API", action: "Validate payload. Check idempotency_key='order_789_shipped' in Redis → new (not dup)", type: "process" },
      { actor: "Notif API → Redis", action: "GET user_prefs:42 → push: ✓ (2 tokens), email: ✓, sms: ✗ (opted out)", type: "auth" },
      { actor: "Notif API", action: "Check rate limit: user_42 push count = 3 this hour (limit: 10) → allowed", type: "process" },
      { actor: "Notif API", action: "Render template: 'Hi Alice, your order ORD-789 has shipped! Track: ...'", type: "process" },
      { actor: "Notif API → Kafka", action: "Publish to topic=notif.high, partition=hash(user_42)%24, for both push and email", type: "success" },
      { actor: "Push Worker", action: "Consume message. Send to APNs HTTP/2: {device_token, alert: 'Your order shipped!'}", type: "request" },
      { actor: "APNs", action: "Returns 200 OK with apns-id. Push delivered to user's iPhone.", type: "success" },
      { actor: "Email Worker", action: "Consume message. Send via SES: from=noreply@example.com, to=alice@gmail.com", type: "request" },
      { actor: "Delivery Tracker", action: "Update Cassandra: notif_xyz → status=sent. Emit metric: notif.sent{channel=push}++", type: "process" },
    ]},
    campaign: { title: "Campaign — Weekly Digest to 50M Users", steps: [
      { actor: "Marketing Tool", action: "POST /notifications/batch {campaign: weekly_digest, segment: active_30d, schedule: 9am local}", type: "request" },
      { actor: "Campaign Scheduler", action: "Query segment: 50M users. Group by timezone (24 cohorts). Schedule each cohort.", type: "process" },
      { actor: "Scheduler (9am EST)", action: "Start EST cohort: 15M users. Feed into Kafka at 100K/min (throttled).", type: "process" },
      { actor: "Notif API", action: "Each notification: check prefs, check rate limit, render template, route to notif.low topic", type: "process" },
      { actor: "Email Worker", action: "Consume from notif.low at throttled rate (5K/sec to SES). Batch where possible.", type: "request" },
      { actor: "SES", action: "Returns 200 for each. Some emails bounce → SES sends bounce webhook.", type: "success" },
      { actor: "Delivery Tracker", action: "Process bounce webhooks: hard bounce → deactivate email address in user_prefs.", type: "check" },
      { actor: "Campaign Monitor", action: "Dashboard: 50M target, 48M sent, 45M delivered, 12M opened (26.7% open rate).", type: "success" },
    ]},
    provider_down: { title: "Provider Failure — APNs Outage", steps: [
      { actor: "Push Worker", action: "Send push to APNs → connection timeout after 5s", type: "error" },
      { actor: "Push Worker", action: "Retry 1: wait 1s + jitter → timeout again. Retry 2: wait 2s → timeout.", type: "error" },
      { actor: "Circuit Breaker", action: "APNs failure count = 5 (threshold). State: CLOSED → OPEN. Stop all APNs sends.", type: "error" },
      { actor: "Push Worker", action: "Circuit open: skip APNs. Move message to retry queue with 60s delay.", type: "process" },
      { actor: "Smart Router", action: "APNs down → for non-critical: escalate to email. For critical (2FA): queue for APNs retry.", type: "check" },
      { actor: "Alert Manager", action: "P1 alert: 'APNs circuit breaker OPEN'. Page on-call engineer.", type: "error" },
      { actor: "Circuit Breaker (60s later)", action: "HALF-OPEN: send 1 probe push to APNs. APNs returns 200 → CLOSED.", type: "success" },
      { actor: "Push Worker", action: "Resume at 10% throughput. Ramp up over 5 min. Drain retry queue.", type: "success" },
    ]},
    critical: { title: "Critical — 2FA Code (Sub-Second SLA)", steps: [
      { actor: "Auth Service", action: "POST /notifications {event: 2fa_code, user: 42, priority: critical, channels: [sms, push]}", type: "request" },
      { actor: "Notif API", action: "Priority = critical → bypass rate limits, skip digest batching, fast-path processing", type: "process" },
      { actor: "Notif API → Kafka", action: "Publish to topic=notif.critical (12 partitions, 20 consumers, over-provisioned)", type: "success" },
      { actor: "SMS Worker", action: "Consume immediately (no lag on critical topic). Call Twilio: send '123456' to +1415...", type: "request" },
      { actor: "Push Worker", action: "Simultaneously: send push with 2FA code to all active device tokens", type: "request" },
      { actor: "Twilio", action: "Returns 201 Created with message_sid. SMS delivered in 1-3 seconds.", type: "success" },
      { actor: "Note", action: "Total e2e: <1 second API-to-provider. SMS delivery depends on carrier (1-30s).", type: "check" },
    ]},
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
          <Label color="#b45309">Kubernetes Deployment — Channel Workers</Label>
          <CodeBlock title="Push Worker — Auto-scaled by Kafka Consumer Lag" code={`apiVersion: apps/v1
kind: Deployment
metadata:
  name: push-worker
spec:
  replicas: 50              # High: handles 400M push/day
  strategy:
    rollingUpdate:
      maxSurge: 10
      maxUnavailable: 5     # Can lose some during deploy
  template:
    spec:
      containers:
      - name: push-worker
        image: push-worker:latest
        env:
        - name: KAFKA_BROKERS
          value: "kafka-cluster.svc:9092"
        - name: KAFKA_TOPIC
          value: "notif.high,notif.medium"
        - name: APNS_TEAM_ID
          valueFrom:
            secretKeyRef:
              name: apns-creds
              key: team-id
        resources:
          requests: {cpu: "500m", memory: "512Mi"}
          limits: {cpu: "2", memory: "1Gi"}
---
# KEDA ScaledObject: scale on Kafka lag
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: push-worker-scaler
spec:
  scaleTargetRef:
    name: push-worker
  minReplicaCount: 20
  maxReplicaCount: 100
  triggers:
  - type: kafka
    metadata:
      topic: notif.high
      lagThreshold: "100"     # Scale up if lag > 100`} />
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Security & Compliance</Label>
          <div className="grid grid-cols-1 gap-4">
            {[
              { layer: "Provider Credentials", what: "APNs auth keys, SES IAM roles, Twilio auth tokens — all stored in Kubernetes Secrets or AWS Secrets Manager. Rotated quarterly.",
                details: ["APNs: token-based auth (not certificate) — easier to rotate", "SES: IAM role per pod (no hardcoded keys)", "Twilio: auth token in K8s Secret, rotated quarterly", "Never log provider credentials or user PII in plain text"] },
              { layer: "User Data & GDPR", what: "Notification content may contain PII. Respect opt-out, right-to-deletion, and data residency.",
                details: ["One-click unsubscribe in every email (CAN-SPAM / GDPR)", "User delete request → purge all notification history from Cassandra", "PII (email, phone) encrypted at rest in PostgreSQL", "Data residency: EU user data stays in EU region"] },
              { layer: "Anti-Abuse", what: "Prevent upstream services from accidentally or maliciously spamming users through the notification system.",
                details: ["Per-service rate limits: 'order-service' can send max 1M/hour", "Per-user global rate limit: max 50 notifications/day total", "Admin kill-switch: disable all sends for a specific event_type", "Anomaly detection: alert if any service sends 10× normal volume"] },
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
    </div>
  );
}

function OpsSection() {
  return (
    <div className="space-y-5">
      <Card accent="#be123c">
        <Label color="#be123c">Scaling Playbook — Component-by-Component</Label>
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
                { bottleneck: "Kafka consumer lag", symptom: "Notifications delayed >30s", fix: "Add consumers (KEDA auto-scale on lag). Increase partitions if consumers = partitions.", pitfall: "Adding partitions requires rebalance — brief pause in consumption." },
                { bottleneck: "APNs throttling", symptom: "HTTP 429 from APNs, push delays", fix: "HTTP/2 multiplex more streams per connection. Add connection pool. Implement token bucket.", pitfall: "Too aggressive retry → APNs bans your team_id temporarily." },
                { bottleneck: "SES rate limit", symptom: "SES returns ThrottlingException", fix: "Request SES limit increase (takes 24-48h). Add sending IPs. Use multiple SES regions.", pitfall: "New IPs need warm-up (2-4 weeks). Sending too fast on cold IP → spam folder." },
                { bottleneck: "Redis memory", symptom: "Evictions, dedup misses", fix: "Reduce dedup TTL. Add Redis shards. Move cold prefs to DB.", pitfall: "Dedup miss → duplicate notification sent. Annoying but not fatal." },
                { bottleneck: "Cassandra write", symptom: "Write latency p99 >100ms", fix: "Add nodes. Check compaction strategy. Increase write consistency to LOCAL_ONE.", pitfall: "Too many tombstones from TTL deletes. Run repairs regularly." },
                { bottleneck: "Campaign overload", symptom: "All channels saturated during campaign", fix: "Throttle campaign ingestion rate. Schedule across hours. Separate campaign Kafka topic.", pitfall: "Campaign traffic should NEVER impact transactional notifications." },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-mono text-stone-700 font-medium">{r.bottleneck}</td>
                  <td className="px-3 py-2 text-red-500">{r.symptom}</td>
                  <td className="px-3 py-2 text-stone-500">{r.fix}</td>
                  <td className="px-3 py-2 text-amber-600 text-[10px]">{r.pitfall}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <Card accent="#be123c">
        <Label color="#be123c">Operational Pitfalls — Production War Stories</Label>
        <div className="grid grid-cols-2 gap-4">
          {[
            { title: "Marketing Campaign Blocks 2FA Codes", symptom: "Users can't log in. 2FA SMS delayed by 10 minutes. Customer support flooded.",
              cause: "50M email campaign used the same Kafka topic as transactional notifications. Campaign messages flooded the queue. 2FA codes stuck behind 50M marketing emails.",
              fix: "Separate Kafka topics for critical/high/low priority. Critical topic has dedicated consumers that are NEVER shared. Campaign traffic goes to 'notif.low' — completely isolated from 'notif.critical'.",
              quote: "Black Friday campaign started at 9am. By 9:05am, 2FA was dead. Users couldn't complete purchases. We lost $2M in 45 minutes." },
            { title: "APNs Token Refresh Storm", symptom: "Push delivery rate drops from 95% to 40% over 2 weeks. No alerts fired because it was gradual.",
              cause: "iOS app update changed the push token for every user on update. 100M tokens refreshed over 2 weeks. Old tokens returned 410 (Unregistered). We didn't process the 410s fast enough.",
              fix: "Process APNs 410 responses in real-time (not batch). On 410: immediately mark token inactive. App on launch: always re-register token. Metric: track invalid_token_rate daily. Alert if > 5%.",
              quote: "Product asked 'why are push open rates dropping?' We said 'users are less engaged.' Turns out 40% of tokens were dead." },
            { title: "Email IP Reputation Destroyed", symptom: "All emails going to spam folder for Gmail users (60% of user base). Open rates drop from 25% to 2%.",
              cause: "New engineer accidentally sent a test campaign to 10M users from a cold IP. Gmail flagged the IP as spam. Reputation takes weeks to recover.",
              fix: "IP warm-up plan: Day 1: 1K emails. Day 2: 5K. Day 7: 100K. Day 14: 1M. Day 21: full volume. Use dedicated IPs for transactional vs marketing. Monitor Sender Score daily. SPF/DKIM/DMARC configured and verified.",
              quote: "Intern deployed to prod on Friday. By Monday, all our emails were in spam. It took 3 weeks to recover Gmail reputation." },
            { title: "SMS Fallback Cost Explosion", symptom: "Twilio bill jumped from $30K/day to $500K/day. Finance team emergency call.",
              cause: "APNs had a 2-hour partial outage. Smart routing escalated 200M push notifications to SMS. At $0.01/SMS, that's $2M in 2 hours.",
              fix: "SMS budget cap: $50K/day hard limit. When exceeded, stop SMS fallback for non-critical. Fall back to email instead. Alert immediately when SMS fallback rate > 5%. Require manual approval to exceed budget.",
              quote: "CFO called CTO at 2am: 'Why is our Twilio bill $500K today?' CTO called VP Eng: 'Why is APNs down?' VP Eng called on-call: 'Why didn't the budget cap trigger?' Answer: we didn't have one." },
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
        { t: "ML-Based Send Time Optimization", d: "Send notifications when each user is most likely to engage, based on historical open patterns.", detail: "Per-user model predicts optimal send hour. 'Alice opens push at 8am, Bob at 6pm.' Increases open rate by 15-30%.", effort: "Hard" },
        { t: "Rich Push (Images, Actions)", d: "Support rich media in push: images, action buttons ('Track Order', 'Leave Review').", detail: "APNs mutable-content + notification service extension. FCM data messages with BigPictureStyle.", effort: "Medium" },
        { t: "Cross-Channel Dedup", d: "If user opens push, cancel the pending email for the same event. Don't send both.", detail: "On push open callback: check if email for same notification_id is still queued. If yes, suppress it.", effort: "Medium" },
        { t: "A/B Testing on Templates", d: "Test different subject lines, body copy, send times. Measure open rate per variant.", detail: "Assign users to variant A/B by hash. Track open/click per variant in ClickHouse. Auto-promote winner.", effort: "Medium" },
        { t: "Webhook Channel Plugin", d: "Let B2B customers receive notifications via webhook to their own servers.", detail: "New channel type: webhook. POST to customer's URL with HMAC signature. Retry with backoff on 5xx.", effort: "Easy" },
        { t: "Notification Inbox API", d: "In-app notification center with read/unread state, infinite scroll, and mark-all-read.", detail: "GET /users/:id/notifications?cursor=...&limit=50. Cassandra query by user_id partition. WebSocket for real-time.", effort: "Easy" },
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
    { q:"How do you guarantee exactly-once delivery?", a:"You can't — truly exactly-once is impossible in a distributed system with external providers. Instead: at-least-once delivery (never lose) + client-side dedup (idempotency_key in Redis with 1hr TTL). If dedup Redis fails, you might send twice, but you'll never silently drop. Providers also dedup: APNs collapses identical pushes, SES deduplicates by message-id.", tags:["design"] },
    { q:"How do you handle user timezone-aware sending?", a:"Store user timezone in preferences. For campaigns: group users by timezone, schedule each cohort at the target local time. Implementation: campaign scheduler creates one Kafka message per timezone cohort, delayed to the right UTC time. For 24 timezones, the campaign naturally spreads over 24 hours, which also helps with load distribution.", tags:["scalability"] },
    { q:"Push vs email vs SMS — how do you decide?", a:"Smart routing hierarchy: (1) Push — free, instant, highest engagement. Use if user has active device token. (2) Email — cheap, good for rich content, lower urgency. (3) SMS — expensive, use only for critical (2FA, fraud). Smart routing checks: does user have a push token? Is it valid (not expired)? If yes → push. If no → email. If critical → SMS. This saves 60-80% on SMS costs.", tags:["algorithm"] },
    { q:"How would you handle 100M notifications in 5 minutes (flash sale)?", a:"Throttled ingestion: campaign controller feeds users into Kafka at 500K/min (controlled rate). Provider-side: batch APIs (FCM: 500/request). SES: pre-warmed IPs at 50K/sec. Pre-scale workers 30 min before launch. Separate 'campaign' Kafka topic so transactional isn't affected. Realistically: 100M in 5 min = 333K/sec — need provider pre-arrangement.", tags:["scalability"] },
    { q:"What if the notification service itself goes down?", a:"Kafka is the buffer. Upstream services produce to Kafka directly (thin API layer). Even if the notification service is completely down, messages are durably stored in Kafka (7 days retention). When service recovers, workers resume from last committed offset. No data loss. For the API layer: run 20+ stateless pods behind a load balancer — very hard to take down entirely.", tags:["availability"] },
    { q:"How do you prevent notification fatigue?", a:"Per-user rate limits (10 push/hr, 5 email/day). Digest batching for high-frequency events ('3 new likes' instead of 3 separate pushes). ML-based relevance scoring: don't send low-relevance notifications. Quiet hours: no pushes between 10pm-8am local time. Category-level opt-out: user can mute 'marketing' but keep 'order updates'. Track uninstall rate — if it spikes after a campaign, you're over-sending.", tags:["design"] },
    { q:"How do you monitor delivery rates across providers?", a:"Track the delivery funnel per channel: ingested → sent → delivered → opened → clicked. Key metric: delivery rate = delivered / sent. Push: expect 85-95% (some tokens stale). Email: expect 95%+ (bounces are normal). SMS: expect 98%+ (carrier issues rare). Alert if any channel drops below threshold. Provider dashboard: per-provider error rate, latency p99, circuit breaker state.", tags:["observability"] },
    { q:"Why Cassandra for notification history and not PostgreSQL?", a:"Scale: 1B writes/day. PostgreSQL handles ~10K writes/sec on a single node — you'd need 100+ shards. Cassandra handles 1B writes/day across a modest 10-node cluster. Access pattern: all queries are by user_id (partition key) + time range — perfect for Cassandra's data model. TTL: 30-day auto-expiry is built-in. Trade-off: no ad-hoc queries, no joins — but notification history doesn't need those.", tags:["data"] },
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

export default function NotificationSystemSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Notification System</h1>
            <Pill bg="#f3e8ff" color="#7c3aed">System Design</Pill>
            <Pill bg="#fef2f2" color="#dc2626">Billions Scale</Pill>
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