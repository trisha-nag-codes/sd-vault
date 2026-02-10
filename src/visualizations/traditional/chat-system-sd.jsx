import { useState, useRef, useEffect } from "react";

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   CHAT SYSTEM (WhatsApp) — System Design Reference
   Pearl white theme · 17 sections (HLD + LLD)
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

const SECTIONS = [
  { id: "concept",       label: "Concept",             icon: "ðŸ’¡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",         icon: "ðŸ“‹", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",  icon: "ðŸ”¢", color: "#7c3aed" },
  { id: "api",           label: "API Design",           icon: "ðŸ”Œ", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",    icon: "ðŸ—ï¸", color: "#9333ea" },
  { id: "algorithm",     label: "Protocol Deep Dive",   icon: "⚙ï¸", color: "#c026d3" },
  { id: "data",          label: "Data Model",           icon: "ðŸ—„ï¸", color: "#dc2626" },
  { id: "scalability",   label: "Scalability",          icon: "ðŸ“ˆ", color: "#059669" },
  { id: "availability",  label: "Availability",         icon: "ðŸ›¡ï¸", color: "#d97706" },
  { id: "observability", label: "Observability",        icon: "ðŸ“Š", color: "#0284c7" },
  { id: "watchouts",     label: "Failure Modes",        icon: "⚠ï¸", color: "#dc2626" },
  { id: "services",      label: "Service Architecture", icon: "ðŸ§©", color: "#0f766e" },
  { id: "flows",         label: "Request Flows",        icon: "ðŸ”€", color: "#7e22ce" },
  { id: "deployment",    label: "Deploy & Security",    icon: "ðŸ”’", color: "#b45309" },
  { id: "ops",           label: "Ops Playbook",         icon: "ðŸ”§", color: "#be123c" },
  { id: "enhancements",  label: "Enhancements",         icon: "ðŸš€", color: "#7c3aed" },
  { id: "followups",     label: "Follow-up Questions",  icon: "â“", color: "#6366f1" },
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
          <div key={i} className={`px-2 rounded ${line.trim().startsWith("#") || line.trim().startsWith("//") ? "text-stone-400" : "text-stone-700"}`}>
            <span className="inline-block w-5 text-right mr-3 text-stone-300 select-none">{line.trim() ? i + 1 : ""}</span>{line}
          </div>
        ))}
      </pre>
    </div>
  );
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   SECTIONS
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

function ConceptSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-7 space-y-5">
          <Card accent="#6366f1">
            <Label>What is a Real-Time Chat System?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A messaging platform that delivers text, media, and group conversations in real-time between users. Think WhatsApp, Facebook Messenger, Slack, or iMessage. The core challenge: deliver messages reliably with sub-second latency to users who may be online, offline, or switching between devices.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Unlike a request-response API, chat requires persistent bidirectional connections (WebSocket), presence tracking, offline message queuing, ordering guarantees, and delivery receipts. These make it fundamentally different from typical web services.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard?</Label>
            <ul className="space-y-2.5">
              <Point icon="ðŸ“¡" color="#0891b2">Persistent connections — millions of open WebSocket connections consuming server memory and file descriptors</Point>
              <Point icon="ðŸ“¬" color="#0891b2">Offline delivery — user is offline when message arrives. Must queue and deliver when they reconnect. Never lose a message.</Point>
              <Point icon="ðŸ”¢" color="#0891b2">Message ordering — messages must appear in correct order even with network delays, retries, and multiple devices</Point>
              <Point icon="ðŸ‘¥" color="#0891b2">Group chat fan-out — one message to a 500-person group = 500 deliveries, each with their own online/offline state</Point>
              <Point icon="ðŸ”" color="#0891b2">End-to-end encryption — server can't read message content. Key exchange, forward secrecy, multi-device key management</Point>
              <Point icon="ðŸŒ" color="#0891b2">Multi-device sync — same user on phone + laptop. Both must see all messages, read receipts, typing indicators</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Scale</Label>
            <div className="space-y-2.5">
              {[
                { co: "WhatsApp", scale: "2B+ users, 100B msgs/day", detail: "~50 engineers at peak. Erlang." },
                { co: "Messenger", scale: "1B+ users, grouped with Instagram DMs", detail: "Massive infra, ML-heavy" },
                { co: "Slack", scale: "32M+ DAU, enterprise focus", detail: "Channels model, not 1:1 only" },
                { co: "Telegram", scale: "800M+ MAU, 100K group limit", detail: "MTProto protocol, supergroups" },
                { co: "WeChat", scale: "1.3B+ MAU, super-app", detail: "Chat + payments + mini-apps" },
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
            <svg viewBox="0 0 360 120" className="w-full">
              <rect x={10} y={10} width={160} height={45} rx={6} fill="#dc262608" stroke="#dc2626" strokeWidth={1.5}/>
              <text x={90} y={28} textAnchor="middle" fill="#dc2626" fontSize="10" fontWeight="700" fontFamily="monospace">HTTP Polling</text>
              <text x={90} y={44} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">âŒ High latency, wasteful</text>

              <rect x={190} y={10} width={160} height={45} rx={6} fill="#05966908" stroke="#059669" strokeWidth={1.5}/>
              <text x={270} y={28} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="700" fontFamily="monospace">WebSocket ★</text>
              <text x={270} y={44} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">✓ Full-duplex, real-time</text>

              <rect x={60} y={68} width={240} height={42} rx={6} fill="#9333ea08" stroke="#9333ea" strokeWidth={1.5}/>
              <text x={180} y={85} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">WebSocket for real-time + HTTP for media upload/download</text>
              <text x={180} y={100} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">Best of both: persistent for msgs, stateless for blobs</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Top 3 most-asked system design question</div>
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
          <span className="text-lg">ðŸ’¡</span>
          <div>
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Scope Is Everything</div>
            <p className="text-[12px] text-stone-500 mt-0.5">"Design WhatsApp" is enormous. Clarify: 1:1 chat only, or group chat too? Text only, or media? E2E encryption? Voice/video calls? For a 45-min interview, focus on <strong>1:1 text chat + group chat + delivery guarantees + online/offline</strong>. E2E encryption and calls are follow-ups.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">1:1 messaging — send and receive text messages in real-time</Point>
            <Point icon="2." color="#059669">Group chat — up to 500 members, all receive every message</Point>
            <Point icon="3." color="#059669">Delivery receipts — sent ✓, delivered ✓✓, read (blue ✓✓)</Point>
            <Point icon="4." color="#059669">Online/offline presence — "last seen 5 min ago"</Point>
            <Point icon="5." color="#059669">Offline message delivery — messages queued and delivered on reconnect</Point>
            <Point icon="6." color="#059669">Chat history — persistent, scrollable, searchable</Point>
            <Point icon="7." color="#059669">Media sharing — images, videos, documents (via CDN)</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Message delivery latency: &lt;500ms for online users</Point>
            <Point icon="2." color="#dc2626">Zero message loss — messages must never be dropped</Point>
            <Point icon="3." color="#dc2626">Message ordering — per-conversation, causal ordering</Point>
            <Point icon="4." color="#dc2626">Scale to 500M+ DAU, 100B messages/day</Point>
            <Point icon="5." color="#dc2626">High availability — 99.99% uptime (chat is critical infra)</Point>
            <Point icon="6." color="#dc2626">Support millions of concurrent WebSocket connections</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "1:1 only or group chat too? Max group size?",
            "Text only, or images/video/voice messages?",
            "End-to-end encryption required?",
            "Multi-device support? (same account on phone + laptop)",
            "Message retention policy? Forever or TTL?",
            "Do we need voice/video calls? (usually separate system)",
            "Read receipts and typing indicators?",
            "What scale? (DAU, messages/day)",
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
            <MathStep step="1" formula="DAU = 500M users" result="500M" note="WhatsApp-scale." />
            <MathStep step="2" formula="Messages sent per user/day = 40" result="40" note="Average across 1:1 + groups. Active users send more." />
            <MathStep step="3" formula="Total messages/day = 500M × 40" result="20B" note="Sent messages. Each received by 1 (1:1) or N (group) recipients." />
            <MathStep step="4" formula="Messages/sec = 20B / 86,400" result="~230K msg/sec" note="Average send rate." final />
            <MathStep step="5" formula="Peak messages/sec = 230K × 3" result="~700K msg/sec" note="New Year's Eve, events, etc." final />
            <MathStep step="6" formula="Concurrent WebSocket connections" result="~150M" note="~30% of DAU online at any time. Each holds 1 WS connection." final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Storage Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Average message size = 100 bytes (text)" result="100 B" note="UTF-8 text. Metadata (sender, timestamp, IDs) adds ~100B more." />
            <MathStep step="2" formula="Total per message with metadata" result="~200 B" note="message_id, conversation_id, sender_id, timestamp, content, status" />
            <MathStep step="3" formula="Daily storage = 20B msgs × 200B" result="~4 TB/day" note="Text messages only. Media stored separately in blob store." />
            <MathStep step="4" formula="Yearly storage = 4 TB × 365" result="~1.5 PB/year" note="Chat history grows forever (WhatsApp model)." final />
            <MathStep step="5" formula="Media: 5% of msgs have media, avg 200KB" result="~200 TB/day" note="Images/videos dominate storage. Stored in S3/blob, not in DB." final />
            <MathStep step="6" formula="5-year retention (text only)" result="~7.3 PB" note="Plus media: 365 PB. This is why media goes to object storage." />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Connection Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Concurrent WS connections = 150M" result="150M" note="Each connection consumes ~10KB memory on server." />
            <MathStep step="2" formula="Memory per connection = 10KB" result="10 KB" note="Connection state, buffers, encryption context." />
            <MathStep step="3" formula="Connections per server (64GB RAM) = ~500K" result="~500K" note="Generous — real systems do 1M+ with tuning (epoll, file descriptors)." />
            <MathStep step="4" formula="Chat servers needed = 150M / 500K" result="~300 servers" note="Just for WebSocket connections. Stateful — connection pinned to server." final />
            <MathStep step="5" formula="File descriptors per server = 500K + overhead" result="~600K" note="ulimit -n must be set high. Default 1024 is not enough." />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Summary — Quick Reference</Label>
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: "Peak Msg Rate", val: "~700K/sec", sub: "Sent messages" },
              { label: "Concurrent Conns", val: "~150M", sub: "Open WebSockets" },
              { label: "Chat Servers", val: "~300", sub: "For WS connections" },
              { label: "Daily Storage", val: "~4 TB", sub: "Text (+ 200TB media)" },
              { label: "Msg Size", val: "~200 B", sub: "With metadata" },
              { label: "Yearly Text", val: "~1.5 PB", sub: "Append-only, sharded" },
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
          <Label color="#2563eb">WebSocket Events (Real-Time)</Label>
          <CodeBlock code={`# Client → Server (send message)
{
  "type": "message.send",
  "request_id": "req_abc123",   # Client-generated, for ack
  "conversation_id": "conv_789",
  "content": "Hey, are you free tonight?",
  "content_type": "text",        # text | image | video | file
  "client_timestamp": 1707500042123
}

# Server → Client (receive message)
{
  "type": "message.new",
  "message_id": "msg_xyz456",    # Server-generated (Snowflake)
  "conversation_id": "conv_789",
  "sender_id": "user_42",
  "content": "Hey, are you free tonight?",
  "content_type": "text",
  "server_timestamp": 1707500042200,
  "sequence_num": 14523          # Per-conversation ordering
}

# Server → Sender (delivery ack)
{
  "type": "message.ack",
  "request_id": "req_abc123",    # Matches client request
  "message_id": "msg_xyz456",
  "status": "sent"               # sent | delivered | read
}

# Client → Server (read receipt)
{
  "type": "message.read",
  "conversation_id": "conv_789",
  "last_read_message_id": "msg_xyz456"
}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">REST APIs (Non-Real-Time)</Label>
          <CodeBlock code={`# GET /v1/conversations
# List user's conversations (inbox)
# Returns: [{conv_id, last_message, unread_count, ...}]

# GET /v1/conversations/:id/messages?cursor=&limit=50
# Paginated chat history (cursor = message_id)
# Returns: {messages: [...], next_cursor, has_more}

# POST /v1/conversations
# Create new 1:1 or group conversation
{
  "type": "group",               # direct | group
  "participants": ["user_42", "user_789", "user_101"],
  "name": "Weekend Plans"        # For groups only
}

# POST /v1/media/upload
# Pre-signed URL for media upload → S3
# Returns: {upload_url, media_id, expires_in: 3600}

# GET /v1/conversations/:id/media/:media_id
# Redirect to CDN URL for media download

# PUT /v1/users/me/presence
# Update online status (or auto-detected by WS connect/disconnect)
{ "status": "online" }           # online | away | offline`} />
          <div className="mt-3 space-y-2">
            {[
              { q: "Why WebSocket for messages but REST for history?", a: "Messages need real-time push (bidirectional). History is a paginated read — standard request-response. No need to keep a socket open for a one-time query." },
              { q: "Why client-generated request_id?", a: "For idempotency. Client retries on timeout → server deduplicates by request_id. Prevents double-send on flaky networks." },
              { q: "Why server-generated message_id + sequence_num?", a: "Server is the single source of truth for ordering. Client timestamps can drift. sequence_num provides total order per conversation." },
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
        <svg viewBox="0 0 720 340" className="w-full">
          {/* Clients */}
          <rect x={10} y={100} width={65} height={34} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={42} y={120} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Phone A</text>
          <rect x={10} y={150} width={65} height={34} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={42} y={170} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Phone B</text>

          {/* LB */}
          <rect x={110} y={120} width={65} height={46} rx={6} fill="#6366f110" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={142} y={140} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">Load</text>
          <text x={142} y={152} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">Balancer</text>

          {/* Chat servers */}
          <rect x={215} y={85} width={90} height={34} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={260} y={106} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Chat Server 1</text>
          <rect x={215} y={130} width={90} height={34} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={260} y={151} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Chat Server 2</text>
          <rect x={215} y={175} width={90} height={34} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={260} y={196} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Chat Server N</text>
          <text x={260} y={225} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">WebSocket handlers</text>
          <text x={260} y={236} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">(stateful — sticky sessions)</text>

          {/* Session Registry */}
          <rect x={350} y={75} width={80} height={38} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={390} y={92} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Session</text>
          <text x={390} y={104} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Registry</text>

          {/* Message Queue */}
          <rect x={350} y={130} width={80} height={38} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={390} y={147} textAnchor="middle" fill="#ea580c" fontSize="9" fontWeight="600" fontFamily="monospace">Kafka</text>
          <text x={390} y={160} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">msg queue</text>

          {/* Presence */}
          <rect x={350} y={185} width={80} height={38} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={390} y={202} textAnchor="middle" fill="#0891b2" fontSize="9" fontWeight="600" fontFamily="monospace">Presence</text>
          <text x={390} y={214} textAnchor="middle" fill="#0891b280" fontSize="7" fontFamily="monospace">online/offline</text>

          {/* Message DB */}
          <rect x={480} y={75} width={80} height={38} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={520} y={92} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Message</text>
          <text x={520} y={104} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">DB</text>

          {/* Push service */}
          <rect x={480} y={130} width={80} height={38} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={520} y={147} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Push</text>
          <text x={520} y={160} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Service</text>

          {/* Media / CDN */}
          <rect x={480} y={185} width={80} height={38} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={520} y={202} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Media</text>
          <text x={520} y={214} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">S3 + CDN</text>

          {/* Group service */}
          <rect x={620} y={100} width={80} height={38} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={660} y={117} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Group</text>
          <text x={660} y={130} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Service</text>

          {/* Notification */}
          <rect x={620} y={160} width={80} height={38} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={660} y={177} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">APNs/FCM</text>
          <text x={660} y={190} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">push notifications</text>

          {/* Arrows */}
          <defs><marker id="ah-hld" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          <line x1={75} y1={117} x2={110} y2={135} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={75} y1={167} x2={110} y2={152} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={175} y1={137} x2={215} y2={102} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={175} y1={143} x2={215} y2={147} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={175} y1={150} x2={215} y2={192} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={305} y1={102} x2={350} y2={94} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={305} y1={147} x2={350} y2={149} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={305} y1={192} x2={350} y2={204} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={430} y1={94} x2={480} y2={94} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={430} y1={149} x2={480} y2={149} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={430} y1={204} x2={480} y2={204} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={560} y1={149} x2={620} y2={179} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={560} y1={94} x2={620} y2={119} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>

          {/* Flow labels */}
          <rect x={10} y={270} width={700} height={60} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={20} y={287} fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Send flow: Client → WS → Chat Server → save to DB + enqueue Kafka → route to recipient's Chat Server → push via WS</text>
          <text x={20} y={302} fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Offline: Recipient not connected → Push Service sends APNs/FCM notification → msg queued → delivered on reconnect</text>
          <text x={20} y={317} fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Session Registry: maps user_id → chat_server_id. "Where is this user connected?" Redis-backed, sub-ms lookup.</text>
        </svg>
      </Card>
      <Card>
        <Label color="#c026d3">Key Architecture Decisions</Label>
        <div className="grid grid-cols-3 gap-4">
          {[
            { q: "Why WebSocket and not long polling?", a: "Full-duplex: server pushes messages instantly. Long polling has ~1s latency (wait for response, re-poll). At 150M connections, long polling creates 150M requests/sec just for keep-alive. WebSocket: one connection, bidirectional, sub-100ms delivery." },
            { q: "Why is the Chat Server stateful?", a: "Each user's WebSocket is pinned to a specific server. The server holds the connection in memory. You can't route a message to a user without knowing WHICH server they're connected to. The Session Registry (Redis) maps user→server." },
            { q: "Why Kafka between Chat Servers?", a: "Sender and recipient may be on different servers. Kafka decouples them. Sender's server publishes message, recipient's server consumes. Also provides durability (message persisted in Kafka before delivery) and replay on failure." },
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
  const [sel, setSel] = useState("transport");
  const topics = {
    transport: { name: "Poll vs Push (Transport)", cx: "WebSocket ★ — The Core Decision",
      desc: "Before designing anything else, you must decide HOW the server delivers messages to the client. This is the first thing to discuss in an interview. There are four options — only one is right for real-time chat.",
      code: `# 1. HTTP Polling (âŒ Don't use)
while True:
    response = HTTP_GET("/messages?since=last_ts")
    # Problem: 150M users × 1 poll/sec = 150M req/sec
    # Most responses are empty (no new messages)
    # Latency: up to 1 second (poll interval)
    sleep(1)

# 2. Long Polling (⚠ï¸ Better, but still not ideal)
while True:
    response = HTTP_GET("/messages?since=last_ts")
    # Server HOLDS the request open until:
    #   a) New message arrives → return immediately, OR
    #   b) Timeout (30s) → return empty, client re-polls
    # Better latency (~instant), but:
    #   - Still half-duplex (client can't send while waiting)
    #   - HTTP overhead on every reconnect (headers, TLS)
    #   - Server holds open connections anyway

# 3. Server-Sent Events / SSE (⚠ï¸ Half solution)
source = EventSource("/messages/stream")
source.onmessage = handle_new_message
# Server pushes, but client can't send through it
# Client sends via separate HTTP POST — two channels
# No binary support, limited browser connections (6 per domain)

# 4. WebSocket (✓ The right answer)
ws = WebSocket("wss://chat.example.com/ws")
ws.onmessage = handle_incoming      # Server → Client
ws.send(JSON.stringify(message))     # Client → Server
# Full-duplex: both directions on ONE TCP connection
# Sub-100ms latency, minimal overhead (2-byte frame header)
# Binary support (protobuf), built-in ping/pong for keepalive
# ONE connection handles: messages, receipts, typing, presence`,
      points: [
        "HTTP Polling: simple but wasteful — 99% of polls return nothing. 150M req/sec for keep-alive alone. Up to 1s latency.",
        "Long Polling: server holds request until data available. Better latency, but still half-duplex. Each response requires new HTTP request (overhead).",
        "SSE (Server-Sent Events): server can push, but client can't send through the same channel. Need separate HTTP POST channel. Limited to text (no binary/protobuf).",
        "WebSocket ★: full-duplex on a single TCP connection. Both sides send anytime. 2-byte frame overhead vs ~800 bytes for HTTP headers. Sub-100ms delivery.",
        "WhatsApp, Slack, Discord, Telegram — ALL use WebSocket (or custom TCP) for real-time messaging.",
        "Hybrid in practice: WebSocket for real-time messages + HTTP REST for media upload, history fetch, search (stateless operations).",
      ] },
    ordering: { name: "Message Ordering", cx: "Per-conversation total order",
      desc: "Messages in a conversation must appear in the same order for all participants. This is harder than it sounds with multiple servers, network delays, and retries.",
      code: `# Server assigns sequence number on receipt
on_message_received(conversation_id, message):
    # Atomic increment — single source of truth
    seq = redis.incr(f"seq:{conversation_id}")
    message.sequence_num = seq
    message.server_timestamp = now()

    # Store and deliver with sequence number
    db.save(message)
    deliver_to_recipients(message)

# Client inserts messages by sequence_num, NOT timestamp
# If msg 14 arrives before msg 13, client waits or re-orders
# Client can request gap fill: GET /messages?after_seq=12&conv=789`,
      points: ["Server-assigned sequence number per conversation (atomic Redis INCR)", "Client-side reordering: if seq 14 arrives before 13, buffer and wait", "Gap detection: client notices missing seq → requests backfill from server", "NOT timestamp-based: client clocks drift, server clock is single source of truth", "For group chat: same approach — single sequence per conversation, not per sender"] },
    delivery: { name: "Delivery Guarantees", cx: "At-least-once with idempotency",
      desc: "Messages must never be lost. But network is unreliable — duplicates are possible. We guarantee at-least-once delivery + client-side deduplication.",
      code: `# Three-phase delivery guarantee
#
# Phase 1: Client → Server (at-least-once)
client.send(message, request_id="req_abc")
# Client retries until it receives ACK
# Server deduplicates by request_id

# Phase 2: Server → Recipient (at-least-once)
if recipient.is_online():
    push_via_websocket(recipient, message)
    # Wait for client ACK within 5s
    # If no ACK → retry 3x → mark for offline delivery
else:
    store_in_offline_queue(recipient, message)
    send_push_notification(recipient)

# Phase 3: Offline → Online (on reconnect)
on_client_reconnect(user_id):
    pending = offline_queue.get_all(user_id)
    for msg in pending:
        push_via_websocket(user_id, msg)
        # Wait for ACK
    offline_queue.clear(user_id)`,
      points: ["Client-generated request_id prevents duplicate sends on retry", "Server ACKs every message — client knows server received it", "Server stores message in DB BEFORE attempting delivery (durability first)", "Offline queue: messages for offline users stored in persistent queue", "On reconnect: drain offline queue, deliver all pending messages", "Client deduplicates by message_id (server-generated, globally unique)"] },
    presence: { name: "Presence / Online Status", cx: "Heartbeat + timeout",
      desc: "How do we know if a user is online? Not as simple as 'connected = online'. Network drops, app backgrounding, and device sleep make this surprisingly hard.",
      code: `# Heartbeat-based presence
HEARTBEAT_INTERVAL = 30  # seconds
PRESENCE_TIMEOUT = 90    # seconds (3 missed heartbeats)

on_websocket_connect(user_id, server_id):
    presence_store.set(user_id, {
        status: "online",
        server_id: server_id,
        last_heartbeat: now()
    })
    notify_contacts(user_id, "online")

on_heartbeat(user_id):
    presence_store.update_heartbeat(user_id, now())

# Background sweeper (every 30s)
sweep_stale_presence():
    stale = presence_store.find(
        last_heartbeat < now() - PRESENCE_TIMEOUT
    )
    for user_id in stale:
        presence_store.set_offline(user_id)
        notify_contacts(user_id, "offline")`,
      points: ["Client sends heartbeat every 30s over WebSocket (tiny payload: 2 bytes)", "If 3 heartbeats missed (90s), mark user as offline", "Store in Redis: user_id → {status, server_id, last_heartbeat}", "Last seen: record timestamp on disconnect or timeout", "Fan-out presence updates: only notify users who have this user in their contact list", "Rate-limit presence updates: don't notify for quick reconnects (< 30s offline)"] },
    e2ee: { name: "End-to-End Encryption", cx: "Signal Protocol (Double Ratchet)",
      desc: "Server cannot read message content. Only sender and recipient have keys. WhatsApp, Signal, and iMessage all use variations of the Signal Protocol.",
      code: `# Signal Protocol — Simplified Flow
#
# 1. Key Registration (one-time)
user.generate_identity_key()      # Long-term
user.generate_signed_prekey()     # Medium-term (rotated monthly)
user.generate_one_time_prekeys(100)  # Ephemeral (used once each)
server.store_public_keys(user)    # Server stores PUBLIC keys only

# 2. Session Setup (X3DH Key Agreement)
sender.fetch_recipient_public_keys()  # From server
shared_secret = X3DH(
    sender.identity_key,
    sender.ephemeral_key,
    recipient.identity_key,
    recipient.signed_prekey,
    recipient.one_time_prekey       # Consumed — never reused
)

# 3. Message Encryption (Double Ratchet)
encrypted = double_ratchet.encrypt(shared_secret, plaintext)
# Each message uses a NEW symmetric key (forward secrecy)
# Compromising one key doesn't reveal past or future messages

# Server sees: {sender, recipient, encrypted_blob, timestamp}
# Server CANNOT decrypt. Only recipient's device can.`,
      points: ["Server stores only public keys — never has access to message content", "X3DH: asynchronous key agreement (works even if recipient is offline)", "Double Ratchet: every message uses a new key (forward secrecy)", "One-time prekeys consumed on use — prevents replay attacks", "Multi-device: each device has its own key pair. Sender encrypts N times (once per device)", "Group E2EE: Sender Key protocol — one encrypt, N decrypts. More efficient than N separate encryptions."] },
  };
  const t = topics[sel];
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Transport Protocol Comparison — Poll vs Push</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[12px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Protocol</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Direction</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Latency</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Overhead</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">150M Users Cost</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Verdict</th>
            </tr></thead>
            <tbody>
              {[
                { n:"HTTP Polling", dir:"Client → Server", lat:"Up to 1s (poll interval)", oh:"~800B headers per poll", cost:"150M req/sec (99% empty)", v:"âŒ", hl:false },
                { n:"Long Polling", dir:"Half-duplex", lat:"~Instant (held request)", oh:"New HTTP req per response", cost:"150M held conns + reconnect overhead", v:"⚠ï¸", hl:false },
                { n:"SSE", dir:"Server → Client only", lat:"~Instant (push)", oh:"Text only, no binary", cost:"150M conns + separate POST channel", v:"⚠ï¸", hl:false },
                { n:"WebSocket ★", dir:"Full-duplex ✓", lat:"<100ms (persistent)", oh:"2-byte frame header", cost:"150M persistent conns (most efficient)", v:"✓", hl:true },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-purple-50" : i%2 ? "bg-stone-50/50" : ""}>
                  <td className={`px-3 py-2.5 font-mono ${r.hl?"text-purple-700 font-bold":"text-stone-600"}`}>{r.n}</td>
                  <td className="text-center px-3 py-2.5 text-stone-500">{r.dir}</td>
                  <td className="text-center px-3 py-2.5 text-stone-500">{r.lat}</td>
                  <td className="text-center px-3 py-2.5 text-stone-500">{r.oh}</td>
                  <td className="text-center px-3 py-2.5 text-stone-400 text-[11px]">{r.cost}</td>
                  <td className="text-center px-3 py-2.5 text-lg">{r.v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3 grid grid-cols-4 gap-3">
          <div className="rounded-lg border border-red-200 bg-red-50/30 p-2.5 text-center">
            <div className="text-[10px] font-bold text-red-600">HTTP Polling</div>
            <div className="text-[20px] mt-1">ðŸ”„</div>
            <div className="text-[9px] text-stone-400 mt-1">Client asks every 1s</div>
            <div className="text-[9px] text-stone-400">"Any new messages?"</div>
            <div className="text-[9px] text-red-500 mt-1 font-medium">99% wasted requests</div>
          </div>
          <div className="rounded-lg border border-amber-200 bg-amber-50/30 p-2.5 text-center">
            <div className="text-[10px] font-bold text-amber-600">Long Polling</div>
            <div className="text-[20px] mt-1">â³</div>
            <div className="text-[9px] text-stone-400 mt-1">Server holds request open</div>
            <div className="text-[9px] text-stone-400">Returns when msg arrives</div>
            <div className="text-[9px] text-amber-600 mt-1 font-medium">Better, but half-duplex</div>
          </div>
          <div className="rounded-lg border border-blue-200 bg-blue-50/30 p-2.5 text-center">
            <div className="text-[10px] font-bold text-blue-600">Server-Sent Events</div>
            <div className="text-[20px] mt-1">ðŸ“¡</div>
            <div className="text-[9px] text-stone-400 mt-1">Server pushes events</div>
            <div className="text-[9px] text-stone-400">Client can't send back</div>
            <div className="text-[9px] text-blue-600 mt-1 font-medium">One-way only</div>
          </div>
          <div className="rounded-lg border border-purple-300 bg-purple-50/50 p-2.5 text-center">
            <div className="text-[10px] font-bold text-purple-700">WebSocket ★</div>
            <div className="text-[20px] mt-1">⚡</div>
            <div className="text-[9px] text-stone-400 mt-1">Both sides send anytime</div>
            <div className="text-[9px] text-stone-400">One persistent TCP connection</div>
            <div className="text-[9px] text-purple-700 mt-1 font-bold">Full-duplex, minimal overhead</div>
          </div>
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
          <CodeBlock code={`-- Messages (sharded by conversation_id)
-- Write-optimized: append-only, time-series-like
CREATE TABLE messages (
  message_id      BIGINT PRIMARY KEY,   -- Snowflake ID
  conversation_id BIGINT NOT NULL,
  sender_id       BIGINT NOT NULL,
  content         BLOB,                 -- Encrypted blob (E2EE)
  content_type    ENUM('text','image','video','file'),
  sequence_num    INT NOT NULL,         -- Per-conversation ordering
  server_ts       TIMESTAMP NOT NULL,
  INDEX idx_conv_seq (conversation_id, sequence_num DESC)
);
-- Shard key: conversation_id
-- All messages in a conversation on same shard (ordering!)

-- Conversations (sharded by conversation_id)
CREATE TABLE conversations (
  conversation_id BIGINT PRIMARY KEY,
  type            ENUM('direct','group'),
  name            VARCHAR(255),          -- NULL for direct
  created_at      TIMESTAMP,
  updated_at      TIMESTAMP              -- Last message time
);

-- Participants (sharded by user_id for inbox queries)
CREATE TABLE participants (
  user_id         BIGINT NOT NULL,
  conversation_id BIGINT NOT NULL,
  joined_at       TIMESTAMP,
  last_read_seq   INT DEFAULT 0,         -- For unread count
  muted_until     TIMESTAMP,
  PRIMARY KEY (user_id, conversation_id),
  INDEX idx_conv (conversation_id)
);
-- Shard by user_id: "list my conversations" is single-shard

-- Session Registry (Redis)
-- Key: session:{user_id}
-- Value: {server_id, connected_at, device_id}
-- TTL: 120s (refreshed by heartbeat)`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Why These Design Choices?</Label>
          <div className="space-y-3">
            {[
              { q: "Shard messages by conversation_id?", a: "All messages for a conversation on the same shard. GET /conversation/:id/messages is a single-shard range scan on (conversation_id, sequence_num). This is THE hot query — must be fast." },
              { q: "Shard participants by user_id?", a: "GET /users/me/conversations (inbox) is the other hot query. Single-shard scan by user_id. Trade-off: 'who's in this group' requires scatter-gather (but rare — cached)." },
              { q: "Why Snowflake IDs?", a: "Time-sortable + globally unique + no coordination. Encode timestamp, machine ID, sequence. Can use message_id for rough ordering (but sequence_num is canonical)." },
              { q: "BLOB for content?", a: "With E2EE, content is an encrypted byte array. Server can't interpret it. Even without E2EE, BLOB handles text, media references, rich messages uniformly." },
              { q: "Why not Cassandra?", a: "Cassandra is actually a great fit: wide-column, time-series, write-optimized, tunable consistency. WhatsApp uses it. Also good: ScyllaDB (C++ Cassandra), HBase, or sharded MySQL." },
              { q: "Unread count?", a: "last_read_seq stored per user per conversation. Unread = max(sequence_num) - last_read_seq. One read, no counting query needed." },
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
          <Label color="#059669">Connection Layer Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Chat Servers are stateful</strong> — each holds WebSocket connections in memory. Can't just round-robin requests. Must use sticky sessions (connection-level, not request-level).</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">~500K connections per server</strong> — limited by memory (~10KB/conn) and file descriptors. Tune ulimit, use epoll/io_uring, minimize per-connection state.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Session Registry</strong> — Redis hash: user_id → server_id. When routing a message, look up recipient's server. O(1) lookup. 150M entries = ~3 GB in Redis.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Graceful drain on deploy</strong> — can't kill a server with 500K connections. Send "reconnect" to all clients, they reconnect to a new server. Drain over 60 seconds.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Message Layer Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Kafka for message routing</strong> — partition by conversation_id. All messages for a conversation go to the same partition (ordering guarantee). Scale partitions independently.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Message DB sharding</strong> — shard by conversation_id. Append-only writes (no updates). Range scan by (conv_id, seq) for history. Cassandra or sharded MySQL.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Group message fan-out</strong> — message to 500-person group: save once in DB, then deliver to each online member's chat server. Offline members: queue. NOT 500 copies in DB.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Read replicas for history</strong> — chat history reads from replicas. Writes (new messages) go to primary. Replication lag OK for history (user scrolling up = few seconds stale is fine).</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Strategy</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Option A: User-Homed Regions ★", d:"Each user is assigned to nearest region. Messages between users in same region: local. Cross-region: routed via backbone.", pros:["Low latency for local conversations","Data locality (GDPR: EU data stays in EU)","Each region is self-contained for local users"], cons:["Cross-region messages have higher latency","User relocation (travel) needs session migration","Cross-region groups are complex"], pick:true },
            { t:"Option B: Follow-the-Sun", d:"All writes go to user's home region. Reads can go to nearest region with replication lag.", pros:["Simple read scaling","No cross-region writes for same-region users"], cons:["Doesn't solve cross-region message routing","Replication lag affects real-time feel"], pick:false },
            { t:"Option C: Global Message Bus", d:"Single Kafka cluster spanning regions. Messages routed globally. Each region has local chat servers.", pros:["Simple routing logic","Single message ordering globally"], cons:["Cross-region Kafka latency (50-150ms)","Single point of failure","Bandwidth costs"], pick:false },
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
      <Card accent="#d97706">
        <Label color="#d97706">Critical: Messages Must Never Be Lost</Label>
        <p className="text-[12px] text-stone-500 mb-4">Chat is the most loss-sensitive system. Users tolerate a slow feed, but a lost message destroys trust. Every design decision prioritizes durability.</p>
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Write-Ahead: Persist Before Deliver</div>
            <p className="text-[11px] text-stone-500">Message saved to DB + Kafka BEFORE attempting WebSocket delivery. If delivery fails, message is still durably stored. Retry from persistent queue.</p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-white p-4">
            <div className="text-[11px] font-bold text-amber-700 mb-1.5">Offline Queue: Never Drop</div>
            <p className="text-[11px] text-stone-500">If recipient is offline, message goes to offline queue (persistent). On reconnect, drain entire queue. Client ACKs each message. Only then remove from queue.</p>
          </div>
          <div className="rounded-lg border border-blue-200 bg-white p-4">
            <div className="text-[11px] font-bold text-blue-700 mb-1.5">Client ACK: Confirm Receipt</div>
            <p className="text-[11px] text-stone-500">Server waits for client ACK after WebSocket push. If no ACK within 5s → retry. After 3 retries → move to offline queue. Belt and suspenders.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#2563eb">Chat Server Failure</Label>
          <ul className="space-y-2.5">
            <Point icon="→" color="#2563eb">Server crashes → all 500K connections drop. Clients detect disconnect.</Point>
            <Point icon="→" color="#2563eb">Clients auto-reconnect with exponential backoff (1s, 2s, 4s, 8s, max 30s).</Point>
            <Point icon="→" color="#2563eb">Load balancer routes to healthy server. New entry in Session Registry.</Point>
            <Point icon="→" color="#2563eb">New server drains offline queue for all reconnecting users.</Point>
            <Point icon="→" color="#2563eb">No messages lost — all were persisted to DB before delivery attempt.</Point>
          </ul>
        </Card>
        <Card>
          <Label color="#d97706">Degradation Ladder</Label>
          <div className="flex flex-col gap-2 mt-1">
            {[
              { label: "Full System", sub: "Real-time WS + presence + receipts", color: "#059669", status: "HEALTHY" },
              { label: "No Presence", sub: "Messages work, online status stale", color: "#d97706", status: "DEGRADED" },
              { label: "Queue Only", sub: "Messages queued, delivered on reconnect", color: "#ea580c", status: "FALLBACK" },
              { label: "Push Only", sub: "APNs/FCM notifications, no WS", color: "#dc2626", status: "EMERGENCY" },
            ].map((t,i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="flex-1 text-center py-2 px-2 rounded-lg border" style={{ borderColor: t.color+"40", background: t.color+"08" }}>
                  <div className="flex items-center justify-center gap-3">
                    <span className="text-[10px] font-mono font-bold" style={{ color: t.color }}>{t.status}</span>
                    <span className="text-[11px] text-stone-600">{t.label}</span>
                    <span className="text-[9px] text-stone-400">{t.sub}</span>
                  </div>
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
      <div className="grid grid-cols-3 gap-5">
        <Card accent="#0284c7">
          <Label color="#0284c7">Key Metrics</Label>
          <div className="space-y-2.5">
            {[
              { metric: "msg.delivery_latency_ms", type: "Histogram", desc: "Time from send to recipient receives. Target: p99 < 500ms for online." },
              { metric: "msg.loss_rate", type: "Counter", desc: "Messages sent but never delivered. Target: 0. Any non-zero is a P0." },
              { metric: "ws.active_connections", type: "Gauge", desc: "Total open WebSockets across all servers. Baseline for capacity planning." },
              { metric: "ws.reconnect_rate", type: "Counter", desc: "Reconnections/sec. Spike = server failure or network issue." },
              { metric: "offline_queue.depth", type: "Gauge", desc: "Pending offline messages. Growing = users not coming back or drain broken." },
              { metric: "presence.heartbeat_miss", type: "Counter", desc: "Missed heartbeats/sec. Spike = network issues or server overload." },
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
              { name: "Message Loss Detected", rule: "msg.loss_rate > 0 for 1min", sev: "P0", action: "ALL HANDS. Check: Kafka lag? DB write failures? WS delivery broken? This is the worst possible failure." },
              { name: "Delivery Latency Spike", rule: "msg.delivery p99 > 2s for 3min", sev: "P2", action: "Check: Kafka consumer lag? Chat server CPU? Session Registry latency? Network between AZs?" },
              { name: "Connection Storm", rule: "ws.reconnect_rate > 10K/sec", sev: "P1", action: "Chat server dying? LB draining? Deploy in progress? Thundering herd on reconnect." },
              { name: "Offline Queue Growing", rule: "offline_queue.depth > 10M for 15min", sev: "P2", action: "Users not reconnecting? Offline drain broken? Push notifications not sending?" },
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
              { q: "User says 'my message didn't deliver'", steps: "Check: message in DB? (yes = persisted). Recipient online? (check session registry). Recipient's chat server received it? (check Kafka consumer lag). Client ACK'd? (check delivery status)." },
              { q: "Messages arriving out of order", steps: "Check: sequence_num in DB correct? Multiple chat servers writing to same conversation? Kafka partition assignment changed? Client reordering logic broken?" },
              { q: "500K reconnections in 1 minute", steps: "Check: which chat server IP are they reconnecting from? Server crash? LB health check failing? SSL cert expired? DNS issue?" },
              { q: "Group messages slow for large groups", steps: "Check: group size (>500?). Fan-out delivery latency. How many members online? Session registry lookup time for N members. Batch lookups." },
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
          { title: "Thundering Herd on Reconnect", sev: "Critical", sevColor: "#dc2626",
            desc: "Chat server crashes. 500K clients reconnect simultaneously to other servers. All at once — CPU spike, session registry flooded, offline queue drain for 500K users.",
            fix: "Client-side jittered exponential backoff: reconnect delay = random(0, min(30, 2^attempt)) seconds. Load balancer connection rate limiting. Chat servers reject above connection threshold (return 503, retry later).",
            code: `Server crash: 500K clients reconnect\n\nWithout jitter:\n  T+0s: 500K connections hit remaining servers\n  T+1s: servers overloaded, crash too\n  T+2s: cascading failure — all servers down\n\nWith jittered backoff:\n  T+0s: ~50K reconnect (random 0-1s)\n  T+1s: ~50K more\n  T+10s: fully recovered, connections spread out` },
          { title: "Message Ordering in Groups", sev: "Critical", sevColor: "#dc2626",
            desc: "User A sends 'Should we go to dinner?' User B sends 'Yes!' — but they arrive in different order for User C. Conversation makes no sense.",
            fix: "Single sequence counter per conversation (Redis INCR, atomic). All messages get a sequence_num from the same counter. Clients display by sequence_num, not timestamp. For Kafka: partition by conversation_id → single partition = total order.",
            code: `Problem:\n  A sends msg (seq=1): "Should we go to dinner?"\n  B sends msg (seq=2): "Yes!"\n  C receives seq=2 before seq=1 (network delay)\n  C sees: "Yes!" then "Should we go to dinner?"\n\nSolution: Client buffers + reorder\n  C receives seq=2, but last_seen_seq=0\n  C detects gap: missing seq=1\n  C buffers seq=2, requests seq=1 from server\n  C renders in order: seq=1, seq=2` },
          { title: "Hot Conversation (Viral Group)", sev: "High", sevColor: "#d97706",
            desc: "Group with 10K members all sending messages during an event. One conversation_id = one DB shard. That shard is overwhelmed with writes + reads simultaneously.",
            fix: "Shard groups >1K members differently: buffer writes in Redis, batch-insert to DB every 100ms. Rate-limit messages in large groups (e.g., 1 msg/sec per member). Read path: cache recent messages for hot groups. Consider separate 'channel' model for very large groups (Telegram, Discord).",
            code: `10K member group, event happening:\n  200 members typing simultaneously\n  200 msg/sec to ONE conversation shard\n  10K members reading → 10K reads/sec\n  Single DB shard: overwhelmed\n\nSolution: Tiered group handling\n  < 500 members: normal group\n  500-5K: buffered writes + rate limit\n  > 5K: channel model (broadcast, not chat)` },
          { title: "Session Registry Inconsistency", sev: "High", sevColor: "#d97706",
            desc: "Session Registry says user is on Server A, but their connection moved to Server B (after reconnect). Messages routed to Server A → dropped. User doesn't receive messages.",
            fix: "Server B registers new session on connect, overwriting Server A's entry (atomic SET). Server A detects stale connection on next write (connection closed → cleanup). Heartbeat in Session Registry with TTL — stale entries auto-expire. On delivery failure → re-lookup Session Registry.",
            code: `Timeline:\n  T=0: User on Server A, registry: user→A\n  T=1: Server A network partition\n  T=2: Client reconnects to Server B\n  T=3: Server B writes registry: user→B\n  T=3: Message arrives, lookup → Server B ✓\n\nEdge case: T=2.5 before registry update:\n  Message routed to Server A → fails\n  Retry: re-lookup registry → now says Server B\n  Deliver to Server B → success\n\nAlways re-lookup on delivery failure.` },
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

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   LLD — Implementation Architecture Sections
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

function ServicesSection() {
  return (
    <div className="space-y-5">
      <Card accent="#0f766e">
        <Label color="#0f766e">Service Decomposition — What We Actually Build</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { name: "Chat Server (Gateway)", owns: "WebSocket connections, message send/receive, heartbeat handling", tech: "Erlang/Go/Rust with epoll/io_uring", api: "WebSocket: message.send, message.ack, heartbeat", scale: "Horizontal — add servers for more connections", stateful: true,
              modules: ["WebSocket Handler (upgrade, read, write frames)", "Connection Manager (per-user state, heartbeat tracking)", "Message Router (lookup recipient → route to correct server)", "Session Registrar (register/deregister in Redis on connect/disconnect)", "Serializer (protobuf encode/decode for wire format)", "Backpressure Controller (slow down client if server overloaded)"] },
            { name: "Message Service", owns: "Persist messages, assign sequence numbers, manage offline queue", tech: "Go/Java + Cassandra/sharded MySQL + Kafka", api: "gRPC: SaveMessage, GetHistory, DrainOffline", scale: "Horizontal — shard by conversation_id", stateful: false,
              modules: ["Message Writer (validate, assign seq, persist to DB)", "Sequence Generator (atomic INCR per conversation in Redis)", "Offline Queue Manager (enqueue for offline users, drain on reconnect)", "History Server (paginated read by conv_id + seq range)", "Deduplication Cache (request_id → message_id, 5min TTL)", "Fanout Coordinator (group: resolve members → enqueue per-member delivery)"] },
            { name: "Presence Service", owns: "Online/offline status, last seen, typing indicators", tech: "Go + Redis (presence store)", api: "gRPC: SetOnline, SetOffline, GetPresence, Heartbeat", scale: "Horizontal — shard presence by user_id", stateful: false,
              modules: ["Heartbeat Processor (update last_heartbeat timestamp)", "Stale Sweeper (background: mark users offline after 90s no heartbeat)", "Presence Publisher (notify contacts on status change)", "Typing Indicator Relay (ephemeral — not persisted, fire-and-forget)", "Contact Resolver (who needs to know about this user's status?)", "Rate Limiter (debounce rapid online/offline flaps)"] },
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
        <Label color="#9333ea">System Architecture — Message Delivery Pipeline</Label>
        <svg viewBox="0 0 720 370" className="w-full">
          {/* Connection Layer */}
          <rect x={5} y={5} width={340} height={165} rx={8} fill="#05966904" stroke="#05966920" strokeWidth={1} strokeDasharray="4,3"/>
          <text x={15} y={22} fill="#059669" fontSize="10" fontWeight="700" fontFamily="monospace">CONNECTION LAYER (Stateful)</text>

          {/* Processing Layer */}
          <rect x={355} y={5} width={360} height={165} rx={8} fill="#2563eb04" stroke="#2563eb20" strokeWidth={1} strokeDasharray="4,3"/>
          <text x={365} y={22} fill="#2563eb" fontSize="10" fontWeight="700" fontFamily="monospace">PROCESSING + STORAGE LAYER</text>

          {/* Data Layer */}
          <rect x={5} y={180} width={710} height={100} rx={8} fill="#d9770604" stroke="#d9770620" strokeWidth={1} strokeDasharray="4,3"/>
          <text x={15} y={197} fill="#d97706" fontSize="10" fontWeight="700" fontFamily="monospace">DATA + EXTERNAL SERVICES</text>

          {/* Phone A */}
          <rect x={15} y={55} width={65} height={36} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={47} y={72} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Phone A</text>
          <text x={47} y={84} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">sender</text>

          {/* Phone B */}
          <rect x={15} y={110} width={65} height={36} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={47} y={127} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Phone B</text>
          <text x={47} y={139} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">recipient</text>

          {/* LB */}
          <rect x={105} y={75} width={55} height={40} rx={6} fill="#6366f110" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={132} y={93} textAnchor="middle" fill="#6366f1" fontSize="8" fontWeight="600" fontFamily="monospace">Load</text>
          <text x={132} y={105} textAnchor="middle" fill="#6366f1" fontSize="8" fontWeight="600" fontFamily="monospace">Balancer</text>

          {/* Chat Server 1 */}
          <rect x={190} y={40} width={85} height={36} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={232} y={57} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Chat Srv 1</text>
          <text x={232} y={70} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">Phone A's WS</text>

          {/* Chat Server 2 */}
          <rect x={190} y={90} width={85} height={36} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={232} y={107} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Chat Srv 2</text>
          <text x={232} y={120} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">Phone B's WS</text>

          {/* Chat Server N */}
          <rect x={190} y={135} width={85} height={28} rx={6} fill="#05966908" stroke="#05966950" strokeWidth={1} strokeDasharray="3,2"/>
          <text x={232} y={153} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">Chat Srv N...</text>

          {/* Session Registry */}
          <rect x={305} y={55} width={80} height={38} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={345} y={72} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Session</text>
          <text x={345} y={84} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Registry</text>
          <text x={345} y={105} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">user → server</text>
          <text x={345} y={115} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">(Redis)</text>

          {/* Message Service */}
          <rect x={370} y={35} width={90} height={40} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
          <text x={415} y={52} textAnchor="middle" fill="#7c3aed" fontSize="9" fontWeight="600" fontFamily="monospace">Message Svc</text>
          <text x={415} y={65} textAnchor="middle" fill="#7c3aed80" fontSize="7" fontFamily="monospace">persist + sequence</text>

          {/* Presence Service */}
          <rect x={370} y={90} width={90} height={36} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={415} y={107} textAnchor="middle" fill="#0891b2" fontSize="9" fontWeight="600" fontFamily="monospace">Presence Svc</text>
          <text x={415} y={120} textAnchor="middle" fill="#0891b280" fontSize="7" fontFamily="monospace">online/offline</text>

          {/* Group Service */}
          <rect x={370} y={135} width={90} height={28} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={415} y={153} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Group Svc</text>

          {/* Kafka */}
          <rect x={490} y={35} width={80} height={40} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={530} y={52} textAnchor="middle" fill="#ea580c" fontSize="9" fontWeight="600" fontFamily="monospace">Kafka</text>
          <text x={530} y={65} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">msg queue</text>

          {/* Push Service */}
          <rect x={490} y={90} width={80} height={36} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={530} y={107} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Push Svc</text>
          <text x={530} y={120} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">APNs / FCM</text>

          {/* Dedup Cache */}
          <rect x={490} y={135} width={80} height={28} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={530} y={153} textAnchor="middle" fill="#78716c" fontSize="8" fontWeight="600" fontFamily="monospace">Dedup Cache</text>

          {/* Key Dist */}
          <rect x={600} y={55} width={80} height={36} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={640} y={72} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Key Dist Svc</text>
          <text x={640} y={84} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">E2EE keys</text>

          {/* Data Layer */}
          <rect x={20} y={205} width={85} height={36} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={62} y={222} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Message DB</text>
          <text x={62} y={234} textAnchor="middle" fill="#78716c80" fontSize="7" fontFamily="monospace">Cassandra</text>

          <rect x={130} y={205} width={85} height={36} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={172} y={222} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">User / Group</text>
          <text x={172} y={234} textAnchor="middle" fill="#78716c80" fontSize="7" fontFamily="monospace">MySQL</text>

          <rect x={240} y={205} width={85} height={36} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={282} y={222} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Presence</text>
          <text x={282} y={234} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">Redis</text>

          <rect x={350} y={205} width={85} height={36} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={392} y={222} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Offline Queue</text>
          <text x={392} y={234} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">Redis / SQS</text>

          <rect x={460} y={205} width={85} height={36} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={502} y={222} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Media Store</text>
          <text x={502} y={234} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">S3 + CDN</text>

          <rect x={570} y={205} width={85} height={36} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={612} y={222} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Key Store</text>
          <text x={612} y={234} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">PostgreSQL</text>

          {/* Arrows */}
          <defs><marker id="ah-svc" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          {/* Client → LB */}
          <line x1={80} y1={73} x2={105} y2={88} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={80} y1={128} x2={105} y2={102} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          {/* LB → Chat Servers */}
          <line x1={160} y1={88} x2={190} y2={58} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={160} y1={98} x2={190} y2={108} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          {/* Chat Server → Session Registry */}
          <line x1={275} y1={58} x2={305} y2={68} stroke="#c026d350" strokeWidth={1} markerEnd="url(#ah-svc)" strokeDasharray="3,2"/>
          <line x1={275} y1={108} x2={305} y2={78} stroke="#c026d350" strokeWidth={1} markerEnd="url(#ah-svc)" strokeDasharray="3,2"/>
          {/* Chat Server 1 → Message Svc */}
          <line x1={275} y1={50} x2={370} y2={50} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          {/* Chat Server 1 → Chat Server 2 (internal routing) */}
          <line x1={245} y1={76} x2={245} y2={90} stroke="#059669" strokeWidth={1.5} markerEnd="url(#ah-svc)"/>
          <text x={255} y={86} fill="#05966990" fontSize="7" fontFamily="monospace">route</text>
          {/* Message Svc → Kafka */}
          <line x1={460} y1={55} x2={490} y2={55} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          {/* Message Svc → DB */}
          <line x1={415} y1={75} x2={62} y2={205} stroke="#78716c40" strokeWidth={1} markerEnd="url(#ah-svc)" strokeDasharray="3,2"/>
          {/* Presence → Redis */}
          <line x1={415} y1={126} x2={282} y2={205} stroke="#d9770640" strokeWidth={1} markerEnd="url(#ah-svc)" strokeDasharray="3,2"/>
          {/* Push Svc → external */}
          <line x1={570} y1={108} x2={600} y2={108} stroke="#d97706" strokeWidth={1} markerEnd="url(#ah-svc)"/>
          <text x={605} y={112} fill="#d97706" fontSize="7" fontFamily="monospace">→ Apple/Google</text>

          {/* Legend */}
          <rect x={15} y={260} width={695} height={95} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={25} y={278} fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Online delivery: Phone A → WS → Chat Srv 1 → Message Svc (persist + seq) → Session Registry (lookup B) → Chat Srv 2 → WS → Phone B</text>
          <text x={25} y={295} fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Offline delivery: ... → Session Registry (B not found) → Offline Queue → Push Notification → Phone B reconnects → drain queue</text>
          <text x={25} y={312} fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Session Registry: user_id → chat_server_id (Redis). THE critical lookup — "which server holds this user's WebSocket?"</text>
          <text x={25} y={329} fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Key insight: Chat Servers are STATEFUL (hold WS connections). All other services are stateless. This is what makes chat different.</text>
          <text x={25} y={346} fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Message stored ONCE (not per-recipient). Group fan-out is delivery-only — storage is single-write to Message DB.</text>
        </svg>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#2563eb">
          <Label color="#2563eb">Supporting Services</Label>
          <div className="space-y-3">
            {[
              { name: "Push Notification Service", role: "Send APNs (iOS) / FCM (Android) notifications for offline users. Batched, rate-limited per device.", tech: "Go workers + APNs HTTP/2 + FCM gRPC", critical: true },
              { name: "Group Service", role: "Create/manage groups. Membership CRUD. Resolve group → member list for fan-out. Cache hot groups.", tech: "Go + MySQL + Redis cache", critical: true },
              { name: "Media Service", role: "Pre-signed upload URLs (S3), thumbnail generation, virus scan, CDN URL generation for download.", tech: "Go + S3 + Lambda (thumbnails) + CloudFront", critical: false },
              { name: "Key Distribution Service", role: "Store and serve public keys for E2EE. X3DH key bundles. One-time prekey replenishment.", tech: "Go + PostgreSQL (key store)", critical: true },
              { name: "Sync Service", role: "Multi-device sync. When new device logs in, catch up on conversation history + state.", tech: "Go + Message DB read path", critical: false },
              { name: "Abuse / Spam Detection", role: "ML model scores messages for spam patterns. Rate-limit flagged senders. Report pipeline.", tech: "Python ML + Kafka consumer", critical: false },
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
                  { route: "Chat Server → Session Registry", proto: "Redis", timeout: "5ms", fail: "Retry 1x, then broadcast (expensive)" },
                  { route: "Chat Server → Message Svc", proto: "gRPC", timeout: "100ms", fail: "Return error to sender, client retries" },
                  { route: "Message Svc → Kafka", proto: "Kafka", timeout: "Async", fail: "Retry 3x, then DLQ" },
                  { route: "Message Svc → DB (write)", proto: "CQL/SQL", timeout: "200ms", fail: "Retry. If persistent: alert P0 (message loss risk)" },
                  { route: "Chat Server → Chat Server", proto: "gRPC/internal", timeout: "50ms", fail: "Message to offline queue" },
                  { route: "Chat Server → Presence Svc", proto: "gRPC", timeout: "50ms", fail: "Stale presence (acceptable)" },
                  { route: "Push Svc → APNs/FCM", proto: "HTTP/2", timeout: "5s", fail: "Retry 3x. Exponential backoff." },
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
  const [flow, setFlow] = useState("send_online");
  const flows = {
    send_online: { title: "1:1 Message — Both Online", steps: [
      { actor: "Sender (Phone A)", action: "User types message, taps send. Client generates request_id='req_abc'", type: "request" },
      { actor: "Phone A → Chat Server 1", action: "WS frame: {type: message.send, request_id: req_abc, conv_id: conv_789, content: 'Hey!'}", type: "request" },
      { actor: "Chat Server 1", action: "Deduplicate by request_id (check Redis cache, 5min TTL)", type: "process" },
      { actor: "Chat Server 1 → Message Svc", action: "gRPC: SaveMessage(conv_789, sender=42, content='Hey!')", type: "request" },
      { actor: "Message Service", action: "Assign sequence_num (Redis INCR seq:conv_789 → 14523). Generate message_id (Snowflake).", type: "process" },
      { actor: "Message Svc → DB", action: "INSERT message (msg_id, conv_789, sender_42, seq=14523, content, ts)", type: "request" },
      { actor: "Message Svc → Kafka", action: "Produce: {msg_id, conv_789, recipient=user_789, seq=14523}", type: "process" },
      { actor: "Chat Server 1 → Sender", action: "WS: {type: message.ack, request_id: req_abc, message_id: msg_xyz, status: 'sent'} — ✓ single tick", type: "success" },
      { actor: "Chat Server 1", action: "Lookup Session Registry: user_789 → Chat Server 2", type: "process" },
      { actor: "Chat Server 1 → Chat Server 2", action: "Internal gRPC: DeliverMessage(user_789, message)", type: "request" },
      { actor: "Chat Server 2 → Phone B", action: "WS push: {type: message.new, message_id, content: 'Hey!', seq: 14523}", type: "success" },
      { actor: "Phone B → Chat Server 2", action: "WS: {type: message.ack, message_id: msg_xyz} — client received it", type: "process" },
      { actor: "Chat Server 2 → Sender", action: "Relay delivery receipt: status = 'delivered' — ✓✓ double tick", type: "success" },
    ]},
    send_offline: { title: "1:1 Message — Recipient Offline", steps: [
      { actor: "Sender (Phone A)", action: "Send message to user_789 who is offline", type: "request" },
      { actor: "Chat Server 1 → Message Svc", action: "Save message (same as online path). Persist first.", type: "request" },
      { actor: "Chat Server 1 → Sender", action: "ACK: status='sent' ✓ (server has it — sender's job is done)", type: "success" },
      { actor: "Chat Server 1", action: "Lookup Session Registry: user_789 → NOT FOUND (offline)", type: "check" },
      { actor: "Message Service", action: "Enqueue in offline queue: offline:{user_789} → [msg_xyz, ...]", type: "process" },
      { actor: "Push Service", action: "Send push notification via APNs/FCM: 'User42: Hey!'", type: "request" },
      { actor: "Phone B", action: "User sees push notification, opens app. Phone connects WebSocket to Chat Server 3.", type: "process" },
      { actor: "Chat Server 3", action: "Register session: user_789 → Server 3. Update presence: online.", type: "process" },
      { actor: "Chat Server 3 → Message Svc", action: "DrainOfflineQueue(user_789)", type: "request" },
      { actor: "Message Service", action: "Return all queued messages: [msg_xyz, ...]. Clear queue after ACK.", type: "process" },
      { actor: "Chat Server 3 → Phone B", action: "WS push: all pending messages in order by sequence_num", type: "success" },
      { actor: "Phone B → Server 3", action: "ACK all messages. Server clears offline queue. Send delivery receipt to sender.", type: "success" },
    ]},
    group: { title: "Group Message — 200 Members", steps: [
      { actor: "Sender", action: "Send message to group_456 (200 members)", type: "request" },
      { actor: "Chat Server → Message Svc", action: "Save message ONCE: conv_id=group_456, seq=N", type: "request" },
      { actor: "Message Service → Group Svc", action: "Resolve members: get_members(group_456) → [user_1, ..., user_200]", type: "request" },
      { actor: "Message Service", action: "For each member: lookup Session Registry → partition into online vs offline", type: "process" },
      { actor: "Delivery", action: "Online (120 users): route message to their respective Chat Servers via internal gRPC", type: "request" },
      { actor: "Delivery", action: "Offline (80 users): enqueue in each user's offline queue. Batch push notifications.", type: "process" },
      { actor: "Note", action: "Message stored ONCE in DB. Delivery is fan-out, but storage is single-write.", type: "check" },
      { actor: "Note", action: "Delivery receipts in groups: only 'delivered' if ALL members received. Often simplified to 'sent' only.", type: "check" },
    ]},
    reconnect: { title: "Client Reconnect + Sync", steps: [
      { actor: "Phone B", action: "App was backgrounded for 2 hours. User opens app. WS connection was closed.", type: "process" },
      { actor: "Phone B", action: "Initiate WebSocket connection to Load Balancer", type: "request" },
      { actor: "Load Balancer", action: "Route to healthy Chat Server (least connections or round-robin)", type: "process" },
      { actor: "Chat Server 4", action: "WS handshake + auth (JWT from stored token). Register in Session Registry.", type: "auth" },
      { actor: "Chat Server 4 → Presence", action: "SetOnline(user_789). Notify contacts.", type: "process" },
      { actor: "Chat Server 4 → Message Svc", action: "DrainOfflineQueue(user_789) — get all messages since disconnect", type: "request" },
      { actor: "Chat Server 4 → Message Svc", action: "GetConversationUpdates(user_789, since=last_sync_timestamp) — any new conversations?", type: "request" },
      { actor: "Chat Server 4 → Phone B", action: "Push: all offline messages + updated conversation list + unread counts", type: "success" },
      { actor: "Phone B", action: "Render messages, update badges, show unread counts. Send ACK for all.", type: "success" },
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
          <Label color="#b45309">K8s Deployment — Chat Server (Stateful!)</Label>
          <CodeBlock title="Chat Servers — StatefulSet with graceful drain" code={`apiVersion: apps/v1
kind: StatefulSet           # Stateful — sticky WS connections
metadata:
  name: chat-server
spec:
  replicas: 300             # 500K connections each = 150M total
  podManagementPolicy: Parallel
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1     # Drain 1 pod at a time
  template:
    spec:
      terminationGracePeriodSeconds: 120  # 2 min drain
      containers:
      - name: chat
        image: chat/server:v3.1
        lifecycle:
          preStop:            # CRITICAL: drain connections
            exec:
              command: ["/drain.sh"]
              # drain.sh:
              # 1. Stop accepting new connections
              # 2. Send "reconnect" frame to all clients
              # 3. Wait for clients to disconnect (60s)
              # 4. Deregister from Session Registry
              # 5. Exit
        ports:
        - containerPort: 8080  # WS port
        resources:
          requests:
            memory: "8Gi"     # ~500K conns × 10KB + overhead
            cpu: "4"
          limits:
            memory: "10Gi"
        env:
        - name: MAX_CONNECTIONS
          value: "500000"
        - name: EPOLL_ENABLED
          value: "true"`} />
          <div className="mt-3 space-y-1.5">
            <Point icon="⚠" color="#b45309">preStop hook sends "reconnect" frame — clients reconnect to other servers gracefully</Point>
            <Point icon="⚠" color="#b45309">terminationGracePeriodSeconds: 120 — long enough for 500K clients to drain</Point>
            <Point icon="⚠" color="#b45309">ulimit -n must be &gt; 600K. Set in pod security context or node config.</Point>
          </div>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Security — E2EE + Transport</Label>
          <div className="space-y-3">
            {[
              { layer: "Client → Server Transport", details: ["TLS 1.3 for WebSocket (wss://)", "Certificate pinning in mobile apps (prevent MITM)", "Token-based auth on WS upgrade (JWT in query param or first frame)", "Re-authenticate on reconnect (token refresh)"] },
              { layer: "End-to-End Encryption", details: ["Signal Protocol (X3DH + Double Ratchet)", "Server stores only encrypted blobs — cannot read content", "Key Distribution Service stores public keys only", "Forward secrecy: each message uses unique key", "Verify identity: QR code / safety number comparison"] },
              { layer: "Data at Rest", details: ["Message DB: encrypted blobs (already E2EE)", "Metadata: sender, recipient, timestamp visible to server (required for routing)", "Metadata minimization: WhatsApp deletes server-side messages after delivery", "Backups: client-side encrypted backup to iCloud/Google Drive (optional)"] },
              { layer: "Abuse Prevention (with E2EE)", details: ["Can't inspect content — but CAN detect patterns", "Rate limiting: max messages/sec per user", "Reported messages: recipient forwards decrypted message to abuse team", "Contact-graph analysis: detect mass-messaging spam patterns", "Device attestation: prevent automated/bot accounts"] },
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
                { b: "WS Connection Capacity", s: "New connections refused, 503s", f: "Add chat servers. Ensure LB least-connections routing.", p: "Each server needs time to fill. Don't over-provision — 500K idle connections waste memory." },
                { b: "Session Registry (Redis)", s: "Lookup latency > 5ms", f: "Shard Redis by user_id. Add read replicas.", p: "Write-heavy during reconnect storms. Pipeline writes. Don't use Cluster mode for this — sentinel is simpler." },
                { b: "Message DB Write Throughput", s: "Write latency > 200ms", f: "More Cassandra nodes. Or buffer writes in Kafka, batch-insert.", p: "Sharded by conv_id: hot groups create hot shards. Consider separate table/shard for groups > 1K." },
                { b: "Kafka Consumer Lag", s: "Messages delayed by seconds/minutes", f: "More partitions + consumers. Or: direct server-to-server delivery, use Kafka only as durability backup.", p: "Adding partitions causes rebalance. Use cooperative-sticky assignor." },
                { b: "Push Notification Rate", s: "APNs/FCM rate limited, notifications delayed", f: "Batch notifications. Collapse multiple notifications per user. Respect per-device rate limits.", p: "APNs has strict per-device limits. Don't spam — aggregate: 'You have 5 new messages' not 5 separate pushes." },
                { b: "Group Fan-Out", s: "Large group messages slow to deliver", f: "Batch session lookups. Pipeline internal gRPC calls. Rate-limit sends in large groups.", p: "10K member group = 10K session lookups + 10K deliveries per message. Consider pub/sub model for huge groups." },
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
            { title: "New Year's Eve Midnight Spike", symptom: "Message volume 20× normal at midnight. Kafka partitions maxed out. Messages delayed by 30+ seconds globally.",
              cause: "Entire world sends 'Happy New Year' simultaneously. 700K msg/sec peaks to 14M msg/sec. Kafka brokers can't handle partition throughput.",
              fix: "Pre-scale 2 weeks before: add Kafka partitions + brokers, add chat servers, pre-warm connection capacity. Rate-limit non-critical features (typing indicators, presence updates). Priority queue: messages first, receipts second, presence last.",
              quote: "We war-gamed New Year's every October. Still got surprised by the volume. Best lesson: disable typing indicators during spike events — they're 40% of WS traffic." },
            { title: "SSL Certificate Expiry Kills All Connections", symptom: "All 150M WebSocket connections drop simultaneously. Reconnection fails — SSL handshake error. Total outage.",
              cause: "TLS certificate expired. Certificate renewal automation failed silently 2 months ago. Alert was misconfigured — nobody noticed until connections dropped.",
              fix: "Certificate monitoring with 30-day, 14-day, 7-day, 1-day alerts on separate channels. Auto-renewal with verification. Multiple certificate providers. Pin to intermediate cert, not leaf. Cert expiry dashboard visible to entire on-call team.",
              quote: "2 billion users couldn't send messages because a cron job for cert renewal had a typo in the path. We now have 6 independent cert monitors." },
            { title: "Message DB Migration Gone Wrong", symptom: "Read latency for chat history spikes 100×. Users can't scroll up in conversations. New messages still deliver (different path).",
              cause: "Online schema migration (adding an index) locked the message table. Cassandra compaction storm during migration consumed all disk I/O.",
              fix: "Never run schema migrations during peak hours. Use ghost tables / online DDL tools. Test migration on production-size replica first. Have a kill switch to abort migration. Throttle Cassandra compaction during migrations.",
              quote: "We added an index at 2pm. Seemed safe — Cassandra is online-DDL-friendly. Except our table had 4 TB and compaction ate all I/O for 3 hours." },
            { title: "Presence Service Storm After Regional Failover", symptom: "Presence shows all users as offline after failover. Then 50M users flip to 'online' simultaneously. Contact list UI freezes on all clients.",
              cause: "Regional failover moved all connections. Session Registry cleared. All users re-register presence. Each re-registration notifies all contacts. 50M users × 200 avg contacts = 10B presence notifications.",
              fix: "Batch presence notifications after failover. Don't notify for each individual user — aggregate: 'refresh your contact list' event after 60s settling window. Client-side: don't animate 200 contacts going online simultaneously — debounce UI updates.",
              quote: "The chat worked fine after failover. But every user's phone froze for 30 seconds rendering 200 contacts going online at once. We killed the client's battery too." },
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
        { t: "End-to-End Encryption", d: "Signal Protocol: X3DH key agreement + Double Ratchet for forward secrecy. Server stores encrypted blobs only.", detail: "Key Distribution Service for public keys. One-time prekeys. Multi-device: encrypt N times. Group: Sender Key protocol.", effort: "Hard" },
        { t: "Voice / Video Calls", d: "WebRTC for peer-to-peer media. STUN/TURN servers for NAT traversal. Signaling via existing WebSocket channel.", detail: "Separate system from messaging. Offer/answer SDP exchange via WS. TURN relay for ~30% of calls (symmetric NAT). SFU for group calls.", effort: "Hard" },
        { t: "Message Reactions & Replies", d: "Emoji reactions on messages. Quote-reply to specific messages. Thread support for groups.", detail: "Reactions: separate event type, not a message. Store as array on message. Quote-reply: include referenced message_id + preview.", effort: "Easy" },
        { t: "Disappearing Messages", d: "Messages auto-delete after 24h/7d/30d. Timer starts on read (not send). Client + server-side enforcement.", detail: "Client deletes locally on timer. Server TTL on message rows. Can't guarantee deletion (screenshots), but removes from DB + UI.", effort: "Medium" },
        { t: "Multi-Device Sync", d: "Same account on phone + tablet + desktop. All devices see all messages, read receipts, and state changes.", detail: "Each device has own WS connection + encryption keys. Server delivers to ALL connected devices. Sync on new device login: replay recent history.", effort: "Hard" },
        { t: "Message Search", d: "Full-text search across all conversations. 'Find that restaurant name from last week.'", detail: "Challenge with E2EE: server can't index content. Client-side search only (or: separate search index with user's permission, not E2EE'd).", effort: "Medium" },
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
    { q:"Why WebSocket and not HTTP long polling or SSE?", a:"WebSocket is full-duplex: both client and server can send at any time without waiting. Long polling has ~1s latency per message and creates enormous request overhead at scale (150M connections × polling = 150M req/sec just for keep-alive). SSE is server-push only — client can't send messages through it. WebSocket: one persistent connection, sub-100ms latency, minimal overhead.", tags:["design"] },
    { q:"How do you handle message ordering?", a:"Server-assigned sequence number per conversation, using atomic Redis INCR. All messages get a monotonically increasing seq from one source. Clients display by seq, not timestamp. If messages arrive out of order (seq 14 before 13), client buffers and reorders. For gap detection: client requests missing messages from server.", tags:["consistency"] },
    { q:"How do you guarantee no message loss?", a:"Three-phase: (1) Client retries until server ACKs (at-least-once send). (2) Server persists to DB BEFORE delivery attempt (durability first). (3) Offline queue for unreachable recipients, drained on reconnect with client ACK. Deduplication by request_id (send) and message_id (receive) prevents duplicates.", tags:["reliability"] },
    { q:"How does group messaging work differently from 1:1?", a:"Message stored ONCE in DB (not per-recipient). Fan-out is delivery-only: resolve members → lookup online/offline → deliver to online via WS, queue for offline. Big difference from news feed fan-out: we don't pre-compute anything. It's message-by-message routing. For very large groups (>5K), consider channel/broadcast model.", tags:["design"] },
    { q:"How do you handle the chat server being stateful?", a:"Session Registry (Redis) maps user→server. On message send: lookup recipient's server, route message there. On server crash: clients reconnect to any server, re-register in registry. Graceful shutdown: drain connections over 60s. Key insight: only the WS connection is stateful — all data is in external stores (DB, Redis, Kafka).", tags:["scalability"] },
    { q:"How does end-to-end encryption work with groups?", a:"Signal Protocol's Sender Key: sender creates a symmetric 'sender key' for the group, distributes it to each member via their individual E2EE channel. Subsequent messages encrypted once with sender key (all members can decrypt). When a member leaves: new sender key generated and distributed. More efficient than encrypting N times per message.", tags:["security"] },
    { q:"How would you implement read receipts?", a:"Recipient's client sends {type: message.read, conv_id, last_read_seq} via WebSocket. Server updates participants.last_read_seq in DB. Routes receipt to sender's chat server. Sender's client shows blue double-tick. For groups: track per-member read status (but showing in UI is optional — WhatsApp shows it, Telegram doesn't).", tags:["design"] },
    { q:"How does typing indicator work?", a:"Ephemeral event — NOT persisted. Client sends {type: typing.start} when user begins typing. Server routes to other participant(s) via WS. Auto-expires after 5s if no typing.stop received. Fire-and-forget: no retry, no persistence, no offline queue. If it's lost, no harm. During load spikes: first thing to disable.", tags:["design"] },
    { q:"WhatsApp uses Erlang — why?", a:"Erlang's BEAM VM excels at: millions of lightweight processes (one per connection), hot code reloading (deploy without disconnecting users), fault tolerance (supervisor trees auto-restart crashed processes), and soft real-time guarantees. WhatsApp famously ran 2M connections per server on Erlang. Alternatives today: Go (goroutines), Rust (async/tokio), Elixir (modern Erlang).", tags:["tech"] },
    { q:"How would you handle message search with E2EE?", a:"Server can't search encrypted content. Options: (1) Client-side search only — index locally on device. Works but slow for large history. (2) User opts in to server-side search index (not E2EE for that index — privacy tradeoff, must be explicit). (3) Encrypted search (homomorphic or encrypted indexes) — research-grade, not production-ready. Most E2EE apps use option 1.", tags:["security"] },
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

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
const SECTION_COMPONENTS = {
  concept: ConceptSection, requirements: RequirementsSection, capacity: CapacitySection,
  api: ApiSection, design: DesignSection, algorithm: AlgorithmSection, data: DataModelSection,
  scalability: ScalabilitySection, availability: AvailabilitySection, observability: ObservabilitySection,
  watchouts: WatchoutsSection, services: ServicesSection, flows: FlowsSection,
  deployment: DeploymentSection, ops: OpsSection,
  enhancements: EnhancementsSection, followups: FollowupsSection,
};

export default function ChatSystemSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Chat System (WhatsApp)</h1>
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
