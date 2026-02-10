import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   GOOGLE DOCS (Collaborative) — System Design Reference
   Pearl white theme · 17 sections (HLD + LLD)
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",             icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",         icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",  icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",           icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",    icon: "🏗️", color: "#9333ea" },
  { id: "algorithm",     label: "OT & CRDT Deep Dive",  icon: "⚙️", color: "#c026d3" },
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
            <Label>What is a Collaborative Document Editor?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A real-time collaborative document editing system where multiple users simultaneously view and edit the same document. Think Google Docs, Notion, or Microsoft Office Online. The core challenge: merging concurrent edits from multiple users in real time without losing anyone's changes or corrupting the document.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Unlike a simple CRUD app, collaborative editing requires conflict resolution algorithms (OT or CRDT), real-time presence and cursor tracking, document versioning, and eventual consistency guarantees — all while maintaining sub-200ms latency so edits feel instantaneous.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard?</Label>
            <ul className="space-y-2.5">
              <Point icon="🔄" color="#0891b2">Conflict resolution — two users type at the same position simultaneously. Their edits must merge correctly without overwriting either change.</Point>
              <Point icon="⏱️" color="#0891b2">Real-time sync — edits must propagate to all collaborators within 200ms. Any noticeable lag destroys the "live editing" feel.</Point>
              <Point icon="📐" color="#0891b2">Intention preservation — if user A inserts "hello" at position 5 while user B deletes position 3, A's insert must shift to position 4. The system must understand what users *intended*.</Point>
              <Point icon="📄" color="#0891b2">Rich document model — not just plain text. Bold, italic, headings, tables, images, comments, suggestions — each with unique merge semantics.</Point>
              <Point icon="💾" color="#0891b2">Version history — full undo/redo per user, document snapshots, ability to revert to any previous version without corrupting collaborators.</Point>
              <Point icon="🌐" color="#0891b2">Scale — a single document with 100 concurrent editors, each sending 2-5 ops/sec = 200-500 operations/sec that must be sequenced and broadcast in order.</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Scale</Label>
            <div className="space-y-2.5">
              {[
                { co: "Google Docs", scale: "1B+ users, real-time collab", detail: "OT-based, Jupiter protocol" },
                { co: "Notion", scale: "100M+ users, block-based", detail: "CRDT-inspired, block model" },
                { co: "MS Office", scale: "400M+ users, OOXML format", detail: "OT + Fluid Framework" },
                { co: "Figma", scale: "4M+ users, design collab", detail: "CRDT (Yjs-like), multiplayer" },
                { co: "Dropbox Paper", scale: "50M+ users, lightweight", detail: "OT-based, Markdown-backed" },
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
              <rect x={10} y={10} width={160} height={45} rx={6} fill="#2563eb08" stroke="#2563eb" strokeWidth={1.5}/>
              <text x={90} y={28} textAnchor="middle" fill="#2563eb" fontSize="10" fontWeight="700" fontFamily="monospace">OT (Op. Transform)</text>
              <text x={90} y={44} textAnchor="middle" fill="#2563eb80" fontSize="8" fontFamily="monospace">✔ Proven, central server</text>

              <rect x={190} y={10} width={160} height={45} rx={6} fill="#05966908" stroke="#059669" strokeWidth={1.5}/>
              <text x={270} y={28} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="700" fontFamily="monospace">CRDT</text>
              <text x={270} y={44} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">✔ Decentralized, P2P-ready</text>

              <rect x={60} y={68} width={240} height={42} rx={6} fill="#9333ea08" stroke="#9333ea" strokeWidth={1.5}/>
              <text x={180} y={85} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">OT ★ for server-centric (Google Docs model)</text>
              <text x={180} y={100} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">Central server = simpler conflict resolution</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Google, Microsoft, Notion, Figma — top-tier SD question</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Scope the Document Model</div>
            <p className="text-[12px] text-stone-500 mt-0.5">"Design Google Docs" is enormous. Clarify: plain text or rich text? Real-time collaboration or just shared editing? Comments/suggestions? Offline editing? For a 45-min interview, focus on <strong>real-time collaborative rich-text editing + conflict resolution + presence</strong>. Version history and offline are follow-ups.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Create, open, and edit documents with rich text formatting (bold, italic, headings, lists)</Point>
            <Point icon="2." color="#059669">Real-time collaborative editing — multiple users see each other's changes instantly</Point>
            <Point icon="3." color="#059669">Cursor presence — see where other collaborators are typing and their selections</Point>
            <Point icon="4." color="#059669">Conflict resolution — concurrent edits merge correctly without data loss</Point>
            <Point icon="5." color="#059669">Version history — view and restore previous document versions</Point>
            <Point icon="6." color="#059669">Access control — owner, editor, commenter, viewer permission levels</Point>
            <Point icon="7." color="#059669">Auto-save — every keystroke persisted, no manual save button</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Edit propagation latency: &lt;200ms for online collaborators</Point>
            <Point icon="2." color="#dc2626">Zero data loss — no edit should ever be silently dropped</Point>
            <Point icon="3." color="#dc2626">Strong eventual consistency — all clients converge to the same document state</Point>
            <Point icon="4." color="#dc2626">Scale to 100M+ documents, 10M+ concurrent editing sessions</Point>
            <Point icon="5." color="#dc2626">High availability — 99.99% uptime (users trust the cloud to hold their data)</Point>
            <Point icon="6." color="#dc2626">Support 50+ concurrent editors per document without degradation</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Plain text or rich text (bold, headings, tables)?",
            "How many concurrent editors per document? (5? 50? 500?)",
            "Do we need offline editing support?",
            "Comments, suggestions, track changes?",
            "Real-time cursor/selection presence?",
            "Version history depth? (last 30 days, or forever?)",
            "Document size limits? (pages? MB?)",
            "What scale? (total docs, DAU, concurrent sessions)",
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
            <MathStep step="1" formula="DAU = 100M users" result="100M" note="Google Docs scale. Not all edit simultaneously." />
            <MathStep step="2" formula="Concurrent editing sessions = 10M" result="10M" note="~10% of DAU are actively editing at any moment." />
            <MathStep step="3" formula="Avg ops per user per second = 2" result="2 ops/s" note="Typing, formatting, cursor moves. Burst: 5-10 ops/s." />
            <MathStep step="4" formula="Total ops/sec = 10M × 2" result="~20M ops/sec" note="Global operation throughput. But each op routes to one doc." final />
            <MathStep step="5" formula="Avg collaborators per doc = 3" result="~3" note="Most docs are solo or 2-3 people. Some have 50+." />
            <MathStep step="6" formula="Active documents = 10M / 3" result="~3.3M docs" note="Concurrently active documents needing real-time sync." final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Storage Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Average document size = 50 KB" result="50 KB" note="Rich text with formatting. Median doc ~5 pages." />
            <MathStep step="2" formula="Total documents = 5B" result="5B" note="All-time documents across all users." />
            <MathStep step="3" formula="Document storage = 5B × 50KB" result="~250 TB" note="Current document snapshots only." final />
            <MathStep step="4" formula="Op log per doc per day = 10 KB" result="10 KB" note="Average editing session generates ~10KB of ops." />
            <MathStep step="5" formula="Active docs per day = 50M" result="50M" note="Docs opened and edited at least once per day." />
            <MathStep step="6" formula="Daily op log storage = 50M × 10KB" result="~500 GB/day" note="Operation logs for versioning and replay." final />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Step 3 — Bandwidth</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Avg operation size = 100 bytes" result="100 B" note="Insert/delete + position + metadata." />
            <MathStep step="2" formula="Ops/sec = 20M" result="20M" note="Each op broadcast to avg 2 other collaborators." />
            <MathStep step="3" formula="Broadcast ops/sec = 20M × 2" result="40M ops/s" note="Total outbound operations to deliver." />
            <MathStep step="4" formula="Bandwidth = 40M × 100B" result="~4 GB/s" note="Outbound bandwidth for real-time sync." final />
          </div>
        </Card>
        <Card>
          <Label color="#dc2626">Key Numbers to Remember</Label>
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "Concurrent Sessions", val: "~10M", sub: "Actively editing" },
              { label: "Ops/sec (global)", val: "~20M", sub: "Typing + formatting" },
              { label: "Active Docs", val: "~3.3M", sub: "Concurrently edited" },
              { label: "Doc Size", val: "~50 KB", sub: "Median rich text" },
              { label: "Total Docs", val: "~5B", sub: "All-time storage" },
              { label: "Op Log", val: "~500 GB/day", sub: "For versioning" },
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
          <Label color="#2563eb">WebSocket Events (Real-Time Editing)</Label>
          <CodeBlock code={`# Client → Server (send operation)
{
  "type": "op.submit",
  "doc_id": "doc_abc123",
  "client_id": "client_42",
  "revision": 1547,            # Client's last known server rev
  "ops": [
    { "type": "insert", "pos": 142, "text": "Hello" },
    { "type": "format", "pos": 142, "len": 5,
      "attrs": { "bold": true } }
  ]
}

# Server → All Clients (broadcast transformed op)
{
  "type": "op.applied",
  "doc_id": "doc_abc123",
  "author_id": "user_42",
  "revision": 1548,            # New server revision
  "ops": [
    { "type": "insert", "pos": 145, "text": "Hello" },
    { "type": "format", "pos": 145, "len": 5,
      "attrs": { "bold": true } }
  ]
}

# Server → Sender (acknowledgement)
{
  "type": "op.ack",
  "revision": 1548              # Sender's op is now canonical
}

# Client → Server (cursor/presence)
{
  "type": "presence.update",
  "doc_id": "doc_abc123",
  "cursor": { "pos": 147, "selection_end": 147 },
  "user": { "name": "Alice", "color": "#e74c3c" }
}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">REST APIs (Non-Real-Time)</Label>
          <CodeBlock code={`# POST /v1/documents
# Create a new document
{
  "title": "Q4 Planning",
  "content": "",                # Empty or template
  "permissions": {
    "owner": "user_42",
    "editors": ["user_789"],
    "viewers": ["team_eng"]
  }
}

# GET /v1/documents/:id
# Load document content + metadata
# Returns: {doc_id, title, content, revision, permissions}

# GET /v1/documents/:id/history?from=1500&to=1548
# Get operation log for version range
# Returns: {ops: [...], snapshots: [...]}

# POST /v1/documents/:id/snapshot
# Create named version / restore point
{ "label": "Before legal review" }

# GET /v1/documents/:id/collaborators
# List active collaborators + cursors
# Returns: [{user, cursor, last_active, color}]

# PUT /v1/documents/:id/permissions
# Update sharing and access control
{ "add_editor": "user_101", "remove_viewer": "user_55" }`} />
          <div className="mt-3 space-y-2">
            {[
              { q: "Why WebSocket for edits but REST for document CRUD?", a: "Edits need real-time broadcast (bidirectional). Creating a doc or changing permissions is a one-time action — standard request-response. Mixing channels keeps each simple." },
              { q: "Why send revision number with every op?", a: "For OT: the server needs to know what state the client was at when it generated the op. If client is at rev 1547 and server is at 1550, the server must transform the op against revisions 1548-1550 before applying." },
              { q: "Why separate cursor presence from document ops?", a: "Cursors are ephemeral — fire-and-forget. If a cursor update is lost, the next one (50ms later) replaces it. Document ops are durable — every single one must be applied. Different reliability guarantees = different channels." },
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
          <rect x={10} y={100} width={70} height={34} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={45} y={120} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Browser A</text>
          <rect x={10} y={150} width={70} height={34} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={45} y={170} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Browser B</text>

          {/* LB */}
          <rect x={115} y={120} width={65} height={46} rx={6} fill="#6366f110" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={147} y={140} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">Load</text>
          <text x={147} y={152} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">Balancer</text>

          {/* Collab servers */}
          <rect x={220} y={85} width={95} height={34} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={267} y={106} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Collab Server 1</text>
          <rect x={220} y={130} width={95} height={34} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={267} y={151} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Collab Server 2</text>
          <rect x={220} y={175} width={95} height={34} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={267} y={196} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Collab Server N</text>
          <text x={267} y={225} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">OT engine + WebSocket</text>
          <text x={267} y={236} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">(stateful — per-doc sessions)</text>

          {/* Doc Registry */}
          <rect x={355} y={75} width={80} height={38} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={395} y={92} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Doc Session</text>
          <text x={395} y={104} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Registry</text>

          {/* Op Log */}
          <rect x={355} y={130} width={80} height={38} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={395} y={147} textAnchor="middle" fill="#ea580c" fontSize="9" fontWeight="600" fontFamily="monospace">Op Log</text>
          <text x={395} y={160} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">append-only</text>

          {/* Presence */}
          <rect x={355} y={185} width={80} height={38} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={395} y={202} textAnchor="middle" fill="#0891b2" fontSize="9" fontWeight="600" fontFamily="monospace">Presence</text>
          <text x={395} y={214} textAnchor="middle" fill="#0891b280" fontSize="7" fontFamily="monospace">cursors/users</text>

          {/* Document DB */}
          <rect x={480} y={75} width={80} height={38} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={520} y={92} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Document</text>
          <text x={520} y={104} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">DB</text>

          {/* Snapshot Service */}
          <rect x={480} y={130} width={80} height={38} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={520} y={147} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Snapshot</text>
          <text x={520} y={160} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Service</text>

          {/* Blob Storage */}
          <rect x={480} y={185} width={80} height={38} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={520} y={202} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Blob Store</text>
          <text x={520} y={214} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">images + files</text>

          {/* Search / Index */}
          <rect x={620} y={100} width={80} height={38} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={660} y={117} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Search</text>
          <text x={660} y={130} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Index</text>

          {/* Notification */}
          <rect x={620} y={160} width={80} height={38} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={660} y={177} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Notify Svc</text>
          <text x={660} y={190} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">email + push</text>

          {/* Arrows */}
          <defs><marker id="ah-hld" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          <line x1={80} y1={117} x2={115} y2={135} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={80} y1={167} x2={115} y2={152} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={180} y1={137} x2={220} y2={102} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={180} y1={143} x2={220} y2={147} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={180} y1={150} x2={220} y2={192} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={315} y1={102} x2={355} y2={94} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={315} y1={147} x2={355} y2={149} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={315} y1={192} x2={355} y2={204} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={435} y1={94} x2={480} y2={94} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={435} y1={149} x2={480} y2={149} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={435} y1={204} x2={480} y2={204} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={560} y1={149} x2={620} y2={179} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={560} y1={94} x2={620} y2={119} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>

          {/* Flow labels */}
          <rect x={10} y={270} width={700} height={60} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={20} y={287} fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Edit flow: Browser → WS → Collab Server → OT transform → persist op → broadcast to all clients on same doc</text>
          <text x={20} y={302} fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Load flow: Browser → REST → Document Service → load latest snapshot + replay ops since snapshot → send full doc</text>
          <text x={20} y={317} fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Doc Registry: maps doc_id → collab_server_id. "Which server owns this document session?" Redis-backed, sub-ms lookup.</text>
        </svg>
      </Card>
      <Card>
        <Label color="#c026d3">Key Architecture Decisions</Label>
        <div className="grid grid-cols-3 gap-4">
          {[
            { q: "Why is the Collab Server stateful?", a: "Each active document has an in-memory OT engine on one server. All editors of that doc connect to the SAME server. The server holds the authoritative document state, pending ops queue, and revision counter. This is the only way OT works — you need a single sequencer per document." },
            { q: "Why OT over CRDT for Google Docs?", a: "OT with a central server is simpler to reason about: one server sequences all ops. No need for vector clocks or tombstones. Google Docs uses this model (Jupiter protocol). Trade-off: single server per doc is a bottleneck, but docs rarely have >50 concurrent editors." },
            { q: "Why separate Op Log from Document DB?", a: "The op log is append-only and high-write. The document DB stores snapshots — updated less frequently (every N ops or every 30s). This separation lets you use different storage: fast append store for ops, reliable DB for snapshots. Replay ops from last snapshot to reconstruct current state." },
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
  const [sel, setSel] = useState("ot_basics");
  const topics = {
    ot_basics: { name: "OT Fundamentals", cx: "The Core Algorithm",
      desc: "Operational Transformation is the backbone of Google Docs. When two users edit simultaneously, their operations must be transformed against each other so both converge to the same document state.",
      code: `# OT Core Idea: Transform concurrent operations
# User A inserts "X" at position 2
# User B deletes character at position 1
# These happened concurrently (both based on same doc state)

# Original document: "ABCD"
# A's op: insert("X", pos=2)  →  "ABXCD"
# B's op: delete(pos=1)       →  "ACD"

# Without OT: applying both naively gives WRONG result
# With OT: transform A's op against B's op
#   B deleted pos 1, so A's insert position shifts: 2 → 1
#   Transformed A's op: insert("X", pos=1)

# Transform function: transform(op_a, op_b) → (op_a', op_b')
def transform_insert_delete(ins, del_op):
    if ins.pos <= del_op.pos:
        # Insert before delete — delete shifts right
        return ins, Delete(del_op.pos + 1)
    else:
        # Insert after delete — insert shifts left
        return Insert(ins.text, ins.pos - 1), del_op

# Result: both users converge to "AXCD" ✓
# A applies B's op:  "ABXCD" → delete(1) → "AXCD"
# B applies A's op': "ACD"   → insert("X",1) → "AXCD"`,
      points: [
        "OT transforms operations against each other so concurrent edits converge. Without transformation, concurrent edits corrupt the document.",
        "Transform function takes two concurrent ops and produces transformed versions. If insert at pos 5 and delete at pos 3 happen concurrently, the insert must shift to pos 4.",
        "Google Docs uses server-side OT: server is the single sequencer. Client sends op + its revision number. Server transforms op against any ops the client hasn't seen yet.",
        "Key insight: OT only works correctly if there's a total order on operations. The central server provides this order (revision numbers).",
        "The transform function must satisfy two mathematical properties: TP1 (convergence) and TP2 (associativity for 3+ users). Getting these right for rich text is extremely difficult.",
      ],
    },
    ot_server: { name: "Server-Side OT (Jupiter)", cx: "Google's Protocol",
      desc: "Google Docs uses the Jupiter protocol: a client-server OT model where the server is the single source of truth. Each client maintains a local copy and applies ops optimistically. The server sequences and transforms.",
      code: `# Jupiter Protocol — Client-Server OT
# Server maintains: { revision, doc_state, op_history[] }
# Client maintains: { server_rev, pending_ops[], buffer_op }

# === CLIENT SIDE ===
def on_local_edit(op):
    apply_locally(op)            # Instant — no lag
    if pending_op is None:
        send_to_server(op, server_rev)
        pending_op = op
    else:
        buffer_op = compose(buffer_op, op)  # Queue behind pending

def on_server_ack(rev):
    server_rev = rev
    pending_op = buffer_op       # Send buffered op
    buffer_op = None
    if pending_op:
        send_to_server(pending_op, server_rev)

def on_server_op(remote_op, rev):
    # Transform remote op against our pending ops
    if pending_op:
        pending_op, remote_op = transform(pending_op, remote_op)
    if buffer_op:
        buffer_op, remote_op = transform(buffer_op, remote_op)
    apply_locally(remote_op)
    server_rev = rev

# === SERVER SIDE ===
def on_client_op(client_op, client_rev):
    # Transform against ops since client_rev
    for op in op_history[client_rev:]:
        client_op, _ = transform(client_op, op)
    revision += 1
    apply_to_doc(client_op)
    op_history.append(client_op)
    send_ack(revision, to=sender)
    broadcast(client_op, revision, to=other_clients)`,
      points: [
        "Client applies edits instantly (optimistic) — no waiting for server. This is why Google Docs feels zero-latency despite network round trips.",
        "At most ONE unacknowledged op in flight from each client. Additional edits are composed into a buffer, sent after the pending op is ACKed.",
        "Server transforms the client's op against all ops it hasn't seen (ops between client_rev and current server_rev). This handles the 'concurrent edit' case.",
        "On receiving a remote op, the client transforms it against its own pending and buffered ops before applying. This ensures the client converges with the server.",
        "Jupiter reduces the general OT problem (N clients) to N instances of the 2-party problem (each client vs. server). Much simpler than peer-to-peer OT.",
      ],
    },
    crdt: { name: "CRDT Alternative", cx: "Decentralized Approach",
      desc: "Conflict-Free Replicated Data Types are the alternative to OT. They encode conflict resolution into the data structure itself. No central server needed — any replica can accept writes and merge with any other.",
      code: `# CRDT for Text — Simplified Yjs/Automerge model
# Each character has a unique ID: (client_id, sequence_num)
# Characters form a linked list with fractional positioning

# Character: { id: (client, seq), value, parent_id, deleted }

# Insert "X" between positions with IDs A and B:
#   new_char = { id: (my_id, seq++), value: "X",
#                left: A, right: B, deleted: false }
# The position is defined by (left, right) — not an index!

# Why this works for concurrent inserts:
# User 1 inserts "X" between A and B: left=A, right=B
# User 2 inserts "Y" between A and B: left=A, right=B
# Both are valid! Tie-break by (client_id, seq) ordering
# Result: "A X Y B" or "A Y X B" — consistent everywhere

# Delete: mark as tombstone (deleted=true), don't remove
# This is why CRDTs use more memory — tombstones accumulate

# Merge: union of all operations, deterministic sort
def merge(replica_a, replica_b):
    all_chars = union(replica_a.chars, replica_b.chars)
    # Sort by (left_id, right_id, char_id) — deterministic
    return sorted(all_chars, key=deterministic_order)

# Key: NO TRANSFORM NEEDED. The data structure itself
# guarantees convergence. Any merge order = same result.`,
      points: [
        "CRDTs embed conflict resolution in the data structure. No need for a central server to sequence operations. Any replica can accept writes.",
        "Each character gets a unique, immutable ID. Position is defined by neighboring IDs, not array indices. This eliminates the index-shifting problem that OT solves with transforms.",
        "Trade-off: tombstones. Deleted characters are marked, not removed. Over time this bloats memory. Periodic garbage collection ('compaction') is needed.",
        "Yjs and Automerge are popular CRDT libraries. Figma uses a CRDT-like approach. Good for P2P (no server needed) and offline-first apps.",
        "For Google Docs (centralized, always-online), OT is simpler and more efficient. CRDTs shine when you need offline editing or decentralized sync.",
      ],
    },
    rich_text: { name: "Rich Text Operations", cx: "Beyond Plain Text",
      desc: "Real documents have formatting (bold, headings), embedded objects (images, tables), and structural elements. Each requires careful merge semantics. This is where OT gets truly complex.",
      code: `# Rich text operations — not just insert/delete
# Three core operation types:

# 1. INSERT: add text at position
{ "type": "insert", "pos": 42, "text": "Hello",
  "attrs": { "bold": true, "font": "Arial" } }

# 2. DELETE: remove characters
{ "type": "delete", "pos": 42, "count": 5 }

# 3. FORMAT: change attributes of existing text
{ "type": "format", "pos": 10, "len": 20,
  "attrs": { "bold": true } }
# Does NOT change text content, only styling

# Transform examples for rich text:
# User A: format(pos=5, len=10, bold=true)
# User B: insert(pos=8, "XYZ")
# Transform: A's format must expand to cover the new text
#   A': format(pos=5, len=13, bold=true)  # 10 + 3 inserted

# Structural ops (harder):
# - Split paragraph (Enter key)
# - Merge paragraphs (Backspace at start)
# - Insert table row/column
# - Move list item (indent/outdent)
# Each needs its own transform rules against every other op

# Google Docs uses "quill delta" style:
# Operations are a sequence of: retain(n), insert(text), delete(n)
# retain(5) → skip 5 chars, insert("Hi") → add "Hi", retain(3)
# This format composes and transforms cleanly`,
      points: [
        "Rich text OT needs three op types: insert, delete, and format (attribute change). Each pair needs a transform function: insert×insert, insert×delete, insert×format, delete×format, etc.",
        "Format ops are 'range-based' — they apply to a range of text. When text is inserted within the range, the format must expand. When text is deleted, it must shrink.",
        "Structural operations (paragraph splits, table edits) are the hardest. Splitting a paragraph while someone inserts text in it requires careful merge. Most systems handle this with 'block-level' OT.",
        "Google Docs uses a 'retain/insert/delete' delta format (similar to Quill). This is more composable than position-based ops and simplifies the transform function.",
        "In an interview, mention rich text complexity but don't try to implement all transform functions. Focus on insert×insert and insert×delete transforms — interviewers want to see you understand the concept.",
      ],
    },
  };
  const t = topics[sel];
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">OT vs CRDT — Comparison</Label>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] border-collapse">
            <thead><tr className="border-b-2 border-stone-200">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Property</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">OT (Operational Transform)</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">CRDT</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Verdict</th>
            </tr></thead>
            <tbody>
              {[
                { n:"Architecture", ot:"Central server required", crdt:"Fully decentralized / P2P", v:"Depends" },
                { n:"Conflict Resolution", ot:"Transform function (server sequences)", crdt:"Built into data structure", v:"OT simpler" },
                { n:"Latency (online)", ot:"Optimistic local apply — instant", crdt:"Optimistic local apply — instant", v:"Tie" },
                { n:"Offline Support", ot:"Difficult (server needed for transform)", crdt:"Native (merge anytime)", v:"CRDT ★" },
                { n:"Memory Overhead", ot:"Minimal (just op log)", crdt:"Tombstones accumulate over time", v:"OT ★" },
                { n:"Complexity", ot:"Transform functions are hard to get right", crdt:"Data structure design is hard", v:"Both hard" },
                { n:"Proven at Scale", ot:"Google Docs (15+ years)", crdt:"Figma, Notion (newer)", v:"OT ★" },
              ].map((r,i) => (
                <tr key={i} className={i%2 ? "bg-stone-50/50" : ""}>
                  <td className="px-3 py-2.5 font-mono text-stone-600 font-medium">{r.n}</td>
                  <td className="text-center px-3 py-2.5 text-stone-500">{r.ot}</td>
                  <td className="text-center px-3 py-2.5 text-stone-500">{r.crdt}</td>
                  <td className="text-center px-3 py-2.5 text-lg">{r.v}</td>
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
          <CodeBlock code={`-- Documents (sharded by doc_id)
CREATE TABLE documents (
  doc_id          BIGINT PRIMARY KEY,     -- Snowflake ID
  owner_id        BIGINT NOT NULL,
  title           VARCHAR(500),
  current_rev     INT NOT NULL DEFAULT 0, -- Latest revision
  snapshot_rev    INT NOT NULL DEFAULT 0, -- Last snapshot rev
  content_ref     VARCHAR(255),           -- Blob ref to snapshot
  created_at      TIMESTAMP,
  updated_at      TIMESTAMP,
  deleted_at      TIMESTAMP               -- Soft delete
);

-- Operations (append-only log, sharded by doc_id)
CREATE TABLE operations (
  doc_id          BIGINT NOT NULL,
  revision        INT NOT NULL,           -- Sequential per doc
  author_id       BIGINT NOT NULL,
  ops_data        BLOB NOT NULL,          -- Serialized op array
  created_at      TIMESTAMP NOT NULL,
  PRIMARY KEY (doc_id, revision)
);
-- Shard key: doc_id
-- All ops for a doc on same shard (replay ordering!)

-- Document Permissions (sharded by doc_id)
CREATE TABLE permissions (
  doc_id          BIGINT NOT NULL,
  user_id         BIGINT NOT NULL,
  role            ENUM('owner','editor','commenter','viewer'),
  granted_at      TIMESTAMP,
  PRIMARY KEY (doc_id, user_id),
  INDEX idx_user (user_id)
);

-- Snapshots (periodic full document state)
CREATE TABLE snapshots (
  doc_id          BIGINT NOT NULL,
  revision        INT NOT NULL,
  content_ref     VARCHAR(255),           -- Blob store reference
  created_at      TIMESTAMP,
  PRIMARY KEY (doc_id, revision)
);
-- Snapshot every 100 revisions — limits replay on doc load

-- Session Registry (Redis)
-- Key: doc_session:{doc_id}
-- Value: {collab_server_id, active_users[], revision}
-- TTL: none (cleared on last user disconnect)`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Why These Design Choices?</Label>
          <div className="space-y-3">
            {[
              { q: "Shard operations by doc_id?", a: "All ops for a document on the same shard. OT replay (load doc = last snapshot + ops since) is a single-shard range scan on (doc_id, revision). This is THE hot query during document open." },
              { q: "Append-only operation log?", a: "Operations are never updated or deleted. They form an immutable event log. This gives us version history for free, enables time-travel debugging, and is write-optimized (no random updates)." },
              { q: "Why periodic snapshots?", a: "Without snapshots, loading a document requires replaying ALL operations from the beginning. A document with 100K revisions would take seconds to load. Snapshot every 100 revisions → max 100 ops to replay." },
              { q: "Why Blob reference for content?", a: "Document content (rich text JSON/HTML) can be large. Storing in blob storage (S3) with a reference in the DB keeps the DB lean. Snapshots are write-once, read-occasionally — perfect for object storage." },
              { q: "Why not just store the latest doc?", a: "You need the op log for: OT (transform against recent ops), version history (show changes), undo/redo (reverse ops), and conflict resolution. The document is really just the 'materialized view' of the op log." },
              { q: "Permissions sharding?", a: "Sharded by doc_id for 'who can access this doc?' (hot path on open). Index on user_id for 'list my documents' (secondary index or separate table). Two access patterns, two indexes." },
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
          <Label color="#059669">Collaboration Layer Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">One Collab Server per active document</strong> — all editors of a document connect to the same server. The OT engine is single-threaded per doc. This is the fundamental constraint.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Doc Session Registry</strong> — Redis hash: doc_id → collab_server_id. When a user opens a doc, look up which server owns it. If none, assign one (consistent hashing or least-loaded).</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">~50K active documents per server</strong> — each doc session uses ~10-50KB memory (in-memory doc state + pending ops). 50K docs × 50KB = 2.5GB. Limited by memory, not CPU.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Session migration on server drain</strong> — on deploy: migrate active doc sessions to other servers. Save current state to DB, notify clients to reconnect. Clients load state from new server.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Storage Layer Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Op log sharding by doc_id</strong> — all operations for a document on the same shard. Write pattern is append-only (sequential writes per shard). Read pattern is range scan (load ops for replay).</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Snapshot compaction</strong> — periodically create a snapshot and optionally archive old ops. Op log retention: 30 days for version history, then only snapshots. Reduces storage 100×.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Hot/cold document tiering</strong> — active docs cached in Redis + in-memory on Collab Server. Inactive docs only in DB. Documents not opened in 90 days → cold storage (S3 Glacier).</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Read replicas for document loading</strong> — opening a doc (read snapshot + ops) can use replicas. Only active editing writes go to primary. Most docs are read-only (viewing) at any moment.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Strategy</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Option A: Document-Homed Regions ★", d:"Each document is assigned to the region where its owner is. Editing from other regions routes through the home region.", pros:["Simple: one OT server per doc, no cross-region OT","Data locality for compliance (GDPR, data residency)","Home region has full consistency"], cons:["Cross-region editors see 50-150ms additional latency","Migration needed if owner moves regions","Popular docs with global editors = high cross-region traffic"], pick:true },
            { t:"Option B: Regional OT Servers", d:"Each region runs its own OT server. Cross-region ops sync via inter-region bridge.", pros:["Low latency for all editors regardless of region","Resilient to regional outages"], cons:["Multi-leader OT is extremely complex","Conflict resolution between regions is an unsolved research problem","Risk of document divergence on partition"], pick:false },
            { t:"Option C: Edge Buffering", d:"OT runs in home region. Edge nodes buffer and batch operations from remote editors, reducing round trips.", pros:["Simpler than multi-leader","Reduces perceived latency with local echo","Home region stays authoritative"], cons:["Still has cross-region latency for conflict resolution","Edge state can diverge briefly","Buffering adds complexity to client"], pick:false },
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
        <Label color="#d97706">Critical: User Edits Must Never Be Lost</Label>
        <p className="text-[12px] text-stone-500 mb-4">Users trust cloud document editors with critical data — contracts, reports, planning docs. Losing even a single edit destroys trust permanently. Google Docs' "saving..." indicator is a sacred promise.</p>
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Write-Ahead: Persist Before ACK</div>
            <p className="text-[11px] text-stone-500">Operation written to op log BEFORE the server sends ACK to the client. If server crashes after ACK but before persist, data is lost. Write-ahead logging prevents this.</p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-white p-4">
            <div className="text-[11px] font-bold text-amber-700 mb-1.5">Client-Side Buffer</div>
            <p className="text-[11px] text-stone-500">Client keeps all unACKed ops in local memory/IndexedDB. On disconnect, ops are retried on reconnect. Client never discards an op until server ACKs it. Belt and suspenders.</p>
          </div>
          <div className="rounded-lg border border-blue-200 bg-white p-4">
            <div className="text-[11px] font-bold text-blue-700 mb-1.5">Periodic Snapshots</div>
            <p className="text-[11px] text-stone-500">Full document snapshot saved every 100 revisions or 30 seconds. If op log is corrupted, recover from last snapshot. Snapshots stored durably in blob storage with replication.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Collab Server Failure Recovery</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Server crashes mid-session</strong> — clients detect disconnect (WebSocket close). They reconnect and are assigned to a new Collab Server. New server loads doc from last snapshot + replays op log. Client resends pending ops.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Doc Session Registry</strong> — Redis marks the old server as dead. New server re-registers. Clients use the registry to find the new server. Recovery time: ~2-5 seconds (snapshot load + op replay).</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Conflict on recovery</strong> — multiple clients might send the same ops. Server deduplicates by (client_id, client_seq). Idempotent replay ensures no duplicate edits.</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Data Durability Guarantees</Label>
          <ul className="space-y-2.5">
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">Op log replication</strong> — synchronous write to primary + at least 1 replica before ACK. Async replication to third copy. RPO = 0 for acknowledged ops.</Point>
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">Snapshot cross-region backup</strong> — snapshots replicated to a secondary region. If entire primary region fails, documents can be recovered from last snapshot (some recent ops may be lost — RPO = up to 100 ops).</Point>
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">Client as last resort</strong> — if server loses data, the client has its local document state + unACKed ops. Client can reconstruct and re-upload. This is the ultimate safety net.</Point>
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">"Saving..." indicator</strong> — only clears after server ACK. If it stays stuck, client alerts user. Never lie about save status — it's the most important UX trust signal.</Point>
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
            { name: "Op Apply Latency", target: "p99 < 50ms", desc: "Time from op received to applied + broadcast", alarm: "> 200ms" },
            { name: "WS→Client Latency", target: "p99 < 200ms", desc: "End-to-end: user A types → user B sees it", alarm: "> 500ms" },
            { name: "Op Log Write Latency", target: "p99 < 20ms", desc: "Time to persist op to durable storage", alarm: "> 100ms" },
            { name: "Snapshot Generation", target: "< 2s per snapshot", desc: "Time to serialize and store document snapshot", alarm: "> 10s" },
            { name: "Active Doc Sessions", target: "< 50K / server", desc: "Documents with active editing sessions", alarm: "> 60K" },
            { name: "Ops/sec per Document", target: "< 500 ops/s", desc: "Single doc throughput. High = hot doc", alarm: "> 1000" },
            { name: "Client Reconnect Rate", target: "< 0.1%/min", desc: "Rate of WS reconnections (indicates instability)", alarm: "> 1%/min" },
            { name: "Transform Queue Depth", target: "< 10 pending", desc: "Ops waiting for OT transform on server", alarm: "> 50" },
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
          <CodeBlock code={`# Trace: User edit → all collaborators see it
trace_id: "edit-abc-123"
spans:
  ├─ [client] local_apply         0ms   # Instant
  ├─ [client] ws_send             2ms   # To server
  ├─ [server] receive_op          25ms  # Network
  ├─ [server] ot_transform        26ms  # Transform vs pending
  ├─ [server] persist_op_log      32ms  # Write to DB
  ├─ [server] apply_to_doc        33ms  # Update in-memory doc
  ├─ [server] broadcast_start     34ms  # Fan-out to clients
  │   ├─ [server] ws_send_client_B  35ms
  │   └─ [server] ws_send_client_C  36ms
  ├─ [client_B] receive_op        60ms  # Network
  ├─ [client_B] ot_transform      61ms  # Transform vs local
  └─ [client_B] render            63ms  # Update DOM
# Total: user A types → user B sees it ≈ 63ms`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Logging & Alerting Strategy</Label>
          <div className="space-y-3">
            {[
              { level: "P0 — Page Immediately", items: ["Op log write failures (data loss risk)", "Collab Server crash (active sessions lost)", "Document corruption detected (checksum mismatch)", "Snapshot service unreachable (recovery path broken)"] },
              { level: "P1 — Alert On-Call", items: ["Op apply latency p99 > 200ms", "Client reconnect rate > 1%/min", "Transform queue depth > 50 (OT bottleneck)", "Doc session count approaching server limit"] },
              { level: "P2 — Dashboard Monitor", items: ["Snapshot generation time trending up", "Cross-region replication lag > 5s", "Presence update latency > 500ms", "Document load time p95 > 3s"] },
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
            { title: "OT Transform Bug → Document Corruption", desc: "A subtle bug in the transform function causes two clients to diverge. Their document states are permanently different. Neither knows.", impact: "HIGH — users see different content in the same doc. Data integrity destroyed.", mitigation: "Periodic checksum comparison: all clients send hash of their doc state. Server compares. If mismatch → force-resync from server state. Extensive property-based testing of transform functions." },
            { title: "Hot Document → Single Server Bottleneck", desc: "A viral document (company all-hands notes) gets 500 concurrent editors. The single Collab Server OT engine can't keep up — ops queue up, latency spikes to seconds.", impact: "MEDIUM — one doc is slow, others unaffected. But high-profile docs are exactly the ones with many editors.", mitigation: "Read-only overflow: after N editors, new users join as viewers with delayed sync. Or: partition document into sections, each on a different server (complex). Alert on ops/sec per doc > 500." },
            { title: "Op Log Storage Full → Writes Rejected", desc: "Op log partition runs out of disk space. New operations can't be persisted. Server must reject edits to avoid data loss.", impact: "HIGH — editing stops for all documents on that shard. Users see 'unable to save' errors.", mitigation: "Aggressive snapshot compaction: archive ops older than 30 days (they're baked into snapshots). Disk space monitoring with 7-day runway alert. Auto-expand or rebalance partitions." },
            { title: "Snapshot Divergence from Op Log", desc: "A bug causes the snapshot to not match the result of replaying all ops. New users loading the doc see a different state than existing editors.", impact: "HIGH — split-brain within the same document. Existing editors have correct state, new joiners have wrong state.", mitigation: "Snapshot verification: after generating snapshot, replay ops from previous snapshot and compare. If mismatch → regenerate. Never serve unverified snapshots." },
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
            { case: "Simultaneous cursor at same position", detail: "Two users type at the exact same position simultaneously. OT must decide ordering — typically by client_id (deterministic tie-break). Both converge but insertion order may feel arbitrary to users." },
            { case: "Delete while someone is typing inside deleted range", detail: "User A selects and deletes a paragraph. User B is typing in the middle of that paragraph. B's inserts must survive — OT transforms insert against delete, keeping B's text at the deletion boundary." },
            { case: "Undo across collaborative edits", detail: "User A types 'hello', user B types 'world' after it, user A hits undo. Should 'hello' be removed? But B's 'world' was positioned relative to 'hello'. Undo must transform the inverse op against intervening ops." },
            { case: "Network partition during editing", detail: "Client loses connection for 30 seconds, accumulates 50 local ops. On reconnect, sends all 50 at once. Server must transform each against ops that happened during the partition. Can be slow for large backlogs." },
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
        <Label color="#0f766e">Service Decomposition</Label>
        <div className="grid grid-cols-3 gap-4">
          {[
            { name: "Collaboration Server (Gateway)", owns: "WebSocket connections, OT engine per document, op sequencing and broadcast", tech: "Go/Rust with epoll, in-memory OT engine", api: "WebSocket: op.submit, op.ack, presence.update", scale: "Horizontal — one server per active doc, ~50K docs/server", stateful: true,
              modules: ["WebSocket Handler (upgrade, read, write frames)", "OT Engine (transform, apply, sequence operations per document)", "Doc State Manager (in-memory document state + revision counter)", "Broadcast Manager (fan-out transformed ops to all doc clients)", "Session Registrar (register/deregister doc→server in Redis)", "Client Sync (resync client on reconnect, send missed ops)"] },
            { name: "Document Service", owns: "Document CRUD, snapshot management, version history, permissions", tech: "Java/Go + PostgreSQL (metadata) + S3 (snapshots)", api: "gRPC: CreateDoc, LoadDoc, SaveSnapshot, GetHistory", scale: "Horizontal — stateless, shard by doc_id", stateful: false,
              modules: ["Document Manager (create, delete, metadata updates)", "Snapshot Writer (serialize doc state, store in blob storage)", "History Service (op log queries, time-travel, diff between revisions)", "Permission Manager (ACL checks, share/unshare, role changes)", "Import/Export (convert to/from DOCX, PDF, Markdown)", "Retention Manager (archive cold docs, purge deleted after 30 days)"] },
            { name: "Presence Service", owns: "Cursor positions, user avatars, active collaborator list, typing indicators", tech: "Go + Redis (ephemeral presence store)", api: "gRPC: UpdateCursor, GetCollaborators, JoinDoc, LeaveDoc", scale: "Horizontal — shard by doc_id", stateful: false,
              modules: ["Cursor Tracker (store cursor position per user per doc)", "Collaborator List (active users in a doc + their colors)", "Heartbeat Monitor (detect stale connections, clean up)", "Broadcast Relay (fan-out presence updates to doc participants)", "Color Assigner (deterministic unique color per user per doc)", "Rate Limiter (throttle cursor updates to 20/sec per user)"] },
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
        <Label color="#9333ea">System Architecture — Document Editing Pipeline</Label>
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

          {/* Browser A */}
          <rect x={15} y={55} width={65} height={36} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={47} y={72} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Browser A</text>
          <text x={47} y={84} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">editor</text>

          {/* Browser B */}
          <rect x={15} y={110} width={65} height={36} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={47} y={127} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Browser B</text>
          <text x={47} y={139} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">editor</text>

          {/* LB */}
          <rect x={105} y={75} width={55} height={40} rx={6} fill="#6366f110" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={132} y={93} textAnchor="middle" fill="#6366f1" fontSize="8" fontWeight="600" fontFamily="monospace">Load</text>
          <text x={132} y={105} textAnchor="middle" fill="#6366f1" fontSize="8" fontWeight="600" fontFamily="monospace">Balancer</text>

          {/* Collab Server 1 */}
          <rect x={190} y={40} width={85} height={36} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={232} y={57} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Collab Srv 1</text>
          <text x={232} y={70} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">Doc A's OT</text>

          {/* Collab Server 2 */}
          <rect x={190} y={90} width={85} height={36} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={232} y={107} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Collab Srv 2</text>
          <text x={232} y={120} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">Doc B's OT</text>

          {/* Collab Server N */}
          <rect x={190} y={135} width={85} height={28} rx={6} fill="#05966908" stroke="#05966950" strokeWidth={1} strokeDasharray="3,2"/>
          <text x={232} y={153} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">Collab Srv N...</text>

          {/* Doc Session Registry */}
          <rect x={305} y={55} width={80} height={38} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={345} y={72} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Doc Session</text>
          <text x={345} y={84} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Registry</text>
          <text x={345} y={105} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">doc → server</text>
          <text x={345} y={115} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">(Redis)</text>

          {/* Document Service */}
          <rect x={370} y={35} width={90} height={40} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
          <text x={415} y={52} textAnchor="middle" fill="#7c3aed" fontSize="9" fontWeight="600" fontFamily="monospace">Document Svc</text>
          <text x={415} y={65} textAnchor="middle" fill="#7c3aed80" fontSize="7" fontFamily="monospace">CRUD + snapshots</text>

          {/* Presence Service */}
          <rect x={370} y={90} width={90} height={36} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={415} y={107} textAnchor="middle" fill="#0891b2" fontSize="9" fontWeight="600" fontFamily="monospace">Presence Svc</text>
          <text x={415} y={120} textAnchor="middle" fill="#0891b280" fontSize="7" fontFamily="monospace">cursors/users</text>

          {/* Search Service */}
          <rect x={370} y={135} width={90} height={28} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={415} y={153} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Search Svc</text>

          {/* Op Log */}
          <rect x={490} y={35} width={80} height={40} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={530} y={52} textAnchor="middle" fill="#ea580c" fontSize="9" fontWeight="600" fontFamily="monospace">Op Log</text>
          <text x={530} y={65} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">append-only</text>

          {/* Notification Service */}
          <rect x={490} y={90} width={80} height={36} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={530} y={107} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Notify Svc</text>
          <text x={530} y={120} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">email / push</text>

          {/* Comment Service */}
          <rect x={490} y={135} width={80} height={28} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={530} y={153} textAnchor="middle" fill="#78716c" fontSize="8" fontWeight="600" fontFamily="monospace">Comment Svc</text>

          {/* Permission Service */}
          <rect x={600} y={55} width={80} height={36} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={640} y={72} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Permission</text>
          <text x={640} y={84} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">ACL + sharing</text>

          {/* Data Layer */}
          <rect x={20} y={205} width={85} height={36} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={62} y={222} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Document DB</text>
          <text x={62} y={234} textAnchor="middle" fill="#78716c80" fontSize="7" fontFamily="monospace">PostgreSQL</text>

          <rect x={130} y={205} width={85} height={36} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={172} y={222} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Op Log Store</text>
          <text x={172} y={234} textAnchor="middle" fill="#78716c80" fontSize="7" fontFamily="monospace">Cassandra</text>

          <rect x={240} y={205} width={85} height={36} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={282} y={222} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Presence</text>
          <text x={282} y={234} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">Redis</text>

          <rect x={350} y={205} width={85} height={36} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={392} y={222} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Doc Sessions</text>
          <text x={392} y={234} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">Redis</text>

          <rect x={460} y={205} width={85} height={36} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={502} y={222} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Blob Store</text>
          <text x={502} y={234} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">S3 (snapshots)</text>

          <rect x={570} y={205} width={85} height={36} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={612} y={222} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Search Index</text>
          <text x={612} y={234} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">Elasticsearch</text>

          {/* Arrows */}
          <defs><marker id="ah-svc" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          <line x1={80} y1={73} x2={105} y2={88} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={80} y1={128} x2={105} y2={102} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={160} y1={88} x2={190} y2={58} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={160} y1={98} x2={190} y2={108} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={275} y1={58} x2={305} y2={68} stroke="#c026d350" strokeWidth={1} markerEnd="url(#ah-svc)" strokeDasharray="3,2"/>
          <line x1={275} y1={108} x2={305} y2={78} stroke="#c026d350" strokeWidth={1} markerEnd="url(#ah-svc)" strokeDasharray="3,2"/>
          <line x1={275} y1={50} x2={370} y2={50} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={245} y1={76} x2={245} y2={90} stroke="#059669" strokeWidth={1.5} markerEnd="url(#ah-svc)"/>
          <text x={255} y={86} fill="#05966990" fontSize="7" fontFamily="monospace">route</text>
          <line x1={460} y1={55} x2={490} y2={55} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={415} y1={75} x2={62} y2={205} stroke="#78716c40" strokeWidth={1} markerEnd="url(#ah-svc)" strokeDasharray="3,2"/>
          <line x1={415} y1={126} x2={282} y2={205} stroke="#d9770640" strokeWidth={1} markerEnd="url(#ah-svc)" strokeDasharray="3,2"/>
          <line x1={570} y1={108} x2={600} y2={108} stroke="#d97706" strokeWidth={1} markerEnd="url(#ah-svc)"/>
          <text x={605} y={112} fill="#d97706" fontSize="7" fontFamily="monospace">→ email/push</text>

          {/* Legend */}
          <rect x={15} y={260} width={695} height={95} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={25} y={278} fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Edit flow: Browser → WS → Collab Srv → OT transform (sequence + apply) → persist to Op Log → broadcast to all doc clients</text>
          <text x={25} y={295} fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Load flow: Browser → REST → Doc Svc → load last snapshot from S3 → replay ops since snapshot from Op Log → send full doc</text>
          <text x={25} y={312} fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Doc Session Registry: doc_id → collab_server_id (Redis). THE critical lookup — "which server owns this document's OT session?"</text>
          <text x={25} y={329} fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Key insight: Collab Servers are STATEFUL (hold OT engine per doc). All other services are stateless. One doc = one server.</text>
          <text x={25} y={346} fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Ops persisted ONCE to append-only log. Snapshots periodic (every 100 revisions). Doc state = last snapshot + replayed ops.</text>
        </svg>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#2563eb">
          <Label color="#2563eb">Supporting Services</Label>
          <div className="space-y-3">
            {[
              { name: "Comment Service", role: "Inline comments anchored to text ranges. Comments must track their anchor position as the document changes (OT must transform comment ranges too)." },
              { name: "Suggestion Service", role: "Track-changes mode: proposed edits shown as colored markup. Accept/reject suggestions. Each suggestion is a deferred operation." },
              { name: "Export Service", role: "Convert document to PDF, DOCX, Markdown, HTML. Runs as an async worker — rendering complex documents to PDF can take seconds." },
              { name: "Notification Service", role: "Email digest of changes ('3 edits since you last viewed'). Real-time notifications for @mentions, comments, sharing changes." },
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
              { comp: "Collab Server", choice: "Go or Rust", why: "High-performance WebSocket handling. Memory-efficient for 50K concurrent doc sessions. Go for simplicity, Rust for absolute performance." },
              { comp: "Op Log Store", choice: "Cassandra or DynamoDB", why: "Append-only, time-series workload. High write throughput. Partition by doc_id. Range scan by revision for replay." },
              { comp: "Document DB", choice: "PostgreSQL", why: "ACID for document metadata, permissions, user data. Low write volume (metadata changes are infrequent). Strong consistency for permission checks." },
              { comp: "Doc Session Registry", choice: "Redis Cluster", why: "Sub-ms lookup: doc_id → server_id. Ephemeral data (session dies when all editors leave). 3.3M active docs × 100 bytes = ~330MB. Fits in memory easily." },
              { comp: "Blob Storage", choice: "S3 / GCS", why: "Snapshots are write-once, read-occasionally. 250TB of documents. Object storage is the cheapest option. CDN for document loading performance." },
              { comp: "Search Index", choice: "Elasticsearch", why: "Full-text search across all user's documents. Inverted index on document content. Updated async via change feed from op log." },
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
  const [sel, setSel] = useState("edit");
  const flows = {
    edit: { name: "Real-Time Edit Flow", steps: [
      { actor: "Browser A", action: "User types 'Hello' at position 42", detail: "Apply locally immediately (optimistic). Send op to server with client_rev=1547." },
      { actor: "Load Balancer", action: "Route to Collab Server for doc_abc", detail: "Sticky routing based on doc_id → Collab Server mapping from Doc Session Registry." },
      { actor: "Collab Server", action: "Receive op, transform against pending ops", detail: "Server is at rev 1550. Transform client's op against ops 1548, 1549, 1550. Adjusted position: 42 → 45." },
      { actor: "Collab Server", action: "Persist transformed op to Op Log", detail: "Append to Op Log: {doc_id, rev=1551, ops, author}. MUST complete before ACK." },
      { actor: "Collab Server", action: "Apply op to in-memory document state", detail: "Update the authoritative in-memory doc. Increment revision to 1551." },
      { actor: "Collab Server", action: "ACK sender + broadcast to others", detail: "Send op.ack{rev=1551} to Browser A. Send op.applied{rev=1551, ops} to Browser B, C, etc." },
      { actor: "Browser B", action: "Receive remote op, transform against local pending", detail: "If B has pending ops, transform remote op against them. Apply transformed op. Update server_rev to 1551." },
    ]},
    load: { name: "Document Load Flow", steps: [
      { actor: "Browser", action: "User opens document doc_abc", detail: "GET /v1/documents/doc_abc → Document Service." },
      { actor: "Document Svc", action: "Check permissions", detail: "Lookup permissions table: does this user have view/edit access? If not, return 403." },
      { actor: "Document Svc", action: "Load latest snapshot + replay ops", detail: "Snapshot at rev 1500 + ops 1501-1551. Apply all ops to snapshot → current state at rev 1551." },
      { actor: "Document Svc", action: "Return document to client", detail: "Return: {content, revision=1551, title, collaborators, permissions}. Browser renders the document." },
      { actor: "Browser", action: "Establish WebSocket to Collab Server", detail: "Lookup Doc Session Registry → find Collab Server for this doc. Connect via WS. Send: {join, doc_id, revision=1551}." },
      { actor: "Collab Server", action: "Register client in doc session", detail: "Add client to doc's editor list. If first editor: load doc into memory. Send missed ops (if any between load and WS connect)." },
      { actor: "Presence Svc", action: "Announce new collaborator", detail: "Broadcast to all other editors: {user joined, name, cursor, color}. Browser shows avatar in toolbar." },
    ]},
    conflict: { name: "Conflict Resolution Flow", steps: [
      { actor: "Browser A", action: "Insert 'X' at position 5 (rev=100)", detail: "Applied locally: doc becomes '...X...' at pos 5. Sent to server." },
      { actor: "Browser B", action: "Delete char at position 3 (rev=100)", detail: "Concurrent! Same base revision. Applied locally: char at pos 3 removed." },
      { actor: "Collab Server", action: "Receive A's op first (arrives first)", detail: "Server at rev 100. No transform needed. Apply insert at pos 5. Rev → 101. Persist + broadcast." },
      { actor: "Collab Server", action: "Receive B's op (client_rev=100, server=101)", detail: "B's op was based on rev 100, but server is now 101. Transform B's delete against A's insert." },
      { actor: "Collab Server", action: "Transform: delete(3) × insert(5) → delete(3)", detail: "Delete at pos 3 is before insert at pos 5. No position shift needed. B's op unchanged. Rev → 102." },
      { actor: "Browser A", action: "Receive B's transformed op", detail: "A already applied its own insert. Receives delete(3). Apply: remove char at pos 3. A has both edits." },
      { actor: "Browser B", action: "Receive A's op, transform against pending", detail: "B already applied its delete. Receives insert(5). Transform: pos 5 → pos 4 (shifted left because B deleted pos 3). Apply insert at pos 4. Both converge. ✓" },
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
          <Label color="#b45309">Kubernetes Deployment — Collab Server</Label>
          <CodeBlock code={`apiVersion: apps/v1
kind: StatefulSet           # StatefulSet for sticky identity
metadata:
  name: collab-server
spec:
  replicas: 200               # 200 servers × 50K docs = 10M docs
  serviceName: collab-server
  template:
    spec:
      terminationGracePeriodSeconds: 120
      containers:
      - name: collab
        image: collab-server:v3.2
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "collab-drain"]
              # 1. Stop accepting new doc sessions
              # 2. Save all in-memory doc states to DB
              # 3. Notify clients to reconnect
              # 4. Deregister from Doc Session Registry
              # 5. Wait for clients to disconnect (90s)
        ports:
        - containerPort: 8080  # WS port
        resources:
          requests:
            memory: "4Gi"     # ~50K docs × 50KB + overhead
            cpu: "4"
          limits:
            memory: "6Gi"
        env:
        - name: MAX_DOC_SESSIONS
          value: "50000"
        - name: SNAPSHOT_INTERVAL_REVS
          value: "100"
        - name: OP_PERSIST_MODE
          value: "sync"        # Write-ahead before ACK`} />
          <div className="mt-3 space-y-1.5">
            <Point icon="⚠" color="#b45309">preStop hook saves all doc states and drains connections over 90 seconds</Point>
            <Point icon="⚠" color="#b45309">StatefulSet gives stable network IDs — clients can reconnect to same server after brief disruption</Point>
            <Point icon="⚠" color="#b45309">Memory-bound: 50K docs × 50KB = 2.5GB. CPU is rarely the bottleneck (OT transforms are fast)</Point>
          </div>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Security — Access Control + Transport</Label>
          <div className="space-y-3">
            {[
              { layer: "Authentication", details: ["OAuth 2.0 / OIDC for user identity", "JWT tokens for session management", "Service-to-service: mTLS between internal services", "WebSocket auth: token in first frame after upgrade"] },
              { layer: "Authorization (Document-Level)", details: ["ACL per document: owner / editor / commenter / viewer", "Link sharing: anyone with link can view/edit", "Organization-level policies: 'all docs default to internal-only'", "Permission check on every op submission (not just on load)"] },
              { layer: "Transport Security", details: ["TLS 1.3 everywhere (HTTPS + WSS)", "Certificate pinning for mobile/desktop apps", "Content Security Policy headers to prevent XSS", "Rate limiting: max 50 ops/sec per client to prevent abuse"] },
              { layer: "Data Protection", details: ["Encryption at rest: AES-256 for document storage", "Customer-managed encryption keys (CMEK) for enterprise", "Audit logging: who accessed/edited what document and when", "Data Loss Prevention (DLP): scan for sensitive content (SSN, credit cards)"] },
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
                { b: "OT Transform Queue", s: "Ops delayed, collab feels laggy. Multiple seconds between type and broadcast.", f: "Optimize transform function. If queue > 50 ops, batch-transform. For very hot docs: degrade gracefully (show 'too many editors' warning).", p: "OT transform is O(n) per op against pending queue. 500 concurrent editors each sending 2 ops/sec = 1000 transforms/sec per doc. CPU bottleneck." },
                { b: "Op Log Write Throughput", s: "Write latency > 100ms. Op persistence blocks ACK → user sees 'saving...' lag.", f: "Batch op writes (collect ops for 10ms window, write batch). Async write with WAL buffer. Shard op log by doc_id.", p: "Sync writes are safer but slower. Async writes risk data loss on crash. WAL + async flush is the sweet spot." },
                { b: "Snapshot Generation", s: "Document load time > 5s. Too many ops to replay since last snapshot.", f: "More frequent snapshots (every 50 revisions instead of 100). Background snapshot worker, don't block editing. Pre-warm popular docs.", p: "Snapshots for 100-page documents with images can be 10MB+. Serialization is CPU-intensive. Don't snapshot on the Collab Server — offload to a worker." },
                { b: "Doc Session Registry", s: "Doc open time > 1s. Registry lookup slow or stale.", f: "Redis Cluster with read replicas. Local cache on LB with 5s TTL. Stale sessions cleaned by heartbeat sweeper.", p: "If registry returns stale server (server died), client connects to dead server. Need fallback: client retries with registry refresh. Use lease-based TTL." },
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
            { title: "Company All-Hands Doc Melts Collab Server", symptom: "CEO shares a doc for all-hands notes. 2,000 employees open it simultaneously. Collab Server OT engine pegged at 100% CPU. Edits delayed by 10+ seconds.",
              cause: "Single Collab Server per doc. 2,000 WebSocket connections + 4,000 ops/sec (2,000 users × 2 ops/sec) overwhelms the OT transform queue. Transform is O(pending_queue_length) per op.",
              fix: "Implement 'overflow mode': after 100 editors, additional users join as viewers with 5-second delayed sync. Partition doc by sections — each section on a separate Collab Server. Pre-scale for known events.",
              quote: "We had 15 minutes warning before all-hands. Not enough to partition the doc. We ended up asking everyone to stop typing so the OT queue could drain. Embarrassing." },
            { title: "Snapshot Corruption After Storage Migration", symptom: "Users opening old documents see garbled content. Some documents show content from other documents. Panic across enterprise customers.",
              cause: "Blob storage migration reordered snapshot references. Document A's content_ref now pointed to Document B's snapshot blob. No checksum validation on snapshot load.",
              fix: "SHA-256 checksum stored alongside every snapshot. Verified on read. If mismatch → fall back to op log replay from the beginning (slow but correct). Added migration verification: read-back every migrated blob and validate checksum.",
              quote: "A lawyer opened their merger agreement and saw someone else's vacation itinerary. We learned that day that blob references MUST be checksummed." },
            { title: "Op Log Partition Hot Spot", symptom: "One Cassandra partition (for a popular doc) grows to 200MB. Read latency for that partition spikes to 5 seconds. Doc becomes nearly unusable.",
              cause: "Document with 500K revisions over 2 years. All ops for that doc on one partition (sharded by doc_id). Cassandra partitions shouldn't exceed ~100MB.",
              fix: "Composite partition key: (doc_id, revision_bucket) where bucket = revision / 10000. Each bucket holds ≤10K ops. Range scan across buckets for replay. Also: aggressive snapshot compaction archives ops older than 30 days.",
              quote: "The most-edited document in the company was a shared glossary. 500K edits over 2 years. We never imagined a single doc would have that many revisions." },
            { title: "WebSocket Connection Storm After Deploy", symptom: "Deploy new Collab Server version. All 50K doc sessions on old server migrate. 200K WebSocket connections hit the LB simultaneously. Connection acceptance rate can't keep up.",
              cause: "Graceful drain sends 'reconnect' to all clients at once. All clients reconnect within 1 second. TLS handshake for 200K connections overwhelms LB.",
              fix: "Jittered reconnect: client adds random delay (0-30 seconds) before reconnecting. Server sends 'reconnect_after' with staggered timestamps. Deploy one server at a time with 60-second gaps.",
              quote: "Our graceful drain was a polite way to DDoS ourselves. Random jitter on reconnect solved it completely." },
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
        { t: "Offline Editing", d: "Allow users to edit documents without internet. Sync changes when reconnected using queued ops + conflict resolution.", detail: "Store ops locally in IndexedDB. On reconnect, send all buffered ops. Server transforms against ops that happened while offline. CRDT is better suited for this than OT.", effort: "Hard" },
        { t: "Comments & Suggestions", d: "Inline comments anchored to text ranges. Suggestion mode shows proposed edits as colored markup.", detail: "Comments have anchor ranges that must be transformed by OT as the document changes. Suggestions are deferred ops — accept/reject applies or discards the op.", effort: "Medium" },
        { t: "Real-Time Cursors & Selections", d: "See where other editors are typing and what they've selected. Color-coded per user.", detail: "Cursor positions are ephemeral — broadcast via presence channel at 20Hz. Selection ranges transformed against incoming ops. Lost cursors replaced by next update.", effort: "Easy" },
        { t: "Document Templates", d: "Pre-built document templates (meeting notes, project briefs, resumes) that users can instantiate.", detail: "Template = a document snapshot with placeholder text. Instantiation copies the snapshot and creates a new doc_id. No special OT handling needed.", effort: "Easy" },
        { t: "AI-Assisted Writing", d: "Inline AI suggestions: autocomplete, summarize, rewrite, translate. Powered by LLM API calls.", detail: "AI-generated text inserted as a special 'suggestion' op. User accepts/rejects. Must handle AI response latency (1-3s) without blocking other edits. Stream AI output as incremental inserts.", effort: "Medium" },
        { t: "Document-Level Permissions Audit", d: "Enterprise feature: audit trail of all permission changes. Who shared with whom, when, and what access level.", detail: "Separate audit log table. Immutable append-only. Required for SOC2/HIPAA compliance. Queryable by admin for compliance reporting.", effort: "Medium" },
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
    { q:"Why OT and not CRDT for Google Docs?", a:"OT with a central server is simpler to implement and reason about. The server provides a total order on operations — no need for vector clocks or logical timestamps. Google has used OT successfully for 15+ years. CRDT's advantage (offline/P2P) isn't needed when users are always online. CRDT also has higher memory overhead (tombstones). For a server-centric system, OT wins on simplicity and efficiency.", tags:["design"] },
    { q:"How do you handle undo/redo with multiple collaborators?", a:"Local undo only — undo YOUR last edit, not everyone's. Implementation: maintain a per-user stack of ops. On undo, compute the inverse op and transform it against all ops that happened since. This is called 'operational undo' and is one of the hardest parts of OT. Example: you insert 'hello' at pos 5, someone else inserts 'world' at pos 3. Your undo must delete 'hello' at pos 10 (shifted by 'world').", tags:["algorithm"] },
    { q:"How do you ensure consistency across all clients?", a:"Strong eventual consistency via the OT protocol: (1) Server assigns a total order (revision numbers). (2) All clients apply ops in the same order (after transformation). (3) Transform functions satisfy TP1: applying A then B' gives the same result as B then A'. (4) Periodic checksum verification: clients send document hash, server compares. If divergence detected, force-resync from server.", tags:["consistency"] },
    { q:"How do you handle a document with 50 concurrent editors?", a:"The single Collab Server handles 50 editors fine — 50 WS connections is trivial, and 100 ops/sec is manageable. The bottleneck is OT transform: each incoming op must be transformed against all concurrent pending ops. At 100 ops/sec, the transform queue stays short. Beyond 100-200 editors, consider: read-only overflow mode, section-based partitioning, or batching/compressing ops.", tags:["scalability"] },
    { q:"How does auto-save work? Is there a save button?", a:"No save button — every op is persisted immediately. The 'Saving...' indicator reflects real-time persistence: (1) Op submitted via WS. (2) Server writes to op log (durable). (3) Server sends ACK. (4) Client shows 'All changes saved'. If ACK doesn't arrive in 5 seconds, show 'Trying to save...'. Client retries on reconnect. This is why Google Docs feels like auto-save — there IS no save, just continuous persistence.", tags:["design"] },
    { q:"How do you implement version history / time travel?", a:"The op log IS the version history. To show the doc at any point in time: start from the nearest snapshot before that timestamp, replay ops up to that timestamp. For 'show changes since Tuesday': diff the snapshot at Tuesday vs. current doc. Named versions (user-created restore points) are just bookmarks pointing to a specific revision number.", tags:["design"] },
    { q:"How do comments survive document edits?", a:"Comments are anchored to text ranges (start_pos, end_pos). When ops modify the document, comment anchors must be transformed too — just like cursor positions. Insert before the comment range: shift right. Delete within the range: shrink it. Delete the entire range: comment becomes 'orphaned' (resolved or shown in margin). This is essentially OT for metadata ranges.", tags:["algorithm"] },
    { q:"What happens during a Collab Server crash?", a:"(1) All clients on that server lose their WS connection. (2) Clients detect disconnect, query Doc Session Registry for a new server. (3) New Collab Server loads the doc: fetch last snapshot + replay op log. (4) Clients reconnect, send their server_rev. (5) New server sends any ops the client missed. (6) Client resends any unACKed ops. Recovery time: 2-5 seconds. Key: op log is the source of truth, not the Collab Server's memory.", tags:["reliability"] },
    { q:"How would you add real-time spell checking?", a:"Run spell-check on the client side (browser's built-in spell checker or a JS library). Don't send text to server for spell-checking — it adds latency and privacy concerns. For grammar checking (Grammarly-style): async API call with the text, results arrive as suggestions (not edits). Apply suggestions via the normal OT pipeline. Rate-limit: don't re-check on every keystroke — debounce by 500ms or check on pause.", tags:["design"] },
    { q:"OT vs CRDT — when would you choose CRDT?", a:"Choose CRDT when: (1) Offline-first is critical (mobile apps, field workers). (2) P2P sync without a server (local-first apps like Ink & Switch). (3) High partition tolerance needed. (4) You want to avoid a central sequencer bottleneck. Figma uses CRDT for its design tool. Notion uses CRDT-inspired data structures. But for a Google Docs clone that's always online with a central server, OT is simpler and proven.", tags:["design"] },
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

export default function GoogleDocsSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Google Docs (Collaborative Editing)</h1>
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