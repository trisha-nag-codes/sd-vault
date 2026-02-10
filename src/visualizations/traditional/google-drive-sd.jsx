import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   GOOGLE DRIVE — System Design Reference
   Pearl white theme · 17 sections (HLD + LLD)
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",             icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",         icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",  icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",           icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",    icon: "🏗️", color: "#9333ea" },
  { id: "algorithm",     label: "Sync & Chunking",      icon: "⚙️", color: "#c026d3" },
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
            <Label>What is Google Drive / Cloud File Storage?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A cloud-based file storage and synchronization service that allows users to upload, store, share, and collaborate on files across devices. Think Google Drive, Dropbox, OneDrive, or iCloud Drive. The core challenge: keep files perfectly synchronized across multiple devices while handling large files, concurrent edits, and unreliable networks.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Unlike a simple blob store, a file sync system requires real-time notifications, chunked uploads for resumability, conflict resolution for concurrent edits, deduplication to save storage, and a rich metadata layer for sharing, permissions, and versioning. These make it fundamentally different from a CDN or object storage.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard?</Label>
            <ul className="space-y-2.5">
              <Point icon="📦" color="#0891b2">Large file handling — users upload 10GB videos. Must support resumable, chunked uploads over flaky connections without re-uploading from scratch.</Point>
              <Point icon="🔄" color="#0891b2">Multi-device sync — edit a file on laptop, see the change on phone instantly. Conflict when two devices edit the same file offline.</Point>
              <Point icon="🧩" color="#0891b2">Chunking & dedup — split files into chunks, deduplicate across users. A 1GB file edited slightly shouldn't re-upload the whole file.</Point>
              <Point icon="👥" color="#0891b2">Sharing & permissions — fine-grained ACLs (viewer, editor, commenter) with link sharing, team drives, and organizational policies.</Point>
              <Point icon="📊" color="#0891b2">Metadata at scale — billions of files, deeply nested folder trees, search across all user content, activity history per file.</Point>
              <Point icon="🌍" color="#0891b2">Consistency — user saves a file, switches devices, must see the latest version immediately. Eventual consistency is not acceptable for the active file.</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Scale</Label>
            <div className="space-y-2.5">
              {[
                { co: "Google Drive", scale: "1B+ users, 2T+ files stored", detail: "Deep G-Suite integration" },
                { co: "Dropbox", scale: "700M+ users, 600M+ files/day synced", detail: "Sync engine pioneer" },
                { co: "OneDrive", scale: "400M+ users, Microsoft 365 backbone", detail: "Deep OS integration" },
                { co: "iCloud Drive", scale: "850M+ users, Apple ecosystem", detail: "Device-first sync model" },
                { co: "Box", scale: "100K+ businesses, enterprise focus", detail: "Content mgmt + compliance" },
              ].map((e,i) => (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-28 shrink-0">{e.co}</span>
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
              <text x={90} y={28} textAnchor="middle" fill="#dc2626" fontSize="10" fontWeight="700" fontFamily="monospace">Whole-File Upload</text>
              <text x={90} y={44} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">✗ Wasteful, no resume</text>

              <rect x={190} y={10} width={160} height={45} rx={6} fill="#05966908" stroke="#059669" strokeWidth={1.5}/>
              <text x={270} y={28} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="700" fontFamily="monospace">Chunked + Dedup ★</text>
              <text x={270} y={44} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">✓ Resumable, efficient</text>

              <rect x={60} y={68} width={240} height={42} rx={6} fill="#9333ea08" stroke="#9333ea" strokeWidth={1.5}/>
              <text x={180} y={85} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Chunk files (4MB blocks) + content-hash dedup</text>
              <text x={180} y={100} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">Upload only changed chunks, resume on failure</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Top 5 most-asked system design question</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Scope Is Everything</div>
            <p className="text-[12px] text-stone-500 mt-0.5">"Design Google Drive" is massive. Clarify: file storage + sync only, or full Google Docs collaboration? Just personal files, or shared drives with permissions? For a 45-min interview, focus on <strong>file upload/download + chunking + sync across devices + sharing</strong>. Real-time collaborative editing (Google Docs) is a separate system.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Upload files — any type, up to 10GB, resumable on network failure</Point>
            <Point icon="2." color="#059669">Download files — stream or direct download with CDN acceleration</Point>
            <Point icon="3." color="#059669">Auto-sync — file changes on one device sync to all linked devices</Point>
            <Point icon="4." color="#059669">File versioning — view and restore previous versions of any file</Point>
            <Point icon="5." color="#059669">Sharing & permissions — share via link or email with viewer/editor roles</Point>
            <Point icon="6." color="#059669">Folder hierarchy — nested folders, move, rename, delete (with trash)</Point>
            <Point icon="7." color="#059669">Search — full-text search across file names, and optionally content</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Upload latency: limited by network speed, not server processing</Point>
            <Point icon="2." color="#dc2626">Zero data loss — files must never be corrupted or lost</Point>
            <Point icon="3." color="#dc2626">Sync latency: changes visible on other devices within 5-10 seconds</Point>
            <Point icon="4." color="#dc2626">Scale to 1B+ users, 1T+ files, 100M+ DAU</Point>
            <Point icon="5." color="#dc2626">High availability — 99.99% uptime (users depend on Drive for work)</Point>
            <Point icon="6." color="#dc2626">Bandwidth efficiency — only transfer changed chunks, not whole files</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Max file size? (typically 5-10GB cap)",
            "Do we need real-time collaborative editing?",
            "Auto-sync client (desktop agent) or web-only?",
            "Versioning — how many versions retained? Forever?",
            "Sharing model — link-based, email-invite, or both?",
            "Offline access support? (sync subset of files locally)",
            "Storage quota per user? (15GB free, expandable?)",
            "What scale? (total users, DAU, files stored)",
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
            <MathStep step="1" formula="Total users = 1B" result="1B" note="Google Drive scale." />
            <MathStep step="2" formula="DAU = 100M (10% of total)" result="100M" note="Active users who open/sync daily." />
            <MathStep step="3" formula="Uploads per DAU/day = 2 files" result="2" note="New files + file modifications (saves)." />
            <MathStep step="4" formula="Total uploads/day = 100M × 2" result="200M" note="Files uploaded or modified per day." final />
            <MathStep step="5" formula="Uploads/sec = 200M / 86,400" result="~2,300/sec" note="Average upload rate." final />
            <MathStep step="6" formula="Peak uploads/sec = 2,300 × 3" result="~7,000/sec" note="Business hours spike." final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Storage Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Average file size = 1 MB" result="1 MB" note="Blended: docs ~100KB, photos ~3MB, videos ~500MB." />
            <MathStep step="2" formula="Daily new storage = 200M × 1MB" result="~200 TB/day" note="Raw new data ingested daily." />
            <MathStep step="3" formula="Dedup savings ≈ 40%" result="~120 TB/day" note="Cross-user dedup (same attachments, templates, etc.)." />
            <MathStep step="4" formula="Yearly storage = 120TB × 365" result="~44 PB/year" note="After dedup. Before replication." final />
            <MathStep step="5" formula="Replication factor = 3" result="~132 PB/year" note="3 copies across regions for durability." final />
            <MathStep step="6" formula="Total stored (5 years)" result="~660 PB" note="Petabyte-scale blob storage (S3/GCS)." />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Step 3 — Bandwidth</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Upload BW = 200M × 1MB / 86,400" result="~2.3 GB/s" note="Average upload bandwidth." />
            <MathStep step="2" formula="Read:Write ratio ≈ 3:1" result="3:1" note="Users download/view more than they upload." />
            <MathStep step="3" formula="Download BW = 2.3 GB/s × 3" result="~7 GB/s" note="CDN handles most download traffic." final />
            <MathStep step="4" formula="Peak total BW (5× avg)" result="~46 GB/s" note="CDN edge caching critical for this." final />
          </div>
        </Card>
        <Card>
          <Label color="#d97706">Quick Reference</Label>
          <div className="grid grid-cols-3 gap-2.5">
            {[
              { label: "DAU", val: "100M", sub: "Active users/day" },
              { label: "Uploads/sec", val: "~2.3K", sub: "Avg; 7K peak" },
              { label: "Daily Ingest", val: "~120 TB", sub: "After dedup" },
              { label: "Avg File", val: "~1 MB", sub: "Blended average" },
              { label: "Read:Write", val: "3:1", sub: "CDN-served reads" },
              { label: "Yearly Storage", val: "~44 PB", sub: "Before replication" },
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
          <Label color="#2563eb">File Upload APIs (Resumable)</Label>
          <CodeBlock code={`# POST /v1/files/upload/init
# Initialize a resumable upload session
{
  "name": "presentation.pptx",
  "parent_folder_id": "folder_abc",
  "size": 52428800,              # 50 MB
  "mime_type": "application/vnd.ms-powerpoint",
  "checksum_sha256": "a1b2c3..."  # Full file hash
}
# Returns: { upload_id: "upl_xyz", chunk_size: 4194304,
#            upload_url: "/v1/files/upload/upl_xyz" }

# PUT /v1/files/upload/:upload_id
# Upload one chunk (repeat for each chunk)
# Headers:
#   Content-Range: bytes 0-4194303/52428800
#   Content-MD5: <chunk_hash>
# Body: raw binary chunk (4 MB)
# Returns: { chunk_index: 0, status: "received" }

# POST /v1/files/upload/:upload_id/complete
# Finalize upload — server assembles chunks
# Returns: { file_id: "file_789", version: 1,
#            created_at: "2025-01-15T10:30:00Z" }

# GET /v1/files/upload/:upload_id/status
# Resume check — which chunks are already uploaded?
# Returns: { received_chunks: [0,1,2],
#            missing_chunks: [3,4,...,12] }`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">File & Folder APIs</Label>
          <CodeBlock code={`# GET /v1/files/:id
# Download a file (or redirect to CDN URL)
# Query: ?version=3   (specific version)
# Returns: 302 redirect to signed CDN URL

# GET /v1/files/:id/metadata
# File metadata without downloading content
# Returns: { file_id, name, size, mime_type,
#   owner, permissions, versions: [...],
#   parent_folder_id, created_at, updated_at }

# GET /v1/folders/:id/contents?cursor=&limit=50
# List folder contents (files + subfolders)
# Returns: { items: [...], next_cursor, has_more }

# POST /v1/files/:id/share
# Share a file with specific users or via link
{
  "grants": [
    { "email": "alice@co.com", "role": "editor" },
    { "email": "bob@co.com", "role": "viewer" }
  ],
  "link_sharing": {
    "enabled": true,
    "role": "viewer",           # anyone with link
    "expires_at": "2025-03-01"
  }
}

# DELETE /v1/files/:id  → move to trash
# DELETE /v1/files/:id?permanent=true → hard delete`} />
          <div className="mt-3 space-y-2">
            {[
              { q: "Why resumable upload instead of single POST?", a: "Large files (GB+) over unreliable networks need resume. Single POST = start over on any failure. Chunked = resume from last successful chunk." },
              { q: "Why redirect to CDN for downloads?", a: "Serving files from origin is expensive and slow. CDN caches popular files at edge. Signed URL prevents unauthorized access. URL expires after a TTL." },
              { q: "Why separate metadata from content?", a: "Listing folders, showing file info, permission checks — all metadata. You don't download 500 files to show a folder. Keep metadata in fast DB, content in blob store." },
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
        <svg viewBox="0 0 720 360" className="w-full">
          {/* Clients */}
          <rect x={10} y={100} width={65} height={34} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={42} y={120} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Desktop</text>
          <rect x={10} y={150} width={65} height={34} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={42} y={170} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Mobile</text>
          <rect x={10} y={200} width={65} height={34} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={42} y={220} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Web App</text>

          {/* LB + API Gateway */}
          <rect x={110} y={140} width={70} height={46} rx={6} fill="#6366f110" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={145} y={158} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">API</text>
          <text x={145} y={170} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">Gateway</text>

          {/* Upload Service */}
          <rect x={220} y={70} width={90} height={34} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={265} y={91} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Upload Svc</text>

          {/* Metadata Service */}
          <rect x={220} y={120} width={90} height={34} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={265} y={141} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">Metadata Svc</text>

          {/* Download Service */}
          <rect x={220} y={170} width={90} height={34} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={265} y={191} textAnchor="middle" fill="#0891b2" fontSize="9" fontWeight="600" fontFamily="monospace">Download Svc</text>

          {/* Sync Service */}
          <rect x={220} y={220} width={90} height={34} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={265} y={241} textAnchor="middle" fill="#ea580c" fontSize="9" fontWeight="600" fontFamily="monospace">Sync Svc</text>

          {/* Chunk Store (Blob) */}
          <rect x={380} y={60} width={90} height={38} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={425} y={77} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Chunk Store</text>
          <text x={425} y={90} textAnchor="middle" fill="#78716c80" fontSize="7" fontFamily="monospace">S3 / GCS</text>

          {/* Metadata DB */}
          <rect x={380} y={115} width={90} height={38} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={425} y={132} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Metadata DB</text>
          <text x={425} y={145} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">PostgreSQL</text>

          {/* Message Queue */}
          <rect x={380} y={170} width={90} height={38} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={425} y={187} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Msg Queue</text>
          <text x={425} y={200} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">Kafka</text>

          {/* Notification Service */}
          <rect x={380} y={225} width={90} height={38} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={425} y={242} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Notification</text>
          <text x={425} y={255} textAnchor="middle" fill="#9333ea80" fontSize="7" fontFamily="monospace">push to clients</text>

          {/* CDN */}
          <rect x={530} y={60} width={80} height={38} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={570} y={77} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">CDN</text>
          <text x={570} y={90} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">CloudFront</text>

          {/* Dedup Service */}
          <rect x={530} y={115} width={80} height={38} rx={6} fill="#0f766e10" stroke="#0f766e" strokeWidth={1.5}/>
          <text x={570} y={132} textAnchor="middle" fill="#0f766e" fontSize="9" fontWeight="600" fontFamily="monospace">Dedup Svc</text>
          <text x={570} y={145} textAnchor="middle" fill="#0f766e80" fontSize="7" fontFamily="monospace">content hash</text>

          {/* Search Service */}
          <rect x={530} y={170} width={80} height={38} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
          <text x={570} y={187} textAnchor="middle" fill="#7c3aed" fontSize="9" fontWeight="600" fontFamily="monospace">Search</text>
          <text x={570} y={200} textAnchor="middle" fill="#7c3aed80" fontSize="7" fontFamily="monospace">Elasticsearch</text>

          {/* Sharing / ACL Service */}
          <rect x={530} y={225} width={80} height={38} rx={6} fill="#be123c10" stroke="#be123c" strokeWidth={1.5}/>
          <text x={570} y={242} textAnchor="middle" fill="#be123c" fontSize="9" fontWeight="600" fontFamily="monospace">Sharing</text>
          <text x={570} y={255} textAnchor="middle" fill="#be123c80" fontSize="7" fontFamily="monospace">ACL engine</text>

          {/* Arrows */}
          <defs><marker id="ah-hld" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          <line x1={75} y1={117} x2={110} y2={152} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={75} y1={167} x2={110} y2={163} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={75} y1={217} x2={110} y2={175} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={180} y1={152} x2={220} y2={87} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={180} y1={158} x2={220} y2={137} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={180} y1={163} x2={220} y2={187} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={180} y1={170} x2={220} y2={237} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={310} y1={87} x2={380} y2={79} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={310} y1={137} x2={380} y2={134} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={310} y1={187} x2={380} y2={189} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={310} y1={237} x2={380} y2={244} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={470} y1={79} x2={530} y2={79} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={470} y1={134} x2={530} y2={134} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={470} y1={189} x2={530} y2={189} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={470} y1={244} x2={530} y2={244} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>

          {/* Flow labels */}
          <rect x={10} y={285} width={700} height={65} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={20} y={302} fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Upload flow: Client chunks file → Upload Svc receives chunks → Dedup Svc checks hashes → Chunk Store (S3) → Metadata DB updated</text>
          <text x={20} y={317} fill="#0891b2" fontSize="8" fontWeight="600" fontFamily="monospace">Download flow: Client requests file → Download Svc → signed CDN URL → CDN serves from edge (cache hit) or origin (S3)</text>
          <text x={20} y={332} fill="#ea580c" fontSize="8" fontWeight="600" fontFamily="monospace">Sync flow: File change → Kafka event → Notification Svc → push to all user's connected devices → clients pull delta</text>
          <text x={20} y={347} fill="#9333ea" fontSize="8" fontWeight="600" fontFamily="monospace">Key insight: Metadata (DB) and Content (Blob) are separate. Metadata is small + hot. Content is large + cold. Never mix them.</text>
        </svg>
      </Card>
      <Card>
        <Label color="#c026d3">Key Architecture Decisions</Label>
        <div className="grid grid-cols-3 gap-4">
          {[
            { q: "Why separate metadata from file content?", a: "Metadata is small (KB), structured, queried heavily (list folder, search, permissions). File content is large (MB-GB), unstructured, accessed infrequently. Different storage engines for each: SQL DB for metadata, blob store (S3) for content." },
            { q: "Why chunk files instead of storing whole files?", a: "Chunking enables: (1) resumable uploads — retry only the failed chunk, (2) deduplication — identical chunks across users stored once, (3) delta sync — only upload changed chunks on file edit, (4) parallel upload/download of chunks." },
            { q: "Why Kafka for sync notifications?", a: "When a file changes, ALL the user's devices need to know. Kafka provides durable, ordered events. Sync Service subscribes per-user. Even if a device is offline, it catches up from its last offset when it reconnects." },
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
  const [sel, setSel] = useState("chunking");
  const topics = {
    chunking: { name: "File Chunking", cx: "4 MB Fixed vs Content-Defined",
      desc: "Files are split into chunks before upload. The chunking strategy determines sync efficiency, dedup ratio, and upload/download parallelism. This is the most important algorithm decision in the system.",
      code: `# Fixed-Size Chunking (simple, fast)
def fixed_chunk(file, chunk_size=4*1024*1024):  # 4MB
    chunks = []
    while data := file.read(chunk_size):
        chunk_hash = sha256(data)
        chunks.append({
            "index": len(chunks),
            "hash": chunk_hash,
            "size": len(data),
            "data": data
        })
    return chunks
# Problem: inserting 1 byte at start shifts ALL chunks
# Every chunk hash changes → full re-upload

# Content-Defined Chunking (Rabin fingerprint) ★
def cdc_chunk(file, min=2MB, max=8MB, avg=4MB):
    # Sliding window Rabin fingerprint
    # Split when fingerprint % avg_size == magic_value
    # Chunk boundaries are DATA-DEPENDENT
    # Insert at start → only first chunk changes
    chunks = []
    buf, window = bytearray(), RabinWindow(48)
    for byte in file:
        buf.append(byte)
        fp = window.slide(byte)
        if len(buf) >= min and (
            fp % avg == MAGIC or len(buf) >= max
        ):
            chunk_hash = sha256(buf)
            chunks.append({"hash": chunk_hash, "data": buf})
            buf = bytearray()
    return chunks
# Result: edit middle of file → only 1-2 chunks change
# 40-60% dedup ratio across similar files`,
      points: [
        "Fixed-size chunks (4MB): simple, fast, good for initial upload. But inserting data shifts all subsequent chunks — bad for sync (every chunk hash changes).",
        "Content-defined chunking (CDC): boundaries determined by content (Rabin fingerprint). Insertions only affect the chunk where the edit happened. Nearby chunks stay identical.",
        "Dropbox uses CDC with ~4MB average. Google Drive uses fixed chunks for simplicity + server-side diff for sync. Both work at scale — CDC is better for delta sync.",
        "Chunk hashes (SHA-256) are the key to deduplication: before uploading a chunk, check if that hash already exists in the store. If yes, skip upload — just reference the existing chunk.",
      ],
    },
    dedup: { name: "Deduplication", cx: "Content-Addressable Storage",
      desc: "Same file (or same chunk) uploaded by different users should be stored only once. This is content-addressable storage: the chunk's hash IS its address. Saves 30-50% storage at scale.",
      code: `# Upload with dedup check
def upload_chunk(chunk_data):
    chunk_hash = sha256(chunk_data)

    # Check if chunk already exists (O(1) lookup)
    existing = chunk_store.lookup(chunk_hash)
    if existing:
        # Skip upload! Just increment reference count
        chunk_store.increment_ref(chunk_hash)
        return {"hash": chunk_hash, "status": "deduped"}

    # New chunk — store it
    chunk_store.put(chunk_hash, chunk_data)
    chunk_store.set_ref_count(chunk_hash, 1)
    return {"hash": chunk_hash, "status": "stored"}

# File = ordered list of chunk hashes (manifest)
# file_manifest = [hash_1, hash_2, ..., hash_N]
# Reconstruct file: fetch chunks by hash, concatenate

# Dedup across users:
# User A uploads report.pdf → chunks [h1, h2, h3]
# User B uploads same report.pdf
#   → hash check: h1 exists, h2 exists, h3 exists
#   → ZERO bytes uploaded. Instant "upload".
#   → User B's manifest points to same chunks

# Reference counting for deletion:
# User A deletes → decrement ref on h1, h2, h3
# Chunks still exist (User B references them)
# Delete chunk only when ref_count = 0`,
      points: [
        "Content-addressable storage: chunk hash = storage key. Identical content always maps to the same key. Global dedup across all users.",
        "Dedup check BEFORE upload: client sends chunk hashes first. Server responds which chunks are needed. Client only uploads new/unique chunks. Saves bandwidth.",
        "Reference counting: each chunk tracks how many file manifests reference it. Only garbage-collected when ref_count drops to zero. Must be atomic (distributed counter).",
        "Security consideration: hash-based dedup can leak info (attacker can probe if a file exists). Mitigate: per-user encryption keys (no cross-user dedup) or accept the trade-off for storage savings.",
      ],
    },
    sync: { name: "Sync Protocol", cx: "How Devices Stay in Sync",
      desc: "The sync engine is the brain of the desktop/mobile client. It watches the local filesystem for changes, computes diffs against the server state, and resolves conflicts. This is what makes Dropbox magical.",
      code: `# Client-side sync engine (simplified)
class SyncEngine:
    def __init__(self):
        self.local_state = {}   # path → {hash, mtime, version}
        self.remote_state = {}  # path → {hash, version}

    def detect_local_changes(self):
        # Watch filesystem (inotify on Linux, FSEvents on macOS)
        # Compare current state vs local_state snapshot
        for path in watched_paths:
            current_hash = compute_hash(path)
            if current_hash != self.local_state[path].hash:
                yield LocalChange(path, "modified")

    def sync_to_server(self, change):
        # 1. Chunk the file (CDC)
        chunks = cdc_chunk(open(change.path))
        # 2. Check which chunks server needs
        needed = server.check_chunks([c.hash for c in chunks])
        # 3. Upload only missing chunks
        for chunk in needed:
            server.upload_chunk(chunk)
        # 4. Update file manifest on server
        server.update_manifest(change.path, chunks, version)

    def handle_remote_change(self, event):
        # Server notifies: file changed by another device
        # 1. Fetch new manifest (list of chunk hashes)
        manifest = server.get_manifest(event.file_id)
        # 2. Diff with local chunks
        needed = [h for h in manifest if not local_cache.has(h)]
        # 3. Download missing chunks
        for h in needed:
            local_cache.store(h, server.download_chunk(h))
        # 4. Reconstruct file from chunks
        reconstruct(event.path, manifest)

    def resolve_conflict(self, local_ver, remote_ver):
        # Both changed since last sync — CONFLICT
        # Strategy: keep both, let user decide
        # Save as "file (conflict copy - Device A).ext"
        save_conflict_copy(local_ver)
        apply_remote(remote_ver)`,
      points: [
        "File system watcher: client monitors local folder for changes (inotify/FSEvents/ReadDirectoryChanges). Batches changes to avoid syncing on every keystroke.",
        "Push + pull model: server pushes change notifications (long-poll or WebSocket). Client pulls the actual data (chunk download). Push is lightweight; pull is heavy.",
        "Delta sync: on file edit, re-chunk → compare chunk hashes with server → upload only changed chunks. A 1GB file with a 100-byte edit uploads ~4MB (one chunk).",
        "Conflict resolution: if two devices edit the same file offline, keep both copies. No automatic merge (too risky for binary files). User resolves manually. Google Docs handles this differently (OT/CRDT).",
      ],
    },
    conflict: { name: "Conflict Resolution", cx: "Last-Write-Wins vs Fork",
      desc: "When two devices edit the same file before syncing, you have a conflict. The resolution strategy varies by file type and product philosophy. There is no perfect answer — only trade-offs.",
      code: `# Conflict detection using version vectors
class VersionVector:
    # Each device has a counter. Increment on edit.
    # device_A: {A: 5, B: 3}
    # device_B: {A: 3, B: 4}
    # Neither dominates → CONFLICT

    def dominates(self, other):
        return all(
            self.versions.get(k, 0) >= other.versions.get(k, 0)
            for k in set(self.versions) | set(other.versions)
        ) and self.versions != other.versions

# Resolution strategies:
# 1. Last-Write-Wins (LWW)
#    Use timestamp: latest edit wins, older is discarded
#    Simple but LOSES DATA. Only for low-value files.
#    Problem: clock skew between devices

# 2. Fork (Conflict Copy) ★ — Dropbox/Google Drive
#    Keep BOTH versions. Rename conflicting file:
#    "report.docx" + "report (conflict copy).docx"
#    User manually merges. Safe but annoying.

# 3. Operational Transform (OT) — Google Docs
#    For text documents: merge concurrent edits
#    character-by-character. Complex but seamless.
#    Only works for structured text, not binary.

# 4. CRDT — Figma, some editors
#    Conflict-free Replicated Data Types
#    Math guarantees convergence. Works offline.
#    Complex to implement for rich document types.

# Practical choice for file storage:
# Binary files → Fork (conflict copy)
# Text files → Fork or simple 3-way merge
# Collaborative docs → OT/CRDT (separate system)`,
      points: [
        "Version vectors (not timestamps): each device maintains a vector of counters. Compare vectors to detect concurrent edits. Timestamps have clock skew — version vectors are reliable.",
        "Fork / conflict copy is the standard for file storage systems. Safe: no data loss. Annoying: user must resolve. But for binary files (images, PDFs), there's no way to auto-merge anyway.",
        "OT (Operational Transform) works for Google Docs-style collaboration. But it's a separate system from file storage. Don't try to use OT for general file sync — scope creep in interviews.",
        "In practice: conflicts are rare (<0.1% of syncs). Most users edit files on one device at a time. Optimize for the common case (no conflict), handle the rare case safely (fork).",
      ],
    },
  };
  const t = topics[sel];
  return (
    <div className="space-y-5">
      <Card className="bg-fuchsia-50/50 border-fuchsia-200">
        <div className="flex items-start gap-3">
          <span className="text-lg">⚙️</span>
          <div>
            <div className="text-[12px] font-bold text-fuchsia-700">Core Algorithms — The Heart of the System</div>
            <p className="text-[12px] text-stone-500 mt-0.5">File chunking, deduplication, and sync are what differentiate a cloud file storage system from a simple blob store. Interviewers dig deep here. Know chunking strategies, dedup trade-offs, and conflict resolution cold.</p>
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
          <CodeBlock code={`-- Files (sharded by owner_id for "my files" queries)
CREATE TABLE files (
  file_id         BIGINT PRIMARY KEY,   -- Snowflake ID
  owner_id        BIGINT NOT NULL,
  parent_folder_id BIGINT,              -- NULL = root
  name            VARCHAR(255) NOT NULL,
  mime_type       VARCHAR(128),
  size_bytes      BIGINT NOT NULL,
  current_version INT NOT NULL DEFAULT 1,
  is_folder       BOOLEAN DEFAULT FALSE,
  is_trashed      BOOLEAN DEFAULT FALSE,
  checksum_sha256 CHAR(64),
  created_at      TIMESTAMP NOT NULL,
  updated_at      TIMESTAMP NOT NULL,
  INDEX idx_parent (parent_folder_id, name),
  INDEX idx_owner  (owner_id, updated_at DESC)
);

-- File Versions (append-only, per file)
CREATE TABLE file_versions (
  version_id      BIGINT PRIMARY KEY,
  file_id         BIGINT NOT NULL,
  version_num     INT NOT NULL,
  size_bytes      BIGINT,
  chunk_manifest  JSONB,     -- ["hash1","hash2",...]
  editor_id       BIGINT,    -- who made this version
  created_at      TIMESTAMP,
  UNIQUE (file_id, version_num)
);

-- Chunks (content-addressable, global dedup)
CREATE TABLE chunks (
  chunk_hash      CHAR(64) PRIMARY KEY, -- SHA-256
  size_bytes      INT NOT NULL,
  ref_count       INT NOT NULL DEFAULT 1,
  storage_key     VARCHAR(512),  -- S3/GCS object key
  created_at      TIMESTAMP
);

-- Sharing / ACL
CREATE TABLE file_shares (
  share_id        BIGINT PRIMARY KEY,
  file_id         BIGINT NOT NULL,
  grantee_id      BIGINT,        -- NULL = link share
  grantee_email   VARCHAR(255),
  role            ENUM('viewer','commenter','editor'),
  link_token      CHAR(32),      -- for link sharing
  expires_at      TIMESTAMP,
  INDEX idx_file (file_id),
  INDEX idx_user (grantee_id)
);`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Why These Design Choices?</Label>
          <div className="space-y-3">
            {[
              { q: "Shard files by owner_id?", a: "GET /users/me/files (file list) is THE hot query. Single-shard scan by owner_id. Trade-off: listing a shared folder with files from many owners requires scatter-gather (but cached heavily)." },
              { q: "Separate file_versions table?", a: "Append-only version history. Never mutate a version. Files table has current_version pointer. To restore: update the pointer. Old versions are immutable." },
              { q: "JSONB for chunk_manifest?", a: "Ordered list of chunk hashes. Read as a whole (to reconstruct file). Never queried by individual chunk. JSONB is flexible: variable number of chunks per file." },
              { q: "Why ref_count on chunks?", a: "Dedup means many files reference the same chunk. ref_count tracks how many manifests reference it. Garbage-collect when ref_count = 0. Must be atomically updated (distributed counter or DB transaction)." },
              { q: "Why PostgreSQL for metadata?", a: "Strong consistency for file operations (rename, move, delete). ACID transactions for multi-row updates (move folder = update parent_folder_id for all children). Rich query support for search. Shard with Citus or Vitess." },
              { q: "Folder as a file row with is_folder?", a: "Simplifies the tree structure. A folder is just a file row with is_folder=true, no chunk_manifest. Parent-child via parent_folder_id. Recursive queries for 'path to root' via CTE or materialized path." },
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
          <Label color="#059669">Metadata Layer Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Shard by owner_id</strong> — each user's file tree lives on one shard. "List my files" is single-shard. 1B users / 1000 shards = 1M users per shard. Each shard ~10M file rows.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Read replicas for shared files</strong> — viewing shared files reads from replicas. Write to primary (owner's shard). Slight replication lag OK for viewers (not the editor).</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Hot folder cache (Redis)</strong> — cache folder listings for frequently accessed shared folders. Invalidate on any child change. Reduces DB reads 10×.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Materialized path for folder tree</strong> — store path as "/root/folder_a/folder_b/file" for fast subtree queries. Alternative: closure table for ancestor lookups.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Storage Layer Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Blob store is the easy part</strong> — S3/GCS scale to exabytes. Object storage is horizontally scalable by design. No sharding decisions needed. Cost-optimize with tiering.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Storage tiering</strong> — hot (frequently accessed, SSD), warm (30-90 days, HDD), cold (90+ days, glacier/archive). Auto-tier based on last access time. 80% of files are cold.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">CDN for downloads</strong> — cache popular/shared files at CDN edge. Short-lived signed URLs for access control. Cache hit ratio 60-80% for shared files.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Chunk-level dedup</strong> — content-hash dedup saves 30-50% storage globally. Same email attachment uploaded by 1000 users = stored once. Requires global chunk index.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Strategy</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Option A: User-Homed Regions ★", d:"Each user's files stored in their nearest region. Metadata and chunks co-located. Cross-region sharing via replication.", pros:["Low latency for file owner","Data residency compliance (GDPR)","Each region self-contained for local users"], cons:["Shared files across regions add latency","Cross-region replication for collaboration","User relocation needs data migration"], pick:true },
            { t:"Option B: Metadata Global, Content Regional", d:"Metadata DB replicated globally. File chunks stored in user's region. Reads from closest replica.", pros:["Fast metadata ops everywhere","Content stays local (data residency)","Simple sharing model for metadata"], cons:["Download latency for cross-region shares","Two separate replication strategies","Metadata DB replication complexity"], pick:false },
            { t:"Option C: Follow-the-Reader", d:"File chunks replicated to regions where they are frequently accessed. Lazy replication on first access.", pros:["Optimizes for read-heavy workloads","Shared files auto-replicate to readers","No upfront region assignment"], cons:["First access is slow (cache miss)","Unbounded replication (popular files everywhere)","Complex garbage collection"], pick:false },
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
        <Label color="#d97706">Critical: Files Must Never Be Lost or Corrupted</Label>
        <p className="text-[12px] text-stone-500 mb-4">Users trust cloud storage with irreplaceable data — family photos, legal documents, business files. Losing a single file destroys trust permanently. Every design decision prioritizes durability over performance.</p>
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">11 Nines Durability (99.999999999%)</div>
            <p className="text-[11px] text-stone-500">S3/GCS provides 11 nines. 3 copies across availability zones. Reed-Solomon erasure coding for cold storage. If you store 10M files, expect to lose 0.000001 files per year.</p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-white p-4">
            <div className="text-[11px] font-bold text-amber-700 mb-1.5">Checksum Everything</div>
            <p className="text-[11px] text-stone-500">SHA-256 checksum on every chunk upload. Verify on download. Background integrity scrubber compares stored chunks against checksums. Detect bit-rot before it's too late.</p>
          </div>
          <div className="rounded-lg border border-blue-200 bg-white p-4">
            <div className="text-[11px] font-bold text-blue-700 mb-1.5">Write-Ahead: Metadata After Content</div>
            <p className="text-[11px] text-stone-500">Only create metadata DB entry AFTER all chunks are durably stored in blob store. If upload fails mid-way, no dangling metadata. Orphan chunks are garbage-collected.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#2563eb">Component Failure Handling</Label>
          <ul className="space-y-2.5">
            <Point icon="→" color="#2563eb">Upload Service crash → client detects timeout → resumes from last successful chunk (upload_id tracks state).</Point>
            <Point icon="→" color="#2563eb">Metadata DB failover → promote read replica to primary in &lt;30s. Writes block briefly. Reads continue from other replicas.</Point>
            <Point icon="→" color="#2563eb">Blob store unavailable (rare) → uploads queue in Kafka. Drain when store recovers. Downloads fail gracefully with retry.</Point>
            <Point icon="→" color="#2563eb">Sync Service crash → clients detect broken long-poll/WS → reconnect → sync from last known version (no data loss, just delayed).</Point>
            <Point icon="→" color="#2563eb">CDN outage → fallback to origin (blob store) directly. Higher latency but functional. CDN is a cache, not the source.</Point>
          </ul>
        </Card>
        <Card>
          <Label color="#d97706">Degradation Ladder</Label>
          <div className="flex flex-col gap-2 mt-1">
            {[
              { label: "Full System", sub: "Upload + download + real-time sync + sharing", color: "#059669", status: "HEALTHY" },
              { label: "No Real-Time Sync", sub: "Upload/download works, sync is delayed", color: "#d97706", status: "DEGRADED" },
              { label: "Read-Only", sub: "Downloads + viewing works, uploads paused", color: "#ea580c", status: "FALLBACK" },
              { label: "Cached Only", sub: "CDN-cached files accessible, origin down", color: "#dc2626", status: "EMERGENCY" },
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
              { metric: "upload.latency_ms", type: "Histogram", desc: "Time from first chunk to upload complete. Target: limited by network, not server." },
              { metric: "upload.failure_rate", type: "Counter", desc: "Uploads that fail after all retries. Target: <0.01%. Any higher = investigate." },
              { metric: "sync.propagation_s", type: "Histogram", desc: "Time from file save on device A to notification on device B. Target: p95 < 10s." },
              { metric: "chunk.dedup_ratio", type: "Gauge", desc: "% of chunks skipped due to dedup. Tracks storage savings. Expect 30-50%." },
              { metric: "download.cdn_hit_rate", type: "Gauge", desc: "% of downloads served from CDN cache. Target: >60% for shared files." },
              { metric: "metadata.query_latency_ms", type: "Histogram", desc: "Folder listing, search, permission check. Target: p99 < 100ms." },
            ].map((m,i) => (
              <div key={i} className="flex items-start gap-2">
                <span className="font-mono text-[10px] text-sky-600 shrink-0 mt-0.5 w-40 truncate">{m.metric}</span>
                <div className="flex-1">
                  <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-sky-100 text-sky-700">{m.type}</span>
                  <p className="text-[10px] text-stone-500 mt-0.5">{m.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Alerting Rules</Label>
          <div className="space-y-2.5">
            {[
              { rule: "upload.failure_rate > 1%", sev: "P0", action: "Page on-call. Check blob store health, upload service errors, disk space." },
              { rule: "sync.propagation_s p95 > 30s", sev: "P1", action: "Kafka lag? Notification service backlogged? Check consumer group health." },
              { rule: "chunk.integrity_failures > 0", sev: "P0", action: "Data corruption detected. Halt affected shard. Restore from replica. RCA required." },
              { rule: "metadata.query_latency p99 > 500ms", sev: "P2", action: "DB slow queries, missing index, or shard hotspot. Check query plans." },
              { rule: "storage.quota_exceeded users > 10K", sev: "P2", action: "Mass quota violations. Check for abuse, bulk uploads, or quota calculation bug." },
            ].map((a,i) => (
              <div key={i} className="rounded-lg border border-stone-100 p-2.5">
                <div className="flex items-center gap-2">
                  <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${a.sev==="P0"?"bg-red-100 text-red-700":a.sev==="P1"?"bg-amber-100 text-amber-700":"bg-blue-100 text-blue-700"}`}>{a.sev}</span>
                  <span className="font-mono text-[10px] text-stone-700">{a.rule}</span>
                </div>
                <p className="text-[10px] text-stone-400 mt-1">{a.action}</p>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Distributed Tracing</Label>
          <div className="space-y-2.5">
            <p className="text-[11px] text-stone-500">Every file operation gets a trace_id. Follow the upload from client through every service:</p>
            {[
              { span: "client.upload_init", svc: "Desktop App", dur: "2ms" },
              { span: "api.auth_check", svc: "API Gateway", dur: "5ms" },
              { span: "upload.receive_chunk", svc: "Upload Svc", dur: "50ms" },
              { span: "dedup.check_hash", svc: "Dedup Svc", dur: "3ms" },
              { span: "blob.put_chunk", svc: "S3/GCS", dur: "120ms" },
              { span: "metadata.update_manifest", svc: "Metadata Svc", dur: "15ms" },
              { span: "sync.publish_change", svc: "Kafka", dur: "8ms" },
              { span: "notify.push_devices", svc: "Notification", dur: "25ms" },
            ].map((s,i) => (
              <div key={i} className="flex items-center gap-2 text-[10px]">
                <span className="font-mono text-purple-600 w-44 shrink-0">{s.span}</span>
                <span className="text-stone-400 w-24 shrink-0">{s.svc}</span>
                <div className="flex-1 h-3 bg-stone-100 rounded-full overflow-hidden">
                  <div className="h-full bg-purple-400 rounded-full" style={{ width: `${Math.min(100, parseInt(s.dur) * 0.7)}%` }} />
                </div>
                <span className="font-mono text-stone-500 w-12 text-right">{s.dur}</span>
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
        <Label color="#dc2626">Failure Modes & Mitigations</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[11px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500 w-40">Failure</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Impact</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Mitigation</th>
            </tr></thead>
            <tbody>
              {[
                { f: "Incomplete upload (chunk missing)", i: "File appears uploaded but is corrupted — missing chunks when downloading.", m: "Server validates chunk_manifest completeness before marking upload as 'done'. Client verifies checksum after download. Background scrubber checks all manifests." },
                { f: "Metadata DB ↔ Blob store inconsistency", i: "Metadata says file exists but chunks are missing, or orphan chunks with no metadata.", m: "Reconciliation job: compare metadata manifests against blob store weekly. Orphan chunks GC'd. Missing chunks → alert + restore from backup." },
                { f: "Dedup reference count drift", i: "ref_count undercount → chunk deleted while still referenced → data loss. Overcount → storage leak.", m: "Use atomic DB transactions for ref_count. Periodic reconciliation: scan all manifests, recompute ref_counts. Soft-delete with 30-day grace before hard delete." },
                { f: "Sync storm (many devices, many changes)", i: "User with 10 devices makes rapid edits. Each generates sync events. N×N notification amplification.", m: "Debounce: batch changes within 2s window. Coalesce notifications: 'folder changed' not 'file1, file2, file3 changed'. Rate-limit per-user sync events." },
                { f: "Large folder listing (10K+ files)", i: "Loading a folder with 50K files is slow — full table scan, huge JSON response, client UI freezes.", m: "Paginate always (cursor-based, 100 items/page). Server-side sort. Cache hot folders. Consider virtual scrolling on client." },
                { f: "Quota enforcement race condition", i: "User uploads 10 files simultaneously, each passes quota check, total exceeds quota.", m: "Optimistic check on upload init (non-blocking). Hard check on upload complete (atomic: increment used_quota in same transaction as metadata write)." },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-mono text-stone-700 font-medium">{r.f}</td>
                  <td className="px-3 py-2 text-red-500">{r.i}</td>
                  <td className="px-3 py-2 text-stone-500">{r.m}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function ServicesSection() {
  return (
    <div className="space-y-5">
      <Card accent="#0f766e">
        <Label color="#0f766e">Detailed Service Architecture</Label>
        <svg viewBox="0 0 720 370" className="w-full">
          {/* Client Layer */}
          <rect x={10} y={30} width={80} height={32} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={50} y={50} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Sync Client</text>

          <rect x={10} y={75} width={80} height={32} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={50} y={95} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Web / Mobile</text>

          {/* API Gateway */}
          <rect x={120} y={50} width={70} height={42} rx={6} fill="#6366f110" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={155} y={68} textAnchor="middle" fill="#6366f1" fontSize="8" fontWeight="600" fontFamily="monospace">API</text>
          <text x={155} y={80} textAnchor="middle" fill="#6366f1" fontSize="8" fontWeight="600" fontFamily="monospace">Gateway</text>

          {/* Core Services */}
          <rect x={225} y={15} width={85} height={32} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={267} y={35} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Upload Svc</text>

          <rect x={225} y={58} width={85} height={32} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={267} y={78} textAnchor="middle" fill="#0891b2" fontSize="8" fontWeight="600" fontFamily="monospace">Download Svc</text>

          <rect x={225} y={101} width={85} height={32} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={267} y={121} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Metadata Svc</text>

          <rect x={225} y={144} width={85} height={32} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={267} y={164} textAnchor="middle" fill="#ea580c" fontSize="8" fontWeight="600" fontFamily="monospace">Sync Svc</text>

          {/* Supporting Services */}
          <rect x={350} y={15} width={85} height={32} rx={6} fill="#0f766e10" stroke="#0f766e" strokeWidth={1.5}/>
          <text x={392} y={35} textAnchor="middle" fill="#0f766e" fontSize="8" fontWeight="600" fontFamily="monospace">Dedup Svc</text>

          <rect x={350} y={58} width={85} height={32} rx={6} fill="#be123c10" stroke="#be123c" strokeWidth={1.5}/>
          <text x={392} y={78} textAnchor="middle" fill="#be123c" fontSize="8" fontWeight="600" fontFamily="monospace">Sharing Svc</text>

          <rect x={350} y={101} width={85} height={32} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={392} y={121} textAnchor="middle" fill="#9333ea" fontSize="8" fontWeight="600" fontFamily="monospace">Notification</text>

          <rect x={350} y={144} width={85} height={32} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={392} y={164} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Search Svc</text>

          {/* Data Layer */}
          <rect x={30} y={210} width={85} height={36} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={72} y={227} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Metadata DB</text>
          <text x={72} y={239} textAnchor="middle" fill="#78716c80" fontSize="7" fontFamily="monospace">PostgreSQL</text>

          <rect x={140} y={210} width={85} height={36} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={182} y={227} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Chunk Store</text>
          <text x={182} y={239} textAnchor="middle" fill="#78716c80" fontSize="7" fontFamily="monospace">S3 / GCS</text>

          <rect x={250} y={210} width={85} height={36} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={292} y={227} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Chunk Index</text>
          <text x={292} y={239} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">Redis</text>

          <rect x={360} y={210} width={85} height={36} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={402} y={227} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Event Queue</text>
          <text x={402} y={239} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">Kafka</text>

          <rect x={470} y={210} width={85} height={36} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
          <text x={512} y={227} textAnchor="middle" fill="#7c3aed" fontSize="9" fontWeight="600" fontFamily="monospace">Search Index</text>
          <text x={512} y={239} textAnchor="middle" fill="#7c3aed80" fontSize="7" fontFamily="monospace">Elasticsearch</text>

          <rect x={580} y={210} width={85} height={36} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={622} y={227} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">CDN Edge</text>
          <text x={622} y={239} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">CloudFront</text>

          {/* Arrows */}
          <defs><marker id="ah-svc" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          <line x1={90} y1={46} x2={120} y2={62} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={90} y1={91} x2={120} y2={78} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={190} y1={62} x2={225} y2={31} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={190} y1={68} x2={225} y2={74} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={190} y1={74} x2={225} y2={117} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={190} y1={80} x2={225} y2={160} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={310} y1={31} x2={350} y2={31} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={310} y1={117} x2={350} y2={74} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>
          <line x1={310} y1={160} x2={350} y2={117} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-svc)"/>

          {/* Legend */}
          <rect x={15} y={268} width={695} height={95} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={25} y={286} fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Upload: Client → API GW → Upload Svc → Dedup check (Redis) → Store chunk (S3) → Update manifest (Metadata DB) → Kafka event</text>
          <text x={25} y={303} fill="#0891b2" fontSize="8" fontWeight="600" fontFamily="monospace">Download: Client → API GW → Download Svc → signed CDN URL → CDN edge (cache hit) or S3 origin (cache miss)</text>
          <text x={25} y={320} fill="#ea580c" fontSize="8" fontWeight="600" fontFamily="monospace">Sync: Kafka event → Sync Svc → Notification Svc → push to all devices → client pulls changed chunks</text>
          <text x={25} y={337} fill="#9333ea" fontSize="8" fontWeight="600" fontFamily="monospace">Key insight: Content (blob) and Metadata (SQL) are FULLY separated. Metadata is hot+small. Content is cold+large.</text>
          <text x={25} y={354} fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Dedup happens at chunk level globally. Chunk hash is the universal key. Same content = same hash = stored once.</text>
        </svg>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#2563eb">
          <Label color="#2563eb">Supporting Services</Label>
          <div className="space-y-3">
            {[
              { name: "Upload Service", role: "Manage resumable upload sessions. Receive chunks, validate checksums, trigger dedup check, write to blob store.", tech: "Go + S3 multipart API + Redis (session state)", critical: true },
              { name: "Dedup Service", role: "Check chunk hash against global index. If exists: skip upload, increment ref_count. Saves 30-50% storage.", tech: "Go + Redis (chunk hash index) + Bloom filter", critical: true },
              { name: "Notification Service", role: "Push file change events to connected clients. Long-poll or WebSocket per device. Fan-out per user.", tech: "Go + Kafka consumer + WebSocket", critical: true },
              { name: "Sharing / ACL Service", role: "Manage permissions (viewer/editor/commenter). Link sharing. Check access on every file operation. Audit log.", tech: "Go + PostgreSQL + Redis (ACL cache)", critical: true },
              { name: "Search Service", role: "Full-text search on file names and metadata. Optional content indexing (OCR, text extraction).", tech: "Elasticsearch + Kafka (index pipeline)", critical: false },
              { name: "Quota Service", role: "Track per-user storage usage. Enforce limits. Handle upgrades, shared drive quotas.", tech: "Go + Redis counter + PostgreSQL", critical: false },
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
                  { route: "Upload Svc → Dedup Svc", proto: "gRPC", timeout: "10ms", fail: "Assume unique, store anyway (over-store, not data loss)" },
                  { route: "Upload Svc → S3/GCS", proto: "HTTPS", timeout: "30s", fail: "Retry 3x. If persistent: fail upload, client retries chunk" },
                  { route: "Upload Svc → Metadata DB", proto: "SQL", timeout: "200ms", fail: "Retry. Critical: no metadata = file doesn't exist" },
                  { route: "Download Svc → CDN", proto: "HTTP redirect", timeout: "N/A", fail: "Fallback to origin (S3) direct download" },
                  { route: "Sync Svc → Kafka", proto: "Kafka", timeout: "Async", fail: "Retry 3x, DLQ. Devices catch up on reconnect" },
                  { route: "Sharing Svc → Metadata DB", proto: "SQL", timeout: "100ms", fail: "Return error. Don't allow access on failure (safe default)" },
                  { route: "Notification → Client", proto: "WS/Long-poll", timeout: "30s poll", fail: "Client reconnects. Sync from last version." },
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
  const [flow, setFlow] = useState("upload");
  const flows = {
    upload: { title: "File Upload (Resumable, Chunked)", steps: [
      { actor: "Client (Desktop)", action: "User saves 50MB file in sync folder. Sync engine detects change via filesystem watcher.", type: "process" },
      { actor: "Client", action: "Chunk file using CDC algorithm → 13 chunks (avg 4MB each). Compute SHA-256 for each chunk.", type: "process" },
      { actor: "Client → API Gateway", action: "POST /v1/files/upload/init {name, size, parent_folder_id, chunk_hashes: [h1..h13]}", type: "request" },
      { actor: "Upload Svc", action: "Create upload session in Redis: upload_id → {file_info, expected_chunks: 13, received: []}. Check chunk_hashes against Dedup index.", type: "process" },
      { actor: "Upload Svc → Client", action: "Return: {upload_id, needed_chunks: [0,2,5,7,8,10,11,12], deduped: [1,3,4,6,9]} — 5 chunks already exist!", type: "success" },
      { actor: "Client → Upload Svc", action: "Upload 8 needed chunks in parallel (3-5 concurrent). Each PUT with Content-Range + Content-MD5.", type: "request" },
      { actor: "Upload Svc", action: "For each chunk: verify MD5 → store in S3 → update session state. If chunk fails: client retries just that chunk.", type: "process" },
      { actor: "Client → Upload Svc", action: "POST /v1/files/upload/:upload_id/complete — all chunks uploaded.", type: "request" },
      { actor: "Upload Svc → Metadata DB", action: "Transaction: INSERT file row + file_version row + update parent folder updated_at. Chunk manifest = [h1..h13].", type: "request" },
      { actor: "Upload Svc → Kafka", action: "Publish event: {type: file.created, file_id, owner_id, parent_folder_id}.", type: "process" },
      { actor: "Sync Svc (Kafka consumer)", action: "Consume event → look up all user's connected devices → push notification to each.", type: "process" },
      { actor: "Other Devices", action: "Receive notification → pull file manifest → download only missing chunks → reconstruct file locally.", type: "success" },
    ]},
    download: { title: "File Download (CDN-Accelerated)", steps: [
      { actor: "Client (Web App)", action: "User clicks 'Download' on a 200MB video file.", type: "request" },
      { actor: "Client → API Gateway", action: "GET /v1/files/:file_id", type: "request" },
      { actor: "API Gateway → Sharing Svc", action: "Check permissions: does user have viewer or editor role? Is link sharing enabled?", type: "process" },
      { actor: "Download Svc", action: "Lookup file metadata → get chunk_manifest → generate signed CDN URL (expires in 1 hour).", type: "process" },
      { actor: "Download Svc → Client", action: "HTTP 302 redirect to signed CDN URL: https://cdn.example.com/chunks/merged/:file_id?sig=...", type: "success" },
      { actor: "CDN Edge", action: "Cache hit? Serve directly from edge (50ms). Cache miss? Fetch from S3 origin, cache it, serve.", type: "process" },
      { actor: "Client", action: "Browser downloads file via CDN. Parallel range requests for large files. Verify checksum on complete.", type: "success" },
    ]},
    sync_edit: { title: "File Edit — Delta Sync", steps: [
      { actor: "Client (Desktop)", action: "User edits an existing 1GB spreadsheet. Saves file. Sync engine detects mtime change.", type: "process" },
      { actor: "Client", action: "Re-chunk file using same CDC algorithm → 256 chunks. Compare chunk hashes with previous version's manifest.", type: "process" },
      { actor: "Client", action: "Diff: 251 chunks unchanged, 5 chunks are new. Only 5 chunks (~20MB) need uploading for a 1GB file!", type: "process" },
      { actor: "Client → Upload Svc", action: "Upload 5 new chunks + new manifest [h1(same)..h251(same), h252(new)..h256(new)].", type: "request" },
      { actor: "Upload Svc → Metadata DB", action: "Create new file_version (version_num +1) with updated manifest. Old version preserved for history.", type: "request" },
      { actor: "Upload Svc → Kafka", action: "Publish: {type: file.updated, file_id, new_version, changed_chunks: [252..256]}.", type: "process" },
      { actor: "Other Devices", action: "Receive sync event → compare manifests → download only 5 new chunks → patch local file. 20MB transfer for a 1GB edit.", type: "success" },
    ]},
    conflict: { title: "Conflict — Two Devices Edit Offline", steps: [
      { actor: "Laptop (offline)", action: "User edits 'report.docx' on airplane. Local version → v3.", type: "process" },
      { actor: "Phone (offline)", action: "Same user edits 'report.docx' on phone (started from v2). Local version → v3.", type: "process" },
      { actor: "Laptop comes online", action: "Sync engine pushes laptop's v3 to server. Server accepts: file is now v3 (laptop version).", type: "request" },
      { actor: "Phone comes online", action: "Sync engine tries to push phone's v3. Server rejects: base version v2 but current is v3. CONFLICT.", type: "check" },
      { actor: "Sync Svc", action: "Detect conflict: phone's base_version (2) < server's current (3). Two concurrent edits from same base.", type: "check" },
      { actor: "Sync Svc", action: "Save phone's version as 'report (conflict copy - Phone).docx' alongside laptop's version.", type: "process" },
      { actor: "All Devices", action: "Sync both files to all devices. User sees both versions and manually merges. No data lost.", type: "success" },
    ]},
  };
  const f = flows[flow];
  return (
    <div className="space-y-5">
      <div className="flex gap-2 flex-wrap">
        {Object.entries(flows).map(([k,v]) => (
          <button key={k} onClick={() => setFlow(k)}
            className={`px-3.5 py-2 rounded-lg text-[12px] font-medium border transition-all ${k===flow?"bg-purple-600 text-white border-purple-600":"bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
            {v.title}
          </button>
        ))}
      </div>
      <Card accent="#7e22ce">
        <Label color="#7e22ce">{f.title}</Label>
        <div className="space-y-0">
          {f.steps.map((s,i) => (
            <div key={i} className={`flex items-start gap-3 py-3 ${i<f.steps.length-1?"border-b border-stone-100":""}`}>
              <span className={`text-[10px] font-bold w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${
                s.type==="success"?"bg-emerald-600 text-white":s.type==="request"?"bg-blue-600 text-white":s.type==="check"?"bg-amber-500 text-white":"bg-stone-200 text-stone-500"
              }`}>{i+1}</span>
              <div className="flex-1 min-w-0">
                <span className="text-[10px] font-bold text-stone-400 uppercase tracking-wider">{s.actor}</span>
                <p className="text-[12px] text-stone-600 mt-0.5">{s.action}</p>
              </div>
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
          <Label color="#b45309">Security Model</Label>
          <ul className="space-y-2.5">
            <Point icon="🔐" color="#b45309"><strong className="text-stone-700">Encryption at rest</strong> — all chunks encrypted with AES-256 in blob store. Per-user encryption keys managed by KMS. Even cloud provider employees can't read user data.</Point>
            <Point icon="🔒" color="#b45309"><strong className="text-stone-700">Encryption in transit</strong> — TLS 1.3 for all API calls and chunk transfers. Certificate pinning on mobile/desktop clients. Mutual TLS between internal services.</Point>
            <Point icon="🪪" color="#b45309"><strong className="text-stone-700">OAuth 2.0 + JWT</strong> — authentication via OAuth 2.0. Short-lived JWTs (15min) + refresh tokens. Scoped tokens for third-party app access.</Point>
            <Point icon="🛡️" color="#b45309"><strong className="text-stone-700">ACL on every operation</strong> — every file read/write checks permissions in Sharing Service. Deny by default. Cache ACLs in Redis (5min TTL) for performance.</Point>
            <Point icon="📋" color="#b45309"><strong className="text-stone-700">Audit log</strong> — every access, share, download, delete is logged. Immutable audit trail for compliance (GDPR, SOX, HIPAA). Retained for 7 years.</Point>
            <Point icon="🗑️" color="#b45309"><strong className="text-stone-700">Soft delete + 30-day trash</strong> — deleted files go to trash for 30 days. Hard delete only after trash + 30-day grace. Protects against accidental deletion and account compromise.</Point>
          </ul>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Deployment Strategy</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Rolling deployments</strong> — deploy service-by-service. Canary to 1% traffic, bake for 30 min, then 10%, 50%, 100%. Automated rollback on error rate spike.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Blue-green for metadata DB migrations</strong> — schema changes are the riskiest. Use online DDL (gh-ost / pt-online-schema-change). Test on prod-replica first.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Feature flags</strong> — new sync algorithm, dedup improvements, sharing changes all behind flags. Gradual rollout. Instant kill switch.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Multi-region active-active</strong> — metadata writes go to user's home region. Replicate async to other regions. Downloads served from nearest CDN edge / region.</Point>
            <Point icon="5." color="#059669"><strong className="text-stone-700">Chaos engineering</strong> — regular blob store failure drills. Kill random Upload/Sync instances. Verify resumable upload works. Verify sync catches up after outage.</Point>
          </ul>
        </Card>
      </div>
    </div>
  );
}

function OpsSection() {
  return (
    <div className="space-y-5">
      <Card accent="#be123c">
        <Label color="#be123c">Operational Bottlenecks</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[11px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500 w-44">Bottleneck</th>
              <th className="text-left px-3 py-2 font-semibold text-red-400 w-44">Signal</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Fix</th>
              <th className="text-left px-3 py-2 font-semibold text-amber-500 w-56">Pro Tip</th>
            </tr></thead>
            <tbody>
              {[
                { b: "Metadata DB write throughput", s: "Write latency > 200ms", f: "Add shards. Batch metadata writes. Write-behind cache for non-critical fields (access_count).", p: "The files table is write-heavy (every upload, rename, move). Shard by owner_id. Keep shard count manageable (hundreds, not thousands)." },
                { b: "Blob store PUT latency", s: "Chunk upload > 5s for 4MB", f: "Upload to nearest regional store. Async replicate to other regions. Use multipart upload API.", p: "S3 PUT can spike during regional overload. Have cross-region fallback. Don't block on replication — async is fine for durability." },
                { b: "Dedup index (Redis) memory", s: "Redis memory > 80%", f: "Shard Redis cluster. Use probabilistic structures (Bloom filter) for first-pass dedup check.", p: "1 trillion chunks × 32 bytes per hash = 32TB. Use Bloom filter to eliminate 99% of lookups, then check DB for positives." },
                { b: "Kafka consumer lag (sync events)", s: "Sync notification delay > 30s", f: "More partitions + consumers. Prioritize: file.created/updated before file.accessed events.", p: "During business hours, sync events spike 10×. Auto-scale consumers. Partition by user_id for locality." },
                { b: "Large folder listing", s: "Folder with 50K+ files loads > 5s", f: "Mandatory cursor pagination. Cache folder listing in Redis. Incremental updates (only fetch changes since last listing).", p: "Some users dump 100K photos in one folder. Paginate on server. Client uses virtual scrolling. Consider suggesting subfolders." },
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
            { title: "Dedup Hash Collision Scare", symptom: "Two different files mapped to the same SHA-256 hash. One user's file content replaced by another's.",
              cause: "Not actually a SHA-256 collision (astronomically unlikely). Root cause: chunk boundary calculation bug caused two different chunks to produce the same slice, which hashed identically.",
              fix: "Added full-file checksum verification on download (not just per-chunk). Added secondary hash (SHA-256 + xxHash) for belt-and-suspenders. Chunk boundary algorithm fix deployed.",
              quote: "We spent 72 hours thinking we broke SHA-256. Turned out it was an off-by-one error in the chunking code. The hash was fine — the data going IN was the same." },
            { title: "Sync Storm After 8-Hour Outage", symptom: "Sync service restored after 8-hour outage. 100M pending sync events. All clients reconnect simultaneously. Thundering herd.",
              cause: "Clients all had stale state from 8 hours ago. Every client requests full sync. Metadata DB overwhelmed with queries. Kafka consumer lag grows further.",
              fix: "Client-side jitter: random 0-300s delay on reconnect. Server-side: priority queue (recent changes first). Rate-limit sync requests per user. Pre-compute delta snapshots during outage.",
              quote: "We restored the sync service and immediately had a second outage — the thundering herd was worse than the original failure. Now we stagger reconnections." },
            { title: "Quota Bypass via Shared Folders", symptom: "Users discovered they could exceed storage quota by accepting shared folders. Quota only counted owned files, not shared content synced locally.",
              cause: "Quota calculation only counted files where owner_id = user. Shared files synced to device consumed local storage but didn't count against cloud quota. Users shared TB-scale folders to bypass limits.",
              fix: "Separate quotas: cloud quota (owned files) + sync quota (total local sync). Shared folders don't sync by default — user opts in. UI shows 'X of Y synced' with warning.",
              quote: "Our free tier users figured out they could get unlimited storage by creating shared circles. Clever. Expensive for us." },
            { title: "Metadata DB Migration Timeout", symptom: "Adding an index to the files table (2 billion rows) caused query timeout for all metadata operations for 4 hours.",
              cause: "Online DDL (gh-ost) was used but it needed to copy the entire table. The copy process consumed all disk I/O. Reads slowed to a crawl due to I/O contention.",
              fix: "Run migrations only during off-peak hours (Sunday 2-6 AM). Throttle gh-ost chunk size (500 rows/batch). Monitor I/O saturation during migration. Have kill switch to abort migration instantly.",
              quote: "We knew the table had 2 billion rows. We thought gh-ost could handle it. It could — it just needed the entire I/O budget to do so. No reads left." },
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
        { t: "Real-Time Collaboration", d: "Google Docs-style simultaneous editing. Multiple cursors, live changes. Operational Transform or CRDT for conflict-free merging.", detail: "Separate system from file storage. OT server per document. WebSocket for real-time. Autosave creates file versions periodically.", effort: "Hard" },
        { t: "Comments & Annotations", d: "Comment on any file. Thread discussions. @mention users. Resolve comments. Activity feed per file.", detail: "Separate comments table: file_id + position (for PDFs/images). Notification on @mention. Integrates with sharing permissions.", effort: "Medium" },
        { t: "Smart Search (AI)", d: "Search inside documents, images (OCR), and PDFs. Natural language queries like 'find my tax documents from 2024'.", detail: "Extract text via Apache Tika / Textract. Index in Elasticsearch. Image labels via Vision API. Embeddings for semantic search.", effort: "Hard" },
        { t: "Offline Mode", d: "Pin files/folders for offline access. Edit offline, sync when back online. Handle conflicts on reconnect.", detail: "Client maintains local SQLite DB + cached chunks. Sync queue for pending uploads. Conflict detection via version vectors.", effort: "Medium" },
        { t: "Team / Shared Drives", d: "Organization-owned drives with admin controls. Files owned by team, not individual. Member management + role hierarchy.", detail: "Shared drive = special owner (org, not user). Team-level quotas. Admin can manage all files. Audit log per team.", effort: "Medium" },
        { t: "Activity Feed & Notifications", d: "See who viewed, edited, commented on your files. Email digest of changes. Real-time activity stream.", detail: "Event sourcing: every action is an event. Build activity feed from events. Aggregate for digest emails. Respect notification preferences.", effort: "Easy" },
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
    { q:"Why chunk files instead of uploading whole files?", a:"Three reasons: (1) Resumability — if upload fails at 90%, you retry one 4MB chunk, not the entire 5GB file. (2) Deduplication — identical chunks across users stored once. Saves 30-50% storage. (3) Delta sync — edit a 1GB file, only changed chunks (~4MB each) re-upload. Without chunking, every save re-uploads the entire file.", tags:["design"] },
    { q:"How does deduplication work with encryption?", a:"Two approaches: (1) Server-side dedup (no per-user encryption): hash content, dedup globally. Google Drive does this. Efficient but server can see content. (2) Convergent encryption: derive encryption key from content hash. Same content → same key → same ciphertext → dedup works on ciphertext. Drawback: vulnerable to confirmation attacks (attacker can check if specific file exists).", tags:["security"] },
    { q:"How do you handle concurrent uploads to the same file?", a:"Optimistic locking with version numbers. Upload includes base_version. Server checks: if current_version == base_version, accept and increment. If current_version > base_version, reject with conflict. Client handles conflict (save as conflict copy or retry). This is last-writer-wins at the version level, conflict-detect at the upload level.", tags:["consistency"] },
    { q:"How would you design the desktop sync client?", a:"Three components: (1) File system watcher (inotify/FSEvents/ReadDirectoryChanges) detects local changes. (2) Sync engine: maintains local DB of file states (hash, mtime, version). Compares local vs server state. Resolves which direction to sync. (3) Transfer manager: uploads/downloads chunks with retry, bandwidth throttling, and prioritization (user-initiated > background sync).", tags:["design"] },
    { q:"Why separate metadata DB from blob store?", a:"Metadata is small (KB per file), structured, queried heavily (list folder = SQL query), and needs strong consistency (rename must be atomic). Content is large (MB-GB), unstructured (arbitrary binary), rarely queried (only on download), and can tolerate eventual consistency for replication. Using the same store for both wastes money and performance. SQL DB for metadata, S3 for content.", tags:["architecture"] },
    { q:"How do you handle a user with 1 million files?", a:"Pagination everywhere: API returns max 1000 items per page with cursor. Metadata DB indexed by (parent_folder_id, name) for folder listing. Consider denormalized count fields (folder.child_count) to avoid COUNT queries. Client-side: virtual scrolling, load on demand. Background: encourage folder organization, warn on large flat folders.", tags:["scalability"] },
    { q:"What's the difference between Google Drive and Dropbox architectures?", a:"Dropbox pioneered content-defined chunking (CDC) with Rabin fingerprinting on the client. Their Magic Pocket storage replaces S3. Block-level sync is done client-side. Google Drive does more server-side processing, integrates with Google Docs (OT-based collab), and uses Google's internal Colossus/Bigtable infrastructure. Dropbox optimizes for sync speed; Google optimizes for collaboration.", tags:["comparison"] },
    { q:"How do you implement file search?", a:"Two levels: (1) Metadata search — file name, owner, type, date. Standard SQL queries with full-text index on name. Fast. (2) Content search — extract text from docs/PDFs (Apache Tika), OCR for images, index in Elasticsearch. Async pipeline: file uploaded → Kafka → extraction worker → ES index. Content search is optional and resource-intensive.", tags:["design"] },
    { q:"How do you handle storage quotas fairly?", a:"Quota = sum of (size_bytes) for all files where owner_id = user. Tracked in a counter (Redis + periodic reconciliation with DB). Enforce on upload: check quota BEFORE accepting chunks (optimistic), hard check in transaction when finalizing upload. Shared files: count against the OWNER's quota, not the viewer. Versioning: count all versions or only latest (product decision).", tags:["design"] },
    { q:"What about data residency and GDPR?", a:"User-homed regions: data stored in the user's assigned region (EU users → EU region). Cross-region sharing: metadata replicated globally, content stays in owner's region (viewer downloads cross-region). GDPR right-to-delete: hard delete all user data within 30 days. Audit log retained per legal requirements. Data processing agreements with cloud provider.", tags:["compliance"] },
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

export default function GoogleDriveSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Google Drive (Cloud File Storage)</h1>
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