import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   WEB CRAWLER — System Design Reference
   Pearl white theme · Reusable section structure
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",             icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",         icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",  icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",           icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",    icon: "🏗️", color: "#9333ea" },
  { id: "algorithm",     label: "Algorithm Deep Dive",  icon: "⚙️", color: "#c026d3" },
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
            <Label>What is a Web Crawler?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A web crawler (spider/bot) is a system that systematically browses the internet, downloading web pages and extracting links to discover new content. It starts from a set of seed URLs, fetches pages, parses them for outgoing links, and adds those links to a queue for future fetching — repeating this loop across billions of pages.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Think of it as an explorer mapping an unknown territory: it starts at a few known locations, follows every road it finds, and builds a map of the entire landscape. The core challenges are scale (billions of pages), politeness (don't overload servers), freshness (re-crawl stale content), and deduplication (avoid wasting resources on duplicate pages).
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard?</Label>
            <ul className="space-y-2.5">
              <Point icon="🌐" color="#0891b2">Massive scale — the web has 100B+ pages; even a focused crawl touches billions of URLs</Point>
              <Point icon="🤖" color="#0891b2">Politeness — must respect robots.txt, rate-limit per domain, and avoid overloading small servers</Point>
              <Point icon="🔄" color="#0891b2">Freshness — pages change constantly; prioritizing re-crawls for high-value or fast-changing content is critical</Point>
              <Point icon="🧩" color="#0891b2">Deduplication — same content behind multiple URLs (www vs non-www, query params, mirrors); must detect and skip</Point>
              <Point icon="🕳️" color="#0891b2">Crawler traps — infinite calendars, session IDs in URLs, auto-generated pages that create unbounded loops</Point>
              <Point icon="⚖️" color="#0891b2">Distributed coordination — thousands of crawl workers must coordinate URL assignment without duplicating effort</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Examples</Label>
            <div className="space-y-2.5">
              {[
                { co: "Googlebot", rule: "Crawls entire web", algo: "PageRank priority" },
                { co: "Bingbot", rule: "Full web crawl", algo: "Priority + freshness" },
                { co: "Common Crawl", rule: "Open dataset, ~3.5B pages/month", algo: "Breadth-first" },
                { co: "Internet Archive", rule: "Archival crawl", algo: "Heritrix framework" },
                { co: "Scrapy/Nutch", rule: "Open-source frameworks", algo: "Configurable" },
              ].map((e,i) => (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-28 shrink-0">{e.co}</span>
                  <span className="text-stone-500 flex-1">{e.rule}</span>
                  <span className="text-stone-400 text-[10px]">{e.algo}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <Label color="#2563eb">Core Crawl Loop</Label>
            <svg viewBox="0 0 360 200" className="w-full">
              <DiagramBox x={180} y={30} w={100} h={34} label="URL Frontier" color="#9333ea"/>
              <DiagramBox x={180} y={90} w={80} h={34} label="Fetcher" color="#2563eb"/>
              <DiagramBox x={180} y={150} w={80} h={34} label="Parser" color="#059669"/>
              <DiagramBox x={310} y={150} w={70} h={34} label="Content\nStore" color="#dc2626"/>
              <DiagramBox x={50} y={90} w={70} h={34} label="Dedup\nFilter" color="#d97706"/>
              <Arrow x1={180} y1={47} x2={180} y2={73} label="next URL" id="cl1"/>
              <Arrow x1={180} y1={107} x2={180} y2={133} label="HTML" id="cl2"/>
              <Arrow x1={220} y1={150} x2={275} y2={150} label="store" id="cl3"/>
              <Arrow x1={140} y1={150} x2={85} y2={107} label="new URLs" id="cl4"/>
              <Arrow x1={50} y1={73} x2={130} y2={35} label="unique" id="cl5"/>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Google, Amazon, Microsoft, Apple, LinkedIn, Pinterest</div>
              </div>
              <span className="text-indigo-500 font-bold text-sm">★★★★☆</span>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Scope the Crawl</div>
            <p className="text-[12px] text-stone-500 mt-0.5">"Web crawler" can mean many things. Clarify: Are we crawling the entire web or a specific domain? Is this for a search engine, an archival system, or a content aggregator? For a 45-min interview, focus on <strong>a general-purpose web crawler that can scale to billions of pages</strong>. Don't try to design the entire search engine.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Given a set of seed URLs, crawl the web by following links to discover new pages</Point>
            <Point icon="2." color="#059669">Download and store the HTML content of each page</Point>
            <Point icon="3." color="#059669">Extract outgoing hyperlinks from each page and add them to the crawl queue</Point>
            <Point icon="4." color="#059669">Respect robots.txt rules and per-domain rate limits (politeness)</Point>
            <Point icon="5." color="#059669">Detect and skip duplicate URLs and duplicate content</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Scalability — must handle billions of URLs and crawl ~1,000 pages/sec per worker</Point>
            <Point icon="2." color="#dc2626">Politeness — never hit a single domain more than 1 req/sec (configurable per domain)</Point>
            <Point icon="3." color="#dc2626">Robustness — handle malformed HTML, timeouts, DNS failures, crawler traps gracefully</Point>
            <Point icon="4." color="#dc2626">Freshness — re-crawl high-priority pages periodically; prioritize pages that change frequently</Point>
            <Point icon="5." color="#dc2626">Extensibility — pluggable modules for parsing, filtering, storage, and URL prioritization</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask the Interviewer</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Is this a general-purpose web crawler or domain-specific?",
            "What's the target scale? Millions or billions of pages?",
            "Do we need to render JavaScript (SPA pages)?",
            "Should we store just HTML or also images/videos/PDFs?",
            "How fresh does the content need to be? Real-time? Daily?",
            "Is this for indexing (search engine) or archival?",
            "Do we need to handle geo-distributed crawling?",
            "What's the budget for bandwidth and storage?",
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
            <div className="text-[12px] font-bold text-violet-700">Interview Tip — Show Your Math</div>
            <p className="text-[12px] text-stone-500 mt-0.5">Walk through estimation out loud. State assumptions clearly: <em>"Let me assume we want to crawl 1 billion pages per month..."</em> — interviewers care about the approach, not exact numbers.</p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Crawl Rate Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Target: 1 billion pages per month" result="1B" note='Assumption — a medium-scale search engine crawl' />
            <MathStep step="2" formula="Pages per day = 1B / 30" result="~33M/day" note="Roughly 33 million pages crawled daily" />
            <MathStep step="3" formula="Pages per second = 33M / 86,400" result="~400 pages/sec" note="Average sustained crawl rate" final />
            <MathStep step="4" formula="Peak (3× average)" result="~1,200 pages/sec" note="Burst during off-peak hours when many domains allow faster crawling" final />
            <MathStep step="5" formula="With 1,000 pages/sec per worker" result="~2 workers" note="At avg; need ~4-5 workers for peak with headroom" />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Storage Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Avg HTML page size (compressed)" result="~100 KB" note="Raw HTML is ~500KB avg; gzip compresses ~5×" />
            <MathStep step="2" formula="Storage per month = 1B × 100 KB" result="100 TB/mo" note="Just for compressed HTML content" final />
            <MathStep step="3" formula="Metadata per URL = ~500 bytes" result="500 B" note="URL, last_crawled, hash, priority, depth, etc." />
            <MathStep step="4" formula="URL metadata = 1B × 500 B" result="500 GB" note="Easily fits in a sharded database" />
            <MathStep step="5" formula="5-year retention = 100TB × 60 months" result="~6 PB" note="Object storage (S3) — ~$21K/mo at $3.50/TB" />
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Bandwidth Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Download bandwidth = 400 pages/sec × 500 KB" result="~200 MB/s" note="Avg uncompressed; ~40 MB/s with compression" />
            <MathStep step="2" formula="In Gbps = 200 MB/s × 8" result="~1.6 Gbps" note="Uncompressed; ~320 Mbps compressed. Easily within cloud egress." final />
            <MathStep step="3" formula="DNS lookups = 400/sec" result="~400 QPS" note="With caching, ~10% are cache misses = ~40 DNS queries/sec" />
            <MathStep step="4" formula="Outbound connections (concurrent)" result="~500-1000" note="Each fetch takes ~1-2s avg; 400/sec × 2s = 800 concurrent connections" />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Infrastructure Cost</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Crawl workers (c5.xlarge, 4vCPU)" result="~5 nodes" note="~$250/mo each on-demand" />
            <MathStep step="2" formula="URL frontier (Redis Cluster)" result="~$500/mo" note="3-node cluster for URL queue + dedup bloom filter" />
            <MathStep step="3" formula="Content storage (S3)" result="~$2,300/mo" note="100 TB × $23/TB (S3 Standard)" final />
            <MathStep step="4" formula="Metadata DB (PostgreSQL RDS)" result="~$800/mo" note="db.r6g.xlarge for URL metadata" />
            <MathStep step="5" formula="Total monthly" result="~$5K/mo" note="Bandwidth largely free within AWS. Very cost-effective." final />
          </div>
          <div className="mt-4 pt-3 border-t border-stone-100">
            <div className="text-[11px] text-stone-500">
              <strong className="text-stone-700">Key point:</strong> Web crawling is I/O-bound, not compute-bound. Most cost goes to storage and network, not CPU. The biggest variable cost is how much content you store and for how long.
            </div>
          </div>
        </Card>
      </div>

      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Crawl Rate", val: "~400/sec", sub: "Peak: ~1,200/sec" },
            { label: "Storage/Month", val: "~100 TB", sub: "Compressed HTML" },
            { label: "Bandwidth", val: "~1.6 Gbps", sub: "~320 Mbps compressed" },
            { label: "Monthly Cost", val: "~$5K", sub: "1B pages/month" },
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
          <Label color="#2563eb">Crawler Core Interface</Label>
          <CodeBlock code={`# Crawl Manager — Controls the crawl lifecycle
class CrawlManager:
    def start_crawl(self, config: CrawlConfig) -> CrawlJob:
        """Start a new crawl job with seed URLs."""
        # config: seed_urls, max_depth, max_pages,
        #         domains_allowlist, politeness_delay
        pass

    def pause_crawl(self, job_id: str) -> bool:
        """Pause a running crawl job."""
        pass

    def resume_crawl(self, job_id: str) -> bool:
        """Resume a paused crawl job."""
        pass

    def get_status(self, job_id: str) -> CrawlStatus:
        """Return progress: pages_crawled, pages_queued,
           errors, throughput, estimated_completion."""
        pass`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">URL Frontier Interface</Label>
          <CodeBlock code={`# URL Frontier — Priority queue of URLs to crawl
class URLFrontier:
    def add_url(self, url: str, priority: int,
                depth: int, parent_url: str) -> bool:
        """Add URL if not seen before. Returns False
           if duplicate."""
        pass

    def get_next_urls(self, domain: str,
                      batch_size: int = 50) -> list[URL]:
        """Get batch of URLs for a domain, respecting
           politeness delay."""
        pass

    def mark_done(self, url: str, status: int,
                  discovered_urls: list[str]) -> None:
        """Mark URL as crawled. Enqueue new URLs."""
        pass

    def size(self) -> int:
        """Number of URLs waiting in the frontier."""
        pass`} />
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#d97706">
          <Label color="#d97706">Fetcher Interface</Label>
          <CodeBlock code={`# Fetcher — Downloads pages with politeness
class Fetcher:
    def fetch(self, url: str) -> FetchResult:
        """Download page. Respects robots.txt, follows
           redirects (max 5), handles timeouts."""
        # Returns: status_code, content, headers,
        #          final_url (after redirects),
        #          content_type, fetch_time_ms
        pass

    def check_robots_txt(self, domain: str) -> RobotsRules:
        """Fetch and parse robots.txt for domain.
           Cache for 24h."""
        pass

# robots.txt compliance is non-negotiable
# Crawl-delay, Disallow, Allow directives`} />
        </Card>
        <Card accent="#c026d3">
          <Label color="#c026d3">Content Processor Interface</Label>
          <CodeBlock code={`# Parser — Extracts links and metadata
class ContentProcessor:
    def parse(self, html: str, base_url: str) -> ParseResult:
        """Extract outgoing links, title, meta tags,
           canonical URL, language."""
        pass

    def compute_fingerprint(self, content: str) -> str:
        """SimHash or MD5 for deduplication."""
        pass

    def normalize_url(self, url: str) -> str:
        """Canonicalize: lowercase host, remove
           fragments, sort query params, resolve
           relative paths."""
        pass`} />
          <div className="mt-4 pt-4 border-t border-stone-100">
            <Label color="#d97706">Design Decisions</Label>
            <ul className="space-y-1.5">
              <Point icon="→" color="#d97706">Internal service APIs — crawler is not user-facing; no REST endpoints</Point>
              <Point icon="→" color="#d97706">Batch processing — fetch URLs in batches per domain for connection reuse</Point>
              <Point icon="→" color="#d97706">Async I/O — use async HTTP clients (aiohttp, Netty) for high concurrency</Point>
              <Point icon="→" color="#d97706">Pluggable processors — add custom extractors for images, structured data, etc.</Point>
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
    { label: "Start Simple", desc: "Single machine: seed URLs in a queue, one thread fetches and parses. Works for small sites. Bottleneck: single thread = ~5 pages/sec. No politeness, no dedup." },
    { label: "Add Politeness", desc: "Separate queues per domain. A scheduler enforces per-domain delays (1 req/sec). robots.txt is fetched and cached. This prevents overloading individual servers." },
    { label: "Distribute Workers", desc: "Multiple crawl workers behind a URL Frontier service. URLs are partitioned by domain hash so each worker owns a set of domains. Dedup via centralized Bloom filter or shared store." },
    { label: "Full Architecture", desc: "Complete: DNS Resolver → Fetcher Workers → Content Store (S3) → Parser Workers → URL Frontier (priority queue + dedup) → Scheduler (politeness). Monitoring, re-crawl scheduler, and trap detection." },
  ];
  const diagrams = [
    () => (
      <svg viewBox="0 0 460 160" className="w-full">
        <DiagramBox x={60} y={65} w={80} h={38} label="Seed URLs" color="#2563eb"/>
        <DiagramBox x={200} y={65} w={90} h={42} label="Single\nCrawler" color="#9333ea"/>
        <DiagramBox x={350} y={65} w={80} h={38} label="File Store" color="#059669"/>
        <Arrow x1={100} y1={65} x2={155} y2={65} id="s0a"/>
        <Arrow x1={245} y1={65} x2={310} y2={65} label="save" id="s0b"/>
        <rect x={130} y={115} width={200} height={22} rx={6} fill="#dc262608" stroke="#dc262630"/>
        <text x={230} y={127} textAnchor="middle" fill="#dc2626" fontSize="9" fontFamily="monospace">❌ ~5 pages/sec, no politeness</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 190" className="w-full">
        <DiagramBox x={60} y={75} w={80} h={38} label="URL Queue" color="#9333ea"/>
        <DiagramBox x={200} y={45} w={80} h={30} label="Domain A\nQueue" color="#d97706"/>
        <DiagramBox x={200} y={105} w={80} h={30} label="Domain B\nQueue" color="#d97706"/>
        <DiagramBox x={350} y={75} w={80} h={42} label="Fetcher\n(1 req/s)" color="#2563eb"/>
        <Arrow x1={100} y1={65} x2={160} y2={50} id="p1"/>
        <Arrow x1={100} y1={85} x2={160} y2={100} id="p2"/>
        <Arrow x1={240} y1={50} x2={310} y2={65} id="p3"/>
        <Arrow x1={240} y1={100} x2={310} y2={85} id="p4"/>
        <rect x={260} y={150} width={160} height={20} rx={4} fill="#05966908" stroke="#05966930"/>
        <text x={340} y={162} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">✓ Polite per-domain</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 200" className="w-full">
        <DiagramBox x={80} y={85} w={85} h={42} label="URL\nFrontier" color="#9333ea"/>
        <DiagramBox x={230} y={40} w={80} h={34} label="Worker 1" color="#2563eb"/>
        <DiagramBox x={230} y={90} w={80} h={34} label="Worker 2" color="#2563eb"/>
        <DiagramBox x={230} y={140} w={80} h={34} label="Worker N" color="#2563eb"/>
        <DiagramBox x={380} y={85} w={80} h={42} label="Bloom\nFilter" color="#dc2626"/>
        <Arrow x1={122} y1={72} x2={190} y2={48} id="d1"/>
        <Arrow x1={122} y1={85} x2={190} y2={90} id="d2"/>
        <Arrow x1={122} y1={98} x2={190} y2={132} id="d3"/>
        <Arrow x1={270} y1={90} x2={340} y2={85} label="dedup" id="d4" dashed/>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 500 240" className="w-full">
        <DiagramBox x={45} y={50} w={58} h={34} label="Seeds" color="#6366f1"/>
        <DiagramBox x={140} y={50} w={80} h={42} label="URL\nFrontier" color="#9333ea"/>
        <DiagramBox x={250} y={50} w={70} h={38} label="Scheduler" color="#d97706"/>
        <DiagramBox x={360} y={50} w={78} h={42} label="Fetcher\nWorkers" color="#2563eb"/>
        <DiagramBox x={470} y={50} w={48} h={34} label="DNS" color="#64748b"/>
        <DiagramBox x={360} y={130} w={78} h={38} label="Parser\nWorkers" color="#059669"/>
        <DiagramBox x={470} y={130} w={48} h={34} label="S3" color="#dc2626"/>
        <DiagramBox x={250} y={130} w={70} h={34} label="Dedup" color="#c026d3"/>
        <DiagramBox x={140} y={200} w={80} h={34} label="Monitor" color="#0284c7"/>
        <Arrow x1={74} y1={50} x2={100} y2={50} id="f0"/>
        <Arrow x1={180} y1={50} x2={215} y2={50} id="f1"/>
        <Arrow x1={285} y1={50} x2={321} y2={50} label="dispatch" id="f2"/>
        <Arrow x1={399} y1={50} x2={446} y2={50} label="resolve" id="f3" dashed/>
        <Arrow x1={360} y1={69} x2={360} y2={111} label="HTML" id="f4"/>
        <Arrow x1={399} y1={130} x2={446} y2={130} label="store" id="f5"/>
        <Arrow x1={321} y1={130} x2={285} y2={130} label="check" id="f6" dashed/>
        <Arrow x1={250} y1={113} x2={180} y2={67} label="new URLs" id="f7"/>
        <Arrow x1={180} y1={71} x2={180} y2={183} label="metrics" id="f8" dashed/>
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
          <Label color="#059669">Crawl Flow — Happy Path</Label>
          <div className="space-y-1.5">
            {[
              { s:"1", t:"URL Frontier pops highest-priority URL from queue", c:"text-purple-600" },
              { s:"2", t:"Scheduler checks politeness: domain last hit >1s ago? Proceed.", c:"text-amber-600" },
              { s:"3", t:"DNS resolver resolves domain → IP (cached if recent)", c:"text-stone-500" },
              { s:"4", t:"Fetcher sends HTTP GET with crawler User-Agent, follows redirects", c:"text-blue-600" },
              { s:"5", t:"Response: 200 OK, Content-Type: text/html, ~150KB body", c:"text-emerald-600" },
              { s:"6", t:"Content stored in S3 with URL hash as key + metadata in DB", c:"text-red-600" },
              { s:"7", t:"Parser extracts 47 outgoing links, normalizes them", c:"text-emerald-600" },
              { s:"8", t:"Dedup filter: 30 are new → added to frontier. 17 already seen → skipped.", c:"text-purple-600" },
            ].map((s,i) => (
              <div key={i} className="flex items-start gap-2 text-[12px]">
                <span className="text-stone-300 font-mono w-4 shrink-0">{s.s}</span>
                <span className={s.c}>{s.t}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <Label color="#dc2626">Crawl Flow — Edge Cases</Label>
          <div className="space-y-1.5">
            {[
              { s:"1", t:"robots.txt disallows /admin/* → URL skipped, marked as blocked", c:"text-red-600" },
              { s:"2", t:"HTTP 301 redirect → follow, update canonical URL, max 5 hops", c:"text-amber-600" },
              { s:"3", t:"HTTP 429 Too Many Requests → back off, increase politeness delay for domain", c:"text-red-600" },
              { s:"4", t:"Timeout after 30s → mark as failed, retry with exponential backoff (max 3)", c:"text-red-600" },
              { s:"5", t:"Content-Type: image/jpeg → skip (only crawling HTML unless configured)", c:"text-stone-400" },
              { s:"6", t:"SimHash matches existing page → near-duplicate, skip storage", c:"text-amber-600" },
              { s:"7", t:"URL depth > max_depth → skip to prevent unbounded crawling", c:"text-stone-400" },
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
  const [sel, setSel] = useState("url_frontier");
  const algos = {
    url_frontier: { name: "URL Frontier ★", cx: "O(log n) / O(n)",
      pros: ["Two-level queue: prioritizer + politeness enforcer","Balances importance (PageRank) with crawl fairness","Industry standard (Mercator architecture)","Prevents domain starvation"],
      cons: ["Complex to implement correctly","Need to balance many domains simultaneously","Memory pressure with billions of URLs"],
      when: "The URL Frontier is the heart of the crawler. Use a two-level architecture: front queues (priority-based) feed into back queues (one per domain, politeness-enforced).",
      code: `# Mercator-style URL Frontier
class URLFrontier:
    def __init__(self):
        # Front queues: priority-based (0=highest)
        self.front_queues = [deque() for _ in range(16)]
        # Back queues: one per domain (politeness)
        self.back_queues = {}  # domain -> deque
        self.domain_heap = []  # (next_fetch_time, domain)
        self.seen = BloomFilter(capacity=10_000_000_000)

    def add(self, url, priority):
        if url in self.seen:
            return False
        self.seen.add(url)
        domain = extract_domain(url)
        self.front_queues[priority].append(url)
        return True

    def get_next(self):
        # Pop domain with earliest fetch time
        while self.domain_heap:
            t, domain = heapq.heappop(self.domain_heap)
            if time.time() >= t and self.back_queues[domain]:
                url = self.back_queues[domain].popleft()
                # Re-schedule domain for next fetch
                delay = self.get_politeness_delay(domain)
                heapq.heappush(self.domain_heap,
                    (time.time() + delay, domain))
                return url
        return None  # all domains in cooldown` },
    bloom_filter: { name: "Bloom Filter Dedup", cx: "O(k) / O(m bits)",
      pros: ["Extremely space-efficient: ~1.2 bytes per URL for 1% FP","O(1) lookup and insert","Handles billions of URLs in memory","No disk I/O needed"],
      cons: ["False positives — may skip a valid URL (~1% with good sizing)","Cannot remove entries (use Counting Bloom for deletion)","Must be sized upfront (or use scalable variant)"],
      when: "Use a Bloom filter as the first-level dedup check. For 10B URLs with 1% false positive rate, need ~11.5 GB — fits in RAM. False positives mean we skip a few valid URLs, which is acceptable.",
      code: `# Bloom Filter for URL Deduplication
import mmh3
import bitarray

class BloomFilter:
    def __init__(self, capacity=10_000_000_000, fp_rate=0.01):
        # m = -(n * ln(p)) / (ln(2))^2
        self.size = int(-capacity * math.log(fp_rate)
                        / (math.log(2) ** 2))
        # k = (m/n) * ln(2)
        self.num_hashes = int(self.size / capacity
                              * math.log(2))
        self.bits = bitarray.bitarray(self.size)
        self.bits.setall(0)

    def add(self, url: str):
        for i in range(self.num_hashes):
            idx = mmh3.hash(url, i) % self.size
            self.bits[idx] = 1

    def __contains__(self, url: str) -> bool:
        return all(
            self.bits[mmh3.hash(url, i) % self.size]
            for i in range(self.num_hashes)
        )
# 10B URLs, 1% FP → ~11.5 GB memory` },
    simhash: { name: "SimHash Content Dedup", cx: "O(n) / O(1)",
      pros: ["Detects near-duplicate content (not just exact match)","64-bit fingerprint per page — tiny storage","Hamming distance = 3 catches ~95% of near-duplicates","Used by Google in production"],
      cons: ["Sensitive to document size differences","Not perfect for very short pages","Need to maintain a lookup index of fingerprints"],
      when: "Use SimHash to detect content-level duplicates. Two pages are near-duplicates if their SimHash fingerprints differ by ≤3 bits (Hamming distance). This catches mirrors, syndicated content, and slightly modified copies.",
      code: `# SimHash for Near-Duplicate Detection
def simhash(text, hash_bits=64):
    tokens = tokenize(text)  # split into words
    v = [0] * hash_bits

    for token in tokens:
        h = hash(token)
        for i in range(hash_bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(hash_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint

def is_near_duplicate(fp1, fp2, threshold=3):
    """Hamming distance <= threshold → duplicate"""
    xor = fp1 ^ fp2
    return bin(xor).count('1') <= threshold

# Example: Two mirror sites
# simhash("Welcome to ExampleStore...") → 0xA3F1
# simhash("Welcome to ExampleShop...") → 0xA3F3
# Hamming distance = 1 → NEAR DUPLICATE` },
    politeness: { name: "Politeness Engine", cx: "O(1) / O(domains)",
      pros: ["Prevents overloading target servers","Respects robots.txt Crawl-delay directives","Adaptive: slows down when server responds slowly","Essential for legal compliance and goodwill"],
      cons: ["Reduces throughput significantly","Complex scheduling across many domains","robots.txt parsing has edge cases"],
      when: "Politeness is non-negotiable. Default: max 1 request per second per domain. Respect Crawl-delay if specified in robots.txt. If server responds with 429 or 503, exponentially back off.",
      code: `# Politeness Engine
class PolitenessScheduler:
    def __init__(self):
        self.domain_state = {}  # domain -> DomainState
        self.robots_cache = {}  # domain -> RobotsRules

    def can_fetch(self, url: str) -> bool:
        domain = extract_domain(url)
        state = self.domain_state.get(domain)

        # 1. Check robots.txt
        robots = self.get_robots(domain)
        if not robots.is_allowed(url, "MyCrawler"):
            return False

        # 2. Check timing
        if state and state.last_fetch:
            delay = robots.crawl_delay or 1.0
            elapsed = time.time() - state.last_fetch
            if elapsed < delay:
                return False  # too soon

        return True

    def record_fetch(self, domain, response_time_ms):
        state = self.domain_state[domain]
        state.last_fetch = time.time()
        # Adaptive: slow down if server is struggling
        if response_time_ms > 2000:
            state.delay = min(state.delay * 2, 30)` },
  };
  const a = algos[sel];
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Algorithm Comparison</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[12px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Component</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Complexity</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Scale</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Purpose</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Critical?</th>
            </tr></thead>
            <tbody>
              {[
                { n:"URL Frontier ★", m:"O(log n)", a:"Billions of URLs", b:"Priority + scheduling", f:"Core", hl:true },
                { n:"Bloom Filter", m:"O(k)", a:"10B+ URLs in ~12GB", b:"URL deduplication", f:"Core" },
                { n:"SimHash", m:"O(n)", a:"Per page", b:"Content dedup", f:"Important" },
                { n:"Politeness Engine", m:"O(1)", a:"Per domain", b:"Rate limiting", f:"Critical" },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-purple-50" : i%2?"bg-stone-50/50":""}>
                  <td className={`px-3 py-2 font-mono ${r.hl?"text-purple-700 font-bold":"text-stone-600"}`}>{r.n}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.m}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.a}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.b}</td>
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
          <Label color="#dc2626">URL Metadata Store (PostgreSQL)</Label>
          <CodeBlock code={`-- URLs table: tracks every discovered URL
CREATE TABLE urls (
    url_hash       BIGINT PRIMARY KEY,   -- 64-bit hash
    url            TEXT NOT NULL,
    domain         VARCHAR(255) NOT NULL,
    status         VARCHAR(20) DEFAULT 'pending',
    -- pending | fetching | fetched | failed | blocked
    priority       SMALLINT DEFAULT 5,   -- 0=highest
    depth          SMALLINT DEFAULT 0,
    last_crawled   TIMESTAMP,
    next_crawl     TIMESTAMP,
    http_status    SMALLINT,
    content_hash   BIGINT,               -- SimHash
    retry_count    SMALLINT DEFAULT 0,
    discovered_by  BIGINT,               -- parent URL hash
    created_at     TIMESTAMP DEFAULT NOW()
);

-- Index for frontier: next URLs to crawl
CREATE INDEX idx_urls_pending
    ON urls(priority, next_crawl)
    WHERE status = 'pending';

-- Index for per-domain scheduling
CREATE INDEX idx_urls_domain
    ON urls(domain, last_crawled);`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Domain Configuration Store</Label>
          <CodeBlock code={`-- Domain state: politeness + robots.txt
CREATE TABLE domains (
    domain         VARCHAR(255) PRIMARY KEY,
    robots_txt     TEXT,
    robots_fetched TIMESTAMP,
    crawl_delay    FLOAT DEFAULT 1.0,    -- seconds
    last_fetched   TIMESTAMP,
    avg_response_ms INT,
    total_pages    INT DEFAULT 0,
    status         VARCHAR(20) DEFAULT 'active',
    -- active | throttled | blocked | blacklisted
    priority       SMALLINT DEFAULT 5
);

-- Content store metadata (actual content in S3)
CREATE TABLE pages (
    url_hash       BIGINT PRIMARY KEY,
    s3_key         VARCHAR(512) NOT NULL,
    content_type   VARCHAR(100),
    content_length INT,
    content_hash   BIGINT,               -- for dedup
    title          TEXT,
    language       VARCHAR(10),
    fetched_at     TIMESTAMP,
    headers_json   JSONB
);`} />
        </Card>
      </div>
      <Card accent="#9333ea">
        <Label color="#9333ea">Storage Architecture — Where Everything Lives</Label>
        <p className="text-[12px] text-stone-500 mb-4">A web crawler has multiple storage tiers. The key insight: URL metadata must be fast (database), raw content can be cold (object storage), and the frontier needs sub-millisecond access (in-memory).</p>
        <div className="grid grid-cols-3 gap-4">
          {[
            { t: "URL Frontier (Redis/Memory)", d: "Active crawl queue. Priority queues per domain. Needs sub-ms read for scheduling.", data: "Current batch: ~10M active URLs", tech: "Redis Sorted Sets or custom in-memory heap", size: "~5 GB (active working set)" },
            { t: "URL Metadata (PostgreSQL)", d: "All discovered URLs + status. Queried for dedup, analytics, recrawl scheduling.", data: "All URLs: billions of rows", tech: "Sharded PostgreSQL or Cassandra for >10B URLs", size: "~500 GB for 1B URLs" },
            { t: "Content Store (S3/HDFS)", d: "Raw HTML content. Keyed by URL hash. Compressed. Write-heavy, read-seldom.", data: "HTML pages + headers", tech: "S3 (or GCS/HDFS). Glacier for archives.", size: "~100 TB/month" },
          ].map((o,i) => (
            <div key={i} className="rounded-lg border border-stone-200 p-4">
              <div className="text-[11px] font-bold text-stone-800 mb-1.5">{o.t}</div>
              <p className="text-[11px] text-stone-500 mb-2">{o.d}</p>
              <div className="space-y-0.5 text-[10px]">
                <div><span className="font-bold text-stone-600">Data:</span> <span className="text-stone-400">{o.data}</span></div>
                <div><span className="font-bold text-stone-600">Tech:</span> <span className="text-stone-400">{o.tech}</span></div>
                <div><span className="font-bold text-stone-600">Size:</span> <span className="font-mono text-stone-400">{o.size}</span></div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 pt-3 border-t border-stone-100">
          <div className="text-[11px] text-stone-500">
            <strong className="text-stone-700">Key insight to mention in interview:</strong> The Bloom filter for URL dedup lives entirely in memory (~12 GB for 10B URLs) and is the critical path. The database is the source of truth but is only consulted when the Bloom filter says "maybe new." This two-tier approach gives O(1) lookups with negligible false positives.
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
          <Label color="#059669">Scaling the Crawl Workers</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Horizontal scaling</strong> — add workers freely. Each worker pulls URLs from the frontier. No shared state between workers (stateless).</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Domain partitioning</strong> — hash domains to workers so the same domain is always handled by the same worker. This makes per-domain politeness trivial (no distributed coordination).</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Async I/O</strong> — each worker maintains ~500 concurrent HTTP connections using asyncio/Netty. Fetching is I/O-bound, not CPU-bound.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Auto-scaling</strong> — scale workers based on frontier queue depth. If queue grows beyond threshold, add workers. If queue shrinks, scale down.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Scaling the URL Frontier</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Shard by domain hash</strong> — partition frontier across N Redis nodes. Domain → consistent hash → shard. Each shard manages its own domains.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">In-memory for active set</strong> — only keep the next ~10M URLs in Redis. Spill overflow to disk-backed queue (RocksDB) for URLs scheduled far in the future.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Bloom filter partitioning</strong> — split the Bloom filter across machines. Route URL checks by hash prefix to the right partition.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Batch operations</strong> — workers pull URLs in batches of 50-100, reducing round-trips to the frontier service.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Crawling Strategy</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Option A: Centralized", d:"All workers in one region. Crawl the entire web from US-East.", pros:["Simplest architecture","Single frontier, no coordination"], cons:["High latency to distant servers","Poor for geo-restricted content"], pick:false },
            { t:"Option B: Geo-Distributed ★", d:"Crawl workers near target servers: US workers for .com, EU workers for .eu/.de, Asia workers for .jp/.cn.", pros:["Lower fetch latency","Access geo-restricted content","Better crawl throughput"], cons:["Need cross-region frontier sync","Higher operational complexity"], pick:true },
            { t:"Option C: Regional Autonomy", d:"Fully independent crawlers per region with periodic dedup sync.", pros:["No cross-region dependency","Each region is self-sufficient"], cons:["Significant duplicate work","Complex merging of results"], pick:false },
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
        <Label color="#d97706">Critical Decision: Crawl Continuity on Failure</Label>
        <p className="text-[12px] text-stone-500 mb-4">What happens when a component fails? Unlike user-facing systems, crawlers can tolerate brief pauses. The key is never losing URLs and never re-crawling excessively.</p>
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Worker Failure (Expected)</div>
            <p className="text-[11px] text-stone-500 mb-2">Worker crashes → URLs it was processing re-enter frontier after timeout</p>
            <ul className="space-y-1"><Point icon="✓" color="#059669">URLs have "leased" state with TTL (5min)</Point><Point icon="✓" color="#059669">If worker doesn't ACK within TTL, URL returns to queue</Point><Point icon="✓" color="#059669">At-least-once semantics — may re-crawl, but never lose a URL</Point></ul>
          </div>
          <div className="rounded-lg border border-red-200 bg-white p-4">
            <div className="text-[11px] font-bold text-red-600 mb-1.5">Frontier Failure (Critical)</div>
            <p className="text-[11px] text-stone-500 mb-2">Redis frontier down → all crawling pauses until restored</p>
            <ul className="space-y-1"><Point icon="⚠" color="#d97706">Redis persistence (AOF) ensures no URL loss on restart</Point><Point icon="⚠" color="#d97706">Redis Sentinel for automatic failover (5-10s)</Point><Point icon="⚠" color="#d97706">Workers retry frontier connection with exponential backoff</Point></ul>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Checkpointing Strategy</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Frontier snapshots</strong> — periodically dump frontier state (Redis RDB) every 5 minutes. On catastrophic failure, restore from last snapshot — at most 5 minutes of re-crawl.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Bloom filter persistence</strong> — serialize Bloom filter to disk hourly. Without this, a restart means re-crawling all URLs (catastrophic waste).</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Crawl progress journal</strong> — write-ahead log of (url, status, timestamp) to Kafka. Can rebuild frontier state by replaying the log.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Idempotent storage</strong> — writing the same page to S3 twice is harmless (same key). Content store is naturally idempotent.</Point>
          </ul>
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Failure Recovery Matrix</Label>
          <div className="overflow-hidden rounded-lg border border-stone-200">
            <table className="w-full text-[11px]">
              <thead><tr className="bg-stone-50">
                <th className="text-left px-3 py-2 font-semibold text-stone-500">Component</th>
                <th className="text-left px-3 py-2 font-semibold text-stone-500">Impact</th>
                <th className="text-left px-3 py-2 font-semibold text-stone-500">Recovery</th>
                <th className="text-left px-3 py-2 font-semibold text-stone-500">RTO</th>
              </tr></thead>
              <tbody>
                {[
                  { c: "Crawl worker", impact: "Reduced throughput", recovery: "Auto-restart + URL re-queue", rto: "<1 min" },
                  { c: "DNS resolver", impact: "Fetches stall", recovery: "Fallback to public DNS (8.8.8.8)", rto: "Instant" },
                  { c: "Redis frontier", impact: "All crawling pauses", recovery: "Sentinel failover → replica promoted", rto: "5-10s" },
                  { c: "Bloom filter", impact: "Dedup disabled → duplicates", recovery: "Rebuild from URL DB (slow)", rto: "10-30 min" },
                  { c: "S3 storage", impact: "Content not persisted", recovery: "AWS-managed, multi-AZ. Re-fetch if lost.", rto: "<1 min" },
                  { c: "PostgreSQL", impact: "No metadata updates", recovery: "RDS Multi-AZ failover", rto: "1-2 min" },
                ].map((r,i) => (
                  <tr key={i} className={i%2?"bg-stone-50/50":""}>
                    <td className="px-3 py-2 font-mono text-stone-700">{r.c}</td>
                    <td className="px-3 py-2 text-stone-500">{r.impact}</td>
                    <td className="px-3 py-2 text-stone-500">{r.recovery}</td>
                    <td className="px-3 py-2 font-mono text-stone-400">{r.rto}</td>
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

function ObservabilitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-5">
        <Card accent="#0284c7">
          <Label color="#0284c7">Key Metrics to Track</Label>
          <div className="space-y-2.5">
            {[
              { m: "pages_crawled_total", t: "Counter", d: "Total pages successfully fetched" },
              { m: "pages_per_second", t: "Gauge", d: "Current crawl throughput" },
              { m: "frontier_queue_size", t: "Gauge", d: "URLs waiting in frontier" },
              { m: "fetch_latency_ms", t: "Histogram", d: "Time to download a page (p50/p99)" },
              { m: "fetch_errors_total", t: "Counter", d: "By type: timeout, DNS, 4xx, 5xx" },
              { m: "duplicate_rate", t: "Gauge", d: "% of URLs already seen (Bloom filter hits)" },
              { m: "domain_crawl_rate", t: "Gauge", d: "Requests/sec per domain (politeness check)" },
              { m: "content_size_bytes", t: "Histogram", d: "Downloaded content size distribution" },
            ].map((m,i) => (
              <div key={i} className="flex items-start gap-2.5 text-[11px]">
                <code className="font-mono text-sky-700 bg-sky-50 px-1.5 py-0.5 rounded shrink-0">{m.m}</code>
                <div className="flex-1">
                  <span className="text-stone-500">{m.d}</span>
                  <span className="text-stone-300 ml-1">({m.t})</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Dashboards</Label>
          <div className="space-y-3">
            {[
              { name: "Crawl Throughput", panels: "Pages/sec, URLs discovered, frontier depth, worker count", tool: "Grafana" },
              { name: "Error Dashboard", panels: "Errors by type (DNS, timeout, 429, 5xx), error rate trend, top failing domains", tool: "Grafana" },
              { name: "Domain Health", panels: "Per-domain crawl rate, avg latency, robots.txt blocks, politeness violations", tool: "Grafana" },
              { name: "Storage Dashboard", panels: "S3 write rate, total stored, dedup savings, DB disk usage", tool: "CloudWatch" },
            ].map((d,i) => (
              <div key={i} className="border border-stone-100 rounded-lg p-3">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-bold text-stone-700">{d.name}</span>
                  <Pill bg="#ecfdf5" color="#059669">{d.tool}</Pill>
                </div>
                <div className="text-[10px] text-stone-400 mt-0.5">{d.panels}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Alerts</Label>
          <div className="space-y-3">
            {[
              { alert: "Crawl rate < 100/sec", severity: "P1", action: "Check worker health, frontier connectivity" },
              { alert: "Frontier queue > 100M", severity: "P2", action: "Scale up workers or investigate slow domains" },
              { alert: "Error rate > 20%", severity: "P1", action: "Check DNS, network, or target site outage" },
              { alert: "Bloom filter memory > 90%", severity: "P2", action: "Resize Bloom filter or rotate to new one" },
              { alert: "Politeness violation detected", severity: "P1", action: "Immediately throttle offending worker" },
              { alert: "S3 write failures", severity: "P1", action: "Check IAM permissions, S3 service health" },
            ].map((a,i) => (
              <div key={i} className="border border-stone-100 rounded-lg p-2.5">
                <div className="flex items-center gap-2">
                  <Pill bg={a.severity==="P1"?"#fef2f2":"#fffbeb"} color={a.severity==="P1"?"#dc2626":"#d97706"}>{a.severity}</Pill>
                  <span className="text-[11px] font-bold text-stone-700">{a.alert}</span>
                </div>
                <div className="text-[10px] text-stone-400 mt-0.5">{a.action}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
      <Card>
        <Label color="#0284c7">Logging Strategy</Label>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-[10px] font-bold text-stone-600 uppercase tracking-wider mb-1.5">Per-URL Log (structured, JSON)</div>
            <CodeBlock code={`{
  "url": "https://example.com/page",
  "url_hash": "a3f1b2c4d5e6",
  "status": "fetched",
  "http_status": 200,
  "fetch_time_ms": 342,
  "content_size": 154230,
  "links_discovered": 47,
  "links_new": 30,
  "content_hash": "0xA3F1B2C4",
  "is_duplicate": false,
  "worker_id": "worker-03",
  "timestamp": "2025-01-15T10:23:45Z"
}`} />
          </div>
          <div>
            <div className="text-[10px] font-bold text-stone-600 uppercase tracking-wider mb-1.5">Log Pipeline</div>
            <ul className="space-y-2">
              <Point icon="1." color="#0284c7">Workers emit structured JSON logs to stdout</Point>
              <Point icon="2." color="#0284c7">FluentBit sidecar ships logs to Kafka topic</Point>
              <Point icon="3." color="#0284c7">Kafka → ClickHouse for analytics (crawl stats, domain performance)</Point>
              <Point icon="4." color="#0284c7">Kafka → Elasticsearch for operational search (find specific URL crawl history)</Point>
              <Point icon="5." color="#0284c7">Retention: 7 days in ES, 90 days in ClickHouse, archived to S3</Point>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}

function WatchoutsSection() {
  return (
    <div className="space-y-5">
      <Card className="bg-red-50/50 border-red-200">
        <div className="flex items-start gap-3">
          <span className="text-lg">⚠️</span>
          <div>
            <div className="text-[12px] font-bold text-red-700">Interview Tip — Show You Know the Gotchas</div>
            <p className="text-[12px] text-stone-500 mt-0.5">Proactively mentioning failure modes shows real-world experience. After presenting the architecture, say: <em>"Now let me walk through what can go wrong..."</em></p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        {[
          { mode: "Crawler Traps", desc: "Infinite URLs generated by dynamic content: calendars with infinite dates, session IDs in URLs, sort/filter permutations.", impact: "Crawler wastes resources on one domain forever.", fix: "Max depth per domain, max pages per domain, URL pattern detection (regex), timeout per domain budget.", severity: "Critical" },
          { mode: "DNS Failures", desc: "DNS resolver overwhelmed, DNS cache expires, domain DNS server unreachable.", impact: "All fetches for affected domains fail. If resolver dies, all fetching stops.", fix: "Local DNS cache (TTL-aware), fallback resolvers (8.8.8.8, 1.1.1.1), circuit breaker per domain.", severity: "High" },
          { mode: "Spider Traps (Malicious)", desc: "Sites intentionally generating infinite pages to consume crawler resources (honeypots for bots).", impact: "Worker stuck on one domain, bandwidth wasted.", fix: "Per-domain page budget, content fingerprinting (detect identical/near-identical pages), blacklist known trap domains.", severity: "High" },
          { mode: "Bloom Filter Saturation", desc: "Bloom filter reaches capacity — false positive rate spikes. New valid URLs incorrectly marked as 'already seen'.", impact: "Crawler stops discovering new pages (silent data loss).", fix: "Monitor fill ratio. When >70% full, create a new Bloom filter and merge. Or use Scalable Bloom Filters.", severity: "Critical" },
          { mode: "Robots.txt Outage", desc: "robots.txt unreachable for a domain. Should crawler proceed or block?", impact: "Crawling without permission risks legal issues. Blocking loses all pages for that domain.", fix: "Cache robots.txt for 24h. If expired and unreachable: use cached version. If never fetched: block. RFC 9309 says treat as allow after reasonable retries.", severity: "Medium" },
          { mode: "Content Explosion", desc: "Single page returns massive content (100MB+ file served as text/html, or infinite streaming response).", impact: "Worker memory OOM, storage spike, other URLs starved.", fix: "Max content size limit (10MB), streaming download with early termination, Content-Length header check.", severity: "Medium" },
        ].map((f,i) => (
          <Card key={i}>
            <div className="flex items-center justify-between mb-2">
              <div className="text-[12px] font-bold text-stone-800">{f.mode}</div>
              <Pill bg={f.severity==="Critical"?"#fef2f2":f.severity==="High"?"#fffbeb":"#f0fdf4"} color={f.severity==="Critical"?"#dc2626":f.severity==="High"?"#d97706":"#059669"}>{f.severity}</Pill>
            </div>
            <p className="text-[12px] text-stone-500 mb-2">{f.desc}</p>
            <div className="text-[11px] text-red-600 mb-1.5"><strong>Impact:</strong> {f.impact}</div>
            <div className="text-[11px] text-emerald-700"><strong>Mitigation:</strong> {f.fix}</div>
          </Card>
        ))}
      </div>
    </div>
  );
}

function ServicesSection() {
  return (
    <div className="space-y-5">
      <Card className="bg-teal-50/50 border-teal-200">
        <div className="flex items-start gap-3">
          <span className="text-lg">🧩</span>
          <div>
            <div className="text-[12px] font-bold text-teal-700">Why This Matters</div>
            <p className="text-[12px] text-stone-500 mt-0.5">In an HLD you draw one "Crawler" box. In production, that box is 6+ independently deployed services — each with its own scaling, failure mode, and team ownership.</p>
          </div>
        </div>
      </Card>

      <Card accent="#0f766e">
        <Label color="#0f766e">Service Decomposition — What We Actually Build</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { name: "URL Frontier Service", owns: "Manages the priority queue of URLs to crawl. Handles dedup, priority assignment, and batch dispatch to workers.", tech: "Custom service (Go/Rust) + Redis Cluster for active queue + RocksDB for overflow", api: "gRPC: GetBatch(domain, n), AddURLs(urls), AckBatch(results)", scale: "Shard by domain hash. Each shard handles ~100K domains.", stateful: true,
              modules: ["Priority Queue Manager (front queues by priority)", "Domain Scheduler (back queues by domain, politeness timing)", "Bloom Filter (in-memory URL dedup, ~12GB for 10B URLs)", "Overflow Spiller (active queue → RocksDB when Redis pressure)", "Checkpoint Writer (periodic frontier state snapshot)", "Metrics Emitter (queue depth, dedup rate, throughput)"] },
            { name: "Fetcher Workers", owns: "Download web pages. Handle DNS, HTTP, redirects, timeouts, robots.txt. Pure I/O workers.", tech: "Python (aiohttp) or Go (net/http) with async I/O. Runs as K8s pods.", api: "Pulls from Frontier via gRPC. Pushes content to S3 + metadata to Kafka.", scale: "Horizontal: add pods to increase throughput. Domain-pinned via consistent hashing.", stateful: false,
              modules: ["HTTP Client (async, connection pooling, redirect following)", "DNS Resolver (local cache + fallback resolvers)", "Robots.txt Manager (fetch, parse, cache per domain)", "Content Validator (check size, type, encoding)", "Retry Handler (exponential backoff, max 3 attempts)", "Politeness Timer (enforce per-domain delay)"] },
            { name: "Content Processor", owns: "Parse HTML, extract links, compute SimHash fingerprint, normalize URLs, extract metadata.", tech: "Python (BeautifulSoup/lxml) or Java (Jsoup). Kafka consumer.", api: "Consumes from Kafka topic: raw_pages. Produces to: discovered_urls, parsed_content.", scale: "Horizontal: add consumers per Kafka partition.", stateful: false,
              modules: ["HTML Parser (link extraction, title, meta tags)", "URL Normalizer (canonicalize, resolve relative, remove fragments)", "SimHash Engine (near-duplicate fingerprint, 64-bit)", "Link Classifier (internal vs external, priority scoring)", "Metadata Extractor (language, charset, canonical URL)", "Output Router (new URLs → Frontier, content → Store)"] },
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
                    <span className="w-1 h-1 rounded-full bg-teal-400 shrink-0"/>
                    {m}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card accent="#9333ea">
        <Label color="#9333ea">Full Service Architecture Diagram</Label>
        <svg viewBox="0 0 720 420" className="w-full">
          {/* Background zones */}
          <rect x={5} y={5} width={710} height={410} rx={10} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>

          {/* Zone labels */}
          <rect x={15} y={15} width={130} height={22} rx={4} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
          <text x={80} y={30} textAnchor="middle" fill="#6366f1" fontSize="8" fontWeight="700" fontFamily="monospace">INGESTION LAYER</text>

          <rect x={155} y={15} width={150} height={22} rx={4} fill="#9333ea08" stroke="#9333ea30" strokeWidth={1}/>
          <text x={230} y={30} textAnchor="middle" fill="#9333ea" fontSize="8" fontWeight="700" fontFamily="monospace">COORDINATION LAYER</text>

          <rect x={315} y={15} width={200} height={22} rx={4} fill="#2563eb08" stroke="#2563eb30" strokeWidth={1}/>
          <text x={415} y={30} textAnchor="middle" fill="#2563eb" fontSize="8" fontWeight="700" fontFamily="monospace">FETCH & PROCESS LAYER</text>

          <rect x={525} y={15} width={185} height={22} rx={4} fill="#dc262608" stroke="#dc262630" strokeWidth={1}/>
          <text x={617} y={30} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="700" fontFamily="monospace">STORAGE LAYER</text>

          {/* Seed URLs */}
          <rect x={25} y={60} width={100} height={40} rx={8} fill="#6366f112" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={75} y={77} textAnchor="middle" fill="#6366f1" fontSize="10" fontWeight="600" fontFamily="monospace">Seed URLs</text>
          <text x={75} y={90} textAnchor="middle" fill="#6366f180" fontSize="7" fontFamily="monospace">sitemap.xml</text>

          {/* Re-Crawl Scheduler */}
          <rect x={25} y={120} width={100} height={40} rx={8} fill="#d9770612" stroke="#d97706" strokeWidth={1.5}/>
          <text x={75} y={137} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Re-Crawl</text>
          <text x={75} y={150} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Scheduler</text>

          {/* URL Frontier */}
          <rect x={170} y={55} width={120} height={55} rx={8} fill="#9333ea12" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={230} y={75} textAnchor="middle" fill="#9333ea" fontSize="11" fontWeight="700" fontFamily="monospace">URL Frontier</text>
          <text x={230} y={90} textAnchor="middle" fill="#9333ea80" fontSize="7" fontFamily="monospace">Priority Queue + Dedup</text>
          <text x={230} y={100} textAnchor="middle" fill="#9333ea80" fontSize="7" fontFamily="monospace">Bloom Filter (12GB)</text>

          {/* Politeness Scheduler */}
          <rect x={170} y={125} width={120} height={40} rx={8} fill="#c026d312" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={230} y={142} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">Politeness</text>
          <text x={230} y={155} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">Scheduler</text>

          {/* DNS Resolver */}
          <rect x={330} y={55} width={90} height={40} rx={8} fill="#64748b12" stroke="#64748b" strokeWidth={1.5}/>
          <text x={375} y={72} textAnchor="middle" fill="#64748b" fontSize="9" fontWeight="600" fontFamily="monospace">DNS Resolver</text>
          <text x={375} y={85} textAnchor="middle" fill="#64748b80" fontSize="7" fontFamily="monospace">cached, fallback</text>

          {/* Fetcher Workers */}
          <rect x={330} y={110} width={90} height={55} rx={8} fill="#2563eb12" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={375} y={130} textAnchor="middle" fill="#2563eb" fontSize="10" fontWeight="700" fontFamily="monospace">Fetcher</text>
          <text x={375} y={143} textAnchor="middle" fill="#2563eb" fontSize="10" fontWeight="700" fontFamily="monospace">Workers</text>
          <text x={375} y={157} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">×10 pods, async I/O</text>

          {/* robots.txt cache */}
          <rect x={330} y={180} width={90} height={32} rx={6} fill="#be123c12" stroke="#be123c" strokeWidth={1}/>
          <text x={375} y={200} textAnchor="middle" fill="#be123c" fontSize="8" fontWeight="600" fontFamily="monospace">robots.txt cache</text>

          {/* Kafka */}
          <rect x={455} y={110} width={85} height={45} rx={8} fill="#05966912" stroke="#059669" strokeWidth={1.5}/>
          <text x={497} y={130} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="600" fontFamily="monospace">Kafka</text>
          <text x={497} y={145} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">raw_pages topic</text>

          {/* Content Processor */}
          <rect x={445} y={175} width={105} height={50} rx={8} fill="#05966912" stroke="#059669" strokeWidth={1.5}/>
          <text x={497} y={195} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="700" fontFamily="monospace">Content</text>
          <text x={497} y={208} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="700" fontFamily="monospace">Processor</text>
          <text x={497} y={220} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">parse, SimHash, links</text>

          {/* S3 Content Store */}
          <rect x={585} y={60} width={120} height={45} rx={8} fill="#dc262612" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={645} y={78} textAnchor="middle" fill="#dc2626" fontSize="10" fontWeight="600" fontFamily="monospace">S3 Content</text>
          <text x={645} y={95} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">HTML pages (100TB/mo)</text>

          {/* PostgreSQL */}
          <rect x={585} y={120} width={120} height={45} rx={8} fill="#d9770612" stroke="#d97706" strokeWidth={1.5}/>
          <text x={645} y={138} textAnchor="middle" fill="#d97706" fontSize="10" fontWeight="600" fontFamily="monospace">PostgreSQL</text>
          <text x={645} y={153} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">URL metadata (1B rows)</text>

          {/* Redis */}
          <rect x={585} y={180} width={120} height={40} rx={8} fill="#dc262612" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={645} y={198} textAnchor="middle" fill="#dc2626" fontSize="10" fontWeight="600" fontFamily="monospace">Redis Cluster</text>
          <text x={645} y={212} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">frontier active queue</text>

          {/* Trap Detector */}
          <rect x={445} y={245} width={105} height={35} rx={6} fill="#be123c12" stroke="#be123c" strokeWidth={1}/>
          <text x={497} y={266} textAnchor="middle" fill="#be123c" fontSize="9" fontWeight="600" fontFamily="monospace">Trap Detector</text>

          {/* Monitoring */}
          <rect x={585} y={240} width={120} height={40} rx={8} fill="#0284c712" stroke="#0284c7" strokeWidth={1.5}/>
          <text x={645} y={257} textAnchor="middle" fill="#0284c7" fontSize="9" fontWeight="600" fontFamily="monospace">Monitoring</text>
          <text x={645} y={270} textAnchor="middle" fill="#0284c780" fontSize="7" fontFamily="monospace">Grafana + Prometheus</text>

          {/* ClickHouse */}
          <rect x={585} y={295} width={120} height={35} rx={6} fill="#0284c712" stroke="#0284c7" strokeWidth={1}/>
          <text x={645} y={316} textAnchor="middle" fill="#0284c7" fontSize="9" fontWeight="600" fontFamily="monospace">ClickHouse (analytics)</text>

          {/* ─── ARROWS ─── */}
          {/* Seeds → Frontier */}
          <line x1={125} y1={80} x2={170} y2={80} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#svc-arr)"/>
          <text x={147} y={74} textAnchor="middle" fill="#64748b" fontSize="7" fontFamily="monospace">inject</text>

          {/* Re-Crawl → Frontier */}
          <line x1={125} y1={140} x2={155} y2={140} stroke="#94a3b8" strokeWidth={1.5}/>
          <line x1={155} y1={140} x2={155} y2={90} stroke="#94a3b8" strokeWidth={1.5}/>
          <line x1={155} y1={90} x2={170} y2={90} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#svc-arr)"/>

          {/* Frontier → Scheduler */}
          <line x1={230} y1={110} x2={230} y2={125} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#svc-arr)"/>

          {/* Scheduler → Fetcher */}
          <line x1={290} y1={140} x2={330} y2={137} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#svc-arr)"/>
          <text x={310} y={132} textAnchor="middle" fill="#64748b" fontSize="7" fontFamily="monospace">dispatch</text>

          {/* Fetcher → DNS */}
          <line x1={375} y1={110} x2={375} y2={95} stroke="#94a3b8" strokeWidth={1.2} strokeDasharray="4,2" markerEnd="url(#svc-arr)"/>
          <text x={392} y={104} fill="#64748b" fontSize="6" fontFamily="monospace">resolve</text>

          {/* Fetcher → robots.txt */}
          <line x1={375} y1={165} x2={375} y2={180} stroke="#94a3b8" strokeWidth={1} strokeDasharray="4,2" markerEnd="url(#svc-arr)"/>

          {/* Fetcher → Kafka */}
          <line x1={420} y1={132} x2={455} y2={132} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#svc-arr)"/>
          <text x={437} y={126} textAnchor="middle" fill="#64748b" fontSize="7" fontFamily="monospace">emit</text>

          {/* Fetcher → S3 */}
          <line x1={420} y1={122} x2={530} y2={80} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#svc-arr)"/>
          <text x={480} y={92} textAnchor="middle" fill="#64748b" fontSize="7" fontFamily="monospace">store HTML</text>

          {/* Kafka → Processor */}
          <line x1={497} y1={155} x2={497} y2={175} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#svc-arr)"/>

          {/* Processor → Frontier (new URLs loop back) */}
          <path d="M 445 200 Q 300 260 230 170" fill="none" stroke="#9333ea" strokeWidth={1.5} strokeDasharray="5,3" markerEnd="url(#svc-arr-purple)"/>
          <text x={310} y={245} textAnchor="middle" fill="#9333ea" fontSize="8" fontWeight="600" fontFamily="monospace">new URLs → Frontier</text>

          {/* Processor → PostgreSQL */}
          <line x1={550} y1={195} x2={585} y2={150} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#svc-arr)"/>
          <text x={572} y={168} textAnchor="middle" fill="#64748b" fontSize="6" fontFamily="monospace">metadata</text>

          {/* Processor → Trap Detector */}
          <line x1={497} y1={225} x2={497} y2={245} stroke="#94a3b8" strokeWidth={1} strokeDasharray="4,2" markerEnd="url(#svc-arr)"/>

          {/* Frontier → Redis */}
          <line x1={290} y1={95} x2={560} y2={200} stroke="#dc2626" strokeWidth={1.2} strokeDasharray="4,2"/>
          <text x={420} y={165} textAnchor="middle" fill="#dc2626" fontSize="7" fontFamily="monospace">active queue</text>

          {/* All → Monitoring */}
          <line x1={540} y1={250} x2={585} y2={255} stroke="#0284c7" strokeWidth={1} strokeDasharray="3,2"/>
          <line x1={645} y1={280} x2={645} y2={295} stroke="#0284c7" strokeWidth={1} strokeDasharray="3,2"/>

          {/* Arrow markers */}
          <defs>
            <marker id="svc-arr" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
              <polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/>
            </marker>
            <marker id="svc-arr-purple" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
              <polygon points="0 0,7 2.5,0 5" fill="#9333ea"/>
            </marker>
          </defs>

          {/* Legend */}
          <rect x={15} y={310} width={540} height={98} rx={8} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={30} y={330} fill="#78716c" fontSize="9" fontWeight="700" fontFamily="monospace">Data Flow Summary</text>
          <text x={30} y={348} fill="#78716c" fontSize="8" fontFamily="monospace">1. Seed URLs / Re-Crawl Scheduler inject URLs → URL Frontier (priority queue + Bloom dedup)</text>
          <text x={30} y={363} fill="#78716c" fontSize="8" fontFamily="monospace">2. Politeness Scheduler enforces per-domain delay → dispatches batch to Fetcher Workers</text>
          <text x={30} y={378} fill="#78716c" fontSize="8" fontFamily="monospace">3. Fetchers resolve DNS, download HTML → store in S3, emit raw_pages event to Kafka</text>
          <text x={30} y={393} fill="#78716c" fontSize="8" fontFamily="monospace">4. Content Processor parses HTML, extracts links, computes SimHash → new URLs loop back to Frontier</text>
        </svg>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#2563eb">
          <Label color="#2563eb">Supporting Services</Label>
          <div className="space-y-3">
            {[
              { name: "DNS Resolver Cache", role: "Local caching DNS resolver to reduce external DNS lookups. Caches results with TTL awareness.", tech: "Unbound DNS + Redis cache", critical: true },
              { name: "Re-Crawl Scheduler", role: "Periodically injects URLs for re-crawling based on page change frequency and importance. Uses historical change data.", tech: "Cron job + PostgreSQL queries → Frontier API", critical: false },
              { name: "Trap Detector", role: "Analyzes crawl patterns per domain. Detects infinite loops, calendar traps, session-ID pollution. Blacklists offending URL patterns.", tech: "Kafka consumer → pattern analysis → domain config update", critical: false },
              { name: "Crawl Dashboard", role: "Real-time visibility into crawl progress: pages/sec, queue depth, errors, per-domain stats, worker health.", tech: "React + Grafana + ClickHouse", critical: false },
            ].map((s,i) => (
              <div key={i} className="flex items-start gap-3 border border-stone-100 rounded-lg p-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-bold text-stone-700">{s.name}</span>
                    {s.critical && <Pill bg="#fef2f2" color="#dc2626">Critical Path</Pill>}
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
            <table className="w-full text-[11px]">
              <thead><tr className="bg-stone-50">
                <th className="text-left px-3 py-2 font-semibold text-stone-500">Caller → Callee</th>
                <th className="text-left px-3 py-2 font-semibold text-stone-500">Protocol</th>
                <th className="text-left px-3 py-2 font-semibold text-stone-500">Timeout</th>
                <th className="text-left px-3 py-2 font-semibold text-stone-500">On Failure</th>
              </tr></thead>
              <tbody>
                {[
                  { route: "Fetcher → Frontier", proto: "gRPC", timeout: "2s", fail: "Retry 3×, then pause worker" },
                  { route: "Fetcher → DNS Cache", proto: "DNS/UDP", timeout: "500ms", fail: "Fallback to 8.8.8.8" },
                  { route: "Fetcher → Target Site", proto: "HTTP/HTTPS", timeout: "30s", fail: "Retry 2×, mark URL failed" },
                  { route: "Fetcher → S3", proto: "HTTPS (AWS SDK)", timeout: "10s", fail: "Retry 3×, buffer to disk" },
                  { route: "Fetcher → Kafka", proto: "TCP (producer)", timeout: "5s", fail: "Buffer in memory, retry" },
                  { route: "Processor → Frontier", proto: "gRPC", timeout: "2s", fail: "Retry, dead-letter queue" },
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
    </div>
  );
}

function FlowsSection() {
  const [flow, setFlow] = useState("happy");
  const flows = {
    happy: {
      title: "Happy Path — URL Discovered → Fetched → Stored",
      steps: [
        { actor: "Frontier", action: "Pops URL https://news.example.com/article/123 (priority=2, depth=3) from domain queue", type: "process" },
        { actor: "Scheduler", action: "Checks domain politeness: last fetch for news.example.com was 1.2s ago. Delay=1s. ✓ Proceed.", type: "check" },
        { actor: "Fetcher", action: "DNS resolve news.example.com → 93.184.216.34 (cached, 0.1ms)", type: "process" },
        { actor: "Fetcher", action: "HTTP GET https://news.example.com/article/123 (User-Agent: MyCrawler/1.0)", type: "request" },
        { actor: "Target Server", action: "Returns 200 OK, Content-Type: text/html, 156KB body, 340ms", type: "success" },
        { actor: "Fetcher", action: "Compress content (gzip) → 31KB. Upload to S3: s3://crawl-data/2025/01/15/a3f1b2c4.html.gz", type: "process" },
        { actor: "Fetcher → Kafka", action: "Emit raw_pages event: {url, s3_key, status:200, fetch_time:340ms, size:156KB}", type: "process" },
        { actor: "Content Processor", action: "Consumes from Kafka. Parses HTML: extracts 47 links, title, meta tags, language=en", type: "process" },
        { actor: "Content Processor", action: "SimHash fingerprint = 0xA3F1B2C4D5E6. Check dedup store: NO match → new content.", type: "check" },
        { actor: "Content Processor → Frontier", action: "Submit 47 links. Bloom filter: 30 are new → enqueued. 17 seen → skipped.", type: "success" },
      ]
    },
    blocked: {
      title: "Blocked by robots.txt",
      steps: [
        { actor: "Frontier", action: "Pops URL https://example.com/admin/users from queue", type: "process" },
        { actor: "Fetcher", action: "Check robots.txt for example.com (cached from 6h ago)", type: "process" },
        { actor: "Robots Parser", action: "Rule: 'Disallow: /admin/' matches /admin/users → BLOCKED", type: "error" },
        { actor: "Fetcher", action: "Mark URL status='blocked'. No HTTP request sent.", type: "error" },
        { actor: "Fetcher → Kafka", action: "Emit event: {url, status:'robots_blocked', reason:'Disallow: /admin/'}", type: "process" },
        { actor: "Metrics", action: "Increment counter: robots_blocked_total{domain='example.com'}", type: "process" },
      ]
    },
    trap: {
      title: "Crawler Trap Detection",
      steps: [
        { actor: "Frontier", action: "Pops URL https://evil.com/calendar/2025/01/15/view?sid=abc123 (depth=47)", type: "process" },
        { actor: "Trap Detector", action: "Check 1: depth=47 > max_depth(20) → ⚠ suspicious", type: "check" },
        { actor: "Trap Detector", action: "Check 2: evil.com has 50,000 pending URLs vs domain budget of 10,000 → ⚠ over budget", type: "check" },
        { actor: "Trap Detector", action: "Check 3: URL pattern /calendar/YYYY/MM/DD/ matches calendar trap pattern → ⚠ trap", type: "error" },
        { actor: "Trap Detector", action: "Verdict: TRAP. Add URL pattern '/calendar/*' to evil.com blacklist.", type: "error" },
        { actor: "Frontier", action: "Purge all pending URLs from evil.com matching /calendar/*. Removed 48,000 URLs.", type: "process" },
        { actor: "Alert", action: "Notify: crawler trap detected for evil.com. Pattern: /calendar/. Auto-blacklisted.", type: "process" },
      ]
    },
    retry: {
      title: "Transient Failure — Retry with Backoff",
      steps: [
        { actor: "Fetcher", action: "HTTP GET https://slow-site.com/data → timeout after 30s (attempt 1/3)", type: "error" },
        { actor: "Fetcher", action: "Record failure. Schedule retry in 60s (exponential backoff: 60, 120, 240).", type: "process" },
        { actor: "Fetcher", action: "Retry #2 → HTTP 503 Service Unavailable (attempt 2/3)", type: "error" },
        { actor: "Fetcher", action: "Schedule retry in 120s. Also increase domain politeness delay: 1s → 3s.", type: "process" },
        { actor: "Fetcher", action: "Retry #3 → HTTP 200 OK, 1.2s response time (attempt 3/3)", type: "success" },
        { actor: "Fetcher", action: "Success! Store content. Domain delay stays elevated (3s) for next 10 minutes.", type: "success" },
        { actor: "Metrics", action: "retry_success_total++. avg_response_time{domain='slow-site.com'} = 1200ms", type: "process" },
      ]
    },
  };
  const f = flows[flow];
  const typeColors = { request:"bg-blue-50 text-blue-700 border-blue-200", success:"bg-emerald-50 text-emerald-700 border-emerald-200", error:"bg-red-50 text-red-700 border-red-200", process:"bg-stone-50 text-stone-600 border-stone-200", auth:"bg-violet-50 text-violet-700 border-violet-200", check:"bg-amber-50 text-amber-700 border-amber-200" };
  return (
    <div className="space-y-5">
      <Card accent="#7e22ce">
        <div className="flex gap-2 mb-4">
          {Object.entries(flows).map(([k,v]) => (
            <button key={k} onClick={() => setFlow(k)}
              className={`px-3 py-1.5 rounded-lg text-[12px] font-medium border transition-all ${k===flow?"bg-purple-600 text-white border-purple-600":"bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
              {v.title.split("—")[0].trim()}
            </button>
          ))}
        </div>
        <Label color="#7e22ce">{f.title}</Label>
        <div className="space-y-1.5 mt-3">
          {f.steps.map((s,i) => (
            <div key={i} className={`flex items-start gap-3 px-4 py-2.5 rounded-lg border ${typeColors[s.type]}`}>
              <span className="text-[10px] font-mono font-bold w-5 shrink-0 mt-0.5">{i+1}</span>
              <span className="text-[11px] font-bold w-36 shrink-0">{s.actor}</span>
              <span className="text-[11px] flex-1">{s.action}</span>
            </div>
          ))}
        </div>
      </Card>
      <Card>
        <Label color="#7e22ce">Crawl State Machine</Label>
        <div className="grid grid-cols-5 gap-2 text-[11px] text-center">
          <div className="bg-stone-50 rounded-lg p-2 border border-stone-200 text-stone-600 font-bold">PENDING</div>
          <div className="bg-blue-50 rounded-lg p-2 border border-blue-200 text-blue-700 font-bold">FETCHING</div>
          <div className="bg-emerald-50 rounded-lg p-2 border border-emerald-200 text-emerald-700 font-bold">FETCHED</div>
          <div className="bg-red-50 rounded-lg p-2 border border-red-200 text-red-700 font-bold">FAILED</div>
          <div className="bg-amber-50 rounded-lg p-2 border border-amber-200 text-amber-700 font-bold">BLOCKED</div>
        </div>
        <div className="grid grid-cols-3 gap-2 mt-2 text-[10px]">
          <div className="bg-blue-50 rounded-lg p-2 text-center text-blue-700">PENDING → FETCHING: worker picks up URL</div>
          <div className="bg-emerald-50 rounded-lg p-2 text-center text-emerald-700">FETCHING → FETCHED: 2xx response received</div>
          <div className="bg-red-50 rounded-lg p-2 text-center text-red-700">FETCHING → FAILED: timeout/error after max retries</div>
          <div className="bg-amber-50 rounded-lg p-2 text-center text-amber-700">PENDING → BLOCKED: robots.txt disallow</div>
          <div className="bg-stone-50 rounded-lg p-2 text-center text-stone-600">FAILED → PENDING: re-crawl scheduler re-enqueues</div>
          <div className="bg-violet-50 rounded-lg p-2 text-center text-violet-700">FETCHING → PENDING: worker dies before ACK (lease timeout)</div>
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
          <Label color="#b45309">Kubernetes Deployment Topology</Label>
          <CodeBlock title="Fetcher Workers — Stateless, horizontally scalable" code={`apiVersion: apps/v1
kind: Deployment
metadata:
  name: crawl-fetcher
spec:
  replicas: 10               # Scale based on queue depth
  strategy:
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1       # Crawling can tolerate brief dip
  template:
    spec:
      containers:
      - name: fetcher
        image: crawler/fetcher:1.4
        env:
        - name: FRONTIER_HOST
          value: "frontier.crawl.svc:50051"
        - name: S3_BUCKET
          value: "crawl-content-prod"
        - name: KAFKA_BROKERS
          value: "kafka.data.svc:9092"
        - name: MAX_CONCURRENT
          value: "500"         # async connections
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "4"
            memory: "4Gi"
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
---
# HPA: scale on frontier queue depth
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fetcher-hpa
spec:
  scaleTargetRef:
    kind: Deployment
    name: crawl-fetcher
  minReplicas: 5
  maxReplicas: 50
  metrics:
  - type: External
    external:
      metric:
        name: frontier_queue_depth
      target:
        type: AverageValue
        averageValue: "100000"`} />
        </Card>
        <Card accent="#0f766e">
          <Label color="#0f766e">Multi-AZ Deployment Layout</Label>
          <svg viewBox="0 0 380 320" className="w-full">
            <rect x={5} y={5} width={370} height={310} rx={10} fill="#0f766e05" stroke="#0f766e30" strokeWidth={1} strokeDasharray="4,3"/>
            <text x={190} y={22} textAnchor="middle" fill="#0f766e" fontSize="10" fontWeight="700" fontFamily="monospace">us-east-1</text>

            {/* NLB */}
            <rect x={120} y={35} width={140} height={24} rx={4} fill="#2563eb12" stroke="#2563eb" strokeWidth={1}/>
            <text x={190} y={51} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Internal Load Balancer</text>

            {/* AZ-a */}
            <rect x={15} y={72} width={110} height={148} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={70} y={87} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-a</text>
            <rect x={22} y={96} width={96} height={24} rx={4} fill="#2563eb15" stroke="#2563eb" strokeWidth={1}/>
            <text x={70} y={112} textAnchor="middle" fill="#2563eb" fontSize="8" fontFamily="monospace">Fetcher ×3</text>
            <rect x={22} y={128} width={96} height={24} rx={4} fill="#05966915" stroke="#059669" strokeWidth={1}/>
            <text x={70} y={144} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">Processor ×2</text>
            <rect x={22} y={160} width={96} height={24} rx={4} fill="#9333ea15" stroke="#9333ea" strokeWidth={1}/>
            <text x={70} y={176} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">Frontier (Pri)</text>
            <rect x={22} y={192} width={96} height={20} rx={4} fill="#dc262615" stroke="#dc2626" strokeWidth={1}/>
            <text x={70} y={205} textAnchor="middle" fill="#dc2626" fontSize="7" fontFamily="monospace">Redis (Primary)</text>

            {/* AZ-b */}
            <rect x={135} y={72} width={110} height={148} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={190} y={87} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-b</text>
            <rect x={142} y={96} width={96} height={24} rx={4} fill="#2563eb15" stroke="#2563eb" strokeWidth={1}/>
            <text x={190} y={112} textAnchor="middle" fill="#2563eb" fontSize="8" fontFamily="monospace">Fetcher ×3</text>
            <rect x={142} y={128} width={96} height={24} rx={4} fill="#05966915" stroke="#059669" strokeWidth={1}/>
            <text x={190} y={144} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">Processor ×2</text>
            <rect x={142} y={160} width={96} height={24} rx={4} fill="#9333ea15" stroke="#9333ea" strokeWidth={1}/>
            <text x={190} y={176} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">Frontier (Rep)</text>
            <rect x={142} y={192} width={96} height={20} rx={4} fill="#0891b215" stroke="#0891b2" strokeWidth={1}/>
            <text x={190} y={205} textAnchor="middle" fill="#0891b2" fontSize="7" fontFamily="monospace">Redis (Replica)</text>

            {/* AZ-c */}
            <rect x={255} y={72} width={110} height={148} rx={6} fill="#6366f108" stroke="#6366f130" strokeWidth={1}/>
            <text x={310} y={87} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">AZ-c</text>
            <rect x={262} y={96} width={96} height={24} rx={4} fill="#2563eb15" stroke="#2563eb" strokeWidth={1}/>
            <text x={310} y={112} textAnchor="middle" fill="#2563eb" fontSize="8" fontFamily="monospace">Fetcher ×4</text>
            <rect x={262} y={128} width={96} height={24} rx={4} fill="#05966915" stroke="#059669" strokeWidth={1}/>
            <text x={310} y={144} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">Processor ×2</text>
            <rect x={262} y={160} width={96} height={24} rx={4} fill="#9333ea15" stroke="#9333ea" strokeWidth={1}/>
            <text x={310} y={176} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">Frontier (Rep)</text>
            <rect x={262} y={192} width={96} height={20} rx={4} fill="#0891b215" stroke="#0891b2" strokeWidth={1}/>
            <text x={310} y={205} textAnchor="middle" fill="#0891b2" fontSize="7" fontFamily="monospace">Redis (Replica)</text>

            {/* Legend */}
            <rect x={15} y={228} width={350} height={82} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
            <text x={190} y={245} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Deployment Rules</text>
            <text x={30} y={263} fill="#78716c" fontSize="8" fontFamily="monospace">• Fetchers: stateless, auto-scaled on queue depth via HPA</text>
            <text x={30} y={278} fill="#78716c" fontSize="8" fontFamily="monospace">• Frontier: 1 primary + 2 replicas. Sentinel for failover.</text>
            <text x={30} y={293} fill="#78716c" fontSize="8" fontFamily="monospace">• Processors: scale with Kafka partition count</text>
            <text x={30} y={308} fill="#78716c" fontSize="8" fontFamily="monospace">• AZ failure: workers auto-redistribute. Redis failover &lt;10s.</text>
          </svg>
        </Card>
      </div>

      <Card accent="#dc2626">
        <Label color="#dc2626">Security Considerations</Label>
        <div className="grid grid-cols-2 gap-5">
          <div>
            <div className="text-[10px] font-bold text-stone-600 uppercase tracking-wider mb-1.5">Crawl Safety</div>
            <ul className="space-y-1.5">
              <Point icon="🛡️" color="#dc2626">Always respect robots.txt — legal compliance and goodwill</Point>
              <Point icon="🛡️" color="#dc2626">Set honest User-Agent: "MyCrawler/1.0 (+https://mysite.com/bot)"</Point>
              <Point icon="🛡️" color="#dc2626">SSRF protection: block private IPs (10.x, 127.x, 169.254.x, 192.168.x) in DNS results</Point>
              <Point icon="🛡️" color="#dc2626">Max response size (10MB) to prevent memory exhaustion from malicious servers</Point>
              <Point icon="🛡️" color="#dc2626">TLS verification: validate certificates, don't follow self-signed certs by default</Point>
            </ul>
          </div>
          <div>
            <div className="text-[10px] font-bold text-stone-600 uppercase tracking-wider mb-1.5">Infrastructure Security</div>
            <ul className="space-y-1.5">
              <Point icon="🔒" color="#b45309">Crawl workers in private subnet — no inbound internet access</Point>
              <Point icon="🔒" color="#b45309">NAT Gateway for outbound — rotate IPs if needed to avoid IP-based blocking</Point>
              <Point icon="🔒" color="#b45309">S3 bucket: private, SSE-S3 encryption, lifecycle policies for archival</Point>
              <Point icon="🔒" color="#b45309">Redis: AUTH enabled, TLS in transit, VPC-only access</Point>
              <Point icon="🔒" color="#b45309">IAM roles per service — fetcher can write S3, processor can read S3, neither can delete</Point>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}

function OpsSection() {
  return (
    <div className="space-y-5">
      <Card accent="#be123c">
        <Label color="#be123c">Runbook — Common Incidents</Label>
        <div className="space-y-3">
          {[
            { incident: "Crawl throughput drops to 0", steps: "1. Check frontier service health (redis connectivity). 2. Verify fetcher pods are running (kubectl get pods). 3. Check DNS resolver health. 4. Look for network partition or AWS outage. 5. Check if Bloom filter service is reachable.", severity: "P1" },
            { incident: "Single domain consuming all resources", steps: "1. Identify domain from dashboard (top domains by queue size). 2. Check for crawler trap patterns. 3. Add domain page budget: UPDATE domains SET max_pages=1000 WHERE domain='evil.com'. 4. Purge excess URLs from frontier. 5. Add URL pattern to trap blacklist.", severity: "P2" },
            { incident: "Duplicate content rate >30%", steps: "1. Check Bloom filter fill ratio (should be <70%). 2. Verify SimHash dedup is running. 3. Check for URL normalization bugs (www vs non-www, trailing slashes). 4. Look for mirror sites flooding the queue. 5. Consider tightening SimHash threshold.", severity: "P2" },
            { incident: "Politeness violations reported", steps: "1. IMMEDIATELY throttle the offending workers. 2. Check per-domain rate tracking. 3. Verify robots.txt cache is fresh (not serving stale allows). 4. Check if domain partitioning is correct (same domain hitting multiple workers). 5. Send courtesy email to affected site admin.", severity: "P1" },
            { incident: "S3 write latency spike", steps: "1. Check AWS S3 service health dashboard. 2. Verify IAM permissions haven't changed. 3. Check if content size distribution changed (larger pages = slower writes). 4. Verify S3 bucket hasn't hit prefix throttling limits. 5. Consider adding S3 Transfer Acceleration.", severity: "P2" },
          ].map((r,i) => (
            <div key={i} className="border border-stone-200 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Pill bg={r.severity==="P1"?"#fef2f2":"#fffbeb"} color={r.severity==="P1"?"#dc2626":"#d97706"}>{r.severity}</Pill>
                <span className="text-[12px] font-bold text-stone-800">{r.incident}</span>
              </div>
              <div className="text-[11px] text-stone-500 leading-relaxed">{r.steps}</div>
            </div>
          ))}
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#be123c">Health Checks</Label>
          <div className="space-y-2">
            {[
              { check: "Frontier Redis ping", interval: "10s", action: "Alert if 3 consecutive failures" },
              { check: "Worker heartbeat", interval: "30s", action: "Restart pod if missed 2 beats" },
              { check: "Kafka consumer lag", interval: "1m", action: "Scale processors if lag > 100K" },
              { check: "Bloom filter fill ratio", interval: "5m", action: "Alert if > 70%, critical > 85%" },
              { check: "S3 write success rate", interval: "1m", action: "Alert if < 99.5%" },
              { check: "DNS resolution time", interval: "30s", action: "Failover resolver if p99 > 100ms" },
            ].map((h,i) => (
              <div key={i} className="flex items-center gap-3 text-[11px] border-b border-stone-100 pb-2">
                <span className="font-mono text-stone-700 w-40 shrink-0">{h.check}</span>
                <span className="text-stone-400 w-12 shrink-0">{h.interval}</span>
                <span className="text-stone-500 flex-1">{h.action}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <Label color="#be123c">Capacity Planning Checklist</Label>
          <ul className="space-y-2">
            <Point icon="📊" color="#be123c">Monitor frontier queue growth trend — if growing faster than draining, add workers</Point>
            <Point icon="💾" color="#be123c">S3 cost projection: current pages/month × avg page size × retention months</Point>
            <Point icon="🔄" color="#be123c">Bloom filter: plan resize when fill ratio > 50% (resize takes ~30 min for 10B capacity)</Point>
            <Point icon="📈" color="#be123c">Database: partition URLs table by domain hash when > 1B rows for query performance</Point>
            <Point icon="🌐" color="#be123c">Bandwidth: monitor egress costs. Consider caching via CDN for re-crawls if available.</Point>
            <Point icon="⏱️" color="#be123c">Re-crawl budget: ensure re-crawl rate doesn't consume > 30% of total crawl capacity</Point>
          </ul>
        </Card>
      </div>
    </div>
  );
}

function EnhancementsSection() {
  return (
    <div className="grid grid-cols-2 gap-5">
      {[
        { t: "JavaScript Rendering (Headless Browser)", d: "Many modern sites use SPAs (React, Vue) where content is rendered client-side. Without JS rendering, crawler misses most content.", detail: "Add a headless Chrome pool (Puppeteer/Playwright). Route URLs marked as 'JS-required' to the rendering pool. Much slower (~5s/page vs 0.3s) and expensive. Only render when needed.", effort: "Hard" },
        { t: "Incremental / Delta Crawling", d: "Instead of re-downloading entire pages, detect what changed. Use ETags and If-Modified-Since headers to get 304 Not Modified responses.", detail: "Send If-Modified-Since and If-None-Match headers. On 304, skip download entirely. Saves 30-50% bandwidth on re-crawls. Track change frequency per URL to optimize re-crawl schedule.", effort: "Medium" },
        { t: "Crawl Prioritization with ML", d: "Use machine learning to predict which URLs are most valuable to crawl next. PageRank is a start; modern crawlers use features like domain authority, content type, freshness signals.", detail: "Train a model on: domain authority, URL depth, path patterns, historical change frequency, content quality scores. Assign priority scores. Continuously retrain on new crawl data.", effort: "Hard" },
        { t: "Distributed Bloom Filter (Cuckoo Filter)", d: "Bloom filters don't support deletion. As URLs expire or are removed, wasted bits accumulate. Cuckoo Filters support deletion and have better space efficiency for low FP rates.", detail: "Replace Bloom filter with Cuckoo filter. Supports delete() operation for removing expired URLs. Slightly better memory at <3% FP rate. More complex implementation.", effort: "Medium" },
        { t: "DNS Pre-Fetching & Caching", d: "DNS resolution is a bottleneck at scale. Pre-resolve domains for queued URLs before the fetcher needs them.", detail: "Background thread resolves DNS for next N domains in the frontier. Cache with TTL awareness. Pool of resolvers across multiple providers. Reduces effective fetch latency by 20-50ms avg.", effort: "Easy" },
        { t: "Content Dedup via MinHash/LSH", d: "SimHash is good for near-duplicates but misses rearranged content. MinHash + Locality Sensitive Hashing (LSH) catches more types of similarity.", detail: "Generate MinHash signatures (128 hash functions). Use LSH bands to find candidate pairs efficiently. Compare candidate pairs with Jaccard similarity. More accurate but 2-3× more compute.", effort: "Hard" },
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
    { q:"How does Google handle the celebrity problem (sites with millions of pages)?", a:"Google uses a page budget per domain — a maximum number of pages it will crawl per site per day. High-authority sites (nytimes.com) get larger budgets. The budget is dynamic based on server responsiveness, content quality, and how often content changes. Low-quality or slow sites get smaller budgets.", tags:["scalability"] },
    { q:"How would you handle JavaScript-rendered pages?", a:"Add a headless Chrome rendering pool (Puppeteer/Playwright). Route URLs to this pool when: (1) initial HTML has minimal content, (2) domain is known SPA framework, (3) discovery heuristic detects JS rendering. It's 10-20× slower than raw HTTP, so only render when needed. Google has a separate rendering queue with lower priority.", tags:["design"] },
    { q:"How do you prevent the crawler from being blocked?", a:"Rotate IPs via a NAT gateway pool. Set an honest User-Agent with contact URL. Respect robots.txt and crawl delays. Spread requests across time. Some sites use CAPTCHAs or browser fingerprinting — for these, either skip or use headless rendering. Don't try to bypass blocks — that's adversarial.", tags:["operations"] },
    { q:"How would you prioritize which URLs to crawl first?", a:"Multi-factor scoring: (1) Domain authority (like PageRank), (2) URL depth (shallower = more important), (3) Content freshness (pages that change often get re-crawled sooner), (4) Inlink count (more pages link to it = more important), (5) Content type (articles > pagination pages). Can use ML for this at scale.", tags:["algorithm"] },
    { q:"How do you handle URL normalization edge cases?", a:"Normalize: lowercase hostname, remove default ports (80/443), remove fragments (#), sort query parameters alphabetically, decode unnecessary percent-encoding, remove trailing slashes, resolve '../' in paths. Special cases: www vs non-www (treat as same if they serve same content), http vs https (prefer https). This is harder than it sounds — many crawlers get this wrong.", tags:["design"] },
    { q:"What if the Bloom filter gives a false positive for an important URL?", a:"Accept it — a 1% FP rate means we miss 1% of URLs. For a web-scale crawler, this is fine. If you need zero misses for critical URLs (homepage, sitemaps), maintain a separate whitelist that bypasses the Bloom filter. You can also use a two-tier approach: Bloom filter first, then check the database for URLs near the 'maybe seen' threshold.", tags:["algorithm"] },
    { q:"How would you implement re-crawling / freshness?", a:"Track per-URL change frequency. Pages that change daily get re-crawled daily; pages that never change get re-crawled monthly. Use a change detection model: compare new SimHash to previous — if different, page changed. HTTP conditional requests (If-Modified-Since) save bandwidth. Allocate 20-30% of crawl capacity to re-crawls.", tags:["design"] },
    { q:"How is this different from Scrapy or Nutch?", a:"Scrapy/Nutch are single-machine or small-cluster frameworks. They handle the crawl loop but not web-scale distribution. Our design adds: (1) distributed URL frontier across many machines, (2) centralized dedup with Bloom filters for billions of URLs, (3) per-domain politeness at scale, (4) fault-tolerant checkpointing, (5) auto-scaling workers. It's the difference between a script and a distributed system.", tags:["design"] },
    { q:"How would you test a web crawler?", a:"Unit: URL normalizer with edge cases, robots.txt parser, SimHash accuracy. Integration: crawl a local test site with known link structure, verify all pages found. Load: seed with 10K real URLs, measure throughput and resource usage. Chaos: kill workers mid-crawl, verify no URL loss. Politeness: verify per-domain rate limits with traffic capture. Trap: create a test trap site, verify crawler escapes.", tags:["testing"] },
    { q:"What about sitemap.xml?", a:"Sitemaps are a gift — they tell you every URL on the site plus change frequency and priority. Fetch sitemap.xml (and sitemap index files) for every domain. Parse and inject URLs into frontier with sitemap-provided priority. Re-fetch sitemaps periodically. But don't rely solely on sitemaps — many sites have incomplete or missing sitemaps.", tags:["design"] },
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

export default function WebCrawlerSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Web Crawler</h1>
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