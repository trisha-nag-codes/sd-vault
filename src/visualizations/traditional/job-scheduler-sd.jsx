import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   JOB SCHEDULER — System Design Reference
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
  { id: "services",      label: "Service Architecture",  icon: "🧩", color: "#0f766e" },
  { id: "flows",         label: "Request Flows",         icon: "🔀", color: "#7e22ce" },
  { id: "deployment",    label: "Deploy & Security",     icon: "🔒", color: "#b45309" },
  { id: "ops",           label: "Ops Playbook",          icon: "🔧", color: "#be123c" },
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

function CodeBlock({ title, code, highlight = [] }) {
  const lines = code.split("\n");
  return (
    <div className="bg-stone-50 border border-stone-200 rounded-lg p-3.5 overflow-x-auto">
      {title && <div className="text-[10px] font-bold text-stone-400 uppercase tracking-[0.1em] mb-2">{title}</div>}
      <pre className="font-mono text-[11.5px] leading-[1.75]" style={{ whiteSpace: "pre" }}>
        {lines.map((line, i) => (
          <div key={i} className={`px-2 rounded ${highlight.includes(i) ? "bg-indigo-50 text-indigo-700" : line.trim().startsWith("#") || line.trim().startsWith("--") || line.trim().startsWith("//") ? "text-stone-400" : "text-stone-700"}`}>
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
            <Label>What is a Job Scheduler?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A distributed job scheduler is a system that accepts, queues, schedules, and reliably executes tasks at a specified time or on a recurring cadence. It's the "cron for the cloud" — but with at-least-once guarantees, horizontal scalability, and visibility into every execution. It decouples <em>what</em> needs to happen from <em>when</em> it happens.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Think of it like an airport control tower: flights (jobs) arrive with scheduled departure times, the tower (scheduler) assigns runways (workers) based on priority and availability, monitors each takeoff (execution), and handles delays, cancellations, and rerouting — all while ensuring no two planes share a runway and every flight eventually departs.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Core Job Types</Label>
            <ul className="space-y-2.5">
              <Point icon="⏰" color="#0891b2"><strong className="text-stone-700">One-time (delayed)</strong> — execute once at a specific future time. Examples: send email in 30 min, expire trial in 14 days, process refund at 5pm.</Point>
              <Point icon="🔄" color="#0891b2"><strong className="text-stone-700">Recurring (cron)</strong> — execute on a schedule: every 5 minutes, daily at midnight, first Monday of month. Examples: generate reports, sync data, billing cycles.</Point>
              <Point icon="📡" color="#0891b2"><strong className="text-stone-700">Event-driven</strong> — triggered by an external event, then scheduled. Examples: user signs up → send welcome email in 1 hour, payment captured → settle in 24h.</Point>
              <Point icon="🔗" color="#0891b2"><strong className="text-stone-700">DAG / Workflow</strong> — jobs with dependencies: Job B runs only after Job A completes. Examples: ETL pipelines, ML training (extract → transform → train → deploy).</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Examples</Label>
            <div className="space-y-2.5">
              {[
                { co: "Airflow", rule: "DAG-based workflow scheduler", algo: "Python DAGs, operator pattern" },
                { co: "Celery Beat", rule: "Periodic task scheduler for Celery", algo: "DB/file-backed cron" },
                { co: "Quartz", rule: "Enterprise Java scheduler", algo: "Cron triggers, clustering" },
                { co: "Temporal", rule: "Durable workflow engine", algo: "Event sourcing, replay" },
                { co: "AWS Step Fn", rule: "Serverless state machine", algo: "JSON state language" },
                { co: "Sidekiq", rule: "Ruby background jobs", algo: "Redis-backed queues" },
                { co: "Uber Cadence", rule: "Fault-tolerant stateful workflows", algo: "Decision + activity tasks" },
              ].map((e,i) => (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-24 shrink-0">{e.co}</span>
                  <span className="text-stone-500 flex-1">{e.rule}</span>
                  <span className="text-stone-400 text-[10px]">{e.algo}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <Label color="#2563eb">Architecture Overview</Label>
            <svg viewBox="0 0 360 180" className="w-full">
              <DiagramBox x={60} y={35} w={80} h={32} label="Client" color="#2563eb"/>
              <DiagramBox x={180} y={35} w={80} h={32} label="API + Store" color="#6366f1"/>
              <DiagramBox x={300} y={35} w={80} h={32} label="Scheduler" color="#9333ea"/>
              <DiagramBox x={180} y={105} w={80} h={32} label="Queue" color="#d97706"/>
              <DiagramBox x={60} y={105} w={80} h={32} label="Worker 1" color="#059669"/>
              <DiagramBox x={300} y={105} w={80} h={32} label="Worker N" color="#059669"/>
              <Arrow x1={100} y1={35} x2={140} y2={35} label="submit" id="c1"/>
              <Arrow x1={220} y1={35} x2={260} y2={35} label="poll" id="c2"/>
              <Arrow x1={300} y1={51} x2={220} y2={89} label="enqueue" id="c3"/>
              <Arrow x1={140} y1={105} x2={100} y2={105} label="dequeue" id="c4"/>
              <Arrow x1={260} y1={105} x2={240} y2={105} label="dequeue" id="c5"/>
              <rect x={80} y={150} width={200} height={18} rx={4} fill="#05966908" stroke="#05966930"/>
              <text x={180} y={160} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">Workers pull jobs → execute → report result</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Amazon, Google, Uber, Stripe, LinkedIn, Airbnb</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Narrow the Scope Fast</div>
            <p className="text-[12px] text-stone-500 mt-0.5">"Design a job scheduler" is deceptively broad — it can mean cron replacement, task queue, workflow engine, or all three. Clarify immediately: (1) One-time delayed jobs, recurring cron, or DAG workflows? (2) At-least-once or exactly-once? (3) What's the execution duration — seconds or hours? For a 45-min interview, focus on <strong>delayed + recurring job scheduling with at-least-once execution guarantees</strong>. DAGs and exactly-once are follow-ups.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Submit jobs — one-time (execute at time T) or recurring (cron expression)</Point>
            <Point icon="2." color="#059669">Reliable execution — every job runs at least once, even after crashes</Point>
            <Point icon="3." color="#059669">Priority support — higher-priority jobs scheduled before lower-priority</Point>
            <Point icon="4." color="#059669">Retry with backoff — failed jobs retried with configurable max attempts</Point>
            <Point icon="5." color="#059669">Job lifecycle — query status (pending, running, succeeded, failed, cancelled)</Point>
            <Point icon="6." color="#059669">Cancel / pause — cancel a pending job or pause a recurring schedule</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">At-least-once delivery — no job silently dropped, even on node failure</Point>
            <Point icon="2." color="#dc2626">Scalable — handle millions of scheduled jobs with thousands of workers</Point>
            <Point icon="3." color="#dc2626">Low scheduling latency — job dispatched within seconds of its scheduled time</Point>
            <Point icon="4." color="#dc2626">High availability — scheduler survives node failures without job loss</Point>
            <Point icon="5." color="#dc2626">Exactly-once semantics (stretch) — idempotent execution for critical jobs</Point>
            <Point icon="6." color="#dc2626">Fairness — one tenant's job flood doesn't starve other tenants</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask the Interviewer</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "One-time delayed jobs, recurring (cron), or both?",
            "What execution duration? Seconds (API calls) or hours (ETL)?",
            "At-least-once or exactly-once delivery guarantees?",
            "How precise? Second-level accuracy or minute-level OK?",
            "Multi-tenant? Fairness/isolation between users?",
            "Do jobs have dependencies (DAG) or are they independent?",
            "Expected scale? Thousands or millions of jobs per day?",
            "What happens on failure? Retry, dead-letter, alert?",
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
            <div className="text-[12px] font-bold text-violet-700">Interview Tip — Show the Bottleneck</div>
            <p className="text-[12px] text-stone-500 mt-0.5">For a job scheduler, the bottleneck is not storage or bandwidth — it's <strong>scheduling throughput</strong> (how many jobs/sec the scheduler can poll, sort, and dispatch) and <strong>worker concurrency</strong> (how many jobs execute in parallel). Show the interviewer you understand that the scheduler is the choke point.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Job Volume</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Total scheduled jobs" result="100M" note="Active jobs in the system at any time" />
            <MathStep step="2" formula="New jobs submitted per day" result="10M/day" note="Mix of one-time and recurring" />
            <MathStep step="3" formula="Jobs due per second (avg)" result="~1,200/s" note="100M jobs, avg fire interval ~24h" />
            <MathStep step="4" formula="Peak job dispatch rate" result="~10,000/s" note="Morning burst: crons + batch jobs at midnight/9am" />
            <MathStep step="5" formula="Recurring jobs (generate next run)" result="~30M" note="30% of jobs are recurring (cron-based)" final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Worker Capacity</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Avg job execution duration" result="~5s" note="Mix of fast (100ms API call) and slow (30s report)" />
            <MathStep step="2" formula="Worker concurrency per node" result="~50" note="50 goroutines/threads per worker node" />
            <MathStep step="3" formula="Jobs/sec per worker node" result="~10" note="50 slots / 5s avg duration" />
            <MathStep step="4" formula="Workers for peak load (10K/s)" result="~1,000 nodes" note="10K / 10 jobs-per-node = 1,000" />
            <MathStep step="5" formula="With 2× headroom" result="~2,000 nodes" note="Buffer for retries and burst" final />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Storage</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Bytes per job record" result="~1 KB" note="job_id, schedule, payload, status, metadata" />
            <MathStep step="2" formula="Active jobs storage = 100M × 1 KB" result="~100 GB" note="Easily fits in a sharded database" />
            <MathStep step="3" formula="Execution history per job (avg 5)" result="~0.3 KB each" note="run_id, start, end, status, error" />
            <MathStep step="4" formula="Daily execution log = 10M × 5 × 0.3 KB" result="~15 GB/day" note="Grows fast — archive after 30 days" />
            <MathStep step="5" formula="30-day hot storage" result="~550 GB" note="100 GB jobs + 450 GB exec history" final />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Scheduling Throughput</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Scheduler poll interval" result="1s" note="Check for due jobs every second" />
            <MathStep step="2" formula="Jobs fetched per poll" result="~1,000" note="Batch fetch: WHERE fire_time &le; NOW LIMIT 1000" />
            <MathStep step="3" formula="Scheduler instances (partitioned)" result="~10" note="Each owns a partition of the time range or job space" />
            <MathStep step="4" formula="Total scheduling throughput" result="~10,000/s" note="10 schedulers × 1,000/s each" />
            <MathStep step="5" formula="Scheduling latency target" result="&lt; 5s" note="From fire_time to job dispatched to worker" final />
          </div>
        </Card>
      </div>
      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Peak Dispatch", val: "~10K/s", sub: "Avg: ~1.2K/s" },
            { label: "Active Jobs", val: "100M", sub: "30M recurring" },
            { label: "Hot Storage", val: "~550 GB", sub: "30-day retention" },
            { label: "Worker Nodes", val: "~2,000", sub: "50 slots each" },
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
          <Label color="#2563eb">Core Scheduler API</Label>
          <CodeBlock code={`# POST /v1/jobs — Create a one-time or recurring job
{
  "name": "send-welcome-email",
  "type": "ONE_TIME",               # ONE_TIME | RECURRING
  "schedule_at": "2025-06-01T09:00:00Z",  # for ONE_TIME
  "cron_expr": null,                 # "*/5 * * * *" for RECURRING
  "payload": {
    "user_id": "usr_abc",
    "template": "welcome"
  },
  "callback_url": "https://api.example.com/hooks/jobs",
  "priority": 5,                     # 0 (lowest) to 10 (highest)
  "max_retries": 3,
  "retry_backoff": "exponential",    # fixed | linear | exponential
  "timeout_seconds": 30,
  "idempotency_key": "welcome-usr_abc",
  "queue": "email"                   # logical queue / topic
}
# Response:
{
  "id": "job_7kx2m...",
  "status": "SCHEDULED",
  "next_fire_time": "2025-06-01T09:00:00Z",
  "created_at": "2025-05-30T14:22:00Z"
}

# GET /v1/jobs/:id — Get job status + execution history
# DELETE /v1/jobs/:id — Cancel a pending job
# PATCH /v1/jobs/:id — Update schedule / pause / resume
# GET /v1/jobs?queue=email&status=FAILED — List / filter`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Key API Endpoints</Label>
          <div className="space-y-3">
            {[
              { op: "POST /v1/jobs", desc: "Submit a new job — one-time delayed or recurring (cron). Returns job ID and next fire time.", perf: "~50ms" },
              { op: "GET /v1/jobs/:id", desc: "Retrieve job status, schedule info, and last N execution results.", perf: "~20ms" },
              { op: "DELETE /v1/jobs/:id", desc: "Cancel a scheduled job. Running jobs can be force-killed or allowed to complete.", perf: "~30ms" },
              { op: "PATCH /v1/jobs/:id", desc: "Update schedule, pause recurring job, change priority, or modify payload.", perf: "~40ms" },
              { op: "GET /v1/jobs", desc: "List jobs with filters: queue, status, priority, time range. Paginated.", perf: "~50ms" },
              { op: "POST /v1/jobs/:id/trigger", desc: "Manually trigger a job immediately, regardless of schedule. For debugging / ops.", perf: "~100ms" },
            ].map((h,i) => (
              <div key={i} className="flex items-start gap-3">
                <code className="text-[11px] font-mono font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded shrink-0">{h.op}</code>
                <div>
                  <div className="text-[12px] text-stone-600">{h.desc}</div>
                  <div className="text-[10px] text-stone-400 font-mono">{h.perf}</div>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-stone-100">
            <Label color="#d97706">Critical Design Decisions</Label>
            <ul className="space-y-1.5">
              <Point icon="→" color="#d97706">Idempotency key prevents duplicate job creation on retries</Point>
              <Point icon="→" color="#d97706">Callback URL (webhook) for push-based result notification</Point>
              <Point icon="→" color="#d97706">Logical queues isolate job types — "email", "report", "billing"</Point>
              <Point icon="→" color="#d97706">Priority + queue combination enables fair, ordered scheduling</Point>
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
    { label: "Polling DB", desc: "Simplest approach: a single scheduler polls the database every N seconds for due jobs (WHERE fire_time ≤ NOW AND status = 'SCHEDULED'), claims them with an UPDATE, and dispatches to workers. Easy to implement but creates hot-spots on the DB and doesn't scale past ~1K jobs/s." },
    { label: "Delay Queue", desc: "Push due jobs into a delay queue (Redis sorted set, SQS with delay, or Kafka with timestamp). Scheduler writes jobs at submit time; the queue makes them visible at fire_time. Workers pull from the queue. Eliminates DB polling, but sorted sets have O(log N) insert and the queue becomes the bottleneck." },
    { label: "Time-Partitioned ★", desc: "Partition the time axis into buckets (e.g., 1-minute windows). Each scheduler instance owns a set of buckets. When a bucket's time arrives, the owning scheduler fetches all jobs in that bucket and enqueues them. Scales horizontally by adding scheduler instances. This is how most production systems work." },
    { label: "Hierarchical (2-Level)", desc: "Two levels: a lightweight Timing Wheel (in-memory) for near-future jobs (next 5 min), backed by a durable store for everything else. A 'loader' periodically promotes jobs from the store into the wheel. Sub-second scheduling precision with massive scale. Used by Uber Cherami and LinkedIn's scheduler." },
  ];
  const diagrams = [
    () => (
      <svg viewBox="0 0 460 130" className="w-full">
        <DiagramBox x={80} y={50} w={90} h={36} label="Scheduler" color="#9333ea" sub="polls every 1s"/>
        <DiagramBox x={230} y={50} w={90} h={36} label="Database" color="#dc2626" sub="jobs table"/>
        <DiagramBox x={380} y={50} w={80} h={36} label="Workers" color="#059669"/>
        <Arrow x1={125} y1={42} x2={185} y2={42} label="SELECT due" id="d1"/>
        <Arrow x1={185} y1={58} x2={125} y2={58} label="rows" id="d2"/>
        <Arrow x1={275} y1={50} x2={340} y2={50} label="dispatch" id="d3" dashed/>
        <rect x={110} y={95} width={230} height={18} rx={4} fill="#dc262608" stroke="#dc262630"/>
        <text x={225} y={105} textAnchor="middle" fill="#dc2626" fontSize="8" fontFamily="monospace">⚠ DB hot-spot on fire_time index, single scheduler bottleneck</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 130" className="w-full">
        <DiagramBox x={60} y={50} w={80} h={36} label="Client" color="#2563eb"/>
        <DiagramBox x={180} y={50} w={80} h={36} label="API" color="#6366f1"/>
        <DiagramBox x={300} y={50} w={90} h={40} label="Delay\nQueue" color="#d97706" sub="Redis ZSET"/>
        <DiagramBox x={420} y={50} w={70} h={36} label="Workers" color="#059669"/>
        <Arrow x1={100} y1={50} x2={140} y2={50} label="submit" id="q1"/>
        <Arrow x1={220} y1={50} x2={255} y2={50} label="ZADD" id="q2"/>
        <Arrow x1={345} y1={50} x2={385} y2={50} label="ZPOP" id="q3"/>
        <rect x={120} y={95} width={220} height={18} rx={4} fill="#05966908" stroke="#05966930"/>
        <text x={230} y={105} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">✓ No polling. Queue reveals jobs at fire_time</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 160" className="w-full">
        <DiagramBox x={60} y={40} w={75} h={30} label="API" color="#6366f1"/>
        <DiagramBox x={60} y={100} w={75} h={30} label="DB" color="#dc2626"/>
        <DiagramBox x={180} y={40} w={85} h={34} label="Scheduler\n0 (T0-T5)" color="#9333ea"/>
        <DiagramBox x={300} y={40} w={85} h={34} label="Scheduler\n1 (T5-T10)" color="#9333ea"/>
        <DiagramBox x={420} y={40} w={70} h={30} label="Queue" color="#d97706"/>
        <DiagramBox x={420} y={100} w={70} h={30} label="Workers" color="#059669"/>
        <Arrow x1={60} y1={55} x2={60} y2={85} label="write" id="tp1"/>
        <Arrow x1={137} y1={40} x2={137} y2={75} label="" id="tp2" dashed/>
        <Arrow x1={257} y1={40} x2={257} y2={75} label="" id="tp3" dashed/>
        <Arrow x1={97} y1={100} x2={148} y2={55} label="fetch bucket" id="tp4"/>
        <Arrow x1={222} y1={40} x2={385} y2={40} label="enqueue" id="tp5"/>
        <Arrow x1={420} y1={55} x2={420} y2={85} label="pull" id="tp6"/>
        <rect x={100} y={135} width={260} height={18} rx={4} fill="#9333ea08" stroke="#9333ea30"/>
        <text x={230} y={145} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">★ Each scheduler owns time partitions. Scales horizontally.</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 160" className="w-full">
        <DiagramBox x={80} y={35} w={95} h={34} label="Timing Wheel\n(in-memory)" color="#c026d3" sub="next 5 min"/>
        <DiagramBox x={250} y={35} w={95} h={34} label="Durable Store\n(DB)" color="#dc2626" sub="all future jobs"/>
        <DiagramBox x={400} y={35} w={80} h={30} label="Workers" color="#059669"/>
        <DiagramBox x={165} y={105} w={80} h={30} label="Loader" color="#6366f1" sub="promotes jobs"/>
        <Arrow x1={127} y1={35} x2={202} y2={35} label="" id="h1" dashed/>
        <Arrow x1={165} y1={90} x2={105} y2={55} label="load near" id="h2"/>
        <Arrow x1={225} y1={105} x2={270} y2={55} label="fetch" id="h3" dashed/>
        <Arrow x1={127} y1={42} x2={360} y2={42} label="dispatch" id="h4"/>
        <rect x={80} y={135} width={280} height={18} rx={4} fill="#c026d308" stroke="#c026d330"/>
        <text x={220} y={145} textAnchor="middle" fill="#c026d3" fontSize="8" fontFamily="monospace">Sub-second precision. Wheel + DB = best of both worlds.</text>
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
        <div className="bg-stone-50 rounded-lg border border-stone-200 p-3" style={{ minHeight: 140 }}>
          {diagrams[phase]()}
        </div>
      </Card>
      <Card>
        <Label color="#c026d3">Scheduling Strategy Comparison</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { t: "DB Polling", d: "SELECT due jobs every N seconds. Simple to implement with any relational database. Single scheduler.", pros: ["Simple, easy to debug","Existing DB infrastructure","ACID guarantees"], cons: ["DB hot-spot on fire_time index","Single scheduler = SPOF","High query load at scale"], pick: false },
            { t: "Delay Queue (Redis ZSET)", d: "Insert job with score=fire_time. Pop jobs where score ≤ now. Near-zero scheduling latency.", pros: ["O(log N) insert, O(1) pop","No polling needed","Sub-second precision"], cons: ["Redis memory limits","Persistence concerns (AOF/RDB)","Single-threaded bottleneck"], pick: false },
            { t: "Time-Partitioned ★", d: "Partition time into buckets. Each scheduler instance owns buckets. Fetch all jobs when bucket time arrives.", pros: ["Scales horizontally","No hot-spots (partitioned)","Natural load balancing","Production-proven"], cons: ["Rebalancing on scheduler join/leave","Bucket granularity limits precision","More complex coordination"], pick: true },
            { t: "Hierarchical (Timing Wheel)", d: "In-memory wheel for near-future, DB for far-future. Loader promotes. Best precision and throughput.", pros: ["Sub-ms scheduling precision","Massive throughput (100K+/s)","Memory-efficient (wheel)"], cons: ["Complex recovery on crash","In-memory state = risk","Two-tier coordination"], pick: false },
          ].map((o,i) => (
            <div key={i} className={`rounded-lg border p-3.5 ${o.pick ? "border-purple-300 bg-purple-50/50" : "border-stone-200"}`}>
              <div className={`text-[11px] font-bold mb-1.5 ${o.pick ? "text-purple-700" : "text-stone-600"}`}>{o.t}</div>
              <p className="text-[10px] text-stone-500 mb-2">{o.d}</p>
              <div className="space-y-0.5 text-[10px]">
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

function AlgorithmSection() {
  const [sel, setSel] = useState("timingWheel");
  const algos = {
    timingWheel: { name: "Timing Wheel ★", cx: "O(1) insert/fire",
      pros: ["O(1) for both inserting and firing jobs — constant time regardless of job count","Memory efficient: array of slots, each slot is a linked list of jobs","Used in production: Linux kernel timers, Netty HashedWheelTimer, Kafka","Hierarchical wheels handle both near and far-future events efficiently"],
      cons: ["Precision limited by tick interval (e.g., 100ms ticks = 100ms precision)","Jobs far in the future waste slots or require overflow/hierarchical wheel","In-memory — must persist to survive crashes (combine with DB)","Single-machine: needs partitioning for distributed use"],
      when: "Best for high-throughput scheduling where you need sub-second precision and O(1) operations. The wheel handles near-future jobs (next few minutes) while a durable store handles far-future jobs. A 'loader' thread periodically promotes jobs from the store into the wheel. This is the standard approach for production schedulers at Uber, LinkedIn, and Kafka's delayed message feature.",
      code: `# Timing Wheel — O(1) insert and fire
# Circular array of slots, each slot = linked list of jobs

class TimingWheel:
    def __init__(self, tick_ms=100, wheel_size=600):
        # 600 slots × 100ms = 60 seconds per rotation
        self.tick_ms = tick_ms
        self.wheel_size = wheel_size
        self.slots = [[] for _ in range(wheel_size)]
        self.current_tick = 0

    def insert(self, job, delay_ms):
        ticks = delay_ms // self.tick_ms
        if ticks >= self.wheel_size:
            # Overflow: store in DB, loader will promote later
            store_in_db(job, fire_time=now() + delay_ms)
            return
        slot_idx = (self.current_tick + ticks) % self.wheel_size
        self.slots[slot_idx].append(job)  # O(1)

    def advance(self):
        # Called every tick_ms by a timer thread
        self.current_tick = (self.current_tick + 1) % self.wheel_size
        due_jobs = self.slots[self.current_tick]
        self.slots[self.current_tick] = []  # Clear slot
        for job in due_jobs:
            dispatch_to_worker(job)  # O(1) per job

# Loader: promotes far-future jobs into wheel
def loader_loop():
    while True:
        near_jobs = db.query(
            "SELECT * FROM jobs WHERE fire_time <= ? "
            "AND fire_time > ? AND status = 'SCHEDULED'",
            now() + 60s, now()  # Next 60 seconds
        )
        for job in near_jobs:
            wheel.insert(job, job.fire_time - now())
            db.update(job.id, status='LOADED')
        sleep(10)  # Reload every 10 seconds` },
    cronParser: { name: "Cron Expression Parser", cx: "O(1) next fire",
      pros: ["Standard, well-understood format (minute hour day month weekday)","Efficiently computes next fire time without enumeration","Extended syntax: @every 5m, @daily, second-level precision","Libraries available in every language (croniter, robfig/cron)"],
      cons: ["Complex expressions can be confusing (day-of-month vs day-of-week)","Timezone handling is tricky (DST transitions)","No support for irregular schedules (every 3rd business day)"],
      when: "Every recurring job needs a cron expression parser. When a recurring job completes, the parser computes the next fire time and the scheduler creates the next execution. The parser is called once per execution — O(1) per call. Key edge case: what happens when cron says 'every day at 2:30 AM' but DST skips 2:00-3:00 AM? Your parser must handle this.",
      code: `# Cron Expression Parser — Compute next fire time
# Format: minute hour day_of_month month day_of_week
# Example: "30 9 * * 1-5" = 9:30 AM every weekday

class CronExpr:
    def __init__(self, expr):
        # "*/5 * * * *" → every 5 minutes
        parts = expr.split()
        self.minute  = parse_field(parts[0], 0, 59)
        self.hour    = parse_field(parts[1], 0, 23)
        self.dom     = parse_field(parts[2], 1, 31)
        self.month   = parse_field(parts[3], 1, 12)
        self.dow     = parse_field(parts[4], 0, 6)

    def next_fire_time(self, after):
        """Find next time matching all fields after 'after'"""
        t = after + 1_minute  # Start from next minute
        for _ in range(366 * 24 * 60):  # Max 1 year search
            if (t.minute in self.minute and
                t.hour in self.hour and
                t.month in self.month and
                (t.day in self.dom or t.weekday in self.dow)):
                return t
            t += 1_minute
        raise NoMatchError("No fire time in next year")

# On job completion, schedule next occurrence:
def on_recurring_job_complete(job):
    cron = CronExpr(job.cron_expr)
    next_time = cron.next_fire_time(after=now())
    db.update(job.id,
              next_fire_time=next_time,
              status='SCHEDULED')

# DST edge case: if 2:30 AM doesn't exist (spring forward),
# fire at 3:00 AM. If 1:30 AM happens twice (fall back),
# fire only on the first occurrence.` },
    leaderElection: { name: "Leader Election (Scheduler HA)", cx: "O(1) heartbeat",
      pros: ["Ensures exactly one scheduler processes each partition — prevents duplicate dispatch","Automatic failover: if leader dies, another instance takes over in seconds","Well-understood pattern with mature libraries (Zookeeper, etcd, Redis Redlock)","Combined with partitioning: each partition has its own leader"],
      cons: ["Split-brain risk if lock expires while leader is still processing (GC pause)","Extra infrastructure dependency (Zookeeper/etcd)","Fencing tokens needed to prevent stale leaders from acting","Adds latency to failover (heartbeat timeout + election)"],
      when: "In a distributed scheduler, multiple instances must coordinate to avoid double-dispatching jobs. Leader election ensures that for each time partition, exactly one scheduler instance is responsible. If that instance crashes, another takes over. Use etcd/Zookeeper lease-based locks with fencing tokens. The key insight for interviews: leader election + partitioning = scalable, fault-tolerant scheduling.",
      code: `# Leader Election with Fencing Tokens
# Each scheduler competes for partition ownership

class SchedulerNode:
    def __init__(self, node_id, partitions):
        self.node_id = node_id
        self.owned = []

    def acquire_partition(self, partition_id):
        # Try to acquire lease in etcd/Zookeeper
        lease = etcd.create_lease(ttl=30)  # 30s TTL
        success = etcd.put_if_absent(
            key=f"/scheduler/partitions/{partition_id}",
            value=self.node_id,
            lease=lease
        )
        if success:
            self.owned.append(partition_id)
            # Start heartbeat to keep lease alive
            self.heartbeat(lease, partition_id)
        return success

    def heartbeat(self, lease, partition_id):
        while True:
            try:
                lease.refresh()  # Extend TTL
                sleep(10)        # Refresh every 10s (TTL=30s)
            except LeaseExpired:
                # Lost ownership! Stop processing this partition
                self.owned.remove(partition_id)
                return

    def process_partition(self, partition_id):
        fence_token = etcd.get_fence_token(partition_id)
        jobs = db.query(
            "SELECT * FROM jobs "
            "WHERE partition = ? AND fire_time <= NOW() "
            "AND status = 'SCHEDULED'",
            partition_id
        )
        for job in jobs:
            # Fence token prevents stale leader from dispatching
            db.update(job.id, status='DISPATCHED',
                      fence_token=fence_token)
            queue.enqueue(job)` },
    priorityQueue: { name: "Priority Scheduling", cx: "O(log N) insert/extract",
      pros: ["Higher-priority jobs execute first — critical for SLAs","Starvation prevention with aging: increase priority over time","Per-queue priority: 'billing' queue always before 'analytics'","Fair scheduling: weighted fair queuing across tenants"],
      cons: ["Priority inversion: low-priority job holds resource needed by high-priority","O(log N) vs O(1) for non-priority queue","Starvation: low-priority jobs may never run without aging","Complex: multiple priority dimensions (urgency, queue, tenant)"],
      when: "When jobs have different importance levels: billing must run before analytics reports. In interviews, show you understand: (1) priority queue (min-heap by fire_time, tie-break by priority), (2) starvation prevention (age-based priority boost), (3) per-queue weights (billing gets 60% of workers, analytics 20%, email 20%). Weighted fair queuing is the production-grade approach used by YARN and Kubernetes.",
      code: `# Priority-Based Job Scheduling
# Combines fire_time with priority for dispatch ordering

import heapq

class PriorityScheduler:
    def __init__(self):
        # Min-heap: (effective_priority, fire_time, job)
        self.heap = []

    def schedule(self, job):
        # Effective priority: lower number = higher priority
        # Priority 10 (highest) → effective = 0
        effective = (10 - job.priority) * 1000 + job.fire_time
        heapq.heappush(self.heap, (effective, job.fire_time, job))

    def next_due(self):
        # Returns highest-priority due job
        while self.heap:
            eff, fire_time, job = self.heap[0]
            if fire_time <= now():
                heapq.heappop(self.heap)
                return job
            break
        return None

# Starvation prevention — age boost
def boost_starving_jobs():
    """Run every 60 seconds"""
    starving = db.query(
        "SELECT * FROM jobs WHERE status = 'SCHEDULED' "
        "AND fire_time < NOW() - INTERVAL '5 minutes'"
    )
    for job in starving:
        job.priority = min(job.priority + 1, 10)
        db.update(job.id, priority=job.priority)
        log.warn(f"Boosted starving job {job.id} to priority {job.priority}")

# Weighted Fair Queue (WFQ)
QUEUE_WEIGHTS = {"billing": 60, "email": 20, "analytics": 20}
def select_next_queue():
    # Deficit round-robin across queues
    for queue in sorted(queues, key=lambda q: q.deficit, reverse=True):
        if queue.has_due_jobs():
            queue.deficit -= 1
            return queue
    # Replenish deficits
    for q in queues:
        q.deficit += QUEUE_WEIGHTS[q.name]` },
  };
  const a = algos[sel];
  return (
    <div className="space-y-5">
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
          <Label color="#dc2626">Core Schema</Label>
          <CodeBlock code={`-- jobs (the unit of scheduled work)
CREATE TABLE jobs (
  id              VARCHAR(26) PRIMARY KEY,  -- job_xxxx (ULID)
  tenant_id       VARCHAR(26) NOT NULL,
  name            VARCHAR(255) NOT NULL,
  type            VARCHAR(10) NOT NULL,     -- ONE_TIME / RECURRING
  status          VARCHAR(20) NOT NULL,     -- SCHEDULED, RUNNING, etc.
  queue           VARCHAR(50) DEFAULT 'default',
  priority        SMALLINT DEFAULT 5,       -- 0-10
  payload         JSONB NOT NULL,           -- arbitrary job data
  callback_url    VARCHAR(2048),
  cron_expr       VARCHAR(100),             -- null for ONE_TIME
  schedule_at     TIMESTAMPTZ,              -- null for RECURRING
  next_fire_time  TIMESTAMPTZ NOT NULL,     -- ★ indexed for scheduling
  timeout_seconds INT DEFAULT 300,
  max_retries     SMALLINT DEFAULT 3,
  retry_count     SMALLINT DEFAULT 0,
  idempotency_key VARCHAR(255),
  partition_key   INT NOT NULL,             -- ★ for scheduler partitioning
  created_at      TIMESTAMPTZ NOT NULL,
  updated_at      TIMESTAMPTZ NOT NULL
);
-- ★ Critical index for scheduler polling:
CREATE INDEX idx_jobs_schedule ON jobs
  (partition_key, next_fire_time)
  WHERE status = 'SCHEDULED';

-- job_executions (one row per run attempt)
CREATE TABLE job_executions (
  id              BIGSERIAL PRIMARY KEY,
  job_id          VARCHAR(26) NOT NULL,
  run_number      INT NOT NULL,
  status          VARCHAR(20) NOT NULL,
  worker_id       VARCHAR(50),
  started_at      TIMESTAMPTZ,
  completed_at    TIMESTAMPTZ,
  duration_ms     INT,
  error_message   TEXT,
  result          JSONB,
  created_at      TIMESTAMPTZ NOT NULL
);`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Job State Machine</Label>
          <CodeBlock code={`# Job Status Transitions (FSM)
#
#  SCHEDULED ──confirm──→ DISPATCHED ──pick_up──→ RUNNING
#      │                      │                      │
#      ↓ (cancel)             ↓ (timeout)            ↓
#   CANCELLED              SCHEDULED ←──retry───── FAILED
#                           (retry_count < max)       │
#                                                     ↓ (max retries)
#  SUCCEEDED ←─ complete ─── RUNNING              DEAD_LETTER
#
# Valid transitions:
TRANSITIONS = {
  "SCHEDULED":   ["DISPATCHED", "CANCELLED"],
  "DISPATCHED":  ["RUNNING", "SCHEDULED"],    # timeout → reschedule
  "RUNNING":     ["SUCCEEDED", "FAILED"],
  "FAILED":      ["SCHEDULED", "DEAD_LETTER"],# retry or give up
  "SUCCEEDED":   [],                          # terminal
  "CANCELLED":   [],                          # terminal
  "DEAD_LETTER": [],                          # terminal (needs manual)
}

# Recurring job: on SUCCEEDED → compute next fire_time → SCHEDULED
def on_job_complete(job):
    if job.type == "RECURRING":
        next_time = CronExpr(job.cron_expr).next_fire_time(now())
        db.update(job.id,
            status="SCHEDULED",
            next_fire_time=next_time,
            retry_count=0)
    else:
        db.update(job.id, status="SUCCEEDED")`} />
          <div className="mt-3 pt-3 border-t border-stone-100">
            <Label color="#059669">Partitioning Strategy</Label>
            <ul className="space-y-1.5">
              <Point icon="★" color="#059669"><strong className="text-stone-700">partition_key = hash(job_id) % N</strong> — each scheduler instance owns a range of partition keys. Jobs are distributed uniformly across partitions.</Point>
              <Point icon="→" color="#059669"><strong className="text-stone-700">Alternative: time-based</strong> — partition by next_fire_time minute bucket. Scheduler 0 owns minutes 0-5, Scheduler 1 owns 6-10, etc.</Point>
              <Point icon="→" color="#059669"><strong className="text-stone-700">Rebalancing</strong> — when a scheduler joins or leaves, partitions are redistributed using consistent hashing. In-flight jobs are not affected.</Point>
            </ul>
          </div>
        </Card>
      </div>
      <Card accent="#9333ea">
        <Label color="#9333ea">Data Design Principles for Job Scheduling</Label>
        <div className="grid grid-cols-3 gap-4">
          {[
            { t: "Index on (partition, fire_time) ★", d: "The scheduler's hot query is: 'give me all SCHEDULED jobs in my partition where fire_time ≤ NOW'. This must be a composite index to be fast.", pros: ["Single index scan per scheduler poll","Partition prefix eliminates full-table scan","Partial index (WHERE status='SCHEDULED') keeps it small"], cons: ["Index maintenance on every status change","Hot writes on fire_time during rescheduling"], pick: true },
            { t: "Execution History Append-Only", d: "Every run attempt is a new row in job_executions. Never update. This gives full retry history and debugging trail.", pros: ["Complete audit trail for every run","Easy to debug: see all attempts, errors, durations","Immutable — no lost data"], cons: ["Table grows fast (10M+ rows/day)","Need TTL/archival policy","Join needed for 'current status'"], pick: false },
            { t: "Soft Claim (Optimistic Lock)", d: "When scheduler dispatches a job, it does: UPDATE SET status='DISPATCHED', claimed_by=scheduler_id WHERE status='SCHEDULED' AND id=job_id. If 0 rows affected → another scheduler already claimed it.", pros: ["No distributed lock needed","DB guarantees atomicity","Simple conflict resolution"], cons: ["Requires serializable isolation or row-level lock","Under contention, many wasted attempts","Must handle race between multiple schedulers"], pick: false },
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
          <Label color="#059669">Scheduler Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Horizontal partitioning</strong> — partition the job space (by hash or time). Each scheduler instance owns a subset. N schedulers = N× throughput. Add more instances as load grows.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Consistent hashing for partition assignment</strong> — when a scheduler joins or leaves, only 1/N partitions are reassigned. Minimizes disruption. Uses etcd/Zookeeper for coordination.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Batch fetching</strong> — each scheduler fetches 1,000 due jobs per poll instead of one at a time. Amortizes DB round-trip cost. Significantly reduces DB load.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Pre-fetching / look-ahead</strong> — load jobs due in the next 60 seconds into memory. When fire_time arrives, dispatch instantly from memory without DB round-trip.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Worker Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Auto-scale on queue depth</strong> — monitor pending jobs per queue. If queue depth exceeds threshold (e.g., 1,000 pending), add worker nodes. Scale down when queues drain.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Per-queue worker pools</strong> — dedicated workers for different queues: "billing" gets 500 workers, "email" gets 300, "analytics" gets 200. Prevents noisy-neighbor.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Concurrency limiting</strong> — per-job-type concurrency limit: "only 10 report-generation jobs running at once" to prevent overwhelming downstream services.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Elastic worker pools</strong> — spot instances / preemptible VMs for burst capacity. Save 70% on compute. Jobs must be idempotent (may be interrupted).</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Database Scaling Strategy</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t: "Shard by Tenant ★", d: "Each tenant's jobs on one shard. Scheduler queries are always scoped to a partition within a shard. Natural isolation between tenants.", pros: ["Natural multi-tenant isolation","Queries always shard-local","Can place large tenants on dedicated shards"], cons: ["Uneven shard sizes (whale tenants)","Cross-tenant queries impossible","Shard rebalancing is complex"], pick: true },
            { t: "Time-Based Partitioning", d: "Partition jobs table by next_fire_time (monthly). Drop old partitions after retention. Current partition is small and fast to query.", pros: ["Old data automatically archived","Current partition stays small","Partition pruning speeds queries"], cons: ["Recurring jobs span many partitions","Rescheduled jobs change partitions","Need migration for updated fire_times"], pick: false },
            { t: "Queue as Message Broker", d: "Use Kafka/SQS as the primary dispatch mechanism. DB only for persistence. Workers consume from topics. Each queue = topic.", pros: ["Near-zero dispatch latency","Built-in consumer groups for scaling","Back-pressure via consumer lag"], cons: ["Two sources of truth (DB + queue)","Kafka not great for delayed delivery","Must reconcile on failure"], pick: false },
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
        <Label color="#d97706">The Core Question: What Happens When a Scheduler Crashes Mid-Dispatch?</Label>
        <p className="text-[12px] text-stone-500 mb-4">Jobs exist in a danger zone between "claimed by scheduler" and "acknowledged by worker." If the scheduler crashes in this window, jobs can be lost or stuck. The system must detect this and recover automatically.</p>
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Lease-Based Claim (Recommended)</div>
            <p className="text-[11px] text-stone-500 mb-2">Scheduler sets status='DISPATCHED' with a claim_expiry (e.g., now + 5 min). If worker doesn't ACK by expiry, job automatically becomes SCHEDULED again.</p>
            <ul className="space-y-1"><Point icon="✓" color="#059669">Self-healing: stuck jobs auto-release on expiry</Point><Point icon="✓" color="#059669">No external coordination needed</Point><Point icon="⚠" color="#d97706">Must set claim_expiry longer than max execution time</Point></ul>
          </div>
          <div className="rounded-lg border border-red-200 bg-white p-4">
            <div className="text-[11px] font-bold text-red-600 mb-1.5">Zombie Scheduler</div>
            <p className="text-[11px] text-stone-500 mb-2">Scheduler crashes but its lease in etcd hasn't expired yet. No one is processing its partitions. Jobs are stuck until lease TTL expires (30s).</p>
            <ul className="space-y-1"><Point icon="→" color="#d97706">Short lease TTL (30s) with frequent heartbeat (10s)</Point><Point icon="→" color="#d97706">Health check probes detect unresponsive schedulers</Point><Point icon="→" color="#d97706">Fencing tokens prevent zombie from acting after recovery</Point></ul>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#2563eb">Scheduler HA — Failover</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#2563eb"><strong className="text-stone-700">Partition-level failover</strong> — each partition has a primary scheduler. If it dies, another instance acquires the partition lease within 30s. No single point of failure.</Point>
            <Point icon="2." color="#2563eb"><strong className="text-stone-700">Heartbeat monitoring</strong> — schedulers heartbeat to etcd every 10s. If 3 consecutive heartbeats missed (30s), the partition is up for re-election.</Point>
            <Point icon="3." color="#2563eb"><strong className="text-stone-700">Graceful shutdown</strong> — on SIGTERM, scheduler releases all partition leases immediately. No 30s wait. Other instances pick up within seconds.</Point>
            <Point icon="4." color="#2563eb"><strong className="text-stone-700">Standby schedulers</strong> — run N+2 scheduler instances. N active, 2 standby. Standby watches for released partitions and acquires them instantly.</Point>
          </ul>
        </Card>
        <Card>
          <Label color="#0891b2">Worker HA — Job Recovery</Label>
          <ul className="space-y-2.5">
            <Point icon="🛡️" color="#0891b2"><strong className="text-stone-700">Visibility timeout</strong> — when worker dequeues a job, it becomes invisible for T seconds. If worker doesn't ACK (completes + reports), job reappears in queue for another worker.</Point>
            <Point icon="📊" color="#0891b2"><strong className="text-stone-700">Worker heartbeat</strong> — long-running jobs send periodic heartbeats to extend visibility timeout. No heartbeat for 60s → job assumed failed → retried.</Point>
            <Point icon="🔄" color="#0891b2"><strong className="text-stone-700">Idempotent execution</strong> — since at-least-once means a job may run twice (after timeout recovery), the job handler must be idempotent. Same input → same effect.</Point>
            <Point icon="📈" color="#0891b2"><strong className="text-stone-700">Dead letter queue</strong> — after max_retries exhausted, job moved to DLQ. Ops team investigates and manually retries or discards. Alert on DLQ growth.</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#9333ea">Degradation Ladder</Label>
        <div className="flex gap-3 items-stretch mt-1">
          {[
            { label: "All Healthy", sub: "Schedulers + Workers nominal", color: "#059669", status: "HEALTHY" },
            { label: "Scheduler Failover", sub: "Partitions reassigned, ~30s gap", color: "#d97706", status: "DEGRADED" },
            { label: "Worker Backpressure", sub: "Queue depth rising, auto-scaling", color: "#ea580c", status: "PRESSURE" },
            { label: "Shedding Low Priority", sub: "Drop P0-P3 jobs, preserve P8-P10", color: "#dc2626", status: "EMERGENCY" },
          ].map((t,i) => (
            <div key={i} className="flex-1 flex items-center gap-2">
              {i > 0 && <span className="text-stone-300 text-lg shrink-0">→</span>}
              <div className="flex-1 rounded-lg border p-3 text-center" style={{ borderColor: t.color+"40", background: t.color+"08" }}>
                <Pill bg={t.color+"20"} color={t.color}>{t.status}</Pill>
                <div className="text-[11px] font-bold text-stone-700 mt-2">{t.label}</div>
                <div className="text-[10px] text-stone-400">{t.sub}</div>
              </div>
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
          <Label color="#0284c7">Key Metrics — Scheduler Dashboard</Label>
          <ul className="space-y-2.5">
            <Point icon="📊" color="#0284c7"><strong className="text-stone-700">Scheduling Latency</strong> — time between fire_time and actual dispatch. Target: p50 &lt; 1s, p99 &lt; 5s. High latency = scheduler overloaded or DB slow.</Point>
            <Point icon="⏱️" color="#0284c7"><strong className="text-stone-700">Queue Depth per Queue</strong> — number of pending jobs. Growing queue = workers can't keep up. Alert at 10× normal depth.</Point>
            <Point icon="💰" color="#0284c7"><strong className="text-stone-700">Job Success Rate</strong> — % of jobs that succeed on first attempt. Target: &gt;95%. Below 90% = systematic issue (downstream failures).</Point>
            <Point icon="🔄" color="#0284c7"><strong className="text-stone-700">Retry Rate</strong> — % of jobs that need retries. High retry rate = flaky downstream or bad job logic. Track per queue and per job type.</Point>
            <Point icon="❌" color="#0284c7"><strong className="text-stone-700">DLQ Size</strong> — dead letter queue count. Should be near zero. Any growth needs investigation. Alert immediately.</Point>
          </ul>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Execution Monitoring</Label>
          <ul className="space-y-2.5">
            <Point icon="📝" color="#059669"><strong className="text-stone-700">Job duration percentiles</strong> — p50, p90, p99 per job type. Sudden increase = downstream degradation or resource contention.</Point>
            <Point icon="🔍" color="#059669"><strong className="text-stone-700">Execution tracing</strong> — each job execution gets a trace_id. Follow through scheduler → queue → worker → downstream calls. Full latency breakdown.</Point>
            <Point icon="📊" color="#059669"><strong className="text-stone-700">Worker utilization</strong> — % of worker slots occupied. Below 30% = over-provisioned. Above 85% = risk of queue backup.</Point>
            <Point icon="🔔" color="#059669"><strong className="text-stone-700">Stale job detection</strong> — jobs stuck in DISPATCHED or RUNNING beyond their timeout. Alert and auto-recover.</Point>
          </ul>
        </Card>
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Alerting Rules</Label>
          <ul className="space-y-2.5">
            <Point icon="🔴" color="#dc2626"><strong className="text-stone-700">P0: Scheduler down</strong> — no heartbeat from scheduler partition for &gt;60s. Auto-failover should trigger. Alert if no failover.</Point>
            <Point icon="🟠" color="#d97706"><strong className="text-stone-700">P1: Scheduling latency</strong> — p99 &gt; 30s means jobs are firing significantly late. Check DB query performance and scheduler load.</Point>
            <Point icon="🟡" color="#d97706"><strong className="text-stone-700">P2: DLQ growth</strong> — any jobs in DLQ need manual attention. Group by job type to find systemic issues.</Point>
            <Point icon="🔵" color="#2563eb"><strong className="text-stone-700">P3: Queue depth spike</strong> — 10× normal depth for any queue. Auto-scale workers or investigate downstream.</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#0284c7">Sample Job Execution Trail</Label>
        <CodeBlock code={`# Job job_7kx2m lifecycle events
{event: "job.created",       ts: "09:00:00.000", status: "SCHEDULED", fire: "09:30:00"}
{event: "job.dispatched",    ts: "09:30:00.150", status: "DISPATCHED", scheduler: "sched-02", partition: 7}
{event: "job.dequeued",      ts: "09:30:00.320", status: "RUNNING", worker: "worker-41", queue: "email"}
{event: "job.heartbeat",     ts: "09:30:15.000", worker: "worker-41", progress: "50%"}
{event: "job.completed",     ts: "09:30:28.400", status: "SUCCEEDED", duration_ms: 28080, result: "ok"}
{event: "job.callback_sent", ts: "09:30:28.600", url: "https://api.example.com/hooks/jobs", http: 200}
# Scheduling latency: 150ms (fire_time to dispatch). Execution: 28s. Total: ~28.6s.`} />
      </Card>
    </div>
  );
}

function WatchoutsSection() {
  return (
    <div className="space-y-5">
      <Card accent="#dc2626">
        <Label color="#dc2626">Critical Failure Modes</Label>
        <div className="grid grid-cols-2 gap-4">
          {[
            { mode: "Duplicate Execution", impact: "CRITICAL", desc: "Worker A times out, job re-queued to Worker B. But Worker A is still running — now two workers execute the same job simultaneously.",
              mitigation: "Idempotent job handlers (same input → same effect). Fencing tokens: worker must present valid token when writing results. If token is stale (another worker was assigned), discard result.",
              example: "Email job times out after 30s. Worker A is still sending the email (slow SMTP server). Worker B picks it up and sends the same email. Customer gets the email twice." },
            { mode: "Thundering Herd at Midnight", impact: "HIGH", desc: "10,000 cron jobs all scheduled for '0 0 * * *' (midnight). All become due simultaneously. Scheduler tries to dispatch 10K jobs in one second, overwhelming the queue and workers.",
              mitigation: "Jitter: add random delay (0-60s) to cron jobs. Spread midnight jobs across the minute. Rate-limit dispatch: enqueue at most 1,000/s. Pre-scale workers before known peaks.",
              example: "Every night at midnight, 15K report-generation jobs fire simultaneously. Queue depth spikes to 15K, workers saturate, some jobs timeout and retry — creating even more load." },
            { mode: "Scheduler Partition Stuck", impact: "HIGH", desc: "Scheduler instance holding partition 7 enters a GC pause for 20 seconds. etcd lease hasn't expired yet. All jobs in partition 7 are delayed by 20s.",
              mitigation: "Short lease TTL (30s) with aggressive heartbeat (10s). GC-tuned JVM or use Go (no long pauses). Monitor scheduling latency per partition — alert if any partition lags.",
              example: "Java-based scheduler hits a 15-second GC pause. 500 jobs in its partition fire 15 seconds late. Some jobs trigger downstream timeouts that cascade." },
            { mode: "Poison Job (Infinite Retry Loop)", impact: "MEDIUM", desc: "A job always fails due to a bug in the handler code. Retry with backoff means it retries 3 times, goes to DLQ, gets manually retried, fails again — forever.",
              mitigation: "Max retries + DLQ (dead letter queue) is baseline. Add: DLQ alert, auto-classify repeated failures, circuit breaker per job type — if 10 consecutive jobs of type X fail, pause the queue.",
              example: "Report-generation job fails because the template was deleted. Retries 3 times, DLQ'd, ops retries, fails again. Repeat for 3 days until someone notices the template is gone." },
            { mode: "Clock Skew Between Nodes", impact: "MEDIUM", desc: "Scheduler node's clock is 5 seconds ahead. It dispatches jobs 5 seconds early. Another scheduler's clock is 3 seconds behind — dispatches 3 seconds late. 8-second inconsistency.",
              mitigation: "Use NTP on all nodes with tight sync (&lt;100ms skew). Use database time (NOW()) for all scheduling decisions, not local clock. Centralized time authority for fire_time comparisons.",
              example: "Two schedulers disagree on whether a job is due. Scheduler A dispatches it (its clock says it's due). Scheduler B also dispatches it later (its clock catches up). Duplicate execution." },
            { mode: "Queue Backpressure Cascade", impact: "HIGH", desc: "Downstream service goes down. Worker jobs start failing and retrying. Retry jobs clog the queue. New jobs can't be processed. All queues back up even if only one downstream is broken.",
              mitigation: "Per-queue isolation: each queue has its own worker pool. Circuit breaker per downstream: if downstream is down, fail fast instead of retrying. Separate retry queue from primary queue.",
              example: "Payment service goes down. All 'billing' jobs retry 3 times each. 1,000 pending billing jobs become 3,000 retries. The shared worker pool is consumed by billing retries, starving email and analytics jobs." },
          ].map((f,i) => (
            <div key={i} className="rounded-lg border border-stone-200 p-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-[12px] font-bold text-stone-800">{f.mode}</span>
                <Pill bg={f.impact==="CRITICAL"?"#fef2f2":"#fffbeb"} color={f.impact==="CRITICAL"?"#dc2626":"#d97706"}>{f.impact}</Pill>
              </div>
              <p className="text-[11px] text-stone-500 mb-2">{f.desc}</p>
              <div className="text-[10px] text-emerald-600 mb-1"><strong>Mitigation:</strong> {f.mitigation}</div>
              <div className="text-[10px] italic text-stone-400">Example: {f.example}</div>
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
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#0f766e">
          <Label color="#0f766e">Service Breakdown</Label>
          <div className="space-y-3">
            {[
              { name: "API Service", role: "REST/gRPC API for job CRUD. Validates input, generates job_id, writes to DB, returns immediately. Stateless.", tech: "Go / Node.js + PostgreSQL", critical: false },
              { name: "Scheduler Service", role: "Polls DB for due jobs in its partitions, claims them (status → DISPATCHED), enqueues to worker queue. The brain of the system.", tech: "Go + etcd (coordination)", critical: true },
              { name: "Queue (Message Broker)", role: "Decouples scheduler from workers. Provides at-least-once delivery, visibility timeout, and per-queue ordering. The backbone.", tech: "Kafka / SQS / Redis Streams", critical: true },
              { name: "Worker Service", role: "Pulls jobs from queue, executes them (HTTP callback, gRPC call, or inline function), reports result. Horizontally scalable.", tech: "Go / Python + auto-scaling", critical: true },
              { name: "Cron Manager", role: "Manages recurring jobs: computes next fire_time after each execution, handles cron expression parsing, timezone logic.", tech: "Part of Scheduler Service", critical: true },
              { name: "Dead Letter Processor", role: "Handles permanently failed jobs. Stores for manual inspection, provides retry API, alerts ops team.", tech: "Separate queue + dashboard", critical: false },
              { name: "Callback / Webhook Service", role: "Delivers job results to client's callback_url. Signs payloads, retries on failure, tracks delivery status.", tech: "Async workers + retry queue", critical: false },
              { name: "Coordination Service", role: "Manages scheduler partition assignment, leader election, and health checks. The orchestrator of schedulers.", tech: "etcd / Zookeeper", critical: true },
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
          <Label color="#9333ea">System Architecture — Block Diagram</Label>
          <svg viewBox="0 0 380 320" className="w-full">
            <rect x={10} y={10} width={360} height={40} rx={6} fill="#6366f108" stroke="#6366f1" strokeWidth={1}/>
            <text x={190} y={28} textAnchor="middle" fill="#6366f1" fontSize="10" fontWeight="600" fontFamily="monospace">API Service (REST / gRPC)</text>
            <text x={190} y={42} textAnchor="middle" fill="#6366f180" fontSize="8" fontFamily="monospace">submit · query · cancel · pause/resume</text>

            <rect x={10} y={60} width={175} height={55} rx={6} fill="#dc262608" stroke="#dc2626" strokeWidth={1}/>
            <text x={97} y={80} textAnchor="middle" fill="#dc2626" fontSize="10" fontWeight="600" fontFamily="monospace">Job Store (DB)</text>
            <text x={97} y={95} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">jobs · executions</text>
            <text x={97} y={107} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">partitioned by tenant</text>

            <rect x={195} y={60} width={175} height={55} rx={6} fill="#7c3aed08" stroke="#7c3aed" strokeWidth={1}/>
            <text x={282} y={80} textAnchor="middle" fill="#7c3aed" fontSize="10" fontWeight="600" fontFamily="monospace">etcd / Zookeeper</text>
            <text x={282} y={95} textAnchor="middle" fill="#7c3aed80" fontSize="8" fontFamily="monospace">partition leases</text>
            <text x={282} y={107} textAnchor="middle" fill="#7c3aed80" fontSize="8" fontFamily="monospace">leader election</text>

            <rect x={10} y={125} width={360} height={50} rx={6} fill="#9333ea08" stroke="#9333ea" strokeWidth={1}/>
            <text x={190} y={145} textAnchor="middle" fill="#9333ea" fontSize="10" fontWeight="600" fontFamily="monospace">Scheduler Cluster (N instances)</text>
            <text x={100} y={163} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">Sched-0 [P0,P1,P2]</text>
            <text x={280} y={163} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">Sched-1 [P3,P4,P5]</text>

            <rect x={10} y={185} width={360} height={40} rx={6} fill="#d9770608" stroke="#d97706" strokeWidth={1}/>
            <text x={190} y={203} textAnchor="middle" fill="#d97706" fontSize="10" fontWeight="600" fontFamily="monospace">Message Queue (Kafka / SQS)</text>
            <text x={190} y={218} textAnchor="middle" fill="#d9770680" fontSize="8" fontFamily="monospace">billing · email · analytics · default</text>

            <rect x={10} y={235} width={175} height={40} rx={6} fill="#05966908" stroke="#059669" strokeWidth={1}/>
            <text x={97} y={255} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="600" fontFamily="monospace">Worker Pool</text>
            <text x={97} y={268} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">auto-scaled per queue</text>

            <rect x={195} y={235} width={175} height={40} rx={6} fill="#0891b208" stroke="#0891b2" strokeWidth={1}/>
            <text x={282} y={255} textAnchor="middle" fill="#0891b2" fontSize="10" fontWeight="600" fontFamily="monospace">Callback Service</text>
            <text x={282} y={268} textAnchor="middle" fill="#0891b280" fontSize="8" fontFamily="monospace">webhook delivery + retry</text>

            <rect x={10} y={285} width={360} height={30} rx={6} fill="#78716c08" stroke="#78716c" strokeWidth={1}/>
            <text x={190} y={303} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Dead Letter Queue · Monitoring · Dashboard</text>

            <line x1={190} y1={50} x2={97} y2={60} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={190} y1={115} x2={190} y2={125} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={190} y1={175} x2={190} y2={185} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={97} y1={225} x2={97} y2={235} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={282} y1={225} x2={282} y2={235} stroke="#94a3b8" strokeWidth={1}/>
          </svg>
        </Card>
      </div>

      <Card>
        <Label color="#0f766e">Service-to-Service Contracts</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[11px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Caller → Callee</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Protocol</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Contract</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Timeout</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">On Failure</th>
            </tr></thead>
            <tbody>
              {[
                { route: "Client → API Service", proto: "REST / gRPC", contract: "CreateJob / GetJob / DeleteJob / ListJobs", timeout: "5s", fail: "Retry with idempotency key" },
                { route: "API → Job Store (DB)", proto: "SQL (pgx)", contract: "INSERT job, SELECT by id, UPDATE status", timeout: "2s", fail: "Return 503, client retries" },
                { route: "Scheduler → Job Store", proto: "SQL (batch)", contract: "SELECT due jobs WHERE partition=X, UPDATE claimed", timeout: "2s", fail: "Skip poll cycle, retry next tick" },
                { route: "Scheduler → Queue", proto: "Kafka producer", contract: "Enqueue(job_id, payload, queue_topic)", timeout: "1s (ack)", fail: "Release claim, job becomes SCHEDULED again" },
                { route: "Queue → Worker", proto: "Consumer pull", contract: "Dequeue → execute → ACK/NACK", timeout: "visibility TO", fail: "Job reappears after visibility timeout" },
                { route: "Worker → Callback URL", proto: "HTTPS POST", contract: "Signed JSON payload, expect 2xx", timeout: "30s", fail: "Retry 3× with backoff, then mark delivered=false" },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-mono text-teal-700 font-medium">{r.route}</td>
                  <td className="px-3 py-2 text-stone-500">{r.proto}</td>
                  <td className="px-3 py-2 text-stone-500">{r.contract}</td>
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
  const [flow, setFlow] = useState("oneTime");
  const flows = {
    oneTime: {
      title: "One-Time Job (Happy Path)",
      steps: [
        { actor: "Client", action: "POST /v1/jobs {name: 'send-email', schedule_at: 'T+30min', payload: {user: 'abc'}}", type: "request" },
        { actor: "API Service", action: "Validate input, generate job_id (ULID), compute partition_key = hash(job_id) % 12", type: "auth" },
        { actor: "Job Store (DB)", action: "INSERT job (status: SCHEDULED, next_fire_time: T+30min, partition: 7)", type: "process" },
        { actor: "API Service", action: "Return 201: {id: 'job_7kx2m', status: 'SCHEDULED', next_fire_time: '...'}", type: "success" },
        { actor: "Scheduler-1", action: "[30 min later] Poll: SELECT FROM jobs WHERE partition IN (6,7,8) AND fire_time ≤ NOW()", type: "check" },
        { actor: "Scheduler-1", action: "Claim job: UPDATE status='DISPATCHED', claimed_by='sched-1', claim_expiry=NOW()+5min", type: "process" },
        { actor: "Scheduler-1", action: "Enqueue to Kafka topic 'email' with job_id and payload", type: "request" },
        { actor: "Worker-41", action: "Dequeue from 'email' topic. Execute: POST https://api.example.com/send-email {user: 'abc'}", type: "process" },
        { actor: "Worker-41", action: "Downstream returns 200 OK. Job execution successful.", type: "success" },
        { actor: "Worker-41", action: "ACK to queue. Write execution record (SUCCEEDED, duration: 1.2s). Update job status.", type: "success" },
        { actor: "Callback Service", action: "POST callback_url with {job_id, status: 'SUCCEEDED', result: {...}}", type: "process" },
      ]
    },
    recurring: {
      title: "Recurring Cron Job",
      steps: [
        { actor: "Client", action: "POST /v1/jobs {name: 'daily-report', type: 'RECURRING', cron: '0 9 * * *', queue: 'reports'}", type: "request" },
        { actor: "API Service", action: "Parse cron '0 9 * * *' → next fire = tomorrow 9:00 AM. Insert with status SCHEDULED.", type: "process" },
        { actor: "Scheduler", action: "[Next day 9:00:00] Job is due. Claim and dispatch to 'reports' queue.", type: "check" },
        { actor: "Worker", action: "Dequeue → Execute report generation → Takes 45 seconds → Success.", type: "process" },
        { actor: "Cron Manager", action: "On completion: parse cron again → next fire = day-after-tomorrow 9:00 AM.", type: "process" },
        { actor: "Job Store", action: "UPDATE next_fire_time = day+2 9:00 AM, status = SCHEDULED, retry_count = 0.", type: "success" },
        { actor: "Note", action: "Cycle repeats daily. If execution fails, retry with backoff. Next cron fire is independent of retries.", type: "check" },
      ]
    },
    retry: {
      title: "Failure → Retry → DLQ",
      steps: [
        { actor: "Worker-12", action: "Dequeue job_7kx2m from 'billing' queue. Execute: call payment service.", type: "process" },
        { actor: "Payment Service", action: "Returns 503 Service Unavailable. Job execution FAILED.", type: "error" },
        { actor: "Worker-12", action: "NACK job. Write execution record: {status: FAILED, error: '503', attempt: 1/3}.", type: "error" },
        { actor: "Retry Logic", action: "retry_count (1) < max_retries (3). Compute backoff: 2^1 × 5s = 10s delay.", type: "check" },
        { actor: "Job Store", action: "UPDATE status=SCHEDULED, next_fire_time=NOW()+10s, retry_count=1.", type: "process" },
        { actor: "Scheduler", action: "[10s later] Job is due again. Dispatch to 'billing' queue.", type: "check" },
        { actor: "Worker-23", action: "Attempt 2: call payment service → 503 again. FAILED. Backoff: 20s.", type: "error" },
        { actor: "Worker-07", action: "Attempt 3: call payment service → 503 again. FAILED. retry_count (3) = max_retries (3).", type: "error" },
        { actor: "DLQ Handler", action: "Move to dead_letter_queue. Alert ops. Status: DEAD_LETTER.", type: "error" },
        { actor: "Ops / Dashboard", action: "Investigate: payment service was in maintenance window. Manually retry from DLQ.", type: "check" },
      ]
    },
    failover: {
      title: "Scheduler Failover",
      steps: [
        { actor: "Scheduler-1", action: "Owns partitions [6,7,8]. Processing normally. Heartbeating to etcd every 10s.", type: "process" },
        { actor: "Scheduler-1", action: "Process crashes (OOM kill). No more heartbeats.", type: "error" },
        { actor: "etcd", action: "[30s later] Lease for Scheduler-1 expires. Partitions [6,7,8] are now unowned.", type: "check" },
        { actor: "Scheduler-3 (standby)", action: "Watches etcd for released partitions. Detects [6,7,8] are free.", type: "check" },
        { actor: "Scheduler-3", action: "Acquires leases for partitions [6,7,8]. Becomes owner. Gets new fence token.", type: "success" },
        { actor: "Scheduler-3", action: "Polls DB for due jobs in [6,7,8]. Finds 50 jobs that were delayed during 30s gap.", type: "process" },
        { actor: "Scheduler-3", action: "Dispatches all 50 jobs to their respective queues. Scheduling resumes normally.", type: "success" },
        { actor: "Recovery Check", action: "Jobs stuck in DISPATCHED (claimed by dead Scheduler-1) auto-release after claim_expiry.", type: "check" },
        { actor: "Alert System", action: "P1 alert: Scheduler-1 crashed. Partitions reassigned in 35s. 50 jobs delayed ~35s.", type: "error" },
      ]
    },
  };
  const colors = { request:"#2563eb", auth:"#7c3aed", process:"#64748b", success:"#059669", error:"#dc2626", check:"#d97706" };
  const f = flows[flow];
  return (
    <div className="space-y-5">
      <div className="flex gap-2 flex-wrap">
        {Object.entries(flows).map(([k,v]) => (
          <button key={k} onClick={() => setFlow(k)}
            className={`px-3.5 py-1.5 rounded-lg text-[12px] font-medium transition-all border ${k===flow?"bg-purple-600 text-white border-purple-600":"bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
            {v.title}
          </button>
        ))}
      </div>
      <Card accent="#7e22ce">
        <Label color="#7e22ce">{f.title}</Label>
        <div className="space-y-0">
          {f.steps.map((s,i) => (
            <div key={i} className="flex items-start gap-3 py-2.5 border-b border-stone-100 last:border-0">
              <span className="text-[10px] font-bold w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5" style={{ background: colors[s.type]+"20", color: colors[s.type] }}>{i+1}</span>
              <span className="text-[11px] font-mono font-bold shrink-0 w-36" style={{ color: colors[s.type] }}>{s.actor}</span>
              <span className="text-[12px] text-stone-600">{s.action}</span>
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
          <Label color="#b45309">Deployment Strategy</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#b45309"><strong className="text-stone-700">Rolling deploy for workers</strong> — workers are stateless; deploy new version gradually. Drain existing jobs before stopping each pod. Zero downtime.</Point>
            <Point icon="2." color="#b45309"><strong className="text-stone-700">Blue-green for scheduler</strong> — scheduler is stateful (holds partition leases). Deploy new version as green cluster, transfer partition leases one-by-one from blue to green. Verify each partition before proceeding.</Point>
            <Point icon="3." color="#b45309"><strong className="text-stone-700">Database migration safety</strong> — only additive schema changes (new columns, new tables). Never drop or rename. Use feature flags to toggle between old and new code paths.</Point>
            <Point icon="4." color="#b45309"><strong className="text-stone-700">Canary for job handlers</strong> — new job handler logic deployed to 1% of workers first. Monitor success rate and latency for 30 min before full rollout.</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Security</Label>
          <ul className="space-y-2.5">
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">Tenant isolation</strong> — each tenant's jobs are logically isolated. tenant_id on every query. No cross-tenant data access. Separate API keys per tenant.</Point>
            <Point icon="🔑" color="#dc2626"><strong className="text-stone-700">Payload encryption</strong> — job payloads may contain sensitive data (API keys, PII). Encrypt at rest with per-tenant keys. Decrypt only in worker at execution time.</Point>
            <Point icon="🛡️" color="#dc2626"><strong className="text-stone-700">Callback URL validation</strong> — only allow HTTPS callback URLs. Validate against allowlist of domains per tenant. Sign all webhook payloads with HMAC.</Point>
            <Point icon="📝" color="#dc2626"><strong className="text-stone-700">Rate limiting per tenant</strong> — limit job creation rate (e.g., 100 jobs/sec per tenant) to prevent abuse. Limit total active jobs per tenant (e.g., 1M).</Point>
            <Point icon="🧱" color="#dc2626"><strong className="text-stone-700">Execution sandboxing</strong> — if workers execute arbitrary code (like serverless functions), run in isolated containers with resource limits (CPU, memory, network).</Point>
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
        <Label color="#be123c">Auto-Scaling &amp; Alerting Triggers</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[11px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Trigger</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Threshold</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Action</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Cooldown</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Pitfall</th>
            </tr></thead>
            <tbody>
              {[
                { trigger: "Queue Depth", thresh: "10× normal for 5min", action: "Auto-scale workers for that queue (+50% pods)", cool: "3 min", pitfall: "May be caused by downstream failure, not capacity. Check error rate first." },
                { trigger: "Scheduling Latency", thresh: "p99 &gt; 30s for 5min", action: "P1 Alert. Check DB query perf, scheduler CPU, partition distribution.", cool: "0 (immed.)", pitfall: "Could be one hot partition. Check per-partition latency." },
                { trigger: "Worker Utilization", thresh: "&gt; 85% slots for 10min", action: "Add worker nodes. Pre-scale if approaching known peak hours.", cool: "5 min", pitfall: "Over-scaling wastes resources. Scale down aggressively when load drops." },
                { trigger: "DLQ Growth", thresh: "Any new DLQ entries", action: "P2 Alert. Group by job type. Check for poison jobs or downstream outage.", cool: "15 min", pitfall: "Single bad job type can flood DLQ. Use circuit breaker per type." },
                { trigger: "Scheduler Heartbeat", thresh: "Missed 3 consecutive", action: "P1 Alert. Verify auto-failover triggered. Check partition reassignment.", cool: "0 (immed.)", pitfall: "Network partition can cause false positive. Verify with multiple probes." },
                { trigger: "Job Success Rate", thresh: "&lt; 90% for 10min", action: "Check downstream health. Enable circuit breaker. Alert on-call.", cool: "5 min", pitfall: "One bad job type can skew overall rate. Check per-type breakdown." },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-mono text-stone-700 font-medium">{r.trigger}</td>
                  <td className="px-3 py-2 font-mono text-amber-700">{r.thresh}</td>
                  <td className="px-3 py-2 text-stone-500">{r.action}</td>
                  <td className="px-3 py-2 text-stone-400">{r.cool}</td>
                  <td className="px-3 py-2 text-red-500 text-[10px]">{r.pitfall}</td>
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
            { title: "The Midnight Thundering Herd", symptom: "Every night at midnight UTC, scheduler latency spikes to 60s. Workers saturate. Some jobs timeout and create cascading retries.",
              cause: "8,000 cron jobs all scheduled for '0 0 * * *'. All become due at exactly 00:00:00. Scheduler tries to dispatch all 8K in one poll cycle.",
              fix: "Added configurable jitter (0-120s) to all cron jobs. '0 0 * * *' now fires randomly between 00:00:00 and 00:02:00. Spread 8K over 120 seconds = 67/s instead of 8K/s. Also pre-scaled workers at 23:55.",
              quote: "Midnight was our scariest time. We called it 'the witching hour.' After adding jitter, our p99 scheduling latency went from 60s to 2s." },
            { title: "The Runaway Retry Storm", symptom: "Worker pool at 100% utilization but queue depth keeps growing. No new jobs can execute. All queues affected.",
              cause: "Payment service went down at 14:00. 2,000 billing jobs failed and entered retry loops. Each retry with exponential backoff still consumed a worker slot. 2K jobs × 3 retries = 6K executions. Shared worker pool was consumed.",
              fix: "Isolated worker pools per queue (billing, email, analytics). Added circuit breaker: if 10 consecutive jobs of same type fail, pause that queue for 30s. Separate retry queue with lower priority than primary queue.",
              quote: "One service going down shouldn't take out the entire scheduler. We learned that isolation between queues is not optional — it's required for production." },
            { title: "The Ghost Partition", symptom: "Jobs in partition 7 are consistently delayed by 5-10 minutes. All other partitions are fine.",
              cause: "Scheduler-2 held the lease for partition 7 but had entered a livelock — it was heartbeating (so etcd didn't revoke the lease) but stuck in an infinite loop in the poll logic due to a corrupted job record.",
              fix: "Added per-partition scheduling latency monitoring. If any partition's p99 exceeds 30s, alert immediately. Added health check that verifies the scheduler is actually processing (not just alive). Kill + restart on livelock detection.",
              quote: "The scheduler was technically alive — it was heartbeating fine. It just wasn't doing any work. We now monitor 'jobs dispatched per partition per minute' not just 'is the process alive.'" },
            { title: "Idempotency Key Wasn't", symptom: "Customer reports getting 3 identical welcome emails after signing up.",
              cause: "Client retried CreateJob 3 times (network timeout). idempotency_key was supposed to prevent duplicates but was scoped to the job_id (which didn't exist yet) instead of the client-provided key. Three different job_ids created.",
              fix: "Idempotency key must be client-provided and stored as a UNIQUE constraint. Check before INSERT. If key exists, return the existing job instead of creating a new one. TTL the idempotency record after 24h.",
              quote: "The irony of a job scheduler sending duplicate jobs because the idempotency check was broken. We had the feature, we just implemented it wrong." },
            { title: "Timezone Daylight Saving Surprise", symptom: "Daily 2:30 AM report didn't run on March 9. On November 2, it ran twice.",
              cause: "Cron job set to '30 2 * * *' in US/Eastern. Spring forward: 2:30 AM doesn't exist (clocks jump 2→3). Fall back: 2:30 AM happens twice (clocks go 2→1→2 again).",
              fix: "Store all fire_times in UTC internally. Convert to local time only for display. For ambiguous times: skip if DST removes it, run once (first occurrence) if DST duplicates it. Document DST behavior in API docs.",
              quote: "CFO called asking why the financial report was missing. Turned out 2:30 AM didn't exist that day. Now every cron job tutorial in our docs starts with 'always think in UTC.'" },
            { title: "The 100M Job Table Slow Query", symptom: "Scheduler poll query takes 8 seconds instead of 50ms. All scheduling delayed.",
              cause: "Jobs table grew to 100M rows (old completed jobs never cleaned up). The partial index on status='SCHEDULED' was only 2M rows but the query planner chose a full table scan due to stale statistics.",
              fix: "Added TTL-based archival: move completed jobs to archive table after 7 days. VACUUM ANALYZE after archival. Partitioned jobs table by created_at month. Index stats stay fresh.",
              quote: "We went from 100M rows to 3M in the hot table. Query went from 8s back to 40ms. Now we have a cron job (ironic) that archives old jobs every night." },
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
        { t: "DAG / Workflow Support", d: "Jobs with dependencies: Job B runs only after Job A succeeds. Directed Acyclic Graph of tasks. Powers ETL pipelines and ML training workflows.", detail: "Each DAG is a set of tasks with edges. Scheduler tracks completed tasks and triggers downstream when all parents complete. Use topological sort for execution order. Airflow and Temporal are purpose-built for this.", effort: "Hard" },
        { t: "Exactly-Once Execution", d: "Guarantee each job runs exactly once, not just at-least-once. Critical for financial operations where duplicates are unacceptable.", detail: "Fencing tokens + transactional outbox. Worker must present valid fence token when writing results. If token is stale (reassigned), result is discarded. Combine with idempotent job handlers for belt-and-suspenders.", effort: "Hard" },
        { t: "Rate-Limited Dispatch", d: "Limit how fast jobs are dispatched to protect downstream services. E.g., max 100 email API calls per second across all workers.", detail: "Token bucket or sliding window per job type or downstream. Scheduler checks rate limit before dispatching. If limit reached, jobs wait until tokens replenish. Distributed rate limiter using Redis.", effort: "Medium" },
        { t: "Job Chaining / Callbacks", d: "On completion, automatically trigger another job. Chain: Job A → Job B → Job C. Lightweight alternative to full DAG support.", detail: "Add 'on_success' and 'on_failure' job IDs to job definition. When job completes, scheduler creates and schedules the next job in the chain. Simple state machine per chain.", effort: "Easy" },
        { t: "Multi-Region Scheduling", d: "Schedule and execute jobs in specific regions for data locality and compliance (e.g., EU jobs run on EU workers).", detail: "Add region field to job. Regional scheduler instances only process jobs for their region. Cross-region replication for job metadata. Region-aware queue routing.", effort: "Hard" },
        { t: "Cron Expression UI Builder", d: "Visual cron expression builder for non-technical users. Shows human-readable description and next 5 fire times for verification.", detail: "Frontend component that translates UI selections (every day, at 9am, on weekdays) to cron syntax. Preview panel shows upcoming fire times. Timezone selector with DST awareness.", effort: "Easy" },
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
    { q: "How do you prevent duplicate job execution in a distributed system?", a: "Three layers: (1) Idempotency key on job creation — prevents duplicate jobs from being created. (2) Lease-based claiming — scheduler marks job as DISPATCHED with a claim_expiry. Only one scheduler can claim a job (optimistic locking in DB). (3) Fencing tokens — each claim gets a monotonically increasing token. When worker writes results, it must present the current fence token. If another worker was assigned (higher token), the stale worker's write is rejected. For true exactly-once, the job handler itself must be idempotent — same input produces same side effects regardless of how many times it runs.", tags: ["design"] },
    { q: "How does the timing wheel compare to Redis sorted sets for scheduling?", a: "Timing wheel: O(1) insert and fire, in-memory, sub-millisecond precision, but volatile (lost on crash). Redis ZSET: O(log N) insert, durable (AOF/RDB), shared across processes, but single-threaded bottleneck at ~100K ops/s. Production approach: use both. Timing wheel for hot scheduling (next 1-5 minutes) for speed, Redis/DB for durable storage of all jobs. A loader thread promotes jobs from durable store to the wheel. This gives you the speed of in-memory with the durability of persistent storage.", tags: ["algorithm"] },
    { q: "How would you handle a tenant that submits 10 million jobs at once?", a: "Rate limiting + queue isolation + backpressure. (1) Rate limit job creation per tenant (e.g., 1,000/sec). Client gets 429 Too Many Requests. (2) Per-tenant queue depth limit (e.g., 1M active jobs max). (3) Weighted fair queuing — tenant's jobs get a proportional share of worker capacity, not all of it. (4) If one tenant's jobs are slow, they don't steal worker slots from other tenants (per-tenant worker pools). The goal is fairness: no single tenant can monopolize the system.", tags: ["scalability"] },
    { q: "What happens to recurring jobs when the scheduler is down for 10 minutes?", a: "When the scheduler recovers (or failover completes), it polls for all jobs where next_fire_time ≤ NOW(). It finds all the 'missed' executions and dispatches them. Key decision: for a job that fires every minute and was missed 10 times, do you run all 10 missed executions or just the latest one? Depends on the use case. For data sync: run all 10 (each covers a different time window). For health check: run only the latest (stale checks are useless). This is configurable via a 'misfire_policy' field: RUN_ALL, RUN_LATEST, or SKIP.", tags: ["availability"] },
    { q: "How do you monitor and debug a job that's stuck?", a: "Every job has a full execution trail: status transitions with timestamps, worker assignment, duration, and error messages. For stuck jobs: (1) Check status — if DISPATCHED for longer than claim_expiry, the scheduler should have auto-released it. If it hasn't, the scheduler may be stuck too. (2) Check worker — is the assigned worker alive? Is it heartbeating? (3) Check downstream — is the job's target service responding? (4) Distributed tracing: each execution gets a trace_id that follows through scheduler → queue → worker → downstream. View the full trace in Jaeger/Zipkin to find where it's stuck.", tags: ["observability"] },
    { q: "How does Airflow's scheduler work internally?", a: "Airflow uses a polling model: the scheduler loop runs every few seconds, queries the metadata DB (PostgreSQL) for DAGs that have tasks ready to run (all upstream dependencies met, schedule interval reached). It creates TaskInstance records, sets status to 'queued', and puts them in a queue (Celery, Kubernetes, or local executor). Workers pull from the queue and execute. Key limitation: single scheduler was a bottleneck until Airflow 2.0 added HA scheduler with row-level locks. Still, it polls the full DAG bag every cycle, which gets slow at scale (1,000+ DAGs).", tags: ["design"] },
    { q: "How would you implement job dependencies (DAG)?", a: "Data model: add a 'depends_on' field (list of job_ids). When a job completes, check all jobs that depend on it. For each dependent: query all its dependencies. If ALL are SUCCEEDED, mark the dependent as SCHEDULED (ready to fire). If ANY dependency FAILED, mark dependent as BLOCKED. This is a simple BFS/topological check. For complex DAGs: store the graph edges in a separate table (dag_edges: parent_id, child_id). On completion, do: UPDATE jobs SET status='SCHEDULED' WHERE id IN (SELECT child_id FROM dag_edges WHERE parent_id = completed_job_id) AND NOT EXISTS (SELECT 1 FROM dag_edges e JOIN jobs j ON e.parent_id = j.id WHERE e.child_id = jobs.id AND j.status != 'SUCCEEDED').", tags: ["design"] },
    { q: "At-least-once vs exactly-once — when does it matter?", a: "At-least-once is sufficient for idempotent operations: sending an email (might send twice but dedup on message_id), updating a cache (same result on re-run), syncing data (re-sync is harmless). Exactly-once matters for non-idempotent operations: charging a credit card (double-charge is bad), decrementing inventory (double-decrement oversells), sending a notification with a unique code. For exactly-once: use fencing tokens + transactional outbox + idempotent handlers. In practice, most systems target at-least-once with idempotent handlers — it's simpler and covers 95% of use cases.", tags: ["design"] },
    { q: "How do you handle timezone and DST for cron jobs?", a: "Rule 1: Store all fire_times in UTC internally. Always. Rule 2: The cron expression is evaluated in the user's specified timezone to compute the 'human' time, then converted to UTC for storage. Rule 3: DST edge cases — when clocks spring forward (2:30 AM doesn't exist), skip that execution or fire at the next valid time (3:00 AM). When clocks fall back (2:30 AM happens twice), fire only once (first occurrence). Rule 4: Recalculate next fire_time after every execution, not ahead of time — this handles DST transitions that occur between now and the next fire.", tags: ["algorithm"] },
    { q: "How would you design this for serverless (AWS Lambda)?", a: "Replace worker nodes with Lambda invocations. Scheduler enqueues to SQS (instead of Kafka). SQS triggers Lambda. Benefits: auto-scaling (Lambda scales to 1,000+ concurrent), no worker management, pay-per-execution. Challenges: (1) Cold start adds 100-500ms latency. (2) 15-minute max execution time. (3) No persistent connections (must reconnect to DB each invocation). (4) Concurrency limits per account (1,000 default). For the scheduler itself: use EventBridge Scheduler (AWS managed) or Step Functions for DAGs. Trade-off: simpler ops but less control and higher per-invocation cost at scale.", tags: ["scalability"] },
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

export default function JobSchedulerSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Job Scheduler</h1>
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