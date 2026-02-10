import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   DISTRIBUTED QUEUE (KAFKA) — System Design Reference
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
            <Label>What is a Distributed Message Queue?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A distributed message queue is a durable, fault-tolerant system that decouples producers (senders) from consumers (receivers) by buffering messages between them. Producers publish messages to topics without knowing who will consume them, and consumers read at their own pace. It transforms synchronous, tightly-coupled services into asynchronous, independently-scalable components.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Think of it like a postal sorting facility: senders drop off mail (messages) into labeled bins (topics). The facility stores the mail durably in sorted order, and mail carriers (consumers) pick up mail from their assigned bins whenever they're ready. The sender doesn't wait for the carrier — they just drop it off and leave. The facility guarantees nothing is lost and everything arrives in the order it was received.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Queues Are Foundational</Label>
            <ul className="space-y-2.5">
              <Point icon="🔗" color="#0891b2"><strong className="text-stone-700">Decoupling</strong> — producers and consumers evolve independently. Adding a new consumer doesn't require changing the producer. Services don't need to know about each other.</Point>
              <Point icon="📈" color="#0891b2"><strong className="text-stone-700">Load leveling</strong> — absorb traffic spikes. Producer writes at 100K/s during peak; consumer processes at 10K/s steadily. The queue buffers the difference.</Point>
              <Point icon="🛡️" color="#0891b2"><strong className="text-stone-700">Durability</strong> — messages survive producer and consumer crashes. If a consumer dies, it resumes from where it left off. No data lost.</Point>
              <Point icon="🔄" color="#0891b2"><strong className="text-stone-700">Fan-out</strong> — one message consumed by multiple independent consumer groups. Order service publishes "order.created"; inventory, email, and analytics all consume it.</Point>
              <Point icon="⏱️" color="#0891b2"><strong className="text-stone-700">Replay</strong> — consumers can re-read historical messages by resetting their offset. Rebuild a cache, reprocess events, fix a bug and replay — all without republishing.</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Systems</Label>
            <div className="space-y-2.5">
              {[
                { co: "Kafka", rule: "Distributed log, high throughput", algo: "Append-only, partitioned" },
                { co: "RabbitMQ", rule: "Traditional broker, AMQP protocol", algo: "Push-based, exchanges" },
                { co: "SQS", rule: "AWS managed, at-least-once", algo: "Visibility timeout" },
                { co: "Pulsar", rule: "Multi-tenant, tiered storage", algo: "Segment-based, BookKeeper" },
                { co: "NATS", rule: "Lightweight, cloud-native", algo: "Subject-based, JetStream" },
                { co: "Redpanda", rule: "Kafka-compatible, no JVM", algo: "C++, thread-per-core" },
                { co: "Google Pub/Sub", rule: "Serverless, global", algo: "Push/pull, seek" },
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
            <Label color="#2563eb">Core Architecture</Label>
            <svg viewBox="0 0 360 190" className="w-full">
              <DiagramBox x={55} y={35} w={70} h={30} label="Producer" color="#2563eb"/>
              <DiagramBox x={55} y={85} w={70} h={30} label="Producer" color="#2563eb"/>
              <DiagramBox x={180} y={35} w={80} h={30} label="Partition 0" color="#9333ea"/>
              <DiagramBox x={180} y={65} w={80} h={30} label="Partition 1" color="#9333ea"/>
              <DiagramBox x={180} y={95} w={80} h={30} label="Partition 2" color="#9333ea"/>
              <DiagramBox x={310} y={35} w={70} h={30} label="Consumer" color="#059669"/>
              <DiagramBox x={310} y={65} w={70} h={30} label="Consumer" color="#059669"/>
              <DiagramBox x={310} y={95} w={70} h={30} label="Consumer" color="#059669"/>
              <Arrow x1={90} y1={35} x2={140} y2={35} id="ca1"/>
              <Arrow x1={90} y1={85} x2={140} y2={65} id="ca2"/>
              <Arrow x1={90} y1={85} x2={140} y2={95} id="ca3"/>
              <Arrow x1={220} y1={35} x2={275} y2={35} id="ca4"/>
              <Arrow x1={220} y1={65} x2={275} y2={65} id="ca5"/>
              <Arrow x1={220} y1={95} x2={275} y2={95} id="ca6"/>
              <rect x={140} y={115} width={80} height={16} rx={4} fill="#9333ea08" stroke="#9333ea30"/>
              <text x={180} y={124} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">Topic "orders"</text>
              <rect x={60} y={145} width={240} height={18} rx={4} fill="#05966908" stroke="#05966930"/>
              <text x={180} y={155} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">Each partition → exactly one consumer per group</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">LinkedIn, Uber, Confluent, Amazon, Meta, Stripe</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Queue vs Log vs Broker</div>
            <p className="text-[12px] text-stone-500 mt-0.5">"Design a message queue" can mean: (1) a <strong>traditional queue</strong> (RabbitMQ — message deleted after consumption, push-based), (2) a <strong>distributed log</strong> (Kafka — append-only, consumer tracks offset, pull-based), or (3) a <strong>managed broker</strong> (SQS — visibility timeout, at-least-once). Clarify immediately. For most interviews, design a <strong>Kafka-style distributed log</strong> — it's the most interesting and covers the most concepts.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Publish messages — producers send messages to a named topic</Point>
            <Point icon="2." color="#059669">Subscribe and consume — consumers read messages from topics in order</Point>
            <Point icon="3." color="#059669">Ordering guarantee — messages within a partition are strictly ordered (FIFO)</Point>
            <Point icon="4." color="#059669">Consumer groups — multiple independent groups each get all messages; within a group, each partition assigned to one consumer</Point>
            <Point icon="5." color="#059669">Message retention — messages retained for a configurable period (e.g., 7 days) regardless of consumption</Point>
            <Point icon="6." color="#059669">Replay — consumers can seek to any offset and re-consume historical messages</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">High throughput — millions of messages per second</Point>
            <Point icon="2." color="#dc2626">Low latency — end-to-end p99 &lt; 10ms for publish, &lt; 100ms for consume</Point>
            <Point icon="3." color="#dc2626">Durability — zero message loss once acknowledged by broker</Point>
            <Point icon="4." color="#dc2626">High availability — survives broker failures without data loss (replication)</Point>
            <Point icon="5." color="#dc2626">Horizontal scalability — add partitions and brokers to increase throughput linearly</Point>
            <Point icon="6." color="#dc2626">At-least-once delivery — every message delivered to consumer at least once (exactly-once as stretch goal)</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask the Interviewer</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Kafka-style log or RabbitMQ-style traditional queue?",
            "Ordering: per-partition FIFO or global total order?",
            "Delivery guarantee: at-most-once, at-least-once, or exactly-once?",
            "What's the expected throughput? Thousands or millions msg/sec?",
            "Average message size? Bytes (events) or megabytes (files)?",
            "Retention period? Hours, days, or infinite (event sourcing)?",
            "Multi-tenancy? Isolation between different teams/services?",
            "Do consumers push or pull? (Kafka = pull, RabbitMQ = push)",
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
            <div className="text-[12px] font-bold text-violet-700">Interview Tip — Throughput Is King</div>
            <p className="text-[12px] text-stone-500 mt-0.5">For a message queue, the critical dimensions are <strong>messages/sec</strong>, <strong>bytes/sec</strong>, and <strong>storage for retention</strong>. Unlike a database, reads don't hit disk (sequential + page cache). The bottleneck is disk write throughput for durability and network bandwidth for replication. Show you understand that sequential I/O is what makes Kafka fast.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Message Volume</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Average messages per second" result="500K/s" note="Aggregate across all topics and producers" />
            <MathStep step="2" formula="Peak messages per second (3×)" result="1.5M/s" note="Flash sales, event bursts, batch jobs" />
            <MathStep step="3" formula="Average message size" result="~1 KB" note="JSON event payload (range: 100B to 10KB)" />
            <MathStep step="4" formula="Average throughput = 500K × 1 KB" result="500 MB/s" note="Sustained write throughput" />
            <MathStep step="5" formula="Peak throughput = 1.5M × 1 KB" result="1.5 GB/s" note="Must sustain without backpressure" final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Storage (Retention)</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Daily volume = 500K/s × 86,400 × 1 KB" result="~43 TB/day" note="Raw message data per day" />
            <MathStep step="2" formula="Retention period" result="7 days" note="Configurable per topic (1 day to ∞)" />
            <MathStep step="3" formula="Raw storage = 43 TB × 7" result="~300 TB" note="Before replication" />
            <MathStep step="4" formula="Replication factor = 3" result="~900 TB" note="Each message stored on 3 brokers" />
            <MathStep step="5" formula="With overhead (indexes, metadata)" result="~1 PB total" note="Across all brokers in the cluster" final />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Broker Cluster Sizing</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Disk throughput per broker (NVMe SSD)" result="~500 MB/s" note="Sequential write, which is what queues do" />
            <MathStep step="2" formula="Brokers for write throughput" result="~9" note="1.5 GB/s peak × 3 replicas / 500 MB/s" />
            <MathStep step="3" formula="Storage per broker = 1 PB / 9" result="~110 TB" note="Large but feasible with JBOD (just a bunch of disks)" />
            <MathStep step="4" formula="Network per broker (25 Gbps NIC)" result="~3.1 GB/s" note="25 Gbps = 3.1 GB/s. Ample for replication + client traffic" />
            <MathStep step="5" formula="Total brokers (with headroom)" result="~15 brokers" note="9 for throughput + headroom for rebalance/failure" final />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Partition Count</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Consumer throughput per partition" result="~10K msg/s" note="Limited by single consumer thread processing" />
            <MathStep step="2" formula="Partitions for consumption" result="~150" note="1.5M/s peak / 10K per partition" />
            <MathStep step="3" formula="Producer throughput per partition" result="~50K msg/s" note="Batching amortizes network overhead" />
            <MathStep step="4" formula="Partitions for production" result="~30" note="1.5M/s / 50K — production is less constrained" />
            <MathStep step="5" formula="Total partitions (consumer-bound)" result="~150-200" note="Consumption is usually the bottleneck" final />
          </div>
        </Card>
      </div>
      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Peak Throughput", val: "1.5M msg/s", sub: "1.5 GB/s" },
            { label: "Total Storage", val: "~1 PB", sub: "7-day, RF=3" },
            { label: "Broker Count", val: "~15", sub: "NVMe SSD, 25G NIC" },
            { label: "Partitions", val: "~150-200", sub: "Consumer-bound" },
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
          <Label color="#2563eb">Producer API</Label>
          <CodeBlock code={`# Produce a message to a topic
producer.send(
  topic    = "orders",
  key      = "user_123",       # partition key (hash → partition)
  value    = '{"order_id":"ord_abc","total":4999}',
  headers  = {"trace_id": "t_xyz", "source": "checkout"},
  timestamp = 1717200000000    # optional, defaults to broker time
)
# Response (async, batched):
{
  "topic": "orders",
  "partition": 3,              # determined by hash(key) % num_partitions
  "offset": 847291,            # monotonically increasing per partition
  "timestamp": 1717200000042
}

# Producer configuration (critical settings):
producer = Producer(
  bootstrap_servers = ["broker1:9092", "broker2:9092"],
  acks = "all",                # wait for all ISR replicas (durability)
  retries = 3,                 # auto-retry on transient failure
  batch_size = 64 * 1024,      # 64 KB batches (throughput)
  linger_ms = 5,               # wait up to 5ms to fill batch
  compression = "lz4",         # compress batches (2-4× savings)
  idempotence = True,          # exactly-once producer semantics
  max_in_flight = 5            # pipelining for throughput
)`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Consumer API</Label>
          <CodeBlock code={`# Subscribe and consume from a topic
consumer = Consumer(
  bootstrap_servers = ["broker1:9092"],
  group_id = "order-processor",  # consumer group
  auto_offset_reset = "earliest",# start from beginning if no commit
  enable_auto_commit = False,    # manual commit for safety
  max_poll_records = 500,        # batch size per poll
  session_timeout_ms = 30000,    # 30s before considered dead
  heartbeat_interval_ms = 10000  # heartbeat every 10s
)
consumer.subscribe(["orders"])

while True:
    records = consumer.poll(timeout_ms=1000)  # pull model
    for record in records:
        process(record)          # business logic
        # record.topic, record.partition, record.offset
        # record.key, record.value, record.timestamp
    consumer.commit()            # commit offsets after processing

# Admin API:
admin.create_topic("orders", partitions=12, replication=3)
admin.alter_topic("orders", retention_ms=604800000)  # 7 days
admin.add_partitions("orders", total=24)  # scale out
admin.list_consumer_groups()
admin.describe_consumer_group("order-processor")  # lag info`} />
        </Card>
      </div>
      <Card>
        <Label color="#d97706">Critical Design Decisions</Label>
        <div className="grid grid-cols-2 gap-4">
          <ul className="space-y-1.5">
            <Point icon="→" color="#d97706"><strong className="text-stone-700">Pull vs Push</strong> — consumers pull (poll) from broker. Consumer controls pace, handles backpressure naturally. Kafka-style.</Point>
            <Point icon="→" color="#d97706"><strong className="text-stone-700">Partition key determines ordering</strong> — messages with same key always go to same partition → guaranteed order per key.</Point>
            <Point icon="→" color="#d97706"><strong className="text-stone-700">Offset-based tracking</strong> — consumer stores its position (offset) per partition. Can seek to any offset for replay.</Point>
          </ul>
          <ul className="space-y-1.5">
            <Point icon="→" color="#d97706"><strong className="text-stone-700">acks=all for durability</strong> — producer waits for all in-sync replicas to acknowledge. No data loss even if leader dies.</Point>
            <Point icon="→" color="#d97706"><strong className="text-stone-700">Batching for throughput</strong> — producer accumulates messages into batches before sending. Amortizes network round-trips.</Point>
            <Point icon="→" color="#d97706"><strong className="text-stone-700">Manual commit for safety</strong> — auto-commit can lose messages. Commit AFTER processing, not before.</Point>
          </ul>
        </div>
      </Card>
    </div>
  );
}

function DesignSection() {
  const [phase, setPhase] = useState(0);
  const phases = [
    { label: "Single Broker", desc: "One broker, one topic, one partition. Producer appends to log file. Consumer reads sequentially. Simple but no fault tolerance, no parallelism, single machine throughput limit. Fine for development; never for production." },
    { label: "Partitioned Topic", desc: "Split topic into N partitions distributed across brokers. Producer hashes message key to select partition. Each partition is an ordered, append-only log. Consumers in a group each own a subset of partitions. Parallelism = number of partitions. This is the core Kafka model." },
    { label: "Replicated ★", desc: "Each partition replicated to R brokers (typically 3). One replica is the leader (handles reads/writes), others are followers (replicate from leader). If leader dies, a follower is promoted. ISR (in-sync replicas) ensures no data loss. This is production Kafka." },
    { label: "Tiered Storage", desc: "Hot data on local SSDs, cold data in object storage (S3). Brokers keep recent segments locally; older segments offloaded to S3. Enables infinite retention at low cost. Consumers seamlessly read from local or remote. Kafka 3.6+ (KIP-405) and Pulsar support this." },
  ];
  const diagrams = [
    () => (
      <svg viewBox="0 0 460 120" className="w-full">
        <DiagramBox x={70} y={50} w={80} h={32} label="Producer" color="#2563eb"/>
        <DiagramBox x={230} y={50} w={110} h={40} label="Broker\n(single log)" color="#9333ea"/>
        <DiagramBox x={400} y={50} w={80} h={32} label="Consumer" color="#059669"/>
        <Arrow x1={110} y1={50} x2={175} y2={50} label="append" id="sb1"/>
        <Arrow x1={285} y1={50} x2={360} y2={50} label="read" id="sb2"/>
        <rect x={140} y={90} width={180} height={16} rx={4} fill="#dc262608" stroke="#dc262630"/>
        <text x={230} y={99} textAnchor="middle" fill="#dc2626" fontSize="8" fontFamily="monospace">✗ No replication, no parallelism, SPOF</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 140" className="w-full">
        <DiagramBox x={60} y={40} w={72} h={28} label="Producer" color="#2563eb"/>
        <DiagramBox x={195} y={25} w={90} h={24} label="Partition 0" color="#9333ea" sub="Broker 1"/>
        <DiagramBox x={195} y={55} w={90} h={24} label="Partition 1" color="#9333ea" sub="Broker 2"/>
        <DiagramBox x={195} y={85} w={90} h={24} label="Partition 2" color="#9333ea" sub="Broker 3"/>
        <DiagramBox x={370} y={25} w={70} h={24} label="C-0" color="#059669"/>
        <DiagramBox x={370} y={55} w={70} h={24} label="C-1" color="#059669"/>
        <DiagramBox x={370} y={85} w={70} h={24} label="C-2" color="#059669"/>
        <Arrow x1={96} y1={35} x2={150} y2={27} label="hash" id="pt1"/>
        <Arrow x1={96} y1={45} x2={150} y2={57} id="pt2"/>
        <Arrow x1={240} y1={25} x2={335} y2={25} id="pt3"/>
        <Arrow x1={240} y1={55} x2={335} y2={55} id="pt4"/>
        <Arrow x1={240} y1={85} x2={335} y2={85} id="pt5"/>
        <rect x={100} y={115} width={250} height={16} rx={4} fill="#05966908" stroke="#05966930"/>
        <text x={225} y={124} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">✓ Parallelism = partitions. Order per partition.</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 150" className="w-full">
        <DiagramBox x={60} y={40} w={72} h={28} label="Producer" color="#2563eb"/>
        <rect x={130} y={15} width={200} height={100} rx={8} fill="#9333ea06" stroke="#9333ea30" strokeDasharray="4,3"/>
        <text x={230} y={28} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">Partition 0 (RF=3)</text>
        <DiagramBox x={175} y={55} w={70} h={24} label="Leader" color="#9333ea"/>
        <DiagramBox x={255} y={55} w={70} h={24} label="Follower" color="#7c3aed"/>
        <DiagramBox x={255} y={85} w={70} h={24} label="Follower" color="#7c3aed"/>
        <Arrow x1={96} y1={40} x2={140} y2={55} label="write" id="rp1"/>
        <Arrow x1={210} y1={50} x2={220} y2={50} id="rp2" label="replicate"/>
        <Arrow x1={210} y1={60} x2={220} y2={80} id="rp3" dashed/>
        <DiagramBox x={400} y={55} w={70} h={28} label="Consumer" color="#059669"/>
        <Arrow x1={210} y1={55} x2={365} y2={55} label="read (leader)" id="rp4"/>
        <rect x={110} y={125} width={240} height={16} rx={4} fill="#9333ea08" stroke="#9333ea30"/>
        <text x={230} y={134} textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="monospace">★ Leader handles R/W. Follower promoted on failure.</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 140" className="w-full">
        <DiagramBox x={80} y={40} w={100} h={34} label="Local Disk\n(hot data)" color="#9333ea" sub="recent segments"/>
        <DiagramBox x={250} y={40} w={110} h={34} label="Object Store\n(cold data)" color="#d97706" sub="S3 / GCS"/>
        <DiagramBox x={370} y={40} w={80} h={30} label="Consumer" color="#059669"/>
        <Arrow x1={130} y1={40} x2={195} y2={40} label="offload" id="ts1"/>
        <Arrow x1={305} y1={40} x2={330} y2={40} label="read old" id="ts2" dashed/>
        <Arrow x1={130} y1={50} x2={330} y2={50} label="read recent" id="ts3"/>
        <rect x={80} y={95} width={280} height={18} rx={4} fill="#d9770608" stroke="#d9770630"/>
        <text x={220} y={105} textAnchor="middle" fill="#d97706" fontSize="8" fontFamily="monospace">∞ retention at $0.02/GB. Hot on SSD, cold on S3.</text>
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
        <div className="bg-stone-50 rounded-lg border border-stone-200 p-3" style={{ minHeight: 130 }}>
          {diagrams[phase]()}
        </div>
      </Card>
      <Card>
        <Label color="#c026d3">Queue Model Comparison</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { t: "Distributed Log (Kafka) ★", d: "Append-only log. Messages retained by time/size. Consumer tracks offset. Pull-based. Fan-out via consumer groups.", pros: ["Highest throughput (millions/s)","Replay + event sourcing built-in","Ordering per partition","Decoupled retention from consumption"], cons: ["Consumer manages offset (more complex)","No per-message routing (topic-level only)","Partition count is hard to change","Head-of-line blocking per partition"], pick: true },
            { t: "Traditional Broker (RabbitMQ)", d: "Push-based. Messages deleted after ACK. Exchanges route to queues. Flexible routing (fanout, topic, headers).", pros: ["Rich routing patterns","Message-level priority","Push-based (lower latency)","Simpler consumer model"], cons: ["Lower throughput (~50K/s)","No replay once consumed","Broker holds state (harder to scale)","Single queue = single consumer"], pick: false },
            { t: "Cloud Managed (SQS)", d: "Fully managed. Visibility timeout for at-least-once. No ordering (standard) or FIFO. Dead letter queue built in.", pros: ["Zero ops — fully managed","Infinite scale (AWS handles it)","DLQ built in","Pay per message"], cons: ["Higher latency (10-50ms)","No ordering in standard queue","No replay","Vendor lock-in"], pick: false },
            { t: "Tiered Log (Pulsar)", d: "Separate compute (brokers) from storage (BookKeeper). Tiered storage to S3 for infinite retention. Multi-tenant by design.", pros: ["Compute/storage separation","Infinite retention via tiered storage","Multi-tenancy built in","Geo-replication"], cons: ["More complex deployment (BookKeeper)","Smaller ecosystem than Kafka","Higher operational burden","Newer, less battle-tested"], pick: false },
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
  const [sel, setSel] = useState("appendLog");
  const algos = {
    appendLog: { name: "Append-Only Log (Write Path) ★", cx: "O(1) append",
      pros: ["O(1) append — constant time regardless of log size, just write to end of file","Sequential disk I/O — 10× faster than random I/O. HDDs: 600 MB/s sequential vs 1 MB/s random","OS page cache gives free read caching — recent messages served from memory, no application-level cache needed","Immutable segments enable zero-copy transfer: sendfile() from disk to network socket, bypassing user space"],
      cons: ["Deletes are expensive — can't remove individual messages, only truncate old segments","Compaction (for key-based retention) requires rewriting segments","Large logs consume disk — need retention policies and tiered storage","Random reads for old data are slow — sequential optimization only helps recent data"],
      when: "This is THE core data structure of Kafka. Every partition is an append-only log stored as a sequence of segment files. The producer appends to the active segment. The consumer reads sequentially from its offset. This is why Kafka achieves millions of messages per second — it leans into what disks are good at (sequential I/O) and avoids what they're bad at (random I/O).",
      code: `# Append-Only Log — The Core of Kafka
# Each partition = sequence of segment files on disk

class Partition:
    def __init__(self, topic, partition_id, segment_size=1_GB):
        self.topic = topic
        self.id = partition_id
        self.active_segment = Segment(base_offset=0)
        self.segments = [self.active_segment]
        self.next_offset = 0

    def append(self, message):
        # O(1) — always append to end of active segment
        offset = self.next_offset
        self.next_offset += 1
        entry = LogEntry(offset, timestamp=now(), key=message.key,
                         value=message.value)

        if self.active_segment.size >= SEGMENT_SIZE:
            self.active_segment.close()  # make immutable
            self.active_segment = Segment(base_offset=offset)
            self.segments.append(self.active_segment)

        self.active_segment.append(entry)  # sequential write
        return offset

    def read(self, offset, max_bytes):
        # Find segment containing offset (binary search on base_offsets)
        segment = self.find_segment(offset)
        # Read sequentially from offset position
        return segment.read_from(offset, max_bytes)

# Segment file on disk:
# /data/orders-3/00000000000000847291.log   (messages)
# /data/orders-3/00000000000000847291.index  (offset → position)
# /data/orders-3/00000000000000847291.timeindex (timestamp → offset)` },
    replication: { name: "ISR Replication Protocol", cx: "O(1) per msg",
      pros: ["Zero data loss when acks=all — message acknowledged only after all ISR replicas have it","Automatic leader election — if leader dies, an ISR follower is promoted in seconds","ISR (in-sync replica set) is dynamic — slow followers are removed, caught-up followers are added back","No consensus needed for every write (unlike Raft) — leader just waits for ISR acknowledgment"],
      cons: ["acks=all adds latency — must wait for slowest ISR follower","Unclean leader election risk — if all ISR replicas die, choose between data loss or unavailability","Network partitions can cause ISR shrinkage — reduced durability during partition","Replication lag means followers serve slightly stale data (if follower reads enabled)"],
      when: "ISR replication is Kafka's core durability mechanism. Each partition has a leader and N-1 followers. The leader maintains an ISR (in-sync replica) set — followers that are caught up within replica.lag.time.max.ms. When acks=all, the producer write is acknowledged only after ALL ISR replicas have the message. If the leader dies, a new leader is elected from the ISR — guaranteeing no data loss. This is the key concept for interview discussions about durability vs availability tradeoff.",
      code: `# ISR (In-Sync Replica) Replication Protocol
# Leader handles all reads and writes. Followers replicate.

class PartitionLeader:
    def __init__(self, replicas):
        self.log = AppendLog()
        self.isr = set(replicas)  # in-sync replicas
        self.high_watermark = 0    # last offset replicated to ALL ISR

    def handle_produce(self, message, acks):
        offset = self.log.append(message)

        if acks == 0:
            return offset  # fire-and-forget (fastest, unsafe)
        elif acks == 1:
            return offset  # leader only (fast, some risk)
        elif acks == "all":
            # Wait for ALL ISR followers to replicate
            self.wait_for_isr(offset)
            self.high_watermark = offset  # advance HW
            return offset  # safest, highest latency

    def handle_fetch(self, follower_id, fetch_offset):
        # Follower pulls from leader (like a consumer)
        data = self.log.read(fetch_offset)
        # Update follower's position
        self.follower_offsets[follower_id] = fetch_offset + len(data)
        # Check if follower is in sync
        if self.follower_offsets[follower_id] < self.log.end - LAG_MAX:
            self.isr.remove(follower_id)  # Too far behind
            log.warn(f"Removed {follower_id} from ISR")
        return data

    def handle_leader_failure(self):
        # Controller detects leader death (ZK session timeout)
        # Elect new leader from ISR
        new_leader = max(self.isr, key=lambda r: r.log_end_offset)
        # new_leader.high_watermark = truncation point
        # Followers truncate to new leader's HW and re-replicate` },
    consumerGroup: { name: "Consumer Group Rebalancing", cx: "O(P) assign",
      pros: ["Automatic partition assignment — consumers join/leave, partitions are redistributed","Exactly one consumer per partition per group — guarantees ordering and no duplicate processing","Coordinator handles failures — if consumer dies (misses heartbeat), its partitions are reassigned","Multiple strategies: range, round-robin, sticky, cooperative sticky"],
      cons: ["Rebalance causes a 'stop-the-world' pause — all consumers stop processing during reassignment","Sticky rebalance minimizes movement but still pauses consumption briefly","Adding consumers beyond partition count is wasteful — idle consumers","Rebalance storms: rapid join/leave causes repeated rebalances"],
      when: "Consumer groups are how Kafka achieves parallel consumption with ordering guarantees. When a consumer joins or leaves a group, the group coordinator triggers a rebalance — reassigning partitions across the remaining consumers. The key insight: max parallelism = number of partitions. If you have 12 partitions and 15 consumers, 3 consumers sit idle. In interviews, discuss: eager vs cooperative rebalance, and why rebalance storms happen.",
      code: `# Consumer Group Rebalance Protocol
# Coordinator (a broker) manages group membership + partition assignment

class GroupCoordinator:
    def __init__(self, group_id, topic_partitions):
        self.group_id = group_id
        self.members = {}           # consumer_id → metadata
        self.assignment = {}         # consumer_id → [partitions]
        self.generation = 0          # increments on each rebalance

    def join_group(self, consumer_id, metadata):
        self.members[consumer_id] = metadata
        self.trigger_rebalance()

    def leave_group(self, consumer_id):
        del self.members[consumer_id]
        self.trigger_rebalance()

    def trigger_rebalance(self):
        self.generation += 1
        consumers = list(self.members.keys())
        partitions = list(range(self.num_partitions))
        # Sticky assignment: minimize partition movement
        self.assignment = sticky_assign(consumers, partitions,
                                         self.assignment)
        # Notify all consumers of new assignment
        for c_id, parts in self.assignment.items():
            notify(c_id, parts, self.generation)

    def heartbeat(self, consumer_id):
        self.members[consumer_id].last_heartbeat = now()

    def check_health(self):
        for c_id, meta in self.members.items():
            if now() - meta.last_heartbeat > SESSION_TIMEOUT:
                log.warn(f"Consumer {c_id} timed out")
                self.leave_group(c_id)  # triggers rebalance

# Sticky assignment minimizes partition movement:
# Before: C0=[P0,P1], C1=[P2,P3]
# C2 joins → C0=[P0,P1], C1=[P2], C2=[P3]  (only P3 moved)` },
    zeroCopy: { name: "Zero-Copy Transfer (sendfile)", cx: "O(1) syscall",
      pros: ["Bypasses user space entirely — data goes from disk page cache to network socket via kernel","4 context switches → 2 context switches, 4 data copies → 2 copies (or 1 with DMA gather)","Critical for consumer throughput — serving cached messages at network line rate","Java NIO FileChannel.transferTo() wraps sendfile() syscall"],
      cons: ["Only works for unmodified data — can't encrypt or transform in zero-copy path","Requires data in page cache — cold reads still hit disk","SSL/TLS breaks zero-copy (need to encrypt in user space)","Not available on all OS/filesystems"],
      when: "Zero-copy is why Kafka consumers can read at near-network speed. Traditional path: disk → kernel buffer → user buffer → socket buffer → NIC (4 copies). Zero-copy path: disk → page cache → NIC (2 copies via DMA). When consumers are reading recent data (still in page cache), the broker serves it at ~GB/s per partition with almost no CPU usage. This is a key differentiator — mention it when discussing why Kafka is fast.",
      code: `# Zero-Copy Transfer — Why Kafka Consumers Are Fast
# Traditional read path (4 copies, 4 context switches):
#   1. read()  : disk → kernel buffer (DMA copy)
#   2. read()  : kernel buffer → user buffer (CPU copy)
#   3. write() : user buffer → socket buffer (CPU copy)
#   4. write() : socket buffer → NIC (DMA copy)

# Zero-copy path (2 copies, 2 context switches):
#   1. sendfile() : disk → kernel buffer (DMA copy)
#   2. sendfile() : kernel buffer → NIC (DMA gather copy)
#   No user space involved at all!

# Java implementation in Kafka:
class LogSegment:
    def read_into_network(self, offset, max_bytes, channel):
        # FileChannel.transferTo → sendfile() syscall
        position = self.index.lookup(offset)
        self.file_channel.transferTo(
            position,       # start position in file
            max_bytes,       # how much to transfer
            channel          # network socket channel
        )
        # Data goes directly from page cache to NIC
        # CPU does almost nothing — just one syscall

# When this works best:
# - Consumer is reading recent data (in page cache) ✓
# - Data is NOT encrypted at rest ✓
# - Consumer is reading large batches ✓

# When this doesn't work:
# - SSL/TLS enabled (must encrypt in user space) ✗
# - Data not in page cache (cold read from disk) ✗
# - Message-level filtering (need to read + filter) ✗` },
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
          <Label color="#dc2626">On-Disk Log Structure</Label>
          <CodeBlock code={`# Partition directory layout:
/data/orders-0/                    # topic "orders", partition 0
  00000000000000000000.log         # segment file (messages)
  00000000000000000000.index       # sparse offset index
  00000000000000000000.timeindex   # timestamp → offset index
  00000000000000847291.log         # next segment (base offset)
  00000000000000847291.index
  00000000000000847291.timeindex
  leader-epoch-checkpoint          # leader epoch history

# Message format (on disk):
┌──────────────┬──────────────────────────┐
│ Offset (8B)  │ monotonically increasing │
│ Size (4B)    │ message byte count       │
│ CRC (4B)     │ checksum for corruption  │
│ Magic (1B)   │ format version           │
│ Timestamp(8B)│ create or broker time    │
│ Key len (4B) │ -1 if null               │
│ Key (var)    │ partition routing key     │
│ Value len(4B)│ payload size             │
│ Value (var)  │ actual message payload   │
│ Headers (var)│ key-value metadata       │
└──────────────┴──────────────────────────┘

# Sparse offset index (every 4KB of messages):
# offset 847291 → file position 0
# offset 847310 → file position 4096
# offset 847329 → file position 8192
# Binary search on index → sequential scan for exact offset`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Metadata &amp; Consumer Offsets</Label>
          <CodeBlock code={`# Topic metadata (stored in controller/ZK):
{
  "topic": "orders",
  "partitions": [
    {
      "id": 0,
      "leader": 1,           # broker_id of leader
      "replicas": [1, 2, 3], # all replicas
      "isr": [1, 2, 3]       # in-sync replicas
    },
    {"id": 1, "leader": 2, "replicas": [2,3,1], "isr": [2,3,1]},
    {"id": 2, "leader": 3, "replicas": [3,1,2], "isr": [3,1,2]}
  ],
  "config": {
    "retention.ms": 604800000,  # 7 days
    "segment.bytes": 1073741824,# 1 GB segments
    "min.insync.replicas": 2,   # acks=all needs 2+ ISR
    "cleanup.policy": "delete"  # delete | compact | compact,delete
  }
}

# Consumer group offsets (__consumer_offsets topic):
# Stored as a compacted Kafka topic (key = group+topic+partition)
{
  "group": "order-processor",
  "topic": "orders",
  "partition": 0,
  "committed_offset": 847291,    # last processed
  "metadata": "",
  "commit_timestamp": 1717200000
}
# Consumer resumes from committed_offset + 1 on restart

# High watermark (HW) vs Log End Offset (LEO):
# LEO = last message written to leader
# HW  = last message replicated to ALL ISR
# Consumers can only read up to HW (not LEO)`} />
        </Card>
      </div>
      <Card accent="#9333ea">
        <Label color="#9333ea">Key Data Design Principles</Label>
        <div className="grid grid-cols-3 gap-4">
          {[
            { t: "Segment-Based Storage ★", d: "Log split into fixed-size segments (1GB). Active segment is writable; closed segments are immutable. Enables efficient retention (delete whole segment files) and compaction.", pros: ["Retention = delete old files (O(1) per segment)","Immutable segments enable zero-copy reads","Compaction only rewrites closed segments","Easy backup: copy segment files"], cons: ["Segment roll adds brief latency spike","Many small segments = many file handles","Active segment must be fsynced for durability"], pick: true },
            { t: "Offset-Based Consumption", d: "Each message has a unique, monotonically increasing offset within its partition. Consumers track their position by offset. Can seek to any offset for replay.", pros: ["Replay from any point in time","Consumer crash: resume from last committed offset","Multiple consumer groups at different positions","No per-message ACK tracking on broker"], cons: ["Consumer must manage offsets (commit after processing)","Gap-free offsets mean compaction creates 'holes'","Offset reset on new consumer: earliest vs latest?"], pick: false },
            { t: "Log Compaction (Key-Based)", d: "Instead of time-based deletion, keep only the latest message per key. Like an upsert table. Used for changelogs, CDC, and state materialization.", pros: ["Infinite retention without unbounded growth","Latest state per key always available","Consumers can rebuild state from compacted log","Works like a key-value store backed by a log"], cons: ["Compaction is expensive (rewrite segments)","Ordering only preserved for same-key messages","Tombstones (null value) needed for deletes","Not useful for event streams (all events matter)"], pick: false },
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
          <Label color="#059669">Throughput Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Add partitions</strong> — more partitions = more parallelism. Each partition is an independent log. Producers spread messages across partitions; consumers in a group each own a subset.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Add brokers</strong> — partitions are distributed across brokers. More brokers = more disk I/O and network bandwidth. Rebalance partitions to new brokers with kafka-reassign-partitions.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Producer batching</strong> — accumulate messages into batches (batch.size=64KB, linger.ms=5). One network round-trip sends hundreds of messages. 10-50× throughput improvement.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Compression</strong> — compress batches with LZ4 or ZStandard. 2-4× reduction in network and disk I/O. Compressed on producer, stored compressed, decompressed on consumer.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Consumer Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Consumers ≤ partitions</strong> — max parallelism = partition count. 12 partitions → max 12 consumers in a group. Adding a 13th consumer makes it idle. Plan partition count for future scale.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Independent consumer groups</strong> — each group gets ALL messages. Group "analytics" and group "search-indexer" both consume from "orders" independently. Adding a group doesn't affect existing groups.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Consumer thread pool</strong> — each consumer can process records in parallel threads (after dequeue). Keeps partition ordering on dequeue, parallelizes processing. Increases throughput without more partitions.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Micro-batching</strong> — consumer polls 500 records at once (max.poll.records=500). Process batch together — amortizes DB round-trips and network calls. 5-10× consumer throughput.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Partition Count — The Critical Decision</Label>
        <p className="text-[12px] text-stone-500 mb-3">Partition count is the most important scaling decision. Too few = bottleneck. Too many = overhead. And you can only increase partitions, never decrease them (messages with the same key would move to different partitions, breaking ordering).</p>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t: "Too Few Partitions", d: "Topic with 3 partitions and 20 consumers: 17 consumers sit idle. Max throughput capped at 3× single-partition throughput. Can't scale consumption.", pros: ["Simple to manage","Lower broker memory/file handles","Fewer rebalance events"], cons: ["Limited parallelism","Can't scale consumers","Hot partitions under load"], pick: false },
            { t: "Right-Sized ★", d: "Rule of thumb: partitions = max(expected consumer count, target throughput / per-partition throughput). Typically 6-24 partitions per topic for most workloads.", pros: ["Good parallelism without waste","Manageable broker overhead","Room for consumer scale-out","Predictable latency"], cons: ["Requires upfront capacity planning","May need to increase later (one-way)"], pick: true },
            { t: "Too Many Partitions", d: "Topic with 10,000 partitions: each partition = open file handles, memory for index, replication overhead. Leader election takes minutes if broker dies (10K elections).", pros: ["Massive parallelism","Never limited by partition count"], cons: ["High broker memory usage","Slow leader election on failure","Increased end-to-end latency","Producer batching less efficient"], pick: false },
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
        <Label color="#d97706">The Fundamental Tradeoff: Durability vs Availability</Label>
        <p className="text-[12px] text-stone-500 mb-4">When a broker (partition leader) dies, the system must choose: wait for the leader to come back (durable but unavailable) or elect a potentially stale follower as leader (available but risk of data loss). This is configured by <code className="text-xs bg-stone-100 px-1 rounded">unclean.leader.election.enable</code> and <code className="text-xs bg-stone-100 px-1 rounded">min.insync.replicas</code>.</p>
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Prioritize Durability (Default) ★</div>
            <p className="text-[11px] text-stone-500 mb-2">unclean.leader.election=false + min.insync.replicas=2 + acks=all. Partition becomes unavailable if fewer than min.insync.replicas are alive. No data loss, ever.</p>
            <ul className="space-y-1"><Point icon="✓" color="#059669">Zero data loss — acknowledged messages never lost</Point><Point icon="✓" color="#059669">Producers get error if durability can't be guaranteed</Point><Point icon="⚠" color="#d97706">Partition unavailable if ISR drops below min.insync</Point></ul>
          </div>
          <div className="rounded-lg border border-red-200 bg-white p-4">
            <div className="text-[11px] font-bold text-red-600 mb-1.5">Prioritize Availability</div>
            <p className="text-[11px] text-stone-500 mb-2">unclean.leader.election=true. If all ISR replicas die, allow an out-of-sync follower to become leader. Partition stays available but may lose messages not yet replicated.</p>
            <ul className="space-y-1"><Point icon="→" color="#d97706">Partition always available (even after all ISR die)</Point><Point icon="→" color="#d97706">Acceptable for metrics/logs (some loss OK)</Point><Point icon="✗" color="#dc2626">Data loss risk: unsynced messages permanently gone</Point></ul>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#2563eb">Broker Failure &amp; Recovery</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#2563eb"><strong className="text-stone-700">Leader dies → controller detects</strong> — ZooKeeper session timeout (default 18s) or KRaft heartbeat. Controller selects new leader from ISR.</Point>
            <Point icon="2." color="#2563eb"><strong className="text-stone-700">New leader election</strong> — highest LEO (log end offset) in ISR wins. Followers truncate logs to new leader's high watermark. Clients re-discover leader via metadata refresh.</Point>
            <Point icon="3." color="#2563eb"><strong className="text-stone-700">Broker rejoins</strong> — fetches missing data from current leader. Once caught up (within lag threshold), added back to ISR. Takes minutes to hours depending on data volume.</Point>
            <Point icon="4." color="#2563eb"><strong className="text-stone-700">Rack-aware placement</strong> — replicas placed on different racks/AZs. Ensures that a rack failure doesn't take out all replicas of any partition.</Point>
          </ul>
        </Card>
        <Card>
          <Label color="#0891b2">Consumer Failure Recovery</Label>
          <ul className="space-y-2.5">
            <Point icon="🛡️" color="#0891b2"><strong className="text-stone-700">Session timeout</strong> — if consumer misses heartbeat for 30s (session.timeout.ms), coordinator considers it dead. Its partitions are reassigned in a rebalance.</Point>
            <Point icon="📊" color="#0891b2"><strong className="text-stone-700">Offset commit recovery</strong> — new consumer assigned the partition starts from the last committed offset. Messages between last commit and crash are re-delivered (at-least-once).</Point>
            <Point icon="🔄" color="#0891b2"><strong className="text-stone-700">Static group membership</strong> — assign group.instance.id to avoid unnecessary rebalances on short restarts. Consumer can rejoin within session.timeout without triggering rebalance.</Point>
            <Point icon="📈" color="#0891b2"><strong className="text-stone-700">Cooperative rebalance</strong> — Kafka 2.4+: only revoke partitions that need to move, not all partitions. Other consumers continue processing during rebalance. Much less disruptive.</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#9333ea">Degradation Ladder</Label>
        <div className="flex gap-3 items-stretch mt-1">
          {[
            { label: "All Healthy", sub: "All brokers up, full ISR", color: "#059669", status: "HEALTHY" },
            { label: "Under-Replicated", sub: "ISR shrunk, still producing", color: "#d97706", status: "DEGRADED" },
            { label: "Leader Election", sub: "Broker died, failover ~10s", color: "#ea580c", status: "FAILOVER" },
            { label: "Partition Offline", sub: "ISR empty, writes rejected", color: "#dc2626", status: "UNAVAILABLE" },
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
          <Label color="#0284c7">Broker Metrics</Label>
          <ul className="space-y-2.5">
            <Point icon="📊" color="#0284c7"><strong className="text-stone-700">Under-Replicated Partitions</strong> — partitions where ISR &lt; replication factor. Non-zero = trouble. A broker is behind or failing.</Point>
            <Point icon="⏱️" color="#0284c7"><strong className="text-stone-700">Request Latency (p99)</strong> — produce and fetch request latency. Produce p99 &lt; 10ms, fetch p99 &lt; 50ms. Spike = disk I/O issue or network saturation.</Point>
            <Point icon="💾" color="#0284c7"><strong className="text-stone-700">Disk I/O Utilization</strong> — % of disk bandwidth used. &gt;80% = need more brokers or faster disks. Sequential write throughput is key metric.</Point>
            <Point icon="🌐" color="#0284c7"><strong className="text-stone-700">Network Throughput</strong> — bytes in/out per broker. Replication doubles write traffic. Approaching NIC limit = scale out.</Point>
            <Point icon="📈" color="#0284c7"><strong className="text-stone-700">Active Controller</strong> — exactly one controller in cluster. Alert if 0 (no controller) or if controller changes frequently (instability).</Point>
          </ul>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Consumer Metrics</Label>
          <ul className="space-y-2.5">
            <Point icon="📝" color="#059669"><strong className="text-stone-700">Consumer Lag</strong> — difference between log end offset and consumer's committed offset. THE most important consumer metric. Growing lag = consumer can't keep up.</Point>
            <Point icon="🔍" color="#059669"><strong className="text-stone-700">Records Consumed/sec</strong> — throughput per consumer. Sudden drop = processing failure or rebalance in progress.</Point>
            <Point icon="📊" color="#059669"><strong className="text-stone-700">Rebalance Rate</strong> — number of rebalances per hour. Frequent rebalances = unstable consumer group (restart loops, timeouts).</Point>
            <Point icon="🔔" color="#059669"><strong className="text-stone-700">Commit Latency</strong> — time to commit offsets. High latency = coordinator overloaded or network issue.</Point>
          </ul>
        </Card>
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Alerting Rules</Label>
          <ul className="space-y-2.5">
            <Point icon="🔴" color="#dc2626"><strong className="text-stone-700">P0: Offline partitions</strong> — any partition with no leader. Produces and consumes fail. Immediate investigation.</Point>
            <Point icon="🔴" color="#dc2626"><strong className="text-stone-700">P0: Under-replicated &gt; 0 for 10min</strong> — replication is failing. Data at risk. Check broker health, disk, network.</Point>
            <Point icon="🟠" color="#d97706"><strong className="text-stone-700">P1: Consumer lag &gt; threshold</strong> — consumer falling behind. Lag &gt; 100K for critical topics. Scale consumers or investigate slow processing.</Point>
            <Point icon="🟡" color="#d97706"><strong className="text-stone-700">P2: Produce latency spike</strong> — p99 &gt; 100ms. Check disk I/O, ISR health, producer batch settings.</Point>
            <Point icon="🔵" color="#2563eb"><strong className="text-stone-700">P3: Disk usage &gt; 80%</strong> — approaching capacity. Reduce retention, add disks, or add brokers before it's too late.</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#0284c7">Consumer Lag Visualization (The Most Important Dashboard)</Label>
        <CodeBlock code={`# Consumer lag = log_end_offset - consumer_committed_offset
# Example: topic "orders", partition 0
#
# Log End Offset (LEO):     847,500  ← latest message written
# Consumer Committed:       847,200  ← last processed by consumer
# Consumer Lag:                 300  ← 300 messages behind
#
# Lag over time:
#   09:00  ████ 50
#   09:10  ██████ 80
#   09:20  ████████████ 200      ← traffic spike
#   09:30  ██████████████████ 350 ← consumer falling behind!
#   09:40  ████████████████ 300   ← auto-scaled, recovering
#   09:50  ██████ 100             ← catching up
#   10:00  ██ 20                  ← healthy again
#
# Alert if:
# - Lag is monotonically increasing for > 5 minutes
# - Lag > 10,000 for any critical topic partition
# - Lag age > 5 minutes (oldest unconsumed message)`} />
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
            { mode: "Consumer Rebalance Storm", impact: "HIGH", desc: "Consumer crashes, triggers rebalance. During rebalance, processing stops. A slow consumer times out, triggers another rebalance. Loop continues — no messages processed.",
              mitigation: "Increase session.timeout.ms (45s). Use static group membership (group.instance.id). Use cooperative sticky rebalance (not eager). Monitor rebalance frequency and alert on > 5/hour.",
              example: "Deploy updated consumer code. Pod takes 40s to start (default session timeout = 30s). Every pod restart triggers a rebalance. 10-pod rolling deploy = 10 rebalances in 5 minutes. No messages processed for 8 minutes." },
            { mode: "Unclean Leader Election (Data Loss)", impact: "CRITICAL", desc: "Leader and all ISR replicas die (e.g., rack power failure). With unclean.leader.election=true, an out-of-sync replica becomes leader. Messages not yet replicated are permanently lost.",
              mitigation: "Set unclean.leader.election.enable=false (default since Kafka 0.11). Use min.insync.replicas=2 with acks=all. Place replicas across racks/AZs. Accept partition unavailability over data loss.",
              example: "Rack A hosts leader and one follower for partition 0. Rack A loses power. Third replica on Rack B is 500 messages behind. Unclean election promotes it — 500 messages permanently lost. Downstream systems miss 500 orders." },
            { mode: "Poison Pill (Bad Message)", impact: "HIGH", desc: "A malformed message causes the consumer to crash on deserialization. Consumer restarts, re-reads the same message, crashes again. Infinite crash loop. The partition is stuck.",
              mitigation: "Catch and log deserialization errors instead of crashing. Send bad messages to a dead-letter topic (DLT). Use consumer error handler: skip after N retries. Schema validation on producer side (Avro + Schema Registry).",
              example: "Producer sends a message with a field type change (string → int). Consumer's JSON parser throws NumberFormatException. Consumer crashes, restarts, reads the same offset, crashes again. 47 restarts in 10 minutes." },
            { mode: "Partition Skew (Hot Partition)", impact: "MEDIUM", desc: "One partition receives 80% of traffic because a popular key (e.g., user_id of a whale customer) hashes to that partition. One consumer is overloaded while others are idle.",
              mitigation: "Use a composite partition key (user_id + random suffix) for high-cardinality distribution. Monitor per-partition throughput. For known hot keys, route to a dedicated 'overflow' partition. Custom partitioner.",
              example: "E-commerce platform: Walmart (one merchant) generates 60% of all order events. All Walmart orders have the same partition key 'merchant_walmart'. One partition gets 60% of all messages. Consumer for that partition is perpetually behind." },
            { mode: "Disk Full — Broker Goes Read-Only", impact: "CRITICAL", desc: "Log retention misconfigured (7 days, but topic produces 100GB/day = 700GB). Disk fills up. Broker stops accepting writes. Producers get errors. If multiple brokers fill up, topics go offline.",
              mitigation: "Monitor disk usage aggressively (alert at 70%, page at 85%). Set log.retention.bytes as a safety cap alongside time-based retention. Enable tiered storage for cold data. Automated alerting before disks fill.",
              example: "New high-throughput topic created with 30-day retention. Nobody noticed it produces 50GB/day. After 20 days: 1TB consumed. Broker A disk 95% full. Emergency retention reduction to 3 days and frantic segment deletion." },
            { mode: "Producer Ordering Violation", impact: "HIGH", desc: "Producer sends messages M1, M2, M3. M1 fails and is retried. With max.in.flight.requests > 1, M2 and M3 may be written before M1's retry succeeds. Result: M2, M3, M1 — out of order.",
              mitigation: "Enable idempotence (enable.idempotence=true), which sets max.in.flight.requests.per.connection=5 with sequence numbers. Broker rejects out-of-order writes. Or set max.in.flight=1 (lower throughput).",
              example: "Financial transaction log requires strict ordering. Producer retries a failed write. Meanwhile, next two messages are sent and written. Downstream sees: txn_102, txn_103, txn_101. Reconciliation fails." },
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
          <Label color="#0f766e">Component Breakdown</Label>
          <div className="space-y-3">
            {[
              { name: "Broker (Data Plane)", role: "Stores partitions on disk. Handles produce requests (append to log), fetch requests (serve to consumers), and replication (fetch from leader). The workhorse.", tech: "Kafka Broker (JVM) / Redpanda (C++)", critical: true },
              { name: "Controller (Control Plane)", role: "Manages cluster metadata: topic creation, partition assignment, leader election, ISR updates, broker health. Exactly one active controller.", tech: "ZooKeeper (legacy) / KRaft (Kafka 3.3+)", critical: true },
              { name: "Producer Client", role: "Serializes messages, selects partition (hash of key), batches messages, compresses, sends to partition leader. Handles retries and idempotence.", tech: "librdkafka / Kafka Java client", critical: true },
              { name: "Consumer Client", role: "Subscribes to topics, polls for messages, deserializes, processes, commits offsets. Manages group membership and heartbeat.", tech: "librdkafka / Kafka Java client", critical: true },
              { name: "Group Coordinator", role: "A broker that manages consumer group membership, triggers rebalances, assigns partitions. Each group has one coordinator (hashed to a broker).", tech: "Built into Kafka broker", critical: true },
              { name: "Schema Registry", role: "Stores and validates message schemas (Avro, Protobuf, JSON Schema). Ensures producers and consumers agree on data format. Prevents poison pills.", tech: "Confluent Schema Registry", critical: false },
              { name: "Kafka Connect", role: "Connector framework for importing data from external systems (CDC from databases) and exporting to sinks (Elasticsearch, S3).", tech: "Source + Sink connectors", critical: false },
              { name: "MirrorMaker / Replicator", role: "Cross-cluster replication for disaster recovery and multi-region. Replicates topics from primary cluster to secondary.", tech: "MirrorMaker 2 / Confluent Replicator", critical: false },
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
          <Label color="#9333ea">Broker Internals — Block Diagram</Label>
          <svg viewBox="0 0 380 320" className="w-full">
            <rect x={10} y={10} width={360} height={40} rx={6} fill="#2563eb08" stroke="#2563eb" strokeWidth={1}/>
            <text x={190} y={28} textAnchor="middle" fill="#2563eb" fontSize="10" fontWeight="600" fontFamily="monospace">Network Layer (Acceptor + Processor Threads)</text>
            <text x={190} y={42} textAnchor="middle" fill="#2563eb80" fontSize="8" fontFamily="monospace">TCP connections · request/response queue · SSL/SASL</text>

            <rect x={10} y={60} width={175} height={50} rx={6} fill="#9333ea08" stroke="#9333ea" strokeWidth={1}/>
            <text x={97} y={80} textAnchor="middle" fill="#9333ea" fontSize="10" fontWeight="600" fontFamily="monospace">Request Handler</text>
            <text x={97} y={95} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">Produce · Fetch · Metadata</text>
            <text x={97} y={105} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">Offsets · GroupCoordinator</text>

            <rect x={195} y={60} width={175} height={50} rx={6} fill="#d9770608" stroke="#d97706" strokeWidth={1}/>
            <text x={282} y={80} textAnchor="middle" fill="#d97706" fontSize="10" fontWeight="600" fontFamily="monospace">Replication Manager</text>
            <text x={282} y={95} textAnchor="middle" fill="#d9770680" fontSize="8" fontFamily="monospace">ISR tracking · HW update</text>
            <text x={282} y={105} textAnchor="middle" fill="#d9770680" fontSize="8" fontFamily="monospace">Follower fetch · Leader epoch</text>

            <rect x={10} y={120} width={360} height={50} rx={6} fill="#dc262608" stroke="#dc2626" strokeWidth={1}/>
            <text x={190} y={140} textAnchor="middle" fill="#dc2626" fontSize="10" fontWeight="600" fontFamily="monospace">Log Manager (Partition Logs)</text>
            <text x={100} y={158} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">Segment files · Index</text>
            <text x={280} y={158} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">Compaction · Retention</text>

            <rect x={10} y={180} width={175} height={45} rx={6} fill="#05966908" stroke="#059669" strokeWidth={1}/>
            <text x={97} y={200} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="600" fontFamily="monospace">Page Cache (OS)</text>
            <text x={97} y={215} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">recent data in memory</text>

            <rect x={195} y={180} width={175} height={45} rx={6} fill="#0891b208" stroke="#0891b2" strokeWidth={1}/>
            <text x={282} y={200} textAnchor="middle" fill="#0891b2" fontSize="10" fontWeight="600" fontFamily="monospace">Disk (NVMe SSD)</text>
            <text x={282} y={215} textAnchor="middle" fill="#0891b280" fontSize="8" fontFamily="monospace">segment files · sequential I/O</text>

            <rect x={10} y={235} width={360} height={35} rx={6} fill="#6366f108" stroke="#6366f1" strokeWidth={1}/>
            <text x={190} y={255} textAnchor="middle" fill="#6366f1" fontSize="10" fontWeight="600" fontFamily="monospace">Controller / KRaft Consensus</text>
            <text x={190} y={265} textAnchor="middle" fill="#6366f180" fontSize="8" fontFamily="monospace">metadata · leader election · partition assignment</text>

            <rect x={10} y={280} width={360} height={30} rx={6} fill="#78716c08" stroke="#78716c" strokeWidth={1}/>
            <text x={190} y={298} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Metrics (JMX) · Audit Log · ACLs</text>

            <line x1={190} y1={50} x2={97} y2={60} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={190} y1={50} x2={282} y2={60} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={190} y1={110} x2={190} y2={120} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={97} y1={170} x2={97} y2={180} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={282} y1={170} x2={282} y2={180} stroke="#94a3b8" strokeWidth={1}/>
          </svg>
        </Card>
      </div>
    </div>
  );
}

function FlowsSection() {
  const [flow, setFlow] = useState("produce");
  const flows = {
    produce: {
      title: "Produce Message (acks=all)",
      steps: [
        { actor: "Producer", action: "Serialize message, hash key 'user_123' → partition 3. Add to batch for partition 3.", type: "request" },
        { actor: "Producer", action: "Batch reaches batch.size (64KB) or linger.ms (5ms) expires. Compress batch with LZ4.", type: "process" },
        { actor: "Producer", action: "Send ProduceRequest to broker hosting partition 3 leader (Broker 2).", type: "request" },
        { actor: "Broker 2 (Leader)", action: "Validate: topic exists, partition is led by this broker, auth check, message CRC.", type: "auth" },
        { actor: "Broker 2 (Leader)", action: "Append batch to active segment file. Assign offsets 847291-847305 (15 messages). Update LEO.", type: "process" },
        { actor: "Broker 1 (Follower)", action: "FetchRequest to leader: 'give me data from offset 847291'. Receives batch, appends locally.", type: "process" },
        { actor: "Broker 3 (Follower)", action: "FetchRequest to leader: same fetch, appends locally. Now all 3 replicas have the data.", type: "process" },
        { actor: "Broker 2 (Leader)", action: "All ISR replicas acknowledged. Advance high watermark to 847305. Batch is now committed.", type: "success" },
        { actor: "Broker 2", action: "Send ProduceResponse: {partition: 3, offset: 847291, timestamp: ...}. Total latency: ~8ms.", type: "success" },
      ]
    },
    consume: {
      title: "Consume Messages (Consumer Group)",
      steps: [
        { actor: "Consumer", action: "consumer.subscribe(['orders']). Send JoinGroupRequest to group coordinator (Broker 1).", type: "request" },
        { actor: "Coordinator", action: "Consumer joins group 'order-processor'. Trigger rebalance. Assign partitions [3,4] to this consumer.", type: "process" },
        { actor: "Consumer", action: "consumer.poll(timeout=1000ms). Send FetchRequest to partition 3 leader: 'give me data from offset 847200'.", type: "request" },
        { actor: "Broker 2 (Leader)", action: "Read from offset 847200. Data is in page cache (recent). Zero-copy transfer to network socket.", type: "process" },
        { actor: "Consumer", action: "Receive 500 records (max.poll.records). Deserialize. Process each record (business logic).", type: "process" },
        { actor: "Consumer", action: "All 500 records processed successfully. consumer.commit() → offset 847700.", type: "success" },
        { actor: "Coordinator", action: "Store committed offset in __consumer_offsets topic: {group, topic, partition:3, offset:847700}.", type: "success" },
        { actor: "Consumer", action: "Next poll(): fetch from offset 847700. Repeat cycle.", type: "request" },
      ]
    },
    leaderElection: {
      title: "Leader Failure → Election",
      steps: [
        { actor: "Broker 2 (Leader)", action: "Hosting leader for partition 'orders-3'. Suddenly crashes (hardware failure).", type: "error" },
        { actor: "Controller", action: "ZooKeeper/KRaft detects Broker 2 session expired (18s timeout). Broker 2 marked as dead.", type: "check" },
        { actor: "Controller", action: "For each partition led by Broker 2: select new leader from ISR. Partition 3 ISR = [1, 3]. Choose Broker 1.", type: "process" },
        { actor: "Controller", action: "Update metadata: partition 3 leader = Broker 1. Propagate to all brokers via UpdateMetadata.", type: "process" },
        { actor: "Broker 1 (New Leader)", action: "Now accepting produce and fetch requests for partition 3. High watermark = last replicated offset.", type: "success" },
        { actor: "Producers", action: "Get 'NOT_LEADER' error on next request to Broker 2. Refresh metadata → discover Broker 1 is new leader.", type: "check" },
        { actor: "Consumers", action: "FetchRequest to Broker 2 fails. Metadata refresh → redirect to Broker 1. Resume from committed offset.", type: "check" },
        { actor: "Broker 3 (Follower)", action: "Now fetching from Broker 1 (new leader) instead of Broker 2. Catches up to new leader's LEO.", type: "process" },
        { actor: "Note", action: "Total disruption: 18s (detection) + 2s (election) ≈ 20s. Producers buffer during this time.", type: "check" },
      ]
    },
    rebalance: {
      title: "Consumer Rebalance (Scale Out)",
      steps: [
        { actor: "Initial State", action: "Group 'order-processor': C1 owns [P0,P1,P2], C2 owns [P3,P4,P5]. 6 partitions, 2 consumers.", type: "process" },
        { actor: "C3 (New Consumer)", action: "Starts up, sends JoinGroupRequest to group coordinator with group_id='order-processor'.", type: "request" },
        { actor: "Coordinator", action: "Triggers cooperative rebalance. Determines new assignment: C1=[P0,P1], C2=[P3,P4], C3=[P2,P5].", type: "process" },
        { actor: "Coordinator", action: "Phase 1: Tell C1 to revoke P2. Tell C2 to revoke P5. C1 and C2 stop consuming revoked partitions.", type: "check" },
        { actor: "C1, C2", action: "Commit offsets for revoked partitions. Continue consuming remaining partitions (P0,P1 and P3,P4).", type: "success" },
        { actor: "Coordinator", action: "Phase 2: Assign P2 to C3, P5 to C3. C3 starts consuming from last committed offsets.", type: "process" },
        { actor: "C3", action: "Fetch from P2 at offset 123456, P5 at offset 789012. Begin processing. Group is rebalanced.", type: "success" },
        { actor: "Note", action: "Cooperative rebalance: only moved partitions pause. C1 and C2 never stopped consuming P0,P1,P3,P4.", type: "check" },
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
              <span className="text-[11px] font-mono font-bold shrink-0 w-40" style={{ color: colors[s.type] }}>{s.actor}</span>
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
            <Point icon="1." color="#b45309"><strong className="text-stone-700">Rolling restart (one broker at a time)</strong> — stop broker, wait for leader elections to complete (~20s), upgrade, restart. Other brokers handle traffic. Zero downtime.</Point>
            <Point icon="2." color="#b45309"><strong className="text-stone-700">Pre-check before restart</strong> — verify under-replicated partitions = 0 before stopping next broker. Don't stop a broker if cluster is already degraded.</Point>
            <Point icon="3." color="#b45309"><strong className="text-stone-700">Controlled shutdown</strong> — broker.controlled.shutdown.enable=true. Broker migrates leaders to other replicas before shutting down. Reduces failover latency.</Point>
            <Point icon="4." color="#b45309"><strong className="text-stone-700">Config-as-code</strong> — topic configs, ACLs, and quotas managed via GitOps (terraform, topicctl). No manual topic creation in production.</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Security</Label>
          <ul className="space-y-2.5">
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">TLS encryption</strong> — encrypt all broker-to-broker (inter-broker) and client-to-broker traffic. Prevents eavesdropping on message content.</Point>
            <Point icon="🔑" color="#dc2626"><strong className="text-stone-700">SASL authentication</strong> — clients authenticate with SASL/SCRAM or mTLS certificates. No unauthenticated access to any topic.</Point>
            <Point icon="🛡️" color="#dc2626"><strong className="text-stone-700">ACLs (authorization)</strong> — fine-grained access control: which principals can produce/consume to which topics. Principle of least privilege.</Point>
            <Point icon="📝" color="#dc2626"><strong className="text-stone-700">Quotas</strong> — per-client produce and consume byte rate limits. Prevent a runaway producer from overwhelming the cluster. Per-broker and per-topic quotas.</Point>
            <Point icon="🧱" color="#dc2626"><strong className="text-stone-700">Network isolation</strong> — brokers on private network. Clients access via load balancer or VPN. No public internet exposure.</Point>
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
                { trigger: "Consumer Lag", thresh: "&gt; 100K msgs for 5min", action: "Scale consumer group (+50% pods). Check for slow downstream.", cool: "3 min", pitfall: "More consumers than partitions = idle consumers. May need to add partitions first." },
                { trigger: "Under-Replicated", thresh: "&gt; 0 for 10min", action: "P0 Alert. Check broker health, disk I/O, network. Potential data loss risk.", cool: "0 (immed.)", pitfall: "Could be a slow broker catching up after restart. Check if it's recovering." },
                { trigger: "Disk Usage", thresh: "&gt; 75% on any broker", action: "Reduce retention, add disks, or add brokers and rebalance partitions.", cool: "1 hour", pitfall: "Adding brokers doesn't help until partitions are moved. Rebalance takes hours." },
                { trigger: "Produce Latency", thresh: "p99 &gt; 50ms for 5min", action: "Check ISR health (acks=all waits for slowest ISR). Check disk I/O.", cool: "5 min", pitfall: "Latency spike could be caused by large messages, not broker issues." },
                { trigger: "Offline Partitions", thresh: "&gt; 0", action: "P0 Alert. Partition has no leader. Check if ISR is empty. May need manual intervention.", cool: "0 (immed.)", pitfall: "If unclean election is disabled and all ISR dead, partition stays offline until recovery." },
                { trigger: "Network Throughput", thresh: "&gt; 80% NIC capacity", action: "Add brokers and redistribute partitions. Or add NIC bandwidth.", cool: "30 min", pitfall: "Replication doubles network usage. 1 GB/s produce = 2 GB/s total (with RF=3)." },
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
            { title: "The 10,000 Partition Meltdown", symptom: "Controller takes 4 minutes to elect leaders after a broker dies. During this time, thousands of partitions are leaderless. Producers error out.",
              cause: "Cluster had 10,000 partitions across 5 brokers. One broker dying means ~2,000 leader elections. Controller processes elections sequentially. 2,000 × 100ms = 200s. System was effectively down for 3+ minutes.",
              fix: "Reduce to 2,000 partitions (many topics were over-partitioned). Spread across more brokers (5→15). Use KRaft instead of ZooKeeper for faster elections. Rule: no more than 4,000 partitions per broker.",
              quote: "We thought more partitions = more throughput. Turned out 10K partitions on 5 brokers meant every broker failure was a 4-minute outage. Less is more." },
            { title: "Consumer Lag That Never Recovers", symptom: "Consumer lag for 'events' topic grows continuously at 500 msg/s. Consumer group has 6 consumers and 6 partitions (fully utilized). Can't add more consumers.",
              cause: "Each message triggers a synchronous HTTP call to an external API (50ms per call). 6 consumers × 20 msg/s = 120 msg/s processing. Production rate: 620 msg/s. Delta of 500 msg/s accumulates as lag.",
              fix: "Async HTTP calls with a thread pool (10 threads per consumer): 6 × 200 = 1,200 msg/s capacity. Also increased partitions from 6 to 12 and scaled to 12 consumers. Lag recovered in 2 hours.",
              quote: "The 'consumer lag keeps growing' ticket took us 3 days to figure out. It was literally a synchronous HTTP call in a loop. 50ms × message = 20 msg/s. Math doesn't lie." },
            { title: "The Poison Pill That Crashed 47 Times", symptom: "Consumer pod restarts 47 times in 10 minutes. Kubernetes gives up and goes into CrashLoopBackOff. All 6 partitions for that group stop processing.",
              cause: "A producer sent a message with a new field type (changed from string to integer). Consumer's protobuf deserializer threw an exception. Consumer crashed, restarted, re-read the same offset, crashed again. Infinite loop.",
              fix: "Added try/catch around deserialization. Failed messages sent to a dead-letter topic (DLT) for inspection. Added schema validation at the producer using Schema Registry with compatibility checks. Added max-restart circuit breaker.",
              quote: "One bad message took down an entire consumer group for 10 minutes. Now we have a rule: if you can't parse it, DLT it. Never crash on bad data." },
            { title: "Rebalance Storm During Deploy", symptom: "Rolling deploy of 12-consumer group. Each pod restart triggers a rebalance. 12 restarts × 30s rebalance = 6 minutes of no consumption. SLA missed.",
              cause: "Default session.timeout.ms=10s. Pod graceful shutdown takes 15s (draining in-flight requests). Old pod times out before new pod starts. Every restart = 2 rebalances (leave + join).",
              fix: "Static group membership: set group.instance.id per pod (stable identity). Increased session.timeout.ms=45s. Cooperative sticky rebalance instead of eager. Deploy time dropped from 6 min to 45s total disruption.",
              quote: "We went from '12 deploys = 24 rebalances = 6 minutes of downtime' to '12 deploys = 0 rebalances = 0 downtime.' Static membership is the single biggest consumer improvement." },
            { title: "The Disk That Ate Everything", symptom: "Broker 3 disk at 98%. Produces to any partition on Broker 3 fail with 'KAFKA_STORAGE_ERROR'. 33% of cluster capacity gone.",
              cause: "A new topic 'raw-events' was created with 30-day retention and no size limit. It produced 80 GB/day. After 25 days: 2 TB consumed. Nobody was monitoring per-topic disk usage, only overall disk.",
              fix: "Added per-topic disk usage monitoring. Set log.retention.bytes=500GB as a safety cap on all topics. Automated alert when any topic exceeds 100GB. Reduced 'raw-events' retention to 3 days and archived to S3.",
              quote: "Our 'infinite retention' policy wasn't infinite — it was 'until the disk fills up and everything breaks.' Now every topic has both time AND size limits." },
            { title: "Split-Brain: Two Leaders, One Partition", symptom: "Producers on different clients are writing to different brokers for the same partition. Consumers see interleaved data from two log segments. Data corruption.",
              cause: "Network partition between ZooKeeper and Broker 2 (old leader). ZooKeeper elects new leader (Broker 1). But Broker 2 didn't get the memo — it still thinks it's leader. Both accept writes.",
              fix: "Enable epoch fencing: each leader has a monotonically increasing epoch number. Followers reject data from leaders with stale epochs. Migration to KRaft (Kafka's built-in consensus) eliminates ZooKeeper dependency and this class of bugs.",
              quote: "Two leaders for the same partition. Both writing different data. Both thinking they're right. Took us 3 hours to figure out and 2 days to reconcile the data. Epoch fencing is now mandatory." },
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
        { t: "Exactly-Once Semantics (EOS)", d: "Guarantee that each message is processed exactly once, end-to-end. No duplicates, no losses. Uses idempotent producer + transactional API.", detail: "Producer assigns sequence numbers to messages. Broker deduplicates. Transactional API: atomic writes across multiple partitions + consumer offset commit. Commit or abort — all or nothing. Used for stream processing (Kafka Streams).", effort: "Hard" },
        { t: "Tiered Storage (Infinite Retention)", d: "Store recent data on local SSDs, old data in object storage (S3). Enables months/years of retention at $0.02/GB instead of $0.10/GB on SSD.", detail: "Active segments stay local. Closed segments offloaded to S3 asynchronously. Consumer transparently reads from either tier. Index in local storage maps offsets to S3 objects. Kafka 3.6+ KIP-405.", effort: "Hard" },
        { t: "Schema Registry + Evolution", d: "Centralized schema store (Avro/Protobuf/JSON Schema). Producers register schemas. Consumers fetch schemas for deserialization. Compatibility checks prevent breaking changes.", detail: "Schema ID embedded in message header (4 bytes). Consumer looks up schema by ID from registry (cached). Backward/forward/full compatibility modes. Prevents poison pills from schema mismatches.", effort: "Medium" },
        { t: "Dead Letter Topic (DLT)", d: "Messages that fail processing after N retries are sent to a DLT for manual inspection. Prevents poison pills from blocking the consumer.", detail: "Consumer catches processing errors, retries 3× with backoff. On final failure, produce message to 'topic.DLT' with error metadata. Dashboard for DLT monitoring, manual replay, and purging.", effort: "Easy" },
        { t: "Multi-Region Replication", d: "Replicate topics across data centers for disaster recovery and geo-local consumption. MirrorMaker 2 or Confluent Replicator.", detail: "Active-passive: primary cluster processes writes, secondary is read-only backup. Active-active: both clusters accept writes, conflict resolution needed. Offset translation for consumer failover between clusters.", effort: "Hard" },
        { t: "Stream Processing (Kafka Streams)", d: "Process data in real-time as it flows through topics. Stateful operations: joins, aggregations, windowing. No separate cluster needed — runs as a library in your app.", detail: "KStream (event stream) and KTable (changelog). Windowed aggregations for metrics (count orders per minute). State stored in local RocksDB with changelog topic for fault tolerance.", effort: "Medium" },
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
    { q: "How does Kafka achieve such high throughput?", a: "Four key techniques: (1) Sequential I/O — append-only log leverages disk sequential write speed (~600 MB/s on SSD vs ~1 MB/s random). No seek overhead. (2) Zero-copy — sendfile() syscall transfers data from page cache directly to NIC, bypassing user space. Consumer reads at near-network speed with zero CPU. (3) Batching — producer accumulates messages into batches (batch.size + linger.ms). One network round-trip sends hundreds of messages. (4) Compression — LZ4/ZStandard compresses batches 2-4×. Less network traffic, less disk I/O. The combination of all four is why Kafka does millions of messages per second on commodity hardware.", tags: ["design"] },
    { q: "Kafka vs RabbitMQ — when to use which?", a: "Kafka: high-throughput event streaming, log-based (retain messages), pull-based consumers, replay capability, consumer groups for parallel processing. Best for: event sourcing, stream processing, CDC, analytics pipelines, high-volume logs. RabbitMQ: traditional message broker, push-based, message deleted after ACK, flexible routing (exchanges: fanout, topic, headers), per-message priority. Best for: task queues, RPC-style messaging, complex routing patterns, lower volume with rich delivery semantics. Rule of thumb: if you need replay, event sourcing, or millions/s → Kafka. If you need flexible routing, priority queues, or simple task distribution → RabbitMQ.", tags: ["design"] },
    { q: "How do you handle message ordering across partitions?", a: "You can't — and that's by design. Kafka only guarantees ordering within a single partition. Cross-partition ordering would require global coordination, destroying throughput. Solutions: (1) Single partition — total order, but throughput limited to one partition. Only for very low-volume topics. (2) Partition key — messages with the same key always go to the same partition. All events for user_123 are ordered. Different users may be in different partitions (and that's fine). (3) External sequencing — embed a sequence number in the message. Consumer buffers and reorders. Complex and adds latency. In practice, partition-key ordering covers 95% of use cases.", tags: ["design"] },
    { q: "What happens when a consumer is slower than the producer?", a: "Consumer lag grows — the gap between the log end offset and the consumer's committed offset widens. Options: (1) Scale consumers — add more consumers to the group (up to partition count). (2) Increase partitions — if you've maxed out consumers, add partitions and then consumers. (3) Optimize processing — batch DB writes, async external calls, parallel processing threads. (4) Backpressure — rate-limit producers (less common, last resort). (5) Accept the lag — if the consumer catches up eventually (batch processing), growing lag is temporary and acceptable. The critical thing to monitor: is lag growing, stable, or shrinking? Growing = problem. Stable = normal. Shrinking = recovering.", tags: ["scalability"] },
    { q: "How does exactly-once semantics (EOS) work?", a: "Three components: (1) Idempotent producer — each message gets a sequence number per partition. Broker deduplicates: if it sees sequence N twice, it discards the duplicate. Prevents duplicate writes on retries. (2) Transactional API — producer starts a transaction, writes to multiple partitions, commits or aborts atomically. Consumer reads only committed messages (isolation.level=read_committed). (3) Consumer offset commit — offset commit is included in the transaction. Process + commit is atomic. If processing fails, both message output AND offset commit are rolled back. Limitation: EOS works within a single Kafka cluster. Cross-system exactly-once (Kafka → database) requires the Outbox pattern or two-phase commit.", tags: ["algorithm"] },
    { q: "How do you monitor and reduce consumer lag?", a: "Monitoring: (1) kafka-consumer-groups.sh --describe shows lag per partition. (2) Burrow (LinkedIn's tool) monitors lag rate of change, not just absolute value. (3) Export lag metrics to Prometheus/Datadog, alert on trends. Reducing lag: (1) Scale consumers (most common fix). (2) Increase max.poll.records for bigger batches. (3) Parallelize processing within consumer (thread pool after dequeue). (4) Reduce per-message processing time (async external calls, batch DB writes). (5) If lag is caused by a slow partition, check for hot keys. Key insight: don't just alert on lag value — alert on lag velocity (rate of change). Lag of 10K that's shrinking is fine. Lag of 1K that's growing is a problem.", tags: ["observability"] },
    { q: "How does Kafka handle backpressure?", a: "Kafka's pull-based model handles backpressure naturally: consumers poll at their own pace. If they're slow, lag grows but the broker is unaffected. For producers: (1) buffer.memory limits in-memory buffer size. If buffer fills (consumer of the network can't keep up), producer.send() blocks or throws. (2) max.block.ms configures how long send() blocks before throwing. (3) Broker-side: quota system limits per-client produce rate. If a client exceeds its quota, the broker throttles (delays) its responses. Unlike push-based systems (RabbitMQ), Kafka doesn't need explicit flow control — the consumer simply polls less frequently when it's overwhelmed.", tags: ["scalability"] },
    { q: "What is log compaction and when should you use it?", a: "Log compaction retains only the latest message per key, deleting older versions. The log becomes like a key-value table. Use cases: (1) CDC (Change Data Capture) — capture every change to a database row. Compacted topic = current state of every row. New consumer reads compacted log = snapshot of the entire database. (2) Materialized views — maintain a cache that reflects the latest state of each entity. On startup, consumer reads compacted topic to rebuild cache. (3) Configuration/state store — latest config per service. When NOT to use: event streams where every event matters (clickstream, logs, financial transactions). Compaction discards old events, keeping only the latest per key.", tags: ["data"] },
    { q: "How would you design a multi-region Kafka setup?", a: "Two patterns: (1) Active-Passive — primary cluster in Region A handles all writes. MirrorMaker 2 replicates to Region B (async). On Region A failure, promote Region B. Challenge: replication lag means some messages may be lost. RPO = replication lag. (2) Active-Active — both regions accept writes to local topics. MirrorMaker 2 replicates bidirectionally. Challenge: conflict resolution for same-key writes in both regions. Topic naming: 'us.orders' and 'eu.orders' avoid conflicts. Consumer reads both. In practice, most use active-passive with careful monitoring of replication lag.", tags: ["availability"] },
    { q: "ZooKeeper vs KRaft — what changed?", a: "ZooKeeper was Kafka's external dependency for metadata, leader election, and cluster coordination. Problems: (1) Separate system to deploy, monitor, and secure. (2) ZooKeeper watches don't scale well with thousands of partitions. (3) Controller failover is slow (new controller must reload all metadata from ZK). KRaft (KIP-500, GA in Kafka 3.3): (1) Metadata stored in an internal Kafka topic (__cluster_metadata) using Raft consensus. (2) No external dependency. (3) Faster controller failover (metadata already in memory). (4) Supports millions of partitions (ZK struggles past 200K). Migration from ZK to KRaft is supported since Kafka 3.6. New clusters should always use KRaft.", tags: ["design"] },
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

export default function DistributedQueueSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Distributed Queue (Kafka)</h1>
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