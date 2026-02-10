import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   PAYMENT SYSTEM (STRIPE) — System Design Reference
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
            <Label>What is a Payment System?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A payment system is the infrastructure that enables the secure transfer of money between buyers, merchants, and financial institutions. It orchestrates the complex flow of authorization, capture, settlement, and reconciliation — ensuring every dollar is accounted for, exactly once, even in the face of network failures, fraud, and regulatory constraints.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Think of it like a postal system for money: the sender (customer) puts money in an envelope (payment intent), the post office (payment processor) verifies the address and stamps (authorization), the carrier (card network) delivers it, and the recipient (merchant) gets a receipt (settlement). But unlike mail, money must arrive exactly once — never lost, never duplicated.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is It Hard?</Label>
            <ul className="space-y-2.5">
              <Point icon="⚡" color="#0891b2">Exactly-once semantics — a payment must never be charged twice or lost; idempotency is non-negotiable</Point>
              <Point icon="🛡️" color="#0891b2">PCI DSS compliance — card data must be encrypted, tokenized, and handled in a certified environment</Point>
              <Point icon="📈" color="#0891b2">High availability — payment downtime = direct revenue loss; even 1 minute of outage costs millions for large merchants</Point>
              <Point icon="🌍" color="#0891b2">Multi-currency, multi-region — support 135+ currencies, local payment methods, cross-border regulations</Point>
              <Point icon="🔒" color="#0891b2">Fraud detection — block fraudulent charges in real-time without blocking legitimate customers</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Examples</Label>
            <div className="space-y-2.5">
              {[
                { co: "Stripe", rule: "Full-stack payments API, 135+ currencies", algo: "Intent-based, 2-phase" },
                { co: "PayPal", rule: "Wallet + merchant gateway, 400M users", algo: "Account-based" },
                { co: "Adyen", rule: "Enterprise payments, single platform", algo: "Unified commerce" },
                { co: "Square", rule: "POS + online, SMB focus", algo: "Omnichannel" },
                { co: "Visa/MC", rule: "Card network (rails), not processor", algo: "4-party model" },
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
            <Label color="#2563eb">The 4-Party Model</Label>
            <svg viewBox="0 0 360 200" className="w-full">
              <DiagramBox x={60} y={40} w={80} h={34} label="Customer" color="#2563eb"/>
              <DiagramBox x={300} y={40} w={80} h={34} label="Merchant" color="#059669"/>
              <DiagramBox x={60} y={130} w={80} h={38} label="Issuing\nBank" color="#9333ea"/>
              <DiagramBox x={300} y={130} w={80} h={38} label="Acquiring\nBank" color="#d97706"/>
              <DiagramBox x={180} y={130} w={70} h={34} label="Card\nNetwork" color="#dc2626"/>
              <Arrow x1={100} y1={40} x2={260} y2={40} label="pays" id="p1"/>
              <Arrow x1={60} y1={57} x2={60} y2={111} label="issued card" id="p2"/>
              <Arrow x1={300} y1={57} x2={300} y2={111} label="acquires" id="p3"/>
              <Arrow x1={100} y1={130} x2={145} y2={130} id="p4"/>
              <Arrow x1={215} y1={130} x2={260} y2={130} id="p5"/>
              <rect x={90} y={172} width={180} height={18} rx={4} fill="#dc262608" stroke="#dc262630"/>
              <text x={180} y={182} textAnchor="middle" fill="#dc2626" fontSize="8" fontFamily="monospace">Visa / Mastercard / Amex rails</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Stripe, Square, PayPal, Amazon, Google, Uber</div>
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
            <p className="text-[12px] text-stone-500 mt-0.5">"Design a payment system" is extremely broad. Clarify immediately: are we designing the <strong>payment gateway/processor</strong> (like Stripe's API), the <strong>merchant integration layer</strong>, or the <strong>internal ledger/settlement system</strong>? For a 45-min interview, focus on <strong>charge creation → authorization → capture → settlement</strong> with idempotency and exactly-once guarantees. Fraud, subscriptions, and multi-currency are follow-ups.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Accept payments — charge a customer's card/bank for a given amount and currency</Point>
            <Point icon="2." color="#059669">Two-phase flow — authorize (hold funds) then capture (collect funds) separately</Point>
            <Point icon="3." color="#059669">Refunds — full or partial refund of a captured payment</Point>
            <Point icon="4." color="#059669">Idempotency — same request retried N times produces exactly one charge</Point>
            <Point icon="5." color="#059669">Webhooks — notify merchants of payment status changes asynchronously</Point>
            <Point icon="6." color="#059669">Payment methods — support cards (Visa, MC, Amex), bank transfers, wallets (Apple Pay, Google Pay)</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Exactly-once processing — never double-charge, never lose a payment</Point>
            <Point icon="2." color="#dc2626">High availability — 99.999% uptime (5 minutes downtime/year)</Point>
            <Point icon="3." color="#dc2626">Low latency — authorization response in &lt;2 seconds</Point>
            <Point icon="4." color="#dc2626">PCI DSS Level 1 compliance — card data encrypted at rest and in transit</Point>
            <Point icon="5." color="#dc2626">Auditability — every state change logged immutably for compliance and reconciliation</Point>
            <Point icon="6." color="#dc2626">Consistency — ledger must always balance; eventual consistency is NOT acceptable for money</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask the Interviewer</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Are we building the processor (like Stripe) or the merchant integration?",
            "Which payment methods? Cards only, or bank transfers and wallets too?",
            "Do we need a two-phase auth/capture or single-step charge?",
            "Multi-currency and cross-border? Or single-currency?",
            "Do we need subscriptions/recurring billing?",
            "What's the expected TPS? Hundreds or millions of transactions/day?",
            "Do we need built-in fraud detection or external (Sift, Riskified)?",
            "Payout/settlement to merchants? Or just the charge side?",
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
            <div className="text-[12px] font-bold text-violet-700">Interview Tip — Money Math Matters</div>
            <p className="text-[12px] text-stone-500 mt-0.5">Payment capacity estimation is about <strong>TPS (transactions per second)</strong> and <strong>total payment volume (TPV)</strong>, not just storage. Focus on: peak TPS, latency budget per hop, ledger row growth, and audit log volume. Show you understand that payment workloads are write-heavy and consistency-critical.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Transaction Volume</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Active merchants on platform" result="500K" note="Mid-large payment platform (Stripe-scale)" />
            <MathStep step="2" formula="Avg transactions per merchant/day" result="~200" note="Wide range: SMB (10) to enterprise (100K)" />
            <MathStep step="3" formula="Total daily transactions = 500K × 200" result="100M/day" note="100 million payment attempts per day" />
            <MathStep step="4" formula="Avg TPS = 100M / 86,400" result="~1,160 TPS" note="Average across the day" />
            <MathStep step="5" formula="Peak TPS = 5× average (Black Friday)" result="~5,800 TPS" note="Flash sales, holiday peaks" final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Payment Volume (TPV)</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Average transaction amount" result="~$65" note="Mix of small ($5 SaaS) and large ($500 e-commerce)" />
            <MathStep step="2" formula="Daily TPV = 100M × $65" result="$6.5B/day" note="Total payment volume processed daily" />
            <MathStep step="3" formula="Annual TPV" result="~$2.4T/year" note="Stripe processes ~$1T/year for reference" />
            <MathStep step="4" formula="Revenue (2.9% + $0.30 per txn)" result="~$11M/day" note="Platform revenue from processing fees" />
            <MathStep step="5" formula="Refund rate (~2% of transactions)" result="~2M/day" note="Each refund = reverse ledger entry" final />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Storage Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Bytes per payment record" result="~2 KB" note="payment_id, amount, status, timestamps, metadata, card fingerprint" />
            <MathStep step="2" formula="Ledger entries per payment (avg 3)" result="~0.5 KB each" note="auth, capture, settle — double-entry each" />
            <MathStep step="3" formula="Daily storage = 100M × (2 + 1.5) KB" result="~350 GB/day" note="Payment records + ledger entries" />
            <MathStep step="4" formula="Audit log per event (~0.8 KB)" result="~240 GB/day" note="~300M events/day × 0.8 KB (immutable append-only)" />
            <MathStep step="5" formula="Yearly storage (data + audit)" result="~215 TB/year" note="Must retain 7+ years for compliance" final />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Latency Budget</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Total auth latency target" result="< 2s" note="End-to-end from API call to response" />
            <MathStep step="2" formula="API Gateway + auth" result="~20ms" note="TLS, API key validation, rate limit" />
            <MathStep step="3" formula="Idempotency check + fraud" result="~50ms" note="Redis lookup + ML model inference" />
            <MathStep step="4" formula="Payment service processing" result="~30ms" note="Validation, ledger write, state machine" />
            <MathStep step="5" formula="Processor/network round-trip" result="~500-1500ms" note="The bottleneck — external card network" final />
          </div>
          <div className="mt-4 pt-3 border-t border-stone-100">
            <div className="text-[11px] text-stone-500">
              <strong className="text-stone-700">Key insight:</strong> The card network round-trip (Visa/MC) dominates latency at 500-1500ms. Everything else is optimized to stay under 100ms. This means internal optimizations have diminishing returns — the critical path goes through external systems you don't control.
            </div>
          </div>
        </Card>
      </div>
      <Card>
        <Label color="#7c3aed">Summary — Quick Reference</Label>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Peak TPS", val: "~5,800", sub: "Avg: ~1,160" },
            { label: "Daily TPV", val: "$6.5B", sub: "Annual: ~$2.4T" },
            { label: "Storage/Year", val: "~215 TB", sub: "7-year retention" },
            { label: "Auth Latency", val: "< 2s", sub: "Network: 500-1500ms" },
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
          <Label color="#2563eb">Core Payment API (Stripe-style)</Label>
          <CodeBlock code={`# POST /v1/payment_intents
# Create a payment intent (2-phase: authorize then capture)
{
  "amount": 5000,               # $50.00 (cents)
  "currency": "usd",
  "payment_method": "pm_card_visa",
  "capture_method": "manual",   # manual = auth only
  "idempotency_key": "order_12345",
  "metadata": { "order_id": "ord_abc" }
}
# Response:
{
  "id": "pi_3abc...",
  "status": "requires_confirmation",
  "amount": 5000,
  "client_secret": "pi_3abc_secret_xyz"
}

# POST /v1/payment_intents/:id/confirm
# Confirm (triggers authorization with card network)
# Response: { "status": "requires_capture" }

# POST /v1/payment_intents/:id/capture
# Capture the authorized amount
{ "amount_to_capture": 5000 }   # Can be ≤ authorized
# Response: { "status": "succeeded" }

# POST /v1/refunds
{
  "payment_intent": "pi_3abc...",
  "amount": 2000                # Partial refund: $20
}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Key API Endpoints</Label>
          <div className="space-y-3">
            {[
              { op: "POST /payment_intents", desc: "Create a payment intent — the core unit of work. Tracks the full lifecycle from creation to settlement.", perf: "~100ms" },
              { op: "POST /:id/confirm", desc: "Confirm intent — triggers authorization against card network. Customer charged only after capture.", perf: "~500-2000ms (network)" },
              { op: "POST /:id/capture", desc: "Capture authorized funds. Can capture full or partial amount. Moves money from hold to collected.", perf: "~200ms" },
              { op: "POST /refunds", desc: "Refund a captured payment. Full or partial. Creates reverse ledger entries.", perf: "~300ms" },
              { op: "GET /payment_intents/:id", desc: "Get current status and full event history of a payment.", perf: "~20ms" },
              { op: "POST /webhooks", desc: "Merchant receives async notifications for payment events (succeeded, failed, refunded).", perf: "Async (retry)" },
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
              <Point icon="→" color="#d97706">Idempotency key on every mutating request — client-generated, stored for 24h</Point>
              <Point icon="→" color="#d97706">Amounts in smallest currency unit (cents) — avoids floating-point errors</Point>
              <Point icon="→" color="#d97706">Payment Intent pattern (not direct charge) — supports SCA, 3DS, async flows</Point>
              <Point icon="→" color="#d97706">Webhooks for all state changes — never rely solely on synchronous responses</Point>
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
    { label: "Simple Charge", desc: "Direct charge: merchant sends card details to processor, gets back success/fail. One-step. No authorization hold, no state machine. Simple but doesn't support SCA (Strong Customer Authentication), 3D Secure, or async payment methods." },
    { label: "2-Phase Auth/Capture", desc: "Authorize (hold funds) → Capture (collect funds). Two separate steps. Allows merchants to verify before collecting. Supports partial capture, void, and tip adjustments. This is how Stripe and most modern processors work." },
    { label: "Payment Intent FSM", desc: "Full state machine: Created → Requires Payment Method → Requires Confirmation → Processing → Requires Action (3DS) → Succeeded/Failed. Handles every edge case: retries, SCA challenges, async payment methods, and disputes." },
    { label: "Full Platform", desc: "Complete payment infrastructure: API Gateway → Payment Service → Ledger → Risk Engine → Processor Router → Card Networks → Settlement → Payout. Includes fraud detection, multi-processor failover, reconciliation, and merchant payouts." },
  ];
  const diagrams = [
    () => (
      <svg viewBox="0 0 460 140" className="w-full">
        <DiagramBox x={60} y={55} w={75} h={36} label="Merchant" color="#059669"/>
        <DiagramBox x={200} y={55} w={85} h={36} label="Processor" color="#9333ea"/>
        <DiagramBox x={350} y={55} w={85} h={38} label="Card\nNetwork" color="#dc2626"/>
        <Arrow x1={97} y1={55} x2={157} y2={55} label="charge" id="s1"/>
        <Arrow x1={242} y1={55} x2={307} y2={55} label="authorize" id="s2"/>
        <rect x={110} y={100} width={230} height={20} rx={4} fill="#dc262608" stroke="#dc262630"/>
        <text x={225} y={111} textAnchor="middle" fill="#dc2626" fontSize="8" fontFamily="monospace">✗ No SCA, no async methods, no partial capture</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 160" className="w-full">
        <DiagramBox x={55} y={55} w={70} h={36} label="Merchant" color="#059669"/>
        <DiagramBox x={175} y={35} w={80} h={30} label="Authorize" color="#9333ea"/>
        <DiagramBox x={175} y={80} w={80} h={30} label="Capture" color="#d97706"/>
        <DiagramBox x={320} y={55} w={80} h={38} label="Card\nNetwork" color="#dc2626"/>
        <Arrow x1={90} y1={45} x2={135} y2={38} label="1. auth" id="t1"/>
        <Arrow x1={90} y1={65} x2={135} y2={78} label="2. capture" id="t2"/>
        <Arrow x1={215} y1={37} x2={280} y2={48} id="t3"/>
        <Arrow x1={215} y1={82} x2={280} y2={62} id="t4"/>
        <rect x={95} y={120} width={260} height={20} rx={4} fill="#05966908" stroke="#05966930"/>
        <text x={225} y={131} textAnchor="middle" fill="#059669" fontSize="8" fontFamily="monospace">✓ Hold → Collect. Supports void, partial capture, tips</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 160" className="w-full">
        {["Created","Confirming","Processing","Action","Succeeded"].map((s,i) => (
          <g key={i}>
            <rect x={15+i*90} y={40} width={78} height={28} rx={6} fill={i===4?"#05966912":"#9333ea12"} stroke={i===4?"#059669":"#9333ea"} strokeWidth={1.2}/>
            <text x={54+i*90} y={56} textAnchor="middle" fill={i===4?"#059669":"#9333ea"} fontSize="8" fontWeight="600" fontFamily="monospace">{s}</text>
            {i < 4 && <Arrow x1={93+i*90} y1={54} x2={15+(i+1)*90} y2={54} id={`fsm${i}`}/>}
          </g>
        ))}
        <rect x={285} y={90} width={78} height={28} rx={6} fill="#dc262612" stroke="#dc2626" strokeWidth={1.2}/>
        <text x={324} y={106} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Failed</text>
        <Arrow x1={234} y1={68} x2={290} y2={95} label="decline" id="fsm_f" dashed/>
        <rect x={60} y={125} width={330} height={18} rx={4} fill="#6366f108" stroke="#6366f130"/>
        <text x={225} y={135} textAnchor="middle" fill="#6366f1" fontSize="8" fontFamily="monospace">State machine handles every edge case: retries, SCA, 3DS, async methods</text>
      </svg>
    ),
    () => (
      <svg viewBox="0 0 460 200" className="w-full">
        <DiagramBox x={45} y={50} w={60} h={30} label="Client" color="#2563eb"/>
        <DiagramBox x={130} y={50} w={65} h={30} label="Gateway" color="#6366f1"/>
        <DiagramBox x={225} y={30} w={70} h={28} label="Payment\nSvc" color="#9333ea"/>
        <DiagramBox x={225} y={75} w={70} h={28} label="Risk\nEngine" color="#dc2626"/>
        <DiagramBox x={330} y={30} w={65} h={28} label="Processor\nRouter" color="#d97706"/>
        <DiagramBox x={420} y={30} w={60} h={28} label="Visa/MC" color="#dc2626"/>
        <DiagramBox x={330} y={75} w={65} h={28} label="Ledger" color="#059669"/>
        <DiagramBox x={420} y={75} w={60} h={28} label="Settle" color="#0891b2"/>
        <Arrow x1={75} y1={50} x2={97} y2={50} id="f1"/>
        <Arrow x1={162} y1={42} x2={190} y2={35} id="f2"/>
        <Arrow x1={260} y1={32} x2={297} y2={32} id="f3"/>
        <Arrow x1={362} y1={32} x2={390} y2={32} id="f4"/>
        <Arrow x1={225} y1={44} x2={225} y2={61} id="f5" dashed/>
        <Arrow x1={260} y1={77} x2={297} y2={77} id="f6"/>
        <Arrow x1={362} y1={77} x2={390} y2={77} id="f7"/>
        <rect x={80} y={120} width={300} height={18} rx={4} fill="#05966908" stroke="#05966930"/>
        <text x={230} y={130} textAnchor="middle" fill="#059669" fontSize="7" fontFamily="monospace">Gateway → Risk → Payment → Processor → Network → Ledger → Settlement</text>
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
        <div className="bg-stone-50 rounded-lg border border-stone-200 p-3" style={{ minHeight: 160 }}>
          {diagrams[phase]()}
        </div>
      </Card>
      <Card>
        <Label color="#c026d3">Payment Lifecycle — The Core Tradeoff</Label>
        <p className="text-[12px] text-stone-500 mb-4">Choosing between sync vs async, one-step vs two-step, and how to handle failures defines the reliability of the entire system.</p>
        <div className="grid grid-cols-4 gap-4">
          {[
            { t: "Payment Intent (Stripe) ★", d: "Intent object tracks full lifecycle. State machine handles auth, capture, 3DS, retries. Client secret for frontend confirmation.", pros: ["Handles every payment flow (cards, bank, wallets)","Built-in SCA/3DS support","Idempotent by design (intent = unit of work)"], cons: ["More API calls for simple charges","Complex state machine to implement","Requires webhook integration"], pick: true },
            { t: "Direct Charge (Legacy)", d: "One API call: send card + amount → get success/fail. Simple but limited. No 3DS, no async methods.", pros: ["Simple — one API call","Low integration effort","Good for low-risk, card-present"], cons: ["No SCA/3DS support","No async payment methods","No partial capture or void"], pick: false },
            { t: "Hosted Checkout", d: "Redirect customer to processor's hosted page. Processor handles card input, 3DS, PCI scope. Merchant never sees card data.", pros: ["Zero PCI scope for merchant","3DS handled automatically","Supports all payment methods"], cons: ["Customer leaves merchant's site","Less brand control","Redirect latency"], pick: false },
            { t: "Tokenized Vault", d: "Card details stored in PCI-compliant vault. Token returned to merchant. All future charges use token, never raw card.", pros: ["One-click payments (stored cards)","Merchant never handles raw card data","Enables subscriptions/recurring"], cons: ["Vault must be PCI Level 1 certified","Token management complexity","Cross-processor portability issues"], pick: false },
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
  const [sel, setSel] = useState("idempotency");
  const algos = {
    idempotency: { name: "Idempotency (Exactly-Once Processing) ★", cx: "O(1) lookup",
      pros: ["Guarantees exactly-once semantics — retries are safe and produce same result","Client-generated key gives caller control over deduplication","Simple Redis/DB lookup before processing — low overhead","Essential for payment reliability — the #1 most important algorithm"],
      cons: ["Keys must be stored for a TTL (24-48h) — storage cost","Race condition: two concurrent requests with same key need locking","Must return cached response, not re-process — need response storage"],
      when: "EVERY mutating payment API call. Non-negotiable. Without idempotency, network retries can double-charge customers. Stripe requires idempotency keys on all POST requests. This is the single most important concept in payment system design.",
      code: `# Idempotency — Exactly-Once Payment Processing
# Client sends Idempotency-Key header with every request

def process_payment(request):
    key = request.headers["Idempotency-Key"]

    # 1. Check if we've seen this key before
    cached = redis.get(f"idempotency:{key}")
    if cached:
        if cached.status == "processing":
            return 409  # Conflict: still in progress
        return cached.response  # Return stored response

    # 2. Acquire lock (prevent concurrent duplicates)
    lock = redis.set(f"idempotency:{key}",
                     {"status": "processing"},
                     NX=True, EX=86400)  # 24h TTL
    if not lock:
        return 409  # Another request owns this key

    # 3. Process the payment
    try:
        result = execute_payment(request)
        # 4. Store result with the idempotency key
        redis.set(f"idempotency:{key}",
                  {"status": "complete", "response": result},
                  EX=86400)
        return result
    except Exception as e:
        redis.delete(f"idempotency:{key}")  # Allow retry
        raise e` },
    ledger: { name: "Double-Entry Ledger", cx: "O(1) per entry",
      pros: ["Every transaction has equal debit and credit — books always balance","Complete audit trail — every cent is traceable","Standard accounting principle — auditors and regulators understand it","Catches bugs: if debits ≠ credits, something is wrong"],
      cons: ["Double the write volume (every transaction = 2+ entries)","Immutable entries — corrections are new entries, not edits","Complex queries for balances (sum of all entries per account)"],
      when: "The ledger is the source of truth for all money movement. Every payment, refund, fee, and payout creates balanced ledger entries. This is how Stripe, Square, and every financial institution tracks money. If your system processes money, you need a double-entry ledger.",
      code: `# Double-Entry Ledger — Every $ has a debit and credit
# Rule: SUM(debits) = SUM(credits) ALWAYS

# Payment of $50 from customer to merchant:
# Entry 1 (Authorization):
INSERT INTO ledger_entries VALUES
  (entry_id, payment_id, 'customer_funding', 'DEBIT',  5000, 'usd'),
  (entry_id, payment_id, 'auth_hold',        'CREDIT', 5000, 'usd');

# Entry 2 (Capture → Settlement):
INSERT INTO ledger_entries VALUES
  (entry_id, payment_id, 'auth_hold',          'DEBIT',  5000, 'usd'),
  (entry_id, payment_id, 'merchant_balance',   'CREDIT', 4855, 'usd'),
  (entry_id, payment_id, 'platform_revenue',   'CREDIT',  145, 'usd');
  # $50.00 - $1.45 fee (2.9%) = $48.55 to merchant

# Refund of $20:
INSERT INTO ledger_entries VALUES
  (entry_id, payment_id, 'merchant_balance',   'DEBIT',  2000, 'usd'),
  (entry_id, payment_id, 'customer_refund',    'CREDIT', 2000, 'usd');

# Invariant check (run periodically):
SELECT SUM(CASE WHEN type='DEBIT' THEN amount ELSE 0 END)
     - SUM(CASE WHEN type='CREDIT' THEN amount ELSE 0 END)
FROM ledger_entries;
-- MUST ALWAYS = 0` },
    fsm: { name: "Payment State Machine (FSM)", cx: "O(1) transition",
      pros: ["Every valid state and transition is explicitly defined — prevents impossible states","Clear error handling: each state knows what failures can occur and what to do","Supports async flows naturally (3DS challenge, pending bank transfer)","Makes the system deterministic — given state + event = exact next state"],
      cons: ["Complex to implement correctly for all edge cases","Adding new payment methods may require new states/transitions","Testing all state paths is combinatorial"],
      when: "The state machine is the brain of the payment service. Every payment intent transitions through well-defined states. Stripe's Payment Intent has ~12 states. This is what makes payment flows reliable and debuggable. An interviewer will be very impressed if you draw the FSM.",
      code: `# Payment Intent State Machine
# States and valid transitions

STATES = {
  "created":
    → "requires_payment_method"  (on: create)
  "requires_payment_method":
    → "requires_confirmation"    (on: attach_method)
  "requires_confirmation":
    → "processing"               (on: confirm)
    → "canceled"                 (on: cancel)
  "processing":
    → "requires_action"          (on: 3DS challenge)
    → "requires_capture"         (on: auth_success, manual capture)
    → "succeeded"                (on: auth_success, auto capture)
    → "failed"                   (on: decline / error)
  "requires_action":
    → "processing"               (on: 3DS complete)
    → "failed"                   (on: 3DS timeout / fail)
  "requires_capture":
    → "succeeded"                (on: capture)
    → "canceled"                 (on: void / expire)
  "succeeded":
    → "refunded"                 (on: full refund)
    → "partially_refunded"       (on: partial refund)
    → "disputed"                 (on: chargeback)
  # Terminal states: succeeded, failed, canceled, refunded
}

def transition(payment, event):
    current = payment.status
    next_state = STATES[current].get(event)
    if next_state is None:
        raise InvalidTransitionError(current, event)
    payment.status = next_state
    audit_log.append(payment.id, current, next_state, event)
    return payment` },
    fraud: { name: "Fraud Detection (Risk Scoring)", cx: "O(1) inference",
      pros: ["Real-time scoring: block fraud before authorization, not after","ML models improve over time with feedback loop from chargebacks","Rule engine for known patterns + ML for novel fraud","Reduces chargebacks (saves ~1-3% of revenue)"],
      cons: ["False positives block legitimate customers (bad UX)","ML model needs large labeled dataset to train","Adversarial: fraudsters adapt to detection patterns","Latency: must score in <50ms to not slow down checkout"],
      when: "Every payment attempt is risk-scored before authorization. Stripe Radar, Adyen Risk, and Sift all do this. In an interview, mention that fraud detection sits on the hot path (pre-auth) and must be fast. Common signals: IP geolocation, device fingerprint, velocity, card BIN, behavioral patterns.",
      code: `# Fraud Detection — Pre-Authorization Risk Scoring
# Must complete in <50ms to stay within latency budget

def score_payment(payment, context):
    features = extract_features(payment, context)
    # Features include:
    # - Card BIN (first 6 digits) → issuing bank/country
    # - IP geolocation vs billing address
    # - Device fingerprint (browser, OS, screen)
    # - Velocity: txns in last 1h/24h for this card/IP/device
    # - Amount: unusually large for this merchant?
    # - Email domain: disposable? New?
    # - Shipping address: known fraud hotspot?

    # Rule engine (fast, deterministic):
    if features.velocity_1h > 10:
        return BLOCK, "velocity_exceeded"
    if features.country_mismatch:
        risk_score += 30

    # ML model (gradient boosted trees / neural net):
    ml_score = fraud_model.predict(features)  # 0.0 - 1.0

    # Combined score:
    final_score = 0.4 * rules_score + 0.6 * ml_score

    if final_score > 0.85:
        return BLOCK, "high_risk"
    elif final_score > 0.6:
        return CHALLENGE_3DS, "medium_risk"
    else:
        return ALLOW, "low_risk"` },
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
          <CodeBlock code={`-- payment_intents (source of truth)
CREATE TABLE payment_intents (
  id                VARCHAR(26) PRIMARY KEY,  -- pi_xxxx (ULID)
  merchant_id       VARCHAR(26) NOT NULL,
  amount            BIGINT NOT NULL,          -- in cents
  currency          CHAR(3) NOT NULL,         -- ISO 4217
  status            VARCHAR(30) NOT NULL,     -- FSM state
  payment_method_id VARCHAR(26),
  capture_method    VARCHAR(10) DEFAULT 'auto',
  idempotency_key   VARCHAR(255) UNIQUE,
  metadata          JSONB,
  created_at        TIMESTAMPTZ NOT NULL,
  updated_at        TIMESTAMPTZ NOT NULL
);

-- ledger_entries (immutable, append-only)
CREATE TABLE ledger_entries (
  id              BIGSERIAL PRIMARY KEY,
  payment_id      VARCHAR(26) NOT NULL,
  account         VARCHAR(50) NOT NULL,
  entry_type      VARCHAR(6) NOT NULL,  -- DEBIT/CREDIT
  amount          BIGINT NOT NULL,
  currency        CHAR(3) NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL
);
-- CONSTRAINT: SUM(DEBIT) = SUM(CREDIT) per payment

-- payment_events (audit trail, immutable)
CREATE TABLE payment_events (
  id              BIGSERIAL PRIMARY KEY,
  payment_id      VARCHAR(26) NOT NULL,
  event_type      VARCHAR(50) NOT NULL,
  previous_status VARCHAR(30),
  new_status      VARCHAR(30),
  processor_ref   VARCHAR(100),
  raw_response    JSONB,
  created_at      TIMESTAMPTZ NOT NULL
);`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Token Vault & Payment Methods</Label>
          <CodeBlock code={`-- payment_methods (tokenized card data)
CREATE TABLE payment_methods (
  id              VARCHAR(26) PRIMARY KEY,  -- pm_xxxx
  customer_id     VARCHAR(26) NOT NULL,
  type            VARCHAR(20) NOT NULL,     -- card, bank, wallet
  card_brand      VARCHAR(10),              -- visa, mc, amex
  card_last4      CHAR(4),
  card_exp_month  SMALLINT,
  card_exp_year   SMALLINT,
  card_fingerprint VARCHAR(64),             -- hash for dedup
  billing_address JSONB,
  processor_token VARCHAR(100),  -- token from card vault
  created_at      TIMESTAMPTZ NOT NULL
);
-- ⚠ Raw card numbers NEVER stored here
-- Processor token references PCI-compliant vault

-- customers
CREATE TABLE customers (
  id              VARCHAR(26) PRIMARY KEY,  -- cus_xxxx
  merchant_id     VARCHAR(26) NOT NULL,
  email           VARCHAR(255),
  name            VARCHAR(255),
  default_pm_id   VARCHAR(26),
  metadata        JSONB,
  created_at      TIMESTAMPTZ NOT NULL
);

-- webhook_endpoints
CREATE TABLE webhook_endpoints (
  id              VARCHAR(26) PRIMARY KEY,
  merchant_id     VARCHAR(26) NOT NULL,
  url             VARCHAR(2048) NOT NULL,
  secret          VARCHAR(64) NOT NULL,     -- HMAC signing
  events          TEXT[] NOT NULL,           -- subscribed events
  status          VARCHAR(10) DEFAULT 'active'
);`} />
          <div className="mt-3 pt-3 border-t border-stone-100">
            <div className="text-[11px] text-stone-500">
              <strong className="text-stone-700">Critical rule:</strong> Raw card numbers (PAN) are NEVER stored in your database. They go directly to a PCI-compliant vault (Stripe's is called "Vault"). Your system only stores tokens (pm_xxx) that reference the vault. This drastically reduces PCI scope — your main database doesn't need PCI Level 1 certification.
            </div>
          </div>
        </Card>
      </div>
      <Card accent="#9333ea">
        <Label color="#9333ea">Data Design Principles for Payments</Label>
        <p className="text-[12px] text-stone-500 mb-4">Payment data has unique constraints that differ from typical application data. Getting these wrong causes financial discrepancies, audit failures, and regulatory issues.</p>
        <div className="grid grid-cols-3 gap-4">
          {[
            { t: "Immutable Ledger ★", d: "Ledger entries are NEVER updated or deleted. Corrections are new entries (reversals). This creates a complete, auditable history.", pros: ["Full audit trail for compliance","Debugging: see every state change","Regulators require immutability"], cons: ["Table grows forever — need archival strategy","Queries for balances are SUMs (can be slow)","Cannot 'undo' — must create reversals"], pick: true },
            { t: "Amounts in Cents (Integers)", d: "Always store money as integers in the smallest currency unit. $50.00 = 5000 cents. Avoids floating-point errors.", pros: ["No floating-point precision bugs","Exact arithmetic always","Standard across Stripe, PayPal, Adyen"], cons: ["Must convert for display","Different currencies have different minor units (JPY has 0)","Easy to forget and cause bugs"], pick: false },
            { t: "Event Sourcing for State", d: "Payment state derived from ordered events, not mutable status field. Events: created → authorized → captured → settled.", pros: ["Replay events to reconstruct any past state","Natural audit log","Enables time-travel debugging"], cons: ["Higher storage (events + snapshots)","Eventual consistency between event store and read model","More complex implementation"], pick: false },
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
          <Label color="#059669">Database Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Shard by merchant_id</strong> — each merchant's payments on one shard. Queries are always scoped to a merchant. No cross-shard joins needed for normal operations.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Separate ledger database</strong> — ledger is write-heavy and append-only. Dedicated database (PostgreSQL with partitioning by date). Archived to cold storage after 90 days, kept in data warehouse for queries.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Read replicas for analytics</strong> — reporting, merchant dashboards, and reconciliation queries hit read replicas. Never query the primary for analytics.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">CQRS pattern</strong> — Command (write) path hits primary DB with strong consistency. Query (read) path hits materialized views or read replicas with eventual consistency. Payment status queries can tolerate ~1s staleness.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Service Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Stateless payment service</strong> — all state in database + Redis. Service instances are interchangeable. Scale horizontally by adding pods.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Processor connection pooling</strong> — each instance maintains a pool of connections to card networks. Pool exhaustion is a common bottleneck — size for peak TPS × avg response time.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Async webhooks</strong> — webhook delivery is decoupled via a message queue (Kafka/SQS). Doesn't block the payment response path. Retries with exponential backoff.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Multi-processor routing</strong> — route payments across multiple processors (Visa Direct, Adyen, etc.) for redundancy and cost optimization. If Processor A is down, fail over to Processor B.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Strategy</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Active-Passive with Failover", d:"Primary region processes all payments. Secondary region has hot standby with real-time replication. Failover in <60s. Simplest for payment systems because of strong consistency requirements.", pros:["Strong consistency — single write path","Simple to reason about (one source of truth)","No split-brain risk for payments"], cons:["Failover is not instant (30-60s)","Secondary region is idle (cost)","Cross-region latency for distant users"], pick:false },
            { t:"Active-Active by Region ★", d:"Payments processed in the region closest to the merchant. Each region has its own DB shard for its merchants. Cross-region replication for reads only.", pros:["Low latency for all merchants","No single region bottleneck","Survives region failure gracefully"], cons:["Complex — must prevent cross-region double-processing","Idempotency must be globally consistent","Settlement/reconciliation across regions"], pick:true },
            { t:"Follow-the-Sun", d:"Route to the region where the merchant's acquiring bank is located. EU merchants → EU region, US merchants → US region. Minimizes cross-border fees.", pros:["Optimal processing fees","Regulatory compliance per region","Natural data residency"], cons:["Merchant-to-region affinity can be complex","Some merchants operate globally","Need all processors in each region"], pick:false },
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
        <Label color="#d97706">Critical Decision: What Happens When the Payment System Is Down?</Label>
        <p className="text-[12px] text-stone-500 mb-4">Payment downtime directly equals lost revenue for every merchant on the platform. A 1-minute outage at Stripe-scale impacts millions of dollars in transactions. The system must degrade gracefully, not fail completely.</p>
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Graceful Degradation (Recommended)</div>
            <p className="text-[11px] text-stone-500 mb-2">Processor down → route to backup processor. Database degraded → queue writes, process async.</p>
            <ul className="space-y-1"><Point icon="✓" color="#059669">Multi-processor failover keeps payments flowing</Point><Point icon="✓" color="#059669">Risk engine degradation: approve low-risk, queue high-risk</Point><Point icon="⚠" color="#d97706">Must reconcile queued transactions when system recovers</Point></ul>
          </div>
          <div className="rounded-lg border border-red-200 bg-white p-4">
            <div className="text-[11px] font-bold text-red-600 mb-1.5">Double-Charge on Recovery</div>
            <p className="text-[11px] text-stone-500 mb-2">Network timeout during authorization → did the charge go through or not? Retry risks double-charge.</p>
            <ul className="space-y-1"><Point icon="→" color="#d97706">Idempotency keys prevent duplicate charges on retry</Point><Point icon="→" color="#d97706">Processor reference ID for reconciliation</Point><Point icon="→" color="#d97706">Void pending authorizations if status unknown after timeout</Point></ul>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card>
          <Label color="#2563eb">Multi-Processor Failover</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#2563eb"><strong className="text-stone-700">Primary/secondary processors</strong> — route payments to primary processor. If it returns errors or timeouts, automatically fail over to secondary within the same request.</Point>
            <Point icon="2." color="#2563eb"><strong className="text-stone-700">Circuit breaker per processor</strong> — if Processor A fails 5 consecutive requests, open circuit for 30s. Route all traffic to Processor B. Half-open probe after 30s.</Point>
            <Point icon="3." color="#2563eb"><strong className="text-stone-700">Smart routing</strong> — route based on card BIN, currency, and merchant location. Visa cards via processor with best Visa rates. AMEX via different processor.</Point>
            <Point icon="4." color="#2563eb"><strong className="text-stone-700">Timeout handling</strong> — processor timeout (&gt;5s) doesn't mean failure. Query processor for transaction status before retrying. Never retry without checking — could double-charge.</Point>
          </ul>
        </Card>
        <Card>
          <Label color="#0891b2">Database HA</Label>
          <ul className="space-y-2.5">
            <Point icon="🛡️" color="#0891b2"><strong className="text-stone-700">Synchronous replication</strong> — primary writes committed to at least one replica before acknowledging. Zero data loss on primary failure. Higher latency (~5ms) but non-negotiable for payments.</Point>
            <Point icon="📊" color="#0891b2"><strong className="text-stone-700">Automated failover</strong> — PostgreSQL Patroni or Aurora for automatic primary promotion. Failover in &lt;30s. Connection pooler (PgBouncer) manages reconnection transparently.</Point>
            <Point icon="🔄" color="#0891b2"><strong className="text-stone-700">Point-in-time recovery</strong> — WAL archiving enables recovery to any second in the past. Critical for investigating discrepancies: "what was the ledger state at 14:32:07 UTC?"</Point>
            <Point icon="📈" color="#0891b2"><strong className="text-stone-700">Backups</strong> — hourly snapshots + continuous WAL streaming. Test restore monthly. Backup verification is as important as the backup itself.</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#9333ea">Degradation Ladder</Label>
        <div className="flex gap-3 items-stretch mt-1">
          {[
            { label: "Full Platform", sub: "All systems nominal", color: "#059669", status: "HEALTHY" },
            { label: "Processor Failover", sub: "Backup processor active", color: "#d97706", status: "DEGRADED" },
            { label: "Risk Bypass", sub: "Low-risk auto-approve", color: "#ea580c", status: "FALLBACK" },
            { label: "Queue & Hold", sub: "Accept, process later", color: "#dc2626", status: "EMERGENCY" },
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
          <Label color="#0284c7">Key Metrics — The Payment Dashboard</Label>
          <ul className="space-y-2.5">
            <Point icon="📊" color="#0284c7"><strong className="text-stone-700">Authorization Rate</strong> — % of auth attempts that succeed. Target: 95-98%. Drop below 90% = investigate immediately.</Point>
            <Point icon="⏱️" color="#0284c7"><strong className="text-stone-700">Auth Latency (p50/p99)</strong> — p50: &lt;800ms, p99: &lt;2s. Includes card network round-trip. Spike = processor issue.</Point>
            <Point icon="💰" color="#0284c7"><strong className="text-stone-700">TPV (Total Payment Volume)</strong> — money processed per hour/day. Sudden drop = checkout broken or processor down.</Point>
            <Point icon="🔄" color="#0284c7"><strong className="text-stone-700">Decline Rate by Reason</strong> — split by: insufficient funds, fraud block, processor error, expired card. Each needs different action.</Point>
            <Point icon="❌" color="#0284c7"><strong className="text-stone-700">Error Rate (5xx)</strong> — internal errors. Must be &lt;0.01%. Any 5xx on payment path = P1 incident.</Point>
          </ul>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Reconciliation & Audit</Label>
          <ul className="space-y-2.5">
            <Point icon="📝" color="#059669"><strong className="text-stone-700">Daily reconciliation</strong> — compare internal ledger totals against processor settlement reports. Every cent must match. Discrepancies investigated within 24h.</Point>
            <Point icon="🔍" color="#059669"><strong className="text-stone-700">Ledger balance check</strong> — SUM(debits) = SUM(credits) verified every hour. Imbalance = critical alert. Means a bug in payment logic.</Point>
            <Point icon="📊" color="#059669"><strong className="text-stone-700">Payment tracing</strong> — unique payment_id traced through every service: API → Risk → Payment → Processor → Ledger → Webhook. Full latency breakdown per hop.</Point>
            <Point icon="🔔" color="#059669"><strong className="text-stone-700">Anomaly detection</strong> — alert on: auth rate drop &gt;5%, latency spike &gt;2×, TPV drop &gt;20%, error rate &gt;0.1%, fraud rate spike.</Point>
          </ul>
        </Card>
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Webhook Monitoring</Label>
          <ul className="space-y-2.5">
            <Point icon="📤" color="#7c3aed"><strong className="text-stone-700">Delivery success rate</strong> — target 99.5%+. Failed webhooks retry with exponential backoff (5s, 30s, 5m, 30m, 2h, up to 72h).</Point>
            <Point icon="⏱️" color="#7c3aed"><strong className="text-stone-700">Delivery latency</strong> — time from payment event to webhook delivery. p50 &lt;1s, p99 &lt;5s. Exclude retries from latency metric.</Point>
            <Point icon="🔄" color="#7c3aed"><strong className="text-stone-700">Dead letter queue</strong> — webhooks that fail after all retries go to DLQ. Alert on DLQ growth. Merchants can manually retry from dashboard.</Point>
            <Point icon="🧪" color="#7c3aed"><strong className="text-stone-700">Webhook testing</strong> — merchants can send test webhooks from dashboard. Verify endpoint reachability and correct response (2xx).</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#0284c7">Sample Payment Event Trail</Label>
        <CodeBlock code={`# Payment pi_3abc lifecycle events (immutable audit trail)
{event: "payment_intent.created",      ts: "14:32:01.000", status: "created"}
{event: "payment_method.attached",      ts: "14:32:01.200", status: "requires_confirmation"}
{event: "payment_intent.confirmed",     ts: "14:32:01.500", status: "processing"}
{event: "risk.scored",                  ts: "14:32:01.520", score: 0.12, action: "allow"}
{event: "processor.auth_request",       ts: "14:32:01.550", processor: "stripe_visa", ref: "ch_xyz"}
{event: "processor.auth_response",      ts: "14:32:02.100", result: "approved", auth_code: "A12345"}
{event: "ledger.entry_created",         ts: "14:32:02.110", entries: 2, balanced: true}
{event: "payment_intent.succeeded",     ts: "14:32:02.120", status: "succeeded"}
{event: "webhook.queued",               ts: "14:32:02.130", endpoint: "https://merchant.com/hooks"}
{event: "webhook.delivered",            ts: "14:32:02.850", http_status: 200}`} />
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
            { mode: "Double Charge", impact: "CRITICAL", desc: "Network timeout during authorization → client retries → card charged twice. Customer sees two $50 charges.",
              mitigation: "Idempotency key on every request. Check idempotency store BEFORE hitting processor. Store processor reference for reconciliation. Void duplicate auths.",
              example: "Mobile client retries on timeout. Server processed the first request successfully but response was lost. Second request charges the card again." },
            { mode: "Inconsistent Ledger", impact: "CRITICAL", desc: "Payment marked as 'succeeded' but ledger entries not written (crash between processor response and ledger write). Money moved but not tracked.",
              mitigation: "Use database transaction: write payment status AND ledger entries atomically. If either fails, both roll back. Transactional outbox pattern for events.",
              example: "Server crashes after receiving auth_success from Visa but before writing ledger entry. $500 charged but no record in ledger. Reconciliation catches it — 24h later." },
            { mode: "Processor Timeout Ambiguity", impact: "HIGH", desc: "Authorization request to Visa times out after 5s. Did the charge go through? Unknown. Can't retry (might double-charge). Can't fail (might lose a valid charge).",
              mitigation: "Query processor for transaction status using the original reference ID. If status unknown, void the potential auth. Log for manual reconciliation.",
              example: "Visa responds after 8 seconds (our timeout is 5s). We timed out and failed the payment, but Visa authorized it. Customer's card shows a $50 hold that was never captured." },
            { mode: "Webhook Delivery Failure", impact: "MEDIUM", desc: "Payment succeeds but webhook to merchant fails. Merchant doesn't know about the payment. Customer paid but order not fulfilled.",
              mitigation: "Retry with exponential backoff (up to 72h). Merchant should poll payment status, not rely solely on webhooks. Dashboard shows pending webhooks.",
              example: "Merchant's server is down for maintenance. 500 webhooks fail. When server comes back, all 500 retry in a burst — overwhelming the merchant's endpoint." },
            { mode: "Currency Conversion Race", impact: "HIGH", desc: "Exchange rate changes between authorization and capture. Authorized $50 = €45, but at capture time $50 = €46. Who absorbs the difference?",
              mitigation: "Lock exchange rate at authorization time. Auth and capture use the same rate. Or: authorize in merchant's settlement currency. Clearly define FX responsibility in API.",
              example: "Authorization at 9am: $100 = €91. Capture at 3pm: $100 = €93. Merchant expected €91 but received €93. Or worse: received €89 due to rate moving the other way." },
            { mode: "PCI Data Breach", impact: "CRITICAL", desc: "Raw card numbers leaked from logs, database, or in-memory. Regulatory fines ($5K-$100K per card), loss of processing ability, reputational damage.",
              mitigation: "NEVER log or store raw card numbers. Tokenize immediately at edge. PCI-compliant vault with hardware security modules. Regular penetration testing. Strict access controls.",
              example: "Developer accidentally logged full card number in debug mode. Log shipped to Splunk. 50,000 card numbers exposed in plaintext in the logging system." },
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
              { name: "API Gateway", role: "TLS termination, API key auth, rate limiting, request routing, idempotency key extraction. First line of defense.", tech: "Envoy / Kong / AWS API Gateway", critical: true },
              { name: "Payment Service", role: "Core orchestrator. Manages payment intent lifecycle (FSM), coordinates between risk, processor, and ledger. The brain.", tech: "Java/Go + PostgreSQL", critical: true },
              { name: "Risk Engine", role: "Pre-auth fraud scoring. Rule engine + ML model. Returns allow/block/challenge decision in <50ms.", tech: "Python ML model + Redis features", critical: true },
              { name: "Processor Router", role: "Routes payment to optimal processor based on card BIN, currency, cost, and health. Handles failover between processors.", tech: "Go + circuit breaker (Hystrix)", critical: true },
              { name: "Ledger Service", role: "Immutable double-entry ledger. Every money movement creates balanced entries. Source of truth for financial data.", tech: "PostgreSQL (append-only) + Kafka", critical: true },
              { name: "Token Vault", role: "PCI-compliant card data storage. Hardware Security Modules (HSM) for encryption. Returns tokens for card references.", tech: "Dedicated PCI environment + HSM", critical: true },
              { name: "Webhook Service", role: "Async event delivery to merchants. Queues events, delivers with retry. Signs payloads with HMAC for verification.", tech: "Kafka → workers → HTTP POST", critical: false },
              { name: "Settlement Service", role: "End-of-day batch: aggregate captured payments per merchant, calculate fees, initiate bank transfers (payouts).", tech: "Batch job (daily) + banking API", critical: false },
              { name: "Reconciliation Service", role: "Compares internal ledger against processor settlement files. Flags discrepancies. Runs daily.", tech: "Spark/Airflow batch job", critical: false },
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
          <Label color="#9333ea">Payment Service Internals — Block Diagram</Label>
          <svg viewBox="0 0 380 320" className="w-full">
            <rect x={10} y={10} width={360} height={45} rx={6} fill="#6366f108" stroke="#6366f1" strokeWidth={1}/>
            <text x={190} y={28} textAnchor="middle" fill="#6366f1" fontSize="10" fontWeight="600" fontFamily="monospace">API Gateway + Auth</text>
            <text x={190} y={43} textAnchor="middle" fill="#6366f180" fontSize="8" fontFamily="monospace">TLS · API key · rate limit · idempotency check</text>

            <rect x={10} y={65} width={360} height={40} rx={6} fill="#9333ea08" stroke="#9333ea" strokeWidth={1}/>
            <text x={190} y={83} textAnchor="middle" fill="#9333ea" fontSize="10" fontWeight="600" fontFamily="monospace">Payment Orchestrator (FSM)</text>
            <text x={190} y={96} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">state machine · intent lifecycle · retry logic</text>

            <rect x={10} y={115} width={175} height={70} rx={6} fill="#dc262608" stroke="#dc2626" strokeWidth={1}/>
            <text x={97} y={138} textAnchor="middle" fill="#dc2626" fontSize="10" fontWeight="600" fontFamily="monospace">Risk Engine</text>
            <text x={97} y={153} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">rule engine + ML model</text>
            <text x={97} y={168} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">allow / block / 3DS</text>

            <rect x={195} y={115} width={175} height={70} rx={6} fill="#d9770608" stroke="#d97706" strokeWidth={1}/>
            <text x={282} y={138} textAnchor="middle" fill="#d97706" fontSize="10" fontWeight="600" fontFamily="monospace">Processor Router</text>
            <text x={282} y={153} textAnchor="middle" fill="#d9770680" fontSize="8" fontFamily="monospace">Visa / MC / Amex rails</text>
            <text x={282} y={168} textAnchor="middle" fill="#d9770680" fontSize="8" fontFamily="monospace">failover · cost routing</text>

            <rect x={10} y={195} width={175} height={55} rx={6} fill="#05966908" stroke="#059669" strokeWidth={1}/>
            <text x={97} y={215} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="600" fontFamily="monospace">Ledger Service</text>
            <text x={97} y={230} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">double-entry bookkeeping</text>
            <text x={97} y={242} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">immutable · append-only</text>

            <rect x={195} y={195} width={175} height={55} rx={6} fill="#0891b208" stroke="#0891b2" strokeWidth={1}/>
            <text x={282} y={215} textAnchor="middle" fill="#0891b2" fontSize="10" fontWeight="600" fontFamily="monospace">Webhook Dispatcher</text>
            <text x={282} y={230} textAnchor="middle" fill="#0891b280" fontSize="8" fontFamily="monospace">Kafka → workers → HTTP</text>
            <text x={282} y={242} textAnchor="middle" fill="#0891b280" fontSize="8" fontFamily="monospace">retry · HMAC signing</text>

            <rect x={10} y={260} width={360} height={50} rx={6} fill="#78716c08" stroke="#78716c" strokeWidth={1}/>
            <text x={190} y={280} textAnchor="middle" fill="#78716c" fontSize="10" fontWeight="600" fontFamily="monospace">Settlement & Reconciliation</text>
            <text x={100} y={298} textAnchor="middle" fill="#78716c80" fontSize="8" fontFamily="monospace">Daily batch: aggregate → payout</text>
            <text x={280} y={298} textAnchor="middle" fill="#78716c80" fontSize="8" fontFamily="monospace">Recon: ledger vs processor files</text>

            <line x1={190} y1={55} x2={190} y2={65} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-ni1)"/>
            <line x1={190} y1={105} x2={97} y2={115} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={190} y1={105} x2={282} y2={115} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={97} y1={185} x2={97} y2={195} stroke="#94a3b8" strokeWidth={1}/>
            <line x1={282} y1={185} x2={282} y2={195} stroke="#94a3b8" strokeWidth={1}/>
            <defs><marker id="ah-ni1" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
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
                { route: "Gateway → Payment Svc", proto: "gRPC (mTLS)", contract: "CreatePaymentIntent / Confirm / Capture", timeout: "10s", fail: "Return 503, client retries with idempotency key" },
                { route: "Payment Svc → Risk Engine", proto: "gRPC", contract: "ScorePayment(features) → allow/block/3DS", timeout: "100ms", fail: "Default to ALLOW for low amounts, BLOCK for high" },
                { route: "Payment Svc → Processor", proto: "HTTPS (ISO 8583)", contract: "Authorize / Capture / Void / Refund", timeout: "5s", fail: "Failover to secondary processor, log for recon" },
                { route: "Payment Svc → Ledger", proto: "gRPC (same DB txn)", contract: "CreateEntries(debit, credit) — atomic", timeout: "500ms", fail: "Rollback payment status, return error" },
                { route: "Payment Svc → Webhook Queue", proto: "Kafka producer", contract: "PaymentEvent → topic (async)", timeout: "100ms (ack)", fail: "Buffer locally, retry. Alert if queue lag > 30s" },
                { route: "Webhook Worker → Merchant", proto: "HTTPS POST", contract: "Signed JSON payload, expect 2xx", timeout: "30s", fail: "Retry 5s/30s/5m/30m/2h up to 72h, then DLQ" },
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
  const [flow, setFlow] = useState("auth");
  const flows = {
    auth: {
      title: "Authorization (Happy Path)",
      steps: [
        { actor: "Client", action: "POST /v1/payment_intents {amount: 5000, currency: 'usd', idempotency_key: 'ord_123'}", type: "request" },
        { actor: "API Gateway", action: "Authenticate API key, check rate limit, extract idempotency key", type: "auth" },
        { actor: "Payment Service", action: "Check idempotency store (Redis): key 'ord_123' not found → proceed", type: "check" },
        { actor: "Payment Service", action: "Create payment_intent (id: pi_abc, status: 'requires_confirmation')", type: "process" },
        { actor: "Client", action: "POST /v1/payment_intents/pi_abc/confirm {payment_method: 'pm_visa_42'}", type: "request" },
        { actor: "Risk Engine", action: "Score payment: amount=$50, card=visa, IP=US → score: 0.08 → ALLOW", type: "check" },
        { actor: "Processor Router", action: "Route to Processor A (primary): Visa auth request, $50.00, card token", type: "request" },
        { actor: "Card Network (Visa)", action: "Issuing bank approves. Auth code: A12345. Response time: 850ms", type: "success" },
        { actor: "Ledger Service", action: "Write double-entry: DEBIT customer_funding $50, CREDIT auth_hold $50", type: "process" },
        { actor: "Payment Service", action: "Update intent status: 'succeeded'. Store in idempotency cache.", type: "success" },
        { actor: "Webhook Queue", action: "Enqueue event: payment_intent.succeeded → merchant webhook endpoint", type: "process" },
      ]
    },
    threeDs: {
      title: "3D Secure Challenge Flow",
      steps: [
        { actor: "Payment Service", action: "Confirm payment → send to processor", type: "request" },
        { actor: "Risk Engine", action: "Score: 0.55 (medium risk) → CHALLENGE_3DS. Or: SCA regulation requires it.", type: "check" },
        { actor: "Processor", action: "Returns 'requires_action' with 3DS URL for customer authentication", type: "process" },
        { actor: "Payment Service", action: "Update status: 'requires_action'. Return 3DS redirect URL to client.", type: "process" },
        { actor: "Client (Browser)", action: "Redirect customer to issuing bank's 3DS page. Customer enters OTP/biometric.", type: "request" },
        { actor: "Issuing Bank", action: "Customer authenticated successfully. Callback to processor with auth result.", type: "success" },
        { actor: "Processor", action: "3DS passed. Continue with authorization. Send auth request to card network.", type: "request" },
        { actor: "Card Network", action: "Authorization approved with 3DS liability shift (fraud liability → issuing bank)", type: "success" },
        { actor: "Payment Service", action: "Status: 'succeeded'. 3DS shifts fraud liability to issuer — lower chargeback risk.", type: "success" },
      ]
    },
    refund: {
      title: "Refund Flow",
      steps: [
        { actor: "Merchant", action: "POST /v1/refunds {payment_intent: 'pi_abc', amount: 2000} (partial: $20)", type: "request" },
        { actor: "Payment Service", action: "Validate: payment is 'succeeded', refund amount ≤ captured - already_refunded", type: "check" },
        { actor: "Payment Service", action: "Create refund object (re_xyz, status: 'pending')", type: "process" },
        { actor: "Processor Router", action: "Send refund request to original processor (must use same processor as charge)", type: "request" },
        { actor: "Card Network", action: "Refund processed. Funds will be returned to cardholder in 5-10 business days.", type: "success" },
        { actor: "Ledger Service", action: "DEBIT merchant_balance $20, CREDIT customer_refund $20 (reverse entries)", type: "process" },
        { actor: "Payment Service", action: "Update payment: 'partially_refunded'. Refund status: 'succeeded'.", type: "success" },
        { actor: "Webhook Queue", action: "Enqueue: charge.refunded event → merchant endpoint", type: "process" },
      ]
    },
    failure: {
      title: "Failure Path — Processor Timeout",
      steps: [
        { actor: "Payment Service", action: "Send auth request to Processor A (primary). Timeout: 5s.", type: "request" },
        { actor: "Processor A", action: "No response after 5 seconds (timeout). Status: UNKNOWN.", type: "error" },
        { actor: "Payment Service", action: "Query Processor A for transaction status using original reference ID.", type: "check" },
        { actor: "Processor A", action: "Status query also times out. Processor may be completely down.", type: "error" },
        { actor: "Payment Service", action: "Circuit breaker OPEN for Processor A. Try Processor B (failover).", type: "process" },
        { actor: "Processor B", action: "Authorization request sent. Approved in 920ms. Auth code: B67890.", type: "success" },
        { actor: "Payment Service", action: "Payment succeeded via Processor B. Record processor used for capture/refund.", type: "success" },
        { actor: "Reconciliation", action: "(Async) Check if Processor A also authorized. If yes → void the duplicate auth.", type: "process" },
        { actor: "Alert System", action: "P2 incident: Processor A timeout rate > 10%. Ops team investigates.", type: "error" },
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
              <span className="text-[11px] font-mono font-bold shrink-0 w-32" style={{ color: colors[s.type] }}>{s.actor}</span>
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
            <Point icon="1." color="#b45309"><strong className="text-stone-700">Zero-downtime deploys</strong> — rolling deployment with blue-green canary. New version handles 1% of traffic for 30 min, monitor auth rate and error rate, then full rollout.</Point>
            <Point icon="2." color="#b45309"><strong className="text-stone-700">Database migrations are backward-compatible</strong> — only additive changes (new columns, new tables). Never rename or drop columns in the same deploy. Two-phase: deploy code that handles both schemas → migrate → remove old code.</Point>
            <Point icon="3." color="#b45309"><strong className="text-stone-700">Feature flags for new payment methods</strong> — new payment method support rolled out behind feature flags. Enable for 1 merchant → 10 → 100 → all. If issues, disable flag instantly.</Point>
            <Point icon="4." color="#b45309"><strong className="text-stone-700">PCI environment isolation</strong> — token vault and card handling services deployed in separate PCI-compliant environment. Different network, different access controls, separate audit trail.</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Security — PCI DSS & Beyond</Label>
          <ul className="space-y-2.5">
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">PCI DSS Level 1</strong> — annual on-site audit by QSA. Network segmentation, encryption at rest (AES-256) and in transit (TLS 1.3). Access logging on all card data access.</Point>
            <Point icon="🔑" color="#dc2626"><strong className="text-stone-700">Tokenization</strong> — raw card numbers enter system, immediately tokenized. Token is a random string (pm_xxx) that maps to encrypted card in vault. Main systems never see raw PAN.</Point>
            <Point icon="🛡️" color="#dc2626"><strong className="text-stone-700">API key security</strong> — secret keys scoped per merchant. Publishable keys for frontend (limited permissions). Key rotation without downtime. IP whitelisting for production keys.</Point>
            <Point icon="📝" color="#dc2626"><strong className="text-stone-700">Webhook signing</strong> — every webhook payload signed with merchant-specific HMAC-SHA256 secret. Merchants verify signature before processing. Prevents forgery.</Point>
            <Point icon="🧱" color="#dc2626"><strong className="text-stone-700">Fraud monitoring</strong> — real-time velocity checks, device fingerprinting, IP reputation. 3D Secure for high-risk transactions. Machine learning model retrained weekly on chargeback data.</Point>
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
        <Label color="#be123c">Auto-Scaling & Alerting Triggers</Label>
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
                { trigger: "Auth Rate Drop", thresh: "< 90% for 5min", action: "P1 Alert. Check processor health, fraud rules, card network status.", cool: "0 (immediate)", pitfall: "Could be processor-side issue (issuer decline rate up), not our bug" },
                { trigger: "Latency p99", thresh: "> 3s for 5min", action: "Check processor response times, DB query latency, connection pool", cool: "5 min", pitfall: "Processor latency is external — we can't fix Visa being slow" },
                { trigger: "Error Rate (5xx)", thresh: "> 0.1% for 3min", action: "P1 Alert. Check logs, recent deploys, DB health, processor status", cool: "0 (immediate)", pitfall: "5xx on payment path = direct revenue loss. Treat as P1 always." },
                { trigger: "TPS Spike", thresh: "> 3× baseline", action: "Auto-scale payment service pods (HPA). Pre-warm DB connection pools.", cool: "3 min", pitfall: "Could be DDoS/enumeration attack, not real traffic. Check fraud signals." },
                { trigger: "Ledger Imbalance", thresh: "SUM(debit) ≠ SUM(credit)", action: "P0 Alert. Halt new deployments. Investigate immediately.", cool: "0 (immediate)", pitfall: "This should never happen. If it does, there's a bug in payment logic." },
                { trigger: "Webhook DLQ Growth", thresh: "> 1000 messages", action: "Investigate merchant endpoint health. Increase retry workers.", cool: "15 min", pitfall: "Bulk DLQ often means one large merchant is down, not a systemic issue" },
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
        <Label color="#be123c">Operational Pitfalls — Production War Stories</Label>
        <p className="text-[12px] text-stone-500 mb-3">These are the things you learn the hard way. Mentioning any of these in an interview signals real operational experience with payment systems.</p>
        <div className="grid grid-cols-2 gap-4">
          {[
            { title: "Floating-Point Money Bug", symptom: "Merchant reports: charged $19.99 but settled $19.98. One cent missing on 3% of transactions.",
              cause: "Developer used float instead of integer cents for calculation. 0.1 + 0.2 = 0.30000000000000004 in IEEE 754. Rounding errors accumulate.",
              fix: "ALL money as integers in smallest currency unit (cents). $19.99 = 1999. No floating point anywhere in the money path. Lint rules to catch float usage in payment code.",
              quote: "We lost $47,000 over 3 months to floating-point rounding. The discrepancy was so small per transaction that nobody noticed until quarterly reconciliation." },
            { title: "Idempotency Key Collision", symptom: "Two different merchants send the same idempotency key. Merchant B's payment returns Merchant A's response.",
              cause: "Idempotency keys were globally scoped, not per-merchant. Both merchants happened to use 'order-001' as their key.",
              fix: "Scope idempotency keys to merchant_id: key = hash(merchant_id + idempotency_key). Always namespace. Never trust client-generated keys to be globally unique.",
              quote: "A merchant called saying they got someone else's payment confirmation. Turned out both used sequential order IDs as idempotency keys." },
            { title: "Settlement Timing Mismatch", symptom: "Merchant balance shows $10,000 but bank transfer is only $9,850. Merchant angry about the 'missing' $150.",
              cause: "Dashboard showed gross amount. Settlement was net of fees ($150 in processing fees deducted). Different systems showed different numbers.",
              fix: "Always show gross, fees, and net separately in every view. Dashboard, API, settlement file — all must show the same fee breakdown. Single source of truth for fee calculation.",
              quote: "Support ticket: 'You're stealing $150 from me.' Resolution: 'That's your processing fee, sir. It was always deducted.' We made the dashboard clearer." },
            { title: "Processor Failover Double-Auth", symptom: "Customer's card shows two $200 holds. Processor A timed out, we failed over to Processor B. But Processor A actually succeeded (delayed response).",
              cause: "Timeout ≠ failure. Processor A was slow but eventually authorized. Our system didn't know because it had already moved on to Processor B.",
              fix: "After processor timeout: query Processor A for status BEFORE trying Processor B. If can't determine status, try B but mark the original for reconciliation. Void orphaned auths within 24h.",
              quote: "Customer's $200 dinner became $400 in pending holds. Both auths eventually captured because nobody voided the orphan. We refunded and added void logic." },
            { title: "Webhook Amplification Attack", symptom: "Single payment generates 10,000 webhook deliveries. Merchant's server overwhelmed. Other webhook deliveries delayed.",
              cause: "Bug in retry logic: failed webhook re-queued immediately instead of with backoff. Each retry failure re-queued again. Exponential growth.",
              fix: "Exponential backoff with jitter (5s, 30s, 5m, 30m, 2h). Max retries cap (15 attempts over 72h). Per-merchant rate limit on webhook deliveries. Circuit breaker per endpoint.",
              quote: "We accidentally DDoS'd our own merchant's server. And it wasn't even a big merchant — 1 failed payment, 10K webhook attempts in 2 minutes." },
            { title: "Time Zone Reconciliation Bug", symptom: "Daily settlement report shows different totals than processor's report. Off by a few hundred thousand dollars.",
              cause: "Our system used UTC day boundary (midnight UTC). Processor used PST day boundary (midnight PST). 8 hours of transactions counted in different 'days.'",
              fix: "Always use UTC internally. When reconciling with processors, convert to their timezone for comparison. Document timezone assumptions. Settlement cutoff time explicitly configured per processor.",
              quote: "We spent 3 days debugging a $340K reconciliation gap. It was literally just a timezone difference between us and the processor. Now we have a sign on the wall: 'UTC or die.'" },
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
        { t: "Subscription / Recurring Billing", d: "Automatically charge customers on a schedule (monthly, annual). Handle plan changes, proration, dunning (retry failed charges).", detail: "Billing engine creates payment intents on schedule. Dunning: retry 3x over 7 days, then pause subscription. Proration: calculate partial charges on plan change.", effort: "Hard" },
        { t: "Multi-Currency & FX", d: "Accept payments in 135+ currencies. Settle to merchants in their preferred currency. Real-time exchange rates with markup.", detail: "Lock FX rate at authorization. Settlement in merchant currency. Transparent FX fee (typically 1-2%). Integration with FX providers for real-time rates.", effort: "Hard" },
        { t: "Dispute / Chargeback Management", d: "Handle cardholder disputes automatically. Collect evidence from merchant, submit to card network. Track win/loss rates.", detail: "Dispute lifecycle: opened → evidence_needed → submitted → won/lost. Auto-collect evidence (receipt, shipping tracking). Webhook to merchant for evidence upload.", effort: "Medium" },
        { t: "Connect Platform (Marketplace)", d: "Split payments between platform and sub-merchants. Platform takes fee, rest goes to seller. Like Stripe Connect.", detail: "PaymentIntent with transfer_group and destination charges. Platform fee calculated per transaction. Separate settlement to each party.", effort: "Hard" },
        { t: "Smart Retry & Recovery", d: "Automatically retry failed payments with intelligent timing. Update expired cards via network tokenization. Recover 5-15% of failed charges.", detail: "Retry at optimal times (payday, low-decline hours). Card account updater to refresh expired cards. ML model predicts best retry window.", effort: "Medium" },
        { t: "Real-Time Fraud ML Pipeline", d: "Streaming ML pipeline for fraud detection. Features computed in real-time from Kafka events. Model retrained daily on chargeback feedback.", detail: "Feature store (Redis) with velocity counters, device fingerprints, behavioral signals. Gradient boosted trees for scoring. A/B test new models with shadow scoring.", effort: "Hard" },
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
    { q:"How does Stripe guarantee exactly-once payment processing?", a:"Three mechanisms: (1) Idempotency keys — client-generated, stored for 24h. Same key = same response, no re-processing. (2) Database transactions — payment status and ledger entries written atomically. If either fails, both roll back. (3) Processor reference tracking — every processor response includes a unique reference. On timeout/retry, query by reference before re-submitting. The combination of all three makes double-charges virtually impossible.", tags:["design"] },
    { q:"Why use a double-entry ledger instead of just updating account balances?", a:"Account balance is a derived number (sum of all entries). If you only store the balance, you lose the 'why.' A double-entry ledger gives you: (1) Complete audit trail — every cent is traceable to a specific event. (2) Self-checking — debits MUST equal credits. If they don't, you have a bug. (3) Compliance — regulators require you to explain every balance change. (4) Debugging — 'why is this merchant's balance $47,291?' → just query their ledger entries. Without a ledger, you're flying blind.", tags:["data"] },
    { q:"How does 3D Secure (3DS) work in the payment flow?", a:"3DS adds a customer authentication step (OTP, biometric, or app approval) during payment. Flow: (1) Merchant submits auth → processor returns 'requires_action' with 3DS URL. (2) Customer redirected to issuing bank's 3DS page. (3) Customer authenticates. (4) Bank sends result back to processor. (5) If successful, authorization continues with liability shift — fraud liability moves from merchant to issuing bank. SCA regulation in EU/UK makes 3DS mandatory for most online payments >€30.", tags:["security"] },
    { q:"How do you handle the 'unknown' state when a processor times out?", a:"This is the hardest problem in payments. Steps: (1) Do NOT retry immediately — might double-charge. (2) Query the processor for transaction status using your reference ID. (3) If processor says 'authorized' → continue normally. (4) If processor says 'not found' → safe to retry (or fail over). (5) If processor also times out on status query → mark as 'unknown,' alert ops, void if possible after investigation. (6) Reconciliation job catches any orphaned authorizations within 24h.", tags:["availability"] },
    { q:"How does settlement work? When does the merchant actually get paid?", a:"Settlement is a batch process: (1) End of day: aggregate all captured payments per merchant. (2) Subtract processing fees (2.9% + $0.30) and refunds. (3) Net amount → initiate bank transfer (ACH, wire, or local rails). (4) Funds arrive in merchant's bank in 1-3 business days. (5) Rolling reserve: hold back 5-10% for new/high-risk merchants to cover chargebacks. Stripe settles daily for most merchants, with a 2-day rolling schedule.", tags:["design"] },
    { q:"How do you prevent and handle chargebacks?", a:"Prevention: (1) Strong fraud detection (ML + rules) to block suspicious charges pre-auth. (2) 3DS to shift liability. (3) Clear billing descriptors so customers recognize charges. Handling: (1) Card network notifies processor of dispute. (2) Processor notifies merchant via webhook. (3) Merchant submits evidence (receipt, delivery proof, customer communication). (4) Card network decides (merchant or cardholder wins). Win rate: ~30-40% industry average. Merchants with good evidence win ~60%. Excessive chargebacks (>1%) risk losing processing ability.", tags:["security"] },
    { q:"How would you design the fraud detection ML pipeline?", a:"Training: labeled data from chargebacks (fraud=1) and successful payments (fraud=0). ~1% fraud rate = highly imbalanced. Features: card BIN, IP geolocation, device fingerprint, velocity (txns/hour), amount deviation, email age, shipping-billing address match. Model: gradient boosted trees (XGBoost/LightGBM) for production, neural nets for experimentation. Serving: feature store in Redis (pre-computed velocity counters), model inference in <50ms. Retraining: daily on new chargeback data. Shadow scoring: new model scores alongside old model, compare before promoting.", tags:["algorithm"] },
    { q:"How does Stripe handle PCI compliance for merchants?", a:"Stripe minimizes merchant PCI scope through: (1) Stripe.js / Elements — card number entered in Stripe's iframe, never touches merchant's server. (2) Tokenization — merchant receives a token (pm_xxx), never raw card data. (3) API calls use tokens — charge, refund, etc. all reference tokens. Result: merchant only needs SAQ-A (self-assessment questionnaire) — the simplest PCI compliance level. Stripe itself maintains PCI Level 1 (the highest level, annual on-site audit) for the vault that stores actual card data. This architecture is a huge competitive advantage — merchants don't need expensive PCI audits.", tags:["security"] },
    { q:"How do you handle multi-currency and cross-border payments?", a:"Three approaches: (1) Presentment currency — charge in customer's currency ($100 USD), settle to merchant in their currency (€91 EUR). FX conversion at settlement with markup (1-2%). (2) Multi-currency pricing — merchant sets prices in each currency. No FX risk for merchant. (3) Dynamic currency conversion — customer sees price in their currency at checkout, charged in merchant's currency. Key decisions: lock FX rate at auth or settlement? Who bears FX risk? How to handle refund FX (original rate or current rate)?", tags:["scalability"] },
    { q:"What's the difference between a payment gateway, processor, and acquirer?", a:"Gateway: the API/software layer that accepts merchant's payment request and routes it (Stripe, Braintree). Processor: the infrastructure that actually sends the auth request to card networks and processes the response (First Data, Worldpay). Acquirer: the bank that underwrites the merchant and receives funds from the card network on behalf of the merchant. Many companies play multiple roles: Stripe is a gateway AND processor AND effectively an acquirer (through banking partners). The 4-party model: Customer → Issuing Bank → Card Network → Acquiring Bank → Merchant.", tags:["design"] },
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

export default function PaymentSystemSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Payment System (Stripe)</h1>
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