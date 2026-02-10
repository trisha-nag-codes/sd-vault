import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   FRAUD / ABUSE DETECTION — ML System Design Reference (Google L6)
   Pearl white theme · 17 sections (HLD + LLD + ML Deep Dive)
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",                icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",            icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",     icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",              icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",       icon: "🏗️", color: "#9333ea" },
  { id: "taxonomy",      label: "Fraud Taxonomy",          icon: "🗂️", color: "#c026d3" },
  { id: "rules",         label: "Rules Engine",            icon: "📏", color: "#ea580c" },
  { id: "models",        label: "ML Models",               icon: "🧠", color: "#dc2626" },
  { id: "features",      label: "Feature Engineering",     icon: "⚙️", color: "#d97706" },
  { id: "graph",         label: "Graph-Based Detection",   icon: "🕸️", color: "#0f766e" },
  { id: "data",          label: "Data Model",              icon: "🗄️", color: "#059669" },
  { id: "training",      label: "Training Pipeline",       icon: "🔄", color: "#7e22ce" },
  { id: "realtime",      label: "Real-Time Scoring",       icon: "⚡", color: "#ea580c" },
  { id: "scalability",   label: "Scalability",             icon: "📈", color: "#059669" },
  { id: "watchouts",     label: "Failure Modes",           icon: "⚠️", color: "#dc2626" },
  { id: "observability", label: "Observability",           icon: "📊", color: "#0284c7" },
  { id: "followups",     label: "Follow-up Questions",     icon: "❓", color: "#6366f1" },
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


/* ═══════════════════════════════════════════════════════════
   SECTIONS
   ═══════════════════════════════════════════════════════════ */

function ConceptSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-7 space-y-5">
          <Card accent="#6366f1">
            <Label>What is a Fraud / Abuse Detection System?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A fraud detection system identifies and blocks malicious activity in real-time — payment fraud, account takeovers, fake accounts, click fraud, abuse of platform features — before the damage is done. At Google's scale, this protects <strong>billions of transactions per day</strong> across Google Pay, Google Ads, Play Store, and Cloud billing.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              The fundamental challenge: make a <em>block-or-allow</em> decision on every transaction in under <strong>50ms</strong>, with extreme class imbalance (fraud is &lt;0.1% of events) and adversarial opponents who constantly adapt their strategies. Unlike content moderation, fraud has a direct, measurable financial cost — every false negative is dollars lost.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard? (Google L6 Depth)</Label>
            <ul className="space-y-2.5">
              <Point icon="💰" color="#0891b2">Extreme class imbalance — fraud is 0.05-0.5% of transactions. A model that predicts "not fraud" for everything gets 99.5% accuracy. You need specialized loss functions and sampling strategies.</Point>
              <Point icon="⚔️" color="#0891b2">Adversarial non-stationarity — fraudsters actively evolve tactics. The distribution isn't just shifting — it's being intentionally manipulated. Yesterday's detection rules are tomorrow's exploits.</Point>
              <Point icon="⚡" color="#0891b2">Latency vs accuracy tradeoff — payment fraud must be decided in &lt;50ms (user is waiting at checkout). But the most powerful signals (network analysis, cross-account patterns) are expensive to compute.</Point>
              <Point icon="🔄" color="#0891b2">Delayed labels — you don't know if a transaction was fraudulent for days or weeks (until chargeback). Some fraud is never reported. Training data is inherently incomplete and delayed.</Point>
              <Point icon="📊" color="#0891b2">Asymmetric costs — false negative (missed fraud): direct financial loss + regulatory fines + customer trust damage. False positive (blocking legitimate user): lost revenue + user friction + churn. Both are expensive but in different ways.</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Scale</Label>
            <div className="space-y-2.5">
              {[
                { co: "Google Pay", scale: "Billions of txns/year", approach: "ML + rules + device trust" },
                { co: "Stripe Radar", scale: "$1T+ processed/yr", approach: "Network-level ML, cross-merchant" },
                { co: "PayPal", scale: "6.5B txns/quarter", approach: "Deep learning + graph analysis" },
                { co: "Visa", scale: "150B+ txns/yr", approach: "Neural network real-time scoring" },
                { co: "Amazon", scale: "Order & account fraud", approach: "ML + behavioral biometrics" },
              ].map((e,i) => (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-24 shrink-0">{e.co}</span>
                  <span className="text-stone-500 flex-1">{e.scale}</span>
                  <span className="text-stone-400 text-[10px]">{e.approach}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <Label color="#2563eb">The Detection Pipeline (Preview)</Label>
            <svg viewBox="0 0 360 185" className="w-full">
              <defs><marker id="ah-fp" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
              <rect x={15} y={10} width={70} height={35} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
              <text x={50} y={25} textAnchor="middle" fill="#2563eb" fontSize="8" fontWeight="700" fontFamily="monospace">Transaction</text>
              <text x={50} y={37} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">event arrives</text>

              <rect x={105} y={10} width={80} height={35} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
              <text x={145} y={25} textAnchor="middle" fill="#ea580c" fontSize="8" fontWeight="700" fontFamily="monospace">Rules Engine</text>
              <text x={145} y={37} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">blocklist + rules</text>

              <rect x={205} y={10} width={75} height={35} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
              <text x={242} y={25} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="700" fontFamily="monospace">ML Scorer</text>
              <text x={242} y={37} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">fraud prob</text>

              <rect x={300} y={10} width={50} height={35} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
              <text x={325} y={25} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="700" fontFamily="monospace">Action</text>
              <text x={325} y={37} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">block/allow</text>

              <line x1={85} y1={28} x2={105} y2={28} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-fp)"/>
              <line x1={185} y1={28} x2={205} y2={28} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-fp)"/>
              <line x1={280} y1={28} x2={300} y2={28} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-fp)"/>

              <rect x={125} y={60} width={80} height={30} rx={6} fill="#0f766e10" stroke="#0f766e" strokeWidth={1.5}/>
              <text x={165} y={79} textAnchor="middle" fill="#0f766e" fontSize="8" fontWeight="600" fontFamily="monospace">Graph Analysis</text>

              <rect x={225} y={60} width={80} height={30} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
              <text x={265} y={79} textAnchor="middle" fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Manual Review</text>

              <line x1={242} y1={45} x2={165} y2={60} stroke="#94a3b8" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-fp)"/>
              <line x1={242} y1={45} x2={265} y2={60} stroke="#94a3b8" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-fp)"/>

              <rect x={15} y={110} width={335} height={65} rx={6} fill="#faf9f7" stroke="#e7e5e4"/>
              <text x={25} y={127} fill="#ea580c" fontSize="7" fontWeight="600" fontFamily="monospace">Layer 1 — Rules: Blocklists, velocity checks, geo-impossible travel. Deterministic, fast (&lt;5ms).</text>
              <text x={25} y={142} fill="#dc2626" fontSize="7" fontWeight="600" fontFamily="monospace">Layer 2 — ML Model: Fraud probability score. Feature-rich deep model. &lt;30ms inference.</text>
              <text x={25} y={157} fill="#0f766e" fontSize="7" fontWeight="600" fontFamily="monospace">Layer 3 — Graph: Network-level patterns. Fraud rings, device sharing. Async or near-real-time.</text>
              <text x={25} y={167} fill="#7c3aed" fontSize="7" fontWeight="600" fontFamily="monospace">Layer 4 — Manual Review: Uncertain cases + high-value transactions. Human decision.</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Google Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Core Trust & Safety / Payments problem at Google</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Scope Like an L6</div>
            <p className="text-[12px] text-stone-500 mt-0.5">"Design a fraud detection system" must be scoped to a domain. Proactively: "I'll design a real-time payment fraud detection system — scoring every transaction at checkout. I'll cover the layered defense architecture: rules engine, ML model, graph analysis, and manual review. I'll focus on the ML model design, feature engineering from transaction sequences, and the adversarial adaptation challenge." Mention the layered defense early — it signals you know fraud systems are never single-model.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Score every transaction with a fraud probability in real-time (block/allow/review)</Point>
            <Point icon="2." color="#059669">Support multiple fraud types: payment fraud, account takeover, fake accounts, promo abuse</Point>
            <Point icon="3." color="#059669">Integrate deterministic rules alongside ML models (rules for known patterns, ML for novel fraud)</Point>
            <Point icon="4." color="#059669">Detect fraud rings and coordinated attacks via network/graph analysis</Point>
            <Point icon="5." color="#059669">Route uncertain cases to manual review with case context and evidence</Point>
            <Point icon="6." color="#059669">Support real-time feedback: when fraud is confirmed, immediately update risk signals</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Scoring latency: p50 &lt;30ms, p99 &lt;100ms (user waiting at checkout)</Point>
            <Point icon="2." color="#dc2626">Throughput: 50K-100K transactions/sec (payment platform scale)</Point>
            <Point icon="3." color="#dc2626">Precision: &gt;90% (most blocked transactions should be truly fraudulent)</Point>
            <Point icon="4." color="#dc2626">Recall: &gt;95% (catch nearly all fraud; each miss = financial loss)</Point>
            <Point icon="5." color="#dc2626">Availability: 99.99% — if fraud system is down, ALL transactions are at risk</Point>
            <Point icon="6." color="#dc2626">Model freshness: adapt to new fraud patterns within hours, not weeks</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask (L6 Signal)</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "What type of fraud? Payments, account takeover, fake accounts, or all?",
            "Inline (block before processing) or async (flag after)?",
            "What's the chargeback rate today? (determines how bad the problem is)",
            "Do we have device fingerprinting? (major signal for fraud detection)",
            "Cross-merchant visibility? (Stripe/Visa see patterns across merchants)",
            "What's the dollar threshold for manual review? ($10 vs $10,000 matters)",
            "Regulatory requirements? (PCI-DSS, PSD2 SCA, KYC/AML)",
            "Can we add friction (2FA, step-up auth) or only binary block/allow?",
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
          <Label color="#7c3aed">Step 1 — Transaction Volume</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Transactions/day (payment platform)" result="~1B/day" note="Google Pay + Play Store + Ads billing + Cloud" />
            <MathStep step="2" formula="TPS = 1B / 86,400" result="~12K TPS" note="Average. Each must be scored in real-time." />
            <MathStep step="3" formula="Peak TPS = 12K x 5" result="~60K TPS" note="Holiday shopping, flash sales, end-of-month billing." final />
            <MathStep step="4" formula="Account events (login, signup)" result="~5B/day" note="Login attempts, password changes, profile updates." />
            <MathStep step="5" formula="Total scorable events/sec" result="~70K/sec" note="Payments + account events combined." final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Fraud Scale & Cost</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Fraud rate (typical payment platform)" result="~0.1%" note="1 in 1000 transactions is fraudulent." />
            <MathStep step="2" formula="Fraudulent transactions/day = 1B x 0.001" result="~1M/day" note="Must catch as many as possible in real-time." />
            <MathStep step="3" formula="Avg fraud transaction value" result="~$200" note="Fraudsters target mid-range amounts to avoid detection." />
            <MathStep step="4" formula="Daily fraud exposure = 1M x $200" result="$200M/day" note="Potential daily loss without any detection." final />
            <MathStep step="5" formula="Target fraud catch rate" result=">95%" note="Catch $190M of $200M. Remaining $10M is the cost of missed fraud." />
            <MathStep step="6" formula="False positive cost" result="~$5/event" note="Lost sale + customer frustration + support cost." />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Latency Budget</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="User-facing latency budget" result="<100ms" note="User is waiting at checkout. Cannot stall." final />
            <MathStep step="2" formula="Rules engine (blocklist + velocity)" result="~5ms" note="In-memory lookups. Deterministic, fast." />
            <MathStep step="3" formula="Feature assembly (user + device + txn)" result="~10ms" note="Parallel batch-fetch from feature store." />
            <MathStep step="4" formula="ML model inference" result="~15ms" note="Deep model on CPU (not GPU — latency too variable)." />
            <MathStep step="5" formula="Decision + action" result="~3ms" note="Threshold logic, enforcement, logging." />
            <MathStep step="6" formula="Graph analysis" result="~200ms (async)" note="Too slow for inline. Runs async, may trigger delayed block." />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Feature & Model Scale</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Feature store: user risk profiles" result="~2B users" note="Each user has rolling aggregations (30+ features)." />
            <MathStep step="2" formula="Feature store size = 2B x 200B/user" result="~400 GB" note="In Redis/Bigtable. Hot users cached." final />
            <MathStep step="3" formula="Device fingerprint DB" result="~5B devices" note="Unique device IDs with trust scores and history." />
            <MathStep step="4" formula="Blocklist size (IPs, cards, devices)" result="~100M entries" note="Bloom filter for fast lookup (~100 MB in-memory)." />
            <MathStep step="5" formula="Training data: labeled events/day" result="~5M" note="Confirmed fraud + confirmed legitimate (with label delay)." />
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
          <Label color="#2563eb">Fraud Scoring API</Label>
          <CodeBlock code={`# Synchronous API — called inline at transaction time
# RPC: FraudService.ScoreTransaction(FraudRequest)
#
# FraudRequest:
{
  "event_id": "evt_abc123",
  "event_type": "payment",   // payment | login | signup | refund
  "timestamp": "2024-02-10T14:30:00Z",
  "transaction": {
    "amount": 249.99,
    "currency": "USD",
    "merchant_id": "merch_xyz",
    "merchant_category": "electronics",
    "payment_method": "card_ending_4242",
    "billing_country": "US",
    "shipping_country": "US",
    "is_digital_goods": false,
  },
  "user": {
    "user_id": "u_def456",
    "email_hash": "a3f2...",
    "account_age_days": 365,
    "is_verified": true,
  },
  "device": {
    "device_id": "dev_ghi789",
    "ip_address": "203.0.113.42",
    "user_agent": "Mozilla/5.0 ...",
    "screen_resolution": "1920x1080",
    "timezone": "America/New_York",
    "has_vpn": false,
  },
  "session": {
    "session_id": "sess_jkl012",
    "pages_viewed": 12,
    "time_on_site_sec": 340,
    "items_in_cart": 2,
  }
}

# FraudResponse:
{
  "event_id": "evt_abc123",
  "decision": "allow",        // allow | block | review | challenge
  "fraud_score": 0.12,        // [0,1] probability
  "risk_level": "low",        // low | medium | high | critical
  "triggered_rules": [],
  "explanation": ["Known device", "Normal purchase pattern"],
  "recommended_action": "none",  // none | 3ds | sms_verify | block
  "latency_ms": 28
}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Design Decisions (L6 Depth)</Label>
          <div className="space-y-3">
            {[
              { q: "Why synchronous (not async)?", a: "Payment fraud must be decided before the transaction is processed. Once money moves, recovery is expensive (chargebacks cost $20-50 each in processing fees alone). Inline scoring blocks bad transactions before the charge is authorized. Some fraud types (account takeover) can be async if you can reverse the action." },
              { q: "Why return 'challenge' as a decision?", a: "Not all suspicious transactions should be blocked. A medium-risk transaction from a known user might just need step-up authentication (3D Secure, SMS verification). This reduces false positives while still catching fraud — the legitimate user passes the challenge, the fraudster doesn't." },
              { q: "Why include device and session signals?", a: "Device fingerprint is the single most powerful fraud signal. If the device has been seen before with this user, risk drops dramatically. Session signals (time on site, pages viewed) distinguish real shopping behavior from bot-driven or stolen-credential fraud (where the session is usually very short)." },
              { q: "Why explanations in the response?", a: "Explainability is critical for three reasons: (1) Manual reviewers need to understand why a transaction was flagged. (2) Merchants want to understand their fraud patterns. (3) Regulators require explanation for blocked transactions (PSD2 SCA). Feature contribution to the fraud score powers the explanation." },
              { q: "Why include recommended_action?", a: "Different risk levels warrant different responses. Score 0.3 = suggest SMS verification (adds friction but catches stolen credentials). Score 0.8 = block outright. Score 0.5 = route to manual review for high-value orders. The action depends on score AND transaction value AND merchant risk tolerance." },
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
        <Label color="#9333ea">Full System Architecture — Layered Defense</Label>
        <svg viewBox="0 0 720 370" className="w-full">
          <defs><marker id="ah-hld" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>

          {/* Transaction */}
          <rect x={10} y={50} width={70} height={40} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={45} y={67} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Event</text>
          <text x={45} y={80} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">txn/login</text>

          {/* Layer 1: Rules */}
          <rect x={100} y={35} width={95} height={65} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={147} y={53} textAnchor="middle" fill="#ea580c" fontSize="9" fontWeight="600" fontFamily="monospace">Layer 1</text>
          <text x={147} y={66} textAnchor="middle" fill="#ea580c" fontSize="8" fontFamily="monospace">Rules Engine</text>
          <text x={147} y={79} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">blocklist, velocity</text>
          <text x={147} y={91} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">geo rules (~5ms)</text>

          {/* Layer 2: ML */}
          <rect x={220} y={35} width={95} height={65} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={267} y={53} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Layer 2</text>
          <text x={267} y={66} textAnchor="middle" fill="#dc2626" fontSize="8" fontFamily="monospace">ML Model</text>
          <text x={267} y={79} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">fraud probability</text>
          <text x={267} y={91} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">deep net (~15ms)</text>

          {/* Decision */}
          <rect x={340} y={40} width={80} height={50} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={380} y={60} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Decision</text>
          <text x={380} y={75} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">block/allow</text>
          <text x={380} y={85} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">challenge/review</text>

          {/* Actions */}
          <rect x={445} y={20} width={70} height={28} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={480} y={38} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Allow</text>
          <rect x={445} y={55} width={70} height={28} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={480} y={73} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Challenge</text>
          <rect x={445} y={90} width={70} height={28} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={480} y={108} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Block</text>

          {/* Layer 3: Graph (async) */}
          <rect x={220} y={125} width={115} height={40} rx={6} fill="#0f766e10" stroke="#0f766e" strokeWidth={1.5}/>
          <text x={277} y={143} textAnchor="middle" fill="#0f766e" fontSize="8" fontWeight="600" fontFamily="monospace">Layer 3 — Graph Analysis</text>
          <text x={277} y={157} textAnchor="middle" fill="#0f766e80" fontSize="7" fontFamily="monospace">fraud rings, device clusters (async)</text>

          {/* Layer 4: Manual Review */}
          <rect x={370} y={125} width={115} height={40} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
          <text x={427} y={143} textAnchor="middle" fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Layer 4 — Manual Review</text>
          <text x={427} y={157} textAnchor="middle" fill="#7c3aed80" fontSize="7" fontFamily="monospace">analyst investigation</text>

          {/* Data stores */}
          <rect x={540} y={30} width={80} height={30} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={580} y={49} textAnchor="middle" fill="#ea580c" fontSize="7" fontWeight="600" fontFamily="monospace">Blocklist DB</text>
          <rect x={540} y={70} width={80} height={30} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={580} y={89} textAnchor="middle" fill="#d97706" fontSize="7" fontWeight="600" fontFamily="monospace">Feature Store</text>
          <rect x={540} y={110} width={80} height={30} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={580} y={129} textAnchor="middle" fill="#0891b2" fontSize="7" fontWeight="600" fontFamily="monospace">Event Log</text>
          <rect x={640} y={70} width={70} height={30} rx={6} fill="#0f766e10" stroke="#0f766e" strokeWidth={1.5}/>
          <text x={675} y={89} textAnchor="middle" fill="#0f766e" fontSize="7" fontWeight="600" fontFamily="monospace">Graph DB</text>

          {/* Arrows */}
          <line x1={80} y1={70} x2={100} y2={67} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={195} y1={67} x2={220} y2={67} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={315} y1={67} x2={340} y2={65} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={420} y1={50} x2={445} y2={34} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={420} y1={65} x2={445} y2={69} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={420} y1={80} x2={445} y2={104} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={267} y1={100} x2={267} y2={125} stroke="#0f766e60" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>
          <line x1={380} y1={90} x2={410} y2={125} stroke="#7c3aed60" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>

          {/* Legend */}
          <rect x={15} y={195} width={700} height={160} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={25} y={215} fill="#ea580c" fontSize="8" fontWeight="600" fontFamily="monospace">Layer 1 — Rules Engine (~5ms): Deterministic checks. Known-bad IPs/cards/devices → instant block.</text>
          <text x={25} y={230} fill="#ea580c" fontSize="8" fontFamily="monospace">           Velocity rules: >5 transactions in 1 minute from same card → block. Geo-impossible: login from NYC then Tokyo in 30 min.</text>
          <text x={25} y={250} fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Layer 2 — ML Model (~15ms): Score every event not blocked by rules. Deep model with 100+ features.</text>
          <text x={25} y={265} fill="#dc2626" fontSize="8" fontFamily="monospace">           Predicts P(fraud). Handles novel patterns that rules can't encode. The core intelligence of the system.</text>
          <text x={25} y={285} fill="#0f766e" fontSize="8" fontWeight="600" fontFamily="monospace">Layer 3 — Graph Analysis (~200ms, async): Find fraud rings — groups of accounts sharing devices, IPs,</text>
          <text x={25} y={300} fill="#0f766e" fontSize="8" fontFamily="monospace">           payment methods. Too slow for inline but catches coordinated attacks. May trigger delayed account action.</text>
          <text x={25} y={320} fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Layer 4 — Manual Review: Human analysts investigate flagged events. High-value ($1K+) or uncertain (score 0.3-0.7).</text>
          <text x={25} y={340} fill="#78716c" fontSize="8" fontFamily="monospace">           KEY INSIGHT: Layers are additive. Each catches fraud the others miss. Rules catch known patterns fast.</text>
          <text x={25} y={350} fill="#78716c" fontSize="8" fontFamily="monospace">           ML catches novel patterns. Graph catches coordinated attacks. Together: >95% detection rate.</text>
        </svg>
      </Card>
    </div>
  );
}

function TaxonomySection() {
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Fraud Taxonomy — Types & Signals</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[11px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Fraud Type</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Description</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Key Signals</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Inline?</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Prevalence</th>
            </tr></thead>
            <tbody>
              {[
                { type: "Stolen Card", desc: "Purchased with stolen credit card numbers from data breaches.", signals: "New device, mismatched geo, shipping ≠ billing, rapid successive transactions", inline: "YES", prev: "40%", hl: true },
                { type: "Account Takeover", desc: "Attacker gains access to legitimate user's account (phishing, credential stuffing).", signals: "New device, new IP/geo, password change + immediate purchase, unusual behavior", inline: "YES", prev: "25%" },
                { type: "Fake Accounts", desc: "Bot-created accounts for promo abuse, money laundering, or platform manipulation.", signals: "Device fingerprint reuse, rapid creation, disposable email, no organic activity", inline: "YES", prev: "15%" },
                { type: "Friendly Fraud", desc: "Legitimate cardholder makes purchase then files chargeback claiming fraud.", signals: "Delivery confirmed, consistent device/IP history, repeat chargeback pattern", inline: "No", prev: "10%" },
                { type: "Promo/Coupon Abuse", desc: "Exploiting referral bonuses, sign-up credits, or discount codes via multiple accounts.", signals: "Same device, similar email patterns, same payment method across accounts", inline: "Partial", prev: "5%" },
                { type: "Money Laundering", desc: "Using the platform to move illicit funds through legitimate-looking transactions.", signals: "High-value, round amounts, rapid in-and-out, layering patterns, known jurisdictions", inline: "No", prev: "3%" },
                { type: "Click Fraud", desc: "Fraudulent clicks on ads to drain competitor budgets or inflate publisher revenue.", signals: "Bot patterns, datacenter IPs, click velocity, no conversion, coordinate timing", inline: "YES", prev: "2%" },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-red-50" : i%2 ? "bg-stone-50/50" : ""}>
                  <td className="px-3 py-2 font-bold text-stone-700">{r.type}</td>
                  <td className="px-3 py-2 text-stone-500 text-[10px]">{r.desc}</td>
                  <td className="px-3 py-2 text-stone-500 font-mono text-[9px]">{r.signals}</td>
                  <td className="text-center px-3 py-2 font-bold" style={{ color: r.inline === "YES" ? "#dc2626" : r.inline === "No" ? "#059669" : "#d97706" }}>{r.inline}</td>
                  <td className="text-center px-3 py-2"><Pill bg="#f3f4f6" color="#374151">{r.prev}</Pill></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function RulesSection() {
  return (
    <div className="space-y-5">
      <Card accent="#ea580c">
        <Label color="#ea580c">Rules Engine — Fast, Deterministic, Interpretable</Label>
        <p className="text-[12px] text-stone-500 mb-4">The rules engine is Layer 1 — it runs before ML. It catches known fraud patterns instantly, provides a baseline even if ML is down, and is immediately deployable when a new fraud pattern is discovered (no model retraining needed).</p>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Rules Engine — Core Rules" code={`# Rules Engine — deterministic, fast, interpretable
class FraudRulesEngine:
    def evaluate(self, event):
        triggered = []

        # BLOCKLISTS (~1ms, bloom filter + exact match)
        if event.ip in ip_blocklist:
            triggered.append(("blocked_ip", "BLOCK", 1.0))
        if event.card_hash in stolen_card_list:
            triggered.append(("stolen_card", "BLOCK", 1.0))
        if event.device_id in banned_devices:
            triggered.append(("banned_device", "BLOCK", 1.0))

        # VELOCITY RULES (~3ms, Redis counters)
        txn_count_1h = redis.get(f"vel:{event.card}:1h")
        if txn_count_1h > 5:
            triggered.append(("card_velocity_1h", "BLOCK", 0.95))

        txn_count_1d = redis.get(f"vel:{event.user}:1d")
        if txn_count_1d > 20:
            triggered.append(("user_velocity_1d", "REVIEW", 0.70))

        # GEO RULES
        last_txn = get_last_transaction(event.user)
        if last_txn:
            distance = haversine(last_txn.geo, event.geo)
            time_diff = event.ts - last_txn.ts
            speed = distance / time_diff.hours
            if speed > 900:  # > 900 km/h = impossible travel
                triggered.append(("impossible_travel", "BLOCK", 0.90))

        # AMOUNT RULES
        if event.amount > user_avg_amount * 10:
            triggered.append(("amount_spike", "REVIEW", 0.60))

        # DEVICE RULES
        if event.device.age_hours < 1 and event.amount > 500:
            triggered.append(("new_device_high_value", "CHALLENGE", 0.65))

        return triggered`} />
          <div className="space-y-4">
            <Card accent="#d97706">
              <Label color="#d97706">Rules vs ML — When to Use Each</Label>
              <div className="space-y-2">
                {[
                  { when: "Known stolen card list", use: "Rule", why: "Deterministic. If the card is on the list, block it. No probability needed." },
                  { when: "Impossible travel speed", use: "Rule", why: "Physics-based. NYC → Tokyo in 30 min is impossible. Simple math, 100% precision." },
                  { when: "Transaction looks slightly unusual", use: "ML", why: "Subtle patterns: amount is a bit high, time is unusual, device is new but not banned. ML weighs all signals together." },
                  { when: "New fraud pattern discovered", use: "Rule first, then ML", why: "Write a rule today (deployed in hours). Train ML model this week (deployed in days). Rule bridges the gap." },
                  { when: "Detecting fraud rings", use: "Graph + ML", why: "Shared devices/IPs across accounts. Rules can't express graph structure. Need GNN or community detection." },
                ].map((r,i) => (
                  <div key={i} className="rounded-lg border border-stone-200 p-2.5">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-[10px] text-stone-600">{r.when}</span>
                      <Pill bg={r.use.includes("Rule")?"#fff7ed":"#fef2f2"} color={r.use.includes("Rule")?"#ea580c":"#dc2626"}>{r.use}</Pill>
                    </div>
                    <p className="text-[10px] text-stone-400">{r.why}</p>
                  </div>
                ))}
              </div>
            </Card>
            <Card className="bg-amber-50/50 border-amber-200">
              <Label color="#d97706">Why Not Just Rules?</Label>
              <p className="text-[11px] text-stone-500">Rules catch known patterns. Fraudsters learn the rules (by probing) and adapt: stay under velocity limits, use residential IPs, match geo perfectly. ML detects subtle combinations of signals that no single rule captures. A slightly high amount + slightly new device + slightly unusual time = individually fine, but collectively suspicious. Only ML weighs all signals together.</p>
            </Card>
          </div>
        </div>
      </Card>
    </div>
  );
}

function ModelsSection() {
  return (
    <div className="space-y-5">
      <Card accent="#dc2626">
        <Label color="#dc2626">ML Model Architecture — Fraud Scoring</Label>
        <p className="text-[12px] text-stone-500 mb-4">The fraud model must be fast (CPU inference, &lt;15ms), handle extreme class imbalance (0.1% fraud), and be robust to adversarial adaptation. Two architectures dominate: gradient-boosted trees (GBDT) for interpretability and speed, and deep models for capturing sequential patterns.</p>
        <div className="grid grid-cols-2 gap-5">
          <Card accent="#dc2626">
            <Label color="#dc2626">GBDT Model (Production Baseline) ★</Label>
            <CodeBlock code={`# XGBoost/LightGBM — the fraud detection workhorse
# Fast inference (CPU), interpretable, handles tabular data well

model = LightGBM(
    objective="binary",
    metric="auc",
    num_leaves=127,
    learning_rate=0.05,
    num_trees=500,
    scale_pos_weight=999,  # 0.1% fraud → 999:1 imbalance
    feature_fraction=0.8,
    bagging_fraction=0.8,
)

# Feature groups (100+ total features)
features = concat([
    transaction_features,    # amount, currency, merchant_cat
    user_features,           # account_age, lifetime_txns, avg_amount
    device_features,         # device_age, device_trust_score, is_vpn
    velocity_features,       # txn_count_1h, unique_merchants_24h
    behavioral_features,     # time_on_site, pages_viewed
    historical_features,     # user_fraud_rate, merchant_fraud_rate
    geo_features,            # distance_from_home, country_risk
    derived_features,        # amount_vs_avg, time_since_last_txn
])

# Training with class imbalance handling
# Option 1: scale_pos_weight (weight fraud examples higher)
# Option 2: SMOTE (synthetic minority oversampling)
# Option 3: Focal loss (down-weight easy negatives)
# Best: scale_pos_weight + hard negative mining

# Serving: ~2ms inference on CPU for 100 features
# Feature importance: SHAP values per prediction → explainability`} />
          </Card>
          <Card accent="#9333ea">
            <Label color="#9333ea">Deep Sequential Model (Advanced)</Label>
            <CodeBlock code={`# Deep model for transaction SEQUENCE analysis
# Captures temporal patterns GBDT misses

class FraudSequenceModel(nn.Module):
    def __init__(self):
        # Transaction embedding
        self.txn_encoder = nn.Sequential(
            nn.Linear(TXN_FEATURES, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        # Sequence encoder (last 50 transactions)
        self.sequence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=128
            ),
            num_layers=2,
        )
        # Context features (user, device, geo)
        self.context_encoder = nn.Linear(CONTEXT_DIM, 64)

        # Fraud prediction head
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, current_txn, txn_history, context):
        # Encode current transaction
        curr_emb = self.txn_encoder(current_txn)

        # Encode transaction history (last 50)
        hist_embs = self.txn_encoder(txn_history)
        # Self-attention over history
        # "Which past transactions are relevant to evaluating this one?"
        seq_emb = self.sequence_encoder(hist_embs)
        seq_summary = seq_emb.mean(dim=1)  # pooling

        # Context features
        ctx_emb = self.context_encoder(context)

        # Combine and classify
        combined = concat([curr_emb, seq_summary, ctx_emb])
        logit = self.classifier(combined)
        return sigmoid(logit)

# WHY SEQUENCE MATTERS:
# Transaction alone: $200 at electronics store → normal
# Sequence: 5 x $200 at different electronics stores in 2 hours
#           → stolen card testing limits before cashing out
# GBDT sees velocity features. Transformer sees the PATTERN.`} />
          </Card>
        </div>
      </Card>
      <Card>
        <Label color="#d97706">Model Comparison — When to Use Each</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[12px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Dimension</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">GBDT (LightGBM) ★</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Deep Sequential</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Ensemble (Both)</th>
            </tr></thead>
            <tbody>
              {[
                { dim: "Inference Latency", gbdt: "~2ms (CPU)", deep: "~12ms (CPU)", ens: "~15ms (parallel)" },
                { dim: "AUC-ROC", gbdt: "0.95-0.97", deep: "0.96-0.98", ens: "0.97-0.99" },
                { dim: "Explainability", gbdt: "High (SHAP)", deep: "Low (black box)", ens: "Medium" },
                { dim: "Handles sequences", gbdt: "No (uses aggregates)", deep: "Yes (attention)", ens: "Yes" },
                { dim: "Training speed", gbdt: "Minutes", deep: "Hours", ens: "Hours" },
                { dim: "Cold start (new users)", gbdt: "OK (contextual features)", deep: "Poor (needs history)", ens: "OK" },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-bold text-stone-700">{r.dim}</td>
                  <td className="text-center px-3 py-2 text-stone-600">{r.gbdt}</td>
                  <td className="text-center px-3 py-2 text-stone-600">{r.deep}</td>
                  <td className="text-center px-3 py-2 text-emerald-700 font-bold">{r.ens}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function FeaturesSection() {
  return (
    <div className="space-y-5">
      <Card accent="#d97706">
        <Label color="#d97706">Feature Categories for Fraud Detection</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[11px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Category</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Example Features</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Computed</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Impact</th>
            </tr></thead>
            <tbody>
              {[
                { cat: "Device Trust", ex: "device_id, device_age, device_fraud_rate, is_emulator, is_rooted, screen_res, timezone_match", comp: "Online", impact: "Very High" },
                { cat: "Velocity / Frequency", ex: "txn_count_1h/6h/24h, unique_merchants_24h, unique_cards_24h, amount_sum_24h", comp: "Real-time (Redis)", impact: "Very High" },
                { cat: "Transaction", ex: "amount, currency, merchant_category, is_digital, is_international, time_of_day", comp: "Online (event)", impact: "High" },
                { cat: "User Behavioral", ex: "avg_txn_amount, txn_frequency, typical_merchants, typical_time_of_day, account_age_days", comp: "Batch (daily)", impact: "Very High" },
                { cat: "Geo / IP", ex: "ip_country, ip_is_datacenter, ip_is_vpn/tor, distance_from_home, geo_impossible_travel", comp: "Online", impact: "High" },
                { cat: "Session / Behavioral", ex: "time_on_site_sec, pages_viewed, mouse_movement_entropy, typing_speed", comp: "Online (session)", impact: "High" },
                { cat: "Cross-Account", ex: "device_shared_with_n_accounts, ip_shared_with_n_accounts, email_similarity_cluster", comp: "Near-real-time", impact: "Very High" },
                { cat: "Historical Risk", ex: "user_chargeback_rate, merchant_fraud_rate, card_BIN_fraud_rate, country_risk_score", comp: "Batch", impact: "High" },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-bold text-stone-700">{r.cat}</td>
                  <td className="px-3 py-2 text-stone-500 font-mono text-[10px]">{r.ex}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.comp}</td>
                  <td className="text-center px-3 py-2"><Pill bg={r.impact.includes("Very")?"#fef2f2":"#fffbeb"} color={r.impact.includes("Very")?"#dc2626":"#d97706"}>{r.impact}</Pill></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Velocity Features — The Most Powerful Signal</Label>
          <CodeBlock code={`# Velocity features capture "how fast is this entity acting?"
# Implemented as streaming counters in Redis

class VelocityFeatureStore:
    # Sliding window counters for multiple time windows
    windows = ["1m", "5m", "1h", "6h", "24h", "7d"]

    def record_event(self, event):
        # Increment counters for multiple aggregation keys
        keys = [
            f"card:{event.card_hash}",
            f"user:{event.user_id}",
            f"device:{event.device_id}",
            f"ip:{event.ip}",
            f"merchant:{event.merchant_id}",
        ]
        for key in keys:
            for window in self.windows:
                redis.incr(f"vel:{key}:{window}")
                redis.expire(f"vel:{key}:{window}", window_to_seconds(window))

    def get_velocities(self, event):
        features = {}
        for entity in ["card", "user", "device", "ip"]:
            entity_key = getattr(event, entity + "_id_or_hash")
            for window in self.windows:
                features[f"{entity}_txn_count_{window}"] = redis.get(
                    f"vel:{entity}:{entity_key}:{window}"
                ) or 0
            # Also: unique counterparties
            features[f"{entity}_unique_merchants_{window}"] = redis.pfcount(
                f"vel:{entity}:{entity_key}:merchants:{window}"
            )
            # And: total amount
            features[f"{entity}_amount_sum_{window}"] = redis.get(
                f"vel:{entity}:{entity_key}:amount:{window}"
            ) or 0
        return features

# WHY VELOCITY IS SO POWERFUL:
# Legitimate user: 2 txns/day, 1 merchant, ~$50 avg
# Stolen card: 10 txns/hour, 5 merchants, $200 each
# The PATTERN screams fraud even if each txn looks normal alone`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Device Fingerprinting</Label>
          <CodeBlock code={`# Device fingerprinting — unique device identification
# Even without cookies, devices can be identified

def compute_device_fingerprint(request):
    # Hardware signals (hard to fake)
    hardware = {
        "screen_resolution": request.screen_res,
        "color_depth": request.color_depth,
        "cpu_cores": request.hardware_concurrency,
        "device_memory_gb": request.device_memory,
        "gpu_renderer": request.webgl_renderer,
        "max_touch_points": request.max_touch_points,
    }

    # Software signals (medium difficulty to fake)
    software = {
        "user_agent": request.user_agent,
        "platform": request.platform,
        "timezone": request.timezone,
        "language": request.language,
        "installed_fonts_hash": hash(request.fonts),
        "canvas_fingerprint": request.canvas_hash,
        "audio_fingerprint": request.audio_context_hash,
    }

    # Network signals (easy to change but informative)
    network = {
        "ip_asn": geoip.get_asn(request.ip),
        "is_datacenter": ip_db.is_datacenter(request.ip),
        "is_vpn": vpn_db.check(request.ip),
        "is_tor": tor_db.check(request.ip),
    }

    # Behavioral biometrics (hardest to fake)
    behavioral = {
        "typing_speed_ms": request.avg_keystroke_interval,
        "mouse_movement_entropy": request.mouse_entropy,
        "scroll_pattern": request.scroll_hash,
    }

    # Combine into stable device ID
    device_id = stable_hash(hardware, software)
    # Note: software changes with updates, so use LSH
    # for fuzzy matching across versions

    return device_id, {**hardware, **software, **network, **behavioral}

# Device trust score:
#   New device → trust = 0.2
#   Seen 10+ times with this user, no fraud → trust = 0.9
#   Seen with 5+ different accounts → trust = 0.05 (likely fraud)`} />
        </Card>
      </div>
    </div>
  );
}

function GraphSection() {
  return (
    <div className="space-y-5">
      <Card accent="#0f766e">
        <Label color="#0f766e">Graph-Based Fraud Detection — Finding Fraud Rings</Label>
        <p className="text-[12px] text-stone-500 mb-4">Individual transactions may look legitimate, but when you connect accounts through shared devices, IPs, payment methods, and addresses, fraud rings become visible. Graph analysis is Layer 3 — too slow for inline scoring but catches coordinated attacks that ML and rules miss.</p>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Fraud Graph Construction" code={`# Build a heterogeneous graph of entities and relationships
# Nodes: users, devices, IPs, cards, addresses, emails
# Edges: "user X used device Y", "device Y had IP Z"

class FraudGraph:
    def build(self, events, window="30d"):
        G = HeterogeneousGraph()

        for event in events.last(window):
            # Add nodes
            G.add_node(event.user_id, type="user")
            G.add_node(event.device_id, type="device")
            G.add_node(event.ip, type="ip")
            G.add_node(event.card_hash, type="card")
            G.add_node(event.email_domain, type="email_domain")

            # Add edges
            G.add_edge(event.user_id, event.device_id, "used_device")
            G.add_edge(event.user_id, event.card_hash, "used_card")
            G.add_edge(event.device_id, event.ip, "had_ip")
            G.add_edge(event.user_id, event.email_domain, "has_email")

        return G

    def detect_fraud_rings(self, G):
        # Community detection — find clusters of connected entities
        communities = louvain_communities(G)

        suspicious = []
        for community in communities:
            users = [n for n in community if G.nodes[n]["type"] == "user"]
            devices = [n for n in community if G.nodes[n]["type"] == "device"]

            # Suspicious: many users sharing few devices
            if len(users) > 3 and len(devices) < len(users):
                suspicious.append({
                    "users": users,
                    "shared_devices": devices,
                    "risk": "fraud_ring",
                    "confidence": min(len(users) / len(devices) / 5, 1.0),
                })

        return suspicious

# Real example:
# 20 accounts, all created in past week
# All share 3 devices and 2 IP ranges
# All made purchases with different stolen cards
# Each individual account looks new-but-plausible
# ONLY the graph reveals they're connected`} />
          <div className="space-y-4">
            <Card accent="#0f766e">
              <Label color="#0f766e">Graph Neural Networks (GNN) for Fraud</Label>
              <CodeBlock code={`# GNN — learns fraud patterns from graph structure
# Node classification: is this user node fraudulent?

class FraudGNN(nn.Module):
    def __init__(self):
        # Graph attention network
        self.conv1 = GATConv(
            in_channels=FEATURE_DIM,
            out_channels=128,
            heads=4,
        )
        self.conv2 = GATConv(128 * 4, 64, heads=1)
        self.classifier = nn.Linear(64, 1)

    def forward(self, node_features, edge_index):
        # Message passing: each node aggregates
        # information from its neighbors
        x = self.conv1(node_features, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # Classify each user node
        fraud_score = sigmoid(self.classifier(x))
        return fraud_score

# WHY GNN > handcrafted graph features:
# Handcrafted: "shares device with N accounts"
# GNN: learns that sharing devices with accounts
#      that ALSO share IPs with OTHER accounts
#      that have high chargeback rates → 3-hop pattern
# GNN captures multi-hop patterns automatically`} />
            </Card>
          </div>
        </div>
      </Card>
    </div>
  );
}

function DataModelSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Core Data Stores</Label>
          <CodeBlock code={`-- Blocklist DB (Redis, in-memory)
-- Known-bad entities for instant blocking
entity_type:entity_hash -> {
  reason: "stolen_card" | "banned_device" | "fraud_ip",
  added_ts: timestamp,
  source: "internal" | "card_network" | "law_enforcement",
  expires_at: timestamp | null,
}
# ~100M entries. Bloom filter for fast negative check.

-- Velocity Counters (Redis, TTL-based)
vel:{entity}:{id}:{window} -> count
vel:{entity}:{id}:amount:{window} -> sum
# Sliding window counters. Auto-expire via Redis TTL.
# Updated synchronously on every transaction.

-- User Risk Profile (Bigtable)
user_id -> {
  trust_score: float,
  account_age_days: int,
  lifetime_txns: int,
  chargeback_count: int,
  chargeback_rate: float,
  avg_txn_amount: float,
  typical_merchants: [category, ...],
  typical_hours: [hour_distribution],
  devices: [device_id, ...],
  last_txn: {amount, merchant, ts, geo},
}

-- Event Log (Kafka -> BigQuery)
(event_id, ts, user_id, event_type, amount,
 merchant_id, device_id, ip, geo,
 fraud_score, decision, triggered_rules,
 features_snapshot,
 label: null | "fraud" | "legitimate",
 label_ts: timestamp | null)
# Labels arrive delayed (days to weeks via chargebacks)

-- Graph Store (Neo4j / Bigtable adjacency)
(entity_a, relationship, entity_b, first_seen, last_seen, count)
# User-device, user-card, device-ip relationships
# Updated in near-real-time via streaming pipeline`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Why These Design Choices?</Label>
          <div className="space-y-3">
            {[
              { q: "Why Redis for both blocklists AND velocity?", a: "Both need sub-millisecond reads in the serving path. Blocklist: simple key lookup. Velocity: atomic increment + TTL-based sliding windows. Redis handles both perfectly. Separate Redis clusters for isolation — blocklist is read-heavy, velocity is write-heavy." },
              { q: "Why log features_snapshot in event log?", a: "Training-serving skew prevention. When the label arrives 2 weeks later, you need the exact features used at scoring time. If you recompute features, user_risk_profile has changed (more transactions, different avg_amount). Logged features ensure training data matches serving conditions exactly." },
              { q: "Why is label nullable?", a: "Most transactions never get a definitive label. Only chargebacks (~0.1%) confirm fraud. Only transactions with no chargeback after 90 days are confidently labeled 'legitimate'. The vast majority (99.9%) are assumed legitimate but never confirmed — inherent label noise in fraud detection." },
              { q: "Why Neo4j for graph AND Bigtable adjacency?", a: "Neo4j (or similar) for complex graph queries: multi-hop traversal, community detection, pattern matching. Bigtable adjacency list for simple lookups: 'what devices has this user used?' Different query patterns, different storage engines. Graph queries are async (Layer 3); adjacency lookups are inline (feature computation)." },
              { q: "Why Kafka before BigQuery?", a: "Event volume is 50K+ per second. BigQuery can't handle real-time inserts at this rate. Kafka buffers events, provides ordering guarantees, and enables real-time consumers (velocity updates, graph updates). BigQuery loads from Kafka in micro-batches (every 1 minute) for training data." },
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

function TrainingSection() {
  return (
    <div className="space-y-5">
      <Card accent="#7e22ce">
        <Label color="#7e22ce">Training Pipeline — Handling Delayed Labels & Class Imbalance</Label>
        <p className="text-[12px] text-stone-500 mb-4">Fraud training is uniquely challenging: labels arrive weeks late (chargebacks), most transactions are never definitively labeled, and the class imbalance is extreme (0.1% fraud). The training pipeline must handle all of this.</p>
        <div className="grid grid-cols-2 gap-5">
          <Card accent="#7e22ce">
            <Label color="#7e22ce">Label Construction & Timing</Label>
            <CodeBlock code={`# Fraud labels arrive with significant delay
# The "label maturity" problem

def construct_training_labels(events, label_window_days=90):
    labeled = []
    for event in events:
        age = days_since(event.timestamp)

        if age < 30:
            continue  # Too recent — label not mature enough

        # Positive label: confirmed fraud
        if event.has_chargeback:
            labeled.append((event, 1))
        elif event.was_manually_confirmed_fraud:
            labeled.append((event, 1))

        # Negative label: assumed legitimate
        elif age > label_window_days and not event.has_chargeback:
            labeled.append((event, 0))
            # After 90 days with no chargeback, very likely legitimate
            # But some fraud is never reported (friendly fraud)

        else:
            continue  # Label not yet mature

    return labeled

# LABEL DELAY TIMELINE:
# Day 0: Transaction occurs, scored in real-time
# Day 1-3: Some fraud detected by issuing bank
# Day 7-14: Most chargebacks filed by cardholders
# Day 30: ~90% of chargebacks have arrived
# Day 90: ~99% of chargebacks have arrived
# Day 120+: Chargeback window closes (Visa/MC rules)
#
# CONSEQUENCE: model is always training on 30-90 day old data
# Fast-moving fraud patterns may not be in training data yet
# → This is why online learning and rules are essential bridges`} />
          </Card>
          <Card accent="#ea580c">
            <Label color="#ea580c">Handling Class Imbalance</Label>
            <CodeBlock code={`# 0.1% fraud rate = 999:1 imbalance
# Standard training would learn to predict "not fraud" for everything

# Strategy 1: Cost-sensitive learning
# Weight fraud examples higher in the loss function
model = LightGBM(
    scale_pos_weight=200,  # fraud examples count 200x
    # Not 999x — that causes overfitting to fraud patterns
    # Tune via validation set to optimize precision@recall=0.95
)

# Strategy 2: Negative downsampling
# Keep ALL fraud examples, downsample negatives
def downsample(events, target_fraud_rate=0.05):
    fraud = [e for e in events if e.label == 1]
    legit = [e for e in events if e.label == 0]
    # Sample legit to get 5% fraud rate
    sample_n = int(len(fraud) / target_fraud_rate - len(fraud))
    legit_sample = random.sample(legit, min(sample_n, len(legit)))
    return fraud + legit_sample
# Must calibrate predictions after: p_true = p_model * k

# Strategy 3: Hard negative mining
# Focus on the HARDEST legitimate transactions
# (the ones that look most like fraud)
def hard_negative_mining(model, legit_events, top_k=50000):
    scores = model.predict(legit_events)
    # Keep the legitimate txns with highest fraud scores
    # These are the cases the model struggles with
    hard_negatives = sorted(
        zip(legit_events, scores), key=lambda x: -x[1]
    )[:top_k]
    return [e for e, s in hard_negatives]

# Strategy 4: Focal Loss (deep models)
# Down-weight easy examples, focus on hard ones
# L = -alpha * (1-p)^gamma * log(p)
# gamma=2 → easy negatives contribute almost nothing to loss`} />
          </Card>
        </div>
      </Card>
    </div>
  );
}

function RealtimeSection() {
  return (
    <div className="space-y-5">
      <Card accent="#ea580c">
        <Label color="#ea580c">Real-Time Scoring — The Serving Path</Label>
        <div className="grid grid-cols-2 gap-5">
          <Card accent="#ea580c">
            <Label color="#ea580c">Scoring Flow</Label>
            <CodeBlock code={`# Real-time fraud scoring — called at checkout
async def score_transaction(event: TransactionEvent):
    t0 = time.now()

    # LAYER 1: Rules Engine (~5ms)
    rule_results = rules_engine.evaluate(event)
    if any(r.action == "BLOCK" for r in rule_results):
        log_decision(event, "block", "rules", rule_results)
        return FraudResponse(decision="block", score=1.0)

    # LAYER 2: Feature Assembly + ML (~20ms)
    # Parallel fetch: user profile + device trust + velocities
    user_profile = feature_store.get_async(event.user_id)
    device_trust = device_store.get_async(event.device_id)
    velocities = velocity_store.get_velocities(event)

    features = assemble_features(
        event=event,
        user=await user_profile,
        device=await device_trust,
        velocities=velocities,
        rule_scores=[r.score for r in rule_results],
    )

    # ML scoring
    fraud_score = model.predict(features)

    # DECISION LOGIC
    if fraud_score > 0.85:
        decision = "block"
    elif fraud_score > 0.5:
        if event.amount > 500:
            decision = "review"
        else:
            decision = "challenge"  # step-up auth
    elif fraud_score > 0.3 and event.amount > 1000:
        decision = "challenge"
    else:
        decision = "allow"

    # UPDATE VELOCITY COUNTERS (async)
    velocity_store.record_event_async(event)

    # LOG EVERYTHING (async, for training data)
    log_decision(event, decision, fraud_score, features)

    elapsed = time.now() - t0  # target: <50ms
    return FraudResponse(
        decision=decision,
        score=fraud_score,
        latency_ms=elapsed,
    )`} />
          </Card>
          <div className="space-y-4">
            <Card accent="#2563eb">
              <Label color="#2563eb">Latency Optimization</Label>
              <div className="space-y-2">
                {[
                  { tech: "Parallel feature fetch", desc: "User profile, device trust, and velocity counters fetched simultaneously. Latency = max(individual fetches), not sum.", saves: "~15ms" },
                  { tech: "CPU-only inference", desc: "GBDT models run on CPU with predictable latency. GPU inference has variable warm-up and kernel launch overhead — dangerous for p99 SLAs.", saves: "Consistency" },
                  { tech: "Feature caching", desc: "Hot user profiles cached in local memory (LRU). 10% of users generate 50% of transactions. Cache hit rate: ~40%.", saves: "~5ms" },
                  { tech: "Model quantization", desc: "LightGBM tree traversal on INT8 quantized features. 2x speedup vs float32. Negligible accuracy loss.", saves: "~2ms" },
                  { tech: "Pre-computed risk scores", desc: "User trust_score, device trust_score computed in batch. Avoids complex computation at serving time.", saves: "~3ms" },
                  { tech: "Async logging", desc: "Decision logging and velocity updates are fire-and-forget to Kafka. Zero impact on response latency.", saves: "~2ms" },
                ].map((t,i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-blue-100 text-blue-700 shrink-0">{t.saves}</span>
                    <div>
                      <div className="text-[11px] font-bold text-stone-700">{t.tech}</div>
                      <div className="text-[10px] text-stone-500">{t.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
            <Card className="bg-red-50/50 border-red-200">
              <Label color="#dc2626">Graceful Degradation</Label>
              <div className="space-y-1.5 text-[11px] text-stone-600">
                <div><strong>Feature store down?</strong> → Use only transaction-level features. Accuracy drops ~15% but system stays online.</div>
                <div><strong>ML model down?</strong> → Fall back to rules engine only. Catches ~60% of fraud (known patterns). Better than nothing.</div>
                <div><strong>Both down?</strong> → Allow all transactions below $100, challenge above $100, block above $5000. Crude but prevents catastrophic loss.</div>
                <div className="text-red-600 font-bold mt-2">NEVER fail open unconditionally — that's what fraudsters wait for.</div>
              </div>
            </Card>
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
          <Label color="#059669">Serving Infrastructure Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Stateless scoring servers</strong> — model + rules loaded in memory at startup. Horizontally scalable. Auto-scale based on TPS. Each server handles ~5K TPS → need ~12 servers baseline, ~60 at peak.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Redis cluster for velocity</strong> — sharded by entity hash (user_id, card_hash). Each shard handles ~10K writes/sec. 60K TPS × 5 counter updates = 300K Redis ops/sec → ~30 shards.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Feature store tiering</strong> — hot user profiles (top 10% by activity) in Redis L1 cache. Warm profiles in Bigtable with row cache. Cold profiles (inactive users) in Bigtable without cache.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Model replication</strong> — same model replicated to all serving regions. Model updates pushed via global distribution service. Canary deployment: 1% traffic → full rollout.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Graph Analysis Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Async batch processing</strong> — graph analysis runs on materialized views, not live queries. Community detection runs hourly on the latest graph snapshot. Results (fraud ring membership scores) are pushed to the feature store.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Graph partitioning</strong> — partition by geography (most fraud rings are regional). US graph, EU graph, Asia graph processed independently. Cross-region edges handled separately.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Incremental graph updates</strong> — don't rebuild the entire graph hourly. Streaming pipeline adds new edges as transactions occur. Community detection runs incrementally on changed subgraphs.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">GNN mini-batch inference</strong> — for node-level fraud scoring, sample k-hop neighborhoods rather than scoring the entire graph. Limits computation to relevant subgraph per user.</Point>
          </ul>
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
          { title: "Adversarial Adaptation", sev: "CRITICAL", desc: "Fraudsters actively probe the system to learn thresholds. Small test transactions that get approved tell them the detection boundary. They then stay just below it. Or they find feature blind spots (new device types, new geos).", fix: "Randomize thresholds slightly (add noise to decision boundary). Don't return exact fraud scores to clients. Retrain models frequently (weekly). Monitor for 'probing patterns' (many small transactions from same entity testing limits). Ensemble models are harder to reverse-engineer than single models.", icon: "🔴" },
          { title: "Label Delay & Noise", sev: "CRITICAL", desc: "Training data is 30-90 days old by the time labels arrive. New fraud patterns emerge and spread before the model sees them in training. Additionally, ~20% of chargebacks are 'friendly fraud' (legitimate user disputes real purchase) — mislabeled training data.", fix: "Rules engine as a fast-response bridge (deploy in hours, not weeks). Semi-supervised learning on unlabeled recent data. Separate 'confirmed fraud' from 'chargeback' labels in training. Weight recent data higher than old data. Active learning: prioritize labeling recent suspicious events.", icon: "🔴" },
          { title: "False Positive Impact on Users", sev: "HIGH", desc: "Blocking a legitimate $2000 purchase is terrible UX. Customer calls support, waits 30 minutes, loses trust. For high-value transactions, the cost of a false positive can exceed the cost of a false negative.", fix: "Step-up authentication instead of hard block for medium-risk scores. Amount-aware thresholds: higher threshold for blocking high-value transactions. Real-time customer notification with one-tap confirmation. Fast unblock process (support can override within minutes).", icon: "🟡" },
          { title: "Concept Drift", sev: "HIGH", desc: "Fraud patterns shift as attackers adapt and technology changes. A model trained on last quarter's fraud may not recognize this quarter's tactics. Seasonal patterns (holiday shopping) also shift feature distributions.", fix: "Monitor model performance daily (AUC, precision, recall on recent labeled data). Automatic retraining trigger when performance degrades below threshold. Feature drift detection: alert when feature distributions shift significantly from training data.", icon: "🟡" },
          { title: "Data Leakage in Training", sev: "MEDIUM", desc: "Using future information in features: e.g., 'user_chargeback_rate' computed over ALL time including after the event. Or velocity features computed with knowledge of future events. Results in over-optimistic offline metrics.", fix: "Strict temporal splits: train on data before time T, evaluate on data after time T. Compute all features point-in-time (as they would have been at event time). Use logged features from serving, not recomputed features.", icon: "🟠" },
          { title: "Model Theft / Reverse Engineering", sev: "MEDIUM", desc: "Sophisticated fraudsters may try to reconstruct the model by observing which transactions get blocked. If they can infer feature importance and thresholds, they craft transactions that specifically evade detection.", fix: "Don't expose fraud scores or specific reasons to end users (only to internal analysts). Add randomness to decision boundaries. Use multiple models in ensemble (harder to reverse-engineer). Rate-limit transaction attempts from suspicious entities.", icon: "🟠" },
        ].map((w,i) => (
          <Card key={i} accent="#dc2626">
            <div className="flex items-center gap-2 mb-2">
              <span>{w.icon}</span>
              <span className="text-[12px] font-bold text-stone-800">{w.title}</span>
              <Pill bg={w.sev==="CRITICAL"?"#fef2f2":w.sev==="HIGH"?"#fffbeb":"#fff7ed"} color={w.sev==="CRITICAL"?"#dc2626":w.sev==="HIGH"?"#d97706":"#ea580c"}>{w.sev}</Pill>
            </div>
            <p className="text-[12px] text-stone-500 mb-2">{w.desc}</p>
            <div className="text-[11px] text-emerald-700 bg-emerald-50 rounded-lg p-2.5">
              <strong>Fix:</strong> {w.fix}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

function ObservabilitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-5">
        <Card accent="#0284c7">
          <Label color="#0284c7">Detection Metrics</Label>
          <div className="space-y-2.5">
            {[
              { metric: "Fraud Detection Rate (Recall)", target: ">95%", why: "% of actual fraud caught. Each miss = direct financial loss." },
              { metric: "Precision @ 95% Recall", target: ">85%", why: "Of blocked transactions, how many are truly fraud?" },
              { metric: "Dollar-Weighted Detection Rate", target: ">98%", why: "High-value fraud matters more. Weight by transaction amount." },
              { metric: "Mean Time to Detection", target: "<1 sec (inline)", why: "Fraud caught before money moves. Post-hoc is much more expensive." },
              { metric: "Chargeback Rate", target: "<0.1%", why: "Industry standard. Above 1% → card network penalties." },
            ].map((m,i) => (
              <div key={i} className="border-b border-stone-100 pb-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold text-stone-700">{m.metric}</span>
                  <span className="text-[10px] font-mono text-cyan-700">{m.target}</span>
                </div>
                <div className="text-[10px] text-stone-400 mt-0.5">{m.why}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">User Impact Metrics</Label>
          <div className="space-y-2.5">
            {[
              { metric: "False Positive Rate", target: "<0.5%", why: "Legitimate users blocked. Each is a support call + lost revenue." },
              { metric: "Challenge Pass Rate", target: ">80%", why: "Of users sent to step-up auth, 80%+ should pass (= were legitimate)." },
              { metric: "Block-to-Appeal Rate", target: "<10%", why: "Blocked users who appeal. High = too many false positives." },
              { metric: "Avg Resolution Time", target: "<15 min", why: "Time for falsely blocked user to get unblocked." },
              { metric: "User Churn After FP", target: "<5%", why: "Users who leave platform after being falsely blocked." },
            ].map((m,i) => (
              <div key={i} className="border-b border-stone-100 pb-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold text-stone-700">{m.metric}</span>
                  <span className="text-[10px] font-mono text-red-600">{m.target}</span>
                </div>
                <div className="text-[10px] text-stone-400 mt-0.5">{m.why}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">System Health & Alerts</Label>
          <div className="space-y-2.5">
            {[
              { alert: "Scoring latency p99 > 100ms", sev: "P0", action: "Scale servers, check feature store" },
              { alert: "Block rate spikes > 2x baseline", sev: "P0", action: "Possible model issue or attack — investigate immediately" },
              { alert: "Block rate drops > 50%", sev: "P0", action: "Model may have failed open — check deployment" },
              { alert: "Chargeback rate > 0.5%", sev: "P1", action: "Model missing new fraud pattern — retrain" },
              { alert: "Rules engine latency > 10ms", sev: "P1", action: "Check Redis cluster, blocklist size" },
            ].map((a,i) => (
              <div key={i} className="border-b border-stone-100 pb-2">
                <div className="flex items-center gap-2">
                  <Pill bg={a.sev==="P0"?"#fef2f2":"#fffbeb"} color={a.sev==="P0"?"#dc2626":"#d97706"}>{a.sev}</Pill>
                  <span className="text-[11px] text-stone-700">{a.alert}</span>
                </div>
                <div className="text-[10px] text-stone-400 mt-0.5 ml-9">→ {a.action}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function FollowupsSection() {
  const [exp, setExp] = useState(null);
  const qas = [
    { q:"Why use a layered defense instead of one powerful ML model?", a:"Four reasons: (1) Speed — rules are 5ms, ML is 15ms, graph is 200ms. Rules catch obvious fraud instantly, before ML even runs. (2) Coverage — rules catch known patterns perfectly (stolen card list = 100% precision). ML catches novel patterns. Graph catches coordinated attacks. Each layer covers what others miss. (3) Deployability — a new fraud pattern discovered at 2am can be blocked by a rule within minutes. Retraining ML takes hours/days. Rules bridge the gap. (4) Resilience — if ML is down, rules still catch ~60% of fraud. If rules are down, ML still works. Layered defense degrades gracefully. A single model is a single point of failure.", tags:["architecture"] },
    { q:"How do you handle the 30-90 day label delay?", a:"Label delay is the hardest operational challenge in fraud. Multi-pronged approach: (1) Train on mature data (30-90 days old) but accept this means the model is always slightly outdated. (2) Use rules engine as a fast bridge for new patterns — deploy rules in hours when new fraud is detected. (3) Semi-supervised learning: use the model's own predictions on recent unlabeled data as pseudo-labels. High-confidence predictions (score > 0.95 or < 0.05) are likely correct. (4) Active learning: route borderline cases to manual review — human labels available within hours, not weeks. Feed these into training immediately. (5) Transfer learning: new fraud patterns often resemble past ones. Pre-train on historical fraud, fine-tune on recent data.", tags:["ml"] },
    { q:"How is fraud detection different from content moderation?", a:"Key differences: (1) Latency — fraud is inline (50ms), moderation is often async (seconds to minutes). (2) Cost asymmetry direction — fraud FN = direct financial loss. Moderation FN = user exposure to harm (indirect). Fraud FP = user friction. Moderation FP = censorship. (3) Adversarial sophistication — fraudsters are financially motivated professionals who probe systematically. Content violators are often impulsive. (4) Label quality — fraud has relatively clear labels (chargeback = fraud). Moderation labels are subjective (is this hate speech?). (5) Feature types — fraud relies heavily on behavioral and transactional features. Moderation relies on content understanding (NLP, vision). (6) Monetization — fraud detection directly protects revenue. Moderation protects reputation and regulatory compliance.", tags:["design"] },
    { q:"How do you detect account takeover (ATO) specifically?", a:"ATO is when a fraudster gains access to a legitimate user's account. Signals: (1) Device change — the most powerful signal. If the login is from a never-seen-before device, risk increases 10x. (2) Behavioral anomalies — different typing speed, different navigation patterns, different purchase categories than the real user. (3) Session anomalies — ATO sessions are typically shorter (fraudster knows what they want), have fewer page views, and go straight to high-value items. (4) Password/email change — changing credentials immediately after login from new device is a strong ATO indicator. (5) Impossible travel — login from NYC, then purchase from Lagos 30 minutes later. (6) Multiple account access — same device accessing many different accounts is a credential stuffing indicator. Response: challenge with MFA. If MFA not set up, temporary account freeze + notification to the user.", tags:["fraud-types"] },
    { q:"How do you A/B test a fraud model without losing money?", a:"Fraud A/B testing is uniquely risky — the control group (old model) may let fraud through, and the treatment group (new model) may block legitimate users. Approach: (1) Shadow scoring: run new model on all traffic in parallel. Compare predictions but DON'T act on new model's decisions. Measure: where do they disagree? (2) Asymmetric experiment: for transactions where old model says 'allow' but new model says 'block', send to manual review instead of blocking. This gives us ground truth labels without blocking legitimate users. (3) Labeled evaluation: wait for chargebacks (30-90 days) on shadow-scored data. Compare AUC, precision, recall. (4) Conservative ramp: 1% → 5% → 20% → 100%, with chargeback rate monitoring at each stage. (5) Guardrails: auto-rollback if false positive rate increases >0.1% or if block rate changes by >20%.", tags:["evaluation"] },
    { q:"What's the role of unsupervised learning in fraud detection?", a:"Supervised learning needs labeled fraud — but new fraud types have no labels. Unsupervised methods fill this gap: (1) Anomaly detection (Isolation Forest, Autoencoders) — model learns 'normal' transaction patterns. Anything far from normal is flagged for review. Catches novel fraud types that the supervised model hasn't seen. (2) Clustering — group similar transactions. Clusters with high post-hoc chargeback rates are likely fraud patterns. Helps discover new fraud types for labeling. (3) Graph community detection — find tightly connected clusters of entities that behave differently from the rest of the graph. Fraud rings form distinctive communities. (4) Sequence anomaly detection — model normal transaction sequences per user. Alert when a user's recent sequence deviates significantly from their historical pattern. Use as auxiliary features fed into the supervised model.", tags:["ml"] },
    { q:"How do you handle cross-border fraud detection?", a:"Cross-border fraud is harder because: (1) Different fraud patterns per country — fraud in Brazil looks different from fraud in Japan. (2) Different regulations — PSD2 in Europe requires Strong Customer Authentication. (3) Different payment methods — credit cards in US, UPI in India, PIX in Brazil. Approach: (1) Regional models: train country-specific models where data volume allows. (2) Transfer learning: pre-train on global data, fine-tune on regional data. Small countries benefit from global patterns. (3) Geo-risk features: country_risk_score based on historical fraud rates. Cross-border transactions (billing country ≠ merchant country) get extra scrutiny. (4) Multi-currency normalization: amount features normalized by local purchasing power, not raw USD. $200 in US is different from $200 in India.", tags:["scalability"] },
    { q:"How do you prevent the model from being biased against certain demographics?", a:"Fraud models can inadvertently discriminate: users from certain countries, age groups, or income levels may be blocked at higher rates. Mitigation: (1) Don't use protected attributes (race, gender, age) as features directly. But proxy features (zip code, name patterns) can encode the same information. (2) Monitor block rates across demographic segments. If segment X is blocked at 3x the rate of segment Y, investigate whether this reflects true fraud rate differences or model bias. (3) Disparate impact testing: run the model on a holdout set and check if precision/recall differs significantly across segments. (4) Fairness constraints in training: add regularization terms that penalize models with large performance gaps across segments. (5) Use causal reasoning: 'Would this transaction have been blocked if only the country were different?' Counterfactual fairness.", tags:["fairness"] },
    { q:"How does the system handle a massive coordinated attack?", a:"Example: a fraud ring launches 100K transactions in 5 minutes using 10K stolen cards. Detection and response: (1) Velocity detection triggers first — unusual spike in transaction volume from correlated entities (same IP range, same device fingerprint cluster). (2) Graph analysis detects the ring structure within minutes (same devices/IPs across many accounts). (3) Automated response: temporarily block ALL transactions from the identified cluster of devices/IPs/cards. Not individual analysis — cluster-level action. (4) Rate limiting: automatically throttle transaction rate from suspicious IP ranges. (5) War room: alert the fraud operations team. Manual analysis of the attack vector. Update rules to block the specific pattern. (6) Post-incident: add all identified cards to blocklist. Score all completed transactions retroactively — reverse any that were approved fraudulently. (7) Retrospective: how did the attack bypass Layer 1 (rules)? Update velocity thresholds and blocklists.", tags:["ops"] },
    { q:"Should the fraud system fail open or fail closed?", a:"This is one of the most important design decisions. Fail open = if the system is down, allow all transactions. Fail closed = if the system is down, block all transactions. The answer is NEITHER extreme. Fail open is unacceptable — fraudsters would intentionally DDoS the fraud system to process stolen cards during the outage. Fail closed is also unacceptable — it stops all legitimate commerce, costing millions per minute. The right answer is degraded operation: (1) If ML is down → use rules engine only (catches ~60% of fraud). (2) If rules are down too → use amount-based policy (allow <$100, challenge $100-$1000, block >$1000). (3) If everything is down → block all new cards and new devices, allow known good user+device pairs. This limits blast radius while maintaining most legitimate commerce.", tags:["availability"] },
  ];
  return (
    <div className="space-y-2.5">
      <Card className="bg-indigo-50/50 border-indigo-200 mb-1">
        <p className="text-[12px] text-stone-500">Common follow-up questions Google L6 interviewers ask about fraud detection. Click to reveal a strong answer.</p>
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
  api: ApiSection, design: DesignSection, taxonomy: TaxonomySection,
  rules: RulesSection, models: ModelsSection, features: FeaturesSection,
  graph: GraphSection, data: DataModelSection, training: TrainingSection,
  realtime: RealtimeSection, scalability: ScalabilitySection,
  watchouts: WatchoutsSection, observability: ObservabilitySection,
  followups: FollowupsSection,
};

export default function FraudDetectionSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Fraud / Abuse Detection</h1>
            <Pill bg="#f3e8ff" color="#7c3aed">ML System Design</Pill>
            <Pill bg="#fef2f2" color="#dc2626">Google L6</Pill>
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