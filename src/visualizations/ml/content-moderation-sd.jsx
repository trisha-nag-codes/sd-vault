import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   CONTENT MODERATION SYSTEM — ML System Design Reference (Google L6)
   Pearl white theme · 17 sections (HLD + LLD + ML Deep Dive)
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",                icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",            icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",     icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",              icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",       icon: "🏗️", color: "#9333ea" },
  { id: "taxonomy",      label: "Policy Taxonomy",         icon: "📜", color: "#c026d3" },
  { id: "models",        label: "Classification Models",   icon: "🧠", color: "#dc2626" },
  { id: "multimodal",    label: "Multi-Modal Pipeline",    icon: "🎬", color: "#d97706" },
  { id: "features",      label: "Feature Engineering",     icon: "⚙️", color: "#0f766e" },
  { id: "data",          label: "Data Model",              icon: "🗄️", color: "#059669" },
  { id: "training",      label: "Training Pipeline",       icon: "🔄", color: "#7e22ce" },
  { id: "humanloop",     label: "Human Review Loop",       icon: "👥", color: "#ea580c" },
  { id: "scalability",   label: "Scalability",             icon: "📈", color: "#059669" },
  { id: "watchouts",     label: "Failure Modes",           icon: "⚠️", color: "#dc2626" },
  { id: "observability", label: "Observability",           icon: "📊", color: "#0284c7" },
  { id: "enhancements",  label: "Enhancements",            icon: "🚀", color: "#7c3aed" },
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
            <Label>What is a Content Moderation System?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              A content moderation system automatically detects and actions policy-violating content — hate speech, violence, spam, nudity, misinformation, child safety — across text, images, video, and audio at platform scale. At YouTube, this system reviews <strong>every single upload</strong> (500 hours of video per minute) and makes enforcement decisions within seconds.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              The fundamental challenge: balance <em>safety</em> (remove harmful content before any human sees it) with <em>fairness</em> (don't wrongly remove legitimate content). False positives silence free expression; false negatives expose users to harm. This is the highest-stakes ML system at any platform.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard? (Google L6 Depth)</Label>
            <ul className="space-y-2.5">
              <Point icon="⚖️" color="#0891b2">Asymmetric costs — a false negative on CSAM (child safety) is catastrophic. A false positive on political speech is censorship. Different violation categories demand different precision/recall tradeoffs.</Point>
              <Point icon="🌍" color="#0891b2">Cultural & linguistic diversity — "hate speech" varies by language, culture, and context. A hand gesture is offensive in one country, benign in another. Must support 100+ languages with different norms.</Point>
              <Point icon="🎭" color="#0891b2">Adversarial evasion — bad actors deliberately modify content to evade classifiers: leetspeak, steganography, subtle edits to images, coded language, out-of-context framing. Constant arms race.</Point>
              <Point icon="🎬" color="#0891b2">Multi-modal — a video's visual track may be benign, but audio contains hate speech. Text overlay on an image changes its meaning. Must fuse signals across modalities.</Point>
              <Point icon="📊" color="#0891b2">Extreme class imbalance — violating content is &lt;1% of all uploads. Most content is fine. But even 0.1% false negative rate at YouTube's scale = thousands of harmful videos slip through daily.</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Scale</Label>
            <div className="space-y-2.5">
              {[
                { co: "YouTube", scale: "500 hrs video/min uploaded", approach: "Multi-stage ML + human review" },
                { co: "Meta/FB", scale: "3B+ users, billions of posts/day", approach: "Whole Post Integrity model" },
                { co: "TikTok", scale: "34M+ videos/day uploaded", approach: "Pre-publish + post-publish ML" },
                { co: "Twitter/X", scale: "500M+ tweets/day", approach: "ML classifiers + Community Notes" },
                { co: "OpenAI", scale: "100M+ users, text+image", approach: "Classifier + rule-based + RLHF" },
              ].map((e,i) => (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-20 shrink-0">{e.co}</span>
                  <span className="text-stone-500 flex-1">{e.scale}</span>
                  <span className="text-stone-400 text-[10px]">{e.approach}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <Label color="#2563eb">The Moderation Pipeline (Preview)</Label>
            <svg viewBox="0 0 360 190" className="w-full">
              <rect x={20} y={10} width={110} height={35} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
              <text x={75} y={25} textAnchor="middle" fill="#ea580c" fontSize="9" fontWeight="700" fontFamily="monospace">Upload / Post</text>
              <text x={75} y={38} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">text + image + video</text>

              <rect x={155} y={10} width={95} height={35} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
              <text x={202} y={25} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="700" fontFamily="monospace">ML Classifiers</text>
              <text x={202} y={38} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">per-policy scores</text>

              <rect x={15} y={60} width={90} height={35} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
              <text x={60} y={75} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="700" fontFamily="monospace">Auto-Allow</text>
              <text x={60} y={88} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">score &lt; low_thresh</text>

              <rect x={120} y={60} width={90} height={35} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
              <text x={165} y={75} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="700" fontFamily="monospace">Human Review</text>
              <text x={165} y={88} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">uncertain zone</text>

              <rect x={225} y={60} width={90} height={35} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
              <text x={270} y={75} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="700" fontFamily="monospace">Auto-Remove</text>
              <text x={270} y={88} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">score &gt; high_thresh</text>

              <defs><marker id="ah-cm" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
              <line x1={130} y1={28} x2={155} y2={28} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-cm)"/>
              <line x1={190} y1={45} x2={60} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-cm)"/>
              <line x1={202} y1={45} x2={165} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-cm)"/>
              <line x1={215} y1={45} x2={270} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-cm)"/>

              <rect x={20} y={115} width={310} height={60} rx={6} fill="#faf9f7" stroke="#e7e5e4"/>
              <text x={30} y={132} fill="#dc2626" fontSize="7" fontWeight="600" fontFamily="monospace">High-confidence violations auto-removed (CSAM: near-zero threshold for false negatives)</text>
              <text x={30} y={147} fill="#d97706" fontSize="7" fontWeight="600" fontFamily="monospace">Uncertain cases routed to human reviewers (~2-5% of all content)</text>
              <text x={30} y={162} fill="#059669" fontSize="7" fontWeight="600" fontFamily="monospace">~95% of content auto-allowed (clear non-violation, no human needed)</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Google Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Trust & Safety is a top Google priority</div>
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
            <p className="text-[12px] text-stone-500 mt-0.5">"Design a content moderation system" is broad. Scope proactively: "I'll focus on multi-modal content moderation for a video platform like YouTube — covering the classification pipeline from upload to enforcement action. I'll design for multiple policy categories with different precision/recall tradeoffs, and include the human review integration. I'll treat content understanding (ASR, OCR) as upstream black boxes." Show L6 ownership by noting the precision/recall asymmetry upfront.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Classify content across multiple policy categories (hate speech, violence, nudity, spam, CSAM, misinformation, harassment)</Point>
            <Point icon="2." color="#059669">Support multi-modal input: text, images, video (frames + audio), and combinations</Point>
            <Point icon="3." color="#059669">Produce per-category violation scores with confidence levels</Point>
            <Point icon="4." color="#059669">Route uncertain cases to human reviewers with prioritization</Point>
            <Point icon="5." color="#059669">Support appeals: users can contest decisions, triggering re-review</Point>
            <Point icon="6." color="#059669">Enforce actions: remove, age-gate, demonetize, reduce distribution, warn</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Latency: pre-publish scan &lt;5s for text/images, &lt;60s for video</Point>
            <Point icon="2." color="#dc2626">Throughput: process 500 hrs of video/min + millions of text posts/sec</Point>
            <Point icon="3." color="#dc2626">CSAM recall: &gt;99.9% — virtually zero false negatives for child safety</Point>
            <Point icon="4." color="#dc2626">Overall precision: &gt;95% — minimize wrongful removals (free speech)</Point>
            <Point icon="5." color="#dc2626">Availability: 99.99% — moderation pipeline cannot be bypassed</Point>
            <Point icon="6." color="#dc2626">Freshness: new policy rules deployable within hours, not weeks</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask (L6 Signal)</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Pre-publish (block before anyone sees) or post-publish (detect and remove after)?",
            "Which modalities: text only, image+text, or full video (frames + audio + text overlay)?",
            "What's the precision/recall priority? Varies by violation category.",
            "How many languages? 10 tier-1 languages or 100+ including low-resource?",
            "What enforcement actions? Binary remove/keep or graduated (warn, limit, demonetize)?",
            "Human reviewer capacity? 10K reviewers or 100K? Determines automation threshold.",
            "Real-time only or also retroactive (policy change applies to existing content)?",
            "Are there regulatory requirements (DSA, COPPA, NetzDG) driving SLAs?",
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
            <div className="text-[12px] font-bold text-violet-700">Interview Tip — Safety-Driven Capacity</div>
            <p className="text-[12px] text-stone-500 mt-0.5">For moderation, the key constraint is that EVERY piece of content must be scanned — you can't sample. This means your throughput must match upload rate with zero backlog tolerance. Frame your estimates around worst-case (peak upload rate) and SLA violations.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Step 1 — Content Volume</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Video uploads = 500 hrs/min" result="~720K hrs/day" note="YouTube scale. Each video needs frame + audio + text analysis." />
            <MathStep step="2" formula="Avg video duration = 7 min" result="~4,300 videos/min" note="Uploaded, not watched. Each triggers the moderation pipeline." />
            <MathStep step="3" formula="Text content (comments, posts)" result="~500M/day" note="Comments, community posts, live chat messages." />
            <MathStep step="4" formula="Images (thumbnails, community)" result="~50M/day" note="Each thumbnail, community post image, profile picture." />
            <MathStep step="5" formula="Total classification requests/sec" result="~10K/sec" note="Videos + text + images combined. Each is a moderation request." final />
            <MathStep step="6" formula="Peak factor" result="3x" note="Events, viral content, coordinated campaigns spike uploads." />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Model Inference Budget</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Video: sample 1 frame/sec, 7 min avg" result="~420 frames/video" note="Image classifier runs on each sampled frame." />
            <MathStep step="2" formula="Image classifications/sec = 4300 × 420 / 60" result="~30K/sec" note="Frame-level image classification throughput." final />
            <MathStep step="3" formula="Audio: transcribe full audio track" result="~4300 ASR jobs/min" note="Speech-to-text, then text classification on transcript." />
            <MathStep step="4" formula="Text classifier (comments + transcripts)" result="~10K/sec" note="BERT-based text classifier on all text content." />
            <MathStep step="5" formula="Per-video total models" result="~8 classifiers" note="Nudity, violence, hate, spam, CSAM, misinfo, harassment, age-appropriateness." />
            <MathStep step="6" formula="Total model inferences/sec" result="~250K/sec" note="30K frames × 8 classifiers + text + audio." final />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Human Review Capacity</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Content needing human review (~3%)" result="~300/min" note="ML uncertain zone: not confident enough to auto-allow or auto-remove." />
            <MathStep step="2" formula="Avg review time per item" result="~3 min" note="Reviewer watches video segment, reads policy, makes decision." />
            <MathStep step="3" formula="Reviewer throughput = 60/3" result="~20 items/hr/reviewer" note="Per reviewer, accounting for breaks and context switching." />
            <MathStep step="4" formula="Reviewers needed = 300 × 60 / 20" result="~900 concurrent" note="To clear the queue in real-time. 24/7 shifts needed." final />
            <MathStep step="5" formula="With 3-shift coverage + buffer" result="~3,000 total reviewers" note="YouTube reportedly has 10K+ content reviewers globally." />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Storage & Hashing</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Perceptual hash database (CSAM, known-bad)" result="~500M hashes" note="PhotoDNA / CSAI Match hash database. Must check every upload." />
            <MathStep step="2" formula="Hash lookup latency requirement" result="<100ms" note="Bloom filter + exact match. Must be in-memory." />
            <MathStep step="3" formula="Hash DB size = 500M × 64B" result="~32 GB" note="Fits in memory. Replicated across all scanning nodes." final />
            <MathStep step="4" formula="Moderation decision log retention" result="7 years" note="Regulatory requirement. Every decision must be auditable." />
            <MathStep step="5" formula="Decision log volume = 10K/sec × 86400" result="~860M records/day" note="Stored in append-only audit log (BigQuery/Spanner)." />
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
          <Label color="#2563eb">Content Moderation API</Label>
          <CodeBlock code={`# Internal API — called on every upload/post
# RPC: ModerationService.ReviewContent(ContentReview)
#
# ContentReviewRequest:
{
  "content_id": "vid_abc123",
  "content_type": "video",        // video | image | text | comment
  "author_id": "user_xyz",
  "upload_timestamp": "2024-02-10T14:30:00Z",
  "content_signals": {
    "video_url": "gs://uploads/vid_abc123.mp4",
    "thumbnail_url": "gs://uploads/thumb_abc123.jpg",
    "title": "How to make homemade fireworks",
    "description": "Fun family activity...",
    "tags": ["DIY", "fireworks", "fun"],
    "language_hint": "en",
    "duration_sec": 420,
  },
  "author_signals": {
    "account_age_days": 15,
    "prior_strikes": 1,
    "subscriber_count": 230,
    "trust_score": 0.4,            // ML-scored author trust
  },
  "review_priority": "standard",   // critical | high | standard
}

# ContentReviewResponse:
{
  "content_id": "vid_abc123",
  "decision": "human_review",      // allow | remove | human_review
  "policy_scores": {
    "violence": {"score": 0.72, "confidence": 0.85},
    "hate_speech": {"score": 0.12, "confidence": 0.91},
    "nudity": {"score": 0.05, "confidence": 0.97},
    "csam": {"score": 0.001, "confidence": 0.99},
    "spam": {"score": 0.08, "confidence": 0.93},
    "dangerous_acts": {"score": 0.81, "confidence": 0.78},
  },
  "triggered_policies": ["violence", "dangerous_acts"],
  "enforcement_action": "hold_for_review",
  "review_queue": "dangerous_acts_high_priority",
  "latency_ms": 3200
}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Design Decisions (L6 Depth)</Label>
          <div className="space-y-3">
            {[
              { q: "Why return per-policy scores, not just a binary?", a: "Different policies have different enforcement actions. A video with nudity gets age-gated. A video with CSAM gets immediately removed + reported to NCMEC. A video with borderline hate speech gets demonetized but stays up. Per-policy scores enable granular enforcement." },
              { q: "Why include author_signals (trust_score)?", a: "Context matters. A new account with prior strikes uploading content titled 'How to make fireworks' is higher risk than a verified educational channel. Author trust shifts the decision threshold — lower trust = more aggressive enforcement. This catches repeat offenders faster." },
              { q: "Why separate confidence from score?", a: "Score = how violating the content is. Confidence = how sure the model is. A score of 0.72 with confidence 0.85 means 'probably violent, fairly sure'. Score 0.72 with confidence 0.50 means 'maybe violent, very uncertain'. Low confidence items go to human review regardless of score." },
              { q: "Why hold_for_review instead of publish-then-review?", a: "For high-severity categories (violence, CSAM, dangerous acts), content should NOT be published before review. Pre-publish hold prevents any exposure. For low-severity categories (spam, mild policy), publish and review async is acceptable — the harm from brief exposure is low." },
              { q: "Why is latency 3200ms (not <10ms like ads)?", a: "Moderation runs on upload, not on view. Users expect upload processing to take a few seconds (encoding, thumbnail generation). 3s for moderation is hidden within the upload pipeline. Video analysis (sampling frames, running ASR) is inherently slower than text classification." },
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
        <Label color="#9333ea">Full System Architecture — Multi-Stage Moderation Pipeline</Label>
        <svg viewBox="0 0 720 370" className="w-full">
          <defs><marker id="ah-hld" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>

          {/* Upload */}
          <rect x={10} y={40} width={65} height={45} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={42} y={56} textAnchor="middle" fill="#ea580c" fontSize="9" fontWeight="600" fontFamily="monospace">Upload</text>
          <text x={42} y={69} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">text/img</text>
          <text x={42} y={79} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">video</text>

          {/* Hash Check */}
          <rect x={95} y={40} width={80} height={45} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={135} y={57} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Hash Check</text>
          <text x={135} y={70} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">PhotoDNA</text>
          <text x={135} y={80} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">CSAI Match</text>

          {/* Content Understanding */}
          <rect x={195} y={30} width={100} height={60} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={245} y={48} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Content</text>
          <text x={245} y={60} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Understanding</text>
          <text x={245} y={74} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">ASR, OCR, frame</text>
          <text x={245} y={84} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">sampling, embeddings</text>

          {/* Policy Classifiers */}
          <rect x={320} y={30} width={100} height={60} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={370} y={48} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Policy</text>
          <text x={370} y={60} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Classifiers</text>
          <text x={370} y={74} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">violence, hate,</text>
          <text x={370} y={84} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">nudity, spam, ...</text>

          {/* Decision Engine */}
          <rect x={445} y={35} width={85} height={50} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={487} y={55} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Decision</text>
          <text x={487} y={67} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Engine</text>
          <text x={487} y={80} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">threshold logic</text>

          {/* Three outputs */}
          <rect x={555} y={10} width={80} height={30} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={595} y={29} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Auto-Allow</text>

          <rect x={555} y={48} width={80} height={30} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={595} y={67} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Human Review</text>

          <rect x={555} y={86} width={80} height={30} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={595} y={105} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Auto-Remove</text>

          {/* Enforcement */}
          <rect x={655} y={48} width={55} height={30} rx={6} fill="#6366f110" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={682} y={67} textAnchor="middle" fill="#6366f1" fontSize="8" fontWeight="600" fontFamily="monospace">Enforce</text>

          {/* Arrows */}
          <line x1={75} y1={62} x2={95} y2={62} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={175} y1={62} x2={195} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={295} y1={60} x2={320} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={420} y1={60} x2={445} y2={60} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={530} y1={50} x2={555} y2={25} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={530} y1={60} x2={555} y2={63} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={530} y1={70} x2={555} y2={101} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={635} y1={63} x2={655} y2={63} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>

          {/* Data stores */}
          <rect x={95} y={140} width={80} height={30} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={135} y={159} textAnchor="middle" fill="#dc2626" fontSize="7" fontWeight="600" fontFamily="monospace">Hash Database</text>

          <rect x={320} y={140} width={80} height={30} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={360} y={159} textAnchor="middle" fill="#c026d3" fontSize="7" fontWeight="600" fontFamily="monospace">Model Registry</text>

          <rect x={445} y={140} width={85} height={30} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={487} y={159} textAnchor="middle" fill="#059669" fontSize="7" fontWeight="600" fontFamily="monospace">Policy Config</text>

          <rect x={555} y={140} width={80} height={30} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={595} y={159} textAnchor="middle" fill="#0891b2" fontSize="7" fontWeight="600" fontFamily="monospace">Audit Log</text>

          {/* Data arrows */}
          <line x1={135} y1={85} x2={135} y2={140} stroke="#dc262640" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>
          <line x1={370} y1={90} x2={360} y2={140} stroke="#c026d340" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>
          <line x1={487} y1={85} x2={487} y2={140} stroke="#05966940" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>
          <line x1={595} y1={116} x2={595} y2={140} stroke="#0891b240" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>

          {/* Legend */}
          <rect x={15} y={195} width={695} height={160} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={25} y={215} fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 1 — Hash Check (&lt;100ms): Compare perceptual hashes against known-bad database (CSAM, terrorist content).</text>
          <text x={25} y={230} fill="#dc2626" fontSize="8" fontFamily="monospace">           Exact match → immediate block + report to authorities. No ML needed — deterministic.</text>
          <text x={25} y={250} fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 2 — Content Understanding (~2s): Extract signals. Video → sample frames, run ASR on audio, OCR on text</text>
          <text x={25} y={265} fill="#d97706" fontSize="8" fontFamily="monospace">           overlays, compute visual/audio embeddings. Output: multi-modal feature vector per content item.</text>
          <text x={25} y={285} fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 3 — Policy Classification (~1s): Run per-policy classifiers on feature vector. Each outputs a</text>
          <text x={25} y={300} fill="#c026d3" fontSize="8" fontFamily="monospace">           violation score [0,1] + confidence. Multi-label: content can violate multiple policies simultaneously.</text>
          <text x={25} y={320} fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Stage 4 — Decision Engine (~10ms): Apply per-policy thresholds. High confidence violations → auto-remove.</text>
          <text x={25} y={335} fill="#059669" fontSize="8" fontFamily="monospace">           Low scores → auto-allow. Uncertain → route to human review with priority and policy context.</text>
          <text x={25} y={350} fill="#78716c" fontSize="8" fontFamily="monospace">           Thresholds are configurable per policy, per country, per content type. No code deploy needed to adjust.</text>
        </svg>
      </Card>
    </div>
  );
}

function TaxonomySection() {
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Policy Taxonomy — Violation Categories & Enforcement</Label>
        <p className="text-[12px] text-stone-500 mb-4">The most critical design decision: each policy category has different precision/recall requirements, different enforcement actions, and different regulatory obligations. An L6 candidate must understand this asymmetry.</p>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[11px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Category</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Recall Target</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Precision Target</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Pre-publish?</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Enforcement</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Why This Tradeoff</th>
            </tr></thead>
            <tbody>
              {[
                { cat: "CSAM", recall: ">99.9%", prec: ">95%", pre: "YES", action: "Remove + report to NCMEC", why: "Legal mandate. False negative = child exploitation. False positive = manual review catches it.", hl: true },
                { cat: "Terrorist Content", recall: ">99%", prec: ">90%", pre: "YES", action: "Remove + report to authorities", why: "EU regulation requires removal within 1 hour. Legal liability." },
                { cat: "Violence/Gore", recall: ">95%", prec: ">90%", pre: "YES", action: "Remove or age-gate", why: "User safety. Some violence is newsworthy — context matters." },
                { cat: "Nudity/Sexual", recall: ">95%", prec: ">92%", pre: "YES", action: "Age-gate or remove", why: "Art vs porn distinction. Cultural variation. Educational exceptions." },
                { cat: "Hate Speech", recall: ">90%", prec: ">85%", pre: "Partial", action: "Remove or reduce distribution", why: "Hardest to define. Satire vs genuine hate. Context-dependent." },
                { cat: "Misinformation", recall: ">80%", prec: ">80%", pre: "No", action: "Label + reduce distribution", why: "Truth is debatable. Prefer labeling over removal. Expert panels." },
                { cat: "Spam", recall: ">95%", prec: ">99%", pre: "Yes", action: "Remove + rate limit account", why: "High volume, low harm per item. High precision to avoid catching legitimate." },
                { cat: "Harassment", recall: ">85%", prec: ">85%", pre: "No", action: "Warn, escalate on repeat", why: "Context-dependent (friends joking vs targeted harassment)." },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-red-50" : i%2 ? "bg-stone-50/50" : ""}>
                  <td className="px-3 py-2 font-bold text-stone-700">{r.cat}</td>
                  <td className="text-center px-3 py-2"><Pill bg="#fef2f2" color="#dc2626">{r.recall}</Pill></td>
                  <td className="text-center px-3 py-2"><Pill bg="#f0fdf4" color="#059669">{r.prec}</Pill></td>
                  <td className="text-center px-3 py-2 font-bold" style={{ color: r.pre === "YES" ? "#dc2626" : r.pre === "No" ? "#059669" : "#d97706" }}>{r.pre}</td>
                  <td className="px-3 py-2 text-stone-600 text-[10px]">{r.action}</td>
                  <td className="px-3 py-2 text-stone-400 text-[10px]">{r.why}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <Card>
        <Label color="#dc2626">Decision Threshold Design (Critical L6 Concept)</Label>
        <CodeBlock code={`# Decision Engine — per-policy, configurable thresholds
# This is where precision/recall tradeoffs are operationalized

POLICY_THRESHOLDS = {
    "csam": {
        "auto_remove": 0.70,    # LOW threshold — remove aggressively
        "human_review": 0.30,    # Review anything remotely suspicious
        "auto_allow": 0.30,      # Very little auto-allowed
        # Result: high recall (catch everything), moderate precision
        #         (humans review false positives)
    },
    "hate_speech": {
        "auto_remove": 0.95,    # HIGH threshold — only remove when very sure
        "human_review": 0.50,    # Large uncertain zone → lots of human review
        "auto_allow": 0.50,
        # Result: moderate recall, high precision for auto-removal
        #         uncertain cases get human judgment
    },
    "spam": {
        "auto_remove": 0.85,    # Moderate threshold
        "human_review": 0.70,    # Narrow review zone — most is clear
        "auto_allow": 0.70,
        # Result: high precision (don't remove legitimate content)
        #         some spam slips through — acceptable (low harm per item)
    },
}

def make_decision(content_id, policy_scores, author_trust):
    decisions = []
    for policy, score_info in policy_scores.items():
        thresholds = POLICY_THRESHOLDS[policy]

        # Author trust shifts thresholds (lower trust = more aggressive)
        trust_adjusted_remove = thresholds["auto_remove"] - (1 - author_trust) * 0.1
        trust_adjusted_review = thresholds["human_review"] - (1 - author_trust) * 0.1

        if score_info["score"] >= trust_adjusted_remove and score_info["confidence"] > 0.8:
            decisions.append(("auto_remove", policy))
        elif score_info["score"] >= trust_adjusted_review or score_info["confidence"] < 0.6:
            decisions.append(("human_review", policy))
        else:
            decisions.append(("auto_allow", policy))

    # Strictest decision wins (remove > review > allow)
    return max(decisions, key=lambda d: {"auto_remove":2, "human_review":1, "auto_allow":0}[d[0]])`} />
      </Card>
    </div>
  );
}

function ModelsSection() {
  const [sel, setSel] = useState("text");
  const models = {
    text: { name: "Text Classifier", cx: "NLP-based",
      desc: "BERT-based multi-label classifier for hate speech, harassment, spam, and toxicity in text. Must handle 100+ languages, slang, coded language, and adversarial misspellings.",
      code: `# Text Policy Classifier — Multi-label BERT
class TextModerationModel:
    def __init__(self):
        self.encoder = MultilingualBERT("bert-base-multilingual")
        # Multi-label heads — each policy is independent
        self.heads = {
            "hate_speech": nn.Linear(768, 1),
            "harassment": nn.Linear(768, 1),
            "spam": nn.Linear(768, 1),
            "self_harm": nn.Linear(768, 1),
            "dangerous_speech": nn.Linear(768, 1),
        }

    def forward(self, text):
        # Encode text with BERT
        embedding = self.encoder(text)  # [CLS] token, 768-dim

        # Score each policy independently (multi-label)
        scores = {}
        for policy, head in self.heads.items():
            logit = head(embedding)
            scores[policy] = sigmoid(logit)
        return scores

    # Training: binary cross-entropy per head
    # Data: human-labeled examples per policy
    # Key challenge: multi-language support
    #   Option A: one multilingual model (simpler, worse on rare languages)
    #   Option B: per-language models (better accuracy, 100x ops cost)
    #   YouTube uses: multilingual base + language-specific fine-tuning

# Adversarial robustness:
#   "I h@te all [group]" → normalize text before classification
#   Character substitution, Unicode tricks, leetspeak
#   Augment training data with adversarial examples
#   Use character-level features alongside word-level` },
    image: { name: "Image Classifier", cx: "Vision-based",
      desc: "CNN/ViT-based classifier for nudity, violence, gore, and disturbing imagery. Must handle edited images, memes with text overlay, and AI-generated content.",
      code: `# Image Policy Classifier — Multi-label Vision Transformer
class ImageModerationModel:
    def __init__(self):
        self.backbone = ViT_Large("google/vit-large-patch16")
        self.heads = {
            "nudity": nn.Linear(1024, 3),  # none, partial, explicit
            "violence": nn.Linear(1024, 3),  # none, mild, graphic
            "gore": nn.Linear(1024, 1),
            "csam_visual": nn.Linear(1024, 1),
            "disturbing": nn.Linear(1024, 1),
        }

    def forward(self, image):
        features = self.backbone(image)  # 1024-dim
        scores = {}
        for policy, head in self.heads.items():
            scores[policy] = softmax(head(features)) if head.out_features > 1 else sigmoid(head(features))
        return scores

# Video processing: sample frames strategically
def sample_video_frames(video_path, strategy="adaptive"):
    if strategy == "uniform":
        # 1 frame per second — simple, baseline
        return extract_frames(video_path, fps=1)
    elif strategy == "adaptive":
        # Scene change detection + uniform baseline
        scenes = detect_scene_changes(video_path)
        frames = []
        for scene in scenes:
            frames.append(scene.first_frame)    # scene start
            frames.append(scene.middle_frame)    # scene content
        # Also sample uniformly at 0.5 fps as baseline
        frames += extract_frames(video_path, fps=0.5)
        return deduplicate(frames)
    elif strategy == "thumbnail_first":
        # Thumbnail + first 30s (most violations are early)
        # Then sparse sampling for rest
        frames = extract_frames(video_path[:30], fps=2)
        frames += extract_frames(video_path[30:], fps=0.2)
        return frames` },
    audio: { name: "Audio Classifier", cx: "ASR + NLP",
      desc: "Processes audio track: speech-to-text transcription → text classification. Also direct audio classifiers for non-speech signals (gunshots, screams, music with explicit lyrics).",
      code: `# Audio Moderation Pipeline
class AudioModerationPipeline:
    def __init__(self):
        self.asr = WhisperLarge()             # speech-to-text
        self.text_classifier = TextModerationModel()  # reuse text model
        self.audio_event_classifier = AudioEventNet()  # non-speech sounds

    def process(self, audio_track):
        results = {}

        # 1. Speech-to-text transcription
        transcript = self.asr.transcribe(audio_track)
        # Output: timestamped segments
        # [{"start": 0.0, "end": 5.2, "text": "...", "lang": "en"}, ...]

        # 2. Run text classifier on transcript
        for segment in transcript.segments:
            text_scores = self.text_classifier(segment.text)
            results[segment.timestamp] = {
                "text_scores": text_scores,
                "language": segment.language,
            }

        # 3. Audio event detection (non-speech)
        events = self.audio_event_classifier(audio_track)
        # Detects: gunshots, explosions, screams, explicit music
        results["audio_events"] = events

        # 4. Aggregate: max score per policy across all segments
        policy_scores = aggregate_max_per_policy(results)
        return policy_scores

    # KEY CHALLENGES:
    # - ASR errors compound: "I love cooking" transcribed as
    #   "I love killing" → false positive
    # - Code-switching (mixing languages mid-sentence)
    # - Background noise / music interference
    # - Multiple speakers with different intents
    # - ASR latency: ~0.3x real-time for Whisper Large
    #   7-min video = ~2 seconds for transcription` },
    fusion: { name: "Multi-Modal Fusion ★", cx: "Combined signal",
      desc: "Combines text, image, audio, and metadata signals into a unified violation score. Critical because single-modal analysis misses cross-modal context (e.g., benign image + hateful text overlay).",
      code: `# Multi-Modal Fusion for Content Moderation
class MultiModalFusion:
    def __init__(self):
        # Per-modality encoders (pre-trained, frozen or fine-tuned)
        self.text_enc = TextEncoder()    # 768-dim
        self.image_enc = ImageEncoder()  # 1024-dim
        self.audio_enc = AudioEncoder()  # 512-dim

        # Cross-attention fusion
        self.cross_attn = CrossModalAttention(
            dims={"text": 768, "image": 1024, "audio": 512},
            output_dim=512
        )

        # Policy heads on fused representation
        self.policy_heads = {
            policy: nn.Linear(512, 1)
            for policy in POLICY_LIST
        }

    def forward(self, text, images, audio, metadata):
        # Encode each modality
        t_emb = self.text_enc(text)         # title + description
        i_emb = self.image_enc(images)      # thumbnail + key frames
        a_emb = self.audio_enc(audio)       # audio features

        # Cross-modal attention
        # "Does the text change the meaning of the image?"
        # Example: benign photo + text "this is how we treat [group]"
        #          = hate speech (neither modality alone is violating)
        fused = self.cross_attn(t_emb, i_emb, a_emb)

        # Append metadata features
        meta_features = encode_metadata(metadata)
        # author_trust, account_age, prior_strikes, geo
        combined = concat([fused, meta_features])

        # Score each policy
        scores = {
            policy: sigmoid(head(combined))
            for policy, head in self.policy_heads.items()
        }
        return scores

# WHY FUSION MATTERS:
# Image-only: puppy photo → safe
# Text-only: "this is how they should all be treated" → ambiguous
# Image + text overlay: puppy being harmed + that text → violent
# ONLY cross-modal attention catches this` },
  };
  const m = models[sel];
  return (
    <div className="space-y-5">
      <Card accent="#dc2626">
        <Label color="#dc2626">Per-Modality Classifier Comparison</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[12px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Model</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Input</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Architecture</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Latency</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Strength</th>
            </tr></thead>
            <tbody>
              {[
                { n:"Text Classifier", inp:"Title, desc, comments", arch:"Multilingual BERT", l:"~20ms", s:"Language understanding", hl:false },
                { n:"Image Classifier", inp:"Frames, thumbnails", arch:"ViT-Large", l:"~30ms/frame", s:"Visual policy violations", hl:false },
                { n:"Audio Classifier", inp:"Audio track", arch:"Whisper + AudioNet", l:"~2s/7min video", s:"Spoken content, sounds", hl:false },
                { n:"Multi-Modal Fusion ★", inp:"All signals combined", arch:"Cross-attention NN", l:"~50ms (post-encoding)", s:"Cross-modal context", hl:true },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-fuchsia-50" : ""}>
                  <td className={`px-3 py-2 font-mono ${r.hl?"text-fuchsia-700 font-bold":"text-stone-600"}`}>{r.n}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.inp}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.arch}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.l}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.s}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-3 space-y-2">
          {Object.entries(models).map(([k,v]) => (
            <button key={k} onClick={() => setSel(k)}
              className={`w-full text-left px-3.5 py-2.5 rounded-lg text-[12px] font-medium border transition-all ${k===sel?"bg-red-600 text-white border-red-600":"bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
              {v.name}
            </button>
          ))}
        </div>
        <div className="col-span-9 space-y-5">
          <Card>
            <div className="flex items-center gap-3 mb-2">
              <span className="text-[14px] font-bold text-stone-800">{m.name}</span>
              <Pill bg="#fdf4ff" color="#c026d3">{m.cx}</Pill>
            </div>
            <p className="text-[12px] text-stone-500">{m.desc}</p>
          </Card>
          <CodeBlock title={`${m.name} — Implementation`} code={m.code} />
        </div>
      </div>
    </div>
  );
}

function MultiModalSection() {
  return (
    <div className="space-y-5">
      <Card accent="#d97706">
        <Label color="#d97706">Multi-Modal Processing Pipeline — Video Moderation</Label>
        <p className="text-[12px] text-stone-500 mb-4">Video moderation is the hardest multi-modal problem. A 7-minute video has ~420 frames, an audio track, text overlays, a title, description, and metadata. The pipeline must process all of these and fuse signals efficiently.</p>
        <CodeBlock title="Video Moderation Pipeline — End to End" code={`# Full video moderation pipeline — processes one uploaded video
async def moderate_video(video: UploadedVideo) -> ModerationResult:
    # STAGE 0: Hash check (instant block for known-bad content)
    hash_match = await hash_db.check(video.perceptual_hash)
    if hash_match:
        return ModerationResult(
            decision="auto_remove",
            reason=f"hash_match:{hash_match.category}",
            report_to=hash_match.report_authority,
        )

    # STAGE 1: Content extraction (parallel)
    frames_future = extract_frames_adaptive(video, max_frames=500)
    audio_future = extract_audio(video)
    text = concat(video.title, video.description, video.tags)
    thumbnail = video.thumbnail

    frames = await frames_future
    audio = await audio_future

    # STAGE 2: Per-modality classification (parallel)
    text_scores = text_classifier.predict(text)
    thumb_scores = image_classifier.predict(thumbnail)
    frame_scores = image_classifier.predict_batch(frames)
    transcript = asr_model.transcribe(audio)
    transcript_scores = text_classifier.predict(transcript.full_text)
    audio_event_scores = audio_event_classifier.predict(audio)

    # STAGE 3: Aggregation per modality
    # Video: max violation score across all sampled frames
    # (one violating frame = entire video is violating)
    max_frame_scores = aggregate_max(frame_scores)

    # Audio: max across all transcript segments
    max_audio_scores = aggregate_max(transcript_scores)

    # STAGE 4: Multi-modal fusion
    # Cross-modal context: does text change meaning of visuals?
    fused_scores = fusion_model.predict(
        text_emb=text_classifier.encode(text),
        image_emb=image_classifier.encode(thumbnail),
        audio_emb=audio_classifier.encode(audio),
    )

    # STAGE 5: Final score = max(per-modality scores, fused scores)
    # Conservative: if ANY signal detects violation, flag it
    final_scores = {}
    for policy in POLICY_LIST:
        final_scores[policy] = max(
            text_scores.get(policy, 0),
            thumb_scores.get(policy, 0),
            max_frame_scores.get(policy, 0),
            max_audio_scores.get(policy, 0),
            fused_scores.get(policy, 0),
        )

    # STAGE 6: Decision engine
    return decision_engine.decide(
        content_id=video.id,
        policy_scores=final_scores,
        author_trust=video.author.trust_score,
    )`} />
      </Card>
    </div>
  );
}

function FeaturesSection() {
  return (
    <div className="space-y-5">
      <Card accent="#0f766e">
        <Label color="#0f766e">Feature Categories for Content Moderation</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[11px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Category</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Example Features</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Signal Type</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Impact</th>
            </tr></thead>
            <tbody>
              {[
                { cat: "Text Content", ex: "title_embedding, description_tokens, detected_language, profanity_count, slur_presence", sig: "Primary", impact: "Very High" },
                { cat: "Visual Content", ex: "frame_embeddings, skin_pixel_ratio, blood_color_ratio, object_detections, face_count_minor", sig: "Primary", impact: "Very High" },
                { cat: "Audio Content", ex: "transcript_embedding, speech_rate, shouting_detected, gunshot_events, explicit_music", sig: "Primary", impact: "High" },
                { cat: "Author Signals", ex: "account_age, prior_strikes, subscriber_count, trust_score, verified_status, upload_frequency", sig: "Context", impact: "High" },
                { cat: "Metadata", ex: "category_tag, upload_time, geo_origin, video_duration, file_metadata", sig: "Context", impact: "Medium" },
                { cat: "Engagement (post-pub)", ex: "report_count, report_rate, dislike_ratio, comment_toxicity_avg", sig: "Post-hoc", impact: "Very High" },
                { cat: "Network Signals", ex: "co-shared_with_violating_content, coordinated_upload_pattern, ring_membership_score", sig: "Graph", impact: "High" },
                { cat: "Perceptual Hash", ex: "PhotoDNA_hash, video_fingerprint, near-duplicate_match_score", sig: "Deterministic", impact: "Critical (CSAM)" },
              ].map((r,i) => (
                <tr key={i} className={i%2?"bg-stone-50/50":""}>
                  <td className="px-3 py-2 font-bold text-stone-700">{r.cat}</td>
                  <td className="px-3 py-2 text-stone-500 font-mono text-[10px]">{r.ex}</td>
                  <td className="text-center px-3 py-2 text-stone-500">{r.sig}</td>
                  <td className="text-center px-3 py-2"><Pill bg={r.impact.includes("Critical")?"#fef2f2":r.impact.includes("Very")?"#fef2f2":r.impact==="High"?"#fffbeb":"#f0fdf4"} color={r.impact.includes("Critical")?"#dc2626":r.impact.includes("Very")?"#dc2626":r.impact==="High"?"#d97706":"#059669"}>{r.impact}</Pill></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Post-Publish Signals (Engagement-Based)</Label>
          <CodeBlock code={`# Post-publish signals dramatically improve accuracy
# Content that slips through pre-publish can be caught by engagement

class PostPublishMonitor:
    def monitor(self, content_id, hours_since_publish):
        # Signal 1: User reports
        report_rate = reports[content_id] / impressions[content_id]
        if report_rate > 0.01:  # 1% of viewers reported
            trigger_re_review(content_id, priority="high")

        # Signal 2: Comment toxicity
        comments = get_comments(content_id, limit=100)
        avg_toxicity = mean([toxicity_model(c) for c in comments])
        if avg_toxicity > 0.7:
            trigger_re_review(content_id, priority="medium")

        # Signal 3: Sharing patterns
        shares = get_share_graph(content_id)
        if shared_in_known_bad_communities(shares):
            trigger_re_review(content_id, priority="high")

        # Signal 4: Anomalous engagement
        # Hate speech often has unusual like/dislike ratios
        if dislike_ratio(content_id) > 0.3:  # 30%+ dislikes
            trigger_re_review(content_id, priority="medium")

# Post-publish monitoring runs continuously
# Catches: evolving context (video becomes relevant to new event),
#   coordinated campaigns, slow-burn violations,
#   content that was borderline at upload but now has evidence`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Author Trust Score</Label>
          <CodeBlock code={`# Author Trust Score — shifts moderation thresholds
# New accounts with no history get low trust (more scrutiny)
# Established creators with clean records get high trust

def compute_author_trust(author):
    features = {
        "account_age_days": author.account_age,
        "prior_strikes": author.strikes,
        "total_uploads": author.upload_count,
        "violation_rate": author.violations / max(author.uploads, 1),
        "subscriber_count": log(author.subscribers + 1),
        "verified": author.is_verified,
        "phone_verified": author.phone_verified,
        "avg_content_quality": author.avg_quality_score,
    }

    # ML model: trained on (author_features → future_violation)
    trust_score = trust_model.predict(features)  # [0, 1]

    # Trust affects moderation thresholds:
    # trust = 0.1 (new, suspicious) → lower thresholds (more aggressive)
    # trust = 0.9 (established, clean) → higher thresholds (more lenient)
    #
    # Example: hate_speech auto_remove threshold
    #   Base: 0.90
    #   trust=0.1: 0.90 - 0.1*(1-0.1) = 0.81 (catches more)
    #   trust=0.9: 0.90 - 0.1*(1-0.9) = 0.89 (about the same)
    #
    # WHY: New accounts are 10x more likely to violate policies.
    # Treating all content equally wastes human reviewer capacity
    # on reviewing established creators' benign content.

    return trust_score`} />
        </Card>
      </div>
    </div>
  );
}

function DataModelSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Core Data Stores</Label>
          <CodeBlock code={`-- Hash Database (in-memory, replicated)
-- Known-bad content hashes (CSAM, terrorist content)
perceptual_hash -> {
  category: enum(csam, terrorism, known_violating),
  source: "NCMEC" | "GIFCT" | "internal",
  added_date: timestamp,
  report_authority: "NCMEC" | "law_enforcement",
}
# 500M+ hashes. Bloom filter for fast negative check.
# Must be replicated to EVERY scanning node.

-- Content Moderation Record (Spanner, strongly consistent)
content_id -> {
  content_type: enum(video, image, text, comment),
  author_id: string,
  upload_ts: timestamp,
  moderation_status: enum(pending, allowed, removed, appealed),
  policy_scores: {policy: {score, confidence}, ...},
  decision: enum(auto_allow, auto_remove, human_review),
  enforcement_action: enum(none, removed, age_gated, demonetized),
  reviewer_id: string | null,
  review_ts: timestamp | null,
  appeal_status: enum(none, pending, upheld, overturned),
  audit_trail: [{action, actor, timestamp, reason}, ...],
}
# MUST be strongly consistent — legal liability
# Retained 7+ years for regulatory compliance

-- Human Review Queue (Redis + Postgres)
queue_entry -> {
  content_id, priority, policy_category,
  ml_scores, author_context,
  assigned_reviewer: string | null,
  created_ts, sla_deadline,
}

-- Author Profile (Bigtable)
author_id -> {
  trust_score, prior_strikes, account_age,
  violation_history: [(content_id, policy, date), ...],
  appeal_history: [(content_id, outcome, date), ...],
}`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Why These Design Choices?</Label>
          <div className="space-y-3">
            {[
              { q: "Why Spanner for moderation records?", a: "Legal requirement: every moderation decision must be auditable, strongly consistent, and globally available. Spanner provides strong consistency with global distribution. If EU regulators ask 'why did you remove this content?', we need an authoritative answer across all regions." },
              { q: "Why in-memory hash database?", a: "Hash check must be sub-100ms and runs on EVERY upload. Disk-based lookup is too slow. 500M hashes × 64 bytes = ~32 GB. Fits in memory easily. Bloom filter pre-check eliminates 99.99% of lookups (most content is NOT in the hash DB)." },
              { q: "Why Redis for review queue?", a: "Real-time priority ordering with sub-millisecond assignment. Reviewers need instant task assignment. Priority changes dynamically (viral content gets bumped up). Redis sorted sets provide O(log N) insert and O(1) pop-min. Postgres backs up for durability." },
              { q: "Why store ML scores in the moderation record?", a: "Three reasons: (1) Audit trail — 'why was this removed?' requires knowing the ML score at decision time. (2) Calibration analysis — compare predicted scores to human review outcomes. (3) Retroactive re-evaluation — when a model improves, you can rescore old content by comparing new scores to stored old scores." },
              { q: "Why separate author profile from moderation record?", a: "Author profile is updated on every upload and every moderation decision. Moderation records are immutable (append-only audit trail). Different access patterns: profile is read on every new upload for trust scoring, while individual moderation records are rarely read (only for appeals/audits)." },
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
        <Label color="#7e22ce">Training Pipeline — Active Learning + Human-in-the-Loop</Label>
        <p className="text-[12px] text-stone-500 mb-4">Content moderation training is unique because: (1) labeling is done by specialized human reviewers, not crowdsourced, (2) the distribution of violations constantly shifts, and (3) new policies emerge frequently. Active learning is essential to make efficient use of expensive reviewer time.</p>
        <div className="grid grid-cols-2 gap-5">
          <Card accent="#7e22ce">
            <Label color="#7e22ce">Training Data Construction</Label>
            <CodeBlock code={`# Training data for moderation classifiers
# Sources: human-reviewed samples + active learning

def construct_training_data():
    data = []

    # Source 1: Random production sample (calibration)
    # Random 0.1% of all uploads, human-reviewed
    # Purpose: unbiased estimate of true violation rate
    random_sample = sample_production(rate=0.001)
    for item in random_sample:
        labels = get_human_labels(item, reviewers=3)
        data.append((item, majority_vote(labels)))

    # Source 2: Active learning (model improvement)
    # Items where the model is most uncertain
    uncertain = get_items_near_threshold(
        threshold_range=(0.4, 0.6),
        limit=10000
    )
    for item in uncertain:
        labels = get_human_labels(item, reviewers=3)
        data.append((item, majority_vote(labels)))

    # Source 3: False positive/negative mining
    # Appeals that were overturned → false positives
    # Reports on auto-allowed content → false negatives
    fp_items = get_overturned_appeals(limit=5000)
    fn_items = get_reported_auto_allowed(limit=5000)
    for item in fp_items + fn_items:
        data.append((item, item.corrected_label))

    # Source 4: Adversarial examples
    # Content that evaded detection but was later caught
    evasion_attempts = get_evasion_examples(limit=5000)
    for item in evasion_attempts:
        data.append((item, "violation"))

    # Label quality: 3 reviewers per item, majority vote
    # Disagreement items get specialist review (4th reviewer)
    # Inter-annotator agreement target: Cohen's kappa > 0.7

    return data`} />
          </Card>
          <Card accent="#ea580c">
            <Label color="#ea580c">Model Update & Deployment</Label>
            <CodeBlock code={`# Model deployment for moderation is HIGH-STAKES
# A bad model can silence millions or expose users to harm

def deploy_new_model(new_model, current_model):
    # Step 1: Offline evaluation on held-out test set
    metrics = evaluate(new_model, test_set)
    for policy in POLICY_LIST:
        assert metrics[policy].recall >= RECALL_TARGETS[policy]
        assert metrics[policy].precision >= PRECISION_TARGETS[policy]

    # Step 2: Side-by-side scoring (shadow mode)
    # Score production traffic with BOTH models
    # Compare decisions: where do they disagree?
    disagreements = shadow_score(new_model, current_model,
                                 traffic_pct=100, duration="24h")

    # Step 3: Human review of disagreements
    # Sample disagreements, have reviewers judge
    # "Which model's decision is correct?"
    review_results = human_review_sample(disagreements, n=1000)
    new_model_correct_rate = review_results.new_correct / len(review_results)
    if new_model_correct_rate < 0.55:
        abort("New model not clearly better on disagreements")

    # Step 4: Gradual rollout with guardrails
    for pct in [1, 5, 20, 50, 100]:
        deploy(new_model, traffic_pct=pct)
        monitor(duration="6h", metrics=[
            "auto_remove_rate",  # should NOT spike (over-enforcement)
            "appeal_overturn_rate",  # should NOT increase
            "report_rate",  # should NOT increase (under-enforcement)
        ])
        if any_guardrail_violated():
            rollback(current_model)
            alert_oncall("Model rollout guardrail triggered")
            return

    # Step 5: Monitor for 2 weeks post-launch
    # Policy changes lag model changes
    return "deployed"`} />
          </Card>
        </div>
      </Card>
    </div>
  );
}

function HumanLoopSection() {
  return (
    <div className="space-y-5">
      <Card accent="#ea580c">
        <Label color="#ea580c">Human Review System — The Critical Integration</Label>
        <p className="text-[12px] text-stone-500 mb-4">The human review loop is not an afterthought — it's a core architectural component. ML handles 95-97% of content automatically; human reviewers handle the remaining 3-5% that the model is uncertain about. The quality of human review directly determines model improvement rate.</p>
        <div className="grid grid-cols-2 gap-5">
          <div className="space-y-4">
            <Card accent="#d97706">
              <Label color="#d97706">Review Queue Prioritization</Label>
              <div className="space-y-2">
                {[
                  { pri: "P0 — Critical", sla: "<1 hour", examples: "Potential CSAM, imminent violence threats, suicide/self-harm content", color: "#dc2626" },
                  { pri: "P1 — High", sla: "<4 hours", examples: "Hate speech, graphic violence, dangerous activities, terrorism content", color: "#ea580c" },
                  { pri: "P2 — Medium", sla: "<24 hours", examples: "Borderline nudity, mild harassment, clickbait, misleading content", color: "#d97706" },
                  { pri: "P3 — Low", sla: "<72 hours", examples: "Spam, copyright (non-urgent), low-severity policy edge cases", color: "#059669" },
                ].map((p,i) => (
                  <div key={i} className="rounded-lg border p-3" style={{ borderLeft: `3px solid ${p.color}` }}>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[11px] font-bold" style={{ color: p.color }}>{p.pri}</span>
                      <Pill bg="#f3f4f6" color="#374151">SLA: {p.sla}</Pill>
                    </div>
                    <p className="text-[10px] text-stone-500">{p.examples}</p>
                  </div>
                ))}
              </div>
            </Card>
          </div>
          <Card accent="#ea580c">
            <Label color="#ea580c">Review Workflow</Label>
            <CodeBlock code={`# Human reviewer workflow
class ReviewSystem:
    def assign_task(self, reviewer):
        # Match reviewer to task based on:
        # 1. Reviewer's language competence
        # 2. Reviewer's policy specialization
        # 3. Queue priority (P0 first)
        # 4. Reviewer's current wellness (CSAM review has time limits)
        task = review_queue.pop_highest_priority(
            languages=reviewer.languages,
            specializations=reviewer.specializations,
            exclude_categories=reviewer.current_cool_down,
        )
        return task

    def submit_review(self, reviewer, task, decision):
        # Decision: violating | non_violating | borderline
        # Borderline → escalate to specialist or team lead

        # Quality check: 5% of reviews are audited
        if should_audit(reviewer, task):
            second_review = assign_to_auditor(task)

        # Disagreement handling
        if task.has_prior_review:
            if decision != task.prior_decision:
                # Disagreement → escalate to specialist
                escalate_to_specialist(task, [decision, task.prior_decision])

        # Update content status
        update_moderation_record(task.content_id, decision, reviewer.id)

        # Feed back to ML model (training signal)
        add_to_training_queue(task.content_id, decision)

    # REVIEWER WELLNESS:
    # - CSAM reviewers: max 4 hours/day, mandatory counseling
    # - Violence/gore reviewers: regular rotation, breaks
    # - Automated content blurring: reviewers see redacted previews
    #   and can choose to view full content if needed
    # - Wellness monitoring: track review speed, accuracy, patterns`} />
          </Card>
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
          <Label color="#059669">ML Pipeline Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">GPU fleet for image/video classification</strong> — 30K frames/sec × 8 classifiers = 240K inferences/sec. Each GPU handles ~1K inferences/sec. Need ~300 GPUs. With redundancy: ~600 GPUs globally.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Async pipeline architecture</strong> — video moderation is NOT in the serving path. It runs async after upload. Use message queues (Pub/Sub) to decouple upload from moderation. Allows backpressure and retry.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Tiered processing</strong> — fast check first (hash, text classifier: 100ms), then deeper analysis (video frames: 3s). If fast check catches it, skip expensive video analysis. Saves 60% of GPU compute.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Model distillation</strong> — ViT-Large is accurate but expensive. Distill to ViT-Small for initial screening. Only run full model on uncertain items. 3x throughput improvement.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Human Review Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">ML confidence → queue routing</strong> — as ML improves, the uncertain zone shrinks. Moving auto-remove threshold from 0.95 to 0.90 reduces human review volume by ~30%. Constant optimization pressure.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Specialist pools</strong> — CSAM-trained reviewers, hate speech linguists, medical misinformation experts. Route tasks to domain experts. Higher accuracy per review, fewer escalations.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Reviewer tooling</strong> — pre-highlighting suspected violation segments in video. ML provides "here's the 3 seconds you should watch." Reduces review time from 5 min to 2 min per item.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Follow-the-sun operations</strong> — review centers in US, Ireland, India, Philippines, Singapore. 24/7 coverage. Language coverage matches regional needs. Peak traffic handled by nearest region.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Retroactive Moderation (Policy Change Handling)</Label>
        <CodeBlock code={`# When a policy changes, existing content must be re-evaluated
# Example: "AI-generated deepfakes" wasn't a policy in 2020, now it is

class RetroactiveModerationPipeline:
    def handle_policy_change(self, new_policy, scope):
        # Step 1: Estimate impact
        # Run new classifier on random sample to estimate volume
        sample_results = score_sample(new_policy.classifier, n=100_000)
        estimated_violations = sample_results.violation_rate * total_content
        print(f"Estimated {estimated_violations} items to review")

        # Step 2: Prioritize by reach
        # High-view content first (most harm from leaving up)
        content_by_views = get_all_content_sorted_by_views()

        # Step 3: Batch re-scoring
        # Process in priority order, batch of 10K
        for batch in content_by_views.batches(size=10_000):
            scores = new_policy.classifier.predict_batch(batch)
            for content, score in zip(batch, scores):
                if score > new_policy.auto_remove_threshold:
                    take_action(content, "remove")
                elif score > new_policy.review_threshold:
                    add_to_review_queue(content, priority="P2")

        # Step 4: Progress tracking
        # Dashboard: % of content re-scanned, violations found, actions taken
        # SLA: re-scan all content with >10K views within 7 days
        # SLA: re-scan all content within 30 days`} />
      </Card>
    </div>
  );
}

function WatchoutsSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        {[
          { title: "Adversarial Evasion", sev: "CRITICAL", desc: "Bad actors deliberately modify content to bypass classifiers: Unicode substitution (replacing letters with similar-looking characters), steganography in images, code words ('pizza' for illegal content), audio pitch shifting, frame-level perturbations.", fix: "Text normalization preprocessing. Adversarial training with augmented evasion examples. Character-level + word-level features. Perceptual hashing (robust to small changes). Behavioral signals (what communities share this content?) as auxiliary signal.", icon: "🔴" },
          { title: "Over-Enforcement (False Positives)", sev: "CRITICAL", desc: "Legitimate content wrongly removed: news reporting on violence, educational content about hate speech, parody/satire, medical content flagged as nudity, war reporting. This is censorship and damages platform trust.", fix: "High precision thresholds for auto-removal. Generous uncertain zone → human review. Context-aware models (news vs user-generated). Allowlisting verified news organizations. Robust appeals process with fast turnaround. Published transparency reports.", icon: "🔴" },
          { title: "Cultural & Linguistic Bias", sev: "HIGH", desc: "Models trained predominantly on English data perform poorly on other languages. Hate speech definitions vary by culture. A gesture offensive in one culture is benign in another. Underserved languages get worse moderation.", fix: "Per-language evaluation metrics. Multilingual training data collection. Regional policy experts involved in label guidelines. Language-specific fine-tuning. Monitor enforcement rates per language — large disparities indicate bias.", icon: "🟡" },
          { title: "Coordinated Campaigns", sev: "HIGH", desc: "Organized groups flood the platform with violating content simultaneously. Goal: overwhelm both ML and human review capacity. Or: coordinate false reports to get legitimate content removed (weaponized reporting).", fix: "Network analysis: detect coordinated upload patterns (same time, same accounts, same content variations). Rate limiting per account/IP during spikes. Graph-based detection of inauthentic behavior rings. Separate queue for coordinated attack content.", icon: "🟡" },
          { title: "Reviewer Burnout & Accuracy Drift", sev: "MEDIUM", desc: "Human reviewers exposed to disturbing content suffer mental health impacts. Fatigued reviewers make more errors. Accuracy drifts over long shifts. Inconsistent labeling degrades model training data quality.", fix: "Mandatory wellness programs and counseling. Shift limits for high-severity content (4 hrs/day for CSAM). Content blurring/redaction by default. Regular calibration exercises. Accuracy monitoring per reviewer with retraining triggers.", icon: "🟠" },
          { title: "Model Staleness on Emerging Threats", sev: "MEDIUM", desc: "New forms of harmful content emerge that models haven't been trained on: AI-generated deepfakes, new slang/coded language, novel scam formats, emerging conspiracy theories.", fix: "Active learning pipeline: route novel/uncertain content to reviewers. Rapid model fine-tuning capability (deploy updated model within hours). Human-authored rules as a fast-response bridge while ML catches up. Threat intelligence team monitoring emerging trends.", icon: "🟠" },
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
          <Label color="#0284c7">Safety Metrics</Label>
          <div className="space-y-2.5">
            {[
              { metric: "Violating View Rate (VVR)", target: "<0.15%", why: "% of total views on content later determined to be violating. Primary safety metric." },
              { metric: "Violative content removal rate", target: ">95%", why: "% of violating content caught (by ML + human). Higher = safer platform." },
              { metric: "Time to action (pre-publish)", target: "<5s text, <60s video", why: "How fast violating content is blocked before anyone sees it." },
              { metric: "CSAM detection recall", target: ">99.9%", why: "Near-zero tolerance. Legal and moral imperative." },
              { metric: "Proactive detection rate", target: ">90%", why: "% of violations caught by ML before any user report. Higher = less user exposure." },
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
        <Card accent="#059669">
          <Label color="#059669">Fairness Metrics</Label>
          <div className="space-y-2.5">
            {[
              { metric: "Appeal overturn rate", target: "<5%", why: "% of removals overturned on appeal. High = too many false positives." },
              { metric: "Enforcement rate parity by language", target: "<2x variance", why: "English vs Hindi enforcement rates shouldn't differ by more than 2x." },
              { metric: "False positive rate by content type", target: "<3%", why: "News, education, satire shouldn't be disproportionately removed." },
              { metric: "Time to appeal resolution", target: "<48 hrs", why: "Users whose content is wrongly removed deserve fast review." },
              { metric: "Auto-remove accuracy (precision)", target: ">95%", why: "Content auto-removed without human review must be truly violating." },
            ].map((m,i) => (
              <div key={i} className="border-b border-stone-100 pb-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold text-stone-700">{m.metric}</span>
                  <span className="text-[10px] font-mono text-emerald-700">{m.target}</span>
                </div>
                <div className="text-[10px] text-stone-400 mt-0.5">{m.why}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">System Health & Alerts</Label>
          <div className="space-y-2.5">
            {[
              { alert: "Moderation pipeline backlog > 10 min", sev: "P0", action: "Scale pipeline, check for outage" },
              { alert: "Auto-remove rate spikes > 2x", sev: "P0", action: "Model may be misfiring — review, possible rollback" },
              { alert: "P0 review queue SLA breach", sev: "P0", action: "Escalate, pull reviewers from P2/P3 tasks" },
              { alert: "Appeal overturn rate > 10%", sev: "P1", action: "Model drift — recalibrate thresholds" },
              { alert: "Hash DB sync lag > 1 hour", sev: "P1", action: "Check NCMEC/GIFCT feed pipeline" },
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

function EnhancementsSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        {[
          { title: "LLM-Powered Context Understanding", d: "Use large language models to understand nuance: sarcasm, satire, quoting vs endorsing hate speech, newsworthy violence vs gratuitous violence. LLMs can reason about intent and context far better than classifiers.", effort: "Hard", detail: "Challenge: LLM inference is expensive (~100ms per item). Use as a second-stage classifier for uncertain items only. Also useful for generating explanations for moderation decisions." },
          { title: "Provenance & Deepfake Detection", d: "Detect AI-generated or manipulated content: deepfake videos, synthetic voices, AI-generated images misrepresented as real. Track content provenance using C2PA metadata standards.", effort: "Hard", detail: "Arms race: generative AI quality improves faster than detectors. Watermarking at generation time (Google SynthID) is more reliable than post-hoc detection." },
          { title: "User-Level Risk Scoring", d: "Aggregate user behavior across content to identify high-risk accounts proactively. Network analysis to detect organized bad-actor rings before they upload violating content.", effort: "Medium", detail: "Graph neural networks on the social/upload graph. Features: account age, upload patterns, association with known bad actors, community membership." },
          { title: "Contextual Moderation", d: "The same content may be violating in one context and allowed in another. A medical image is fine in a health community, violating in a general feed. Context-aware models condition on where content is shared.", effort: "Medium", detail: "Requires modeling the community/forum context. Content + community embedding space. Different threshold sets per community type." },
          { title: "Explainable Decisions", d: "Generate human-readable explanations for every moderation decision: 'This video was age-restricted because frames at 2:34-2:41 contain graphic violence.' Helps creators understand and helps with appeals.", effort: "Medium", detail: "Attention visualization for which frames/words triggered the decision. LLM-generated natural language explanations. Critical for regulatory compliance (DSA Article 17)." },
          { title: "Cross-Platform Intelligence Sharing", d: "Platforms share hashes and signals about violating content through consortiums like GIFCT (Global Internet Forum to Counter Terrorism). Content removed on YouTube shouldn't resurface on Twitter.", effort: "Hard", detail: "Privacy constraints: can't share user data. Share content hashes and policy signals only. Standardization challenges across platforms with different policies." },
        ].map((e,i) => (
          <Card key={i}>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="text-[12px] font-bold text-stone-800">{e.title}</span>
              <Pill bg={e.effort==="Medium"?"#fffbeb":"#fef2f2"} color={e.effort==="Medium"?"#d97706":"#dc2626"}>{e.effort}</Pill>
            </div>
            <p className="text-[12px] text-stone-600 mb-1.5">{e.d}</p>
            <p className="text-[11px] text-stone-400">{e.detail}</p>
          </Card>
        ))}
      </div>
    </div>
  );
}

function FollowupsSection() {
  const [exp, setExp] = useState(null);
  const qas = [
    { q:"How do you handle satire and parody that looks like hate speech?", a:"This is the hardest problem in content moderation. Approaches: (1) Context features — is the author a known satirist? Is the content posted in a comedy/satire community? (2) Audience signals — satire typically gets likes and laughing emojis, not angry reactions and reports. (3) LLM reasoning — prompt an LLM with the content and ask 'is this satirical or genuine hate speech?' with examples. (4) Lower confidence for satire-like content → route to specialist human reviewers who are trained on the satire/hate speech boundary. (5) Accept that this is a problem without a perfect solution — there will always be edge cases. The goal is to minimize harm while minimizing censorship of legitimate expression.", tags:["policy"] },
    { q:"What happens when your model makes a high-profile mistake?", a:"Example: auto-removing a major news organization's video of a war zone. Process: (1) Incident response: detect quickly via spikes in appeals or PR escalation. (2) Immediately restore content if wrongly removed. (3) Root cause: was it a model error, policy ambiguity, or reviewer error? (4) If model error: add the case to the training set as a hard negative. Adjust thresholds for the specific content type. (5) If policy ambiguity: clarify the policy with the Trust & Safety team. Update reviewer guidelines. (6) Communicate: transparency report, public acknowledgment. (7) Allowlisting: consider verified news organizations getting higher trust scores to prevent recurrence. The key insight: high-profile mistakes have outsized reputational damage, so fast detection and response matter more than preventing 100% of them.", tags:["ops"] },
    { q:"How do you handle content in languages you don't have training data for?", a:"Low-resource language problem. Layered approach: (1) Multilingual models (mBERT, XLM-R) provide baseline coverage — they transfer some knowledge from high-resource to low-resource languages. (2) Image/video classifiers are language-independent — visual violations (nudity, violence) work across all languages. (3) Translation + classification: translate to English, then classify. Noisy but catches many violations. (4) Community reporting: rely more heavily on user reports in low-resource languages. Lower auto-remove thresholds, higher human review rates. (5) Targeted data collection: partner with local organizations to build labeled datasets for critical languages. (6) Monitor enforcement rates per language — if a language has significantly lower violation detection, it indicates a model gap.", tags:["ml"] },
    { q:"How does the system handle a sudden policy change?", a:"Example: a new policy banning AI-generated deepfakes. Timeline: (1) Day 0: Policy team defines the new policy with detailed guidelines and examples. (2) Day 1-3: Create initial training dataset from curated examples. Rule-based detector as a bridge (metadata signals, known deepfake generators). (3) Day 3-7: Train initial classifier on small labeled set. Deploy in shadow mode alongside human review. (4) Week 2-4: Active learning loop — model routes uncertain cases to reviewers, labels feed back into training. Model improves rapidly. (5) Month 2: Model is accurate enough for auto-enforcement. Adjust thresholds. (6) Ongoing: retroactive scan of existing content using batch pipeline. The key: rules-based systems provide immediate coverage while ML ramps up. This hybrid approach means there's never a gap in enforcement.", tags:["ops"] },
    { q:"How do you prevent the system from being weaponized (mass false reporting)?", a:"Coordinated false reporting is when groups mass-report legitimate content to trigger automated removal. Defenses: (1) Reporter trust scores — accounts that frequently make false reports have their reports downweighted. Reports from trusted reporters (verified, high accuracy history) are prioritized. (2) Report velocity detection — if 1000 reports come in on the same content within 5 minutes, flag as potentially coordinated. Don't auto-act, route to specialist review. (3) Content-based assessment over report-based — the ML classifier's score matters more than the number of reports. If the classifier says 'safe' with high confidence, many reports don't override it. (4) Network analysis — detect if reporters are part of a coordinated ring (same IP ranges, created at similar times, report similar content). (5) Rate limiting — cap reports per account per hour.", tags:["adversarial"] },
    { q:"Why not use a single large multi-modal model instead of separate classifiers?", a:"Practical and technical reasons: (1) Different policies have different update frequencies. Hate speech evolves with language trends (weekly). Nudity classifiers are stable (monthly). A monolithic model couples all update cycles. (2) Different precision/recall requirements. CSAM needs 99.9% recall; spam needs 99% precision. A single model can't optimize both. (3) Debugging — when a specific policy has accuracy issues, you can retrain just that classifier. A monolithic model is a black box. (4) Resource allocation — some classifiers need GPU (image), others CPU-only (text). Separate models allow heterogeneous deployment. (5) That said, the fusion model IS a cross-modal model — but it sits on top of per-modality classifiers, combining their outputs. You get modularity AND cross-modal reasoning.", tags:["architecture"] },
    { q:"How do you measure the true false negative rate?", a:"You can't directly measure what you don't catch. Approaches: (1) Random sampling: take a random 0.1% of ALL content (including auto-allowed), have humans review it. Calculate: violations found in auto-allowed / total violations. This is your estimated false negative rate. Expensive but necessary — it's the only unbiased estimate. (2) Delayed signal analysis: content auto-allowed at upload but later reported by users or found via post-publish monitoring. This gives a lower bound on false negatives. (3) Red team exercises: internal teams create violating content and test if the system catches it. Measures recall for known violation types but doesn't discover unknown failure modes. (4) Cross-platform signals: content removed on other platforms that's still live on yours. Requires intelligence sharing agreements.", tags:["evaluation"] },
    { q:"How do you handle the reviewer wellbeing problem at scale?", a:"Content moderation has documented mental health impacts on reviewers, including PTSD. Comprehensive program: (1) Exposure limits: max 4 hours/day reviewing CSAM, 6 hours for graphic violence. Rotate to lower-severity content for the rest of the shift. (2) Content blurring: ML pre-blurs suspected graphic content. Reviewers see a blurred version first and can choose to view full content if needed for judgment. Reduces unnecessary exposure. (3) Mandatory wellness support: on-site counselors, regular check-ins, opt-out for severe categories. (4) AI-assisted review: ML highlights the specific 3-second segment that's violating. Reviewer doesn't need to watch the entire 10-minute video. (5) Gradual onboarding: new reviewers start with low-severity categories and gradually move to more severe. (6) Exit support: mental health resources available after leaving the role. This is a moral obligation, not just an operational concern.", tags:["ops"] },
    { q:"How does the system handle edge cases like war reporting or medical content?", a:"Context-dependent exceptions require special handling: (1) Newsworthy exceptions: content that would normally violate (graphic violence) may be allowed for news reporting. Signal: is the uploader a verified news organization? Is the content tagged as 'news'? Is it being shared in news-related contexts? (2) Educational exceptions: medical imagery (nudity, surgical procedures) allowed in educational contexts. Signal: uploader's channel category, video description keywords, audience demographics. (3) Implementation: a 'context classifier' that predicts the purpose of content (entertainment vs news vs education vs malicious). This score modifies the enforcement threshold — newsworthy content gets a higher bar for removal. (4) Human review: edge cases ALWAYS go to human review. ML flags the potential exception type; the reviewer makes the final call. (5) Policy clarity: published guidelines with examples of what's allowed and what isn't. Reduces inconsistency.", tags:["policy"] },
    { q:"How would you architect this system for a brand new platform with no training data?", a:"Cold-start for moderation is dangerous — a new platform without moderation attracts bad actors immediately. Approach: (1) Start with external APIs: Google Cloud Vision SafeSearch, OpenAI Moderation API, Perspective API for toxicity. Expensive per-call but immediate coverage. (2) Adopt existing hash databases: join NCMEC hash-sharing program (mandatory in many jurisdictions for CSAM), GIFCT for terrorism. Deterministic coverage for known-bad content. (3) Rule-based systems: keyword blocklists, regex patterns for common violations. Low accuracy but fast to deploy. (4) Community-driven: prominent 'Report' button. Rely on user reports more heavily initially. Hire a small team of reviewers. (5) Bootstrapping ML: use reports + reviews as labeled data. After 3-6 months, you have enough data to train custom classifiers. (6) Transfer learning: fine-tune open-source models (LLaMA Guard, Detoxify) on your platform's specific data. Dramatically reduces the data needed to build accurate classifiers.", tags:["design"] },
  ];
  return (
    <div className="space-y-2.5">
      <Card className="bg-indigo-50/50 border-indigo-200 mb-1">
        <p className="text-[12px] text-stone-500">Common follow-up questions Google L6 interviewers ask about content moderation. Click to reveal a strong answer.</p>
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
  models: ModelsSection, multimodal: MultiModalSection, features: FeaturesSection,
  data: DataModelSection, training: TrainingSection, humanloop: HumanLoopSection,
  scalability: ScalabilitySection, watchouts: WatchoutsSection,
  observability: ObservabilitySection,
  enhancements: EnhancementsSection, followups: FollowupsSection,
};

export default function ContentModerationSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Content Moderation System</h1>
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