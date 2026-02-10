import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   ML TRAINING + SERVING INFRASTRUCTURE — ML System Design Reference (Google L6)
   Pearl white theme · 17 sections (HLD + LLD + ML Deep Dive)
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",                icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",            icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",     icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",              icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",       icon: "🏗️", color: "#9333ea" },
  { id: "training",      label: "Training Pipeline",       icon: "🏋️", color: "#c026d3" },
  { id: "distributed",   label: "Distributed Training",    icon: "🔀", color: "#dc2626" },
  { id: "featurestore",  label: "Feature Store",           icon: "🗃️", color: "#d97706" },
  { id: "modelreg",      label: "Model Registry",          icon: "📦", color: "#0f766e" },
  { id: "serving",       label: "Model Serving",           icon: "⚡", color: "#ea580c" },
  { id: "canary",        label: "Deployment & Canary",     icon: "🐦", color: "#059669" },
  { id: "data",          label: "Data Model",              icon: "🗄️", color: "#7e22ce" },
  { id: "monitoring",    label: "Model Monitoring",        icon: "📊", color: "#0284c7" },
  { id: "scalability",   label: "Scalability",             icon: "📈", color: "#059669" },
  { id: "watchouts",     label: "Failure Modes",           icon: "⚠️", color: "#dc2626" },
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
            <Label>What is ML Training + Serving Infrastructure?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              ML infrastructure is the platform that enables every ML-powered product at a company: the systems to train models at scale, manage their lifecycle, serve predictions in real-time, and monitor them in production. At Google, this is the backbone behind Search ranking, Ads CTR, YouTube recommendations, Translate, and every other ML-driven product — thousands of models running simultaneously.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Google builds this as <strong>Vertex AI</strong> (cloud platform), internally as <strong>TFX</strong> (TensorFlow Extended) pipelines, <strong>Borg</strong> for scheduling, and custom TPU infrastructure. The challenge: build a unified platform where any team can train, validate, deploy, and monitor models with minimal friction — from a tiny click model to a 100B-parameter LLM.
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard? (Google L6 Depth)</Label>
            <ul className="space-y-2.5">
              <Point icon="🔀" color="#0891b2">Heterogeneous workloads — small tabular models (LightGBM, 2 seconds to train) and massive transformers (months on hundreds of TPUs). The platform must serve both without over-provisioning.</Point>
              <Point icon="📊" color="#0891b2">Training-serving skew — the #1 silent killer of ML systems. Features computed differently in training vs serving. The model works great offline but degrades in production. Must guarantee feature parity across pipelines.</Point>
              <Point icon="⚡" color="#0891b2">Serving at extreme scale — Google Ads scores 30M predictions/sec at p99 &lt;10ms. Model serving must be rock-solid: autoscaling, graceful degradation, zero-downtime deployments, multi-region.</Point>
              <Point icon="🔄" color="#0891b2">Continuous retraining — production models must be retrained daily/weekly on fresh data, validated automatically, and deployed without human intervention. Any failure in this loop degrades the product.</Point>
              <Point icon="🧪" color="#0891b2">Experimentation velocity — ML teams need to iterate fast. Training a model shouldn't take a week of DevOps. The platform must abstract away infrastructure complexity while giving power users full control.</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Platforms</Label>
            <div className="space-y-2.5">
              {[
                { co: "Google (Vertex AI)", scale: "Thousands of models", approach: "TFX pipelines, TPU pods, Borg" },
                { co: "Meta (FBLearner)", scale: "10K+ production models", approach: "PyTorch + custom infra" },
                { co: "Uber (Michelangelo)", scale: "10K+ models", approach: "Spark + custom serving" },
                { co: "Amazon (SageMaker)", scale: "Cloud ML platform", approach: "Managed training + endpoints" },
                { co: "Netflix", scale: "100s of models", approach: "Metaflow + custom serving" },
              ].map((e,i) => (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-28 shrink-0">{e.co}</span>
                  <span className="text-stone-500 flex-1">{e.scale}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <Label color="#2563eb">The ML Lifecycle (Preview)</Label>
            <svg viewBox="0 0 360 170" className="w-full">
              <defs><marker id="ah-ml" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
              {[
                { x:25, y:10, w:70, h:32, label:"Data Prep", sub:"features, labels", c:"#059669" },
                { x:115, y:10, w:70, h:32, label:"Train", sub:"GPU/TPU cluster", c:"#dc2626" },
                { x:205, y:10, w:70, h:32, label:"Validate", sub:"offline metrics", c:"#d97706" },
                { x:25, y:60, w:70, h:32, label:"Register", sub:"model registry", c:"#0f766e" },
                { x:115, y:60, w:70, h:32, label:"Deploy", sub:"canary rollout", c:"#7c3aed" },
                { x:205, y:60, w:70, h:32, label:"Serve", sub:"real-time/batch", c:"#ea580c" },
                { x:115, y:110, w:70, h:32, label:"Monitor", sub:"drift, latency", c:"#0284c7" },
              ].map((b,i) => (
                <g key={i}>
                  <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={6} fill={b.c+"10"} stroke={b.c} strokeWidth={1.5}/>
                  <text x={b.x+b.w/2} y={b.y+14} textAnchor="middle" fill={b.c} fontSize="8" fontWeight="700" fontFamily="monospace">{b.label}</text>
                  <text x={b.x+b.w/2} y={b.y+26} textAnchor="middle" fill={b.c+"80"} fontSize="7" fontFamily="monospace">{b.sub}</text>
                </g>
              ))}
              {/* Flow arrows */}
              <line x1={95} y1={26} x2={115} y2={26} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-ml)"/>
              <line x1={185} y1={26} x2={205} y2={26} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-ml)"/>
              <line x1={240} y1={42} x2={240} y2={55} stroke="#94a3b8" strokeWidth={1} markerEnd="url(#ah-ml)"/>
              <line x1={205} y1={76} x2={185} y2={76} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-ml)"/>
              <line x1={115} y1={76} x2={95} y2={76} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-ml)"/>
              <line x1={150} y1={92} x2={150} y2={110} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-ml)"/>
              {/* Feedback loop */}
              <path d="M115,126 Q30,126 30,42" stroke="#0284c7" strokeWidth={1} strokeDasharray="3,2" fill="none" markerEnd="url(#ah-ml)"/>
              <text x={15} y={90} fill="#0284c7" fontSize="6" fontFamily="monospace" transform="rotate(-90,15,90)">retrain</text>

              <rect x={10} y={150} width={280} height={15} rx={4} fill="#faf9f7" stroke="#e7e5e4"/>
              <text x={150} y={161} textAnchor="middle" fill="#78716c" fontSize="7" fontFamily="monospace">Continuous loop: data → train → validate → deploy → monitor → retrain</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Google Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Core infra for every ML team at Google</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Scope Like an L6</div>
            <p className="text-[12px] text-stone-500 mt-0.5">"Design ML infrastructure" is massive. Scope: "I'll design a platform for the full ML lifecycle — feature management, distributed training, model registry, real-time serving, and production monitoring. I'll focus on the serving path (latency-critical) and the training-serving consistency problem (feature store). I'll treat the scheduler (Borg/K8s) and hardware (TPU/GPU) as given." This signals infrastructure breadth with clear depth focus.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Train models of any size: tabular (minutes) to LLMs (weeks on multi-node GPU/TPU)</Point>
            <Point icon="2." color="#059669">Feature store: compute features once, share across training and serving (eliminating skew)</Point>
            <Point icon="3." color="#059669">Model registry: version, stage (dev/staging/prod), lineage, and artifact management</Point>
            <Point icon="4." color="#059669">Real-time serving: low-latency prediction endpoints with autoscaling</Point>
            <Point icon="5." color="#059669">Batch prediction: offline scoring for large datasets (periodic ranking, pre-computation)</Point>
            <Point icon="6." color="#059669">Automated retraining: scheduled pipelines that retrain, validate, and deploy without human intervention</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Serving latency: p50 &lt;5ms, p99 &lt;20ms for inline models (ads, search ranking)</Point>
            <Point icon="2." color="#dc2626">Serving throughput: 100K+ predictions/sec per model, millions platform-wide</Point>
            <Point icon="3." color="#dc2626">Training efficiency: &gt;80% GPU/TPU utilization across the cluster</Point>
            <Point icon="4." color="#dc2626">Zero-downtime deployments: model updates with no serving interruption</Point>
            <Point icon="5." color="#dc2626">Availability: 99.99% for serving (if model serving is down, products are down)</Point>
            <Point icon="6." color="#dc2626">Multi-tenancy: thousands of teams sharing infrastructure with resource isolation</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask (L6 Signal)</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "What model types? Tabular (GBDT), deep learning (TF/PyTorch), LLMs, or all?",
            "Online serving only, or also batch prediction pipelines?",
            "Feature store: real-time features (streaming) or batch features only?",
            "Multi-framework (TensorFlow + PyTorch + JAX) or single-framework?",
            "What's the GPU/TPU budget? Shared cluster or dedicated per team?",
            "Regulatory: model explainability, audit trails, data lineage requirements?",
            "Self-service (ML engineers deploy their own) or centralized ML ops team?",
            "Existing infra: Kubernetes, Borg, or building from scratch?",
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
          <Label color="#7c3aed">Step 1 — Training Cluster</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Active ML teams" result="~500 teams" note="At Google scale. Each team trains 1-20 models." />
            <MathStep step="2" formula="Concurrent training jobs" result="~2,000" note="Mix of small (1 GPU, 10 min) and large (256 GPUs, days)." />
            <MathStep step="3" formula="GPU-hours consumed/day" result="~100K GPU-hrs" note="$2-3/GPU-hr → $200-300K/day in compute." final />
            <MathStep step="4" formula="Peak TPU pod-hours (LLMs)" result="~5K TPU-hrs/day" note="TPU v4 pods for large model training. 10x cost/hr vs GPU." />
            <MathStep step="5" formula="Target cluster utilization" result=">80%" note="Below 80% = wasted money. Above 95% = queuing delays." />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Serving Infrastructure</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Total production models" result="~5,000" note="Across all teams. Each needs a serving endpoint." />
            <MathStep step="2" formula="Aggregate prediction QPS" result="~10M QPS" note="Ads (30M/s), Search (500K/s), Recs (1M/s), others." final />
            <MathStep step="3" formula="Serving machines (CPU-based models)" result="~20K servers" note="Most models serve on CPU. Each handles ~500 QPS." />
            <MathStep step="4" formula="GPU serving instances (deep models)" result="~2,000 GPUs" note="Large embedding models, transformers, LLMs need GPU." />
            <MathStep step="5" formula="Model artifact storage" result="~50 TB" note="5K models × 10 versions × 1 GB avg = 50 TB in registry." />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#dc2626">
          <Label color="#dc2626">Step 3 — Feature Store</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Unique feature definitions" result="~50K features" note="Across all teams. Many shared (user features, device features)." />
            <MathStep step="2" formula="Online feature lookups/sec" result="~50M/sec" note="Every prediction request fetches 50-200 features." final />
            <MathStep step="3" formula="Online store size (Redis/Bigtable)" result="~5 TB" note="Latest feature values for all entities. Must be in-memory tier." />
            <MathStep step="4" formula="Offline store size (BigQuery/Hive)" result="~10 PB" note="Historical feature values for training data construction." />
            <MathStep step="5" formula="Feature freshness requirement" result="<1 min (real-time)" note="Streaming features updated per-event. Batch features daily." />
          </div>
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Step 4 — Cost Breakdown</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Training compute" result="~$250K/day" note="100K GPU-hrs at $2.50/hr. Dominant cost." final />
            <MathStep step="2" formula="Serving compute" result="~$150K/day" note="20K CPU servers + 2K GPUs. Always-on." />
            <MathStep step="3" formula="Feature store (online)" result="~$50K/day" note="Redis/Bigtable for 50M lookups/sec. Memory-intensive." />
            <MathStep step="4" formula="Storage (models + features + logs)" result="~$30K/day" note="50TB model registry + 10PB feature history." />
            <MathStep step="5" formula="Total platform cost" result="~$175M/year" note="Justifies a dedicated ML platform team." final />
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
          <Label color="#2563eb">Platform APIs</Label>
          <CodeBlock code={`# 1. TRAINING API — submit a training job
# POST /v1/training/jobs
{
  "job_name": "ads_ctr_model_v42",
  "framework": "tensorflow",       // tensorflow | pytorch | jax | xgboost
  "entry_point": "gs://ml-code/ads_ctr/train.py",
  "config": {
    "accelerator": "nvidia-a100",   // a100 | t4 | tpu-v4
    "num_workers": 8,
    "gpus_per_worker": 4,
    "distributed_strategy": "data_parallel",
  },
  "data": {
    "train": "gs://ml-data/ads_ctr/train/2024-02-*",
    "eval": "gs://ml-data/ads_ctr/eval/2024-02-10",
    "feature_set": "fs://ads/ctr_features_v3",  # feature store ref
  },
  "hyperparams": {"lr": 0.001, "batch_size": 2048, "epochs": 5},
  "schedule": "daily_04:00_utc",    // null for one-off
  "auto_deploy": {
    "enabled": true,
    "validation_criteria": {
      "auc_roc": {"min": 0.78, "max_regression": 0.005},
      "calibration_error": {"max": 0.02},
    },
    "canary_pct": 5,
    "canary_duration_hours": 6,
  }
}

# 2. SERVING API — create a prediction endpoint
# POST /v1/endpoints
{
  "endpoint_name": "ads_ctr_serving",
  "model_version": "registry://ads_ctr/v42",
  "serving_config": {
    "min_replicas": 10,
    "max_replicas": 500,
    "target_latency_p99_ms": 10,
    "accelerator": "cpu",       // cpu | gpu | tpu
    "batch_config": {"max_batch_size": 64, "timeout_ms": 5},
  }
}

# 3. PREDICTION API — real-time inference
# POST /v1/predict/ads_ctr_serving
{
  "instances": [
    {"user_id": "u_123", "query": "shoes", "ad_id": "a_456"},
  ]
}
# Response: {"predictions": [{"ctr": 0.032, "latency_ms": 3.2}]}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Design Decisions (L6 Depth)</Label>
          <div className="space-y-3">
            {[
              { q: "Why feature_set reference instead of raw data paths?", a: "The feature store decouples feature definition from consumption. Training and serving both reference the SAME feature definition (fs://ads/ctr_features_v3). This guarantees training-serving consistency. If training reads raw data files while serving computes features differently, you get skew — the #1 production ML bug." },
              { q: "Why auto_deploy with validation criteria?", a: "Continuous deployment for ML: the model is retrained daily. If it passes validation gates (AUC didn't regress, calibration is good), it's automatically deployed via canary. No human in the loop for routine retraining. Humans only intervene when validation fails. This is how Google operates 5,000+ production models." },
              { q: "Why batch_config on the serving endpoint?", a: "Dynamic batching: accumulate individual prediction requests and batch them for GPU inference. A single GPU processes 64 predictions in ~5ms total, vs 64 × 3ms = 192ms individually. 40x throughput improvement. The timeout_ms (5ms) limits how long we wait to fill a batch before sending a partial one." },
              { q: "Why min/max replicas instead of fixed?", a: "Traffic varies 10x between peak and off-peak. Fixed replicas waste 80% of compute at off-peak or cause latency spikes at peak. Autoscaling based on QPS and latency targets adapts automatically. HPA (Horizontal Pod Autoscaler) with custom metrics (p99 latency) is the standard." },
              { q: "Why support multiple frameworks?", a: "Different teams use different frameworks: TensorFlow for production models with SavedModel serving, PyTorch for research, JAX for TPU workloads, XGBoost for tabular models. A platform that forces one framework loses adoption. The abstraction layer (training API + serving API) is framework-agnostic." },
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
        <Label color="#9333ea">Full Platform Architecture</Label>
        <svg viewBox="0 0 720 380" className="w-full">
          <defs><marker id="ah-hld" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>

          {/* Training Path (top) */}
          <text x={20} y={18} fill="#dc2626" fontSize="9" fontWeight="700" fontFamily="monospace">TRAINING PATH</text>
          <rect x={15} y={25} width={80} height={40} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={55} y={42} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Data Sources</text>
          <text x={55} y={55} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">logs, events</text>

          <rect x={115} y={25} width={90} height={40} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={160} y={42} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Feature Store</text>
          <text x={160} y={55} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">offline (batch)</text>

          <rect x={225} y={25} width={85} height={40} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={267} y={42} textAnchor="middle" fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Data Pipeline</text>
          <text x={267} y={55} textAnchor="middle" fill="#c026d380" fontSize="7" fontFamily="monospace">TFX / Dataflow</text>

          <rect x={330} y={25} width={100} height={40} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={380} y={42} textAnchor="middle" fill="#dc2626" fontSize="8" fontWeight="600" fontFamily="monospace">Training Cluster</text>
          <text x={380} y={55} textAnchor="middle" fill="#dc262680" fontSize="7" fontFamily="monospace">GPU / TPU pods</text>

          <rect x={450} y={25} width={80} height={40} rx={6} fill="#7c3aed10" stroke="#7c3aed" strokeWidth={1.5}/>
          <text x={490} y={42} textAnchor="middle" fill="#7c3aed" fontSize="8" fontWeight="600" fontFamily="monospace">Evaluator</text>
          <text x={490} y={55} textAnchor="middle" fill="#7c3aed80" fontSize="7" fontFamily="monospace">offline metrics</text>

          <rect x={550} y={25} width={80} height={40} rx={6} fill="#0f766e10" stroke="#0f766e" strokeWidth={1.5}/>
          <text x={590} y={42} textAnchor="middle" fill="#0f766e" fontSize="8" fontWeight="600" fontFamily="monospace">Model Reg.</text>
          <text x={590} y={55} textAnchor="middle" fill="#0f766e80" fontSize="7" fontFamily="monospace">versioned</text>

          {/* Training arrows */}
          <line x1={95} y1={45} x2={115} y2={45} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={205} y1={45} x2={225} y2={45} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={310} y1={45} x2={330} y2={45} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={430} y1={45} x2={450} y2={45} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={530} y1={45} x2={550} y2={45} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>

          {/* Serving Path (middle) */}
          <text x={20} y={98} fill="#ea580c" fontSize="9" fontWeight="700" fontFamily="monospace">SERVING PATH</text>
          <rect x={15} y={105} width={80} height={40} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={55} y={122} textAnchor="middle" fill="#2563eb" fontSize="8" fontWeight="600" fontFamily="monospace">Request</text>
          <text x={55} y={135} textAnchor="middle" fill="#2563eb80" fontSize="7" fontFamily="monospace">user + context</text>

          <rect x={115} y={105} width={90} height={40} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={160} y={122} textAnchor="middle" fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Feature Store</text>
          <text x={160} y={135} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">online (real-time)</text>

          <rect x={225} y={105} width={105} height={40} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={277} y={122} textAnchor="middle" fill="#ea580c" fontSize="8" fontWeight="600" fontFamily="monospace">Model Serving</text>
          <text x={277} y={135} textAnchor="middle" fill="#ea580c80" fontSize="7" fontFamily="monospace">TF Serving / Triton</text>

          <rect x={350} y={105} width={80} height={40} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={390} y={122} textAnchor="middle" fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Response</text>
          <text x={390} y={135} textAnchor="middle" fill="#05966980" fontSize="7" fontFamily="monospace">prediction</text>

          {/* Serving arrows */}
          <line x1={95} y1={125} x2={115} y2={125} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={205} y1={125} x2={225} y2={125} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={330} y1={125} x2={350} y2={125} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>

          {/* Model Registry to Serving */}
          <path d="M590,65 L590,100 Q590,115 575,115 L330,115" stroke="#0f766e60" strokeWidth={1.2} strokeDasharray="3,2" fill="none" markerEnd="url(#ah-hld)"/>

          {/* Feature Store shared between training and serving */}
          <line x1={160} y1={65} x2={160} y2={105} stroke="#d97706" strokeWidth={2} strokeDasharray="4,2"/>
          <text x={170} y={88} fill="#d97706" fontSize="7" fontWeight="600" fontFamily="monospace">SHARED</text>

          {/* Monitoring */}
          <rect x={450} y={105} width={90} height={40} rx={6} fill="#0284c710" stroke="#0284c7" strokeWidth={1.5}/>
          <text x={495} y={122} textAnchor="middle" fill="#0284c7" fontSize="8" fontWeight="600" fontFamily="monospace">Monitoring</text>
          <text x={495} y={135} textAnchor="middle" fill="#0284c780" fontSize="7" fontFamily="monospace">drift, latency</text>
          <line x1={390} y1={145} x2={450} y2={135} stroke="#0284c760" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#ah-hld)"/>

          {/* Legend */}
          <rect x={15} y={170} width={700} height={195} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={25} y={190} fill="#d97706" fontSize="8" fontWeight="700" fontFamily="monospace">Feature Store (SHARED between training + serving) — THE KEY COMPONENT</text>
          <text x={25} y={205} fill="#78716c" fontSize="8" fontFamily="monospace">  Offline store: batch-computed features materialized to BigQuery/Hive. Used for training data construction.</text>
          <text x={25} y={218} fill="#78716c" fontSize="8" fontFamily="monospace">  Online store: latest feature values in Redis/Bigtable. Used for real-time serving. Same feature definitions.</text>
          <text x={25} y={231} fill="#78716c" fontSize="8" fontFamily="monospace">  This dual-write architecture GUARANTEES training-serving consistency. Same code computes both.</text>
          <text x={25} y={250} fill="#dc2626" fontSize="8" fontWeight="700" fontFamily="monospace">Training Path: data → feature store (offline) → data pipeline → GPU/TPU training → evaluate → register</text>
          <text x={25} y={265} fill="#ea580c" fontSize="8" fontWeight="700" fontFamily="monospace">Serving Path: request → feature store (online) → model server → response</text>
          <text x={25} y={280} fill="#0f766e" fontSize="8" fontWeight="700" fontFamily="monospace">Model Registry: central storage for all model versions with metadata, lineage, and promotion stages</text>
          <text x={25} y={295} fill="#0284c7" fontSize="8" fontWeight="700" fontFamily="monospace">Monitoring: tracks prediction drift, feature drift, latency, throughput. Triggers retraining if degraded.</text>
          <text x={25} y={315} fill="#6366f1" fontSize="8" fontWeight="600" fontFamily="monospace">L6 KEY INSIGHT: The feature store bridging training and serving is the most important architectural decision.</text>
          <text x={25} y={330} fill="#6366f1" fontSize="8" fontFamily="monospace">Without it, every team implements features twice (Python for training, C++ for serving) and skew is inevitable.</text>
          <text x={25} y={345} fill="#6366f1" fontSize="8" fontFamily="monospace">Google's solution: feature definitions are compiled to BOTH offline (Flume/Beam) and online (C++ serving) code.</text>
          <text x={25} y={360} fill="#6366f1" fontSize="8" fontFamily="monospace">This "write once, execute anywhere" approach eliminated the single largest class of ML production bugs at Google.</text>
        </svg>
      </Card>
    </div>
  );
}

function TrainingSection() {
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Training Pipeline — From Data to Model Artifact</Label>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Automated Training Pipeline (TFX-style)" code={`# End-to-end training pipeline — runs daily
class TrainingPipeline:
    def run(self, config):
        # Step 1: Data Validation
        # Check for schema drift, missing values, distribution shift
        data_stats = StatisticsGen(config.train_data)
        anomalies = SchemaValidator(data_stats, config.expected_schema)
        if anomalies.has_critical():
            alert_oncall("Data validation failed", anomalies)
            return PipelineResult(status="DATA_ERROR")

        # Step 2: Feature Engineering
        # Transform raw data using feature store definitions
        features = FeatureTransform(
            raw_data=config.train_data,
            feature_set=config.feature_set,  # from feature store
            # Point-in-time join: features as they were at event time
            # NOT current values (would cause data leakage)
        )

        # Step 3: Training
        model = Trainer(
            framework=config.framework,
            hyperparams=config.hyperparams,
            train_data=features.train,
            eval_data=features.eval,
            accelerator=config.accelerator,
            distributed=config.distributed_strategy,
        )

        # Step 4: Model Evaluation
        metrics = Evaluator(model, features.eval)
        # Compare against:
        #   a) Absolute thresholds (AUC > 0.78)
        #   b) Current production model (no regression > 0.5%)
        #   c) Baseline model (better than simple heuristic)

        if not metrics.passes_gates(config.validation_criteria):
            alert_oncall("Model validation failed", metrics)
            return PipelineResult(status="VALIDATION_FAILED")

        # Step 5: Model Registration
        model_registry.register(
            model=model,
            metrics=metrics,
            lineage={
                "training_data": config.train_data,
                "feature_set": config.feature_set,
                "hyperparams": config.hyperparams,
                "pipeline_run_id": self.run_id,
            },
        )

        # Step 6: Auto-Deploy (if enabled)
        if config.auto_deploy.enabled:
            deployer.canary_deploy(
                model=model,
                endpoint=config.endpoint,
                canary_pct=config.auto_deploy.canary_pct,
            )

        return PipelineResult(status="SUCCESS", model=model)`} />
          <div className="space-y-4">
            <Card accent="#c026d3">
              <Label color="#c026d3">Training Pipeline Components</Label>
              <div className="space-y-2">
                {[
                  { comp: "Data Validation (TFDV)", role: "Detect schema changes, missing values, distribution shift in training data BEFORE training starts. Catches data pipeline bugs early.", why: "A model trained on corrupted data looks fine in metrics but fails in production." },
                  { comp: "Point-in-Time Feature Join", role: "When constructing training data, join features as they existed AT THE TIME of the event, not current values.", why: "Using current features for past events is data leakage. The model learns to use 'future information' not available at serving time." },
                  { comp: "Model Evaluator", role: "Compare new model against production model on held-out evaluation set. Gate deployment on no-regression criteria.", why: "Without auto-validation, bad models reach production. With it, only improvements get deployed. The human only reviews failures." },
                  { comp: "Lineage Tracking", role: "Record exactly which data, features, code, and hyperparams produced each model. Full reproducibility.", why: "When a model degrades in production, you need to trace back: was it the data? The features? A code change? Lineage enables root-cause analysis." },
                ].map((c,i) => (
                  <div key={i} className="rounded-lg border border-stone-200 p-2.5">
                    <div className="text-[11px] font-bold text-stone-700">{c.comp}</div>
                    <div className="text-[10px] text-stone-500 mt-0.5">{c.role}</div>
                    <div className="text-[10px] text-fuchsia-600 mt-0.5"><strong>Why:</strong> {c.why}</div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </Card>
    </div>
  );
}

function DistributedSection() {
  return (
    <div className="space-y-5">
      <Card accent="#dc2626">
        <Label color="#dc2626">Distributed Training Strategies</Label>
        <p className="text-[12px] text-stone-500 mb-4">When a model doesn't fit on one GPU, or training takes too long on one machine, you need distributed training. The strategy depends on what's too big — the data or the model.</p>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Data Parallelism (Most Common)" code={`# Data Parallelism: same model on every worker, different data
# Used when: model fits on 1 GPU, but training is slow
# Example: CTR model, 10B training examples, 8 GPUs

# Each GPU gets batch_size/num_gpus examples
strategy = tf.distribute.MirroredStrategy(devices=8)
with strategy.scope():
    model = build_ctr_model()  # replicated to all 8 GPUs

# Training loop:
# 1. Each GPU processes its mini-batch independently
# 2. Compute gradients locally
# 3. All-Reduce: average gradients across all GPUs
# 4. Each GPU updates its copy of the model identically
#
# All-Reduce communication pattern:
#   Ring AllReduce: each GPU sends to neighbor in a ring
#   Bandwidth-optimal: each GPU sends N/P data total
#   Latency: O(P) steps for P GPUs
#
# SCALING EFFICIENCY:
#   8 GPUs: ~7.5x speedup (93% efficiency)
#   64 GPUs: ~50x speedup (78% efficiency)
#   256 GPUs: ~150x speedup (58% efficiency)
#   Efficiency drops due to communication overhead
#
# LARGE BATCH TRAINING TRICKS:
# - Linear learning rate scaling (lr * num_gpus)
# - Gradual warmup (5 epochs at lower lr)
# - LARS/LAMB optimizer (layer-wise adaptive lr)
# - Mixed precision (FP16 compute, FP32 accumulate)`} />
          <CodeBlock title="Model Parallelism (Large Models)" code={`# Model Parallelism: split model across GPUs
# Used when: model TOO LARGE for 1 GPU (LLMs, huge embeddings)
# Example: 100B parameter LLM, 80GB per GPU → need 20+ GPUs

# TENSOR PARALLELISM (within a layer):
# Split weight matrices across GPUs
# Example: Linear(4096, 4096) on 4 GPUs
# Each GPU: Linear(4096, 1024) — 1/4 of output dim
# All-gather at output: combine partial results
# Used in: Megatron-LM, Google PaLM

# PIPELINE PARALLELISM (across layers):
# Layer 0-11 on GPU 0, layers 12-23 on GPU 1, etc.
# Micro-batching: split batch into micro-batches
# GPU 0 processes micro-batch 1, forwards to GPU 1,
# then starts micro-batch 2 (pipeline overlap)
# Used in: GPipe, PipeDream

# FULLY SHARDED DATA PARALLEL (FSDP / ZeRO):
# Shard model parameters, gradients, AND optimizer states
# across all GPUs. Each GPU stores only 1/N of everything.
# All-gather parameters just-in-time before forward pass.
# ZeRO Stage 3: reduces memory per GPU by N×
#   64 GPUs → model can be 64× larger than 1-GPU memory
# Used in: PyTorch FSDP, DeepSpeed ZeRO

# Google TPU Pod Training:
# TPU v4 pod: 4096 chips, 1.1 exaFLOPS
# 3D torus interconnect: high-bandwidth chip-to-chip
# Data parallel across pod slices
# Model parallel within each slice
# PaLM 540B: trained on 6144 TPU v4 chips`} />
        </div>
      </Card>
      <Card>
        <Label color="#d97706">When to Use Each Strategy</Label>
        <div className="overflow-hidden rounded-lg border border-stone-200">
          <table className="w-full text-[12px]">
            <thead><tr className="bg-stone-50">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Strategy</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">When</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Scaling</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Complexity</th>
            </tr></thead>
            <tbody>
              {[
                { s:"Data Parallel ★", when:"Model fits on 1 GPU. Training is slow due to data volume. (Most production ML)", scale:"2-64 GPUs", cx:"Low" },
                { s:"Model Parallel (Tensor)", when:"Model weights don't fit on 1 GPU. (LLMs, huge embeddings)", scale:"2-8 GPUs per model", cx:"High" },
                { s:"Pipeline Parallel", when:"Deep models with many sequential layers. Combine with data parallel.", scale:"4-32 GPUs", cx:"High" },
                { s:"FSDP / ZeRO", when:"Model + optimizer state too large. (LLMs, large batch training)", scale:"8-1000+ GPUs", cx:"Medium" },
                { s:"Hybrid (Data + Model)", when:"Both model is huge AND data is huge. (PaLM, GPT-4 class)", scale:"100-10,000 GPUs", cx:"Very High" },
              ].map((r,i) => (
                <tr key={i} className={i===0?"bg-fuchsia-50":""}>
                  <td className="px-3 py-2 font-bold text-stone-700">{r.s}</td>
                  <td className="px-3 py-2 text-stone-500 text-[11px]">{r.when}</td>
                  <td className="text-center px-3 py-2 text-stone-600">{r.scale}</td>
                  <td className="text-center px-3 py-2 text-stone-600">{r.cx}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function FeatureStoreSection() {
  return (
    <div className="space-y-5">
      <Card accent="#d97706">
        <Label color="#d97706">Feature Store — Eliminating Training-Serving Skew</Label>
        <p className="text-[12px] text-stone-500 mb-4">The feature store is the single most important component for ML infrastructure reliability. It provides a unified abstraction: define a feature once, compute it consistently for both training (batch) and serving (real-time), and version it alongside models.</p>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Feature Store Architecture" code={`# Feature definition: write once, use for training AND serving
@feature_set(name="user_features_v3", entity="user_id")
class UserFeatures:
    # Feature definitions with transformations
    @feature
    def avg_purchase_amount_30d(self, user_id):
        # This SAME logic runs in both:
        #   Batch (BigQuery SQL for training data)
        #   Streaming (Flink/Beam for online store)
        return purchases.where(
            user_id=user_id,
            window=last_30_days
        ).amount.mean()

    @feature
    def purchase_count_7d(self, user_id):
        return purchases.where(
            user_id=user_id,
            window=last_7_days
        ).count()

    @feature
    def days_since_last_login(self, user_id):
        return (now() - logins.where(
            user_id=user_id
        ).max_timestamp()).days

# DUAL MATERIALIZATION:
# 1. Offline store (BigQuery): feature values at EVERY point in time
#    Training query: "give me user features AS OF Feb 1, 2024"
#    Point-in-time correct: no future data leakage
#
# 2. Online store (Redis/Bigtable): LATEST feature values only
#    Serving query: "give me current user features for user_123"
#    Sub-millisecond latency. Updated via streaming pipeline.
#
# BOTH stores use the SAME feature transformation code
# Compiled to SQL (offline) and streaming (online) automatically
# This GUARANTEES training-serving consistency`} />
          <div className="space-y-4">
            <Card accent="#d97706">
              <Label color="#d97706">Training-Serving Skew: The #1 ML Bug</Label>
              <CodeBlock code={`# WITHOUT feature store (skew-prone):
#
# Training (Python):
#   avg_amount = df.groupby("user_id")["amount"].mean()
#   # Includes ALL purchases (even those after the event!)
#
# Serving (C++):
#   avg_amount = redis.get(f"user:{user_id}:avg_amount")
#   # Only includes purchases before the current moment
#
# RESULT: model trained on "average including future"
#         but served on "average up to now"
# The feature values DON'T MATCH → model accuracy drops 5-20%
# Impossible to debug because both look "correct" in isolation

# COMMON SKEW SOURCES:
# 1. Temporal leakage: training uses future data
# 2. Different aggregation logic (mean vs median)
# 3. Different null handling (0 vs NaN vs drop)
# 4. Different data sources (training: BigQuery, serving: Redis)
# 5. Stale online features (not updated fast enough)

# WITH feature store:
# Same code → same logic → same values → no skew
# The feature store is an ARCHITECTURAL GUARANTEE
# not just a "best practice"`} />
            </Card>
          </div>
        </div>
      </Card>
    </div>
  );
}

function ModelRegistrySection() {
  return (
    <div className="space-y-5">
      <Card accent="#0f766e">
        <Label color="#0f766e">Model Registry — Version Control for Models</Label>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Model Registry Operations" code={`# Model Registry: the "Git for models"
class ModelRegistry:
    def register(self, model_name, version, artifact, metadata):
        entry = {
            "model_name": model_name,
            "version": version,
            "artifact_uri": f"gs://model-artifacts/{model_name}/{version}/",
            "framework": metadata.framework,
            "metrics": metadata.metrics,  # AUC, loss, calibration
            "lineage": {
                "training_data": metadata.data_path,
                "feature_set": metadata.feature_set,
                "hyperparams": metadata.hyperparams,
                "training_job_id": metadata.job_id,
                "code_commit": metadata.git_sha,
                "parent_model": metadata.parent_version,
            },
            "stage": "development",
            # Stages: development → staging → canary → production
            "created_at": now(),
            "created_by": metadata.user,
            "model_size_bytes": artifact.size,
            "serving_signature": artifact.input_output_spec,
        }
        self.store.put(entry)

    def promote(self, model_name, version, target_stage):
        # Promotion gates by stage:
        if target_stage == "staging":
            # Requires: offline metrics pass threshold
            assert self.passes_offline_gates(model_name, version)
        elif target_stage == "canary":
            # Requires: staging metrics + shadow scoring comparison
            assert self.passes_staging_gates(model_name, version)
        elif target_stage == "production":
            # Requires: canary period passed with no regression
            assert self.passes_canary_gates(model_name, version)

        self.store.update_stage(model_name, version, target_stage)

    def rollback(self, model_name):
        # Instantly rollback to previous production version
        prev = self.get_previous_production(model_name)
        self.promote(model_name, prev.version, "production")
        alert_oncall(f"Rollback: {model_name} to {prev.version}")`} />
          <Card accent="#0f766e">
            <Label color="#0f766e">Model Lifecycle Stages</Label>
            <div className="space-y-2">
              {[
                { stage: "Development", desc: "Model trained and evaluated offline. Metrics recorded. Lives in the registry but not deployed anywhere.", gate: "Offline metrics pass absolute thresholds", color: "#78716c" },
                { stage: "Staging", desc: "Deployed to a staging environment with production-like traffic (shadow scoring). Compared against current production model.", gate: "No regression vs production on shadow traffic", color: "#d97706" },
                { stage: "Canary", desc: "Serves 1-5% of live production traffic. Monitored for latency, prediction distribution, business metrics.", gate: "No metric degradation after 6-24 hours", color: "#ea580c" },
                { stage: "Production", desc: "Serves 100% of traffic. Continuously monitored. Previous version kept warm for instant rollback.", gate: "N/A (already serving)", color: "#059669" },
                { stage: "Archived", desc: "Previous production versions. Kept for 30 days for rollback capability. Then garbage collected.", gate: "Time-based retention policy", color: "#94a3b8" },
              ].map((s,i) => (
                <div key={i} className="rounded-lg border p-2.5" style={{ borderLeft: `3px solid ${s.color}` }}>
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-[11px] font-bold" style={{ color: s.color }}>{s.stage}</span>
                  </div>
                  <p className="text-[10px] text-stone-500">{s.desc}</p>
                  <p className="text-[9px] text-stone-400 mt-0.5">Gate: {s.gate}</p>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </Card>
    </div>
  );
}

function ServingSection() {
  return (
    <div className="space-y-5">
      <Card accent="#ea580c">
        <Label color="#ea580c">Model Serving — Real-Time Predictions at Scale</Label>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Serving Infrastructure" code={`# Model Serving — latency-critical path
class ModelServer:
    def __init__(self, config):
        # Load model artifact from registry
        self.model = load_model(config.model_uri)

        # Dynamic batching: accumulate requests, batch for GPU
        self.batcher = DynamicBatcher(
            max_batch_size=config.max_batch_size,
            max_latency_ms=config.batch_timeout_ms,
        )

        # Feature fetcher (parallel async)
        self.feature_store = OnlineFeatureStore(config.feature_set)

    async def predict(self, request):
        t0 = time.now()

        # Step 1: Fetch features from online store (~3ms)
        features = await self.feature_store.get_features(
            entity_ids=request.entity_ids,
            feature_names=self.model.required_features,
        )

        # Step 2: Assemble input tensor
        input_tensor = self.model.preprocess(request, features)

        # Step 3: Batch inference (~2ms amortized)
        prediction = await self.batcher.predict(input_tensor)
        # DynamicBatcher accumulates multiple requests
        # Sends batch to GPU when full or timeout reached
        # Returns individual results to each caller

        # Step 4: Post-process
        result = self.model.postprocess(prediction)

        latency = time.now() - t0
        metrics.record("serving_latency_ms", latency)
        return result

# SERVING FRAMEWORKS:
# TF Serving: SavedModel format, gRPC/REST, dynamic batching
# NVIDIA Triton: multi-framework, GPU-optimized, ensemble support
# TorchServe: PyTorch native, model archiving
# vLLM: LLM-specialized, PagedAttention, continuous batching
#
# Model format matters for serving speed:
# SavedModel (TF) → TF Serving (fastest for TF models)
# TorchScript → TorchServe
# ONNX → ONNX Runtime (cross-framework, optimized)
# TensorRT → maximum GPU throughput (NVIDIA-only)`} />
          <div className="space-y-4">
            <Card accent="#ea580c">
              <Label color="#ea580c">Serving Optimization Techniques</Label>
              <div className="space-y-2">
                {[
                  { tech: "Dynamic Batching", desc: "Accumulate individual requests, batch for GPU. Single batch of 64 is faster than 64 individual inferences. Max latency timeout prevents starvation.", saves: "5-40x throughput" },
                  { tech: "Model Quantization", desc: "INT8 or FP16 inference instead of FP32. Halves model size and doubles throughput. <1% accuracy loss for most models.", saves: "2x throughput, 50% memory" },
                  { tech: "Embedding Caching", desc: "Cache frequently accessed embedding lookups (popular items, power users). LRU cache with high hit rate (>40% for top entities).", saves: "~30% latency reduction" },
                  { tech: "Model Distillation", desc: "Train a smaller 'student' model to mimic the large 'teacher'. Serve the student in production. 10x smaller, 5x faster, ~2% accuracy loss.", saves: "5-10x latency" },
                  { tech: "Graph Optimization", desc: "TF graph optimization: constant folding, operator fusion, dead code elimination. Automatic via XLA compiler or TF-TRT.", saves: "10-30% latency" },
                  { tech: "Pre-computation", desc: "Pre-compute scores for popular queries/items during off-peak. Serve from cache. Only compute real-time for long-tail queries.", saves: "Eliminates inference for top queries" },
                ].map((t,i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-orange-100 text-orange-700 shrink-0 whitespace-nowrap">{t.saves}</span>
                    <div>
                      <div className="text-[11px] font-bold text-stone-700">{t.tech}</div>
                      <div className="text-[10px] text-stone-500">{t.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </Card>
    </div>
  );
}

function CanarySection() {
  return (
    <div className="space-y-5">
      <Card accent="#059669">
        <Label color="#059669">Deployment Strategy — Zero-Downtime Model Updates</Label>
        <div className="grid grid-cols-2 gap-5">
          <CodeBlock title="Canary Deployment Pipeline" code={`# Canary deployment: gradual model rollout with auto-rollback
class CanaryDeployer:
    def deploy(self, model, endpoint, config):
        old_model = endpoint.current_model

        # Phase 1: Shadow Scoring (0% live traffic)
        # Score production traffic with new model in parallel
        # Compare predictions: where do they disagree?
        shadow = self.shadow_score(model, endpoint, duration="2h")
        if shadow.disagreement_rate > 0.20:
            alert("High disagreement rate — review before proceeding")
            return

        # Phase 2: Canary (1-5% live traffic)
        endpoint.set_traffic_split({
            old_model: 1.0 - config.canary_pct,
            model: config.canary_pct,
        })

        # Monitor for canary_duration
        for hour in range(config.canary_hours):
            time.sleep(3600)
            metrics = self.compare_metrics(old_model, model)

            # Auto-rollback triggers:
            if metrics.latency_p99_regression > 5:
                self.rollback(endpoint, old_model, "Latency regression")
                return
            if metrics.prediction_drift > 0.1:
                self.rollback(endpoint, old_model, "Prediction drift")
                return
            if metrics.business_metric_regression > 0.02:
                self.rollback(endpoint, old_model, "Business metric drop")
                return

        # Phase 3: Gradual Rollout
        for pct in [10, 25, 50, 100]:
            endpoint.set_traffic_split({
                old_model: 1.0 - pct/100,
                model: pct/100,
            })
            time.sleep(1800)  # 30 min between steps
            if self.check_guardrails(endpoint) == FAIL:
                self.rollback(endpoint, old_model, "Guardrail failure")
                return

        # Phase 4: Cleanup
        # Keep old model warm for 24h (fast rollback)
        endpoint.keep_warm(old_model, duration="24h")
        return DeployResult(status="SUCCESS")`} />
          <div className="space-y-4">
            <Card accent="#059669">
              <Label color="#059669">Deployment Strategies Compared</Label>
              <div className="space-y-2">
                {[
                  { strat: "Blue-Green", desc: "Two identical environments. Switch traffic instantly. Fast rollback (switch back). Requires 2x resources during transition.", when: "Simple models, infrequent updates" },
                  { strat: "Canary ★", desc: "Gradual traffic shift with monitoring. 1% → 5% → 25% → 100%. Auto-rollback on regression. Standard for Google ML.", when: "Default for all production models" },
                  { strat: "Shadow (Dark Launch)", desc: "New model scores traffic in parallel but doesn't serve results. Compare predictions offline. Zero risk to users.", when: "Major model architecture changes" },
                  { strat: "Multi-Armed Bandit", desc: "Automatically allocate more traffic to the better-performing model. Optimizes exploration/exploitation.", when: "A/B testing competing models" },
                ].map((s,i) => (
                  <div key={i} className="rounded-lg border border-stone-200 p-2.5">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-[11px] font-bold text-stone-800">{s.strat}</span>
                      <span className="text-[9px] text-stone-400">{s.when}</span>
                    </div>
                    <p className="text-[10px] text-stone-500">{s.desc}</p>
                  </div>
                ))}
              </div>
            </Card>
            <Card className="bg-red-50/50 border-red-200">
              <Label color="#dc2626">Auto-Rollback Guardrails</Label>
              <div className="space-y-1.5 text-[11px] text-stone-600">
                <div><strong>Latency p99 increases &gt;5ms</strong> → model too complex or bad optimization</div>
                <div><strong>Prediction distribution shift &gt;10%</strong> → model learning different patterns</div>
                <div><strong>Business metric drops &gt;2%</strong> → CTR, revenue, engagement regression</div>
                <div><strong>Error rate increases &gt;0.1%</strong> → serving errors, feature failures</div>
                <div className="text-red-600 font-bold mt-2">All guardrails are automatic — no human approval needed for rollback.</div>
              </div>
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
      <Card accent="#7e22ce">
        <Label color="#7e22ce">Core Data Stores</Label>
        <CodeBlock code={`-- Model Registry (Spanner / Cloud SQL)
model_name, version -> {
  artifact_uri: string,           # GCS path to model files
  framework: enum(tf, pytorch, jax, xgboost, lightgbm),
  stage: enum(development, staging, canary, production, archived),
  metrics: {auc, loss, calibration, latency_ms_estimate},
  lineage: {data_path, feature_set, hyperparams, git_sha, job_id},
  serving_signature: {input_schema, output_schema},
  model_size_bytes: int,
  created_at, created_by, promoted_at, promoted_by,
}

-- Feature Store — Offline (BigQuery)
(entity_id, feature_name, feature_value, event_timestamp)
# Partitioned by event_timestamp for point-in-time queries
# "Give me user_123's features as of Jan 15, 2024"

-- Feature Store — Online (Redis cluster)
entity_type:entity_id:feature_name -> feature_value
# TTL-based expiration. Updated via streaming pipeline.
# Sub-millisecond reads. ~50M lookups/sec.

-- Training Job Store (Spanner)
job_id -> {
  model_name, config, status, start_time, end_time,
  resource_usage: {gpu_hours, peak_memory_gb},
  metrics_history: [{epoch, train_loss, eval_loss, auc}, ...],
  artifacts: [checkpoint_uris],
  logs_uri: string,
}

-- Serving Endpoint Store (Spanner)
endpoint_name -> {
  models: [{model_uri, traffic_pct, status}, ...],
  autoscale_config: {min, max, target_latency},
  current_qps, current_latency_p50, current_latency_p99,
}

-- Prediction Log (Kafka → BigQuery)
(timestamp, endpoint, model_version, request_features_hash,
 prediction, latency_ms, features_snapshot)
# Logged for: retraining data, monitoring, debugging
# features_snapshot: exact features used (prevents training-serving skew in analysis)`} />
      </Card>
    </div>
  );
}

function MonitoringSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-5">
        <Card accent="#0284c7">
          <Label color="#0284c7">Model Quality Metrics</Label>
          <div className="space-y-2.5">
            {[
              { metric: "Prediction Drift", target: "KL div < 0.05", why: "Is the model's output distribution shifting? Compare today's predictions to last week's. Large drift = model or data changed." },
              { metric: "Feature Drift", target: "PSI < 0.1 per feature", why: "Are input features changing? If user_avg_amount shifts, model may be stale. Population Stability Index per feature." },
              { metric: "Business Metric Correlation", target: "Within 2% of baseline", why: "Does model prediction still correlate with business outcome? CTR model predictions vs actual click rates." },
              { metric: "Label Drift", target: "Monitor, no fixed target", why: "Is the relationship between features and labels changing? If so, model needs retraining." },
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
        <Card accent="#ea580c">
          <Label color="#ea580c">Serving Health Metrics</Label>
          <div className="space-y-2.5">
            {[
              { metric: "Latency p50 / p99", target: "p50<5ms, p99<20ms", why: "Direct user experience impact. P99 matters more — tail latency affects 1% of requests." },
              { metric: "Throughput (QPS)", target: "Within autoscale range", why: "Current QPS vs capacity. Approaching max = scale up. Underutilized = scale down (save cost)." },
              { metric: "Error Rate", target: "< 0.01%", why: "Failed predictions. Feature store timeout, model error, OOM. Each error = degraded user experience." },
              { metric: "GPU/CPU Utilization", target: "60-80%", why: "Below 60% = over-provisioned (wasting money). Above 90% = risk of latency spikes under load." },
            ].map((m,i) => (
              <div key={i} className="border-b border-stone-100 pb-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold text-stone-700">{m.metric}</span>
                  <span className="text-[10px] font-mono text-orange-600">{m.target}</span>
                </div>
                <div className="text-[10px] text-stone-400 mt-0.5">{m.why}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Alerts & Actions</Label>
          <div className="space-y-2.5">
            {[
              { alert: "Prediction drift KL > 0.1", sev: "P1", action: "Investigate data pipeline. Possible feature store issue." },
              { alert: "Serving latency p99 > 50ms", sev: "P0", action: "Scale up. Check feature store latency. Model too complex?" },
              { alert: "Training pipeline failed", sev: "P1", action: "Auto-retry once. Then alert ML engineer. No deployment." },
              { alert: "Canary rollback triggered", sev: "P1", action: "Investigate regression. Was it data, model, or serving?" },
              { alert: "Feature store stale > 1 hour", sev: "P0", action: "Streaming pipeline issue. Models serving on stale features." },
            ].map((a,i) => (
              <div key={i} className="border-b border-stone-100 pb-2">
                <div className="flex items-center gap-2">
                  <Pill bg={a.sev==="P0"?"#fef2f2":"#fffbeb"} color={a.sev==="P0"?"#dc2626":"#d97706"}>{a.sev}</Pill>
                  <span className="text-[10px] text-stone-700">{a.alert}</span>
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

function ScalabilitySection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Training Cluster Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Priority-based scheduling</strong> — production retraining jobs get highest priority (preempt research jobs). Research jobs run on spare capacity. Ensures production models are always fresh.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Spot/preemptible instances</strong> — research and experimentation jobs run on spot instances (60-80% cheaper). Production training uses on-demand for reliability. Checkpointing every 30 min for spot job recovery.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Elastic scaling</strong> — cluster auto-scales based on job queue depth. Peak Monday (everyone starts experiments) vs quiet weekends. Right-sizing prevents over-provisioning.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">Multi-tenant resource quotas</strong> — each team gets a guaranteed base quota + burst capacity from shared pool. Prevents one team from monopolizing the cluster.</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Serving Scaling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Autoscaling on custom metrics</strong> — scale on p99 latency target, not just CPU utilization. If p99 approaches target, add replicas proactively. HPA with custom metrics via Prometheus/Monarch.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Model co-location</strong> — small models that share similar feature requirements can be co-located on the same server. Reduces feature fetch overhead and improves utilization.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Tiered serving</strong> — hot models (high QPS, low latency) on dedicated GPU/CPU pools. Warm models (moderate QPS) on shared pools with autoscaling. Cold models (rare requests) scale-to-zero with cold start on first request.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Multi-region serving</strong> — models replicated to all serving regions. Model updates propagated globally. Regional traffic routed to nearest replicas. Failover to backup region if primary degrades.</Point>
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
          { title: "Training-Serving Skew", sev: "CRITICAL", desc: "Features computed differently in training vs serving. Model trained on one distribution, served on another. Silently degrades accuracy by 5-20%. Extremely common and hard to detect — metrics look fine offline but business metrics drop online.", fix: "Feature store with dual materialization (write once, execute in batch + streaming). Log serving features and compare to training features periodically. Automated skew detection: sample serving requests, re-compute features in batch mode, compare.", icon: "🔴" },
          { title: "Silent Model Degradation", sev: "CRITICAL", desc: "Model accuracy slowly declines as the world changes (concept drift). Engagement patterns shift, new products launch, user behavior evolves. Without monitoring, the model serves increasingly stale predictions for weeks.", fix: "Continuous monitoring of prediction drift, feature drift, and business metric correlation. Automated retraining on schedule (daily/weekly). Drift thresholds that trigger retraining. Always compare against a simple baseline — if the model can't beat a heuristic, something is wrong.", icon: "🔴" },
          { title: "GPU Cluster Underutilization", sev: "HIGH", desc: "GPUs are expensive ($2-3/hr each). A 1000-GPU cluster at 50% utilization wastes $1M/month. Common causes: fragmentation (jobs need 8 GPUs but only 4 are free), poor scheduling, long queue times.", fix: "Gang scheduling for multi-GPU jobs (allocate all GPUs atomically). Preemptible priority tiers. Cluster utilization dashboard with real-time monitoring. Job profiling: identify and fix jobs that request 8 GPUs but only use 4 effectively.", icon: "🟡" },
          { title: "Bad Model Deployed to Production", sev: "HIGH", desc: "A model that passes offline validation but hurts business metrics online. Possible causes: evaluation data not representative, overfitting to eval set, feature interaction not captured in offline metrics.", fix: "Multi-stage deployment gates (offline → staging → canary → production). Canary deployment with automatic rollback on business metric regression. Shadow scoring before any live traffic. Keep previous model warm for instant rollback. Never deploy without canary period.", icon: "🟡" },
          { title: "Feature Store Outage", sev: "CRITICAL", desc: "If the online feature store is down, model serving returns errors or degrades to feature-less predictions. Feature store is a single point of failure for ALL models.", fix: "Multi-layer caching: local cache on serving nodes (stale but available), Redis replicas, fallback to pre-computed feature snapshots. Graceful degradation: serve with cached features (slightly stale) rather than failing. Feature store SLA: 99.99% availability, p99 < 5ms.", icon: "🔴" },
          { title: "Data Pipeline Corruption", sev: "HIGH", desc: "Upstream data pipeline produces bad data (null values, wrong schema, duplicated events). Model trained on corrupted data produces wrong predictions. May take days to detect via business metrics.", fix: "Data validation at every pipeline stage (TFDV/Great Expectations). Schema enforcement: reject data that doesn't match expected schema. Data quality dashboards with anomaly detection. Automatic pipeline halt on critical data quality issues.", icon: "🟡" },
        ].map((w,i) => (
          <Card key={i} accent="#dc2626">
            <div className="flex items-center gap-2 mb-2">
              <span>{w.icon}</span>
              <span className="text-[12px] font-bold text-stone-800">{w.title}</span>
              <Pill bg={w.sev==="CRITICAL"?"#fef2f2":"#fffbeb"} color={w.sev==="CRITICAL"?"#dc2626":"#d97706"}>{w.sev}</Pill>
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

function EnhancementsSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        {[
          { title: "AutoML / Hyperparameter Optimization", d: "Automated hyperparameter search (Vizier, Optuna). Bayesian optimization or population-based training to find optimal hyperparams.", effort: "Medium", detail: "Vizier (Google): service that manages hyperparameter trials across distributed training jobs. Integrates with training pipeline for zero-code HPO." },
          { title: "Experiment Tracking & Comparison", d: "Track every training run with full metadata: hyperparams, data version, metrics curves, feature importance. Compare experiments side-by-side.", effort: "Medium", detail: "MLflow, Weights & Biases, or Vertex AI Experiments. Essential for ML team productivity. 'What changed between v41 and v42?'" },
          { title: "LLM Serving Optimization", d: "Specialized serving for large language models: PagedAttention (vLLM), continuous batching, speculative decoding, KV-cache management.", effort: "Hard", detail: "LLMs require fundamentally different serving patterns: long context, autoregressive decoding, memory-bound (not compute-bound). Separate serving stack from traditional ML." },
          { title: "Edge/On-Device Serving", d: "Deploy models to mobile devices or edge servers. TensorFlow Lite, ONNX Runtime Mobile, Core ML. Requires model compression.", effort: "Hard", detail: "Challenges: limited memory, no GPU, battery constraints. Solutions: quantization to INT8, pruning, model distillation to tiny models." },
          { title: "Cost Attribution & Optimization", d: "Track GPU/CPU/storage costs per team, per model, per experiment. Identify expensive models that could be optimized.", effort: "Medium", detail: "Chargeback model: teams pay for resources consumed. Incentivizes efficiency. 'This model costs $50K/month to serve — can we distill it?'" },
          { title: "Federated Learning", d: "Train models across distributed data sources without centralizing data. Privacy-preserving: data stays on device, only gradients are shared.", effort: "Hard", detail: "Google uses for Gboard next-word prediction, Smart Reply. Challenges: non-IID data distribution, communication efficiency, privacy guarantees (differential privacy)." },
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
    { q:"What is training-serving skew and how do you prevent it?", a:"Training-serving skew is when features used for training differ from features used for serving — the #1 production ML bug. Example: training computes 'avg_purchase_amount' over all time, but serving computes it over the last 30 days. The model learned one distribution but sees another. Prevention: (1) Feature store with unified definitions: same code generates features for both training (batch) and serving (streaming). (2) Point-in-time correctness: training features computed AS OF the event timestamp, not current time. (3) Feature logging: log exact features used at serving time. Periodically compare logged serving features to what the training pipeline would compute for the same events. Any difference is skew. (4) Schema enforcement: training and serving must use the same feature schema version. (5) Integration tests: before deployment, run the model on a sample of logged serving features and compare to expected outputs.", tags:["architecture"] },
    { q:"How do you decide between retraining daily vs weekly vs on-demand?", a:"Depends on the rate of change in the data distribution: (1) Daily retraining: for models where the world changes fast — ads CTR (ad campaigns change daily), search ranking (new content every minute), fraud detection (attackers adapt). Cost: high (compute + pipeline). Benefit: model always reflects current patterns. (2) Weekly retraining: for models with moderate drift — recommendations, content moderation. The underlying patterns change but not hourly. (3) On-demand / triggered retraining: when monitoring detects drift beyond a threshold. More efficient than fixed schedules but requires robust monitoring. (4) Never retrain: for stable models like image classifiers, language models on fixed tasks. Retrain only when you have new labeled data or architecture improvements. The L6 answer: use monitoring-triggered retraining as the default. Daily schedule as a safety net (retrain at least weekly even if drift isn't detected). This balances freshness with compute cost.", tags:["operations"] },
    { q:"How do you handle feature store latency at serving time?", a:"Online feature store latency is on the critical serving path — every millisecond matters. Strategies: (1) Batch feature fetch: request all features for a prediction in one round-trip, not one-by-one. Reduces network overhead from N round-trips to 1. (2) Local caching: cache hot entity features on the serving node itself (LRU cache). Top 10% of users generate 50%+ of traffic. Cache hit rate: 30-50%. (3) Feature pre-computation: for features that don't change per-request (user demographics, account age), pre-compute and co-locate with the model. Only fetch dynamic features (real-time counters) from the online store. (4) Async feature fetch: start fetching features as soon as the request arrives, before any other processing. Overlap with other pre-processing. (5) Tiered storage: Redis (sub-ms) for hot features, Bigtable (1-5ms) for warm features, pre-computed snapshots for cold features. (6) Feature fallback: if a feature lookup times out after 3ms, use a default value rather than failing the prediction.", tags:["serving"] },
    { q:"How do you run A/B tests on ML models?", a:"ML A/B testing has unique challenges compared to product A/B tests: (1) Setup: deploy control (current model) and treatment (new model) behind a traffic splitter. Randomize at the user level (not request level) to avoid inconsistent experiences. (2) Metrics: primary metric (business KPI: CTR, revenue, engagement), guardrail metrics (latency, error rate, coverage), and model metrics (AUC, calibration). The business metric is what matters — a model with better AUC but worse CTR is a bad model. (3) Duration: ML A/B tests often need 1-2 weeks because some effects are delayed (recommendation diversity affects long-term engagement). (4) Interference: models in the same pipeline can interfere. If the ranking model changes, the CTR model sees different training data. Test one model change at a time. (5) Interleaving: for ranking models, interleave results from both models in the same result page. More sensitive than split testing with fewer samples needed.", tags:["evaluation"] },
    { q:"What's the difference between model serving on CPU vs GPU?", a:"CPU serving: (1) Predictable latency — no GPU kernel launch overhead. (2) Lower throughput per server but linear scaling (add more CPU servers). (3) Best for: tabular models (XGBoost, LightGBM), small neural nets, latency-sensitive endpoints. (4) Cost-effective for most production ML. GPU serving: (1) Much higher throughput via parallel computation. (2) Requires dynamic batching to amortize GPU overhead (one inference is wasteful; batch of 64 is efficient). (3) Higher latency variance (batch accumulation, memory transfer). (4) Best for: large embedding models, transformers, vision models, LLMs. (5) More expensive per server but fewer servers needed for high-throughput workloads. Decision framework: if the model is <100MB and latency < 10ms is required, use CPU. If the model is >1GB or needs to process images/sequences, use GPU. Many Google models serve on CPU — GBDT models for ads, small ranking models.", tags:["serving"] },
    { q:"How do you handle the cold start problem for new models on the platform?", a:"Cold start has two meanings: (1) New model / new team: no training data, no infrastructure setup. Platform solution: templates. Provide cookiecutter templates for common model types (tabular GBDT, deep ranking, text classifier). Each template includes: data pipeline config, feature store setup, training script, serving config, monitoring dashboard. A team can go from zero to production model in 1-2 days instead of weeks. (2) New serving endpoint: first request after scale-to-zero or new deployment. Model must be loaded into memory, features cached, JIT compilation warmed up. Solution: pre-warm endpoints before switching traffic. Load model + run warm-up requests (dummy predictions to fill caches) before receiving real traffic. Keep a minimum of 1 replica always running for latency-sensitive models.", tags:["platform"] },
    { q:"How do you manage 5,000 production models without a huge ML ops team?", a:"Automation and self-service are the only way: (1) Automated pipelines: every model has a pipeline config that defines: data source, feature set, training schedule, validation criteria, deployment strategy. Once configured, the pipeline runs without human intervention. Humans only review failures. (2) Standardized templates: 80% of models follow one of 5 patterns (tabular, ranking, embedding, classifier, regressor). Templates handle all infrastructure concerns. (3) Centralized monitoring: single dashboard showing all models' health — prediction drift, latency, business metric correlation. Alert only on anomalies. (4) Self-service platform: ML engineers own their models end-to-end. Platform team provides the tools, not the operations. 5 platform engineers support 500 ML teams. (5) Graduated autonomy: new models get stricter guardrails (auto-rollback, limited traffic). Mature models get wider autonomy (auto-deploy to 100%). This is Google's actual approach — TFX pipelines manage thousands of models with a small platform team.", tags:["operations"] },
    { q:"Should the platform support TensorFlow AND PyTorch AND JAX?", a:"Yes, but with a framework-agnostic abstraction layer. Reasons: (1) Different teams have different expertise and preferences. Forcing one framework kills adoption. (2) Different frameworks excel at different things: JAX for TPU/research, PyTorch for flexibility/debugging, TF for production serving (SavedModel ecosystem), XGBoost for tabular. (3) Implementation: the platform APIs (training submission, model registry, serving endpoint) are framework-agnostic. The model artifact is framework-specific but wrapped in a standard interface. Training: accepts any Docker container with a standard entry point. Serving: standardize on ONNX Runtime for cross-framework serving, or Triton (supports TF, PyTorch, ONNX, TensorRT). Registry: stores any artifact format with metadata. (4) Cost of multi-framework: more testing, more serving infrastructure variants, harder to optimize. But the adoption benefit outweighs the cost.", tags:["platform"] },
  ];
  return (
    <div className="space-y-2.5">
      <Card className="bg-indigo-50/50 border-indigo-200 mb-1">
        <p className="text-[12px] text-stone-500">Common follow-up questions Google L6 interviewers ask about ML infrastructure. Click to reveal a strong answer.</p>
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
  api: ApiSection, design: DesignSection, training: TrainingSection,
  distributed: DistributedSection, featurestore: FeatureStoreSection,
  modelreg: ModelRegistrySection, serving: ServingSection, canary: CanarySection,
  data: DataModelSection, monitoring: MonitoringSection,
  scalability: ScalabilitySection, watchouts: WatchoutsSection,
  enhancements: EnhancementsSection, followups: FollowupsSection,
};

export default function MLInfraSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">ML Training + Serving Infrastructure</h1>
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