import { useState, useEffect, lazy, Suspense } from "react";

/* ═══════════════════════════════════════════════════════════
   SD VAULT — System Design Interview Reference Portal
   Two sections: Traditional SD + ML System Design
   ═══════════════════════════════════════════════════════════ */

// ─── Traditional SD lazy imports ───
const RateLimiterSD = lazy(() => import("./visualizations/traditional/rate-limiter-sd.jsx"));
const UrlShortenerSD = lazy(() => import("./visualizations/traditional/url-shortener-sd.jsx"));
const NewsFeedSD = lazy(() => import("./visualizations/traditional/news-feed-sd.jsx"));
const ChatSystemSD = lazy(() => import("./visualizations/traditional/chat-system-sd.jsx"));
const NotificationSystemSD = lazy(() => import("./visualizations/traditional/notification-system-sd.jsx"));
const DistributedCacheSD = lazy(() => import("./visualizations/traditional/distributed-cache-sd.jsx"));
const CdnSD = lazy(() => import("./visualizations/traditional/cdn-sd.jsx"));
const MessageQueueSD = lazy(() => import("./visualizations/traditional/message-queue-sd.jsx"));
const JobSchedulerSD = lazy(() => import("./visualizations/traditional/job-scheduler-sd.jsx"));
const SearchAutocompleteSD = lazy(() => import("./visualizations/traditional/search-autocomplete-sd.jsx"));
const PaymentSystemSD = lazy(() => import("./visualizations/traditional/payment-system-sd.jsx"));
const WebCrawlerSD = lazy(() => import("./visualizations/traditional/web-crawler-sd.jsx"));
const GoogleDriveSD = lazy(() => import("./visualizations/traditional/google-drive-sd.jsx"));
const GoogleDocsSD = lazy(() => import("./visualizations/traditional/google-docs-sd.jsx"));
const InstagramSD = lazy(() => import("./visualizations/traditional/instagram-sd.jsx"));
const GoogleMapsSD = lazy(() => import("./visualizations/traditional/google-maps-sd.jsx"));

// ─── ML SD lazy imports ───
const SearchRankingSD = lazy(() => import("./visualizations/ml/search-ranking-sd.jsx"));
const YoutubeRecommendationsSD = lazy(() => import("./visualizations/ml/youtube-recommendations-sd.jsx"));
const AdsCtrPredictionSD = lazy(() => import("./visualizations/ml/ads-ctr-prediction-sd.jsx"));
const ContentModerationSD = lazy(() => import("./visualizations/ml/content-moderation-sd.jsx"));
const FraudAbuseDetectionSD = lazy(() => import("./visualizations/ml/fraud-abuse-detection-sd.jsx"));
const RagEnterpriseSearchSD = lazy(() => import("./visualizations/ml/rag-enterprise-search-sd.jsx"));
const MlTrainingServingSD = lazy(() => import("./visualizations/ml/ml-training-serving-sd.jsx"));
const NotificationFeedRankingSD = lazy(() => import("./visualizations/ml/notification-feed-ranking-sd.jsx"));

// ─── Topic registry ───
interface Topic {
  key: string;
  label: string;
  emoji: string;
  component: React.LazyExoticComponent<any>;
  tier: string;
}

const TRADITIONAL: Topic[] = [
  { key: "rate-limiter",        label: "Rate Limiter",           emoji: "🚦", component: RateLimiterSD,        tier: "Tier 1" },
  { key: "url-shortener",       label: "URL Shortener",          emoji: "🔗", component: UrlShortenerSD,       tier: "Tier 1" },
  { key: "news-feed",           label: "News Feed / Timeline",   emoji: "📰", component: NewsFeedSD,           tier: "Tier 1" },
  { key: "chat-system",         label: "Chat / Messaging",       emoji: "💬", component: ChatSystemSD,         tier: "Tier 1" },
  { key: "notification-system", label: "Notification System",    emoji: "🔔", component: NotificationSystemSD, tier: "Tier 1" },
  { key: "distributed-cache",   label: "Distributed Cache",      emoji: "⚡", component: DistributedCacheSD,   tier: "Tier 2" },
  { key: "cdn",                 label: "Content Delivery Network",emoji: "🌐", component: CdnSD,               tier: "Tier 2" },
  { key: "message-queue",       label: "Message Queue (Kafka)",  emoji: "📨", component: MessageQueueSD,       tier: "Tier 2" },
  { key: "search-autocomplete", label: "Search Autocomplete",    emoji: "🔍", component: SearchAutocompleteSD, tier: "Tier 2" },
  { key: "web-crawler",         label: "Web Crawler",            emoji: "🕷️", component: WebCrawlerSD,         tier: "Tier 2" },
  { key: "job-scheduler",       label: "Job Scheduler",          emoji: "⏰", component: JobSchedulerSD,       tier: "Tier 3" },
  { key: "payment-system",      label: "Payment System (Stripe)",emoji: "💳", component: PaymentSystemSD,      tier: "Tier 3" },
  { key: "google-drive",        label: "Google Drive / Dropbox", emoji: "📁", component: GoogleDriveSD,        tier: "Tier 3" },
  { key: "google-docs",         label: "Google Docs (Collab)",   emoji: "📝", component: GoogleDocsSD,         tier: "Tier 3" },
  { key: "instagram",           label: "Instagram / Social",     emoji: "📸", component: InstagramSD,          tier: "Tier 3" },
  { key: "google-maps",         label: "Google Maps (Collab)",   emoji: "🗺️", component: GoogleMapsSD,         tier: "Tier 3" },
];

const ML: Topic[] = [
  { key: "search-ranking",            label: "Search Ranking",          emoji: "🔎", component: SearchRankingSD,           tier: "Pattern 1" },
  { key: "youtube-recommendations",   label: "YouTube Recommendations", emoji: "▶️",  component: YoutubeRecommendationsSD,  tier: "Pattern 1" },
  { key: "ads-ctr-prediction",        label: "Ads Click Prediction",    emoji: "🎯", component: AdsCtrPredictionSD,        tier: "Pattern 1" },
  { key: "content-moderation",        label: "Content Moderation",      emoji: "🛡️", component: ContentModerationSD,       tier: "Pattern 4" },
  { key: "fraud-abuse-detection",     label: "Fraud / Abuse Detection", emoji: "🚨", component: FraudAbuseDetectionSD,     tier: "Pattern 4" },
  { key: "rag-enterprise-search",     label: "RAG / Enterprise Search", emoji: "🧠", component: RagEnterpriseSearchSD,     tier: "Pattern 5" },
  { key: "ml-training-serving",       label: "ML Training & Serving",   emoji: "⚙️",  component: MlTrainingServingSD,       tier: "Pattern 6" },
  { key: "notification-feed-ranking", label: "Feed Ranking",            emoji: "📊", component: NotificationFeedRankingSD, tier: "Pattern 2" },
];

const ALL_TOPICS = [...TRADITIONAL, ...ML];

// ─── Loading spinner ───
function LoadingSpinner() {
  return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: "#faf9f7" }}>
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-stone-300 border-t-indigo-500 rounded-full animate-spin" />
        <span className="text-xs text-stone-400 tracking-wide">Loading system design...</span>
      </div>
    </div>
  );
}

// ─── Tier badge colors ───
function tierColor(tier: string): string {
  if (tier.includes("1")) return "#059669";
  if (tier.includes("2")) return "#d97706";
  if (tier.includes("3")) return "#7c3aed";
  if (tier.includes("4")) return "#dc2626";
  if (tier.includes("5")) return "#2563eb";
  if (tier.includes("6")) return "#0891b2";
  return "#6b7280";
}

// ─── App ───
export default function App() {
  const [route, setRoute] = useState<string>(window.location.hash.slice(1) || "");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [section, setSection] = useState<"traditional" | "ml">("traditional");

  useEffect(() => {
    const handler = () => {
      const hash = window.location.hash.slice(1) || "";
      setRoute(hash);
      // Auto-switch section tab when navigating
      if (ML.some(t => t.key === hash)) setSection("ml");
      else if (TRADITIONAL.some(t => t.key === hash)) setSection("traditional");
    };
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, []);

  const activeTopic = ALL_TOPICS.find(t => t.key === route);

  if (activeTopic) {
    const Comp = activeTopic.component;
    return (
      <div className="flex h-screen overflow-hidden" style={{ background: "#faf9f7" }}>
        {/* Sidebar */}
        <aside
          className="h-screen flex flex-col border-r border-stone-200 bg-white transition-all duration-200 overflow-hidden shrink-0"
          style={{ width: sidebarOpen ? 260 : 0, minWidth: sidebarOpen ? 260 : 0 }}
        >
          {/* Logo */}
          <div
            className="flex items-center gap-2.5 px-4 py-3 border-b border-stone-100 cursor-pointer hover:bg-stone-50 transition-colors shrink-0"
            onClick={() => { window.location.hash = ""; setRoute(""); }}
          >
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
              <span className="text-white text-xs font-bold">SD</span>
            </div>
            <span className="font-semibold text-stone-800 text-sm tracking-tight">SD Vault</span>
          </div>

          {/* Section toggle */}
          <div className="flex border-b border-stone-100 shrink-0">
            {(["traditional", "ml"] as const).map(s => (
              <button
                key={s}
                onClick={() => setSection(s)}
                className="flex-1 py-2 text-[11px] font-semibold uppercase tracking-wider transition-colors"
                style={{
                  color: section === s ? "#4f46e5" : "#a8a29e",
                  borderBottom: section === s ? "2px solid #4f46e5" : "2px solid transparent",
                  background: section === s ? "#eef2ff" : "transparent",
                }}
              >
                {s === "traditional" ? "System Design" : "ML Design"}
              </button>
            ))}
          </div>

          {/* Topic list */}
          <div className="flex-1 overflow-y-auto py-1">
            {(section === "traditional" ? TRADITIONAL : ML).map(t => (
              <a
                key={t.key}
                href={`#${t.key}`}
                className="flex items-center gap-2.5 px-4 py-2 text-[12.5px] transition-colors hover:bg-stone-50"
                style={{
                  background: route === t.key ? "#eef2ff" : undefined,
                  color: route === t.key ? "#4338ca" : "#57534e",
                  fontWeight: route === t.key ? 600 : 400,
                  borderRight: route === t.key ? "3px solid #4f46e5" : "3px solid transparent",
                }}
              >
                <span className="text-sm shrink-0">{t.emoji}</span>
                <span className="truncate">{t.label}</span>
                <span
                  className="ml-auto text-[9px] font-bold px-1.5 py-0.5 rounded-full shrink-0"
                  style={{ background: tierColor(t.tier) + "18", color: tierColor(t.tier) }}
                >
                  {t.tier}
                </span>
              </a>
            ))}
          </div>
        </aside>

        {/* Content */}
        <div className="flex-1 overflow-y-auto relative">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="fixed top-3 z-50 w-8 h-8 rounded-lg bg-white border border-stone-200 shadow-sm flex items-center justify-center hover:bg-stone-50 transition-colors"
            style={{ left: sidebarOpen ? 268 : 8 }}
          >
            <span className="text-stone-500 text-xs">{sidebarOpen ? "◀" : "▶"}</span>
          </button>

          <Suspense fallback={<LoadingSpinner />}>
            <Comp />
          </Suspense>
        </div>
      </div>
    );
  }

  // ─── Landing Page ───
  return <LandingPage />;
}

function LandingPage() {
  return (
    <div className="min-h-screen" style={{ background: "#faf9f7", fontFamily: "'DM Sans', system-ui, sans-serif" }}>
      {/* Hero */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 opacity-[0.03]" style={{
          backgroundImage: "radial-gradient(circle at 1px 1px, #000 1px, transparent 0)",
          backgroundSize: "32px 32px",
        }} />
        <div className="max-w-5xl mx-auto px-6 pt-16 pb-12 relative">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-200">
              <span className="text-white text-lg font-bold">SD</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-stone-900 tracking-tight">SD Vault</h1>
              <p className="text-xs text-stone-400 tracking-wide">System Design Interview Reference</p>
            </div>
          </div>
          <p className="text-sm text-stone-500 max-w-2xl leading-relaxed">
            24 interactive system design guides — each with 17 sections covering concept, requirements, capacity estimation,
            API design, architecture evolution, algorithms, data model, scalability, availability, failure modes,
            service architecture, request flows, deployment, ops playbook, and follow-up questions.
          </p>

          {/* Stats */}
          <div className="flex gap-8 mt-8">
            {[
              { n: "16", label: "System Design", color: "#4f46e5" },
              { n: "8", label: "ML Design", color: "#7c3aed" },
              { n: "17", label: "Sections Each", color: "#059669" },
              { n: "408", label: "Total Sections", color: "#d97706" },
            ].map((s, i) => (
              <div key={i} className="flex flex-col">
                <span className="text-2xl font-bold" style={{ color: s.color }}>{s.n}</span>
                <span className="text-[10px] text-stone-400 tracking-wide uppercase">{s.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Traditional SD Section */}
      <div className="max-w-5xl mx-auto px-6 pb-6">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-8 h-8 rounded-lg bg-indigo-50 flex items-center justify-center">
            <span className="text-indigo-600 text-sm">🏗️</span>
          </div>
          <div>
            <h2 className="text-lg font-bold text-stone-800">System Design</h2>
            <p className="text-[11px] text-stone-400">Traditional infrastructure &amp; distributed systems</p>
          </div>
          <span className="ml-auto text-xs font-semibold text-indigo-600 bg-indigo-50 px-2.5 py-1 rounded-full">16 topics</span>
        </div>

        <div className="grid grid-cols-4 gap-3">
          {TRADITIONAL.map(t => (
            <a
              key={t.key}
              href={`#${t.key}`}
              className="group bg-white rounded-xl border border-stone-200 p-4 hover:border-indigo-300 hover:shadow-md hover:shadow-indigo-50 transition-all duration-200 cursor-pointer"
            >
              <div className="flex items-start justify-between mb-2">
                <span className="text-xl">{t.emoji}</span>
                <span
                  className="text-[9px] font-bold px-1.5 py-0.5 rounded-full"
                  style={{ background: tierColor(t.tier) + "18", color: tierColor(t.tier) }}
                >
                  {t.tier}
                </span>
              </div>
              <h3 className="text-[13px] font-semibold text-stone-800 group-hover:text-indigo-700 transition-colors leading-tight">
                {t.label}
              </h3>
              <div className="flex items-center gap-1 mt-2">
                <span className="text-[9px] text-stone-400">17 sections</span>
                <span className="text-stone-300 text-[9px]">→</span>
              </div>
            </a>
          ))}
        </div>
      </div>

      {/* ML SD Section */}
      <div className="max-w-5xl mx-auto px-6 pb-16 pt-4">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-8 h-8 rounded-lg bg-violet-50 flex items-center justify-center">
            <span className="text-violet-600 text-sm">🤖</span>
          </div>
          <div>
            <h2 className="text-lg font-bold text-stone-800">ML System Design</h2>
            <p className="text-[11px] text-stone-400">Machine learning pipelines, ranking, and inference</p>
          </div>
          <span className="ml-auto text-xs font-semibold text-violet-600 bg-violet-50 px-2.5 py-1 rounded-full">8 topics</span>
        </div>

        <div className="grid grid-cols-4 gap-3">
          {ML.map(t => (
            <a
              key={t.key}
              href={`#${t.key}`}
              className="group bg-white rounded-xl border border-stone-200 p-4 hover:border-violet-300 hover:shadow-md hover:shadow-violet-50 transition-all duration-200 cursor-pointer"
            >
              <div className="flex items-start justify-between mb-2">
                <span className="text-xl">{t.emoji}</span>
                <span
                  className="text-[9px] font-bold px-1.5 py-0.5 rounded-full"
                  style={{ background: tierColor(t.tier) + "18", color: tierColor(t.tier) }}
                >
                  {t.tier}
                </span>
              </div>
              <h3 className="text-[13px] font-semibold text-stone-800 group-hover:text-violet-700 transition-colors leading-tight">
                {t.label}
              </h3>
              <div className="flex items-center gap-1 mt-2">
                <span className="text-[9px] text-stone-400">17 sections</span>
                <span className="text-stone-300 text-[9px]">→</span>
              </div>
            </a>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-stone-200 py-6 text-center">
        <p className="text-[10px] text-stone-400">
          SD Vault — Staff+ System Design Interview Prep • 24 topics • 408 sections
        </p>
      </div>
    </div>
  );
}
