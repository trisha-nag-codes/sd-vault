# SD Vault — Interactive System Design Reference

24 interactive system design guides organized into two sections, each with 17 deep-dive sections.

## Sections

### System Design (16 topics)
Rate Limiter, URL Shortener, News Feed, Chat/Messaging, Notification System, Distributed Cache, CDN, Message Queue (Kafka), Search Autocomplete, Web Crawler, Job Scheduler, Payment System (Stripe), Google Drive, Google Docs, Instagram, Google Maps

### ML System Design (8 topics)
Search Ranking, YouTube Recommendations, Ads Click Prediction, Content Moderation, Fraud/Abuse Detection, RAG/Enterprise Search, ML Training & Serving, Feed Ranking

## Each Topic: 17 Sections

Concept → Requirements → Capacity → API Design → HLD → Algorithm Deep Dive → Data Model → Scalability → Availability → Observability → Failure Modes → Service Architecture → Request Flows → Deploy & Security → Ops Playbook → Enhancements → Follow-up Questions

## Getting Started

```bash
npm install
npm run dev
```

## Deploy to GitHub Pages

Pre-configured for GitHub Pages via GitHub Actions.

1. Push to a repo named `sd-vault`
2. **Settings → Pages → Source → GitHub Actions**
3. Push to `main` — auto-deploys to `https://yourname.github.io/sd-vault/`

> Different repo name? Update `base` in `vite.config.ts`. Custom domain? Set `base: "/"`.

## Tech Stack

React 18 + TypeScript + Vite + Tailwind CSS. Hash-based routing. Lazy-loaded chunks.
