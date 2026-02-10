import { useState, useRef, useEffect } from "react";

/* ═══════════════════════════════════════════════════════════
   TICKET BOOKING (Ticketmaster) — System Design Reference
   Pearl white theme · 17 sections (HLD + LLD)
   ═══════════════════════════════════════════════════════════ */

const SECTIONS = [
  { id: "concept",       label: "Concept",             icon: "💡", color: "#6366f1" },
  { id: "requirements",  label: "Requirements",         icon: "📋", color: "#0891b2" },
  { id: "capacity",      label: "Capacity Estimation",  icon: "🔢", color: "#7c3aed" },
  { id: "api",           label: "API Design",           icon: "🔌", color: "#2563eb" },
  { id: "design",        label: "High-Level Design",    icon: "🏗️", color: "#9333ea" },
  { id: "algorithm",     label: "Seat Booking Deep Dive",icon: "⚙️", color: "#c026d3" },
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
            <Label>What is a Ticket Booking System?</Label>
            <p className="text-[14px] text-stone-600 leading-relaxed">
              An online platform that lets users browse events, view venue seat maps, select seats, and purchase tickets — all under extreme concurrency. Think Ticketmaster, BookMyShow, or StubHub. The core challenge: thousands of users competing for the same limited seats at the exact same moment, with no double-selling and no lost revenue.
            </p>
            <p className="text-[13px] text-stone-500 leading-relaxed mt-3">
              Unlike a typical e-commerce system, ticket booking has unique constraints: fixed, finite inventory (seats), time-sensitive holds (temporary reservations), flash-sale traffic patterns (10,000× normal load in seconds), and strict consistency requirements (a seat sold to one person cannot also be sold to another).
            </p>
          </Card>
          <Card>
            <Label color="#0891b2">Why Is This Hard?</Label>
            <ul className="space-y-2.5">
              <Point icon="🎟️" color="#0891b2">Inventory contention — 50,000 users trying to book 500 seats simultaneously. Every seat is a unique, non-fungible resource with only one valid owner.</Point>
              <Point icon="⚡" color="#0891b2">Flash-sale thundering herd — ticket sales open at 10:00 AM sharp. Traffic spikes from 100 req/sec to 500,000 req/sec in under 5 seconds.</Point>
              <Point icon="🔒" color="#0891b2">No double-booking — the #1 invariant. If seat A7 is sold to User X, it MUST NOT be sold to User Y. Race conditions are the enemy.</Point>
              <Point icon="⏱️" color="#0891b2">Temporary holds — user selects seats and has 7 minutes to pay. Seat must be locked from others but released if payment fails or times out.</Point>
              <Point icon="💰" color="#0891b2">Payment coordination — 2-phase process: hold seat → process payment → confirm booking. Payment failures must release seats atomically.</Point>
              <Point icon="🤖" color="#0891b2">Bot prevention — scalpers use bots to grab tickets in bulk. Must distinguish legitimate users from automated scrapers without degrading real-user experience.</Point>
            </ul>
          </Card>
        </div>
        <div className="col-span-5 space-y-5">
          <Card accent="#d97706">
            <Label color="#d97706">Real-World Scale</Label>
            <div className="space-y-2.5">
              {[
                { co: "Ticketmaster", scale: "500M+ tickets/year", detail: "Live Nation. Largest globally." },
                { co: "BookMyShow", scale: "200M+ users, India market", detail: "Movies + concerts + sports" },
                { co: "StubHub", scale: "Secondary market leader", detail: "Resale + primary sales" },
                { co: "SeatGeek", scale: "MLS/NFL official partner", detail: "Interactive seat maps" },
                { co: "Eventbrite", scale: "4M+ events/year", detail: "Self-serve event creation" },
              ].map((e,i) => (
                <div key={i} className="flex items-center gap-2 text-[12px]">
                  <span className="font-mono font-bold text-stone-800 w-24 shrink-0">{e.co}</span>
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
              <text x={90} y={28} textAnchor="middle" fill="#dc2626" fontSize="10" fontWeight="700" fontFamily="monospace">Optimistic Locking</text>
              <text x={90} y={44} textAnchor="middle" fill="#dc262680" fontSize="8" fontFamily="monospace">⚠ High contention = retries</text>

              <rect x={190} y={10} width={160} height={45} rx={6} fill="#05966908" stroke="#059669" strokeWidth={1.5}/>
              <text x={270} y={28} textAnchor="middle" fill="#059669" fontSize="10" fontWeight="700" fontFamily="monospace">Pessimistic Lock ★</text>
              <text x={270} y={44} textAnchor="middle" fill="#05966980" fontSize="8" fontFamily="monospace">✔ Hold seat, then pay</text>

              <rect x={60} y={68} width={240} height={42} rx={6} fill="#9333ea08" stroke="#9333ea" strokeWidth={1.5}/>
              <text x={180} y={85} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Temporary Hold + 2-Phase Booking ★</text>
              <text x={180} y={100} textAnchor="middle" fill="#9333ea80" fontSize="8" fontFamily="monospace">Lock seat → pay → confirm OR release</text>
            </svg>
          </Card>
          <Card className="border-indigo-200 bg-indigo-50/50">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Interview Frequency</div>
                <div className="text-[11px] text-stone-500 mt-1">Google, Amazon, Uber, Stripe — classic concurrency problem</div>
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
            <div className="text-[12px] font-bold text-sky-700">Interview Tip — Clarify the Booking Model</div>
            <p className="text-[12px] text-stone-500 mt-0.5">"Design Ticketmaster" is broad. Clarify: assigned seats (concerts, sports) or general admission? Primary sales only, or resale too? How long do holds last? Is there a virtual queue (waiting room)? For a 45-min interview, focus on <strong>assigned-seat booking + temporary holds + payment flow + flash-sale concurrency</strong>. Resale and recommendations are follow-ups.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669">Browse events — search by artist, venue, date, genre, location</Point>
            <Point icon="2." color="#059669">View venue seat map — interactive map showing available/taken/held seats with pricing</Point>
            <Point icon="3." color="#059669">Select seats — pick specific seats and add to cart (temporary hold for 7 minutes)</Point>
            <Point icon="4." color="#059669">Checkout & payment — purchase held seats within the hold window</Point>
            <Point icon="5." color="#059669">Booking confirmation — e-ticket with QR code / barcode delivered via email and app</Point>
            <Point icon="6." color="#059669">View my tickets — list upcoming events and past orders</Point>
            <Point icon="7." color="#059669">Cancel / transfer — cancel for refund (if policy allows) or transfer ticket to another user</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Non-Functional Requirements</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#dc2626">Zero double-booking — a seat must never be sold to two different users</Point>
            <Point icon="2." color="#dc2626">Seat selection latency: &lt;300ms (seat map must feel instant)</Point>
            <Point icon="3." color="#dc2626">Handle flash-sale spikes: 0 → 500K req/sec in seconds</Point>
            <Point icon="4." color="#dc2626">Checkout completion rate: &gt;95% (minimize drop-offs due to timeouts/errors)</Point>
            <Point icon="5." color="#dc2626">High availability — 99.99% uptime during on-sale windows</Point>
            <Point icon="6." color="#dc2626">Fairness — first-come-first-served, no advantage for bots or scripted buyers</Point>
          </ul>
        </Card>
      </div>
      <Card>
        <Label color="#c026d3">Clarifying Questions to Ask</Label>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {[
            "Assigned seats or general admission (GA)?",
            "How long should a seat hold last? (5 min? 10 min?)",
            "Do we need a virtual queue / waiting room?",
            "Primary sales only, or resale marketplace too?",
            "Multiple ticket types? (VIP, GA, accessible)",
            "Payment providers? (Stripe, PayPal, Apple Pay)",
            "How many seats per venue? (5K arena? 80K stadium?)",
            "Bot prevention requirements? (CAPTCHA, device fingerprint)",
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
            <MathStep step="1" formula="Events on sale at any time = 50,000" result="50K" note="Across all venues, dates, genres." />
            <MathStep step="2" formula="DAU (browsing) = 10M users" result="10M" note="Searching events, checking prices, browsing seat maps." />
            <MathStep step="3" formula="Avg page views per user = 5" result="5" note="Search → event page → seat map → checkout." />
            <MathStep step="4" formula="Read requests/sec = 10M × 5 / 86,400" result="~580 req/s" note="Average. But distribution is wildly uneven." final />
            <MathStep step="5" formula="Hot event on-sale spike" result="500K req/s" note="Taylor Swift tickets: 14M users in queue. 500K concurrent." final />
            <MathStep step="6" formula="Booking transactions/sec (peak)" result="~10K tx/s" note="Of 500K viewers, ~2% attempt booking simultaneously." final />
          </div>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Step 2 — Storage Estimation</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Events per year = 500K" result="500K" note="Concerts, sports, theater, festivals." />
            <MathStep step="2" formula="Avg seats per event = 10,000" result="10K" note="Small venue: 500. Stadium: 80,000. Average ~10K." />
            <MathStep step="3" formula="Total seat inventory/year = 500K × 10K" result="5B seats" note="Each is a row in seat inventory table." />
            <MathStep step="4" formula="Per-seat row size = 200 bytes" result="200 B" note="seat_id, event_id, section, row, number, status, price, hold info." final />
            <MathStep step="5" formula="Seat inventory storage = 5B × 200B" result="~1 TB/year" note="Active inventory. Archivable after event date." final />
            <MathStep step="6" formula="Booking records/year = 500M" result="500M" note="Tickets sold. ~100 bytes each = 50 GB/year." />
          </div>
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Step 3 — The Flash-Sale Problem</Label>
          <div className="space-y-0">
            <MathStep step="1" formula="Users in queue for hot event" result="2M+" note="Taylor Swift Eras Tour: 14M in Ticketmaster queue." />
            <MathStep step="2" formula="Concurrent seat-map viewers" result="~500K" note="Users actively viewing available seats at once." />
            <MathStep step="3" formula="Seat hold attempts/sec" result="~50K" note="Users clicking 'hold seat' simultaneously." />
            <MathStep step="4" formula="Contention per popular seat" result="100:1" note="100 users trying to hold the same seat at the same moment." final />
          </div>
        </Card>
        <Card>
          <Label color="#dc2626">Key Numbers to Remember</Label>
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "Avg Read QPS", val: "~580/s", sub: "Normal browsing" },
              { label: "Peak Spike", val: "500K/s", sub: "Hot on-sale event" },
              { label: "Peak Bookings", val: "~10K tx/s", sub: "Concurrent purchases" },
              { label: "Seat Rows/yr", val: "~5B", sub: "Inventory records" },
              { label: "Hold Duration", val: "7 min", sub: "Temp reservation" },
              { label: "Contention Ratio", val: "100:1", sub: "Users per hot seat" },
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
          <Label color="#2563eb">Core Booking APIs</Label>
          <CodeBlock code={`# GET /v1/events/:id/seats
# View available seats for an event (seat map)
# Query: ?section=A&price_min=50&price_max=200
# Returns: {seats: [{seat_id, section, row, number,
#           price, status, view_quality}], held_count}
# Status: available | held | sold

# POST /v1/events/:id/hold
# Temporarily hold selected seats (start checkout timer)
{
  "seat_ids": ["seat_A7", "seat_A8"],
  "user_id": "user_42",
  "hold_duration_sec": 420      # 7 minutes
}
# Returns: {hold_id, expires_at, total_price}
# 409 if any seat already held/sold

# POST /v1/bookings
# Complete purchase (within hold window)
{
  "hold_id": "hold_abc123",
  "payment_method_id": "pm_stripe_xyz",
  "user_id": "user_42"
}
# Returns: {booking_id, tickets: [{ticket_id, qr_code}],
#           receipt_url}
# 410 if hold expired

# DELETE /v1/holds/:hold_id
# Release held seats (user cancels before paying)
# Returns: {released_seats: ["seat_A7","seat_A8"]}

# GET /v1/users/:id/bookings
# List user's tickets (upcoming + past)
# Returns: {bookings: [{booking_id, event, seats,
#           status, ticket_urls}]}`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Supporting APIs</Label>
          <CodeBlock code={`# GET /v1/events?q=taylor+swift&city=NYC&date=2026-03
# Search events with filters
# Returns: {events: [{id, name, venue, date, price_range,
#           availability_pct}], total, next_cursor}

# GET /v1/events/:id
# Event detail page
# Returns: {event_id, name, artist, venue, date,
#           description, seat_map_url, price_tiers,
#           total_seats, available_seats, on_sale_at}

# POST /v1/events/:id/waitlist
# Join waitlist when sold out
{ "user_id": "user_42", "max_price": 200,
  "seat_preference": "any" }

# POST /v1/bookings/:id/cancel
# Cancel booking (refund per policy)
{ "reason": "can_no_longer_attend" }
# Returns: {refund_amount, refund_status}

# POST /v1/bookings/:id/transfer
# Transfer ticket to another user
{ "recipient_email": "friend@example.com" }
# Returns: {transfer_id, new_ticket_id}

# GET /v1/events/:id/queue-status
# Virtual queue position (for flash sales)
# Returns: {position: 4521, estimated_wait_min: 12,
#           ahead_of_you: 4520, total_in_queue: 50000}`} />
          <div className="mt-3 space-y-2">
            {[
              { q: "Why a separate hold endpoint?", a: "The hold creates a temporary reservation — a pessimistic lock on those seats. It starts a timer (7 min). If payment doesn't complete in time, hold auto-releases. This prevents seats being locked forever by abandoned carts." },
              { q: "Why not just book directly?", a: "Payment processing takes 2-10 seconds and can fail. If we locked the seat only during payment, a 10-second window creates race conditions. The hold-then-pay pattern gives users time to enter payment details while guaranteeing their seats." },
              { q: "Why 409 Conflict for held seats?", a: "If another user already holds or bought the seat, return 409 immediately. Client can suggest nearby available seats. Don't make the user wait — fail fast with alternatives." },
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
        <svg viewBox="0 0 720 340" className="w-full">
          {/* Clients */}
          <rect x={10} y={100} width={70} height={34} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={45} y={120} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Web App</text>
          <rect x={10} y={150} width={70} height={34} rx={6} fill="#2563eb10" stroke="#2563eb" strokeWidth={1.5}/>
          <text x={45} y={170} textAnchor="middle" fill="#2563eb" fontSize="9" fontWeight="600" fontFamily="monospace">Mobile App</text>

          {/* CDN / Queue */}
          <rect x={110} y={85} width={65} height={34} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={142} y={106} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">CDN</text>
          <rect x={110} y={135} width={65} height={34} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={142} y={156} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Queue</text>
          <text x={142} y={166} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">waiting room</text>

          {/* API Gateway */}
          <rect x={205} y={110} width={70} height={46} rx={6} fill="#6366f110" stroke="#6366f1" strokeWidth={1.5}/>
          <text x={240} y={130} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">API</text>
          <text x={240} y={142} textAnchor="middle" fill="#6366f1" fontSize="9" fontWeight="600" fontFamily="monospace">Gateway</text>

          {/* Services */}
          <rect x={310} y={60} width={90} height={34} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={355} y={81} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Event Service</text>

          <rect x={310} y={110} width={90} height={34} rx={6} fill="#c026d310" stroke="#c026d3" strokeWidth={1.5}/>
          <text x={355} y={131} textAnchor="middle" fill="#c026d3" fontSize="9" fontWeight="600" fontFamily="monospace">Booking Svc</text>

          <rect x={310} y={160} width={90} height={34} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={355} y={181} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Inventory Svc</text>

          <rect x={310} y={210} width={90} height={34} rx={6} fill="#0891b210" stroke="#0891b2" strokeWidth={1.5}/>
          <text x={355} y={231} textAnchor="middle" fill="#0891b2" fontSize="9" fontWeight="600" fontFamily="monospace">Payment Svc</text>

          {/* Data stores */}
          <rect x={460} y={60} width={80} height={34} rx={6} fill="#78716c10" stroke="#78716c" strokeWidth={1.5}/>
          <text x={500} y={81} textAnchor="middle" fill="#78716c" fontSize="9" fontWeight="600" fontFamily="monospace">Event DB</text>

          <rect x={460} y={110} width={80} height={34} rx={6} fill="#ea580c10" stroke="#ea580c" strokeWidth={1.5}/>
          <text x={500} y={127} textAnchor="middle" fill="#ea580c" fontSize="9" fontWeight="600" fontFamily="monospace">Seat</text>
          <text x={500} y={139} textAnchor="middle" fill="#ea580c" fontSize="9" fontWeight="600" fontFamily="monospace">Inventory</text>

          <rect x={460} y={160} width={80} height={34} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={500} y={177} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Redis</text>
          <text x={500} y={189} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">holds + locks</text>

          <rect x={460} y={210} width={80} height={34} rx={6} fill="#9333ea10" stroke="#9333ea" strokeWidth={1.5}/>
          <text x={500} y={231} textAnchor="middle" fill="#9333ea" fontSize="9" fontWeight="600" fontFamily="monospace">Booking DB</text>

          {/* External */}
          <rect x={600} y={110} width={90} height={34} rx={6} fill="#05966910" stroke="#059669" strokeWidth={1.5}/>
          <text x={645} y={131} textAnchor="middle" fill="#059669" fontSize="9" fontWeight="600" fontFamily="monospace">Stripe/PayPal</text>

          <rect x={600} y={160} width={90} height={34} rx={6} fill="#d9770610" stroke="#d97706" strokeWidth={1.5}/>
          <text x={645} y={177} textAnchor="middle" fill="#d97706" fontSize="9" fontWeight="600" fontFamily="monospace">Notification</text>
          <text x={645} y={189} textAnchor="middle" fill="#d9770680" fontSize="7" fontFamily="monospace">email + push</text>

          <rect x={600} y={210} width={90} height={34} rx={6} fill="#dc262610" stroke="#dc2626" strokeWidth={1.5}/>
          <text x={645} y={227} textAnchor="middle" fill="#dc2626" fontSize="9" fontWeight="600" fontFamily="monospace">Search (ES)</text>

          {/* Arrows */}
          <defs><marker id="ah-hld" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#94a3b8"/></marker></defs>
          <line x1={80} y1={117} x2={110} y2={102} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={80} y1={167} x2={110} y2={152} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={175} y1={102} x2={205} y2={125} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={175} y1={152} x2={205} y2={138} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={275} y1={127} x2={310} y2={77} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={275} y1={133} x2={310} y2={127} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={275} y1={138} x2={310} y2={177} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={275} y1={143} x2={310} y2={227} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={400} y1={77} x2={460} y2={77} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={400} y1={127} x2={460} y2={127} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={400} y1={177} x2={460} y2={177} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={400} y1={227} x2={460} y2={227} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={540} y1={127} x2={600} y2={127} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={540} y1={177} x2={600} y2={177} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>
          <line x1={540} y1={77} x2={600} y2={227} stroke="#94a3b8" strokeWidth={1.2} markerEnd="url(#ah-hld)"/>

          {/* Flow labels */}
          <rect x={10} y={270} width={700} height={60} rx={6} fill="#faf9f7" stroke="#e7e5e4" strokeWidth={1}/>
          <text x={20} y={287} fill="#059669" fontSize="8" fontWeight="600" fontFamily="monospace">Hold flow: Client → API GW → Inventory Svc → Redis lock (SET NX) + DB status update → return hold_id + timer</text>
          <text x={20} y={302} fill="#c026d3" fontSize="8" fontWeight="600" fontFamily="monospace">Book flow: Client → Booking Svc → verify hold → Payment Svc → Stripe charge → confirm seat → issue ticket</text>
          <text x={20} y={317} fill="#d97706" fontSize="8" fontWeight="600" fontFamily="monospace">Flash sale: Client → Waiting Room Queue → rate-limited admission → API GW → normal flow. Prevents thundering herd.</text>
        </svg>
      </Card>
      <Card>
        <Label color="#c026d3">Key Architecture Decisions</Label>
        <div className="grid grid-cols-3 gap-4">
          {[
            { q: "Why a virtual queue / waiting room?", a: "During flash sales, millions hit the site simultaneously. Without a queue, all requests hit the Inventory Service → DB melts. The waiting room absorbs the spike: users get a queue position, admitted at a controlled rate (e.g., 5,000/sec). Fair ordering + system protection." },
            { q: "Why Redis for seat holds?", a: "Seat holds are short-lived (7 min) and high-contention. Redis SET NX (set-if-not-exists) is atomic and sub-millisecond. Perfect for distributed locking. SET seat:A7 hold_abc123 EX 420 — atomic hold with auto-expiry. No need for manual cleanup." },
            { q: "Why separate Inventory and Booking services?", a: "Inventory handles real-time availability (hot path — sub-100ms). Booking handles the transactional payment flow (slower, 2-10s). Separating them isolates the fast read path from the slow write path. Inventory can be cached aggressively; Booking requires ACID guarantees." },
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
  const [sel, setSel] = useState("hold_pattern");
  const topics = {
    hold_pattern: { name: "Temporary Hold Pattern", cx: "The Core Booking Mechanism",
      desc: "The heart of ticket booking: a pessimistic lock that temporarily reserves seats while the user completes payment. This 2-phase approach prevents double-booking without requiring instant payment.",
      code: `# Phase 1: HOLD — Reserve seats atomically
def hold_seats(event_id, seat_ids, user_id, ttl=420):
    # Step 1: Acquire distributed lock per seat using Redis
    hold_id = generate_hold_id()
    locked_seats = []

    for seat_id in seat_ids:
        key = f"hold:{event_id}:{seat_id}"
        # SET NX = set only if not exists (atomic!)
        acquired = redis.set(key, hold_id, nx=True, ex=ttl)
        if not acquired:
            # Seat already held by someone else
            release_all(locked_seats)  # Rollback
            return Error(409, "Seat unavailable", seat_id)
        locked_seats.append(seat_id)

    # Step 2: Update DB (async or sync)
    db.update_seats(seat_ids, status="held",
                    hold_id=hold_id, held_by=user_id,
                    expires_at=now() + ttl)

    # Step 3: Schedule auto-release
    delay_queue.enqueue(release_hold, hold_id, delay=ttl)

    return {hold_id, expires_at: now() + ttl}

# Phase 2: BOOK — Convert hold to confirmed booking
def complete_booking(hold_id, payment_method_id):
    hold = db.get_hold(hold_id)
    if hold.expired():
        return Error(410, "Hold expired")

    # Step 1: Charge payment
    charge = payment.charge(hold.total, payment_method_id)
    if charge.failed():
        release_hold(hold_id)
        return Error(402, "Payment failed")

    # Step 2: Confirm seats (within transaction)
    with db.transaction():
        db.update_seats(hold.seat_ids, status="sold",
                        booking_id=booking_id)
        booking = db.create_booking(hold, charge)
        redis.delete(*[f"hold:{hold.event_id}:{s}"
                       for s in hold.seat_ids])

    # Step 3: Issue tickets
    tickets = generate_tickets(booking)
    notify_user(booking, tickets)
    return {booking_id, tickets}`,
      points: [
        "Redis SET NX (set-if-not-exists) is the atomic primitive. If two users try to hold the same seat at the exact same microsecond, only one succeeds. The other gets 409 immediately.",
        "Hold TTL (7 minutes) is set on the Redis key itself (EX parameter). If user doesn't pay, Redis auto-deletes the key — seat becomes available again. No background job needed for normal expiry.",
        "Backup: a delayed queue job fires at TTL to clean up DB state (update seat status back to 'available'). Belt and suspenders — handles cases where Redis expiry and DB state diverge.",
        "Multi-seat atomicity: if holding 3 seats and the 3rd fails, MUST release the first 2. This is the 'all-or-nothing' guarantee. Loop with rollback on failure.",
        "Payment failure releases the hold immediately — don't make the user wait for TTL expiry. Other users can grab the seats right away.",
      ],
    },
    contention: { name: "Handling High Contention", cx: "When 1000 Users Want 1 Seat",
      desc: "During flash sales, hundreds of users click on the same popular seat at the same instant. Only one can win. The system must handle this gracefully without overloading the DB or frustrating users.",
      code: `# Problem: 1000 concurrent SET NX on the same key
# Only 1 wins. The other 999 need to:
#   1. Find out immediately (fail fast)
#   2. Get suggested alternatives
#   3. Not retry the same seat

# Strategy 1: Redis-first with fast rejection
def try_hold_with_suggestions(event_id, seat_id, user_id):
    key = f"hold:{event_id}:{seat_id}"
    acquired = redis.set(key, user_id, nx=True, ex=420)

    if acquired:
        return Success(seat_id)

    # Fast path: seat taken, suggest nearby seats
    nearby = get_nearby_available(event_id, seat_id, limit=5)
    return Conflict(seat_id, suggestions=nearby)

# Strategy 2: Batch availability with optimistic UI
# Client loads seat map → cached snapshot (1s stale)
# User clicks seat → real-time check + hold attempt
# If failed → animate seat turning red + show alternatives

# Strategy 3: "Best Available" for general admission
def hold_best_available(event_id, section, count, user_id):
    # Don't let user pick individual seats
    # System picks best available from section
    # Uses Redis sorted set: score = seat quality
    seats = redis.zpopmin(f"avail:{event_id}:{section}",
                          count=count)
    if len(seats) < count:
        return Error(409, "Not enough seats")
    # Hold all selected seats
    return hold_seats(event_id, seats, user_id)

# Strategy 4: Virtual queue for extremely hot events
# Decouple "arrival" from "seat selection"
# Users enter queue → admitted N at a time
# Admission rate = system capacity (e.g., 5000/sec)
# Queue position = arrival order (fair!)
def enter_queue(event_id, user_id):
    position = redis.incr(f"queue:{event_id}")
    return {position, est_wait: position / ADMISSION_RATE}`,
      points: [
        "Redis SET NX handles contention at the lock layer — only one thread/request wins per key. All losers fail instantly (sub-millisecond). No DB contention at all.",
        "Suggest alternatives immediately. Don't just say '409 seat taken'. Return 5 nearby available seats so the user can try again without loading the full seat map.",
        "'Best available' mode bypasses individual seat selection entirely. System picks the best remaining seats in a section. Reduces contention because users aren't all clicking the same seat.",
        "Virtual queue is the nuclear option for mega-events. Users wait in a fair queue; the system admits them at a controlled rate. This converts a thundering herd into a steady stream.",
        "Seat map freshness: serve a cached snapshot (1-2s stale) for display. On click, do a real-time check. Slightly stale map is acceptable — the hold attempt is the source of truth.",
      ],
    },
    payment: { name: "Payment Orchestration", cx: "2-Phase Commit with External Systems",
      desc: "The payment flow coordinates between your system (hold → confirm) and an external payment provider (Stripe/PayPal). This is a distributed transaction across system boundaries — one of the hardest patterns in system design.",
      code: `# The challenge: coordinate seat + payment atomically
# Can't use a DB transaction across Stripe + your DB
# Solution: Saga pattern with compensating actions

# State machine for a booking:
# HOLD → PAYMENT_PENDING → PAYMENT_SUCCESS → CONFIRMED
# HOLD → PAYMENT_PENDING → PAYMENT_FAILED → RELEASED
# HOLD → EXPIRED → RELEASED

def booking_saga(hold_id, payment_method_id):
    # Step 1: Verify hold is still valid
    hold = db.get_hold(hold_id)
    if hold.status != "ACTIVE" or hold.expired():
        return Error("Hold expired or invalid")

    # Step 2: Transition to PAYMENT_PENDING
    db.update_hold(hold_id, status="PAYMENT_PENDING")

    # Step 3: Create payment intent (Stripe)
    try:
        intent = stripe.create_payment_intent(
            amount=hold.total_cents,
            currency="usd",
            payment_method=payment_method_id,
            idempotency_key=hold_id,     # Prevents double-charge
            metadata={"hold_id": hold_id}
        )
        charge_result = stripe.confirm(intent.id)
    except PaymentError as e:
        # COMPENSATE: release seats
        release_hold(hold_id)
        return Error(402, "Payment failed", e.message)

    # Step 4: Confirm booking (idempotent)
    with db.transaction():
        db.update_seats(hold.seat_ids, status="sold")
        booking = db.create_booking(hold, charge_result)
        db.delete_hold(hold_id)

    # Step 5: Async — generate tickets, send email
    queue.enqueue(generate_and_send_tickets, booking.id)

    return Success(booking)

# Idempotency key = hold_id
# If network fails after Stripe charges but before DB update:
#   - User retries → same idempotency_key → Stripe returns
#     the same charge (no double-charge)
#   - Our code sees charge succeeded → confirms booking`,
      points: [
        "Idempotency key on the payment intent prevents double-charging. If the network fails after Stripe charges but before your DB confirms, the retry uses the same key — Stripe returns the existing charge.",
        "Saga pattern with compensating actions: if payment fails, the compensating action releases the seat hold. There's no 'rollback' across Stripe and your DB — you compensate.",
        "State machine (HOLD → PENDING → CONFIRMED/RELEASED) provides clear audit trail and recovery. On server restart, check pending bookings: verify with Stripe if charge went through.",
        "Stripe webhook as backup: even if your server crashes after payment, Stripe sends a webhook (payment_intent.succeeded). Webhook handler confirms the booking. Eventual consistency.",
        "Never charge and then check the hold. Always verify hold first, then charge. If you charge first and the hold expired, you've taken money for seats that were given to someone else.",
      ],
    },
    queue: { name: "Virtual Queue / Waiting Room", cx: "Flash-Sale Protection",
      desc: "When millions of users arrive simultaneously for a hot event, the virtual queue absorbs the spike and admits users at a controlled rate. This prevents system overload while maintaining fairness.",
      code: `# Virtual Queue Architecture
# 1. User arrives at event page → redirected to queue
# 2. Queue assigns position (arrival order)
# 3. Users admitted at controlled rate (e.g., 5000/sec)
# 4. Admitted users get a time-limited access token
# 5. Token required for all booking APIs

# Queue implementation:
class VirtualQueue:
    def __init__(self, event_id, admission_rate=5000):
        self.event_id = event_id
        self.admission_rate = admission_rate  # users/sec

    def enqueue(self, user_id):
        # Atomic increment → fair position
        position = redis.incr(f"queue:{self.event_id}:counter")
        redis.zadd(f"queue:{self.event_id}:waiting",
                   {user_id: position})
        est_wait = position / self.admission_rate
        return {position, est_wait_sec: est_wait}

    def admit_batch(self):
        # Called by scheduler every 1 second
        # Pop next N users from the waiting set
        batch = redis.zpopmin(
            f"queue:{self.event_id}:waiting",
            count=self.admission_rate
        )
        for user_id, _ in batch:
            token = generate_access_token(user_id,
                ttl=600)  # 10 min to complete purchase
            redis.set(f"admitted:{self.event_id}:{user_id}",
                      token, ex=600)
            notify_user(user_id, "You're in! Start shopping.")
        return len(batch)

    def verify_admission(self, user_id, token):
        stored = redis.get(
            f"admitted:{self.event_id}:{user_id}")
        return stored == token

# API Gateway middleware:
def queue_middleware(request):
    event = get_event(request.event_id)
    if event.requires_queue:
        token = request.headers.get("X-Queue-Token")
        if not queue.verify_admission(request.user_id, token):
            return Redirect("/queue/" + event.id)
    return proceed(request)`,
      points: [
        "Redis INCR for position assignment is atomic — no two users get the same position. Perfectly fair: arrival order determines queue position.",
        "Admission rate is tunable: 5,000/sec is typical. Can be adjusted based on real-time system health (if DB is struggling, reduce admission rate).",
        "Access token after admission: time-limited (10 min). User must complete their purchase within that window. Prevents admitted users from sitting idle while others wait.",
        "Queue page is static (served from CDN) with a polling endpoint for position updates. The queue page itself has near-zero backend cost — critical when millions are waiting.",
        "Ticketmaster's 'Smart Queue' uses this exact pattern. The Eras Tour had 14M users in queue — no system can handle 14M concurrent booking attempts. Queue converts it to 5K/sec steady flow.",
      ],
    },
  };
  const t = topics[sel];
  return (
    <div className="space-y-5">
      <Card accent="#c026d3">
        <Label color="#c026d3">Seat Locking Strategy — Comparison</Label>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] border-collapse">
            <thead><tr className="border-b-2 border-stone-200">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Strategy</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">How It Works</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Pros</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Cons</th>
              <th className="text-center px-3 py-2 font-semibold text-stone-500">Verdict</th>
            </tr></thead>
            <tbody>
              {[
                { n:"Optimistic (version check)", how:"Read seat, pay, write if version unchanged", pros:"No locks, high throughput for low contention", cons:"High contention = constant retries, user frustration", v:"⚠️" },
                { n:"Pessimistic (DB row lock)", how:"SELECT FOR UPDATE on seat row", pros:"Strong guarantee, simple", cons:"DB lock held during payment (seconds) → deadlocks at scale", v:"❌" },
                { n:"Redis Temp Hold ★", how:"SET NX with TTL, then pay, then confirm in DB", pros:"Sub-ms atomic lock, auto-expiry, no DB contention", cons:"Redis-DB consistency gap (solvable with WAL)", v:"✔", hl:true },
                { n:"Distributed Lock (Redlock)", how:"Acquire lock on majority of Redis nodes", pros:"Fault-tolerant across Redis failures", cons:"Complexity, slower than single-node SET NX", v:"⚠️" },
              ].map((r,i) => (
                <tr key={i} className={r.hl ? "bg-purple-50" : i%2 ? "bg-stone-50/50" : ""}>
                  <td className={`px-3 py-2.5 font-mono ${r.hl?"text-purple-700 font-bold":"text-stone-600"}`}>{r.n}</td>
                  <td className="text-center px-3 py-2.5 text-stone-500">{r.how}</td>
                  <td className="text-center px-3 py-2.5 text-emerald-600 text-[10px]">{r.pros}</td>
                  <td className="text-center px-3 py-2.5 text-red-500 text-[10px]">{r.cons}</td>
                  <td className="text-center px-3 py-2.5 text-lg">{r.v}</td>
                </tr>
              ))}
            </tbody>
          </table>
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
          <CodeBlock code={`-- Events (sharded by event_id)
CREATE TABLE events (
  event_id        BIGINT PRIMARY KEY,
  name            VARCHAR(500) NOT NULL,
  artist_id       BIGINT,
  venue_id        BIGINT NOT NULL,
  event_date      TIMESTAMP NOT NULL,
  on_sale_at      TIMESTAMP,            -- When tickets go live
  status          ENUM('draft','on_sale','sold_out','completed'),
  total_seats     INT NOT NULL,
  available_seats INT NOT NULL,          -- Denormalized counter
  created_at      TIMESTAMP
);

-- Seats / Inventory (sharded by event_id)
-- One row per physical seat per event
CREATE TABLE seats (
  seat_id         BIGINT PRIMARY KEY,    -- Snowflake ID
  event_id        BIGINT NOT NULL,
  section         VARCHAR(50),           -- "Floor", "Balcony A"
  row_name        VARCHAR(10),           -- "A", "B", "AA"
  seat_number     INT,
  price_cents     INT NOT NULL,
  tier            ENUM('vip','premium','standard','ga'),
  status          ENUM('available','held','sold','blocked'),
  hold_id         BIGINT,                -- FK if currently held
  booking_id      BIGINT,                -- FK if sold
  INDEX idx_event_status (event_id, status),
  INDEX idx_event_section (event_id, section, status)
);
-- Shard by event_id: all seats for one event on same shard

-- Holds (temporary reservations)
CREATE TABLE holds (
  hold_id         BIGINT PRIMARY KEY,
  event_id        BIGINT NOT NULL,
  user_id         BIGINT NOT NULL,
  seat_ids        JSON NOT NULL,         -- ["seat_A7","seat_A8"]
  total_cents     INT NOT NULL,
  status          ENUM('active','converted','expired','released'),
  created_at      TIMESTAMP,
  expires_at      TIMESTAMP NOT NULL,
  INDEX idx_user (user_id),
  INDEX idx_expires (expires_at, status)
);

-- Bookings (confirmed purchases)
CREATE TABLE bookings (
  booking_id      BIGINT PRIMARY KEY,
  event_id        BIGINT NOT NULL,
  user_id         BIGINT NOT NULL,
  hold_id         BIGINT,
  seat_ids        JSON NOT NULL,
  total_cents     INT NOT NULL,
  payment_id      VARCHAR(100),          -- Stripe charge ID
  status          ENUM('confirmed','cancelled','refunded'),
  created_at      TIMESTAMP,
  INDEX idx_user (user_id),
  INDEX idx_event (event_id)
);

-- Redis: Seat Locks (ephemeral)
-- Key: hold:{event_id}:{seat_id}
-- Value: hold_id
-- TTL: 420 seconds (7 minutes)`} />
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Why These Design Choices?</Label>
          <div className="space-y-3">
            {[
              { q: "Shard seats by event_id?", a: "All seats for one event on the same shard. 'Show available seats for event X' is a single-shard query. This is THE hot query — must be fast. No cross-shard joins needed." },
              { q: "Denormalized available_seats counter?", a: "Counting available seats by scanning the seats table is O(n) and expensive during flash sales. A denormalized counter on the events table gives O(1) availability checks. Updated atomically with seat status changes." },
              { q: "Why both Redis holds AND DB holds table?", a: "Redis is the source of truth for 'is this seat locked RIGHT NOW?' (sub-ms). DB holds table is the durable audit trail. Redis auto-expires; DB needs a cleanup job. Redis is fast but volatile; DB is slow but durable." },
              { q: "Why JSON for seat_ids in holds/bookings?", a: "A hold/booking groups multiple seats (user buys 4 tickets together). Normalized (separate table) adds JOINs on the hot path. JSON is simple, and we rarely query individual seats within a booking." },
              { q: "Why Snowflake IDs?", a: "Time-sortable, globally unique, no coordination. seat_id encodes (timestamp, shard, sequence). Can generate IDs client-side or server-side without hitting a central ID service." },
              { q: "Index on (event_id, status)?", a: "Optimizes 'show available seats for event X' — the most common query. Also enables 'count available seats' quickly. Partial index on status='available' would be even better (PostgreSQL)." },
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
          <Label color="#059669">Read Path Scaling (Seat Map)</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Seat map cached in Redis/CDN</strong> — full seat availability snapshot cached per event. Invalidated on seat status change. 1-2 second staleness acceptable for display; real-time check on hold attempt.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Read replicas for event browsing</strong> — event search, artist pages, venue info served from read replicas. Only booking writes go to primary. 95% of traffic is reads.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">CDN for static assets</strong> — venue maps (SVG), event images, seat map JS bundle. All served from CDN edge. Backend doesn't serve any static content.</Point>
            <Point icon="4." color="#059669"><strong className="text-stone-700">WebSocket for live availability</strong> — during flash sales, push seat status changes to connected clients via WS. Better than polling every second (which would amplify the load).</Point>
          </ul>
        </Card>
        <Card accent="#0891b2">
          <Label color="#0891b2">Write Path Scaling (Booking)</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#0891b2"><strong className="text-stone-700">Shard inventory by event_id</strong> — each event's seats on one DB shard. Contention is per-event, not cross-event. Two hot events don't interfere with each other.</Point>
            <Point icon="2." color="#0891b2"><strong className="text-stone-700">Redis for the hot lock layer</strong> — SET NX is O(1) and handles 100K+ ops/sec per node. The DB never sees seat-level contention; Redis absorbs it all.</Point>
            <Point icon="3." color="#0891b2"><strong className="text-stone-700">Virtual queue limits write rate</strong> — admission rate caps the number of concurrent booking attempts. System never receives more write traffic than it can handle. Auto-scales admission rate based on system health.</Point>
            <Point icon="4." color="#0891b2"><strong className="text-stone-700">Async ticket generation</strong> — after booking confirmed, ticket PDF/QR generation is async (queue worker). Doesn't block the checkout response. User gets booking confirmation immediately; tickets arrive in email within 60 seconds.</Point>
          </ul>
        </Card>
      </div>
      <Card accent="#7c3aed">
        <Label color="#7c3aed">Multi-Region Strategy</Label>
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { t:"Option A: Event-Homed Region ★", d:"Each event is assigned to the region closest to its venue. All booking transactions for that event run in that region.", pros:["Strong consistency — single-leader per event","Low latency for local buyers (most are local)","Simple: no cross-region coordination for bookings"], cons:["Remote buyers have higher latency (100-200ms)","Popular global events (Taylor Swift) bottleneck one region","Regional failure = event unavailable"], pick:true },
            { t:"Option B: Global Queue + Regional Processing", d:"Virtual queue is global (CDN-edge). After admission, route to the event's home region for booking.", pros:["Queue absorbs global traffic at edge","Booking still single-region (consistent)","Users worldwide get fair queue positions"], cons:["Still single-region for writes","Queue-to-booking handoff adds complexity"], pick:false },
            { t:"Option C: Active-Active (Multi-Region Inventory)", d:"Inventory replicated across regions. Any region can accept bookings. Cross-region coordination for consistency.", pros:["Lowest latency for all users","Survives regional failure","Scale reads infinitely"], cons:["Distributed locking across regions (Redlock?)","Risk of double-booking on partition","Extremely complex — not worth it for ticket booking"], pick:false },
          ].map((o,i) => (
            <div key={i} className={`rounded-lg border p-4 ${o.pick ? "border-purple-300 bg-purple-50/50" : "border-stone-200 bg-stone-50/30"}`}>
              <div className={`text-[11px] font-bold mb-1.5 ${o.pick ? "text-purple-700" : "text-stone-600"}`}>{o.t}</div>
              <p className="text-[11px] text-stone-500 mb-2">{o.d}</p>
              <div className="space-y-0.5 text-[11px]">
                {o.pros.map((p,j) => <div key={j} className="text-emerald-600">✔ {p}</div>)}
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
        <Label color="#d97706">Critical: No Double-Booking, No Lost Revenue</Label>
        <p className="text-[12px] text-stone-500 mb-4">Double-booking is an existential risk: two people show up for the same seat. Lost revenue from false 'sold out' is nearly as bad. The system must be correct AND highly available, especially during the brief on-sale window that generates most revenue.</p>
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="text-[11px] font-bold text-emerald-700 mb-1.5">Redis + DB Dual-Write</div>
            <p className="text-[11px] text-stone-500">Seat lock in Redis (speed) AND status update in DB (durability). If Redis fails, DB is the fallback source of truth. If DB fails, Redis still prevents double-booking short-term.</p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-white p-4">
            <div className="text-[11px] font-bold text-amber-700 mb-1.5">Payment Idempotency</div>
            <p className="text-[11px] text-stone-500">Every payment uses an idempotency key (hold_id). Network failures, retries, duplicate clicks — none can cause double-charging. Stripe/payment provider deduplicates on their end.</p>
          </div>
          <div className="rounded-lg border border-blue-200 bg-white p-4">
            <div className="text-[11px] font-bold text-blue-700 mb-1.5">Hold Expiry Safety Net</div>
            <p className="text-[11px] text-stone-500">If the booking service crashes mid-checkout, the hold TTL expires automatically (Redis) + a sweeper job cleans up DB state. Seats return to available — no permanent orphan locks.</p>
          </div>
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#059669">
          <Label color="#059669">Redis Failure Handling</Label>
          <ul className="space-y-2.5">
            <Point icon="1." color="#059669"><strong className="text-stone-700">Redis Sentinel for automatic failover</strong> — primary fails → Sentinel promotes replica within 5-10 seconds. During failover: seat holds may be temporarily unavailable. Accept brief unavailability over double-booking.</Point>
            <Point icon="2." color="#059669"><strong className="text-stone-700">Fallback to DB-level locking</strong> — if Redis is completely down, fall back to SELECT FOR UPDATE on the seats table. Slower (10ms vs 0.5ms) but correct. Circuit breaker triggers fallback automatically.</Point>
            <Point icon="3." color="#059669"><strong className="text-stone-700">Redis persistence (AOF)</strong> — append-only file with fsync=everysec. On crash, lose at most 1 second of holds. Those holds were &lt;1 second old — users will retry immediately.</Point>
          </ul>
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Consistency Guarantees</Label>
          <ul className="space-y-2.5">
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">Seat → one owner invariant</strong> — enforced by Redis SET NX (atomic) + DB unique constraint (seat_id, event_id, status='sold'). Two layers of protection against double-booking.</Point>
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">Hold → booking atomicity</strong> — converting a hold to a booking is a single DB transaction: update seats + create booking + delete hold. If any step fails, transaction rolls back.</Point>
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">Payment → booking consistency</strong> — Stripe webhook confirms payment even if the booking service crashes. Reconciliation job checks: 'Stripe says paid but no booking in DB' → create booking retroactively.</Point>
            <Point icon="🔒" color="#dc2626"><strong className="text-stone-700">Availability counter accuracy</strong> — available_seats counter decremented atomically with seat status change (same transaction). If counter drifts, a nightly reconciliation job recounts from seat table.</Point>
          </ul>
        </Card>
      </div>
    </div>
  );
}

function ObservabilitySection() {
  return (
    <div className="space-y-5">
      <Card accent="#0284c7">
        <Label color="#0284c7">Key Metrics Dashboard</Label>
        <div className="grid grid-cols-4 gap-3">
          {[
            { name: "Hold Acquisition Time", target: "p99 < 10ms", desc: "Time to SET NX in Redis for a seat hold", alarm: "> 50ms" },
            { name: "Checkout Completion Rate", target: "> 95%", desc: "% of holds that convert to bookings", alarm: "< 85%" },
            { name: "Hold Timeout Rate", target: "< 15%", desc: "% of holds that expire without payment", alarm: "> 30%" },
            { name: "Double-Booking Count", target: "= 0 ALWAYS", desc: "Two bookings for the same seat. NEVER acceptable.", alarm: "> 0" },
            { name: "Queue Wait Time (p50)", target: "< 5 min", desc: "Median time from entering queue to admission", alarm: "> 15 min" },
            { name: "Payment Latency", target: "p99 < 5s", desc: "Time from 'pay now' click to confirmation", alarm: "> 10s" },
            { name: "Seat Map Load Time", target: "p95 < 500ms", desc: "Time to render seat availability for an event", alarm: "> 2s" },
            { name: "Bot Detection Rate", target: "> 99%", desc: "% of automated booking attempts blocked", alarm: "< 95%" },
          ].map((m,i) => (
            <div key={i} className="rounded-lg border border-stone-200 p-3 bg-stone-50/30">
              <div className="text-[11px] font-bold text-stone-800">{m.name}</div>
              <div className="text-[12px] font-mono font-bold text-sky-700 mt-1">{m.target}</div>
              <div className="text-[10px] text-stone-400 mt-0.5">{m.desc}</div>
              <div className="text-[10px] text-red-500 mt-1">🚨 Alert: {m.alarm}</div>
            </div>
          ))}
        </div>
      </Card>
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#7c3aed">
          <Label color="#7c3aed">Distributed Tracing</Label>
          <CodeBlock code={`# Trace: User clicks "Hold Seats" → confirmation
trace_id: "book-xyz-789"
spans:
  ├─ [client] click_hold_button     0ms
  ├─ [cdn] queue_check              5ms   # Already admitted?
  ├─ [gateway] auth + rate_limit    8ms
  ├─ [inventory] check_availability 10ms  # Redis GET
  ├─ [inventory] acquire_hold       12ms  # Redis SET NX
  ├─ [inventory] update_db_status   25ms  # DB write
  ├─ [client] show_checkout_form    30ms  # User fills payment
  │   ... user enters payment info (60 seconds)
  ├─ [booking] verify_hold          60030ms
  ├─ [payment] create_intent        60050ms # Stripe API
  ├─ [payment] confirm_charge       62000ms # 2s Stripe latency
  ├─ [booking] confirm_seats_db     62020ms # DB transaction
  ├─ [booking] enqueue_ticket_gen   62025ms # Async
  └─ [client] show_confirmation     62030ms
# Hold: 30ms. Payment: ~2s. Total checkout: ~62s (user time)`} />
        </Card>
        <Card accent="#059669">
          <Label color="#059669">Logging & Alerting Strategy</Label>
          <div className="space-y-3">
            {[
              { level: "P0 — Page Immediately", items: ["Double-booking detected (count > 0)", "Redis cluster unreachable (no seat locking)", "Payment provider outage (Stripe down)", "Hold acquisition failure rate > 50%"] },
              { level: "P1 — Alert On-Call", items: ["Queue wait time p50 > 15 minutes", "Checkout completion rate < 85%", "Seat map cache miss rate > 20%", "Hold expiry rate > 30% (checkout flow broken?)"] },
              { level: "P2 — Dashboard Monitor", items: ["Bot detection blocks trending up", "Payment retry rate increasing", "Available seats counter drift detected", "Ticket generation queue backing up"] },
            ].map((l,i) => (
              <div key={i} className="rounded-lg border border-stone-200 p-3">
                <div className={`text-[10px] font-bold mb-1 ${i===0?"text-red-600":i===1?"text-amber-600":"text-sky-600"}`}>{l.level}</div>
                <div className="space-y-0.5">
                  {l.items.map((item,j) => (
                    <div key={j} className="text-[10px] text-stone-500 flex items-center gap-1.5">
                      <span className={`w-1 h-1 rounded-full shrink-0 ${i===0?"bg-red-400":i===1?"bg-amber-400":"bg-sky-400"}`}/>{item}
                    </div>
                  ))}
                </div>
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
        <Label color="#dc2626">Critical Failure Scenarios</Label>
        <div className="grid grid-cols-2 gap-4">
          {[
            { title: "Redis Crash → Orphan Holds in DB", desc: "Redis loses all seat locks on crash. DB still shows seats as 'held'. Users can't hold those seats (DB says held) but no one owns the holds (Redis lost them).", impact: "HIGH — seats stuck in 'held' state. Effectively sold out when they shouldn't be. Revenue loss.", mitigation: "Hold sweeper job runs every 60 seconds: any hold in DB past its expires_at → mark as expired + set seat status to 'available'. Redis is the speed layer; DB + sweeper is the correctness layer." },
            { title: "Stripe Webhook Missed → Payment Without Booking", desc: "User's payment succeeded in Stripe but our server crashed before creating the booking. Stripe webhook fails to deliver (our endpoint is down). User charged but has no ticket.", impact: "HIGH — user charged money but no ticket. Support nightmare. Potential chargeback.", mitigation: "Reconciliation job every 5 minutes: query Stripe for recent charges → compare with bookings table. Any charge without a matching booking → create booking retroactively and email the user. Also: Stripe retries webhooks for 72 hours." },
            { title: "Hold Expiry During Slow Payment", desc: "User's 7-minute hold expires while Stripe processes their payment (Stripe is slow, 10+ seconds). Hold released, seats become available. Another user grabs them. Original user's payment succeeds for seats they no longer hold.", impact: "HIGH — double-booking risk. Or: payment charged for seats given to someone else.", mitigation: "Extend hold by 60 seconds when payment is initiated (before calling Stripe). Check hold validity AFTER Stripe response. If hold expired during payment → refund immediately and apologize. Never confirm seats without verifying hold is still active." },
            { title: "Available Seats Counter Drift", desc: "available_seats counter on events table drifts from actual count of available seats. Shows 'sold out' when seats exist, or shows availability when sold out.", impact: "MEDIUM — false 'sold out' loses revenue. False availability frustrates users (click through to empty seat map).", mitigation: "Counter updated atomically in same transaction as seat status change. Nightly reconciliation job: SELECT COUNT(*) FROM seats WHERE status='available' GROUP BY event_id → compare with events.available_seats → fix drift." },
          ].map((f,i) => (
            <div key={i} className="rounded-lg border border-stone-200 p-4">
              <div className="text-[12px] font-bold text-stone-800 mb-1.5">{f.title}</div>
              <p className="text-[11px] text-stone-500 mb-2">{f.desc}</p>
              <div className="text-[11px]"><span className="font-bold text-red-600">Impact:</span> <span className="text-stone-500">{f.impact}</span></div>
              <div className="text-[11px] mt-1"><span className="font-bold text-emerald-600">Mitigation:</span> <span className="text-stone-500">{f.mitigation}</span></div>
            </div>
          ))}
        </div>
      </Card>
      <Card>
        <Label color="#d97706">Race Conditions & Edge Cases</Label>
        <div className="space-y-3">
          {[
            { case: "Two users click the same seat at the same millisecond", detail: "Redis SET NX is atomic — exactly one wins, one gets 409. No DB-level contention. The 'loser' is shown alternative seats within 50ms. This is the most common race condition and the simplest to handle." },
            { case: "User opens two browser tabs, holds same seats twice", detail: "Second hold attempt fails (seat already held by same user in first tab). Deduplicate by user_id — if user already holds a seat, return the existing hold_id instead of creating a new one. Or: enforce max 1 active hold per user per event." },
            { case: "Network timeout during payment — did it charge?", detail: "Client shows spinner forever. Stripe may or may not have charged. Solution: idempotency key ensures retry is safe. Client retries with same hold_id. Stripe returns the same charge result. Show outcome based on Stripe's authoritative response." },
            { case: "Scalper bot sends 10,000 hold requests in 1 second", detail: "Rate limit per user: max 5 hold requests/minute. Device fingerprinting + CAPTCHA for suspicious patterns. Queue system naturally throttles — bots wait in queue like everyone else. Block known bot IPs/user agents." },
          ].map((c,i) => (
            <div key={i} className="flex gap-3 items-start">
              <span className="text-amber-500 text-sm mt-0.5">⚡</span>
              <div>
                <div className="text-[11px] font-bold text-stone-700">{c.case}</div>
                <div className="text-[11px] text-stone-500 mt-0.5">{c.detail}</div>
              </div>
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
      <Card accent="#0f766e">
        <Label color="#0f766e">Service Decomposition</Label>
        <div className="grid grid-cols-3 gap-4">
          {[
            { name: "Inventory Service", owns: "Seat availability, seat holds (Redis locks), seat status management, seat map cache", tech: "Go + Redis (locks) + PostgreSQL (seats) + Redis (cache)", api: "gRPC: GetSeats, HoldSeats, ReleaseHold, GetAvailability", scale: "Horizontal — shard by event_id", stateful: false,
              modules: ["Seat Map Builder (generate availability snapshot for event)", "Hold Manager (acquire/release/extend Redis locks)", "Availability Counter (atomic increment/decrement)", "Hold Sweeper (background: clean expired holds from DB)", "Cache Invalidator (update cached seat map on status change)", "Suggestion Engine (nearby available seats when hold fails)"] },
            { name: "Booking Service", owns: "Checkout flow, payment orchestration, booking records, ticket generation", tech: "Java/Go + PostgreSQL + Stripe SDK + SQS", api: "gRPC: CreateBooking, CancelBooking, GetUserBookings", scale: "Horizontal — stateless, shard bookings by user_id", stateful: false,
              modules: ["Checkout Orchestrator (verify hold → charge → confirm)", "Payment Adapter (Stripe, PayPal, Apple Pay abstraction)", "Booking Writer (create booking record + update seats in txn)", "Ticket Generator (QR code, PDF, Apple Wallet pass — async)", "Refund Processor (cancel booking → reverse payment → release seats)", "Reconciliation Worker (match Stripe charges with bookings)"] },
            { name: "Queue Service", owns: "Virtual waiting room, admission control, fair ordering, access tokens", tech: "Go + Redis (sorted sets) + CDN (queue page)", api: "REST: EnterQueue, GetPosition, VerifyAdmission", scale: "Horizontal — per-event queues, Redis-backed", stateful: false,
              modules: ["Position Assigner (atomic INCR for fair ordering)", "Admission Controller (pop N users/sec from waiting set)", "Token Generator (JWT access token with event + TTL)", "Queue Page Renderer (static page with position polling)", "Rate Adjuster (auto-tune admission rate based on system health)", "Queue Analytics (conversion rate, drop-off, wait times)"] },
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
                    <span className="w-1 h-1 rounded-full bg-teal-400 shrink-0"/>{m}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-5">
        <Card accent="#2563eb">
          <Label color="#2563eb">Supporting Services</Label>
          <div className="space-y-3">
            {[
              { name: "Event Service", role: "Event CRUD, venue management, pricing tiers, on-sale scheduling. Mostly admin operations. Feeds event data to search index (Elasticsearch) for user-facing search." },
              { name: "Search Service", role: "Full-text event search with filters (artist, city, date, genre, price range). Powered by Elasticsearch. Updated async via CDC from Event DB. Also powers 'recommended events'." },
              { name: "Notification Service", role: "Email confirmations, push notifications (ticket ready, event reminder), SMS for 2FA. Async via SQS. Handles retries and delivery tracking." },
              { name: "User Service", role: "User accounts, authentication, payment methods on file, purchase history. OAuth 2.0 for identity. Stores no payment credentials (Stripe handles PCI compliance)." },
            ].map((s,i) => (
              <div key={i} className="rounded-lg border border-stone-200 p-3">
                <div className="text-[11px] font-bold text-stone-800 mb-0.5">{s.name}</div>
                <div className="text-[10px] text-stone-500">{s.role}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card accent="#d97706">
          <Label color="#d97706">Technology Choices</Label>
          <div className="space-y-3">
            {[
              { comp: "Seat Locks", choice: "Redis 7+ (Cluster)", why: "SET NX with TTL — atomic, sub-ms, auto-expiry. 100K+ ops/sec per node. The speed layer for seat contention. Sentinel for HA." },
              { comp: "Inventory DB", choice: "PostgreSQL (sharded)", why: "ACID for seat status transitions. Partial indexes on status='available'. Sharded by event_id. Strong consistency for the booking transaction." },
              { comp: "Booking DB", choice: "PostgreSQL", why: "Same cluster, different schema. ACID for payment + booking atomicity. Foreign keys to seats table (same shard since sharded by event_id)." },
              { comp: "Virtual Queue", choice: "Redis Sorted Set + CDN", why: "ZADD/ZPOPMIN for fair ordering at 100K ops/sec. CDN serves static queue page — near-zero backend cost for millions of waiting users." },
              { comp: "Search", choice: "Elasticsearch", why: "Full-text search, geo-filtering (events near me), faceted search (by genre, price). Updated async via CDC. Not on the booking hot path." },
              { comp: "Payment", choice: "Stripe", why: "PCI-compliant payment processing. Idempotency keys prevent double-charges. Webhooks for async confirmation. Apple Pay / Google Pay support." },
            ].map((t,i) => (
              <div key={i}>
                <div className="text-[11px]"><span className="font-bold text-stone-700">{t.comp}:</span> <span className="font-mono text-amber-700">{t.choice}</span></div>
                <div className="text-[10px] text-stone-400 mt-0.5">{t.why}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function FlowsSection() {
  const [sel, setSel] = useState("booking");
  const flows = {
    booking: { name: "Complete Booking Flow", steps: [
      { actor: "Client", action: "User selects seats A7, A8 on seat map", detail: "Client sends POST /v1/events/:id/hold with seat_ids=['A7','A8']. Seat map was loaded from cache (1s stale max)." },
      { actor: "API Gateway", action: "Authenticate + rate limit + queue check", detail: "Verify JWT token. Check rate limit (5 hold attempts/min). Verify queue admission token if flash sale event." },
      { actor: "Inventory Svc", action: "Attempt Redis SET NX for each seat", detail: "SET hold:evt123:A7 hold_abc NX EX 420 → success. SET hold:evt123:A8 hold_abc NX EX 420 → success. Both locked!" },
      { actor: "Inventory Svc", action: "Update DB + decrement counter", detail: "UPDATE seats SET status='held', hold_id=... WHERE seat_id IN (A7,A8). UPDATE events SET available_seats = available_seats - 2." },
      { actor: "Client", action: "User fills in payment details (within 7 min)", detail: "Countdown timer shown. User enters credit card or selects saved payment method. 'Complete Purchase' button." },
      { actor: "Booking Svc", action: "Verify hold → create Stripe payment intent", detail: "Check hold_id is still active + not expired. Create Stripe PaymentIntent with idempotency_key=hold_id. Confirm charge." },
      { actor: "Booking Svc", action: "Confirm booking in DB (single transaction)", detail: "BEGIN TXN: UPDATE seats SET status='sold', booking_id=... + INSERT INTO bookings + DELETE hold + COMMIT." },
      { actor: "Booking Svc", action: "Return confirmation + enqueue ticket generation", detail: "Return booking_id + receipt. Enqueue async: generate QR codes, PDF tickets, send confirmation email." },
    ]},
    flash_sale: { name: "Flash Sale (Queue) Flow", steps: [
      { actor: "Client", action: "User arrives at event page 10 min before on-sale", detail: "Event page shows countdown timer. At on-sale time (10:00 AM), user clicks 'Get Tickets'." },
      { actor: "CDN / Edge", action: "Redirect to virtual queue", detail: "Static queue page served from CDN. User gets position via Redis INCR. Page polls /queue-status every 3 seconds." },
      { actor: "Queue Service", action: "Assign position #4,521 in queue", detail: "Redis INCR queue:evt123:counter → 4521. ZADD queue:evt123:waiting user_42 4521. Estimated wait: ~1 minute at 5K/sec admission." },
      { actor: "Queue Service", action: "Admission controller pops batch every 1s", detail: "ZPOPMIN queue:evt123:waiting 5000 → admit 5000 users. Generate JWT access token (TTL=10min) for each admitted user." },
      { actor: "Client", action: "User admitted — redirected to seat map", detail: "Queue page detects admission (polling response includes token). Redirect to event page with X-Queue-Token header." },
      { actor: "API Gateway", action: "Verify queue token on all booking APIs", detail: "Middleware checks X-Queue-Token JWT: valid? not expired? correct event_id? If invalid → redirect back to queue." },
      { actor: "Inventory Svc", action: "Normal hold flow (rate-limited by queue)", detail: "System receives at most 5000 new users/sec (queue admission rate). Seat holds proceed normally — no thundering herd." },
    ]},
    cancel: { name: "Cancellation & Refund Flow", steps: [
      { actor: "Client", action: "User clicks 'Cancel Booking' on their ticket", detail: "POST /v1/bookings/:id/cancel. Client shows refund policy (full refund if >48h before event, 50% if &lt;48h, no refund &lt;24h)." },
      { actor: "Booking Svc", action: "Verify cancellation eligibility", detail: "Check booking status (must be 'confirmed'). Check refund policy based on time until event. Calculate refund amount." },
      { actor: "Payment Svc", action: "Initiate Stripe refund", detail: "stripe.refunds.create(charge_id, amount). Partial or full refund. Stripe processes in 5-10 business days to user's card." },
      { actor: "Booking Svc", action: "Update booking + release seats", detail: "BEGIN TXN: UPDATE booking SET status='cancelled' + UPDATE seats SET status='available', booking_id=NULL + UPDATE events.available_seats += N." },
      { actor: "Notification Svc", action: "Send cancellation confirmation", detail: "Email with refund details and timeline. Push notification confirming cancellation. Remove ticket from user's wallet." },
      { actor: "Inventory Svc", action: "Invalidate seat map cache", detail: "Cancelled seats now show as available. Update cached seat map. If event had a waitlist, notify waitlisted users that seats are available." },
    ]},
  };
  const f = flows[sel];
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-3 space-y-2">
          {Object.entries(flows).map(([k,v]) => (
            <button key={k} onClick={() => setSel(k)}
              className={`w-full text-left px-3.5 py-2.5 rounded-lg text-[12px] font-medium border transition-all ${k===sel?"bg-purple-600 text-white border-purple-600":"bg-white text-stone-500 border-stone-200 hover:border-stone-300"}`}>
              {v.name}
            </button>
          ))}
        </div>
        <div className="col-span-9">
          <Card accent="#7e22ce">
            <Label color="#7e22ce">{f.name}</Label>
            <div className="space-y-0">
              {f.steps.map((s,i) => (
                <div key={i} className={`flex gap-4 py-3 ${i < f.steps.length-1 ? "border-b border-stone-100" : ""}`}>
                  <div className="flex flex-col items-center">
                    <span className="w-7 h-7 rounded-full bg-purple-100 text-purple-700 text-[11px] font-bold flex items-center justify-center">{i+1}</span>
                    {i < f.steps.length-1 && <div className="w-px flex-1 bg-purple-200 mt-1"/>}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono font-bold text-purple-600 bg-purple-50 px-2 py-0.5 rounded">{s.actor}</span>
                      <span className="text-[12px] font-medium text-stone-700">{s.action}</span>
                    </div>
                    <div className="text-[11px] text-stone-400 mt-0.5">{s.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function DeploymentSection() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-5">
        <Card accent="#b45309">
          <Label color="#b45309">Kubernetes Deployment — Inventory Service</Label>
          <CodeBlock code={`apiVersion: apps/v1
kind: Deployment
metadata:
  name: inventory-service
spec:
  replicas: 50                  # Scale up before flash sales
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 0         # Zero downtime during deploy
  template:
    spec:
      containers:
      - name: inventory
        image: inventory-svc:v4.1
        ports:
        - containerPort: 8080   # gRPC
        resources:
          requests:
            memory: "512Mi"
            cpu: "1"
          limits:
            memory: "1Gi"
            cpu: "2"
        env:
        - name: REDIS_CLUSTER
          value: "redis-seat-locks.internal:6379"
        - name: DB_POOL_SIZE
          value: "50"
        - name: HOLD_TTL_SEC
          value: "420"
        - name: MAX_SEATS_PER_HOLD
          value: "8"            # Prevent bulk holds
        readinessProbe:
          grpc:
            port: 8080
          periodSeconds: 5
        livenessProbe:
          grpc:
            port: 8080
          periodSeconds: 10

# HPA: auto-scale on CPU + custom metric (hold_requests/sec)
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inventory-hpa
spec:
  scaleTargetRef:
    name: inventory-service
  minReplicas: 10
  maxReplicas: 200
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 60`} />
        </Card>
        <Card accent="#dc2626">
          <Label color="#dc2626">Security — Anti-Bot + Access Control</Label>
          <div className="space-y-3">
            {[
              { layer: "Bot Prevention (Critical)", details: ["CAPTCHA challenge before entering queue (reCAPTCHA v3 / hCaptcha)", "Device fingerprinting (browser, screen, plugins)", "Behavioral analysis (mouse movement, typing speed)", "Rate limiting: max 5 hold attempts/min per user", "IP reputation scoring — block known bot IPs/ASNs", "Account age check: new accounts flagged for extra verification"] },
              { layer: "Authentication & Authorization", details: ["OAuth 2.0 / OIDC for user identity", "JWT tokens with short TTL (15 min) + refresh tokens", "Queue admission tokens: event-scoped JWT", "Admin APIs: separate auth with MFA required"] },
              { layer: "Payment Security", details: ["PCI DSS compliance: never store card numbers (Stripe tokenization)", "3D Secure for high-value transactions", "Fraud detection: velocity checks, address verification", "Chargeback monitoring and automated dispute response"] },
              { layer: "Data Protection", details: ["TLS 1.3 everywhere (API, DB connections, Redis)", "Encryption at rest for booking and user data (AES-256)", "PII minimization: store user_id references, not full names in logs", "GDPR: data export and deletion on user request"] },
            ].map((s,i) => (
              <div key={i} className="rounded-lg border border-stone-200 p-3">
                <div className="text-[11px] font-bold text-stone-800 mb-1">{s.layer}</div>
                <div className="space-y-0.5">
                  {s.details.map((d,j) => (
                    <div key={j} className="text-[10px] text-stone-500 flex items-center gap-1.5">
                      <span className="w-1 h-1 rounded-full bg-red-400 shrink-0"/>{d}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function OpsSection() {
  return (
    <div className="space-y-5">
      <Card accent="#be123c">
        <Label color="#be123c">Bottleneck Analysis</Label>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] border-collapse">
            <thead><tr className="border-b-2 border-stone-200">
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Bottleneck</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Symptom</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Fix</th>
              <th className="text-left px-3 py-2 font-semibold text-stone-500">Pitfall</th>
            </tr></thead>
            <tbody>
              {[
                { b: "Redis SET NX Throughput", s: "Hold acquisition latency > 10ms. Redis CPU at 100%.", f: "Redis Cluster: shard keys by event_id across nodes. Each event's locks on one node. Hot events still bottleneck one node — add read replicas for availability checks.", p: "Redis is single-threaded per shard. One mega-event can saturate one node. Consider: Lua script to batch multi-seat holds into one round trip." },
                { b: "DB Write Contention (seats table)", s: "Seat status updates > 50ms. Connection pool exhausted.", f: "Batch seat updates: UPDATE seats SET status='held' WHERE seat_id IN (...). Shard by event_id. Reduce DB writes by making Redis authoritative for holds (DB update async).", p: "Async DB update risks Redis-DB inconsistency. Use WAL + reconciliation. Don't sacrifice consistency for performance on the hold confirmation path." },
                { b: "Stripe API Latency", s: "Payment confirmation takes > 5s. User thinks checkout is stuck.", f: "Can't speed up Stripe — it's external. Show progress indicator. Implement webhook-based async confirmation. Client polls for booking status. Don't block the user thread on Stripe response.", p: "Stripe p99 latency can spike to 15s during high load. Your hold TTL must account for this. Extend hold when payment is initiated." },
                { b: "Seat Map Cache Invalidation", s: "Stale seat maps show available seats that are actually held/sold. Users get 409 on every hold attempt.", f: "WebSocket push for real-time seat status. Or: short TTL cache (2-3s). On 409, force-refresh seat map from source. Accept that during flash sales, the map is always slightly stale.", p: "Don't try for real-time seat map accuracy at 500K concurrent users. The hold attempt is the source of truth. Seat map is a 'hint' — show freshness timestamp." },
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
            { title: "Taylor Swift Eras Tour Breaks Ticketmaster", symptom: "14 million users hit the queue simultaneously. Queue system overwhelmed. Site shows errors. Verified fans can't access seats. Congressional hearing follows.",
              cause: "Queue designed for 1M concurrent users, not 14M. Redis queue counter hit memory limits. Admission controller couldn't generate tokens fast enough. CDN cache stampede on queue status endpoint.",
              fix: "10× queue capacity pre-provisioning for mega-events. Separate Redis cluster per mega-event. CDN caching for queue position (stale by design — user doesn't need real-time position). Pre-computed admission batches instead of individual token generation.",
              quote: "We load-tested for 1.5M. 14M showed up. No amount of auto-scaling helps when 14M connections arrive in 30 seconds. You have to pre-scale." },
            { title: "Double-Booking During Redis Failover", symptom: "Redis primary fails during on-sale. Sentinel promotes replica. During the 5-second window, 23 seats are double-booked. Two people show up for the same seat at the venue.",
              cause: "Redis replication is async. Replica was 200ms behind primary. Primary accepted SET NX → crashed → replica promoted without those locks. New requests succeeded for already-held seats.",
              fix: "WAIT command after SET NX: block until at least 1 replica acknowledges the write. Adds 1-2ms latency but eliminates the async replication window. Also: DB-level unique constraint catches any remaining edge cases (booking fails if seat already has status='sold').",
              quote: "23 double-bookings. 23 angry customers. 23 refunds + free upgrades. The 1-2ms latency for WAIT is the cheapest insurance we ever bought." },
            { title: "Hold Sweeper Kills DB During Peak Load", symptom: "Hold sweeper job scans for expired holds every 30 seconds. During flash sale, 500K expired holds pile up. Sweeper query takes 45 seconds, locks the seats table, blocks new bookings.",
              cause: "SELECT * FROM holds WHERE expires_at < NOW() AND status='active' does a full table scan on 500K rows during peak. Then 500K individual UPDATE statements to release seats.",
              fix: "Sweeper uses batched updates: UPDATE seats SET status='available' WHERE hold_id IN (SELECT hold_id FROM holds WHERE expires_at < NOW() LIMIT 1000). Process 1000 at a time with 100ms sleep between batches. Also: Redis TTL is the primary expiry — DB sweeper is a background consistency check, not the hot path.",
              quote: "Our safety net (the sweeper) became the threat. A background job doing 500K writes during a flash sale. We rate-limited our own cleanup to stop killing ourselves." },
            { title: "Payment Provider Rate-Limited Us", symptom: "Stripe returns 429 (rate limited) during peak checkout. 30% of payments fail. Users see 'payment failed' but their seats are held — can't retry with new payment method.",
              cause: "Stripe rate limit: 100 requests/sec per account. We hit it during a 10K tx/sec flash sale. Stripe's Enterprise tier needed, not standard.",
              fix: "Stripe Enterprise plan (higher rate limits). Payment request queue with backpressure: if Stripe returns 429, queue the request and retry with exponential backoff. Show user 'processing...' not 'failed'. Allow retry with different payment method without releasing hold.",
              quote: "We learned that our 'unlimited' Stripe plan had a limit after all. At 10K checkout/sec, everything has a limit. We now pre-coordinate with Stripe before every major on-sale." },
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
        { t: "Dynamic Pricing", d: "Adjust ticket prices based on demand in real-time. High demand → prices increase. Low demand → discounts to fill seats.", detail: "ML model predicts demand curve. Price updated every N minutes. Floor/ceiling prices prevent extremes. Transparent to users ('prices may change based on demand').", effort: "Hard" },
        { t: "Waitlist & Auto-Purchase", d: "When sold out, users join a waitlist. If a ticket is cancelled or released, the next waitlisted user is auto-notified or auto-charged.", detail: "Priority queue (FIFO or highest-bid). On seat release: dequeue next user → hold seat → charge saved payment method → confirm. Time-limited auto-purchase (user has 5 min to decline).", effort: "Medium" },
        { t: "Resale Marketplace", d: "Allow ticket holders to resell tickets at or below face value (or market price). Verified transfer between users.", detail: "Seller lists ticket → marketplace. Buyer purchases → original ticket voided, new ticket issued. Prevents counterfeiting. Platform takes commission (10-15%). Price cap options for artists.", effort: "Hard" },
        { t: "Interactive 3D Seat Map", d: "View from your seat before purchasing. 360° venue photos rendered per seat section.", detail: "Pre-rendered images per section (not real-time 3D). Loaded lazily on hover/click. Hosted on CDN. Major differentiator for premium events.", effort: "Medium" },
        { t: "Group Booking", d: "Allow groups to coordinate — one person selects seats, others join the booking and pay individually.", detail: "Shared hold for the group (extended TTL — 15 min). Each member pays their share. Booking confirmed only when all pay. Partial payment → partial booking (configurable).", effort: "Medium" },
        { t: "Ticket Insurance", d: "Users can buy refund protection at checkout. Guaranteed refund if they can't attend, regardless of event's refund policy.", detail: "Third-party insurance provider integration. Premium = 10-15% of ticket price. Automatic refund processing on claim. Profitable for the platform (claims < premiums).", effort: "Easy" },
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
    { q:"How do you prevent double-booking?", a:"Three layers of defense: (1) Redis SET NX — atomic 'set if not exists'. At the lock layer, only one request wins per seat. (2) DB status check in the booking transaction — UPDATE seats SET status='sold' WHERE status='held' AND hold_id=X. If another booking somehow slipped through, this WHERE clause fails. (3) DB unique constraint — (seat_id, event_id) with status='sold' can have at most one row. The DB itself prevents duplicates even if application logic has a bug.", tags:["consistency"] },
    { q:"Why temporary holds instead of immediate booking?", a:"Payment takes 2-10 seconds (Stripe API call). If you lock the seat only during payment processing, you have a 2-10 second window where the seat is in limbo — showing as available but mid-purchase. The temporary hold (7 min) gives users time to enter payment details while guaranteeing the seat is reserved. Without holds, users would race to complete checkout faster than others — terrible UX.", tags:["design"] },
    { q:"How does the virtual queue work?", a:"Redis INCR for atomic position assignment (perfectly fair). Users see a static CDN-hosted queue page that polls for position. Admission controller runs every 1 second: ZPOPMIN pops the next N users from the waiting sorted set. Admitted users get a JWT access token (10 min TTL). All booking APIs require this token via middleware. This converts a thundering herd (14M simultaneous) into a controlled stream (5K/sec).", tags:["scalability"] },
    { q:"What if the user's hold expires while they're paying?", a:"When the user clicks 'Pay Now', extend the hold by 60 seconds (covers Stripe latency). After Stripe responds, verify the hold is still active before confirming. If the hold expired during the extended window (extremely rare — >60 second Stripe call), refund immediately and show 'Sorry, your session expired'. The key principle: never confirm booking without verifying the hold. Check hold validity AFTER payment, not just before.", tags:["reliability"] },
    { q:"How do you handle the 'bot' problem?", a:"Multi-layered: (1) CAPTCHA before entering queue (stops simple scripts). (2) Device fingerprinting + behavioral analysis (mouse movement patterns — bots move linearly). (3) Rate limiting per user (5 hold attempts/min). (4) Account age and history checks (new accounts flagged). (5) IP reputation (block known datacenter IPs). (6) The queue itself helps — bots wait like everyone else. Ticketmaster's 'Verified Fan' program pre-registers users and gives priority — known humans first.", tags:["security"] },
    { q:"How do you handle a sold-out event with cancelled tickets?", a:"When a booking is cancelled: (1) Seats return to 'available'. (2) If a waitlist exists, notify the next N waitlisted users ('Tickets available! You have 5 minutes to purchase.'). (3) If no waitlist, seats appear on the seat map for general purchase. For high-demand events, the waitlist auto-purchase mode can automatically charge the next user's saved payment method (with their pre-consent).", tags:["design"] },
    { q:"What database would you use for the seats table?", a:"PostgreSQL, sharded by event_id. Why: (1) ACID transactions for the hold→sold status transition (critical for no double-booking). (2) Partial indexes: CREATE INDEX ON seats(event_id) WHERE status='available' — only indexes available seats, much smaller than full index. (3) SELECT FOR UPDATE as a fallback if Redis is down. (4) Strong consistency — eventual consistency is unacceptable for seat inventory. Not Cassandra (no ACID). Not DynamoDB (limited transaction support across items).", tags:["data"] },
    { q:"What happens if Redis goes down entirely?", a:"Circuit breaker detects Redis is unreachable. Fallback: DB-level locking with SELECT FOR UPDATE on the seat row. This is 10× slower than Redis (10ms vs 1ms) and creates DB contention, but it's correct. System degrades: lower throughput, higher latency, but no double-booking. The queue automatically reduces admission rate when it detects elevated hold latency, naturally reducing load on the DB fallback path.", tags:["availability"] },
    { q:"How do you scale for a stadium with 80,000 seats?", a:"80K seats is one event, one DB shard. The seats table has 80K rows — trivially small for PostgreSQL. The challenge isn't data size, it's contention: 80K users trying to hold seats simultaneously. Redis handles this: 80K SET NX operations complete in &lt;1 second. The seat map is the bottleneck — rendering 80K seats in the browser. Solution: section-level aggregation (show sections, not individual seats) until the user zooms into a section.", tags:["scalability"] },
    { q:"How do you handle partial failures in the booking saga?", a:"State machine: HOLD → PAYMENT_PENDING → CONFIRMED or RELEASED. If crash at any point: (1) HOLD state + no Stripe charge → hold expires naturally, seats released. (2) PAYMENT_PENDING + Stripe charge succeeded → reconciliation job detects orphaned charge → creates booking. (3) PAYMENT_PENDING + Stripe charge failed → release hold immediately. (4) Stripe webhook as backup: if our server is completely down, Stripe webhooks retry for 72 hours. On recovery, webhook handler creates the booking.", tags:["reliability"] },
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

export default function TicketBookingSD() {
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
            <h1 className="text-xl font-bold text-stone-800 tracking-tight">Ticket Booking (Ticketmaster)</h1>
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