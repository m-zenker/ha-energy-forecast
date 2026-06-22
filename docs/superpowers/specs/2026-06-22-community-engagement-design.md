# Community Engagement Campaign — Design Spec

**Date:** 2026-06-22
**Scope:** GitHub Discussions + HA Community Forum
**Goal:** Grow the user base and collect structured product feedback simultaneously.

---

## Context

The project is mirrored to [GitHub](https://github.com/m-zenker/ha-energy-forecast) with Discussions already enabled. As of 2026-06-22: 8 stars, 2 forks, zero existing Discussion threads. Engagement has been close to zero. The primary development repo is on a private Forgejo instance; GitHub is the public-facing surface.

**Constraints:**
- Maintainer capacity: bi-weekly Discussion posts; replies as frequent as needed
- No budget for promotion — organic + HA Community Forum posts only
- Audience: HA power users with smart meters and AppDaemon

---

## Section 1: GitHub Discussions Structure

Four categories (three active tracks + default Q&A):

| Category | Purpose |
|---|---|
| **General / Q&A** | Installation questions, troubleshooting (GitHub default) |
| **Accuracy & Benchmarks** | MAE sharing, model diagnostics, seasonal accuracy trends |
| **Use Cases & Setups** | Automations, dashboards, config tips, hardware setups |
| **Roadmap & Feature Requests** | Vote on backlog items, propose new features |

Three active categories is the right size for a cold-start community — enough structure to organise signal, not so many that any one looks abandoned.

---

## Section 2: Three-Track Rotation

Bi-weekly posts cycling through three tracks. Each post: 2–3 sentences of framing, one concrete question, maintainer's own answer as a seed reply.

| Slot | Track | Example prompt |
|---|---|---|
| Week 1 | Accuracy & Benchmarks | "Share your MAE — what is your model reporting?" |
| Week 3 | Use Cases & Setups | "How are you using the forecast? Show your automations." |
| Week 5 | Roadmap vote | "What should I build next — #83 predicted day total vs #84 legionella hour?" |
| Week 7 | Accuracy & Benchmarks | "Seasonal check-in: how has your MAE changed since winter?" |
| Week 9 | Use Cases & Setups | "Dashboard showcase — post your Lovelace cards." |
| Week 11 | Roadmap vote | "Physics-ML hybrid spec is written — feedback before I start?" |

**Rule:** the maintainer always replies first in every post. An empty thread is a dead thread.

Replies to install questions, bug reports, and general comments happen as frequently as needed — no cadence constraint.

---

## Section 3: HA Community Forum Launch Posts

Two posts, staggered by one week. GitHub Discussions are seeded (Section 4) before either post goes live.

### Post 1 — Project Showcase
**Category:** *Projects & Integration Showcase*
**Hook:** "I built an ML energy consumption forecaster for Home Assistant"
**Content:** What it does, a dashboard screenshot, key features (48h forecast, SHAP explainability, anomaly detection, scenario API). End with a direct invitation: "I've set up GitHub Discussions for anyone who wants to compare accuracy or share how they use it — would love to hear what MAE values others are seeing."
**Primary goal:** drive users to the Accuracy & Benchmarks Discussion thread.

### Post 2 — Practical How-To
**Category:** *Automation & Scripts* or *Community Guides*
**Hook:** "Automating heat pump scheduling with a 48-hour energy forecast"
**Content:** A concrete automation example — using `energy_forecast_tomorrow` to shift DHW boost timing for better COP. Practical and searchable; targets people looking for heat pump automation, not just energy nerd content.
**Primary goal:** grow user base via search; links back to repo and Discussions for questions.

**Timing:** Post 1 first, Post 2 one week later.

---

## Section 4: Seed Discussion Posts (Launch Day)

Create all four before the first HA forum post goes live.

### 1. Pinned Welcome (General)
Short intro: what the project does, what the three categories are for, maintainer's commitment to respond. Sets the community tone.

### 2. Accuracy & Benchmarks seed
**Title:** "What's your MAE? Share your model diagnostics"
**Maintainer reply template (go first):**
- 7d MAE and 30d MAE values
- Hardware (Pi / x86 / other)
- How long the model has been running
- Which optional features are enabled (heat pump sub-sensor, thermal pressure, regimes, etc.)

Providing the template as a first reply gives others a format to copy, lowering the participation barrier significantly.

### 3. Use Cases & Setups seed
**Title:** "How are you using the forecast? Show your setup"
**Maintainer reply:** describe the DHW/heat pump automation in use, paste a dashboard screenshot, list enabled optional features.

### 4. Roadmap Vote seed
**Title:** "What should I build next?"
**Content:** List 3–4 backlog items with one-line descriptions (e.g. #83 predicted day total, #84 legionella hour, #87 trend deviation feature, #16 HACS packaging). Ask for 👍 reactions or comments with reasoning. Explicitly state: "this is how I prioritise."

---

## Success Metrics

No vanity targets — track these qualitatively after 3 months:

- At least 5 distinct users have posted their MAE in the benchmark thread
- At least one use-case or automation shared by someone other than the maintainer
- At least one roadmap item prioritised or deprioritised based on Discussion votes
- HA forum posts each receive ≥ 10 replies or ≥ 50 views

---

## What This Design Does Not Cover

- HACS submission (separate initiative; tracked on roadmap as #16)
- Automated posting or bots
- Reddit (r/homeassistant) — potential second wave after HA forum traction is established
