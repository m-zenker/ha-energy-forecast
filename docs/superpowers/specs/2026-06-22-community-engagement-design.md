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

### Post 1 Draft

> **Title:** I built an ML energy consumption forecaster to schedule loads around solar surplus
>
> **Category:** Projects & Integration Showcase
>
> With solar panels it's not enough to know when the sun will shine — you also need to know when your household will *consume*. I built a Home Assistant app that predicts hourly energy consumption 48 hours ahead, so I can schedule my washing machine, dishwasher, and EV charger in the windows where solar surplus is largest.
>
> The model trains entirely on your own HA meter history — no cloud service, no generic averages. It learns your household's patterns (daily routines, heat pump cycles, seasonal swings) and publishes them as standard HA sensors.
>
> **Key features**
> - 48h hourly forecast with calibrated prediction intervals
> - Scenario API: "what if I run the dishwasher at 14:00?" — returns the delta against the baseline forecast
> - SHAP explainability: know *why* the model predicted high consumption today
> - Rolling 7d/30d MAE sensors so you can track real-world accuracy over time
> - Works on Raspberry Pi (automatic LightGBM → scikit-learn fallback)
>
> [screenshot: assets/dashboard_overview.png]
>
> GitHub: https://github.com/m-zenker/ha-energy-forecast
>
> I've just set up GitHub Discussions — curious what MAE values others are seeing and how you're using the forecast in automations. Happy to answer questions here or there.

---

## Section 4: Seed Discussion Posts (Launch Day)

Create all four before the first HA forum post goes live.

### 1. Pinned Welcome (General)

> **Title:** Welcome — what this project is and how to use Discussions
>
> HA Energy Forecast is a Home Assistant app that predicts your household's hourly energy consumption 48 hours ahead, using a LightGBM model trained entirely on your own meter data. I use it mainly to schedule deferrable loads (dishwasher, washing machine, EV charger) in windows where solar surplus is largest — but people use it for bill prediction, anomaly detection, and heat pump scheduling too.
>
> This Discussions space has three active categories:
>
> - **Accuracy & Benchmarks** — share your MAE, compare model diagnostics, track accuracy over time
> - **Use Cases & Setups** — show your automations, dashboards, and config; ask for setup advice
> - **Roadmap & Feature Requests** — vote on what I should build next; propose your own ideas
>
> I check in here regularly and will respond to everything. Install questions and bug reports are also welcome here (or as GitHub Issues if you want them tracked).
>
> To get started: [→ What's your MAE?](link) · [→ How are you using it?](link) · [→ What should I build next?](link)

*(Update the three links once the other seed posts are live.)*

### 2. Accuracy & Benchmarks seed

> **Title:** What's your MAE? Share your model diagnostics
>
> The rolling MAE sensors (`sensor.energy_forecast_mae_7d`, `mae_30d`, `relative_mae_7d`) are the best way to gauge whether your model is well-fitted for your household. I'm curious how others compare — hardware, setup complexity, and time running all affect accuracy.
>
> **My numbers (to give you a reference):**
> - 7d MAE: 0.16 kWh/h · 30d MAE: 0.20 kWh/h
> - Relative 7d MAE: 60% — but this is misleading in summer: solar covers most consumption, so grid import averages ~0.26 kWh/h and even a small absolute error looks large as a percentage. Absolute MAE is the more honest metric when you have solar.
> - Running since: October 2025 (~8 months of training data)
> - Hardware: Raspberry Pi 5 8 GB (LightGBM, not sklearn fallback)
> - Optional features enabled: heat pump sub-sensor, thermal pressure, DHW pressure, daily regime clustering, EV detection, solar + battery target correction
>
> Share whatever you have — even a rough number is useful. If your MAE feels high, mention your setup and we can troubleshoot.

### 3. Use Cases & Setups seed

> **Title:** How are you using the forecast? Show your setup
>
> The forecast is most useful when it drives something — an automation, a scheduling decision, a dashboard alert. I'm curious what people have built.
>
> **My main use case:** the consumption forecast feeds into a companion app, [ha-energy-manager](https://github.com/m-zenker/ha-energy-manager), which combines it with a solar production forecast to compute expected hourly surplus and automatically schedules deferrable loads (dishwasher, washing machine, heat pump DHW boost) in the best windows. The scenario API is useful for sanity-checking a candidate schedule: "if I run the washer at 14:00, what does total consumption look like?"
>
> But you don't need the energy manager to get value — `energy_forecast_tomorrow` and the 3-hour block sensors work directly in HA automations. A simple threshold automation ("if surplus forecast for 13:00–15:00 > 1.5 kWh, start the dishwasher") goes a long way.
>
> What are you doing with yours? Paste an automation, a dashboard card, or just describe the decision you're automating. Happy to help adapt the sensor setup for your use case.

### 4. Roadmap Vote seed

> **Title:** What should I build next? Vote on the roadmap
>
> Here are the top candidates for the next release. React with 👍 on the ones you'd find most useful, or comment if you have a reason to prioritise (or skip) one.
>
> **#16 — HACS support**
> Make the app installable directly from HACS. No code changes needed — mostly packaging work. Would significantly lower the install barrier.
>
> **#87 — Recent consumption trend feature**
> Add `trend_deviation` (24h rolling mean minus 7d rolling mean) as a model feature. A simulation on my own data shows ~18% daily MAE improvement on ordinary days. About 1h of work.
>
> **#15 — HVAC flow setpoint projection**
> For heat pump households: project the heating curve forward using forecast outdoor temperatures, giving the model a more accurate thermal load signal for the full 48h window. High impact if you have a heat pump.
>
> **#10 — School holidays**
> Add a school holiday flag (configurable per country/region) so the model learns that daytime consumption rises during school breaks. Medium impact for families.
>
> This is genuinely how I prioritise — your votes influence what goes in next.

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
