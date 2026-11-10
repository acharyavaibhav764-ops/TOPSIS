# TOPSIS

<p align="center">
  <img src="https://img.shields.io/badge/TOPSIS-Market%20Ranking%20Engine-0EA5E9?style=for-the-badge&labelColor=0F172A" alt="TOPSIS Market Ranking Engine" />
</p>

<p align="center">
  <strong>Next-generation Multi-Criteria Decision Making pipeline for international market selection</strong><br/>
  Engineered for leveraged derivatives trading platforms — auditable, probabilistic, and production-ready.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Algorithm-TOPSIS--RAD-0EA5E9?style=flat-square&labelColor=0F172A" />
  <img src="https://img.shields.io/badge/Algorithm-Bayesian%20BWM-6366F1?style=flat-square&labelColor=0F172A" />
  <img src="https://img.shields.io/badge/Algorithm-CRITIC--M-10B981?style=flat-square&labelColor=0F172A" />
  <img src="https://img.shields.io/badge/Algorithm-SMAA-F59E0B?style=flat-square&labelColor=0F172A" />
  <img src="https://img.shields.io/badge/tested%20with-pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white" />
  <img src="https://img.shields.io/badge/license-MIT-22C55E?style=flat-square" />
</p>

---

## 🚀 Overview

**TOPSIS** is a next-generation **Multi-Criteria Decision Making (MCDM)** pipeline engineered to rank and sequence international markets for a **leveraged derivatives trading platform** (India-domiciled).

It fuses:
- 🤖 **LLM-powered De Jure regulatory extraction** — structured JSON from raw legal/regulatory text
- 🎲 **Bayesian BWM (Best-Worst Method)** — probabilistic weight elicitation from expert pairwise comparisons
- 📊 **CRITIC-M** — objective, variance-based criteria weighting from live data
- 🛡️ **TOPSIS-RAD** — rank-reversal-resistant scoring (a robust extension of classic TOPSIS)
- 🔬 **SMAA** — Stochastic Multicriteria Acceptability Analysis for confidence bands over rankings
- 🎒 **Knapsack Portfolio Optimiser** — constraint-aware market portfolio selection

...into a single, fully **auditable** pipeline with live data ingestion, staleness detection, and Monte Carlo robustness simulation.

---

## ✨ Features

| Feature | Description |
|---|---|
| **TOPSIS-RAD** | Rank-reversal-resistant TOPSIS variant — stable under market addition/removal |
| **Bayesian BWM** | MCMC-based probabilistic weight elicitation; outputs full posterior distributions |
| **CRITIC-M** | Objective criteria weighting using correlation-adjusted standard deviations |
| **SMAA** | Monte Carlo stochastic acceptability; produces rank acceptability indices & confidence bands |
| **De Jure LLM Pipeline** | Extracts structured regulatory attributes (leverage limits, margin rules, tax treaties) via LLM |
| **Knockout Filter** | Hard constraint elimination before scoring (e.g. regulatory bans, liquidity floors) |
| **Knapsack Optimiser** | Selects optimal market portfolio under capital/operational constraints |
| **Staleness Detection** | Flags and quarantines data feeds that breach freshness thresholds |
| **Live Data Feeds** | World Bank API, Governance Indices, real-time market data scrapers |
| **Monte Carlo Robustness** | Full robustness simulation over weight and data uncertainty |
| **Web Console** | Browser-based UI for running queries and visualising ranked outputs |

---

## 🏗️ Architecture

```
acharyavaibhav764-ops/TOPSIS/
│
├── ranking_engine/          # Core MCDM Python package
│   ├── topsis_rad.py        # Rank-reversal-resistant TOPSIS
│   ├── bayesian_bwm.py      # Bayesian Best-Worst Method (MCMC)
│   ├── critic_m.py          # CRITIC-M objective weighting
│   ├── smaa.py              # Stochastic Multicriteria Acceptability Analysis
│   ├── knockout.py          # Hard-constraint knockout filter
│   ├── knapsack.py          # Portfolio optimisation (Knapsack)
│   ├── de_jure_llm.py       # LLM regulatory extraction pipeline
│   ├── staleness.py         # Data freshness / staleness detection
│   └── monte_carlo.py       # Robustness Monte Carlo simulation
│
├── data_feeds/              # Live data scrapers & API clients
│   ├── world_bank.py        # World Bank Open Data API client
│   ├── governance.py        # Governance / WGI index feed
│   └── market_data.py       # Real-time market data feed
│
├── tools/                   # CLI utilities
│   ├── pipeline_runner.py   # End-to-end pipeline executor
│   ├── endpoint_probe.py    # Data source health checks
│   └── data_refresh.py      # Scheduled feed refresh utility
│
├── console/                 # Web console (HTML/CSS/JS)
├── index.html               # Entry point for the web UI
└── tests/                   # Test suites v0–v9
```

---

## 🔄 Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│          Live Data Feeds                                     │
│  World Bank · Governance Indices · Real-Time Market APIs     │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  De Jure LLM        │  ← Regulatory text in
              │  Extraction         │  → Structured JSON out
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Staleness Check    │  ← Flag stale / quarantine
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Knockout Filter    │  ← Hard constraint gates
              │  (hard constraints) │    (banned, illiquid, etc.)
              └──────────┬──────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
  ┌────────▼────────┐       ┌──────────▼────────┐
  │  Bayesian BWM   │       │    CRITIC-M        │
  │  (subjective    │       │    (objective      │
  │   weights)      │       │     weights)       │
  └────────┬────────┘       └──────────┬─────────┘
           │                           │
           └─────────────┬─────────────┘
                         │
                 Fused weight vector
                         │
              ┌──────────▼──────────┐
              │    TOPSIS-RAD       │  ← Rank-reversal resistant
              │    Scoring          │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  SMAA               │  ← Monte Carlo robustness
              │  Robustness         │    Rank acceptability indices
              │  Analysis           │    Confidence bands
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Knapsack Portfolio │  ← Constraint-aware
              │  Optimiser          │    market selection
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────────────────┐
              │  Ranked Market Shortlist         │
              │  + Confidence Bands              │
              │  + Audit Trail (JSON)            │
              └──────────────────────────────────┘
```

---

## 📦 Installation

> **Requirements:** Python 3.12+

```bash
# Clone the repository
git clone https://github.com/acharyavaibhav764-ops/TOPSIS.git
cd TOPSIS

# (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the full pipeline

```bash
python tools/pipeline_runner.py
```

### Refresh data feeds

```bash
python tools/data_refresh.py
```

### Probe data source health

```bash
python tools/endpoint_probe.py
```

### Web Console

Open `index.html` in your browser, or serve via:

```bash
python -m http.server 8080
# → http://localhost:8080
```

### Run tests

```bash
pytest tests/
```

---

## 🔌 API / Module Reference

### `ranking_engine`

| Module | Class / Function | Description |
|---|---|---|
| `topsis_rad` | `TopsisRAD` | Rank-reversal-resistant TOPSIS scorer |
| `bayesian_bwm` | `BayesianBWM` | MCMC weight elicitation from pairwise comparisons |
| `critic_m` | `CriticM` | Objective criteria weighting |
| `smaa` | `SMAA` | Stochastic rank acceptability analysis |
| `knockout` | `KnockoutFilter` | Hard-constraint market eliminator |
| `knapsack` | `KnapsackOptimiser` | Portfolio selection under constraints |
| `de_jure_llm` | `DeJurePipeline` | LLM-based regulatory attribute extraction |
| `staleness` | `StalenessDetector` | Data freshness monitoring & quarantine |
| `monte_carlo` | `RobustnessSim` | Monte Carlo robustness simulation |

### `data_feeds`

| Module | Description |
|---|---|
| `world_bank` | World Bank Open Data API client |
| `governance` | Governance / WGI index scraper |
| `market_data` | Real-time market data feed client |

---

## 🧩 Design Notes

### Why TOPSIS-RAD?
Classic TOPSIS suffers from **rank reversal** — adding or removing an alternative can change the relative ranking of all others. TOPSIS-RAD addresses this with a reference-alternative design that keeps rankings stable under market universe changes.

### Why Bayesian BWM?
Traditional BWM yields point estimates for weights. Bayesian BWM treats expert preferences as likelihood functions, producing full **posterior distributions** over weights. This enables uncertainty propagation into SMAA.

### Why CRITIC-M?
Pure subjective weighting ignores information already present in the data. CRITIC-M extracts **objective weight signals** from the variance and inter-correlation structure of the criteria matrix, complementing expert-elicited weights.

### Why SMAA?
Deterministic TOPSIS gives a single ranking with no uncertainty quantification. SMAA runs thousands of Monte Carlo trials over the joint weight uncertainty (from Bayesian BWM posteriors) to produce **rank acceptability indices** — the probability that each market occupies each rank position.

### De Jure vs. De Facto
Regulatory data exists in unstructured text (legislation, circulars, exchange rules). The **De Jure LLM pipeline** systematically extracts structured attributes (leverage caps, margin requirements, withholding tax treaties, short-selling rules) from raw documents, enabling regulatory criteria to be first-class inputs to the MCDM pipeline.

---

## 📄 License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

<p align="center">
  Built with rigour · Designed for decision-making under uncertainty
</p>