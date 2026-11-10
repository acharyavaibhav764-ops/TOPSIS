# TOPSIS

<p align="center">
  <img src="https://img.shields.io/badge/TOPSIS-Market%20Ranking%20Engine-0EA5E9?style=for-the-badge&labelColor=0F172A" alt="TOPSIS Market Ranking Engine" />
</p>

<p align="center">
  <strong>Next-generation Multi-Criteria Decision Making pipeline for international market selection.</strong><br/>
  Engineered for leveraged derivatives trading platforms -- auditable, probabilistic, and production-ready.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Algorithm-TOPSIS--RAD-0EA5E9?style=flat-square&labelColor=0F172A" />
  <img src="https://img.shields.io/badge/Algorithm-Bayesian%20BWM-6366F1?style=flat-square&labelColor=0F172A" />
  <img src="https://img.shields.io/badge/Algorithm-CRITIC--M-10B981?style=flat-square&labelColor=0F172A" />
  <img src="https://img.shields.io/badge/Algorithm-SMAA-F59E0B?style=flat-square&labelColor=0F172A" />
  <img src="https://img.shields.io/badge/tested%20with-pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white" />
</p>

---

## Overview

**TOPSIS** is a next-generation **Multi-Criteria Decision Making (MCDM)** pipeline engineered to rank and sequence international markets for a **leveraged derivatives trading platform** (India-domiciled).

It fuses LLM-powered regulatory extraction (De Jure pipeline), probabilistic weight elicitation (Bayesian BWM), objective variance weighting (CRITIC-M), rank-reversal-resistant scoring (TOPSIS-RAD), and stochastic acceptability analysis (SMAA) into a single, fully auditable pipeline with live data ingestion, staleness detection, and Monte Carlo robustness simulation.

---

## Features

| Feature | Description |
|---|---|
| **TOPSIS-RAD** | Rank-reversal-resistant TOPSIS variant -- stable under market addition/removal |
| **Bayesian BWM** | MCMC-based probabilistic weight elicitation; outputs full posterior distributions |
| **CRITIC-M** | Objective criteria weighting using correlation-adjusted standard deviations |
| **SMAA** | Monte Carlo stochastic acceptability; produces rank acceptability indices and confidence bands |
| **De Jure LLM Pipeline** | Extracts structured regulatory attributes (leverage limits, margin rules, tax treaties) via LLM |
| **Knockout Filter** | Hard constraint elimination before scoring (regulatory bans, liquidity floors, etc.) |
| **Knapsack Optimiser** | Selects optimal market portfolio under capital and operational constraints |
| **Staleness Detection** | Flags and quarantines data feeds that breach freshness thresholds |
| **Live Data Feeds** | World Bank API, Governance Indices, real-time market data scrapers |
| **Monte Carlo Robustness** | Full robustness simulation over weight and data uncertainty |
| **Web Console** | Browser-based UI for running queries and visualising ranked outputs |

---

## Architecture

```
TOPSIS/
|-- ranking_engine/          # Core MCDM Python package
|   |-- topsis_rad.py        # Rank-reversal-resistant TOPSIS
|   |-- weighting_bwm.py     # Bayesian Best-Worst Method (MCMC)
|   |-- weighting_bwm_bayes.py
|   |-- weighting_critic_m.py# CRITIC-M objective weighting
|   |-- robustness_smaa.py   # Stochastic Multicriteria Acceptability Analysis
|   |-- robustness_mc.py     # Monte Carlo robustness simulation
|   |-- knockout.py          # Hard-constraint knockout filter
|   |-- knapsack.py          # Portfolio optimisation (Knapsack)
|   |-- dejure_pipeline.py   # LLM regulatory extraction pipeline
|   |-- staleness.py         # Data freshness / staleness detection
|   |-- pipeline.py          # End-to-end pipeline orchestrator
|   |-- criterion_mapping.py # Criteria schema and mappings
|   |-- results_store.py     # Results persistence
|   `-- tests/               # Test suites v0 through v9
|
|-- data_feeds/              # Live data scrapers and API clients
|   |-- worldbank_client.py  # World Bank Open Data API client
|   |-- governance_index_client.py
|   |-- scoring.py
|   `-- raw_fact_adapter.py
|
|-- tools/                   # CLI utilities
|   |-- run_pipeline.py      # End-to-end pipeline executor
|   |-- run_full_pipeline.py
|   |-- refresh_data_feeds.py# Scheduled feed refresh utility
|   `-- live_endpoint_check.py
|
|-- console/                 # Web console (HTML/CSS/JS)
|-- index.html               # Entry point for the web UI
`-- requirements.txt
```

---

## Pipeline

```
+---------------------------------------------------------------+
|                      Live Data Feeds                          |
|   World Bank  .  Governance Indices  .  Real-Time Market APIs |
+-------------------------------+-------------------------------+
                                |
                                v
                   +------------+------------+
                   |   De Jure LLM           |
                   |   Extraction Pipeline   |  Regulatory text in
                   |                         |  Structured JSON out
                   +------------+------------+
                                |
                                v
                   +------------+------------+
                   |   Staleness Detection   |  Flag stale / quarantine
                   +------------+------------+
                                |
                                v
                   +------------+------------+
                   |   Knockout Filter       |  Hard constraint gates
                   |   (hard constraints)    |  Banned, illiquid, etc.
                   +------------+------------+
                                |
               +----------------+----------------+
               |                                 |
    +----------+----------+          +-----------+-----------+
    |   Bayesian BWM      |          |     CRITIC-M          |
    |   Subjective        |          |     Objective         |
    |   weight elicitation|          |     variance weights  |
    +----------+----------+          +-----------+-----------+
               |                                 |
               +-----------------+---------------+
                                 |
                         Fused weight vector
                                 |
                                 v
                   +------------+------------+
                   |     TOPSIS-RAD          |  Rank-reversal resistant
                   |     Scoring             |  scoring engine
                   +------------+------------+
                                |
                                v
                   +------------+------------+
                   |   SMAA Robustness       |  Monte Carlo simulation
                   |   Analysis              |  Rank acceptability indices
                   |                         |  Confidence bands
                   +------------+------------+
                                |
                                v
                   +------------+------------+
                   |   Knapsack Portfolio    |  Constraint-aware
                   |   Optimiser             |  market selection
                   +------------+------------+
                                |
                                v
                   +-----------------------------------+
                   |   Ranked Market Shortlist         |
                   |   + Confidence Bands              |
                   |   + Full Audit Trail (JSON)       |
                   +-----------------------------------+
```

---

## Installation

```bash
git clone https://github.com/acharyavaibhav764-ops/TOPSIS.git
cd TOPSIS

python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline

```bash
python tools/run_pipeline.py
```

### Refresh data feeds

```bash
python tools/refresh_data_feeds.py
```

### Check data source health

```bash
python tools/live_endpoint_check.py
```

### Web Console

Open `index.html` in your browser or serve it locally:

```bash
python -m http.server 8080
# -> http://localhost:8080
```

### Run tests

```bash
pytest ranking_engine/tests/
```

---

## Module Reference

| Module | Description |
|---|---|
| `topsis_rad` | Rank-reversal-resistant TOPSIS scorer |
| `weighting_bwm_bayes` | MCMC Bayesian BWM weight elicitation |
| `weighting_critic_m` | Objective CRITIC-M criteria weighting |
| `robustness_smaa` | Stochastic rank acceptability analysis |
| `robustness_mc` | Monte Carlo robustness simulation |
| `knockout` | Hard-constraint market eliminator |
| `knapsack` | Portfolio selection under constraints |
| `dejure_pipeline` | LLM-based regulatory attribute extraction |
| `staleness` | Data freshness monitoring and quarantine |
| `pipeline` | End-to-end pipeline orchestrator |

---

## Design Notes

**Why TOPSIS-RAD?**
Classic TOPSIS suffers from rank reversal -- adding or removing one alternative can reshuffle all others. TOPSIS-RAD uses reference alternatives to keep rankings stable under market universe changes.

**Why Bayesian BWM?**
Traditional BWM yields point estimates for weights. Bayesian BWM treats expert preferences as likelihood functions, producing full posterior distributions over weights -- enabling uncertainty propagation into SMAA.

**Why CRITIC-M?**
Pure subjective weighting ignores information already present in the data. CRITIC-M extracts objective weight signals from the variance and inter-correlation structure of the criteria matrix, complementing expert-elicited weights.

**Why SMAA?**
Deterministic TOPSIS gives a single ranking with no uncertainty quantification. SMAA runs thousands of Monte Carlo trials over the joint weight uncertainty to produce rank acceptability indices -- the probability that each market occupies each rank position.

**De Jure vs. De Facto**
Regulatory data exists in unstructured text (legislation, circulars, exchange rules). The De Jure LLM pipeline systematically extracts structured attributes from raw documents, making regulatory criteria first-class inputs to the MCDM pipeline.

---

<p align="center">
  Built with rigour &middot; Designed for decision-making under uncertainty
</p>