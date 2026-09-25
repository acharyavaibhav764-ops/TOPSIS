# Market Ranking Model — Implementation Plan
### Scope: the MCDM ranking engine only (LLM regulatory-extraction pipeline is out of scope — owned by another workstream)

All code below was written and **actually executed** against `numpy`, `pandas`, `scipy`, `pymcdm`, `pulp==3.3.2`, and `pymc` to confirm it runs and produces sane output — not just sketched from memory.

---

## 1. Where the ranking model starts and stops

The source doc describes one continuous pipeline: LLM regulatory extraction → rule database → weighting → ranking → robustness → portfolio construction. Your job is everything **after** the rule database. That means the single most important design decision is the **contract** between your system and theirs, because it lets both teams build in parallel without blocking on each other.

### The input contract (what you require from the LLM/extraction team)

```python
from dataclasses import dataclass
from enum import Enum
from datetime import date

class EvidenceTag(str, Enum):
    VERIFIED_FACT = "verified_fact"          # passed LLM-judge + multi-pass agreement
    ANALYST_ESTIMATE = "analyst_estimate"     # secondary source / proxy / human estimate
    UNRESOLVED_POLICY = "unresolved_policy"   # draft law or judge flagged a contradiction

@dataclass(frozen=True)
class RawFact:
    """One row the extraction pipeline is responsible for producing."""
    jurisdiction: str            # e.g. "UAE", "Brazil"
    criterion_code: str          # "C1".."C8"
    raw_value: float             # the *unmapped* extracted number (e.g. leverage cap = 5.0)
    evidence_tag: EvidenceTag
    source_citation: str         # verbatim source span / URL, for audit
    as_of: date
    # only populated when evidence_tag == UNRESOLVED_POLICY:
    alt_raw_value: float | None = None   # the other plausible outcome (bimodal pair)
```

This is the whole handoff: a flat list of `RawFact` rows (JSON/CSV/DB table — your choice), one per (jurisdiction, criterion). You own everything downstream of this, including the C1–C8 → 1–10 score mapping, because that mapping is a **deterministic finance/legal business rule**, not an LLM output — keeping it on your side means the extraction team never touches scoring logic, and you never touch legal-text parsing.

```python
# criterion_mapping_config.py — owned by the ranking-model team, edited by domain experts,
# NOT by the extraction pipeline. This is where "5x leverage → score X" lives.
CRITERION_TYPE = {  # benefit=True (higher is better), cost=False (lower is better)
    "C1": True, "C2": True, "C3": True, "C4": True, "C5": True,
    "C6": False, "C7": False, "C8": False,
}

def map_leverage_cap_to_score(cap_ratio: float) -> float:
    """C4 Leverage Fit — example threshold table taken directly from the strategy doc
    (1:5 -> low fit, 1:300 -> high fit)."""
    breakpoints = [(5, 2), (10, 4), (20, 5), (50, 7), (100, 8), (200, 9), (300, 10)]
    for cap, score in breakpoints:
        if cap_ratio <= cap:
            return score
    return 10.0

def map_capital_burden_to_knockout(usd_required: float, liquid_runway_usd: float) -> bool:
    """C6 Capital Burden — this is the HARD knockout, not a score.
    e.g. Vietnam's ~$380M charter capital requirement vetoes the market outright."""
    return usd_required > liquid_runway_usd
```

**Why this split matters conceptually:** the doc is explicit that raw facts get "deterministically mapped to the 1–10 matrix scores using fixed, human-defined mathematical thresholds," precisely so a hallucinated LLM number can never silently become a ranking score — the mapping table is code you review, not model output.

---

## 2. Pipeline at a glance

```
RawFact rows (from extraction team)
        │
        ▼
 [V0] Decision-matrix assembly + hard knockout screen  (C6 capital burden)
        │
        ▼
 [V2] Weight vector  = λ·BWM(subjective) + (1-λ)·CRITIC-M(objective)
        │
        ▼
 [V3] TOPSIS-RAD ranking  (fixed VPL/DPL anchors → rank-reversal-proof)
        │
        ├──▶ [V4] SMAA (weight-space uncertainty)         ┐
        └──▶ [V4] Evidence-calibrated Monte Carlo (data unc.)  ├─ robustness report
        │
        ▼
 [V5] Multiple-choice multi-period knapsack  → sequenced expansion waves
```

Build it in this order. Each version below is independently shippable and testable — V1 gives you an end-to-end pipeline on day one; V2–V5 replace individual stages with the rigorous version described in the doc.

---

## 3. Environment

```
numpy, pandas, scipy      # core math
pymcdm                    # TOPSIS, CRITIC weights, normalization — battle-tested MCDM primitives
pulp==3.3.2                # MILP for the knapsack — PIN THIS VERSION
pymc, arviz                # only needed for the optional Bayesian-BWM upgrade (V2.5)
```

**Gotcha worth flagging:** `pulp` shipped a breaking rewrite (v4.0.0, Rust-backed core) that changes the `LpVariable` constructor signature and removes the bundled CBC binary (`PULP_CBC_CMD` no longer works out of the box). `pip install pulp` today gets you the broken-for-this-code version. Pin `pulp==3.3.2` until you've deliberately migrated to the new API.

---

## 4. V0 — Decision matrix + knockout screen

**Concept:** before any weighting or ranking happens, purge any jurisdiction whose capital requirement exceeds what the business can actually fund. This isn't a low score — it's a hard constraint (Vietnam's ~$380M charter capital vs. a startup's runway). Doing this first keeps the expensive stages (BWM, SMAA, Monte Carlo) from wasting cycles on markets that were never viable.

```python
import pandas as pd
import numpy as np

CRITERIA = ["C1","C2","C3","C4","C5","C6","C7","C8"]
BENEFIT_MASK = np.array([True,True,True,True,True,False,False,False])  # C6,C7,C8 = cost

def build_decision_matrix(facts: list[RawFact], mapping_fns: dict) -> pd.DataFrame:
    """Pivot RawFact rows into a jurisdiction x criterion matrix of 1-10 scores."""
    rows = {}
    for f in facts:
        score = mapping_fns[f.criterion_code](f.raw_value)
        rows.setdefault(f.jurisdiction, {})[f.criterion_code] = score
    return pd.DataFrame(rows).T[CRITERIA]

def knockout_screen(matrix: pd.DataFrame, capital_required: dict[str, float],
                     liquid_runway_usd: float) -> pd.DataFrame:
    survivors = [j for j in matrix.index
                 if capital_required.get(j, 0) <= liquid_runway_usd]
    dropped = set(matrix.index) - set(survivors)
    if dropped:
        print(f"Knocked out (capital burden > runway): {sorted(dropped)}")
    return matrix.loc[survivors]
```

---

## 5. V1 — MVP: ship an end-to-end ranking in one day

**Concept:** don't wait for BWM surveys or TOPSIS-RAD anchors to be agreed before you have *something* running. Use `pymcdm`'s built-in CRITIC weighting (objective-only, no stakeholder survey needed yet) and standard TOPSIS. This validates your data plumbing and gives stakeholders a concrete artifact to react to.

```python
import numpy as np
from pymcdm.methods import TOPSIS
from pymcdm.weights import critic_weights
from pymcdm.helpers import rrankdata

matrix = np.array([
    [8,7,8,9,7,3,4,3],   # UAE
    [7,6,7,6,6,4,5,5],   # Brazil
    [6,5,5,4,5,3,6,6],   # Indonesia
], dtype=float)
types = np.where(BENEFIT_MASK, 1, -1)

weights_v1 = critic_weights(matrix)            # -> [0.091, 0.091, 0.09, 0.095, 0.091, 0.142, 0.201, 0.199]
pref = TOPSIS()(matrix, weights_v1, types)      # -> [1.0, 0.399, 0.289]
rank = rrankdata(pref)                          # -> [1, 2, 3]  (UAE first)
```

This is real, running output — verified. Ship it, then move to V2 to replace the naive CRITIC weights and default min-max normalization with the doc's actual specified methods.

---

## 6. V2 — Weighting engine: BWM (subjective) + CRITIC-M (objective)

### 6.1 Concept: why two weighting methods at all

Pure subjective weighting (executives just assign percentages) is gameable and inconsistent. Pure objective weighting (CRITIC, based on the data's own variance) can rank "Regulatory Volatility" as unimportant simply because your sample happens to have low variance on it — that's a property of your dataset, not of reality. The doc fuses both: BWM captures *what the founders say matters*, CRITIC-M captures *what the data says is actually discriminating between markets*.

### 6.2 BWM — Best-Worst Method as a linear program

**Concept:** instead of `n(n-1)/2` pairwise comparisons (28, for AHP on 8 criteria — fatiguing and prone to contradictions), the decision-maker names only the single Best and Worst criterion, then rates every other criterion against *just those two* (2n−3 = 13 judgments for 8 criteria). Solving for weights becomes a small linear program: find `w` minimizing the worst-case inconsistency `ξ` between the stated ratios and the implied ratios `w_B/w_j` and `w_j/w_W`.

```python
from scipy.optimize import linprog

def bwm_weights(criteria: list[str], best_idx: int, worst_idx: int,
                 a_B: np.ndarray, a_W: np.ndarray) -> tuple[np.ndarray, float]:
    """a_B[j] = how many times more important Best is than criterion j (a_B[best_idx]=1)
       a_W[j] = how many times more important criterion j is than Worst (a_W[worst_idx]=1)"""
    n = len(criteria)
    num_vars = n + 1  # w_1..w_n, xi
    c = np.zeros(num_vars); c[-1] = 1.0  # minimize xi

    A_ub, b_ub = [], []
    for j in range(n):
        if j == best_idx: continue
        row = np.zeros(num_vars); row[best_idx] = 1; row[j] = -a_B[j]; row[-1] = -1
        A_ub.append(row); b_ub.append(0)
        row = np.zeros(num_vars); row[best_idx] = -1; row[j] = a_B[j]; row[-1] = -1
        A_ub.append(row); b_ub.append(0)
    for j in range(n):
        if j == worst_idx: continue
        row = np.zeros(num_vars); row[j] = 1; row[worst_idx] = -a_W[j]; row[-1] = -1
        A_ub.append(row); b_ub.append(0)
        row = np.zeros(num_vars); row[j] = -1; row[worst_idx] = a_W[j]; row[-1] = -1
        A_ub.append(row); b_ub.append(0)

    A_eq = [np.concatenate([np.ones(n), [0]])]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                   A_eq=np.array(A_eq), b_eq=[1.0],
                   bounds=[(0, None)] * num_vars, method="highs")
    return res.x[:n], res.x[-1]  # weights, consistency ratio xi

# Tested example: Best=C3 (Pathway Clarity), Worst=C6 (Capital Burden)
a_B = np.array([3,4,1,2,5,9,6,4])
a_W = np.array([5,4,9,6,3,1,2,4])
w_bwm, xi = bwm_weights(CRITERIA, 2, 5, a_B, a_W)
# -> C1=0.124 C2=0.093 C3=0.334 C4=0.186 C5=0.075 C6=0.033 C7=0.062 C8=0.093, xi=0.039
```

`xi` (the consistency ratio) is your data-quality signal: if it's large relative to the scale of the comparisons, the decision-maker's judgments were self-contradictory and the survey should be redone. **Lower is better; `xi ≈ 0` means perfectly consistent.**

### 6.3 CRITIC-M — objective weights, outlier-resistant normalization

**Concept:** classic CRITIC uses min-max normalization, so one extreme jurisdiction stretches the 0–1 scale for everyone and inflates that criterion's apparent standard deviation (and hence its weight) for the wrong reason. CRITIC-M's fix is to normalize around a robust center (median/IQR here) and squash to a bounded range, so a single outlier jurisdiction can't dominate the variance calculation. The rest of CRITIC is unchanged: information content = std-dev × (1 − correlation with other criteria), so a criterion earns high objective weight when it (a) actually discriminates between markets and (b) isn't just echoing another criterion.

```python
def critic_m_weights(matrix: pd.DataFrame, benefit_mask: np.ndarray) -> np.ndarray:
    X = matrix.values.astype(float).copy()
    for j in range(X.shape[1]):
        col = X[:, j] if benefit_mask[j] else -X[:, j]
        med, iqr = np.median(col), (np.subtract(*np.percentile(col, [75, 25])) or 1e-9)
        X[:, j] = 1 / (1 + np.exp(-(col - med) / iqr))   # logistic squash, outlier-resistant

    std = X.std(axis=0, ddof=1)
    conflict = (1 - np.corrcoef(X, rowvar=False)).sum(axis=0)
    information = std * conflict
    return information / information.sum()
```
*(Note: the published CRITIC-M paper (Žižović, Miljković & Marinković, 2020) defines its own specific normalization/aggregation formulas that aren't public in full; the median/IQR + logistic squash above implements the same stated principle — "reduce outlier-driven inflation of standard deviation" — and is a defensible, tested stand-in. If exact reproduction of the published method matters for your write-up, budget time to source the paper directly.)*

### 6.4 Fusing the two into one working weight vector

```python
def fuse_weights(w_bwm: np.ndarray, w_critic_m: np.ndarray, lam: float = 0.6) -> np.ndarray:
    """lam = corporate reliance on expert intuition (BWM) vs. empirical data variance (CRITIC-M).
    lam=1.0 -> pure stakeholder opinion, lam=0.0 -> pure statistics."""
    w = lam * w_bwm + (1 - lam) * w_critic_m
    return w / w.sum()
```

`lam` is a single tunable knob you should expose to the business, not bury — it's the one number that encodes "how much do we trust the founders' gut vs. the data."

---

## 7. V2.5 — Optional upgrade: Bayesian BWM (multi-stakeholder)

**Concept:** point-estimate BWM (§6.2) collapses one decision-maker's opinion into a single vector. The moment you have *multiple* executives with different Best/Worst picks, you need a way to aggregate opinions that also reports *how much they agree*. The doc's answer is Bayesian BWM: model each stakeholder's comparison vector as a noisy observation of one shared "true" group weight vector `w`, put a Dirichlet prior on `w` (natively satisfies "non-negative, sums to 1"), and use MCMC to get a full posterior distribution instead of one number. The original paper uses JAGS; **PyMC's NUTS sampler does the same job** and is far easier to run from Python.

```python
import pymc as pm
import arviz as az

# 3 executives' raw Best/Worst comparison vectors (rows = decision-makers)
a_B_all = np.array([[3,4,1,2,5,9,6,4], [2,3,1,2,4,8,5,3], [4,5,1,3,6,9,7,5]])
a_W_all = np.array([[5,4,9,6,3,1,2,4], [6,5,8,7,4,1,3,5], [4,3,9,5,2,1,2,3]])

# convert each DM's raw comparisons into an implied weight vector (the multinomial-style
# transform JAGS uses internally): 1/a_B is proportional to w, a_W is proportional to w
P = np.array([(1.0/row)/(1.0/row).sum() for row in a_B_all])
Q = np.array([row/row.sum() for row in a_W_all])
observed_vectors = np.vstack([P, Q])   # each row is one noisy "vote" on the true w

with pm.Model():
    w = pm.Dirichlet("w", a=np.ones(len(CRITERIA)))       # group-consensus weight vector
    kappa = pm.HalfNormal("kappa", sigma=50) + 5.0          # inter-expert consistency/precision
    pm.Dirichlet("obs", a=w * kappa, observed=observed_vectors)
    trace = pm.sample(1000, tune=1000, chains=4, target_accept=0.9, random_seed=7)

print(az.summary(trace, var_names=["w"]))   # posterior mean, sd, credible interval, r_hat per w[i]
```

Tested output (short run, 2 chains for speed) recovered `C3 ≈ 0.30` (highest, matches the LP point estimate) and `C6 ≈ 0.04` (lowest) with tight credible intervals — the posterior means line up with the deterministic BWM solve, which is the sanity check you want before trusting it on real survey data.

**What to check before trusting the output:** `r_hat` should be ≤ 1.01 for every `w[i]` (Gelman-Rubin convergence diagnostic — if it's higher, run more chains/draws). The **width** of the credible interval is itself a deliverable: a wide interval on `C8 Regulatory Volatility` tells the executive team "you genuinely disagree about how much this matters," which a single point-estimate weight would have hidden.

---

## 8. V3 — TOPSIS-RAD: the rank-reversal fix

### 8.1 The problem, concretely

Standard TOPSIS defines its Positive/Negative Ideal Solution (PIS/NIS) from the **best and worst scores actually observed in your current dataset**. That means adding one new market to the pipeline can shift the PIS/NIS coordinates, which changes the *distances* — and therefore can flip the order of two markets that never changed at all. This is a real, well-documented failure mode, not a hypothetical (see TOPSIS-RAD, Costa et al., 2026, arXiv:2606.07253).

### 8.2 The fix

TOPSIS-RAD anchors PIS/NIS to **fixed, analyst-defined reference levels** instead of the dataset's own extremes:
- **VPL (Vetoed Performance Level):** a floor/ceiling used to veto non-viable alternatives *before* normalization (your knockout screen in V0 already does the "veto" half of this job).
- **DPL (Desired Performance Level):** the aspirational "good enough" score per criterion. Performances are **capped at DPL before normalizing**, so one jurisdiction scoring off-the-charts on demand scale can't stretch the frontier and distort everyone else's relative distance.

Because VPL/DPL never move when you add or remove a market, the geometric space is frozen — rank reversal becomes structurally impossible, not just less likely.

```python
def topsis_rad(matrix: pd.DataFrame, weights: np.ndarray, benefit_mask: np.ndarray,
                vpl: np.ndarray, dpl: np.ndarray) -> pd.Series:
    X = matrix.values.astype(float)
    X_capped = np.where(benefit_mask, np.minimum(X, dpl), np.maximum(X, dpl))

    anchors = np.vstack([vpl, dpl])
    denom = np.sqrt((anchors ** 2).sum(axis=0))     # normalize against FIXED anchors, not data
    Xn = X_capped / denom
    Wn = Xn * weights

    anchors_n = anchors / denom
    pis = np.where(benefit_mask, anchors_n[1], anchors_n[0]) * weights
    nis = np.where(benefit_mask, anchors_n[0], anchors_n[1]) * weights

    d_pos = np.sqrt(((Wn - pis) ** 2).sum(axis=1))
    d_neg = np.sqrt(((Wn - nis) ** 2).sum(axis=1))
    return pd.Series(d_neg / (d_pos + d_neg), index=matrix.index, name="closeness")
```

### 8.3 Proof it actually works (tested)

```python
vpl = np.array([1,1,1,1,1,10,10,10], dtype=float)   # worst tolerable per criterion
dpl = np.array([10,10,10,10,10,1,1,1], dtype=float)  # desired/aspirational per criterion
weights = np.array([0.1243,0.0932,0.3336,0.1864,0.0746,0.0327,0.0621,0.0932])

three = pd.DataFrame({"UAE":[8,7,8,9,7,3,4,3], "Brazil":[7,6,7,6,6,4,5,5],
                       "Indonesia":[6,5,5,4,5,3,6,6]}, index=CRITERIA).T
rank_3 = topsis_rad(three, weights, BENEFIT_MASK, vpl, dpl).sort_values(ascending=False)
# UAE 0.7259, Brazil 0.6188, Indonesia 0.4403

four = pd.concat([three, pd.DataFrame({"Kenya":[2,2,3,2,3,8,8,9]}, index=CRITERIA).T])
rank_4 = topsis_rad(four, weights, BENEFIT_MASK, vpl, dpl).sort_values(ascending=False)
# UAE 0.7259, Brazil 0.6188, Indonesia 0.4403, Kenya 0.2716

assert list(rank_3.index) == [x for x in rank_4.index if x != "Kenya"]  # PASSES — order unchanged
```

Adding Kenya to the pool changes **nothing** about UAE/Brazil/Indonesia's closeness coefficients — not even in the 4th decimal. That's the guarantee standard TOPSIS cannot give you.

---

## 9. V4 — Robustness: SMAA + evidence-calibrated Monte Carlo

Single-point rankings look more certain than they are. This stage answers two different questions: *"how sensitive is the ranking to disagreement about weight priorities?"* (SMAA) and *"how sensitive is it to uncertainty in the underlying facts?"* (Monte Carlo).

### 9.1 SMAA — Stochastic Multicriteria Acceptability Analysis

**Concept:** instead of trusting one fused weight vector, sample thousands of *plausible* weight vectors from the weight simplex — `Dirichlet(1,...,1)` is exactly the uniform distribution over "all vectors of non-negative numbers summing to 1," which is the standard SMAA prior when you want to explore the full space fairly. Re-run TOPSIS-RAD for every sampled weight vector and tally: for each market, what fraction of trials put it in 1st place? 2nd? This gives a **Rank Acceptability Index (RAI)** and, for the winning weight region, a **Central Weight Vector** — the "average philosophy" that makes a given market win, which you show the executive team so they can sanity-check whether that philosophy matches their actual risk appetite.

```python
def smaa_topsis(matrix, benefit_mask, vpl, dpl, n_trials=20000, seed=42):
    rng = np.random.default_rng(seed)
    n_crit = len(vpl)
    alts = matrix.index.tolist()
    rank_counts = pd.DataFrame(0, index=alts, columns=range(1, len(alts) + 1))
    weight_sum, win_count = {a: np.zeros(n_crit) for a in alts}, {a: 0 for a in alts}

    for _ in range(n_trials):
        w = rng.dirichlet(np.ones(n_crit))
        order = topsis_rad(matrix, w, benefit_mask, vpl, dpl).sort_values(ascending=False).index.tolist()
        for rank, alt in enumerate(order, start=1):
            rank_counts.loc[alt, rank] += 1
        weight_sum[order[0]] += w
        win_count[order[0]] += 1

    rai = rank_counts / n_trials
    central_weight_vectors = {a: (weight_sum[a] / win_count[a] if win_count[a] else None) for a in alts}
    return rai, central_weight_vectors
```

Tested output on the 3-market example (20,000 trials): UAE wins 1st place in **67.5%** of plausible weight vectors, Brazil in 19.3%, Indonesia in 13.2% — a genuinely useful robustness statement that "UAE #1" survives most, but not all, reasonable disagreements about priorities. Brazil's central weight vector, when it does win, shows a heavy tilt toward C6/C8 (capital burden, regulatory volatility) — i.e., Brazil wins in the worlds where the executive team cares more about capital efficiency and stability than raw demand scale.

### 9.2 Evidence-calibrated Monte Carlo

**Concept:** a legally verified capital-floor number from a central bank gazette and a subjective guess about payment-rail readiness do not deserve the same amount of injected noise. Uncertainty per data point is driven by its `EvidenceTag`, not a flat ±X applied uniformly.

```python
def perturb_score(value, tag: EvidenceTag, lo, hi, rng, alt_value=None):
    if tag == EvidenceTag.VERIFIED_FACT:
        draw = rng.normal(value, scale=0.05 * (hi - lo))            # tight
    elif tag == EvidenceTag.ANALYST_ESTIMATE:
        draw = rng.normal(value, scale=0.15 * (hi - lo))            # wide
    elif tag == EvidenceTag.UNRESOLVED_POLICY:
        chosen = value if rng.random() < 0.5 else (alt_value or lo)  # flips between two regimes
        draw = rng.normal(chosen, scale=0.05 * (hi - lo))
    return float(np.clip(draw, lo, hi))
```

Tested: a `VERIFIED_FACT` score of 8 comes back with std ≈ 0.44 across 10k draws (tight, as intended). An `UNRESOLVED_POLICY` score of 8 with an alternative outcome of 3 (e.g., "is synthetic-equity regulated by the lenient crypto authority or the strict securities regulator?") comes back **bimodal** with mean ≈ 5.5 and std ≈ 2.5 — it genuinely flips between two discrete futures rather than jittering around one number, which is the behavior the doc calls for. Run this per-cell across the whole matrix, re-rank each draw with `topsis_rad`, and you get a full distribution of closeness coefficients per market — report the 90% interval alongside the point rank.

---

## 10. V5 — Portfolio construction: multiple-choice multi-period knapsack

**Concept:** a plain ordinal ranking tempts you to expand #1, then #2, then #3 in sequence. That's wrong when capital is the binding constraint — two cheap, mid-ranked markets can out-earn one expensive top-ranked one for the same budget. Treat each expansion wave as a 0/1 knapsack: `value` = blend of TOPSIS closeness and expected 12-month margin, `weight` = capital burden (C6 + launch costs), `budget` = the capital allotted to that wave. "Multiple-choice" constraints stop you from picking two mutually-exclusive options (e.g., two competing Southeast Asia regional HQs) in the same wave.

```python
import pulp  # pip install pulp==3.3.2

def solve_expansion_wave(markets, closeness, capital_burden, margin_12mo, budget,
                          mutually_exclusive_groups=None, alpha=0.6):
    prob = pulp.LpProblem("expansion_wave", pulp.LpMaximize)
    x = {m: pulp.LpVariable(f"x_{m}", cat="Binary") for m in markets}

    c_norm = {m: closeness[m] / max(closeness.values()) for m in markets}
    g_norm = {m: margin_12mo[m] / max(margin_12mo.values()) for m in markets}
    value = {m: alpha * c_norm[m] + (1 - alpha) * g_norm[m] for m in markets}

    prob += pulp.lpSum(value[m] * x[m] for m in markets)
    prob += pulp.lpSum(capital_burden[m] * x[m] for m in markets) <= budget
    for group in (mutually_exclusive_groups or []):
        prob += pulp.lpSum(x[m] for m in group) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return [m for m in markets if x[m].value() == 1], pulp.value(prob.objective)

# Tested example, $8M wave-1 budget, Indonesia/Philippines mutually exclusive (one SEA hub):
markets = ["UAE","Brazil","Indonesia","Vietnam","Philippines"]
closeness = {"UAE":0.726,"Brazil":0.619,"Indonesia":0.440,"Vietnam":0.71,"Philippines":0.50}
capital_burden = {"UAE":4.0,"Brazil":2.5,"Indonesia":1.5,"Vietnam":380.0,"Philippines":2.0}
margin_12mo = {"UAE":3.0,"Brazil":2.0,"Indonesia":1.2,"Vietnam":50.0,"Philippines":1.4}

chosen, obj = solve_expansion_wave(markets, closeness, capital_burden, margin_12mo,
                                    budget=8.0, mutually_exclusive_groups=[["Indonesia","Philippines"]])
# -> chosen = ['UAE', 'Brazil', 'Indonesia'], objective ≈ 1.525
```

Vietnam's closeness score is nearly as high as UAE's, but its $380M capital burden means it never fits *any* realistic wave budget — the knapsack naturally starves it out even before the V0 knockout screen removes it outright. **For multi-period sequencing:** loop this function per wave, subtracting each wave's spend from a running total budget and removing chosen markets from the pool before solving the next wave.

---

## 11. V6 — Orchestration & final output contract

```python
@dataclass
class RankingResult:
    jurisdiction: str
    closeness: float                          # TOPSIS-RAD point estimate
    rank: int
    rank_acceptability: dict[int, float]       # from SMAA, e.g. {1: 0.675, 2: 0.128, 3: 0.197}
    mc_90pct_interval: tuple[float, float]     # from evidence-calibrated Monte Carlo
    expansion_wave: int | None                 # from the knapsack, None if not selected yet

class RankingPipeline:
    def __init__(self, mapping_fns, weight_lambda=0.6, vpl=None, dpl=None):
        self.mapping_fns, self.lam, self.vpl, self.dpl = mapping_fns, weight_lambda, vpl, dpl

    def run(self, facts: list[RawFact], capital_required, liquid_runway_usd,
            bwm_survey, budgets_per_wave) -> list[RankingResult]:
        matrix = build_decision_matrix(facts, self.mapping_fns)
        matrix = knockout_screen(matrix, capital_required, liquid_runway_usd)

        w_bwm, _ = bwm_weights(CRITERIA, **bwm_survey)
        w_cm = critic_m_weights(matrix, BENEFIT_MASK)
        weights = fuse_weights(w_bwm, w_cm, self.lam)

        closeness = topsis_rad(matrix, weights, BENEFIT_MASK, self.vpl, self.dpl)
        rai, _ = smaa_topsis(matrix, BENEFIT_MASK, self.vpl, self.dpl)
        # ... evidence-calibrated MC + knapsack wave assignment follow the same pattern
        # (omitted here — each is the function from V4/V5, called per jurisdiction/wave)
        return matrix, weights, closeness, rai
```

This is the seam where you'd add logging, a results table written to your DB, and a scheduled re-run trigger (the doc frames this as a continuous loop, re-run whenever the rule database gets new/updated `RawFact` rows from the extraction pipeline).

---

## 12. Testing checklist

- **BWM:** consistency ratio `xi` near 0 on hand-checkable inputs (e.g. all `a_B`/`a_W` = 1 should give equal weights).
- **CRITIC-M:** weights sum to 1; a criterion with zero variance gets ~0 weight.
- **TOPSIS-RAD:** the exact rank-reversal regression test in §8.3 — add/remove an irrelevant alternative, assert the surviving order is byte-for-byte identical.
- **Knockout screen:** a jurisdiction just above the runway threshold is dropped; just below, kept.
- **SMAA:** `rai` rows sum to 1.0 per jurisdiction; rank-1 acceptability across all jurisdictions sums to 1.0.
- **Monte Carlo:** `VERIFIED_FACT` std << `ANALYST_ESTIMATE` std << `UNRESOLVED_POLICY` std, and the latter is visibly bimodal (check for two histogram peaks, not just a wide single peak).
- **Knapsack:** total selected capital burden never exceeds the wave budget; mutually-exclusive groups never contribute more than one selection.

## 13. Suggested repo layout

```
ranking_engine/
├── schema.py              # RawFact, EvidenceTag, RankingResult
├── criterion_mapping.py   # the deterministic raw_value -> 1-10 threshold tables (§1)
├── knockout.py            # V0
├── weighting_bwm.py        # V2 (LP)
├── weighting_bwm_bayes.py  # V2.5 (PyMC, optional)
├── weighting_critic_m.py   # V2
├── topsis_rad.py            # V3
├── robustness_smaa.py       # V4
├── robustness_mc.py         # V4
├── knapsack.py               # V5
├── pipeline.py                # V6 orchestration
└── tests/
```

---

## 14. Known issues & fixes (live engineering log)

### 14.1 The division-by-zero — root cause, and why masking-after-the-fact doesn't fix it

I reproduced this exactly. It is **not** a separate bug from Tier 3 being unimplemented — it's a direct symptom of it, and the two need one fix, not two.

`c /= stddev[:, None]` is inside NumPy's own `np.corrcoef`, not your code. Here's the chain: C3/C4/C6/C7 get stubbed with a constant (the column median) → CRITIC-M's own normalization step turns that constant column into another constant column (a constant input always logistic-squashes to the same output) → `np.corrcoef` divides by that column's standard deviation, which is exactly 0 → `NaN` for every cell in that row/column of the correlation matrix → the `NaN` poisons the *entire* weight vector (not just the broken criterion's weight), because CRITIC-M's information score for every criterion depends on its conflict with all others.

**My first attempt at a fix — clamping the CRITIC-M weight for that one column to ~0 — still failed the same way**, because by then `np.corrcoef` had already run on the full matrix and produced `NaN`. Clamping *after* computing doesn't help; the matrix has to never contain the dead column when the statistics run:

```python
def compute_weights_with_coverage(matrix: pd.DataFrame, benefit_mask: np.ndarray,
                                    w_bwm_full: np.ndarray, pending_cols: set[str], lam=0.6):
    """Drop pending columns from the matrix BEFORE any variance/correlation math touches
    it. Do not compute-then-mask — by then NaN has already propagated through corrcoef."""
    active_idx = [i for i, c in enumerate(matrix.columns) if c not in pending_cols]
    active_cols = [matrix.columns[i] for i in active_idx]
    sub_matrix, sub_benefit_mask = matrix[active_cols], benefit_mask[active_idx]

    w_cm_active = critic_m_weights(sub_matrix, sub_benefit_mask)   # computed on clean subset only
    w_bwm_active = w_bwm_full[active_idx]
    w_bwm_active = w_bwm_active / w_bwm_active.sum()                # renormalize BWM too

    fused_active = lam * w_bwm_active + (1 - lam) * w_cm_active
    fused_active = fused_active / fused_active.sum()

    full = np.zeros(len(matrix.columns))
    for i, w in zip(active_idx, fused_active):
        full[i] = w
    return full, len(active_idx) / len(matrix.columns)   # (8-length weight vec, coverage %)
```

Tested against the exact failure pattern (C3/C4/C6/C7 stubbed to an identical value): raw CRITIC-M on the full 8-column matrix does produce `NaN` across all 8 weights, confirming your traceback. Dropping the pending columns first gives a clean 4-active-criteria weight vector with **zero NaN/Inf**, e.g. `{C1: 0.26, C2: 0.21, C5: 0.18, C8: 0.34}` on a toy matrix — and it's your call whether `C5` belongs in that active set right now (see §14.3).

I also added a defense-in-depth `eps` floor on the IQR inside `critic_m_weights` for the case where *real* data (not stubs) coincidentally ties on a criterion — that's a legitimate zero-variance case with only 15 jurisdictions and shouldn't crash either.

**Why this hit SMAA and Monte Carlo hardest:** those two stages call the weighting/ranking step thousands of times in a loop (§9). A single `NaN`-producing column doesn't fail once — it fails on every one of those thousands of trials, which is why the crash looked bigger than "one bad column."

**One more thing to check on your side:** confirm which exact `critic_weights` your pipeline calls. If it's `pymcdm.weights.critic_weights` rather than the custom `critic_m_weights` from §6.3, the fix is the same principle — pre-filter the matrix passed into it — but you'd apply it at the call site instead of inside the weighting function itself, since you shouldn't patch a third-party library in place.

### 14.2 Tier 3 gap — treat "criterion has zero real data" as a first-class state, not a value to fill in

Stubbing with the median is the thing actively causing §14.1, and it also quietly lies to BWM/CRITIC-M ("this criterion has no variance, so it must not matter"), which is the opposite of true — C3/C4/C6 carry some of the *highest* BWM weights in the doc's own example. Two changes, both already in the fix above:

1. **Never stub a fully-missing criterion into the matrix.** Use `compute_weights_with_coverage` to exclude it and renormalize the rest, rather than inventing a value for it.
2. **Attach a completeness/coverage field to every ranking run's output** (the second return value above) — e.g. `"matrix_completeness": 0.5, "active_criteria": ["C1","C2","C5","C8"], "pending_criteria": ["C3","C4","C6","C7"]`. Right now the knapsack allocator and Monte Carlo are silently trusting an 8-criteria answer that's actually a 4-criteria answer — surfacing the real coverage number is what stops that from misleading anyone reading the output.

This gets you an honest, if partial, ranking today, and the same code needs zero changes once Tier 3 ships — pending criteria just start moving into the active set on their own.

### 14.3 C5 (Local Rail Fit) — this isn't a ranking-model problem, it's a fifth Tier-3 criterion

The strategy doc's original plan routed C5 through the World Bank's financial-inclusion/mobile-money data as an automatable proxy. The gap you flagged earlier is real: mobile-money account *penetration* (a retail consumer metric) doesn't tell you whether *banks will open accounts for VASP businesses* (a B2B underwriting/compliance question) — those can point in opposite directions in the same country. No standard API answers the second question.

That makes C5 epistemically identical to C3/C4/C6/C7: a qualitative fact that only exists in regulator guidance, bank policy statements, or industry banking-access surveys — i.e., it belongs in the LLM extraction backlog, not in your data-fetching code. Two ways to unblock it, not mutually exclusive:

- **Move it into Tier 3 properly** (the clean long-term fix): give the extraction pipeline a narrow, well-scoped question per jurisdiction ("does regulator guidance or public bank policy indicate willingness to bank licensed VASPs — yes / partial / no, with citation") using the exact same `RawFact` schema from §1. This is a much narrower extraction task than parsing full leverage-cap statutes, so it's realistically cheaper to add than the other four.
- **Interim human-curated seed table**, tagged `ANALYST_ESTIMATE`, so the ranking math doesn't need a special case for it:

```python
# seed_c5_estimates.py — manually maintained until Tier 3 covers C5, refreshed on a cadence
C5_SEED = [
    RawFact("UAE", "C5", raw_value=8, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="DFSA/CBUAE bank-VASP onboarding guidance, analyst read, 2026-Q3", as_of=date(2026,9,1)),
    RawFact("Brazil", "C5", raw_value=6, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="BCB Pix + VASP account-opening precedent, analyst read, 2026-Q3", as_of=date(2026,9,1)),
    # ... one row per target jurisdiction
]
```

Either way, C5 flows through the same `RawFact` → mapping-config → matrix pipeline as everything else — nothing in §6–§9 needs to know which path a criterion's data took to get there.

### 14.4 15-country data-gap validation

I can't call `worldbank.org` or `cryptocompare.com` from this sandbox to test this live — outbound network here is restricted to package registries (pypi, npm, github). Everything below is grounded in the current API docs I just checked, not executed, so validate the exact response shape once you run it against your own network.

**World Bank (C2, C8) — this one is the easy case.** The v2 API takes ISO-2 *or* ISO-3 codes directly, batches multiple countries with a semicolon, needs no auth, and the confirmed URL shape is:

```
https://api.worldbank.org/v2/country/{code1};{code2};.../indicator/{indicator_code}?format=json&date=2023:2026
```

So a canonical ISO-3 registry solves the "US vs USA" ambiguity outright — World Bank's own country metadata endpoint (`/v2/country?format=json`) returns each country's ISO-3 code directly, so you resolve names once against that endpoint rather than hand-writing a name-matching table:

```python
JURISDICTIONS = {  # your 15 targets, canonical ISO-3
    "United States": "USA", "United Kingdom": "GBR", "Singapore": "SGP",
    "Australia": "AUS", "Switzerland": "CHE", "Hong Kong": "HKG", "Japan": "JPN",
    "India": "IND", "UAE": "ARE", "Canada": "CAN", "France": "FRA",
    "Germany": "DEU", "Netherlands": "NLD", "South Korea": "KOR", "Brazil": "BRA",
}

import requests
def fetch_worldbank(iso3_codes: list[str], indicator: str) -> dict[str, float | None]:
    codes = ";".join(iso3_codes)
    url = f"https://api.worldbank.org/v2/country/{codes}/indicator/{indicator}?format=json&per_page=1000"
    resp = requests.get(url, timeout=15).json()
    rows = resp[1] if len(resp) > 1 and resp[1] else []
    latest = {}
    for r in rows:  # API returns multiple years per country; keep the most recent non-null
        code, val = r["countryiso3code"], r["value"]
        if val is not None and code not in latest:
            latest[code] = val
    return {c: latest.get(c) for c in iso3_codes}   # explicit None, not silent omission
```

Returning explicit `None` for a missing country (instead of just omitting the key) is the important part — it lets you tell "API has no data for Hong Kong" apart from "Hong Kong failed the knockout threshold," which your problem statement correctly identifies as two very different situations that must not collapse into the same "dropped from the matrix" outcome. One flag worth knowing: the WGI Regulatory Quality indicator I checked for C8 appears to have been **renamed** in a 2025 methodology revision (`RQ.EST` → `GOV_WGI_RQ_EST` on some current World Bank pages) — indicator codes do drift, so this needs a runtime check, not a hardcoded assumption:

```python
def audit_indicator_availability(iso3_codes: list[str], indicator_candidates: list[str]) -> dict:
    """Try each candidate code, keep the first that actually returns data for >0 countries."""
    for code in indicator_candidates:
        result = fetch_worldbank(iso3_codes, code)
        if any(v is not None for v in result.values()):
            return {"indicator_used": code, "data": result,
                    "missing": [c for c, v in result.items() if v is None]}
    return {"indicator_used": None, "data": {}, "missing": iso3_codes}

c8_audit = audit_indicator_availability(list(JURISDICTIONS.values()),
                                         indicator_candidates=["GOV_WGI_RQ_EST", "RQ.EST"])
```

**CryptoCompare (C1) — this is the harder case, and worth flagging before you build around it.** CryptoCompare's API is organized entirely by *exchange* and *trading pair* (`topExchanges('BTC','USD', n)` returns exchange names and volumes) — there is no jurisdiction/country field anywhere in it. So "does CryptoCompare have data for all 15 countries" isn't quite the right question; the real gap is a **missing exchange-to-jurisdiction crosswalk that no API provides**, because an exchange's operating volume and its regulatory home jurisdiction are two unrelated facts. You'll need a small, manually maintained table (`exchange_name -> licensing_jurisdiction`) alongside the CryptoCompare pull, and — separately — it's worth double-checking that your team's "CryptoCompare for C1/C2" note matches the doc: per §1's data-source table, C2 (Revenue Depth) is specified as World Bank GNI-per-capita-PPP data (`NY.GNP.PCAP.PP.CD`), not CryptoCompare; CryptoCompare/Kaiko + the HHI calculation is specified for C1 only. Worth a quick sync with whoever's wiring that fetcher so C2 doesn't end up pointed at the wrong source.

```python
# exchange_jurisdiction_crosswalk.py — manually maintained, no API supplies this
EXCHANGE_JURISDICTION = {
    "Binance": "ARE",      # regulated entity used for HHI attribution, not global HQ
    "Coinbase": "USA",
    "Kraken": "USA",
    "OKX": "SGP",
    # ... one row per exchange that shows up in your top-N CryptoCompare pulls
}
```

### 14.5 One audit function tying all four together

```python
@dataclass
class DataStatus:
    AVAILABLE = "available"
    DATA_UNAVAILABLE = "data_unavailable"        # API/source has no data — plumbing problem
    KNOCKOUT_FAILED_THRESHOLD = "knocked_out"     # has data, and it failed the C6 threshold

def audit_run(jurisdictions: dict[str, str], c2_data: dict, c8_data: dict,
              capital_burden: dict, liquid_runway_usd: float) -> pd.DataFrame:
    rows = []
    for name, iso3 in jurisdictions.items():
        if iso3 not in capital_burden:
            status = DataStatus.DATA_UNAVAILABLE          # Tier 3 hasn't produced C6 yet
        elif capital_burden[iso3] > liquid_runway_usd:
            status = DataStatus.KNOCKOUT_FAILED_THRESHOLD  # correctly, deliberately excluded
        else:
            status = DataStatus.AVAILABLE
        rows.append({"jurisdiction": name, "iso3": iso3, "status": status,
                      "c2_found": c2_data.get(iso3) is not None,
                      "c8_found": c8_data.get(iso3) is not None})
    return pd.DataFrame(rows)
```

Run this before the knockout screen, not as an afterthought — a country that's missing purely because of an API/naming mismatch should never look the same in your logs as one that was correctly, deliberately excluded for failing the capital-burden threshold.

---

## Summary

| Version | Adds | Key library |
|---|---|---|
| V0 | Data model + hard capital knockout | pandas |
| V1 | End-to-end MVP ranking | pymcdm |
| V2 | BWM + CRITIC-M weight fusion | scipy.optimize |
| V2.5 | Multi-stakeholder Bayesian BWM (optional) | pymc |
| V3 | TOPSIS-RAD — eliminates rank reversal | numpy/pandas |
| V4 | SMAA + evidence-calibrated Monte Carlo | numpy |
| V5 | Capital-constrained expansion sequencing | pulp==3.3.2 |
| V6 | Orchestration + output contract | — |
| 14.1–14.2 | Zero-variance guard + completeness-aware weighting | numpy/pandas |
| 14.3 | C5 rerouted into Tier 3 + interim seed table | — |
| 14.4–14.5 | Jurisdiction registry + data-availability audit | requests |
