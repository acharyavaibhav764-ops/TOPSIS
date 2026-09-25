# Ranking Engine — Final Fix List, Test Plan, Output Contract & Remaining Roadmap

Status as of this document: two rounds of review completed against actual code (not summaries). Two bugs confirmed and genuinely fixed. One new bug found, caused by the fix itself, still open. This doc is the single source of truth going forward — hand it to whoever/whatever writes the next round of code.

---

## 1. Where things stand right now

| # | Issue | Status |
|---|---|---|
| 1 | `weighting_critic_m.py` re-inverted C6–C8 a second time (double inversion) | **Fixed & verified** — negation removed, docstring now states the direction contract |
| 2 | `robustness_mc.py` fed raw `alt_raw_value` into `perturb_score` instead of mapping it first | **Fixed & verified** — now runs `MAPPING_FUNCTIONS[crit](alt_raw_value)` before use |
| 3 | `topsis_rad.py` still branches on `benefit_mask` for capping + PIS/NIS. New DPL/VPL values (needed to fix #1's side effect) break that branch: `max(X, dpl_cost=10.0)` forces every C6/C7/C8 score to a constant `10.0` for every jurisdiction. Verified with real code: two jurisdictions differing only in C6 (score 10 vs score 2) get **identical** closeness (0.4817 both). Same root cause as #1, just relocated — `topsis_rad.py` still assumes it receives raw-direction data when the matrix is already direction-corrected upstream. `robustness_smaa.py` and `robustness_mc.py` both call `topsis_rad` internally with the same mask, so this is very likely also live in SMAA rank-acceptability and Monte Carlo intervals, not just the point ranking. | **Open — fix below** |
| 4 | No test in the 76-test suite would catch #3: `test_c6_weight_not_dominant` checks CRITIC-M weight only (a different, now-fixed stage); `test_uae_ranks_first` passes by coincidence because UAE dominates C1–C5 regardless of what happens to C6–C8 | **Open — fix below** |

**Pattern to watch for going forward:** every reported fix so far has needed independent verification against the actual diff — stated numbers and "60/60 passed" / "76/76 passed" claims have not reliably matched what the code does. Don't accept the next round's summary at face value either; re-run the checks in §5 against the real files.

---

## 2. Required code fixes

### Fix A — `topsis_rad.py`: remove the `benefit_mask`-conditional direction handling

Same category of fix as #1, applied to the file it was missed in. The contract, once and for all: **`criterion_mapping.py` is the only place direction-correction happens.** Every module downstream of `build_decision_matrix()` receives a matrix where higher = better, full stop, for all 8 columns. No downstream module should branch on `benefit_mask` for math again.

Concretely, in `topsis_rad()`:

- **Capping (step 1).** Replace
  `X_capped = np.where(benefit_mask, np.minimum(X, dpl), np.maximum(X, dpl))`
  with
  `X_capped = np.minimum(X, dpl)` — applied uniformly, no branch.
- **PIS/NIS (step 3).** Replace
  `pis = np.where(benefit_mask, anchors_n[1], anchors_n[0]) * weights`
  `nis = np.where(benefit_mask, anchors_n[0], anchors_n[1]) * weights`
  with
  `pis = anchors_n[1] * weights` (always DPL)
  `nis = anchors_n[0] * weights` (always VPL) — no branch.
- Update the docstring with the same "Direction contract" note added to `weighting_critic_m.py`: matrix is pre-corrected, `benefit_mask` kept in the signature only so callers/tests can see raw-data semantics, not consumed for computation.
- `smaa_topsis()` and `monte_carlo_robustness()` need no code change — they just pass `benefit_mask`/`vpl`/`dpl` through — but both must be re-verified after this fix (see Fix C).

### Fix B — `pipeline.py`: simplify VPL/DPL to one convention, permanently

Once Fix A lands, cost vs. benefit no longer needs separate VPL/DPL values — the matrix is uniform. Replace the current split arrays with one pair used for all 8 columns, e.g.:

```python
vpl: np.ndarray = field(default_factory=lambda: np.array([1.0]*8))
dpl: np.ndarray = field(default_factory=lambda: np.array([8.5]*8))
```

This removes the whole class of bug (#3 was caused by tuning one half of a two-part config without touching the code that depended on the old relationship between the halves). One array, one meaning, nothing to keep in sync.

### Fix C — cross-cutting monotonicity test (the actual verification mechanism)

This is the single test type that would have caught **both** bug #1 and bug #3, and will catch the same class of bug anywhere else it's hiding. Apply it to every consumer of `benefit_mask`: `critic_m_weights`, `topsis_rad`, `smaa_topsis`, `monte_carlo_robustness`.

For each of the 8 criteria, independently:
1. Take two jurisdictions identical on every criterion except one.
2. Set that one criterion so jurisdiction A is genuinely better (real-world sense — cheaper capital, lower friction, higher leverage cap, whatever C's actual meaning is), jurisdiction B genuinely worse.
3. Run it through `build_decision_matrix` → the module under test.
4. Assert: A's output is strictly better than B's (not equal — equal means the criterion was silently zeroed, which is exactly what happened here) and in the correct direction.

This is a direct generalization of the manual check that caught bug #3 (`CheapCapital` vs `ExpensiveCapital`, real code, real mapping) — just run for all 8 criteria instead of one, and run against `critic_m_weights`/`smaa_topsis`/`monte_carlo_robustness` too, not just `topsis_rad`.

### Fix D — stage-by-stage disk dump for independent re-verification

This is what you asked for directly: a way to see every stage's output in a form that can be re-checked without re-running the pipeline. `pipeline.run()` already builds all of this in memory (the `artifacts` dict) — this just serializes it.

Add an optional `dump_dir: str | None = None` parameter to `RankingPipeline.run()`. When set, write:

| File | Contents |
|---|---|
| `00_raw_facts.json` | The input `facts` list, verbatim — the audit trail starts from raw truth |
| `01_decision_matrix.csv` | Post-mapping matrix, before knockout |
| `02_post_knockout_matrix.csv` | Matrix after knockout, plus a `knocked_out.json` listing who was dropped and why |
| `03_weights.json` | `w_bwm`, `w_critic_m`, fused `weights`, BWM consistency `xi` |
| `04_topsis_closeness.csv` | Per-jurisdiction closeness + rank |
| `05_smaa_rai.csv`, `05_smaa_central_weight_vectors.json` | SMAA outputs |
| `06_mc_intervals.csv` | Monte Carlo p5/p50/p95/std |
| `07_knapsack_allocation.json` | Wave assignments |
| `08_final_results.json` | The full `RankingResult` list |

Each numbered file should be independently loadable and checkable against the previous one — e.g., someone (a human in a spreadsheet, or a fresh review session) should be able to take `02_post_knockout_matrix.csv` + `03_weights.json`, hand-recompute TOPSIS closeness, and diff it against `04_topsis_closeness.csv` without touching Python. That's the actual point of this: verification that doesn't depend on trusting the same code that produced the number.

---

## 3. Test checklist — consolidated, file by file

| File | Covers | Action needed |
|---|---|---|
| `test_v0_knockout.py` | Matrix assembly, knockout boundary | None |
| `test_v1_mvp.py` | Legacy MVP scratch path (not on production path — `pipeline.py` doesn't call it) | None; just don't mistake its passing for production coverage |
| `test_v2_weighting.py` | BWM + CRITIC-M unit tests | Add a variant that runs `critic_m_weights` on a matrix built via real `build_decision_matrix` + `MAPPING_FUNCTIONS`, not just a hand-typed fixture — this is what would've caught bug #1 faster |
| `test_v3_topsis_rad.py` | Rank-reversal regression, capping | Re-verify the existing capping test still passes after Fix A; **add the Fix C monotonicity test for C6/C7/C8 specifically** — this is the direct regression test for bug #3 |
| `test_v4_robustness.py` | SMAA + MC unit tests | Add Fix C monotonicity tests routed through `smaa_topsis` and `monte_carlo_robustness` |
| `test_v5_knapsack.py` | Budget, mutual exclusion | None |
| `test_v6_pipeline.py` | Integration, real `RawFact` → `RankingPipeline.run()` | **Add:** a fixture where two jurisdictions are equal everywhere except C6 (or C7/C8), and assert the ranking actually responds — this closes the exact gap that let bug #3 through 76/76 |
| `test_v7_reverify.py` (new) | Independent cross-check | Run `pipeline.run(..., dump_dir=...)`, then **re-implement TOPSIS-RAD from scratch in the test file itself** (not calling `topsis_rad()` at all — a completely separate, deliberately naive numpy implementation) on the dumped stage-02/03 data, and assert it matches `04_topsis_closeness.csv`. This is the strongest test in the suite: it checks the module against an independent reference, not against itself. |

---

## 4. Remaining roadmap (full picture, not just the ranking engine)

**Tier 1 — this document.** Fixes A–D above, plus the test additions. Bounded, code-only, you already know exactly what's wrong.

**Tier 2 — production-hardening of the ranking engine itself.**
- Real VPL/DPL values decided by the actual management team, not placeholder numbers (the doc calls for this explicitly — "established independently by the management team")
- A real BWM survey process to collect actual Best/Worst judgments from real stakeholders (currently hardcoded in `PipelineConfig`); decide if you need the already-written-but-unwired `weighting_bwm_bayes.py` for multi-stakeholder aggregation
- Persistence layer + scheduled re-run trigger — `pipeline.py`'s own docstring flags this as an open seam ("logging, a results table written to your DB, and a scheduled re-run trigger")

**Tier 3 — the De Jure LLM regulatory-extraction pipeline (currently 0% built).**
This is the only thing that can ever produce real `RawFact` rows — everything tested so far has been hand-typed numbers. Per the strategy doc: scraping/ingesting raw regulator PDFs/HTML → Markdown normalization preserving legal structure → RAG-constrained extraction into the strict JSON schema (with verbatim source span) → the "LLM-as-a-judge" verification against 19 criteria → evidence tagging (`VERIFIED_FACT`/`ANALYST_ESTIMATE`/`UNRESOLVED_POLICY`) → a versioned rule database. Realistically the largest remaining body of work in the project — bigger than everything built so far.

**Tier 4 — automated data feeds** for the criteria that don't come from legal text: World Bank Open Data API (C2, C5), exchange volume + HHI calculation (C1), WGI/B-READY indices (C8). Currently all hand-entered.

**Tier 5 — non-engineering, parallel track.** GIFT City Category III AIF setup, IFSCA FME registration, FEMA-compliant capital structuring. Doesn't block the code, but the doc frames the model's output as only "legally deployable" once this exists — worth someone owning it now rather than later.

---

## 5. How to re-verify the next round of "it's fixed" claims

Don't trust the summary. Do this instead, in order:
1. Diff every changed file against the previous version — confirm the described change is actually what changed, and that nothing else changed silently.
2. For any bug that involved a direction/sign/unit issue, run the two-jurisdiction differential check: identical on everything except the one criterion in question, real values through real `MAPPING_FUNCTIONS`, and check the final output — not an intermediate value — actually moves, and moves the right way.
3. Check which specific test would fail if the bug were reintroduced. If you can't name one, the fix isn't tested — it's just claimed.
4. Re-run the actual test suite yourself; don't take the pass count on faith.
