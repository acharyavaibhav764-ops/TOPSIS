"""
pipeline.py — V6: Full pipeline orchestration.

Ties V0 through V5 together in the correct order and exposes a single
RankingPipeline.run() entry point.  The output is a list of RankingResult
objects — one per surviving jurisdiction — which is the final contract the
business layer, reporting layer, and DB writer consume.

Re-run triggers (as the doc specifies, this runs as a continuous loop):
  - New/updated RawFact rows arrive from the extraction pipeline → re-run everything
  - Management team submits a new BWM survey → re-run V2 onwards
  - A new jurisdiction is scoped in → V3/V4 are unaffected due to TOPSIS-RAD's
    fixed anchors (adding a market doesn't change existing closeness coefficients)
"""

import numpy as np
import pandas as pd
import json
import os
import hashlib
import dataclasses
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from .schema import RawFact, EvidenceTag, RankingResult, StakeholderSurvey
from .criterion_mapping import CRITERIA, BENEFIT_MASK, MAPPING_FUNCTIONS
from .knockout import build_decision_matrix, knockout_screen, extract_capital_required
from .weighting_bwm import bwm_weights
from .weighting_critic_m import critic_m_weights, fuse_weights
from .topsis_rad import topsis_rad
from .robustness_smaa import smaa_topsis
from .robustness_mc import monte_carlo_robustness
from .knapsack import solve_multi_period

try:
    from .weighting_bwm_bayes import bayesian_bwm_weights
    _BBWM_AVAILABLE = True
except ImportError:
    _BBWM_AVAILABLE = False


@dataclass
class PipelineConfig:
    """All configuration that the business layer can tune without touching math code."""

    # --- Hard knockout ---
    liquid_runway_usd: float = 5_000_000.0  # USD available for any single jurisdiction

    # --- Fixed TOPSIS-RAD anchors (VPL / DPL) ---
    # These are analyst-defined constants, never derived from the dataset.
    # The matrix is fully direction-corrected (higher = better for ALL 8 columns),
    # so VPL and DPL are uniform — no split between benefit and cost criteria.
    #
    # VPL = 1.0 for all: the worst conceivable score on the 1-10 scale.
    # DPL = 8.5 for all: any market scoring above 8.5 is capped at 8.5 —
    #   one outstanding criterion cannot geometrically dominate the TOPSIS space.
    #   Setting DPL = 10.0 (scale boundary) makes capping a no-op; keep it < 10.
    #
    # These values must be decided by the actual management team before production use
    # (the plan explicitly calls for this — these are reasonable analyst defaults only).
    vpl: np.ndarray = field(
        default_factory=lambda: np.array([1.0] * 8, dtype=float)
    )
    dpl: np.ndarray = field(
        default_factory=lambda: np.array([8.5] * 8, dtype=float)
    )

    # --- Weighting ---
    weight_lambda: float = 0.6  # λ=1 → pure BWM, λ=0 → pure CRITIC-M

    # --- BWM survey inputs (single-DM point-estimate path) ---
    # These are used when surveys=[] (the default). When surveys is populated
    # with real StakeholderSurvey objects, the pipeline routes through
    # bayesian_bwm_weights() instead, and these fields are ignored.
    bwm_best_idx:  int = 2                                         # C3 = Pathway Clarity
    bwm_worst_idx: int = 5                                         # C6 = Capital Burden
    bwm_a_B: list = field(default_factory=lambda: [3,4,1,2,5,9,6,4])
    bwm_a_W: list = field(default_factory=lambda: [5,4,9,6,3,1,2,4])

    # --- Multi-stakeholder Bayesian BWM (Phase 2A) ---
    # Populate this list with real StakeholderSurvey objects to activate BBWM.
    # When len(surveys) >= 1, the pipeline uses MCMC (PyMC) instead of the
    # single-DM linear BWM above. Requires: pip install pymc arviz
    surveys: list = field(default_factory=list)

    # Bayesian MCMC settings (only used when surveys is non-empty):
    bbwm_n_draws:       int   = 2000
    bbwm_n_tune:        int   = 1000
    bbwm_n_chains:      int   = 4
    bbwm_target_accept: float = 0.90
    bbwm_seed:          int   = 42

    # --- Robustness ---
    smaa_n_trials: int = 20_000
    mc_n_trials:   int = 10_000
    smaa_seed:     int = 42
    mc_seed:       int = 42

    # --- Knapsack ---
    wave_budgets:             list = field(default_factory=lambda: [8_000_000, 5_000_000])
    mutually_exclusive_groups: list = field(default_factory=list)
    knapsack_alpha:           float = 0.6

    # -------------------------------------------------------------------------
    # Phase 2B helpers
    # -------------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        vpl_dpl:  str | None = None,
        surveys:  str | None = None,
        **kwargs,
    ) -> "PipelineConfig":
        """
        Load VPL/DPL anchors and/or stakeholder surveys from JSON files.

        Usage::

            cfg = PipelineConfig.from_files(
                vpl_dpl="ranking_engine/config/vpl_dpl.json",
                surveys="ranking_engine/config/surveys.json",
            )

        vpl_dpl.json format::

            {
              "C1": {"vpl": 2.0, "dpl": 8.0},
              "C2": {"vpl": 1.5, "dpl": 9.0},
              ...
            }

        surveys.json format: list of StakeholderSurvey-serializable dicts.
        """
        init_kwargs = {**kwargs}

        if vpl_dpl:
            data = json.loads(Path(vpl_dpl).read_text())
            vpl_arr = np.array([data[c]["vpl"] for c in CRITERIA], dtype=float)
            dpl_arr = np.array([data[c]["dpl"] for c in CRITERIA], dtype=float)
            _validate_vpl_dpl(vpl_arr, dpl_arr)
            init_kwargs["vpl"] = vpl_arr
            init_kwargs["dpl"] = dpl_arr

        if surveys:
            raw = json.loads(Path(surveys).read_text())
            init_kwargs["surveys"] = [
                StakeholderSurvey(
                    best_idx=s["best_idx"],
                    worst_idx=s["worst_idx"],
                    a_B=s["a_B"],
                    a_W=s["a_W"],
                    dm_id=s["dm_id"],
                    submitted=datetime.fromisoformat(s["submitted"]),
                )
                for s in raw
            ]

        return cls(**init_kwargs)


def _validate_vpl_dpl(vpl: np.ndarray, dpl: np.ndarray) -> None:
    """
    Validate VPL/DPL arrays before constructing a PipelineConfig.
    Raises ValueError on structural violations; warns on boundary values.
    """
    import warnings
    if not (vpl >= 1.0).all():
        raise ValueError("VPL values must be >= 1.0 (minimum of the 1-10 score scale)")
    if not (dpl <= 10.0).all():
        raise ValueError("DPL values must be <= 10.0 (maximum of the 1-10 score scale)")
    if not (vpl < dpl).all():
        bad = [CRITERIA[i] for i in np.where(~(vpl < dpl))[0]]
        raise ValueError(
            f"VPL must be strictly less than DPL for all criteria. Violated for: {bad}"
        )
    if (dpl == 10.0).any():
        boundary = [CRITERIA[i] for i in np.where(dpl == 10.0)[0]]
        warnings.warn(
            f"DPL=10.0 for {boundary}: setting DPL at the scale boundary makes the "
            "anti-stretching cap a no-op for these criteria. Consider DPL < 10.0.",
            UserWarning, stacklevel=2,
        )

def _dump_stage_artifacts(
    dump_dir: str,
    facts: list,
    pre_knockout_matrix: "pd.DataFrame",
    post_knockout_matrix: "pd.DataFrame",
    knocked_out: list[str],
    w_bwm: np.ndarray,
    w_cm: np.ndarray,
    weights: np.ndarray,
    xi: float,
    criteria: list[str],
    closeness: "pd.Series",
    rai: "pd.DataFrame",
    cwv: dict,
    mc_results: "pd.DataFrame",
    allocation: dict,
    results: list,
) -> None:
    """
    Write pipeline intermediates to numbered CSV/JSON files in dump_dir.
    Each file is independently loadable — no Python needed to read them.
    """
    os.makedirs(dump_dir, exist_ok=True)

    # 00 — raw facts (the audit trail starts here)
    with open(os.path.join(dump_dir, "00_raw_facts.json"), "w") as f:
        json.dump(
            [{
                "jurisdiction": r.jurisdiction,
                "criterion_code": r.criterion_code,
                "raw_value": r.raw_value,
                "evidence_tag": r.evidence_tag.value,
                "source_citation": r.source_citation,
                "as_of": r.as_of.isoformat() if isinstance(r.as_of, date) else str(r.as_of),
                "alt_raw_value": r.alt_raw_value,
            } for r in facts],
            f, indent=2,
        )

    # 01 — decision matrix before knockout
    pre_knockout_matrix.to_csv(os.path.join(dump_dir, "01_decision_matrix.csv"), float_format="%.4f")

    # 02 — post-knockout matrix + who was dropped
    post_knockout_matrix.to_csv(os.path.join(dump_dir, "02_post_knockout_matrix.csv"), float_format="%.4f")
    with open(os.path.join(dump_dir, "02_knocked_out.json"), "w") as f:
        json.dump({"knocked_out": knocked_out}, f, indent=2)

    # 03 — weight vectors + consistency ratio
    with open(os.path.join(dump_dir, "03_weights.json"), "w") as f:
        json.dump({
            "w_bwm":      dict(zip(criteria, w_bwm.tolist())),
            "w_critic_m": dict(zip(criteria, w_cm.tolist())),
            "w_fused":    dict(zip(criteria, weights.tolist())),
            "bwm_consistency_xi": xi,
        }, f, indent=2)

    # 04 — TOPSIS closeness + rank
    cc_df = pd.DataFrame({
        "closeness": closeness,
        "rank": closeness.rank(ascending=False).astype(int),
    })
    cc_df.to_csv(os.path.join(dump_dir, "04_topsis_closeness.csv"), float_format="%.6f")

    # 05 — SMAA rank acceptability index + central weight vectors
    rai.to_csv(os.path.join(dump_dir, "05_smaa_rai.csv"), float_format="%.4f")
    with open(os.path.join(dump_dir, "05_smaa_central_weight_vectors.json"), "w") as f:
        json.dump(
            {jur: (v.tolist() if v is not None else None) for jur, v in cwv.items()},
            f, indent=2,
        )

    # 06 — Monte Carlo intervals
    mc_results.to_csv(os.path.join(dump_dir, "06_mc_intervals.csv"), float_format="%.6f")

    # 07 — Knapsack allocation
    with open(os.path.join(dump_dir, "07_knapsack_allocation.json"), "w") as f:
        json.dump({f"wave_{k+1}": v for k, v in allocation.items()}, f, indent=2)

    # 08 — Final results (the full RankingResult list)
    with open(os.path.join(dump_dir, "08_final_results.json"), "w") as f:
        json.dump([
            dataclasses.asdict(r) for r in results
        ], f, indent=2, default=str)




class RankingPipeline:
    """End-to-end market ranking pipeline from raw facts to RankingResult list."""

    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()

    def run(
        self,
        facts: list[RawFact],
        margin_12mo: dict[str, float] | None = None,
        verbose: bool = True,
        dump_dir: str | None = None,
        store=None,     # ResultsStore | None  (type-hinted as string to avoid circular import)
    ) -> tuple[list[RankingResult], dict]:
        """
        Execute the full pipeline and return ranked results + intermediate artifacts.

        Parameters
        ----------
        facts        : list of RawFact from the extraction pipeline
        margin_12mo  : dict jurisdiction → expected 12-month contribution margin (USD)
                       Required for the knapsack value function; can be omitted to skip V5.
        verbose      : whether to print stage-by-stage progress
        dump_dir     : optional directory path.  When set, every pipeline stage is
                       serialized to numbered CSV/JSON files (00_raw_facts.json …
                       08_final_results.json).  Each file is independently loadable
                       and re-verifiable without re-running the pipeline.

        Returns
        -------
        results      : list[RankingResult] sorted by rank
        artifacts    : dict of intermediate outputs (matrix, weights, rai, mc, allocation)
                       for debugging, logging, and downstream reporting
        """
        cfg = self.cfg

        # ---- V0: Matrix assembly + hard knockout -----------------------------------------
        if verbose: print("=== V0: Building decision matrix + knockout screen ===")
        matrix = build_decision_matrix(facts, MAPPING_FUNCTIONS)
        capital_required = extract_capital_required(facts)
        matrix = knockout_screen(matrix, capital_required, cfg.liquid_runway_usd)

        if matrix.empty:
            raise ValueError("All jurisdictions were knocked out by the capital screen. "
                             "Increase liquid_runway_usd or review C6 raw values.")

        # ---- V0-C: Partial-criteria mode -------------------------------------------------
        # Drop any criterion column whose values are ALL NaN (e.g. C3/C4/C6/C7 when Tier 3
        # is not yet built). Missing criteria cannot contribute signal — including them as
        # NaN propagates through CRITIC-M variance calculation and poisons all weights and
        # closeness scores. We warn loudly and rank on the available subset.
        all_null_cols = [c for c in matrix.columns if matrix[c].isna().all()]
        if all_null_cols:
            if verbose:
                print(f"  [PARTIAL-CRITERIA MODE] Dropping {len(all_null_cols)} criteria with "
                      f"no data yet: {all_null_cols}")
                print(f"  Ranking will proceed on {len(matrix.columns) - len(all_null_cols)} "
                      f"available criteria. Add Tier-3 data to include the rest.")
            matrix = matrix.drop(columns=all_null_cols)

        # Map active column positions back to the full CRITERIA list
        active_criteria  = list(matrix.columns)
        active_idx       = [CRITERIA.index(c) for c in active_criteria]
        active_benmask   = np.array([BENEFIT_MASK[i] for i in active_idx])
        active_vpl       = cfg.vpl[active_idx]
        active_dpl       = cfg.dpl[active_idx]

        # Any rows still with partial NaN get imputed with the column median
        # (prevents a single missing cell from silently zeroing a jurisdiction's rank).
        if matrix.isna().any().any():
            medians = matrix.median()
            matrix  = matrix.fillna(medians)
            if verbose:
                print("  [IMPUTATION] Remaining NaN cells filled with column medians.")

        # ---- V2: Weight vector -----------------------------------------------------------
        if verbose: print("=== V2: BWM + CRITIC-M weight fusion ===")

        if cfg.surveys and _BBWM_AVAILABLE:
            # --- Bayesian BWM path (multi-stakeholder, Phase 2A) ---
            a_B_all = np.array([s.a_B for s in cfg.surveys], dtype=float)  # (n_DM, n_criteria)
            a_W_all = np.array([s.a_W for s in cfg.surveys], dtype=float)
            bbwm_result = bayesian_bwm_weights(
                a_B_all=a_B_all,
                a_W_all=a_W_all,
                n_criteria=len(CRITERIA),
                n_draws=cfg.bbwm_n_draws,
                n_tune=cfg.bbwm_n_tune,
                n_chains=cfg.bbwm_n_chains,
                target_accept=cfg.bbwm_target_accept,
                random_seed=cfg.bbwm_seed,
            )
            w_bwm_full = bbwm_result["posterior_mean"]    # ndarray (n_criteria,)
            xi         = float(bbwm_result["r_hat"].max())
            bbwm_diagnostics = bbwm_result
            if verbose:
                print(f"  BBWM max r_hat = {xi:.4f}  "
                      f"({'converged' if xi <= 1.01 else 'NOT CONVERGED — increase n_draws'})")
        elif cfg.surveys and not _BBWM_AVAILABLE:
            raise ImportError(
                "surveys are set but PyMC is not installed. "
                "Install with: pip install pymc arviz"
            )
        else:
            # --- Point-estimate BWM path (single DM, default) ---
            w_bwm_full, xi = bwm_weights(
                CRITERIA, cfg.bwm_best_idx, cfg.bwm_worst_idx,
                np.array(cfg.bwm_a_B, dtype=float),
                np.array(cfg.bwm_a_W, dtype=float),
            )
            bbwm_diagnostics = None
            if verbose:
                print(f"  BWM consistency ratio xi = {xi:.4f}  "
                      f"({'OK' if xi < 0.1 else 'HIGH — consider re-surveying'})")

        # Slice BWM weights to active criteria and re-normalise so they sum to 1
        w_bwm = w_bwm_full[active_idx]
        w_bwm = w_bwm / w_bwm.sum()

        w_cm   = critic_m_weights(matrix, active_benmask)
        weights = fuse_weights(w_bwm, w_cm, cfg.weight_lambda)
        if verbose:
            for c, w in zip(active_criteria, weights):
                print(f"  {c}: {w:.4f}")

        # ---- V3: TOPSIS-RAD point ranking -----------------------------------------------
        if verbose: print("=== V3: TOPSIS-RAD ranking ===")
        closeness = topsis_rad(matrix, weights, active_benmask, active_vpl, active_dpl)
        ranked    = closeness.sort_values(ascending=False)
        if verbose:
            for rank, (jur, cc) in enumerate(ranked.items(), start=1):
                print(f"  #{rank} {jur}: closeness={cc:.4f}")

        # ---- V4a: SMAA ------------------------------------------------------------------
        if verbose: print(f"=== V4a: SMAA ({cfg.smaa_n_trials:,} trials) ===")
        rai, cwv = smaa_topsis(
            matrix, active_benmask, active_vpl, active_dpl,
            n_trials=cfg.smaa_n_trials, seed=cfg.smaa_seed
        )

        # ---- V4b: Evidence-calibrated Monte Carlo ----------------------------------------
        if verbose: print(f"=== V4b: Monte Carlo ({cfg.mc_n_trials:,} trials) ===")
        mc_results = monte_carlo_robustness(
            base_matrix=matrix,
            facts=facts,
            weights=weights,
            benefit_mask=active_benmask,
            vpl=active_vpl,
            dpl=active_dpl,
            n_trials=cfg.mc_n_trials,
            seed=cfg.mc_seed,
        )

        # ---- V5: Multi-period knapsack --------------------------------------------------
        allocation: dict[int, list[str]] = {}
        if margin_12mo is not None and cfg.wave_budgets:
            if verbose: print(f"=== V5: Knapsack ({len(cfg.wave_budgets)} waves) ===")
            allocation = solve_multi_period(
                closeness=closeness,
                capital_burden=capital_required,
                margin_12mo={m: margin_12mo.get(m, 0.0) for m in matrix.index},
                wave_budgets=cfg.wave_budgets,
                mutually_exclusive_groups=cfg.mutually_exclusive_groups,
                alpha=cfg.knapsack_alpha,
            )

        # ---- Assemble RankingResult list ------------------------------------------------
        market_to_wave: dict[str, int | None] = {}
        for wave_idx, chosen in allocation.items():
            for m in chosen:
                market_to_wave[m] = wave_idx + 1  # 1-indexed waves

        results: list[RankingResult] = []
        for rank, (jur, cc) in enumerate(ranked.items(), start=1):
            mc_row = mc_results.loc[jur] if jur in mc_results.index else None
            results.append(RankingResult(
                jurisdiction=jur,
                closeness=float(cc),
                rank=rank,
                rank_acceptability={int(col): float(rai.loc[jur, col]) for col in rai.columns},
                mc_90pct_interval=(
                    float(mc_row["p5"]) if mc_row is not None else 0.0,
                    float(mc_row["p95"]) if mc_row is not None else 0.0,
                ),
                expansion_wave=market_to_wave.get(jur, None),
            ))

        artifacts = {
            "decision_matrix": matrix,
            "active_criteria": active_criteria,
            "weights": dict(zip(active_criteria, weights)),
            # Fix #3: expose raw BWM and CRITIC-M vectors so ResultsStore can
            # write per-criterion breakdown to the weights table without reading
            # private locals. These keys are stable and documented.
            "w_bwm":  dict(zip(active_criteria, w_bwm.tolist())),
            "w_cm":   dict(zip(active_criteria, w_cm.tolist())),
            "bwm_consistency_xi": xi,
            "bbwm_diagnostics": bbwm_diagnostics,
            "closeness": closeness,
            "rai": rai,
            "central_weight_vectors": cwv,
            "mc_intervals": mc_results,
            "expansion_allocation": allocation,
        }

        # ---- Fix D: optional stage dump -------------------------------------------------
        if dump_dir is not None:
            pre_knockout = build_decision_matrix(facts, MAPPING_FUNCTIONS)
            knocked_out_list = [j for j in pre_knockout.index if j not in matrix.index]
            _dump_stage_artifacts(
                dump_dir=dump_dir,
                facts=facts,
                pre_knockout_matrix=pre_knockout,
                post_knockout_matrix=matrix,
                knocked_out=knocked_out_list,
                w_bwm=w_bwm,
                w_cm=w_cm,
                weights=weights,
                xi=xi,
                criteria=active_criteria,
                closeness=closeness,
                rai=rai,
                cwv=cwv,
                mc_results=mc_results,
                allocation=allocation,
                results=results,
            )
            if verbose:
                print(f"=== Stage artifacts written to: {dump_dir} ===")

        # ---- Phase 2C: optional persistence to ResultsStore ----------------------------
        run_id = None
        if store is not None:
            run_id = store.save_run(
                facts=facts,
                results=results,
                artifacts=artifacts,
                cfg=self.cfg,
            )
            if verbose:
                print(f"=== Results persisted (run_id={run_id}) ===")

        artifacts["run_id"] = run_id
        return results, artifacts
