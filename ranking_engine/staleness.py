"""
staleness.py — Detect when a pipeline re-run is needed.

The pipeline is expensive (SMAA + Monte Carlo). Running it on unchanged data
wastes compute. Running it too infrequently leaves stale rankings driving
decisions.

A re-run is needed if ANY of:
  1. A RawFact has an as_of date newer than the last run timestamp.
  2. A new jurisdiction appears in the current facts that wasn't in the last run.
  3. The PipelineConfig has changed since the last run.
  4. Forced: the caller passes force=True.

Fix #1: The original plan had `from .criterion_mapping import build_decision_matrix`
  (wrong module — it lives in knockout.py). That import was also unused in the
  function body. It has been removed entirely.

Fix #2: The original config_hash() used a scalar-only isinstance filter, which
  silently dropped vpl (ndarray), dpl (ndarray), and surveys (list) — the three
  fields that Phases 2A and 2B exist to make configurable. The fix explicitly
  serializes vpl, dpl, and surveys before the scalar filter runs, so any change
  to these fields changes the hash.
"""

import dataclasses
import hashlib
import json
from datetime import date, datetime

from .results_store import ResultsStore


def config_hash(cfg) -> str:
    """
    SHA-256 fingerprint of a PipelineConfig.

    Fix #2: vpl, dpl (ndarrays) and surveys (list of StakeholderSurvey) are
    explicitly serialized before the isinstance scalar filter runs. Without this,
    changing VPL/DPL (Phase 2B) or stakeholder surveys (Phase 2A) would not
    register as a config change and would never trigger a re-run.

    The hash is deterministic: same config values → same hash, always.
    """
    raw = dataclasses.asdict(cfg)

    payload = {}

    # --- Explicit array/list fields (would be dropped by scalar filter) ---
    # vpl / dpl: ndarray → list[float]
    if "vpl" in raw:
        payload["vpl"] = list(float(x) for x in raw["vpl"])
    if "dpl" in raw:
        payload["dpl"] = list(float(x) for x in raw["dpl"])

    # surveys: list of StakeholderSurvey dicts (dataclasses.asdict already recursed)
    # Each survey dict has: best_idx, worst_idx, a_B, a_W, dm_id, submitted (datetime)
    if "surveys" in raw and raw["surveys"]:
        surveys_payload = []
        for s in raw["surveys"]:
            surveys_payload.append({
                "best_idx":  s["best_idx"],
                "worst_idx": s["worst_idx"],
                "a_B":       s["a_B"],
                "a_W":       s["a_W"],
                "dm_id":     s["dm_id"],
                # datetime is not JSON-serializable by default
                "submitted": (
                    s["submitted"].isoformat()
                    if isinstance(s["submitted"], datetime)
                    else str(s["submitted"])
                ),
            })
        payload["surveys"] = surveys_payload
    else:
        payload["surveys"] = []

    # bwm_a_B / bwm_a_W are now list[float] (changed in pipeline.py), so they
    # pass the scalar filter below — but to be safe, keep them explicit too.
    if "bwm_a_B" in raw:
        payload["bwm_a_B"] = list(raw["bwm_a_B"])
    if "bwm_a_W" in raw:
        payload["bwm_a_W"] = list(raw["bwm_a_W"])

    # --- All remaining scalar fields ---
    for k, v in raw.items():
        if k in payload:
            continue  # already handled above
        if isinstance(v, (int, float, str, bool)):
            payload[k] = v

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def needs_rerun(
    store: ResultsStore,
    current_facts: list,
    current_cfg_hash: str,
    force: bool = False,
) -> tuple[bool, str]:
    """
    Check whether a pipeline re-run is warranted.

    Parameters
    ----------
    store             : ResultsStore to compare against
    current_facts     : the RawFact list the caller is about to use
    current_cfg_hash  : SHA-256 from config_hash(cfg) — NOT a raw cfg object
                        (lets callers cache the hash)
    force             : if True, always returns (True, "forced re-run requested")

    Returns
    -------
    (should_rerun: bool, reason: str)

    Fix #1: the original version had an unused `from .criterion_mapping import
    build_decision_matrix` at the point of check #2 ("new jurisdictions?").
    That import was from the wrong module (it lives in knockout.py) AND was
    never used in the function body. It has been removed.
    """
    if force:
        return True, "forced re-run requested"

    run_id = store.latest_run_id()
    if run_id is None:
        return True, "no previous run found in store"

    # ---- Check 1: any fact newer than the last run? ----
    row = store._conn.execute(
        "SELECT run_at FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    last_run_at = datetime.fromisoformat(row["run_at"].rstrip("Z"))

    newest_fact_date = max(
        (
            f.as_of if isinstance(f.as_of, date)
            else date.fromisoformat(str(f.as_of))
        )
        for f in current_facts
    )
    if newest_fact_date > last_run_at.date():
        return True, (
            f"new facts dated {newest_fact_date} "
            f"(last run: {last_run_at.date()})"
        )

    # ---- Check 2: new jurisdictions in current facts? ----
    # Fix #1: no import needed here — we just compare sets of jurisdiction strings.
    last_result_jurs = {r["jurisdiction"] for r in store.latest_results()}
    current_jurs     = {f.jurisdiction for f in current_facts}
    new_jurs         = current_jurs - last_result_jurs
    if new_jurs:
        return True, f"new jurisdictions in facts: {sorted(new_jurs)}"

    # ---- Check 3: PipelineConfig changed? ----
    # The runs table stores the config_hash string (not the scalar dict), so
    # Fix #2 applies: any change to vpl, dpl, or surveys will produce a
    # different hash and correctly trigger a re-run.
    last_cfg_hash_row = store._conn.execute(
        "SELECT config_json FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if last_cfg_hash_row and last_cfg_hash_row["config_json"]:
        if last_cfg_hash_row["config_json"] != current_cfg_hash:
            return True, "PipelineConfig has changed since last run"

    return False, "data is current — skipping re-run"
