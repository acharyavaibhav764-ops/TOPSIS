"""
results_store.py — SQLite-backed persistence for RankingResult lists.

Schema design
-------------
Every pipeline run gets a UUID run_id and a timestamp. Results are stored as
rows (run_id, jurisdiction, rank, closeness, ...). Raw facts and weights are
stored as JSON blobs and per-criterion rows per run, so the full audit trail
is self-contained in a single .db file.

Querying "how did Brazil's rank change over time" is a two-line SQL query:

    SELECT r.run_at, res.rank, res.closeness
    FROM runs r JOIN results res USING (run_id)
    WHERE res.jurisdiction = 'Brazil'
    ORDER BY r.run_at;

No ORM — raw sqlite3 only. Zero new dependencies beyond the standard library.
"""

import sqlite3
import json
import uuid
from datetime import date, datetime
from pathlib import Path

from .schema import RankingResult


# ---------------------------------------------------------------------------
# DDL — tables are created on first open, idempotent thereafter
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    run_at      TEXT NOT NULL,      -- ISO-8601 UTC timestamp
    n_facts     INTEGER,            -- number of RawFacts consumed
    n_markets   INTEGER,            -- number of markets that survived knockout
    config_json TEXT                -- JSON snapshot of scalar PipelineConfig fields
);

CREATE TABLE IF NOT EXISTS results (
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    jurisdiction    TEXT NOT NULL,
    rank            INTEGER NOT NULL,
    closeness       REAL    NOT NULL,
    mc_p5           REAL,
    mc_p95          REAL,
    expansion_wave  INTEGER,
    PRIMARY KEY (run_id, jurisdiction)
);

CREATE TABLE IF NOT EXISTS weights (
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    criterion   TEXT NOT NULL,
    w_bwm       REAL,
    w_critic_m  REAL,
    w_fused     REAL,
    PRIMARY KEY (run_id, criterion)
);

CREATE TABLE IF NOT EXISTS facts_snapshots (
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    facts_json  TEXT NOT NULL    -- full JSON array of RawFacts used in this run
);
"""


class ResultsStore:
    """
    Thin wrapper around a SQLite database for ranking results persistence.

    Usage::

        store = ResultsStore("ranking_engine/data/results.db")
        run_id = store.save_run(facts=facts, results=results, artifacts=artifacts, cfg=cfg)

        # History for one jurisdiction across all runs:
        history = store.jurisdiction_history("Brazil")

        # Latest run's results:
        latest  = store.latest_results()

        store.close()
    """

    def __init__(self, db_path: str = "ranking_engine/data/results.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(_DDL)

    def save_run(
        self,
        facts: list,
        results: list[RankingResult],
        artifacts: dict,
        cfg,                      # PipelineConfig — inspected by config_hash() separately
        run_id: str | None = None,
    ) -> str:
        """
        Persist one complete pipeline run. Returns the run_id.

        Reads from artifacts:
            "weights"  : dict[str, float]  — fused weight per criterion
            "w_bwm"    : dict[str, float]  — BWM weight per criterion   (Fix #3)
            "w_cm"     : dict[str, float]  — CRITIC-M weight per criterion (Fix #3)

        All three keys are written by pipeline.py's V2 block. If a key is
        missing the column will be NULL — but the test_v8_persistence.py suite
        explicitly asserts non-null, so a regression here will be caught.
        """
        from .staleness import config_hash  # local import to avoid circular

        run_id = run_id or str(uuid.uuid4())
        run_at = datetime.utcnow().isoformat() + "Z"

        # Scalar-only snapshot for change detection (excludes ndarray/list — Fix #2
        # is in config_hash, not here; we store the full hash, not the filtered dict).
        cfg_hash_str = config_hash(cfg)

        with self._conn:
            self._conn.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                (run_id, run_at, len(facts), len(results), cfg_hash_str),
            )

            for r in results:
                self._conn.execute(
                    "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id, r.jurisdiction, r.rank, r.closeness,
                        r.mc_90pct_interval[0] if r.mc_90pct_interval else None,
                        r.mc_90pct_interval[1] if r.mc_90pct_interval else None,
                        r.expansion_wave,
                    ),
                )

            # Fix #3: read the keys that pipeline.py now guarantees to produce.
            w_bwm_d  = artifacts.get("w_bwm", {})   # dict[criterion -> float]
            w_cm_d   = artifacts.get("w_cm",  {})
            w_fused  = artifacts.get("weights", {})

            for crit in w_fused:
                self._conn.execute(
                    "INSERT INTO weights VALUES (?, ?, ?, ?, ?)",
                    (
                        run_id, crit,
                        w_bwm_d.get(crit),    # non-null after Fix #3 is in place
                        w_cm_d.get(crit),
                        w_fused.get(crit),
                    ),
                )

            # Facts snapshot
            facts_payload = []
            for f in facts:
                facts_payload.append({
                    "jurisdiction":   f.jurisdiction,
                    "criterion_code": f.criterion_code,
                    "raw_value":      f.raw_value,
                    "evidence_tag":   f.evidence_tag.value,
                    "as_of": (
                        f.as_of.isoformat()
                        if isinstance(f.as_of, (date, datetime))
                        else str(f.as_of)
                    ),
                })
            self._conn.execute(
                "INSERT INTO facts_snapshots VALUES (?, ?)",
                (run_id, json.dumps(facts_payload)),
            )

        return run_id

    # -------------------------------------------------------------------------
    # Query helpers
    # -------------------------------------------------------------------------

    def jurisdiction_history(self, jurisdiction: str) -> list[dict]:
        """
        Rank/closeness history for one jurisdiction across all runs, oldest-first.

        Returns list of dicts: [{run_id, run_at, rank, closeness, mc_p5, mc_p95}]
        """
        rows = self._conn.execute("""
            SELECT r.run_id, r.run_at, res.rank, res.closeness, res.mc_p5, res.mc_p95
            FROM runs r
            JOIN results res ON r.run_id = res.run_id
            WHERE res.jurisdiction = ?
            ORDER BY r.run_at ASC
        """, (jurisdiction,)).fetchall()
        return [dict(row) for row in rows]

    def latest_run_id(self) -> str | None:
        """Return the run_id of the most recent run, or None if the store is empty."""
        row = self._conn.execute(
            "SELECT run_id FROM runs ORDER BY run_at DESC LIMIT 1"
        ).fetchone()
        return row["run_id"] if row else None

    def latest_results(self) -> list[dict]:
        """Return all results from the most recent run, sorted by rank."""
        run_id = self.latest_run_id()
        if not run_id:
            return []
        rows = self._conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY rank ASC",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def weights_for_run(self, run_id: str) -> list[dict]:
        """Return per-criterion weight breakdown for a given run."""
        rows = self._conn.execute(
            "SELECT * FROM weights WHERE run_id = ? ORDER BY criterion ASC",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._conn.close()
