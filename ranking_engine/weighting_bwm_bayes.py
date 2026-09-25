"""
weighting_bwm_bayes.py — V2.5 (Optional): Bayesian BWM via PyMC.

Concept
-------
Point-estimate BWM (weighting_bwm.py) collapses each decision-maker's survey
into one weight vector.  The moment you have multiple executives with different
Best/Worst picks, you need a way to:
  (a) aggregate conflicting opinions, and
  (b) report *how much they actually agree*.

The doc's answer is Bayesian BWM:
  - Model the "true" group weight vector w as unknown, with a Dirichlet prior
    (natively satisfies "non-negative, sums to 1").
  - Treat each stakeholder's comparison ratios as a noisy observation of w.
  - Use MCMC (NUTS via PyMC) to recover the posterior distribution of w.

Output is not one number but a full probability distribution per weight.
The *width* of the credible interval is itself a deliverable:
  wide interval on w[C8] → the executives genuinely disagree about how much
  Regulatory Volatility matters.  A single point estimate would hide this.

Dependency: pip install pymc arviz  (only needed for this file)
"""

import numpy as np

try:
    import pymc as pm
    import arviz as az
    _PYMC_AVAILABLE = True
except ImportError:
    _PYMC_AVAILABLE = False


def bayesian_bwm_weights(
    a_B_all: np.ndarray,
    a_W_all: np.ndarray,
    n_criteria: int,
    n_draws: int = 2000,
    n_tune: int = 1000,
    n_chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> dict:
    """
    Fit the Bayesian BWM model and return posterior statistics.

    Parameters
    ----------
    a_B_all    : 2-D array (n_decision_makers, n_criteria) — each row is one
                 DM's Best-reference comparison vector (a_B[best_idx] = 1)
    a_W_all    : 2-D array (n_decision_makers, n_criteria) — Worst-reference vectors
    n_criteria : number of criteria (8 for C1-C8)
    n_draws    : MCMC draws per chain (more → tighter credible intervals)
    n_tune     : warm-up draws discarded before sampling
    n_chains   : number of independent chains (≥4 recommended for r_hat diagnostics)
    target_accept : NUTS acceptance rate target (0.9 is safe for this model)
    random_seed   : reproducibility seed

    Returns
    -------
    dict with keys:
        "posterior_mean"  : np.ndarray (n_criteria,) — point estimate to use in fuse_weights
        "posterior_std"   : np.ndarray (n_criteria,) — per-weight uncertainty
        "hdi_94"          : np.ndarray (n_criteria, 2) — 94% highest-density interval [lo, hi]
        "r_hat"           : np.ndarray (n_criteria,) — Gelman-Rubin convergence (must be ≤ 1.01)
        "trace"           : az.InferenceData — full posterior for further analysis / plotting
    """
    if not _PYMC_AVAILABLE:
        raise ImportError(
            "pymc and arviz are required for Bayesian BWM.\n"
            "Install with:  pip install pymc arviz\n"
            "If you don't need multi-stakeholder aggregation, use bwm_weights() from weighting_bwm.py instead."
        )

    # Transform raw comparison vectors into implied weight proportions.
    # 1/a_B[j] is proportional to w[j] (how important j is relative to Best),
    # so normalizing 1/a_B gives a "noisy vote" for the true weight vector.
    P = np.array([(1.0 / row) / (1.0 / row).sum() for row in a_B_all])
    Q = np.array([row / row.sum() for row in a_W_all])

    # Each row of observed_vectors is one noisy observation of the true group w.
    observed_vectors = np.vstack([P, Q]).astype(float)

    with pm.Model():
        # Prior: uniform over the weight simplex (each alpha=1)
        w = pm.Dirichlet("w", a=np.ones(n_criteria))

        # Concentration: captures how tightly each DM's vote clusters around the group center.
        # HalfNormal + 5 keeps kappa > 5 (reasonable minimum inter-expert consistency).
        kappa = pm.HalfNormal("kappa", sigma=50) + 5.0

        # Likelihood: each observed vote is a Dirichlet draw centered on w
        pm.Dirichlet("obs", a=w * kappa, observed=observed_vectors)

        trace = pm.sample(
            draws=n_draws,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
        )

    summary = az.summary(trace, var_names=["w"])
    w_posterior = trace.posterior["w"].values  # shape: (chains, draws, n_criteria)
    flat = w_posterior.reshape(-1, n_criteria)

    r_hat_values = summary["r_hat"].values

    # Warn loudly if chains haven't converged — r_hat > 1.01 means don't trust the output
    if np.any(r_hat_values > 1.01):
        bad = np.where(r_hat_values > 1.01)[0]
        print(
            f"[WARNING] Bayesian BWM: r_hat > 1.01 for criteria indices {bad}. "
            "Run more chains/draws before using these weights in production."
        )

    hdi = az.hdi(trace, var_names=["w"], hdi_prob=0.94)["w"].values  # (n_criteria, 2)

    return {
        "posterior_mean": flat.mean(axis=0),
        "posterior_std":  flat.std(axis=0),
        "hdi_94":         hdi,
        "r_hat":          r_hat_values,
        "trace":          trace,
    }
