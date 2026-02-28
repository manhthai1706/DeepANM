"""
Adaptive LASSO-based Edge Selection for DeepANM.

Replaces the brittle hard threshold (|W| > 0.015) with a statistically
principled model selection method borrowed from LiNGAM.

Adaptive LASSO procedure for each variable Xj:
    1. Candidate parents = all variables that precede j in causal order.
    2. Initial OLS fit → get |β_OLS|.
    3. Adaptive weights = 1 / (|β_OLS| + ε).
    4. LASSO with adaptive weights (equivalent to L1 penalty proportional to 1/|β_OLS|).
       → Coefficients of weak/irrelevant parents are pushed exactly to 0.
    5. Non-zero coefficient → confirmed edge parent → j.

Why this works better than fixed threshold:
    - OLS magnitudes adapt to each variable's scale automatically.
    - LASSO provides exact sparsity (zero vs non-zero), not noisy near-threshold values.
    - Cross-validation selects the optimal regularization per variable.
"""

import numpy as np


def _ols_fit(X_parents: np.ndarray, Xj: np.ndarray) -> np.ndarray:
    """OLS coefficients of Xj ~ Xparents (with intercept removed via centering)."""
    Xc = X_parents - X_parents.mean(axis=0)
    yc = Xj - Xj.mean()
    try:
        beta, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
        return beta
    except Exception:
        return np.ones(X_parents.shape[1])


def _adaptive_lasso_column(X_parents: np.ndarray, Xj: np.ndarray,
                            gamma: float = 1.0) -> np.ndarray:
    """
    Fit Xj ~ Xparents with Adaptive LASSO (cross-validated alpha).

    Parameters
    ----------
    X_parents : (n, p) array — potential parent variables
    Xj        : (n,) array — target variable
    gamma     : exponent for adaptive weights (1.0 is standard)

    Returns
    -------
    coef : (p,) array — zero entries mean no edge, non-zero mean confirmed edge
    """
    try:
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler

        n, p = X_parents.shape
        if p == 0:
            return np.array([])

        # 1. Scale for numerical stability
        scaler = StandardScaler()
        Xp_scaled = scaler.fit_transform(X_parents)
        yj_scaled = (Xj - Xj.mean()) / (Xj.std() + 1e-8)

        # 2. OLS for adaptive weights
        beta_ols = _ols_fit(Xp_scaled, yj_scaled)
        adap_weights = 1.0 / (np.abs(beta_ols) ** gamma + 1e-6)

        # 3. Re-weight columns: X_tilde[:,i] = X[:,i] / adaptive_weight[i]
        X_tilde = Xp_scaled / adap_weights[np.newaxis, :]

        # 4. LASSO with cross-validation on the re-weighted design matrix
        cv_folds = min(5, max(3, n // 50))
        lasso = LassoCV(cv=cv_folds, max_iter=2000, n_alphas=20,
                        fit_intercept=True, random_state=42)
        lasso.fit(X_tilde, yj_scaled)

        # 5. Back-transform coefficients to original space
        raw_coef = lasso.coef_ / adap_weights

        return raw_coef

    except Exception:
        # Fallback: plain OLS if sklearn fails
        return _ols_fit(X_parents, Xj)


def adaptive_lasso_dag(X: np.ndarray, causal_order: list) -> np.ndarray:
    """
    Build a binary DAG adjacency matrix using Adaptive LASSO edge selection.

    For each variable j (processed in causal order), run Adaptive LASSO of Xj
    on all its topological ancestors. Non-zero coefficients → edges.

    Parameters
    ----------
    X            : (n_samples, n_vars) array — observational data
    causal_order : list of int, root-first (from toposort module)

    Returns
    -------
    W_bin : (n_vars, n_vars) binary float array
            W_bin[i, j] = 1 means edge i → j was confirmed
    """
    n_vars = X.shape[1]
    W_bin = np.zeros((n_vars, n_vars), dtype=np.float32)

    for step, j in enumerate(causal_order):
        # All variables that precede j in the causal order are potential parents
        potential_parents = causal_order[:step]   # root-first slice
        if len(potential_parents) == 0:
            continue   # Root variable: no parents

        X_parents = X[:, potential_parents]
        Xj = X[:, j]

        coef = _adaptive_lasso_column(X_parents, Xj)

        for k, parent in enumerate(potential_parents):
            if abs(coef[k]) > 1e-8:    # Adaptive LASSO exact zero check
                W_bin[parent, j] = 1.0

    return W_bin


def adaptive_lasso_from_ate(ATE: np.ndarray, X: np.ndarray,
                             causal_order: list) -> np.ndarray:
    """
    Combine Neural ATE signal with Adaptive LASSO for more robust edge selection.

    Strategy:
        1. Run Adaptive LASSO on raw X to get structural edges.
        2. Intersect with ATE > eps to confirm only edges with measurable effect.

    This double-gate approach keeps edges that are:
        (a) statistically nonzero in regression AND
        (b) have detectable causal effect magnitude

    Returns
    -------
    W_bin : (n_vars, n_vars) binary float array
    """
    W_lasso = adaptive_lasso_dag(X, causal_order)
    # ATE signal as soft gate (low threshold — not for sparsity, just sanity check)
    ATE_gate = (np.abs(ATE) > 1e-3).astype(float)
    return (W_lasso * ATE_gate).astype(float)
