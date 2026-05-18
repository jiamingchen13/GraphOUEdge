"""
================================================================================
grOU / MCAR / OU Drift Estimation Toolkit
================================================================================

This module implements parameter estimation routines for three classes of
Lévy-driven continuous-time autoregressive processes, as developed in:

    Chen & Veraart (2026): "Edge-Indexed Network Time Series with Graph
    Ornstein-Uhlenbeck Dynamics".

The three model classes supported are:

1. **MCAR** (Multivariate Continuous-time AutoRegressive):
   A general p-th order multivariate CAR process driven by a Lévy process
   with triplet (b, Σ, F). The drift matrix Q ∈ R^{d×d} is estimated
   component-wise via a discretized maximum likelihood approach.

2. **OU** (Ornstein-Uhlenbeck):
   A diagonal-restricted MCAR process where Q is constrained to be diagonal,
   yielding a univariate OU model for each component.

3. **grOU** (Graph Ornstein-Uhlenbeck):
   A network-structured extension of the OU model for edge-indexed time series,
   as defined in Definition 2.1 of Chen & Veraart (2026). The drift matrix Q
   is parametrized by scalar autoregressive coefficients (alpha) and network
   interaction coefficients (beta) acting through weighted neighborhood matrices
   W^{(r)} at each lag l and stage r.

--------------------------------------------------------------------------------
Estimation Strategy (Theorem 3.2, Chen & Veraart 2026):
--------------------------------------------------------------------------------
Given discrete observations Y on a fine partition P and a coarser partition Q:

  - Time derivatives D^k Y are approximated by forward finite differences on P.
  - Stochastic integrals are approximated by left-hand Riemann sums on Q.
  - Jump contamination is removed via component-wise thresholding:
        ΔY_thresh = ΔY · 1{|ΔY| ≤ ν}
    where ν is a data-driven threshold sequence chosen to isolate the
    continuous martingale part of the Lévy process.
  - The Lévy covariance Σ is estimated iteratively via the disentanglement
    procedure of Gegler (2011), separating Brownian from jump increments.
  - The full estimation loop iterates between drift and Lévy triplet estimation
    until convergence (or a maximum iteration count is reached).

--------------------------------------------------------------------------------
Key Functions:
--------------------------------------------------------------------------------
  state_space            - Construct finite-difference state-space representation
  estimate_grOU_drift    - Estimate grOU drift parameters (global or not)
  estimate_OU_drift      - Estimate OU drift parameters (global or not)
  estimate_MCAR_drift    - Estimate MCAR drift parameters
  estimate_MCAR          - Full iterative estimation loop (drift + Lévy triplet)
  recover_BDLP           - Recover background driving Lévy process increments
  estimate_Sigma_L       - Iterative Brownian covariance estimation
  estimate_Sigma         - Covariance from (irregularly spaced) BM increments
  disentangle_BM         - Identify Brownian-only increments via thresholding
  estimate_integral_Levy_measure - Estimate functionals of the Lévy measure
  choose_nu              - Data-driven threshold sequence selection
  inv_vectorize_AA       - Reshape vectorized parameters into matrix list
  estimator_to_QQ        - Convert grOU parameter vector to matrix form
  states_close           - Check convergence of iterative parameter estimates

--------------------------------------------------------------------------------
References:
--------------------------------------------------------------------------------
  - Chen & Veraart (2026): Edge-indexed network time series with graph
  Ornstein-Uhlenbeck dynamics
  - Lucchese, Pakkanen & Veraart (2026): Estimation and Inference for Multivariate
  Continuous Time Autoregressive Processes
  - Gegler (2011): Estimation of the characteristics of a Lévy process
  - Marquardt & Stelzer (2007): Multivariate CARMA Processes
================================================================================
"""

import numpy as np
from typing import Callable, List
import warnings

# ============================================================
# STATE-SPACE CONSTRUCTION
# ============================================================

def state_space(Y: np.ndarray, p: int, P: np.ndarray) -> np.ndarray:
    """
    Construct state-space representation of process from observation Y up to order p-1 via finite differences.

    Given a d-dimensional process Y observed at N+1 time points on partition P,
    this function builds the pd-dimensional state vector:
        X = [Y, D^1 Y, D^2 Y, ..., D^{p-1} Y]
    where D^k Y denotes the k-th order forward finite difference approximation
    of the k-th derivative of Y, as described in Appendix C.1 of Chen & Veraart (2026).

    :param Y: MCAR process observation, (d, N+1) np.ndarray
    :param p: state space order (number of lags), int
    :param P: partition over which to simulate the MCAR process [0=t_0, t_1, ..., t_N = T], (N+1, ) np.ndarray
    :return X: state space representation (first d entries are Y), (pd, N+1) np.ndarray
    """
    d, N = Y.shape
    # Allocate state matrix: p blocks of d rows, N columns (drop last time point)
    X = np.zeros((p * d, N))
    # The 0-th block is the process itself
    X[:d, :] = Y

    # Iteratively compute finite-difference approximations of derivatives D^1 Y, ..., D^{p-1} Y
    # Each derivative is approximated as: D^k Y_t ≈ (D^{k-1} Y_{t+1} - D^{k-1} Y_t) / (t_{n+1} - t_n)
    for i in range(1, p):
        X[d * i:d * (i + 1), :-1] = np.diff(X[d * (i - 1):d * i, :], axis=1) / (P[1:] - P[:-1])
    return X

# ============================================================
# grOU DRIFT ESTIMATION
# ============================================================

def estimate_grOU_drift(Y: np.ndarray, A: np.ndarray, W_C: List, R_L: List, C: int, P: np.ndarray, Q: np.ndarray, b: np.ndarray,
                         gamma: float, nu: np.ndarray = None, Sigma: np.ndarray = None, with_cov: bool = False,
                        is_global: bool = False, lam: float = 1e-8):
    """
    Estimate grOU drift parameters from the realization of a grOU process with driving Levy triplet (b, Sigma, F).

    Implements the discretized MLE from Theorem 3.2 of Chen & Veraart (2026), which approximates:
        K_t     = -∫ H_s Σ^{-1} dD^{L-1}Y^c_{Q(0),s}      (stochastic integral term)
        [K]_t   =  ∫ H_s Σ^{-1} H_s^T ds                   (quadratic variation term)
        θ_hat   = [K]_t^{-1} K_t                             (MLE estimator)

    The Lévy drift parameter b corresponds to no truncation t(x) = 0 when the process has
    finite jump activity, thus E[L_1] = b + ∫_{R^d} z F(dz).

    Two estimation modes are supported:
      - is_global=True:  Scalar autoregressive coefficients (alpha_l shared across all edges at lag l),
                         yielding a compact parameter vector of length p + sum(R_L) * C.
      - is_global=False: Component-wise coefficients (alpha_{e,l} per edge),
                         yielding a parameter vector of length p*d + sum(R_L) * C.

    :param Y:        grOU process observation, (d, N+1) np.ndarray
    :param A:        Graph adjacency matrix, maximum stage of (d, d) np.ndarray
    :param W_C:      Edge weights for Covariates, list of np.ndarray
    :param R_L:      Maximum stage for each Lag, list of int
    :param C:        Number of Covariates, int
    :param P:        Finer partition [0=s_0,...,s_N=t], (N+1, ) np.ndarray
    :param Q:        Coarser partition [0=u_0,...,u_M=t], (M+1, ) np.ndarray
    :param b:        Drift of Levy process (no-truncation convention), (d, ) np.ndarray
    :param gamma:    Hyperparameter for defining the critical region (must be > 1), float
    :param nu:       Thresholding sequence; if None, chosen from data, (d, M) np.ndarray
    :param Sigma:    Covariance matrix of the driving Levy process; if None and with_cov=True,
                     estimated from data, (d, d) np.ndarray
    :param with_cov: If True, return the estimated covariance matrix [K] of the estimator, bool
    :param is_global:If True, return the global (scalar alpha) estimator, bool
    :param lam:      Ridge regularization parameter to avoid singularity, float
    :return:
        if with_cov is True and Sigma is not None:
            vec_theta_hat: vectorized estimated grCAR parameters
            KK:            estimated Fisher information matrix [K]
            K:             estimated score vector K
        if with_cov is True and Sigma is None:
            vec_theta_hat, KK, Sigma_hat (estimated Brownian covariance)
        else:
            vec_theta_hat: vectorized estimated grCAR parameters
    """
    # Number of lags
    p = len(R_L)
    d = Y.shape[0]

    # ---- Validate that Q is a coarsening of P ----
    if not all([u in P for u in Q]):
        raise ValueError('P must be refinement of Q.')

    # Build boolean index to restrict fine-grid state-space to coarse-grid Q points
    indices = [False] * len(P)
    j = 0
    for i, s in enumerate(P):
        if s == Q[j]:
            indices[i] = True
            j += 1

    # Build state-space representation on fine grid P, then restrict to coarse grid Q
    X = state_space(Y, p, P)
    X = X[:, indices]
    x0 = X[:, 0]    # Initial state vector

    # ---- Determine inverse of Sigma (or estimate it) ----
    if with_cov and Sigma is not None:
        # Sigma is known: use pseudo-inverse for numerical stability
        Sigma_inv = np.linalg.pinv(Sigma)
    elif with_cov and Sigma is None:
        # Sigma is unknown: estimate it from the recovered BDLP increments
        # Step 1: Get a preliminary drift estimate (without Sigma weighting)
        global_QQ_hat = estimate_grOU_drift(Y, A, W_C, R_L, C, P, Q, b, gamma, with_cov=False, is_global=is_global)
        # Step 2: Convert to matrix form and recover driving Levy increments
        QQ_hat = estimator_to_QQ(global_QQ_hat, R_L, C, W_C, A, d)
        DeltaL = recover_BDLP(Y, p, P, Q, QQ_hat)
        # Step 3: Iteratively estimate Brownian covariance Sigma via thresholding
        Sigma_hat = estimate_Sigma_L(DeltaL, Q, gamma)
        Sigma_inv = np.linalg.pinv(Sigma_hat)
    else:
        # Covariance weighting not required: use identity (equivalent to unweighted OLS)
        Sigma_inv = np.eye(d)

    # ---- Compute thresholded increments of the (p-1)-th derivative ----
    # DY approximates the increments of the continuous martingale part of D^{p-1}Y,
    # after subtracting the initial value and the Levy drift contribution b*dt
    DY = np.diff(X[-d:, :] - x0[-d:].reshape(-1, 1) - b.reshape(-1, 1) * Q.reshape(1, -1))

    # Choose thresholding sequence nu from data if not provided
    # nu_{i,m} ≈ sqrt(2 * gamma * Sigma_ii * Delta_m * log(M)) separates BM from jumps
    if nu is None:
        nu = np.vstack([choose_nu(DY[i:i + 1, :], Q, gamma) for i in range(d)])

    # Apply threshold: keep only increments below nu (Brownian-dominated increments)
    # This implements the indicator 1{|ΔY| ≤ ν} from Theorem 3.2
    DY_thresh = DY * (np.abs(DY) < nu)

    # ============================================================
    # GLOBAL ESTIMATOR (scalar alpha_l shared across all edges)
    # ============================================================
    if is_global:
        # ---- Compute score vector K ----
        # K[idx]         = -tr(Sigma_inv @ ∫ X_k dY_thresh^T)   for baseline alpha_l
        # K[idx+c*R+r+1] = -tr((W_C[c] ⊙ A[r])^T Sigma_inv @ integral)  for beta_{l,r,c}
        K = np.zeros(p + sum(R_L) * C)
        idx = 0
        for k in range(p):
            integral = np.zeros((d, d))
            # Select the appropriate lag block of the state vector
            if k == 0:
                integrand = X[-d:, :]   # D^0 Y = Y (lag 0)
            else:
                integrand = X[-d - k * d:-k * d, :]     # D^k Y (lag k)
            for i in range(d):
                for j in range(d):
                    # Left-hand Riemann sum approximating the Itô integral ∫ X_i dY^c_j
                    # (converges in L2 to the Itô integral)
                    integral[i, j] = np.sum(integrand[i, :-1] * DY_thresh[j, :])

            # Baseline autoregressive score contribution: -tr(Σ^{-1} ∫ X_k dY^T)
            K[idx] = -np.sum(np.multiply(Sigma_inv, integral))
            # Network interaction score contributions for each covariate c and stage r
            for c in range(C):
                for r in range(R_L[k]):
                    K[idx + c * R_L[k] + r + 1] = -np.sum(np.multiply(np.matmul(np.multiply(W_C[c], A[r]).T, Sigma_inv), integral))
            idx += C * R_L[k] + 1

        # ---- Compute Fisher information matrix [K] ----
        # [K][k,l] = tr(Sigma_inv @ ∫ X_k X_l^T dt)  (Riemann integral, converges a.s.)
        KK = np.zeros((p + sum(R_L) * C, p + sum(R_L) * C))
        idx_k = 0
        for k in range(p):
            idx_l = 0
            for l in range(p):
                # Select lag-k and lag-l integrand blocks
                if k == 0:
                    integrand_one = X[-d:, :]
                else:
                    integrand_one = X[-d - k * d:-k * d, :]
                if l == 0:
                    integrand_two = X[-d:, :]
                else:
                    integrand_two = X[-d - l * d:-l * d, :]
                integral = np.zeros((d, d))
                for i in range(d):
                    for j in range(d):
                        # Left-hand Riemann sum approximating ∫ X_k,i X_l,j dt
                        # (converges a.s. to the Riemann integral for any midpoint choice)
                        integral[i, j] = np.sum(integrand_one[i, :-1] * integrand_two[j, :-1] * (Q[1:] - Q[:-1]))

                # Fill [K] block for (k,l) pair: baseline and network interaction terms
                KK[idx_k, idx_l] = np.sum(np.multiply(Sigma_inv, integral))
                for c in range(C):
                    for r in range(R_L[k]):
                        KK[idx_k + c * R_L[k] + r + 1, idx_l] = np.sum(np.multiply(np.matmul(np.multiply(W_C[c], A[r]).T, Sigma_inv), integral))
                for e in range(C):
                    for s in range(R_L[l]):
                        KK[idx_k, idx_l + e * R_L[l] + s + 1] = np.sum(np.multiply(np.matmul(Sigma_inv, np.multiply(W_C[e], A[s])), integral))
                for c in range(C):
                    for r in range(R_L[k]):
                        for e in range(C):
                            for s in range(R_L[l]):
                                KK[idx_k + c * R_L[k] + r + 1, idx_l + e * R_L[l] + s + 1] = np.sum(np.multiply(np.matmul(np.matmul(np.multiply(W_C[c], A[r]).T, Sigma_inv), np.multiply(W_C[e], A[s])), integral))
                idx_l += C * R_L[l] + 1
            idx_k += C * R_L[k] + 1

        # ---- Solve regularized linear system: θ_hat = ([K] + λI)^{-1} K ----
        vec_theta_hat = np.linalg.solve(KK + lam * np.eye(KK.shape[0]), K)

        # ---- Expand global (scalar) alpha to per-component (d-dimensional) form ----
        # The global estimator returns one scalar alpha_l per lag, which is broadcast
        # to all d edge components to form the local-format parameter vector.
        vec_theta_hat_ext = np.zeros(p * d + sum(R_L) * C)
        idx_old = 0
        idx_new = 0
        for k in range(p):
            # Number of network parameters at lag k
            Lk = int(C * R_L[k])
            # Duplicate the scalar baseline alpha_k to all d components
            vec_theta_hat_ext[idx_new: idx_new + d] = vec_theta_hat[idx_old]
            # Copy the network interaction parameters (beta) for lag k
            vec_theta_hat_ext[idx_new + d: idx_new + d + Lk] = vec_theta_hat[idx_old + 1: idx_old + 1 + Lk]
            idx_old += 1 + Lk
            idx_new += d + Lk
        vec_theta_hat = vec_theta_hat_ext

    # ============================================================
    # LOCAL ESTIMATOR (per-component alpha_{e,l} for each edge e)
    # ============================================================
    else:
        # ---- Compute score vector K (local/component-wise) ----
        K = np.zeros(p * d + sum(R_L) * C)
        idx = 0
        for k in range(p):
            # Select lag-k integrand block
            if k == 0:
                integrand = X[-d:, :]
            else:
                integrand = X[-d - k * d:-k * d, :]

            # Compute the full d×d integral matrix for network terms
            integral_ = integrand[:, :-1] @ DY_thresh.T

            # Compute per-component score for the diagonal (alpha) parameters
            for i in range(d):
                integral = np.zeros(d)
                integrand_row = integrand[i, :]
                for j in range(d):
                    # Left-hand Riemann sum: ∫ X_{k,i} dY^c_j
                    integral[j] = np.sum(integrand_row[:-1] * DY_thresh[j, :])
                # Score for the i-th diagonal entry: -(Σ^{-1} ∫ X_{k,i} dY^c)_i
                K[idx + i] = -np.matmul(Sigma_inv, integral)[i]

            # Score for the i-th diagonal entry: -(Σ^{-1} ∫ X_{k,i} dY^c)_i
            base_offset = idx + d
            for c in range(C):
                for r in range(R_L[k]):
                    K[base_offset + c * R_L[k] + r] = -np.sum(np.multiply(np.matmul(np.multiply(W_C[c], A[r]).T, Sigma_inv), integral_))
            idx += C * R_L[k] + d

        # ---- Compute Fisher information matrix [K] (local) ----
        KK = np.zeros((p * d + sum(R_L) * C, p * d + sum(R_L) * C))
        idx_k = 0
        for k in range(p):
            if k == 0:
                integrand_one = X[-d:, :]
            else:
                integrand_one = X[-d - k * d:-k * d, :]
            idx_l = 0
            for l in range(p):
                if l == 0:
                    integrand_two = X[-d:, :]
                else:
                    integrand_two = X[-d - l * d:-l * d, :]

                # Vectorized Riemann sum: ∫ X_{k,i} X_{l,j} dt for all (i,j) pairs
                integral = (integrand_one[:, :-1] * (Q[1:] - Q[:-1])) @ integrand_two[:, :-1].T

                # Fill diagonal (alpha × alpha) block
                KK[idx_k: idx_k + d, idx_l: idx_l + d] = np.multiply(Sigma_inv, integral)

                # Fill off-diagonal (beta × alpha) blocks
                for c in range(C):
                    for r in range(R_L[k]):
                        KK[idx_k + c * R_L[k] + r + d, idx_l: idx_l + d] = np.sum(
                            np.multiply(np.matmul(np.multiply(W_C[c], A[r]).T, Sigma_inv), integral), axis=0)
                # Fill off-diagonal (alpha × beta) blocks
                for e in range(C):
                    for s in range(R_L[l]):
                        KK[idx_k: idx_k + d, idx_l + e * R_L[l] + s + d] = np.sum(
                            np.multiply(np.matmul(Sigma_inv, np.multiply(W_C[e], A[s])), integral), axis=1)
                # Fill (beta × beta) blocks
                for c in range(C):
                    for r in range(R_L[k]):
                        for e in range(C):
                            for s in range(R_L[l]):
                                KK[idx_k + c * R_L[k] + r + d, idx_l + e * R_L[l] + s + d] = np.sum(np.multiply(
                                    np.matmul(np.matmul(np.multiply(W_C[c], A[r]).T, Sigma_inv),
                                              np.multiply(W_C[e], A[s])), integral))
                idx_l += C * R_L[l] + d
            idx_k += C * R_L[k] + d

        # Solve regularized system: θ_hat = ([K] + λI)^{-1} K
        vec_theta_hat = np.linalg.solve(KK + lam * np.eye(KK.shape[0]), K)

    # ---- Return results ----
    if with_cov and Sigma is not None:
        return vec_theta_hat, KK, K
    elif with_cov and Sigma is None:
        return vec_theta_hat, KK, Sigma_hat
    else:
        return vec_theta_hat

# ============================================================
# OU DRIFT ESTIMATION
# ============================================================

def estimate_OU_drift(Y: np.ndarray, R_L: List, C: int, P: np.ndarray, Q: np.ndarray, b: np.ndarray, gamma: float,
                      nu: np.ndarray = None, Sigma: np.ndarray = None, with_cov: bool = False, is_global: bool = False,
                      lam = 1e-8):
    """
    Estimate OU drift parameters from the realization of a process with driving Levy triplet (b, Sigma, F).

    This is a diagonal-restricted version of the MCAR estimator: the drift matrix Q is
    constrained to be diagonal, so each component evolves as an independent OU process.
    The estimation follows the same discretized MLE framework as estimate_grOU_drift,
    but without network interaction terms (no beta parameters).

    Here the Levy drift parameter b corresponds to no truncation t(x) = 0 when the process
    has finite jump activity, thus E[L_1] = b + ∫_{R^d} z F(dz).

    :param Y:        Process observation, (d, N+1) np.ndarray
    :param R_L:      Maximum stage for each Lag (used for lag structure only), List
    :param C:        Number of Covariates, int
    :param P:        Finer partition [0=s_0,...,s_N=t], (N+1, ) np.ndarray
    :param Q:        Coarser partition [0=u_0,...,u_M=t], (M+1, ) np.ndarray
    :param b:        Drift of Levy process (no-truncation convention), (d, ) np.ndarray
    :param gamma:    Hyperparameter for defining the critical region (must be > 1), float
    :param nu:       Thresholding sequence; if None, chosen from data, (d, M) np.ndarray
    :param Sigma:    Covariance matrix of the driving Levy process; if None and with_cov=True,
                     estimated from data, (d, d) np.ndarray
    :param with_cov: If True, return the estimated covariance matrix [K], bool
    :param is_global:If True, return the global (scalar per lag) estimator, bool
    :param lam:      Ridge regularization parameter, float
    :return:
        if with_cov is True and Sigma is not None:
            vec_theta_hat: estimated OU parameters (shape (p, d) or (p, ) depending on is_global)
            KK:            estimated Fisher information matrix
            K:             estimated score vector
        if with_cov is True and Sigma is None:
            vec_theta_hat, KK, Sigma_hat
        else:
            vec_theta_hat
    """
    p = len(R_L)
    d = Y.shape[0]

    # Validate that Q is a coarsening of P
    if not all([u in P for u in Q]):
        raise ValueError('P must be refinement of Q.')

    # Build index to restrict state-space to coarse-grid Q
    indices = [False] * len(P)
    j = 0
    for i, s in enumerate(P):
        if s == Q[j]:
            indices[i] = True
            j += 1

    # Build state-space on fine grid, then restrict to Q
    X = state_space(Y, p, P)
    X = X[:, indices]
    x0 = X[:, 0]

    # ---- Determine Sigma_inv ----
    if with_cov and Sigma is not None:
        Sigma_inv = np.linalg.pinv(Sigma)
    elif with_cov and Sigma is None:
        # Estimate Sigma: get preliminary diagonal drift estimate, recover BDLP, then estimate Sigma
        theta_hat = estimate_OU_drift(Y, R_L, C, P, Q, b, gamma, with_cov=False, is_global=is_global)
        QQ_hat = np.zeros((p, d, d))
        idx = np.arange(d)
        # Diagonal restriction: place theta_hat values on diagonal of each lag matrix
        QQ_hat[:, idx, idx] = theta_hat
        DeltaL = recover_BDLP(Y, p, P, Q, QQ_hat)
        Sigma_hat = estimate_Sigma_L(DeltaL, Q, gamma)
        Sigma_inv = np.linalg.pinv(Sigma_hat)
    else:
        # No Sigma weighting needed
        Sigma_inv = np.eye(d)

    # ---- Compute thresholded increments of D^{p-1}Y ----
    # Subtract initial value and Levy drift to isolate the continuous martingale part
    DY = np.diff(X[-d:, :] - x0[-d:].reshape(-1, 1) - b.reshape(-1, 1) * Q.reshape(1, -1))

    # Data-driven threshold selection if not provided
    if nu is None:
        nu = np.vstack([choose_nu(DY[i:i + 1, :], Q, gamma) for i in range(d)])

    # Apply threshold to isolate Brownian increments
    DY_thresh = DY * (np.abs(DY) < nu)

    # ============================================================
    # GLOBAL OU ESTIMATOR (one scalar theta_l per lag)
    # ============================================================
    if is_global:
        # Score vector K: one entry per lag
        K = np.zeros(p)
        for k in range(p):
            integral = np.zeros((d, d))
            if k == 0:
                integrand = X[-d:, :]
            else:
                integrand = X[-d - k * d:-k * d, :]
            for i in range(d):
                for j in range(d):
                    # Left-hand Riemann sum: ∫ X_{k,i} dY^c_j
                    integral[i, j] = np.sum(integrand[i, :-1] * DY_thresh[j, :])
            # Global score: -tr(Σ^{-1} ∫ X_k dY^T)
            K[k] = -np.sum(np.multiply(Sigma_inv, integral))

        # Fisher information matrix [K]: p × p matrix
        KK = np.zeros((p, p))
        for k in range(p):
            for l in range(p):
                if k == 0:
                    integrand_one = X[-d:, :]
                else:
                    integrand_one = X[-d - k * d:-k * d, :]
                if l == 0:
                    integrand_two = X[-d:, :]
                else:
                    integrand_two = X[-d - l * d:-l * d, :]
                integral = np.zeros((d, d))
                for i in range(d):
                    for j in range(d):
                        # Left-hand Riemann sum: ∫ X_{k,i} X_{l,j} dt
                        integral[i, j] = np.sum(integrand_one[i, :-1] * integrand_two[j, :-1] * (Q[1:] - Q[:-1]))

                KK[k, l] = np.sum(np.multiply(Sigma_inv, integral))

        # Solve and broadcast scalar theta_l to all d components
        vec_theta_hat = np.repeat(np.linalg.solve(KK + lam * np.eye(KK.shape[0]), K)[:, np.newaxis], d, axis=1)

    # ============================================================
    # LOCAL OU ESTIMATOR (per-component theta_{i,l} for each component i)
    # ============================================================
    else:
        # Score vector K: p*d entries (one per component per lag)
        K = np.zeros(p * d)
        for k in range(p):
            for i in range(d):
                integral = np.zeros(d)
                # i-th row of the lag-k state block: D^{p-k-1}Y_i
                integrand = X[-(k + 1) * d + i, :]
                for j in range(d):
                    # Left-hand Riemann sum: ∫ X_{k,i} dY^c_j
                    integral[j] = np.sum(integrand[:-1] * DY_thresh[j, :])
                # Score for component i at lag k: -(Σ^{-1} integral)_i
                K[k * d + i] = - np.matmul(Sigma_inv, integral)[i]

        # Fisher information matrix [K]: (p*d) × (p*d)
        KK = np.zeros((p * d, p * d))

        for k in range(p):
            for i in range(d):
                integrand_one = X[-(k + 1) * d + i, :]
                for l in range(p):
                    for j in range(d):
                        integrand_two = X[-(l + 1) * d + j, :]
                        # Riemann sum: ∫ X_{k,i} X_{l,j} dt
                        integral = np.sum(integrand_one[:-1] * integrand_two[:-1] * np.diff(Q))
                        # Diagonal Sigma weighting (OU diagonal structure)
                        KK[k * d + i, l * d + j] = Sigma_inv[i, j] * integral

        # Solve regularized system and reshape to (p, d)
        vec_theta_hat = np.linalg.solve(KK + lam * np.eye(KK.shape[0]), K).reshape(p, d)

    # ---- Return results ----
    if with_cov and Sigma is not None:
        return vec_theta_hat, KK, K
    elif with_cov and Sigma is None:
        return vec_theta_hat, KK, Sigma_hat
    else:
        return vec_theta_hat

# ============================================================
# MCAR DRIFT ESTIMATION
# ============================================================

def estimate_MCAR_drift(Y: np.ndarray, R_L: List, P: np.ndarray, Q: np.ndarray, b: np.ndarray, gamma: float, nu: np.ndarray = None,
                        with_cov: bool = False, Sigma: np.ndarray = None):
    """
    Estimate MCAR drift parameters from the realisation of a MCAR process with driving Levy triplet (b, Sigma, F).

    This is the fully general (unconstrained) version of the drift estimator, where the
    drift matrix Q_k ∈ R^{d×d} at each lag k is estimated without any network or diagonal
    structure. The estimator follows the same discretized MLE framework as the grOU/OU
    estimators, but operates on the full d²-dimensional vectorization of each Q_k.

    Here the Levy drift parameter b corresponds to no truncation t(x) = 0 when the process
    has finite jump activity, thus E[L_1] = b + ∫_{R^d} z F(dz).

    :param Y:        MCAR process observation, (d, N+1) np.ndarray
    :param R_L:      Maximum stage for each Lag (used for lag count p = len(R_L)), List
    :param P:        Finer partition [0=s_0,...,s_N=t], (N+1, ) np.ndarray
    :param Q:        Coarser partition [0=u_0,...,u_M=T], (M+1, ) np.ndarray
    :param b:        Drift of Levy process (no-truncation convention), (d, ) np.ndarray
    :param gamma:    Hyperparameter for defining the critical region (must be > 1), float
    :param nu:       Thresholding sequence; if None, chosen from data, (d, M) np.ndarray
    :param with_cov: If True, return the estimated covariance matrix [K], bool
    :param Sigma:    Covariance matrix of the driving Levy process; if None and with_cov=True,
                     estimated from data, (d, d) np.ndarray
    :return:
        if with_cov is True and Sigma is not None:
            vec_QQ_hat: vectorized estimated MCAR parameters, (p*d², ) np.ndarray
            KK:         estimated Fisher information matrix, (p*d², p*d²) np.ndarray
            K:          estimated score vector, (p*d², ) np.ndarray
        if with_cov is True and Sigma is None:
            vec_QQ_hat, KK, Sigma_hat
        else:
            QQ_hat: list of p estimated (d, d) drift matrices
    """
    p = len(R_L)
    d = Y.shape[0]

    # Validate that Q is a coarsening of P
    if not all([u in P for u in Q]):
        raise ValueError('P must be refinement of Q.')

    # Build index to restrict state-space to coarse-grid Q
    indices = [False] * len(P)
    j = 0
    for i, s in enumerate(P):
        if s == Q[j]:
            indices[i] = True
            j += 1

    # Build state-space on fine grid, then restrict to Q
    X = state_space(Y, p, P)
    X = X[:, indices]
    x0 = X[:, 0]

    # ---- Determine Sigma_inv ----
    if with_cov and Sigma is not None:
        Sigma_inv = np.linalg.pinv(Sigma)
    elif with_cov and Sigma is None:
        # Estimate Sigma from a preliminary MCAR drift estimate
        QQ_hat = estimate_MCAR_drift(Y, R_L, P, Q, b, gamma, with_cov=False)
        DeltaL = recover_BDLP(Y, p, P, Q, QQ_hat)
        Sigma_hat = estimate_Sigma_L(DeltaL, Q, gamma)
        Sigma_inv = np.linalg.pinv(Sigma_hat)
    else:
        Sigma_inv = np.eye(d)

    # ---- Compute thresholded increments of D^{p-1}Y ----
    DY = np.diff(X[-d:, :] - x0[-d:].reshape(-1, 1) - b.reshape(-1, 1) * Q.reshape(1, -1))

    if nu is None:
        nu = np.vstack([choose_nu(DY[i:i + 1, :], Q, gamma) for i in range(d)])

    DY_thresh = DY * (np.abs(DY) < nu)

    # ---- Compute score vector K ----
    # K has p*d² entries: for each lag k and each row i of Q_k, a d-dimensional block
    # K[k*d²+i*d : k*d²+(i+1)*d] = -Σ^{-1} ∫ X_{k,i} dY^c
    K = np.zeros(p * d ** 2)
    for k in range(p):
        for i in range(d):
            integral = np.zeros((d,))
            # i-th row of lag-k state block: D^{p-k-1}Y_i
            integrand = X[-(k + 1) * d + i, :]
            for j in range(d):
                # Left-hand Riemann sum: ∫ X_{k,i} dY^c_j
                integral[j] = np.sum(integrand[:-1] * DY_thresh[j, :])
            # Score block for row i of Q_k
            K[k * d ** 2 + i * d:k * d ** 2 + (i + 1) * d] = - np.matmul(Sigma_inv, integral).reshape(-1)

    # ---- Compute Fisher information matrix [K] ----
    # KK has shape (p*d², p*d²): for each pair (k,i) and (l,j),
    # the (d×d) block is Σ^{-1} * ∫ X_{k,i} X_{l,j} dt
    KK = np.zeros((p * d ** 2, p * d ** 2))
    for k in range(p):
        for i in range(d):
            for l in range(p):
                for j in range(d):
                    integrand_one = X[-(k + 1) * d + i, :]
                    integrand_two = X[-(l + 1) * d + j, :]
                    # Left-hand Riemann sum: ∫ X_{k,i} X_{l,j} dt
                    integral = np.sum(integrand_one[:-1] * integrand_two[:-1] * np.diff(Q))
                    # Fill d×d block with Sigma_inv * integral (scalar broadcast)
                    KK[k * d ** 2 + i * d:k * d ** 2 + (i + 1) * d, l * d ** 2 + j * d:l * d ** 2 + (
                            j + 1) * d] = Sigma_inv * integral

    # Solve the linear system (no ridge here; KK should be invertible for MCAR)
    vec_QQ_hat = np.linalg.solve(KK, K)

    # ---- Return results ----
    if with_cov and Sigma is not None:
        return vec_QQ_hat, KK, K
    elif with_cov and Sigma is None:
        return vec_QQ_hat, KK, Sigma_hat
    else:
        # Reshape vectorized parameters back to list of (d, d) matrices
        QQ_hat = inv_vectorize_AA(vec_QQ_hat, p)
        return QQ_hat

# ============================================================
# FULL ITERATIVE MCAR ESTIMATION (DRIFT + LÉVY TRIPLET)
# ============================================================

def estimate_MCAR(Model: str, Y: np.ndarray, P: np.ndarray, Q: np.ndarray, A: np.ndarray, W_C: List, R_L: List, C: int,
                gamma: float, epsilon: float = 0.01, max_iter: int = 200, QQ: np.ndarray = None, b: np.ndarray = None,
                Sigma: np.ndarray = None, f: np.ndarray = None, mu: np.ndarray = None, mu_estimator: str = 'MCAR_mean',
                is_global: bool = False):
    """
    Full iterative estimation of MCAR/OU/grOU parameters and Lévy triplet (b, Sigma).

    Implements the iterative procedure described in Section H.1 of Lucchese et al. (2026):
      1. Initialize b_hat = 0 (or use provided b).
      2. Estimate drift matrices QQ_hat using current b_hat.
      3. Recover BDLP increments ΔL from QQ_hat.
      4. Estimate Sigma_hat from ΔL via iterative thresholding (estimate_Sigma_L).
      5. Estimate mu_hat (mean of L_1) and f_hat (jump component of mean).
      6. Update b_hat = mu_hat - f_hat.
      7. Repeat steps 2-6 until convergence or max_iter is reached.
         Loop detection: if parameters revisit a previous state, average over the loop.

    The Lévy drift parameter b corresponds to no truncation t(x) = 0 when the process
    has finite jump activity, so mu := E[L_1] = b + ∫_{R^d} z F(dz) =: b + f.

    :param Model:         Model type: 'OU', 'grOU', or 'MCAR', str
    :param Y:             MCAR process observation, (d, N+1) np.ndarray
    :param P:             Finer partition [0=s_0,...,s_N=t], (N+1, ) np.ndarray
    :param Q:             Coarser partition [0=u_0,...,u_M=t], (M+1, ) np.ndarray
    :param A:             Graph adjacency matrix, (d, d) np.ndarray
    :param W_C:           Edge weights for Covariates, List of np.ndarray
    :param R_L:           Maximum stage for each Lag, List of int
    :param C:             Number of Covariates, int
    :param gamma:         Hyperparameter for defining the critical region (must be > 1), float
    :param epsilon:       Convergence tolerance for iterative scheme, float
    :param max_iter:      Maximum number of iterations before forced termination, int
    :param QQ:            Fixed drift matrices (if None, estimated), (p, d, d) np.ndarray
    :param b:             Fixed Levy drift (if None, estimated), (d, ) np.ndarray
    :param Sigma:         Fixed Brownian covariance (if None, estimated), (d, d) np.ndarray
    :param f:             Fixed jump mean ∫ z F(dz) (if None, estimated), (d, ) np.ndarray
    :param mu:            Fixed total Levy mean E[L_1] (if None, estimated), (d, ) np.ndarray
    :param mu_estimator:  Method for estimating mu: 'Levy_slope' or 'MCAR_mean', str
    :param is_global:     If True, use global (scalar alpha) estimator for OU/grOU, bool
    :return:
        QQ_hat:          Estimated drift matrices, (p, d, d) np.ndarray
        (b_hat, Sigma_hat): Estimated Levy drift and Brownian covariance
    """
    p = len(R_L)
    d = Y.shape[0]

    # ---- Initialize parameters ----
    b_hat = np.zeros(d) if b is None else b     # Start with zero drift
    QQ_hat = np.zeros((p, d, d))
    idx = np.arange(d)

    # ---- Initial drift estimate ----
    if Model == 'OU':
        theta_hat = estimate_OU_drift(Y, R_L, C, P, Q, b_hat, gamma, with_cov=False, is_global=is_global) if QQ is None else QQ
        QQ_hat[:, idx, idx] = theta_hat   # Place on diagonal for OU model
    elif Model == 'grOU':
        global_QQ_hat = estimate_grOU_drift(Y, A, W_C, R_L, C, P, Q, b_hat, gamma) if QQ is None else QQ
        QQ_hat = estimator_to_QQ(global_QQ_hat, R_L, C, W_C, A, d)    # Convert to matrix form
    elif Model == 'MCAR':
        QQ_hat = estimate_MCAR_drift(Y, R_L, P, Q, b_hat, gamma) if QQ is None else QQ
    else:
        raise ValueError('Model not recognized')

    # ---- Initial Lévy triplet estimates ----
    DeltaL = recover_BDLP(Y, p, P, Q, QQ_hat)
    Sigma_hat = estimate_Sigma_L(DeltaL, Q, gamma) if Sigma is None else Sigma

    # Estimate mu = E[L_1] using chosen method
    if mu_estimator == 'Levy_slope':
        # Slope of the recovered Lévy process: L_T / T
        mu_hat = DeltaL.sum(axis=1) / P[-1] if mu is None else mu
    elif mu_estimator == 'MCAR_mean':
        # MCAR stationary mean: Q_p^{-1} E[Y] (from Proposition 2.1)
        mu_hat = QQ_hat[p-1].dot(Y.mean(axis=1)) if mu is None else mu
    else:
        raise ValueError("mu_estimator must be 'Levy_slope' or 'MCAR_mean'.")

    # Estimate jump mean f = ∫ z F(dz) and compute b = mu - f
    f_hat = estimate_integral_Levy_measure(DeltaL, Q, gamma, Sigma_hat, K=lambda x: True, f=lambda x: x) if f is None else f
    b_hat = mu_hat - f_hat if b is None else b
    converged = False

    # ---- Track parameter history for loop detection ----
    QQs = [QQ_hat]
    bs = [b_hat]
    Sigmas = [Sigma_hat]
    n_iter = 0

    # ---- Main iteration loop ----
    while not converged:
        QQ_hat_new = np.zeros((p, d, d))
        n_iter += 1

        # Step 2: Update drift estimate with current b_hat
        if Model == 'OU':
            theta_hat_new = estimate_OU_drift(Y, R_L, C, P, Q, b_hat, gamma, with_cov=False, is_global=is_global) if QQ is None else QQ
            QQ_hat_new[:, idx, idx] = theta_hat_new
        elif Model == 'grOU':
            global_QQ_hat_new = estimate_grOU_drift(Y, A, W_C, R_L, C, P, Q, b_hat, gamma) if QQ is None else QQ
            QQ_hat_new = estimator_to_QQ(global_QQ_hat_new, R_L, C, W_C, A, d)
        elif Model == "MCAR":
            QQ_hat_new = estimate_MCAR_drift(Y, R_L, P, Q, b_hat, gamma) if QQ is None else QQ

        # Step 3-4: Recover BDLP and re-estimate Sigma
        DeltaL = recover_BDLP(Y, p, P, Q, QQ_hat_new)
        Sigma_hat = estimate_Sigma_L(DeltaL, Q, gamma) if Sigma is None else Sigma

        # Step 5: Re-estimate mu and f
        if mu_estimator == 'Levy_slope':
            mu_hat = DeltaL.sum(axis=1) / P[-1] if mu is None else mu
        elif mu_estimator == 'MCAR_mean':
            mu_hat = QQ_hat_new[p-1].dot(Y.mean(axis=1)) if mu is None else mu
        f_hat = estimate_integral_Levy_measure(DeltaL, Q, gamma, Sigma_hat, K=lambda x: True, f=lambda x: x) if f is None else f

        # Step 6: Update b_hat
        b_hat = mu_hat - f_hat if b is None else b

        # Check convergence: sum of Frobenius norms of drift matrix changes
        converged = np.sum([np.linalg.norm(QQ_hat_new[i] - QQ_hat[i]) for i in range(p)]) <= epsilon
        QQ_hat = QQ_hat_new.copy()

        # Force termination if max iterations exceeded
        if n_iter >= max_iter:
            converged = True
            warnings.warn('maximum number of iterations exceeded.')

        # ---- Loop detection: average over loop iterates if cycle is found ----
        if any(states_close(b_hat, Sigma_hat, QQ_hat, b_, Sigma_, QQ_, tol=1e-6) for b_, Sigma_, QQ_ in zip(bs, Sigmas, QQs)):
            converged = True
            # Find where the loop starts and average over the cycle
            index = np.argmax(np.array([states_close(b_hat, Sigma_hat, QQ_hat, b_, Sigma_, QQ_, tol=1e-6) for b_, Sigma_, QQ_ in zip(bs, Sigmas, QQs)]))
            QQ_hat = [np.mean(np.array(QQs)[index:, i, :, :], axis=0) for i in range(p)]
            b_hat = np.mean(np.array(bs)[index:, :], axis=0)
            Sigma_hat = np.mean(np.array(Sigmas)[index:, :, :], axis=0)
        else:
            QQs.append(QQ_hat.copy())
            bs.append(b_hat)
            Sigmas.append(Sigma_hat)
    return QQ_hat, (b_hat, Sigma_hat)

# ============================================================
# BDLP RECOVERY
# ============================================================

def recover_BDLP(Y: np.ndarray, p: int, P: np.ndarray, Q: np.ndarray, QQ: np.ndarray):
    """
    Recover the increments of the background driving Lévy process (BDLP) L over coarse partition Q.

    Given MCAR observations Y on fine partition P and drift matrices QQ = [Q_1,...,Q_p],
    this function inverts the MCAR dynamics to recover the Lévy increments ΔL on Q:
        ΔX = -drift_contribution + ΔL  (approximately)

    Specifically:
      1. Compute the state-space X on the fine grid P.
      2. Approximate the drift contribution ∫ Q_k D^k Y dt on P via a Riemann sum.
      3. Coarsen the cumulative drift to Q by summing over the fine-grid intervals.
      4. Return ΔL = ΔX[-d:] + drift_Q (the last d components of the state increment
         plus the drift correction).

    :param Y:   MCAR process observation, (d, N+1) np.ndarray
    :param p:   MCAR order (number of lags), int
    :param P:   Finer partition [0=s_0,...,s_N=t], (N+1, ) np.ndarray
    :param Q:   Coarser partition [0=u_0,...,u_M=T], (M+1, ) np.ndarray
    :param QQ:  MCAR drift matrices, (p, d, d) np.ndarray
    :return DeltaL_Q: Increments of the BDLP on partition Q, (d, M) np.ndarray
    """
    d = Y.shape[0]
    # Build full state-space on fine grid P
    X_P = state_space(Y, p, P)
    Delta_P = np.diff(P)    # Fine-grid time increments

    # Find indices of Q-points within P
    indices = [False] * len(P)
    j = 0
    for i, s in enumerate(P):
        if s == Q[j]:
            indices[i] = True
            j += 1

    # Restrict state-space to coarse grid Q and compute increments
    X_Q = X_P[:, indices]
    DeltaX_Q = np.diff(X_Q, axis=1)

    # ---- Approximate the drift on the fine grid P ----
    # drift_P[:,n] ≈ Σ_k Q_{p-k-1} D^k Y_{t_n} * Δt_n
    drift_P = np.zeros((d, len(Delta_P)))
    for j in range(p):
        DjY = X_P[j * d:(j + 1) * d, :-1]
        drift_P += QQ[p - j - 1].dot(DjY) * Delta_P.T

    # ---- Coarsen cumulative drift to Q ----
    # Sum fine-grid drift contributions within each coarse interval [u_m, u_{m+1}]
    cum_drift = np.cumsum(drift_P, axis=1)
    drift_Q = np.diff(np.concatenate([np.zeros((d, 1)), cum_drift[:, indices[:-1]]], axis=1), axis=1)

    # ---- Recover BDLP increments ----
    # ΔL_m ≈ ΔX[-d:,m] + drift_Q[:,m]  (last d components = highest-order derivative block)
    DeltaL_Q = DeltaX_Q[-d:, :] + drift_Q

    return DeltaL_Q

# ============================================================
# BROWNIAN COVARIANCE ESTIMATION
# ============================================================

def estimate_Sigma_L(DeltaL: np.ndarray, Q: np.ndarray, gamma: float, epsilon: float = 0.01):
    """
    Estimate the Brownian covariance Sigma from (irregularly spaced) Lévy increments.

    Implements the iterative thresholding procedure of Gegler (2011), Section 4:
      1. Initialize Σ_hat using all increments.
      2. Use current Σ_hat to identify Brownian-dominated increments via disentangle_BM.
      3. Re-estimate Σ_hat from the identified Brownian increments.
      4. Repeat until convergence (||Σ_new - Σ_old|| ≤ epsilon) or a loop is detected.
         If a loop is detected, return the average of the loop iterates.

    :param DeltaL:  Increments of Lévy process on partition Q, (d, M) np.ndarray
    :param Q:       Partition [0=u_1,...,u_M=T], (M+1, ) np.ndarray
    :param gamma:   Hyperparameter for defining the critical region (must be > 1), float
    :param epsilon: Convergence tolerance, float
    :return Sigma_hat: Estimated Brownian covariance, (d, d) np.ndarray
    """
    # Initialize with the sample covariance of all increments
    DeltaQ_W = np.diff(Q)
    DeltaW = DeltaL
    Sigma_hat_new = estimate_Sigma(DeltaW, DeltaQ_W)
    converged = False

    # Track all visited Sigma estimates to detect loops
    Sigmas = [Sigma_hat_new]
    while not converged:
        Sigma_hat_old = Sigma_hat_new
        # Identify Brownian-dominated increments using current Sigma estimate
        DeltaW, DeltaQ_W = disentangle_BM(DeltaL, Q, Sigma_hat_old, gamma)
        # Re-estimate Sigma from the identified Brownian increments
        Sigma_hat_new = estimate_Sigma(DeltaW, DeltaQ_W)
        converged = np.linalg.norm(Sigma_hat_new - Sigma_hat_old) <= epsilon

        # Detect if the iteration has entered a cycle
        if any(np.array_equal(Sigma_hat_new, Sigma) for Sigma in Sigmas):
            converged = True
            # Average over the loop iterates to break the cycle
            index = np.argmax(np.array([np.array_equal(Sigma_hat_new, Sigma) for Sigma in Sigmas]))
            Sigma_hat_new = np.mean(np.array(Sigmas)[index:, :, :], axis=0)
        else:
            Sigmas.append(Sigma_hat_new)
    return Sigma_hat_new

def estimate_Sigma(DeltaW: np.ndarray, DeltaQ: np.ndarray):
    """
    Estimate the covariance matrix Sigma from (irregularly spaced) Brownian increments.

    Given Brownian increments ΔW_m ~ N(0, Σ * Δt_m), normalize by sqrt(Δt_m) to obtain
    i.i.d. standard increments, then compute the sample covariance:
        Σ_hat = (1/M') Σ_m (ΔW_m/√Δt_m)(ΔW_m/√Δt_m)^T

    :param DeltaW:  Brownian increments, (d, M') np.ndarray
    :param DeltaQ:  Time increments corresponding to DeltaW, (M', ) np.ndarray
    :return Sigma_hat: Estimated covariance matrix, (d, d) np.ndarray
    """
    d = DeltaW.shape[0]
    # Return zero matrix if no Brownian increments were identified
    if len(DeltaQ) == 0:
        return np.zeros((d, d))
    # Normalize increments by sqrt(Δt) to get approximately i.i.d. standard normals
    DeltaZ = DeltaW / np.sqrt(DeltaQ)
    # Sample covariance of normalized increments
    Sigma_hat = np.einsum('ki,ji->kj', DeltaZ, DeltaZ) / DeltaZ.shape[1]
    return Sigma_hat

# ============================================================
# BROWNIAN COMPONENT DISENTANGLEMENT
# ============================================================

def disentangle_BM(DeltaL: np.ndarray, Q: np.ndarray, Sigma: np.ndarray, gamma: float):
    """
    Disentangle Brownian-only increments from Lévy increments on partition Q.

    Implements the critical-region thresholding approach of Gegler (2011), generalized
    to irregularly spaced data. An increment ΔL_m is classified as Brownian-dominated
    (i.e., not a jump) if it falls within the ellipsoidal critical region:
        ΔL_m^T Σ^{-1} ΔL_m ≤ 2 * gamma * Δt_m * log(M)
    where M is the number of increments and Δt_m is the m-th time step.

    :param DeltaL: Increments of Lévy process on partition Q, (d, M) np.ndarray
    :param Q:      Partition [0=u_0,...,u_M=T], (M+1, ) np.ndarray
    :param Sigma:  Covariance of the Brownian part of the Lévy process, (d, d) np.ndarray
    :param gamma:  Hyperparameter for defining the critical region (must be > 1), float
    :return:
        DeltaW:   Subset of DeltaL corresponding to Brownian-dominated increments, (d, M') np.ndarray
        DeltaQ_W: Corresponding time increments, (M', ) np.ndarray
    """
    DeltaQ = np.diff(Q)
    N = len(DeltaQ)
    DeltaX = DeltaL
    try:
        # Compute Mahalanobis distance for each increment and compare to threshold
        subset = np.einsum('ki,ki->i', DeltaX, np.linalg.pinv(Sigma).dot(DeltaX)) <= 2 * gamma * DeltaQ * np.log(N)
    except np.linalg.LinAlgError:
        # Regularize Sigma if singular
        eps = 1e-6
        Sigma_reg = Sigma + eps * np.eye(Sigma.shape[0])
        subset = np.einsum('ki,ki->i', DeltaX, np.linalg.pinv(Sigma_reg).dot(DeltaX)) <= 2 * gamma * DeltaQ * np.log(N)

    # Return only the Brownian-dominated increments and their time steps
    DeltaQ_W = DeltaQ[subset]
    DeltaW = DeltaX[:, subset]
    return DeltaW, DeltaQ_W

# ============================================================
# LÉVY MEASURE FUNCTIONAL ESTIMATION
# ============================================================

def estimate_integral_Levy_measure(DeltaL: np.ndarray, Q: np.ndarray, gamma: float, Sigma: np.ndarray = None, epsilon: float = 0.01, K: Callable = lambda x: np.linalg.norm(x) > 1e-3, f: Callable = lambda x: x):
    """
    Estimate the functional f(F) = ∫_K f(z) F(dz) of the Lévy measure F.

    Uses the jump-identification approach of Gegler (2011): increments outside the
    Brownian critical region are classified as jump-dominated, and the functional
    is estimated by summing f(ΔL_m) over these jump increments, normalized by T.

    Theoretical guarantees are given for functions of the form f(z) = (z^T A z)^l.
    If Sigma is known, uses the rejection-region estimator (Gegler 2011, Section 2);
    otherwise, uses the iterative approach (Gegler 2011, Section 4).

    :param DeltaL:  Increments of Lévy process on partition Q, (d, M) np.ndarray
    :param Q:       Partition [0=u_1,...,u_M=T], (M+1, ) np.ndarray
    :param gamma:   Hyperparameter for defining the critical region (must be > 1), float
    :param Sigma:   Covariance of Brownian component; if None, estimated iteratively, (d, d) np.ndarray
    :param epsilon: Convergence tolerance for iterative Sigma estimation, float
    :param K:       Integration region indicator: K(z) = True if z is in the integration domain
    :param f:       Integrand function: maps (d, ) np.ndarray to float or np.ndarray
    :return f_hat:  Estimated functional ∫_K f(z) F(dz), float or np.ndarray
    """
    # Estimate Sigma iteratively if not provided
    if Sigma is None:
        Sigma = estimate_Sigma_L(DeltaL, Q, gamma, epsilon)

    d = Sigma.shape[0]
    DeltaQ = np.diff(Q)
    N = len(DeltaQ)

    # Identify jump-dominated increments: those OUTSIDE the Brownian critical region
    # and within the integration domain K
    subset = np.logical_and(
        np.einsum('ki,ki->i', np.linalg.pinv(Sigma).dot(DeltaL), DeltaL) > 2 * gamma * DeltaQ * np.log(N),
        [K(DeltaL[:, i]) for i in range(DeltaL.shape[1])]
    )
    DeltaZ = DeltaL[:, subset]
    if DeltaZ.shape[1] < 1:
        # No jump increments identified: return zero functional
        f_hat = np.zeros(f(np.zeros(d)).shape) if isinstance(f(np.zeros(d)), np.ndarray) else 0
    else:
        # Estimate ∫_K f(z) F(dz) ≈ (1/T) Σ_m f(ΔL_m) 1{m is jump-dominated, ΔL_m ∈ K}
        f_hat = np.sum(np.array([f(DeltaZ[:, i]) for i in range(DeltaZ.shape[1])]), axis=0) / (Q[-1] - Q[0])
    return f_hat

# ============================================================
# THRESHOLD SEQUENCE SELECTION
# ============================================================

def choose_nu(DeltaL: np.ndarray, Q: np.ndarray, gamma: float, epsilon: float = 0.01, Sigma: np.ndarray = None):
    """
    Choose component-wise thresholds nu to separate the continuous (Brownian) component
    from the jump component of a one-dimensional Lévy process.

    The threshold is set to:
        nu_m = sqrt(2 * gamma * Sigma_hat * Δt_m * log(M))
    which is the square root of the Brownian critical region boundary for a 1D process.
    This ensures that increments |ΔL_m| ≤ nu_m are classified as Brownian-dominated.

    :param DeltaL:  Increments of Lévy process on partition Q, (1, M) np.ndarray
    :param Q:       Partition [0=u_1,...,u_M=T], (M+1, ) np.ndarray
    :param gamma:   Hyperparameter for defining the critical region (must be > 1), float
    :param epsilon: Convergence tolerance for iterative Sigma estimation, float
    :param Sigma:   Known covariance of Brownian component; if None, estimated, (1, 1) np.ndarray
    :return nu:     Thresholding vector, (1, M) np.ndarray
    """
    DeltaQ = np.diff(Q)
    # Estimate or use provided Brownian variance
    Sigma_hat = estimate_Sigma_L(DeltaL, Q, gamma, epsilon) if Sigma is None else Sigma
    # Threshold: sqrt(2 * gamma * Sigma * Δt * log(M))
    nu = np.sqrt(DeltaQ * 2 * gamma * np.log(len(DeltaQ)) * Sigma_hat)
    return nu

# ============================================================
# UTILITY: RESHAPE VECTORIZED PARAMETERS
# ============================================================

def inv_vectorize_AA(vec_AA: np.ndarray, p: int):
    """
    Reshape a vectorized parameter array back into a list of p square matrices.

    Given a vector of length p*d², this function reconstructs the p drift matrices
    Q_1, ..., Q_p ∈ R^{d×d} by splitting and reshaping (with transpose to recover
    column-major ordering used during vectorization).

    :param vec_AA: Vectorized parameter array, (p*d², ) np.ndarray
    :param p:      Number of lags, int
    :return:       Array of p drift matrices, (p, d, d) np.ndarray
    """
    d = int(np.sqrt(len(vec_AA) // p))
    AA = []
    for i in range(p):
        # Reshape and transpose to recover the original matrix orientation
        AA.append(vec_AA[i*d**2:(i+1)*d**2].reshape((d, d)).T)
    return np.array(AA)

# ============================================================
# grOU PARAMETER CONVERSION UTILITY
# ============================================================

def estimator_to_QQ(V: np.ndarray, R_L: List, C: int, W_C: List, Adj: np.ndarray, K: int) -> np.ndarray:
    """
    Convert the estimated grOU parameter vector theta to drift matrices QQ.

    Reconstructs the p drift matrices Q_1, ..., Q_p ∈ R^{K×K} from the vectorized
    parameter theta = [alpha_1, beta_1, ..., alpha_L, beta_L] using the grOU structure:
        Q_l = diag(alpha_l, ..., alpha_l) + Σ_{c,r} beta_{l,c,r} (W_C[c] ⊙ Adj[r])
    as defined in Section 2.3 of Chen & Veraart (2026).

    :param V:    Estimated parameter vector, (L + sum(R_L)*C, ) np.ndarray
    :param R_L:  Maximum stage for each Lag, list of int
    :param C:    Number of Covariates, int
    :param W_C:  Edge weights for Covariates, list of (K, K) np.ndarray
    :param Adj:  Graph adjacency matrix (maximum stage), (d, d) np.ndarray
    :param K:    Number of edges (dimension of the process), int
    :return:     Estimated drift matrices, (L, K, K) np.ndarray
    """
    alphas = []     # Scalar autoregressive coefficients per lag
    betas = []      # Network interaction coefficient vectors per lag
    idx = 0
    L = len(R_L)

    # Extract alpha_l and beta_l for each lag l
    for l in range(L):
        alphas.append(V[idx])   # Scalar alpha at lag l
        idx += 1
        n_beta = C * R_L[l]     # Number of beta parameters at lag l
        betas.append(V[idx:idx + n_beta])
        idx += n_beta

    # Build weighted neighborhood matrices W[l] = [W_C[c] ⊙ Adj[r] for each (c,r) at lag l]
    W = [[np.multiply(W_C[c], Adj[stage]) for c in range(C) for stage in range(R_i)] for R_i in R_L]
    Q = []
    l = 0
    while l < L:
        Q.append(np.diag([alphas[l]] * K) + np.hstack(W[l]) @ np.kron(betas[l], np.eye(K)).T)
        l += 1
    return np.array(Q)

# ============================================================
# CONVERGENCE CHECK UTILITY
# ============================================================

def states_close(b1, S1, Q1, b2, S2, Q2, tol=1e-6):
    """
    Check whether two parameter states (b, Sigma, QQ) are close within tolerance.

    Used for loop detection in the iterative estimation procedure: if the current
    parameter state matches a previously visited state (up to tolerance), the
    iteration has entered a cycle.

    :param b1, b2:  Levy drift vectors to compare, (d, ) np.ndarray
    :param S1, S2:  Brownian covariance matrices to compare, (d, d) np.ndarray
    :param Q1, Q2:  Lists of p drift matrices to compare, list of (d, d) np.ndarray
    :param tol:     Tolerance for closeness check, float
    :return:        True if all components are within tolerance, bool
    """
    b_close = np.linalg.norm(b1 - b2) < tol
    S_close = np.linalg.norm(S1 - S2, ord='fro') < tol
    Q_close = sum(np.linalg.norm(Q1[i] - Q2[i], ord='fro') for i in range(len(Q1))) < tol
    return b_close and S_close and Q_close