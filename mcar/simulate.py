"""
================================================================================
MCAR Process Simulation Toolkit
================================================================================

This module provides utilities for simulating Lévy-driven Multivariate
Continuous-time AutoRegressive (MCAR) processes, as studied in:

    Chen & Veraart (2026): "Edge-Indexed Network Time Series with Graph
    Ornstein-Uhlenbeck Dynamics".

--------------------------------------------------------------------------------
Model Overview:
--------------------------------------------------------------------------------
An MCAR(p) process {Y_t, t ≥ 0} satisfies the formal differential equation:

    D^p Y_t + Q_1 D^{p-1} Y_t + ... + Q_{p-1} D Y_t + Q_p Y_t = D L_t

where D denotes differentiation with respect to t, Q_1,...,Q_p ∈ R^{d×d} are
the drift (autoregressive) matrices, and {L_t} is a d-dimensional Lévy process
with triplet (a, Σ, F):
    - a ∈ R^d:          drift (no-truncation convention, so E[L_1] = a + ∫ z F(dz))
    - Σ ∈ S^+_d:        Brownian covariance
    - F:                Lévy measure (jump distribution)

This is equivalently represented via the pd-dimensional state-space SDE:
    dX_t = Q X_t dt + E dL_t,   Y_t = A X_t
where:
    - X_t = (Y_t, DY_t, ..., D^{p-1}Y_t)^T ∈ R^{pd}  is the state vector
    - Q ∈ R^{pd×pd}  is the MCAR structural matrix (companion form)
    - E ∈ R^{pd×d}   is the noise injection matrix (zeros except last d rows = I_d)
    - A = [I_d, 0, ..., 0] ∈ R^{d×pd}  extracts the first d components

The exact solution over an interval [s, t] is:
    X_t = e^{Q(t-s)} X_s + ∫_s^t e^{Q(t-r)} E dL_r

where the three integral components are:
    - Drift:   ∫_s^t e^{Q(t-r)} E a dr = (e^{Q(t-s)} - I) Q^{-1} E a
    - Brownian: ∫_s^t e^{Q(t-r)} E dW_r ~ N(0, V(t-s))
      with V(h) computed via the block matrix exponential:
          exp([Q, EΣE^T; 0, -Q^T] * h) = [e^{Qh}, V(h); 0, e^{-Q^T h}]
    - Jump:    ∫_s^t e^{Q(t-r)} E dJ_r = Σ_{s<τ≤t} e^{Q(t-τ)} E ΔJ_τ

--------------------------------------------------------------------------------
Simulation Methods:
--------------------------------------------------------------------------------
Two simulation strategies are provided:

1. **Euler-Maruyama approximation** (`simulate_MCAR_approx`):
   Uses the first-order Euler scheme:
       X_{n+1} ≈ X_n + Q X_n Δt_n + E ΔL_n
   Suitable for infinite Lévy measure (e.g., Gamma processes) where exact
   simulation of the jump component is intractable.

2. **Exact simulation** (`simulate_MCAR_compound_poisson`):
   Uses the exact solution via matrix exponentials and the block matrix
   covariance formula. Suitable for finite-activity (compound Poisson) jumps.

Both methods support:
    - Uniform and non-uniform time grids
    - Multiple output formats: observed process Y, full state X, Lévy process L

Stationary distribution sampling is also provided via:
    - `simulate_MCAR_stat_distr_approx`:           Riemann-sum approximation
    - `simulate_MCAR_stat_distr_compound_poisson`: Exact (compound Poisson)

--------------------------------------------------------------------------------
Key Functions:
--------------------------------------------------------------------------------
  reshape_array                           - Safe array reshaping utility
  MCAR_Q                                  - Build MCAR companion matrix Q
  Q_QQ_inverse                            - Compute Q^{-1} analytically
  simulate_MCAR_stat_distr_approx         - Sample from stationary distribution (approx)
  simulate_MCAR_stat_distr_compound_poisson - Sample from stationary distribution (exact)
  simulate_MCAR_approx                    - Euler-Maruyama path simulation
  simulate_MCAR_compound_poisson          - Exact path simulation (compound Poisson)
  gamma_increments                        - Generate symmetric Gamma Lévy increments

--------------------------------------------------------------------------------
References:
--------------------------------------------------------------------------------
  - Lucchese, Pakkanen & Veraart (2026): Estimation and Inference for
    Multivariate Continuous-time Autoregressive Processes
  - Chen & Veraart (2026): Edge-Indexed Network Time Series with Graph
    Ornstein-Uhlenbeck Dynamics
  - Marquardt & Stelzer (2007): Multivariate CARMA Processes
  - Van Loan (1978): Computing integrals involving the matrix exponential
================================================================================
"""

import numpy as np
import scipy
import scipy.linalg
import scipy.stats
from typing import Optional, Callable

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def reshape_array(array, d):
    """
    Safely reshape an array to ensure consistent 2D matrix orientation.

    Handles three cases:
      - d == 1:             Reshape to a column vector (n, 1)
      - 1D array, d > 1:   Reshape to a row vector (1, n), treating as a single sample
      - 2D array:           Return unchanged

    This utility is used to standardize jump size arrays that may be returned
    as scalars, 1D vectors, or 2D matrices depending on the number of samples.

    :param array: Input array to reshape, np.ndarray
    :param d:     Dimension of the process (number of components), int
    :return:      Reshaped 2D array, np.ndarray
    """
    if d == 1:
        # Univariate case: ensure column vector shape (n, 1)
        return array.reshape(-1, 1)
    elif len(array.shape) == 1:
        # 1D array with d > 1: treat as a single d-dimensional sample, shape (1, d)
        return array.reshape(1, -1)
    else:
        # Already 2D: return as-is
        return array

# ============================================================
# MCAR STRUCTURAL MATRIX CONSTRUCTION
# ============================================================

def MCAR_Q(QQ: np.ndarray) -> np.ndarray:
    """
    Construct the MCAR companion (structural) matrix Q from drift matrices Q_1,...,Q_p.

    The MCAR(p) process satisfies the state-space SDE dX_t = Q X_t dt + E dL_t,
    where the companion matrix Q ∈ R^{pd×pd} takes the block companion form:

        Q = [ 0    I    0   ...  0  ]
            [ 0    0    I   ...  0  ]
            [ .    .    .   ...  .  ]
            [ 0    0    0   ...  I  ]
            [-Q_p -Q_{p-1} ...  -Q_1]

    The last block row encodes the autoregressive coefficients in reverse order,
    consistent with the formal differential equation:
        D^p Y + Q_1 D^{p-1} Y + ... + Q_p Y = D L

    :param QQ: List/array of p drift matrices [Q_1, ..., Q_p], each (d, d) np.ndarray
    :return Q_QQ: MCAR companion matrix, (pd, pd) np.ndarray
    """
    p = len(QQ)
    d = QQ[0].shape[0]
    Q_QQ = np.zeros((p * d, p * d))

    # Fill the super-diagonal blocks with identity matrices (shift operator)
    # Q_QQ[i*d:(i+1)*d, (i+1)*d:(i+2)*d] = I_d  for i = 0, ..., p-2
    for i in range(d, p * d, d):
        Q_QQ[(i - d):i, i:(i + d)] = np.eye(d)

    # Fill the last block row with -Q_p, -Q_{p-1}, ..., -Q_1 (reversed order)
    for i in range(p):
        Q_QQ[-d:, i * d:(i + 1) * d] = - QQ[p - i - 1]

    return Q_QQ

def Q_QQ_inverse(Q_QQ: np.ndarray, K: int) -> np.ndarray:
    """
    Compute the inverse of the MCAR companion matrix Q analytically.

    Uses the closed-form expression from Proposition 2.1 of Chen & Veraart (2026).
    Given the companion matrix Q with last block row [-Q_p, -Q_{p-1}, ..., -Q_1],
    the inverse has the structure:

        Q^{-1} = [-Q_p^{-1}Q_{p-1}, ..., -Q_p^{-1}Q_1, -Q_p^{-1}]
                 [I,   0,   ...   0,   0                           ]
                 [0,   I,   ...   0,   0                           ]
                 [.    .    ...   .    .                            ]
                 [0,   0,   ...   I,   0                           ]

    The first block row is computed by solving Q_p X = -Q_j for each j,
    and the remaining rows form a lower-shifted identity structure.

    :param Q_QQ: MCAR companion matrix, (pd, pd) np.ndarray
    :param K:    Dimension of each block (number of edges/components), int
    :return Q_inv: Inverse of the companion matrix, (pd, pd) np.ndarray
    """
    p = Q_QQ.shape[0] // K

    # Extract the p block matrices from the last block row of Q_QQ
    # These are [-Q_p, -Q_{p-1}, ..., -Q_1] (note: stored with sign)
    last_row_blocks = [Q_QQ[(p - 1) * K:p * K, j * K:(j + 1) * K] for j in range(p)]

    Q_inv = np.zeros_like(Q_QQ)

    # ---- First block row of Q^{-1} ----
    # Entries: -Q_p^{-1} Q_j for j = 1,...,p-1, and -Q_p^{-1} for the last block
    # Computed by solving: last_row_blocks[0] @ X = last_row_blocks[j+1]
    # (last_row_blocks[0] = -Q_p, so solving gives Q_p^{-1} Q_j with correct sign)
    for j in range(p - 1):
        Q_inv[0:K, j * K:(j + 1) * K] = -np.linalg.solve(last_row_blocks[0], last_row_blocks[j + 1])
    # Last block in first row: -Q_p^{-1} = solve(-Q_p, I)
    Q_inv[0:K, (p - 1) * K:p * K] = np.linalg.solve(last_row_blocks[0], np.eye(K))

    # ---- Lower block rows: identity blocks on the first subdiagonal ----
    # Q^{-1}[i*K:(i+1)*K, (i-1)*K:i*K] = I_K  for i = 1, ..., p-1
    for i in range(1, p):
        Q_inv[i * K:(i + 1) * K, (i - 1) * K:i * K] = np.eye(K)

    return Q_inv

# ============================================================
# STATIONARY DISTRIBUTION SAMPLING (APPROXIMATE)
# ============================================================

def simulate_MCAR_stat_distr_approx(Q_QQ: np.ndarray, a: np.ndarray, Sigma: np.ndarray, jumps: Optional[Callable[[float, float], np.ndarray]]):
    """
    Sample from the stationary distribution of an MCAR process via Riemann-sum approximation.

    The stationary state X_0 has the representation (Remark 2.2, Chen & Veraart 2026):
        X_0 = ∫_{-∞}^{0} e^{-sQ} E dL_s

    This integral is approximated by a finite Riemann sum over [0, T], where T is chosen
    large enough that e^{QT} ≈ 0 (i.e., the integrand has decayed to negligible size):
        T = -log(1e-8) / ||Q||

    The integral is discretized as:
        X_0 ≈ Σ_{n=0}^{N-1} e^{Q t_n} E ΔL_n

    where ΔL_n ~ N(a Δt, Σ Δt) + jump_n are i.i.d. Lévy increments over [t_n, t_{n+1}].

    :param Q_QQ:  MCAR companion matrix, (pd, pd) np.ndarray
    :param a:     Lévy drift (no-truncation), E[L_1] = a + ∫ z F(dz), (d, ) np.ndarray
    :param Sigma: Brownian covariance of Lévy process, (d, d) np.ndarray
    :param jumps: Function jumps(delta_t, n) → (d, n) np.ndarray generating n jump
                  increments over an interval of size delta_t; None for pure diffusion
    :return x:    Sample from the stationary distribution, (pd, ) np.ndarray
    """
    pd = Q_QQ.shape[0]
    d = a.shape[0]

    # Noise injection matrix: E = [0; ...; 0; I_d] ∈ R^{pd×d}
    E = np.zeros([pd, d])
    E[-d:,:] = np.eye(d)

    # Choose truncation horizon T so that ||e^{Q*T}|| ≈ 1e-8 (integrand negligible)
    T = - np.log(1e-8) / np.linalg.norm(Q_QQ)
    N = 1000    # Number of Riemann sum steps
    P = np.linspace(0, T, N + 1)
    delta_t = T/N

    # Generate Lévy increments ΔL_n = a*Δt + ΔW_n + ΔJ_n
    if jumps is None:
        delta_jump_L = np.zeros([d, N])
    else:
        delta_jump_L = jumps(delta_t, N)
    delta_L = scipy.stats.multivariate_normal(mean=a * delta_t, cov=Sigma * delta_t).rvs(size=N).T + delta_jump_L

    # Compute Riemann sum: X_0 ≈ Σ_n e^{Q*t_n} E ΔL_n
    # np.tensordot computes e^{Q*t_n} for each t_n ∈ P[:-1], then contracts with E*ΔL
    return np.tensordot(scipy.linalg.expm(np.tensordot(P[:-1], Q_QQ, axes=0)), np.matmul(E, delta_L), axes=[[0, 2], [1, 0]])

# ============================================================
# STATIONARY DISTRIBUTION SAMPLING (EXACT, COMPOUND POISSON)
# ============================================================

def simulate_MCAR_stat_distr_compound_poisson(Q_QQ: np.ndarray, a: np.ndarray, Sigma: np.ndarray, rate: float, jump_F: Optional[scipy.stats._multivariate.multi_rv_frozen]):
    """
    Sample from the stationary distribution of a compound-Poisson-driven MCAR process.

    Uses the exact decomposition of the stationary state X_0 = ∫_{-∞}^0 e^{-sQ} E dL_s
    into three independent components (drift, Brownian, jump), each computed exactly
    over a truncation horizon T = -log(1e-8) / ||Q||:

    1. **Drift component**:
           a_component = -Q^{-1} E a
       (Closed-form from the stationary mean formula, Proposition 2.1)

    2. **Brownian component**:
           W_component ~ N(0, V)
       where V = ∫_0^T e^{Qs} E Σ E^T e^{Q^T s} ds is computed via the Van Loan (1978)
       block matrix exponential trick:
           exp([Q, EΣE^T; 0, -Q^T] * T) = [e^{QT}, V e^{-Q^T T}; 0, e^{-Q^T T}]
       so V = exp(M*T)[:pd, pd:] @ e^{QT}^T

    3. **Jump component** (compound Poisson):
           J_component = Σ_{τ_i ∈ [0,T]} e^{Q τ_i} E ΔJ_{τ_i}
       where jump times τ_i ~ Uniform[0, T] and jump sizes ΔJ_i ~ jump_F.

    :param Q_QQ:   MCAR companion matrix, (pd, pd) np.ndarray
    :param a:      Lévy drift (no-truncation), (d, ) np.ndarray
    :param Sigma:  Brownian covariance of Lévy process, (d, d) np.ndarray
    :param rate:   Poisson jump arrival rate (jumps per unit time), float
    :param jump_F: Distribution of individual jump sizes; None if rate == 0,
                   scipy.stats._multivariate.multi_rv_frozen
    :return x:     Sample from the stationary distribution, (pd, ) np.ndarray
    """
    if rate > 0 and jump_F is None:
        raise ValueError("jump_F must be provided if rate > 0.")

    pd = Q_QQ.shape[0]
    d = a.shape[0]

    # Noise injection matrix E and lifted covariance Σ̃ = E Σ E^T ∈ R^{pd×pd}
    E = np.zeros([pd, d])
    E[-d:,:] = np.eye(d)
    Sigma_tilde = E.dot(Sigma).dot(E.T)

    # Truncation horizon: e^{Q*T} ≈ 0 beyond this point
    T = - np.log(1e-8) / np.linalg.norm(Q_QQ)

    # ---- Brownian covariance V via Van Loan block matrix exponential ----
    # Build 2pd × 2pd block matrix M = [Q, Σ̃; 0, -Q^T]
    # Then V = exp(M*T)[:pd, pd:] @ exp(Q*T)^T
    M = np.zeros((2 * pd, 2 * pd))
    M[:pd, :pd] = Q_QQ
    M[:pd, pd:] = Sigma_tilde
    M[pd:, pd:] = -Q_QQ.T
    V = scipy.linalg.expm(M * T)[:pd, pd:].dot(scipy.linalg.expm(Q_QQ * T).T)

    # ---- Drift component: -Q^{-1} E a (stationary mean, Proposition 2.1) ----
    Q_inv = Q_QQ_inverse(Q_QQ, d)
    a_component = -Q_inv.dot(E).dot(a)

    # ---- Brownian component: sample from N(0, V) ----
    W_component = scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=1).T

    # ---- Jump component: sum e^{Q*τ_i} E ΔJ_i over Poisson arrivals in [0, T] ----
    if rate > 0:
        # Sample number of jumps in [0, T] from Poisson(rate * T)
        N_T = scipy.stats.poisson(mu=rate * T).rvs(size=1)[0]
        # Sample jump times uniformly over [0, T]
        jump_times = np.sort(scipy.stats.uniform().rvs(size=N_T) * T)
        # Sample jump sizes from the specified distribution
        jump_sizes = reshape_array(jump_F.rvs(size=N_T), d).T
        # Compute J = Σ_i e^{Q*τ_i} E ΔJ_i using einsum for efficiency
        J_component = np.einsum('ijk,ki->j',scipy.linalg.expm(np.tensordot(jump_times, Q_QQ, axes=0)), E.dot(jump_sizes))
    else:
        J_component = np.zeros(pd)

    # Combine all three components
    x = a_component + W_component + J_component
    return x

# ============================================================
# PATH SIMULATION: EULER-MARUYAMA (INFINITE LÉVY MEASURE)
# ============================================================

def simulate_MCAR_approx(P: np.ndarray, Q_QQ: np.ndarray, x0: np.ndarray, a: np.ndarray, Sigma: np.ndarray, jumps: Optional[Callable[[float, float], np.ndarray]], output_format: str = 'MCAR', uniform = False):
    """
    Simulate MCAR process paths using the Euler-Maruyama discretization scheme.

    Implements the first-order approximation:
        X_{n+1} ≈ X_n + Q X_n Δt_n + E ΔL_n

    where ΔL_n = a Δt_n + ΔW_n + ΔJ_n with:
        - ΔW_n ~ N(0, Σ Δt_n):  Brownian increment
        - ΔJ_n = jumps(Δt_n, 1): Jump increment (if provided)

    This approximation is suitable for Lévy processes with infinite Lévy measure
    (e.g., Gamma, Variance Gamma) where exact simulation of jumps is intractable.
    For finite-activity (compound Poisson) jumps, use `simulate_MCAR_compound_poisson`
    which implements the exact solution via matrix exponentials.

    The uniform flag enables a vectorized implementation that pre-generates all
    Brownian increments at once, which is significantly faster for large N.

    :param P:             Time partition [t_0, t_1, ..., t_N], (N+1, ) np.ndarray
    :param Q_QQ:          MCAR companion matrix, (pd, pd) np.ndarray
    :param x0:            Initial state vector X_0, (pd, ) np.ndarray
    :param a:             Lévy drift (no-truncation), E[L_1] = a + ∫ z F(dz), (d, ) np.ndarray
    :param Sigma:         Brownian covariance of Lévy process, (d, d) np.ndarray
    :param jumps:         Function jumps(delta_t, n) → (d, n) np.ndarray; None for pure diffusion
    :param output_format: Output selection:
                            'MCAR'          → Y = X[:d, :],          (d, N+1) np.ndarray
                            'SS'            → X (full state),         (pd, N+1) np.ndarray
                            'SS + L'        → (X, L),                 tuple
                            'SS + L + jump_L' → (X, L, jump_L),      tuple
    :param uniform:       If True, use vectorized uniform-grid implementation, bool
    :return:              See output_format description above
    """
    # Extract dimensions
    d = len(a)      # Dimension of the observed process Y
    pd = len(x0)    # Dimension of the full state vector X
    N = len(P)      # Number of time points (including t_0)

    # Noise injection matrix: E = [0; ...; 0; I_d] ∈ R^{pd×d}
    E = np.zeros([pd, d])
    E[-d:, :] = np.eye(d)

    # Initialize state, Lévy process, and jump process arrays
    X = np.zeros([pd, N])
    L = np.zeros([d, N])    # Cumulative Lévy process
    jump_L = np.zeros([d, N])   # Cumulative jump component of L
    X[:, 0] = x0

    # Pre-generate all Brownian increments: ΔW_n ~ N(0, Σ * Δt_n)
    # Scale by sqrt(Δt_n) to account for non-uniform spacing
    delta_W = np.sqrt(np.diff(P))[np.newaxis, :] * scipy.stats.multivariate_normal(cov=Sigma).rvs(size=N - 1).T
    delta_L = np.zeros((d, N - 1))

    # ============================================================
    # UNIFORM GRID: VECTORIZED IMPLEMENTATION
    # ============================================================
    if uniform:
        delta_t = P[1] - P[0]   # Constant time step

        # Compute all Lévy increments: ΔL_n = a*Δt + ΔW_n [+ ΔJ_n]
        delta_L = a.reshape(-1, 1) * delta_t + delta_W
        if jumps is not None:
            # Generate all jump increments at once for efficiency
            delta_jump_L = jumps(delta_t, N - 1)
            delta_L += delta_jump_L
            # Cumulative jump process: jump_L[:,n] = Σ_{k<n} ΔJ_k
            jump_L[:, 1:] = delta_jump_L.cumsum(axis=1)

        # Cumulative Lévy process: L[:,n] = Σ_{k<n} ΔL_k
        L[:, 1:] = delta_L.cumsum(axis=1)

        # Euler-Maruyama evolution: X_{n+1} = X_n + Q X_n Δt + E ΔL_n
        for n in range(N - 1):
            X[:, n + 1] = X[:, n] + np.matmul(Q_QQ, X[:, n]) * delta_t + np.matmul(E, delta_L[:, n])

    # ============================================================
    # NON-UNIFORM GRID: STEP-BY-STEP IMPLEMENTATION
    # ============================================================
    else:
        for n in range(N - 1):
            delta_t = P[n + 1] - P[n]   # Variable time step

            # Compute Lévy increment for this step: ΔL_n = a*Δt + ΔW_n [+ ΔJ_n]
            delta_L[:, n] = a.reshape(-1, 1) * delta_t + delta_W[:, n]
            if jumps is not None:
                # Generate a single jump increment for this interval
                delta_jump_L = jumps(delta_t, 1).reshape(-1, 1)
                delta_L[:, n] += delta_jump_L
                # Update cumulative jump process
                jump_L[:, n + 1] = jump_L[:, n] + delta_jump_L

            # Update cumulative Lévy process
            L[:, n + 1] = L[:, n] + delta_L[:, n]

            # Euler-Maruyama step: X_{n+1} = X_n + Q X_n Δt + E ΔL_n
            X[:, n + 1] = X[:, n] + np.matmul(Q_QQ, X[:, n]) * delta_t + np.matmul(E, delta_L[:, n])

    # ---- Return requested output format ----
    if output_format == 'MCAR':
        return X[:d, :]     # Observed process Y = first d components of X
    elif output_format == 'SS':
        return X            # Full state vector X
    elif output_format == 'SS + L':
        return X, L         # State vector + cumulative Lévy process
    elif output_format == 'SS + L + jump_L':
        return X, L, jump_L # State vector + Lévy process + jump component
    else:
        raise ValueError("output_format must be one of ['MCAR', 'SS', 'SS + L', 'SS + L + jump_L']")

# ============================================================
# PATH SIMULATION: EXACT (COMPOUND POISSON JUMPS)
# ============================================================

def simulate_MCAR_compound_poisson(P: np.ndarray, Q_QQ: np.ndarray, x0: np.ndarray, a: np.ndarray, Sigma: np.ndarray, rate: float, jump_F: Optional[scipy.stats._multivariate.multi_rv_frozen], output_format: str = 'MCAR', uniform: bool = False):
    """
    Simulate MCAR process paths using the exact solution for compound Poisson jumps.

    Implements the exact transition over each interval [t_n, t_{n+1}]:
        X_{n+1} = e^{Q Δt_n} X_n + drift_n + Brownian_n + Jump_n

    where each component is computed exactly:

    1. **Drift increment**:
           drift_n = (e^{Q Δt_n} - I) Q^{-1} E a
       using the analytical inverse Q^{-1} from `Q_QQ_inverse`.

    2. **Brownian increment**:
           Brownian_n ~ N(0, V(Δt_n))
       where V(h) = exp([Q, EΣE^T; 0, -Q^T] * h)[:pd, pd:] @ e^{QT}^T
       via the Van Loan (1978) block matrix exponential method.
       For uniform grids, V is precomputed once; for non-uniform grids, it is
       recomputed at each step.

    3. **Jump increment** (compound Poisson):
           Jump_n = Σ_{τ_i ∈ (t_n, t_{n+1}]} e^{Q(t_{n+1} - τ_i)} E ΔJ_i
       where the number of jumps N_Δ ~ Poisson(rate * Δt_n), jump times
       τ_i ~ Uniform[t_n, t_{n+1}], and jump sizes ΔJ_i ~ jump_F.

    :param P:             Time partition [t_0, t_1, ..., t_N], (N+1, ) np.ndarray
    :param Q_QQ:          MCAR companion matrix, (pd, pd) np.ndarray
    :param x0:            Initial state vector X_0, (pd, ) np.ndarray
    :param a:             Lévy drift (no-truncation), E[L_1] = a + ∫ z F(dz), (d, ) np.ndarray
    :param Sigma:         Brownian covariance of Lévy process, (d, d) np.ndarray
    :param rate:          Poisson jump arrival rate (jumps per unit time), float
    :param jump_F:        Distribution of individual jump sizes; None if rate == 0,
                          scipy.stats._multivariate.multi_rv_frozen
    :param output_format: Output selection:
                            'MCAR'       → Y = X[:d, :],              (d, N+1) np.ndarray
                            'SS'         → X (full state),             (pd, N+1) np.ndarray
                            'SS + jumps' → (X, jump_times, jump_sizes), tuple
    :param uniform:       If True, precompute matrix exponentials for uniform grid, bool
    :return:              See output_format description above
    """
    if rate > 0 and jump_F is None:
        raise ValueError("jump_F must be provided if rate > 0")

    d = len(a)      # Dimension of the observed process
    pd = len(x0)    # Dimension of the full state vector
    N = len(P)      # Number of time points
    T = P[-1] - P[0]    # Total time horizon

    # ---- Noise injection matrix and lifted covariance ----
    # E = [0; ...; 0; I_d] ∈ R^{pd×d}: injects noise into the last d components
    E = np.zeros((pd, d))
    E[-d:,:] = np.eye(d)
    # Σ̃ = E Σ E^T ∈ R^{pd×pd}: lifted Brownian covariance in state-space
    Sigma_tilde = E.dot(Sigma).dot(E.T)
    
    # Initialize state array
    X = np.zeros((pd, N))
    X[:, 0] = x0

    # ---- Precompute matrix exponential e^Q and block matrix e^M ----
    # These are used to compute e^{Q*h} and V(h) for any h via fractional powers
    M = np.zeros((2 * pd, 2 * pd))
    M[:pd, :pd] = Q_QQ
    M[:pd, pd:] = Sigma_tilde
    M[pd:, pd:] = -Q_QQ.T
    eM = scipy.linalg.expm(M)   # e^M (used as base for fractional powers)
    eQ = scipy.linalg.expm(Q_QQ)    # e^Q (used as base for fractional powers)

    # Analytical inverse of the companion matrix (for drift computation)
    Q_inv = Q_QQ_inverse(Q_QQ, d)

    # ============================================================
    # UNIFORM GRID: PRECOMPUTE ALL MATRIX QUANTITIES ONCE
    # ============================================================
    if uniform:
        delta_t = P[1] - P[0]

        # e^{Q*Δt}: state transition matrix for one step
        eQt = scipy.linalg.fractional_matrix_power(eQ, delta_t).real

        # Drift increment: (e^{Q*Δt} - I) Q^{-1} E a  (same for all steps)
        a_increment = (eQt - np.eye(pd)).dot(Q_inv).dot(E).dot(a)

        # Brownian covariance V(Δt) via Van Loan block matrix trick
        # V(Δt) = e^{M*Δt}[:pd, pd:] @ e^{Q*Δt}^T
        V = scipy.linalg.fractional_matrix_power(eM, delta_t).real[:pd, pd:].dot(eQt.T)

        # Handle potential numerical non-PSD issues in V
        try:
            W_increments = np.atleast_2d(scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=N-1)).T
        except ValueError:
            # Repair V by projecting to nearest PSD matrix
            eigvals, eigvecs = np.linalg.eigh(V)
            eigvals = np.maximum(eigvals, 1e-16)
            V = eigvecs @ np.diag(eigvals) @ eigvecs.T
            W_increments = np.atleast_2d(scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=N - 1)).T

        # Ensure correct shape for 1D state space
        if pd == 1:
            W_increments = W_increments.reshape(1, -1)

        # ---- Precompute jump increments for all intervals ----
        J_increments = np.zeros((pd, N-1))
        if rate > 0:
            # Sample total number of jumps over [0, T] from Poisson(rate * T)
            N_T = scipy.stats.poisson(mu=rate * T).rvs(size=1)[0]
            # Sample jump times uniformly over [0, T]
            jump_times = np.sort(scipy.stats.uniform().rvs(size=N_T) * T)
            # Sample jump sizes from the specified distribution
            jump_sizes = reshape_array(jump_F.rvs(size=N_T), d).T
            # Assign each jump to its corresponding interval index
            jump_indices = (jump_times // delta_t).astype(int)
            for i, jump_index in enumerate(jump_indices):
                # Exact contribution: e^{Q*(t_{n+1} - τ_i)} E ΔJ_i
                # Time remaining in interval after jump: (jump_index+1)*Δt - τ_i
                J_increments[:, jump_index] += scipy.linalg.fractional_matrix_power(eQ, (jump_index + 1) * delta_t - jump_times[i]).real.dot(E).dot(jump_sizes[:, i])

        # ---- Evolve state: X_{n+1} = e^{Q*Δt} X_n + drift + BM + jump ----
        for n in range(N - 1):
            X[:, n + 1] = eQt.dot(X[:, n]) + a_increment + W_increments[:, n] + J_increments[:, n]

    # ============================================================
    # NON-UNIFORM GRID: RECOMPUTE MATRIX QUANTITIES AT EACH STEP
    # ============================================================
    else:
        # Accumulate jump times and sizes across all intervals
        jump_times = np.array([])
        jump_sizes = np.array([])
        for n in range(N - 1):
            delta_t = P[n + 1] - P[n]   # Variable time step

            # e^{Q*Δt_n}: state transition matrix for this step
            eQt = scipy.linalg.fractional_matrix_power(eQ, delta_t).real

            # Drift increment: (e^{Q*Δt_n} - I) Q^{-1} E a
            a_increment = (eQt - np.eye(pd)).dot(Q_inv).dot(E).dot(a)
            
            # Brownian covariance V(Δt_n) for this step
            V = scipy.linalg.fractional_matrix_power(eM, delta_t).real[:pd, pd:].dot(eQt.T)

            # Sample Brownian increment for this step, with PSD repair if needed
            try:
                W_increment = scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=1).T
            except ValueError:
                eigvals, eigvecs = np.linalg.eigh(V)
                eigvals = np.maximum(eigvals, 1e-16)
                V = eigvecs @ np.diag(eigvals) @ eigvecs.T
                W_increment = scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=1).T

            # ---- Jump increment for interval (t_n, t_{n+1}] ----
            if rate > 0:
                # Number of jumps in this interval: N_Δ ~ Poisson(rate * Δt_n)
                N_delta = int(scipy.stats.poisson(mu=rate * delta_t).rvs(size=1))
                # Sample jump sizes and times within this interval
                jump_sizes_delta = reshape_array(jump_F.rvs(size=N_delta), d)
                jump_times_delta = np.sort(scipy.stats.uniform().rvs(size=N_delta) * delta_t)
                # Exact jump contribution: Σ_i e^{Q*(Δt_n - τ_i)} E ΔJ_i
                J_increment = np.zeros(pd)
                for _ in range(N_delta):
                    J_increment += scipy.linalg.fractional_matrix_power(eQ, delta_t - jump_times_delta[_]).real.dot(E).dot(jump_sizes_delta[_, :])

                # Record absolute jump times and sizes for output
                jump_times = np.append(jump_times, P[n+1] - jump_times_delta)
                jump_sizes = np.append(jump_sizes, jump_sizes_delta)

            else:
                J_increment = np.zeros(pd)

            # Exact evolution: X_{n+1} = e^{Q*Δt_n} X_n + drift + BM + jump
            X[:, n + 1] = eQt.dot(X[:, n]) + a_increment + W_increment + J_increment

    # ---- Return requested output format ----
    if output_format == 'MCAR':
        return X[:d, :]                     # Observed process Y = first d components
    elif output_format == 'SS':
        return X                            # Full state vector X
    elif output_format == 'SS + jumps':
        return X, jump_times, jump_sizes    # State + jump times + jump sizes
    else:
        raise ValueError("output_format must be one of ['MCAR', 'SS', 'SS + jumps']")

# ============================================================
# LÉVY INCREMENT GENERATION: SYMMETRIC GAMMA
# ============================================================

def gamma_increments(delta_t: float, n: int, d: int, shape: float, scale: float = 1):
    """
    Generate increments of a d-dimensional symmetric Gamma Lévy process.

    Each component is an independent symmetric Gamma process, constructed as the
    difference of two independent Gamma processes:
        ΔX_i = Gamma(shape * Δt, scale) - Gamma(shape * Δt, scale)

    This gives a symmetric distribution with mean 0 and variance 2 * shape * Δt * scale².
    The symmetric Gamma process is used as a tractable infinite-activity Lévy process
    in the simulation study of Chen & Veraart (2026), Section 4.1.

    Note: The shape parameter scales with Δt to ensure the process has stationary
    independent increments (i.e., X_{t+s} - X_t ~ Gamma(shape*s, scale) - Gamma(shape*s, scale)).

    :param delta_t: Time step size for each increment, float
    :param n:       Number of increments to generate, int
    :param d:       Dimension of the process (number of independent components), int
    :param shape:   Shape parameter of each Gamma distribution (before scaling by Δt), float
    :param scale:   Scale parameter of each Gamma distribution (default 1.0), float
    :return increments: Generated increments, (d, n) np.ndarray
    """
    # Generate d*n positive Gamma samples and subtract independent copies for symmetry
    increments = (scipy.stats.gamma.rvs(a=shape * delta_t, scale=scale, size=d * n) - scipy.stats.gamma.rvs(a=shape * delta_t, scale=scale, size=d * n)).reshape((d, -1))
    return increments