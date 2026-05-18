"""
================================================================================
GraphOUEdge: Graph Ornstein-Uhlenbeck Model for Edge-Indexed Network Time Series
================================================================================

This module provides the `GraphOUEdge` class, which implements the grOU(L, [R_1,...,R_L])
model for edge-indexed network time series introduced in:

    Chen & Veraart (2026): "Edge-Indexed Network Time Series with Graph
    Ornstein-Uhlenbeck Dynamics".

--------------------------------------------------------------------------------
Model Overview:
--------------------------------------------------------------------------------
The grOU edge model is a special case of the MCAR(p) framework in which the
drift matrices Q_1, ..., Q_L ∈ R^{K×K} are parametrized by the network structure:

    Q_l = diag(α_{e_1,l}, ..., α_{e_K,l}) + Σ_{r=1}^{R_l} β_{l,r} W^{(r)}

where:
    - K = |E|:              Number of edges in the network graph G = (V, E)
    - L:                    Maximum lag order
    - R_l:                  Maximum neighborhood stage at lag l
    - α_{e,l} ∈ R:          Standard autoregressive coefficient for edge e at lag l
    - β_{l,r} ∈ R:          Network interaction coefficient at lag l, stage r
    - W^{(r)} ∈ R^{K×K}:   Weighted r-stage neighborhood matrix (Definition 2.1)

The full state-space representation is:
    dX_t = Q X_t dt + E dL_t,   Y_t = A X_t

where X_t ∈ R^{LK} is the state vector, Q ∈ R^{LK×LK} is the MCAR companion
matrix (built from Q_1,...,Q_L), E injects Lévy noise into the last K components,
and L_t is a K-dimensional Lévy process with triplet (b, Σ, F).

--------------------------------------------------------------------------------
Supported Noise Regimes:
--------------------------------------------------------------------------------
Three Lévy noise regimes are supported for simulation:

  1. **BM** (Pure Brownian Motion):
     L_t = b*t + W_t, where W_t ~ BM(0, Σ).
     Uses exact simulation via matrix exponentials (rate = 0).

  2. **CP** (Compound Poisson):
     L_t = b*t + W_t + J_t, where J_t is a compound Poisson process with
     rate λ and Gaussian jump sizes N(0, scale * I_K).
     Uses exact simulation via matrix exponentials.

  3. **Gamma** (Symmetric Gamma):
     L_t has independent symmetric Gamma components (infinite activity).
     Uses Euler-Maruyama approximation.

--------------------------------------------------------------------------------
Parallelization:
--------------------------------------------------------------------------------
Monte Carlo simulation is parallelized using Python's `multiprocessing.Pool`,
with each sample path simulated independently in a separate worker process.
Each worker uses a deterministic seed (base_seed + sample_idx) for reproducibility.

--------------------------------------------------------------------------------
Usage Example:
--------------------------------------------------------------------------------
    model = GraphOUEdge(R_L=[1], K=2, C=1, Adj=[A], W_C=[W], gamma=1.5,
                        data_dir='./data', scale=1.0, T=8)
    model.alpha_set(np.array([[4.0, 3.0]]))
    model.beta_set([[1.0]])
    model.Q_matrix(strict=True)
    paths = model.simulate(regime='CP', n_samples=100, seed=42, rate=1.0)

--------------------------------------------------------------------------------
References:
--------------------------------------------------------------------------------
  - Chen & Veraart (2026): Edge-Indexed Network Time Series with Graph OU Dynamics
  - Lucchese, Pakkanen & Veraart (2026): Estimation and Inference for MCAR Processes
  - Mantziou et al. (2023): The GNAR-edge model: a network autoregressive model for networks with
time-varying edge weights
================================================================================
"""

from multiprocessing import Pool, cpu_count
from functools import partial
from mcar.simulate import *
from mcar.estimate import *

class GraphOUEdge:
    """
    Graph Ornstein-Uhlenbeck model for edge-indexed network time series.

    Implements the grOU(L, [R_1,...,R_L]) model of Chen & Veraart (2026), where
    the MCAR drift matrices are parametrized by a network graph structure via
    autoregressive coefficients (alpha) and network interaction coefficients (beta).

    The class supports:
      - Setting model parameters (alpha, beta) and constructing the companion matrix Q
      - Simulating Monte Carlo paths under three Lévy noise regimes (BM, CP, Gamma)
      - Parallel simulation using Python multiprocessing

    Attributes
    ----------
    L : int
        Maximum lag order (number of autoregressive lags).
    R_L : List[int]
        Maximum neighborhood stage at each lag l = 1, ..., L.
    K : int
        Number of edges in the network (dimension of the observed process Y_t).
    C : int
        Number of edge-weight covariate matrices.
    Adj : np.ndarray
        Adjacency matrices for each neighborhood stage, shape (max_stage, K, K).
    W_C : List[np.ndarray]
        Edge-weight matrices for each covariate, each of shape (K, K).
    gamma : float
        Hyperparameter for the jump-filtering critical region (must be > 1).
    alpha : np.ndarray
        Standard autoregressive coefficients, shape (L, K).
        alpha[l, e] = α_{e,l} is the self-autoregressive coefficient for edge e at lag l.
    beta : List[List[float]]
        Network interaction coefficients. beta[l] contains the C * R_L[l] coefficients
        [β_{l,1,1}, ..., β_{l,R_l,C}] for lag l.
    W : List[List[np.ndarray]]
        Weighted neighborhood matrices W^{(r)} ⊙ W_C[c] for each lag, stage, and covariate.
    Q : List[np.ndarray]
        Drift matrices [Q_1, ..., Q_L], each of shape (K, K). Populated by Q_matrix().
    Q_Q : np.ndarray
        MCAR companion matrix, shape (LK, LK). Populated by Q_matrix().
    b : np.ndarray
        Lévy drift parameter (no-truncation convention), shape (K,).
    Sigma : np.ndarray
        Brownian covariance matrix of the Lévy process, shape (K, K).
    jump_F : scipy distribution
        Jump size distribution for compound Poisson regime (finite activity).
    jumps : Callable
        Jump increment generator for Gamma regime (infinite activity).
    T : int
        Total simulation time horizon.
    N : int
        Number of fine-grid time steps (= T^7, satisfying the mesh condition Δ_P = T^{-6}).
    data_dir : str
        Directory for saving/loading simulation data.
    """

    def __init__(self, R_L: List, K: int, C: int, Adj: np.ndarray, W_C: List, gamma: float, data_dir: str, scale: float = 1, T: int = 8):
        """
        Initialize the GraphOUEdge model with network structure and default parameters.

        Sets up the model architecture (network graph, lag structure, covariate weights)
        and initializes all model parameters to default values. The companion matrix Q_Q
        is initialized to zero and must be populated by calling Q_matrix() before simulation.

        The fine-grid mesh is set to N = T^7 steps, which satisfies the asymptotic conditions Δ_P = T^{-6} and Δ_Q = T^{-2}
        required by Theorem 3.2 of Chen & Veraart (2026).

        :param R_L:      Maximum neighborhood stage at each lag, List[int] of length L
        :param K:        Number of edges (dimension of the observed process), int
        :param C:        Number of edge-weight covariate matrices, int
        :param Adj:      Adjacency matrices for each neighborhood stage, np.ndarray
        :param W_C:      Edge-weight matrices for each covariate, List[np.ndarray]
        :param gamma:    Critical region hyperparameter for jump filtering (must be > 1), float
        :param data_dir: Directory path for saving/loading simulation outputs, str
        :param scale:    Scale parameter for the default Gaussian jump distribution, float
        :param T:        Total simulation time horizon (also controls grid resolution), int
        """
        # Fix global random seed for reproducibility of initialization
        np.random.seed(123)

        # ---- Network and lag structure ----
        self.L = len(R_L)   # Number of autoregressive lags
        self.R_L = R_L      # Maximum neighborhood stage per lag
        self.K = K          # Number of edges (process dimension)
        self.C = C          # Number of covariate weight matrices

        # ---- Network graph structure ----
        self.Adj = Adj      # Adjacency matrices: Adj[r] is the r-stage adjacency matrix
        self.W_C = W_C      # Edge-weight matrices: W_C[c] is the c-th covariate weight

        # ---- Jump filtering hyperparameter ----
        self.gamma = gamma  # Critical region parameter γ > 1 (Theorem 3.2)

        # ---- Model parameters (initialized to zero/default) ----
        # alpha[l, e]: autoregressive coefficient for edge e at lag l
        self.alpha = np.zeros((self.L, self.K))

        # beta[l]: list of C * R_L[l] network interaction coefficients at lag l
        # Initialized to zero vectors of the appropriate length
        self.beta = [[0] * stage * C for stage in R_L]

        # W[l][c*R_l+r]: weighted neighborhood matrix W^{(r)} ⊙ W_C[c] at lag l, stage r
        # Used to construct Q_l = diag(alpha[l]) + [W matrices] @ beta[l]
        self.W = [[np.multiply(W_C[c], Adj[stage]) for c in range(C) for stage in range(R_i)] for R_i in R_L]  # Edge weights

        # ---- Drift matrices and companion matrix (populated by Q_matrix()) ----
        self.Q = []     # List of L drift matrices Q_1,...,Q_L
        self.Q_Q = np.zeros((self.L * K, self.L * K))  # MCAR companion matrix (LK × LK)

        # ---- Lévy triplet parameters ----
        self.b = np.zeros(K)    # Drift parameter b (no-truncation: E[L_1] = b + ∫ z F(dz))
        self.Sigma = np.eye(K)  # Brownian covariance Σ (initialized to identity)

        # Default jump distribution for compound Poisson regime (finite activity):
        # ΔJ ~ N(0, scale * I_K)
        self.jump_F = scipy.stats.multivariate_normal(mean=np.zeros(K), cov=scale * np.eye(K))  # Initial jump distribution for finitely many jumps

        # Default jump increment generator for Gamma regime (infinite activity):
        # Uses symmetric Gamma increments with shape=1, scale=1
        self.jumps = partial(gamma_increments, shape=1, scale=1, d=K)  # Initial jump increment function for infinitely many jumps

        # ---- Simulation grid parameters ----
        # log2T controls the number of time horizons in consistency experiments
        self.log2T = 3
        self.T = T      # Total time horizon T

        # Fine-grid size: N = T^7 steps → mesh Δ_P = T/N = T^{-6}
        # This satisfies the asymptotic condition Δ_P = o(t^{-1} Δ_Q^2) from Appendix C
        self.N = self.T ** 7 # Number of partitions

        # Output directory for saving simulation results
        self.data_dir = data_dir

    # ============================================================
    # PARAMETER SETTERS
    # ============================================================

    def alpha_set(self, alpha: np.ndarray):
        """
        Set the standard autoregressive coefficients alpha.

        :param alpha: Autoregressive coefficients, shape (L, K) np.ndarray.
                      alpha[l, e] = α_{e,l} is the self-autoregressive coefficient
                      for edge e at lag l+1 (0-indexed).
        """
        self.alpha = alpha

    def beta_set(self, beta: List):
        """
        Set the network interaction (neighborhood power) coefficients beta.

        :param beta: Network interaction coefficients, List of L sublists.
                     beta[l] is a list of C * R_L[l] floats:
                     [β_{l,1,1}, β_{l,1,2}, ..., β_{l,R_l,C}]
                     where β_{l,r,c} is the coefficient for lag l+1, stage r, covariate c.
        """
        self.beta = beta

    def Q_get(self):
        """
        Return the current list of drift matrices [Q_1, ..., Q_L].

        :return: List of L drift matrices, each (K, K) np.ndarray.
                 Returns an empty list if Q_matrix() has not been called yet.
        """
        return self.Q

    # ============================================================
    # COMPANION MATRIX CONSTRUCTION
    # ============================================================

    def Q_matrix(self, strict: bool = True):
        """
        Construct the MCAR companion matrix Q_Q from the current alpha and beta parameters.

        Builds each drift matrix Q_l using the grOU parametrization (Section 2.3):
            Q_l = diag(α_{e_1,l}, ..., α_{e_K,l})
                  + [W^{(1)}⊙W_C[1], ..., W^{(R_l)}⊙W_C[C]] @ kron(β_l, I_K)^T

        Then assembles the LK × LK MCAR companion matrix via MCAR_Q():
            Q_Q = [ 0    I    0   ...  0  ]
                  [ 0    0    I   ...  0  ]
                  [ .    .    .   ...  .  ]
                  [-Q_L -Q_{L-1} ...  -Q_1]

        Optionally asserts that all eigenvalues of Q_Q have strictly negative real parts,
        which is required for the process to be stationary (Hurwitz condition,
        Definition 2.1 of Chen & Veraart 2026).

        :param strict: If True, assert that Q_Q is a Hurwitz matrix (all eigenvalues
                       have strictly negative real parts), required for stationarity. bool
        :return Q_Q:   MCAR companion matrix, (LK, LK) np.ndarray
        :raises AssertionError: If strict=True and Q_Q is not Hurwitz.
        """
        self.Q = []
        l = 0
        while l < self.L:
            # Build Q_l = diag(alpha[l]) + [W matrices] @ kron(beta[l], I_K)^T
            # np.concatenate(self.W[l], axis=1): horizontal stack of all W^{(r)}⊙W_C[c] at lag l
            # np.tensordot(beta[l], I_K, axes=0): outer product giving kron(beta[l], I_K)
            # np.concatenate(..., axis=0): vertical stack to form the Kronecker structure
            self.Q.append(np.diag(self.alpha[l]) + np.matmul(np.concatenate(self.W[l], axis=1), np.concatenate(
                np.tensordot(np.array(self.beta[l]).T, np.identity(self.K), axes=0), axis=0)))
            l += 1

        # Assemble the MCAR companion matrix from Q_1, ..., Q_L
        self.Q_Q = MCAR_Q(np.array(self.Q))

        # ---- Stationarity check (Hurwitz condition) ----
        # The grOU process is stationary iff all eigenvalues of Q_Q have Re(λ) < 0
        # (Remark 2.2, Chen & Veraart 2026)
        if strict:
            evals, evecs = np.linalg.eig(self.Q_Q)
            assert (np.real(evals) < 0).all()

        return self.Q_Q

    # ============================================================
    # PARALLEL SIMULATION WORKER
    # ============================================================

    def _simulate_one(self, args):
        """
        Simulate a single MCAR path for one Monte Carlo replicate.

        This is the worker function called by each subprocess in the parallel
        simulation pool. It sets a deterministic per-sample random seed to ensure
        reproducibility while allowing parallel execution.

        Three noise regimes are supported:
          - 'BM':    Pure Brownian motion (rate=0, exact simulation)
          - 'CP':    Compound Poisson jumps (rate>0, exact simulation)
          - 'Gamma': Symmetric Gamma noise (infinite activity, Euler-Maruyama)

        For each regime, the procedure is:
          1. Sample the initial state x0 from the stationary distribution.
          2. Simulate a path of length N+1 on the uniform grid [0, T].

        :param args: Tuple (regime, sample_idx, rate, seed) where:
                       - regime (str):     Noise regime ('BM', 'CP', or 'Gamma')
                       - sample_idx (int): Index of this sample in the output array
                       - rate (float):     Poisson jump rate (used for 'CP' regime only)
                       - seed (int):       Random seed for this replicate
        :return: Tuple (sample_idx, path) where path is (K, N+1) np.ndarray,
                 or None if an invalid regime is specified.
        """
        regime, sample_idx, rate, seed = args

        # Set per-sample seed for reproducibility across parallel workers
        np.random.seed(seed)

        if regime == 'BM':
            # Pure Brownian motion: rate=0, no jumps
            # Sample stationary initial condition exactly
            x0 = simulate_MCAR_stat_distr_compound_poisson(self.Q_Q, self.b, self.Sigma, 0, self.jump_F)
            # Simulate exact path using matrix exponentials
            path = simulate_MCAR_compound_poisson(np.linspace(0, self.T, self.N + 1), self.Q_Q, x0, self.b, self.Sigma, 0, self.jump_F, uniform=True)

        elif regime == 'CP':
            # Compound Poisson jumps: rate > 0, Gaussian jump sizes N(0, scale*I_K)
            # Sample stationary initial condition exactly
            x0 = simulate_MCAR_stat_distr_compound_poisson(self.Q_Q, self.b, self.Sigma, rate, self.jump_F)
            # Simulate exact path using matrix exponentials
            path = simulate_MCAR_compound_poisson(np.linspace(0, self.T, self.N + 1), self.Q_Q, x0, self.b, self.Sigma, rate, self.jump_F, uniform=True)

        elif regime == 'Gamma':
            # Symmetric Gamma noise (infinite activity Lévy process)
            # Sample stationary initial condition via Riemann-sum approximation
            x0 = simulate_MCAR_stat_distr_approx(self.Q_Q, self.b, self.Sigma, self.jumps)
            # Simulate path using Euler-Maruyama approximation
            path = simulate_MCAR_approx(np.linspace(0, self.T, self.N + 1), self.Q_Q, x0, self.b, self.Sigma, self.jumps, uniform=True)

        else:
            print("Regime must be 'BM', 'CP' or 'Gamma'.")
            return None
        return sample_idx, path

    # ============================================================
    # PARALLEL MONTE CARLO SIMULATION
    # ============================================================

    def simulate(self, regime: str, n_samples: int, seed: int, rate: float = 0, n_jobs: int = None):
        """
        Simulate n_samples independent MCAR paths in parallel using multiprocessing.

        Each sample path is simulated independently on the uniform time grid
        [0, T] with N+1 points, starting from a draw from the stationary distribution.
        Parallelization is achieved via Python's multiprocessing.Pool, with each
        worker receiving a unique deterministic seed (seed + sample_idx) to ensure
        both reproducibility and statistical independence across replicates.

        The companion matrix Q_Q must be constructed via Q_matrix() before calling
        this method. The Lévy triplet parameters b and Sigma should also be set
        to the desired values before simulation.

        :param regime:    Noise regime for the driving Lévy process:
                            'BM'    - Pure Brownian motion (no jumps)
                            'CP'    - Compound Poisson jumps (requires rate > 0)
                            'Gamma' - Symmetric Gamma noise (infinite activity)
        :param n_samples: Number of independent Monte Carlo paths to simulate, int
        :param seed:      Base random seed; worker i uses seed + i for reproducibility, int
        :param rate:      Poisson jump arrival rate (used only for 'CP' regime), float
        :param n_jobs:    Number of parallel worker processes; defaults to cpu_count(), int
        :return MCAR:     Simulated paths array, shape (n_samples, K, N+1) np.ndarray,
                          where MCAR[i, :, :] is the i-th simulated path of Y_t ∈ R^K.
                          Returns None if Q_matrix() has not been called.
        """
        # Guard: ensure companion matrix has been constructed before simulation
        if np.array_equal(self.Q_Q, np.zeros((self.L * self.K, self.L * self.K))):
            print('Run method Q_matrix first before simulation. (set b and Sigma as well)')
            return None

        # Pre-allocate output array: (n_samples, K, N+1)
        MCAR = np.zeros((n_samples, self.K, self.N + 1))

        # Default to using all available CPU cores
        if n_jobs is None:
            n_jobs = cpu_count()

        # Build argument list for parallel workers:
        # Each worker receives (regime, sample_index, rate, unique_seed)
        args_list = [(regime, i, rate, seed + i) for i in range(n_samples)]

        # Launch parallel simulation pool
        # imap with chunksize=10 balances overhead vs. granularity for large n_samples
        with Pool(processes = n_jobs) as pool:
            for sample_idx, path in pool.imap(self._simulate_one, args_list, chunksize=10):
                # Store each completed path in the pre-allocated output array
                MCAR[sample_idx, :, :] = path

        return MCAR