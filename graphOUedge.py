from multiprocessing import Pool, cpu_count
from functools import partial
from mcar.simulate import *
from mcar.estimate import *

class GraphOUEdge:
    def __init__(self, R_L: List, K: int, C: int, Adj: np.ndarray, W_C: List, gamma: float, data_dir: str, scale: float = 1, T: int = 8):
        self.L = len(R_L)  # Lags
        self.R_L = R_L  # Maximum stage for each Lag
        self.K = K  # Edges
        self.C = C  # Covariates
        self.Adj = Adj  # Adjacency matrices for different stages
        self.W_C = W_C  # Edge weights for Covariates
        self.gamma = gamma # Hyperparameter for defining the critical region >1
        self.alpha = np.zeros((self.L, self.K))    # Initialize standard autoregressive parameters
        self.beta = [[0] * stage * C for stage in R_L]  # Initialize network power parameters
        self.W = [[np.multiply(W_C[c], Adj[stage]) for c in range(C) for stage in range(R_i)] for R_i in R_L]  # Edge weights
        self.Q = []   # Set of parameters
        self.Q_Q = np.zeros((self.L * K, self.L * K))  # Initialize state-transition matrix
        self.b = np.zeros(K)    # Initial b in the Levy triplet
        self.Sigma = np.eye(K)  # Initial Sigma in the Levy triplet
        self.jump_F = scipy.stats.multivariate_normal(mean=np.zeros(K), cov=scale * np.eye(K))  # Initial jump distribution for finitely many jumps
        self.jumps = partial(gamma_increments, shape=1, scale=1, d=K)  # Initial jump increment function for infinitely many jumps
        self.log2T = 3  # Number of different time lengths
        self.T = T # Total duration
        self.N = self.T ** 7 # Number of partitions
        self.data_dir = data_dir

    def alpha_set(self, alpha: np.ndarray):
        self.alpha = alpha

    def beta_set(self, beta: List):
        self.beta = beta

    def Q_get(self):
        return self.Q

    def Q_matrix(self, strict: bool = True):
        """Compute state-transition matrix with given network structure and parameters."""
        self.Q = []
        l = 0
        while l < self.L:
            self.Q.append(np.diag(self.alpha[l]) + np.matmul(np.concatenate(self.W[l], axis=1), np.concatenate(
                np.tensordot(np.array(self.beta[l]).T, np.identity(self.K), axes=0), axis=0)))
            l += 1
        self.Q_Q = MCAR_Q(np.array(self.Q))

        if strict:
            evals, evecs = np.linalg.eig(self.Q_Q)
            assert (np.real(evals) < 0).all()

        return self.Q_Q

    def _simulate_one(self, args):
        """Helper function for parallel execution."""
        regime, sample_idx, rate, seed = args
        np.random.seed(seed)
        if regime == 'BM':
            x0 = simulate_MCAR_stat_distr_compound_poisson(self.Q_Q, self.b, self.Sigma, 0, self.jump_F)
            path = simulate_MCAR_compound_poisson(np.linspace(0, self.T, self.N + 1), self.Q_Q, x0, self.b, self.Sigma, 0, self.jump_F, uniform=True)
        elif regime == 'CP':
            x0 = simulate_MCAR_stat_distr_compound_poisson(self.Q_Q, self.b, self.Sigma, rate, self.jump_F)
            path = simulate_MCAR_compound_poisson(np.linspace(0, self.T, self.N + 1), self.Q_Q, x0, self.b, self.Sigma, rate, self.jump_F, uniform=True)
        elif regime == 'Gamma':
            x0 = simulate_MCAR_stat_distr_approx(self.Q_Q, self.b, self.Sigma, self.jumps)
            path = simulate_MCAR_approx(np.linspace(0, self.T, self.N + 1), self.Q_Q, x0, self.b, self.Sigma, self.jumps, uniform=True)
        else:
            print("Regime must be 'BM', 'CP' or 'Gamma'.")
            return None
        return sample_idx, path

    def simulate(self, regime: str, n_samples: int, seed: int, rate: float = 0, n_jobs: int = None):
        """Simulate and save the sample paths."""
        if np.array_equal(self.Q_Q, np.zeros((self.L * self.K, self.L * self.K))):
            print('Run method Q_matrix first before simulation. (set b and Sigma as well)')
            return None
        MCAR = np.zeros((n_samples, self.K, self.N + 1))
        if n_jobs is None:
            n_jobs = cpu_count()
        args_list = [(regime, i, rate, seed + i) for i in range(n_samples)]
        with Pool(processes = n_jobs) as pool:
            for sample_idx, path in pool.imap_unordered(self._simulate_one, args_list, chunksize=10):
                MCAR[sample_idx, :, :] = path

        return MCAR