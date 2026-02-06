import os
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from mcar.simulate import *
from mcar.estimate import *

class GraphOUEdge:
    def __init__(self, R_L: np.ndarray, K: int, C: int, Adj, W_C, data_dir: str, plots_dir: str):
        self.L = len(R_L)  # Lags
        self.R_L = R_L  # Maximum stage for each Lag
        self.K = K  # Edges
        self.C = C  # Covariates
        self.Adj = Adj  # Adjacency matrices for different stages
        self.W_C = W_C  # Edge weights for Covariates
        self.alpha = np.zeros((self.L, self.K))    # Initialize standard autoregressive parameters
        self.beta = [[0] * stage * C for stage in R_L]  # Initialize network power parameters
        self.W = [[np.multiply(W_C[c], Adj[stage]) for c in range(C) for stage in range(R_i)] for R_i in R_L]  # Edge weights
        self.Q = []   # Set of parameters
        self.Q_Q = np.zeros((self.L * K, self.L * K))  # Initialize state-transition matrix
        self.b = np.zeros(K)    # Initial b in the Levy triplet
        self.Sigma = np.eye(K)  # Initial Sigma in the Levy triplet
        self.jump_F = scipy.stats.multivariate_normal(mean=np.zeros(K), cov=np.eye(K))  # Initial jump distribution for finitely many jumps
        self.jumps = partial(gamma_increments, shape=1, scale=1, d=K)   # Initial jump increment function for infinitely many jumps
        self.log2T = 3  # Number of different time lengths
        self.T = 2 ** self.log2T # Total duration
        self.N = self.T ** 7 # Number of partitions
        self.P = np.linspace(0, self.T, self.N + 1)  # delta_P = T^(-6)
        self.data_dir = data_dir
        self.plots_dir = plots_dir

    def W_get(self):
        return self.W

    def alpha_get(self):
        return self.alpha

    def alpha_set(self, alpha: np.ndarray):
        self.alpha = alpha

    def beta_get(self):
        return self.beta

    def beta_set(self, beta: List):
        self.beta = beta

    def Q_matrix(self, strict: bool = True):
        """Compute state-transition matrix with given network structure and parameters."""
        self.Q = []
        l = 0
        while l < self.L:
            self.Q.append(np.diag(self.alpha[l]) + np.matmul(np.concatenate(self.W[l], axis=1), np.concatenate(
                np.tensordot(np.array(self.beta[l]).T, np.identity(self.K), axes=0), axis=0)))
            l += 1
        self.Q_Q = MCAR_A(np.array(self.Q))

        if strict:
            evals, evecs = np.linalg.eig(self.Q_Q)
            assert (np.real(evals) < 0).all()

        return self.Q_Q

    def LevyProcess_get(self):
        return [self.b, self.Sigma, self.jump_F, self.jumps]

    def LevyProcess_set(self, b: np.ndarray, Sigma: np.ndarray, jump_F: Optional[scipy.stats._multivariate.multi_rv_frozen] = None, jumps: Optional[Callable[[float, float], np.ndarray]] = None):
        self.b = b
        self.Sigma = Sigma
        if jump_F is not None:
            self.jump_F = jump_F
        if jumps is not None:
            self.jumps = jumps

    def _simulate_one(self, args):
        """Helper function for parallel execution."""
        regime, sample_idx, rate = args
        if regime == 'BM':
            x0 = simulate_MCAR_stat_distr_compound_poisson(self.Q_Q, self.b, self.Sigma, 0, self.jump_F)
            path = simulate_MCAR_compound_poisson(self.P, self.Q_Q, x0, self.b, self.Sigma, 0, self.jump_F, uniform=True)
        elif regime == 'CP':
            try:
                assert rate > 0
            except AssertionError:
                print('Need positive rate.')
                return None
            x0 = simulate_MCAR_stat_distr_compound_poisson(self.Q_Q, self.b, self.Sigma, rate, self.jump_F)
            path = simulate_MCAR_compound_poisson(self.P, self.Q_Q, x0, self.b, self.Sigma, rate, self.jump_F, uniform=True)
        elif regime == 'Gamma':
            x0 = simulate_MCAR_stat_distr_approx(self.Q_Q, self.b, self.Sigma, self.jumps)
            path = simulate_MCAR_approx(self.P, self.Q_Q, x0, self.b, self.Sigma, self.jumps, uniform=True)
        else:
            print("Regime must be 'BM', 'CP' or 'Gamma'.")
            return None
        return sample_idx, path

    def simulate(self, regime: str, n_samples: int, rate: float = 0, n_jobs: int = None):
        """Simulate and save the sample paths."""
        if np.array_equal(self.Q_Q, np.zeros((self.L * self.K, self.L * self.K))):
            print('Run method Q_matrix first before simulation. (set b and Sigma as well)')
            return
        MCAR = np.zeros((n_samples, self.K, self.N + 1))
        if n_jobs is None:
            n_jobs = cpu_count()
        args_list = [(regime, i, rate) for i in range(n_samples)]
        with Pool(processes = n_jobs) as pool:
            for sample_idx, path in pool.imap_unordered(self._simulate_one, args_list, chunksize=10):
                MCAR[sample_idx, :, :] = path
        with open(os.path.join(self.data_dir, f'MCAR_{regime}.npy'), 'wb') as f:
            np.save(f, MCAR)

    def _estimate_one(self, args):
        """Helper function for parallel execution."""
        sample_idx, Y, beta = args

        estimates_sample = np.zeros((self.log2T, self.L * self.K ** 2))
        estimates_cov_sample = np.zeros((self.log2T, self.L * self.K ** 2, self.L * self.K ** 2))

        for n in range(1, self.log2T + 1):
            T = 2 ** n
            subset_P = slice(0, int(self.N / 2 ** (self.log2T - n)) + 1, 2 ** (6 * (self.log2T - n)))
            P_ = self.P[subset_P]  # delta_P = T^(-6)
            Q_ = P_[::T ** 4]  # delta_Q = T^(-2)

            nu = np.diff(Q_) ** beta

            Y_ = Y[:, subset_P]
            QQ_hat, HH = estimate_MCAR_drift(Y_, self.L, P_, Q_, self.b, nu, with_cov=True, Sigma=self.Sigma)

            estimates_sample[n - 1, :] = QQ_hat
            estimates_cov_sample[n - 1, :, :] = HH

        return sample_idx, estimates_sample, estimates_cov_sample

    def estimate_with_known_Levy(self, jump_regimes: List, betas: List, n_jobs=None):
        """Estimate and save the parameter matrix with known Levy process."""
        for jump_regime, beta in zip(jump_regimes, betas):
            try:
                with open(os.path.join(self.data_dir, f'MCAR_{jump_regime}.npy'), 'rb') as f:
                    MCAR = np.load(f)
            except FileNotFoundError:
                print("Jump regimes must be a list of 'BM', 'CP' or 'Gamma'.")
                return
            n_samples = MCAR.shape[0]
            estimates = np.zeros((n_samples, self.log2T, self.L * self.K ** 2))
            estimates_Cov = np.zeros((n_samples, self.log2T, self.L * self.K ** 2, self.L * self.K ** 2))

            if n_jobs is None:
                n_jobs = cpu_count()

            args_list = [(i, MCAR[i, :, :], beta) for i in range(n_samples)]

            with Pool(processes = n_jobs) as pool:
                for idx, est, cov in pool.imap_unordered(self._estimate_one, args_list, chunksize=10):
                    estimates[idx, :, :] = est
                    estimates_Cov[idx, :, :, :] = cov

            with open(os.path.join(self.data_dir, f'MCAR_{jump_regime}_estimates.npy'), 'wb') as f:
                np.save(f, estimates)

            with open(os.path.join(self.data_dir, f'MCAR_{jump_regime}_estimates_Cov.npy'), 'wb') as f:
                np.save(f, estimates_Cov)

    def _estimate_one_global(self, args):
        """Helper function for parallel execution."""
        sample_idx, Y, beta = args

        estimates_sample = np.zeros((self.log2T, self.L + sum(self.R_L) * self.C))
        estimates_cov_sample = np.zeros((self.log2T, self.L + sum(self.R_L) * self.C, self.L + sum(self.R_L) * self.C))

        for n in range(1, self.log2T + 1):
            T = 2 ** n
            subset_P = slice(0, int(self.N / 2 ** (self.log2T - n)) + 1, 2 ** (6 * (self.log2T - n)))
            P_ = self.P[subset_P]  # delta_P = T^(-6)
            Q_ = P_[::T ** 4]  # delta_Q = T^(-2)

            nu = np.diff(Q_) ** beta

            Y_ = Y[:, subset_P]
            theta_hat, KK, _ = estimate_grCAR_drift(Y_, self.Adj, self.W_C, self.L, self.R_L, self.C, P_, Q_, self.b, nu, self.Sigma, with_cov=True)

            estimates_sample[n - 1, :] = theta_hat
            estimates_cov_sample[n - 1, :, :] = KK

        return sample_idx, estimates_sample, estimates_cov_sample

    def estimate_with_known_Levy_global(self, jump_regimes: List, betas: List, n_jobs=None):
        """Estimate and save the parameter matrix in global model with known Levy process."""
        for jump_regime, beta in zip(jump_regimes, betas):
            try:
                with open(os.path.join(self.data_dir, f'MCAR_{jump_regime}.npy'), 'rb') as f:
                    MCAR = np.load(f)
            except FileNotFoundError:
                print("Jump regimes must be a list of 'BM', 'CP' or 'Gamma'.")
                return
            n_samples = MCAR.shape[0]
            estimates = np.zeros((n_samples, self.log2T, self.L + sum(self.R_L) * self.C))
            estimates_Cov = np.zeros((n_samples, self.log2T, self.L + sum(self.R_L) * self.C, self.L + sum(self.R_L) * self.C))

            if n_jobs is None:
                n_jobs = cpu_count()

            args_list = [(i, MCAR[i, :, :], beta) for i in range(n_samples)]

            with Pool(processes = n_jobs) as pool:
                for idx, est, cov in pool.imap_unordered(self._estimate_one_global, args_list, chunksize=10):
                    estimates[idx, :, :] = est
                    estimates_Cov[idx, :, :, :] = cov

            with open(os.path.join(self.data_dir, f'grCAR_{jump_regime}_estimates.npy'), 'wb') as f:
                np.save(f, estimates)

            with open(os.path.join(self.data_dir, f'grCAR_{jump_regime}_estimates_Cov.npy'), 'wb') as f:
                np.save(f, estimates_Cov)

    def estimate_unknown_Levy(self, Y, P_, Q_, model, jump_activity: str = 'finite', is_global=True):
        """Estimate and save the parameters of unknown Levy process."""

        QQ_hat, (b_hat, Sigma_hat) = estimate_MCAR(Model = model, Y=Y, P=P_, Q=Q_, A=self.Adj, W_C=self.W_C, R_L=self.R_L, C=self.C, jump_activity=jump_activity, is_global=is_global)
        if model == 'OU':
            theta_hat, KK, K = estimate_OU_drift(Y, self.R_L, self.C, P_, Q_, b_hat, None, Sigma_hat, with_cov=True, is_global=is_global)
        elif model == 'grOU':
            theta_hat, KK, K = estimate_grCAR_drift(Y, self.Adj, self.W_C, self.L, self.R_L, self.C, P_, Q_, b_hat, None, Sigma_hat, with_cov=True)
        else:
            raise ValueError('Model not recognized')
        DeltaL = recover_BDLP(Y=Y, p=self.L, P=P_, Q=Q_, AA=QQ_hat)

        if jump_activity == 'finite':
            F_mean_hat = estimate_integral_Levy_measure(DeltaL, Q=Q_, Sigma=Sigma_hat, K=lambda x: True, f=lambda x: x)
            F_squared_hat = estimate_integral_Levy_measure(DeltaL, Q=Q_, Sigma=Sigma_hat, K=lambda x: True, f=lambda x: np.outer(x, x))
        elif jump_activity == 'infinite':
            F_mean_hat = estimate_integral_Levy_measure(DeltaL, Q=Q_, Sigma=Sigma_hat, K=lambda x: np.linalg.norm(x) > 1e-3, f=lambda x: x)
            F_squared_hat = estimate_integral_Levy_measure(DeltaL, Q=Q_, Sigma=Sigma_hat, K=lambda x: np.linalg.norm(x) > 1e-3, f=lambda x: np.outer(x, x))
        else:
            raise ValueError("jump_activity must be 'finite' or 'infinite.")

        with open(os.path.join(self.data_dir, f'MCAR_estimates_Sigma_{jump_activity}.npy'), 'wb') as f:
            np.save(f, Sigma_hat)

        with open(os.path.join(self.data_dir, f'MCAR_estimates_b_{jump_activity}.npy'), 'wb') as f:
            np.save(f, b_hat)

        with open(os.path.join(self.data_dir, f'MCAR_estimates_F_squared_{jump_activity}.npy'), 'wb') as f:
            np.save(f, F_squared_hat)

        with open(os.path.join(self.data_dir, f'MCAR_estimates_F_mean_{jump_activity}.npy'), 'wb') as f:
            np.save(f, F_mean_hat)

        return np.matmul(theta_hat, K) - 0.5 * np.matmul(theta_hat, np.matmul(KK, theta_hat))

    def compute_beta(self, Q_hat, lag):
        Q_hat = Q_hat.reshape(self.K, -1)
        np.fill_diagonal(Q_hat, 0)
        A = np.concatenate(self.W[lag - 1], axis=1)
        stage = self.R_L[lag - 1]

        # Build M = [vec(A1) ... vec(A_{2L})] in R^{K^2 x 2L}
        blocks = [A[:, i * self.K:(i + 1) * self.K] for i in range(self.C * stage)]
        M = np.column_stack([blk.reshape(-1) for blk in blocks])
        rhs = Q_hat.reshape(-1)

        # Least-squares solution for b in R^{2L}
        b, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        return b.reshape(self.C * lag, 1)

    def plot_alpha(self, jump_regimes: List, plot_type: str = 'kde_contour'):
        """Plot and save joint distribution of estimated alphas with known Levy process in 'histogram', 'kde' or 'kde_contour'."""
        try:
            assert self.Q
        except AssertionError:
            print('Run method Q_matrix first before plotting.')
            return
        for jump_regime in jump_regimes:
            try:
                with open(os.path.join(self.data_dir, f'MCAR_{jump_regime}_estimates.npy'), 'rb') as f:
                    estimates = np.load(f)
            except FileNotFoundError:
                print("Jump regimes must be a list of 'BM', 'CP' or 'Gamma'.")
                return
            fig, axes = plt.subplots(4, self.log2T, figsize=(24, 18), squeeze=False)
            for log2t in range(1, self.log2T + 1):
                for n in range(self.L):
                    for m in range(self.K // 2):
                        ax = axes[2 * n + m, log2t - 1]
                        x = estimates[:, log2t - 1, n * self.K ** 2 + (self.K + 1) * 2 * m]
                        y = estimates[:, log2t - 1, n * self.K ** 2 + (self.K + 1) * (2 * m + 1)]

                        if plot_type == 'histogram':
                            ax.hexbin(x=x, y=y, cmap='Blues_r', gridsize=18, extent=[-1, 4, -1, 4])
                        elif plot_type[:3] == 'kde':
                            k = scipy.stats.gaussian_kde(np.vstack([x, y]))
                            xi, yi = np.mgrid[0:8:100 * 1j, 0:8:100 * 1j]
                            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                            if plot_type == 'kde':
                                ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='Blues_r', shading='auto')
                            elif plot_type == 'kde_contour':
                                ax.contourf(xi, yi, zi.reshape(xi.shape), cmap='Blues_r')

                        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
                        ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
                        ax.set_xlabel(r'$\alpha_{%i,%i}$' % (n + 1, 2 * m + 1), fontsize=15)
                        ax.set_ylabel(r'$\alpha_{%i,%i}$' % (n + 1, 2 * m + 2), fontsize=15)
                        ax.plot(self.Q[n][2 * m][2 * m], self.Q[n][2 * m + 1][2 * m + 1], color='red', marker='.', markersize=15)
                        ax.set_box_aspect(1)

                        if n == 0 and m == 0:
                            ax.set_title(r'$t = %i$' % 2 ** log2t, fontsize=20)

            fig.savefig(os.path.join(self.plots_dir, f'MCAR_{plot_type}_{jump_regime}_alpha.eps'), bbox_inches='tight',
                        format='eps')

    def plot_beta(self, jump_regimes: List, plot_type: str = 'kde_contour'):
        """Plot and save joint distribution of estimated betas with known Levy process in 'histogram', 'kde' or 'kde_contour'"""
        for jump_regime in jump_regimes:
            try:
                with open(os.path.join(self.data_dir, f'MCAR_{jump_regime}_estimates.npy'), 'rb') as f:
                    estimates = np.load(f)
            except FileNotFoundError:
                print("Jump regimes must be a list of 'BM', 'CP' or 'Gamma'.")
                return
            fig, axes = plt.subplots(2, self.log2T, figsize=(24, 18), squeeze=False)
            for log2t in range(1, self.log2T + 1):
                num_of_plot = 0
                for n in range(self.L):
                    for m in range(self.R_L[n]):
                        if num_of_plot > 1:
                            continue
                        ax = axes[num_of_plot, log2t - 1]
                        if self.R_L[n] == 1:
                            x = [self.compute_beta(estimates[i, log2t - 1, n * self.K ** 2:(n + 1) * self.K ** 2], n + 1)[m].item()
                                 for i in range(estimates.shape[0])]
                            y = [self.compute_beta(estimates[i, log2t - 1, n * self.K ** 2:(n + 1) * self.K ** 2], n + 1)[m].item()
                                 for i in range(estimates.shape[0])] + 1e-6 * np.random.randn(len(x))
                            ax.set_xlabel(r'$\beta_{%i,%i}$' % (n + 1, m + 1), fontsize=15)
                            ax.set_ylabel(r'$\beta_{%i,%i}$' % (n + 1, m + 1), fontsize=15)
                            ax.plot(self.beta[n][m], self.beta[n][m], color='red', marker='.', markersize=15)
                        else:
                            x = [self.compute_beta(estimates[i, log2t - 1, n * self.K ** 2:(n + 1) * self.K ** 2], n + 1)[m].item()
                                 for i in range(estimates.shape[0])]
                            y = [self.compute_beta(estimates[i, log2t - 1, n * self.K ** 2:(n + 1) * self.K ** 2], n + 1)[
                                     m + 1].item() for i in range(estimates.shape[0])]
                            ax.set_xlabel(r'$\beta_{%i,%i}$' % (n + 1, m + 1), fontsize=15)
                            ax.set_ylabel(r'$\beta_{%i,%i}$' % (n + 1, m + 2), fontsize=15)
                            ax.plot(self.beta[n][m], self.beta[n][m + 1], color='red', marker='.', markersize=15)

                        if plot_type == 'histogram':
                            ax.hexbin(x=x, y=y, cmap='Blues_r', gridsize=18, extent=[-1, 4, -1, 4])
                        elif plot_type[:3] == 'kde':
                            k = scipy.stats.gaussian_kde(np.vstack([x, y]))
                            xi, yi = np.mgrid[-2:2:100 * 1j, -2:2:100 * 1j]
                            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                            if plot_type == 'kde':
                                ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='Blues_r', shading='auto')
                            elif plot_type == 'kde_contour':
                                ax.contourf(xi, yi, zi.reshape(xi.shape), cmap='Blues_r')
                        else:
                            print("Print type must be 'histogram', 'kde' or 'kde_contour'.")
                            return
                        ax.set_xticks([-1.25, -.75, -.25, .25, .75, 1.25, 1.75])
                        ax.set_yticks([-1.25, -.75, -.25, .25, .75, 1.25, 1.75])
                        ax.set_box_aspect(1)

                        if n == 0 and m == 0:
                            ax.set_title(r'$t = %i$' % 2 ** log2t, fontsize=20)
                        num_of_plot += 1
            fig.savefig(os.path.join(self.plots_dir, f'MCAR_{plot_type}_{jump_regime}_beta.eps'), bbox_inches='tight',
                        format='eps')

    def plot_alpha_global(self, jump_regimes: List, plot_type: str = 'kde_contour'):
        """Plot and save joint distribution of estimated alphas in global model with known Levy process in 'histogram', 'kde' or 'kde_contour'."""
        for jump_regime in jump_regimes:
            try:
                with open(os.path.join(self.data_dir, f'grCAR_{jump_regime}_estimates.npy'), 'rb') as f:
                    estimates = np.load(f)
            except FileNotFoundError:
                print("Jump regimes must be a list of 'BM', 'CP' or 'Gamma'.")
                return

            fig, axes = plt.subplots(self.L // 2, self.log2T, figsize=(24, 18), squeeze=False)

            for log2t in range(1, self.log2T + 1):
                alphas = []
                idx = 0
                for r in self.R_L:
                    alphas.append(estimates[:, log2t - 1, idx])  # alpha_i
                    idx += 1 + r * self.C
                for n in range(self.L // 2):
                    ax = axes[n, log2t - 1]
                    x = alphas[2 * n]
                    y = alphas[2 * n + 1]

                    if plot_type == 'histogram':
                        ax.hexbin(x=x, y=y, cmap='Blues_r', gridsize=18, extent=[-1, 4, -1, 4])
                    elif plot_type[:3] == 'kde':
                        k = scipy.stats.gaussian_kde(np.vstack([x, y]))
                        xi, yi = np.mgrid[0:8:100 * 1j, 0:8:100 * 1j]
                        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                        if plot_type == 'kde':
                            ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='Blues_r', shading='auto')
                        elif plot_type == 'kde_contour':
                            ax.contourf(xi, yi, zi.reshape(xi.shape), cmap='Blues_r')

                    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
                    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
                    ax.set_xlabel(r'$\alpha_{%i}$' % (2 * n + 1), fontsize=15)
                    ax.set_ylabel(r'$\alpha_{%i}$' % (2 * n + 2), fontsize=15)
                    ax.plot(self.Q[2 * n][0][0], self.Q[2 * n + 1][0][0], color='red', marker='.', markersize=15)
                    ax.set_box_aspect(1)

                    if n == 0:
                        ax.set_title(r'$t = %i$' % 2 ** log2t, fontsize=20)

            fig.savefig(os.path.join(self.plots_dir, f'grCAR_{plot_type}_{jump_regime}_alpha.eps'), bbox_inches='tight',
                        format='eps')

    def plot_beta_global(self, jump_regimes: List, plot_type: str = 'kde_contour'):
        """Plot and save joint distribution of estimated betas in global model with known Levy process in 'histogram', 'kde' or 'kde_contour'."""
        for jump_regime in jump_regimes:
            try:
                with open(os.path.join(self.data_dir, f'grCAR_{jump_regime}_estimates.npy'), 'rb') as f:
                    estimates = np.load(f)
            except FileNotFoundError:
                print("Jump regimes must be a list of 'BM', 'CP' or 'Gamma'.")
                return

            fig, axes = plt.subplots(2, self.log2T, figsize=(24, 18), squeeze=False)

            for log2t in range(1, self.log2T + 1):
                betas_c1 = []
                idx = 0
                for r in self.R_L:
                    idx += 1
                    betas_c1.append(estimates[:, log2t - 1, idx: idx + r].T)
                    idx += r * self.C
                for n in range(self.L):
                    ax = axes[n, log2t - 1]
                    if self.R_L[n] == 1:
                        x = betas_c1[n][0]
                        y = x + 1e-6 * np.random.randn(len(x))
                        ax.set_xlabel(r'$\beta_{%i,%i}$' % (n + 1, 1), fontsize=15)
                        ax.set_ylabel(r'$\beta_{%i,%i}$' % (n + 1, 1), fontsize=15)
                        ax.plot(self.beta[n][0], self.beta[n][0], color='red', marker='.', markersize=15)
                    else:
                        for m in range(self.R_L[n] - 1):
                            x = betas_c1[n][m]
                            y = betas_c1[n][m + 1]
                            ax.set_xlabel(r'$\beta_{%i,%i}$' % (n + 1, m + 1), fontsize=15)
                            ax.set_ylabel(r'$\beta_{%i,%i}$' % (n + 1, m + 2), fontsize=15)
                            ax.plot(self.beta[n][m], self.beta[n][m + 1], color='red', marker='.', markersize=15)

                    if plot_type == 'histogram':
                        ax.hexbin(x=x, y=y, cmap='Blues_r', gridsize=18, extent=[-1, 4, -1, 4])
                    elif plot_type[:3] == 'kde':
                        k = scipy.stats.gaussian_kde(np.vstack([x, y]))
                        xi, yi = np.mgrid[-2:2:100 * 1j, -2:2:100 * 1j]
                        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                        if plot_type == 'kde':
                            ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='Blues_r', shading='auto')
                        elif plot_type == 'kde_contour':
                            ax.contourf(xi, yi, zi.reshape(xi.shape), cmap='Blues_r')

                    ax.set_xticks([-1.25, -.75, -.25, .25, .75, 1.25, 1.75])
                    ax.set_yticks([-1.25, -.75, -.25, .25, .75, 1.25, 1.75])
                    ax.set_box_aspect(1)

                    if n == 0:
                        ax.set_title(r'$t = %i$' % 2 ** log2t, fontsize=20)

            fig.savefig(os.path.join(self.plots_dir, f'grCAR_{plot_type}_{jump_regime}_beta.eps'), bbox_inches='tight',
                        format='eps')