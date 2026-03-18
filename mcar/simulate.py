import numpy as np
import scipy
import scipy.linalg
import scipy.stats
from typing import Optional, Callable

def reshape_array(array, d):
    if d == 1:
        return array.reshape(-1, 1)
    elif len(array.shape) == 1:
        return array.reshape(1, -1)
    else:
        return array

def MCAR_Q(QQ: np.ndarray) -> np.ndarray:
    """
    Construct MCAR structural matrix from parameters Q_1, ..., Q_p.
    :param QQ: MCAR parameters, list of p (d, d) np.ndarray
    :return Q_QQ: MCAR structural matrix, (pd, pd) np.ndarray
    [ 0 I 0... 0
      0 0 I... 0...
    -Q_p...      -Q_1]
    """
    p = len(QQ)
    d = QQ[0].shape[0]
    Q_QQ = np.zeros((p * d, p * d))

    for i in range(d, p * d, d):
        Q_QQ[(i - d):i, i:(i + d)] = np.eye(d)
    for i in range(p):
        Q_QQ[-d:, i * d:(i + 1) * d] = - QQ[p - i - 1]
    return Q_QQ

def simulate_MCAR_stat_distr_approx(Q_QQ: np.ndarray, a: np.ndarray, Sigma: np.ndarray, jumps: Optional[Callable[[float, float], np.ndarray]]):
    """
    Simulate from stationary distribution of MCAR process by approximating integral discretely.
    :param Q_QQ: MCAR state-transition matrix, (pd, pd) np.ndarray
    :param a: drift of Levy process with truncation function t(z) = 0, i.e. E[L_1] = a + \int_{R^d} x F(dx), (d, ) np.ndarray
    :param Sigma: covariance of Brownian component of Levy process, (d, d) np.ndarray
    :param jumps: function for generating n jump increments over an interval delta_t, Callable[[float, float], np.ndarray] instead of distribution of jumps
    :return x: sample from stationary distribution, (pd, ) np.ndarray
    """
    pd = Q_QQ.shape[0]
    d = a.shape[0]
    E = np.zeros([pd, d])
    E[-d:,:] = np.eye(d)

    # approximate integral, choose T depending on |Q_QQ| (depending on how fast e^Q_QQs decays)
    T = - np.log(1e-8) / np.linalg.norm(Q_QQ)
    N = 1000
    P = np.linspace(0, T, N+1)
    delta_t = T/N
    if jumps is None:
        delta_jump_L = np.zeros([d, N])
    else:
        delta_jump_L = jumps(delta_t, N)
    delta_L = scipy.stats.multivariate_normal(mean=a*delta_t, cov=Sigma*delta_t).rvs(size=N).T + delta_jump_L
    return np.tensordot(scipy.linalg.expm(np.tensordot(P[:-1], Q_QQ, axes=0)), np.matmul(E, delta_L), axes=[[0, 2], [1, 0]])

def simulate_MCAR_stat_distr_compound_poisson(Q_QQ: np.ndarray, a: np.ndarray, Sigma: np.ndarray, rate: float, jump_F: Optional[scipy.stats._multivariate.multi_rv_frozen]):
    """
    Approximate a sample from the stationary distribution for a finite-activity MCAR (compound Poisson jumps) by simulating over a truncation horizon T.
    :param Q_QQ: MCAR state-transition matrix, (pd, pd) np.ndarray
    :param a: drift of Levy process with truncation function t(z) = 0, i.e. E[L_1] = a + \int_{R^d} x F(dx), (d, ) np.ndarray
    :param Sigma: covariance of Brownian component of Levy process, (d, d) np.ndarray
    :param rate: Poisson jump rate, float
    :param jump_F: distribution of jumps, scipy.stats._multivariate.multi_rv_frozen
    :return x: sample from stationary distribution, (pd, ) np.ndarray
    """
    if rate > 0 and jump_F is None:
        raise ValueError("jump_F must be provided if rate > 0.")

    pd = Q_QQ.shape[0]
    d = a.shape[0]

    # Noise coefficient and lifted covariance
    E = np.zeros([pd, d])
    E[-d:,:] = np.eye(d)
    Sigma_tilde = E.dot(Sigma).dot(E.T)

    T = - np.log(1e-8) / np.linalg.norm(Q_QQ)
    # Block matrix for covariance integral
    M = np.zeros((2 * pd, 2 * pd))
    M[:pd, :pd] = Q_QQ
    M[:pd, pd:] = Sigma_tilde
    M[pd:, pd:] = -Q_QQ.T
    V = scipy.linalg.expm(M * T)[:pd, pd:].dot(scipy.linalg.expm(Q_QQ * T).T)

    # Stationary drift component
    Q_inv = np.linalg.pinv(Q_QQ)
    a_component = -Q_inv.dot(E).dot(a)

    # Brownian component
    W_component = scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=1).T

    # Increment due to jumps
    if rate > 0:
        N_T = scipy.stats.poisson(mu=rate * T).rvs(size=1)[0]
        jump_times = np.sort(scipy.stats.uniform().rvs(size=N_T) * T)
        jump_sizes = reshape_array(jump_F.rvs(size=N_T), d).T
        J_component = np.einsum('ijk,ki->j',scipy.linalg.expm(np.tensordot(jump_times, Q_QQ, axes=0)), E.dot(jump_sizes))
    else:
        J_component = np.zeros(pd)

    x = a_component + W_component + J_component
    return x

def simulate_MCAR_approx(P: np.ndarray, Q_QQ: np.ndarray, x0: np.ndarray, a: np.ndarray, Sigma: np.ndarray, jumps: Optional[Callable[[float, float], np.ndarray]], output_format: str = 'MCAR', uniform = False):
    """
    Simulate (discrete) paths from a MCAR model with state-transition matrix Q_QQ (with infinite Levy measure)
    :param P: partition over which to simulate the MCAR process [0 = t_0, t_1, ..., t_N = T], (N+1, ) np.ndarray
    :param Q_QQ: MCAR state-transition matrix, (pd, pd) np.ndarray
    :param x0: state space initial condition, (pd, ) np.ndarray
    :param a: drift of Levy process with truncation function t(z) = 0, i.e. E[L_1] = a + \int_{R^d} x F(dx), (d, ) np.ndarray
    :param Sigma: covariance of Brownian component of Levy process, (d, d) np.ndarray
    :param jumps: function for generating n jump increments over a time interval delta_t, function(delta_t, n) instead of distribution of jumps
    :param uniform: if True, treats grid as uniform
    :param output_format: format of output, str in ['MCAR', 'SS', 'SS + L', 'SS + L + jump_L']
    :return Y: MCAR simulation, (d, N+1) np.ndarray
            X: MCAR state space simulation (first d entries are Y), (pd, N+1) np.ndarray
            L: driving Levy process, (d, N+1) np.ndarray
            jump_L: jumps in the driving Levy process, (d, N+1) np.ndarray
    """
    # get dimensions and time step
    d = len(a)
    pd = len(x0)
    N = len(P)

    # compute noise coefficient E
    E = np.zeros([pd, d])
    E[-d:, :] = np.eye(d)

    # initialize
    X = np.zeros([pd, N])
    L = np.zeros([d, N])
    jump_L = np.zeros([d, N])
    X[:, 0] = x0

    # simulate driving Brownian noise
    delta_W = np.sqrt(np.diff(P))[np.newaxis, :] * scipy.stats.multivariate_normal(cov=Sigma).rvs(size=N - 1).T
    delta_L = np.zeros((d, N - 1))

    if uniform:
        delta_t = P[1] - P[0]
        delta_L = a.reshape(-1, 1) * delta_t + delta_W
        if jumps is not None:
            delta_jump_L = jumps(delta_t, N - 1)
            delta_L += delta_jump_L
            jump_L[:, 1:] = delta_jump_L.cumsum(axis=1)

        # get Levy process
        L[:, 1:] = delta_L.cumsum(axis=1)

        # evolve state space
        for n in range(N - 1):
            X[:, n + 1] = X[:, n] + np.matmul(Q_QQ, X[:, n]) * delta_t + np.matmul(E, delta_L[:, n])
    else:
        for n in range(N - 1):
            delta_t = P[n + 1] - P[n]

            # Levy increment (continuous and jump parts)
            delta_L[:, n] = a.reshape(-1, 1) * delta_t + delta_W[:, n]
            if jumps is not None:
                delta_jump_L = jumps(delta_t, 1).reshape(-1, 1)
                delta_L[:, n] += delta_jump_L
                jump_L[:, n + 1] = jump_L[:, n] + delta_jump_L

            # evolve the processes
            L[:, n + 1] = L[:, n] + delta_L[:, n]

            # evolve state space
            X[:, n + 1] = X[:, n] + np.matmul(Q_QQ, X[:, n]) * delta_t + np.matmul(E, delta_L[:, n])

    if output_format == 'MCAR':
        return X[:d, :]
    elif output_format == 'SS':
        return X
    elif output_format == 'SS + L':
        return X, L
    elif output_format == 'SS + L + jump_L':
        return X, L, jump_L
    else:
        raise ValueError("output_format must be one of ['MCAR', 'SS', 'SS + L', 'SS + L + jump_L']")

def simulate_MCAR_compound_poisson(P: np.ndarray, Q_QQ: np.ndarray, x0: np.ndarray, a: np.ndarray, Sigma: np.ndarray, rate: float, jump_F: Optional[scipy.stats._multivariate.multi_rv_frozen], output_format: str = 'MCAR', uniform: bool = False):
    """
    Simulate (discrete) paths from a MCAR model with state-transition matrix Q_QQ and driving Levy process (with finite Levy measure) with triplet (a, Sigma, F).
    X_t = e^{A(t-s)} X_s + int_s^t e^{A(t-r)} dL_r = e^{A(t-s)}X_s + int_s^t e^{A(t-r)} (a dr + dW_r + dJ_r)
    where int_s^t e^{A(t-r)} a dr = (e^{A(t-s)} - I) A^{-1} a
          int_s^t e^{A(t-r)} dW_r ~ N(0, F(t-s) e^{A^T(t-s)}) where e^Mh = [e^{Ah}, F(h) // 0, e^{-A^Th}] with M = [A, Σ // 0, -A^T] for W~N(0, Σ)
          int_s^t e^{A(t-r)} dJ_r = sum_{r\in[s,t]} e^{A(t-r)} ΔJ_{r}
    recall need to multiply E b, E Sigma E^T, and E jumps
    :param P: partition over which to simulate the MCAR process [0 = t_0, t_1, ..., t_N = T], (N+1, ) np.ndarray
    :param Q_QQ: MCAR state-transition matrix, (pd, pd) np.ndarray
    :param x0: state space initial condition, (pd, ) np.ndarray
    :param a: drift of Levy process with truncation function t(z) = 0, i.e. E[L_1] = a + \int_{R^d} x F(dx), (d, ) np.ndarray
    :param Sigma: covariance of Brownian component of Levy process, (d, d) np.ndarray
    :param rate: jump rate of Levy process, float
    :param jump_F: distribution of jumps, scipy.stats._multivariate.multi_rv_frozen
    :param output_format: format of output, str in ['MCAR', 'SS', 'SS + jumps']
    :param uniform: if True, treats grid as uniform
    :return if output_format = 'MCAR':
                Y: MCAR simulation, (d, N+1) np.ndarray
            elif output_format = 'SS':
                X: MCAR state space simulation (first d entries are Y), (pd, N+1) np.ndarray
            elif output_format = 'SS + jumps':
                X: MCAR state space simulation (first d entries are Y), (pd, N+1) np.ndarray
                jump_times: jump times of the p-1 derivative of Y, (T, ) np.ndarray
                jump_sizes: jump magnitudes of the p-1 derivative of Y at jump_times, (d, T) np.ndarray
    """
    if rate > 0 and jump_F is None:
        raise ValueError("jump_F must be provided if rate > 0")

    d = len(a)
    pd = len(x0)
    N = len(P)
    T = P[-1] - P[0]
    
    # Noise coefficient E, Sigma_tilde
    E = np.zeros((pd, d))
    E[-d:,:] = np.eye(d)
    Sigma_tilde = E.dot(Sigma).dot(E.T)
    
    # Initialize
    X = np.zeros((pd, N))
    X[:, 0] = x0

    # Precompute block matrix
    M = np.zeros((2 * pd, 2 * pd))
    M[:pd, :pd] = Q_QQ
    M[:pd, pd:] = Sigma_tilde
    M[pd:, pd:] = -Q_QQ.T
    eM = scipy.linalg.expm(M)

    eQ = scipy.linalg.expm(Q_QQ)
    Q_inv = np.linalg.pinv(Q_QQ)
    
    if uniform:
        delta_t = P[1] - P[0]
        eQt = scipy.linalg.fractional_matrix_power(eQ, delta_t).real

        # Increment due to b
        a_increment = (eQt - np.eye(pd)).dot(Q_inv).dot(E).dot(a)

        # Increment due to W
        V = scipy.linalg.fractional_matrix_power(eM, delta_t).real[:pd, pd:].dot(eQt.T)

        # Repair PSD if needed
        try:
            W_increments = np.atleast_2d(scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=N-1)).T
        except ValueError:
            eigvals, eigvecs = np.linalg.eigh(V)
            eigvals = np.maximum(eigvals, 1e-16)
            V = eigvecs @ np.diag(eigvals) @ eigvecs.T
            W_increments = np.atleast_2d(scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=N - 1)).T

        if pd == 1:
            W_increments = W_increments.reshape(1, -1)

        # Increment due to jumps
        J_increments = np.zeros((pd, N-1))
        if rate > 0:
            N_T = scipy.stats.poisson(mu=rate * T).rvs(size=1)[0]
            jump_times = np.sort(scipy.stats.uniform().rvs(size=N_T) * T)
            jump_sizes = reshape_array(jump_F.rvs(size=N_T), d).T
            jump_indices = (jump_times // delta_t).astype(int)
            for i, jump_index in enumerate(jump_indices):
                J_increments[:, jump_index] += scipy.linalg.fractional_matrix_power(eQ, (jump_index + 1) * delta_t - jump_times[i]).real.dot(E).dot(jump_sizes[:, i])

        # Evolve the process
        for n in range(N - 1):
            X[:, n + 1] = eQt.dot(X[:, n]) + a_increment + W_increments[:, n] + J_increments[:, n]
    else:
        jump_times = np.array([])
        jump_sizes = np.array([])
        for n in range(N - 1):
            delta_t = P[n + 1] - P[n]
            eQt = scipy.linalg.fractional_matrix_power(eQ, delta_t).real

            # Increment due to b
            a_increment = (eQt - np.eye(pd)).dot(Q_inv).dot(E).dot(a)
            
            # Increment due to W
            V = scipy.linalg.fractional_matrix_power(eM, delta_t).real[:pd, pd:].dot(eQt.T)

            try:
                W_increment = scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=1).T
            except ValueError:
                eigvals, eigvecs = np.linalg.eigh(V)
                eigvals = np.maximum(eigvals, 1e-16)
                V = eigvecs @ np.diag(eigvals) @ eigvecs.T
                W_increment = scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=1).T

            # Increment due to jumps
            if rate > 0:
                N_delta = int(scipy.stats.poisson(mu=rate * delta_t).rvs(size=1))
                jump_sizes_delta = reshape_array(jump_F.rvs(size=N_delta), d)
                jump_times_delta = np.sort(scipy.stats.uniform().rvs(size=N_delta) * delta_t)
                J_increment = np.zeros(pd)
                for _ in range(N_delta):
                    J_increment += scipy.linalg.fractional_matrix_power(eQ, delta_t - jump_times_delta[_]).real.dot(E).dot(jump_sizes_delta[_, :])

                # Append jump times and sizes
                jump_times = np.append(jump_times, P[n+1] - jump_times_delta)
                jump_sizes = np.append(jump_sizes, jump_sizes_delta)

            else:
                J_increment = np.zeros(pd)

            # Evolve the process
            X[:, n + 1] = eQt.dot(X[:, n]) + a_increment + W_increment + J_increment

    if output_format == 'MCAR':
        return X[:d, :]
    elif output_format == 'SS':
        return X
    elif output_format == 'SS + jumps':
        return X, jump_times, jump_sizes
    else:
        raise ValueError("output_format must be one of ['MCAR', 'SS', 'SS + jumps']")

def gamma_increments(delta_t: float, n: int, d: int, shape: float, scale: float = 1):
    """
    Generate n increments of a d-dimensional process with independent symmetric Gamma components over an interval of size delta_t.
    :param delta_t: time step of increment, float
    :param n: number of increments to generate, int
    :param shape: shape of Gamma distribution, float
    :param scale: scale of Gamma distribution, float
    :param d: dimension of process, int
    :return increments: the increments of the process, (d, n) np.ndarray
    """
    increments = (scipy.stats.gamma.rvs(a=shape*delta_t, scale=scale, size=d*n) - scipy.stats.gamma.rvs(a=shape*delta_t, scale=scale, size=d*n)).reshape((d, -1))
    return increments