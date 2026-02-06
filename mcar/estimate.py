import numpy as np
from typing import Callable, List
import warnings

def state_space(Y: np.ndarray, p: int, P: np.ndarray) -> np.ndarray:
    """
    Construct state-space representation of process from observation Y up to order p-1.
    :param Y: MCAR process observation, (d, N+1) np.ndarray
    :param p: state space order, int
    :param P: partition over which to simulate the MCAR process [0=t_0, t_1, ..., t_N = T], (N+1, ) np.ndarray
    :return X: state space representation (first d entries are Y), (pd, N+1) np.ndarray
    """
    # get dimensions
    d, N = Y.shape
    X = np.zeros([p*d, N])
    X[:d, :] = Y

    # differentiate p-1 times to get state space representation
    for i in range(1, p):
        X[d*i:d*(i+1), :-1] = np.diff(X[d*(i-1):d*i, :], axis=1) / (P[1:] - P[:-1])
    return X

def estimate_MCAR_drift(Y: np.ndarray, p: int, P: np.ndarray, Q: np.ndarray, b: np.ndarray, nu: np.ndarray = None, with_cov: bool = False, Sigma: np.ndarray = None):
    """
    Estimate MCAR drift parameters from the realisation of a MCAR process with driving Levy triplet (b, Sigma, F).
    Here the Levy drift parameter b corresponds to
        - no truncation t(x) = 0 when the process has finite jump activity, thus E[L_1] = b + \int_{R^d} z F(dz).
        - classic truncation t(x) = 1_{|x|<=1} when the process has infinite jump activity, thus E[L_1] = b + \int_{|z|>1} z F(dz).

    :param Y: MCAR process observation, (d, N+1) np.ndarray
    :param p: MCAR parameter, int
    :param P: finer partition over which we observe the MCAR process [0 = s_0, ..., s_N = T], (N+1, ) np.ndarray
    :param Q: coarser partition over which to approximate the integrals [0 = u_0, ..., u_M = T], (M+1, ) np.ndarray
    :param b: drift parameter of Levy process, (d, ) np.ndarray
    :param nu: thresholding sequence, (d, M) np.ndarray
    :param with_cov: whether to return the vectorized estimator with its estimated covariance, bool
    :param Sigma: covariance matrix of the driving Levy process - only needed if with_cov is True, (d, d) np.ndarray
    :return if with_cov is True:
                vec_AA_hat: vectorized estimated MCAR drift parameters, (pd**2, ) np.ndarray
                HH: estimated covariance of the estimated MCAR parameters, (pd**2, pd**2) np.ndarray
            else:
                AA_hat: estimated MCAR drift parameters, list of p (d, d) np.ndarray
    """
    # get dimension
    d = Y.shape[0]

    if with_cov and Sigma is not None:
        # compute inverse of Sigma
        Sigma_inv = np.linalg.pinv(Sigma)
    elif with_cov and Sigma is None:
        # estimate Sigma from data
        AA_hat = estimate_MCAR_drift(Y, p, P, Q, b, nu, with_cov=False)
        DeltaL = recover_BDLP(Y, p, P, Q, AA_hat)
        Sigma_hat = estimate_Sigma_L(DeltaL, Q)
        Sigma_inv = np.linalg.pinv(Sigma_hat)
    else:
        # no need to use Sigma if cov is not needed
        Sigma_inv = np.eye(d)

    # compute H and its quadratic variation
    H, HH = compute_H_HH(Y, p, P, Q, b, nu, Sigma_inv)

    # compute vec_AA_hat
    vec_AA_hat = np.matmul(np.linalg.pinv(HH), H)

    if with_cov and Sigma is not None:
        return vec_AA_hat, HH
    elif with_cov and Sigma is None:
        return vec_AA_hat, HH, Sigma_hat
    else:
        AA_hat = inv_vectorize_AA(vec_AA_hat, p)
        return AA_hat

def estimate_grCAR_drift(Y: np.ndarray, A, W_C, p: int, R_L: np.ndarray, C: int, P: np.ndarray, Q: np.ndarray, b: np.ndarray, nu: np.ndarray = None, Sigma: np.ndarray = None, with_cov: bool = False):
    """
    Estimate grCAR drift parameters from the realisation of a grCAR process with driving Levy triplet (b, Sigma, F).
    Here the Levy drift parameter b corresponds to
        - no truncation t(x) = 0 when the process has finite jump activity, thus E[L_1] = b + \int_{R^d} z F(dz).
        - classic truncation t(x) = 1_{|x|<=1} when the process has infinite jump activity, thus E[L_1] = b + \int_{|z|>1} z F(dz).
    :param Y: MCAR process observation, (d, N+1) np.ndarray
    :param A: Graph adjacency matrix, maximum stage of (d, d) np.ndarray
    :param W_C: Edge weights for Covariates
    :param p: Number of considered lags, int
    :param R_L: Maximum stage for each Lag, list of int
    :param C: Covariates, int
    :param P: Finer partition over which we observe the GrCAR process [0 = s_0, ..., s_N = t], (N+1, ) np.ndarray
    :param Q: Coarser partition over which to approximate the integrals [0 = u_0, ..., u_M = t], (M+1, ) np.ndarray
    :param b: Drift of Levy process, (d, ) np.ndarray
    :param nu: Thresholding sequence, (d, M) np.ndarray
    :param Sigma: Covariance matrix of the driving Levy process, (d, d) np.ndarray
    :param with_cov: whether to return the vectorized estimator with its estimated covariance, bool
    :return if with_cov is True:
                vec_theta_hat: vectorized estimated grCAR parameters, (p + sum(R_L) * C, ) np.ndarray
                KK: estimated covariance of the estimated grCAR parameters, (p + sum(R_L) * C, p + sum(R_L) * C) np.ndarray
    """
    # approximate the state space representation by finite differences over P
    X = state_space(Y, p, P)

    # approximate integrals on Q
    if not all([u in P for u in Q]):
        raise ValueError('P must be refinement of Q.')
    indices = [False] * len(P)
    j = 0
    for i, s in enumerate(P):
        if s == Q[j]:
            indices[i] = True
            j += 1
    X = X[:, indices]

    # get dimensions and time horizon
    d = Y.shape[0]
    x0 = X[:, 0]

    if with_cov and Sigma is not None:
        # compute inverse of Sigma
        Sigma_inv = np.linalg.pinv(Sigma)
    elif with_cov and Sigma is None:
        # estimate Sigma from data
        global_AA_hat = estimate_grCAR_drift(Y, A, W_C, p, R_L, C, P, Q, b, with_cov=False)
        AA_hat = estimator_to_QQ(global_AA_hat, R_L, C, W_C, A, d)
        DeltaL = recover_BDLP(Y, p, P, Q, AA_hat)
        Sigma_hat = estimate_Sigma_L(DeltaL, Q)
        Sigma_inv = np.linalg.pinv(Sigma_hat)
    else:
        # no need to use Sigma if cov is not needed
        Sigma_inv = np.eye(d)

    # get P_0-continuous martingale part of last d entries of state space representation, i.e. of the p-1-th derivative of grCAR realisation
    DY = np.diff(X[-d:, :] - x0[-d:].reshape(-1, 1) - b.reshape(-1, 1) * Q.reshape(1, -1))

    # if nu is not provided choose from data
    if nu is None:
        nu = np.zeros_like(DY)
        for i in range(d):
            nu[i, :] = choose_nu(DY[i:i+1, :], Q)

    # get thresholded increments
    DY_thresh = DY * (np.abs(DY) < nu)

    # compute K
    K = np.zeros(p + sum(R_L) * C)
    idx = 0
    for k in range(p):
        integral = np.zeros((d, d))
        if k == 0:
            integrand = X[-d:, :]
        else:
            integrand = X[-d-k*d:-k*d, :]
        for i in range(d):
            for j in range(d):
                # left-hand Riemann sum (converges in L2 to Ito Integral)
                integral[i, j] = np.sum(integrand[i, :-1] * DY_thresh[j, :])

        K[idx] = -np.sum(np.multiply(Sigma_inv, integral))
        for c in range(C):
            for r in range(R_L[k]):
                K[idx + c * R_L[k] + r + 1] = -np.sum(np.multiply(np.matmul(np.multiply(W_C[c], A[r]).T, Sigma_inv), integral))
        idx += C * R_L[k] + 1

    # compute [K]
    KK = np.zeros((p + sum(R_L) * C, p + sum(R_L) * C))
    idx_k = 0
    for k in range(p):
        idx_l = 0
        for l in range(p):
            if k == 0:
                integrand_one = X[-d:, :]
            else:
                integrand_one = X[-d-k*d:-k*d, :]
            if l == 0:
                integrand_two = X[-d:, :]
            else:
                integrand_two = X[-d-l*d:-l*d, :]
            integral = np.zeros((d, d))
            for i in range(d):
                for j in range(d):
                    # left-hand Riemann sum (converges a.s. to Riemann Integral, for any choice of mid-point)
                    integral[i, j] = np.sum(integrand_one[i, :-1] * integrand_two[j, :-1] * (Q[1:] - Q[:-1]))
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

    # compute and return theta_hat
    vec_theta_hat = np.matmul(np.linalg.pinv(KK), K)

    if with_cov and Sigma is not None:
        return vec_theta_hat, KK, K
    elif with_cov and Sigma is None:
        return vec_theta_hat, KK, Sigma_hat
    else:
        return vec_theta_hat

def estimate_OU_drift(Y: np.ndarray, R_L: np.ndarray, C: int, P: np.ndarray, Q: np.ndarray, b: np.ndarray,
                      nu: np.ndarray = None, Sigma: np.ndarray = None, with_cov: bool = False, is_global: bool = True):
    """
    Estimate OU drift parameters from the realisation of a process with driving Levy triplet (b, Sigma, F).
    Here the Levy drift parameter b corresponds to
        - no truncation t(x) = 0 when the process has finite jump activity, thus E[L_1] = b + \int_{R^d} z F(dz).
        - classic truncation t(x) = 1_{|x|<=1} when the process has infinite jump activity, thus E[L_1] = b + \int_{|z|>1} z F(dz).
    :param Y: Process observation, (d, N+1) np.ndarray
    :param R_L: Maximum stage for each Lag
    :param C: Covariates
    :param P: finer partition over which we observe the GrCAR process [0 = s_0, ..., s_N = t], (N+1, ) np.ndarray
    :param Q: coarser partition over which to approximate the integrals [0 = u_0, ..., u_M = t], (M+1, ) np.ndarray
    :param b: drift of Levy process, (d, ) np.ndarray
    :param nu: thresholding sequence, (d, M) np.ndarray
    :param Sigma: covariance matrix of the driving Levy process, (d, d) np.ndarray
    :param with_cov: whether to return the vectorized estimator with its estimated covariance, bool
    :param is_global: whether to return the global estimator, bool
    :return if with_cov is True:
                vec_theta_hat: vectorized estimated OU parameters
                KK: estimated covariance of the estimated OU parameters
    """
    # approximate the state space representation by finite differences over P
    p = len(R_L)
    X = state_space(Y, p, P)

    # approximate integrals on Q
    if not all([u in P for u in Q]):
        raise ValueError('P must be refinement of Q.')
    indices = [False] * len(P)
    j = 0
    for i, s in enumerate(P):
        if s == Q[j]:
            indices[i] = True
            j += 1
    X = X[:, indices]

    # get dimensions and time horizon
    d = Y.shape[0]
    x0 = X[:, 0]

    if with_cov and Sigma is not None:
        # compute inverse of Sigma
        Sigma_inv = np.linalg.pinv(Sigma)
    elif with_cov and Sigma is None:
        # estimate Sigma from data
        theta_hat = estimate_OU_drift(Y, R_L, C, P, Q, b, with_cov=False, is_global=is_global)
        AA_hat = np.zeros((p, d, d))
        idx = np.arange(d)
        AA_hat[:, idx, idx] = theta_hat
        DeltaL = recover_BDLP(Y, p, P, Q, AA_hat)
        Sigma_hat = estimate_Sigma_L(DeltaL, Q)
        Sigma_inv = np.linalg.pinv(Sigma_hat)
    else:
        # no need to use Sigma if cov is not needed
        Sigma_inv = np.eye(d)

    # get P_0-continuous martingale part of last d entries of state space representation, i.e. of the p-1-th derivative of grCAR realisation
    DY = np.diff(X[-d:, :] - x0[-d:].reshape(-1, 1) - b.reshape(-1, 1) * Q.reshape(1, -1))

    # if nu is not provided choose from data
    if nu is None:
        nu = np.zeros_like(DY)
        for i in range(d):
            nu[i, :] = choose_nu(DY[i:i + 1, :], Q)

    # get thresholded increments
    DY_thresh = DY * (np.abs(DY) < nu)

    # compute K
    if is_global:
        K = np.zeros(p)
        for k in range(p):
            integral = np.zeros((d, d))
            if k == 0:
                integrand = X[-d:, :]
            else:
                integrand = X[-d - k * d:-k * d, :]
            for i in range(d):
                for j in range(d):
                    # left-hand Riemann sum (converges in L2 to Ito Integral)
                    integral[i, j] = np.sum(integrand[i, :-1] * DY_thresh[j, :])

            K[k] = -np.sum(np.multiply(Sigma_inv, integral))

        # compute [K]
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
                        # left-hand Riemann sum (converges a.s. to Riemann Integral, for any choice of mid-point)
                        integral[i, j] = np.sum(integrand_one[i, :-1] * integrand_two[j, :-1] * (Q[1:] - Q[:-1]))

                KK[k, l] = np.sum(np.multiply(Sigma_inv, integral))

        # compute and return theta_hat
        vec_theta_hat = np.repeat(np.matmul(np.linalg.pinv(KK), K)[:, np.newaxis], d, axis=1)

    else:
        K = np.zeros(p * d)
        for k in range(p):
            integral = np.zeros((d, d))
            if k == 0:
                integrand = X[-d:, :]
            else:
                integrand = X[-d - k * d:-k * d, :]
            for i in range(d):
                for j in range(d):
                    # left-hand Riemann sum (converges in L2 to Ito Integral)
                    integral[i, j] = np.sum(integrand[i, :-1] * DY_thresh[j, :])

            K[k] = -np.sum(np.multiply(Sigma_inv, integral))

        # compute [K]
        KK = np.zeros((p * d, p * d))
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
                        # left-hand Riemann sum (converges a.s. to Riemann Integral, for any choice of mid-point)
                        integral[i, j] = np.sum(integrand_one[i, :-1] * integrand_two[j, :-1] * (Q[1:] - Q[:-1]))

                KK[k, l] = np.sum(np.multiply(Sigma_inv, integral))

        vec_theta_hat = np.matmul(np.linalg.pinv(KK), K).reshape(p, d)

    if with_cov and Sigma is not None:
        return vec_theta_hat, KK, K
    elif with_cov and Sigma is None:
        return vec_theta_hat, KK, Sigma_hat
    else:
        return vec_theta_hat

def recover_BDLP(Y: np.ndarray, p: int, P: np.ndarray, Q: np.ndarray, AA: np.ndarray):
    """
    Recover the increments of the background driving Levy process L over coarser partition Q from MCAR observation Y over finer partition P with parameters AA.
    :param Y: MCAR process observation, (d, N+1) np.ndarray
    :param p: MCAR parameter, int
    :param P: finer partition over which we observe the MCAR process [0 = s_0, ..., s_N = t], (N+1, ) np.ndarray
    :param Q: coarser partition over which we recover the Levy increments [0 = u_0, ..., u_M = T], (M+1, ) np.ndarray
    :param AA: MCAR parameters, list of p (d, d) np.ndarray
    :return DeltaL_Q: increments of the background driving Levy process L on the partition Q, (d, M) np.ndarray
    """
    # get dimensions and time horizon
    d = Y.shape[0]

    # approximate the state space representation by finite differences over P
    X_P = state_space(Y, p, P)
    Delta_P = np.diff(P)

    # find elements of Q in P
    indices = [False] * len(P)
    j = 0
    for i, s in enumerate(P):
        if s == Q[j]:
            indices[i] = True
            j += 1

    # restrict X to Q
    X_Q = X_P[:, indices]
    DeltaX_Q = np.diff(X_Q, axis = 1)

    # approximate drift on P
    drift_P = np.zeros((d, len(Delta_P)))
    for j in range(p):
        DjY = X_P[j*d:(j+1)*d, :-1]
        drift_P += AA[p-j-1].dot(DjY) * Delta_P.T

    # coarsen drift to Q
    cum_drift = np.cumsum(drift_P, axis=1)
    drift_Q = np.diff(np.concatenate([np.zeros((d, 1)), cum_drift[:, indices[:-1]]], axis=1), axis=1)

    # return L on Q
    DeltaL_Q = DeltaX_Q[-d:, :] + drift_Q

    return DeltaL_Q

def disentangle_BM(DeltaL: np.ndarray, Q: np.ndarray, Sigma: np.ndarray, gamma: float = 1.3):
    """
    Disentangle Brownian-only increments on the partition Q.
    To define the critical region B_N we generalize the approach in (Gegler, 2011) to irregularly spaced data.
    :param DeltaL: increments of Levy process on the partition Q, (d, N) np.ndarray
    :param Q: partition, [0 = u_0, ..., u_{N} = T], (N+1, ) np.ndarray
    :param Sigma: covariance of the Brownian part of the Levy process, (d, d) np.ndarray
    :param gamma: hyperparameter for defining the critical region > 1, float
    :return DeltaW: subset of elements of DeltaL corresponding to the diffusion part only, (d, M) np.ndarray with M <= N
            DeltaQ_W: time increments corresponding to the elements in DeltaW, (M, ) np.ndarray
    """
    # identify critical region x^T Sigma_inv x <= beta*2*Delta*log(N)
    DeltaQ = np.diff(Q)
    N = len(DeltaQ)
    DeltaX = DeltaL
    subset = np.einsum('ki,ki->i', DeltaX, np.linalg.pinv(Sigma).dot(DeltaX)) <= 2*gamma*DeltaQ*np.log(N)

    # return Brownian increments
    DeltaQ_W = DeltaQ[subset]
    DeltaW = DeltaX[:, subset]
    return DeltaW, DeltaQ_W

def estimate_Sigma(DeltaW: np.ndarray, DeltaQ: np.ndarray):
    """
    Estimate the covariance matrix Sigma from the (irregularly spaced) Brownian increments DeltaW.
    :param DeltaW: Brownian increments, (d, M)
    :param DeltaQ: time increments corresponding to the elements in DeltaW, (M, ) np.ndarray
    """
    d = DeltaW.shape[0]
    if len(DeltaQ) == 0:
        return np.zeros((d, d))
    DeltaZ = DeltaW / np.sqrt(DeltaQ)
    Sigma_hat = np.einsum('ki,ji->kj', DeltaZ, DeltaZ) / DeltaZ.shape[1]
    return Sigma_hat

def estimate_Sigma_L(DeltaL: np.ndarray, Q: np.ndarray, epsilon: float = 0.01, gamma: float = 1.3):
    """
    Estimate the covariance matrix Sigma of the Brownian component from (irregularly spaced) Levy increments.
    Use the iterative approach in (Gegler, 2011) Section 4.
    :param DeltaL: increments of Levy process on the partition Q, (d, N) np.ndarray
    :param Q: partition, [0 = u_1, ..., u_N = T], (N+1, ) np.ndarray
    :param epsilon: tolerance level for defining convergence of iterative scheme, float
    :param gamma: hyperparameter for defining the critical region >1, float
    :return Sigma_hat: estimated covariance of the Brownian component, (d, d) np.ndarray
    """
    # initialize procedure
    DeltaQ_W = np.diff(Q)
    DeltaW = DeltaL
    Sigma_hat_new = estimate_Sigma(DeltaW, DeltaQ_W)
    converged = False

    # keep track of Sigmas to see if enter a loop
    Sigmas = [Sigma_hat_new]
    while not converged:
        Sigma_hat_old = Sigma_hat_new
        DeltaW, DeltaQ_W = disentangle_BM(DeltaL, Q, Sigma_hat_old, gamma=gamma)
        Sigma_hat_new = estimate_Sigma(DeltaW, DeltaQ_W)
        converged = np.linalg.norm(Sigma_hat_new - Sigma_hat_old) <= epsilon

        # check if entering a loop
        if any(np.array_equal(Sigma_hat_new, Sigma) for Sigma in Sigmas):
            converged = True
            # take average of loop values
            index = np.argmax(np.array([np.array_equal(Sigma_hat_new, Sigma) for Sigma in Sigmas]))
            Sigma_hat_new = np.mean(np.array(Sigmas)[index:, :, :], axis = 0)
        else:
            Sigmas.append(Sigma_hat_new)
    return Sigma_hat_new

def estimate_integral_Levy_measure(DeltaL: np.ndarray, Q: np.ndarray, Sigma: np.ndarray = None, epsilon: float = 0.01, gamma: float = 1.3, K: Callable = lambda x: np.linalg.norm(x) > 1e-3, f: Callable = lambda x: x):
    """
    Estimate the functional f(F) = \int_K f(z) F(dz) where f: R^d -> R^n of the Levy measure F from (irregularly spaced) Levy increments DeltaL.
    Theoretical guarantees of convergence are given in (Gegler, 2011) for functions of the form f(z) = (z^T A z)^l for matrix A and power l.
    If Sigma is known use the rejection-region based estimator in (Gegler, 2011) Section 2, otherwise use the iterative approach in (Gegler, 2011) Section 4.
    :param DeltaL: increments of Levy process on the partition Q, (d, N) np.ndarray
    :param Q: partition, [0 = u_1, ..., u_N = T], (N+1, ) np.ndarray
    :param Sigma: covariance of Brownian component of L, (d, d) np.ndarray
    :param epsilon: tolerance level for defining convergence of iterative scheme, float
    :param gamma: hyperparameter for defining the critical region > 1, float
    :param K: integration region in functional, function mapping (d, ) np.ndarray to bool
    :param f: integrand function, function mapping (d, ) np.ndarray to float
    :return f_hat: estimated functional f of the Levy measure, float
    """
    if Sigma is None:
        Sigma = estimate_Sigma_L(DeltaL, Q, epsilon, gamma)

    d = Sigma.shape[0]

    DeltaQ = np.diff(Q)
    N = len(DeltaQ)

    # identify critical region x^T Sigma_inv x > beta*2*Delta*log(N)
    subset = np.logical_and(
        np.einsum('ki,ki->i', np.linalg.pinv(Sigma).dot(DeltaL), DeltaL) > 2*gamma*DeltaQ*np.log(N),
        [K(DeltaL[:, i]) for i in range(DeltaL.shape[1])]
    )
    DeltaZ = DeltaL[:, subset]
    if DeltaZ.shape[1] < 1:
        f_hat = np.zeros(f(np.zeros(d)).shape) if isinstance(f(np.zeros(d)), np.ndarray) else 0
    else:
        # check this is OK also for non-uniform partition
        f_hat = np.sum(np.array([f(DeltaZ[:, i]) for i in range(DeltaZ.shape[1])]), axis=0) / (Q[-1] - Q[0])
    return f_hat

def choose_nu(DeltaL: np.ndarray, Q: np.ndarray, epsilon: float = 0.01, gamma: float = 1.3, Sigma: np.ndarray = None):
    """
    This function chooses thresholds nu to disentangle the continuous component from the jump component of a one dimensional Levy process.
    :param DeltaL: increments of Levy process on the partition Q, (1, N) np.ndarray
    :param Q: partition, [0 = u_1, ..., u_N = T], (N+1, ) np.ndarray
    :param epsilon: tolerance level for defining convergence of iterative scheme, float
    :param gamma: hyperparameter for defining the critical region > 1, float
    :param Sigma: covariance of Brownian component of L, (1, 1) np.ndarray
    :return nu: thresholding vector, (1, N) np.ndarray
    """
    DeltaQ = np.diff(Q)
    Sigma_hat = estimate_Sigma_L(DeltaL, Q, epsilon, gamma) if Sigma is None else Sigma
    nu = np.sqrt(DeltaQ * 2 * gamma * np.log(len(DeltaQ)) * Sigma_hat)
    return nu

def estimate_MCAR(
        Model: str,
        Y: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        A: np.ndarray,
        W_C: np.ndarray,
        R_L: np.ndarray,
        C: int,
        epsilon: float = 0.01,
        max_iter: int = 200,
        AA: np.ndarray = None,
        b: np.ndarray = None,
        Sigma: np.ndarray = None,
        f: np.ndarray = None,
        mu: np.ndarray = None,
        jump_activity: str = 'infinite',
        mu_estimator: str = 'MCAR_mean',
        is_global: bool = True
    ):
    """
    Estimate MCAR parameters from the realisation of a MCAR process. Estimate both drift parameter and Levy triplet parameters (b, Sigma).
    Here the Levy drift parameter b corresponds to
        - no truncation t(x) = 0 when the process has finite jump activity, thus mu := E[L_1] = b + \int_{R^d} z F(dz) =: b + f.
        - classic truncation t(x) = 1_{|x|<=1} when the process has infinite jump activity, thus mu := E[L_1] = b + \int_{|z|>1} z F(dz) =: b + f.
    If some parameters are known or want to be fixed, these can be provided as arguments. Can also fix mu or f (e.g. if jump component is assumed symmetric set f=0).
    Use iterative procedure:
        1. Start with b_hat = 0.
        2. Estimate AA_hat using b_hat.
        3. Estimate (b_hat, Sigma_hat) using AA_hat to recover the driving Levy process.
           To estimate Sigma_hat use iterative thresholding as in (Gegler, 2011).
           To estimate b_hat = mu_hat - f_hat we use:
            - for mu_hat can either:
                - use the recovered Levy process slope, i.e. mu_hat = L_t / t
                - use the MCAR mean, i.e. mu_hat = A_hat[p-1] Y.mean()
            - for f_hat use Sigma_hat and thresholding as in (Gegler, 2011).
        4. Iterate 2. and 3. until convergence criterion is met.

    :param Model: Can either be OU or grOU
    :param Y: MCAR process observation, (d, N+1) np.ndarray
    :param P: finer partition over which we observe the MCAR process [0 = s_0, ..., s_N = t], (N+1, ) np.ndarray
    :param Q: coarser partition over which to approximate the integrals [0 = u_0, ..., u_M = t], (M+1, ) np.ndarray
    :param A: graph adjacency matrix, maximum stage of (d, d) np.ndarray
    :param W_C: Edge weights for Covariates
    :param R_L: Maximum stage for each Lag
    :param C: Covariates
    :param epsilon: tolerance level for defining convergence of iterative scheme, float
    :param max_iter: maximum number of iterations, int
    :param b: drift parameter of Levy process, (d, ) np.ndarray
    :param Sigma: covariance matrix of the driving Levy process - only needed if with_cov is True, (d, d) np.ndarray
    :param f: Levy measure component of drift parameter f:= \int_{|z|>1} z F(dz) or \int_{R^d} z F(dz), (d, ) np.ndarray
    :param mu: 'actual' drift of Levy process mu := E[L_1], (d, ) np.ndarray
    :param is_global: whether to return the global estimator, bool
    :return AA_hat: estimated MCAR drift parameters, list of p (d, d) np.ndarray
            b_hat: estimated driving Levy process drift parameter, (d, ) np.ndarray
            Sigma_hat: estimated driving Levy process covariance parameter, (d, ) np.ndarray
    """
    p = len(R_L)
    d = Y.shape[0]
    b_hat = np.zeros(d) if b is None else b
    AA_hat = np.zeros((p, d, d))
    idx = np.arange(d)
    if Model == 'OU':
        theta_hat = estimate_OU_drift(Y, R_L, C, P, Q, b, with_cov=False, is_global=is_global) if AA is None else AA
        AA_hat[:, idx, idx] = theta_hat
    elif Model == 'grOU':
        global_AA_hat = estimate_grCAR_drift(Y, A, W_C, p, R_L, C, P, Q, b_hat) if AA is None else AA
        AA_hat = estimator_to_QQ(global_AA_hat, R_L, C, W_C, A, d)
    else:
        raise ValueError('Model not recognized')
    DeltaL = recover_BDLP(Y, p, P, Q, AA_hat)
    Sigma_hat = estimate_Sigma_L(DeltaL, Q) if Sigma is None else Sigma
    if mu_estimator == 'Levy_slope':
        mu_hat = DeltaL.sum(axis=1) / P[-1] if mu is None else mu
    elif mu_estimator == 'MCAR_mean':
        mu_hat = AA_hat[p-1].dot(Y.mean(axis=1)) if mu is None else mu
    else:
        raise ValueError("mu_estimator must be 'Levy_slope' or 'MCAR_mean'.")
    if jump_activity == 'finite':
        f_hat = estimate_integral_Levy_measure(DeltaL, Q, Sigma_hat, K = lambda x: True, f= lambda x: x) if f is None else f
    elif jump_activity == 'infinite':
        f_hat = estimate_integral_Levy_measure(DeltaL, Q, Sigma_hat, K = lambda x: np.linalg.norm(x) > 1e-3, f= lambda x: x) if f is None else f
    else:
        raise ValueError("jump_activity must be 'finite' or 'infinite'.")
    b_hat = mu_hat - f_hat if b is None else b
    converged = False

    # keep track of parameters to check if we enter a loop
    AAs = [AA_hat]
    bs = [b_hat]
    Sigmas = [Sigma_hat]
    n_iter = 0
    AA_hat_new = np.zeros((p, d, d))

    while not converged:
        n_iter += 1
        if Model == 'OU':
            theta_hat_new = estimate_OU_drift(Y, R_L, C, P, Q, b_hat, with_cov=False, is_global=is_global) if AA is None else AA
            AA_hat_new[:, idx, idx] = theta_hat_new
        elif Model == 'grOU':
            global_AA_hat_new = estimate_grCAR_drift(Y, A, W_C, p, R_L, C, P, Q, b_hat) if AA is None else AA
            AA_hat_new = estimator_to_QQ(global_AA_hat_new, R_L, C, W_C, A, d)
        DeltaL = recover_BDLP(Y, p, P, Q, AA_hat_new)
        Sigma_hat = estimate_Sigma_L(DeltaL, Q) if Sigma is None else Sigma
        if mu_estimator == 'Levy_slope':
            mu_hat = DeltaL.sum(axis=1) / P[-1] if mu is None else mu
        elif mu_estimator == 'MCAR_mean':
            mu_hat = AA_hat_new[p-1].dot(Y.mean(axis=1)) if mu is None else mu
        if jump_activity == 'finite':
            f_hat = estimate_integral_Levy_measure(DeltaL, Q, Sigma_hat, K = lambda x: True, f= lambda x: x) if f is None else f
        elif jump_activity == 'infinite':
            f_hat = estimate_integral_Levy_measure(DeltaL, Q, Sigma_hat, K = lambda x: np.linalg.norm(x) > 1e-3, f= lambda x: x) if f is None else f
        b_hat = mu_hat - f_hat if b is None else b

        converged = np.sum([np.linalg.norm(AA_hat_new[i] - AA_hat[i]) for i in range(p)]) <= epsilon
        AA_hat = AA_hat_new

        if n_iter >= max_iter:
            converged = True
            warnings.warn('maximum number of iterations exceeded.')

        # check if entering a loop
        if any(np.array_equal(np.append(b_hat, [Sigma_hat, *AA_hat]), np.append(b_, [Sigma_, *AA_])) for b_, Sigma_, AA_ in zip(bs, Sigmas, AAs)):
            converged = True
            # take average of loop values
            index = np.argmax(np.array([np.array_equal(np.append(b_hat, [Sigma_hat, *AA_hat]), np.append(b_, [Sigma_, *AA_])) for b_, Sigma_, AA_ in zip(bs, Sigmas, AAs)]))
            AA_hat = [np.mean(np.array(AAs)[index:, i, :, :], axis = 0) for i in range(p)]
            b_hat = np.mean(np.array(bs)[index:, :], axis = 0)
            Sigma_hat = np.mean(np.array(Sigmas)[index:, :, :], axis = 0)
        else:
            AAs.append(AA_hat)
            bs.append(b_hat)
            Sigmas.append(Sigma_hat)
    return AA_hat, (b_hat, Sigma_hat)

def compute_H_HH(Y: np.ndarray, p: int, P: np.ndarray, Q: np.ndarray, b: np.ndarray, nu: np.ndarray = None, Sigma_inv: np.ndarray = None):
    # get dimension
    d = Y.shape[0]

    # approximate the state space representation by finite differences over P
    X = state_space(Y, p, P)

    # approximate integrals on Q
    if not all([u in P for u in Q]):
        raise ValueError('P must be refinement of Q.')
    indices = [False] * len(P)
    j = 0
    for i, s in enumerate(P):
        if s == Q[j]:
            indices[i] = True
            j += 1
    X = X[:, indices]
    x0 = X[:, 0]

    # get P_0-continuous martingale part of last d entries of state space representation, i.e. of the p-1-th derivative of MCAR realisation
    DY = np.diff(X[-d:, :] - x0[-d:].reshape(-1, 1) - b.reshape(-1, 1) * Q.reshape(1, -1))

    # if nu is not provided choose from data
    if nu is None:
        nu = np.zeros_like(DY)
        for i in range(d):
            nu[i, :] = choose_nu(DY[i:i+1, :], Q)

    # get thresholded increments
    DY_thresh = DY * (np.abs(DY) < nu)

    # compute H
    H = np.zeros(p*d**2)
    for k in range(p):
        for i in range(d):
            integral = np.zeros((d,))
            # (i+1)-th entry of D^{p-k}Y
            integrand = X[-(k+1)*d+i, :]
            for j in range(d):
                # left-hand Riemann sum (converges in L2 to Ito Integral)
                integral[j] = np.sum(integrand[:-1] * DY_thresh[j, :])
            H[k*d**2+i*d:k*d**2+(i+1)*d] = - np.matmul(Sigma_inv, integral).reshape(-1)

    # compute [H]
    HH = np.zeros((p*d**2, p*d**2))
    for k in range(p):
        for i in range(d):
            for l in range(p):
                for j in range(d):
                    # (i+1)-th entry of D^{p-k}Y
                    integrand_one = X[-(k+1)*d+i, :]
                    # (j+1)-th entry of D^{p-l}Y
                    integrand_two = X[-(l+1)*d+j, :]
                    # left-hand Riemann sum (converges a.s. to Riemann Integral, for any choice of mid-point)
                    integral = np.sum(integrand_one[:-1] * integrand_two[:-1] * np.diff(Q))
                    HH[k*d**2+i*d:k*d**2+(i+1)*d, l*d**2+j*d:l*d**2+(j+1)*d] = Sigma_inv * integral
    return H, HH

def inv_vectorize_AA(vec_AA: np.ndarray, p: int) -> List:
    d = int(np.sqrt(len(vec_AA) // p))
    AA = []
    for i in range(p):
        AA.append(vec_AA[i*d**2:(i+1)*d**2].reshape((d, d)).T)
    return AA

def estimator_to_QQ(V: np.ndarray, R_L: np.ndarray, C: int, W_C, Adj, K: int) -> np.ndarray:
    alphas = []
    betas = []
    idx = 0
    L = len(R_L)

    for l in range(L):
        # extract alpha_l
        alphas.append(V[idx])
        idx += 1

        # extract beta_l block
        n_beta = C * R_L[l]
        betas.append(V[idx:idx + n_beta])
        idx += n_beta

    W = [[np.multiply(W_C[c], Adj[stage]) for c in range(C) for stage in range(R_i)] for R_i in R_L]
    Q = []
    l = 0
    while l < L:
        Q.append(np.diag([alphas[l]] * K) + np.hstack(W[l]) @ np.kron(betas[l], np.eye(K)).T)
        l += 1
    return np.array(Q)