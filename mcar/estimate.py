import numpy as np
from typing import Callable, List
import warnings

def state_space(Y: np.ndarray, p: int, P: np.ndarray) -> np.ndarray:
    """
    Construct state-space representation of process from observation Y up to order p-1 via finite differences.
    :param Y: MCAR process observation, (d, N+1) np.ndarray
    :param p: state space order, int
    :param P: partition over which to simulate the MCAR process [0=t_0, t_1, ..., t_N = T], (N+1, ) np.ndarray
    :return X: state space representation (first d entries are Y), (pd, N+1) np.ndarray
    """
    d, N = Y.shape
    X = np.zeros((p * d, N))
    X[:d, :] = Y

    # Differentiate p-1 times to get state space representation
    for i in range(1, p):
        X[d * i:d * (i + 1), :-1] = np.diff(X[d * (i - 1):d * i, :], axis=1) / (P[1:] - P[:-1])
    return X

def estimate_grOU_drift(Y: np.ndarray, A: np.ndarray, W_C: List, R_L: List, C: int, P: np.ndarray, Q: np.ndarray, b: np.ndarray,
                         gamma: float, nu: np.ndarray = None, Sigma: np.ndarray = None, with_cov: bool = False,
                        is_global: bool = False):
    """
    Estimate grOU drift parameters from the realization of a grOU process with driving Levy triplet (b, Sigma, F).
    Here the Levy drift parameter b corresponds to no truncation t(x) = 0 when the process has finite jump activity, thus E[L_1] = b + \int_{R^d} z F(dz).
    :param Y: grOU process observation, (d, N+1) np.ndarray
    :param A: Graph adjacency matrix, maximum stage of (d, d) np.ndarray
    :param W_C: Edge weights for Covariates, list of np.ndarray
    :param R_L: Maximum stage for each Lag, list of int
    :param C: Covariates, int
    :param P: Finer partition over which we observe the GrCAR process [0 = s_0, ..., s_N = t], (N+1, ) np.ndarray
    :param Q: Coarser partition over which to approximate the integrals [0 = u_0, ..., u_M = t], (M+1, ) np.ndarray
    :param b: Drift of Levy process, (d, ) np.ndarray
    :param gamma: hyperparameter for defining the critical region > 1, float
    :param nu: Thresholding sequence, (d, M) np.ndarray
    :param Sigma: Covariance matrix of the driving Levy process, (d, d) np.ndarray
    :param with_cov: whether to return the vectorized estimator with its estimated covariance, bool
    :param is_global: whether to return the global estimator, bool
    :return if with_cov is True:
                vec_theta_hat: vectorized estimated grCAR parameters, (p + sum(R_L) * C, ) np.ndarray
                KK: estimated covariance of the estimated grCAR parameters, (p + sum(R_L) * C, p + sum(R_L) * C) np.ndarray
    """
    p = len(R_L)
    d = Y.shape[0]

    if not all([u in P for u in Q]):
        raise ValueError('P must be refinement of Q.')
    indices = np.isin(P, Q)

    X = state_space(Y, p, P)
    X = X[:, indices]
    x0 = X[:, 0]

    if with_cov and Sigma is not None:
        # Compute inverse of Sigma
        Sigma_inv = np.linalg.pinv(Sigma)
    elif with_cov and Sigma is None:
        # Estimate Sigma from data
        global_QQ_hat = estimate_grOU_drift(Y, A, W_C, R_L, C, P, Q, b, gamma, with_cov=False, is_global=is_global)
        QQ_hat = estimator_to_QQ(global_QQ_hat, R_L, C, W_C, A, d)
        DeltaL = recover_BDLP(Y, p, P, Q, QQ_hat)
        Sigma_hat = estimate_Sigma_L(DeltaL, Q, gamma)
        Sigma_inv = np.linalg.pinv(Sigma_hat)
    else:
        # No need to use Sigma if cov is not needed
        Sigma_inv = np.eye(d)

    # Get P_0-continuous martingale part of last d entries of state space representation, i.e. of the p-1-th derivative of grOU realisation
    DY = np.diff(X[-d:, :] - x0[-d:].reshape(-1, 1) - b.reshape(-1, 1) * Q.reshape(1, -1))

    # If nu is not provided choose from data
    if nu is None:
        nu = np.vstack([choose_nu(DY[i:i + 1, :], Q, gamma) for i in range(d)])

    # Get thresholded increments
    DY_thresh = DY * (np.abs(DY) < nu)

    if is_global:
        # Compute K
        K = np.zeros(p + sum(R_L) * C)
        idx = 0
        for k in range(p):
            integral = np.zeros((d, d))
            if k == 0:
                integrand = X[-d:, :]
            else:
                integrand = X[-d - k * d:-k * d, :]
            for i in range(d):
                for j in range(d):
                    # Left-hand Riemann sum (converges in L2 to Ito Integral)
                    integral[i, j] = np.sum(integrand[i, :-1] * DY_thresh[j, :])

            K[idx] = -np.sum(np.multiply(Sigma_inv, integral))
            for c in range(C):
                for r in range(R_L[k]):
                    K[idx + c * R_L[k] + r + 1] = -np.sum(np.multiply(np.matmul(np.multiply(W_C[c], A[r]).T, Sigma_inv), integral))
            idx += C * R_L[k] + 1

        # Compute [K]
        KK = np.zeros((p + sum(R_L) * C, p + sum(R_L) * C))
        idx_k = 0
        for k in range(p):
            idx_l = 0
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
                        # Left-hand Riemann sum (converges a.s. to Riemann Integral, for any choice of mid-point)
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

        # Return theta_hat
        vec_theta_hat = np.linalg.solve(KK, K)
        vec_theta_hat_ext = np.zeros(p * d + sum(R_L) * C)

        idx_old = 0
        idx_new = 0
        for k in range(p):
            # Number of local scalars at lag k
            Lk = int(C * R_L[k])

            # 1) Duplicate baseline_k d times
            vec_theta_hat_ext[idx_new: idx_new + d] = vec_theta_hat[idx_old]

            # 2) Copy locals for lag k
            vec_theta_hat_ext[idx_new + d: idx_new + d + Lk] = vec_theta_hat[idx_old + 1: idx_old + 1 + Lk]

            # Advance indices
            idx_old += 1 + Lk
            idx_new += d + Lk
        vec_theta_hat = vec_theta_hat_ext

    else:
        # Compute K
        K = np.zeros(p * d + sum(R_L) * C)
        idx = 0
        for k in range(p):
            if k == 0:
                integrand = X[-d:, :]
            else:
                integrand = X[-d - k * d:-k * d, :]
            integral_ = integrand[:, :-1] @ DY_thresh.T
            K[idx:idx + d] = -np.diag(Sigma_inv @ integral_.T)
            base_offset = idx + d
            for c in range(C):
                for r in range(R_L[k]):
                    K[base_offset + c * R_L[k] + r] = -np.sum(np.multiply(np.matmul(np.multiply(W_C[c], A[r]).T, Sigma_inv), integral_))
            idx += C * R_L[k] + d

        # Compute [K]
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
                integral = (integrand_one[:, :-1] * (Q[1:] - Q[:-1])) @ integrand_two[:, :-1].T
                KK[idx_k: idx_k + d, idx_l: idx_l + d] = np.multiply(Sigma_inv, integral)
                for c in range(C):
                    for r in range(R_L[k]):
                        KK[idx_k + c * R_L[k] + r + d, idx_l: idx_l + d] = np.sum(
                            np.multiply(np.matmul(np.multiply(W_C[c], A[r]).T, Sigma_inv), integral), axis=0)
                for e in range(C):
                    for s in range(R_L[l]):
                        KK[idx_k: idx_k + d, idx_l + e * R_L[l] + s + d] = np.sum(
                            np.multiply(np.matmul(Sigma_inv, np.multiply(W_C[e], A[s])), integral), axis=1)
                for c in range(C):
                    for r in range(R_L[k]):
                        for e in range(C):
                            for s in range(R_L[l]):
                                KK[idx_k + c * R_L[k] + r + d, idx_l + e * R_L[l] + s + d] = np.sum(np.multiply(
                                    np.matmul(np.matmul(np.multiply(W_C[c], A[r]).T, Sigma_inv),
                                              np.multiply(W_C[e], A[s])), integral))
                idx_l += C * R_L[l] + d
            idx_k += C * R_L[k] + d
        vec_theta_hat = np.linalg.solve(KK, K)
            
    if with_cov and Sigma is not None:
        return vec_theta_hat, KK, K
    elif with_cov and Sigma is None:
        return vec_theta_hat, KK, Sigma_hat
    else:
        return vec_theta_hat

def estimate_OU_drift(Y: np.ndarray, R_L: List, C: int, P: np.ndarray, Q: np.ndarray, b: np.ndarray, gamma: float,
                      nu: np.ndarray = None, Sigma: np.ndarray = None, with_cov: bool = False, is_global: bool = False,
                      lam = 1e-4):
    """
    Estimate OU drift parameters from the realization of a process with driving Levy triplet (b, Sigma, F). Here the Levy drift parameter b corresponds to
    no truncation t(x) = 0 when the process has finite jump activity, thus E[L_1] = b + \int_{R^d} z F(dz).
    :param Y: Process observation, (d, N+1) np.ndarray
    :param R_L: Maximum stage for each Lag, List
    :param C: Covariates, int
    :param P: finer partition over which we observe the GrCAR process [0 = s_0, ..., s_N = t], (N+1, ) np.ndarray
    :param Q: coarser partition over which to approximate the integrals [0 = u_0, ..., u_M = t], (M+1, ) np.ndarray
    :param b: drift of Levy process, (d, ) np.ndarray
    :param gamma: hyperparameter for defining the critical region >1, float
    :param nu: thresholding sequence, (d, M) np.ndarray
    :param Sigma: covariance matrix of the driving Levy process, (d, d) np.ndarray
    :param with_cov: whether to return the vectorized estimator with its estimated covariance, bool
    :param is_global: whether to return the global estimator, bool
    :param lam: Regularization parameter, float
    :return if with_cov is True:
                vec_theta_hat: vectorized estimated OU parameters
                KK: estimated covariance of the estimated OU parameters
    """
    p = len(R_L)
    d = Y.shape[0]

    if not all([u in P for u in Q]):
        raise ValueError('P must be refinement of Q.')
    indices = np.isin(P, Q)

    X = state_space(Y, p, P)
    X = X[:, indices]
    x0 = X[:, 0]

    if with_cov and Sigma is not None:
        # Compute inverse of Sigma
        Sigma_inv = np.linalg.pinv(Sigma)
    elif with_cov and Sigma is None:
        # Estimate Sigma from data
        theta_hat = estimate_OU_drift(Y, R_L, C, P, Q, b, gamma, with_cov=False, is_global=is_global)
        QQ_hat = np.zeros((p, d, d))
        idx = np.arange(d)
        QQ_hat[:, idx, idx] = theta_hat
        DeltaL = recover_BDLP(Y, p, P, Q, QQ_hat)
        Sigma_hat = estimate_Sigma_L(DeltaL, Q, gamma)
        Sigma_inv = np.linalg.pinv(Sigma_hat)
    else:
        # No need to use Sigma if cov is not needed
        Sigma_inv = np.eye(d)

    # Get P_0-continuous martingale part of last d entries of state space representation, i.e. of the p-1-th derivative of OU realisation
    DY = np.diff(X[-d:, :] - x0[-d:].reshape(-1, 1) - b.reshape(-1, 1) * Q.reshape(1, -1))

    # If nu is not provided choose from data
    if nu is None:
        nu = np.vstack([choose_nu(DY[i:i + 1, :], Q, gamma) for i in range(d)])

    # Get thresholded increments
    DY_thresh = DY * (np.abs(DY) < nu)

    if is_global:
        # Compute K
        K = np.zeros(p)
        for k in range(p):
            integral = np.zeros((d, d))
            if k == 0:
                integrand = X[-d:, :]
            else:
                integrand = X[-d - k * d:-k * d, :]
            for i in range(d):
                for j in range(d):
                    # Left-hand Riemann sum (converges in L2 to Ito Integral)
                    integral[i, j] = np.sum(integrand[i, :-1] * DY_thresh[j, :])

            K[k] = -np.sum(np.multiply(Sigma_inv, integral))

        # Compute [K]
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

        # Return theta_hat
        vec_theta_hat = np.repeat(np.linalg.solve(KK + lam * np.eye(KK.shape[0]), K)[:, np.newaxis], d, axis=1)

    else:
        # Build lag blocks:
        # X_blocks[k] has shape (d, len(Q))
        # and corresponds to rows X[-(k+1)*d : -k*d] in forward lag order
        X_blocks = np.stack([X[-(k + 1) * d: -k * d if k > 0 else None, :] for k in range(p)], axis=0)

        # ----------------------------------------------------------
        # Compute K
        # ----------------------------------------------------------
        K = np.zeros(p * d)

        for k in range(p):
            # integral_mat[i, j] = sum_t X_blocks[k, i, t] * DY_thresh[j, t]
            integral_mat = X_blocks[k, :, :-1] @ DY_thresh.T

            # original:
            # K[k*d + i] = -(Sigma_inv @ integral_row_i)[i]
            K[k * d:(k + 1) * d] = -np.diag(Sigma_inv @ integral_mat.T)

        # ----------------------------------------------------------
        # Compute [K]
        # ----------------------------------------------------------
        KK = np.zeros((p * d, p * d))

        for k in range(p):
            A_k = X_blocks[k, :, :-1] * np.diff(Q)

            for l in range(p):
                B_l = X_blocks[l, :, :-1]

                # integral_mat[i, j] = sum_t integrand_one[i,t] * integrand_two[j,t] * dQ[t]
                integral_mat = A_k @ B_l.T

                KK[k * d:(k + 1) * d, l * d:(l + 1) * d] = Sigma_inv * integral_mat

        vec_theta_hat = np.linalg.solve(KK + lam * np.eye(KK.shape[0]), K).reshape(p, d)

    if with_cov and Sigma is not None:
        return vec_theta_hat, KK, K
    elif with_cov and Sigma is None:
        return vec_theta_hat, KK, Sigma_hat
    else:
        return vec_theta_hat

def estimate_MCAR_drift(Y: np.ndarray, R_L: List, P: np.ndarray, Q: np.ndarray, b: np.ndarray, gamma: float, nu: np.ndarray = None,
                        with_cov: bool = False, Sigma: np.ndarray = None):
    """
    Estimate MCAR drift parameters from the realisation of a MCAR process with driving Levy triplet (b, Sigma, F).
    Here the Levy drift parameter b corresponds to no truncation t(x) = 0 when the process has finite jump activity, thus E[L_1] = b + \int_{R^d} z F(dz).

    :param Y: MCAR process observation, (d, N+1) np.ndarray
    :param R_L: Maximum stage for each Lag, List
    :param P: finer partition over which we observe the MCAR process [0 = s_0, ..., s_N = t], (N+1, ) np.ndarray
    :param Q: coarser partition over which to approximate the integrals [0 = u_0, ..., u_M = T], (M+1, ) np.ndarray
    :param b: drift parameter of Levy process, (d, ) np.ndarray
    :param gamma: hyperparameter for defining the critical region >1, float
    :param nu: thresholding sequence, (d, M) np.ndarray
    :param with_cov: whether to return the vectorized estimator with its estimated covariance, bool
    :param Sigma: covariance matrix of the driving Levy process - only needed if with_cov is True, (d, d) np.ndarray
    :return if with_cov is True:
                vec_QQ_hat: vectorized estimated MCAR drift parameters, (pd ** 2, ) np.ndarray
                KK: estimated covariance of the estimated MCAR parameters, (pd ** 2, pd ** 2) np.ndarray
            else:
                QQ_hat: estimated MCAR drift parameters, list of p (d, d) np.ndarray
    """
    p = len(R_L)
    d = Y.shape[0]

    if not all([u in P for u in Q]):
        raise ValueError('P must be refinement of Q.')
    indices = np.isin(P, Q)

    X = state_space(Y, p, P)
    X = X[:, indices]
    x0 = X[:, 0]

    if with_cov and Sigma is not None:
        # Compute inverse of Sigma
        Sigma_inv = np.linalg.pinv(Sigma)
    elif with_cov and Sigma is None:
        # Estimate Sigma from data
        QQ_hat = estimate_MCAR_drift(Y, R_L, P, Q, b, gamma, with_cov=False)
        DeltaL = recover_BDLP(Y, p, P, Q, QQ_hat)
        Sigma_hat = estimate_Sigma_L(DeltaL, Q, gamma)
        Sigma_inv = np.linalg.pinv(Sigma_hat)
    else:
        # No need to use Sigma if cov is not needed
        Sigma_inv = np.eye(d)

    # Get P_0-continuous martingale part of last d entries of state space representation, i.e. of the p-1-th derivative of OU realisation
    DY = np.diff(X[-d:, :] - x0[-d:].reshape(-1, 1) - b.reshape(-1, 1) * Q.reshape(1, -1))

    # If nu is not provided choose from data
    if nu is None:
        nu = np.vstack([choose_nu(DY[i:i + 1, :], Q, gamma) for i in range(d)])

    # Get thresholded increments
    DY_thresh = DY * (np.abs(DY) < nu)

    # Build lag blocks:
    # X_blocks[k] has shape (d, len(Q))
    X_blocks = np.stack([X[-(k + 1) * d: -k * d if k > 0 else None, :] for k in range(p)], axis=0)  # shape (p, d, len(Q))

    # ----------------------------------------------------------
    # Compute K
    # ----------------------------------------------------------
    # G[k, i, j] = sum_t X_blocks[k, i, t] * DY_thresh[j, t]
    G = X_blocks[:, :, :-1] @ DY_thresh.T  # shape (p, d, d)

    # Apply Sigma_inv on the left of each row-vector-as-column:
    # for fixed (k,i), original block is -Sigma_inv @ G[k,i,:]
    K = -(G @ Sigma_inv.T).reshape(p * d * d)

    # ----------------------------------------------------------
    # Compute [K]
    # ----------------------------------------------------------
    # H[k, i, l, j] = sum_t X_blocks[k, i, t] * X_blocks[l, j, t] * dQ[t]
    H = np.einsum('kit,ljt,t->kilj', X_blocks[:, :, :-1], X_blocks[:, :, :-1], np.diff(Q))
    # shape (p, d, p, d)

    # Each scalar H[k,i,l,j] multiplies the full matrix Sigma_inv
    KK = H[:, :, :, :, None, None] * Sigma_inv[None, None, None, None, :, :]
    # shape (p, d, p, d, d, d)

    # Rearrange into 2D block matrix:
    KK = KK.transpose(0, 1, 4, 2, 3, 5).reshape(p * d * d, p * d * d)

    vec_QQ_hat = np.linalg.solve(KK, K)

    if with_cov and Sigma is not None:
        return vec_QQ_hat, KK, K
    elif with_cov and Sigma is None:
        return vec_QQ_hat, KK, Sigma_hat
    else:
        QQ_hat = inv_vectorize_AA(vec_QQ_hat, p)
        return QQ_hat

def inv_vectorize_AA(vec_AA: np.ndarray, p: int):
    d = int(np.sqrt(len(vec_AA) // p))
    AA = []
    for i in range(p):
        AA.append(vec_AA[i*d**2:(i+1)*d**2].reshape((d, d)).T)
    return np.array(AA)

def recover_BDLP(Y: np.ndarray, p: int, P: np.ndarray, Q: np.ndarray, QQ: np.ndarray):
    """
    Recover the increments of the background driving Levy process L over coarser partition Q from MCAR observation Y over finer partition P with parameters QQ.
    :param Y: MCAR process observation, (d, N+1) np.ndarray
    :param p: MCAR parameter, int
    :param P: finer partition over which we observe the MCAR process [0 = s_0, ..., s_N = t], (N+1, ) np.ndarray
    :param Q: coarser partition over which we recover the Levy increments [0 = u_0, ..., u_M = T], (M+1, ) np.ndarray
    :param QQ: MCAR parameters, list of p (d, d) np.ndarray
    :return DeltaL_Q: increments of the background driving Levy process L on the partition Q, (d, M) np.ndarray
    """
    d = Y.shape[0]
    X_P = state_space(Y, p, P)
    Delta_P = np.diff(P)

    # Find elements of Q in P
    indices = [False] * len(P)
    j = 0
    for i, s in enumerate(P):
        if s == Q[j]:
            indices[i] = True
            j += 1

    # Restrict X to Q
    X_Q = X_P[:, indices]
    DeltaX_Q = np.diff(X_Q, axis = 1)

    # Approximate drift on P
    drift_P = np.zeros((d, len(Delta_P)))
    for j in range(p):
        DjY = X_P[j * d:(j + 1) * d, :-1]
        drift_P += QQ[p - j - 1].dot(DjY) * Delta_P.T

    # Coarsen drift to Q
    cum_drift = np.cumsum(drift_P, axis=1)
    drift_Q = np.diff(np.concatenate([np.zeros((d, 1)), cum_drift[:, indices[:-1]]], axis=1), axis=1)

    # Return L on Q
    DeltaL_Q = DeltaX_Q[-d:, :] + drift_Q

    return DeltaL_Q

def disentangle_BM(DeltaL: np.ndarray, Q: np.ndarray, Sigma: np.ndarray, gamma: float):
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
    # Identify critical region x^T Sigma_inv x <= beta*2*Delta*log(N)
    DeltaQ = np.diff(Q)
    N = len(DeltaQ)
    DeltaX = DeltaL
    try:
        subset = np.einsum('ki,ki->i', DeltaX, np.linalg.pinv(Sigma).dot(DeltaX)) <= 2 * gamma * DeltaQ * np.log(N)
    except:
        eps = 1e-6
        Sigma_reg = Sigma + eps * np.eye(Sigma.shape[0])
        subset = np.einsum('ki,ki->i', DeltaX, np.linalg.pinv(Sigma_reg).dot(DeltaX)) <= 2 * gamma * DeltaQ * np.log(N)
    # Return Brownian increments
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

def estimate_Sigma_L(DeltaL: np.ndarray, Q: np.ndarray, gamma: float, epsilon: float = 0.01):
    """
    Estimate the covariance matrix Sigma of the Brownian component from (irregularly spaced) Levy increments. Use the iterative approach in (Gegler, 2011) Section 4.
    :param DeltaL: increments of Levy process on the partition Q, (d, N) np.ndarray
    :param Q: partition, [0 = u_1, ..., u_N = T], (N+1, ) np.ndarray
    :param gamma: hyperparameter for defining the critical region >1, float
    :param epsilon: tolerance level for defining convergence of iterative scheme, float
    :return Sigma_hat: estimated covariance of the Brownian component, (d, d) np.ndarray
    """
    # Initialize procedure
    DeltaQ_W = np.diff(Q)
    DeltaW = DeltaL
    Sigma_hat_new = estimate_Sigma(DeltaW, DeltaQ_W)
    converged = False

    # Keep track of Sigmas to see if enter a loop
    Sigmas = [Sigma_hat_new]
    while not converged:
        Sigma_hat_old = Sigma_hat_new
        DeltaW, DeltaQ_W = disentangle_BM(DeltaL, Q, Sigma_hat_old, gamma)
        Sigma_hat_new = estimate_Sigma(DeltaW, DeltaQ_W)
        converged = np.linalg.norm(Sigma_hat_new - Sigma_hat_old) <= epsilon

        # Check if entering a loop
        if any(np.array_equal(Sigma_hat_new, Sigma) for Sigma in Sigmas):
            converged = True
            # Take average of loop values
            index = np.argmax(np.array([np.array_equal(Sigma_hat_new, Sigma) for Sigma in Sigmas]))
            Sigma_hat_new = np.mean(np.array(Sigmas)[index:, :, :], axis = 0)
        else:
            Sigmas.append(Sigma_hat_new)
    return Sigma_hat_new

def estimate_integral_Levy_measure(DeltaL: np.ndarray, Q: np.ndarray, gamma: float, Sigma: np.ndarray = None, epsilon: float = 0.01, K: Callable = lambda x: np.linalg.norm(x) > 1e-3, f: Callable = lambda x: x):
    """
    Estimate the functional f(F) = \int_K f(z) F(dz) where f: R^d -> R^n of the Levy measure F from (irregularly spaced) Levy increments DeltaL.
    Theoretical guarantees of convergence are given in (Gegler, 2011) for functions of the form f(z) = (z^T A z)^l for matrix A and power l.
    If Sigma is known use the rejection-region based estimator in (Gegler, 2011) Section 2, otherwise use the iterative approach in (Gegler, 2011) Section 4.
    :param DeltaL: increments of Levy process on the partition Q, (d, N) np.ndarray
    :param Q: partition, [0 = u_1, ..., u_N = T], (N+1, ) np.ndarray
    :param gamma: hyperparameter for defining the critical region > 1, float
    :param Sigma: covariance of Brownian component of L, (d, d) np.ndarray
    :param epsilon: tolerance level for defining convergence of iterative scheme, float
    :param K: integration region in functional, function mapping (d, ) np.ndarray to bool
    :param f: integrand function, function mapping (d, ) np.ndarray to float
    :return f_hat: estimated functional f of the Levy measure, float
    """
    if Sigma is None:
        Sigma = estimate_Sigma_L(DeltaL, Q, gamma, epsilon)

    d = Sigma.shape[0]
    DeltaQ = np.diff(Q)
    N = len(DeltaQ)

    # Identify critical region x^T Sigma_inv x > beta*2*Delta*log(N)
    subset = np.logical_and(
        np.einsum('ki,ki->i', np.linalg.pinv(Sigma).dot(DeltaL), DeltaL) > 2 * gamma * DeltaQ * np.log(N),
        [K(DeltaL[:, i]) for i in range(DeltaL.shape[1])]
    )
    DeltaZ = DeltaL[:, subset]
    if DeltaZ.shape[1] < 1:
        f_hat = np.zeros(f(np.zeros(d)).shape) if isinstance(f(np.zeros(d)), np.ndarray) else 0
    else:
        # Check this is OK also for non-uniform partition
        f_hat = np.sum(np.array([f(DeltaZ[:, i]) for i in range(DeltaZ.shape[1])]), axis=0) / (Q[-1] - Q[0])
    return f_hat

def choose_nu(DeltaL: np.ndarray, Q: np.ndarray, gamma: float, epsilon: float = 0.01, Sigma: np.ndarray = None):
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
    Sigma_hat = estimate_Sigma_L(DeltaL, Q, gamma, epsilon) if Sigma is None else Sigma
    nu = np.sqrt(DeltaQ * 2 * gamma * np.log(len(DeltaQ)) * Sigma_hat)
    return nu

def estimate_MCAR(
        Model: str,
        Y: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        A: np.ndarray,
        W_C: List,
        R_L: List,
        C: int,
        gamma: float,
        epsilon: float = 0.01,
        max_iter: int = 200,
        QQ: np.ndarray = None,
        b: np.ndarray = None,
        Sigma: np.ndarray = None,
        f: np.ndarray = None,
        mu: np.ndarray = None,
        mu_estimator: str = 'MCAR_mean',
        is_global: bool = False
    ):
    """
    Estimate MCAR parameters from the realization of a MCAR process. Estimate both drift parameter and Levy triplet parameters (b, Sigma).
    Here the Levy drift parameter b corresponds to no truncation t(x) = 0 when the process has finite jump activity, thus mu := E[L_1] = b + \int_{R^d} z F(dz) =: b + f.
    If some parameters are known or want to be fixed, these can be provided as arguments. Can also fix mu or f (e.g. if jump component is assumed symmetric set f=0).
    Use iterative procedure:
        1. Start with b_hat = 0.
        2. Estimate QQ_hat using b_hat.
        3. Estimate (b_hat, Sigma_hat) using QQ_hat to recover the driving Levy process.
           To estimate Sigma_hat use iterative thresholding as in (Gegler, 2011).
           To estimate b_hat = mu_hat - f_hat we use:
            - for mu_hat can either:
                - use the recovered Levy process slope, i.e. mu_hat = L_t / t
                - use the MCAR mean, i.e. mu_hat = QQ_hat[p-1] Y.mean()
            - for f_hat use Sigma_hat and thresholding as in (Gegler, 2011).
        4. Iterate 2. and 3. until convergence criterion is met.

    :param Model: Can either be OU or grOU, str
    :param Y: MCAR process observation, (d, N+1) np.ndarray
    :param P: finer partition over which we observe the MCAR process [0 = s_0, ..., s_N = t], (N+1, ) np.ndarray
    :param Q: coarser partition over which to approximate the integrals [0 = u_0, ..., u_M = t], (M+1, ) np.ndarray
    :param A: graph adjacency matrix, maximum stage of (d, d) np.ndarray
    :param W_C: Edge weights for Covariates, List
    :param R_L: Maximum stage for each Lag, List
    :param C: Covariates, int
    :param gamma: hyperparameter for defining the critical region >1, float
    :param epsilon: tolerance level for defining convergence of iterative scheme, float
    :param max_iter: maximum number of iterations, int
    :param QQ: MCAR parameters, list of p (d, d) np.ndarray
    :param b: drift parameter of Levy process, (d, ) np.ndarray
    :param Sigma: covariance matrix of the driving Levy process - only needed if with_cov is True, (d, d) np.ndarray
    :param f: Levy measure component of drift parameter f:= \int_{|z|>1} z F(dz) or \int_{R^d} z F(dz), (d, ) np.ndarray
    :param mu: 'actual' drift of Levy process mu := E[L_1], (d, ) np.ndarray
    :param is_global: whether to return the global estimator, bool
    :return QQ_hat: estimated MCAR drift parameters, list of p (d, d) np.ndarray
            b_hat: estimated driving Levy process drift parameter, (d, ) np.ndarray
            Sigma_hat: estimated driving Levy process covariance parameter, (d, ) np.ndarray
    """
    p = len(R_L)
    d = Y.shape[0]
    b_hat = np.zeros(d) if b is None else b
    QQ_hat = np.zeros((p, d, d))
    idx = np.arange(d)
    if Model == 'OU':
        theta_hat = estimate_OU_drift(Y, R_L, C, P, Q, b_hat, gamma, with_cov=False, is_global=is_global) if QQ is None else QQ
        QQ_hat[:, idx, idx] = theta_hat
    elif Model == 'grOU':
        global_QQ_hat = estimate_grOU_drift(Y, A, W_C, R_L, C, P, Q, b_hat, gamma) if QQ is None else QQ
        QQ_hat = estimator_to_QQ(global_QQ_hat, R_L, C, W_C, A, d)
    elif Model == 'MCAR':
        QQ_hat = estimate_MCAR_drift(Y, R_L, P, Q, b_hat, gamma) if QQ is None else QQ
    else:
        raise ValueError('Model not recognized')
    DeltaL = recover_BDLP(Y, p, P, Q, QQ_hat)
    Sigma_hat = estimate_Sigma_L(DeltaL, Q, gamma) if Sigma is None else Sigma
    if mu_estimator == 'Levy_slope':
        mu_hat = DeltaL.sum(axis=1) / P[-1] if mu is None else mu
    elif mu_estimator == 'MCAR_mean':
        mu_hat = QQ_hat[p-1].dot(Y.mean(axis=1)) if mu is None else mu
    else:
        raise ValueError("mu_estimator must be 'Levy_slope' or 'MCAR_mean'.")

    f_hat = estimate_integral_Levy_measure(DeltaL, Q, gamma, Sigma_hat, K = lambda x: True, f= lambda x: x) if f is None else f
    b_hat = mu_hat - f_hat if b is None else b
    converged = False

    # Keep track of parameters to check if we enter a loop
    QQs = [QQ_hat]
    bs = [b_hat]
    Sigmas = [Sigma_hat]
    n_iter = 0

    while not converged:
        QQ_hat_new = np.zeros((p, d, d))
        n_iter += 1
        if Model == 'OU':
            theta_hat_new = estimate_OU_drift(Y, R_L, C, P, Q, b_hat, gamma, with_cov=False, is_global=is_global) if QQ is None else QQ
            QQ_hat_new[:, idx, idx] = theta_hat_new
        elif Model == 'grOU':
            global_QQ_hat_new = estimate_grOU_drift(Y, A, W_C, R_L, C, P, Q, b_hat, gamma) if QQ is None else QQ
            QQ_hat_new = estimator_to_QQ(global_QQ_hat_new, R_L, C, W_C, A, d)
        elif Model == "MCAR":
            QQ_hat_new = estimate_MCAR_drift(Y, R_L, P, Q, b_hat, gamma) if QQ is None else QQ
        DeltaL = recover_BDLP(Y, p, P, Q, QQ_hat_new)
        Sigma_hat = estimate_Sigma_L(DeltaL, Q, gamma) if Sigma is None else Sigma
        if mu_estimator == 'Levy_slope':
            mu_hat = DeltaL.sum(axis=1) / P[-1] if mu is None else mu
        elif mu_estimator == 'MCAR_mean':
            mu_hat = QQ_hat_new[p-1].dot(Y.mean(axis=1)) if mu is None else mu
        f_hat = estimate_integral_Levy_measure(DeltaL, Q, gamma, Sigma_hat, K = lambda x: True, f= lambda x: x) if f is None else f
        b_hat = mu_hat - f_hat if b is None else b

        converged = np.sum([np.linalg.norm(QQ_hat_new[i] - QQ_hat[i]) for i in range(p)]) <= epsilon
        QQ_hat = QQ_hat_new.copy()

        if n_iter >= max_iter:
            converged = True
            warnings.warn('maximum number of iterations exceeded.')

        # Check if entering a loop
        if any(states_close(b_hat, Sigma_hat, QQ_hat, b_, Sigma_, QQ_, tol=1e-6) for b_, Sigma_, QQ_ in zip(bs, Sigmas, QQs)):
            converged = True
            # Take average of loop values
            index = np.argmax(np.array([states_close(b_hat, Sigma_hat, QQ_hat, b_, Sigma_, QQ_, tol=1e-6) for b_, Sigma_, QQ_ in zip(bs, Sigmas, QQs)]))
            QQ_hat = [np.mean(np.array(QQs)[index:, i, :, :], axis = 0) for i in range(p)]
            b_hat = np.mean(np.array(bs)[index:, :], axis = 0)
            Sigma_hat = np.mean(np.array(Sigmas)[index:, :, :], axis = 0)
        else:
            QQs.append(QQ_hat.copy())
            bs.append(b_hat)
            Sigmas.append(Sigma_hat)
    return QQ_hat, (b_hat, Sigma_hat)

def states_close(b1, S1, Q1, b2, S2, Q2, tol=1e-6):
    b_close = np.linalg.norm(b1 - b2) < tol
    S_close = np.linalg.norm(S1 - S2, ord='fro') < tol
    Q_close = sum(np.linalg.norm(Q1[i] - Q2[i], ord='fro') for i in range(len(Q1))) < tol
    return b_close and S_close and Q_close

def estimator_to_QQ(V: np.ndarray, R_L: List, C: int, W_C: List, Adj: np.ndarray, K: int) -> np.ndarray:
    """
    Convert the estimated vectorized parameters to the estimated MCAR parameters
    :param V: Estimated parameters, np.ndarray
    :param R_L: Maximum stage for each Lag, list of int
    :param C: Covariates, int
    :param W_C: Edge weights for Covariates, list of np.ndarray
    :param Adj: Graph adjacency matrix, maximum stage of (d, d) np.ndarray
    :param K: Number of edges, int
    :return: Estimated MCAR parameters, list of p (d, d) np.ndarray
    """
    alphas = []
    betas = []
    idx = 0
    L = len(R_L)

    for l in range(L):
        # Extract alpha_l
        alphas.append(V[idx])
        idx += 1

        # Extract beta_l block
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