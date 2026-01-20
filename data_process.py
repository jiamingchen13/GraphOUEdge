import os
import glob
import math
import re
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from joblib import Parallel, delayed
from numpy.linalg import det
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from multiprocessing import Pool
from itertools import combinations
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR
from graphOUedge import GraphOUEdge
from mcar.estimate import *
from mcar.simulate import *

def multiprocess_orderbooks(TICKER, input_path, output_path, log_path, check_for_processed_data=True):
    """
    Pre-process LOBSTER data into feature-response pairs parallely. The data must be stored in the input_path
    directory as daily message book and order book files. For details of processing steps see description of process_orderbook function.
    A log file is produced tracking:
    - order book files with problems
    - message book files with problems
    - trading days with unusual open - close times
    - trading days with crossed quotes
    :param TICKER: the TICKER to be considered, str
    :param input_path: the path where the order book and message book files are stored, order book files have shape (:, 4*levels):
                       "ASKp1", "ASKs1", "BIDp1",  "BIDs1", ..., "ASKplevels", "ASKslevels", "BIDplevels",  "BIDslevels", str
    :param output_path: the path where we wish to save the processed datasets (as .npz files), str
    :param log_path: the path where processing logs are saved, str
    :param check_for_processed_data: check if processed .npz files already exist in output_path and skip these dates for processing, bool
    :return: saves the aggregated mid-quote price in output_path, as .npz files with numpy attributes
             "orderbook_features" (:, 1).
    """
    csv_file_list = glob.glob(os.path.join(input_path, "*.{}".format("csv")))
    csv_orderbook = [name for name in csv_file_list if "orderbook" in name]
    csv_message = [name for name in csv_file_list if "message" in name]
    csv_orderbook.sort()
    csv_message.sort()

    # do not process already processed data
    if check_for_processed_data:
        npz_file_list = sorted(glob.glob(os.path.join(output_path, "*.{}".format("npz"))))
        processed_dates = [re.search(r'\d{4}-\d{2}-\d{2}', file).group() for file in npz_file_list]
        csv_orderbook = [file for file in csv_orderbook if
                         re.search(r'\d{4}-\d{2}-\d{2}', file).group() not in processed_dates]
        csv_message = [file for file in csv_message if
                       re.search(r'\d{4}-\d{2}-\d{2}', file).group() not in processed_dates]

    # check if exactly half of the files are order book and exactly half are messages
    assert (len(csv_message) == len(csv_orderbook))

    print("started multiprocessing")

    try:
        pool = Pool(os.cpu_count())
        engine = ProcessOrderbook(TICKER, output_path)
        logs = pool.map(engine, csv_orderbook)
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    print("finished multiprocessing")

    with open(log_path + "/processing_logs.txt", "w") as f:
        for log in logs:
            f.write(log + "\n")

    print("please check processing logs.")


class ProcessOrderbook(object):
    """
    Multiprocessing engine for pre-processing LOBSTER data into feature-response npz files.
    """

    def __init__(self, TICKER, output_path):
        self.TICKER = TICKER
        self.output_path = output_path

    def __call__(self, orderbook_name):
        output = process_orderbook(orderbook_name, self.TICKER, self.output_path)
        return output


def process_orderbook(orderbook_name, TICKER, output_path):
    """
    Function carrying out processing of single order book orderbook_name. Features are not normalized.
    The data is treated in the following way:
    - order book states with crossed quotes are removed.
    - each state in the orderbook is time-stamped, with states occurring at the same time collapsed
      onto the last state.
    - the first and last 60 minutes of market activity (inside usual opening times) are dropped.
    :param orderbook_name: order book to process, of shape (T, 4):
            "ASKp1", "ASKs1", "BIDp1",  "BIDs1" str
    :param TICKER: the TICKER to be considered, str
    :param output_path: the path where we wish to save the processed datasets (as .npz files), str
    :return: saves the aggregated mid-quote price in output_path, as .npz files with numpy attributes
             "orderbook_features" (:, 1).
    """
    print(orderbook_name)
    log = ''

    ### load LOBSTER orderbook and message files
    try:
        df_orderbook = pd.read_csv(orderbook_name, header=None)
    except:
        return orderbook_name + ' skipped. Error: failed to read orderbook.'

    # compute the mid-quote price
    df_orderbook.columns = ["ASKp1", "ASKs1", "BIDp1", "BIDs1"]
    df_orderbook.insert(0, "mid", (df_orderbook["ASKp1"] + df_orderbook["BIDp1"]) / 2)

    # get date
    match = re.findall(r"\d{4}-\d{2}-\d{2}", orderbook_name)[-1]
    date = datetime.strptime(match, "%Y-%m-%d")

    # read times from message file. keep a record of problematic files
    message_name = orderbook_name.replace("orderbook", "message")
    try:
        df_message = pd.read_csv(message_name, usecols=[0, 1, 2, 3, 4, 5], header=None)
    except:
        return orderbook_name + ' skipped. Error: failed to read messagebook.'

    # check the two df have the same length
    assert (len(df_message) == len(df_orderbook))

    # add column names to message book
    df_message.columns = ["seconds", "event type", "order ID", "volume", "price", "direction"]

    ### process potential data anomalies

    # 1. remove trading halts
    trading_halts_start = df_message[(df_message["event type"] == 7) & (df_message["price"] == -1)].index
    trading_halts_end = df_message[(df_message["event type"] == 7) & (df_message["price"] == 1)].index
    trading_halts_index = np.array([])
    for halt_start, halt_end in zip(trading_halts_start, trading_halts_end):
        trading_halts_index = np.append(trading_halts_index, df_message.index[
            (df_message.index >= halt_start) & (df_message.index < halt_end)])
    if len(trading_halts_index) > 0:
        for halt_start, halt_end in zip(trading_halts_start, trading_halts_end):
            log = log + 'Warning: trading halt between ' + str(df_message.loc[halt_start, "seconds"]) + ' and ' + str(
                df_message.loc[halt_end, "seconds"]) + ' in ' + orderbook_name + '.\n'
    df_orderbook = df_orderbook.drop(trading_halts_index)
    df_message = df_message.drop(trading_halts_index)

    # 2. remove crossed quotes
    crossed_quotes_index = df_orderbook[(df_orderbook["BIDp1"] > df_orderbook["ASKp1"])].index
    if len(crossed_quotes_index) > 0:
        log = log + 'Warning: ' + str(
            len(crossed_quotes_index)) + ' crossed quotes removed in ' + orderbook_name + '.\n'
    df_orderbook = df_orderbook.drop(crossed_quotes_index)
    df_message = df_message.drop(crossed_quotes_index)

    # add the seconds since midnight column to the order book from the message book
    df_orderbook.insert(0, "seconds", df_message["seconds"])

    # 3. one conceptual event (e.g. limit order modification which is implemented as a cancellation followed
    # by an immediate new arrival, single market order executing against multiple resting limit orders) may
    # appear as multiple rows in the message file, all with the same timestamp.
    # We hence group the order book data by unique timestamps and take the last entry.
    df_orderbook = df_orderbook.groupby(["seconds"]).tail(1)

    # 4. check market opening times for strange values
    market_open = int(df_orderbook["seconds"].iloc[0] / 60) / 60  # open at minute before first transaction
    market_close = (int(df_orderbook["seconds"].iloc[-1] / 60) + 1) / 60  # close at minute after last transaction

    if not (market_open == 9.5 and market_close == 16):
        log = log + 'Warning: unusual opening times in ' + orderbook_name + ': ' + str(market_open) + ' - ' + str(
            market_close) + '.\n'

    # drop values outside of market hours
    df_orderbook = df_orderbook.loc[(df_orderbook["seconds"] >= 34200) &
                                    (df_orderbook["seconds"] <= 57600)]

    # 5. drop first and last 60 minutes of trading using seconds (do this after orderflow computation)
    market_open_seconds = market_open * 60 * 60 + 60 * 60
    market_close_seconds = market_close * 60 * 60 - 60 * 60
    df_orderbook = df_orderbook.loc[(df_orderbook["seconds"] >= market_open_seconds) &
                                    (df_orderbook["seconds"] <= market_close_seconds)]

    # 6. downsample tick-by-tick mid-quote price data into 1-second frequency, keeping only the last tick in each second with forward-fill
    df_orderbook['timestamp'] = pd.to_datetime(df_orderbook['seconds'], unit='s', origin=pd.Timestamp(date))
    df_orderbook = df_orderbook.set_index('timestamp')
    df_orderbook_sec = df_orderbook.resample('1s').last().ffill()

    # save
    output_name = os.path.join(output_path, TICKER + "_" + "data" + "_" + str(date.date()) + ".npz")
    np.savez(output_name, timestamps=df_orderbook_sec.index.astype('datetime64[s]'), mid=df_orderbook_sec['mid'].values)

    return log + orderbook_name + ' completed.'


def process_day_parallel(arg):
    """
    Load all assets for a single day in parallel.
    Returns a DataFrame with timestamps as index and assets as columns.
    """
    date, assets, base_dir = arg
    results = []
    for asset in assets:
        file_path = base_dir + asset + '/' + asset + '_data_' + date + '.npz'
        data = np.load(file_path)
        ts = pd.to_datetime(data["timestamps"])
        mid = data["mid"]
        results.append(pd.DataFrame({asset: np.log(mid)}, index=ts))

    # Align timestamps across assets
    day_df = pd.concat(results, axis=1, join='inner').sort_index()
    return day_df


def aggregate_assets(assets, base_dir, out_dir, max_workers_days=None):
    """
    Aggregate log mid-quote price for multiple assets and days using full parallelization.
    - max_workers_days: number of parallel processes for days
    - max_workers_assets: number of parallel processes for assets within a day
    """
    os.makedirs(out_dir, exist_ok=True)

    csv_file_list = glob.glob(base_dir + assets[0] + '/' + "*.{}".format("npz"))
    dates = [re.findall(r"\d{4}-\d{2}-\d{2}", a)[-1] for a in csv_file_list]

    day_inputs = [(date, assets, base_dir) for date in dates]

    # Process each day in parallel
    with ProcessPoolExecutor(max_workers=max_workers_days) as executor:
        results = list(executor.map(process_day_parallel, day_inputs))

    # Concatenate all days
    log_price_data = pd.concat(results)

    # Ensure consistent asset order
    log_price_data = log_price_data[assets]

    # Save aggregated dataset
    log_price_path = out_dir + 'log_price.npz'
    np.savez(log_price_path, timestamps=log_price_data.index.astype('datetime64[s]'), **{col: log_price_data[col].values for col in log_price_data.columns})


def MRC(Y):
    """
    Compute the Modulated Realized Covariance (MRC) estimator.
    """
    assets = Y.columns
    Y = np.asarray(Y)
    n, d = Y.shape
    if n == 0:
        return None

    N = math.ceil((n - 1) ** .61 * 1)
    k_n = N if N % 2 == 0 else N + 1
    half = k_n // 2
    m = n - k_n + 1  # number of valid indices i

    # Compute bar{Y_i} for all i in a vectorized way
    barY = np.zeros((m, d))

    for i in range(m):
        # forward block
        forward = Y[i + half: i + k_n].sum(axis=0)
        # backward block
        backward = Y[i: i + half].sum(axis=0)
        barY[i] = (forward - backward) / k_n

    # Sum of outer products barY_i barY_i^T and multiply the scaling factor
    S = sum(np.outer(barY[i], barY[i]) for i in range(m)) * ((n - 1) / (n - k_n + 1)) * (12 / k_n)

    # Flatten covariance matrix
    iu = np.triu_indices(d)
    labels = [f"{assets[i]}_{assets[j]}" for i, j in zip(iu[0], iu[1])]
    data = S[iu]

    return pd.Series(data, index=labels)


def plot_realizedCov(cov_data, rules, out_dir='./data/LOBSTERdata/output/aggregate/'):
    fig, axes = plt.subplots(cov_data.shape[1], 1, figsize=(12, 3 * cov_data.shape[1]), sharex=True)

    for j in range(cov_data.shape[1]):
        ax = axes[j]
        for rule in rules:
            df_cov = cov_data.resample(rule).sum()
            df_cov = df_cov.loc[df_cov.index.intersection(cov_data.index)]
            df_cov.index = df_cov.index.strftime('%Y-%m-%d %H:%M')
            ax.plot(df_cov.iloc[:, j], label=rule)

        ax.set_title(f"{cov_data.columns[j]}")
        ax.set_ylabel("Covariance/Variance")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.xticks(df_cov.index[::6], rotation=45)

    fig.savefig(out_dir + 'plot.eps', bbox_inches='tight', format='eps')


def adjacency_matrices(A, n):
    """
    Compute all adjacency matrices up to distance n.
    :param A: binary adjacency matrix (NxN)
    :param n: maximum distance
    :return: list of binary matrices [A1, A2, ..., An]
    """
    reach = (A > 0).astype(int)       # distance <= 1
    results = [reach.copy()]          # list of distance-exact adjacency
    cumulative = reach.copy()
    for k in range(2, n+1):
        # paths of length k
        Ak = (reach @ A > 0).astype(int)

        # remove already reachable
        Ak = Ak * (1 - cumulative)

        # zero diagonal for no loops
        np.fill_diagonal(Ak, 0)

        results.append(Ak.copy())
        cumulative |= Ak
        reach = Ak.copy()  # DP update (only extend from new layer)

    return results


def edge_weight(adjs):
    """
    Compute edge weight for each edge in the network (for only one covariate now).
    :param adjs: list of binary matrices [A1, A2, ..., An]
    :return: list of numpy matrix [W]
    """
    N = adjs[0].shape[0]
    result = np.zeros((N, N))
    for adj in adjs:
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        result += adj / row_sums

    return [result]


def adj_to_line_graph(adj, TICKERS):
    adjacency = pd.DataFrame(adj, index=TICKERS, columns=TICKERS)
    n = adjacency.shape[0]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adjacency.iloc[i, j] == 1]
    m = len(edges)
    edges_ = [adjacency.columns[edge[0]] + '_' + adjacency.columns[edge[1]] for edge in edges]
    M = np.zeros((n, m), dtype=int)
    for e_idx, (i, j) in enumerate(edges):
        M[i, e_idx] = 1
        M[j, e_idx] = 1

    B = M.T @ M
    np.fill_diagonal(B, 0)
    return [B, edges_]


def line_graph(cov_data, TICKERS, corr_thres = 0.25, ratio_thres = 0.1):
    """
    Construct the line graph adjacency matrix given covariance/variance data
    :param cov_data: covariance/variance data
    :param TICKERS: list of tickers
    :param corr_thres: predefined correlation threshold, float
    :param ratio_thres: predefined ratio threshold, float
    :return: [line graph adjacency matrix, asset pair for each edge]
    """
    pairs = list(combinations(TICKERS, 2))
    corr_list = []

    # Record the correlation between assets as the value of edges
    for t, row in cov_data.iterrows():
        r = row.values
        corr_vals = {pair[0] + '_' + pair[1]: r[cov_data.columns.get_loc(pair[0] + '_' + pair[1])] / (
                    (r[cov_data.columns.get_loc(pair[0] + '_' + pair[0])] * r[cov_data.columns.get_loc(pair[1] + '_' + pair[1])]) ** .5) for pair in pairs}
        corr_list.append(pd.Series(corr_vals, name=t))
    corr_df = pd.DataFrame(corr_list)

    # Compute the ratio of the number of realized correlations exceed the correlation threshold
    s = (abs(corr_df) > corr_thres).sum(axis=0) / corr_df.shape[0]
    edges = []
    for pair, value in s.items():
        i, j = pair.split('_')
        edges.append((i, j, value))

    df_edges = pd.DataFrame(edges, columns=['from', 'to', 'weight'])

    # Initialize adjacency matrix
    adjacency = pd.DataFrame(0, index=TICKERS, columns=TICKERS)

    # Fill in adjacency matrix where ratio > ratio_thres
    for _, row in df_edges.iterrows():
        if abs(row['weight']) > ratio_thres:
            adjacency.loc[row['from'], row['to']] = 1
            adjacency.loc[row['to'], row['from']] = 1  # make symmetric

    n = adjacency.shape[0]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adjacency.iloc[i, j] == 1]
    m = len(edges)
    edges_ = [adjacency.columns[edge[0]] + '_' + adjacency.columns[edge[1]] for edge in edges]
    M = np.zeros((n, m), dtype=int)
    for e_idx, (i, j) in enumerate(edges):
        M[i, e_idx] = 1
        M[j, e_idx] = 1

    B = M.T @ M
    np.fill_diagonal(B, 0)
    return [B, edges_]


def log_data_to_cov_data(path: str, rule: str):
    # Load empirical data
    data = np.load(path + 'log_price.npz')
    index = pd.to_datetime(data["timestamps"])
    columns = [key for key in data.files if key != "timestamps"]
    log_price_data = pd.DataFrame({col: data[col] for col in columns}, index=index)
    data.close()

    # Compute modulated realised covariance
    out = {}

    for timestamp, group in log_price_data.resample(rule):
        res = MRC(group)
        if res is None:
            continue
        out[timestamp] = res

    cov_data = pd.DataFrame.from_dict(out, orient='index')
    np.savez(path + 'cov.npz', timestamps=cov_data.index.astype('datetime64[s]'),
             **{col: cov_data[col].values for col in cov_data.columns})


def hurwitz_principal_minors(a):
    #a = [a_1, a_2,..., a_l] length l
    n = len(a)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            idx = 2 * j - i + 1
            if 0 < idx <= n:
                H[i,j] = a[idx - 1]
            elif idx == 0:
                H[i,j] = 1   # coefficient of lambda^n (monic) when appears
            else:
                H[i,j] = 0
    # compute leading principal minors Delta_k
    deltas = [det(H[:k, :k]) for k in range(1, n+1)]
    return np.array(deltas)


def estimator_to_QQ_cp(V, R_L: np.ndarray, C: int, W_C, Adj, K: int):
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
        betas.append(cp.reshape(V[idx:idx + n_beta], (n_beta, 1), order='F'))
        idx += n_beta

    W = [[np.multiply(W_C[c], Adj[stage]) for c in range(C) for stage in range(R_i)] for R_i in R_L]
    Q = []
    for l in range(L):
        alpha_mat = cp.multiply(alphas[l], np.eye(K))
        kron_beta = cp.kron(betas[l], np.eye(K))
        Q.append(alpha_mat + cp.hstack(W[l]) @ kron_beta)

    top_blocks = []
    for i in range(L - 1):
        row = [np.zeros((K, K)) for _ in range(L)]
        row[i + 1] = np.eye(K)
        top_blocks.append(np.hstack(row))
        # Last row: -A_p, ..., -A_1
    last_row = [-Q[L - i - 1] for i in range(L)]
    all_rows = cp.vstack(top_blocks + [cp.hstack(last_row)])
    return cp.bmat(all_rows)

def kalman_filter_mcar(Y, A, Sigma, P_):
    """
    Y: (d, N+1) observations at time grid
    A: (pd, pd) MCAR matrix
    Sigma: (d, d) Brownian covariance of Levy process
    P_: (N+1, ) observation times

    Returns m_filtered, P_filtered: lists of length n+1
    """
    d, n = Y.shape[0], Y.shape[1] - 1
    pd = A.shape[0]

    # Build Sigma_tilde = E Sigma E^T, with E[-d:,:] = I_d
    E = np.zeros((pd, d))
    E[-d:, :] = np.eye(d)
    Sigma_tilde = E @ Sigma @ E.T

    M = np.zeros((2 * pd, 2 * pd))
    M[:pd, :pd] = A
    M[:pd, pd:] = Sigma_tilde
    M[pd:, pd:] = -A.T
    H = np.zeros((d, pd))
    H[:, :d] = np.eye(d)

    delta = P_[1] - P_[0]
    expM = scipy.linalg.expm(M * delta)
    F_k = scipy.linalg.expm(A * delta)
    Q_k = expM[:pd, pd:] @ F_k.T

    # Initial prior
    m = np.zeros(pd)
    P = np.eye(pd)
    m_filtered = [m.copy()]

    for k in range(n):
        # 1) Prediction
        m_pred = F_k @ m
        P_pred = F_k @ P @ F_k.T + Q_k

        # 2) Update
        v = Y[:, k + 1] - H @ m_pred
        S = H @ P_pred @ H.T
        K = P_pred @ H.T @ np.linalg.pinv(S)

        m = m_pred + K @ v
        P = P_pred - K @ S @ K.T

        m_filtered.append(m.copy())

    return m_filtered[-1]


def kalman_filter_mcar_BM(Y, A, Sigma, P_, Q_, AA_hat, gamma):
    """
    Y: (d, N+1) observations at time grid
    A: (pd, pd) MCAR matrix
    Sigma: (d, d) Brownian covariance of Levy process
    P_: (N+1, ) observation times

    Returns m_filtered, P_filtered: lists of length n+1
    """
    d= Y.shape[0]
    pd = A.shape[0]
    p = int(pd / d)

    DeltaL_Q = recover_BDLP(Y, p, P_, Q_, AA_hat)
    DeltaW, DeltaQ_W = disentangle_BM(DeltaL_Q, Q_, Sigma, gamma)
    Q_index = np.searchsorted(Q_, np.cumsum(DeltaQ_W), side="left")
    Y_Q = Y[:, Q_index]

    # Build Sigma_tilde = E Sigma E^T, with E[-d:,:] = I_d
    E = np.zeros((pd, d))
    E[-d:, :] = np.eye(d)
    Sigma_tilde = E @ Sigma @ E.T

    M = np.zeros((2 * pd, 2 * pd))
    M[:pd, :pd] = A
    M[:pd, pd:] = Sigma_tilde
    M[pd:, pd:] = -A.T
    H = np.zeros((d, pd))
    H[:, :d] = np.eye(d)

    # Initial prior
    m = np.zeros(pd)
    P = np.eye(pd)
    m_filtered = [m.copy()]

    for k in range(len(DeltaQ_W)):
        expM = scipy.linalg.expm(M * DeltaQ_W[k])
        F_k = scipy.linalg.expm(A * DeltaQ_W[k])
        Q_k = expM[:pd, pd:] @ F_k.T

        # 1) Prediction
        m_pred = F_k @ m
        P_pred = F_k @ P @ F_k.T + Q_k

        # 2) Update
        v = Y_Q[:, k] - H @ m_pred
        S = H @ P_pred @ H.T
        K = P_pred @ H.T @ np.linalg.pinv(S)

        m = m_pred + K @ v
        P = P_pred - K @ S @ K.T

        m_filtered.append(m.copy())

    return m_filtered[-1]


def UAR_model(Y, test_set, K, test_size, forecast_step, lag = None):
    # Univariate AR Edge Model

    df_AR = np.array(Y.copy())
    aic_values = []
    best_p = np.zeros((K,))
    for i in range(K):
        for p in range(1, 7):
            model = AutoReg(df_AR[i, :], lags=p).fit()
            aic_values.append((p, model.aic))
        best_p[i] = min(aic_values, key=lambda x: x[1])[0]

    forecast_AR = np.zeros((test_size, K, forecast_step))
    for i in range(test_size):
        for j in range(K):
            window_data = df_AR[j, i:]
            if lag is None:
                model = AutoReg(window_data, lags=best_p[j]).fit()
            else:
                model = AutoReg(window_data, lags=lag).fit()
            forecast_AR[i, j, :] = model.predict(start=len(window_data), end=len(window_data) + forecast_step - 1)
        df_AR = np.append(df_AR, test_set[i, :, 0:1], axis=1)

    with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', 'forecast_AR.npy'), 'wb') as f:
        np.save(f, forecast_AR)

    RMSE_AR = np.sqrt(np.mean((forecast_AR - test_set) ** 2))
    print(f'RMSE for UAR Edge model is: {RMSE_AR}')

    last_ = np.append(np.array(Y.copy())[:, -1].reshape(-1, 1)[np.newaxis, :, :], test_set[:-1, :, 0:1], axis=0)
    forecast_AR_dir = np.sign(np.diff(np.append(last_, forecast_AR, axis=2), axis=2))
    test_set_dir = np.sign(np.diff(np.append(last_, test_set, axis=2), axis=2))
    print(f'Direction success rate for UAR Edge model is: {(test_set_dir == forecast_AR_dir).mean()}')


def VAR_model(Y, test_set, K, test_size, forecast_step):
    # VAR Edge Model

    df_VAR = np.array(Y.T.copy())
    model = VAR(df_VAR)

    # automatic lag selection by AIC
    best_lag_var = model.select_order(maxlags=6).aic

    forecast_VAR = np.zeros((test_size, K, forecast_step))
    for i in range(test_size):
        window_data = df_VAR[i:, :]
        model = VAR(window_data).fit(best_lag_var)
        forecast_VAR[i, :, :] = model.forecast(window_data[-model.k_ar:, :], steps=forecast_step).T
        df_VAR = np.append(df_VAR, test_set[i, :, 0:1].T, axis=0)

    with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', 'forecast_VAR.npy'), 'wb') as f:
        np.save(f, forecast_VAR)

    RMSE_VAR = np.sqrt(np.mean((forecast_VAR - test_set) ** 2))
    print(f'RMSE for VAR Edge model is: {RMSE_VAR}')

    last_ = np.append(np.array(Y.copy())[:, -1].reshape(-1, 1)[np.newaxis, :, :], test_set[:-1, :, 0:1], axis=0)
    forecast_VAR_dir = np.sign(np.diff(np.append(last_, forecast_VAR, axis=2), axis=2))
    test_set_dir = np.sign(np.diff(np.append(last_, test_set, axis=2), axis=2))
    print(f'Direction success rate for VAR Edge model is: {(test_set_dir == forecast_VAR_dir).mean()}')


def NA_model(Y, test_set, forecast_step):
    # Naive Model

    forecast_NA = np.repeat(np.append(np.array(Y.copy())[:, -1].reshape(-1, 1)[np.newaxis, :, :], test_set[:-1, :, 0:1], axis=0), forecast_step, axis=2)

    RMSE_NA = np.sqrt(np.mean((forecast_NA - test_set) ** 2))
    print(f'RMSE for Naive model is: {RMSE_NA}')


def OU_edge_model(Y, test_set, K, test_size, forecast_step, R_L, C, jump_activity, P_, Q_):
    # OU Edge model

    forecast_OU = np.zeros((test_size, K, forecast_step))
    df_OU = np.array(Y.copy())
    Adj_L_OU = [np.zeros((K, K)) for _ in range(max(R_L))]
    W_C_OU = edge_weight(Adj_L_OU)
    X_OU = GraphOUEdge(R_L, K, C, Adj_L_OU, W_C_OU, 'data\\LOBSTERdata\\output\\train\\aggregate', 'plots')
    X_OU.estimate_unknown_Levy(Y, P_, Q_, 'OU',f'{jump_activity}')
    b_OU = np.load(f'./data/LOBSTERdata/output/train/aggregate/MCAR_estimates_b_{jump_activity}.npy')
    Sigma_OU = np.load(f'./data/LOBSTERdata/output/train/aggregate/MCAR_estimates_Sigma_{jump_activity}.npy')
    F_mean_OU = np.load(f'./data/LOBSTERdata/output/train/aggregate/MCAR_estimates_F_mean_{jump_activity}.npy')
    F_squared_OU = np.load(f'./data/LOBSTERdata/output/train/aggregate/MCAR_estimates_F_squared_{jump_activity}.npy')

    jump_F_OU = scipy.stats.multivariate_normal(mean=F_mean_OU, cov=(F_squared_OU - np.outer(F_mean_OU, F_mean_OU)), allow_singular=True)
    # alphas_OU = []

    for i in range(test_size):
        window_data = df_OU[:, i:]
        theta_hat_OU, HH, H = estimate_OU_drift(window_data, R_L, K, C, P_, Q_, b_OU, None, Sigma_OU, with_cov=True)
        X_OU.alpha_set(theta_hat_OU)
        X_OU.beta_set([[0] * stage * C for stage in R_L])
        try:
            Q_Q = X_OU.Q_matrix()
        except:
            # for l = 1 or 2
            if len(R_L) < 3:
                theta_hat_OU = minimize(lambda x: -np.matmul(x, H) + 0.5 * np.matmul(x, np.matmul(HH, x)), theta_hat_OU,
                                        bounds=[(0.0, None)] * len(theta_hat_OU), method='L-BFGS-B').x
            # for l >= 3
            else:
                n = len(theta_hat_OU)
                hurwitz_constraint = NonlinearConstraint(hurwitz_principal_minors, lb=np.full(n, 1e-16), ub=np.full(n, np.inf))
                theta_hat_OU = minimize(lambda x: -np.matmul(x, H) + 0.5 * np.matmul(x, np.matmul(HH, x)), theta_hat_OU,
                                        bounds=Bounds(lb=np.zeros(n), ub=np.full(n, np.inf), keep_feasible=True),
                                        constraints=[hurwitz_constraint], method='trust-constr').x
            X_OU.alpha_set(theta_hat_OU)
            X_OU.beta_set([[0] * stage * C for stage in R_L])
            Q_Q = X_OU.Q_matrix(False)

        # alphas_OU.append(theta_hat_OU)

        if len(R_L) == 1:
            x0 = window_data[:, -1]
        else:

        # Backward - differences
            #flip_window = state_space(np.flip(window_data[:, -len(R_L):], axis=1), len(R_L), P_[-len(R_L):])
            #x0 = flip_window[:, 0] * np.repeat((-1) ** np.arange(len(R_L)), flip_window.shape[0] / len(R_L))

        # Naive initialization
            x0 =np.append(window_data[:, -1], [0] * K * (len(R_L) - 1))

        # Kalman - filter
            #x0 = kalman_filter_mcar(window_data[:, -6:], Q_Q, Sigma_OU, P_[-6:])
            #x0[:K] = window_data[:, -1]

            #AA_hat = [np.diag([x] * K) for x in theta_hat_OU]
            #x0 = kalman_filter_mcar_BM(window_data, Q_Q, Sigma_OU, P_, Q_, AA_hat, 1.01)
            #x0[:K] = window_data[:, -1]

        # Make predictions
        def run_single_simulation():
            """Wrapper function for single simulation"""
            return simulate_MCAR_compound_poisson(
                [j * P_[1] for j in range(forecast_step + 1)],
                Q_Q,
                x0,
                b_OU,
                Sigma_OU,
                1.0,
                jump_F_OU,
                uniform=True
            )[:, 1:]

        paths = Parallel(n_jobs=-1)(
            delayed(run_single_simulation)() for _ in range(1000)
        )

        forecast_OU[i, :, :] = np.mean(np.array(paths), axis=0)
        df_OU = np.append(df_OU, test_set[i, :, 0:1], axis=1)

    with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', f'forecast_OU_{len(R_L)}.npy'), 'wb') as f:
        np.save(f, forecast_OU)

    RMSE_OU = np.sqrt(np.mean((forecast_OU - test_set) ** 2))
    print(f'RMSE for OU({len(R_L)}) Edge model is: {RMSE_OU}')

    last_ = np.append(np.array(Y.copy())[:, -1].reshape(-1, 1)[np.newaxis, :, :], test_set[:-1, :, 0:1], axis=0)
    forecast_OU_dir = np.sign(np.diff(np.append(last_, forecast_OU, axis=2), axis=2))
    test_set_dir = np.sign(np.diff(np.append(last_, test_set, axis=2), axis=2))
    print(f'Direction success rate for OU({len(R_L)}) Edge model is: {(test_set_dir == forecast_OU_dir).mean()}')


def grOU_edge_model(Y, Adj, test_set, K, test_size, forecast_step, R_L, C, jump_activity, P_, Q_):
    # graph OU Edge model

    # Derive the adjacency matrix for edge graph
    Adj_L = adjacency_matrices(Adj[0], max(R_L))

    # Compute the edge weight
    W_C = edge_weight(Adj_L)

    X = GraphOUEdge(R_L, K, C, Adj_L, W_C, 'data\\LOBSTERdata\\output\\train\\aggregate', 'plots')
    aic = X.estimate_unknown_Levy(Y, P_, Q_, 'grOU',f'{jump_activity}') * -2 + 2 * (len(R_L) + sum(R_L) * C)
    b = np.load(f'./data/LOBSTERdata/output/train/aggregate/MCAR_estimates_b_{jump_activity}.npy')
    Sigma = np.load(f'./data/LOBSTERdata/output/train/aggregate/MCAR_estimates_Sigma_{jump_activity}.npy')
    F_mean = np.load(f'./data/LOBSTERdata/output/train/aggregate/MCAR_estimates_F_mean_{jump_activity}.npy')
    F_squared = np.load(f'./data/LOBSTERdata/output/train/aggregate/MCAR_estimates_F_squared_{jump_activity}.npy')

    forecast_grOU = np.zeros((test_size, K, forecast_step))
    df_grOU = np.array(Y.copy())
    jump_F = scipy.stats.multivariate_normal(mean=F_mean, cov=(F_squared - np.outer(F_mean, F_mean)), allow_singular=True)
    # alphas = []
    # betas = []

    for i in range(test_size):
        window_data = df_grOU[:, i:]
        theta_hat, HH, H = estimate_grCAR_drift(window_data, Adj_L, W_C, len(R_L), R_L, C, P_, Q_, b, None, Sigma, with_cov=True)

        # Fetch parameters
        idx = 0
        alpha_pred = []
        beta_pred = []
        for r in R_L:
            alpha_pred.append(theta_hat[idx])
            beta_pred.append(theta_hat[(idx + 1):(idx + 1 + r * C)])
            idx += 1 + r * C
        X.alpha_set(alpha_pred)
        X.beta_set(beta_pred)

        try:
            Q_Q = X.Q_matrix()
        except:
            x = cp.Variable(len(R_L) + sum(R_L))

            # Objective function
            objective = cp.Minimize(-H.T @ x + 0.5 * cp.quad_form(x, HH))

            # Constraint: the symmetric part of Q_Q is negative definite
            epsilon = 0
            constraints = [estimator_to_QQ_cp(x, R_L, C, W_C, Adj_L, K) << -epsilon * np.eye(len(R_L) * K)]

            # Problem setup
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver=cp.CVXOPT, abstol=1e-5, reltol=1e-5, feastol=1e-5)
                theta_hat = x.value
                if theta_hat is None:
                    prob.solve(solver=cp.MOSEK)
                    theta_hat = x.value
            except:
                prob.solve(solver=cp.MOSEK)
                theta_hat = x.value
                if theta_hat is None:
                    prob.solve(solver=cp.SCS)
                    theta_hat = x.value
            idx = 0
            alpha_pred = []
            beta_pred = []
            for r in R_L:
                alpha_pred.append(theta_hat[idx])
                beta_pred.append(theta_hat[(idx + 1):(idx + 1 + r * C)])
                idx += 1 + r * C
            X.alpha_set(alpha_pred)
            X.beta_set(beta_pred)
            Q_Q = X.Q_matrix(False)

        # alphas.append(alpha_pred)
        # betas.append(beta_pred)

        flip_window = state_space(np.flip(window_data[:, -len(R_L):], axis=1), len(R_L), P_[-len(R_L):])
        x0 = flip_window[:, 0] * np.repeat((-1) ** np.arange(len(R_L)), flip_window.shape[0] / len(R_L))

        # Make predictions
        def run_single_simulation():
            """Wrapper function for single simulation"""
            return simulate_MCAR_compound_poisson(
                [i * P_[1] for i in range(forecast_step + 1)],
                Q_Q,
                np.append(window_data[:, -1], [0] * K * (len(R_L) - 1)),
                b,
                Sigma,
                1.0,
                jump_F,
                uniform=True
            )[:, 1:]

        paths = Parallel(n_jobs=-1)(
            delayed(run_single_simulation)() for _ in range(1000)
        )

        forecast_grOU[i, :, :] = np.mean(np.array(paths), axis=0)
        df_grOU = np.append(df_grOU, test_set[i, :, 0:1], axis=1)

    with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', f'forecast_grOU_{len(R_L)}_{R_L}.npy'), 'wb') as f:
        np.save(f, forecast_grOU)

    RMSE_grOU = np.sqrt(np.mean((forecast_grOU - test_set) ** 2))
    print(f'RMSE for graph OU({len(R_L)}_{R_L}) Edge model is: {RMSE_grOU}')

    last_ = np.append(np.array(Y.copy())[:, -1].reshape(-1, 1)[np.newaxis, :, :], test_set[:-1, :, 0:1], axis=0)
    forecast_grOU_dir = np.sign(np.diff(np.append(last_, forecast_grOU, axis=2), axis=2))
    test_set_dir = np.sign(np.diff(np.append(last_, test_set, axis=2), axis=2))
    print(f'Direction success rate for graph OU({len(R_L)}_{R_L}) Edge model is: {(test_set_dir == forecast_grOU_dir).mean()}')