import os
import glob
import math
import re
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from numpy.lib.stride_tricks import sliding_window_view
from multiprocessing import Pool
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR
from graphOUedge import GraphOUEdge
from mcar.estimate import *
from mcar.simulate import *

def multiprocess_orderbooks(TICKER, input_path, output_path, log_path, check_for_processed_data=True, sample_interval = '1s'):
    """
    Pre-process LOBSTER data parallely. The data must be stored in the input_path directory as daily message book and order book files.
    For details of processing steps see description of process_orderbook function.
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
    :param sample_interval: sample interval, str
    :return: saves the aggregated mid-quote price in output_path, as .npz files with numpy attributes
             "orderbook_features" (:, 1).
    """
    csv_file_list = glob.glob(os.path.join(input_path, "*.{}".format("csv")))
    csv_orderbook = [name for name in csv_file_list if "orderbook" in name]
    csv_message = [name for name in csv_file_list if "message" in name]
    csv_orderbook.sort()
    csv_message.sort()

    # Do not process already processed data
    if check_for_processed_data:
        npz_file_list = sorted(glob.glob(os.path.join(output_path, "*.{}".format("npz"))))
        processed_dates = [re.search(r'\d{4}-\d{2}-\d{2}', file).group() for file in npz_file_list]
        csv_orderbook = [file for file in csv_orderbook if
                         re.search(r'\d{4}-\d{2}-\d{2}', file).group() not in processed_dates]
        csv_message = [file for file in csv_message if
                       re.search(r'\d{4}-\d{2}-\d{2}', file).group() not in processed_dates]

    # Check if exactly half of the files are order book and exactly half are messages
    assert (len(csv_message) == len(csv_orderbook))

    print("started multiprocessing")

    pool = Pool(os.cpu_count())
    try:
        engine = ProcessOrderbook(TICKER, output_path, sample_interval)
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

    def __init__(self, TICKER, output_path, sample_interval):
        self.sample_interval = sample_interval
        self.TICKER = TICKER
        self.output_path = output_path

    def __call__(self, orderbook_name):
        output = process_orderbook(orderbook_name, self.TICKER, self.output_path, self.sample_interval)
        return output

def process_orderbook(orderbook_name, TICKER, output_path, sample_interval):
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
    :param sample_interval: sample interval, str
    :return: saves the aggregated mid-quote price in output_path, as .npz files with numpy attributes
             "orderbook_features" (:, 1).
    """
    print(orderbook_name)
    log = ''

    ## Load LOBSTER orderbook and message files
    try:
        df_orderbook = pd.read_csv(orderbook_name, header=None)
    except:
        return orderbook_name + ' skipped. Error: failed to read orderbook.'

    # Compute the mid-quote price
    df_orderbook.columns = ["ASKp1", "ASKs1", "BIDp1", "BIDs1"]
    df_orderbook.insert(0, "mid", (df_orderbook["ASKp1"] + df_orderbook["BIDp1"]) / 2)

    # Get date
    match = re.findall(r"\d{4}-\d{2}-\d{2}", orderbook_name)[-1]
    date = datetime.strptime(match, "%Y-%m-%d")

    # Read times from message file. keep a record of problematic files
    message_name = orderbook_name.replace("orderbook", "message")
    try:
        df_message = pd.read_csv(message_name, usecols=[0, 1, 2, 3, 4, 5], header=None)
    except:
        return orderbook_name + ' skipped. Error: failed to read messagebook.'

    # Check the two df have the same length
    assert (len(df_message) == len(df_orderbook))

    # Add column names to message book
    df_message.columns = ["seconds", "event type", "order ID", "volume", "price", "direction"]

    ## Process potential data anomalies

    # 1. Remove trading halts
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

    # 2. Remove crossed quotes
    crossed_quotes_index = df_orderbook[(df_orderbook["BIDp1"] > df_orderbook["ASKp1"])].index
    if len(crossed_quotes_index) > 0:
        log = log + 'Warning: ' + str(
            len(crossed_quotes_index)) + ' crossed quotes removed in ' + orderbook_name + '.\n'
    df_orderbook = df_orderbook.drop(crossed_quotes_index)
    df_message = df_message.drop(crossed_quotes_index)

    # Add the seconds since midnight column to the order book from the message book
    df_orderbook.insert(0, "seconds", df_message["seconds"])

    # 3. One conceptual event (e.g. limit order modification which is implemented as a cancellation followed
    # by an immediate new arrival, single market order executing against multiple resting limit orders) may
    # appear as multiple rows in the message file, all with the same timestamp.
    # We hence group the order book data by unique timestamps and take the last entry.
    df_orderbook = df_orderbook.groupby(["seconds"]).tail(1)

    # 4. Check market opening times for strange values
    market_open = int(df_orderbook["seconds"].iloc[0] / 60) / 60  # open at minute before first transaction
    market_close = (int(df_orderbook["seconds"].iloc[-1] / 60) + 1) / 60  # close at minute after last transaction

    if not (market_open == 9.5 and market_close == 16):
        log = log + 'Warning: unusual opening times in ' + orderbook_name + ': ' + str(market_open) + ' - ' + str(
            market_close) + '.\n'

    # Drop values outside of market hours
    df_orderbook = df_orderbook.loc[(df_orderbook["seconds"] >= 34200) &
                                    (df_orderbook["seconds"] <= 57600)]

    # 5. Drop first and last 60 minutes of trading using seconds (do this after orderflow computation)
    market_open_seconds = market_open * 60 * 60 + 60 * 60
    market_close_seconds = market_close * 60 * 60 - 60 * 60
    df_orderbook = df_orderbook.loc[(df_orderbook["seconds"] >= market_open_seconds) &
                                    (df_orderbook["seconds"] <= market_close_seconds)]

    # 6. Downsample tick-by-tick mid-quote price data into sampling, keeping only the last tick in each sample interval with forward-fill
    df_orderbook['timestamp'] = pd.to_datetime(df_orderbook['seconds'], unit='s', origin=pd.Timestamp(date))
    df_orderbook = df_orderbook.set_index('timestamp')
    df_orderbook_min = df_orderbook.resample(sample_interval).last().ffill()

    # Save
    output_name = os.path.join(output_path, TICKER + "_" + "data" + "_" + str(date.date()) + ".npz")
    np.savez(output_name, timestamps=df_orderbook_min.index.astype('datetime64[s]'), mid=df_orderbook_min['mid'].values)

    return log + orderbook_name + ' completed.'

def process_day_parallel(arg):
    """
    Load all assets for a single day in parallel. Returns a DataFrame with timestamps as index and assets as columns.
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

def MRC(Y, delta, theta, is_corr=True):
    """
    Compute the Modulated Realized Correlation (MRC) estimator.
    """
    assets = Y.columns
    Y = np.asarray(Y)
    n, d = Y.shape
    if n == 0:
        return None

    N = math.ceil((n - 1) ** delta * theta)
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

    if is_corr:
        # Convert covariance to correlation
        diag = np.sqrt(np.diag(S))
        denom = np.outer(diag, diag)

        # Numerical safety
        S = np.divide(S, denom, out=np.zeros_like(S), where=denom > 0)

    # Flatten correlation matrix
    iu = np.triu_indices(d)
    labels = [f"{assets[i]}_{assets[j]}" for i, j in zip(iu[0], iu[1])]
    data = S[iu]

    return pd.Series(data, index=labels)

def log_data_to_corr_data(path: str, rule: str, delta=.6, theta=1, is_corr=True):
    # Load empirical data
    data = np.load(path + 'log_price.npz')
    index = pd.to_datetime(data["timestamps"])
    columns = [key for key in data.files if key != "timestamps"]
    log_price_data = pd.DataFrame({col: data[col] for col in columns}, index=index)
    data.close()

    # Compute modulated realized correlation
    out = {}

    for timestamp, group in log_price_data.resample(rule):
        res = MRC(group, delta, theta, is_corr)
        if res is None:
            continue
        out[timestamp] = res

    corr_data = pd.DataFrame.from_dict(out, orient='index')
    if is_corr:
        np.savez(path + 'corr.npz', timestamps=corr_data.index,
             **{col: corr_data[col].values for col in corr_data.columns})
    else:
        np.savez(path + 'cov.npz', timestamps=corr_data.index,
                 **{col: corr_data[col].values for col in corr_data.columns})

def adjacency_matrices(A, n):
    """
    Compute all adjacency matrices up to distance n.
    :param A: binary adjacency matrix (NxN)
    :param n: maximum distance
    :return: list of binary matrices [A1, A2, ..., An]
    """
    # Distance <= 1
    reach = (A > 0).astype(int)
    # List of distance-exact adjacency
    results = [reach.copy()]
    cumulative = reach.copy()
    for k in range(2, n + 1):
        # Paths of length k
        Ak = (reach @ A > 0).astype(int)

        # Remove already reachable
        Ak = Ak * (1 - cumulative)

        # Zero diagonal for no loops
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

def kalman_filter_mcar_BM(Y, Q_QQ, Sigma, P_, Q_, QQ_hat, gamma):
    """
    Y: (d, N+1) observations at time grid
    Q_QQ: (pd, pd) MCAR matrix
    Sigma: (d, d) Brownian covariance of Levy process
    P_: (N+1, ) observation times
    Q_: (M+1, ) coarser grid
    QQ_hat: estimated MCAR parameters, list of p (d, d) np.ndarray
    gamma: hyperparameter for defining the critical region
    Returns the last value of m_filtered, (pd, ) np.ndarray
    """
    d = Y.shape[0]
    pd = Q_QQ.shape[0]
    p = int(pd / d)

    DeltaL_Q = recover_BDLP(Y, p, P_, Q_, QQ_hat)
    DeltaW, DeltaQ_W = disentangle_BM(DeltaL_Q, Q_, Sigma, gamma)
    Q_index = np.searchsorted(Q_, np.cumsum(DeltaQ_W), side="left")
    Y_Q = Y[:, Q_index]

    # Build Sigma_tilde = E Sigma E^T, with E[-d:,:] = I_d
    E = np.zeros((pd, d))
    E[-d:, :] = np.eye(d)
    Sigma_tilde = E @ Sigma @ E.T

    M = np.zeros((2 * pd, 2 * pd))
    M[:pd, :pd] = Q_QQ
    M[:pd, pd:] = Sigma_tilde
    M[pd:, pd:] = -Q_QQ.T
    H = np.zeros((d, pd))
    H[:, :d] = np.eye(d)

    # Initial prior
    m = np.zeros(pd)
    P = np.eye(pd)
    m_filtered = [m.copy()]

    for k in range(len(DeltaQ_W)):
        expM = scipy.linalg.expm(M * DeltaQ_W[k])
        F_k = scipy.linalg.expm(Q_QQ * DeltaQ_W[k])
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

def worker(arg):
    _data, test_data_, TICKERS, R_L, P_, Q_, forecast_step, C, gamma, i = arg

    np.random.seed(i)
    A = np.triu(np.random.randint(0, 2, (8, 8)), 1)
    A += A.T

    # Derive network structure
    Adj = adj_to_line_graph(A, TICKERS)

    # Number of edges
    K = Adj[0].shape[0]

    # Training data
    Y = _data[Adj[1]].T

    # Testing data
    test_Y = test_data_[Adj[1]].T
    test_set = sliding_window_view(test_Y, (K, forecast_step))[0]
    test_size = test_set.shape[0]

    forecast_AR = UAR_model(Y, test_set, test_size, forecast_step, lag=len(R_L))
    _, direction_acc_AR = evaluate_forecast(Y, forecast_AR, test_set)
    forecast_grOU =  grOU_edge_model(Y, Adj, test_set, test_size, forecast_step, R_L, C, P_, Q_, gamma)
    _, direction_acc_grOU = evaluate_forecast(Y, forecast_grOU, test_set)
    return direction_acc_grOU - direction_acc_AR, A, (direction_acc_grOU - direction_acc_AR)/direction_acc_AR

def process_one_sim(arg):
    P, R_L, K, C, Y, adj_L_sim, W_C_sim, gamma = arg
    result = np.zeros((3, K * len(R_L) + sum(R_L)))

    for n in range(1, 4):
        T = 2 ** n
        subset_P = slice(0, int(8 ** 7 / 2 ** (3 - n)) + 1, 2 ** (6 * (3 - n)))
        P_ = P[subset_P]  # delta_P = T^(-6)
        Q_ = P_[::T ** 4]  # delta_Q = T^(-2)
        Y_ = Y[:, subset_P]

        _, (b_sim_sol, Sigma_sim_sol) = estimate_MCAR('grOU', Y_, P_, Q_, adj_L_sim, W_C_sim, R_L, C, gamma)
        theta_hat_sim_sol, _, _ = estimate_grOU_drift(Y_, adj_L_sim, W_C_sim, R_L, C, P_, Q_, b_sim_sol, gamma,
                                                      None, Sigma_sim_sol, with_cov=True, is_global=False)
        result[n - 1, :] = theta_hat_sim_sol.reshape(-1)
    return result

def model_pred_single(args):
    index_, sim_Y, sim_test, adj_sim, forecast_step, P_Q_ratio, gamma = args

    sim_test_set = sliding_window_view(sim_test, (sim_Y.shape[0], forecast_step))[0]
    sim_test_size = sim_test_set.shape[0]
    P_sim = np.array([i for i in range(sim_Y.shape[1])])
    Q_sim = np.append(P_sim[::P_Q_ratio], P_sim[-1]) if (sim_Y.shape[1] - 1) % P_Q_ratio != 0 else P_sim[::P_Q_ratio]

    RMSE_row = np.zeros(7)
    direction_row = np.zeros(6)
    comp_row = np.zeros(6)

    start = time.perf_counter()
    forecast = UAR_model(sim_Y, sim_test_set, sim_test_size, forecast_step, 1)
    comp_row[0] = time.perf_counter() - start
    RMSE_row[0], direction_row[0] = evaluate_forecast(sim_Y, forecast, sim_test_set)

    start = time.perf_counter()
    forecast = VAR_model(sim_Y, sim_test_set, sim_test_size, forecast_step, 1)
    comp_row[1] = time.perf_counter() - start
    RMSE_row[1], direction_row[1] = evaluate_forecast(sim_Y, forecast, sim_test_set)

    start = time.perf_counter()
    forecast = GNAR_model(sim_Y, adj_sim, [1], sim_test_set, sim_test_size, forecast_step)
    comp_row[2] = time.perf_counter() - start
    RMSE_row[2], direction_row[2] = evaluate_forecast(sim_Y, forecast, sim_test_set)

    start = time.perf_counter()
    forecast = OU_edge_model(sim_Y, sim_test_set, sim_test_size, forecast_step, [1],
                                    1, P_sim, Q_sim, gamma, is_global=False)
    comp_row[3] = time.perf_counter() - start
    RMSE_row[3], direction_row[3] = evaluate_forecast(sim_Y, forecast, sim_test_set)

    start = time.perf_counter()
    forecast = MCAR_edge_model(sim_Y, sim_test_set, sim_test_size, forecast_step, P_sim, Q_sim, [1], 1, gamma)
    comp_row[4] = time.perf_counter() - start
    RMSE_row[4], direction_row[4] = evaluate_forecast(sim_Y, forecast, sim_test_set)

    start = time.perf_counter()
    forecast = grOU_edge_model(sim_Y, adj_sim, sim_test_set, sim_test_size, forecast_step,
                                            [1], 1, P_sim, Q_sim, gamma, is_global=False)
    comp_row[5] = time.perf_counter() - start
    RMSE_row[5], direction_row[5] = evaluate_forecast(sim_Y, forecast, sim_test_set)

    _, RMSE_row[6] = NA_model(sim_Y, sim_test_set, forecast_step)

    return index_, RMSE_row, direction_row, comp_row

def model_pred(data, test_size, adj_sim, forecast_step, P_Q_ratio, gamma, label, scenario):
    if label == 'missing_5':
        data = data[:, 1:, :]
    elif label == 'missing_1':
        data = data[:, :-1, :]

    n = data.shape[0]
    RMSE = np.zeros((n, 7))
    direction_acc = np.zeros((n, 6))
    comp_time = np.zeros((n, 6))

    args_list = [(i, data[i, :, :-test_size], data[i, :, -test_size:], adj_sim, forecast_step, P_Q_ratio,
                  gamma) for i in range(n)]
    with Pool(os.cpu_count()) as pool:
        for idx, rmse_row, dir_row, comp_row in pool.imap_unordered(model_pred_single, args_list):
            RMSE[idx] = rmse_row
            direction_acc[idx] = dir_row
            comp_time[idx] = comp_row

    with open('./data/LOBSTERdata/output/sim/' + scenario + '/rmse_' + label + '.npy', 'wb') as f:
        np.save(f, RMSE)
    with open('./data/LOBSTERdata/output/sim/' + scenario + '/dir_' + label + '.npy', 'wb') as f:
        np.save(f, direction_acc)
    with open('./data/LOBSTERdata/output/sim/' + scenario + '/time_' + label + '.npy', 'wb') as f:
        np.save(f, comp_time)

def boxplot_(sim, scenario):
    plt.plot(sim[0, :, :].T)
    plt.savefig('./data/LOBSTERdata/output/sim/' + scenario + '/sample_path.eps', bbox_inches='tight', format='eps')

    grOU_RMSE = np.zeros((sim.shape[0], 3))
    grOU_direction_acc = np.zeros((sim.shape[0], 3))

    for i, label in enumerate(['correct', 'missing_5', 'missing_1']):
        with open('./data/LOBSTERdata/output/sim/' + scenario + '/rmse_' + label + '.npy', 'rb') as f:
            RMSE = np.load(f)
            RMSE = RMSE[~np.isnan(RMSE).any(axis=1)]
        grOU_RMSE[:, i] = RMSE[:, 5]

        with open('./data/LOBSTERdata/output/sim/' + scenario + '/dir_' + label + '.npy', 'rb') as f:
            direction_acc = np.load(f)
            direction_acc = direction_acc[~np.isnan(direction_acc).any(axis=1)]
        grOU_direction_acc[:, i] = direction_acc[:, 5]

        with open('./data/LOBSTERdata/output/sim/' + scenario + '/time_' + label + '.npy', 'rb') as f:
            comp_time = np.load(f)
            comp_time = comp_time[~np.isnan(comp_time).any(axis=1)]

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        positions = np.arange(1, 8)
        ax[0].boxplot(RMSE[:, :], positions=positions, widths=0.3, showfliers=False)

        for j in range(7):
            kde = gaussian_kde(RMSE[:, j], bw_method=0.4)
            low, high = np.percentile(RMSE[:, j], [5, 95])
            x_vals = np.linspace(low, high, 100)
            density = kde(x_vals)
            density = density / density.max() * 0.25
            # Shift density to the right of box
            ax[0].plot(positions[j] + density, x_vals)

        ax[0].set_xticks(positions)
        ax[0].set_xticklabels(['UAR', 'VAR', 'GNAR', 'OU', 'MCAR', 'grOU', 'Baseline'])
        ax[0].set_ylabel('RMSE')

        positions = np.arange(1, 7)
        ax[1].boxplot(direction_acc, positions=positions, widths=0.3)

        for k in range(6):
            kde = gaussian_kde(direction_acc[:, k], bw_method=0.4)
            low, high = np.percentile(direction_acc[:, k], [5, 95])
            x_vals = np.linspace(low, high, 100)
            density = kde(x_vals)
            density = density / density.max() * 0.25
            # Shift density to the right of box
            ax[1].plot(positions[k] + density, x_vals)

        ax[1].set_xticks(positions)
        ax[1].set_xticklabels(['UAR', 'VAR', 'GNAR', 'OU', 'MCAR', 'grOU'])
        ax[1].set_ylabel('Directional accuracy')

        positions = np.arange(1, 7)
        ax[2].boxplot(comp_time, positions=positions, widths=0.3)

        for l in range(6):
            kde = gaussian_kde(comp_time[:, l], bw_method=0.4)
            low, high = np.percentile(comp_time[:, l], [5, 95])
            x_vals = np.linspace(low, high, 100)
            density = kde(x_vals)
            density = density / density.max() * 0.25
            # Shift density to the right of box
            ax[2].plot(positions[l] + density, x_vals)

        ax[2].set_xticks(positions)
        ax[2].set_xticklabels(['UAR', 'VAR', 'GNAR', 'OU', 'MCAR', 'grOU'])
        ax[2].set_ylabel('Computational Time')

        plt.show()
        fig.savefig('./data/LOBSTERdata/output/sim/' + scenario + '/boxplot_' + label + '.eps', bbox_inches='tight',
                    format='eps')

        df = pd.DataFrame(
            [np.mean(RMSE, axis=0), np.std(RMSE, axis=0), np.mean(direction_acc, axis=0), np.std(direction_acc, axis=0),
             np.mean(comp_time, axis=0), np.std(comp_time, axis=0)],
            columns=['UAR', 'VAR', 'GNAR', 'OU', 'MCAR', 'grOU', 'NA'],
            index=['RMSE_mean', 'RMSE_std', 'Directional_accuracy_mean', 'Directional_accuracy_std',
                   'Computational_time_mean', 'Computational_time_std'])
        print(label)
        print(df)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    positions = np.arange(1, 4)
    ax[0].boxplot(grOU_RMSE[:, :], positions=positions, widths=0.3, showfliers=False)

    for i in range(3):
        kde = gaussian_kde(grOU_RMSE[:, i], bw_method=0.4)
        low, high = np.percentile(grOU_RMSE[:, i], [5, 95])
        x_vals = np.linspace(low, high, 100)
        density = kde(x_vals)
        density = density / density.max() * 0.25
        # Shift density to the right of box
        ax[0].plot(positions[i] + density, x_vals)

        ax[0].set_xticks(positions)
        ax[0].set_xticklabels(['correct', 'missing_5', 'missing_1'])
        ax[0].set_ylabel('RMSE')

    positions = np.arange(1, 4)
    ax[1].boxplot(grOU_direction_acc, positions=positions, widths=0.3)

    for i in range(3):
        kde = gaussian_kde(grOU_direction_acc[:, i], bw_method=0.4)
        low, high = np.percentile(grOU_direction_acc[:, i], [5, 95])
        x_vals = np.linspace(low, high, 100)
        density = kde(x_vals)
        density = density / density.max() * 0.25
        # Shift density to the right of box
        ax[1].plot(positions[i] + density, x_vals)

        ax[1].set_xticks(positions)
        ax[1].set_xticklabels(['correct', 'missing_5', 'missing_1'])
        ax[1].set_ylabel('Directional accuracy')

    plt.show()
    fig.savefig('./data/LOBSTERdata/output/sim/' + scenario + '/boxplot_grOU.eps', bbox_inches='tight', format='eps')

def evaluate_forecast(Y, forecast, test_set):
    rmse = np.sqrt(np.mean((forecast - test_set) ** 2))
    last_ = np.append(np.array(Y.copy())[:, -1].reshape(-1, 1)[np.newaxis, :, :], test_set[:-1, :, 0:1], axis=0)
    forecast_dir = np.sign(np.diff(np.append(last_, forecast, axis=2), axis=2))
    test_dir = np.sign(np.diff(np.append(last_, test_set, axis=2), axis=2))
    acc = 0.5 if (last_==forecast).all() else (test_dir == forecast_dir).mean()
    return rmse, acc

def UAR_model(Y, test_set, test_size, forecast_step, lag=None, save=False):
    # Univariate AR Edge Model

    K, T_train = Y.shape
    total_T = T_train + test_size

    data = np.empty((K, total_T))
    data[:, :T_train] = Y
    data[:, T_train:] = test_set[:, :, 0].T

    forecast_AR = np.zeros((test_size, K, forecast_step))

    # -------------------------------------------------
    # Choose one common lag if lag is None
    # -------------------------------------------------
    if lag is None:
        mean_aic_by_p = []

        for p in range(1, 13):
            aics = []
            for j in range(K):
                try:
                    fit = AutoReg(data[j, :T_train], lags=p, old_names=False).fit()
                    aics.append(fit.aic)
                except Exception:
                    pass

            if len(aics) == 0:
                mean_aic_by_p.append((p, np.inf))
            else:
                mean_aic_by_p.append((p, float(np.mean(aics))))

        p = min(mean_aic_by_p, key=lambda x: x[1])[0]
    else:
        p = int(lag)

    n_rows = T_train - p
    if n_rows <= 0:
        raise ValueError(f"lag={p} too large for T_train={T_train}")

    q = p + 1
    n_total_rows = total_T - p

    # -------------------------------------------------
    # Build all regression rows for all nodes at once
    # X_all shape: (K, n_total_rows, q)
    # y_all shape: (K, n_total_rows)
    # -------------------------------------------------
    X_all = np.empty((K, n_total_rows, q))
    X_all[:, :, 0] = 1.0

    for s in range(1, p + 1):
        X_all[:, :, s] = data[:, p - s: total_T - s]

    y_all = data[:, p: total_T]

    # -------------------------------------------------
    # Initial rolling window cross-products
    # XtX shape: (K, q, q)
    # Xty shape: (K, q)
    # -------------------------------------------------
    X0 = X_all[:, :n_rows, :]      # (K, n_rows, q)
    y0 = y_all[:, :n_rows]         # (K, n_rows)

    XtX = np.einsum('knq,knr->kqr', X0, X0)
    Xty = np.einsum('knq,kn->kq', X0, y0)

    # -------------------------------------------------
    # Rolling windows
    # -------------------------------------------------
    for i in range(test_size):
        # Solve all node regressions in batch
        beta = np.linalg.solve(XtX, Xty[..., None]).squeeze(-1)   # (K, q)

        # Forecast recursively for all nodes
        hist = np.empty((K, T_train + forecast_step))
        hist[:, :T_train] = data[:, i:T_train + i]

        for s in range(forecast_step):
            # Build batched predictor rows for all nodes
            x = np.empty((K, q))
            x[:, 0] = 1.0
            for ell in range(1, p + 1):
                x[:, ell] = hist[:, T_train + s - ell]

            y_next = np.einsum('kq,kq->k', x, beta)
            hist[:, T_train + s] = y_next
            forecast_AR[i, :, s] = y_next

        # Rolling update
        if i < test_size - 1:
            x_out = X_all[:, i, :]             # (K, q)
            y_out = y_all[:, i]                # (K,)
            x_in = X_all[:, i + n_rows, :]     # (K, q)
            y_in = y_all[:, i + n_rows]        # (K,)

            XtX -= np.einsum('kq,kr->kqr', x_out, x_out)
            XtX += np.einsum('kq,kr->kqr', x_in, x_in)

            Xty -= x_out * y_out[:, None]
            Xty += x_in * y_in[:, None]

    if save:
        with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', 'forecast_AR.npy'), 'wb') as f:
            np.save(f, forecast_AR)

    return forecast_AR

def VAR_model(Y, test_set, test_size, forecast_step, lag=None, save=False):
    # VAR Edge Model

    K, T_train = Y.shape
    total_T = T_train + test_size

    df_VAR = np.empty((total_T, K))
    df_VAR[:T_train, :] = Y.T
    df_VAR[T_train:, :] = test_set[:, :, 0]

    forecast_VAR = np.zeros((test_size, K, forecast_step))

    # -----------------------------------------
    # Choose lag once on initial training sample
    # -----------------------------------------
    if lag is None:
        model0 = VAR(df_VAR[:T_train, :])
        best_lag_var = model0.select_order(maxlags=12).aic
        p = int(best_lag_var)
    else:
        p = int(lag)

    n_rows = T_train - p
    if n_rows <= 0:
        raise ValueError(f"lag={p} too large for T_train={T_train}")

    q = 1 + p * K
    n_total_rows = total_T - p

    # -----------------------------------------
    # Build all regression rows once
    # X_g = [1, y_{g+p-1}', y_{g+p-2}',..., y_g']
    # target_g = y_{g+p}
    # -----------------------------------------
    X_all = np.empty((n_total_rows, q))
    X_all[:, 0] = 1.0

    col = 1
    for s in range(1, p + 1):
        X_all[:, col:col + K] = df_VAR[p - s:total_T - s, :]
        col += K

    Y_all = df_VAR[p:total_T, :]  # shape (n_total_rows, K)

    # Initial rolling window corresponds to rows [0 : n_rows)
    X0 = X_all[:n_rows, :]
    Y0 = Y_all[:n_rows, :]

    XtX = X0.T @ X0  # shape (q, q)
    XtY = X0.T @ Y0  # shape (q, K)

    # -----------------------------------------
    # Rolling windows
    # -----------------------------------------
    for i in range(test_size):
        # Fit coefficients for current window
        B = np.linalg.solve(XtX, XtY)  # shape (q, K)

        # Forecast recursively
        hist = np.empty((T_train + forecast_step, K))
        hist[:T_train, :] = df_VAR[i:T_train + i, :]

        preds = np.zeros((forecast_step, K))

        for s in range(forecast_step):
            x = np.empty(q)
            x[0] = 1.0
            x[1:] = hist[T_train + s - p:T_train + s, :][::-1, :].reshape(-1)

            y_next = x @ B
            preds[s, :] = y_next
            hist[T_train + s, :] = y_next

        forecast_VAR[i, :, :] = preds.T

        # Update XtX, XtY for next rolling window
        if i < test_size - 1:
            x_out = X_all[i, :]  # outgoing row
            y_out = Y_all[i, :]  # outgoing target

            x_in = X_all[i + n_rows, :]  # incoming row
            y_in = Y_all[i + n_rows, :]  # incoming target

            XtX -= np.outer(x_out, x_out)
            XtX += np.outer(x_in, x_in)

            XtY -= np.outer(x_out, y_out)
            XtY += np.outer(x_in, y_in)

    if save:
        with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', 'forecast_VAR.npy'), 'wb') as f:
            np.save(f, forecast_VAR)

    return forecast_VAR

def GNAR_model(Y, Adj, R_L, test_set, test_size, forecast_step, is_global = False, save=False):
    # GNAR Edge Model

    p = len(R_L)
    K, T_train = Y.shape
    total_T = T_train + test_size
    n_rows = T_train - p

    if n_rows <= 0:
        raise ValueError(f"Need p < T_train, got p={p}, T_train={T_train}")

    # -------------------------------------------------
    # Full data buffer: time x node
    # -------------------------------------------------
    data = np.empty((total_T, K), dtype=np.float64)
    data[:T_train, :] = Y.T
    data[T_train:, :] = test_set[:, :, 0]

    # -------------------------------------------------
    # Graph weights
    # -------------------------------------------------
    max_r = max(R_L) if len(R_L) > 0 else 0
    Adj_L = np.array(adjacency_matrices(Adj[0], max_r))
    W_C_raw = edge_weight(Adj_L)

    W_C = []
    for w in W_C_raw:
        W_C.append(np.asarray(w, dtype=np.float64))

    # -------------------------------------------------
    # Precompute lag blocks over the whole timeline
    #
    # lag_all[l-1, g, k] = data[p-l+g, k]
    # target_all[g, k]   = data[p+g, k]
    #
    # g = 0,..., total_T-p-1
    # -------------------------------------------------
    n_total_rows = total_T - p

    lag_all = np.empty((p, n_total_rows, K), dtype=np.float64)
    for l in range(1, p + 1):
        lag_all[l - 1] = data[p - l: total_T - l, :]

    target_all = data[p: total_T, :]  # (n_total_rows, K)

    # -------------------------------------------------
    # Precompute all neighbor blocks
    # neigh_all[l][r] shape = (n_total_rows, K)
    # -------------------------------------------------
    neigh_all = []
    for l in range(p):
        lag_block = lag_all[l]
        neigh_l = []
        for r in range(R_L[l]):
            neigh_l.append(lag_block @ W_C[r].T)
        neigh_all.append(neigh_l)

    # -------------------------------------------------
    # Parameter indexing
    # -------------------------------------------------
    if is_global:
        q_alpha = p
    else:
        q_alpha = p * K

    q_beta = sum(R_L)
    q = q_alpha + q_beta

    beta_offsets = np.cumsum([0] + R_L[:-1])
    beta_base = q_alpha

    forecast_GNAR = np.zeros((test_size, K, forecast_step), dtype=np.float64)

    # -------------------------------------------------
    # Build per-row feature vectors Phi[g]
    #
    # For global:
    #   phi_g = [
    #       sum_k Z[0,g,k]?  NO
    #   ]
    #
    # Careful:
    # We are fitting stacked equations across nodes.
    # So row g contributes K equation rows.
    #
    # Instead of materializing all K rows, we build sufficient-stat contributions
    # per time g directly:
    #
    # XtX += Phi_g
    # Xty += psi_g
    #
    # where Phi_g already includes summation over nodes.
    # -------------------------------------------------

    def row_contribution(g):
        """
        Returns (Sxx_g, Sxy_g) contribution of time row g,
        already summed over nodes/equations.
        """
        Sxx_g = np.zeros((q, q), dtype=np.float64)
        Sxy_g = np.zeros(q, dtype=np.float64)
        y_g = target_all[g]  # (K,)

        if is_global:
            # alpha features: each lag contributes full vector over nodes
            alpha_blocks = [lag_all[l, g] for l in range(p)]  # each (K,)
            beta_blocks = []
            for l in range(p):
                for r in range(R_L[l]):
                    beta_blocks.append(neigh_all[l][r][g])  # each (K,)

            # alpha-target
            for l in range(p):
                Sxy_g[l] = np.dot(alpha_blocks[l], y_g)

            # alpha-alpha
            for l in range(p):
                for m in range(l, p):
                    val = np.dot(alpha_blocks[l], alpha_blocks[m])
                    Sxx_g[l, m] = val
                    Sxx_g[m, l] = val

            # beta-target
            for b, Nb in enumerate(beta_blocks):
                idxb = beta_base + b
                Sxy_g[idxb] = np.dot(Nb, y_g)

            # beta-beta
            for b1, N1 in enumerate(beta_blocks):
                idx1 = beta_base + b1
                for b2 in range(b1, len(beta_blocks)):
                    idx2 = beta_base + b2
                    val = np.dot(N1, beta_blocks[b2])
                    Sxx_g[idx1, idx2] = val
                    Sxx_g[idx2, idx1] = val

            # alpha-beta
            for l in range(p):
                for b, Nb in enumerate(beta_blocks):
                    idxb = beta_base + b
                    val = np.dot(alpha_blocks[l], Nb)
                    Sxx_g[l, idxb] = val
                    Sxx_g[idxb, l] = val

        else:
            # Node-specific alpha
            # alpha coordinates are ordered by lag-major:
            # [alpha_(l=1,k=1..K), alpha_(l=2,k=1..K),...]
            #
            # alpha-target
            for l in range(p):
                start = l * K
                z = lag_all[l, g]  # (K,)
                Sxy_g[start:start + K] = z * y_g

            # alpha-alpha
            # only same node survives, so blocks are diagonal
            for l in range(p):
                z_l = lag_all[l, g]
                sl = l * K
                for m in range(p):
                    z_m = lag_all[m, g]
                    sm = m * K
                    Sxx_g[sl:sl + K, sm:sm + K] = np.diag(z_l * z_m)

            # beta blocks
            beta_blocks = []
            for l in range(p):
                for r in range(R_L[l]):
                    beta_blocks.append(neigh_all[l][r][g])  # (K,)

            # beta-target
            for b, Nb in enumerate(beta_blocks):
                idxb = beta_base + b
                Sxy_g[idxb] = np.dot(Nb, y_g)

            # beta-beta
            for b1, N1 in enumerate(beta_blocks):
                idx1 = beta_base + b1
                for b2 in range(b1, len(beta_blocks)):
                    idx2 = beta_base + b2
                    val = np.dot(N1, beta_blocks[b2])
                    Sxx_g[idx1, idx2] = val
                    Sxx_g[idx2, idx1] = val

            # alpha-beta
            for l in range(p):
                sl = l * K
                z_l = lag_all[l, g]  # (K,)
                for b, Nb in enumerate(beta_blocks):
                    idxb = beta_base + b
                    val_vec = z_l * Nb
                    Sxx_g[sl:sl + K, idxb] = val_vec
                    Sxx_g[idxb, sl:sl + K] = val_vec

        return Sxx_g, Sxy_g

    # -------------------------------------------------
    # Initial sufficient statistics
    # -------------------------------------------------
    XtX = np.zeros((q, q), dtype=np.float64)
    Xty = np.zeros(q, dtype=np.float64)

    row_cache_xx = []
    row_cache_xy = []

    for g in range(n_rows):
        Sxx_g, Sxy_g = row_contribution(g)
        row_cache_xx.append(Sxx_g)
        row_cache_xy.append(Sxy_g)
        XtX += Sxx_g
        Xty += Sxy_g

    # -------------------------------------------------
    # Rolling windows
    # -------------------------------------------------
    for i in range(test_size):
        theta = np.linalg.solve(XtX, Xty)

        # Forecast recursively
        hist = np.empty((T_train + forecast_step, K), dtype=np.float64)
        hist[:T_train, :] = data[i:T_train + i, :]
        preds = np.zeros((forecast_step, K), dtype=np.float64)

        for s in range(forecast_step):
            x_next = np.zeros(K, dtype=np.float64)

            # alpha part
            if is_global:
                for l in range(1, p + 1):
                    x_next += theta[l - 1] * hist[T_train + s - l, :]
            else:
                for l in range(1, p + 1):
                    alpha_vec = theta[(l - 1) * K:l * K]
                    x_next += alpha_vec * hist[T_train + s - l, :]

            # beta part
            flat_beta_idx = 0
            for l in range(1, p + 1):
                x_lag = hist[T_train + s - l, :]
                for r in range(R_L[l - 1]):
                    idxb = beta_base + flat_beta_idx
                    x_next += theta[idxb] * (W_C[r] @ x_lag)
                    flat_beta_idx += 1

            preds[s, :] = x_next
            hist[T_train + s, :] = x_next

        forecast_GNAR[i, :, :] = preds.T

        # rolling update
        if i < test_size - 1:
            # outgoing row index = i
            XtX -= row_cache_xx[i]
            Xty -= row_cache_xy[i]

            # incoming row index = i + n_rows
            g_in = i + n_rows
            Sxx_in, Sxy_in = row_contribution(g_in)
            row_cache_xx.append(Sxx_in)
            row_cache_xy.append(Sxy_in)

            XtX += Sxx_in
            Xty += Sxy_in

    if save:
        with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', 'forecast_GNAR.npy'), 'wb') as f:
            np.save(f, forecast_GNAR)

    return forecast_GNAR

def NA_model(Y, test_set, forecast_step):
    # Naive Model

    forecast_NA = np.repeat(np.append(np.array(Y.copy())[:, -1].reshape(-1, 1)[np.newaxis, :, :], test_set[:-1, :, 0:1], axis=0), forecast_step, axis=2)
    RMSE_NA = np.sqrt(np.mean((forecast_NA - test_set) ** 2))

    return forecast_NA, RMSE_NA

def MCAR_edge_model(Y, test_set, test_size, forecast_step, P_, Q_, R_L, C, gamma, method='naive', save=False):
    # MCAR edge model

    K, T_train = Y.shape
    forecast_MCAR = np.zeros((test_size, K, forecast_step))
    A = np.array([np.zeros((K, K)) for _ in range(max(R_L))])
    W_C = edge_weight(A)
    df_MCAR = np.empty((K, T_train + test_size))
    df_MCAR[:, :T_train] = Y
    df_MCAR[:, T_train:] = test_set[:, :, 0].T
    _, (b_MCAR, Sigma_MCAR) = estimate_MCAR('MCAR', Y, P_, Q_, A, W_C, R_L, C, gamma)

    for i in range(test_size):
        window_data = df_MCAR[:, i:T_train + i]
        QQ_hat = estimate_MCAR_drift(Y, R_L, P_, Q_, b_MCAR, gamma, Sigma = Sigma_MCAR)
        Q_Q = MCAR_Q(QQ_hat)

        if len(R_L) == 1:
            x0 = window_data[:, -1]
        else:
            if method == 'backward':
                # Backward - differences
                flip_window = state_space(np.flip(window_data[:, -len(R_L):], axis=1), len(R_L), P_[-len(R_L):])
                x0 = flip_window[:, 0] * np.repeat((-1) ** np.arange(len(R_L)), flip_window.shape[0] / len(R_L))
            elif method == 'naive':
                # Naive initialization
                x0 =np.append(window_data[:, -1], [0] * K * (len(R_L) - 1))
            elif method == 'kalman':
                # Kalman - filter
                x0 = kalman_filter_mcar_BM(window_data, Q_Q, Sigma_MCAR, P_, Q_, QQ_hat, gamma)
                x0[:K] = window_data[:, -1]
            else:
                raise NotImplementedError("method must be among 'backward', 'naive' or 'kalman'.")

        # Make prediction
        # Compute noise coefficient E, Sigma_tilde
        E = np.zeros([K * len(R_L), K])
        E[-K:, :] = np.eye(K)

        # Initialize
        X = np.zeros([K * len(R_L), forecast_step + 1])
        X[:, 0] = x0

        # Precompute for speed-up
        eQ = scipy.linalg.expm(Q_Q)
        try:
            Q_inv = Q_QQ_inverse(Q_Q, K)
        except:
            Q_inv = np.linalg.pinv(Q_Q)
        eQt = scipy.linalg.fractional_matrix_power(eQ, P_[1] - P_[0]).real

        for n in range(forecast_step):
            X[:, n + 1] = eQt.dot(X[:, n]) + (eQt - np.eye(K * len(R_L))).dot(Q_inv).dot(E).dot(b_MCAR)
        forecast_MCAR[i, :, :] = np.array(X[:K, 1:])
    if save:
        with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', f'forecast_MCAR_{len(R_L)}_{method}.npy'), 'wb') as f:
            np.save(f, forecast_MCAR)

    return forecast_MCAR

def OU_edge_model(Y, test_set, test_size, forecast_step, R_L, C, P_, Q_, gamma, method = 'naive',
                  is_global = False, print_param = False, is_test=False, save=False):
    # OU Edge model

    K, T_train = Y.shape
    forecast_OU = np.zeros((test_size, K, forecast_step))
    df_OU = np.empty((K, T_train + test_size))
    df_OU[:, :T_train] = Y
    df_OU[:, T_train:] = test_set[:, :, 0].T
    Adj_L_OU = np.array([np.zeros((K, K)) for _ in range(max(R_L))])
    W_C_OU = edge_weight(Adj_L_OU)
    X_OU = GraphOUEdge(R_L, K, C, Adj_L_OU, W_C_OU, gamma, 'data\\LOBSTERdata\\output\\train\\aggregate')
    _, (b_OU, Sigma_OU) = estimate_MCAR('OU', Y, P_, Q_, Adj_L_OU, W_C_OU, R_L, C, gamma, is_global=is_global)
    alphas_OU = []

    if is_test:
        test_size = 1

    for i in range(test_size):
        window_data = df_OU[:, i:T_train + i]
        theta_hat_OU, _, _ = estimate_OU_drift(window_data, R_L, C, P_, Q_, b_OU, gamma, None, Sigma_OU, with_cov=True, is_global=is_global)
        X_OU.alpha_set(theta_hat_OU)
        X_OU.beta_set([[0] * stage * C for stage in R_L])
        Q_Q = X_OU.Q_matrix(False)
        alphas_OU.append(theta_hat_OU)

        if len(R_L) == 1:
            x0 = window_data[:, -1]
        else:
            if method == 'backward':
                # Backward - differences
                flip_window = state_space(np.flip(window_data[:, -len(R_L):], axis=1), len(R_L), P_[-len(R_L):])
                x0 = flip_window[:, 0] * np.repeat((-1) ** np.arange(len(R_L)), flip_window.shape[0] / len(R_L))
            elif method == 'naive':
                # Naive initialization
                x0 =np.append(window_data[:, -1], [0] * K * (len(R_L) - 1))
            elif method == 'kalman':
                # Kalman - filter
                QQ_hat = [np.diag([x] * K) for x in theta_hat_OU]
                x0 = kalman_filter_mcar_BM(window_data, Q_Q, Sigma_OU, P_, Q_, QQ_hat, gamma)
                x0[:K] = window_data[:, -1]
            else:
                raise NotImplementedError("method must be among 'backward', 'naive' or 'kalman'.")

        # Make prediction
        # Compute noise coefficient E, Sigma_tilde
        E = np.zeros([K * len(R_L), K])
        E[-K:, :] = np.eye(K)

        # Initialize
        X = np.zeros([K * len(R_L), forecast_step + 1])
        X[:, 0] = x0

        # Precompute for speed-up
        eQ = scipy.linalg.expm(Q_Q)
        try:
            Q_inv = Q_QQ_inverse(Q_Q, K)
        except:
            Q_inv = np.linalg.pinv(Q_Q)
        eQt = scipy.linalg.fractional_matrix_power(eQ, P_[1] - P_[0]).real

        for n in range(forecast_step):
            X[:, n + 1] = eQt.dot(X[:, n]) + (eQt - np.eye(K * len(R_L))).dot(Q_inv).dot(E).dot(b_OU)
        forecast_OU[i, :, :] = np.array(X[:K, 1:])
    if save:
        with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', f'forecast_OU_{len(R_L)}_{method}_global_{is_global}.npy'), 'wb') as f:
            np.save(f, forecast_OU)

    if print_param:
        return np.array(alphas_OU)
    else:
        return forecast_OU

def grOU_edge_model(Y, Adj, test_set, test_size, forecast_step, R_L, C, P_, Q_, gamma, method = 'naive',
                    is_global=False, print_param=False, is_test=False, save=False):
    # graph OU Edge model

    K, T_train = Y.shape
    # Derive the adjacency matrix for edge graph
    Adj_L = np.array(adjacency_matrices(Adj[0], max(R_L)))

    # Compute the edge weight
    W_C = edge_weight(Adj_L)

    X = GraphOUEdge(R_L, K, C, Adj_L, W_C, gamma, 'data\\LOBSTERdata\\output\\train\\aggregate')
    _, (b, Sigma) = estimate_MCAR('grOU', Y, P_, Q_, Adj_L, W_C, R_L, C, gamma, is_global=is_global)
    forecast_grOU = np.zeros((test_size, K, forecast_step))
    df_grOU = np.empty((K, T_train + test_size))
    df_grOU[:, :T_train] = Y
    df_grOU[:, T_train:] = test_set[:, :, 0].T
    alphas = []
    betas = []

    if is_test:
        test_size = 1

    for i in range(test_size):
        window_data = df_grOU[:, i:T_train + i]
        theta_hat, KK, _ = estimate_grOU_drift(window_data, Adj_L, W_C, R_L, C, P_, Q_, b, gamma, None, Sigma, with_cov=True, is_global=is_global)

        # Fetch parameters
        idx = 0
        alpha_pred = []
        beta_pred = []
        for r in R_L:
            alpha_pred.append(theta_hat[idx : idx + K])
            beta_pred.append(theta_hat[(idx + K):(idx + K + r * C)])
            idx += K + r * C
        X.alpha_set(np.array(alpha_pred))
        X.beta_set(beta_pred)
        Q_Q = X.Q_matrix(False)
        if print_param:
            alphas.append(alpha_pred)
            betas.append(beta_pred)

        if len(R_L) == 1:
            x0 = window_data[:, -1]
        else:
            if method == 'backward':
                # Backward - differences
                flip_window = state_space(np.flip(window_data[:, -len(R_L):], axis=1), len(R_L), P_[-len(R_L):])
                x0 = flip_window[:, 0] * np.repeat((-1) ** np.arange(len(R_L)), flip_window.shape[0] / len(R_L))
            elif method == 'naive':
                # Naive initialization
                x0 =np.append(window_data[:, -1], [0] * K * (len(R_L) - 1))
            elif method == 'kalman':
                # Kalman - filter
                QQ_hat = estimator_to_QQ(theta_hat, R_L, C, W_C, Adj_L, K)
                x0 = kalman_filter_mcar_BM(window_data, Q_Q, Sigma, P_, Q_, QQ_hat, gamma)
                x0[:K] = window_data[:, -1]
            else:
                raise NotImplementedError("method must be among 'backward', 'naive' or 'kalman'.")

        # Make prediction
        # Compute noise coefficient E, Sigma_tilde
        E = np.zeros([K * len(R_L), K])
        E[-K:, :] = np.eye(K)

        # Initialize
        X_ = np.zeros([K * len(R_L), forecast_step + 1])
        X_[:, 0] = x0

        # Precompute for speed-up
        eQ = scipy.linalg.expm(Q_Q)
        try:
            Q_inv = Q_QQ_inverse(Q_Q, K)
        except:
            Q_inv = np.linalg.pinv(Q_Q)
        eQt = scipy.linalg.fractional_matrix_power(eQ, P_[1] - P_[0]).real

        for n in range(forecast_step):
            X_[:, n + 1] = eQt.dot(X_[:, n]) + (eQt - np.eye(K * len(R_L))).dot(Q_inv).dot(E).dot(b)
        forecast_grOU[i, :, :] = np.array(X_[:K, 1:])
    if save:
        with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', f'forecast_grOU_{len(R_L)}_{R_L}_{method}_global_{is_global}.npy'), 'wb') as f:
            np.save(f, forecast_grOU)

    if print_param:
       return alphas, betas, -theta_hat.T @ KK @ theta_hat + (len(R_L) * K + sum(R_L)) * np.log(len(Q_))
    else:
       return forecast_grOU

def is_better(candidate, incumbent, tol=1e-4):
    """
    Compare two results of form:
    [rate_abs_diff, A, rate_rel_diff]
    """
    if incumbent is None:
        return True

    cand_abs, _, cand_rel = candidate
    inc_abs, _, inc_rel = incumbent

    if cand_abs > inc_abs + tol:
        return True
    elif abs(cand_abs - inc_abs) <= tol and cand_rel > inc_rel:
        return True
    return False

def find_best_network(_data, test_data_, TICKERS, RL, P_, Q_, forecast_step, C, gamma, n_candidates=100, tol=1e-4):
    """
    Search candidate networks and return the best one under a fixed RL.
    """
    best_result = None

    arg_list = [(_data, test_data_, TICKERS, RL, P_, Q_, forecast_step, C, gamma, i) for i in range(n_candidates)]

    with Pool(os.cpu_count()) as pool:
        for candidate in pool.imap_unordered(worker, arg_list):
            candidate = list(candidate)
            if is_better(candidate, best_result, tol=tol):
                best_result = candidate

    return best_result

def select_best_model_for_network(_data, test_data_, TICKERS, A, P_, Q_, forecast_step, C, gamma, max_order=5):
    """
    Given a network A, select the best RL via BIC.
    """
    Adj = adj_to_line_graph(A, TICKERS)
    K = Adj[0].shape[0]

    # Training data
    Y = _data[Adj[1]].T

    # Testing data
    test_Y = test_data_[Adj[1]].T
    test_set = sliding_window_view(test_Y, (K, forecast_step))[0]
    test_size = test_set.shape[0]

    models = [[value] * length for length in range(1, max_order + 1) for value in (1, 2)]

    bic = [grOU_edge_model(Y, Adj, test_set, test_size, forecast_step, model, C, P_, Q_, gamma, is_global=False, print_param=True, is_test=True)[2] for model in models]

    best_idx = min(range(len(bic)), key=bic.__getitem__)
    best_model = models[best_idx]
    best_bic = bic[best_idx]

    return best_model, best_bic, Adj

def iterative_grou_search_with_rollback(_data, test_data_, TICKERS, start_RL, P_, Q_, forecast_step, C, gamma, n_candidates=100, max_order=5, max_iter=20, tol=1e-4):
    """
    Iterative search:
    - search best network for current RL
    - use BIC to propose new RL
    - accept new RL only if it improves out-of-sample performance
    - otherwise revert to previous RL and stop
    """
    current_RL = start_RL.copy()

    # Baseline result under starting RL
    current_result = find_best_network(_data, test_data_, TICKERS, current_RL, P_, Q_, forecast_step, C, gamma, n_candidates=n_candidates, tol=tol)
    current_A = current_result[1]

    history = [{
        "iteration": 0,
        "RL": current_RL.copy(),
        "result": current_result,
        "accepted": True,
        "note": "initial model"
    }]

    print(f"Initial RL: {current_RL}")
    print(f"Initial best result: {current_result}")

    for iteration in range(1, max_iter + 1):
        print(f"\n===== Iteration {iteration} =====")
        print(f"Current RL: {current_RL}")

        # Step 4: BIC proposes a new RL using current best network
        proposed_RL, best_bic, Adj = select_best_model_for_network(_data, test_data_, TICKERS, current_A, P_, Q_, forecast_step, C, gamma, max_order=max_order)

        print(f"BIC proposed RL: {proposed_RL}, BIC={best_bic}")

        # If BIC gives the same RL, stop
        if proposed_RL == current_RL:
            print("Stop: BIC selected the same RL.")
            history.append({
                "iteration": iteration,
                "RL": current_RL.copy(),
                "result": current_result,
                "accepted": False,
                "note": "BIC selected same RL; stop"
            })
            break

        # Evaluate the proposed RL by re-running network search
        proposed_result = find_best_network(_data, test_data_, TICKERS, proposed_RL, P_, Q_, forecast_step, C, gamma, n_candidates=n_candidates, tol=tol)

        print(f"Proposed RL result: {proposed_result}")
        print(f"Current RL result:  {current_result}")

        # Accept only if proposed RL improves performance
        if is_better(proposed_result, current_result, tol=tol):
            print("Accept new RL: out-of-sample performance improved.")
            current_RL = proposed_RL
            current_result = proposed_result
            current_A = proposed_result[1]

            history.append({
                "iteration": iteration,
                "RL": current_RL.copy(),
                "result": current_result,
                "accepted": True,
                "note": "accepted BIC-proposed RL"
            })
        else:
            print("Reject new RL: no improvement. Revert to previous RL and stop.")
            history.append({
                "iteration": iteration,
                "RL": proposed_RL.copy() if isinstance(proposed_RL, list) else proposed_RL,
                "result": proposed_result,
                "accepted": False,
                "note": "rejected; reverted to previous RL"
            })
            break

    return {
        "final_RL": current_RL,
        "final_network": current_A,
        "final_result": current_result,
        "history": history,
    }