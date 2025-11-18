import os
import glob
import math
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    :return: saves the aggregated mid quote price in output_path, as .npz files with numpy attributes
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
    :return: saves the aggregated mid quote price in output_path, as .npz files with numpy attributes
             "orderbook_features" (:, 1).
    """
    print(orderbook_name)
    log = ''

    ### load LOBSTER orderbook and message files
    try:
        df_orderbook = pd.read_csv(orderbook_name, header=None)
    except:
        return orderbook_name + ' skipped. Error: failed to read orderbook.'

    # compute the mid quote price
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

    # 6. downsample tick-by-tick mid quote price data into 1-second frequency, keeping only the last tick in each second with forward-fill
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
    Aggregate log mid quote price for multiple assets and days using full parallelization.
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

    Parameters
    ----------
    Y : np.ndarray
        Log-price series, shape (n, d).

    Returns
    -------
    C : np.ndarray
        Estimated covariance matrix, shape (d, d).
    """
    assets = Y.columns
    Y = np.asarray(Y)
    n, d = Y.shape
    if n == 0:
        return

    N = math.ceil((n - 1) ** .6 * 1)
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

    # Compute the correlation between assets as the value of edges
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

    return cov_data


def UAR_model(Y, K, test_size, forecast_step):
    # Univariate AR Edge Model

    df_AR = np.array(Y.copy())
    aic_values = []
    best_p = np.zeros((K,))
    for i in range(K):
        for p in range(1, 6):
            model = AutoReg(df_AR[i, :], lags=p).fit()
            aic_values.append((p, model.aic))
        best_p[i] = min(aic_values, key=lambda x: x[1])[0]

    forecast_AR = np.zeros((test_size, K, forecast_step))
    for i in range(test_size):
        for j in range(K):
            window_data = df_AR[j, i:]
            model = AutoReg(window_data, lags=best_p[j]).fit()
            forecast_AR[i, j, :] = model.predict(start=len(window_data), end=len(window_data) + forecast_step - 1)
        df_AR = np.append(df_AR, forecast_AR[i, :, 0:1], axis=1)

    with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', 'forecast_AR.npy'), 'wb') as f:
        np.save(f, forecast_AR)


def VAR_model(Y, K, test_size, forecast_step):
    # VAR Edge Model

    df_VAR = np.array(Y.T.copy())
    model = VAR(df_VAR)
    # automatic lag selection by AIC
    best_lag_var = model.select_order(maxlags=5).aic

    forecast_VAR = np.zeros((test_size, K, forecast_step))
    for i in range(test_size):
        window_data = df_VAR[i:, :]
        model = VAR(window_data).fit(best_lag_var)
        forecast_VAR[i, :, :] = model.forecast(window_data[-model.k_ar:, :], steps=forecast_step).T
        df_VAR = np.append(df_VAR, forecast_VAR[i, :, 0:1].T, axis=0)

    with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', 'forecast_VAR.npy'), 'wb') as f:
        np.save(f, forecast_VAR)

def OU_model(Y, K, test_size, forecast_step, R_L, C, jump_activity, P_, Q_):
    # OU Edge model

    forecast_OU = np.zeros((test_size, K, forecast_step))
    df_OU = np.array(Y.copy())
    Adj_L_OU = [np.zeros((K, K)) for _ in range(max(R_L))]
    W_C_OU = edge_weight(Adj_L_OU)
    X_OU = GraphOUEdge(R_L, K, C, Adj_L_OU, W_C_OU, 'data\\LOBSTERdata\\output\\train\\aggregate', 'plots')
    X_OU.estimate_unknown_Levy(Y, f'{jump_activity}')
    b_OU = np.load(f'./data/LOBSTERdata/output/train/aggregate/MCAR_estimates_b_{jump_activity}.npy')
    Sigma_OU = np.load(f'./data/LOBSTERdata/output/train/aggregate/MCAR_estimates_Sigma_{jump_activity}.npy')
    F_mean_OU = np.load(f'./data/LOBSTERdata/output/train/aggregate/MCAR_estimates_F_mean_{jump_activity}.npy')
    F_squared_OU = np.load(f'./data/LOBSTERdata/output/train/aggregate/MCAR_estimates_F_squared_{jump_activity}.npy')

    jump_F_OU = scipy.stats.multivariate_normal(mean=F_mean_OU, cov=(F_squared_OU - np.outer(F_mean_OU, F_mean_OU)), allow_singular=True)
    alphas_OU = []

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
                                        constraints=({'type': 'ineq', 'fun': lambda x: x}), method='SLSQP').x
            # for l >= 3
            else:
                theta_hat_OU = minimize(lambda x: -np.matmul(x, H) + 0.5 * np.matmul(x, np.matmul(HH, x)), theta_hat_OU,
                                        constraints=({'type': 'ineq', 'fun': lambda x: hurwitz_principal_minors(x)}, {'type': 'ineq', 'fun': lambda x: x}), method='SLSQP').x
            theta_hat_OU = np.clip(theta_hat_OU, 1e-16, None)
            X_OU.alpha_set(theta_hat_OU)
            X_OU.beta_set([[0] * stage * C for stage in R_L])
            Q_Q = X_OU.Q_matrix()

        alphas_OU.append(theta_hat_OU)
        forecast_OU[i, :, :] = np.mean(np.array([simulate_MCAR_compound_poisson([i for i in range(forecast_step + 1)],
                                                                                Q_Q, np.append(window_data[:, -1], [0] * K * (len(R_L) - 1)), b_OU, Sigma_OU, 1, jump_F_OU, uniform=True)[:, 1:] for _ in range(1000)]), axis=0)
        df_OU = np.append(df_OU, forecast_OU[i, :, 0:1], axis=1)

    with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', 'forecast_OU.npy'), 'wb') as f:
        np.save(f, forecast_OU)