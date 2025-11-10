import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from multiprocessing import Pool
from itertools import combinations
from datetime import datetime

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
    :return: saves the aggregated VWAP in output_path, as .npz files with numpy attributes
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
    :return: saves the aggregated VWAP in output_path, as .npz files with numpy attributes
             "orderbook_features" (:, 1).
    """
    print(orderbook_name)
    log = ''

    ### load LOBSTER orderbook and message files
    try:
        df_orderbook = pd.read_csv(orderbook_name, header=None)
    except:
        return orderbook_name + ' skipped. Error: failed to read orderbook.'

    # compute the VWAP
    df_orderbook.columns = ["ASKp1", "ASKs1", "BIDp1", "BIDs1"]
    df_orderbook.insert(0, "VWAP", (df_orderbook["ASKp1"] * df_orderbook["BIDs1"] + df_orderbook["BIDp1"] *
                                    df_orderbook["ASKs1"]) / (df_orderbook["BIDs1"] + df_orderbook["ASKs1"]))

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

    # 6. downsample tick-by-tick VWAP data into 1-second frequency, keeping only the last tick in each second with forward-fill
    df_orderbook['timestamp'] = pd.to_datetime(df_orderbook['seconds'], unit='s', origin=pd.Timestamp(date))
    df_orderbook = df_orderbook.set_index('timestamp')
    df_orderbook_sec = df_orderbook.resample('1s').last().ffill()

    # save
    output_name = os.path.join(output_path, TICKER + "_" + "data" + "_" + str(date.date()) + ".npz")
    np.savez(output_name, timestamps=df_orderbook_sec.index.astype('datetime64[s]'), VWAP=df_orderbook_sec['VWAP'].values)

    return log + orderbook_name + ' completed.'


def aggregate_assets(assets, base_dir, out_dir):
    """
    Aggregate VWAP data for multiple assets into a single file.
    :param assets : list of Asset names (e.g., ["asset1", "asset2", "asset3", "asset4"]).
    :param base_dir : Base directory containing asset folders.
    :param out_dir : Output directory for aggregated files.
    """
    os.makedirs(out_dir, exist_ok=True)

    csv_file_list = glob.glob(base_dir + assets[0] + '/' + "*.{}".format("npz"))
    dates = [re.findall(r"\d{4}-\d{2}-\d{2}", a)[-1] for a in csv_file_list]
    day_data = []
    for date in dates:
        dfs = []
        for asset in assets:
            file_path = base_dir + asset + '/' + asset + '_data_' + date + '.npz'
            data = np.load(file_path)

            # assume .npz contains 'timestamp' and 'vwap' arrays
            ts = pd.to_datetime(data["timestamps"])
            vwap = data["VWAP"]

            df = pd.DataFrame({f"{asset}": vwap}, index=ts)
            dfs.append(df)

        # align on timestamp
        agg_df = pd.concat(dfs, axis=1).sort_index()

        # compute log return and covariance matrices for multiple intervals
        returns = np.log(agg_df).diff().dropna()

        assets = returns.columns
        pairs = list(combinations(assets, 2))

        outer_list = []

        for t, row in returns.iterrows():
            r = row.values
            cov_vals = {pair[0] + '_' + pair[1]: r[assets.get_loc(pair[0])] * r[assets.get_loc(pair[1])] for pair in pairs}
            var_vals = {asset: r[assets.get_loc(asset)] * r[assets.get_loc(asset)] for asset in assets}
            cov_vals.update(var_vals)
            outer_list.append(pd.Series(cov_vals, name=t))

        outer_df = pd.DataFrame(outer_list)
        day_data.append(outer_df)

    # save aggregated dataset
    cov_data = pd.concat(day_data)
    cov_path = out_dir + 'cov.npz'
    np.savez(cov_path, timestamp=cov_data.index.astype('datetime64[s]'), **{col: cov_data[col].values for col in cov_data.columns})


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


def line_graph(cov_data, TICKERS, rule, corr_thres = 0.25, ratio_thres = 0.1):
    """
    Construct the line graph adjacency matrix given covariance/variance data
    :param cov_data: covariance/variance data
    :param TICKERS: list of tickers
    :param rule: aggregation rule for realized covariations, str
    :param corr_thres: predefined correlation threshold, float
    :param ratio_thres: predefined ratio threshold, float
    :return: [line graph adjacency matrix, asset pair for each edge]
    """
    pairs = list(combinations(TICKERS, 2))
    df_cov = cov_data.resample(rule).sum()
    df_cov = df_cov.loc[df_cov.index.intersection(cov_data.index)]
    corr_list = []

    # Compute the correlation between assets as the value of edges
    for t, row in df_cov.iterrows():
        r = row.values
        corr_vals = {pair[0] + '_' + pair[1]: r[df_cov.columns.get_loc(pair[0] + '_' + pair[1])] / (
                    (r[df_cov.columns.get_loc(pair[0])] * r[df_cov.columns.get_loc(pair[1])]) ** .5) for pair in pairs}
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