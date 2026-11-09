"""
================================================================================
Empirical Analysis & Forecasting Pipeline for grOU Edge Models
================================================================================

This module implements the full empirical pipeline for the paper:

    Chen & Veraart (2026): "Edge-Indexed Network Time Series with Graph
    Ornstein-Uhlenbeck Dynamics", Imperial College London.

It covers five major components:

--------------------------------------------------------------------------------
1. LOBSTER Data Pre-Processing (Section 5)
--------------------------------------------------------------------------------
Functions for loading, cleaning, and resampling high-frequency limit order book
(LOB) data from the LOBSTER database into synchronized log mid-quote price series:

  - multiprocess_orderbooks:  Parallel pre-processing of raw LOBSTER CSV files
  - ProcessOrderbook:         Multiprocessing engine wrapper class
  - process_orderbook:        Single-day order book cleaning pipeline
  - process_day_parallel:     Load and align one day of data across all assets
  - aggregate_assets:         Aggregate log mid-quote prices across assets and days

Cleaning steps applied per trading day:
  (i)   Remove trading halts (event type 7)
  (ii)  Remove crossed quotes (BID > ASK)
  (iii) Collapse simultaneous timestamps (keep last state per second)
  (iv)  Restrict to regular market hours (9:30–16:00)
  (v)   Drop first and last 60 minutes of trading
  (vi)  Downsample to a fixed sampling interval (e.g., 1 second) with forward-fill

--------------------------------------------------------------------------------
2. Covariance Estimation via MRC (Appendix F)
--------------------------------------------------------------------------------
Implementation of the Modulated Realized Covariance (MRC) estimator of
Christensen, Kinnebrock & Podolskij (2010), which pre-averages high-frequency
price increments to reduce microstructure noise before computing covariance:

  - MRC:                  Compute MRC estimator for a single time window
  - log_data_to_corr_data: Apply MRC over a rolling window to produce a
                            time series of realized correlations/covariances

--------------------------------------------------------------------------------
3. Network Graph Utilities
--------------------------------------------------------------------------------
Functions for constructing and manipulating the edge-indexed network structure:

  - adjacency_matrices:  Compute r-stage adjacency matrices A^{(1)},...,A^{(n)}
  - edge_weight:         Compute normalized edge weights W^{(r)} from adjacency
  - adj_to_line_graph:   Convert a node-level adjacency matrix to a line graph
                         (edge-level adjacency matrix B = M^T M, where M is the
                         incidence matrix), returning edge labels and B

--------------------------------------------------------------------------------
4. Forecasting Models (Section 4.2)
--------------------------------------------------------------------------------
One-step-ahead forecasting implementations for all benchmark models:

  - NA_model:         Naive model (Y_{t+1} = Y_t)
  - UAR_model:        Univariate AR model (independent per edge, OLS)
  - VAR_model:        Vector AR model (joint, OLS)
  - GNAR_model:       Generalized Network AR edge model (Mantziou et al. 2023)
  - OU_edge_model:    Diagonal OU model (no network structure)
  - MCAR_edge_model:  Full MCAR model (unconstrained drift matrices)
  - grOU_edge_model:  Graph OU edge model (network-structured drift, main model)

All continuous-time models (OU, MCAR, grOU) use the exact prediction formula
from Proposition 2.2 of Chen & Veraart (2026):
    E[Y_{t+h} | F_t] = E[Y_0] + A e^{Qh} [X_t - E[X_0]]

Evaluation metrics:
  - evaluate_forecast: Computes RMSE and directional accuracy (DirAcc)

--------------------------------------------------------------------------------
5. Joint Graph-Model Selection (Section 5.3)
--------------------------------------------------------------------------------
Implements the two-stage joint network and model selection procedure:

  Stage 1 (Screening): Generate n_candidates Erdős-Rényi random graphs,
                        evaluate each using a simple grOU(1,[1]) model,
                        and retain the top_k networks by directional accuracy
                        (with BIC as tiebreaker).

  Stage 2 (Selection): For each retained network, search over candidate lag/
                        neighborhood specifications and select the best model
                        using directional accuracy and BIC.

  Functions:
  - worker:                        Evaluate one candidate network (parallel worker)
  - screen_candidate_networks:     Stage 1: parallel network screening
  - evaluate_models_for_network:   Stage 2: model selection for a fixed network
  - joint_graph_model_selection:   Full two-stage procedure
  - is_better:                     Comparison function for (DirAcc, BIC) pairs

--------------------------------------------------------------------------------
6. Simulation Study Utilities (Section 4)
--------------------------------------------------------------------------------
  - process_one_sim:    Estimate grOU parameters at multiple time horizons
                        (for consistency/convergence experiments)
  - model_pred_single:  Evaluate all forecasting models for one simulated path
  - model_pred:         Parallel evaluation across all Monte Carlo replicates
  - boxplot_:           Generate half-violin boxplots for RMSE, DirAcc, and
                        computation time across models and misspecification scenarios

--------------------------------------------------------------------------------
References:
--------------------------------------------------------------------------------
  - Chen & Veraart (2026): Edge-indexed network time series with graph Ornstein-Uhlenbeck dynamics
  - Lucchese, Pakkanen & Veraart (2026): Estimation and Inference for Multivariate
Continuous-time Autoregressive Processes
  - Mantziou et al. (2023): The GNAR-edge model: a network autoregressive model for networks with
time-varying edge weights
  - Christensen, Kinnebrock & Podolskij (2010): Pre-averaging estimators of the
ex-post covariance matrix in noisy diffusion models with non-synchronous data
  - Lucchese, Pakkanen & Veraart (2024): The short-term predictability of
returns in order book markets: A deep learning perspective
================================================================================
"""

import os
import glob
import math
import re
import time
import heapq
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from numpy.lib.stride_tricks import sliding_window_view
from multiprocessing import Pool
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR
from graphOUedge import GraphOUEdge
from mcar.estimate import *
from mcar.simulate import *

# ============================================================
# LOBSTER DATA PRE-PROCESSING
# ============================================================

def multiprocess_orderbooks(TICKER, input_path, output_path, log_path, check_for_processed_data=True, sample_interval = '1s'):
    """
    Pre-process LOBSTER limit order book data in parallel across trading days.

    Discovers all order book and message book CSV files in `input_path`, matches
    them by date, and distributes processing across CPU cores using multiprocessing.
    Each trading day is processed independently by `process_orderbook`, which
    applies the full cleaning pipeline (see that function for details).

    A processing log is written to `log_path/processing_logs.txt` recording:
      - Files that failed to load (read errors)
      - Days with crossed quotes (BID > ASK)
      - Days with unusual market opening/closing times
      - Days with trading halts

    :param TICKER:                   Ticker symbol to process (e.g., 'AAPL'), str
    :param input_path:               Directory containing raw LOBSTER CSV files.
                                     Order book files must contain 'orderbook' in the filename;
                                     message files must contain 'message'. str
    :param output_path:              Directory where processed .npz files are saved, str
    :param log_path:                 Directory where the processing log is written, str
    :param check_for_processed_data: If True, skip dates already present as .npz files
                                     in output_path (allows resuming interrupted runs), bool
    :param sample_interval:          Pandas-compatible resampling interval (e.g., '1s', '5s'), str
    :return:                         None. Saves one .npz file per trading day to output_path,
                                     each containing 'timestamps' and 'mid' arrays.
    """
    # Discover all CSV files and separate into order book and message book lists
    csv_file_list = glob.glob(os.path.join(input_path, "*.{}".format("csv")))
    csv_orderbook = [name for name in csv_file_list if "orderbook" in name]
    csv_message = [name for name in csv_file_list if "message" in name]
    csv_orderbook.sort()
    csv_message.sort()

    # Skip already-processed dates to allow resuming interrupted runs
    if check_for_processed_data:
        npz_file_list = sorted(glob.glob(os.path.join(output_path, "*.{}".format("npz"))))
        processed_dates = [re.search(r'\d{4}-\d{2}-\d{2}', file).group() for file in npz_file_list]
        csv_orderbook = [file for file in csv_orderbook if
                         re.search(r'\d{4}-\d{2}-\d{2}', file).group() not in processed_dates]
        csv_message = [file for file in csv_message if
                       re.search(r'\d{4}-\d{2}-\d{2}', file).group() not in processed_dates]

    # Sanity check: each order book file must have a corresponding message file
    assert (len(csv_message) == len(csv_orderbook))

    print("started multiprocessing")

    # Launch parallel processing pool: one worker per order book file
    pool = Pool(os.cpu_count())
    try:
        engine = ProcessOrderbook(TICKER, output_path, sample_interval)
        logs = pool.map(engine, csv_orderbook)
    finally:
        # Ensure pool is closed even if errors occur during processing
        pool.close()
        pool.join()

    print("finished multiprocessing")

    # Write all processing logs to a single text file
    with open(log_path + "/processing_logs.txt", "w") as f:
        for log in logs:
            f.write(log + "\n")

    print("please check processing logs.")

class ProcessOrderbook(object):
    """
    Callable wrapper class for multiprocessing-compatible order book processing.

    Python's multiprocessing.Pool.map requires a picklable callable. Since
    instance methods are not directly picklable, this class wraps the
    `process_orderbook` function as a callable object that can be passed to Pool.

    Each worker process receives the same TICKER, output_path, and sample_interval,
    but processes a different order book file.
    """

    def __init__(self, TICKER, output_path, sample_interval):
        """
        :param TICKER:          Ticker symbol (e.g., 'AAPL'), str
        :param output_path:     Directory for saving processed .npz files, str
        :param sample_interval: Pandas resampling interval (e.g., '1s'), str
        """
        self.sample_interval = sample_interval
        self.TICKER = TICKER
        self.output_path = output_path

    def __call__(self, orderbook_name):
        """
        Process a single order book file.

        :param orderbook_name: Path to the order book CSV file, str
        :return:               Processing log string for this file, str
        """
        output = process_orderbook(orderbook_name, self.TICKER, self.output_path, self.sample_interval)
        return output

def process_orderbook(orderbook_name, TICKER, output_path, sample_interval):
    """
    Clean and resample a single day of LOBSTER limit order book data.

    Applies the following cleaning pipeline (following Lucchese et al. 2024):

    1. **Load data**: Read order book (ASK/BID prices and sizes) and message book
       (timestamps, event types) from CSV files.
    2. **Compute mid-quote**: mid = (ASKp1 + BIDp1) / 2
    3. **Remove trading halts**: Drop all rows between event type 7 entries with
       price=-1 (halt start) and price=+1 (halt end).
    4. **Remove crossed quotes**: Drop rows where BIDp1 > ASKp1.
    5. **Collapse simultaneous timestamps**: Group by second and keep the last
       order book state per second (handles multi-record conceptual events).
    6. **Check market hours**: Log a warning if open/close times differ from
       the standard 9:30–16:00 window.
    7. **Restrict to market hours**: Keep only rows with timestamps in [34200, 57600]
       seconds since midnight (9:30–16:00).
    8. **Drop opening/closing periods**: Remove the first and last 60 minutes of
       trading to reduce microstructure noise and non-stationarity.
    9. **Resample**: Downsample to `sample_interval` frequency, forward-filling
       gaps (last-tick interpolation).
    10. **Save**: Write timestamps and mid-quote prices to a .npz file.

    :param orderbook_name:  Path to the LOBSTER order book CSV file, str
                            Expected columns (no header): ASKp1, ASKs1, BIDp1, BIDs1
    :param TICKER:          Ticker symbol used for naming the output file, str
    :param output_path:     Directory for saving the processed .npz file, str
    :param sample_interval: Pandas resampling interval (e.g., '1s'), str
    :return:                Log string describing the outcome (warnings or completion), str
    """
    print(orderbook_name)
    log = ''

    # ---- Step 1: Load order book and message book ----
    try:
        df_orderbook = pd.read_csv(orderbook_name, header=None)
    except:
        return orderbook_name + ' skipped. Error: failed to read orderbook.'

    # Assign column names and compute mid-quote price
    df_orderbook.columns = ["ASKp1", "ASKs1", "BIDp1", "BIDs1"]
    df_orderbook.insert(0, "mid", (df_orderbook["ASKp1"] + df_orderbook["BIDp1"]) / 2)

    # Extract trading date from filename (format: YYYY-MM-DD)
    match = re.findall(r"\d{4}-\d{2}-\d{2}", orderbook_name)[-1]
    date = datetime.strptime(match, "%Y-%m-%d")

    # Read times from message file and keep a record of problematic files
    message_name = orderbook_name.replace("orderbook", "message")
    try:
        df_message = pd.read_csv(message_name, usecols=[0, 1, 2, 3, 4, 5], header=None)
    except:
        return orderbook_name + ' skipped. Error: failed to read messagebook.'

    # Verify alignment between order book and message book
    assert (len(df_message) == len(df_orderbook))

    # Add column names to message book
    df_message.columns = ["seconds", "event type", "order ID", "volume", "price", "direction"]

    # ---- Step 3: Remove trading halts (event type 7) ----
    # Halts are marked by event type 7 with price=-1 (start) and price=+1 (end)
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

    # ---- Step 4: Remove crossed quotes (BID > ASK) ----
    crossed_quotes_index = df_orderbook[(df_orderbook["BIDp1"] > df_orderbook["ASKp1"])].index
    if len(crossed_quotes_index) > 0:
        log = log + 'Warning: ' + str(
            len(crossed_quotes_index)) + ' crossed quotes removed in ' + orderbook_name + '.\n'
    df_orderbook = df_orderbook.drop(crossed_quotes_index)
    df_message = df_message.drop(crossed_quotes_index)

    # Add timestamp column from message book to order book
    df_orderbook.insert(0, "seconds", df_message["seconds"])

    # ---- Step 5: Collapse simultaneous timestamps ----
    # A single conceptual event (e.g., limit order modification = cancel + resubmit,
    # or a market order matching multiple resting orders) may appear as multiple rows
    # with the same timestamp. Keep only the last state per unique second.
    df_orderbook = df_orderbook.groupby(["seconds"]).tail(1)

    # ---- Step 6: Check market opening times ----
    # Standard US equity market hours: 9:30 (34200s) to 16:00 (57600s)
    market_open = int(df_orderbook["seconds"].iloc[0] / 60) / 60
    market_close = (int(df_orderbook["seconds"].iloc[-1] / 60) + 1) / 60
    if not (market_open == 9.5 and market_close == 16):
        log = log + 'Warning: unusual opening times in ' + orderbook_name + ': ' + str(market_open) + ' - ' + str(
            market_close) + '.\n'

    # ---- Step 7: Restrict to standard market hours ----
    df_orderbook = df_orderbook.loc[(df_orderbook["seconds"] >= 34200) &
                                    (df_orderbook["seconds"] <= 57600)]

    # ---- Step 8: Drop first and last 60 minutes of trading ----
    # Reduces opening/closing auction effects and microstructure noise
    market_open_seconds = market_open * 60 * 60 + 60 * 60
    market_close_seconds = market_close * 60 * 60 - 60 * 60
    df_orderbook = df_orderbook.loc[(df_orderbook["seconds"] >= market_open_seconds) &
                                    (df_orderbook["seconds"] <= market_close_seconds)]

    # ---- Step 9: Resample to fixed interval with forward-fill ----
    # Convert seconds-since-midnight to proper datetime index, then resample
    df_orderbook['timestamp'] = pd.to_datetime(df_orderbook['seconds'], unit='s', origin=pd.Timestamp(date))
    df_orderbook = df_orderbook.set_index('timestamp')
    # Keep last tick in each interval; forward-fill gaps (no-trade periods)
    df_orderbook_min = df_orderbook.resample(sample_interval).last().ffill()

    # ---- Step 10: Save processed data ----
    output_name = os.path.join(output_path, TICKER + "_" + "data" + "_" + str(date.date()) + ".npz")
    np.savez(output_name, timestamps=df_orderbook_min.index.astype('datetime64[s]'), mid=df_orderbook_min['mid'].values)

    return log + orderbook_name + ' completed.'

def process_day_parallel(arg):
    """
    Load and align log mid-quote prices for all assets on a single trading day.

    Reads pre-processed .npz files (produced by `process_orderbook`) for each
    asset on the given date, converts mid-quote prices to log scale, and aligns
    all assets to a common timestamp index using an inner join (refresh-time
    synchronization: only timestamps present for all assets are retained).

    :param arg: Tuple (date, assets, base_dir) where:
                  - date (str):       Trading date in 'YYYY-MM-DD' format
                  - assets (List[str]): List of ticker symbols to load
                  - base_dir (str):   Base directory containing per-asset subdirectories
    :return:    DataFrame with timestamps as index and log mid-quote prices as columns,
                shape (T_day, n_assets), pd.DataFrame
    """
    date, assets, base_dir = arg
    results = []
    for asset in assets:
        # Load pre-processed .npz file for this asset and date
        file_path = base_dir + asset + '/' + asset + '_data_' + date + '.npz'
        data = np.load(file_path)
        ts = pd.to_datetime(data["timestamps"])
        mid = data["mid"]
        # Convert to log mid-quote price (log prices are approximately stationary in differences)
        results.append(pd.DataFrame({asset: np.log(mid)}, index=ts))

    # Inner join aligns timestamps: only keep timestamps present for ALL assets
    # This implements refresh-time sampling to handle asynchronous trading
    day_df = pd.concat(results, axis=1, join='inner').sort_index()
    return day_df

def aggregate_assets(assets, base_dir, out_dir, max_workers_days=None):
    """
    Aggregate log mid-quote prices across all assets and trading days in parallel.

    Discovers all available trading dates from the first asset's directory,
    processes each day in parallel using `process_day_parallel`, concatenates
    the results, and saves the aggregated dataset as a single .npz file.

    :param assets:           List of ticker symbols to aggregate, List[str]
    :param base_dir:         Base directory containing per-asset subdirectories, str
    :param out_dir:          Output directory for the aggregated 'log_price.npz' file, str
    :param max_workers_days: Maximum number of parallel processes for day-level loading;
                             defaults to the number of available CPUs, int or None
    :return:                 None. Saves 'log_price.npz' to out_dir with arrays:
                             'timestamps' and one array per asset column.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Discover all available dates from the first asset's directory
    csv_file_list = glob.glob(base_dir + assets[0] + '/' + "*.{}".format("npz"))
    dates = [re.findall(r"\d{4}-\d{2}-\d{2}", a)[-1] for a in csv_file_list]

    # Build argument list for parallel day loading
    day_inputs = [(date, assets, base_dir) for date in dates]

    # Process each trading day in parallel
    with ProcessPoolExecutor(max_workers=max_workers_days) as executor:
        results = list(executor.map(process_day_parallel, day_inputs))

    # Concatenate all days into a single time series DataFrame
    log_price_data = pd.concat(results)

    # Ensure consistent asset column ordering
    log_price_data = log_price_data[assets]

    # Save aggregated log price data
    log_price_path = out_dir + 'log_price.npz'
    np.savez(log_price_path, timestamps=log_price_data.index.astype('datetime64[s]'), **{col: log_price_data[col].values for col in log_price_data.columns})

# ============================================================
# MODULATED REALIZED COVARIANCE (MRC) ESTIMATOR
# ============================================================

def MRC(Y, delta, theta, is_corr=True):
    """
    Compute the Modulated Realized Covariance (MRC) estimator for a price panel.

    Implements the pre-averaging estimator of Christensen, Kinnebrock & Podolskij (2010),
    which reduces the impact of market microstructure noise by smoothing short blocks of
    high-frequency observations before computing the covariance matrix.

    For a window of n observations and d assets, the algorithm:
    1. Computes the window length k_n = ceil((n-1)^delta * theta), rounded to even.
    2. For each valid index i = 0,...,m-1 (where m = n - k_n + 1), computes the
       pre-averaged return:
           Ȳ_i = (1/k_n) [Σ_{j=i+k_n/2}^{i+k_n-1} Y_j - Σ_{j=i}^{i+k_n/2-1} Y_j]
       (difference of forward and backward half-window sums, scaled by k_n)
    3. Estimates the covariance as:
           S = [(n-1)/(n-k_n+1)] * (12/k_n) * Σ_i Ȳ_i Ȳ_i^T
       The scaling factor (n-1)/(n-k_n+1) * 12/k_n corrects for the window length
       and the reduced number of valid indices.
    4. Optionally converts S to a correlation matrix.
    5. Returns the upper-triangular elements as a labeled pd.Series.

    :param Y:        Panel of log price levels, shape (n, d), pd.DataFrame
                     with asset tickers as column names
    :param delta:    Exponent controlling window size growth: k_n ~ (n-1)^delta, float
    :param theta:    Scaling constant for window size: k_n = ceil((n-1)^delta * theta), float
    :param is_corr:  If True, return correlation matrix; if False, return covariance, bool
    :return:         Upper-triangular elements of S as a labeled pd.Series,
                     with labels 'ASSET_i_ASSET_j' for i ≤ j. Returns None if n == 0.
    """
    assets = Y.columns
    Y = np.asarray(Y)
    n, d = Y.shape
    if n == 0:
        return None

    # ---- Compute pre-averaging window size k_n ----
    # k_n must be even for symmetric half-window construction
    N = math.ceil((n - 1) ** delta * theta)
    k_n = N if N % 2 == 0 else N + 1
    half = k_n // 2
    m = n - k_n + 1  # Number of valid pre-averaged returns

    # ---- Compute pre-averaged returns Ȳ_i for all i ----
    # Ȳ_i = (1/k_n) * [forward_half_sum - backward_half_sum]
    # This smoothing cancels microstructure noise (which fluctuates rapidly)
    # while preserving the slower-moving efficient price component
    barY = np.zeros((m, d))
    for i in range(m):
        forward = Y[i + half: i + k_n].sum(axis=0)      # Σ_{j=i+k_n/2}^{i+k_n-1} Y_j
        backward = Y[i: i + half].sum(axis=0)           # Σ_{j=i}^{i+k_n/2-1} Y_j
        barY[i] = (forward - backward) / k_n

    # ---- Compute MRC covariance estimate ----
    # S = scaling * Σ_i Ȳ_i Ȳ_i^T
    # Scaling corrects for window length and reduced sample size
    S = sum(np.outer(barY[i], barY[i]) for i in range(m)) * ((n - 1) / (n - k_n + 1)) * (12 / k_n)

    # ---- Optionally convert to correlation matrix ----
    if is_corr:
        diag = np.sqrt(np.diag(S))
        denom = np.outer(diag, diag)
        # Safe division: avoid division by zero for zero-variance assets
        S = np.divide(S, denom, out=np.zeros_like(S), where=denom > 0)

    # ---- Extract upper-triangular elements with asset-pair labels ----
    iu = np.triu_indices(d)
    labels = [f"{assets[i]}_{assets[j]}" for i, j in zip(iu[0], iu[1])]
    data = S[iu]

    return pd.Series(data, index=labels)

def log_data_to_corr_data(path: str, rule: str, delta=.6, theta=1, is_corr=True):
    """
    Compute a time series of MRC estimates from aggregated log price data.

    Loads the 'log_price.npz' file from `path`, resamples into non-overlapping
    windows of length `rule` (e.g., '1H' for hourly), applies the MRC estimator
    to each window, and saves the resulting time series of covariance/correlation
    estimates as 'corr.npz' or 'cov.npz'.

    :param path:     Directory containing 'log_price.npz', str
    :param rule:     Pandas resampling rule for grouping (e.g., '1H', '1D'), str
    :param delta:    MRC window size exponent (default 0.6 as in Appendix F), float
    :param theta:    MRC window size scaling constant (default 1), float
    :param is_corr:  If True, compute correlation; if False, compute covariance, bool
    :return:         None. Saves 'corr.npz' or 'cov.npz' to `path`.
    """
    # Load aggregated log price data
    data = np.load(path + 'log_price.npz')
    index = pd.to_datetime(data["timestamps"])
    columns = [key for key in data.files if key != "timestamps"]
    log_price_data = pd.DataFrame({col: data[col] for col in columns}, index=index)
    data.close()

    # Apply MRC estimator to each resampling window
    out = {}
    for timestamp, group in log_price_data.resample(rule):
        res = MRC(group, delta, theta, is_corr)
        if res is None:
            continue    # Skip empty windows (e.g., non-trading periods)
        out[timestamp] = res

    # Assemble time series of MRC estimates
    corr_data = pd.DataFrame.from_dict(out, orient='index')

    # Save as .npz with one array per asset pair
    if is_corr:
        np.savez(path + 'corr.npz', timestamps=corr_data.index,
             **{col: corr_data[col].values for col in corr_data.columns})
    else:
        np.savez(path + 'cov.npz', timestamps=corr_data.index,
                 **{col: corr_data[col].values for col in corr_data.columns})

# ============================================================
# NETWORK GRAPH UTILITIES
# ============================================================

def adjacency_matrices(A, n):
    """
    Compute r-stage adjacency matrices A^{(1)}, A^{(2)}, ..., A^{(n)} for a graph.

    Implements the r-stage neighborhood definition of Knight et al. (2020),
    extended to the edge-indexed setting of Chen & Veraart (2026):
      - A^{(1)}[i,j] = 1 if there is a direct edge from i to j (1-stage neighbor)
      - A^{(r)}[i,j] = 1 if j is reachable from i in exactly r steps,
                        but NOT reachable in fewer than r steps

    Uses a BFS-style dynamic programming approach:
      - `reach`: nodes reachable from the current frontier in one step
      - `cumulative`: all nodes reachable in ≤ r steps (to exclude closer neighbors)
      - At each stage r, A^{(r)} = (reach @ A > 0) AND NOT cumulative

    :param A: Binary node-level adjacency matrix (N×N), where A[i,j]=1 if edge (i,j) exists
    :param n: Maximum neighborhood stage to compute
    :return:  List of n binary matrices [A^{(1)}, A^{(2)}, ..., A^{(n)}], each (N,N) np.ndarray
    """
    # Stage 1: direct neighbors (1-hop reachability)
    reach = (A > 0).astype(int)
    results = [reach.copy()]
    cumulative = reach.copy()   # Tracks all nodes reachable in ≤ current stage
    for k in range(2, n + 1):
        # Extend frontier: nodes reachable from current stage in one more hop
        Ak = (reach @ A > 0).astype(int)

        # Remove nodes already reachable at closer stages (exact-distance condition)
        Ak = Ak * (1 - cumulative)

        # Remove self-loops (no edge from a node to itself)
        np.fill_diagonal(Ak, 0)

        results.append(Ak.copy())
        cumulative |= Ak    # Update cumulative reachability
        reach = Ak.copy()   # DP update (only extend from new layer)

    return results

def edge_weight(adjs):
    """
    Compute normalized edge weight matrix W from a list of adjacency matrices.

    For each node i and each stage r, the weight of edge (i,j) at stage r is
    w_{ij}^{(r)} = 1 / |N^{(r)}(i)|, i.e., uniform weights normalized by the
    number of r-stage neighbors of i (as in Section 2.2 of Chen & Veraart 2026).

    The total weight matrix W = Σ_r A^{(r)} / |N^{(r)}(·)| aggregates contributions
    from all neighborhood stages. This corresponds to a single covariate (C=1).

    :param adjs: List of binary adjacency matrices [A^{(1)}, ..., A^{(n)}], each (N,N)
    :return:     List containing a single weight matrix W, shape (N,N) np.ndarray.
                 W[i,j] is the total normalized weight of the interaction from i to j.
    """
    N = adjs[0].shape[0]
    result = np.zeros((N, N))
    for adj in adjs:
        # Row sums give the number of neighbors at this stage for each node
        row_sums = adj.sum(axis=1, keepdims=True)
        # Avoid division by zero for isolated nodes (no neighbors at this stage)
        row_sums[row_sums == 0] = 1
        # Add normalized adjacency: each row sums to 1 (or 0 if no neighbors)
        result += adj / row_sums

    return [result]

def adj_to_line_graph(adj, TICKERS):
    """
    Convert a node-level adjacency matrix to a line graph (edge-level adjacency).

    The line graph L(G) of a graph G has one node per edge of G, with two nodes
    of L(G) connected if the corresponding edges of G share a common endpoint.
    This is computed via the incidence matrix M ∈ {0,1}^{N×K}:
        M[i, e] = 1 if node i is an endpoint of edge e
        B = M^T M,  diag(B) = 0  (line graph adjacency matrix)

    Only undirected edges (i,j) with i < j are included (upper triangle of adj).

    :param adj:     Binary node-level adjacency matrix (N×N), np.ndarray
    :param TICKERS: List of N asset ticker symbols (node labels), List[str]
    :return:        List [B, edges_] where:
                      B (np.ndarray):      Line graph adjacency matrix (K×K)
                      edges_ (List[str]):  Edge labels 'TICKER_i_TICKER_j' for each edge
    """
    adjacency = pd.DataFrame(adj, index=TICKERS, columns=TICKERS)
    n = adjacency.shape[0]

    # Extract upper-triangular edges (undirected: only include each pair once)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adjacency.iloc[i, j] == 1]
    m = len(edges)

    # Create readable edge labels
    edges_ = [adjacency.columns[edge[0]] + '_' + adjacency.columns[edge[1]] for edge in edges]

    # Build incidence matrix M ∈ {0,1}^{N×K}: M[i,e]=1 if node i is endpoint of edge e
    M = np.zeros((n, m), dtype=int)
    for e_idx, (i, j) in enumerate(edges):
        M[i, e_idx] = 1
        M[j, e_idx] = 1

    # Line graph adjacency: B = M^T M, with diagonal zeroed (no self-loops)
    B = M.T @ M
    np.fill_diagonal(B, 0)
    return [B, edges_]

# ============================================================
# FORECASTING MODELS
# ============================================================

def NA_model(Y, test_set, forecast_step):
    """
    Naive (random walk) forecasting model: predict the last observed value.

    Implements the naive benchmark Ŷ_{t+h} = Y_t for all h = 1,...,forecast_step.
    This model achieves DirAcc = 0.5 by construction (no directional information).

    :param Y:             Training data (used to get the last observed value), (K, T_train)
    :param test_set:      True future values, shape (test_size, K, forecast_step) np.ndarray
    :param forecast_step: Number of steps ahead to forecast, int
    :return:              Tuple (forecast_NA, RMSE_NA) where:
                            forecast_NA (np.ndarray): Naive forecast, (test_size, K, forecast_step)
                            RMSE_NA (float):          RMSE of the naive forecast
    """
    # Repeat the last observed value (or first value of each test window) for all forecast steps
    forecast_NA = np.repeat(np.append(np.array(Y.copy())[:, -1].reshape(-1, 1)[np.newaxis, :, :], test_set[:-1, :, 0:1], axis=0), forecast_step, axis=2)
    RMSE_NA = np.sqrt(np.mean((forecast_NA - test_set) ** 2))

    return forecast_NA, RMSE_NA

def UAR_model(Y, test_set, test_size, forecast_step, lag=None, save=False):
    """
    Univariate Autoregressive (UAR) edge model: fit independent AR models per edge.

    Each edge is forecast independently using an AR(lag) model estimated by OLS.
    If lag is None, the optimal lag is selected by AIC over lags 1–12.
    Forecasts are generated recursively for multistep horizons.

    This corresponds to the 'AR' benchmark in Section 4.2 of Chen & Veraart (2026).

    :param Y:             Training data, shape (K, T_train) np.ndarray
    :param test_set:      Test windows, shape (test_size, K, forecast_step) np.ndarray
    :param test_size:     Number of test windows, int
    :param forecast_step: Number of steps ahead to forecast, int
    :param lag:           AR lag order; if None, selected by AIC, int or None
    :param save:          If True, save forecast array to disk, bool
    :return:              Forecast array, shape (test_size, K, forecast_step) np.ndarray
    """
    K, T_train = Y.shape

    # Concatenate training and test data for rolling-window evaluation
    df_AR = np.empty((K, T_train + test_size))
    df_AR[:, :T_train] = Y
    df_AR[:, T_train:] = test_set[:, :, 0].T
    forecast_AR = np.zeros((test_size, K, forecast_step))

    # ---- AIC-based lag selection (if lag not specified) ----
    if lag is None:
        aic_values = []
        best_p = np.zeros((K,))
        for i in range(K):
            for p in range(1, 13):
                model = AutoReg(df_AR[i, :T_train], lags=p).fit()
                aic_values.append((p, model.aic))
            best_p[i] = min(aic_values, key=lambda x: x[1])[0]

    # ---- Rolling-window forecasting ----
    for i in range(test_size):
        for j in range(K):
            window_data = df_AR[j, i:T_train + i]   # Rolling window of length T_train

            if lag is None:
                # Use statsmodels AutoReg with AIC-selected lag
                model = AutoReg(window_data, lags=best_p[j]).fit()
                forecast_AR[i, j, :] = model.predict(start=len(window_data), end=len(window_data) + forecast_step - 1)
            else:
                # Manual OLS AR(lag): build design matrix X with intercept and lagged values
                target = window_data[lag:]
                X = np.empty((T_train - lag, lag + 1))
                X[:, 0] = 1     # Intercept
                for s in range(1, lag + 1):
                    X[:, s] = window_data[lag - s:T_train - s]  # Lag-s values

                # OLS estimate: β = (X^T X)^{-1} X^T y
                beta = np.linalg.solve(X.T @ X, X.T @ target)

                # Recursive multistep forecast using rolling lag buffer
                recent = window_data[-lag:].copy()
                preds = np.zeros(forecast_step)
                for s in range(forecast_step):
                    x = np.concatenate(([1.0], recent[::-1]))   # [1, y_{t}, y_{t-1}, ...]
                    y_next = x @ beta
                    preds[s] = y_next
                    recent = np.append(recent[1:], y_next)      # Shift buffer

                forecast_AR[i, j, :] = preds

    if save:
        with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', 'forecast_AR.npy'), 'wb') as f:
            np.save(f, forecast_AR)

    return forecast_AR

def VAR_model(Y, test_set, test_size, forecast_step, lag=None, save=False):
    """
    Vector Autoregressive (VAR) edge model: joint AR model for all edges.

    All K edges are forecast jointly using a VAR(lag) model estimated by OLS.
    If lag is None, the optimal lag is selected by AIC over lags 1–12.
    Forecasts are generated recursively for multistep horizons.

    This corresponds to the 'VAR' benchmark in Section 4.2 of Chen & Veraart (2026).

    :param Y:             Training data, shape (K, T_train) np.ndarray
    :param test_set:      Test windows, shape (test_size, K, forecast_step) np.ndarray
    :param test_size:     Number of test windows, int
    :param forecast_step: Number of steps ahead to forecast, int
    :param lag:           VAR lag order; if None, selected by AIC, int or None
    :param save:          If True, save forecast array to disk, bool
    :return:              Forecast array, shape (test_size, K, forecast_step) np.ndarray
    """
    K, T_train = Y.shape

    # Concatenate training and test data (VAR expects shape (T, K))
    df_VAR = np.empty((T_train + test_size, K))
    df_VAR[:T_train, :] = Y.T
    df_VAR[T_train:, :] = test_set[:, :, 0]
    model = VAR(df_VAR[:T_train, :])
    forecast_VAR = np.zeros((test_size, K, forecast_step))

    # ---- AIC-based lag selection ----
    if lag is None:
        best_lag_var = model.select_order(maxlags=12).aic

    # ---- Rolling-window forecasting ----
    for i in range(test_size):
        window_data = df_VAR[i:T_train + i, :]  # Rolling window of length T_train

        if lag is None:
            # Use statsmodels VAR with AIC-selected lag
            model = VAR(window_data).fit(best_lag_var)
            forecast_VAR[i, :, :] = model.forecast(window_data[-model.k_ar:, :], steps=forecast_step).T
        else:
            # Manual OLS VAR(lag): build design matrix X with intercept and lagged values
            target = window_data[lag:, :]  # Shape (T_train - lag, K)
            X = np.empty((T_train - lag, 1 + lag * K))
            X[:, 0] = 1     # Intercept
            for row in range(T_train - lag):
                for s in range(1, lag + 1):
                    X[row, 1 + (s - 1) * K:1 + s * K] = window_data[lag - s + row, :]

            # OLS estimate: B = (X^T X)^{-1} X^T Y, shape (1+lag*K, K)
            B = np.linalg.solve(X.T @ X, X.T @ target)  # (1+lag*K, K)

            # Recursive multi-step forecast
            recent = window_data[-lag:, :].copy()
            preds = np.zeros((forecast_step, K))
            for s in range(forecast_step):
                x = np.concatenate(([1.0], recent[::-1, :].reshape(-1)))
                y_next = x @ B
                preds[s, :] = y_next
                recent = np.vstack([recent[1:, :], y_next])

            forecast_VAR[i, :, :] = preds.T

    if save:
        with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', 'forecast_VAR.npy'), 'wb') as f:
            np.save(f, forecast_VAR)

    return forecast_VAR

def GNAR_model(Y, Adj, R_L, test_set, test_size, forecast_step, is_global = False, save=False):
    """
    Generalized Network Autoregressive (GNAR) edge model.

    Implements the GNAR edge model of Mantziou et al. (2023) adapted to the
    edge-indexed setting. The model at each lag l includes:
      - Own-lag terms: α_{e,l} Y_{e,t-l} (per-edge or shared, depending on is_global)
      - Neighbor-stage terms: β_{l,r} Σ_{e' ∈ N^{(r)}(e)} w_{e,e'} Y_{e',t-l}

    Parameters are estimated by OLS on the vectorized design matrix.

    :param Y:             Training data, shape (K, T_train) np.ndarray
    :param Adj:           Line graph adjacency structure [B, edge_labels]
    :param R_L:           Maximum neighborhood stage per lag, List[int]
    :param test_set:      Test windows, shape (test_size, K, forecast_step) np.ndarray
    :param test_size:     Number of test windows, int
    :param forecast_step: Number of steps ahead to forecast, int
    :param is_global:     If True, use one shared α per lag (global); otherwise per-edge, bool
    :param save:          If True, save forecast array to disk, bool
    :return:              Forecast array, shape (test_size, K, forecast_step) np.ndarray
    """
    p = len(R_L)
    K, T_train = Y.shape

    # Build edge-level adjacency matrices and normalized edge weights
    Adj_L = np.array(adjacency_matrices(Adj[0], max(R_L)))
    W_C = edge_weight(Adj_L)

    # Concatenate training and test data
    df_GNAR = np.empty((T_train + test_size, K))
    df_GNAR[:T_train, :] = Y.T
    df_GNAR[T_train:, :] = test_set[:, :, 0]
    forecast_GNAR = np.zeros((test_size, K, forecast_step))

    n_rows = T_train - p
    n_neigh = sum(R_L)

    # Parameter count: global uses p + Σ R_l; local uses p*K + Σ R_l
    if is_global:
        n_params = p + n_neigh
    else:
        n_params = p * K + n_neigh

    # ---- Rolling-window forecasting ----
    for i in range(test_size):
        window_data = df_GNAR[i:T_train + i, :]
        Y_vec = window_data[p:, :].reshape(-1)      # Vectorized response

        # ---- Build design matrix D ----
        # D has n_rows * K rows (one per time step per edge) and n_params columns
        D = np.zeros((n_rows * K, n_params))
        row_idx = np.arange(n_rows * K).reshape(n_rows, K)
        rows = row_idx.reshape(-1)
        node_cols = np.tile(np.arange(K), n_rows)
        col_idx = 0

        for j in range(1, p + 1):
            x_lag = window_data[p - j:T_train - j, :]   # Lag-j values, shape (n_rows, K)

            if is_global:
                # One shared own-lag coefficient: same column for all edges
                D[:, col_idx] = x_lag.reshape(-1)
                col_idx += 1
            else:
                # Per-edge own-lag coefficients: each edge gets its own column
                D[rows, node_cols + col_idx] = x_lag.reshape(-1)
                col_idx += K

            # Neighbor-stage effects: β_{l,r} * (W^{(r)} Y_{t-l})
            for r in range(R_L[j - 1]):
                D[:, col_idx] = (x_lag @ W_C[r].T).reshape(-1)
                col_idx += 1

        # OLS parameter estimation: θ = (D^T D)^{-1} D^T Y_vec
        theta = np.linalg.solve(D.T @ D, D.T @ Y_vec)

        # OLS parameter estimation: θ = (D^T D)^{-1} D^T Y_vec
        hist = window_data.copy()
        y_fc = np.zeros((forecast_step, K))

        for s in range(forecast_step):
            x_next = np.zeros(K)
            idx = 0
            for j in range(1, p + 1):
                x_lag = hist[-j]        # Most recent lag-j value
                if is_global:
                    x_next += theta[idx] * x_lag
                    idx += 1
                else:
                    x_next += theta[idx:idx + K] * x_lag
                    idx += K
                for r in range(R_L[j - 1]):
                    x_next += theta[idx] * (W_C[r] @ x_lag)
                    idx += 1
            y_fc[s] = x_next
            hist = np.vstack((hist, x_next))    # Append forecast to history buffer

        forecast_GNAR[i, :, :] = y_fc.T

    if save:
        with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', 'forecast_GNAR.npy'), 'wb') as f:
            np.save(f, forecast_GNAR)

    return forecast_GNAR

def OU_edge_model(Y, test_set, test_size, forecast_step, R_L, C, P_, Q_, gamma, method = 'naive',
                  is_global = False, print_param = False, save=False):
    """
    Diagonal OU edge model: forecast using component-wise Ornstein-Uhlenbeck.

    Estimates a diagonal drift matrix Q = diag(α_1,...,α_K) (no cross-edge dependence)
    and generates forecasts using the exact conditional mean (Proposition 2.2).
    This is the 'OU' benchmark in Section 4.2 of Chen & Veraart (2026).

    :param Y:             Training data, shape (K, T_train) np.ndarray
    :param test_set:      Test windows, shape (test_size, K, forecast_step) np.ndarray
    :param test_size:     Number of test windows, int
    :param forecast_step: Number of steps ahead to forecast, int
    :param R_L:           Lag structure, List[int]
    :param C:             Number of covariates (unused, for API consistency), int
    :param P_:            Fine-grid partition, (N+1, ) np.ndarray
    :param Q_:            Coarse-grid partition, (M+1, ) np.ndarray
    :param gamma:         Critical region hyperparameter, float
    :param method:        Forecast method label (for file naming only), str
    :param is_global:     If True, use global (scalar) OU estimator, bool
    :param print_param:   If True, return estimated alpha values instead of forecasts, bool
    :param save:          If True, save forecast array to disk, bool
    :return:              If print_param=False: forecast array, shape (test_size, K, forecast_step)
                          If print_param=True:  array of estimated alpha values
    """
    K, T_train = Y.shape
    forecast_OU = np.zeros((test_size, K, forecast_step))

    # Concatenate training and test data
    df_OU = np.empty((K, T_train + test_size))
    df_OU[:, :T_train] = Y
    df_OU[:, T_train:] = test_set[:, :, 0].T

    # No network structure for OU: use zero adjacency matrices
    Adj_L_OU = np.array([np.zeros((K, K)) for _ in range(max(R_L))])
    W_C_OU = edge_weight(Adj_L_OU)
    X_OU = GraphOUEdge(R_L, K, C, Adj_L_OU, W_C_OU, gamma, 'data\\LOBSTERdata\\output\\train\\aggregate')

    # Estimate Lévy triplet (b, Σ) using full iterative procedure
    _, (b_OU, Sigma_OU) = estimate_MCAR('OU', Y, P_, Q_, Adj_L_OU, W_C_OU, R_L, C, gamma, is_global=is_global)
    alphas_OU = []

    # ---- Rolling-window forecasting ----
    for i in range(test_size):
        window_data = df_OU[:, i:T_train + i]

        # Estimate OU drift parameters for this window
        theta_hat_OU, _, _ = estimate_OU_drift(window_data, R_L, C, P_, Q_, b_OU, gamma, None, Sigma_OU, with_cov=True, is_global=is_global)

        # Fall back to naive forecast if drift is numerically zero
        if np.all(np.abs(theta_hat_OU) < 1e-12):
            forecast_OU[i, :, :] = np.repeat(window_data[:, -1][:, None], forecast_step, axis=1)
            continue

        # Build companion matrix with diagonal OU structure (no network terms)
        X_OU.alpha_set(theta_hat_OU)
        X_OU.beta_set([[0] * stage * C for stage in R_L])
        Q_Q = X_OU.Q_matrix(False)
        alphas_OU.append(theta_hat_OU)

        # Initial state: observed Y + zero higher-order derivatives
        x0 = np.append(window_data[:, -1], [0] * K * (len(R_L) - 1))

        # Noise injection matrix
        E = np.zeros([K * len(R_L), K])
        E[-K:, :] = np.eye(K)

        # Initialize state trajectory
        X = np.zeros([K * len(R_L), forecast_step + 1])
        X[:, 0] = x0

        # Precompute matrix exponential and Q^{-1}
        eQt = scipy.linalg.expm(Q_Q * (P_[1] - P_[0]))
        Q_inv = Q_QQ_inverse(Q_Q, K)

        # Iterative exact prediction
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

def MCAR_edge_model(Y, test_set, test_size, forecast_step, P_, Q_, R_L, C, gamma, method='naive', save=False):
    """
    Full MCAR edge model: forecast using unconstrained multivariate continuous-time AR.

    Estimates the full drift matrices Q_1,...,Q_p ∈ R^{K×K} without any network
    or diagonal structure, then generates forecasts using the exact conditional mean
    from Proposition 2.2 of Chen & Veraart (2026):
        E[Y_{t+h} | F_t] = E[Y_0] + A e^{Qh} [X_t - E[X_0]]

    The prediction is computed iteratively:
        X_{n+1} = e^{Q*Δt} X_n + (e^{Q*Δt} - I) Q^{-1} E b

    If the estimated drift is numerically zero (all entries < 1e-12), falls back
    to the naive forecast (last observed value).

    :param Y:             Training data, shape (K, T_train) np.ndarray
    :param test_set:      Test windows, shape (test_size, K, forecast_step) np.ndarray
    :param test_size:     Number of test windows, int
    :param forecast_step: Number of steps ahead to forecast, int
    :param P_:            Fine-grid partition, (N+1, ) np.ndarray
    :param Q_:            Coarse-grid partition, (M+1, ) np.ndarray
    :param R_L:           Lag structure (used for lag count p = len(R_L)), List[int]
    :param C:             Number of covariates (unused for MCAR, included for API consistency), int
    :param gamma:         Critical region hyperparameter, float
    :param method:        Forecast method label (for file naming only), str
    :param save:          If True, save forecast array to disk, bool
    :return:              Forecast array, shape (test_size, K, forecast_step) np.ndarray
    """
    K, T_train = Y.shape
    forecast_MCAR = np.zeros((test_size, K, forecast_step))

    # No network structure for MCAR: use zero adjacency matrices
    A = np.array([np.zeros((K, K)) for _ in range(max(R_L))])
    W_C = edge_weight(A)

    # Concatenate training and test data for rolling-window evaluation
    df_MCAR = np.empty((K, T_train + test_size))
    df_MCAR[:, :T_train] = Y
    df_MCAR[:, T_train:] = test_set[:, :, 0].T

    # Estimate Lévy triplet (b, Σ) using full iterative procedure
    _, (b_MCAR, Sigma_MCAR) = estimate_MCAR('MCAR', Y, P_, Q_, A, W_C, R_L, C, gamma)

    # ---- Rolling-window forecasting ----
    for i in range(test_size):
        window_data = df_MCAR[:, i:T_train + i]

        # Estimate MCAR drift matrices for this window
        QQ_hat = estimate_MCAR_drift(window_data, R_L, P_, Q_, b_MCAR, gamma, Sigma=Sigma_MCAR)

        # Fall back to naive forecast if drift is numerically zero
        if np.all(np.abs(QQ_hat) < 1e-12):
            forecast_MCAR[i, :, :] = np.repeat(window_data[:, -1][:, None], forecast_step, axis=1)
            continue

        # Build companion matrix and initial state
        Q_Q = MCAR_Q(QQ_hat)
        x0 = np.append(window_data[:, -1], [0] * K * (len(R_L) - 1))

        # Noise injection matrix E ∈ R^{pK×K}
        E = np.zeros([K * len(R_L), K])
        E[-K:, :] = np.eye(K)

        # Initialize state trajectory
        X = np.zeros([K * len(R_L), forecast_step + 1])
        X[:, 0] = x0

        # Precompute matrix exponential and Q^{-1} for efficiency
        eQt = scipy.linalg.expm(Q_Q * (P_[1] - P_[0]))
        Q_inv = Q_QQ_inverse(Q_Q, K)

        # Iterative exact prediction: X_{n+1} = e^{QΔt} X_n + (e^{QΔt} - I) Q^{-1} E b
        for n in range(forecast_step):
            X[:, n + 1] = eQt.dot(X[:, n]) + (eQt - np.eye(K * len(R_L))).dot(Q_inv).dot(E).dot(b_MCAR)

        forecast_MCAR[i, :, :] = np.array(X[:K, 1:])    # Extract first K components (Y)

    if save:
        with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', f'forecast_MCAR_{len(R_L)}_{method}.npy'), 'wb') as f:
            np.save(f, forecast_MCAR)

    return forecast_MCAR

def grOU_edge_model(Y, Adj, test_set, test_size, forecast_step, R_L, C, P_, Q_, gamma, method = 'naive',
                    is_global=False, print_param=False, is_test=False, save=False):
    """
    Graph OU (grOU) edge model: forecast using the network-structured OU model.

    Implements the main forecasting model of Chen & Veraart (2026). Estimates the
    grOU drift parameters (alpha, beta) using the discretized MLE (Theorem 3.2),
    constructs the companion matrix Q, and generates forecasts via the exact
    conditional mean (Proposition 2.2):
        E[Y_{t+h} | F_t] = E[Y_0] + A e^{Qh} [X_t - E[X_0]]

    Also supports BIC computation for model selection (Section 5.2, Equation 5.1):
        BIC = -θ^T [K] θ + (LK + Σ R_l) * log(N)

    :param Y:             Training data, shape (K, T_train) np.ndarray
    :param Adj:           Line graph adjacency structure [B, edge_labels]
    :param test_set:      Test windows, shape (test_size, K, forecast_step) np.ndarray
    :param test_size:     Number of test windows, int
    :param forecast_step: Number of steps ahead to forecast, int
    :param R_L:           Lag structure, List[int]
    :param C:             Number of covariates, int
    :param P_:            Fine-grid partition, (N+1, ) np.ndarray
    :param Q_:            Coarse-grid partition, (M+1, ) np.ndarray
    :param gamma:         Critical region hyperparameter, float
    :param method:        Forecast method label (for file naming only), str
    :param is_global:     If True, use global (scalar alpha) estimator, bool
    :param print_param:   If True, return (alphas, betas, BIC) instead of forecasts, bool
    :param is_test:       If True, only evaluate the first test window (for BIC), bool
    :param save:          If True, save forecast array to disk, bool
    :return:              If print_param=False: forecast array, shape (test_size, K, forecast_step)
                          If print_param=True:  Tuple (alphas, betas, BIC)
    """
    K, T_train = Y.shape

    # Build edge-level adjacency matrices and normalized edge weights
    Adj_L = np.array(adjacency_matrices(Adj[0], max(R_L)))
    W_C = edge_weight(Adj_L)

    # Initialize GraphOUEdge model structure
    X = GraphOUEdge(R_L, K, C, Adj_L, W_C, gamma, 'data\\LOBSTERdata\\output\\train\\aggregate')

    # Estimate Lévy triplet (b, Σ) using full iterative procedure
    _, (b, Sigma) = estimate_MCAR('grOU', Y, P_, Q_, Adj_L, W_C, R_L, C, gamma, is_global=is_global)

    forecast_grOU = np.zeros((test_size, K, forecast_step))
    df_grOU = np.empty((K, T_train + test_size))
    df_grOU[:, :T_train] = Y
    df_grOU[:, T_train:] = test_set[:, :, 0].T
    alphas = []
    betas = []

    # For BIC computation, only evaluate the first test window
    if is_test:
        test_size = 1

    # ---- Rolling-window forecasting ----
    for i in range(test_size):
        window_data = df_grOU[:, i:T_train + i]

        # Estimate grOU drift parameters θ = [α_1, β_1, ..., α_L, β_L] for this window
        theta_hat, KK, _ = estimate_grOU_drift(window_data, Adj_L, W_C, R_L, C, P_, Q_, b, gamma, None, Sigma, with_cov=True, is_global=is_global)

        # Fall back to naive forecast if drift is numerically zero
        if np.all(np.abs(theta_hat) < 1e-12):
            forecast_grOU[i, :, :] = np.repeat(window_data[:, -1][:, None], forecast_step, axis=1)
            continue

        # ---- Extract alpha and beta from the vectorized parameter vector ----
        # theta_hat layout: [α_1 (K), β_1 (R_1*C), α_2 (K), β_2 (R_2*C), ...]
        idx = 0
        alpha_pred = []
        beta_pred = []
        for r in R_L:
            alpha_pred.append(theta_hat[idx : idx + K])                 # α_l: K autoregressive coefficients
            beta_pred.append(theta_hat[(idx + K):(idx + K + r * C)])    # β_l: R_l*C network coefficients
            idx += K + r * C

        # Build companion matrix from estimated parameters
        X.alpha_set(np.array(alpha_pred))
        X.beta_set(beta_pred)
        Q_Q = X.Q_matrix(False)

        if print_param:
            alphas.append(alpha_pred)
            betas.append(beta_pred)

        # Initial state: observed Y + zero higher-order derivatives
        x0 = np.append(window_data[:, -1], [0] * K * (len(R_L) - 1))

        # Noise injection matrix
        E = np.zeros([K * len(R_L), K])
        E[-K:, :] = np.eye(K)

        # Initialize state trajectory
        X_ = np.zeros([K * len(R_L), forecast_step + 1])
        X_[:, 0] = x0

        # Precompute matrix exponential and Q^{-1}
        eQt = scipy.linalg.expm(Q_Q * (P_[1] - P_[0]))
        Q_inv = Q_QQ_inverse(Q_Q, K)

        # Iterative exact prediction: X_{n+1} = e^{QΔt} X_n + (e^{QΔt} - I) Q^{-1} E b
        for n in range(forecast_step):
            X_[:, n + 1] = eQt.dot(X_[:, n]) + (eQt - np.eye(K * len(R_L))).dot(Q_inv).dot(E).dot(b)

        forecast_grOU[i, :, :] = np.array(X_[:K, 1:])

    if save:
        with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', f'forecast_grOU_{len(R_L)}_{R_L}_{method}_global_{is_global}.npy'), 'wb') as f:
            np.save(f, forecast_grOU)

    if print_param:
        # BIC = -θ^T [K] θ + (LK + Σ R_l) * log(N)  (Equation 5.1)
       return alphas, betas, -theta_hat.T @ KK @ theta_hat + (len(R_L) * K + sum(R_L)) * np.log(len(Q_))
    else:
       return forecast_grOU

# ============================================================
# FORECASTING MODEL EVALUATION
# ============================================================

def evaluate_forecast(Y, forecast, test_set):
    """
    Evaluate forecast accuracy using RMSE and directional accuracy (DirAcc).

    Computes two metrics as defined in Section 4.2 of Chen & Veraart (2026):

    1. **RMSE** (Root Mean Squared Error):
           RMSE = sqrt( (1/TK) Σ_{t,k} (Y^{(k)}_t - Ŷ^{(k)}_t)^2 )

    2. **DirAcc** (Directional Accuracy):
           DirAcc = (1/TK) Σ_{t,k} 1{ sgn(Y^{(k)}_t - Y^{(k)}_{t-1}) = sgn(Ŷ^{(k)}_t - Y^{(k)}_{t-1}) }
       Measures the fraction of correctly predicted directions of change.
       By convention, the naive model achieves DirAcc = 0.5.
       Returns 0.5 if the forecast is constant (equal to the last observed value).

    :param Y:        Training data (used to get the last observed value Y_{T_train}),
                     shape (K, T_train) np.ndarray
    :param forecast: Forecast array, shape (test_size, K, forecast_step) np.ndarray
    :param test_set: True future values, shape (test_size, K, forecast_step) np.ndarray
    :return:         Tuple (rmse, acc) where:
                       rmse (float): Root mean squared error
                       acc (float):  Directional accuracy in [0, 1]
    """
    # RMSE: average squared error across all time steps, edges, and forecast horizons
    rmse = np.sqrt(np.mean((forecast - test_set) ** 2))

    # Build the "last observed value" array for directional comparison:
    # last_[0] = Y[:, -1] (last training observation)
    # last_[t] = test_set[t-1, :, 0] (first element of previous test window)
    last_ = np.append(np.array(Y.copy())[:, -1].reshape(-1, 1)[np.newaxis, :, :], test_set[:-1, :, 0:1], axis=0)

    # Compute sign of forecast increments and true increments
    forecast_dir = np.sign(np.diff(np.append(last_, forecast, axis=2), axis=2))
    test_dir = np.sign(np.diff(np.append(last_, test_set, axis=2), axis=2))

    # DirAcc: fraction of correctly predicted directions
    # Return 0.5 (naive baseline) if forecast is constant (no directional information)
    acc = 0.5 if (last_ == forecast).all() else (test_dir == forecast_dir).mean()

    return rmse, acc

# ============================================================
# JOINT GRAPH-MODEL SELECTION
# ============================================================

def worker(arg):
    """
    Evaluate one candidate Erdős-Rényi network using the screening grOU model.

    Parallel worker function for the network screening stage (Stage 1) of the
    joint graph-model selection procedure. For each candidate network:
      1. Generate a random Erdős-Rényi graph G(n, p) with seed=i
      2. Derive the edge-level line graph structure
      3. Evaluate the screening grOU model (specified by screening_RL)
      4. Compute the BIC for the selected model

    :param arg: Tuple (_data, test_data_, TICKERS, R_L, P_, Q_, forecast_step,
                       C, gamma, p, i) where:
                  - _data:        Training data DataFrame (assets as columns)
                  - test_data_:   Test data DataFrame
                  - TICKERS:      List of asset ticker symbols
                  - R_L:          Lag structure for screening model
                  - P_, Q_:       Fine and coarse grid partitions
                  - forecast_step: Number of steps ahead to forecast
                  - C:            Number of covariates
                  - gamma:        Critical region hyperparameter
                  - p:            Erdős-Rényi edge probability
                  - i:            Random seed for graph generation
    :return:    Tuple (direction_acc_grOU, A, bic) or None if graph is empty
    """
    _data, test_data_, TICKERS, R_L, P_, Q_, forecast_step, C, gamma, p, i = arg

    # Generate a random Erdős-Rényi graph with seed i for reproducibility
    A = nx.to_numpy_array(nx.erdos_renyi_graph(len(TICKERS), p, seed=i))

    # Skip empty graphs (no edges → no network structure to exploit)
    if np.all(A == 0):
        return None

    # Derive edge-level line graph structure from node-level adjacency
    Adj = adj_to_line_graph(A, TICKERS)
    K = Adj[0].shape[0]

    # Extract training and test data for the edges in this network
    Y = _data[Adj[1]].T
    test_Y = test_data_[Adj[1]].T
    test_set = sliding_window_view(test_Y, (K, forecast_step))[0]
    test_size = test_set.shape[0]

    # Evaluate screening model and compute directional accuracy
    forecast_grOU =  grOU_edge_model(Y, Adj, test_set, test_size, forecast_step, R_L, C, P_, Q_, gamma)
    _, direction_acc_grOU = evaluate_forecast(Y, forecast_grOU, test_set)

    # Compute BIC for the screening model (used as tiebreaker)
    bic = grOU_edge_model(Y, Adj, test_set, test_size, forecast_step, R_L, C, P_, Q_, gamma, is_global=False, print_param=True, is_test=True)[2]

    return direction_acc_grOU, A, bic

def screen_candidate_networks(_data, test_data_, TICKERS, screening_RL, P_, Q_, forecast_step, C, gamma, n_candidates,
    top_k, tol, p):
    """
    Stage 1 of joint graph-model selection: screen candidate networks in parallel.

    Generates n_candidates Erdős-Rényi random graphs, evaluates each using a
    simple screening grOU model, and retains the top_k networks by directional
    accuracy (with BIC as tiebreaker). Uses a min-heap of size top_k for
    memory-efficient online selection.

    The heap key is (round(DirAcc / tol), -BIC), so that:
      - Higher DirAcc → larger first component → better
      - Among ties in DirAcc: lower BIC → larger second component → better

    :param _data:         Training data DataFrame (assets as columns), pd.DataFrame
    :param test_data_:    Test data DataFrame, pd.DataFrame
    :param TICKERS:       List of asset ticker symbols, List[str]
    :param screening_RL:  Lag structure for the screening model (e.g., [1]), List[int]
    :param P_:            Fine-grid partition, (N+1, ) np.ndarray
    :param Q_:            Coarse-grid partition, (M+1, ) np.ndarray
    :param forecast_step: Number of steps ahead to forecast, int
    :param C:             Number of covariates, int
    :param gamma:         Critical region hyperparameter, float
    :param n_candidates:  Number of random candidate networks to generate, int
    :param top_k:         Number of top networks to retain, int
    :param tol:           Tolerance for DirAcc comparison (used in heap key), float
    :param p:             Erdős-Rényi edge probability, float
    :return:              List of top_k candidate results [DirAcc, A, BIC],
                          sorted from best to worst
    """
    arg_list = [(_data, test_data_, TICKERS, screening_RL, P_, Q_, forecast_step, C, gamma, p, i) for i in range(n_candidates)]

    # Min-heap of size top_k: heap[0] is the worst among the current top_k
    heap = []
    with Pool(os.cpu_count()) as pool:
        for idx, candidate in enumerate(pool.imap_unordered(worker, arg_list)):
            if candidate is None:
                continue    # Skip empty graphs

            # Heap key: (DirAcc_rounded, -BIC) — higher is better
            key = (round(candidate[0] / tol), -int(candidate[2]))

            if len(heap) < top_k:
                # Heap not full yet: add candidate
                heapq.heappush(heap, (key, idx, candidate))
            else:
                # Replace the worst if candidate is better
                if key > heap[0][0]:
                    heapq.heapreplace(heap, (key, idx, candidate))

    # Return top_k results sorted from the best to the worst
    top_results = [item[2] for item in sorted(heap, key=lambda x: x[0], reverse=True)]
    return top_results

def evaluate_models_for_network(_data, test_data_, TICKERS, A, P_, Q_, forecast_step, C, gamma, max_order, tol):
    """
    Stage 2 of joint graph-model selection: select the best grOU model for a fixed network.

    For a given network adjacency matrix A, evaluates all candidate grOU model
    specifications (varying lag order L and neighborhood stage R_l) and selects
    the best specification using directional accuracy (primary) and BIC (tiebreaker).

    Candidate models: [[1]*L, [2]*L] for L = 1,...,max_order
    (i.e., all lags with the same neighborhood stage, either 1 or 2)

    :param _data:         Training data DataFrame, pd.DataFrame
    :param test_data_:    Test data DataFrame, pd.DataFrame
    :param TICKERS:       List of asset ticker symbols, List[str]
    :param A:             Node-level adjacency matrix for this network, (N,N) np.ndarray
    :param P_:            Fine-grid partition, (N+1, ) np.ndarray
    :param Q_:            Coarse-grid partition, (M+1, ) np.ndarray
    :param forecast_step: Number of steps ahead to forecast, int
    :param C:             Number of covariates, int
    :param gamma:         Critical region hyperparameter, float
    :param max_order:     Maximum lag order to consider, int
    :param tol:           Tolerance for DirAcc comparison, float
    :return:              Tuple (best_model, best_result) where:
                            best_model (List[int]): Best R_L specification
                            best_result (List):     [DirAcc, A, BIC] for best model
    """
    # Derive edge-level line graph structure
    Adj = adj_to_line_graph(A, TICKERS)
    K = Adj[0].shape[0]

    # Extract training and test data for this network's edges
    Y = _data[Adj[1]].T
    test_Y = test_data_[Adj[1]].T
    test_set = sliding_window_view(test_Y, (K, forecast_step))[0]
    test_size = test_set.shape[0]

    # Generate candidate model specifications: [r]*L for L=1,...,max_order, r∈{1,2}
    candidate_models = [[value] * length for length in range(1, max_order + 1) for value in (1, 2)]

    best_model = None
    best_result = None

    for model in candidate_models:
        # Evaluate this model specification
        forecast_grOU = grOU_edge_model(Y, Adj, test_set, test_size, forecast_step, model, C, P_, Q_, gamma)
        _, direction_acc_grOU = evaluate_forecast(Y, forecast_grOU, test_set)

        # Compute BIC for this specification
        bic = grOU_edge_model(Y, Adj, test_set, test_size, forecast_step, model, C, P_, Q_, gamma, is_global=False,
                              print_param=True, is_test=True)[2]
        result = [direction_acc_grOU, bic]

        # Build comparable object for is_better
        comparable = [result[0], A, result[1]]

        if is_better(comparable, best_result, tol=tol):
            best_result = comparable
            best_model = model

    return best_model, best_result

def joint_graph_model_selection(_data, test_data_, TICKERS, screening_RL, P_, Q_, forecast_step, C, gamma,
                                n_candidates=1000, top_k=50, max_order=3, tol=1e-2, p=0.4):
    """
    Full two-stage joint graph-model selection procedure (Section 5.3).

    Implements the procedure described in Section 5.3 of Chen & Veraart (2026):

    **Stage 1 (Network Screening)**:
      - Generate n_candidates Erdős-Rényi graphs G(|TICKERS|, p)
      - Evaluate each using the screening model (screening_RL)
      - Retain the top_k networks by directional accuracy (BIC as tiebreaker)

    **Stage 2 (Model Selection)**:
      - For each of the top_k retained networks:
        Search over candidate grOU specifications ([1]*L and [2]*L for L=1,...,max_order)
        and select the best model by directional accuracy and BIC

    **Final Selection**:
      - Among all (network, model) pairs from Stage 2, select the pair (Ĝ, M̂)
        with the best directional accuracy (BIC as tiebreaker)

    :param _data:         Training data DataFrame (assets as columns), pd.DataFrame
    :param test_data_:    Test data DataFrame, pd.DataFrame
    :param TICKERS:       List of asset ticker symbols, List[str]
    :param screening_RL:  Lag structure for the Stage 1 screening model, List[int]
    :param P_:            Fine-grid partition, (N+1, ) np.ndarray
    :param Q_:            Coarse-grid partition, (M+1, ) np.ndarray
    :param forecast_step: Number of steps ahead to forecast, int
    :param C:             Number of covariates, int
    :param gamma:         Critical region hyperparameter, float
    :param n_candidates:  Number of random candidate networks to generate, int
    :param top_k:         Number of top networks to retain after Stage 1, int
    :param max_order:     Maximum lag order for Stage 2 model search, int
    :param tol:           Tolerance for DirAcc comparison in is_better, float
    :param p:             Erdős-Rényi edge probability for candidate generation, float
    :return:              Dictionary with keys:
                            'screening_RL':       Screening model specification
                            'all_pair_summaries': List of all (network, model) pairs
                            'Selected model':     Best R_L specification
                            'Selected final result': [DirAcc, A, BIC] for best pair
    """
    print("=== Step 1-4: Network screening ===")
    screened_networks = screen_candidate_networks(_data, test_data_, TICKERS, screening_RL, P_, Q_, forecast_step, C,
                                                  gamma, n_candidates, top_k, tol, p)

    best_pair = None
    best_summary = None
    pair_summaries = []

    print("\n=== Step 5: Model selection within each retained network ===")
    for idx, screened_result in enumerate(screened_networks, start=1):
        _, A, bic = screened_result

        # Select the best grOU model specification for this network
        best_model, model_result = evaluate_models_for_network(_data, test_data_, TICKERS, A,
                                                                    P_, Q_, forecast_step, C, gamma, max_order, tol)

        pair_summary = {
            "network_rank": idx,            # Rank from Stage 1 screening
            "network": A,                   # Node-level adjacency matrix
            "best_model": best_model,       # Best R_L specification for this network
            "final_result": model_result,   # [DirAcc, A, BIC] for best model
        }

        pair_summaries.append(pair_summary)

        # Update global best if this (network, model) pair is better
        if is_better(model_result, best_pair, tol=tol):
            best_pair = model_result
            best_summary = pair_summary

    print("\n=== Final selection ===")
    print(f"Selected model: {best_summary['best_model']}")
    print(f"Selected final result: {best_summary['final_result']}")

    return {
        "screening_RL": screening_RL,
        "all_pair_summaries": pair_summaries,
        "Selected model": best_summary['best_model'],
        "Selected final result": best_summary['final_result']
    }

def is_better(candidate, incumbent, tol):
    """
    Compare two (DirAcc, network, BIC) results for model/network selection.

    Implements the two-criterion comparison used in the joint graph-model
    selection procedure (Section 5.3 of Chen & Veraart 2026):
      1. Primary criterion: directional accuracy (higher is better)
         A candidate is better if DirAcc_candidate > DirAcc_incumbent + tol
      2. Tiebreaker: BIC (lower is better)
         If |DirAcc_candidate - DirAcc_incumbent| ≤ tol, prefer lower BIC

    :param candidate:  List [DirAcc, network, BIC] for the candidate model
    :param incumbent:  List [DirAcc, network, BIC] for the current best (or None)
    :param tol:        Tolerance for considering two DirAcc values equal, float
    :return:           True if candidate is better than incumbent, bool
    """
    # Any candidate is better than no incumbent
    if incumbent is None:
        return True

    cand, _, cand_bic = candidate
    inc, _, inc_bic = incumbent

    if cand > inc + tol:
        # Candidate has meaningfully higher directional accuracy
        return True
    elif abs(cand - inc) <= tol and cand_bic < inc_bic:
        # Tie in DirAcc: prefer lower BIC (more parsimonious model)
        return True
    return False

# ============================================================
# SIMULATION STUDY UTILITIES
# ============================================================

def process_one_sim(arg):
    """
    Estimate grOU drift parameters at multiple time horizons for one simulated path.

    Used in the finite-sample consistency experiment (Section 4.1) to assess
    whether the discretized MLE concentrates around the true parameter values
    as the observation horizon T increases.

    For each time horizon T ∈ {2, 4, 8} (= 2^1, 2^2, 2^3):
      - Extracts a sub-path of length T^7 from the full simulation
      - Sets the fine grid P with mesh Δ_P = T^{-6}
      - Sets the coarse grid Q with mesh Δ_Q = T^{-2}
      - Runs the full iterative estimation (estimate_MCAR) to get (b, Σ)
      - Runs estimate_grOU_drift with the estimated (b, Σ) to get θ_hat

    :param arg: Tuple (P, R_L, K, C, Y, adj_L_sim, W_C_sim, gamma) where:
                  - P (np.ndarray):      Full fine-grid partition for T=8
                  - R_L (List[int]):     Lag structure
                  - K (int):             Number of edges
                  - C (int):             Number of covariates
                  - Y (np.ndarray):      Simulated path, shape (K, N+1)
                  - adj_L_sim:           Line graph adjacency structure
                  - W_C_sim:             Edge weight matrices
                  - gamma (float):       Critical region hyperparameter
    :return:    result: Array of estimated parameters at 3 time horizons,
                shape (3, K*len(R_L) + sum(R_L)) np.ndarray
    """
    P, R_L, K, C, Y, adj_L_sim, W_C_sim, gamma = arg
    result = np.zeros((3, K * len(R_L) + sum(R_L)))

    for n in range(1, 4):
        T = 2 ** n      # Time horizon: T ∈ {2, 4, 8}

        # Extract sub-path of length T^7 with mesh Δ_P = T^{-6}
        # Slice: take every 2^{6*(3-n)} points from the full T=8 path
        subset_P = slice(0, int(8 ** 7 / 2 ** (3 - n)) + 1, 2 ** (6 * (3 - n)))
        P_ = P[subset_P]    # Fine grid: Δ_P = T^{-6}
        Q_ = P_[::T ** 4]   # Coarse grid: Δ_Q = T^{-2} (every T^4-th fine point)
        Y_ = Y[:, subset_P] # Sub-path

        # Step 1: Full iterative estimation to get (b, Σ)
        _, (b_sim_sol, Sigma_sim_sol) = estimate_MCAR('grOU', Y_, P_, Q_, adj_L_sim, W_C_sim, R_L, C, gamma)
        # Step 2: Estimate grOU drift θ_hat with estimated (b, Σ)
        theta_hat_sim_sol, _, _ = estimate_grOU_drift(Y_, adj_L_sim, W_C_sim, R_L, C, P_, Q_, b_sim_sol, gamma,
                                                      None, Sigma_sim_sol, with_cov=True, is_global=False)
        result[n - 1, :] = theta_hat_sim_sol.reshape(-1)

    return result

def model_pred_single(args):
    """
    Evaluate all forecasting models on a single simulated path.

    For one Monte Carlo replicate, fits and evaluates all six benchmark models
    (UAR, VAR, GNAR, OU, MCAR, grOU) plus the naive model (NA), recording
    RMSE, directional accuracy, and computation time for each.

    :param args: Tuple (index_, sim_Y, sim_test, adj_sim, forecast_step, P_Q_ratio, gamma) where:
                   - index_ (int):         Sample index in the output arrays
                   - sim_Y (np.ndarray):   Training data, shape (K, T_train)
                   - sim_test (np.ndarray):Test data, shape (K, T_test)
                   - adj_sim:              Line graph adjacency structure for grOU/GNAR
                   - forecast_step (int):  Number of steps ahead to forecast
                   - P_Q_ratio (int):      Ratio of fine-grid to coarse-grid step sizes
                   - gamma (float):        Critical region hyperparameter
    :return:    Tuple (index_, RMSE_row, direction_row, comp_row) where:
                  - RMSE_row (np.ndarray):      RMSE for 7 models (6 + NA), shape (7, )
                  - direction_row (np.ndarray): DirAcc for 6 models, shape (6, )
                  - comp_row (np.ndarray):      Computation time for 6 models (s), shape (6, )
    """
    index_, sim_Y, sim_test, adj_sim, forecast_step, P_Q_ratio, gamma = args

    # Build sliding-window test set: shape (test_size, K, forecast_step)
    sim_test_set = sliding_window_view(sim_test, (sim_Y.shape[0], forecast_step))[0]
    sim_test_size = sim_test_set.shape[0]

    # Construct fine grid P and coarse grid Q from the training data length
    # Δ_P = (N-1)^{-6/7} satisfies the mesh condition from Appendix C
    P_sim = np.array([i * ((sim_Y.shape[1] - 1) ** (-6 / 7)) for i in range(sim_Y.shape[1])])
    # Coarse grid Q: every P_Q_ratio-th point of P, with last point appended if needed
    Q_sim = np.append(P_sim[::P_Q_ratio], P_sim[-1]) if (sim_Y.shape[1] - 1) % P_Q_ratio != 0 else P_sim[::P_Q_ratio]

    RMSE_row = np.zeros(7)      # RMSE for UAR, VAR, GNAR, OU, MCAR, grOU, NA
    direction_row = np.zeros(6) # DirAcc for UAR, VAR, GNAR, OU, MCAR, grOU
    comp_row = np.zeros(6)      # Computation time for UAR, VAR, GNAR, OU, MCAR, grOU

    # ---- UAR: Univariate AR (independent per edge) ----
    start = time.perf_counter()
    forecast = UAR_model(sim_Y, sim_test_set, sim_test_size, forecast_step, 1)
    comp_row[0] = time.perf_counter() - start
    RMSE_row[0], direction_row[0] = evaluate_forecast(sim_Y, forecast, sim_test_set)

    # ---- VAR: Vector Autoregression ----
    start = time.perf_counter()
    forecast = VAR_model(sim_Y, sim_test_set, sim_test_size, forecast_step, 1)
    comp_row[1] = time.perf_counter() - start
    RMSE_row[1], direction_row[1] = evaluate_forecast(sim_Y, forecast, sim_test_set)

    # ---- GNAR: Generalized Network AR edge model ----
    start = time.perf_counter()
    forecast = GNAR_model(sim_Y, adj_sim, [1], sim_test_set, sim_test_size, forecast_step)
    comp_row[2] = time.perf_counter() - start
    RMSE_row[2], direction_row[2] = evaluate_forecast(sim_Y, forecast, sim_test_set)

    # ---- OU: Diagonal Ornstein-Uhlenbeck edge model ----
    start = time.perf_counter()
    forecast = OU_edge_model(sim_Y, sim_test_set, sim_test_size, forecast_step, [1],
                                    1, P_sim, Q_sim, gamma, is_global=False)
    comp_row[3] = time.perf_counter() - start
    RMSE_row[3], direction_row[3] = evaluate_forecast(sim_Y, forecast, sim_test_set)

    # ---- MCAR: Full multivariate continuous-time AR model ----
    start = time.perf_counter()
    forecast = MCAR_edge_model(sim_Y, sim_test_set, sim_test_size, forecast_step, P_sim, Q_sim, [1], 1, gamma)
    comp_row[4] = time.perf_counter() - start
    RMSE_row[4], direction_row[4] = evaluate_forecast(sim_Y, forecast, sim_test_set)

    # ---- grOU: Graph OU edge model (main model) ----
    start = time.perf_counter()
    forecast = grOU_edge_model(sim_Y, adj_sim, sim_test_set, sim_test_size, forecast_step,
                                            [1], 1, P_sim, Q_sim, gamma, is_global=False)
    comp_row[5] = time.perf_counter() - start
    RMSE_row[5], direction_row[5] = evaluate_forecast(sim_Y, forecast, sim_test_set)

    # ---- NA: Naive model (Y_{t+1} = Y_t) ----
    _, RMSE_row[6] = NA_model(sim_Y, sim_test_set, forecast_step)

    return index_, RMSE_row, direction_row, comp_row

def model_pred(data, test_size, adj_sim, forecast_step, P_Q_ratio, gamma, label, scenario):
    """
    Evaluate all forecasting models across all Monte Carlo replicates in parallel.

    Supports three network misspecification scenarios (Section 4.3):
      - 'correct':   Use the full network (all K edges)
      - 'missing_5': Remove the edge with α=5 (dominant autoregressive edge)
      - 'missing_1': Remove the edge with α=1 (weakly persistent edge)

    Results are saved as .npy files for subsequent analysis and plotting.

    :param data:          Simulated paths, shape (n_samples, K, T_total) np.ndarray
    :param test_size:     Number of time steps reserved for out-of-sample evaluation, int
    :param adj_sim:       Line graph adjacency structure for grOU/GNAR models
    :param forecast_step: Number of steps ahead to forecast, int
    :param P_Q_ratio:     Ratio of fine-grid to coarse-grid step sizes, int
    :param gamma:         Critical region hyperparameter, float
    :param label:         Misspecification scenario: 'correct', 'missing_5', or 'missing_1', str
    :param scenario:      Simulation scenario name (used for output directory), str
    :return:              None. Saves 'rmse_{label}.npy', 'dir_{label}.npy',
                          'time_{label}.npy' to the scenario output directory.
    """
    # Apply network misspecification: remove one edge from the data
    if label == 'missing_5':
        data = data[:, 1:, :]   # Remove first edge (α=5, dominant)
    elif label == 'missing_1':
        data = data[:, :-1, :]  # Remove last edge (α=1, weakly persistent)

    n = data.shape[0]
    RMSE = np.zeros((n, 7))
    direction_acc = np.zeros((n, 6))
    comp_time = np.zeros((n, 6))

    # Build argument list for parallel workers: split train/test for each replicate
    args_list = [(i, data[i, :, :-test_size], data[i, :, -test_size:], adj_sim, forecast_step, P_Q_ratio,
                  gamma) for i in range(n)]

    # Parallel evaluation across all Monte Carlo replicates
    with Pool(os.cpu_count()) as pool:
        for idx, rmse_row, dir_row, comp_row in pool.imap_unordered(model_pred_single, args_list):
            RMSE[idx] = rmse_row
            direction_acc[idx] = dir_row
            comp_time[idx] = comp_row

    # Save results for plotting
    with open('./data/LOBSTERdata/output/sim/' + scenario + '/rmse_' + label + '.npy', 'wb') as f:
        np.save(f, RMSE)
    with open('./data/LOBSTERdata/output/sim/' + scenario + '/dir_' + label + '.npy', 'wb') as f:
        np.save(f, direction_acc)
    with open('./data/LOBSTERdata/output/sim/' + scenario + '/time_' + label + '.npy', 'wb') as f:
        np.save(f, comp_time)

def boxplot_(sim, scenario):
    """
    Generate half-violin boxplots for RMSE, directional accuracy, and computation time.

    Produces the figures shown in Sections 4.2 and 4.3 of Chen & Veraart (2026):
      - For each misspecification scenario ('correct', 'missing_5', 'missing_1'):
        A 3-panel figure with boxplots + half-violin KDE overlays for RMSE,
        directional accuracy, and computation time across all models.
      - A 2-panel summary figure comparing grOU RMSE and directional accuracy
        across the three misspecification scenarios.

    The half-violin overlay is constructed by:
      1. Fitting a Gaussian KDE to the 5th–95th percentile range of each model's results
      2. Normalizing the density to a maximum width of 0.25 (in plot units)
      3. Plotting the density to the right of each box

    :param sim:      Simulated paths array, shape (n_samples, K, T_total) np.ndarray
                     (used only for plotting a representative sample path)
    :param scenario: Simulation scenario name (used for output directory and file names), str
    :return:         None. Saves .eps figures to the scenario output directory.
    """
    # ---- Plot a representative sample path ----
    plt.ylim([-6, 6])
    plt.plot(sim[0, :, :].T)
    plt.tick_params(axis='both', labelsize=20)
    plt.savefig('./data/LOBSTERdata/output/sim/' + scenario + '/sample_path.eps', bbox_inches='tight', format='eps')

    # Arrays to accumulate grOU results across scenarios for the summary plot
    grOU_RMSE = np.zeros((sim.shape[0], 3))
    grOU_direction_acc = np.zeros((sim.shape[0], 3))

    # ---- Per-scenario plots ----
    for i, label in enumerate(['correct', 'missing_5', 'missing_1']):
        # Load results for this scenario
        with open('./data/LOBSTERdata/output/sim/' + scenario + '/rmse_' + label + '.npy', 'rb') as f:
            RMSE = np.load(f)
        grOU_RMSE[:, i] = RMSE[:, 5]
        RMSE = RMSE[np.isfinite(RMSE).all(axis=1)]

        with open('./data/LOBSTERdata/output/sim/' + scenario + '/dir_' + label + '.npy', 'rb') as f:
            direction_acc = np.load(f)
        grOU_direction_acc[:, i] = direction_acc[:, 5]
        direction_acc = direction_acc[np.isfinite(direction_acc).all(axis=1)]

        with open('./data/LOBSTERdata/output/sim/' + scenario + '/time_' + label + '.npy', 'rb') as f:
            comp_time = np.load(f)
            comp_time = comp_time[~np.isnan(comp_time).any(axis=1)]

        # ---- 3-panel figure: RMSE | DirAcc | Computation Time ----
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: RMSE boxplots + half-violin KDE
        positions = np.arange(1, 8)
        ax[0].boxplot(RMSE[:, :], positions=positions, widths=0.3, showfliers=False)
        for j in range(7):
            kde = gaussian_kde(RMSE[:, j], bw_method=0.4)
            low, high = np.percentile(RMSE[:, j], [5, 95])
            x_vals = np.linspace(low, high, 100)
            density = kde(x_vals)
            density = density / density.max() * 0.25        # Normalize to max width 0.25
            ax[0].plot(positions[j] + density, x_vals)      # Plot to the right of box
        ax[0].set_xticks(positions)
        ax[0].set_xticklabels(['UAR', 'VAR', 'GNAR', 'OU', 'MCAR', 'grOU', 'NA'])
        ax[0].set_ylabel('RMSE', fontsize=16)
        ax[0].tick_params(axis='both', labelsize=14)

        # Panel 2: Directional accuracy boxplots + half-violin KDE
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
        ax[1].set_ylabel('Directional accuracy', fontsize=16)
        ax[1].tick_params(axis='both', labelsize=14)

        # Panel 3: Computation time boxplots + half-violin KDE
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
        ax[2].set_ylabel('Computational Time', fontsize=16)
        ax[2].tick_params(axis='both', labelsize=14)

        plt.tight_layout()
        plt.show()
        fig.savefig('./data/LOBSTERdata/output/sim/' + scenario + '/boxplot_' + label + '.eps', bbox_inches='tight',
                    format='eps')

        # Print summary statistics table
        df = pd.DataFrame(
            [np.mean(RMSE, axis=0), np.std(RMSE, axis=0), np.mean(direction_acc, axis=0), np.std(direction_acc, axis=0),
             np.mean(comp_time, axis=0), np.std(comp_time, axis=0)],
            columns=['UAR', 'VAR', 'GNAR', 'OU', 'MCAR', 'grOU', 'NA'],
            index=['RMSE_mean', 'RMSE_std', 'Directional_accuracy_mean', 'Directional_accuracy_std',
                   'Computational_time_mean', 'Computational_time_std'])
        print(label)
        print(df)

    # ---- Summary figure: grOU across misspecification scenarios ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    positions = np.arange(1, 4)
    ax[0].boxplot(grOU_RMSE[:, :], positions=positions, widths=0.3, showfliers=False)
    ax[0].tick_params(axis='both', labelsize=14)
    for i in range(3):
        x = grOU_RMSE[:, i]
        x = x[np.isfinite(x)]
        kde = gaussian_kde(x, bw_method=0.4)
        low, high = np.percentile(x, [5, 95])
        x_vals = np.linspace(low, high, 100)
        density = kde(x_vals)
        density = density / density.max() * 0.25
        ax[0].plot(positions[i] + density, x_vals)
    ax[0].set_xticks(positions)
    ax[0].set_xticklabels(['correct', 'missing_5', 'missing_1'])
    ax[0].set_ylabel('RMSE', fontsize=16)

    positions = np.arange(1, 4)
    ax[1].boxplot(grOU_direction_acc, positions=positions, widths=0.3)
    ax[1].tick_params(axis='both', labelsize=14)
    for i in range(3):
        y = grOU_direction_acc[:, i]
        y = y[np.isfinite(y)]
        kde = gaussian_kde(y, bw_method=0.4)
        low, high = np.percentile(y, [5, 95])
        x_vals = np.linspace(low, high, 100)
        density = kde(x_vals)
        density = density / density.max() * 0.25
        ax[1].plot(positions[i] + density, x_vals)
    ax[1].set_xticks(positions)
    ax[1].set_xticklabels(['correct', 'missing_5', 'missing_1'])
    ax[1].set_ylabel('Directional accuracy', fontsize=16)

    plt.tight_layout()
    plt.show()
    fig.savefig('./data/LOBSTERdata/output/sim/' + scenario + '/boxplot_grOU.eps', bbox_inches='tight', format='eps')