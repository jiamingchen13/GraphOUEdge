from graphOUedge import GraphOUEdge
from graphOUedgehpc import GraphOUEdgehpc
from data_process import *
import numpy as np
import os

# set global parameters
R = [1, 2]
K = 4
C = 1
alpha = [2, 1]
W_C = [[[0, 1/3, 1/3, 1/3], [1/2, 0, 1/2, 1], [1/3, 1/3, 0, 1/3], [1/2, 1, 1/2, 0]]]
Adj = [[[0, 1, 1, 1], [1, 0, 1, 0], [1, 1 ,0 ,1], [1, 0, 1, 0]], [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]]]
beta = [[1], [0.75, 0.5]]
rate = 1
n_samples = 20
betas = [-np.inf]
jump_regimes = ['BM']
data_dir = 'data_' + str(n_samples)
os.makedirs(data_dir, exist_ok=True)
plots_dir = 'plots_' + str(n_samples)
os.makedirs(plots_dir, exist_ok=True)

'''
TICKERS = ['SPY', 'JPM', 'KO', 'MSFT']
rules = ['5min', '15min', '30min', '60min']
'''
if __name__ == "__main__":

    X = GraphOUEdge(R, K, C, Adj, W_C, data_dir, plots_dir)
    X.alpha_set(alpha)
    X.beta_set(beta)
    X.Q_matrix()
    X.simulate('BM', n_samples)
    #X.simulate('CP', n_samples, rate)
    #X.simulate('Gamma', n_samples)
    X.estimate_with_known_Levy_global(jump_regimes, betas)
    X.plot_alpha_global(jump_regimes)
    X.plot_beta_global(jump_regimes)
    '''
    for TICKER in TICKERS:
        input_path = './data/LOBSTERdata/_data_dwn_16_88__' + TICKER + '_2025-08-01_2025-08-31_1'
        log_path = './data/LOBSTERdata/log/' + TICKER + '_processing_logs'
        os.makedirs(log_path, exist_ok=True)
        output_path = './data/LOBSTERdata/output/' + TICKER
        os.makedirs(output_path, exist_ok=True)
        multiprocess_orderbooks(TICKER=TICKER,
                            input_path=input_path,
                            output_path=output_path,
                            log_path=log_path,
                            check_for_processed_data=True)
    
    aggregate_assets(TICKERS)
    '''
















