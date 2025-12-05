from data_process import *
from mcar.simulate import *
from numpy.lib.stride_tricks import sliding_window_view

# Initialization

# Tickers
TICKERS = ['SPY', 'JPM', 'KO', 'MSFT']

# Maximum neighbor for each lag
R_L = [1]

# Number of covariate
C = 1

# Aggregation interval for realized covariance
rule = '5min'

# Set jumps to be a compounded Poisson process
jump_activity = 'finite'

# Forecast steps
forecast_step = 1

cov_data = log_data_to_cov_data('./data/LOBSTERdata/output/train/aggregate/', rule)

# Derive network structure
Adj = line_graph(cov_data, TICKERS)

# Number of edges
K = Adj[0].shape[0]

# Derive the adjacency matrix for edge graph
Adj_L = adjacency_matrices(Adj[0], max(R_L))

# Compute the edge weight
W_C = edge_weight(Adj_L)

# Training data
Y = cov_data[Adj[1]].T

# Set time grids
n = int(np.ceil((Y.shape[1] - 1) ** (1 / 7)))
P_ = np.linspace(0, (Y.shape[1] - 1), Y.shape[1])
Q_ = np.append(P_[::n ** 4], P_[-1]) if (Y.shape[1] - 1) % n ** 4 != 0 else P_[::n ** 4]

# Testing data
test_cov_data = log_data_to_cov_data('./data/LOBSTERdata/output/test/aggregate/', rule)
test_Y = test_cov_data[Adj[1]].T
test_set = sliding_window_view(test_Y, (K, forecast_step))[0]
test_size = test_set.shape[0]

if __name__ == "__main__":
    pass














