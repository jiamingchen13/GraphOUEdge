from data_process import *
from mcar.estimate import *
from mcar.simulate import *
from graphOUedge import GraphOUEdge
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
forecast_step = 5

# Load empirical data
data = np.load('./data/LOBSTERdata/output/train/aggregate/cov.npz')
index = pd.to_datetime(data["timestamp"])
columns = [key for key in data.files if key != "timestamp"]
cov_data = pd.DataFrame({col: data[col] for col in columns}, index=index)
data.close()

# Derive network structure
Adj = line_graph(cov_data, TICKERS, rule)

# Number of edges
K = Adj[0].shape[0]

# Derive the adjacency matrix for edge graph
Adj_L = adjacency_matrices(Adj[0], max(R_L))

# Compute the edge weight
W_C = edge_weight(Adj_L)

# Training data
df_cov = cov_data.resample(rule).sum()
df_cov = df_cov.loc[df_cov.index.intersection(cov_data.index)]
Y = df_cov[Adj[1]].T

# Set time grids
n = int(np.ceil((Y.shape[1] - 1) ** (1 / 7)))
P_ = np.linspace(0, (Y.shape[1] - 1), Y.shape[1])
Q_ = np.append(P_[::n ** 4], P_[-1]) if (Y.shape[1] - 1) % n ** 4 != 0 else P_[::n ** 4]

# Testing data
test_data = np.load('./data/LOBSTERdata/output/test/aggregate/cov.npz')
test_index = pd.to_datetime(test_data["timestamp"])
test_columns = [key for key in test_data.files if key != "timestamp"]
test_cov_data = pd.DataFrame({col: test_data[col] for col in test_columns}, index=test_index)
test_data.close()

df_test_cov = test_cov_data.resample(rule).sum()
df_test_cov = df_test_cov.loc[df_test_cov.index.intersection(test_cov_data.index)]
test_Y = df_test_cov[Adj[1]].T
test_set = sliding_window_view(test_Y, (K, forecast_step))[0]
test_size = test_set.shape[0]

if __name__ == "__main__":
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

    jump_F_OU = scipy.stats.multivariate_normal(mean=F_mean_OU, cov=(F_squared_OU - np.outer(F_mean_OU, F_mean_OU)),
                                                allow_singular=True)
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
                                        constraints=({'type': 'ineq', 'fun': lambda x: hurwitz_principal_minors(x)},
                                                     {'type': 'ineq', 'fun': lambda x: x}), method='SLSQP').x
            theta_hat_OU = np.clip(theta_hat_OU, 1e-16, None)
            X_OU.alpha_set(theta_hat_OU)
            X_OU.beta_set([[0] * stage * C for stage in R_L])
            Q_Q = X_OU.Q_matrix()

        alphas_OU.append(theta_hat_OU)
        forecast_OU[i, :, :] = np.mean(np.array([simulate_MCAR_compound_poisson([i for i in range(forecast_step + 1)],
                               Q_Q, np.append(window_data[:, -1], [0] * K * (len(R_L) - 1)), b_OU, Sigma_OU, 1, jump_F_OU, uniform=True)[:, 1:] for _ in range(1)]), axis=0)
        df_OU = np.append(df_OU, forecast_OU[i, :, 0:1], axis=1)

    with open(os.path.join('data\\LOBSTERdata\\output\\train\\aggregate', 'forecast_OU.npy'), 'wb') as f:
        np.save(f, forecast_OU)

    RMSE_OU = np.sqrt(np.mean((forecast_OU - test_set) ** 2))
    print(f'RMSE for OU Edge model is: {RMSE_OU}')














