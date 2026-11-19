"""Training-only surrogate search over discrete grOU tuning choices.

Replace validation_loss() with your own rolling-validation score. It must
return one finite scalar, with smaller meaning better. Do not use the final
held-out test period while running this search.
"""
import csv
from itertools import product

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

X_VALUES = [1, 0.1, 0.01, 0.001, 0.0001]
Y_VALUES = [18, 54, 81, 108, 144]
Z_VALUES = [1.01, 1.5, 2.5, 5, 10]
BUDGET = 20  # expensive evaluations, including initial settings
SEED = 42


def validation_loss(x, y, z):
    """Use only the training period; average the same rolling splits each call."""
    # Example if model_sensitive RETURNS a scalar validation RMSE:
    # return float(model_sensitive(x, y, z))
    # If it SAVES arrays and returns None, read the validation output here
    # instead; do not read final-test RMSE or optimize on jump labels.
    raise NotImplementedError("Connect to training-only model_sensitive output")


def feasible(x, y, z):
    """Exclude settings unsupported by actual observations or too few coarse bins."""
    # E.g. reject if x cannot be constructed from observed timestamps or
    # if the resulting training M is <= the covariance dimension K.
    return True


candidates = [tuple(c) for c in product(X_VALUES, Y_VALUES, Z_VALUES)
              if feasible(*c)]
if not candidates:
    raise ValueError("No feasible combinations")

# Log-scale positive coordinates; map each dimension to [0, 1].
raw = np.log(np.asarray(candidates, dtype=float))
lo, hi = raw.min(axis=0), raw.max(axis=0)
features = (raw - lo) / np.maximum(hi - lo, 1.0e-12)
index = {candidate: i for i, candidate in enumerate(candidates)}
rng = np.random.default_rng(SEED)

# Start with diverse feasible configurations; the baseline is included if feasible.
initial = []
for candidate in [(0.01, 81, 1.5), (1, 18, 1.01),
                  (0.0001, 144, 10)]:
    if candidate in index and candidate not in initial:
        initial.append(candidate)
remaining = [c for c in candidates if c not in initial]
rng.shuffle(remaining)
initial.extend(remaining[:max(0, min(10, BUDGET) - len(initial))])

observed = {}  # candidate -> measured validation loss

def evaluate(candidate):
    value = float(validation_loss(*candidate))
    if not np.isfinite(value):
        raise ValueError(f"Non-finite validation loss at {candidate}: {value}")
    observed[candidate] = value
    print(f"{len(observed):3d}/{min(BUDGET, len(candidates))}: "
          f"x={candidate[0]:g}, y={candidate[1]}, z={candidate[2]:g}, loss={value:.6g}")


for candidate in initial[:min(BUDGET, len(candidates))]:
    evaluate(candidate)

while len(observed) < min(BUDGET, len(candidates)):
    seen = np.array([index[c] for c in observed], dtype=int)
    unseen = np.array([i for i, c in enumerate(candidates) if c not in observed], dtype=int)
    scores = np.array(list(observed.values()), dtype=float)
    # Normalize the response to keep GP noise regularization comparable across metrics.
    scale = max(float(scores.std()), 1.0e-12)
    scaled_scores = (scores - scores.mean()) / scale
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.ones(3), length_scale_bounds=(0.05, 10.0), nu=2.5
    )
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4,
                                  normalize_y=False, n_restarts_optimizer=2,
                                  random_state=SEED)
    gp.fit(features[seen], scaled_scores)
    mean, sd = gp.predict(features[unseen], return_std=True)
    improvement = scaled_scores.min() - mean - 0.01
    u = improvement / np.maximum(sd, 1e-12)
    expected_improvement = improvement * norm.cdf(u) + sd * norm.pdf(u)
    evaluate(candidates[unseen[np.argmax(expected_improvement)]])

best = min(observed, key=observed.get)
with open("surrogate_tuning_results.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["x", "y", "z", "validation_loss"])
    writer.writerows([(*c, loss) for c, loss in observed.items()])
print(f"Best TESTED configuration: {best}; validation loss: {observed[best]:.6g}")
print(f"Evaluated {len(observed)} of {len(candidates)} feasible configurations")
