# GraphOUEdge

This repository contains code for simulating and estimating graph Ornstein-Uhlenbeck (grOU) processes for edge-indexed network time series. For an overview of multivariate CAR(MA) processes, see [Marquardt and Stelzer, 2007](https://doi.org/10.1016/j.spa.2006.05.014) and [Lucchese, Pakkanen and Veraart, 2026](https://doi.org/10.1214/25-AAP2244). The grOU model and the estimators implemented in this repository are based on the results of [Chen and Veraart, 2026](https://arxiv.org/abs/2605.15907).

## Contents

The main methods are located in `mcar.simulate.py`, `mcar.estimate.py`, `graphOUedge.py` and `data_process.py`. Examples of their usage can be found in `grOU-simulation-study.ipynb`. Specifically:

- `grOU-simulation-study.ipynb`: Contains code for a simulation study to evaluate the finite-sample empirical properties of the estimators and an empirical application to high-frequency limit order book data from the [LOBSTER](https://lobsterdata.com/) database.
- `data/LOBSTERdata/output`: This directory contains all precomputed results and generated figures.

The authors would like to acknowledge Claude Sonnet 4.6 for its assistance in polishing and restructuring the comments in those code files.