import cProfile
import logging
import math
import multiprocessing
import multiprocessing.shared_memory
import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd
from combinations import generate_portfolio_combinations
from generate_weights import generate_weights

NUM_SIMULATIONS_PER_PORTFOLIO = 1000
MAX_WEIGHT_PER_ASSET = 0.2
ASSETS_PER_PORTFOLIO = 25
NUM_PORTFOLIOS = 1_000
LOGGER_NAME = 'maximize_sharpe'
DATA_FILE = 'data.pkl'
DELTA = 1e-5
NUM_SIMULATIONS = NUM_SIMULATIONS_PER_PORTFOLIO * NUM_PORTFOLIOS

NP_SHARED_NAME = 'shared_returns'


def maximize_sharpe_matrix(
        tickers: list[str],
        W: np.ndarray,
        daily_returns_by_ticker: dict[str, pd.Series],
        Rf_yearly: float = 0.05) -> tuple[float, np.ndarray]:

    ANNUALIZATION_FACTOR = 252

    R_daily = np.stack([daily_returns_by_ticker[ticker]
                       for ticker in tickers], axis=0).T

    Rp_daily = R_daily @ W.T
    Rp_yearly = np.mean(Rp_daily, axis=0) * ANNUALIZATION_FACTOR
    ER_yearly = Rp_yearly - Rf_yearly

    cov_matrix = np.cov(R_daily, rowvar=False)
    # sigma2 = np.einsum('ij,jk,ik->i', W, Sigma, W)
    Var_daily = np.diag(W @ cov_matrix @ W.T)
    Vol_yearly = np.sqrt(Var_daily * ANNUALIZATION_FACTOR)

    SR = ER_yearly / Vol_yearly

    optimal_idx = np.argmax(SR)

    return SR[optimal_idx], W[optimal_idx]


def maximize_sharpe_loop(
        tickers: list[str],
        W: np.ndarray,
        daily_returns_by_ticker: dict[str, pd.Series],
        Rf_yearly: float = 0.05) -> tuple[float, np.ndarray]:

    ANNUALIZATION_FACTOR = 252

    R_daily = np.stack([daily_returns_by_ticker[ticker]
                       for ticker in tickers], axis=0).T

    cov_matrix = np.cov(R_daily, rowvar=False)

    optimal_sharpe = -np.inf
    optimal_weights = np.zeros(0)

    for w in W:
        Rp_daily = R_daily @ w.T
        Rp_yearly = np.mean(Rp_daily) * ANNUALIZATION_FACTOR
        ER_yearly = Rp_yearly - Rf_yearly
        Var_daily = w.T @ cov_matrix @ w
        Vol_yearly = np.sqrt(Var_daily * ANNUALIZATION_FACTOR)

        ratio = ER_yearly / Vol_yearly

        if ratio > optimal_sharpe:
            optimal_sharpe = ratio
            optimal_weights = w

    return optimal_sharpe, optimal_weights


def test_outputs_dummy():
    daily_returns_by_ticker = {
        'AAPL': pd.Series([0.001, 0.002, -0.001, 0.003, 0.000]),
        'GOOGL': pd.Series([0.002, 0.001, 0.000, 0.002, -0.001]),
        'MSFT': pd.Series([-0.001, 0.000, 0.002, -0.002, 0.001])
    }

    w = np.array([[0.4, 0.4, 0.2],
                  [0.2, 0.5, 0.3]])
    tickers = list(daily_returns_by_ticker.keys())
    Rf = 0.00

    sharpe_matrix, weights_matrix = maximize_sharpe_matrix(
        tickers, w, daily_returns_by_ticker, Rf)

    sharpe_loop, weights_loop = maximize_sharpe_loop(
        tickers, w, daily_returns_by_ticker, Rf)

    print(sharpe_matrix, sharpe_loop, sharpe_matrix ==
          sharpe_loop, sharpe_loop - sharpe_matrix)
    print(weights_matrix, weights_loop, np.all(weights_matrix == weights_loop))


def aux_multiprocessing(function, num_portfolios, tickers, weights, returns):
    weights = generate_weights(
        ASSETS_PER_PORTFOLIO, MAX_WEIGHT_PER_ASSET, NUM_SIMULATIONS_PER_PORTFOLIO)

    for _ in range(num_portfolios):
        __ = function(
            tickers, weights, returns)


def run_matrix_multiprocessing(
        daily_returns_by_ticker: dict[str, pd.Series],
        weights: np.ndarray,
        tickers: list[str]):

    n_processes = multiprocessing.cpu_count()
    chunk_size = math.ceil(NUM_PORTFOLIOS / n_processes)
    start_time = time.time()

    with multiprocessing.Pool() as pool:
        pool.starmap(
            aux_multiprocessing,
            [(chunk_size, tickers, weights, daily_returns_by_ticker)
                for _ in range(n_processes)]
        )

    end_time = time.time()

    total_time = end_time - start_time

    logger.info(
        f"Matrix method with multiprocessing took {total_time:.2f} seconds for {NUM_SIMULATIONS} simulations")


def run_matrix_single_process(
        daily_returns_by_ticker: dict[str, pd.Series],
        weights: np.ndarray,
        tickers: list[str]):

    start_time = time.time()
    for _ in range(NUM_PORTFOLIOS):
        __ = maximize_sharpe_matrix(
            tickers, weights, daily_returns_by_ticker)
    end_time = time.time()

    total_time = end_time - start_time

    logger.info(
        f"Matrix method in single process took {total_time:.2f} seconds for {NUM_SIMULATIONS} simulations")


def run_loop_multiprocessing(
        daily_returns_by_ticker: dict[str, pd.Series],
        weights: np.ndarray,
        tickers: list[str]):

    n_processes = multiprocessing.cpu_count()
    chunk_size = math.ceil(NUM_PORTFOLIOS / n_processes)

    start_time = time.time()
    with multiprocessing.Pool() as pool:
        pool.starmap(
            aux_multiprocessing,
            [(chunk_size, tickers, weights, daily_returns_by_ticker)
             for _ in range(n_processes)]
        )
    end_time = time.time()

    total_time = end_time - start_time

    logger.info(
        f"Loop method with multiprocessing took {total_time:.2f} seconds for {NUM_SIMULATIONS} simulations")


def run_loop_single_process(
        daily_returns_by_ticker: dict[str, pd.Series],
        weights: np.ndarray,
        tickers: list[str]):

    start_time = time.time()
    for _ in range(NUM_PORTFOLIOS):
        __ = maximize_sharpe_loop(
            tickers, weights, daily_returns_by_ticker)
    end_time = time.time()

    total_time = end_time - start_time

    logger.info(
        f"Loop method in single process took {total_time:.2f} seconds for {NUM_SIMULATIONS} simulations")

########################################################################################


def aux_real_values(function, chunk, max_asset_weight, num_samples, returns):
    optimal_ratio = 0
    optimal_tickers = []
    optimal_weights = np.zeros(0)

    for tickers_idxs in chunk:
        W = generate_weights(len(tickers_idxs), max_asset_weight, num_samples)
        sharpe, weights = function(
            tickers_idxs, W, returns)

        if sharpe > optimal_ratio:
            optimal_ratio = sharpe
            optimal_tickers = tickers_idxs
            optimal_weights = weights

    return optimal_ratio, optimal_tickers, optimal_weights


def maximize_sharpe_matrix_array(
        tickers_idxs: np.ndarray,
        W: np.ndarray,
        daily_returns_matrix: np.ndarray,
        Rf_yearly: float = 0.05) -> tuple[float, np.ndarray]:

    ANNUALIZATION_FACTOR = 252

    R_daily = daily_returns_matrix[tickers_idxs].T
    Rp_daily = R_daily @ W.T
    Rp_yearly = np.mean(Rp_daily, axis=0) * ANNUALIZATION_FACTOR
    ER_yearly = Rp_yearly - Rf_yearly

    cov_matrix = np.cov(R_daily, rowvar=False)
    Var_daily = np.diag(W @ cov_matrix @ W.T)
    Vol_yearly = np.sqrt(Var_daily * ANNUALIZATION_FACTOR)

    SR = ER_yearly / Vol_yearly

    optimal_idx = np.argmax(SR)

    return SR[optimal_idx], W[optimal_idx]


def maximize_sharpe_loop_array(
        tickers_idxs: np.ndarray,
        W: np.ndarray,
        daily_returns_matrix: np.ndarray,
        Rf_yearly: float = 0.05) -> tuple[float, np.ndarray]:

    ANNUALIZATION_FACTOR = 252

    R_daily = daily_returns_matrix[tickers_idxs].T
    cov_matrix = np.cov(R_daily, rowvar=False)

    optimal_sharpe = -np.inf
    optimal_weights = np.zeros(0)

    for w in W:
        Rp_daily = R_daily @ w.T
        Rp_yearly = np.mean(Rp_daily) * ANNUALIZATION_FACTOR
        ER_yearly = Rp_yearly - Rf_yearly
        Var_daily = w.T @ cov_matrix @ w
        Vol_yearly = np.sqrt(Var_daily * ANNUALIZATION_FACTOR)

        ratio = ER_yearly / Vol_yearly

        if ratio > optimal_sharpe:
            optimal_sharpe = ratio
            optimal_weights = w

    return optimal_sharpe, optimal_weights


def run_matrix_array_multiprocessing(
        array_returns: np.ndarray,
        tickers_idxs: np.ndarray):

    n_processes = multiprocessing.cpu_count()
    chunk_size = math.ceil(NUM_PORTFOLIOS / n_processes)
    start_time = time.time()

    with multiprocessing.Pool() as pool:
        pool.starmap(
            aux_multiprocessing,
            [(maximize_sharpe_matrix_array, chunk_size, tickers_idxs, array_returns)
                for _ in range(n_processes)]
        )

    end_time = time.time()
    total_time = end_time - start_time

    logger.info(
        f"Matrix method with multiprocessing with array inputs took {total_time:.2f} seconds for {NUM_SIMULATIONS:_} simulations")


def run_matrix_array_single_process(
        array_returns: np.ndarray,
        weights: np.ndarray,
        tickers_idxs: np.ndarray):

    start_time = time.time()
    for _ in range(NUM_PORTFOLIOS):
        __ = maximize_sharpe_matrix_array(
            tickers_idxs, weights, array_returns)
    end_time = time.time()
    total_time = end_time - start_time

    logger.info(
        f"Matrix method in single process with array inputs took {total_time:.2f} seconds for {NUM_SIMULATIONS:_} simulations")

########################################################################################


def aux_multiprocessing_shared(  # noqa: PLR0913, PLR0917
        function, num_portfolios, tickers_idxs, array_shape, array_type, shm_name=NP_SHARED_NAME):

    shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
    returns = np.ndarray(
        shape=array_shape,
        dtype=array_type,
        buffer=shm.buf)

    weights = generate_weights(
        ASSETS_PER_PORTFOLIO, MAX_WEIGHT_PER_ASSET, NUM_SIMULATIONS_PER_PORTFOLIO)

    for _ in range(num_portfolios):
        __ = function(tickers_idxs, weights, returns)


def create_shared_memory_nparray(returns: np.ndarray, name: str = NP_SHARED_NAME):

    d_size = np.dtype(returns.dtype).itemsize * np.prod(returns.shape)

    shm = multiprocessing.shared_memory.SharedMemory(
        create=True, size=d_size, name=name)
    # numpy array on shared memory buffer
    dst = np.ndarray(shape=returns.shape, dtype=returns.dtype, buffer=shm.buf)
    dst[:] = returns[:]
    print(f'NP SIZE: {(dst.nbytes / 1024) / 1024}')
    return shm


def release_shared(name: str = NP_SHARED_NAME):
    shm = multiprocessing.shared_memory.SharedMemory(name=name)
    shm.close()
    shm.unlink()


def run_matrix_shared_multiprocessing(
        array_returns: np.ndarray,
        tickers_idxs: np.ndarray):

    _ = create_shared_memory_nparray(array_returns)

    n_processes = multiprocessing.cpu_count()
    chunk_size = math.ceil(NUM_PORTFOLIOS / n_processes)
    start_time = time.time()

    with multiprocessing.Pool() as pool:
        pool.starmap(
            aux_multiprocessing_shared,
            [(maximize_sharpe_matrix_array, chunk_size, tickers_idxs, array_returns.shape, array_returns.dtype)
                for _ in range(n_processes)]
        )

    end_time = time.time()
    total_time = end_time - start_time

    release_shared()

    logger.info(
        f"Matrix method with multiprocessing with array inputs took {total_time:.2f} seconds for {NUM_SIMULATIONS:_} simulations")


########################################################################################


def run_profiled(
        function,
        returns,
        weights,
        tickers):

    profiler = cProfile.Profile()
    profiler.enable()

    function(
        returns, weights, tickers)

    profiler.disable()
    profiler.print_stats(sort='cumtime')


def run_profiled_no_weights(
        function,
        returns,
        tickers):

    profiler = cProfile.Profile()
    profiler.enable()

    function(
        returns, tickers)

    profiler.disable()
    profiler.print_stats(sort='cumtime')


if __name__ == '__main__':
    script_dir = pathlib.Path(sys.argv[0]).parent.resolve()
    log_path = script_dir / f'{LOGGER_NAME}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        filename=log_path
    )
    logger = logging.getLogger(LOGGER_NAME)

    with open(script_dir / DATA_FILE, 'rb') as f:
        daily_returns_by_ticker: dict[str, pd.Series] = pickle.load(f)

    tickers = list(daily_returns_by_ticker.keys())

    tickers_idxs = generate_portfolio_combinations(
        ASSETS_PER_PORTFOLIO, len(tickers))

    array_returns = np.stack([daily_returns_by_ticker[ticker]
                             for ticker in tickers])

    tested_tickers = tickers_idxs[:NUM_PORTFOLIOS]

    optimal_sharpe, optimal_tickers_idxs, optimal_weights = aux_real_values(
        maximize_sharpe_matrix_array, tested_tickers, MAX_WEIGHT_PER_ASSET, NUM_SIMULATIONS_PER_PORTFOLIO, array_returns)

    print(f'Optimal Sharpe Ratio: {optimal_sharpe}')
    print(f'Optimal Tickers Indices: {optimal_tickers_idxs}')
    print(f'Optimal Weights: {optimal_weights}')

    optimal_tickers = [tickers[i] for i in optimal_tickers_idxs]

    print(f'Optimal Tickers: {optimal_tickers}')

    res = maximize_sharpe_matrix(
        optimal_tickers,
        np.array([optimal_weights]),
        daily_returns_by_ticker
    )
    print(res)

    # s1, w1 = maximize_sharpe_matrix_array(
    #     tickers_idxs, weights, array_returns)
    # s2, w2 = maximize_sharpe_matrix(
    #     tickers, weights, daily_returns_by_ticker
    # )

    # run_loop_multiprocessing(
    # daily_returns_by_ticker, weights, tickers)
    # run_loop_single_process(
    # daily_returns_by_ticker, weights, tickers)

    # run_matrix_multiprocessing(
    #     daily_returns_by_ticker, weights, tickers)
    # run_matrix_single_process(
    #     daily_returns_by_ticker, weights, tickers)

    # run_matrix_single_process(
    #     daily_returns_by_ticker, weights, tickers)
    # run_loop_single_process(
    #     daily_returns_by_ticker, weights, tickers)
