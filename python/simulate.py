import logging
import math
import multiprocessing

import numpy as np
import pandas as pd
from numpy import float64, ndarray

logger = logging.getLogger(__name__)


def generate_weights(number_of_assets: int, max_weight: float, num_samples: int) -> ndarray:
    if 1 / number_of_assets > max_weight:
        raise ValueError('Impossible Condition')

    weights_matrix = np.empty((num_samples, number_of_assets), dtype=float64)
    portfolio_size = np.ones(number_of_assets)
    BATCH_SIZE_MULTIPLIER = 1.2

    i = 0

    while i < num_samples:
        spots_left = num_samples - i
        batch_size = int(BATCH_SIZE_MULTIPLIER * spots_left)
        batch = np.random.dirichlet(alpha=portfolio_size, size=batch_size)
        valid_weights = batch[np.all(batch <= max_weight, axis=1)]
        spots_to_be_filled = min(len(valid_weights), num_samples - i)

        if spots_to_be_filled > 0:
            weights_matrix[i:i +
                           spots_to_be_filled] = valid_weights[:spots_to_be_filled]
            i += spots_to_be_filled

    return weights_matrix


def simulate_portfolio(
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


def simulate_portfolios(chunk, daily_returns_by_ticker, max_asset_weight, num_samples):

    optimal_ratio = 0
    optimal_tickers = []
    optimal_weights = np.zeros(0)

    for tickers in chunk:
        W = generate_weights(len(tickers), max_asset_weight, num_samples)
        sharpe, weights = simulate_portfolio(
            tickers, W, daily_returns_by_ticker)

        if sharpe > optimal_ratio:
            optimal_ratio = sharpe
            optimal_tickers = tickers
            optimal_weights = weights

    return optimal_ratio, optimal_tickers, optimal_weights


def run(
        portfolios_tickers: list[list[str]],
        daily_returns_by_ticker: dict[str, pd.DataFrame],
        max_asset_weight: float,
        num_samples: int) -> tuple[float, list, ndarray]:

    n_processes = multiprocessing.cpu_count()
    chunk_size = math.ceil(len(portfolios_tickers) / n_processes)
    chunks = [portfolios_tickers[i:i + chunk_size]
              for i in range(0, len(portfolios_tickers), chunk_size)]

    args = [
        (chunk.copy(), daily_returns_by_ticker, max_asset_weight, num_samples)
        for chunk in chunks
    ]

    with multiprocessing.Pool() as pool:
        results = pool.starmap(simulate_portfolios, args)

    optimal_overall_sharpe, optimal_overall_tickers, optimal_overall_weights = 0, [
        ""], np.zeros(0)
    for sharpe, tickers, weights in results:
        if sharpe > optimal_overall_sharpe:
            optimal_overall_sharpe = sharpe
            optimal_overall_tickers = tickers
            optimal_overall_weights = weights

    return optimal_overall_sharpe, optimal_overall_tickers, optimal_overall_weights


if __name__ == '__main__':
    import main
    main.main()
