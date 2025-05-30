import pandas as pd
import numpy as np
from numpy import ndarray, float64

import logging
import multiprocessing

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


def simulate_optimal_weights_matrix(tickers: list[str], W: ndarray, daily_returns_by_ticker: dict[str, pd.DataFrame], Rf: float) -> tuple[float, ndarray]:   

    R_daily = np.stack([daily_returns_by_ticker[ticker] for ticker in tickers], axis=0).T
    Rp_daily = R_daily @ W.T
    Rp_yearly = Rp_daily.mean(axis=0) * 252
    ER_yearly = Rp_yearly - Rf  
    
    Sigma = np.cov(R_daily, rowvar=False)
    sigma2 = np.einsum('ij,jk,ik->i', W, Sigma, W)
    sigma_yearly = np.sqrt(sigma2 * 252)

    SR = ER_yearly / sigma_yearly 

    optimal_idx = np.argmax(SR)

    return SR[optimal_idx], W[optimal_idx]


def simulate_portfolios(args_list):
    tickers, daily_returns_by_ticker, yearly_risk_free_rate, max_asset_weight, num_samples = args_list
    weights_list = generate_weights(len(tickers), max_asset_weight, num_samples)
    sharpe, weights = simulate_optimal_weights_matrix(tickers, weights_list, daily_returns_by_ticker, yearly_risk_free_rate)
    
    return sharpe, tickers, weights

def run(
        portfolios_tickers: list[list[str]],
        daily_returns_by_ticker: dict[str, pd.DataFrame],
        yearly_risk_free_rate: float,
        max_asset_weight: float,
        num_samples: int) -> tuple[float, list, ndarray]:

    args_list = [
        (tickers, daily_returns_by_ticker, yearly_risk_free_rate, max_asset_weight, num_samples)
        for tickers in portfolios_tickers
    ]

    results = [simulate_portfolios(args) for args in args_list]

    with multiprocessing.Pool() as pool:
        results = pool.map(simulate_portfolios, args_list)

    optimal_overall_sharpe, optimal_overall_tickers, optimal_overall_weights = 0, [""], np.zeros(0)
    for sharpe, tickers, weights in results:
        if sharpe > optimal_overall_sharpe:
            optimal_overall_sharpe = sharpe
            optimal_overall_tickers = tickers
            optimal_overall_weights = weights

    return optimal_overall_sharpe, optimal_overall_tickers, optimal_overall_weights
        

if __name__ == '__main__':
    import main
    main.main()
