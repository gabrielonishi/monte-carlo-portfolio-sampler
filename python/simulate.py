import pandas as pd
import numpy as np
from numpy import ndarray, float64
from typing import Annotated

import logging

logger = logging.getLogger(__name__)


def generate_weights(n: int, max_weight: float, num_samples: int) -> ndarray:
    weights_matrix = np.empty((num_samples, n), dtype=float64)
    portfolio_size = np.ones(n)
    i = 0

    while i < num_samples:
        spots_left = num_samples - i
        batch_size = 2 * spots_left
        batch = np.random.dirichlet(alpha=portfolio_size, size=batch_size)
        valid_weights = batch[np.all(batch <= max_weight, axis=1)]
        
        spots_to_be_filled = min(len(valid_weights), num_samples - i)

        if spots_to_be_filled > 0:
            weights_matrix[i:i+spots_to_be_filled] = valid_weights[:spots_to_be_filled]
            i += spots_to_be_filled

    return weights_matrix


def calculate_sharpe_ratio(tickers: list[str], weights: ndarray) -> float:
    ...


def run(
        portfolios_tickers: list[list[str]], 
        financial_data_by_ticker: dict[str, pd.DataFrame], 
        max_asset_weight: float, 
        num_samples: int) -> tuple[list, ndarray, float]:
    
    if len(portfolios_tickers) == 0:
        return [], np.empty(shape=(0,), dtype=float64), 0

    head_tickers, tail_tickers = portfolios_tickers[0], portfolios_tickers[1:]

    top_tail_tickers, top_tail_weights, top_tail_sharpe = run(
        tail_tickers, financial_data_by_ticker, max_asset_weight, num_samples)

    head_weights = generate_weights(
        len(portfolios_tickers), max_asset_weight, num_samples)
    
    head_sharpe = calculate_sharpe_ratio(head_tickers, head_weights)

    if head_sharpe > top_tail_sharpe:
        return head_tickers, head_weights, head_sharpe
    else:
        return top_tail_tickers, top_tail_weights, top_tail_sharpe


if __name__ == '__main__':
    t = generate_weights(25, 0.2, num_samples=1000)
    print(t)
