from itertools import combinations

import numpy as np


def generate_portfolio_combinations(assets_per_portfolio: int, total_assets) -> np.ndarray:
    '''
    Based on the number of assets in the portfolio and the total number of assets,
    generate all possible combinations of portfolios.
    '''
    if assets_per_portfolio > total_assets:
        raise ValueError(
            "Number of assets in portfolio cannot exceed total number of assets.")
    return np.array(list(combinations(range(total_assets), assets_per_portfolio)))
