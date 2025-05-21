import logging
import pathlib
import sys
import time
from functools import lru_cache
from datetime import date

import data_loader
import simulate

ASSETS_PER_PORTFOLIO = 25
SIMULATION_START = date(2024, 8, 1)
SIMULATION_END = date(2024, 12, 31)
MAX_WEIGHT_PER_ASSET = 0.2
NUM_SIMULATIONS_PER_PORTFOLIO = 5
RISK_FREE_YF_TICKER = None
RISK_FREE_RATE = None


script_dir = pathlib.Path(sys.argv[0]).parent.resolve()
log_path = script_dir / 'app.log'
dev_data_path = script_dir / 'financial_history_data_dev.pkl'

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=log_path
)


@lru_cache
def generate_portfolio_combinations(tickers: tuple[str], portfolio_size: int) -> list[list[str]]:
    if portfolio_size == 0:
        return [[]]

    if len(tickers) < portfolio_size:
        return []

    head, tail = tickers[0], tickers[1:]

    tail_combinations = generate_portfolio_combinations(tail, portfolio_size)
    head_combinations_aux = generate_portfolio_combinations(
        tail, portfolio_size - 1)
    head_combinations = [[head] + aux for aux in head_combinations_aux]

    return head_combinations + tail_combinations


def main():
    start = time.time()
    daily_returns_by_ticker, yearly_risk_free_rate = data_loader.run(
        SIMULATION_START, SIMULATION_END, stage='DEV')

    portfolios = generate_portfolio_combinations(
        tuple(daily_returns_by_ticker.keys()), ASSETS_PER_PORTFOLIO)

    portfolios = portfolios[:20000]

    optimal_sharpe, optimal_tickers, optimal_weights = simulate.run(portfolios, daily_returns_by_ticker,
                                                                    yearly_risk_free_rate, MAX_WEIGHT_PER_ASSET, NUM_SIMULATIONS_PER_PORTFOLIO)
    
    print('Optimal Sharpe:', optimal_sharpe)
    for t, w in zip(optimal_tickers, optimal_weights):
        print(t, "{:.2f}".format(w))

    print('Time: ', "{:.2f}".format(time.time() - start), 's')

if __name__ == '__main__':
    main()
