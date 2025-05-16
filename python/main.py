import logging
import pathlib
import pickle
import sys
from functools import lru_cache
from datetime import date

import pandas as pd

import data_loader
import simulate

ASSETS_PER_PORTFOLIO = 25
NUM_SIMULATIONS = 1000
SIMULATION_START = date(2024, 8, 1)
SIMULATION_END = date(2024, 12, 31)
MAX_WEIGHT_PER_ASSET = 0.2
NUM_SAMPLES_PER_PORTFOLIO = 1000

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
    daily_returns_by_ticker = data_loader.run(SIMULATION_START, SIMULATION_END, 'DEV')
    portfolios = generate_portfolio_combinations(
        tuple(daily_returns_by_ticker.keys()), ASSETS_PER_PORTFOLIO)
    top_portfolio = simulate.run(portfolios, daily_returns_by_ticker, MAX_WEIGHT_PER_ASSET, NUM_SAMPLES_PER_PORTFOLIO)


if __name__ == '__main__':
    main()
