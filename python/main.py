import logging
import pathlib
import pickle
import sys
from functools import lru_cache
from datetime import date

import pandas as pd

import data_loader
import simulate

PROD = False
ASSETS_PER_PORTFOLIO = 25
NUM_COMPANIES = 30
NUM_SIMULATIONS = 1000
SIMULATION_START = date(2024, 8, 1)
SIMULATION_END = date(2024, 12, 31)

script_dir = pathlib.Path(sys.argv[0]).parent.resolve()
log_path = script_dir / 'app.log'
dev_data_path = script_dir / 'financial_history_data_dev.pkl'

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=log_path
)


def fetch_data() -> tuple[list[str], dict[str, pd.DataFrame]]:
    if PROD:
        tickers = data_loader.get_djia_tickers()
        financial_data_by_ticker = data_loader.load_data_from_yf(
            tickers, SIMULATION_START, SIMULATION_END)
    else:
        with open(file=dev_data_path, mode='rb') as f:
            financial_data_by_ticker = pickle.load(file=f)
            tickers = list(financial_data_by_ticker.keys())[:NUM_COMPANIES]
            financial_data_by_ticker = {
                ticker: financial_data_by_ticker[ticker] for ticker in tickers}

    return tickers, financial_data_by_ticker


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
    tickers, financial_data_by_ticker = fetch_data()
    portfolios = generate_portfolio_combinations(
        tuple(tickers), ASSETS_PER_PORTFOLIO)
    top_portfolio = simulate.run(portfolios, financial_data_by_ticker)


if __name__ == '__main__':
    main()
