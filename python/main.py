import logging
import pathlib
import pickle
import sys
from multiprocessing import Pool
from time import time
from functools import lru_cache

import pandas as pd

import data_loader as dl

PROD = False
ASSETS_PER_PORTFOLIO = 25
NUM_COMPANIES = 30

script_dir = pathlib.Path(sys.argv[0]).parent.resolve()
log_path = script_dir / 'app.log'
dev_data_path = script_dir / 'financial_history_data_dev.pkl'

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=log_path
)


def fetch_data() -> dict[str : pd.DataFrame]:
    if PROD:
        companies_tickers = dl.get_djia_companies()
        companies_dfs = dl.load_data_from_yf(companies_tickers)
    else:
        with open(file=dev_data_path, mode='rb') as f:
            companies_dfs = pickle.load(file=f)
            companies = list(companies_dfs.keys())[:NUM_COMPANIES]
            companies_dfs = {ticker: companies_dfs[ticker] for ticker in companies}

    return companies_dfs


@lru_cache
def generate_portfolio_combinations(companies: tuple[str, ...], portfolio_len: int) -> list[list[str]]:
    if portfolio_len == 0:
        return [[]]

    if len(companies) < portfolio_len:
        return []

    head, tail = companies[0], companies[1:]

    tail_combinations = generate_portfolio_combinations(tail, portfolio_len)
    head_combinations_aux = generate_portfolio_combinations(tail, portfolio_len - 1)
    head_combinations = [[head] + aux for aux in head_combinations_aux]

    return head_combinations + tail_combinations


def main():
    companies_dfs = fetch_data()
    companies = tuple(companies_dfs.keys())
    start = time()
    portfolios = generate_portfolio_combinations(companies, ASSETS_PER_PORTFOLIO)
    print(portfolios)
    print(time() - start)

if __name__ == '__main__':
    main()
