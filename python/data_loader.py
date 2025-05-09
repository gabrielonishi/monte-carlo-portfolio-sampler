"""Fetches data from Yahoo Finance."""

import logging
import pathlib
import pickle
import sys

import pandas as pd
import requests
import yfinance as yf

DOW_JONES_WIKI_URL = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
SIMULATION_START_DATE = '2024-08-01'
SIMULATION_END_DATE = '2024-12-31'
TIMEOUT = 10

logger = logging.getLogger(__name__)


def get_djia_companies() -> list[str]:
    """Get Tickers from Dow Jones components by accessing Wikipedia article."""
    try:
        response = requests.get(DOW_JONES_WIKI_URL, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.exception("Couldn't fetch data from Wikipedia", extra={'exception': str(e)})
        return []

    tables = pd.read_html(response.content)

    for table in tables:
        if 'Company' in table and 'Exchange' in table and 'Symbol' in table:
            logger.info('Fetched information from Wikipedia succesfully')
            return list(table['Symbol'])

    logger.error('Could not identify companies from Wikipedia tables')

    return []


def load_data_from_yf(tickers: list[str]) -> dict[str : pd.DataFrame | None]:
    """Get data from YahooFinance using tickers and simulation period."""
    dfs = {}

    for ticker in tickers:
        try:
            df_ticker = yf.download(
                ticker, start=SIMULATION_START_DATE, end=SIMULATION_END_DATE, progress=False, timeout=TIMEOUT
            )
        except TimeoutError as e:
            logger.exception("Couldn't fetch data from Yahoo Finance", extra={'error': e})
        if df_ticker is None:
            logger.exception("Couldn't fetch data from Yahoo Finance", extra={'company': ticker})
        dfs[ticker] = df_ticker
    return dfs


if __name__ == '__main__':
    script_dir = pathlib.Path(sys.argv[0]).parent.resolve()
    log_path = script_dir / 'app.log'
    dev_data_path = script_dir / 'financial_history_data_dev.pkl'

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=log_path
    )

    companies = get_djia_companies()
    companies_data = load_data_from_yf(companies)

    with open(dev_data_path, mode='wb') as f:
        pickle.dump(companies_data, file=f)
