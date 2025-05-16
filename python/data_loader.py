"""Fetches data from Yahoo Finance."""

import logging
from datetime import date

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


def get_djia_tickers() -> list[str]:
    """Get Tickers from Dow Jones components by accessing Wikipedia article."""

    DOW_JONES_WIKI_URL = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'

    try:
        response = requests.get(DOW_JONES_WIKI_URL)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.critical("Couldn't fetch data from Wikipedia",
                        extra={'exception': str(e)})
        return [""]

    tables: list[pd.DataFrame] = pd.read_html(response.content.decode('utf-8'))

    for table in tables:
        if 'Company' in table and 'Exchange' in table and 'Symbol' in table:
            logger.info('Fetched information from Wikipedia succesfully')
            return list(table['Symbol'])

    logger.critical('Could not identify companies from Wikipedia tables')

    return [""]


def load_data_from_yf(tickers: list[str], simulation_start: date, simulation_end: date) -> dict[str, pd.DataFrame]:
    """Get data from YahooFinance using tickers and simulation period."""
    dfs = {}

    start = simulation_start.isoformat()
    end = simulation_end.isoformat()

    for ticker in tickers:
        df_ticker = None
        try:
            df_ticker = yf.download(
                ticker, start=start, end=end, progress=False
            )
        except TimeoutError as e:
            logger.exception(
                "Couldn't fetch data from Yahoo Finance", extra={'error': e})

        if df_ticker is None:
            logger.exception("Couldn't fetch data from Yahoo Finance", extra={
                             'company': ticker})
        else:
            dfs[ticker] = df_ticker

    return dfs
