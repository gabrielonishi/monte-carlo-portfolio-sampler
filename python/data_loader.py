"""Fetches data from Yahoo Finance."""

import logging

import pandas as pd
import requests
import yfinance as yf

DOW_JONES_WIKI_URL = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
SIMULATION_START_DATE = "2024-08-01"
SIMULATION_END_DATE = "2024-12-31"


def get_djia_components(logger: logging.Logger) -> list[str]:
    """Get Dow Jones components from its Wikipedia article."""
    try:
        response = requests.get(DOW_JONES_WIKI_URL, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.exception("Couldn't fetch data from Wikipedia", extra={"exception": str(e)})
        return []

    tables = pd.read_html(response.content)

    for table in tables:
        if "Company" in table and "Exchange" in table and "Symbol" in table:
            return list(table["Symbol"])

    logger.error("Could not identify companies from Wikipedia tables")

    return []


def load_data_from_yf(tickers: list[str], logger: logging.Logger) -> dict[str : pd.DataFrame | None]:
    """Get data from YahooFinance using tickers and simulation period."""
    dfs = {}

    for ticker in tickers:
        df_ticker = yf.download(ticker, start=SIMULATION_START_DATE, end=SIMULATION_END_DATE)
        if df_ticker is None:
            logger.exception("Couldn't fetch data from Yahoo Finance", extra={"company": ticker})
        dfs[ticker] = df_ticker

    return dfs


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    tickers = get_djia_components(logger)
    data = load_data_from_yf(tickers, logger)
    ...
