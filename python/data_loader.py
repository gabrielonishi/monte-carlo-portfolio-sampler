"""Fetches data from Yahoo Finance."""

import logging
import pickle
import sys
from datetime import date, timedelta

import pathlib
import pandas as pd
import numpy as np
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


def load_data_from_yf(ticker: str, simulation_start: date, simulation_end: date) -> pd.DataFrame:
    """Get data from YahooFinance using tickers and simulation period."""

    start = simulation_start.isoformat()
    end = simulation_end.isoformat()

    df_ticker = None

    try:
        df_ticker = yf.download(
            ticker, start=start, end=end, progress=False
        )
    except TimeoutError as e:
        logger.exception(
            "Couldn't fetch data from Yahoo Finance", extra={'error': e})

    if df_ticker is None or not isinstance(df_ticker, pd.DataFrame):
        logger.exception("Invalid data fetched from Yahoo Finance", extra={
            'company': ticker})
        return pd.DataFrame()

    if 'Close' not in df_ticker.columns:
        logger.exception("Missing 'Close' column in fetched data", extra={
            'company': ticker})
        return pd.DataFrame()

    return df_ticker[['Close']]


def load_history_by_ticker(tickers: list[str], simulation_start: date, simulation_end: date) -> dict[str, pd.DataFrame]:
    price_history_by_ticker = {}

    for ticker in tickers:
        price_history_by_ticker[ticker] = load_data_from_yf(
            ticker, simulation_start, simulation_end)

    return price_history_by_ticker


def calculate_daily_returns(price_history_by_ticker: dict[str, pd.DataFrame]) -> dict:
    daily_returns_by_ticker = {}
    for ticker in price_history_by_ticker:
        daily_returns_by_ticker[ticker] = price_history_by_ticker[ticker] / \
            price_history_by_ticker[ticker].shift(1) - 1

    return daily_returns_by_ticker


def run(simulation_start: date, simulation_end: date, risk_free_yf_ticker: str, stage: str = 'PROD') -> tuple[dict[str, pd.DataFrame], float]:
    if stage == 'PROD':
        tickers = get_djia_tickers()
        price_history_by_ticker = load_history_by_ticker(
            tickers, simulation_start, simulation_end)
    elif stage == 'DEV':
        NUM_COMPANIES = 30

        script_dir = pathlib.Path(sys.argv[0]).parent.resolve()
        dev_data_path = script_dir / 'financial_history_data_dev.pkl'

        with open(file=dev_data_path, mode='rb') as f:
            price_history_by_ticker = pickle.load(file=f)
            tickers = list(price_history_by_ticker.keys())[:NUM_COMPANIES]
            price_history_by_ticker = {
                ticker: price_history_by_ticker[ticker]['Close', ticker] for ticker in tickers}
    else:
        raise ValueError(f'{stage} is not a valid stage. Should be PROD, DEV')

    daily_returns_by_ticker = calculate_daily_returns(price_history_by_ticker)

    risk_free_df = load_data_from_yf(
        risk_free_yf_ticker, simulation_start, simulation_start + timedelta(days=1))

    DECIMAL_PLACES = 2

    yearly_risk_free_rate = risk_free_df.iloc[0].iloc[0] / 10**DECIMAL_PLACES

    return daily_returns_by_ticker, yearly_risk_free_rate
