import logging
import math
import multiprocessing
import pathlib
import sys
import time

import numpy as np
from numpy import float64, ndarray

NUM_SIMULATIONS_PER_PORTFOLIO = 1000
MAX_WEIGHT_PER_ASSET = 0.2
ASSETS_PER_PORTFOLIO = 25
NUM_PORTFOLIOS = 142_000
LOGGER_NAME = 'generate_weights'


def generate_weights(number_of_assets: int, max_weight: float, num_samples: int) -> ndarray:
    if 1 / number_of_assets > max_weight:
        raise ValueError('Impossible Condition')

    weights_matrix = np.empty((num_samples, number_of_assets), dtype=float64)
    portfolio_size = np.ones(number_of_assets)
    BATCH_SIZE_MULTIPLIER = 1.2

    i = 0

    while i < num_samples:
        spots_left = num_samples - i
        batch_size = int(BATCH_SIZE_MULTIPLIER * spots_left)
        batch = np.random.dirichlet(alpha=portfolio_size, size=batch_size)
        valid_weights = batch[np.all(batch <= max_weight, axis=1)]
        spots_to_be_filled = min(len(valid_weights), num_samples - i)

        if spots_to_be_filled > 0:
            weights_matrix[i:i +
                           spots_to_be_filled] = valid_weights[:spots_to_be_filled]
            i += spots_to_be_filled

    return weights_matrix


def aux(num_portfolios, assets_per_portfolio, max_weight_per_asset, num_simulations_per_portfolio):
    for i in range(num_portfolios):
        _ = generate_weights(
            assets_per_portfolio, max_weight_per_asset, num_simulations_per_portfolio)


def run_multiprocessing():
    n_processes = multiprocessing.cpu_count()
    chunk_size = math.ceil(NUM_PORTFOLIOS / n_processes)

    with multiprocessing.Pool() as pool:
        pool.starmap(
            aux,
            [(chunk_size, ASSETS_PER_PORTFOLIO, MAX_WEIGHT_PER_ASSET, NUM_SIMULATIONS_PER_PORTFOLIO)
             for _ in range(n_processes)]
        )


def run_single_process():
    for i in range(NUM_PORTFOLIOS):
        _ = generate_weights(
            ASSETS_PER_PORTFOLIO, MAX_WEIGHT_PER_ASSET, NUM_SIMULATIONS_PER_PORTFOLIO)


if __name__ == "__main__":
    script_dir = pathlib.Path(sys.argv[0]).parent.resolve()
    log_path = script_dir / f'{LOGGER_NAME}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_path
    )
    logger = logging.getLogger(LOGGER_NAME)

    logger.info("Starting weight generation with multiprocessing")
    start_time = time.time()
    run_multiprocessing()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Multiprocessing completed in {elapsed_time:.2f} seconds")

    logger.info("Starting weight generation with single process...")
    start_time = time.time()
    run_single_process()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Single process completed in {elapsed_time:.2f} seconds")
