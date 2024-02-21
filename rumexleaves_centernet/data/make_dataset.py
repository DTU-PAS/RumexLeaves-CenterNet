# -*- coding: utf-8 -*-
import glob
import logging
import shutil


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from raw data")

    datasets = glob.glob("data/raw/*.zip")

    for dataset in datasets:
        shutil.unpack_archive(dataset, "data/processed", "zip")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
