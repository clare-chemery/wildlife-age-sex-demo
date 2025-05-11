import logging
import argparse
from typing import Optional
import tomli
from wildlifeml.io import load, save
from wildlifeml.preprocess import preprocess_data, split_data
from pathlib import Path


def main(
    working_dir: str,
    raw_data_filepath: str,
    train_data_filepath: str,
    test_data_filepath: str,
    preprocess_kwargs: dict,
    **kwargs,
):
    # Load data
    data = load(filepath=Path(working_dir) / Path(raw_data_filepath))

    # Preprocess data
    preprocessed_data = preprocess_data(data, **preprocess_kwargs)

    # Split data, save to disk
    for train, test in split_data(
        preprocessed_data, stratify_by=preprocess_kwargs.get("stratify_by", None)
    ):
        save(train, filepath=Path(working_dir) / Path(train_data_filepath))
        save(test, filepath=Path(working_dir) / Path(test_data_filepath))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/test__config.toml")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        args = tomli.load(f)
    logging.basicConfig(**args.get("logging", {}))

    main(
        args["globals"]["working_dir"],
        **args["io"]["data"],
        **args.get("preprocess", {}),
    )
