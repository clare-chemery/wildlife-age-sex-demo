from pathlib import Path
import argparse


def get_io_args(working_dir: str):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-path",
        type=str,
        default=working_dir / TRAIN_DATA_PATH,
        help="Path to the train data",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        default=working_dir / TEST_DATA_PATH,
        help="Path to the test data",
    )
    parser.add_argument(
        "--model-path", type=str, default=working_dir / MODEL_DIRECTORY, help="Path to the model"
    )
    parser.add_argument(
        "--eval-results-path",
        type=str,
        default=working_dir / EVAL_RESULTS_DIRECTORY,
        help="Path to the eval results",
    )
    args = parser.parse_args()

    return args


def new_func(working_dir, parser):
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default=working_dir / TEST__RAW_DATA_PATH,
        help="Path to the raw data",
    )


# Bounding box data paths
TEST__RAW_DATA_PATH = Path("test_data" / "raw" / "labeled_bbox_data.parquet")
RAW_DATA_PATH = "..."

# Output data paths
TEST_DATA_PATH = Path("data" / "test.parquet")
TRAIN_DATA_PATH = Path("data" / "train.parquet")

# Output modelpaths
EVAL_RESULTS_DIRECTORY = Path("results" / "evals")
MODEL_DIRECTORY = Path("results" / "models")
