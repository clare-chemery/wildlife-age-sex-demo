import argparse
import logging
from datetime import datetime
from typing import Literal


def get_session_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--working-dir",
        type=str,
        help="Path to the working directory containing data and results directories.",
    )
    parser.add_argument(
        "--log-level",
        type=Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument(
        "--use-log-file",
        type=bool,
        default=True,
        help="""
        Whether to use a log file. If True, a log file will be created in the working directory.
        If False, logs will be printed to the console.
        """,
    )
    args = parser.parse_args()

    return {
        "level": getattr(logging, args.log_level.upper()),
        "filename": args.working_dir
        / "logs"
        / f"wildlifeml_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.log"
        if args.use_log_file
        else None,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
        "working_dir": args.working_dir,
    }
