"""
CLI argument parser for Refrakt.
"""

import argparse
from typing import List, Tuple


def setup_argument_parser() -> Tuple[argparse.Namespace, List[str]]:
    """
    Setup argument parser for CLI with all required arguments.

    This function creates and configures the argument parser with all
    necessary arguments for the Refrakt CLI, including configuration
    file path, logging options, and override capabilities.

    Returns:
        Tuple of parsed arguments and unknown arguments from parse_known_args()
    """
    parser = argparse.ArgumentParser(
        description="Refrakt CLI - ML/DL Training Framework"
    )

    # Required arguments
    parser.add_argument("--config", required=True, help="Path to configuration file")

    # Optional arguments
    parser.add_argument(
        "--mode",
        choices=["train", "test", "inference", "pipeline"],
        default="train",
        help="Pipeline mode to execute",
    )

    parser.add_argument(
        "--model-path", help="Path to model checkpoint (for test/inference modes)"
    )

    parser.add_argument("--log-dir", help="Override log directory path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--override",
        nargs="+",
        help="Specify multiple override values (format: path.to.param=value).",
    )

    return parser.parse_known_args()
