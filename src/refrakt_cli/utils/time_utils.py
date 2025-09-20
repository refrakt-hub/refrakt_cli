"""
Time-related utility functions.
"""

from datetime import datetime


def get_experiment_id() -> str:
    """
    Generates a unique experiment ID based on the current timestamp.

    Returns:
        A string representing the timestamp in YYYYMMDD_HHMMSS format.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
