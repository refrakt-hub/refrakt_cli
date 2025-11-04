import logging
import os
import random
import time
from typing import Optional, Tuple


def get_retry_parameters(max_retries: int) -> Tuple[int, float]:
    """
    Retrieve retry parameters from environment variables.

    Args:
        max_retries: Default maximum retries

    Returns:
        Tuple of maximum retries and base delay
    """
    max_retries = int(os.environ.get("LLM_MAX_RETRIES", max_retries))
    base_delay = float(os.environ.get("LLM_BASE_DELAY", "3.0"))
    return max_retries, base_delay


def handle_rate_limit_error(
    attempt: int,
    max_retries: int,
    base_delay: float,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Handle rate limit errors with exponential backoff and jitter.

    Args:
        attempt: Current retry attempt
        max_retries: Maximum retry attempts
        base_delay: Base delay for exponential backoff
        logger: Optional logger

    Returns:
        None
    """
    wait_time = (base_delay * (2**attempt)) + random.uniform(0, 2)
    msg = (
        f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
        f"Waiting {wait_time:.2f}s before retry..."
    )
    if logger:
        logger.warning(msg)
    else:
        print(msg)
    time.sleep(wait_time)
