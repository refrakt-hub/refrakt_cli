import logging
import os
from typing import Optional

import vertexai


def initialize_vertex_ai(logger: Optional[logging.Logger] = None) -> None:
    """
    Initialize Vertex AI with project and location from environment variables.

    Args:
        logger: Optional logger

    Returns:
        None
    """
    try:
        PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "refrakt-xai")
        LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        if logger:
            logger.info(
                f"Initialized Vertex AI with project: {PROJECT_ID}, "
                f"location: {LOCATION}"
            )
    except Exception as e:
        if logger:
            logger.error(f"Failed to initialize Vertex AI: {e}")
        raise
