import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from vertexai.generative_models import Part

from refrakt_cli.helpers.llm_helpers import (
    find_latest_experiment_dir,
    generate_and_save_explanations,
    setup_llm_environment,
)
from refrakt_cli.helpers.metadata_helpers import (
    extract_dataset_metadata,
    extract_model_metadata,
)
from refrakt_cli.utils.explanation_utils import (
    extract_file_context,
    extract_metadata_context,
)
from refrakt_cli.utils.gemini_utils import add_images_to_content, build_system_prompt


def _retry_logic(
    attempt: int,
    max_retries: int,
    base_delay: float,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Handle retry logic for API calls."""
    if logger:
        logger.warning(
            f"Retrying after {base_delay} seconds "
            f"(attempt {attempt + 1}/{max_retries + 1})"
        )
    time.sleep(base_delay * (2**attempt) + random.uniform(0, 1))


def _handle_api_error(e: Exception, logger: Optional[logging.Logger] = None) -> str:
    """Handle API errors and return appropriate error messages."""
    error_msg = str(e).lower()
    if (
        "429" in error_msg
        or "resource exhausted" in error_msg
        or "quota exceeded" in error_msg
    ):
        return "RATE_LIMIT_ERROR"
    return "GENERIC_ERROR"


def run_llm_explanations(logger: Optional[logging.Logger] = None) -> None:
    """
    Generate LLM explanations for each XAI component separately, then combine
    them into a comprehensive explanation. This function handles multiple XAI
    methods by organizing files by method and generating explanations for each.
    """
    explanations_dir = setup_llm_environment(logger)
    if not explanations_dir:
        return

    exp_dir = find_latest_experiment_dir(explanations_dir, logger)
    if not exp_dir:
        return

    if logger:
        logger.info(f"Processing most recent experiment directory: {exp_dir}")

    try:
        generate_and_save_explanations(exp_dir, logger)
    except Exception as e:
        if logger:
            logger.error(f"Failed to process experiment directory {exp_dir}: {e}")


def build_gemini_content(
    metadata: Dict[str, Any],
    npy_files: List[str],
    png_files: List[str],
    config_files: List[str],
    logger: Optional[logging.Logger] = None,
) -> List[Part]:
    """
    Build structured content for Gemini API using the multimodal approach.
    """
    content_parts = []

    # Add system prompt as text
    system_prompt = build_system_prompt(logger)

    # Build structured context
    context = build_structured_context(metadata, npy_files, png_files, config_files)

    # Combine system prompt and context
    full_prompt = f"{system_prompt}\n\n{context}"
    content_parts.append(Part.from_text(full_prompt))

    # Add images as multimodal content
    add_images_to_content(png_files, content_parts, logger)

    return content_parts


def build_structured_context(
    metadata: Dict[str, Any],
    npy_files: List[str],
    png_files: List[str],
    config_files: List[str],
) -> str:
    """
    Build a structured context string from metadata and file lists.
    """
    # Extract metadata context
    metadata_context = extract_metadata_context(metadata)

    # Extract file context
    file_context = extract_file_context(npy_files, config_files)

    # Combine metadata and file context
    return f"{metadata_context}\n\n{file_context}"


def extract_experiment_metadata(
    exp_dir: str, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Extract metadata from experiment directory."""
    metadata = {
        "experiment_dir": exp_dir,
        "model_name": (
            os.path.basename(exp_dir).split("_")[0]
            if "_" in os.path.basename(exp_dir)
            else "unknown"
        ),
        "experiment_id": (
            os.path.basename(exp_dir).split("_", 1)[1]
            if "_" in os.path.basename(exp_dir)
            else "unknown"
        ),
        "has_train": os.path.exists(os.path.join(exp_dir, "train")),
        "has_inference": os.path.exists(os.path.join(exp_dir, "inference")),
    }

    if logger:
        exp_name = f"{metadata['model_name']}_{metadata['experiment_id']}"
        logger.info(f"Extracting metadata for experiment: {exp_name}")

    # Extract model metadata
    model_metadata = extract_model_metadata(exp_dir, logger)
    metadata.update(model_metadata)

    # Extract dataset metadata
    dataset_metadata = extract_dataset_metadata(exp_dir, logger)
    metadata.update(dataset_metadata)

    return metadata


def generate_method_explanation_simple(
    exp_dir: str,
    method_name: str,
    metadata: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> str:
    """Generate a simple explanation for a method based on available files."""
    train_dir = os.path.join(exp_dir, "train", method_name)
    inference_dir = os.path.join(exp_dir, "inference", method_name)

    explanation_parts = []
    explanation_parts.append(f"## {method_name.upper()} Analysis")

    if os.path.exists(train_dir):
        train_files = os.listdir(train_dir)
        explanation_parts.append(f"Training files found: {len(train_files)}")

    if os.path.exists(inference_dir):
        inference_files = os.listdir(inference_dir)
        explanation_parts.append(f"Inference files found: {len(inference_files)}")

    explanation_parts.append(f"Model: {metadata.get('model_name', 'unknown')}")
    explanation_parts.append(
        f"Experiment ID: {metadata.get('experiment_id', 'unknown')}"
    )

    return "\n".join(explanation_parts)
