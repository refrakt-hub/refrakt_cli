import logging
import os
from typing import List, Optional

from vertexai.generative_models import Part  # Corrected import for Part


def build_system_prompt(logger: Optional[logging.Logger] = None) -> str:
    """
    Build the system prompt for Gemini API by loading from explainer_prompt.txt.
    """
    if logger:
        logger.info("Building system prompt for Gemini API.")

    # Try to load the detailed system prompt from the file
    # Look for the prompt file in multiple possible locations
    possible_paths = [
        os.path.join(
            os.path.dirname(__file__), "..", "EXPLAINER_PROMPT.txt"
        ),  # relative to helpers
        os.path.join(
            os.path.dirname(__file__), "..", "..", "EXPLAINER_PROMPT.txt"
        ),  # relative to refrakt_cli
        os.path.join(
            os.getcwd(),
            "external",
            "refrakt_cli",
            "src",
            "refrakt_cli",
            "EXPLAINER_PROMPT.txt",
        ),  # from workspace root
        os.path.join(os.getcwd(), "EXPLAINER_PROMPT.txt"),  # in current directory
        # Also try lowercase for backward compatibility
        os.path.join(os.path.dirname(__file__), "..", "explainer_prompt.txt"),
        os.path.join(os.path.dirname(__file__), "..", "..", "explainer_prompt.txt"),
        os.path.join(
            os.getcwd(),
            "external",
            "refrakt_cli",
            "src",
            "refrakt_cli",
            "explainer_prompt.txt",
        ),
        os.path.join(os.getcwd(), "explainer_prompt.txt"),
    ]

    prompt_file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            prompt_file_path = path
            break
    if prompt_file_path is None:
        if logger:
            logger.warning(
                "System prompt file not found in any of the expected "
                "locations, using default prompt"
            )
        default_prompt = (
            "You are an expert AI explainer for deep learning models "
            "specializing in Explainable AI (XAI) analysis. Your task is to "
            "generate comprehensive, accurate, and insightful natural language "
            "explanations for model behavior using the provided XAI "
            "visualizations, attribution data, and model metadata."
        )
        return default_prompt

    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        if logger:
            logger.info(f"Loaded system prompt from {prompt_file_path}")
        return system_prompt
    except Exception as e:
        if logger:
            logger.error(f"Error loading system prompt from {prompt_file_path}: {e}")
        default_prompt = (
            "You are an expert AI explainer for deep learning models "
            "specializing in Explainable AI (XAI) analysis. Your task is to "
            "generate comprehensive, accurate, and insightful natural language "
            "explanations for model behavior using the provided XAI "
            "visualizations, attribution data, and model metadata."
        )
        return default_prompt


def add_images_to_content(
    png_files: List[str],
    content_parts: List[Part],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Add image files as multimodal content parts.
    """
    for png_file in png_files:
        if logger:
            logger.info(f"Adding image {png_file} to content parts.")
        try:
            # For now, we'll skip image addition to avoid type issues
            # TODO: Implement proper image handling with Vertex AI
            if logger:
                logger.info(
                    f"Skipping image {png_file} - image handling not yet implemented"
                )
        except Exception as e:
            if logger:
                logger.warning(f"Failed to add image {png_file}: {e}")
