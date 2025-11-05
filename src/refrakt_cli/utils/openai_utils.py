import base64
import logging
import os
from typing import Any, Dict, List, Optional


def build_system_prompt(logger: Optional[logging.Logger] = None) -> str:
    """
    Build the system prompt for OpenAI API by loading from explainer_prompt.txt.
    """
    if logger:
        logger.info("Building system prompt for OpenAI API.")

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


def add_images_to_messages(
    png_files: List[str],
    messages: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Add image files to OpenAI messages using vision API format.
    Images are encoded as base64 and added to the last user message.

    Args:
        png_files: List of paths to PNG image files
        messages: List of messages in OpenAI format
        logger: Optional logger instance
    """
    if not png_files:
        return

    # Find or create the last user message
    user_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg
            break

    if user_message is None:
        # Create a new user message if none exists
        user_message = {"role": "user", "content": []}
        messages.append(user_message)

    # Convert content to list format if it's a string
    if isinstance(user_message["content"], str):
        user_message["content"] = [{"type": "text", "text": user_message["content"]}]
    elif not isinstance(user_message["content"], list):
        user_message["content"] = []

    # Add images to the message
    for png_file in png_files:
        if logger:
            logger.info(f"Adding image {png_file} to OpenAI messages")
        try:
            # Read and encode image as base64
            with open(png_file, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_url = f"data:image/png;base64,{image_data}"
                user_message["content"].append(
                    {"type": "image_url", "image_url": {"url": image_url}}
                )
        except Exception as e:
            if logger:
                logger.warning(f"Failed to add image {png_file}: {e}")
