import os
import glob
import base64
import json
import yaml
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

import vertexai
from vertexai.generative_models import GenerativeModel, Part
from refrakt_core.error_handling import XAINotSupportedError
from refrakt_cli.helpers.retry_helpers import get_retry_parameters, handle_rate_limit_error
from refrakt_cli.helpers.vertex_ai_helpers import initialize_vertex_ai
from refrakt_cli.helpers.metadata_helpers import extract_experiment_metadata_helper, determine_train_inference, initialize_metadata_structure, extract_performance_metrics, filter_na_metrics, load_config_files, merge_training_results, extract_experiment_id, determine_model_type, extract_model_metadata, extract_dataset_metadata, extract_metadata_from_config, extract_run_metadata
from refrakt_cli.helpers.log_helpers import sort_log_files_by_time, extract_key_metrics_from_logs, filter_recent_logs
from refrakt_cli.helpers.xai_helpers import process_xai_files, save_explanation_to_markdown, save_comprehensive_report
from refrakt_cli.helpers.gemini_helpers import build_system_prompt, add_images_to_content
from refrakt_cli.helpers.context_helpers import extract_metadata_context, extract_file_context
from refrakt_cli.helpers.core_helpers import generate_method_explanation, combine_method_explanations, organize_xai_files_by_method
from refrakt_cli.helpers.llm_helpers import find_latest_experiment_dir, generate_and_save_explanations, setup_llm_environment
from refrakt_cli.helpers.shared_core import extract_comprehensive_metadata

def _retry_logic(attempt: int, max_retries: int, base_delay: float, logger=None):
    """Handle retry logic for API calls."""
    if logger:
        logger.warning(f"Retrying after {base_delay} seconds (attempt {attempt + 1}/{max_retries + 1})")
    time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))

def _handle_api_error(e: Exception, logger=None) -> str:
    """Handle API errors and return appropriate error messages."""
    error_msg = str(e).lower()
    if "429" in error_msg or "resource exhausted" in error_msg or "quota exceeded" in error_msg:
        return "RATE_LIMIT_ERROR"
    return "GENERIC_ERROR"

def call_gemini_with_retry(model: GenerativeModel, content_parts: Any, max_retries: int = 3, logger=None) -> str:
    """
    Call Gemini with exponential backoff retry logic for rate limiting.

    Args:
        model: GenerativeModel instance
        content_parts: Content parts to send to the model
        max_retries: Maximum number of retry attempts
        logger: Optional logger

    Returns:
        Response text from the model

    Raises:
        Exception: If all retries are exhausted
    """
    max_retries, base_delay = get_retry_parameters(max_retries)

    for attempt in range(max_retries + 1):
        try:
            if logger:
                logger.info(f"Calling Gemini API (attempt {attempt + 1}/{max_retries + 1})")
            response = model.generate_content(content_parts)
            if logger:
                logger.info(f"Successfully received response from Gemini API")
            return response.text
        except Exception as e:
            error_type = _handle_api_error(e, logger)
            if error_type == "RATE_LIMIT_ERROR" and attempt < max_retries:
                _retry_logic(attempt, max_retries, base_delay, logger)
                continue
            elif error_type == "RATE_LIMIT_ERROR":
                if logger:
                    logger.error(f"Rate limit exceeded after {max_retries + 1} attempts. Skipping LLM explanation.")
                return f"[RATE_LIMIT_ERROR] Could not generate explanation due to API rate limits: {e}"
            else:
                if logger:
                    logger.error(f"Error calling Gemini: {e}")
                return f"[ERROR] Failed to generate explanation: {e}"

    return "[ERROR] Unexpected error in retry logic"

def run_llm_explanations(logger=None):
    """
    Generate LLM explanations for each XAI component separately, then combine them into a comprehensive explanation.
    This function handles multiple XAI methods by organizing files by method and generating explanations for each.
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

def extract_metrics_from_logs(log_files: List[str]) -> Dict[str, Any]:
    """
    Extract performance metrics from log files.
    Prioritizes the most recent log files and filters by relevance.
    """
    metrics = {
        "best_accuracy": "N/A",
        "final_loss": "N/A",
        "epochs_completed": "N/A",
        "training_time": "N/A"
    };
    
    # Sort log files by modification time (most recent first)
    log_files_with_time = sort_log_files_by_time(log_files)
    
    latest_log_time = None
    latest_metrics = None
    latest_loss = None
    epochs_completed = 0
    
    # Process log files in order of recency
    for log_file, log_time in log_files_with_time:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Skip log files that are too old (more than 24 hours)
            current_time = time.time()
            # Ensure log_time is converted to a float for comparison
            if current_time - float(log_time) > 86400:  # 24 hours in seconds
                continue
            
            # Extract key metrics from log lines
            metrics = extract_key_metrics_from_logs(lines, metrics)
            
            # If we found the key metrics, we can stop
            if metrics["best_accuracy"] != "N/A" and latest_loss is not None:
                break
        except Exception:
            continue
    
    if latest_loss is not None:
        metrics['final_loss'] = str(latest_loss)
    if epochs_completed > 0:
        metrics['epochs_completed'] = str(epochs_completed)
    
    return metrics


def build_gemini_content(metadata: Dict[str, Any], npy_files: List[str], png_files: List[str], config_files: List[str], logger=None) -> List[Part]:
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


def build_structured_context(metadata: Dict[str, Any], npy_files: List[str], png_files: List[str], config_files: List[str]) -> str:
    """
    Build a structured context string from metadata and file lists.
    """
    # Extract metadata context
    metadata_context = extract_metadata_context(metadata)

    # Extract file context
    file_context = extract_file_context(npy_files, config_files)

    # Combine metadata and file context
    return f"{metadata_context}\n\n{file_context}"





def extract_experiment_metadata(exp_dir, logger=None):
    """Extract metadata from experiment directory."""
    metadata = {
        "experiment_dir": exp_dir,
        "model_name": os.path.basename(exp_dir).split('_')[0] if '_' in os.path.basename(exp_dir) else "unknown",
        "experiment_id": os.path.basename(exp_dir).split('_', 1)[1] if '_' in os.path.basename(exp_dir) else "unknown",
        "has_train": os.path.exists(os.path.join(exp_dir, 'train')),
        "has_inference": os.path.exists(os.path.join(exp_dir, 'inference')),
    }

    if logger:
        logger.info(f"Extracting metadata for experiment: {metadata['model_name']}_{metadata['experiment_id']}")

    # Extract model metadata
    model_metadata = extract_model_metadata(exp_dir, logger)
    metadata.update(model_metadata)

    # Extract dataset metadata
    dataset_metadata = extract_dataset_metadata(exp_dir, logger)
    metadata.update(dataset_metadata)

    return metadata


def generate_method_explanation_simple(exp_dir, method_name, metadata, logger=None):
    """Generate a simple explanation for a method based on available files."""
    train_dir = os.path.join(exp_dir, 'train', method_name)
    inference_dir = os.path.join(exp_dir, 'inference', method_name)
    
    explanation_parts = []
    explanation_parts.append(f"## {method_name.upper()} Analysis")
    
    if os.path.exists(train_dir):
        train_files = os.listdir(train_dir)
        explanation_parts.append(f"Training files found: {len(train_files)}")
    
    if os.path.exists(inference_dir):
        inference_files = os.listdir(inference_dir)
        explanation_parts.append(f"Inference files found: {len(inference_files)}")
    
    explanation_parts.append(f"Model: {metadata.get('model_name', 'unknown')}")
    explanation_parts.append(f"Experiment ID: {metadata.get('experiment_id', 'unknown')}")
    
    return "\n".join(explanation_parts)