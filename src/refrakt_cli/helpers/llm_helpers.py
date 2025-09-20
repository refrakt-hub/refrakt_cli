import logging
import os
from typing import List, Optional, Tuple


def setup_llm_environment(logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Initializes Vertex AI and validates the explanations directory.

    Args:
        logger: An optional logger instance.

    Returns:
        The path to the explanations directory if setup is successful, otherwise None.
    """
    from refrakt_cli.utils.vertex_ai_utils import initialize_vertex_ai

    try:
        if logger:
            logger.debug("Initializing Vertex AI...")
        initialize_vertex_ai(logger)
    except Exception as e:
        if logger:
            logger.error(f"Failed to initialize Vertex AI: {e}")
        return None

    base_dir = os.getcwd()
    explanations_dir = os.path.join(base_dir, "explanations")

    if not os.path.exists(explanations_dir):
        if logger:
            logger.warning(
                "No explanations directory found. Skipping LLM explanations."
            )
        return None

    if logger:
        logger.debug(f"Base directory: {base_dir}")
        logger.debug(f"Explanations directory: {explanations_dir}")

    return explanations_dir


def _get_experiment_directories(
    explanations_dir: str, logger: Optional[logging.Logger] = None
) -> List[Tuple[str, float]]:
    """
    Find and return experiment directories with their modification times.

    Args:
        explanations_dir: Path to the explanations directory
        logger: Optional logger

    Returns:
        List of tuples containing directory paths and their modification times
    """
    experiment_dirs = []
    for item in os.listdir(explanations_dir):
        item_path = os.path.join(explanations_dir, item)
        if os.path.isdir(item_path):
            train_subdir = os.path.join(item_path, "train")
            inference_subdir = os.path.join(item_path, "inference")
            if os.path.exists(train_subdir) or os.path.exists(inference_subdir):
                try:
                    mtime = os.path.getmtime(item_path)
                    experiment_dirs.append((item_path, mtime))
                except Exception:
                    experiment_dirs.append((item_path, 0))
    return experiment_dirs


def find_latest_experiment_dir(
    explanations_dir: str, logger: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    Finds and returns the path to the most recent experiment directory.

    Args:
        explanations_dir: The directory where experiment explanations are stored.
        logger: An optional logger instance.

    Returns:
        The path to the latest experiment directory, or None if not found.
    """
    experiment_dirs = _get_experiment_directories(explanations_dir, logger)

    if not experiment_dirs:
        if logger:
            logger.warning(
                "No experiment directories with explanations found in explanations/. "
                "Skipping LLM explanations."
            )
        return None

    experiment_dirs.sort(key=lambda x: x[1], reverse=True)
    exp_dir, _ = experiment_dirs[0]
    return exp_dir


def generate_and_save_explanations(
    exp_dir: str, logger: Optional[logging.Logger] = None
) -> None:
    """
    Generates and saves explanations for a given experiment directory.

    Args:
        exp_dir: The path to the experiment directory.
        logger: An optional logger instance.
    """
    import json

    from refrakt_cli.helpers.xai_helpers import (
        process_xai_files,
        save_comprehensive_report,
        save_explanation_to_markdown,
    )
    from refrakt_cli.utils.explanation_utils import combine_method_explanations
    from refrakt_cli.utils.gemini_utils import build_system_prompt
    from refrakt_cli.utils.llm_utils import (
        generate_method_explanation,
        organize_xai_files,
    )

    # Find and organize XAI-related files
    npy_files, png_files, config_files = process_xai_files(exp_dir, logger)

    # Organize files by XAI method
    xai_files_by_method = organize_xai_files(npy_files + png_files, logger=logger)

    # Debug output
    if logger:
        logger.info(
            f"DEBUG: Found {len(npy_files)} npy files and {len(png_files)} png files"
        )
        logger.info(f"DEBUG: Organized into {len(xai_files_by_method)} methods:")
        for method_name, method_data in xai_files_by_method.items():
            logger.info(
                f"DEBUG:   {method_name}: {len(method_data.get('files', []))} files"
            )

    # Read metadata from summary_metrics.json
    exp_name = os.path.basename(exp_dir)
    checkpoints_dir = os.path.join(
        os.path.dirname(os.path.dirname(exp_dir)), "checkpoints"
    )
    summary_metrics_path = os.path.join(
        checkpoints_dir, exp_name, "explanations", "summary_metrics.json"
    )

    metadata = {}
    if os.path.exists(summary_metrics_path):
        try:
            with open(summary_metrics_path, "r") as f:
                metadata = json.load(f)
            if logger:
                logger.info(f"Loaded metadata from {summary_metrics_path}")
        except Exception as e:
            if logger:
                logger.warning(
                    f"Failed to load metadata from {summary_metrics_path}: {e}"
                )
            # TODO: Move extract_comprehensive_metadata to utils during Phase 4
            from refrakt_cli.helpers.shared_core import extract_comprehensive_metadata

            base_dir = os.path.dirname(exp_dir)
            metadata = extract_comprehensive_metadata(
                config_files, base_dir, checkpoints_dir, logger=logger
            )
    else:
        if logger:
            logger.warning(
                f"summary_metrics.json not found at {summary_metrics_path}, "
                "falling back to config file extraction"
            )
        # TODO: Move extract_comprehensive_metadata to utils during Phase 4
        from refrakt_cli.helpers.shared_core import extract_comprehensive_metadata

        base_dir = os.path.dirname(exp_dir)
        metadata = extract_comprehensive_metadata(
            config_files, base_dir, checkpoints_dir, logger=logger
        )

    # System prompt for LLM
    system_prompt = build_system_prompt(logger)
    model_name = os.environ.get("GEMINI_MODEL", "")

    method_explanations = {}
    for method_name, method_files in xai_files_by_method.items():
        explanation = generate_method_explanation(
            method_name,
            method_files,
            metadata,
            config_files,
            system_prompt,
            model_name,
            logger=logger,
        )
        method_explanations[method_name] = explanation
        save_explanation_to_markdown(method_name, explanation, exp_dir, logger)

    # Write comprehensive report
    combined_explanation = combine_method_explanations(
        method_explanations, metadata, logger=logger
    )
    save_comprehensive_report(exp_dir, combined_explanation, logger)
