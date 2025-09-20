import glob
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from refrakt_cli.utils.explanation_utils import combine_method_explanations
from refrakt_cli.utils.llm_utils import organize_xai_files


def filter_files_by_extension(
    file_paths: List[str], extensions: List[str]
) -> List[str]:
    """
    Filter files by their extensions.

    Args:
        file_paths: List of file paths
        extensions: List of allowed extensions (e.g., ['.npy', '.png'])

    Returns:
        List of file paths with the specified extensions
    """
    return [
        file_path
        for file_path in file_paths
        if any(file_path.endswith(ext) for ext in extensions)
    ]


def organize_files_by_xai_method(
    npy_files: List[str], png_files: List[str], logger: Optional[logging.Logger] = None
) -> Dict[str, List[str]]:
    """Organize files by XAI method."""
    organized_files = organize_xai_files(npy_files, png_files, logger=logger)
    flattened_files = {
        method: file_list
        for method, file_dict in organized_files.items()
        for file_list in file_dict.values()
    }
    return flattened_files


def generate_and_save_explanations(
    xai_files_by_method: Dict[str, List[str]],
    exp_dir: str,
    metadata: Dict[str, Any],
    config_files: List[str],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """Generate explanations for each XAI method and save them as markdown files."""
    from refrakt_cli.utils.gemini_utils import build_system_prompt
    from refrakt_cli.utils.llm_utils import generate_method_explanation

    system_prompt = build_system_prompt(logger)
    model_name = os.environ.get("GEMINI_MODEL", "")

    method_explanations = {}
    for method_name, method_files in xai_files_by_method.items():
        if logger:
            logger.info(
                f"Generating explanation for XAI method: {method_name} "
                f"with {len(method_files)} files"
            )

        explanation = generate_method_explanation(
            method_name,
            {"files": method_files},
            metadata,
            config_files,
            system_prompt,
            model_name,
            logger=logger,
        )
        method_explanations[method_name] = explanation

        # Write individual method explanation to markdown file
        save_explanation_to_markdown(method_name, explanation, exp_dir, logger)

    return method_explanations


def write_comprehensive_report(
    exp_dir: str,
    method_explanations: Dict[str, str],
    metadata: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Write a comprehensive report."""
    combined_explanation = combine_method_explanations(
        method_explanations, metadata, logger=logger
    )
    save_comprehensive_report(exp_dir, combined_explanation, logger)


def process_xai_files(
    exp_dir: str, logger: Optional[logging.Logger] = None
) -> Tuple[List[str], List[str], List[str]]:
    """Find and organize XAI-related files in the experiment directory."""
    npy_files = glob.glob(os.path.join(exp_dir, "**", "*.npy"), recursive=True)
    png_files = glob.glob(os.path.join(exp_dir, "**", "*.png"), recursive=True)

    # Look for config files in the checkpoints directory
    exp_name = os.path.basename(exp_dir)
    checkpoints_dir = os.path.join(
        os.path.dirname(os.path.dirname(exp_dir)), "checkpoints"
    )
    checkpoint_exp_dir = os.path.join(checkpoints_dir, exp_name)
    config_files = glob.glob(os.path.join(checkpoint_exp_dir, "*.yaml"))

    if logger:
        logger.info(f"Found config files: {config_files}")

    return npy_files, png_files, config_files


def save_explanation_to_markdown(
    method_name: str,
    explanation: str,
    exp_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Save the explanation for a method to a markdown file in the checkpoints
    directory.
    """
    # Extract experiment info from the exp_dir path
    # exp_dir should be like: explanations/autoencoder_simple_20250807_111150
    exp_name = os.path.basename(exp_dir)

    # Construct the checkpoints path
    checkpoints_dir = os.path.join(
        os.path.dirname(os.path.dirname(exp_dir)), "checkpoints"
    )
    explanations_dir = os.path.join(checkpoints_dir, exp_name, "explanations")

    # Create the explanations directory if it doesn't exist
    os.makedirs(explanations_dir, exist_ok=True)

    # Save the explanation file
    md_path = os.path.join(explanations_dir, f"{method_name}.md")
    try:
        with open(md_path, "w") as f:
            f.write(explanation)
        if logger:
            logger.info(
                f"[XAI] Wrote markdown explanation for {method_name} to {md_path}"
            )
    except Exception as e:
        if logger:
            logger.warning(f"[XAI] Failed to write markdown for {method_name}: {e}")


def save_comprehensive_report(
    exp_dir: str, combined_explanation: str, logger: Optional[logging.Logger] = None
) -> None:
    """Save the comprehensive report to a markdown file in the checkpoints directory."""
    # Extract experiment info from the exp_dir path
    # exp_dir should be like: explanations/autoencoder_simple_20250807_111150
    exp_name = os.path.basename(exp_dir)

    # Construct the checkpoints path
    checkpoints_dir = os.path.join(
        os.path.dirname(os.path.dirname(exp_dir)), "checkpoints"
    )
    explanations_dir = os.path.join(checkpoints_dir, exp_name, "explanations")

    # Create the explanations directory if it doesn't exist
    os.makedirs(explanations_dir, exist_ok=True)

    # Save the comprehensive report
    report_path = os.path.join(explanations_dir, "comprehensive_report.md")
    try:
        with open(report_path, "w") as f:
            f.write(combined_explanation)
        if logger:
            logger.info(f"[XAI] Wrote comprehensive report to {report_path}")
    except Exception as e:
        if logger:
            logger.warning(f"[XAI] Failed to write comprehensive report: {e}")
