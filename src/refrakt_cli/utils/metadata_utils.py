import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from refrakt_cli.helpers.metadata_helpers import extract_performance_metrics


def create_experiment_info(
    experiment_metadata: Dict[str, Any], has_train: bool, has_inference: bool
) -> Dict[str, Any]:
    experiment_id = experiment_metadata.get("experiment_id")
    if isinstance(experiment_id, dict):
        experiment_id = (
            experiment_id.get("experiment_id")
            or experiment_id.get("experiment_dir", "").split("_")[-1]
        )
    if not isinstance(experiment_id, str):
        experiment_id = "unknown_experiment"

    # Create timestamp from experiment_id if it contains datetime info
    timestamp = None
    if isinstance(experiment_id, str) and "_" in experiment_id:
        # experiment_id format: "20250814_195335"
        timestamp = experiment_id.split("_")[0]  # Extract "20250814"

    result = {
        "experiment_id": experiment_id,  # Keep full format: "20250814_195335"
        "has_train": has_train,
        "has_inference": has_inference,
    }

    # Add timestamp if extracted
    if timestamp:
        result["timestamp"] = timestamp

    return result


def _extract_training_results_dict(
    training_results: Optional[Any],
) -> Dict[str, Any]:
    """Extract dict from training_results, handling summary/callable/dict/None."""
    if training_results is None:
        return {}
    if (
        not isinstance(training_results, dict)
        and hasattr(training_results, "summary")
        and callable(training_results.summary)
    ):
        summary = training_results.summary()
        if isinstance(summary, dict):
            return summary
        return {}
    if isinstance(training_results, dict):
        return training_results
    return {}


def _merge_metrics_dicts(
    latest_metrics: Dict[str, Any], training_results_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two metrics dicts, normalizing types as needed."""
    for k, v in training_results_dict.items():
        if isinstance(v, (int, float)) or (
            isinstance(v, str) and v.replace(".", "", 1).isdigit()
        ):
            latest_metrics[k] = str(v)
        elif v not in [None, "", "N/A"] and k not in latest_metrics:
            latest_metrics[k] = v
    return latest_metrics


def merge_performance_metrics(
    base_dir: str,
    training_results: Optional[Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    latest_metrics = extract_performance_metrics(base_dir, training_results, logger)
    training_results_dict = _extract_training_results_dict(training_results)
    return _merge_metrics_dicts(latest_metrics, training_results_dict)


def _collect_explanation_files(explanations_dir: str) -> Tuple[List[str], List[str]]:
    """Collect all npy and png files under explanations_dir."""
    npy_files, png_files = [], []
    for root, _, files in os.walk(explanations_dir):
        for file in files:
            path = os.path.join(root, file)
            rel = os.path.relpath(path, explanations_dir)
            if file.endswith(".npy"):
                npy_files.append(rel)
            elif file.endswith(".png"):
                png_files.append(rel)
    return npy_files, png_files


def _filter_files_by_experiment_id(files: List[str], experiment_id: str) -> List[str]:
    """Filter files to train/inference for experiment_id, return relative paths."""
    # Handle different experiment ID formats found in the file paths
    experiment_patterns = [
        f"{experiment_id}/train/",  # Direct format: 184818/train/
        f"{experiment_id}/inference/",  # Direct format: 184818/inference/
        f"convnext_{experiment_id}/train/",  # Full: convnext_184818/train/
        f"convnext_{experiment_id}/inference/",  # Full: convnext_184818/inference/
    ]

    filtered_files = []
    for file in files:
        for pattern in experiment_patterns:
            if pattern in file:
                # Extract method dir and filename (e.g., "layer_gradcam/sample_1.png")
                # from full path like "convnext_20250814_220650/train/..."
                parts = file.split(pattern, 1)
                if len(parts) == 2:
                    relative_path = parts[
                        1
                    ]  # This gives us "layer_gradcam/sample_1.png"
                    filtered_files.append(relative_path)
                break

    return filtered_files


def collect_run_metadata(
    checkpoints_dir: str,
    experiment_id: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Collect run metadata including NPY and PNG files from explanations dirs."""
    # First, try the local explanations directory (within checkpoints)
    explanations_dir = os.path.join(checkpoints_dir, "explanations")
    npy_files, png_files = _collect_explanation_files(explanations_dir)

    # Also check the global explanations directory (sibling to checkpoints)
    global_explanations_dir = "./explanations"
    if os.path.exists(global_explanations_dir):
        global_npy, global_png = _collect_explanation_files(global_explanations_dir)
        npy_files.extend(global_npy)
        png_files.extend(global_png)

    # Filter by experiment ID
    filtered_npy = _filter_files_by_experiment_id(npy_files, experiment_id)
    filtered_png = _filter_files_by_experiment_id(png_files, experiment_id)

    if logger:
        logger.debug(
            f"[DEBUG] Found {len(filtered_npy)} NPY files and "
            f"{len(filtered_png)} PNG files for experiment {experiment_id}"
        )
        logger.debug(f"[DEBUG] NPY files: {filtered_npy}")
        logger.debug(f"[DEBUG] PNG files: {filtered_png}")

    return {"npy_files": filtered_npy, "png_files": filtered_png}
