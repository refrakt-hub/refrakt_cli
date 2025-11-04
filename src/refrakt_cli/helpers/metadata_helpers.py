import glob
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

from refrakt_cli.utils.log_utils import (
    extract_key_metrics_from_logs,
    sort_log_files_by_time,
)


def _read_yaml_file(config_file: str) -> Dict[str, Any]:
    """Read and parse a YAML file, returning its contents as a dict."""
    with open(config_file, "r") as f:
        data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"YAML file {config_file} did not return a dict.")
        return data


def _log_config_event(logger: Optional[logging.Logger], message: str) -> None:
    if logger:
        logger.info(message)


def load_config_files(
    config_files: List[str], logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Load and parse configuration files.

    Args:
        config_files: List of configuration file paths
        logger: Optional logger

    Returns:
        Dictionary containing merged configuration data
    """
    config_data = {}
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                config_data.update(_read_yaml_file(config_file))
                _log_config_event(logger, f"Loaded config file: {config_file}")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not parse config file: {e}")
    return config_data


def merge_training_results(
    latest_metrics: Dict[str, Any],
    training_results: Optional[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Merge training results into the latest metrics.

    Args:
        latest_metrics: Dictionary of latest metrics
        training_results: Optional dictionary of training results
        logger: Optional logger

    Returns:
        Updated metrics dictionary
    """
    if training_results:
        if logger:
            logger.info(
                f"Merging training_results into latest_metrics: {training_results}"
            )
        for key, value in training_results.items():
            if value is not None:
                latest_metrics[key] = str(value)
        if "final_loss" in training_results and (
            "best_accuracy" not in latest_metrics
            or latest_metrics["best_accuracy"] == "N/A"
        ):
            try:
                final_loss = float(training_results["final_loss"])
                if final_loss <= 10.0:
                    accuracy_proxy = max(0.0, 1.0 - (final_loss / 10.0))
                    latest_metrics["best_accuracy"] = f"{accuracy_proxy:.4f}"
            except (ValueError, TypeError):
                pass
    return latest_metrics


def _parse_experiment_id_from_name(exp_dir_name: str) -> Tuple[str, str]:
    """Parse experiment ID and model name from directory name."""
    parts = exp_dir_name.split("_", 1)
    model_name = parts[0] if len(parts) > 0 else "unknown"
    # Keep the full datetime as experiment_id (e.g., "20250814_192304")
    experiment_id = parts[1] if len(parts) > 1 else "unknown"
    return model_name, experiment_id


def extract_experiment_id(
    exp_dir: str, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Extract metadata from experiment directory."""
    exp_dir_name = os.path.basename(exp_dir)
    model_name, experiment_id = _parse_experiment_id_from_name(exp_dir_name)

    metadata = {
        "experiment_dir": exp_dir,
        "model_name": model_name,
        "experiment_id": experiment_id,
        "has_train": os.path.exists(os.path.join(exp_dir, "train")),
        "has_inference": os.path.exists(os.path.join(exp_dir, "inference")),
    }

    if logger:
        logger.info(f"Extracting metadata for experiment: {model_name}_{experiment_id}")

    return metadata


def _validate_model_type(model_type: str) -> str:
    """Validate and return a standardized model type."""
    valid_types = {"classification", "regression", "clustering"}
    return model_type if model_type in valid_types else "unknown"


def determine_model_type(
    config: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    Determine the model type from the configuration.
    Only return type for autoencoder variants.
    """
    model_name = config.get("name", "").lower()
    model_wrapper = config.get("wrapper", "").lower()

    # Only return type for autoencoder variants
    autoencoder_patterns = ["autoencoder"]

    is_autoencoder = any(
        pattern in model_name or pattern in model_wrapper
        for pattern in autoencoder_patterns
    )

    if is_autoencoder:
        model_type = config.get("type", "unknown")
        validated_type = _validate_model_type(model_type)
        if logger:
            logger.info(f"Determined autoencoder model type: {validated_type}")
        return validated_type

    # For non-autoencoder models, return None (no type field)
    if logger:
        logger.info(
            f"Non-autoencoder model detected: {model_name}, no type field added"
        )
    return None


def extract_model_metadata(
    exp_dir: str, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Extract metadata related to the model from the experiment directory.

    Args:
        exp_dir: Path to the experiment directory.
        logger: Optional logger for logging information.

    Returns:
        A dictionary containing model metadata.
    """
    model_metadata = {
        "name": "unknown",
        "type": "unknown",
        "params": {},
        "wrapper": "unknown",
    }
    model_config_path = os.path.join(exp_dir, "model_config.yaml")

    if os.path.exists(model_config_path):
        try:
            with open(model_config_path, "r") as f:
                model_config = yaml.safe_load(f)
                model_metadata.update(
                    {
                        "name": model_config.get("name", "unknown"),
                        "params": model_config.get("params", {}),
                        "wrapper": model_config.get("wrapper", "unknown"),
                    }
                )

                # Only add 'type' field for autoencoder models
                model_type = determine_model_type(model_config, logger)
                if model_type is not None:
                    model_metadata["type"] = model_type

                if logger:
                    logger.info(f"Extracted model metadata from {model_config_path}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to extract model metadata: {e}")

    return model_metadata


def extract_dataset_metadata(
    exp_dir: str, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Extract metadata related to the dataset from the experiment directory.

    Args:
        exp_dir: Path to the experiment directory.
        logger: Optional logger for logging information.

    Returns:
        A dictionary containing dataset metadata.
    """
    dataset_metadata = {
        "name": "unknown",
        "type": "unknown",
        "params": {},
        "transforms": [],
    }
    dataset_config_path = os.path.join(exp_dir, "dataset_config.yaml")

    if os.path.exists(dataset_config_path):
        try:
            with open(dataset_config_path, "r") as f:
                dataset_config = yaml.safe_load(f)
                dataset_metadata.update(
                    {
                        "name": dataset_config.get("name", "unknown"),
                        "type": dataset_config.get("type", "unknown"),
                        "params": dataset_config.get("params", {}),
                        "transforms": dataset_config.get("transform", []),
                    }
                )
                if logger:
                    logger.info(
                        f"Extracted dataset metadata from {dataset_config_path}"
                    )
        except Exception as e:
            if logger:
                logger.warning(f"Failed to extract dataset metadata: {e}")

    return dataset_metadata


def extract_experiment_metadata_helper(
    checkpoints_dir: str, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Extract experiment metadata such as experiment_id."""
    experiment_id = extract_experiment_id(checkpoints_dir, logger)
    if logger:
        logger.debug(f"[DEBUG] Extracted experiment_id: {experiment_id}")
    return {"experiment_id": experiment_id}


def _normalize_filename(config_file: str) -> str:
    """Return the lowercase basename of a config file path."""
    return os.path.basename(config_file).lower()


def _detect_train_flag(fname: str) -> bool:
    return "train" in fname or "fit" in fname


def _detect_inference_flag(fname: str) -> bool:
    return any(x in fname for x in ["inference", "infer", "test", "eval"])


def determine_train_inference(config_files: List[str]) -> Tuple[bool, bool]:
    """Determine if training and inference are present based on config files."""
    has_train = has_inference = False
    for config_file in config_files:
        fname = _normalize_filename(config_file)
        if _detect_train_flag(fname):
            has_train = True
        if _detect_inference_flag(fname):
            has_inference = True
    if not (has_train or has_inference):
        has_train = has_inference = True
    return has_train, has_inference


def initialize_metadata_structure(
    experiment_metadata: Dict[str, Any], has_train: bool, has_inference: bool
) -> Dict[str, Any]:
    """Initialize the metadata structure with basic experiment info."""
    return {
        "experiment_info": {
            **experiment_metadata,
            "has_train": has_train,
            "has_inference": has_inference,
        },
        "model_info": {},
        "training_info": {},
        "dataset_info": {},
        "xai_info": {},
        "performance_metrics": {},
        "run_metadata": {},
    }


def _merge_training_results_into_metrics(
    latest_metrics: Dict[str, Any],
    training_results: Optional[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    if training_results:
        if logger:
            logger.debug(
                "[DEBUG] Merging training_results into latest_metrics: "
                f"{training_results}"
            )
        for key, value in training_results.items():
            if value is not None:
                # Only update if current value is N/A or empty,
                # or if new value is better (non-N/A)
                current_val = latest_metrics.get(key, "N/A")
                if current_val in ["N/A", "", None] or str(value) != "N/A":
                    latest_metrics[key] = str(value)
    return latest_metrics


def _add_proxy_accuracy_if_needed(
    latest_metrics: Dict[str, Any], training_results: Optional[Dict[str, Any]]
) -> None:
    if training_results and (
        "final_loss" in training_results
        and (
            "best_accuracy" not in latest_metrics
            or latest_metrics["best_accuracy"] == "N/A"
        )
    ):
        try:
            final_loss = float(training_results["final_loss"])
            if final_loss <= 10.0:
                accuracy_proxy = max(0.0, 1.0 - (final_loss / 10.0))
                latest_metrics["best_accuracy"] = f"{accuracy_proxy:.4f}"
        except (ValueError, TypeError):
            pass


def extract_performance_metrics(
    base_dir: str,
    training_results: Optional[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Extract performance metrics from logs and training results."""
    log_files = glob.glob(os.path.join(base_dir, "logs", "**", "*.log"), recursive=True)
    latest_metrics = extract_key_metrics_from_logs(log_files, logger)
    if logger:
        logger.debug(f"[DEBUG] latest_metrics after log extraction: {latest_metrics}")
    latest_metrics = _merge_training_results_into_metrics(
        latest_metrics, training_results, logger
    )
    _add_proxy_accuracy_if_needed(latest_metrics, training_results)
    if logger:
        logger.debug(f"[DEBUG] Final performance_metrics: {latest_metrics}")
    return latest_metrics


def _normalize_training_results(
    training_results: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Normalize training_results to a dict if possible."""
    if training_results is None:
        return None
    if not isinstance(training_results, dict):
        if hasattr(training_results, "summary") and callable(training_results.summary):  # type: ignore[unreachable]
            return training_results.summary()
        return None
    return training_results


def _update_metrics_with_numeric_values(
    latest_metrics: Dict[str, Any], training_results: Dict[str, Any]
) -> None:
    """Update metrics with numeric or numeric-string values from training_results."""
    for k, v in training_results.items():
        if isinstance(v, (int, float)) or (
            isinstance(v, str) and v.replace(".", "", 1).isdigit()
        ):
            latest_metrics[k] = str(v)


def _update_metrics_with_non_numeric_values(
    latest_metrics: Dict[str, Any], training_results: Dict[str, Any]
) -> None:
    """
    Update metrics with non-numeric, non-empty values from training_results
    if not already present.
    """
    for k, v in training_results.items():
        if v not in [None, "", "N/A"] and k not in latest_metrics:
            latest_metrics[k] = v


def _merge_metrics_dicts(
    latest_metrics: Dict[str, Any], training_results: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    norm_results = _normalize_training_results(training_results)
    if norm_results is None:
        return latest_metrics
    _update_metrics_with_numeric_values(latest_metrics, norm_results)
    _update_metrics_with_non_numeric_values(latest_metrics, norm_results)
    return latest_metrics


def merge_training_results_with_metrics(
    training_results: Optional[Dict[str, Any]],
    latest_metrics: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    return _merge_metrics_dicts(latest_metrics, training_results)


def _filter_recent_log_files(
    log_files_with_time: List[Tuple[str, float]], max_age_seconds: int = 86400
) -> List[Tuple[str, float]]:
    current_time = time.time()
    return [
        item
        for item in log_files_with_time
        if current_time - float(item[1]) <= max_age_seconds
    ]


def _extract_metrics_from_log_file(
    log_file: str, metrics: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()

        # Extract metrics from lines
        for line in lines:
            from refrakt_cli.utils.log_utils import (
                _extract_accuracy,
                _extract_epoch,
                _extract_loss,
            )

            acc_val = _extract_accuracy(line)
            if acc_val is not None:
                metrics["best_accuracy"] = str(acc_val)

            loss_val = _extract_loss(line)
            if loss_val is not None:
                metrics["final_loss"] = str(loss_val)

            epoch_val = _extract_epoch(line)
            if epoch_val is not None:
                metrics["epochs_completed"] = str(epoch_val)

        # Extract training time from all lines
        from refrakt_cli.utils.log_utils import _extract_training_time

        training_time = _extract_training_time(lines)
        if training_time is not None:
            metrics["training_time"] = str(round(training_time, 2))

        return metrics
    except Exception as e:
        if logger:
            logger.warning(f"Failed to extract metrics from {log_file}: {e}")
        return metrics


def filter_na_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out metrics with value 'N/A'."""
    return {k: v for k, v in metrics.items() if v != "N/A"}


def _extract_model_info(config_data: Dict[str, Any]) -> Dict[str, Any]:
    if "model" not in config_data:
        return {}
    model = config_data["model"]

    model_info = {
        "name": model.get("name", "unknown"),
        "wrapper": model.get("wrapper", "unknown"),
        "params": model.get("params", {}),
    }

    # Only add type field for autoencoder variants
    model_type = determine_model_type(model)
    if model_type is not None:
        model_info["type"] = model_type

    # Add fusion information if available
    if "fusion" in model:
        fusion_info = model["fusion"]
        model_info["fusion"] = {
            "type": fusion_info.get("type", "unknown"),
            "model": fusion_info.get("model", "unknown"),
            "params": fusion_info.get("params", {}),
        }

    return model_info


def _extract_trainer_info(config_data: Dict[str, Any]) -> Dict[str, Any]:
    if "trainer" not in config_data:
        return {}
    trainer = config_data["trainer"]
    return {
        "name": trainer.get("name", "unknown"),
        "params": trainer.get("params", {}),
        "epochs": trainer.get("params", {}).get("num_epochs", "N/A"),
    }


def _extract_dataset_info(config_data: Dict[str, Any]) -> Dict[str, Any]:
    if "dataset" not in config_data:
        return {}
    dataset = config_data["dataset"]
    return {
        "name": dataset.get("name", "unknown"),
        "params": dataset.get("params", {}),
        "transforms": dataset.get("transform", []),
    }


def _extract_xai_viz_info(config_data: Dict[str, Any]) -> Tuple[Any, ...]:
    xai_info = {}
    viz_info = {}
    if "runtime" in config_data and "hooks" in config_data["runtime"]:
        hooks = config_data["runtime"]["hooks"]
        explainability_configs = hooks.get("explainability", [])
        xai_methods, method_configs, layer_info = _extract_xai_methods_and_configs(
            explainability_configs
        )
        xai_info = _build_xai_info(xai_methods, method_configs, layer_info, hooks)
        viz_info = _build_viz_info(hooks)
    return xai_info, viz_info


def _extract_xai_methods_and_configs(explainability_configs):
    xai_methods = []
    method_configs = {}
    layer_info = {}
    for config in explainability_configs:
        if isinstance(config, dict) and "method" in config:
            method_name = config["method"]
            xai_methods.append(method_name)
            if "params" in config:
                params = config["params"]
                method_configs[method_name] = params
                if "layer" in params:
                    layer_info["target_layer"] = params["layer"]
                elif "target_layer" in params:
                    layer_info["target_layer"] = params["target_layer"]
        elif isinstance(config, str):
            xai_methods.append(config)
    return xai_methods, method_configs, layer_info


def _build_xai_info(xai_methods, method_configs, layer_info, hooks):
    xai_info = {"methods": xai_methods, "explain_flag": hooks.get("explain", False)}
    if method_configs:
        xai_info["method_configs"] = method_configs
    if layer_info:
        xai_info["layer_info"] = layer_info
    return xai_info


def _build_viz_info(hooks):
    return {"methods": hooks.get("visualizations", [])}


def extract_metadata_from_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from configuration data."""
    metadata = {}
    model_info = _extract_model_info(config_data)
    if model_info:
        metadata["model_info"] = model_info
    trainer_info = _extract_trainer_info(config_data)
    if trainer_info:
        metadata["training_info"] = trainer_info
    dataset_info = _extract_dataset_info(config_data)
    if dataset_info:
        metadata["dataset_info"] = dataset_info
    xai_info, viz_info = _extract_xai_viz_info(config_data)
    if xai_info:
        metadata["xai_info"] = xai_info
    if viz_info:
        metadata["viz_info"] = viz_info
    return metadata


def extract_run_metadata(
    config_files: List[str], checkpoints_dir: str
) -> Dict[str, Any]:
    """
    Extract run metadata such as file lists, including from explanations
    directory.
    """
    # Check both checkpoints_dir and explanations directory
    npy_files = []
    png_files = []

    # First, look in checkpoints_dir
    npy_files.extend(
        glob.glob(os.path.join(checkpoints_dir, "**", "*.npy"), recursive=True)
    )
    png_files.extend(
        glob.glob(os.path.join(checkpoints_dir, "**", "*.png"), recursive=True)
    )

    # Also look in the explanations directory (sibling to checkpoints)
    explanations_dir = os.path.join(os.path.dirname(checkpoints_dir), "explanations")
    if os.path.exists(explanations_dir):
        npy_files.extend(
            glob.glob(os.path.join(explanations_dir, "**", "*.npy"), recursive=True)
        )
        png_files.extend(
            glob.glob(os.path.join(explanations_dir, "**", "*.png"), recursive=True)
        )

    # Fallback: look in global explanations directory
    global_explanations_dir = "./explanations"
    if os.path.exists(global_explanations_dir):
        npy_files.extend(
            glob.glob(
                os.path.join(global_explanations_dir, "**", "*.npy"), recursive=True
            )
        )
        png_files.extend(
            glob.glob(
                os.path.join(global_explanations_dir, "**", "*.png"), recursive=True
            )
        )

    # Extract experiment ID from checkpoints_dir for filtering
    experiment_id = (
        os.path.basename(checkpoints_dir).split("_")[-1]
        if "_" in os.path.basename(checkpoints_dir)
        else "unknown"
    )

    # Filter files by experiment ID using the same logic as collect_run_metadata
    def filter_by_experiment(files: list[str], exp_id: str) -> list[str]:
        experiment_patterns: list[str] = [
            f"{exp_id}/train",
            f"{exp_id}/inference",
            f"convnext_{exp_id}/train",
            f"convnext_{exp_id}/inference",
        ]
        filtered: list[str] = []
        for file in files:
            for pattern in experiment_patterns:
                if pattern in file:
                    filtered.append(file)
                    break
        return filtered

    filtered_npy = filter_by_experiment(npy_files, experiment_id)
    filtered_png = filter_by_experiment(png_files, experiment_id)

    return {
        "config_files": [os.path.basename(f) for f in config_files],
        "npy_files": [os.path.basename(f) for f in filtered_npy],
        "png_files": [os.path.basename(f) for f in filtered_png],
    }


def fix_experiment_info_structure(
    experiment_metadata: Dict[str, Any], has_train: bool, has_inference: bool
) -> Dict[str, Any]:
    """Fix the structure of experiment_info in metadata."""
    experiment_id = experiment_metadata.get("experiment_id")
    if isinstance(experiment_id, dict):
        experiment_id = (
            experiment_id.get("experiment_id")
            or experiment_id.get("experiment_dir", "").split("_")[-1]
        )
    if isinstance(experiment_id, str):
        # Keep full experiment ID with datetime (e.g., "20250814_192304")
        experiment_id = experiment_id if "_" in experiment_id else experiment_id
    else:
        experiment_id = "unknown_experiment"
    return {
        "experiment_id": experiment_id,
        "has_train": has_train,
        "has_inference": has_inference,
    }


def robust_merge_performance_metrics(
    base_dir: str,
    training_results: Optional[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    latest_metrics = extract_performance_metrics(base_dir, training_results, logger)
    return _merge_metrics_dicts(latest_metrics, training_results)


def _sort_log_files_by_time(log_files: List[str]) -> List[Tuple[str, float]]:
    raw = sort_log_files_by_time(log_files)
    result = []
    for x in raw:
        if isinstance(x, tuple) and len(x) == 2:
            fname, tval = x
            try:
                result.append((str(fname), float(tval)))
            except Exception:
                continue
    return result


def _extract_metrics_from_log_lines_helper(
    lines: List[str], metrics: Dict[str, Any]
) -> Dict[str, Any]:
    # Assuming logger is not available here, so just pass lines and metrics as
    # required by the actual function signature.
    # This function should extract metrics from log lines directly, not via
    # extract_key_metrics_from_logs.
    # Implement a simple placeholder that returns the metrics dict unchanged.
    # TODO: Implement actual extraction logic if needed.
    return metrics


def _collect_files_by_extension(
    explanations_dir: str, extensions: Tuple[str, ...]
) -> Dict[str, List[str]]:
    files: Dict[str, List[str]] = {ext: [] for ext in extensions}
    for root, _, fs in os.walk(explanations_dir):
        for file in fs:
            for ext in extensions:
                if file.endswith(ext):
                    rel_path = os.path.relpath(
                        os.path.join(root, file), explanations_dir
                    )
                    files[ext].append(rel_path)
    return files


def _filter_files_by_experiment_id(files: List[str], experiment_id: str) -> List[str]:
    exp_prefix_train = f"{experiment_id}/train"
    exp_prefix_inference = f"{experiment_id}/inference"
    return [
        f
        for f in files
        if f.startswith(exp_prefix_train) or f.startswith(exp_prefix_inference)
    ]


def extract_run_files_metadata(
    explanations_dir: str, experiment_id: str
) -> Dict[str, List[str]]:
    files = _collect_files_by_extension(explanations_dir, (".npy", ".png"))
    filtered_npy = _filter_files_by_experiment_id(files[".npy"], experiment_id)
    filtered_png = _filter_files_by_experiment_id(files[".png"], experiment_id)
    return {
        "npy_files": filtered_npy if filtered_npy else files[".npy"],
        "png_files": filtered_png if filtered_png else files[".png"],
    }


def collect_explanations_run_metadata(
    checkpoints_dir: str, experiment_id: str
) -> Dict[str, Any]:
    explanations_dir = os.path.join(checkpoints_dir, "explanations")
    files = _collect_files_by_extension(explanations_dir, (".npy", ".png"))
    filtered_npy = _filter_files_by_experiment_id(files[".npy"], experiment_id)
    filtered_png = _filter_files_by_experiment_id(files[".png"], experiment_id)
    return {
        "npy_files": filtered_npy if filtered_npy else files[".npy"],
        "png_files": filtered_png if filtered_png else files[".png"],
    }
