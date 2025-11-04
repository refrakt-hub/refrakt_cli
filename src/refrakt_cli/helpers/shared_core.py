"""
Metadata extraction utilities for Refrakt CLI.

This module contains the extract_comprehensive_metadata function and related
utilities that were previously part of shared_core.py.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from refrakt_cli.helpers.metadata_helpers import (
    determine_train_inference,
    extract_experiment_metadata_helper,
    extract_metadata_from_config,
    initialize_metadata_structure,
    load_config_files,
)
from refrakt_cli.utils.metadata_utils import (
    collect_run_metadata,
    create_experiment_info,
    merge_performance_metrics,
)


def extract_comprehensive_metadata(
    config_files: List[str],
    base_dir: str,
    checkpoints_dir: str,
    logger: Optional[logging.Logger] = None,
    training_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from config files, logs, and training results.
    Implements smart merging to preserve good metrics from previous phases.
    """
    _log_metadata_call(logger, training_results)
    summary_metrics_path = os.path.join(
        checkpoints_dir, "explanations", "summary_metrics.json"
    )
    existing_metadata = _load_existing_summary_metrics(summary_metrics_path, logger)
    experiment_metadata = extract_experiment_metadata_helper(checkpoints_dir, logger)
    has_train, has_inference = determine_train_inference(config_files)
    metadata = initialize_metadata_structure(
        experiment_metadata, has_train, has_inference
    )
    metadata["experiment_info"] = create_experiment_info(
        experiment_metadata, has_train, has_inference
    )
    experiment_id = metadata["experiment_info"]["experiment_id"]
    config_data = load_config_files(config_files, logger)
    new_performance_metrics = merge_performance_metrics(
        base_dir, training_results, logger
    )
    if existing_metadata and existing_metadata.get("performance_metrics"):
        metadata["performance_metrics"] = _merge_performance_metrics(
            existing_metadata["performance_metrics"], new_performance_metrics, logger
        )
    else:
        metadata["performance_metrics"] = new_performance_metrics
    if config_data:
        config_metadata = extract_metadata_from_config(config_data)
        if existing_metadata:
            metadata = _merge_config_metadata(
                metadata, config_metadata, existing_metadata
            )
        else:
            metadata.update(config_metadata)
    metadata["run_metadata"] = collect_run_metadata(
        checkpoints_dir, experiment_id, logger
    )
    _log_run_metadata_files(metadata, logger)
    runtime_xai_info = collect_runtime_xai_info(checkpoints_dir, logger)
    metadata = _update_runtime_xai_info(metadata, runtime_xai_info)
    metadata = _cleanup_metadata_keys(metadata)
    _log_about_to_write_summary(metadata, logger)
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict[str, Any]")
    return metadata


def _log_metadata_call(logger, training_results):
    if logger:
        logger.info(
            "extract_comprehensive_metadata called with training_results: "
            f"{training_results}"
        )


def _load_existing_summary_metrics(summary_metrics_path, logger):
    existing_metadata = None
    if os.path.exists(summary_metrics_path):
        try:
            with open(summary_metrics_path, "r") as f:
                existing_metadata = json.load(f)
                if logger:
                    logger.debug(
                        "[DEBUG] Found existing metadata, preserving good values"
                    )
        except Exception as e:
            if logger:
                logger.warning(f"Could not read existing summary_metrics.json: {e}")
    return existing_metadata


def _merge_performance_metrics(existing_perf, new_perf, logger):
    merged_perf = {}
    for key, value in existing_perf.items():
        if value not in ["N/A", "", None]:
            merged_perf[key] = value
    for key, value in new_perf.items():
        if value not in ["N/A", "", None]:
            merged_perf[key] = value
        elif key not in merged_perf:
            merged_perf[key] = value
    if logger:
        logger.debug(f"[DEBUG] Merged performance metrics: {merged_perf}")
    return merged_perf


def _merge_config_metadata(metadata, config_metadata, existing_metadata):
    for key in ["model_info", "training_info", "dataset_info", "xai_info", "viz_info"]:
        if key in config_metadata:
            if not existing_metadata.get(key) or existing_metadata[key] == {}:
                metadata[key] = config_metadata[key]
            else:
                metadata[key] = existing_metadata[key]
        elif existing_metadata.get(key):
            metadata[key] = existing_metadata[key]
    return metadata


def _log_run_metadata_files(metadata, logger):
    if logger:
        logger.debug(
            f"[DEBUG] run_metadata.npy_files: {metadata['run_metadata']['npy_files']}"
        )
        logger.debug(
            f"[DEBUG] run_metadata.png_files: {metadata['run_metadata']['png_files']}"
        )


def _update_runtime_xai_info(metadata, runtime_xai_info):
    if runtime_xai_info and "xai_info" in metadata:
        if "runtime_info" not in metadata["xai_info"]:
            metadata["xai_info"]["runtime_info"] = runtime_xai_info
        else:
            metadata["xai_info"]["runtime_info"].update(runtime_xai_info)
    elif runtime_xai_info:
        if "xai_info" not in metadata:
            metadata["xai_info"] = {}
        metadata["xai_info"]["runtime_info"] = runtime_xai_info
    return metadata


def _cleanup_metadata_keys(metadata):
    if "config_files" in metadata["run_metadata"]:
        del metadata["run_metadata"]["config_files"]
    return metadata


def _log_about_to_write_summary(metadata, logger):
    if logger:
        logger.debug(
            "[DEBUG] About to write summary: experiment_id="
            f"{metadata['experiment_info']['experiment_id']}, "
            f"metadata={json.dumps(metadata, indent=2)}"
        )


def collect_runtime_xai_info(
    checkpoints_dir: str, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Collect runtime XAI method information including layer details.
    This function looks for XAI instances that may have been saved or can be
    introspected.
    """
    runtime_xai_info = {}

    try:
        # Look for XAI hook configurations in config files or runtime data
        # This is a placeholder for future enhancement where we can introspect
        # actual XAI method instances and extract their runtime parameters

        # For now, we'll return an empty dict but structure is ready for enhancement
        # When XAI methods are executed, they could save their configuration info
        # to a runtime_xai_info.json file that we could read here

        runtime_info_path = os.path.join(
            checkpoints_dir, "explanations", "runtime_xai_info.json"
        )
        if os.path.exists(runtime_info_path):
            with open(runtime_info_path, "r") as f:
                runtime_xai_info = json.load(f)
                if logger:
                    logger.info(f"Loaded runtime XAI info from {runtime_info_path}")
    except Exception as e:
        if logger:
            logger.warning(f"Error collecting runtime XAI info: {e}")

    return runtime_xai_info


def save_runtime_xai_info(
    xai_instance: Any,
    method_name: str,
    params: Dict[str, Any],
    base_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Save runtime XAI method information to a JSON file.
    This captures method parameters, layer information, and other runtime details.
    """
    try:
        runtime_info = {
            "method": method_name,
            "config_params": params,
        }
        runtime_info = _collect_layer_info(
            xai_instance, method_name, logger, runtime_info
        )
        runtime_info = _get_resolved_layer_type(xai_instance, runtime_info)
        explanations_dir = os.path.join(base_dir, "explanations")
        os.makedirs(explanations_dir, exist_ok=True)
        runtime_file = os.path.join(explanations_dir, "runtime_xai_info.json")
        existing_info = _load_existing_runtime_info(runtime_file)
        existing_info = _update_runtime_info(existing_info, method_name, runtime_info)
        _save_runtime_info(runtime_file, existing_info)
        if logger:
            logger.debug(f"Saved runtime XAI info for {method_name} to {runtime_file}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save runtime XAI info for {method_name}: {e}")


def _collect_layer_info(xai_instance, method_name, logger, runtime_info):
    if hasattr(xai_instance, "get_target_layer_info"):
        try:
            layer_info = xai_instance.get_target_layer_info()
            if layer_info:
                runtime_info["layer_info"] = layer_info
        except Exception as e:
            if logger:
                logger.warning(f"Failed to get layer info for {method_name}: {e}")
    return runtime_info


def _get_resolved_layer_type(xai_instance, runtime_info):
    if hasattr(xai_instance, "layer") and xai_instance.layer is not None:
        runtime_info["resolved_layer_type"] = type(xai_instance.layer).__name__
    return runtime_info


def _load_existing_runtime_info(runtime_file):
    existing_info = {}
    if os.path.exists(runtime_file):
        try:
            with open(runtime_file, "r") as f:
                existing_info = json.load(f)
        except Exception:
            existing_info = {}
    return existing_info


def _update_runtime_info(existing_info, method_name, runtime_info):
    if "methods" not in existing_info:
        existing_info["methods"] = {}
    existing_info["methods"][method_name] = runtime_info
    return existing_info


def _save_runtime_info(runtime_file, existing_info):
    with open(runtime_file, "w") as f:
        json.dump(existing_info, f, indent=2)
