"""
Pipeline orchestration for Refrakt CLI.
"""

import os
from typing import Any, Optional

from omegaconf import DictConfig
from refrakt_core.api.train import train
from refrakt_core.api.test import test
from refrakt_core.api.inference import inference


def execute_pipeline_mode(
    mode: str, cfg: DictConfig, model_path: Optional[str], logger: Any
) -> None:
    """
    Execute the appropriate pipeline based on mode.

    This function routes to the correct pipeline execution function based
    on the specified mode, handling different pipeline types with appropriate
    validation and error handling.

    Args:
        mode: Pipeline execution mode ('train', 'test', 'inference', 'pipeline')
        cfg: Configuration object for the pipeline
        model_path: Model path for inference mode
        logger: Logger instance for status messages

    Raises:
        ValueError: If model_path is required but not provided for inference mode
    """
    if mode == "train":
        logger.info("🚀 Starting training pipeline")
        train(cfg, logger=logger)

    elif mode == "test":
        logger.info("🧪 Starting testing pipeline")
        test(cfg, model_path=model_path, logger=logger)

    elif mode == "inference":
        if not model_path:
            raise ValueError(
                "model_path must be provided for inference mode"
            )
        logger.info("🔮 Starting inference pipeline")
        inference(cfg, model_path, logger=logger)

    elif mode == "pipeline":
        logger.info("🔁 Starting full pipeline (train → test → inference)")
        execute_full_pipeline(cfg, logger)

    else:
        raise ValueError(f"Invalid mode: {mode}")


def execute_full_pipeline(cfg: DictConfig, logger: Any) -> None:
    """
    Execute the full pipeline (train → test → inference).

    Args:
        cfg: Configuration object
        logger: Logger instance
    """
    # Resolve model name and path
    model_name = cfg.model.name
    if model_name == "autoencoder":
        variant = cfg.model.params.get("variant", "simple")
        resolved_model_name = f"autoencoder_{variant}"
    else:
        resolved_model_name = model_name

    # Check if using custom dataset
    dataset_params = (
        cfg.dataset.params
        if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "params")
        else {}
    )
    dataset_path = dataset_params.get("path", "") or dataset_params.get("zip_path", "")
    if dataset_path and str(dataset_path).endswith(".zip"):
        resolved_model_name = f"{resolved_model_name}_custom"

    save_dir = cfg.trainer.params.save_dir
    model_path = os.path.join(save_dir, f"{resolved_model_name}.pth")

    # Execute pipeline phases
    logger.info("🚀 Training phase started")
    train(cfg, logger=logger)
    
    logger.info("🧪 Testing phase started")
    test(cfg, model_path=model_path, logger=logger)
    
    logger.info("🔮 Inference phase started")
    inference(cfg, model_path=model_path, logger=logger)
