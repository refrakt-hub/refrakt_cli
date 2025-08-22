from typing import Any, Optional
from omegaconf import DictConfig
from refrakt_cli.utils.time_utils import get_experiment_id
from refrakt_core.api.train import train
from refrakt_core.api.test import test
from refrakt_core.api.inference import inference
from refrakt_core.api.utils.pipeline_utils import parse_runtime_hooks
from refrakt_cli.llm_explainer import run_llm_explanations
from refrakt_core.error_handling import XAINotSupportedError

def _resolve_model_name_train(cfg: DictConfig):
    """Resolve model name for training pipeline, including variant for autoencoders."""
    model = cfg.get("model", {})
    name = model.get("name", "unknown")
    if name == "autoencoder":
        params = model.get("params", {})
        variant = params.get("variant", "simple")
        resolved_model_name = f"autoencoder_{variant}"
    else:
        resolved_model_name = name
    # Optionally handle custom dataset suffix if needed (see core logic)
    dataset_params = cfg.get("dataset", {}).get("params", {})
    dataset_path = dataset_params.get("path", "") or dataset_params.get("zip_path", "")
    if dataset_path and str(dataset_path).endswith(".zip"):
        resolved_model_name = f"{resolved_model_name}_custom"
    return resolved_model_name

def _handle_post_pipeline_llm(cfg, logger):
    from omegaconf import OmegaConf
    from refrakt_core.api.utils.pipeline_utils import parse_runtime_hooks
    from refrakt_core.error_handling import XAINotSupportedError
    from refrakt_cli.llm_explainer import run_llm_explanations
    try:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            cfg_dict = {}
        cfg_dict = dict(cfg_dict)
        _, _, explain_flag = parse_runtime_hooks(cfg_dict)  # type: ignore
        model_name = cfg_dict.get("model", {}).get("name", "").lower()
        model_type = cfg_dict.get("model", {}).get("type", "").lower()
        wrapper_name = cfg_dict.get("model", {}).get("wrapper", "").lower()
        contrastive_indicators = ['simclr', 'dino', 'msn', 'contrastive']
        is_contrastive = any(indicator in model_name or indicator in model_type or indicator in wrapper_name for indicator in contrastive_indicators)
        if explain_flag:
            if is_contrastive:
                logger.info("⚠️  XAI components are currently not supported for contrastive family models (SimCLR, DINO, MSN) in refrakt v1. Skipping LLM explainer for contrastive model.")
            else:
                logger.info("... triggering the llm explainer (post-pipeline)")
                run_llm_explanations(logger=logger)
    except XAINotSupportedError as e:
        logger.warning(f"XAI not supported for contrastive models: {e}")
        logger.info("Skipping LLM explainer for contrastive model")
    except Exception as e:
        logger.error(f"Failed to run LLM explainer: {e}")

def execute_pipeline_mode(
    mode: str, cfg, model_path, logger, config_path=None
) -> None:
    """
    Execute the appropriate pipeline based on mode.
    """
    experiment_id = get_experiment_id()
    logger.info(f"🔬 Experiment ID: {experiment_id}")
    
    PIPELINE_MODES = {
    "train": lambda: train(cfg, logger=logger, experiment_id=experiment_id),
    "test": lambda: test(cfg, model_path=model_path, logger=logger, experiment_id=experiment_id, config_path=config_path),
    "inference": lambda: inference(cfg, model_path, logger=logger, experiment_id=experiment_id, config_path=config_path),
    "pipeline": lambda: execute_full_pipeline(cfg, logger, experiment_id, config_path=config_path)
    }

    if mode in PIPELINE_MODES:
        PIPELINE_MODES[mode]()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    _handle_post_pipeline_llm(cfg, logger)

def execute_full_pipeline(cfg: DictConfig, logger: Any, experiment_id: str, config_path: Optional[str] = None) -> None:
    """
    Execute the full pipeline (train → test → inference).

    Args:
        cfg: Configuration object
        logger: Logger instance
        experiment_id: Unique experiment ID for consistent directory naming
    """
    # Train
    logger.info("🚀 Training phase started")
    train(cfg, logger=logger, experiment_id=experiment_id)
    logger.info("✅ Training completed successfully!")
    
    # Test - use the same experiment ID and construct the model path
    logger.info("🧪 Testing phase started")
    
    # Resolve the model name to include variant information (e.g., autoencoder_vae)
    resolved_model_name = _resolve_model_name_train(cfg)
    
    # Construct the model path based on the experiment directory structure
    model_path = f"./checkpoints/{resolved_model_name}_{experiment_id}/weights/{resolved_model_name}.pth"
    test(cfg, model_path=model_path, logger=logger, experiment_id=experiment_id, config_path=config_path)
    logger.info("✅ Testing completed successfully!")
    
    # Inference
    logger.info("🔮 Inference phase started")
    inference(cfg, model_path, logger=logger, experiment_id=experiment_id, config_path=config_path)
    # Note: "Inference completed successfully!" is already logged in the inference function
