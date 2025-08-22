"""
Configuration override handling for Refrakt CLI.
"""

from typing import Any, Dict, List, Optional, cast

from omegaconf import DictConfig, OmegaConf

from refrakt_cli.hooks.hyperparameter_override import (
    apply_overrides,
    extract_overrides_from_args,
)


def extract_overrides(args: Any, remaining: List[str]) -> List[str]:
    """
    Extract and combine all overrides from arguments.

    This function combines explicit --override flags with positional overrides
    to create a comprehensive list of configuration overrides to apply.

    Args:
        args: Parsed command-line arguments
        remaining: Remaining positional arguments that may contain overrides

    Returns:
        Combined list of all override strings to apply

    Note:
        This function handles both explicit --override flags and positional
        overrides, ensuring all override methods are properly combined.
    """
    positional_overrides, _ = extract_overrides_from_args(remaining)

    # Combine explicit --override flags with positional overrides
    all_overrides: List[str] = []
    if args.override:
        all_overrides.extend(args.override)
    all_overrides.extend(positional_overrides)

    return all_overrides


def apply_config_overrides(cfg: Any, all_overrides: List[str]) -> Any:
    """
    Apply overrides to configuration.

    This function applies configuration overrides to the main configuration
    object, enabling runtime parameter modifications without changing
    configuration files.

    Args:
        cfg: Configuration object to modify
        all_overrides: List of override strings to apply

    Returns:
        Updated configuration object with overrides applied

    Note:
        The function includes debug logging to track override application
        and ensure proper parameter modification.
    """
    if all_overrides:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg_dict, dict):
            cfg_dict = apply_overrides(OmegaConf.create(cfg_dict), all_overrides)
            cfg = OmegaConf.create(cfg_dict)
    return cfg


def extract_runtime_config(cfg: DictConfig) -> Dict[str, Any]:
    """
    Extract runtime configuration from config.

    This function extracts the runtime configuration section from the main
    configuration object, which contains pipeline execution settings.

    Args:
        cfg: Configuration object containing runtime settings

    Returns:
        Runtime configuration dictionary

    Raises:
        TypeError: If the configuration cannot be converted to a dictionary
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(
            f"Config must be a dict after OmegaConf.to_container, "
            f"got type: {type(cfg_dict)}"
        )
    cfg_dict = cast(Dict[str, Any], cfg_dict)
    runtime_config = cfg_dict.get("runtime", {})
    if isinstance(runtime_config, dict):
        return runtime_config
    return {}


def setup_logging_config(
    runtime_cfg: Dict[str, Any], args_log_dir: Optional[str] = None
) -> tuple[str, str, list[str], bool, Optional[str], bool]:
    """
    Setup logging configuration from runtime config.

    This function extracts and validates logging configuration parameters
    from the runtime configuration, handling various parameter types and
    providing sensible defaults.

    Args:
        runtime_cfg: Runtime configuration dictionary
        args_log_dir: Optional log directory override from command line

    Returns:
        Tuple containing:
        - mode: Pipeline execution mode
        - log_dir: Log directory path
        - log_types: List of logging backends
        - console: Whether to enable console logging
        - model_path: Optional model path for inference
        - debug: Whether debug logging is enabled
    """
    mode = runtime_cfg.get("mode", "train")
    log_dir = args_log_dir or runtime_cfg.get("log_dir", "./logs")

    # Handle log_types - accept list or single string
    log_types = runtime_cfg.get("log_type", [])
    if isinstance(log_types, str):
        log_types = [log_types]  # Convert single string to list
    elif log_types is None:
        log_types = []  # Convert None to empty list

    console = runtime_cfg.get("console", True)
    model_path = runtime_cfg.get("model_path", None)
    debug = runtime_cfg.get("debug", False)
    return mode, log_dir, log_types, console, model_path, debug
