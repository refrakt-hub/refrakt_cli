"""
Refrakt CLI entry point for training, testing, and inference.

This module serves as the command-line interface entry point for the Refrakt framework.
It parses command-line arguments, sets up logging and configuration, and dispatches
to the appropriate pipeline stage (train, test, or inference).

The module handles:
- Command-line argument parsing and validation
- Configuration loading and override application
- Logging setup and configuration
- Pipeline mode execution and error handling
- Memory cleanup and resource management
"""

import gc

import torch
from omegaconf import OmegaConf
from refrakt_core.api.utils.pipeline_utils import setup_logger_and_config
from refrakt_cli.helpers.config_overrides import (
    setup_logging_config, 
    extract_runtime_config, 
    apply_config_overrides, 
    extract_overrides
)

from refrakt_cli.helpers.argument_parser import setup_argument_parser
from refrakt_cli.helpers.pipeline_orchestrator import execute_pipeline_mode

def main() -> None:
    """
    Main entry point for the Refrakt CLI.

    This function serves as the primary command-line interface for the Refrakt \
        framework. It parses command-line arguments, sets up logging and \
        configuration, and dispatches to the appropriate pipeline stage \
        (train, test, or inference).

    The function handles the complete CLI workflow including:
    - Argument parsing and validation
    - Configuration loading and override application
    - Logging setup and configuration
    - Pipeline execution with error handling
    - Resource cleanup and memory management

    Raises:
        SystemExit: If the pipeline fails due to configuration errors, argument
                   validation issues, or other critical failures. The function will
                   log detailed error information before exiting.
    """
    try:
        print("==> Refrakt CLI launched")

        parser = setup_argument_parser()
        args, remaining = parser.parse_known_args()

        all_overrides = extract_overrides(args, remaining)

        cfg = OmegaConf.load(args.config)
        cfg = apply_config_overrides(cfg, all_overrides)

        runtime_cfg = extract_runtime_config(cfg)
        mode, log_dir, log_types, console, model_path, debug = setup_logging_config(
            runtime_cfg, args.log_dir
        )

        debug = args.debug or debug

        if OmegaConf.is_config(cfg):
            model_name = cfg.model.name
        else:
            model_name = cfg.get("model", {}).get("name", "unknown")

        # Setup logger
        logger = setup_logger_and_config(
            cfg, model_name, log_dir, log_types, console, debug, all_overrides
        )

        try:
            execute_pipeline_mode(mode, cfg, model_path or "", logger)

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            logger.info("Finalizing and saving logs...")
            logger.close()

    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
