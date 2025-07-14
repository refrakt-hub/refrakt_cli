import sys
from unittest.mock import patch, Mock
import pytest
from omegaconf import OmegaConf

from refrakt_cli.cli import main

class TestCLIMinimal:
    @patch("refrakt_cli.helpers.pipeline_orchestrator.train")
    @patch("refrakt_cli.cli.setup_argument_parser")
    @patch("refrakt_cli.cli.extract_overrides")
    @patch("refrakt_cli.cli.apply_config_overrides")
    @patch("refrakt_cli.cli.setup_logger_and_config")
    @patch("refrakt_cli.cli.OmegaConf.load")
    def test_main_success(
        self, mock_load, mock_logger, mock_apply, mock_extract, mock_parser, mock_train
    ):
        # Setup mocks
        mock_args = Mock()
        mock_args.config = "test_config.yaml"
        mock_args.mode = "train"
        mock_args.model_path = None
        mock_args.log_dir = None
        mock_args.debug = False
        mock_args.override = None

        mock_remaining = []
        mock_parser.return_value.parse_known_args.return_value = (mock_args, mock_remaining)
        mock_cfg = OmegaConf.create({"model": {"name": "test"}})
        mock_load.return_value = mock_cfg
        mock_extract.return_value = []
        mock_apply.return_value = mock_cfg
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        mock_train.return_value = {"status": "completed"}

        with patch("sys.argv", ["refrakt", "--config", "test_config.yaml"]):
            with patch("builtins.print"):
                main()

        mock_parser.assert_called_once()
        mock_extract.assert_called_once_with(mock_args, mock_remaining)
        mock_load.assert_called_once_with("test_config.yaml")
        mock_apply.assert_called_once_with(mock_cfg, [])
        mock_logger.assert_called_once()
        mock_train.assert_called_once_with(mock_cfg, logger=mock_logger_instance)

    @patch("refrakt_cli.helpers.pipeline_orchestrator.train")
    @patch("refrakt_cli.cli.setup_argument_parser")
    @patch("refrakt_cli.cli.extract_overrides")
    @patch("refrakt_cli.cli.OmegaConf.load")
    def test_main_config_error(self, mock_load, mock_extract, mock_parser, mock_train):
        mock_args = Mock()
        mock_args.config = "invalid_config.yaml"
        mock_args.override = []
        mock_remaining = []
        mock_parser.return_value.parse_known_args.return_value = (mock_args, mock_remaining)
        mock_load.side_effect = FileNotFoundError("Config file not found")

        with patch("sys.argv", ["refrakt", "--config", "invalid_config.yaml"]):
            with patch("builtins.print"):
                with pytest.raises(FileNotFoundError):
                    main()

    @patch("refrakt_cli.helpers.pipeline_orchestrator.train")
    @patch("refrakt_cli.cli.setup_argument_parser")
    @patch("refrakt_cli.cli.extract_overrides")
    @patch("refrakt_cli.cli.apply_config_overrides")
    @patch("refrakt_cli.cli.setup_logger_and_config")
    @patch("refrakt_cli.cli.OmegaConf.load")
    def test_main_keyboard_interrupt(
        self, mock_load, mock_logger, mock_apply, mock_extract, mock_parser, mock_train
    ):
        mock_args = Mock()
        mock_args.config = "test_config.yaml"
        mock_args.mode = "train"
        mock_args.model_path = None
        mock_args.log_dir = None
        mock_args.debug = False
        mock_args.override = None

        mock_remaining = []
        mock_parser.return_value.parse_known_args.return_value = (mock_args, mock_remaining)
        mock_cfg = OmegaConf.create({"model": {"name": "test"}})
        mock_load.return_value = mock_cfg
        mock_extract.return_value = []
        mock_apply.return_value = mock_cfg
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        mock_train.side_effect = KeyboardInterrupt()

        with patch("sys.argv", ["refrakt", "--config", "test_config.yaml"]):
            with patch("builtins.print"):
                main()

        mock_logger_instance.warning.assert_called_with("Training interrupted by user")

    @patch("refrakt_cli.helpers.pipeline_orchestrator.train")
    @patch("refrakt_cli.cli.setup_argument_parser")
    @patch("refrakt_cli.cli.extract_overrides")
    @patch("refrakt_cli.cli.apply_config_overrides")
    @patch("refrakt_cli.cli.setup_logger_and_config")
    @patch("refrakt_cli.cli.OmegaConf.load")
    def test_main_pipeline_error(
        self, mock_load, mock_logger, mock_apply, mock_extract, mock_parser, mock_train
    ):
        mock_args = Mock()
        mock_args.config = "test_config.yaml"
        mock_args.mode = "train"
        mock_args.model_path = None
        mock_args.log_dir = None
        mock_args.debug = False
        mock_args.override = None

        mock_remaining = []
        mock_parser.return_value.parse_known_args.return_value = (mock_args, mock_remaining)
        mock_cfg = OmegaConf.create({"model": {"name": "test"}})
        mock_load.return_value = mock_cfg
        mock_extract.return_value = []
        mock_apply.return_value = mock_cfg
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        mock_train.side_effect = RuntimeError("Pipeline failed")

        with patch("sys.argv", ["refrakt", "--config", "test_config.yaml"]):
            with patch("builtins.print"):
                with pytest.raises(RuntimeError):
                    main()

        mock_logger_instance.error.assert_called_with("Pipeline failed: Pipeline failed") 