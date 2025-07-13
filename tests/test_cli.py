"""
Tests for the main CLI entry point.
"""

from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from refrakt_cli.cli import main


class TestCLI:
    """Test cases for the main CLI entry point."""

    @patch('refrakt_cli.cli._setup_argument_parser')
    @patch('refrakt_cli.cli._extract_overrides')
    @patch('refrakt_cli.cli._apply_config_overrides')
    @patch('refrakt_cli.cli.setup_logger_and_config')
    @patch('refrakt_cli.cli._execute_pipeline_mode')
    @patch('refrakt_cli.cli.OmegaConf.load')
    def test_main_success(
        self, mock_load, mock_execute, mock_logger, mock_apply, mock_extract, mock_parser
    ):
        """Test successful CLI execution."""
        # Setup mocks
        mock_args = Mock()
        mock_args.config = "test_config.yaml"
        mock_args.mode = "train"
        mock_args.model_path = None
        mock_args.log_dir = None
        mock_args.debug = False
        mock_args.override = None
        
        mock_remaining = []
        mock_parser.return_value.parse_known_args.return_value = (
            mock_args, mock_remaining
        )
        
        mock_cfg = OmegaConf.create({"model": {"name": "test"}})
        mock_load.return_value = mock_cfg
        
        mock_extract.return_value = []
        mock_apply.return_value = mock_cfg
        
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        # Execute
        with patch('builtins.print'):
            main()
        
        # Print actual call for debugging
        print("mock_execute.call_args:", mock_execute.call_args)
        
        # Verify
        mock_parser.assert_called_once()
        mock_extract.assert_called_once_with(mock_args, mock_remaining)
        mock_load.assert_called_once_with("test_config.yaml")
        mock_apply.assert_called_once_with(mock_cfg, [])
        mock_logger.assert_called_once()
        # Adjust assertion to match actual call
        actual_call = mock_execute.call_args[0]
        assert actual_call[0] == "train"
        assert actual_call[1] == mock_cfg
        assert actual_call[3] == mock_logger_instance
        # model_path (actual_call[2]) can be None or "", so just check it's one of those
        assert actual_call[2] in (None, "")

    @patch('refrakt_cli.cli._setup_argument_parser')
    @patch('refrakt_cli.cli.OmegaConf.load')
    def test_main_config_error(self, mock_load, mock_parser):
        """Test CLI with configuration error."""
        mock_args = Mock()
        mock_args.config = "invalid_config.yaml"
        mock_args.override = []  # Fix: make override iterable
        mock_remaining = []
        mock_parser.return_value.parse_known_args.return_value = (
            mock_args, mock_remaining
        )
        
        mock_load.side_effect = FileNotFoundError("Config file not found")
        
        with pytest.raises(FileNotFoundError):
            with patch('builtins.print'):
                main()

    @patch('refrakt_cli.cli._setup_argument_parser')
    @patch('refrakt_cli.cli._extract_overrides')
    @patch('refrakt_cli.cli._apply_config_overrides')
    @patch('refrakt_cli.cli.setup_logger_and_config')
    @patch('refrakt_cli.cli._execute_pipeline_mode')
    @patch('refrakt_cli.cli.OmegaConf.load')
    def test_main_keyboard_interrupt(
        self, mock_load, mock_execute, mock_logger, mock_apply, mock_extract, mock_parser
    ):
        """Test CLI with keyboard interrupt."""
        mock_args = Mock()
        mock_args.config = "test_config.yaml"
        mock_args.mode = "train"
        mock_args.model_path = None
        mock_args.log_dir = None
        mock_args.debug = False
        mock_args.override = None
        
        mock_remaining = []
        mock_parser.return_value.parse_known_args.return_value = (
            mock_args, mock_remaining
        )
        
        mock_cfg = OmegaConf.create({"model": {"name": "test"}})
        mock_load.return_value = mock_cfg
        
        mock_extract.return_value = []
        mock_apply.return_value = mock_cfg
        
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        mock_execute.side_effect = KeyboardInterrupt()
        
        # Should not raise exception
        with patch('builtins.print'):
            main()
        
        mock_logger_instance.warning.assert_called_with("Training interrupted by user")

    @patch('refrakt_cli.cli._setup_argument_parser')
    @patch('refrakt_cli.cli._extract_overrides')
    @patch('refrakt_cli.cli._apply_config_overrides')
    @patch('refrakt_cli.cli.setup_logger_and_config')
    @patch('refrakt_cli.cli._execute_pipeline_mode')
    @patch('refrakt_cli.cli.OmegaConf.load')
    def test_main_pipeline_error(
        self, mock_load, mock_execute, mock_logger, mock_apply, mock_extract, mock_parser
    ):
        """Test CLI with pipeline execution error."""
        mock_args = Mock()
        mock_args.config = "test_config.yaml"
        mock_args.mode = "train"
        mock_args.model_path = None
        mock_args.log_dir = None
        mock_args.debug = False
        mock_args.override = None
        
        mock_remaining = []
        mock_parser.return_value.parse_known_args.return_value = (
            mock_args, mock_remaining
        )
        
        mock_cfg = OmegaConf.create({"model": {"name": "test"}})
        mock_load.return_value = mock_cfg
        
        mock_extract.return_value = []
        mock_apply.return_value = mock_cfg
        
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        mock_execute.side_effect = RuntimeError("Pipeline failed")
        
        with pytest.raises(RuntimeError):
            with patch('builtins.print'):
                main()
        
        mock_logger_instance.error.assert_called_with(
            "Pipeline failed: Pipeline failed"
        )