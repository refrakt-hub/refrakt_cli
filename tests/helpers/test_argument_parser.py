"""
Tests for CLI argument parser.
"""

import argparse
from unittest.mock import Mock, patch

import pytest

from refrakt_cli.helpers.argument_parser import parse_args, setup_argument_parser


class TestArgumentParser:
    """Test cases for argument parser functionality."""

    def test_setup_argument_parser(self):
        """Test argument parser setup."""
        parser = setup_argument_parser()
        
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.description == "Refrakt CLI - ML/DL Training Framework"

    def test_required_arguments(self):
        """Test that required arguments are present."""
        parser = setup_argument_parser()
        
        # Check that --config is required
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_optional_arguments(self):
        """Test optional arguments."""
        parser = setup_argument_parser()
        
        args = parser.parse_args([
            "--config", "test.yaml",
            "--mode", "test",
            "--model-path", "model.pth",
            "--log-dir", "/tmp/logs",
            "--debug",
            "--override", "model.name=ResNet", "optimizer.lr=0.001"
        ])
        
        assert args.config == "test.yaml"
        assert args.mode == "test"
        assert args.model_path == "model.pth"
        assert args.log_dir == "/tmp/logs"
        assert args.debug is True
        assert args.override == ["model.name=ResNet", "optimizer.lr=0.001"]

    def test_default_values(self):
        """Test default argument values."""
        parser = setup_argument_parser()
        
        args = parser.parse_args(["--config", "test.yaml"])
        
        assert args.mode == "train"
        assert args.model_path is None
        assert args.log_dir is None
        assert args.debug is False
        assert args.override is None

    def test_mode_choices(self):
        """Test mode argument choices."""
        parser = setup_argument_parser()
        
        valid_modes = ["train", "test", "inference", "pipeline"]
        for mode in valid_modes:
            args = parser.parse_args(["--config", "test.yaml", "--mode", mode])
            assert args.mode == mode
        
        # Test invalid mode
        with pytest.raises(SystemExit):
            parser.parse_args(["--config", "test.yaml", "--mode", "invalid"])

    @patch('refrakt_cli.helpers.argument_parser.setup_argument_parser')
    def test_parse_args(self, mock_setup):
        """Test parse_args function."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.config = "test.yaml"
        mock_args.mode = "train"
        mock_remaining = ["override1=value1"]
        
        mock_parser.parse_known_args.return_value = (mock_args, mock_remaining)
        mock_setup.return_value = mock_parser
        
        args, remaining = parse_args()
        
        assert args == mock_args
        assert remaining == mock_remaining
        mock_parser.parse_known_args.assert_called_once()