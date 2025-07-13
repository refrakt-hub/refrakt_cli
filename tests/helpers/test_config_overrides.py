"""
Tests for configuration override handling.
"""

from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from refrakt_cli.helpers.config_overrides import (
    apply_config_overrides,
    extract_overrides,
    extract_runtime_config,
    setup_logging_config,
)


class TestConfigOverrides:
    """Test cases for configuration override functionality."""

    def test_extract_overrides_with_explicit_flags(self):
        """Test extracting overrides from explicit --override flags."""
        mock_args = Mock()
        mock_args.override = ["model.name=ResNet", "optimizer.lr=0.001"]
        
        remaining = []
        
        with patch(
            'refrakt_cli.helpers.config_overrides.extract_overrides_from_args'
        ) as mock_extract:
            mock_extract.return_value = ([], [])
            
            result = extract_overrides(mock_args, remaining)
            
            assert result == ["model.name=ResNet", "optimizer.lr=0.001"]
            mock_extract.assert_called_once_with(remaining)

    def test_extract_overrides_with_positional(self):
        """Test extracting overrides from positional arguments."""
        mock_args = Mock()
        mock_args.override = None
        
        remaining = ["model.name=ResNet", "optimizer.lr=0.001"]
        
        with patch('refrakt_cli.helpers.config_overrides.extract_overrides_from_args') as mock_extract:
            mock_extract.return_value = (["model.name=ResNet", "optimizer.lr=0.001"], [])
            
            result = extract_overrides(mock_args, remaining)
            
            assert result == ["model.name=ResNet", "optimizer.lr=0.001"]

    def test_extract_overrides_combined(self):
        """Test combining explicit and positional overrides."""
        mock_args = Mock()
        mock_args.override = ["model.name=ResNet"]
        
        remaining = ["optimizer.lr=0.001"]
        
        with patch('refrakt_cli.helpers.config_overrides.extract_overrides_from_args') as mock_extract:
            mock_extract.return_value = (["optimizer.lr=0.001"], [])
            
            result = extract_overrides(mock_args, remaining)
            
            assert result == ["model.name=ResNet", "optimizer.lr=0.001"]

    def test_apply_config_overrides(self):
        """Test applying overrides to configuration."""
        cfg = OmegaConf.create({
            "model": {"name": "default"},
            "optimizer": {"lr": 0.01}
        })
        overrides = ["model.name=ResNet", "optimizer.lr=0.001"]
        result = apply_config_overrides(cfg, overrides)
        assert result["model"]["name"] == "ResNet"
        assert result["optimizer"]["lr"] == 0.001

    def test_apply_config_overrides_empty(self):
        """Test applying empty overrides."""
        cfg = OmegaConf.create({"model": {"name": "default"}})
        
        result = apply_config_overrides(cfg, [])
        
        assert result == cfg

    def test_extract_runtime_config(self):
        """Test extracting runtime configuration."""
        cfg = OmegaConf.create({
            "runtime": {
                "mode": "train",
                "log_dir": "./logs",
                "debug": True
            },
            "model": {"name": "test"}
        })
        
        result = extract_runtime_config(cfg)
        
        assert result["mode"] == "train"
        assert result["log_dir"] == "./logs"
        assert result["debug"] is True

    def test_extract_runtime_config_missing(self):
        """Test extracting runtime config when missing."""
        cfg = OmegaConf.create({"model": {"name": "test"}})
        
        result = extract_runtime_config(cfg)
        
        assert result == {}

    def test_extract_runtime_config_invalid(self):
        """Test extracting runtime config with invalid type."""
        cfg = OmegaConf.create([1, 2, 3])  # ListConfig instead of DictConfig
        
        with pytest.raises(TypeError):
            extract_runtime_config(cfg)

    def test_setup_logging_config_defaults(self):
        """Test logging config setup with defaults."""
        runtime_cfg = {}
        
        mode, log_dir, log_types, console, model_path, debug = setup_logging_config(runtime_cfg)
        
        assert mode == "train"
        assert log_dir == "./logs"
        assert log_types == []
        assert console is True
        assert model_path is None
        assert debug is False

    def test_setup_logging_config_custom(self):
        """Test logging config setup with custom values."""
        runtime_cfg = {
            "mode": "test",
            "log_dir": "/custom/logs",
            "log_type": ["file", "console"],
            "console": False,
            "model_path": "/path/to/model.pth",
            "debug": True
        }
        
        mode, log_dir, log_types, console, model_path, debug = setup_logging_config(runtime_cfg)
        
        assert mode == "test"
        assert log_dir == "/custom/logs"
        assert log_types == ["file", "console"]
        assert console is False
        assert model_path == "/path/to/model.pth"
        assert debug is True

    def test_setup_logging_config_with_override(self):
        """Test logging config with command line override."""
        runtime_cfg = {"log_dir": "./default/logs"}
        args_log_dir = "/override/logs"
        
        mode, log_dir, log_types, console, model_path, debug = setup_logging_config(runtime_cfg, args_log_dir)
        
        assert log_dir == "/override/logs"

    def test_setup_logging_config_log_types_string(self):
        """Test logging config with string log_type."""
        runtime_cfg = {"log_type": "file"}
        
        mode, log_dir, log_types, console, model_path, debug = setup_logging_config(runtime_cfg)
        
        assert log_types == ["file"]

    def test_setup_logging_config_log_types_none(self):
        """Test logging config with None log_type."""
        runtime_cfg = {"log_type": None}
        
        mode, log_dir, log_types, console, model_path, debug = setup_logging_config(runtime_cfg)
        
        assert log_types == []