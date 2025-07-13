"""
Tests for pipeline orchestration.
"""
import os 
import pytest
from unittest.mock import Mock, patch, MagicMock
from omegaconf import OmegaConf

from refrakt_cli.helpers.pipeline_orchestrator import (
    execute_pipeline_mode,
    execute_full_pipeline
)


class TestPipelineOrchestrator:
    """Test cases for pipeline orchestration functionality."""

    @patch('refrakt_cli.helpers.pipeline_orchestrator.train')
    def test_execute_pipeline_mode_train(self, mock_train):
        """Test executing training pipeline mode."""
        cfg = OmegaConf.create({"model": {"name": "test"}})
        logger = Mock()
        
        execute_pipeline_mode("train", cfg, None, logger)
        
        mock_train.assert_called_once_with(cfg, logger=logger)
        logger.info.assert_called_with("🚀 Starting training pipeline")

    @patch('refrakt_cli.helpers.pipeline_orchestrator.test')
    def test_execute_pipeline_mode_test(self, mock_test):
        """Test executing testing pipeline mode."""
        cfg = OmegaConf.create({"model": {"name": "test"}})
        logger = Mock()
        model_path = "/path/to/model.pth"
        
        execute_pipeline_mode("test", cfg, model_path, logger)
        
        mock_test.assert_called_once_with(cfg, model_path=model_path, logger=logger)
        logger.info.assert_called_with("🧪 Starting testing pipeline")

    @patch('refrakt_cli.helpers.pipeline_orchestrator.inference')
    def test_execute_pipeline_mode_inference(self, mock_inference):
        """Test executing inference pipeline mode."""
        cfg = OmegaConf.create({"model": {"name": "test"}})
        logger = Mock()
        model_path = "/path/to/model.pth"
        
        execute_pipeline_mode("inference", cfg, model_path, logger)
        
        mock_inference.assert_called_once_with(cfg, model_path, logger=logger)
        logger.info.assert_called_with("�� Starting inference pipeline")

    def test_execute_pipeline_mode_inference_no_model_path(self):
        """Test inference mode without model path."""
        cfg = OmegaConf.create({"model": {"name": "test"}})
        logger = Mock()
        
        with pytest.raises(ValueError, match="model_path must be provided for inference mode"):
            execute_pipeline_mode("inference", cfg, None, logger)

    @patch('refrakt_cli.helpers.pipeline_orchestrator.execute_full_pipeline')
    def test_execute_pipeline_mode_pipeline(self, mock_full_pipeline):
        """Test executing full pipeline mode."""
        cfg = OmegaConf.create({"model": {"name": "test"}})
        logger = Mock()
        
        execute_pipeline_mode("pipeline", cfg, None, logger)
        
        mock_full_pipeline.assert_called_once_with(cfg, logger)
        logger.info.assert_called_with("🔁 Starting full pipeline (train → test → inference)")

    def test_execute_pipeline_mode_invalid(self):
        """Test executing invalid pipeline mode."""
        cfg = OmegaConf.create({"model": {"name": "test"}})
        logger = Mock()
        
        with pytest.raises(ValueError, match="Invalid mode: invalid"):
            execute_pipeline_mode("invalid", cfg, None, logger)

    @patch('refrakt_cli.helpers.pipeline_orchestrator.train')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.test')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.inference')
    @patch('os.path.join')
    def test_execute_full_pipeline(self, mock_join, mock_inference, mock_test, mock_train):
        """Test executing full pipeline."""
        cfg = OmegaConf.create({
            "model": {"name": "autoencoder", "params": {"variant": "complex"}},
            "dataset": {"params": {"path": "data.zip"}},
            "trainer": {"params": {"save_dir": "/save/dir"}}
        })
        logger = Mock()
        
        mock_join.return_value = "/save/dir/autoencoder_complex_custom.pth"
        
        execute_full_pipeline(cfg, logger)
        
        # Verify all pipeline phases were called
        mock_train.assert_called_once_with(cfg, logger=logger)
        mock_test.assert_called_once_with(cfg, model_path="/save/dir/autoencoder_complex_custom.pth", logger=logger)
        mock_inference.assert_called_once_with(cfg, model_path="/save/dir/autoencoder_complex_custom.pth", logger=logger)
        
        # Verify logging
        assert logger.info.call_count == 3
        logger.info.assert_any_call("🚀 Training phase started")
        logger.info.assert_any_call("🧪 Testing phase started")
        logger.info.assert_any_call("🔮 Inference phase started")

    @patch('refrakt_cli.helpers.pipeline_orchestrator.train')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.test')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.inference')
    @patch('os.path.join')
    def test_execute_full_pipeline_resnet(self, mock_join, mock_inference, mock_test, mock_train):
        """Test executing full pipeline with ResNet model."""
        cfg = OmegaConf.create({
            "model": {"name": "resnet"},
            "trainer": {"params": {"save_dir": "/save/dir"}}
        })
        logger = Mock()
        
        mock_join.return_value = "/save/dir/resnet.pth"
        
        execute_full_pipeline(cfg, logger)
        
        mock_join.assert_called_once_with("/save/dir", "resnet.pth")

    @patch('refrakt_cli.helpers.pipeline_orchestrator.train')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.test')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.inference')
    @patch('os.path.join')
    def test_execute_full_pipeline_autoencoder_simple(self, mock_join, mock_inference, mock_test, mock_train):
        """Test executing full pipeline with simple autoencoder."""
        cfg = OmegaConf.create({
            "model": {"name": "autoencoder", "params": {}},  # No variant specified
            "trainer": {"params": {"save_dir": "/save/dir"}}
        })
        logger = Mock()
        
        mock_join.return_value = "/save/dir/autoencoder_simple.pth"
        
        execute_full_pipeline(cfg, logger)
        
        mock_join.assert_called_once_with("/save/dir", "autoencoder_simple.pth")

    @patch('refrakt_cli.helpers.pipeline_orchestrator.train')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.test')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.inference')
    @patch('os.path.join')
    def test_execute_full_pipeline_custom_dataset(self, mock_join, mock_inference, mock_test, mock_train):
        """Test executing full pipeline with custom dataset."""
        cfg = OmegaConf.create({
            "model": {"name": "resnet"},
            "dataset": {"params": {"zip_path": "custom_data.zip"}},
            "trainer": {"params": {"save_dir": "/save/dir"}}
        })
        logger = Mock()
        
        mock_join.return_value = "/save/dir/resnet_custom.pth"
        
        execute_full_pipeline(cfg, logger)
        
        mock_join.assert_called_once_with("/save/dir", "resnet_custom.pth")