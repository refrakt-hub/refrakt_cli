"""
Tests for pipeline orchestration.
"""
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from refrakt_cli.helpers.pipeline_orchestrator import (
    execute_full_pipeline,
    execute_pipeline_mode,
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
        logger.info.assert_called_with("🔮 Starting inference pipeline")

    def test_execute_pipeline_mode_inference_no_model_path(self):
        """Test inference mode without model path."""
        cfg = OmegaConf.create({"model": {"name": "test"}})
        logger = Mock()
        
        with pytest.raises(
            ValueError, match="model_path must be provided for inference mode"
        ):
            execute_pipeline_mode("inference", cfg, None, logger)

    @patch('refrakt_cli.helpers.pipeline_orchestrator.execute_full_pipeline')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.datetime')
    def test_execute_pipeline_mode_pipeline(self, mock_datetime, mock_full_pipeline):
        """Test executing full pipeline mode."""
        cfg = OmegaConf.create({"model": {"name": "test"}})
        logger = Mock()
        
        # Mock the datetime to return a fixed experiment ID
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000"
        
        execute_pipeline_mode("pipeline", cfg, None, logger)
        
        mock_full_pipeline.assert_called_once_with(cfg, logger, "20250101_120000")
        logger.info.assert_called_with(
            "🔁 Starting full pipeline (train → test → inference)"
        )

    def test_execute_pipeline_mode_invalid(self):
        """Test executing invalid pipeline mode."""
        cfg = OmegaConf.create({"model": {"name": "test"}})
        logger = Mock()
        
        with pytest.raises(ValueError, match="Invalid mode: invalid"):
            execute_pipeline_mode("invalid", cfg, None, logger)

    @patch('refrakt_cli.helpers.pipeline_orchestrator.train')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.test')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.inference')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.datetime')
    def test_execute_full_pipeline(
        self, mock_datetime, mock_inference, mock_test, mock_train
    ):
        """Test executing full pipeline."""
        cfg = OmegaConf.create({
            "model": {"name": "autoencoder", "params": {"variant": "complex"}},
            "dataset": {"params": {"path": "data.zip"}},
            "trainer": {"params": {"save_dir": "/save/dir"}}
        })
        logger = Mock()
        
        # Mock the datetime to return a fixed experiment ID
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000"
        
        execute_full_pipeline(cfg, logger, "20250101_120000")
        
        # Verify all pipeline phases were called
        mock_train.assert_called_once_with(cfg, logger=logger, experiment_id="20250101_120000")
        mock_test.assert_called_once_with(
            cfg, model_path="./checkpoints/autoencoder_complex_custom_20250101_120000/weights/autoencoder_complex_custom.pth", logger=logger, experiment_id="20250101_120000"
        )
        mock_inference.assert_called_once_with(
            cfg, model_path="./checkpoints/autoencoder_complex_custom_20250101_120000/weights/autoencoder_complex_custom.pth", logger=logger, experiment_id="20250101_120000"
        )
        
        # Verify logging
        assert logger.info.call_count == 4  # Including experiment ID log
        logger.info.assert_any_call("🔬 Experiment ID: 20250101_120000")
        logger.info.assert_any_call("🚀 Training phase started")
        logger.info.assert_any_call("🧪 Testing phase started")
        logger.info.assert_any_call("🔮 Inference phase started")

    @patch('refrakt_cli.helpers.pipeline_orchestrator.train')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.test')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.inference')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.datetime')
    def test_execute_full_pipeline_resnet(
        self, mock_datetime, mock_inference, mock_test, mock_train
    ):
        """Test executing full pipeline with ResNet model."""
        cfg = OmegaConf.create({
            "model": {"name": "resnet"},
            "trainer": {"params": {"save_dir": "/save/dir"}}
        })
        logger = Mock()
        
        # Mock the datetime to return a fixed experiment ID
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000"
        
        execute_full_pipeline(cfg, logger, "20250101_120000")
        
        # Verify the model path construction
        mock_test.assert_called_once_with(
            cfg, model_path="./checkpoints/resnet_20250101_120000/weights/resnet.pth", logger=logger, experiment_id="20250101_120000"
        )

    @patch('refrakt_cli.helpers.pipeline_orchestrator.train')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.test')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.inference')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.datetime')
    def test_execute_full_pipeline_autoencoder_simple(
        self, mock_datetime, mock_inference, mock_test, mock_train
    ):
        """Test executing full pipeline with simple autoencoder."""
        cfg = OmegaConf.create({
            "model": {"name": "autoencoder", "params": {}},  # No variant specified
            "trainer": {"params": {"save_dir": "/save/dir"}}
        })
        logger = Mock()
        
        # Mock the datetime to return a fixed experiment ID
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000"
        
        execute_full_pipeline(cfg, logger, "20250101_120000")
        
        # Verify the model path construction
        mock_test.assert_called_once_with(
            cfg, model_path="./checkpoints/autoencoder_simple_20250101_120000/weights/autoencoder_simple.pth", logger=logger, experiment_id="20250101_120000"
        )

    @patch('refrakt_cli.helpers.pipeline_orchestrator.train')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.test')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.inference')
    @patch('refrakt_cli.helpers.pipeline_orchestrator.datetime')
    def test_execute_full_pipeline_custom_dataset(
        self, mock_datetime, mock_inference, mock_test, mock_train
    ):
        """Test executing full pipeline with custom dataset."""
        cfg = OmegaConf.create({
            "model": {"name": "resnet"},
            "dataset": {"params": {"zip_path": "custom_data.zip"}},
            "trainer": {"params": {"save_dir": "/save/dir"}}
        })
        logger = Mock()
        
        # Mock the datetime to return a fixed experiment ID
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000"
        
        execute_full_pipeline(cfg, logger, "20250101_120000")
        
        # Verify the model path construction
        mock_test.assert_called_once_with(
            cfg, model_path="./checkpoints/resnet_custom_20250101_120000/weights/resnet_custom.pth", logger=logger, experiment_id="20250101_120000"
        )