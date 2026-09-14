"""
Unit tests for standard Python entry-point plugin discovery for methods and evaluators.
"""

from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch
import pytest

from viforge.config.schemas import StageMetrics, TrainingStageConfig
from viforge.methods.base import BaseTrainingMethod, MethodRegistry
from viforge.evaluation.suites import EvaluatorRegistry


class DummyExternalMethod(BaseTrainingMethod):
    @property
    def method_name(self) -> str:
        return "dummy_external"

    def prepare_model(self, model: Any, stage_config: TrainingStageConfig) -> Any:
        return model

    def execute_stage(
        self,
        model: Any,
        tokenizer: Any,
        train_data_path: Path,
        eval_data_path: Optional[Path],
        stage_config: TrainingStageConfig,
        output_dir: Path,
    ) -> StageMetrics:
        return StageMetrics(
            stage_id="test",
            method=self.method_name,
            training_loss=0.1,
            eval_loss=0.1,
            tokens_processed=100,
            tokens_per_second=100.0,
            peak_vram_gb=1.0,
            trainable_parameters=100,
            total_parameters=100,
            trainable_ratio_pct=100.0,
            wall_clock_seconds=1.0,
            estimated_stage_cost_usd=0.01,
        )


class DummyExternalEvaluator:
    @property
    def name(self) -> str:
        return "dummy_eval"

    def evaluate(self, inference_backend: Any, output_dir: Path, sampling_params: Any) -> Any:
        return {"score": 1.0}


@pytest.mark.unit
def test_method_entrypoint_discovery():
    mock_ep = MagicMock()
    mock_ep.name = "dummy_external"
    mock_ep.load.return_value = DummyExternalMethod

    mock_eps = MagicMock()
    mock_eps.select.return_value = [mock_ep]

    with patch("importlib.metadata.entry_points", return_value=mock_eps):
        registry = MethodRegistry()
        assert "dummy_external" in registry.list_all()
        inst = registry.get("dummy_external")
        assert isinstance(inst, DummyExternalMethod)
        assert inst.method_name == "dummy_external"


@pytest.mark.unit
def test_method_entrypoint_invalid_subclass_skipped():
    class NotAMethod:
        pass

    mock_ep = MagicMock()
    mock_ep.name = "invalid_plugin"
    mock_ep.load.return_value = NotAMethod

    mock_eps = MagicMock()
    mock_eps.select.return_value = [mock_ep]

    with patch("importlib.metadata.entry_points", return_value=mock_eps):
        registry = MethodRegistry()
        assert "invalid_plugin" not in registry.list_all()


@pytest.mark.unit
def test_evaluator_entrypoint_discovery():
    mock_ep = MagicMock()
    mock_ep.name = "dummy_eval"
    mock_ep.load.return_value = DummyExternalEvaluator

    mock_eps = MagicMock()
    mock_eps.select.return_value = [mock_ep]

    with patch("importlib.metadata.entry_points", return_value=mock_eps):
        registry = EvaluatorRegistry()
        assert "dummy_eval" in registry.list_all()
        evaluator = registry.get("dummy_eval")
        assert hasattr(evaluator, "evaluate")
