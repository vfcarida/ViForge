"""
Contract tests for ViForge Plugin Interfaces (TrainingMethods, EvaluationSuites, InferenceBackends).
"""

from viforge.methods.base import method_registry, BaseTrainingMethod
from viforge.evaluation.suites import evaluator_registry
from viforge.inference.backends import backend_registry, BaseInferenceBackend


def test_training_methods_registered():
    methods = method_registry.list_all()
    assert "sft" in methods
    assert "qlora" in methods
    assert "lora" in methods
    assert "cpt" in methods
    assert "dpo" in methods
    assert "grpo" in methods

    sft_inst = method_registry.get("sft")
    assert isinstance(sft_inst, BaseTrainingMethod)
    assert sft_inst.method_name == "sft"


def test_evaluator_suites_registered():
    evals = evaluator_registry.list_all()
    assert "humaneval_plus" in evals
    assert "swe_bench_lite" in evals
    assert "mmlu_pro" in evals

    suite = evaluator_registry.get("humaneval_plus")
    assert hasattr(suite, "evaluate")
    assert suite.name == "humaneval_plus"


def test_inference_backends_registered():
    mock_backend = backend_registry.get("mock")
    assert isinstance(mock_backend, BaseInferenceBackend)
