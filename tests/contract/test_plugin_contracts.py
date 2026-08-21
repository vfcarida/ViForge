"""
Contract tests for ViForge Plugin Interfaces (TrainingMethods, EvaluationSuites, InferenceBackends).
"""

import pytest

from viforge.evaluation.suites import evaluator_registry
from viforge.inference.backends import BaseInferenceBackend, backend_registry
from viforge.methods.base import BaseTrainingMethod, method_registry


@pytest.mark.contract
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


@pytest.mark.contract
def test_evaluator_suites_registered():
    evals = evaluator_registry.list_all()
    assert "humaneval_plus" in evals
    assert "swe_bench_lite" in evals
    assert "mmlu_pro" in evals

    suite = evaluator_registry.get("humaneval_plus")
    assert hasattr(suite, "evaluate")
    assert suite.name == "humaneval_plus"


@pytest.mark.contract
def test_inference_backends_registered():
    mock_backend = backend_registry.get("mock")
    assert isinstance(mock_backend, BaseInferenceBackend)
    assert hasattr(mock_backend, "generate")
    assert hasattr(mock_backend, "load_model")


@pytest.mark.contract
def test_all_evaluation_suites_contract():
    required_suites = [
        "humaneval_plus",
        "swe_bench_lite",
        "mbpp_plus",
        "mmlu_pro",
        "gsm8k",
        "arc_challenge",
    ]
    registered = evaluator_registry.list_all()
    for req in required_suites:
        assert req in registered
        suite = evaluator_registry.get(req)
        assert suite.name == req
        assert hasattr(suite, "evaluate")
        assert hasattr(suite, "load_problems")
