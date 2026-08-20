"""
Unit tests for Real Benchmark Dataset Loading, Retention Gold Answer Verification, and Degradation Checks.
"""

from pathlib import Path

import pytest

from viforge.config.schemas import (
    BenchmarkItemConfig,
    BenchmarkResult,
    EvaluationConfig,
    SamplingParams,
)
from viforge.evaluation.harness import EvaluationHarness, check_retention
from viforge.evaluation.suites.humaneval_plus import HumanEvalPlusSuite
from viforge.evaluation.suites.retention import (
    ARCChallengeRetentionSuite,
    GSM8KRetentionSuite,
    MMLUProRetentionSuite,
)
from viforge.evaluation.suites.swe_bench import SWEBenchSuite
from viforge.inference.backends import MockInferenceBackend


@pytest.mark.unit
def test_offline_fallback_problems():
    he = HumanEvalPlusSuite()
    he_probs = he.load_problems(limit=2)
    assert len(he_probs) == 2
    assert "prompt" in he_probs[0]
    assert "test" in he_probs[0]

    swe = SWEBenchSuite()
    swe_probs = swe.load_problems(limit=2)
    assert len(swe_probs) == 2
    assert "problem_statement" in swe_probs[0]

    gsm = GSM8KRetentionSuite()
    gsm_probs = gsm.load_problems(limit=2)
    assert len(gsm_probs) == 2
    assert "answer" in gsm_probs[0]

    arc = ARCChallengeRetentionSuite()
    arc_probs = arc.load_problems(limit=1)
    assert len(arc_probs) == 1
    assert "answer" in arc_probs[0]

    mmlu = MMLUProRetentionSuite()
    mmlu_probs = mmlu.load_problems(limit=2)
    assert len(mmlu_probs) == 2
    assert "answer" in mmlu_probs[0]


@pytest.mark.unit
def test_gsm8k_answer_extraction():
    gsm = GSM8KRetentionSuite()

    assert gsm._extract_answer("Step by step reasoning... #### 42") == "42"
    assert gsm._extract_answer("Total revenue was #### 1,250") == "1250"
    assert gsm._extract_answer("Therefore, the answer is 15.") == "15"

    assert gsm.evaluate_completion("Step 1... #### 42", "42") is True
    assert gsm.evaluate_completion("Calculated $1,250 dollars.", "1250") is True
    assert gsm.evaluate_completion("The answer is 99", "42") is False


@pytest.mark.unit
def test_arc_and_mmlu_answer_extraction():
    arc = ARCChallengeRetentionSuite()
    assert arc._extract_answer("Based on this, choice (B) is correct.") == "B"
    assert arc._extract_answer("Answer: C") == "C"
    assert arc._extract_answer("Option 1") == "A"
    assert arc.evaluate_completion("The answer is (A)", "A") is True
    assert arc.evaluate_completion("The answer is (B)", "A") is False

    mmlu = MMLUProRetentionSuite()
    assert mmlu._extract_answer("The best option is (G)") == "G"
    assert mmlu._extract_answer("Answer: J") == "J"
    assert mmlu.evaluate_completion("Option (D) is clearly correct", "D") is True
    assert mmlu.evaluate_completion("Option (A)", "D") is False


@pytest.mark.unit
def test_retention_degradation_check():
    base_results = [
        BenchmarkResult(
            benchmark_name="gsm8k",
            pass_at_k={"acc": 0.80},
            total_problems=100,
            passed_problems=80,
            failed_problems=20,
            execution_time_seconds=10.0,
        ),
        BenchmarkResult(
            benchmark_name="arc_challenge",
            pass_at_k={"acc": 0.70},
            total_problems=100,
            passed_problems=70,
            failed_problems=30,
            execution_time_seconds=10.0,
        ),
    ]

    # Mild acceptable drop (gsm8k: 0.80 -> 0.78, arc: 0.70 -> 0.68) < 10%
    spec_results_mild = [
        BenchmarkResult(
            benchmark_name="gsm8k",
            pass_at_k={"acc": 0.78},
            total_problems=100,
            passed_problems=78,
            failed_problems=22,
            execution_time_seconds=10.0,
        ),
        BenchmarkResult(
            benchmark_name="arc_challenge",
            pass_at_k={"acc": 0.68},
            total_problems=100,
            passed_problems=68,
            failed_problems=32,
            execution_time_seconds=10.0,
        ),
    ]
    ret_res_mild = check_retention(base_results, spec_results_mild, threshold=0.10)
    assert len(ret_res_mild) == 2
    assert all(r.within_threshold for r in ret_res_mild)

    # Severe unacceptable drop (gsm8k: 0.80 -> 0.50, > 30% drop)
    spec_results_severe = [
        BenchmarkResult(
            benchmark_name="gsm8k",
            pass_at_k={"acc": 0.50},
            total_problems=100,
            passed_problems=50,
            failed_problems=50,
            execution_time_seconds=10.0,
        ),
        BenchmarkResult(
            benchmark_name="arc_challenge",
            pass_at_k={"acc": 0.70},
            total_problems=100,
            passed_problems=70,
            failed_problems=30,
            execution_time_seconds=10.0,
        ),
    ]
    ret_res_severe = check_retention(base_results, spec_results_severe, threshold=0.10)
    gsm_res = next(r for r in ret_res_severe if r.benchmark == "gsm8k")
    assert gsm_res.within_threshold is False
    assert gsm_res.degradation > 0.10


@pytest.mark.unit
def test_evaluation_config_and_harness(tmp_path: Path):
    config = EvaluationConfig(
        sampling=SamplingParams(max_new_tokens=128),
        domain_benchmarks=["humaneval_plus", "swe_bench_lite"],
        retention_benchmarks=["gsm8k", "arc_challenge"],
        max_problems=2,
    )
    assert len(config.domain_benchmarks) == 2
    assert len(config.retention_benchmarks) == 2
    assert config.max_problems == 2

    harness = EvaluationHarness(config)
    mock_backend = MockInferenceBackend()
    mock_backend.load_model("gpt2")

    domain_res, ret_res = harness.run_all(mock_backend, output_dir=tmp_path / "eval")
    assert len(domain_res) == 2
    assert len(ret_res) == 2
    assert all(r.total_problems <= 2 for r in domain_res + ret_res)
