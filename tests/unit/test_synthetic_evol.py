"""
Unit tests for Evol-Instruct Prompt Evolution and Self-Play Preference Pair Curating.
"""

from pathlib import Path
import pytest

from viforge.methods.synthetic import EvolStrategy, SyntheticDataPipeline
from viforge.preprocessing.contamination import ContaminationDetector


@pytest.mark.unit
def test_evol_prompt_mutation():
    original = "Write a function to search a binary tree."
    deepened = SyntheticDataPipeline.mutate_prompt(original, EvolStrategy.DEEPEN_CONSTRAINTS)
    assert original in deepened
    assert "production constraints" in deepened

    broadened = SyntheticDataPipeline.mutate_prompt(original, EvolStrategy.BROADEN_DOMAIN)
    assert "enterprise" in broadened or "distributed" in broadened

    reasoning = SyntheticDataPipeline.mutate_prompt(original, EvolStrategy.ADD_REASONING_STEPS)
    assert "reasoning" in reasoning.lower()

    concretized = SyntheticDataPipeline.mutate_prompt(original, EvolStrategy.CONCRETIZE)
    assert "bug report" in concretized.lower() or "scaffold" in concretized.lower()


@pytest.mark.unit
def test_self_play_preference_pair_generation():
    pipeline = SyntheticDataPipeline(min_code_lines=2)

    prompt = "Write a Python function to compute factorial."
    valid_clean = (
        "```python\ndef factorial(n: int) -> int:\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n```"
    )
    invalid_syntax = (
        "```python\ndef factorial(n):\n"
        "    return ??? syntax error\n```"
    )

    # Valid vs Invalid
    pair = pipeline.create_preference_pair(prompt, valid_clean, invalid_syntax)
    assert pair is not None
    assert pair["chosen"] == valid_clean
    assert pair["rejected"] == invalid_syntax
    assert pair["margin"] > 0.5
    assert pair["provenance"]["length_normalization_applied"] is True


@pytest.mark.unit
def test_self_play_both_invalid_discarded():
    pipeline = SyntheticDataPipeline(min_code_lines=2)
    prompt = "Test prompt"
    res = pipeline.create_preference_pair(prompt, "invalid code {[[", "also invalid ((((")
    assert res is None


@pytest.mark.unit
def test_export_synthetic_dataset_and_manifest(tmp_path: Path):
    pipeline = SyntheticDataPipeline()
    samples = [
        {"instruction": "Write add", "response": "```python\ndef add(a, b):\n    return a + b\n```"},
        {"instruction": "Write sub", "response": "```python\ndef sub(a, b):\n    return a - b\n```"},
    ]

    out_file = tmp_path / "synthetic_train.jsonl"
    saved_path, manifest = pipeline.export_to_dataset(
        examples=samples,
        output_path=out_file,
        dataset_id="unit-test-synthetic",
    )

    assert saved_path.exists()
    assert (tmp_path / "synthetic_train.manifest.json").exists()
    assert manifest.dataset_id == "unit-test-synthetic"
    assert manifest.num_samples == 2
    assert manifest.contamination_checked is True
    assert len(manifest.content_sha256) == 64


@pytest.mark.unit
def test_synthetic_contamination_rejected():
    detector = ContaminationDetector(ngram_size=4, max_allowed_overlap=0.10)
    detector.register_benchmark_corpus(
        "humaneval",
        ["def benchmark_secret_function(x): return x * 42"],
    )

    pipeline = SyntheticDataPipeline(min_code_lines=1, contamination_detector=detector)

    clean_example = {
        "instruction": "Write a clean function",
        "response": "```python\ndef my_func(a):\n    return a + 1\n```",
    }
    clean_res = pipeline.verify_example(clean_example)
    assert clean_res["is_accepted"] is True
    assert clean_res["is_contaminated"] is False

    contaminated_example = {
        "instruction": "def benchmark_secret_function(x):",
        "response": "```python\ndef benchmark_secret_function(x):\n    return x * 42\n```",
    }
    contaminated_res = pipeline.verify_example(contaminated_example)
    assert contaminated_res["is_accepted"] is False
    assert contaminated_res["is_contaminated"] is True
    assert "humaneval" in contaminated_res["contamination_info"]

