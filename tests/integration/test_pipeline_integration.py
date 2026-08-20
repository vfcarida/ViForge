"""
Integration tests for ViForge Synthetic Pipeline, Manifest Management, and Adapter Merging.
"""

from pathlib import Path

import pytest

from viforge.datasets.manifest import ManifestManager
from viforge.methods.synthetic import SyntheticDataPipeline


@pytest.mark.integration
def test_synthetic_data_pipeline_integration():
    pipeline = SyntheticDataPipeline()
    candidates = [
        {"prompt": "Write add", "response": "```python\ndef add(a, b):\n    return a + b\n```"},
        {"prompt": "Write broken", "response": "```python\ndef broken(:\n```"},
    ]
    accepted, stats = pipeline.process_and_filter_candidates(
        candidates, system_prompt="Generate clean python."
    )
    assert len(accepted) == 1
    assert stats["rejected_samples"] == 1
    assert "provenance" in accepted[0]


@pytest.mark.integration
def test_dataset_manifest_integrity(tmp_output_dir: Path):
    sample_file = tmp_output_dir / "data.jsonl"
    sample_file.write_text('{"text": "def test(): pass"}\n', encoding="utf-8")

    manifest = ManifestManager.create_manifest(
        dataset_id="test_ds",
        file_path=sample_file,
        source_url="https://local.test/data.jsonl",
    )
    assert ManifestManager.verify_integrity(manifest, sample_file) is True
