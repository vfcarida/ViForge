"""
Integration tests for ViForge Synthetic Pipeline, Manifest Management, and Adapter Merging.
"""

import tempfile
from pathlib import Path
from viforge.methods.synthetic import SyntheticDataPipeline
from viforge.datasets.manifest import ManifestManager


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


def test_dataset_manifest_integrity():
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_file = Path(tmpdir) / "data.jsonl"
        sample_file.write_text('{"text": "def test(): pass"}\n', encoding="utf-8")

        manifest = ManifestManager.create_manifest(
            dataset_id="test_ds",
            file_path=sample_file,
            source_url="https://local.test/data.jsonl",
        )
        assert ManifestManager.verify_integrity(manifest, sample_file) is True
