"""
Unit tests for chunked streaming ingestion and processing in ViForge.
"""

import json
from pathlib import Path
import pytest
from viforge.config.schemas import DatasetConfig
from viforge.datasets.adapters import DatasetAdapter
from viforge.datasets.preparer import DatasetPreparer


@pytest.fixture
def sample_jsonl_dataset(tmp_path: Path) -> Path:
    jsonl_path = tmp_path / "stream_data.jsonl"
    records = [
        {
            "instruction": f"Implement helper algorithm {i} in Python.",
            "output": f"def algo_{i}():\n    return {i} * 2\n",
        }
        for i in range(15)
    ]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return jsonl_path


def test_dataset_adapter_stream_local_records_jsonl(sample_jsonl_dataset: Path):
    chunks = list(DatasetAdapter.stream_local_records(sample_jsonl_dataset, chunk_size=4))
    assert len(chunks) == 4
    assert len(chunks[0]) == 4
    assert len(chunks[1]) == 4
    assert len(chunks[2]) == 4
    assert len(chunks[3]) == 3


def test_dataset_preparer_prepare_chunked(sample_jsonl_dataset: Path):
    cfg = DatasetConfig(
        domain="software_engineering",
        source=str(sample_jsonl_dataset),
        format="sft",
        chunk_size=5,
        dedup=True,
        quality_filter=False,
        decontaminate=False,
        scan_secrets=False,
        scan_pii=False,
    )
    preparer = DatasetPreparer(cfg)
    chunks = list(preparer.prepare_chunked())
    assert len(chunks) == 3
    for chunk in chunks:
        assert isinstance(chunk, list)
        for rec in chunk:
            assert "messages" in rec
            assert len(rec["messages"]) == 2


def test_dataset_preparer_save_chunked_jsonl(sample_jsonl_dataset: Path, tmp_path: Path):
    cfg = DatasetConfig(
        domain="software_engineering",
        source=str(sample_jsonl_dataset),
        format="sft",
        chunk_size=5,
        quality_filter=False,
        decontaminate=False,
    )
    preparer = DatasetPreparer(cfg)
    out_file = tmp_path / "output_streamed.jsonl"
    saved_path = preparer.save(out_file)
    assert saved_path.exists()

    # Verify line count
    lines = [line for line in open(saved_path, "r", encoding="utf-8") if line.strip()]
    assert len(lines) == 15
