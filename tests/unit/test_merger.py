"""
Unit tests for ViForge AdapterMerger multi-adapter model merging.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from viforge.artifacts.merger import AdapterMerger


def test_merger_empty_adapters_raises(tmp_path: Path):
    """Test that merging with an empty adapters list raises ValueError."""
    out_dir = tmp_path / "merged_out"
    with pytest.raises(ValueError, match="At least one adapter must be provided"):
        AdapterMerger.merge_multiple_adapters(
            base_model_id="mock/base-model",
            adapters=[],
            output_dir=out_dir,
        )


def test_merger_single_adapter_mocked(tmp_path: Path):
    """Test single adapter merging workflow with mocked backend."""
    out_dir = tmp_path / "single_out"
    dummy_adapter = tmp_path / "adapter_dummy"
    dummy_adapter.mkdir()

    mock_base = MagicMock()
    mock_peft = MagicMock()
    mock_merged = MagicMock()
    mock_peft.merge_and_unload.return_value = mock_merged

    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
        patch("peft.PeftModel.from_pretrained", return_value=mock_peft),
    ):
        res = AdapterMerger.merge_and_export(
            base_model_id="mock/base-model",
            adapter_path=dummy_adapter,
            output_dir=out_dir,
        )
        assert res == out_dir
        assert mock_peft.merge_and_unload.called
        assert mock_merged.save_pretrained.called


def test_merger_multiple_adapters_mocked(tmp_path: Path):
    """Test multi-adapter weighted linear combination merging."""
    out_dir = tmp_path / "multi_out"
    ad1 = tmp_path / "ad1"
    ad2 = tmp_path / "ad2"
    ad1.mkdir()
    ad2.mkdir()

    mock_base = MagicMock()
    mock_peft = MagicMock()
    mock_merged = MagicMock()
    mock_peft.merge_and_unload.return_value = mock_merged

    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_base),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
        patch("peft.PeftModel.from_pretrained", return_value=mock_peft),
    ):
        res = AdapterMerger.merge_multiple_adapters(
            base_model_id="mock/base-model",
            adapters=[(ad1, 0.7), (ad2, 0.3)],
            output_dir=out_dir,
            combination_type="linear",
        )
        assert res == out_dir
        assert mock_peft.load_adapter.called
        assert mock_peft.add_weighted_adapter.called
        assert mock_peft.set_adapter.called
        assert mock_peft.merge_and_unload.called
        assert mock_merged.save_pretrained.called
