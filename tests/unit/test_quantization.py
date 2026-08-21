"""
Unit tests for ViForge AWQQuantizer post-training quantization engine.
"""

import json
from pathlib import Path
import pytest
from viforge.artifacts.quantization import AWQQuantizer


def test_awq_quantizer_validation(tmp_path: Path):
    """Test that invalid bits or group_size raise ValueError."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_dir = tmp_path / "quant_out"

    with pytest.raises(ValueError, match="Unsupported bit precision"):
        AWQQuantizer.quantize(model_path=model_dir, output_dir=out_dir, bits=2)

    with pytest.raises(ValueError, match="Unsupported group size"):
        AWQQuantizer.quantize(model_path=model_dir, output_dir=out_dir, bits=4, group_size=256)


def test_awq_quantizer_metadata_export(tmp_path: Path):
    """Test standard fallback metadata export when AutoAWQ is not executing on GPU."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_dir = tmp_path / "quant_out"

    res = AWQQuantizer.quantize(
        model_path=model_dir,
        output_dir=out_dir,
        bits=4,
        group_size=128,
        zero_point=True,
    )

    assert res["bits"] == 4
    assert res["group_size"] == 128
    assert Path(res["config_path"]).exists()
    assert Path(res["weights_path"]).exists()

    with open(res["config_path"], "r", encoding="utf-8") as f:
        config = json.load(f)
    assert config["quant_method"] == "awq"
    assert config["bits"] == 4
    assert config["group_size"] == 128
