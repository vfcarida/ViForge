"""
Unit tests for GGUF Exporter, Ollama Modelfile Generation, and AWQ Quantization.
"""

import tempfile
from pathlib import Path
import pytest
from viforge.artifacts.gguf import GGUFExporter
from viforge.artifacts.quantization import AWQQuantizer


def test_gguf_export_and_modelfile():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "test_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        export_dir = Path(tmpdir) / "exports" / "gguf"

        res = GGUFExporter.export(
            merged_model_dir=model_dir,
            output_gguf_dir=export_dir,
            quant_type="Q4_K_M",
            generate_ollama=True,
            system_prompt="Test System Prompt",
        )

        assert Path(res["gguf_path"]).exists()
        assert Path(res["modelfile_path"]).exists()
        assert res["quant_type"] == "Q4_K_M"

        # Verify Modelfile content
        with open(res["modelfile_path"], "r", encoding="utf-8") as f:
            content = f.read()
            assert "FROM " in content
            assert "Test System Prompt" in content
            assert "PARAMETER temperature 0.0" in content


def test_gguf_invalid_quant():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "test_model"
        export_dir = Path(tmpdir) / "exports"

        with pytest.raises(ValueError, match="Unsupported quantization"):
            GGUFExporter.export(
                merged_model_dir=model_dir,
                output_gguf_dir=export_dir,
                quant_type="INVALID_QUANT_TYPE",
            )


def test_awq_quantizer():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "test_model"
        export_dir = Path(tmpdir) / "exports" / "awq"

        res = AWQQuantizer.quantize(
            model_path=model_dir,
            output_dir=export_dir,
            bits=4,
            group_size=128,
        )

        assert Path(res["config_path"]).exists()
        assert res["bits"] == 4
        assert res["group_size"] == 128
