"""
Unit tests for GGUF Exporter, Ollama Modelfile Generation, AWQ Quantization,
and vLLM inference backend.
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from viforge.artifacts.gguf import GGUFExporter, _find_llama_cpp_root
from viforge.artifacts.quantization import AWQQuantizer
from viforge.config.schemas import SamplingParams
from viforge.inference.backends import (
    VLLMInferenceBackend,
    backend_registry,
)


# --------------------------------------------------------------------------- #
# GGUF Exporter - mock fallback (no llama.cpp)                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_gguf_export_and_modelfile(tmp_output_dir: Path):
    """Mock-path export (no llama.cpp) must produce a file + Modelfile."""
    model_dir = tmp_output_dir / "test_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    export_dir = tmp_output_dir / "exports" / "gguf"

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
    # Without llama.cpp the result is a stub
    assert res["is_real_gguf"] is False

    with open(res["modelfile_path"], "r", encoding="utf-8") as f:
        content = f.read()
        assert "FROM " in content
        assert "Test System Prompt" in content
        assert "PARAMETER temperature 0.0" in content


@pytest.mark.unit
def test_gguf_invalid_quant(tmp_output_dir: Path):
    model_dir = tmp_output_dir / "test_model"
    export_dir = tmp_output_dir / "exports"

    with pytest.raises(ValueError, match="Unsupported quantization"):
        GGUFExporter.export(
            merged_model_dir=model_dir,
            output_gguf_dir=export_dir,
            quant_type="INVALID_QUANT_TYPE",
        )


@pytest.mark.unit
def test_gguf_mock_stub_content(tmp_output_dir: Path):
    """Mock stub must contain the MOCK marker and source model dir."""
    model_dir = tmp_output_dir / "my_specialist"
    model_dir.mkdir(parents=True, exist_ok=True)
    export_dir = tmp_output_dir / "exports" / "stub"

    res = GGUFExporter.export(
        merged_model_dir=model_dir,
        output_gguf_dir=export_dir,
        quant_type="Q5_K_M",
        generate_ollama=False,
    )

    with open(res["gguf_path"], "r", encoding="utf-8") as f:
        content = f.read()
    assert "MOCK" in content or "GGUF_MAGIC" in content


@pytest.mark.unit
def test_gguf_real_export_with_mock_subprocess(tmp_output_dir: Path):
    """When llama.cpp root is set, the real pipeline must be called."""
    model_dir = tmp_output_dir / "hf_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    export_dir = tmp_output_dir / "exports" / "real"

    fake_llama_root = tmp_output_dir / "llama_cpp"
    fake_llama_root.mkdir(parents=True, exist_ok=True)
    (fake_llama_root / "convert_hf_to_gguf.py").write_text("# fake converter\n")

    # Simulate convert_hf_to_gguf.py creating the F16 gguf
    def fake_run(cmd, **kwargs):
        # Write a fake F16 GGUF file
        outfile = next((c for c in cmd if c.endswith(".gguf")), None)
        if outfile:
            Path(outfile).write_bytes(b"GGUF_FAKE_BINARY")
        return MagicMock(returncode=0)

    with (
        patch("viforge.artifacts.gguf._find_llama_cpp_root", return_value=fake_llama_root),
        patch("viforge.artifacts.gguf._find_quantize_binary", return_value=None),
        patch("viforge.artifacts.gguf.subprocess.run", side_effect=fake_run),
    ):
        res = GGUFExporter.export(
            merged_model_dir=model_dir,
            output_gguf_dir=export_dir,
            quant_type="F16",
            generate_ollama=False,
        )

    assert Path(res["gguf_path"]).exists()
    assert res["is_real_gguf"] is True


@pytest.mark.unit
def test_gguf_real_export_fallback_on_subprocess_error(tmp_output_dir: Path):
    """If subprocess fails, export must fall back to mock stub without crashing."""
    model_dir = tmp_output_dir / "hf_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    export_dir = tmp_output_dir / "exports" / "fallback"

    fake_llama_root = tmp_output_dir / "llama_cpp_broken"
    fake_llama_root.mkdir(parents=True, exist_ok=True)
    (fake_llama_root / "convert_hf_to_gguf.py").write_text("# fake\n")

    with (
        patch("viforge.artifacts.gguf._find_llama_cpp_root", return_value=fake_llama_root),
        patch(
            "viforge.artifacts.gguf.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "convert", stderr="error"),
        ),
    ):
        res = GGUFExporter.export(
            merged_model_dir=model_dir,
            output_gguf_dir=export_dir,
            quant_type="Q4_K_M",
            generate_ollama=False,
        )

    # Must fall back gracefully
    assert Path(res["gguf_path"]).exists()
    assert res["is_real_gguf"] is False


# --------------------------------------------------------------------------- #
# AWQ Quantizer                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_awq_quantizer(tmp_output_dir: Path):
    model_dir = tmp_output_dir / "test_model"
    export_dir = tmp_output_dir / "exports" / "awq"

    res = AWQQuantizer.quantize(
        model_path=model_dir,
        output_dir=export_dir,
        bits=4,
        group_size=128,
    )

    assert Path(res["config_path"]).exists()
    assert res["bits"] == 4
    assert res["group_size"] == 128


# --------------------------------------------------------------------------- #
# vLLM Backend                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_vllm_backend_registered():
    """vLLM backend must be registered in the global registry."""
    assert "vllm" in backend_registry._backends
    assert backend_registry._backends["vllm"] is VLLMInferenceBackend


@pytest.mark.unit
def test_vllm_backend_import_error_without_package():
    """VLLMInferenceBackend must raise ImportError when vllm not installed."""
    with patch.dict("sys.modules", {"vllm": None}):
        with pytest.raises(ImportError, match="vLLM is not installed"):
            VLLMInferenceBackend()


@pytest.mark.unit
def test_vllm_backend_with_mock(tmp_output_dir: Path):
    """When vllm is mocked, load_model and generate must call real vllm APIs."""
    mock_vllm = MagicMock()
    mock_llm_instance = MagicMock()
    mock_vllm.LLM.return_value = mock_llm_instance
    mock_vllm.SamplingParams.return_value = MagicMock()

    # Mock generate output shape
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="def solution():\n    return 42")]
    mock_llm_instance.generate.return_value = [mock_output]

    with patch.dict(
        "sys.modules",
        {"vllm": mock_vllm, "vllm.lora": MagicMock(), "vllm.lora.request": MagicMock()},
    ):
        backend = VLLMInferenceBackend(
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            dtype="bfloat16",
        )
        backend.load_model("deepseek-ai/DeepSeek-V4-Pro")

    mock_vllm.LLM.assert_called_once_with(
        model="deepseek-ai/DeepSeek-V4-Pro",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=False,
    )

    # Verify generate works
    sp = SamplingParams(temperature=0.0, max_new_tokens=512)
    with patch.dict(
        "sys.modules",
        {"vllm": mock_vllm, "vllm.lora": MagicMock(), "vllm.lora.request": MagicMock()},
    ):
        results = backend.generate(["Write a function"], sp)

    assert results == ["def solution():\n    return 42"]
