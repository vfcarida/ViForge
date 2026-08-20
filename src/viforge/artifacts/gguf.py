"""
ViForge GGUF Exporter and Ollama Modelfile Generator.

Export strategy (defense-in-depth):
1. If llama.cpp's ``convert_hf_to_gguf.py`` is discoverable on PATH or via
   ``LLAMA_CPP_PATH`` env var, run the real conversion + quantization.
2. Otherwise fall back to a JSON manifest stub so existing tests and CI
   continue working without llama.cpp installed.

Environment variables:
  LLAMA_CPP_PATH   – absolute path to the llama.cpp checkout root
                     (should contain ``convert_hf_to_gguf.py`` and
                     ``build/bin/llama-quantize`` or ``llama-quantize.exe``).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from viforge.utils.logging import logger


# --------------------------------------------------------------------------- #
# llama.cpp discovery                                                           #
# --------------------------------------------------------------------------- #


def _find_llama_cpp_root() -> Optional[Path]:
    """Return the llama.cpp root directory if discoverable, else None."""
    # 1. Explicit env var
    env_path = os.environ.get("LLAMA_CPP_PATH")
    if env_path:
        root = Path(env_path)
        if (root / "convert_hf_to_gguf.py").exists():
            return root

    # 2. ``llama-quantize`` on PATH implies llama.cpp is installed
    if shutil.which("llama-quantize") or shutil.which("llama-quantize.exe"):
        # Can't determine root from PATH alone — caller must set LLAMA_CPP_PATH
        # to use the Python converter.  Return None so we skip conversion step.
        return None

    return None


def _find_quantize_binary(llama_root: Optional[Path]) -> Optional[Path]:
    """Return the llama-quantize binary path, or None."""
    # Check on PATH first
    for name in ("llama-quantize", "llama-quantize.exe"):
        found = shutil.which(name)
        if found:
            return Path(found)

    # Check common build dirs inside llama.cpp root
    if llama_root:
        candidates = [
            llama_root / "build" / "bin" / "llama-quantize",
            llama_root / "build" / "bin" / "llama-quantize.exe",
            llama_root / "build" / "llama-quantize",
        ]
        for c in candidates:
            if c.exists():
                return c

    return None


# --------------------------------------------------------------------------- #
# GGUFExporter                                                                  #
# --------------------------------------------------------------------------- #


class GGUFExporter:
    """
    Handles model conversion to GGUF format and Ollama Modelfile generation.

    When llama.cpp is available (``LLAMA_CPP_PATH`` set):
    - Runs ``convert_hf_to_gguf.py`` to produce an F16 GGUF binary
    - Runs ``llama-quantize`` to produce the target quantized GGUF

    When llama.cpp is NOT available:
    - Writes a JSON manifest stub so the rest of the pipeline continues
    - Logs a clear warning so the user knows what to install
    """

    SUPPORTED_QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0", "F16", "BF16"]

    # llama-quantize ftype IDs (must match the binary's internal enum)
    _QUANT_FTYPE: Dict[str, str] = {
        "Q4_K_M": "q4_k_m",
        "Q5_K_M": "q5_k_m",
        "Q8_0": "q8_0",
        "F16": "f16",
        "BF16": "bf16",
    }

    @classmethod
    def generate_modelfile(
        cls,
        gguf_file_path: Path,
        output_modelfile_path: Path,
        system_prompt: str = "You are ViForge Specialist, an expert software engineering and reasoning AI.",
        temperature: float = 0.0,
        stop_tokens: Optional[List[str]] = None,
        context_length: int = 8192,
    ) -> str:
        """Generate an Ollama Modelfile configured with parameters, template, and system prompt."""
        stops = stop_tokens or ["<|endoftext|>", "<|im_end|>", "</s>", "```\n"]
        gguf_rel_or_abs = str(gguf_file_path.resolve())

        modelfile_content = [
            f"FROM {gguf_rel_or_abs}",
            f"PARAMETER temperature {temperature}",
            f"PARAMETER num_ctx {context_length}",
        ]

        for stop in stops:
            modelfile_content.append(f'PARAMETER stop "{stop}"')

        modelfile_content.extend(
            [
                f'SYSTEM "{system_prompt}"',
                "",
                "# Template for Chat/Instruction formatting",
                'TEMPLATE """{{ if .System }}<|im_start|>system',
                "{{ .System }}<|im_end|>",
                "{{ end }}{{ if .Prompt }}<|im_start|>user",
                "{{ .Prompt }}<|im_end|>",
                "{{ end }}<|im_start|>assistant",
                '{{ .Response }}<|im_end|>"""',
            ]
        )

        full_text = "\n".join(modelfile_content) + "\n"

        output_modelfile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_modelfile_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        logger.info(f"Ollama Modelfile generated at {output_modelfile_path}")
        return full_text

    @classmethod
    def _run_real_export(
        cls,
        merged_model_dir: Path,
        output_gguf_dir: Path,
        quant_type: str,
        llama_root: Path,
    ) -> Path:
        """
        Run the real llama.cpp conversion pipeline:
        1. convert_hf_to_gguf.py → F16 GGUF
        2. llama-quantize → quantized GGUF

        Returns the final GGUF path.
        Raises subprocess.CalledProcessError on failure.
        """
        model_name = merged_model_dir.name or "viforge_specialist"
        f16_gguf = output_gguf_dir / f"{model_name}-f16.gguf"
        final_gguf = output_gguf_dir / f"{model_name}-{quant_type.lower()}.gguf"

        convert_script = llama_root / "convert_hf_to_gguf.py"
        logger.info(f"GGUF: running convert_hf_to_gguf.py for '{merged_model_dir}'...")
        subprocess.run(
            [
                sys.executable,
                str(convert_script),
                str(merged_model_dir),
                "--outtype",
                "f16",
                "--outfile",
                str(f16_gguf),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"GGUF: F16 conversion complete → {f16_gguf}")

        # Skip quantization step for F16/BF16 — conversion is already the output
        if quant_type in ("F16", "BF16"):
            return f16_gguf

        quantize_bin = _find_quantize_binary(llama_root)
        if quantize_bin is None:
            logger.warning(
                "llama-quantize binary not found; skipping quantization step. "
                "GGUF will be F16 only."
            )
            return f16_gguf

        ftype = cls._QUANT_FTYPE[quant_type]
        logger.info(f"GGUF: quantizing to {quant_type} ({ftype})...")
        subprocess.run(
            [str(quantize_bin), str(f16_gguf), str(final_gguf), ftype],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"GGUF: quantization complete → {final_gguf}")

        # Remove intermediate F16 to save disk space
        if f16_gguf != final_gguf and f16_gguf.exists():
            f16_gguf.unlink()

        return final_gguf

    @classmethod
    def _run_mock_export(
        cls,
        merged_model_dir: Path,
        output_gguf_dir: Path,
        quant_type: str,
    ) -> Path:
        """
        Write a JSON descriptor stub when llama.cpp is not available.
        Always clearly marked as a mock so callers can detect it.
        """
        model_name = merged_model_dir.name or "viforge_specialist"
        gguf_path = output_gguf_dir / f"{model_name}-{quant_type.lower()}.gguf"

        manifest = {
            "format": "GGUF-v3-MOCK",
            "source_model_dir": str(merged_model_dir),
            "quantization_type": quant_type,
            "architecture": "llama_or_deepseek",
            "tensor_count": 290,
            "note": (
                "This is a JSON stub, not a real GGUF binary. "
                "Set LLAMA_CPP_PATH to enable real conversion."
            ),
        }

        with open(gguf_path, "w", encoding="utf-8") as f:
            f.write(f"GGUF_MAGIC_V3_MOCK:{json.dumps(manifest)}")

        logger.warning(
            "llama.cpp not found — wrote GGUF manifest stub. "
            "To enable real export, install llama.cpp and set LLAMA_CPP_PATH."
        )
        return gguf_path

    @classmethod
    def export(
        cls,
        merged_model_dir: Path,
        output_gguf_dir: Path,
        quant_type: str = "Q4_K_M",
        generate_ollama: bool = True,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Export a merged HF model checkpoint to GGUF and optionally create an Ollama Modelfile.

        Returns a dict with keys:
          - ``gguf_path`` (str)
          - ``modelfile_path`` (str | None)
          - ``quant_type`` (str)
          - ``is_real_gguf`` (bool) — False when llama.cpp was not available
          - ``manifest`` (dict)
        """
        if quant_type not in cls.SUPPORTED_QUANTS:
            raise ValueError(
                f"Unsupported quantization: {quant_type}. Supported: {cls.SUPPORTED_QUANTS}"
            )

        output_gguf_dir.mkdir(parents=True, exist_ok=True)
        modelfile_path = output_gguf_dir / "Modelfile"

        # Try real conversion first
        llama_root = _find_llama_cpp_root()
        is_real_gguf = False
        gguf_path: Path

        if llama_root is not None:
            try:
                gguf_path = cls._run_real_export(
                    merged_model_dir, output_gguf_dir, quant_type, llama_root
                )
                is_real_gguf = True
            except subprocess.CalledProcessError as exc:
                logger.error(
                    f"llama.cpp conversion failed (returncode={exc.returncode}). "
                    f"Falling back to mock export.\nstderr: {exc.stderr}"
                )
                gguf_path = cls._run_mock_export(merged_model_dir, output_gguf_dir, quant_type)
            except Exception as exc:
                logger.error(f"Unexpected error during GGUF conversion: {exc}. Using mock.")
                gguf_path = cls._run_mock_export(merged_model_dir, output_gguf_dir, quant_type)
        else:
            gguf_path = cls._run_mock_export(merged_model_dir, output_gguf_dir, quant_type)

        if generate_ollama:
            sys_p = (
                system_prompt or "You are ViForge Specialist, an expert software engineering AI."
            )
            cls.generate_modelfile(
                gguf_file_path=gguf_path,
                output_modelfile_path=modelfile_path,
                system_prompt=sys_p,
            )

        logger.info(
            f"GGUF export complete: {gguf_path} "
            f"({'real binary' if is_real_gguf else 'manifest stub'})"
        )

        manifest = {
            "format": "GGUF-v3" if is_real_gguf else "GGUF-v3-MOCK",
            "source_model_dir": str(merged_model_dir),
            "quantization_type": quant_type,
            "is_real_gguf": is_real_gguf,
        }

        return {
            "gguf_path": str(gguf_path),
            "modelfile_path": str(modelfile_path) if generate_ollama else None,
            "quant_type": quant_type,
            "is_real_gguf": is_real_gguf,
            "manifest": manifest,
        }
