"""
ViForge x ViPym Execution Orchestrator: Programmatically executes ViPym compression pipelines on ViForge models.
"""

import importlib.util
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from viforge.utils.logging import logger


def is_vipym_available() -> bool:
    """Check whether vipym is importable in the current environment."""
    try:
        return importlib.util.find_spec("vipym") is not None
    except Exception:
        return False


class ViPymRunner:
    """
    Orchestrates the execution of downstream ViPym compression workflows.
    """

    @classmethod
    def run_compression(
        cls,
        vipym_config_path: Union[str, Path],
        work_dir: Optional[Union[str, Path]] = None,
        mock: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a ViPym compression pipeline on a specialized model.
        """
        cfg_path = Path(vipym_config_path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"ViPym config file not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}

        exp_id = cfg_data.get("experiment_id", "vipym_compression")
        out_work_dir = Path(work_dir or "runs/vipym") / exp_id
        out_work_dir.mkdir(parents=True, exist_ok=True)

        stages = cfg_data.get("compression_pipeline", [])
        method = stages[0].get("method", "quantization") if stages else "quantization"
        scheme = stages[0].get("scheme", "W8A8") if stages else "W8A8"

        start_time = time.time()

        if mock:
            logger.info(f"ViPymRunner: Executing simulated compression pipeline ({method} {scheme})...")
            elapsed = 1.5
            size_reduction = 75.0 if "4" in scheme else 50.0
            result: Dict[str, Any] = {
                "status": "Completed",
                "is_mock": True,
                "experiment_id": exp_id,
                "method": method,
                "scheme": scheme,
                "output_dir": str(out_work_dir),
                "compressed_model_path": str(out_work_dir / "compressed_model"),
                "size_reduction_pct": size_reduction,
                "latency_speedup": 1.95 if "W8A8" in scheme else 1.45,
                "vram_saved_gb": 18.5 if "W4" in scheme else 12.0,
                "execution_time_seconds": elapsed,
            }
            summary_path = out_work_dir / "compression_summary.json"
            summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result

        if not is_vipym_available():
            raise RuntimeError(
                "ViPym is not installed in the active environment. "
                "Install it with 'pip install vipym' or run with mock=True."
            )

        logger.info(f"ViPymRunner: Launching live ViPym compression with config: {cfg_path}")
        try:
            cmd = ["vipym", "compress", str(cfg_path)]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            elapsed = time.time() - start_time
            return {
                "status": "Completed" if res.returncode == 0 else "Failed",
                "is_mock": False,
                "experiment_id": exp_id,
                "method": method,
                "scheme": scheme,
                "output_dir": str(out_work_dir),
                "returncode": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "execution_time_seconds": round(elapsed, 2),
            }
        except Exception as err:
            logger.warning(f"ViPym subprocess launch fallback: {err}")
            return {
                "status": "Failed",
                "is_mock": False,
                "error": str(err),
            }
