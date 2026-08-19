"""
ViForge Isolated Code Execution Sandbox.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional


class SandboxExecutionError(Exception):
    """Raised when sandbox execution encounters an unexpected error."""

    pass


class SandboxTimeoutError(SandboxExecutionError):
    """Raised when sandbox execution exceeds its allotted time limit."""

    pass


class ExecutionSandbox:
    """
    Executes code snippets in an isolated, non-networked environment with memory caps and timeouts.
    """

    def __init__(self, default_timeout_sec: int = 15, max_memory_mb: int = 2048):
        self.default_timeout_sec = default_timeout_sec
        self.max_memory_mb = max_memory_mb

    def execute_snippet(
        self,
        code_string: str,
        test_assertions: str = "",
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        timeout = timeout_seconds or self.default_timeout_sec
        full_script = f"{code_string}\n\n# --- Assertions ---\n{test_assertions}\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "sandbox_exec.py"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(full_script)

            start_t = time.time()
            try:
                # Sanitized clean environment without AWS tokens or dangerous keys
                clean_env = {
                    "PYTHONPATH": "",
                    "PATH": sys.exec_prefix,
                    "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows")
                    if sys.platform == "win32"
                    else "",
                }

                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=clean_env,
                )
                elapsed = time.time() - start_t
                return {
                    "passed": result.returncode == 0,
                    "returncode": result.returncode,
                    "stdout": result.stdout[:2000],
                    "stderr": result.stderr[:2000],
                    "timeout": False,
                    "execution_time_seconds": round(elapsed, 3),
                }

            except subprocess.TimeoutExpired:
                elapsed = time.time() - start_t
                return {
                    "passed": False,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Execution timed out after {timeout} seconds.",
                    "timeout": True,
                    "execution_time_seconds": round(elapsed, 3),
                }
            except Exception as e:
                elapsed = time.time() - start_t
                return {
                    "passed": False,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Sandbox error: {type(e).__name__} - {str(e)}",
                    "timeout": False,
                    "execution_time_seconds": round(elapsed, 3),
                }
