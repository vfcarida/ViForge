"""
ViForge Hardened Code Execution Sandbox.
Provides multi-layered defense-in-depth:
1. Static AST validation (blocking dangerous modules & builtins)
2. Ephemeral Docker container isolation (optional, zero-network, read-only root)
3. Subprocess execution with resource limits (RLIMIT_AS, RLIMIT_CPU, RLIMIT_NPROC, RLIMIT_FSIZE)
"""

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import resource
except ImportError:
    resource = None

BLOCKED_MODULES = {
    "os",
    "subprocess",
    "shutil",
    "socket",
    "http",
    "urllib",
    "ftplib",
    "smtplib",
    "ctypes",
    "multiprocessing",
    "signal",
}

BLOCKED_BUILTINS = {"exec", "eval", "compile"}


def validate_ast(code: str) -> List[str]:
    """
    Return list of security violations found in Python code AST.
    Checks for blocked imports and dangerous built-in execution functions.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    violations: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_pkg = alias.name.split(".")[0]
                if base_pkg in BLOCKED_MODULES:
                    violations.append(f"Blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_pkg = node.module.split(".")[0]
                if base_pkg in BLOCKED_MODULES:
                    violations.append(f"Blocked import: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS:
                violations.append(f"Blocked builtin: {node.func.id}")

    return violations


def set_resource_limits() -> None:
    """
    Configure POSIX resource limits for untrusted process isolation:
    - 256MB virtual address space limit
    - 30s CPU time limit
    - 0 child processes (no fork bomb)
    - 1MB maximum file write limit
    """
    if resource is None:
        return

    # Memory limit (256MB)
    try:
        mem_limit = 256 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
    except (ValueError, getattr(resource, "error", Exception)):
        pass

    # CPU time limit (30s)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    except (ValueError, getattr(resource, "error", Exception)):
        pass

    # Process limit (no fork)
    try:
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except (ValueError, getattr(resource, "error", Exception)):
        pass

    # File size limit (1MB)
    try:
        fsize_limit = 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_limit, fsize_limit))
    except (ValueError, getattr(resource, "error", Exception)):
        pass


class SandboxExecutionError(Exception):
    """Raised when sandbox execution encounters an unexpected error."""

    pass


class SandboxTimeoutError(SandboxExecutionError):
    """Raised when sandbox execution exceeds its allotted time limit."""

    pass


class HardenedSandbox:
    """
    Layered Hardened Sandbox for executing untrusted LLM-generated code.
    Layer 1: AST pre-validation
    Layer 2: Ephemeral Docker container isolation (if available and enabled)
    Layer 3: Subprocess execution with environment stripping and resource limits (fallback)
    """

    def __init__(
        self,
        default_timeout_sec: int = 15,
        max_memory_mb: int = 256,
        use_docker: bool = False,
        enable_ast_validation: bool = True,
        docker_image: str = "python:3.11-slim",
    ):
        self.default_timeout_sec = default_timeout_sec
        self.max_memory_mb = max_memory_mb
        self.use_docker = use_docker
        self.enable_ast_validation = enable_ast_validation
        self.docker_image = docker_image

    @classmethod
    def is_docker_available(cls) -> bool:
        """Check if docker CLI is present and the daemon is reachable."""
        if not shutil.which("docker"):
            return False
        try:
            res = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return res.returncode == 0
        except Exception:
            return False

    def execute_snippet(
        self,
        code_string: str,
        test_assertions: str = "",
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        timeout = timeout_seconds or self.default_timeout_sec
        full_script = f"{code_string}\n\n# --- Assertions ---\n{test_assertions}\n"

        # 1. AST Validation
        if self.enable_ast_validation:
            violations = validate_ast(full_script)
            if violations:
                return {
                    "passed": False,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"AST Validation Failed: {'; '.join(violations)}",
                    "timeout": False,
                    "execution_time_seconds": 0.0,
                    "violations": violations,
                    "backend": "ast_blocked",
                }

        # 2. Docker Execution (if available and enabled)
        if self.use_docker and self.is_docker_available():
            try:
                return self._execute_docker(full_script, timeout)
            except Exception:
                # Graceful fallback to subprocess
                pass

        # 3. Subprocess Execution (with resource limits)
        return self._execute_subprocess(full_script, timeout)

    def _execute_docker(self, full_script: str, timeout: int) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "sandbox_exec.py"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(full_script)

            docker_cmd = [
                "docker",
                "run",
                "--rm",
                f"--memory={self.max_memory_mb}m",
                "--network=none",
                "--read-only",
                "-v",
                f"{Path(tmpdir).resolve()}:/sandbox:ro",
                "-w",
                "/sandbox",
                self.docker_image,
                "python",
                "/sandbox/sandbox_exec.py",
            ]

            start_t = time.time()
            try:
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                elapsed = time.time() - start_t
                return {
                    "passed": result.returncode == 0,
                    "returncode": result.returncode,
                    "stdout": result.stdout[:2000],
                    "stderr": result.stderr[:2000],
                    "timeout": False,
                    "execution_time_seconds": round(elapsed, 3),
                    "backend": "docker",
                }
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start_t
                return {
                    "passed": False,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Docker execution timed out after {timeout} seconds.",
                    "timeout": True,
                    "execution_time_seconds": round(elapsed, 3),
                    "backend": "docker",
                }

    def _execute_subprocess(self, full_script: str, timeout: int) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "sandbox_exec.py"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(full_script)

            start_t = time.time()
            try:
                clean_env = {
                    "PYTHONPATH": "",
                    "PATH": sys.exec_prefix,
                    "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows")
                    if sys.platform == "win32"
                    else "",
                }

                preexec = (
                    set_resource_limits
                    if sys.platform != "win32" and resource is not None
                    else None
                )

                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=clean_env,
                    preexec_fn=preexec,
                )
                elapsed = time.time() - start_t
                return {
                    "passed": result.returncode == 0,
                    "returncode": result.returncode,
                    "stdout": result.stdout[:2000],
                    "stderr": result.stderr[:2000],
                    "timeout": False,
                    "execution_time_seconds": round(elapsed, 3),
                    "backend": "subprocess",
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
                    "backend": "subprocess",
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
                    "backend": "subprocess",
                }


# Backwards compatibility alias
ExecutionSandbox = HardenedSandbox
