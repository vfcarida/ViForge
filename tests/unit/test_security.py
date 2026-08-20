"""
Unit tests for ViForge Security: Secrets scanning, PII redaction, and Hardened Execution Sandbox.
"""

import pytest

from viforge.security.pii_filter import PIIFilter
from viforge.security.sandbox import (
    BLOCKED_BUILTINS,
    BLOCKED_MODULES,
    ExecutionSandbox,
    HardenedSandbox,
    set_resource_limits,
    validate_ast,
)
from viforge.security.secrets_scanner import SecretsScanner


@pytest.mark.unit
def test_secrets_scanner():
    leak = "AWS_ACCESS_KEY_ID = 'AKIA1234567890ABCDEF'\nGH_TOKEN = 'ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'"
    findings = SecretsScanner.scan(leak)
    assert len(findings) >= 2
    redacted = SecretsScanner.redact(leak)
    assert "AKIA1234567890ABCDEF" not in redacted
    assert "<REDACTED_SECRET_" in redacted


@pytest.mark.unit
def test_pii_filter():
    pii_text = "Contact john.doe@realcompany.com or call SSN 123-45-6789 at IP 192.168.1.100."
    findings = PIIFilter.scan(pii_text)
    assert len(findings) >= 2
    redacted = PIIFilter.redact(pii_text)
    assert "john.doe@realcompany.com" not in redacted
    assert "<REDACTED_PII_" in redacted


@pytest.mark.unit
def test_execution_sandbox():
    sandbox = ExecutionSandbox(default_timeout_sec=5)
    result = sandbox.execute_snippet(
        code_string="def add(a, b): return a + b",
        test_assertions="assert add(2, 3) == 5\nassert add(-1, 1) == 0\n",
    )
    assert result["passed"] is True
    assert result["returncode"] == 0

    fail_res = sandbox.execute_snippet(
        code_string="def add(a, b): return a * b",
        test_assertions="assert add(2, 3) == 5",
    )
    assert fail_res["passed"] is False


@pytest.mark.unit
def test_sandbox_ast_blocked_imports():
    for blocked_mod in ["os", "subprocess", "socket", "shutil", "http", "urllib"]:
        code = f"import {blocked_mod}\ndef exploit(): pass"
        violations = validate_ast(code)
        assert len(violations) >= 1
        assert any(f"Blocked import: {blocked_mod}" in v for v in violations)

        sandbox = HardenedSandbox(default_timeout_sec=5)
        res = sandbox.execute_snippet(code_string=code)
        assert res["passed"] is False
        assert "AST Validation Failed" in res["stderr"]
        assert res["backend"] == "ast_blocked"

    from_import_code = "from subprocess import Popen\ndef exploit(): Popen(['ls'])"
    from_violations = validate_ast(from_import_code)
    assert any("Blocked import: subprocess" in v for v in from_violations)


@pytest.mark.unit
def test_sandbox_ast_blocked_builtins():
    for builtin_fn in ["eval", "exec", "compile"]:
        code = f"def exploit(s): return {builtin_fn}(s)"
        violations = validate_ast(code)
        assert len(violations) >= 1
        assert any(f"Blocked builtin: {builtin_fn}" in v for v in violations)

        sandbox = HardenedSandbox(default_timeout_sec=5)
        res = sandbox.execute_snippet(code_string=code)
        assert res["passed"] is False
        assert "AST Validation Failed" in res["stderr"]


@pytest.mark.unit
def test_sandbox_ast_allowed_code():
    safe_code = """
import math
import itertools
from collections import Counter
from typing import List

def solve(nums: List[int]) -> int:
    counts = Counter(nums)
    return max(counts.values(), default=0)
"""
    violations = validate_ast(safe_code)
    assert len(violations) == 0

    sandbox = HardenedSandbox(default_timeout_sec=5)
    res = sandbox.execute_snippet(
        code_string=safe_code,
        test_assertions="assert solve([1, 2, 2, 3, 3, 3]) == 3",
    )
    assert res["passed"] is True
    assert res["returncode"] == 0


@pytest.mark.unit
def test_sandbox_resource_limits():
    # Calling set_resource_limits should not crash on any platform
    set_resource_limits()


@pytest.mark.unit
def test_sandbox_docker_fallback():
    sandbox = HardenedSandbox(use_docker=True, default_timeout_sec=5)
    res = sandbox.execute_snippet(
        code_string="def mul(a, b): return a * b",
        test_assertions="assert mul(4, 5) == 20",
    )
    assert res["passed"] is True
    assert res["returncode"] == 0
    assert res["backend"] in ("docker", "subprocess")
