"""
Unit tests for ViForge Security: Secrets scanning, PII redaction, and Execution Sandbox.
"""

from viforge.security.secrets_scanner import SecretsScanner
from viforge.security.pii_filter import PIIFilter
from viforge.security.sandbox import ExecutionSandbox


def test_secrets_scanner():
    leak = "AWS_ACCESS_KEY_ID = 'AKIA1234567890ABCDEF'\nGH_TOKEN = 'ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'"
    findings = SecretsScanner.scan(leak)
    assert len(findings) >= 2
    redacted = SecretsScanner.redact(leak)
    assert "AKIA1234567890ABCDEF" not in redacted
    assert "<REDACTED_SECRET_" in redacted


def test_pii_filter():
    pii_text = "Contact john.doe@realcompany.com or call SSN 123-45-6789 at IP 192.168.1.100."
    findings = PIIFilter.scan(pii_text)
    assert len(findings) >= 2
    redacted = PIIFilter.redact(pii_text)
    assert "john.doe@realcompany.com" not in redacted
    assert "<REDACTED_PII_" in redacted


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
