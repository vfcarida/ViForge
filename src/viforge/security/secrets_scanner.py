"""
ViForge Secrets and Credentials Scanner.
"""

import re
from typing import Dict, List, Tuple


class SecretsScanner:
    """Detects credentials, AWS keys, GitHub tokens, and private keys."""

    PATTERNS: Dict[str, re.Pattern] = {
        "aws_access_key": re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
        "aws_secret_key": re.compile(r"(?i)aws_secret_access_key\s*=\s*['\"][A-Za-z0-9/+=]{40}['\"]"),
        "github_token": re.compile(r"\b(gh[pousr]_[A-Za-z0-9_]{36,255})\b"),
        "generic_api_key": re.compile(r"(?i)(api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][A-Za-z0-9-_]{20,80}['\"]"),
        "rsa_private_key": re.compile(r"-----BEGIN (?:RSA )?PRIVATE KEY-----"),
        "slack_token": re.compile(r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24,32}"),
    }

    @classmethod
    def scan(cls, text: str) -> List[Tuple[str, str]]:
        findings = []
        for name, pattern in cls.PATTERNS.items():
            matches = pattern.findall(text)
            for m in matches:
                findings.append((f"secret:{name}", str(m)[:10] + "..."))
        return findings

    @classmethod
    def redact(cls, text: str) -> str:
        sanitized = text
        for name, pattern in cls.PATTERNS.items():
            sanitized = pattern.sub(f"<REDACTED_SECRET_{name.upper()}>", sanitized)
        return sanitized
