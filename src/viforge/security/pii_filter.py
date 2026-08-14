"""
ViForge Personal Identifiable Information (PII) Filter.
"""

import re
from typing import Dict, List, Tuple


class PIIFilter:
    """Detects and redacts personal identifiable information."""

    PII_PATTERNS: Dict[str, re.Pattern] = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
        "ipv4": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    }

    @classmethod
    def scan(cls, text: str) -> List[Tuple[str, str]]:
        findings = []
        for name, pattern in cls.PII_PATTERNS.items():
            if name == "ipv4":
                for m in pattern.findall(text):
                    if not (m.startswith("127.") or m.startswith("0.0.") or m == "255.255.255.255"):
                        findings.append((f"pii:{name}", m))
            elif name == "email":
                for m in pattern.findall(text):
                    if not any(d in m.lower() for d in ["example.com", "test.com", "placeholder.org", "domain.com"]):
                        findings.append((f"pii:{name}", m))
            else:
                for m in pattern.findall(text):
                    findings.append((f"pii:{name}", str(m)))
        return findings

    @classmethod
    def redact(cls, text: str) -> str:
        sanitized = text
        for name, pattern in cls.PII_PATTERNS.items():
            sanitized = pattern.sub(f"<REDACTED_PII_{name.upper()}>", sanitized)
        return sanitized
