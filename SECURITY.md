# Security Policy

## Reporting Security Vulnerabilities

We take the security of ViForge seriously. If you discover a vulnerability or security flaw, please report it responsibly.

**Do NOT report security vulnerabilities via public GitHub issues.**

Please send security reports to **security@viforge.ai** with:
1. Description of the vulnerability
2. Steps to reproduce or proof-of-concept
3. Potential impact on training clusters or host systems

We will acknowledge receipt within 48 hours and work with you to remediate the issue promptly.

---

## Built-In Security Guardrails

ViForge implements several native security mechanisms:
* **Secrets Scanning:** Automated regex and entropy scanning for AWS tokens, GitHub keys, and private certificates before dataset ingestion.
* **PII Redaction:** Automated masking of email addresses, IP ranges, and SSNs.
* **Isolated Sandbox:** Benchmark execution is performed in non-networked sandboxes with strict memory and time quotas.
* **IAM Least Privilege:** Dedicated SageMaker and S3 roles with fine-grained scoping.
