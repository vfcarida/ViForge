"""
ViForge Automated Software Bill of Materials (SBOM) Generator.
Generates CycloneDX v1.5 JSON formatted SBOM from pinned requirements lockfiles.
"""

import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_lockfile(lockfile_path: Path) -> List[Dict[str, Any]]:
    """Parse pip-compile formatted lockfile with SHA-256 hashes."""
    if not lockfile_path.exists():
        print(f"Error: Lockfile not found at {lockfile_path}", file=sys.stderr)
        return []

    components: List[Dict[str, Any]] = []
    current_pkg: Dict[str, Any] = {}

    with open(lockfile_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue

            # Match package name and version: "package==1.2.3 \"
            pkg_match = re.match(r"^([a-zA-Z0-9_\-\.]+)=+([a-zA-Z0-9_\-\.]+)", line_str)
            if pkg_match:
                if current_pkg and "name" in current_pkg:
                    components.append(current_pkg)
                pkg_name = pkg_match.group(1).lower()
                pkg_version = pkg_match.group(2)
                current_pkg = {
                    "type": "library",
                    "name": pkg_name,
                    "version": pkg_version,
                    "purl": f"pkg:pypi/{pkg_name}@{pkg_version}",
                    "hashes": [],
                }

            # Match SHA-256 hash: "--hash=sha256:abcd..."
            hash_match = re.search(r"--hash=sha256:([a-fA-F0-9]{64})", line_str)
            if hash_match and current_pkg:
                current_pkg["hashes"].append({"alg": "SHA-256", "content": hash_match.group(1)})

        if current_pkg and "name" in current_pkg:
            components.append(current_pkg)

    return components


def generate_sbom(
    lockfile_path: Path, output_path: Path, project_name: str = "viforge", version: str = "0.1.0"
) -> Path:
    """Generate CycloneDX v1.5 JSON SBOM."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    components = parse_lockfile(lockfile_path)

    sbom: Dict[str, Any] = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tools": [
                {
                    "vendor": "ViForge Authors",
                    "name": "viforge-sbom-generator",
                    "version": version,
                }
            ],
            "component": {
                "type": "application",
                "name": project_name,
                "version": version,
                "description": "ViForge: Systematic Post-Training & Evaluation Framework",
                "licenses": [{"license": {"id": "MIT"}}],
                "purl": f"pkg:pypi/{project_name}@{version}",
            },
        },
        "components": components,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sbom, f, indent=2)

    return output_path
