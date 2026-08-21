"""
ViForge Automated Software Bill of Materials (SBOM) Generator Script.
"""

from pathlib import Path
from viforge.security.sbom import generate_sbom


def main():
    root_dir = Path(__file__).parent.parent
    lockfile = root_dir / "requirements" / "all.lock"
    if not lockfile.exists():
        lockfile = root_dir / "requirements" / "base.lock"

    out_file = root_dir / "dist" / "sbom.cyclonedx.json"
    generate_sbom(lockfile, out_file)
    print(f"SBOM successfully generated: {out_file}")


if __name__ == "__main__":
    main()
