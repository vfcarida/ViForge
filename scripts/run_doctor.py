"""
ViForge Doctor Script: Standalone Environment Checker.
"""

from viforge.utils.doctor import SystemDoctor
import json


def main():
    report = SystemDoctor.diagnose()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
