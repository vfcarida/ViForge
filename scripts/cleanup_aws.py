"""
ViForge AWS Orphaned Resource Cleanup Script.
"""

import sys
from viforge.aws.sagemaker import AWSCleanupManager


def main():
    dry_run = "--dry-run" in sys.argv
    print(f"Scanning for orphaned ViForge training jobs and instances (dry_run={dry_run})...")
    res = AWSCleanupManager.terminate_orphaned_jobs(experiment_prefix="viforge-", dry_run=dry_run)
    print(f"Cleanup finished. Result: {res}")


if __name__ == "__main__":
    main()
