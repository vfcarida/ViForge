"""
ViForge AWS Orphaned Resource Cleanup Script.
"""

from viforge.aws.cleanup import AWSCleanupManager


def main():
    print("Scanning for orphaned ViForge training jobs and instances...")
    res = AWSCleanupManager.terminate_orphaned_jobs(experiment_prefix="viforge-")
    print(f"Cleanup finished. Result: {res}")


if __name__ == "__main__":
    main()
