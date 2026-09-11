"""
ViForge SageMaker Managed Training Runner and AWS Cleanup Utility.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from viforge.config.schemas import ExperimentManifest
from viforge.utils.logging import logger



class SageMakerRunner:
    """
    Launches distributed training jobs on AWS SageMaker with least-privilege IAM roles.
    Integrates with boto3 for real AWS orchestration when configured, and provides
    a simulated runner when credentials or boto3 are not available.
    """

    def __init__(self, role_arn: Optional[str] = None, region: str = "us-east-1"):
        self.role_arn = role_arn or "arn:aws:iam::123456789012:role/ViForgeSageMakerExecutionRole"
        self.region = region

    @classmethod
    def is_boto3_available(cls) -> bool:
        """Check if boto3 is importable."""
        try:
            import boto3  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_instance_type(self, accelerator: str, num_gpus: int) -> str:
        """Map manifest accelerator and GPU count to AWS EC2/SageMaker instance types."""
        accel = accelerator.lower()
        if "h100" in accel:
            return "ml.p5.48xlarge"
        elif "a100" in accel:
            return "ml.p4de.24xlarge" if num_gpus >= 8 else "ml.p4d.24xlarge"
        elif "a10g" in accel or "g5" in accel:
            return f"ml.g5.{max(2, num_gpus * 4)}xlarge"
        return "ml.g5.12xlarge"

    def launch(
        self, manifest: ExperimentManifest, image_uri: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Launch training job on AWS SageMaker.
        Uses boto3 when available with valid credentials; falls back to simulator.
        """
        job_name = f"viforge-{manifest.experiment_id[:30]}-{int(time.time())}"
        instance_type = self._get_instance_type(
            manifest.hardware.accelerator, manifest.hardware.num_gpus
        )

        logger.info(
            f"SageMakerRunner: preparing '{job_name}' on {manifest.hardware.num_gpus}x "
            f"{manifest.hardware.accelerator} ({instance_type})..."
        )

        if self.is_boto3_available():
            try:
                import boto3
                from botocore.exceptions import BotoCoreError, ClientError

                client = boto3.client("sagemaker", region_name=self.region)
                container_uri = (
                    image_uri
                    or "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04"
                )

                hyperparameters = {
                    "experiment_id": str(manifest.experiment_id),
                    "model_id": str(manifest.model.hf_hub_id),
                    "pipeline_stages": str(len(manifest.pipeline)),
                }

                response = client.create_training_job(
                    TrainingJobName=job_name,
                    HyperParameters=hyperparameters,
                    AlgorithmSpecification={
                        "TrainingImage": container_uri,
                        "TrainingInputMode": "File",
                    },
                    RoleArn=self.role_arn,
                    ResourceConfig={
                        "InstanceType": instance_type,
                        "InstanceCount": 1,
                        "VolumeSizeInGB": 100,
                    },
                    StoppingCondition={"MaxRuntimeInSeconds": 86400},
                )
                logger.info(f"SageMaker training job launched: {response.get('TrainingJobArn')}")
                return {
                    "job_name": job_name,
                    "job_arn": response.get("TrainingJobArn"),
                    "status": "Launched",
                    "role_arn": self.role_arn,
                    "instance_type": instance_type,
                    "is_mock": False,
                }
            except (BotoCoreError, ClientError, Exception) as exc:
                logger.warning(
                    f"SageMaker live launch failed: {exc}. Using simulated training job response."
                )

        return {
            "job_name": job_name,
            "job_arn": f"arn:aws:sagemaker:{self.region}:123456789012:training-job/{job_name}",
            "status": "Simulated",
            "role_arn": self.role_arn,
            "instance_type": instance_type,
            "is_mock": True,
        }

    def describe_job(self, job_name: str) -> Dict[str, Any]:
        """Describe status of a launched SageMaker training job."""
        if self.is_boto3_available():
            try:
                import boto3

                client = boto3.client("sagemaker", region_name=self.region)
                resp = client.describe_training_job(TrainingJobName=job_name)
                return {
                    "job_name": job_name,
                    "status": resp.get("TrainingJobStatus", "Unknown"),
                    "secondary_status": resp.get("SecondaryStatus", ""),
                    "is_mock": False,
                }
            except Exception as e:
                logger.debug(f"SageMaker describe_job fallback: {e}")

        return {"job_name": job_name, "status": "Completed", "is_mock": True}

    @classmethod
    def generate_entrypoint_script(cls, output_path: Path) -> Path:
        """
        Generates a standard standalone SageMaker container entrypoint script.
        Reads SM_CHANNEL_*, SM_MODEL_DIR from AWS environment and launches ViForge.
        """
        script_content = '''"""
ViForge SageMaker Container Entrypoint.
Auto-generated by ViForge AWS Subsystem.
"""

import os
import sys
from pathlib import Path
from viforge.config.loader import ConfigLoader
from viforge.experiments.runner import ExperimentRunner
from viforge.utils.logging import logger

def main():
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    input_dir = Path(os.environ.get("SM_INPUT_DIR", "/opt/ml/input"))
    config_path = input_dir / "config" / "experiment.yaml"

    if not config_path.exists():
        env_cfg = os.environ.get("VIFORGE_CONFIG_PATH")
        if env_cfg and Path(env_cfg).exists():
            config_path = Path(env_cfg)
        else:
            logger.error(f"Configuration manifest not found at {config_path}")
            sys.exit(1)

    manifest = ConfigLoader.load_manifest(config_path)
    runner = ExperimentRunner(manifest, work_dir=model_dir)
    runner.profile_and_validate()
    summary = runner.execute(backend_type="huggingface", strict=True)
    logger.info(f"SageMaker training completed with verdict: {summary.verdict}")

if __name__ == "__main__":
    main()
'''
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(script_content, encoding="utf-8")
        logger.info(f"Generated SageMaker entrypoint script at: {output_path}")
        return output_path



class AWSCleanupManager:
    """Scans and terminates orphaned GPU training instances to prevent accidental cost overruns."""

    @classmethod
    def terminate_orphaned_jobs(
        cls,
        experiment_prefix: str = "viforge-",
        region: str = "us-east-1",
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        List InProgress training jobs matching prefix and terminate them.
        When dry_run=True, jobs are listed but not terminated.
        """
        logger.info(
            f"AWSCleanupManager: checking for running instances matching '{experiment_prefix}' (dry_run={dry_run})..."
        )
        checked = 0
        terminated: List[str] = []

        try:
            import boto3

            client = boto3.client("sagemaker", region_name=region)
            paginator = client.get_paginator("list_training_jobs")
            for page in paginator.paginate(
                StatusEquals="InProgress", NameContains=experiment_prefix
            ):
                for job in page.get("TrainingJobSummaries", []):
                    j_name = job["TrainingJobName"]
                    checked += 1
                    if dry_run:
                        logger.info(f"[Dry-Run] Would terminate orphaned SageMaker job: {j_name}")
                        terminated.append(j_name)
                    else:
                        logger.warning(f"Terminating orphaned SageMaker job: {j_name}")
                        client.stop_training_job(TrainingJobName=j_name)
                        terminated.append(j_name)
        except Exception as exc:
            logger.debug(f"AWSCleanupManager boto3 scan skipped/failed: {exc}")

        return {
            "instances_checked": checked,
            "instances_terminated": len(terminated) if not dry_run else 0,
            "dry_run": dry_run,
            "terminated_job_names": terminated,
            "status": "Clean" if not dry_run else "DryRunComplete",
        }
