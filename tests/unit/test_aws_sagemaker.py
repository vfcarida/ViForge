"""
Unit tests for ViForge AWS SageMaker runner and AWSCleanupManager.
"""

from unittest.mock import MagicMock, patch
from viforge.aws.sagemaker import AWSCleanupManager, SageMakerRunner
from viforge.config.schemas import (
    ExperimentManifest,
    HardwareConfig,
    ModelConfig,
    TrainingStageConfig,
)


def _make_dummy_manifest() -> ExperimentManifest:
    model_cfg = ModelConfig(
        name="TestModel",
        hf_hub_id="test/model",
        total_parameters=7.0,
    )
    stage_cfg = TrainingStageConfig(
        stage_id="stage_1",
        method="sft",
        dataset={"id": "ds1", "split": "train"},
    )
    return ExperimentManifest(
        experiment_id="test-exp-123",
        model=model_cfg,
        hardware=HardwareConfig(accelerator="A100-80GB", num_gpus=8),
        pipeline=[stage_cfg],
    )


def test_sagemaker_runner_offline_simulation():
    """Test SageMaker runner simulation when boto3 is not connected."""
    manifest = _make_dummy_manifest()
    runner = SageMakerRunner()

    with patch.object(SageMakerRunner, "is_boto3_available", return_value=False):
        res = runner.launch(manifest)
        assert res["status"] == "Simulated"
        assert res["is_mock"] is True
        assert "viforge-test-exp-123" in res["job_name"]
        assert res["instance_type"] == "ml.p4de.24xlarge"


def test_sagemaker_runner_boto3_launch():
    """Test SageMaker runner launch with mocked boto3 client."""
    manifest = _make_dummy_manifest()
    runner = SageMakerRunner(region="us-west-2")

    mock_client = MagicMock()
    mock_client.create_training_job.return_value = {
        "TrainingJobArn": "arn:aws:sagemaker:us-west-2:123456:training-job/viforge-test-exp-123"
    }

    with (
        patch("boto3.client", return_value=mock_client),
        patch.object(SageMakerRunner, "is_boto3_available", return_value=True),
    ):
        res = runner.launch(manifest)
        assert res["status"] == "Launched"
        assert res["is_mock"] is False
        assert (
            res["job_arn"] == "arn:aws:sagemaker:us-west-2:123456:training-job/viforge-test-exp-123"
        )
        assert mock_client.create_training_job.called


def test_aws_cleanup_manager_terminate_orphaned():
    """Test terminating orphaned SageMaker training jobs."""
    mock_client = MagicMock()
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {
            "TrainingJobSummaries": [
                {"TrainingJobName": "viforge-test-job-1"},
                {"TrainingJobName": "viforge-test-job-2"},
            ]
        }
    ]
    mock_client.get_paginator.return_value = mock_paginator

    with patch("boto3.client", return_value=mock_client):
        res = AWSCleanupManager.terminate_orphaned_jobs(experiment_prefix="viforge-")
        assert res["instances_checked"] == 2
        assert res["instances_terminated"] == 2
        assert "viforge-test-job-1" in res["terminated_job_names"]
        assert mock_client.stop_training_job.call_count == 2


def test_aws_cleanup_manager_dry_run():
    """Test dry_run mode for AWSCleanupManager."""
    mock_client = MagicMock()
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {"TrainingJobSummaries": [{"TrainingJobName": "viforge-test-job-1"}]}
    ]
    mock_client.get_paginator.return_value = mock_paginator

    with patch("boto3.client", return_value=mock_client):
        res = AWSCleanupManager.terminate_orphaned_jobs(experiment_prefix="viforge-", dry_run=True)
        assert res["instances_checked"] == 1
        assert res["instances_terminated"] == 0
        assert res["dry_run"] is True
        assert res["status"] == "DryRunComplete"
        assert not mock_client.stop_training_job.called
