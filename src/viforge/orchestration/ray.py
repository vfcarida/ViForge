"""
ViForge Ray Train and KubeRay Job Manifest Generator.
Generates Kubernetes RayJob custom resources and standalone Ray Train distributed runners.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from viforge.config.schemas import ExperimentManifest
from viforge.utils.logging import logger


class RayJobGenerator:
    """
    Generates Kubernetes RayJob custom resource manifests and Ray Train distributed submission scripts.
    """

    @classmethod
    def generate_rayjob_dict(
        cls,
        manifest: ExperimentManifest,
        config_path: str = "configs/experiment.yaml",
        job_name: Optional[str] = None,
        namespace: str = "default",
        image: str = "viforge/viforge:latest",
        num_workers: int = 1,
        gpus_per_worker: Optional[int] = None,
        cpus_per_worker: int = 8,
        memory_per_worker_gb: int = 64,
    ) -> Dict[str, Any]:
        """
        Constructs a valid KubeRay (ray.io/v1) RayJob custom resource dictionary.
        """
        j_name = job_name or f"rayjob-{manifest.experiment_id[:24]}".lower().replace("_", "-")
        gpus = gpus_per_worker or max(1, manifest.hardware.num_gpus)

        manifest_dict: Dict[str, Any] = {
            "apiVersion": "ray.io/v1",
            "kind": "RayJob",
            "metadata": {
                "name": j_name,
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": "viforge",
                    "viforge.ai/experiment": manifest.experiment_id,
                },
            },
            "spec": {
                "entrypoint": f"python -m viforge.cli.main run {config_path} --live",
                "shutdownAfterJobFinishes": True,
                "ttlSecondsAfterFinished": 3600,
                "rayClusterSpec": {
                    "rayVersion": "2.35.0",
                    "headGroupSpec": {
                        "rayStartParams": {"dashboard-host": "0.0.0.0"},
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "ray-head",
                                        "image": image,
                                        "resources": {
                                            "limits": {"cpu": "4", "memory": "16Gi"},
                                            "requests": {"cpu": "2", "memory": "8Gi"},
                                        },
                                    }
                                ]
                            }
                        },
                    },
                    "workerGroupSpecs": [
                        {
                            "groupName": "gpu-workers",
                            "replicas": num_workers,
                            "minReplicas": num_workers,
                            "maxReplicas": num_workers,
                            "rayStartParams": {},
                            "template": {
                                "spec": {
                                    "containers": [
                                        {
                                            "name": "ray-worker",
                                            "image": image,
                                            "resources": {
                                                "limits": {
                                                    "cpu": str(cpus_per_worker),
                                                    "memory": f"{memory_per_worker_gb}Gi",
                                                    "nvidia.com/gpu": str(gpus),
                                                },
                                                "requests": {
                                                    "cpu": str(max(1, cpus_per_worker // 2)),
                                                    "memory": f"{memory_per_worker_gb // 2}Gi",
                                                    "nvidia.com/gpu": str(gpus),
                                                },
                                            },
                                        }
                                    ]
                                }
                            },
                        }
                    ],
                },
            },
        }
        return manifest_dict

    @classmethod
    def export_rayjob_yaml(
        cls,
        manifest: ExperimentManifest,
        output_path: Union[str, Path],
        config_path: str = "configs/experiment.yaml",
        **kwargs,
    ) -> Path:
        """
        Generates and saves the RayJob YAML manifest.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        job_dict = cls.generate_rayjob_dict(manifest, config_path=config_path, **kwargs)
        with open(out_p, "w", encoding="utf-8") as f:
            yaml.dump(job_dict, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Generated KubeRay RayJob manifest at: {out_p}")
        return out_p
