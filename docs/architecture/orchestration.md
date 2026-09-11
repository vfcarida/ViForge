# Enterprise Cluster & Cloud Orchestration

## Overview

Deploying post-training jobs at enterprise scale requires flexible, vendor-agnostic execution targets. **ViForge** natively supports three premier computing targets:

1. **HPC Clusters via Slurm**: Automated `.sbatch` script generation with multi-node `torchrun`, Apptainer/Singularity container execution, or native virtual environments.
2. **Kubernetes via KubeRay**: Production-grade `RayJob` Custom Resource Definitions (CRDs) for cloud-native orchestration with autoscaling and persistent volume tracking.
3. **AWS SageMaker**: Managed training execution with automated entrypoint generation, input data channels, and S3 artifact synchronization.

---

## Target Matrix

| Orchestrator | Target Environment | Runtime Isolation | Inter-Node Networking | Typical Deployment |
| :--- | :--- | :--- | :--- | :--- |
| **Slurm** | On-premise Supercomputers, HPC Clusters | Apptainer / Singularity / venv | InfiniBand, RoCE, MPI | Academic & Enterprise On-Prem |
| **KubeRay** | Kubernetes (EKS, GKE, AKS, OpenShift) | OCI Docker Containers | Kubernetes CNI, Calico | Modern Cloud-Native AI Platforms |
| **AWS SageMaker** | Fully Managed AWS Cloud | AWS Deep Learning Containers (DLC) | AWS EFA (Elastic Fabric Adapter) | Managed Enterprise Workflows |

---

## Slurm Job Generation (`.sbatch`)

ViForge's `SlurmJobGenerator` synthesizes production-ready batch submission scripts conforming to Slurm HPC standards:

### Key Features
- **Dynamic Node & GPU Topology**: Computes `srun` and `torchrun` parameters automatically (`--nnodes`, `--nproc_per_node`, rendezvous endpoints).
- **Dual Container Runtimes**: Supports Apptainer / Singularity (`--nv` GPU acceleration and `--bind` paths) or native Python virtual environments.
- **Log & Error Partitioning**: Auto-configures `%j.out` and `%j.err` log paths.

### Generated Script Structure Example

```bash
#!/bin/bash
#SBATCH --job-name=viforge-swe-train
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-a100
#SBATCH --output=slurm_logs/viforge-swe-train_%j.out
#SBATCH --error=slurm_logs/viforge-swe-train_%j.err

# Network Rendezvous Setup
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500

srun torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m viforge.cli.main train --config configs/experiments/software_engineering.yaml
```

---

## Kubernetes KubeRay (`RayJob`)

For Kubernetes infrastructures, ViForge generates declarative Kubernetes manifests adhering to `ray.io/v1` `RayJob` specifications:

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: viforge-train-job
  namespace: ml-training
spec:
  entrypoint: python -m viforge.cli.main train --config configs/experiments/software_engineering.yaml
  rayClusterSpec:
    rayVersion: "2.35.0"
    headGroupSpec:
      rayStartParams:
        dashboard-host: "0.0.0.0"
      template:
        spec:
          containers:
            - name: ray-head
              image: ghcr.io/viforge/viforge:latest
              resources:
                limits:
                  cpu: "8"
                  memory: "32Gi"
                requests:
                  cpu: "4"
                  memory: "16Gi"
    workerGroupSpecs:
      - groupName: gpu-workers
        replicas: 2
        minReplicas: 2
        maxReplicas: 2
        rayStartParams: {}
        template:
          spec:
            containers:
              - name: ray-worker
                image: ghcr.io/viforge/viforge:latest
                resources:
                  limits:
                    cpu: "16"
                    memory: "64Gi"
                    nvidia.com/gpu: "4"
                  requests:
                    cpu: "16"
                    memory: "64Gi"
                    nvidia.com/gpu: "4"
```

---

## Usage Examples

### CLI Command: Slurm Job Creation

```bash
viforge slurm-job \
  --job-name domain-specialist-cpt \
  --config configs/experiments/software_engineering.yaml \
  --num-nodes 2 \
  --gpus-per-node 4 \
  --partition gpu-a100 \
  --time-limit 24:00:00 \
  --output-path slurm_submit.sbatch
```

### CLI Command: KubeRay Manifest Generation

```bash
viforge ray-job \
  --job-name domain-specialist-ray \
  --config configs/experiments/software_engineering.yaml \
  --num-workers 2 \
  --gpus-per-worker 4 \
  --image ghcr.io/viforge/viforge:v1.0 \
  --output-path k8s_rayjob.yaml
```

### Python API

```python
from viforge.orchestration.slurm import SlurmJobGenerator
from viforge.orchestration.ray import RayJobGenerator

# Generate Slurm Batch Script
slurm_gen = SlurmJobGenerator(
    job_name="finetune-swe",
    num_nodes=2,
    gpus_per_node=4,
    partition="accelerated",
    container_image="/containers/viforge_latest.sif",
)
sbatch_path = slurm_gen.generate_sbatch_script(
    command="python -m viforge.cli.main train --config configs/swe.yaml",
    output_path="submit.sbatch",
)

# Generate KubeRay Manifest
ray_gen = RayJobGenerator(
    job_name="k8s-finetune-swe",
    num_workers=4,
    gpus_per_worker=8,
    image="custom-registry.io/viforge:prod",
)
ray_path = ray_gen.generate_ray_job_manifest(
    entrypoint="python -m viforge.cli.main train --config configs/swe.yaml",
    output_path="rayjob.yaml",
)
```
