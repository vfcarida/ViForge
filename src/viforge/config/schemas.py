"""
ViForge Pydantic v2 Configuration Schemas & Data Contracts.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ModelType(str, Enum):
    CAUSAL_LM = "causal_lm"
    MOE = "moe"
    ENCODER_DECODER = "encoder_decoder"


class TrainingMethodType(str, Enum):
    SFT = "sft"
    LORA = "lora"
    QLORA = "qlora"
    FULL_FT = "full_ft"
    CPT = "cpt"
    DPO = "dpo"
    KTO = "kto"
    GRPO = "grpo"
    SYNTHETIC_SFT = "synthetic_sft"


class QuantizationType(str, Enum):
    NONE = "none"
    NF4 = "nf4"
    FP4 = "fp4"
    INT8 = "int8"


class ComputeDtype(str, Enum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class BasePricingConfig(BaseModel):
    prompt_usd_per_1m: float = Field(0.14, description="Base API cost per 1M prompt tokens")
    completion_usd_per_1m: float = Field(0.28, description="Base API cost per 1M completion tokens")


class ModelConfig(BaseModel):
    name: str = Field(..., description="Human-readable model name (e.g. DeepSeek V4 Pro)")
    hf_hub_id: str = Field(..., description="Hugging Face repo ID")
    revision: str = Field("main", description="Git commit hash or branch")
    model_type: ModelType = Field(ModelType.CAUSAL_LM, description="Model architecture type")
    total_parameters: float = Field(..., description="Total parameters in billions (e.g. 14.0)")
    active_parameters: Optional[float] = Field(None, description="Active parameters for MoE architectures")
    context_window: int = Field(4096, description="Maximum native context length")
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    expected_license: str = Field("MIT", description="Expected license")
    pricing: BasePricingConfig = Field(default_factory=BasePricingConfig)
    trust_remote_code: bool = Field(False, description="Whether to trust remote code")


class DatasetGovernanceManifest(BaseModel):
    dataset_id: str = Field(..., description="Unique dataset identifier")
    source_url: str = Field(..., description="Dataset origin URL or repo")
    source_type: Literal["huggingface", "git_repo", "synthetic_distill", "local_file", "s3"]
    revision_hash: str = Field(..., description="Git commit SHA or Hugging Face dataset revision")
    license: str = Field("MIT", description="SPDX License identifier")
    collection_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_sha256: str = Field(..., description="SHA-256 hash of dataset content")
    preprocessing_version: str = Field("1.0.0", description="Semver of preprocessing logic")
    filtering_version: str = Field("1.0.0", description="Semver of filtering/dedup logic")
    split_strategy: Literal["repo_stratified", "random", "temporal_cutoff", "author_disjoint"] = "repo_stratified"
    contamination_checked: bool = Field(False, description="Whether decontamination scan passed")
    benchmark_overlap_ratio: float = Field(0.0, ge=0.0, le=1.0)
    pii_scan_clean: bool = Field(False, description="Whether PII scan passed")
    secrets_scan_clean: bool = Field(False, description="Whether secret/token scan passed")
    num_samples: int = Field(0, ge=0)
    num_tokens_packed: int = Field(0, ge=0)


class DatasetReference(BaseModel):
    id: str = Field(..., description="Dataset ID matching dataset registry")
    path: Optional[str] = Field(None, description="Local path or S3 URI")
    split: str = Field("train", description="Dataset split (train/validation/test)")
    manifest_hash: Optional[str] = Field(None, description="Expected SHA-256 manifest hash")


class HyperparametersConfig(BaseModel):
    learning_rate: float = Field(2.0e-4, gt=0)
    lr_scheduler: str = Field("cosine")
    warmup_ratio: float = Field(0.03, ge=0.0, le=1.0)
    num_epochs: int = Field(3, ge=1)
    max_seq_len: int = Field(4096, ge=128)
    per_device_batch_size: int = Field(4, ge=1)
    gradient_accumulation_steps: int = Field(4, ge=1)
    gradient_checkpointing: bool = Field(True)
    weight_decay: float = Field(0.01, ge=0.0)
    optimizer: str = Field("paged_adamw_8bit")

    # PEFT / LoRA / QLoRA
    quantization: QuantizationType = Field(QuantizationType.NONE)
    double_quant: bool = Field(True)
    lora_rank: int = Field(64, ge=0)
    lora_alpha: int = Field(128, ge=1)
    lora_dropout: float = Field(0.05, ge=0.0, le=1.0)
    target_modules: Optional[List[str]] = Field(None)

    # Preference Optimization (DPO/GRPO/KTO)
    beta: Optional[float] = Field(0.1, description="DPO temperature beta")
    max_prompt_len: Optional[int] = Field(2048)
    max_completion_len: Optional[int] = Field(2048)
    num_generations: Optional[int] = Field(4, description="GRPO candidate rollouts per prompt")


class TrainingStageConfig(BaseModel):
    stage_id: str = Field(..., description="Unique stage identifier")
    method: TrainingMethodType = Field(..., description="Training method")
    dataset: DatasetReference
    hyperparameters: HyperparametersConfig = Field(default_factory=HyperparametersConfig)
    depends_on: Optional[str] = Field(None, description="Stage ID that must complete before this stage")
    output_adapter_dir: Optional[str] = Field(None)


class HardwareConfig(BaseModel):
    accelerator: str = Field("NVIDIA_A100_80GB", description="GPU device type")
    num_gpus: int = Field(1, ge=1)
    vram_per_gpu_gb: float = Field(80.0, gt=0)
    compute_dtype: ComputeDtype = Field(ComputeDtype.BFLOAT16)
    enable_flash_attention_2: bool = Field(True)
    deepspeed_config: Optional[Dict[str, Any]] = None
    fsdp_config: Optional[Dict[str, Any]] = None


class SamplingParams(BaseModel):
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    max_new_tokens: int = Field(2048, ge=1)
    seed: int = Field(42)
    stop_tokens: List[str] = Field(default_factory=lambda: ["<|endoftext|>", "</s>", "```\n"])


class BenchmarkItemConfig(BaseModel):
    name: str = Field(..., description="Benchmark identifier (e.g. humaneval_plus, swe_bench_lite)")
    version: Optional[str] = Field("latest")
    split: str = Field("test")
    pass_k: List[int] = Field(default_factory=lambda: [1])
    timeout_seconds: int = Field(15, ge=1)
    extra_args: Dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    sampling: SamplingParams = Field(default_factory=SamplingParams)
    domain_benchmarks: List[BenchmarkItemConfig] = Field(default_factory=list)
    general_retention_benchmarks: List[BenchmarkItemConfig] = Field(default_factory=list)
    sandbox_backend: Literal["docker", "gvisor", "subprocess"] = "docker"


class CostTrackingConfig(BaseModel):
    ec2_hourly_rate_usd: float = Field(7.40, ge=0.0)
    sagemaker_hourly_rate_usd: Optional[float] = None
    storage_gb_month_usd: float = Field(0.023, ge=0.0)
    log_to_mlflow: bool = Field(False)
    log_to_wandb: bool = Field(False)
    log_to_cloudwatch: bool = Field(False)
    mlflow_experiment_name: str = Field("ViForge_Specialization")


class ExperimentManifest(BaseModel):
    schema_version: str = Field("2.0")
    experiment_id: str = Field(..., description="Unique experiment run ID")
    tags: List[str] = Field(default_factory=list)
    model: ModelConfig
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    pipeline: List[TrainingStageConfig] = Field(..., description="Sequential or DAG training stages")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    cost_tracking: CostTrackingConfig = Field(default_factory=CostTrackingConfig)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BenchmarkResult(BaseModel):
    benchmark_name: str
    pass_at_k: Dict[str, float] = Field(default_factory=dict, description="e.g. {'pass@1': 0.682}")
    total_problems: int
    passed_problems: int
    failed_problems: int
    execution_time_seconds: float
    raw_metrics: Dict[str, Any] = Field(default_factory=dict)


class StatisticalDelta(BaseModel):
    metric_name: str
    baseline_value: float
    specialized_value: float
    absolute_delta: float
    relative_delta_pct: float
    ci_lower: float
    ci_upper: float
    p_value: Optional[float] = None
    is_significant: bool = False


class StageMetrics(BaseModel):
    stage_id: str
    method: str
    training_loss: float
    eval_loss: Optional[float] = None
    tokens_processed: int
    tokens_per_second: float
    peak_vram_gb: float
    trainable_parameters: int
    total_parameters: int
    trainable_ratio_pct: float
    wall_clock_seconds: float
    estimated_stage_cost_usd: float


class ParetoPoint(BaseModel):
    model_name: str
    stage_or_variant: str
    domain_score: float = Field(..., description="Weighted domain score")
    general_retention_score: float = Field(..., description="General capability retention score")
    training_cost_usd: float
    inference_cost_per_1m: float
    latency_p50_ms: float
    memory_footprint_gb: float
    capability_per_dollar: float = Field(..., description="Metric score / Total amortized cost")
    is_pareto_optimal: bool = False


class ExperimentSummaryReport(BaseModel):
    experiment_id: str
    model_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    baseline_domain_score: float
    specialized_domain_score: float
    domain_gain_pct: float
    baseline_retention_score: float
    specialized_retention_score: float
    retention_delta_pct: float
    total_training_cost_usd: float
    total_wall_clock_hours: float
    stages: List[StageMetrics] = Field(default_factory=list)
    benchmark_results: List[BenchmarkResult] = Field(default_factory=list)
    statistical_deltas: List[StatisticalDelta] = Field(default_factory=list)
    pareto_frontier: List[ParetoPoint] = Field(default_factory=list)
    verdict: str = Field(..., description="Research conclusion answering the core hypothesis")
    limitations_and_disclosures: List[str] = Field(default_factory=list)
