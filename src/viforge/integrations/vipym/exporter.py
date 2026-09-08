"""
ViForge x ViPym Exporter: Translates ViForge specialist checkpoints and manifests into ViPym compression experiments.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from viforge.config.loader import ConfigLoader
from viforge.config.schemas import ExperimentManifest
from viforge.integrations.vipym.recipes import (
    RECIPE_CATALOG,
    CompressionRecipe,
    RecipeDefinition,
)
from viforge.utils.logging import logger

BENCHMARK_TO_VIPYM_SUITE: Dict[str, str] = {
    "humaneval_plus": "humaneval",
    "humaneval": "humaneval",
    "swe_bench_lite": "swe_bench",
    "swe_bench": "swe_bench",
    "gsm8k": "gsm8k",
    "arc_challenge": "arc",
    "mmlu_pro": "mmlu",
    "mbpp_plus": "mbpp",
    "mbpp": "mbpp",
    "livecodebench": "humaneval",
}


class ViPymExporter:
    """
    Exports ViForge specialized model checkpoints or manifests to validated ViPym compression experiment configurations.
    """

    @classmethod
    def get_recipe_definition(cls, recipe: Union[CompressionRecipe, str]) -> RecipeDefinition:
        """Resolve CompressionRecipe enum or string to RecipeDefinition."""
        if isinstance(recipe, str):
            try:
                recipe_enum = CompressionRecipe(recipe.lower())
            except ValueError:
                valid = [r.value for r in CompressionRecipe]
                raise ValueError(f"Unknown compression recipe '{recipe}'. Valid recipes: {valid}")
        else:
            recipe_enum = recipe

        return RECIPE_CATALOG[recipe_enum]

    @classmethod
    def map_benchmarks(cls, benchmark_names: List[str]) -> List[str]:
        """Translate ViForge benchmark names to ViPym evaluation suite identifiers."""
        mapped = []
        for b in benchmark_names:
            suite = BENCHMARK_TO_VIPYM_SUITE.get(b.lower(), "humaneval")
            if suite not in mapped:
                mapped.append(suite)
        return mapped or ["humaneval"]

    @classmethod
    def build_vipym_config_dict(
        cls,
        model_id: str,
        recipe: Union[CompressionRecipe, str],
        experiment_id: Optional[str] = None,
        student_model_id: Optional[str] = None,
        calibration_dataset: str = "wikitext",
        calibration_samples: int = 512,
        evaluation_suites: Optional[List[str]] = None,
        extra_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Construct a dictionary conforming to ViPym's ViPymExperimentConfig schema.
        """
        recipe_def = cls.get_recipe_definition(recipe)
        exp_id = experiment_id or f"vipym_{recipe_def.recipe.value}_{Path(model_id).name or 'specialist'}"

        params = dict(recipe_def.default_parameters)
        if extra_parameters:
            params.update(extra_parameters)

        target_model_id = model_id
        # For distillation, ViForge model is teacher; student model is target
        if recipe_def.recipe == CompressionRecipe.DISTILL_LOGIT:
            if not student_model_id:
                student_model_id = "Qwen/Qwen2.5-Coder-1.5B"
            params["teacher_model"] = model_id
            target_model_id = student_model_id

        eval_suites = evaluation_suites or ["humaneval"]

        stage_config: Dict[str, Any] = {
            "stage_id": f"stage_{recipe_def.method}",
            "method": recipe_def.method,
            "scheme": recipe_def.scheme,
            "calibration": {
                "dataset_name": calibration_dataset,
                "dataset_config": "wikitext-2-raw-v1" if calibration_dataset == "wikitext" else None,
                "split": "train",
                "num_samples": calibration_samples,
                "sequence_length": 2048,
                "shuffle": True,
                "seed": 42,
            },
            "parameters": params,
            "dependencies": [],
        }

        vipym_config: Dict[str, Any] = {
            "experiment_id": exp_id,
            "seed": 42,
            "description": f"ViPym downstream compression for ViForge specialist model '{model_id}' ({recipe_def.description})",
            "model": {
                "id": str(target_model_id),
                "revision": "main",
                "trust_remote_code": False,
                "device_map": "auto",
            },
            "compression_pipeline": [stage_config],
            "serving": {
                "backend": "vllm",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.9,
                "kv_cache_dtype": "auto",
                "enable_speculative_decoding": False,
                "max_num_seqs": 256,
            },
            "evaluation": {
                "suites": eval_suites,
                "timeout_per_task_sec": 15,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 2048,
                "pass_k_values": [1],
                "isolate_with_gvisor": False,
                "allow_unsafe_execution": True,
            },
            "cost_assumptions": {
                "aws_ec2_hourly_rate": 32.77,
                "s3_storage_cost_per_gb_month": 0.023,
                "data_transfer_per_gb": 0.09,
                "active_gpu_count": 1,
            },
            "infrastructure": {
                "provider": "local",
                "aws_region": "us-east-1",
                "auto_teardown": True,
            },
        }

        # Validate against ViPym Pydantic schema if vipym is importable
        try:
            from vipym.core.config import ViPymExperimentConfig

            ViPymExperimentConfig.model_validate(vipym_config)
            logger.info("Validated export config against live ViPym Pydantic schema.")
        except ImportError:
            logger.debug("vipym not installed in active environment; skipping live schema validation.")
        except Exception as err:
            logger.warning(f"ViPym schema validation warning: {err}")

        return vipym_config

    @classmethod
    def export_from_model_dir(
        cls,
        model_dir: Union[str, Path],
        recipe: Union[CompressionRecipe, str],
        output_yaml_path: Union[str, Path],
        student_model_id: Optional[str] = None,
        calibration_dataset: str = "wikitext",
        calibration_samples: int = 512,
        evaluation_suites: Optional[List[str]] = None,
        extra_parameters: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export a merged model directory into a ViPym compression experiment manifest.
        """
        model_path = Path(model_dir).resolve()
        out_path = Path(output_yaml_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = cls.build_vipym_config_dict(
            model_id=str(model_path),
            recipe=recipe,
            student_model_id=student_model_id,
            calibration_dataset=calibration_dataset,
            calibration_samples=calibration_samples,
            evaluation_suites=evaluation_suites,
            extra_parameters=extra_parameters,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, sort_keys=False)

        logger.info(f"ViPym experiment manifest exported to: {out_path}")
        return out_path

    @classmethod
    def export_from_manifest(
        cls,
        manifest_or_path: Union[str, Path, ExperimentManifest],
        recipe: Union[CompressionRecipe, str],
        output_yaml_path: Union[str, Path],
        merged_model_dir: Optional[Union[str, Path]] = None,
        student_model_id: Optional[str] = None,
        calibration_dataset: str = "wikitext",
        calibration_samples: int = 512,
        extra_parameters: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export from a ViForge ExperimentManifest into a ViPym compression experiment manifest.
        """
        if isinstance(manifest_or_path, (str, Path)):
            manifest = ConfigLoader.load_manifest(Path(manifest_or_path))
        else:
            manifest = manifest_or_path

        model_target = (
            str(Path(merged_model_dir).resolve())
            if merged_model_dir
            else manifest.model.hf_hub_id
        )

        # Extract evaluated benchmarks from manifest
        benchmarks: List[str] = []
        for b in manifest.evaluation.domain_benchmarks:
            name = b.name if hasattr(b, "name") else str(b)
            benchmarks.append(name)
        for b in manifest.evaluation.retention_benchmarks:
            name = b.name if hasattr(b, "name") else str(b)
            benchmarks.append(name)

        mapped_suites = cls.map_benchmarks(benchmarks)

        return cls.export_from_model_dir(
            model_dir=model_target,
            recipe=recipe,
            output_yaml_path=output_yaml_path,
            student_model_id=student_model_id,
            calibration_dataset=calibration_dataset,
            calibration_samples=calibration_samples,
            evaluation_suites=mapped_suites,
            extra_parameters=extra_parameters,
        )
