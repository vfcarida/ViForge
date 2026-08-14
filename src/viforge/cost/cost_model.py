"""
ViForge Economic Cost Model & Financial Analytics Engine.
"""

from viforge.config.schemas import BasePricingConfig


class CostModel:
    """Calculates comprehensive training, generation, storage, and inference costs."""

    @classmethod
    def calculate_training_cost(
        cls,
        wall_clock_seconds: float,
        hourly_rate: float = 7.40,
        num_instances: int = 1,
    ) -> float:
        hours = wall_clock_seconds / 3600.0
        return round(hours * num_instances * hourly_rate, 2)

    @classmethod
    def calculate_data_generation_cost(
        cls,
        num_queries: int,
        avg_prompt_tokens: int = 500,
        avg_completion_tokens: int = 800,
        teacher_prompt_usd_per_1m: float = 3.00,
        teacher_comp_usd_per_1m: float = 15.00,
    ) -> float:
        total_prompt_tokens = num_queries * avg_prompt_tokens
        total_comp_tokens = num_queries * avg_completion_tokens
        p_cost = (total_prompt_tokens / 1e6) * teacher_prompt_usd_per_1m
        c_cost = (total_comp_tokens / 1e6) * teacher_comp_usd_per_1m
        return round(p_cost + c_cost, 2)

    @classmethod
    def calculate_storage_cost(
        cls,
        size_gb: float,
        months: float = 1.0,
        rate_per_gb_month: float = 0.023,
    ) -> float:
        return round(size_gb * months * rate_per_gb_month, 4)

    @classmethod
    def calculate_inference_cost(
        cls,
        prompt_tokens: int,
        completion_tokens: int,
        pricing: BasePricingConfig,
    ) -> float:
        p_cost = (prompt_tokens / 1e6) * pricing.prompt_usd_per_1m
        c_cost = (completion_tokens / 1e6) * pricing.completion_usd_per_1m
        return round(p_cost + c_cost, 6)

    @classmethod
    def compute_capability_per_dollar(
        cls,
        domain_gain: float,
        retention_delta: float,
        total_experiment_cost: float,
        amortized_query_volume: int = 1_000_000,
        infer_cost_per_query: float = 0.0004,
        forgetting_weight: float = 0.5,
    ) -> float:
        forgetting_penalty = max(0.0, -retention_delta) * forgetting_weight
        net_gain = domain_gain - forgetting_penalty
        total_cost = max(1.0, total_experiment_cost + (amortized_query_volume * infer_cost_per_query))
        return round((net_gain / total_cost) * 1000.0, 4)
