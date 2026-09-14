"""
ViForge Unified Pareto Engine: End-to-End Specialization & Downstream Compression Multi-Objective Optimization.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from viforge.utils.logging import logger


class UnifiedParetoPoint(BaseModel):
    """
    Represents a candidate model variant along the end-to-end lifecycle:
    Base -> Specialized (ViForge) -> Compressed (ViPym/AWQ/GGUF).
    """

    model_name: str = Field(..., description="Base model family or name")
    variant_type: str = Field(
        ...,
        description="Type category: base, specialist, or compressed_<recipe>",
    )
    variant_label: str = Field(..., description="Human-readable label for plotting and tables")
    domain_score: float = Field(..., description="Normalized domain benchmark score (0.0 - 1.0)")
    domain_gain_pct: float = Field(default=0.0, description="Relative gain over base baseline (%)")
    general_retention_score: float = Field(
        ..., description="Normalized retention benchmark score (0.0 - 1.0)"
    )
    retention_delta_pct: float = Field(
        default=0.0, description="Relative retention delta from base baseline (%)"
    )
    training_cost_usd: float = Field(default=0.0, description="Post-training specialization cost")
    compression_cost_usd: float = Field(
        default=0.0, description="Downstream quantization/compression cost"
    )
    total_cost_usd: float = Field(default=0.0, description="Cumulative compute cost in USD")
    serving_memory_gb: float = Field(..., description="Peak serving VRAM footprint in GB")
    serving_latency_ms: float = Field(..., description="Inference latency P50 per token in ms")
    serving_throughput_tok_s: float = Field(
        default=0.0, description="Estimated inference throughput (tokens/sec)"
    )
    serving_cost_per_1m_tokens: float = Field(
        default=0.0, description="Estimated serving inference cost per 1M tokens"
    )
    capability_per_dollar: float = Field(
        default=0.0, description="Domain & retention score / Total amortized cost"
    )
    efficiency_index: float = Field(
        default=0.0, description="Combined index: (DomainScore * Throughput) / (VRAM * TotalCost)"
    )
    is_pareto_optimal: bool = Field(
        default=False, description="Whether this variant is non-dominated on the 5D Pareto frontier"
    )


class SweetSpotRecommendation(BaseModel):
    category: str
    point: UnifiedParetoPoint
    rationale: str


class UnifiedParetoEngine:
    """
    Computes multi-objective non-dominated Pareto frontiers and discovers deployment sweet spots
    spanning both ViForge specialization and ViPym downstream compression.
    """

    @classmethod
    def calculate_metrics(
        cls,
        domain_score: float,
        general_retention_score: float,
        total_cost_usd: float,
        serving_memory_gb: float,
        serving_latency_ms: float,
        base_domain: float,
        base_retention: float,
    ) -> Dict[str, float]:
        """Compute derived metrics: gain, throughput, cap/dollar, and efficiency index."""
        domain_gain = (
            ((domain_score - base_domain) / max(base_domain, 1e-6)) * 100.0 if base_domain else 0.0
        )
        retention_delta = (
            ((general_retention_score - base_retention) / max(base_retention, 1e-6)) * 100.0
            if base_retention
            else 0.0
        )
        throughput = 1000.0 / max(serving_latency_ms, 1e-3)
        effective_cost = max(total_cost_usd, 1.0)
        effective_mem = max(serving_memory_gb, 0.5)

        cap_per_dollar = ((domain_score * 0.7) + (general_retention_score * 0.3)) / effective_cost * 100.0
        # Efficiency index: rewards high quality and throughput while penalizing VRAM and cost
        efficiency = (domain_score * throughput) / (effective_mem * effective_cost) * 100.0

        # Approximate serving cost per 1M tokens based on memory-proportional GPU instance rental
        serving_cost_1m = (effective_mem / 80.0) * 2.50 + 0.10

        return {
            "domain_gain_pct": round(domain_gain, 2),
            "retention_delta_pct": round(retention_delta, 2),
            "serving_throughput_tok_s": round(throughput, 1),
            "serving_cost_per_1m_tokens": round(serving_cost_1m, 3),
            "capability_per_dollar": round(cap_per_dollar, 2),
            "efficiency_index": round(efficiency, 2),
        }

    @classmethod
    def build_unified_points(
        cls,
        model_name: str,
        base_domain_score: float = 0.50,
        base_retention_score: float = 0.65,
        specialized_domain_score: float = 0.68,
        specialized_retention_score: float = 0.65,
        training_cost_usd: float = 24.50,
        custom_variants: Optional[List[Dict[str, Any]]] = None,
    ) -> List[UnifiedParetoPoint]:
        """
        Construct a default complete lifecycle cohort of variants (Base -> Specialist -> Compressed).
        """
        # 1. Base Model
        base_calc = cls.calculate_metrics(
            domain_score=base_domain_score,
            general_retention_score=base_retention_score,
            total_cost_usd=0.0,
            serving_memory_gb=28.0,
            serving_latency_ms=120.0,
            base_domain=base_domain_score,
            base_retention=base_retention_score,
        )
        base_point = UnifiedParetoPoint(
            model_name=model_name,
            variant_type="base",
            variant_label=f"{model_name} (Base Baseline)",
            domain_score=base_domain_score,
            domain_gain_pct=0.0,
            general_retention_score=base_retention_score,
            retention_delta_pct=0.0,
            training_cost_usd=0.0,
            compression_cost_usd=0.0,
            total_cost_usd=0.0,
            serving_memory_gb=28.0,
            serving_latency_ms=120.0,
            serving_throughput_tok_s=base_calc["serving_throughput_tok_s"],
            serving_cost_per_1m_tokens=base_calc["serving_cost_per_1m_tokens"],
            capability_per_dollar=base_calc["capability_per_dollar"],
            efficiency_index=base_calc["efficiency_index"],
        )

        # 2. Specialized Model (ViForge: CPT + QLoRA + DPO)
        spec_calc = cls.calculate_metrics(
            domain_score=specialized_domain_score,
            general_retention_score=specialized_retention_score,
            total_cost_usd=training_cost_usd,
            serving_memory_gb=28.1,
            serving_latency_ms=124.0,
            base_domain=base_domain_score,
            base_retention=base_retention_score,
        )
        spec_point = UnifiedParetoPoint(
            model_name=model_name,
            variant_type="specialist",
            variant_label=f"{model_name} (Specialized FP16)",
            domain_score=specialized_domain_score,
            domain_gain_pct=spec_calc["domain_gain_pct"],
            general_retention_score=specialized_retention_score,
            retention_delta_pct=spec_calc["retention_delta_pct"],
            training_cost_usd=training_cost_usd,
            compression_cost_usd=0.0,
            total_cost_usd=training_cost_usd,
            serving_memory_gb=28.1,
            serving_latency_ms=124.0,
            serving_throughput_tok_s=spec_calc["serving_throughput_tok_s"],
            serving_cost_per_1m_tokens=spec_calc["serving_cost_per_1m_tokens"],
            capability_per_dollar=spec_calc["capability_per_dollar"],
            efficiency_index=spec_calc["efficiency_index"],
        )

        points = [base_point, spec_point]

        # 3. Downstream Compressed Variants (ViPym / AWQ)
        default_compressions: List[Dict[str, Any]] = [
            {
                "type": "compressed_smoothquant_w8a8",
                "label": "Specialist + SmoothQuant W8A8",
                "domain_score": specialized_domain_score * 0.993,
                "retention_score": specialized_retention_score * 0.997,
                "compression_cost": 0.20,
                "memory_gb": 14.2,
                "latency_ms": 63.0,
            },
            {
                "type": "compressed_autoround_w4a16",
                "label": "Specialist + AutoRound W4A16",
                "domain_score": specialized_domain_score * 0.988,
                "retention_score": specialized_retention_score * 0.991,
                "compression_cost": 0.35,
                "memory_gb": 7.2,
                "latency_ms": 85.0,
            },
            {
                "type": "compressed_fp8_kv",
                "label": "Specialist + FP8 KV Cache",
                "domain_score": specialized_domain_score * 0.997,
                "retention_score": specialized_retention_score * 0.998,
                "compression_cost": 0.05,
                "memory_gb": 18.0,
                "latency_ms": 98.0,
            },
            {
                "type": "compressed_distill",
                "label": "Specialist + Logit Distill (1.5B)",
                "domain_score": specialized_domain_score * 0.925,
                "retention_score": specialized_retention_score * 0.938,
                "compression_cost": 6.50,
                "memory_gb": 3.5,
                "latency_ms": 38.0,
            },
            {
                "type": "compressed_awq_w4a16",
                "label": "Specialist + AWQ W4A16",
                "domain_score": specialized_domain_score * 0.974,
                "retention_score": specialized_retention_score * 0.985,
                "compression_cost": 0.15,
                "memory_gb": 7.5,
                "latency_ms": 42.0,
            },
            {
                "type": "compressed_gguf_q4_k_m",
                "label": "Specialist + GGUF Q4_K_M (Ollama)",
                "domain_score": specialized_domain_score * 0.965,
                "retention_score": specialized_retention_score * 0.980,
                "compression_cost": 0.02,
                "memory_gb": 8.0,
                "latency_ms": 45.0,
            },
        ]

        variants_to_build: List[Dict[str, Any]] = (
            custom_variants if custom_variants is not None else default_compressions
        )

        for var in variants_to_build:
            d_score = float(var["domain_score"])
            r_score = float(var.get("retention_score", specialized_retention_score))
            comp_cost = float(var.get("compression_cost", 0.0))
            tot_cost = training_cost_usd + comp_cost
            mem_gb = float(var["memory_gb"])
            lat_ms = float(var["latency_ms"])

            calc = cls.calculate_metrics(
                domain_score=d_score,
                general_retention_score=r_score,
                total_cost_usd=tot_cost,
                serving_memory_gb=mem_gb,
                serving_latency_ms=lat_ms,
                base_domain=base_domain_score,
                base_retention=base_retention_score,
            )

            pt = UnifiedParetoPoint(
                model_name=model_name,
                variant_type=var["type"],
                variant_label=f"{model_name} ({var['label']})",
                domain_score=round(d_score, 4),
                domain_gain_pct=calc["domain_gain_pct"],
                general_retention_score=round(r_score, 4),
                retention_delta_pct=calc["retention_delta_pct"],
                training_cost_usd=training_cost_usd,
                compression_cost_usd=comp_cost,
                total_cost_usd=round(tot_cost, 2),
                serving_memory_gb=mem_gb,
                serving_latency_ms=lat_ms,
                serving_throughput_tok_s=calc["serving_throughput_tok_s"],
                serving_cost_per_1m_tokens=calc["serving_cost_per_1m_tokens"],
                capability_per_dollar=calc["capability_per_dollar"],
                efficiency_index=calc["efficiency_index"],
            )
            points.append(pt)

        return cls.identify_pareto_frontier(points)

    @classmethod
    def identify_pareto_frontier(cls, points: List[UnifiedParetoPoint]) -> List[UnifiedParetoPoint]:
        """
        Compute non-dominated Pareto frontier across 5 dimensions:
        1. domain_score (maximize)
        2. general_retention_score (maximize)
        3. total_cost_usd (minimize)
        4. serving_memory_gb (minimize)
        5. serving_latency_ms (minimize)
        """
        if not points:
            return []

        results = []
        for i, pt_a in enumerate(points):
            is_dominated = False
            for j, pt_b in enumerate(points):
                if i == j:
                    continue

                # pt_b is no worse than pt_a in all 5 criteria
                b_better_or_equal = (
                    pt_b.domain_score >= pt_a.domain_score
                    and pt_b.general_retention_score >= pt_a.general_retention_score
                    and pt_b.total_cost_usd <= pt_a.total_cost_usd
                    and pt_b.serving_memory_gb <= pt_a.serving_memory_gb
                    and pt_b.serving_latency_ms <= pt_a.serving_latency_ms
                )

                # pt_b is strictly better than pt_a in at least one criterion
                b_strictly_better = (
                    pt_b.domain_score > pt_a.domain_score
                    or pt_b.general_retention_score > pt_a.general_retention_score
                    or pt_b.total_cost_usd < pt_a.total_cost_usd
                    or pt_b.serving_memory_gb < pt_a.serving_memory_gb
                    or pt_b.serving_latency_ms < pt_a.serving_latency_ms
                )

                if b_better_or_equal and b_strictly_better:
                    is_dominated = True
                    break

            pt_copy = pt_a.model_copy()
            pt_copy.is_pareto_optimal = not is_dominated
            results.append(pt_copy)

        optimal_count = sum(1 for p in results if p.is_pareto_optimal)
        logger.info(
            f"Unified Pareto analysis: {optimal_count}/{len(results)} points on the 5D optimal frontier."
        )
        return results

    @classmethod
    def find_sweet_spots(cls, points: List[UnifiedParetoPoint]) -> List[SweetSpotRecommendation]:
        """
        Identify canonical deployment sweet spots on the Pareto frontier.
        """
        recs = []

        # 1. Maximum Capability Specialist
        max_cap = max(points, key=lambda p: p.domain_score)
        recs.append(
            SweetSpotRecommendation(
                category="Max Domain Capability",
                point=max_cap,
                rationale=(
                    f"Highest domain accuracy ({max_cap.domain_score:.3f}, {max_cap.domain_gain_pct:+.1f}% gain) "
                    f"for mission-critical accuracy where VRAM ({max_cap.serving_memory_gb} GB) is unconstrained."
                ),
            )
        )

        # 2. Edge / Consumer GPU Sweet Spot (VRAM <= 8GB)
        edge_candidates = [p for p in points if p.serving_memory_gb <= 8.5]
        if edge_candidates:
            best_edge = max(edge_candidates, key=lambda p: p.domain_score)
            recs.append(
                SweetSpotRecommendation(
                    category="Edge / Low-VRAM Sweet Spot (<=8GB)",
                    point=best_edge,
                    rationale=(
                        f"Retains {best_edge.domain_score:.3f} domain score while fitting entirely into "
                        f"consumer GPUs (RTX 4060/3070 or T4) at only {best_edge.serving_memory_gb:.1f} GB VRAM."
                    ),
                )
            )

        # 3. Ultra-Low Latency & High-Throughput Sweet Spot
        lowest_latency = min(points, key=lambda p: p.serving_latency_ms)
        recs.append(
            SweetSpotRecommendation(
                category="Ultra-Throughput & Low Latency",
                point=lowest_latency,
                rationale=(
                    f"Fastest token generation ({lowest_latency.serving_latency_ms:.1f} ms/token, "
                    f"{lowest_latency.serving_throughput_tok_s:.0f} tok/s) ideal for interactive real-time agent loops."
                ),
            )
        )

        # 4. Maximum Efficiency / ROI Sweet Spot
        highest_eff = max(points, key=lambda p: p.efficiency_index)
        recs.append(
            SweetSpotRecommendation(
                category="Maximum Efficiency Index (Quality*Throughput / VRAM*Cost)",
                point=highest_eff,
                rationale=(
                    f"Highest balanced efficiency score ({highest_eff.efficiency_index:.1f}), maximizing capability "
                    f"and serving speed per gigabyte of VRAM and training dollar."
                ),
            )
        )

        return recs

    @classmethod
    def export_json(cls, points: List[UnifiedParetoPoint], output_path: Path) -> Path:
        """Export Pareto frontier to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_points": len(points),
            "pareto_optimal_count": sum(1 for p in points if p.is_pareto_optimal),
            "frontier": [p.model_dump() for p in points],
            "sweet_spots": [
                {"category": r.category, "point": r.point.variant_label, "rationale": r.rationale}
                for r in cls.find_sweet_spots(points)
            ],
        }
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def generate_html_chart(cls, points: List[UnifiedParetoPoint], output_path: Path) -> Path:
        """
        Generate standalone HTML report with interactive Plotly visualization.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            # Separate Pareto-optimal and Dominated points
            opt_pts = [p for p in points if p.is_pareto_optimal]
            dom_pts = [p for p in points if not p.is_pareto_optimal]

            if dom_pts:
                fig.add_trace(
                    go.Scatter(
                        x=[p.serving_memory_gb for p in dom_pts],
                        y=[p.domain_score for p in dom_pts],
                        mode="markers+text",
                        name="Dominated Variants",
                        text=[p.variant_label.split(" (")[-1].replace(")", "") for p in dom_pts],
                        textposition="top center",
                        marker=dict(
                            size=[max(p.total_cost_usd, 5.0) * 0.8 for p in dom_pts],
                            color="#94a3b8",
                            opacity=0.6,
                        ),
                        hovertext=[
                            f"<b>{p.variant_label}</b><br>Domain: {p.domain_score:.3f}<br>VRAM: {p.serving_memory_gb}GB<br>Latency: {p.serving_latency_ms}ms<br>Cost: ${p.total_cost_usd:.2f}"
                            for p in dom_pts
                        ],
                        hoverinfo="text",
                    )
                )

            if opt_pts:
                fig.add_trace(
                    go.Scatter(
                        x=[p.serving_memory_gb for p in opt_pts],
                        y=[p.domain_score for p in opt_pts],
                        mode="markers+lines+text",
                        name="Pareto Optimal Frontier",
                        text=[p.variant_label.split(" (")[-1].replace(")", "") for p in opt_pts],
                        textposition="bottom right",
                        line=dict(color="#6366f1", width=2, dash="dot"),
                        marker=dict(
                            size=[max(p.total_cost_usd, 5.0) * 0.9 for p in opt_pts],
                            color="#10b981",
                            symbol="star",
                            line=dict(width=2, color="#047857"),
                        ),
                        hovertext=[
                            f"<b>{p.variant_label} (OPTIMAL)</b><br>Domain: {p.domain_score:.3f}<br>VRAM: {p.serving_memory_gb}GB<br>Latency: {p.serving_latency_ms}ms<br>Cost: ${p.total_cost_usd:.2f}"
                            for p in opt_pts
                        ],
                        hoverinfo="text",
                    )
                )

            fig.update_layout(
                title="<b>ViForge x ViPym Unified Pareto Frontier</b><br><sup>Specialization Accuracy vs Serving VRAM (Bubble Size = Compute Cost $)</sup>",
                xaxis_title="Serving VRAM Memory Footprint (GB) [Lower is Better]",
                yaxis_title="Domain Benchmark Capability Score [Higher is Better]",
                template="plotly_dark",
                hovermode="closest",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                font=dict(family="Inter, sans-serif", color="#f8fafc"),
            )

            fig.write_html(str(output_path), include_plotlyjs="cdn")
            return output_path

        except ImportError:
            # Fallback simple HTML if plotly is not importable
            html = f"""<!DOCTYPE html>
<html>
<head><title>ViForge Unified Pareto Frontier</title>
<style>body {{ font-family: sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; }} table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid #334155; padding: 8px 12px; text-align: left; }} th {{ background: #1e293b; }} tr.opt {{ background: #064e3b; }}</style>
</head>
<body>
<h2>ViForge x ViPym Unified Pareto Frontier</h2>
<table>
<tr><th>Variant</th><th>Domain Score</th><th>VRAM (GB)</th><th>Latency (ms)</th><th>Total Cost ($)</th><th>Optimal</th></tr>
{"".join(f'<tr class="{"opt" if p.is_pareto_optimal else ""}"><td>{p.variant_label}</td><td>{p.domain_score:.3f}</td><td>{p.serving_memory_gb}</td><td>{p.serving_latency_ms}</td><td>${p.total_cost_usd:.2f}</td><td>{"YES" if p.is_pareto_optimal else "NO"}</td></tr>' for p in points)}
</table>
</body></html>"""
            output_path.write_text(html, encoding="utf-8")
            return output_path


class MultiCampaignParetoComparator:
    """
    Cross-model benchmark comparison engine. Computes global multi-objective Pareto frontiers
    across different model families, sizes, and specialization campaigns.
    """

    MODEL_PALETTES = [
        "#6366f1",  # Indigo
        "#06b6d4",  # Cyan
        "#f59e0b",  # Amber
        "#ec4899",  # Pink
        "#10b981",  # Emerald
        "#8b5cf6",  # Violet
        "#ef4444",  # Red
    ]

    @classmethod
    def load_campaign(cls, path: Union[str, Path]) -> List[UnifiedParetoPoint]:
        """Load unified pareto points from a JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Pareto campaign file not found: {path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_list = data.get("frontier", data) if isinstance(data, dict) else data
        return [UnifiedParetoPoint(**item) for item in raw_list]

    @classmethod
    def compute_global_frontier(
        cls, campaigns: Dict[str, List[UnifiedParetoPoint]]
    ) -> List[UnifiedParetoPoint]:
        """
        Compute global Pareto optimality across all models and variants in all campaigns.
        """
        all_points: List[UnifiedParetoPoint] = []
        for _, points in campaigns.items():
            for p in points:
                all_points.append(p.model_copy())

        for i, pt_a in enumerate(all_points):
            is_dominated = False
            for j, pt_b in enumerate(all_points):
                if i == j:
                    continue
                b_better_or_equal = (
                    pt_b.domain_score >= pt_a.domain_score - 1e-4
                    and pt_b.general_retention_score >= pt_a.general_retention_score - 1e-4
                    and pt_b.serving_memory_gb <= pt_a.serving_memory_gb + 1e-4
                    and pt_b.serving_latency_ms <= pt_a.serving_latency_ms + 1e-4
                    and pt_b.total_cost_usd <= pt_a.total_cost_usd + 1e-4
                )
                b_strictly_better = (
                    pt_b.domain_score > pt_a.domain_score + 1e-4
                    or pt_b.general_retention_score > pt_a.general_retention_score + 1e-4
                    or pt_b.serving_memory_gb < pt_a.serving_memory_gb - 1e-4
                    or pt_b.serving_latency_ms < pt_a.serving_latency_ms - 1e-4
                    or pt_b.total_cost_usd < pt_a.total_cost_usd - 1e-4
                )
                if b_better_or_equal and b_strictly_better:
                    is_dominated = True
                    break
            pt_a.is_pareto_optimal = not is_dominated

        return all_points

    @classmethod
    def export_multi_model_json(
        cls,
        campaigns: Dict[str, List[UnifiedParetoPoint]],
        global_points: List[UnifiedParetoPoint],
        output_path: Path,
    ) -> Path:
        """Export cross-model comparison results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "campaign_count": len(campaigns),
            "models": list(campaigns.keys()),
            "total_variants_evaluated": len(global_points),
            "global_pareto_optimal_count": sum(1 for p in global_points if p.is_pareto_optimal),
            "campaign_summaries": {
                name: {
                    "variants_count": len(pts),
                    "best_domain_score": max((p.domain_score for p in pts), default=0.0),
                    "best_retention_score": max((p.general_retention_score for p in pts), default=0.0),
                    "min_vram_gb": min((p.serving_memory_gb for p in pts), default=0.0),
                    "min_latency_ms": min((p.serving_latency_ms for p in pts), default=0.0),
                }
                for name, pts in campaigns.items()
            },
            "global_sweet_spots": [
                {"category": r.category, "point": r.point.variant_label, "rationale": r.rationale}
                for r in UnifiedParetoEngine.find_sweet_spots(global_points)
            ],
            "global_frontier": [p.model_dump() for p in global_points if p.is_pareto_optimal],
            "all_points": [p.model_dump() for p in global_points],
        }
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def generate_multi_model_html_chart(
        cls,
        campaigns: Dict[str, List[UnifiedParetoPoint]],
        global_points: List[UnifiedParetoPoint],
        output_path: Path,
    ) -> Path:
        """Generate interactive Plotly visualization overlaying multiple models and global Pareto frontier."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            # Plot each campaign's points
            for idx, (model_name, points) in enumerate(campaigns.items()):
                color = cls.MODEL_PALETTES[idx % len(cls.MODEL_PALETTES)]
                sorted_pts = sorted(points, key=lambda p: p.serving_memory_gb)

                fig.add_trace(
                    go.Scatter(
                        x=[p.serving_memory_gb for p in sorted_pts],
                        y=[p.domain_score for p in sorted_pts],
                        mode="markers+lines",
                        name=f"{model_name}",
                        line=dict(color=color, width=2, dash="dash"),
                        marker=dict(
                            size=[max(p.total_cost_usd, 5.0) * 0.7 for p in sorted_pts],
                            color=color,
                            opacity=0.75,
                        ),
                        hovertext=[
                            f"<b>{p.variant_label}</b><br>Model: {model_name}<br>Domain: {p.domain_score:.3f}<br>VRAM: {p.serving_memory_gb}GB<br>Latency: {p.serving_latency_ms}ms<br>Cost: ${p.total_cost_usd:.2f}"
                            for p in sorted_pts
                        ],
                        hoverinfo="text",
                    )
                )

            # Global Pareto frontier
            global_opt = [p for p in global_points if p.is_pareto_optimal]
            if global_opt:
                sorted_global = sorted(global_opt, key=lambda p: p.serving_memory_gb)
                fig.add_trace(
                    go.Scatter(
                        x=[p.serving_memory_gb for p in sorted_global],
                        y=[p.domain_score for p in sorted_global],
                        mode="markers+lines+text",
                        name="🏆 Global Pareto Frontier",
                        text=[p.variant_label.split(" (")[-1].replace(")", "") for p in sorted_global],
                        textposition="top right",
                        line=dict(color="#10b981", width=3),
                        marker=dict(
                            size=16,
                            color="#10b981",
                            symbol="star",
                            line=dict(width=2, color="#ffffff"),
                        ),
                        hovertext=[
                            f"<b>{p.variant_label} (GLOBAL OPTIMAL)</b><br>Model: {p.model_name}<br>Domain: {p.domain_score:.3f}<br>VRAM: {p.serving_memory_gb}GB<br>Latency: {p.serving_latency_ms}ms<br>Cap/$: {p.capability_per_dollar:.2f}"
                            for p in sorted_global
                        ],
                        hoverinfo="text",
                    )
                )

            fig.update_layout(
                title="<b>ViForge Cross-Model Multi-Campaign Pareto Benchmark</b><br><sup>Cross-Architecture Evaluation: Accuracy vs Serving VRAM (Bubble Size = Total Cost)</sup>",
                xaxis_title="Serving VRAM Footprint (GB) [Lower is Better]",
                yaxis_title="Domain Benchmark Capability Score [Higher is Better]",
                template="plotly_dark",
                hovermode="closest",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                font=dict(family="Inter, sans-serif", color="#f8fafc"),
            )

            fig.write_html(str(output_path), include_plotlyjs="cdn")
            return output_path

        except ImportError:
            # Fallback simple HTML if plotly is not importable
            html = f"""<!DOCTYPE html>
<html>
<head><title>ViForge Cross-Model Pareto Benchmark</title>
<style>body {{ font-family: sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; }} table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid #334155; padding: 8px 12px; text-align: left; }} th {{ background: #1e293b; }} tr.opt {{ background: #064e3b; font-weight: bold; }}</style>
</head>
<body>
<h2>ViForge Cross-Model Multi-Campaign Pareto Benchmark</h2>
<table>
<tr><th>Model</th><th>Variant</th><th>Domain Score</th><th>VRAM (GB)</th><th>Latency (ms)</th><th>Cost ($)</th><th>Global Optimal</th></tr>
{"".join(f'<tr class="{"opt" if p.is_pareto_optimal else ""}"><td>{p.model_name}</td><td>{p.variant_label}</td><td>{p.domain_score:.3f}</td><td>{p.serving_memory_gb}</td><td>{p.serving_latency_ms}</td><td>${p.total_cost_usd:.2f}</td><td>{"YES (GLOBAL)" if p.is_pareto_optimal else "NO"}</td></tr>' for p in global_points)}
</table>
</body></html>"""
            output_path.write_text(html, encoding="utf-8")
            return output_path

