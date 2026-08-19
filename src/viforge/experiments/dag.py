"""
ViForge Experiment Stage DAG Resolver.
"""

from typing import List
from viforge.config.schemas import TrainingStageConfig


class StageDAGResolver:
    """Validates and sorts training stages according to their depends_on dependencies."""

    @classmethod
    def resolve_execution_order(
        cls, stages: List[TrainingStageConfig]
    ) -> List[TrainingStageConfig]:
        stage_map = {s.stage_id: s for s in stages}
        ordered: List[TrainingStageConfig] = []
        visited = set()

        def visit(s: TrainingStageConfig, stack: set):
            if s.stage_id in stack:
                raise ValueError(f"Cyclic dependency detected at stage: {s.stage_id}")
            if s.stage_id not in visited:
                stack.add(s.stage_id)
                if s.depends_on:
                    if s.depends_on not in stage_map:
                        raise ValueError(
                            f"Dependency '{s.depends_on}' not found for stage '{s.stage_id}'"
                        )
                    visit(stage_map[s.depends_on], stack)
                stack.remove(s.stage_id)
                visited.add(s.stage_id)
                ordered.append(s)

        for stg in stages:
            visit(stg, set())

        return ordered
