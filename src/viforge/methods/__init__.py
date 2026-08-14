"""
ViForge Specialization Methods module: SFT, LoRA, QLoRA, CPT, DPO, GRPO, SyntheticSFT.
"""

from viforge.methods.base import BaseTrainingMethod, MethodRegistry, method_registry
from viforge.methods.sft import SFTMethod
from viforge.methods.peft_lora import LoRAMethod, QLoRAMethod
from viforge.methods.cpt import CPTMethod
from viforge.methods.preference import DPOMethod, GRPOMethod
from viforge.methods.synthetic import SyntheticDataPipeline, SyntheticSFTMethod

__all__ = [
    "BaseTrainingMethod",
    "MethodRegistry",
    "method_registry",
    "SFTMethod",
    "LoRAMethod",
    "QLoRAMethod",
    "CPTMethod",
    "DPOMethod",
    "GRPOMethod",
    "SyntheticDataPipeline",
    "SyntheticSFTMethod",
]
