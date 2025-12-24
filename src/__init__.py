"""
AI Scientist for SCI - Complete Implementation v3.0
An AI Scientist system for SCI domain based on Kosmos and CIAS-X algorithms

Key Features:
- AnalysisAgent uses LLM for intelligent analysis
- Supports all OpenAI-compatible LLM services
- Complete Pareto verification, trend analysis and experiment recommendations
"""

__version__ = "3.0.0"
__author__ = "AI Scientist Team"

from .agents.sci.structures import (
    SCIConfiguration,
    ForwardConfig,
    ReconParams,
    TrainConfig,
    Metrics,
    Artifacts,
    ExperimentResult,
    ReconFamily,
    UQScheme,
)
from .agents.sci.world_model import WorldModel
from .agents.sci.planner import PlannerAgent
from .agents.sci.executor import ExecutorAgent
from .agents.sci.analysis import AnalysisAgent
from .agents.sci.reviewer import PlanReviewerAgent
from .llm.client import LLMClient

__all__ = [
    # Data Structures
    "SCIConfiguration",
    "ForwardConfig",
    "ReconParams",
    "TrainConfig",
    "Metrics",
    "Artifacts",
    "ExperimentResult",
    "ReconFamily",
    "UQScheme",
    # Models
    "WorldModel",
    # Agents
    "PlannerAgent",
    "ExecutorAgent",
    "AnalysisAgent",
    "PlanReviewerAgent",
    # LLM
    "LLMClient",
]
