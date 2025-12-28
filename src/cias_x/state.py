"""
CIAS-X AgentState Definition

Global State for the AI Scientist Multi-Agent Workflow (LangGraph)
"""

from typing import List, Dict, Any, TypedDict


class AgentState(TypedDict):
    """
    Global State for the AI Scientist Multi-Agent Workflow (LangGraph)
    """
    executed_experiment_count: int

    # Static / Read-only
    design_space: Dict[str, Any]
    budget_remaining: int

    # Design info
    design_id: int

    # Planner
    configs: List[Any]  # Proposed configs for current plan cycle

    # Executor
    experiments: List[Any]  # Experiment results for current plan cycle

    # Analyst
    pareto_frontiers: List[Any]  # Pareto frontiers from WorldModel
    global_summary: str  # Global summary from WorldModel

    # Workflow Control
    status: str  # "planning", "executing", "analyzing", "end"
