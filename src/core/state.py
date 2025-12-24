from typing import List, Dict, Any, TypedDict, Annotated, Optional
import operator
from src.agents.sci.structures import SCIConfiguration, ExperimentResult

class AgentState(TypedDict):
    """
    Global State for the AI Scientist Multi-Agent Workflow (LangGraph)
    """
    cycle: int

    # Static / Read-only
    design_space: Dict[str, Any]
    budget_remaining: int

    # Planner & Reviewer State
    current_plan: List[SCIConfiguration]  # Proposed experiments for current cycle
    plan_feedback: str                    # Critique from Reviewer
    # Use operator.add to append history instead of overwriting, if desired.
    # For now, let's keep it simple: we might not need full history in state if WorldModel exists.
    # plan_history: Annotated[List[List[SCIConfiguration]], operator.add]

    # Execution State
    # Results are appended to the global list of experiments
    experiments: Annotated[List[ExperimentResult], operator.add]
    last_batch_results: List[ExperimentResult] # Temp storage for persistence

    # Analysis State
    insights: Dict[str, Any]             # Latest analysis insights
    new_insights: Optional[Dict[str, Any]] # Temp storage for persistence

    # Workflow Control
    status: str  # "planning", "reviewing", "executing", "analyzing", "end"
