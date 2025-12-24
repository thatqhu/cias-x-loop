import logging
import asyncio
from typing import Dict, Any, Callable, Type
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    # Fallback for when langgraph is not installed yet
    class StateGraph:
        def __init__(self, state_schema): pass
        def add_node(self, name, func): pass
        def set_entry_point(self, name): pass
        def add_edge(self, src, dst): pass
        def add_conditional_edges(self, src, cond, path_map): pass
        def compile(self): return self
    END = "END"
    print("WARNING: langgraph not installed. Install with `pip install langgraph`")

from src.core.state import AgentState
from src.core.world_model_base import WorldModelBase

logger = logging.getLogger(__name__)

# --- Base Node ---

class BaseNode:
    """Base class for all workflow nodes"""
    def __init__(self, agent: Any):
        self.agent = agent

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute the node logic asynchronously.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

# --- Concrete Node Implementations ---

class PlannerNode(BaseNode):
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        cycle = state.get("cycle", 1)
        feedback = state.get("plan_feedback", "")
        logger.info(f"--- Node: Planner (Cycle {cycle}) ---")

        budget = min(3, state.get("budget_remaining", 0))
        design_space = state.get("design_space", {})

        # Access WorldModel via agent
        if not self.agent.world_model:
            logger.error("Planner agent missing world_model reference")
            return {"status": "error"}

        summary = self.agent.world_model.summarize()

        configs = self.agent.plan_experiments(
            world_summary=summary,
            design_space=design_space,
            budget=budget,
            feedback=feedback
        )

        return {
            "current_plan": configs,
            "status": "reviewing"
        }

class ReviewerNode(BaseNode):
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- Node: Reviewer ---")

        configs = state.get("current_plan", [])
        if not configs:
            logger.warning("Empty plan submitted to reviewer")
            return {"plan_feedback": "Empty plan", "status": "planning_retry"}

        context = {
            "cycle": state.get("cycle", 1),
        }

        review_result = await self.agent.review_plan(configs, context)

        if review_result.approved:
            logger.info(f"Plan APPROVED ({len(review_result.approved_configs)} configs)")
            return {
                "current_plan": review_result.approved_configs,
                "plan_feedback": "",
                "status": "executing"
            }
        else:
            logger.warning("Plan REJECTED")
            feedback_msg = f"{review_result.feedback}\nCritique: {review_result.critique}"
            return {
                "plan_feedback": feedback_msg,
                "status": "planning_retry"
            }

class ExecutorNode(BaseNode):
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- Node: Executor ---")

        configs = state.get("current_plan", [])

        # Async execution
        results = await self.agent.run_experiments_async(configs)

        cost = len(results)
        logger.info(f"Executed {cost} experiments")

        return {
            "experiments": results,
            "current_plan": [],
            "budget_remaining": state.get("budget_remaining", 0) - cost,
            "last_batch_results": results,
            "status": "persisting"
        }

class PersistenceNode(BaseNode):
    """
    Special node that doesn't map 1:1 to an agent but uses an agent's
    inherited world_model reference to perform data syncing.
    """
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- Node: Persistence (Sync) ---")

        updates = {}
        wm = self.agent.world_model if hasattr(self.agent, 'world_model') else None

        if wm:
            # 1. Persist Experiment Results
            new_results = state.get("last_batch_results", [])
            if new_results:
                count = 0
                for result in new_results:
                    wm.add_experiment(result)
                    count += 1
                logger.info(f"Persisted {count} experiments to World Model (Storage)")
                # Clear the batch so we don't re-persist
                updates["last_batch_results"] = []

            # 2. Persist Insights
            new_insights = state.get("new_insights", None)
            if new_insights:
                logger.info("Persisting Analysis Insights to World Model")
                wm.update_with_insights(new_insights)
                # Clear flag/data
                updates["new_insights"] = None
                # Also ensure they are in the main state if not already
                updates["insights"] = new_insights

        else:
            logger.warning("WorldModel reference unavailable or invalid for persistence")

        # Status returned might need to be dynamic or ignored if edge handles it
        # but for safety let's return a generic status that edges can ignore if they want
        return updates

class AnalyzerNode(BaseNode):
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- Node: Analyzer ---")

        cycle = state.get("cycle", 1)
        pareto_ids, insights = await self.agent.run_analysis(cycle)

        # Instead of returning logic status here, we populate 'new_insights'
        # and let the subsequent PersistenceNode handle the sync.
        return {
            "new_insights": insights,
            "cycle": cycle + 1,
            # We don't set status here, leaving it to the topology or next node
        }

# --- Workflow Builder ---

class WorkflowBuilder:
    """Builder for constructing the workflow graph"""

    def __init__(self, state_schema=AgentState):
        self.graph = StateGraph(state_schema)

    def add_node(self, name: str, node: BaseNode):
        """Add a processing node"""
        self.graph.add_node(name, node)
        return self

    def add_edge(self, src: str, dst: str):
        """Add a direct edge"""
        self.graph.add_edge(src, dst)
        return self

    def add_conditional(self, src: str, router: Callable, paths: Dict[str, str]):
        """Add a conditional edge"""
        self.graph.add_conditional_edges(src, router, paths)
        return self

    def set_entry_point(self, name: str):
        self.graph.set_entry_point(name)
        return self

    def build(self):
        """Compile the graph"""
        return self.graph.compile()
