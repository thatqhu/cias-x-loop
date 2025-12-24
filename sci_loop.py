#!/usr/bin/env python3
"""
AI Scientist for SCI - Main Entry Point
"""

import os
import argparse
from pathlib import Path

import yaml
from loguru import logger
from dotenv import load_dotenv
from src.llm.client import LLMClient

# Load environment variables from .env file
load_dotenv()

import asyncio
import asyncio
# from src.core.workflow_graph import create_workflow # Moved local
from src.core.workflow_graph import (
    WorkflowBuilder,
    PlannerNode,
    ReviewerNode,
    ExecutorNode,
    PersistenceNode,
    AnalyzerNode
)
from src.core.state import AgentState
from src.agents.sci.world_model import WorldModel
from src.agents.sci.planner import PlannerAgent
from src.agents.sci.executor import ExecutorAgent
from src.agents.sci.analysis import AnalysisAgent
from src.agents.sci.reviewer import PlanReviewerAgent
try:
    from langgraph.graph import END
except ImportError:
    END = "END"


def create_workflow(planner, reviewer, executor, analyzer):
    """
    Creates the AI Scientist workflow topology
    """
    builder = WorkflowBuilder()

    # 1. Register Nodes
    builder.add_node("planner", PlannerNode(planner))
    builder.add_node("reviewer", ReviewerNode(reviewer))
    builder.add_node("executor", ExecutorNode(executor))
    # Persistence uses Planner's WM reference (shared singleton)
    builder.add_node("persistence", PersistenceNode(planner))
    builder.add_node("persistence_insights", PersistenceNode(planner)) # Second instance for insights
    builder.add_node("analyst", AnalyzerNode(analyzer))

    # 2. Define Topology
    builder.set_entry_point("planner")

    builder.add_edge("planner", "reviewer")
    builder.add_edge("executor", "persistence")
    builder.add_edge("persistence", "analyst")
    builder.add_edge("analyst", "persistence_insights")

    # 3. Define Logic
    def should_continue_review(state: AgentState):
        if state["status"] == "planning_retry":
            return "retry"
        return "execute"

    def should_continue_cycle(state: AgentState):
        if state["budget_remaining"] <= 0:
            logger.info("Budget exhausted. Stopping workflow.")
            return "end"
        return "continue"

    builder.add_conditional("reviewer", should_continue_review, {
        "retry": "planner",
        "execute": "executor"
    })

    builder.add_conditional("persistence_insights", should_continue_cycle, {
        "continue": "planner",
        "end": END
    })

    return builder.build()


def load_config(config_path: str) -> dict:
    """Load configuration file"""
    if not Path(config_path).exists():
        logger.warning(f"Config not found: {config_path}")
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Parse environment variables
    if 'llm' in config and 'api_key' in config['llm']:
        api_key = config['llm']['api_key']
        if api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            config['llm']['api_key'] = os.environ.get(env_var, '')

    return config


async def run_loop(args):
    """Async Main Loop"""
    config = load_config(args.config)

    # Design space
    design_space = config.get('design_space', {
        "compression_ratios": [8, 16, 24],
        "mask_types": ["random", "optimized"],
        "num_stages": [5, 7, 9],
        "num_features": [32, 64, 128],
        "num_blocks": [2, 3, 4],
        "learning_rates": [1e-4, 5e-5],
        "activations": ["ReLU", "LeakyReLU"]
    })

    # LLM configuration
    llm_config = config.get('llm', {
        'base_url': 'https://api.openai.com/v1',
        'api_key': os.environ.get('OPENAI_API_KEY', ''),
        'model': 'gpt-4-turbo-preview',
    })
    llm_client = LLMClient(llm_config)

    # Experiment settings
    exp_cfg = config.get('experiment', {})
    budget_max = args.budget or exp_cfg.get('budget_max', 20)
    max_cycles = args.cycles or exp_cfg.get('max_cycles', 5)
    mock_mode = args.mock or exp_cfg.get('mock_mode', True)
    db_path = config.get('database', {}).get('path', 'world_model_v3.db')

    # Initialize components
    world_model = WorldModel(db_path)

    planner = PlannerAgent(
        config=config.get('planner', {}),
        world_model=world_model,
        llm_client=llm_client
    )

    # Executor config (merge with mock mode setting)
    executor_config = config.get('executor', {})
    executor_config['mock'] = mock_mode
    executor = ExecutorAgent(executor_config, llm_client=llm_client, world_model=world_model)

    analyzer = AnalysisAgent(llm_client=llm_client, world_model=world_model)
    reviewer = PlanReviewerAgent(
        design_space=design_space,
        llm_client=llm_client,
        world_model=world_model
    )

    # --- Construct Workflow Graph ---
    logger.info("Initializing AI Scientist Workflow Graph...")
    app = create_workflow(planner, reviewer, executor, analyzer)

    if app is None:
        logger.error("Failed to initialize LangGraph workflow (missing dependency?)")
        return

    # --- Run Workflow ---
    logger.info(f"Starting Research Loop (Budget: {budget_max}, Cycles: {max_cycles})...")

    # Initial State
    initial_state = {
        "cycle": 1,
        "design_space": design_space,
        "budget_remaining": budget_max,
        "current_plan": [],
        "plan_feedback": "",
        "experiments": [],
        "last_batch_results": [],
        "insights": {},
        "new_insights": {},
        "status": "planning"
    }

    try:
        final_state = await app.ainvoke(initial_state)

        # Results Analysis
        logger.info(f"\nWorklow Completed (Status: {final_state.get('status')})")
        logger.info(f"Budget Remaining: {final_state.get('budget_remaining')}")

        insights = final_state.get('insights', {})
        if 'pareto_front_ids' in insights:
             logger.info(f"Pareto Front Size: {len(insights['pareto_front_ids'])}")

        if 'trends' in insights:
            logger.info(f"Findings: {insights['trends'].get('key_findings', [])[:3]}")

    except Exception as e:
        logger.exception(f"Workflow execution failed: {e}")



def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AI Scientist for SCI v3.0 (LangGraph Edition)")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--mock", action="store_true", help="Mock mode")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--cycles", type=int, default=None)
    args = parser.parse_args()

    try:
        asyncio.run(run_loop(args))
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")


if __name__ == "__main__":
    main()
