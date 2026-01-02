#!/usr/bin/env python3
"""
CIAS-X: Autonomous AI Scientist for SCI Reconstruction

Main entry point for running the CIAS-X workflow.
"""

import asyncio
import logging
import os
import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.llm.client import LLMClient
from src.cias_x.world_model import CIASWorldModel
from src.cias_x.planner import CIASPlannerAgent
from src.cias_x.executor import CIASExecutorAgent
from src.cias_x.analyst import CIASAnalystAgent
from src.cias_x.workflow import create_cias_workflow
from src.cias_x.state import AgentState
from src.cias_x.structures import AppConfig
from src.cias_x.evaluator import PlanEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()


def load_config(config_path: str = "config/default.yaml") -> AppConfig:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config not found: {config_path}")
        return AppConfig()

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Parse environment variables
    if 'llm' in config and 'api_key' in config['llm']:
        api_key = config['llm']['api_key']
        if api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            config['llm']['api_key'] = os.environ.get(env_var, '')

    try:
        return AppConfig(**config)
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
        return AppConfig()

async def run_workflow(_args):
    """Run the CIAS-X workflow."""
    # Load config
    config = load_config()

    # Design space
    design_space = config.design_space

    # LLM configuration
    llm_config = config.llm

    # Initialize LLM client
    llm_client = None
    if llm_config.api_key:
        try:
            llm_client = LLMClient(llm_config)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")

    # planner config
    max_configs_per_plan = config.planner.max_configs_per_plan
    design_id = config.planner.design_id

    # Budget and execution mode
    budget = config.experiment.budget_max
    execution_mode = "remote" if config.experiment.mock_mode else "local"
    service_url = config.executor.api_base_url
    max_token = config.experiment.max_tokens

    # Initialize components
    logger.info(f"Initializing CIAS-X with database: {config.database.path}, top_k={config.pareto.top_k}")
    world_model = CIASWorldModel(
        db_path=config.database.path,
        top_k=config.pareto.top_k
    )
    llm_client = LLMClient(config.llm)

    # Initialize Evaluator
    evaluator = PlanEvaluator()

    planner_agent = CIASPlannerAgent(llm_client=llm_client, world_model=world_model, max_configs_per_plan=max_configs_per_plan)
    executor_agent = CIASExecutorAgent(llm_client=llm_client, world_model=world_model, execution_mode=execution_mode, service_url=service_url)
    analyst_agent = CIASAnalystAgent(llm_client=llm_client, world_model=world_model, evaluator=evaluator)


    # Create workflow
    app = create_cias_workflow(planner_agent, executor_agent, analyst_agent)

    # Initial state
    initial_state: AgentState = {
        "executed_experiment_count": 0,
        "design_space": design_space,
        "budget_remaining": budget,
        "token_remaining": max_token,
        "top_k": config.pareto.top_k,
        "design_id": design_id,  # Planner will initialize
        "design_goal": config.design_goal,
        "configs": [],
        "experiments": [],
        "pareto_frontiers": [],
        "global_summary": "",
        "status": "planning"
    }

    mode_desc = f"mode={execution_mode}, service={service_url}" if execution_mode == "remote" else f"mode={execution_mode}"
    logger.info(f"Starting CIAS-X workflow (budget={budget}, {mode_desc}, top_k={config.pareto.top_k})")
    logger.info("=" * 60)

    # Run workflow
    try:
        final_state = await app.ainvoke(initial_state)

        # Report results
        logger.info("=" * 60)
        logger.info("CIAS-X Workflow Completed")
        logger.info(f"  Final Status: {final_state.get('status')}")
        logger.info(f"  Total Executed Experiments: {final_state.get('executed_experiment_count', 0)}")
        logger.info(f"  Budget Remaining: {final_state.get('budget_remaining')}")
        logger.info(f"  Token Remaining: {final_state.get('token_remaining')}")

        pareto_frontiers = final_state.get('pareto_frontiers', [])
        logger.info(f"  Pareto Frontiers: {len(pareto_frontiers)}")

        if pareto_frontiers:
            best = max(pareto_frontiers, key=lambda x: x.get('metrics', {}).get('psnr', 0))
            logger.info(f"  Best PSNR: {best['metrics'].get('psnr', 0):.2f}dB")
            logger.info(f"  Best Config Rank: {best.get('rank', '?')}")
            logger.info(f"  Best Config Strata: {best.get('strata', 'N/A')}")

        global_summary = final_state.get('global_summary', '')
        if global_summary:
            logger.info(f"  Global Summary: {global_summary[:200]}...")

        return final_state

    except Exception as e:
        logger.exception(f"Workflow failed: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CIAS-X: Autonomous AI Scientist for SCI Reconstruction"
    )
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--db", type=str, default=None,
                        help="Database path")
    parser.add_argument("--budget", type=int, default=None,
                        help="Experiment budget")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k Pareto frontiers per strata")
    parser.add_argument("--remote", action="store_true",
                        help="Use remote execution mode (calls FastAPI service)")
    parser.add_argument("--service-url", type=str, default=None,
                        help="URL of the FastAPI training service (for remote mode)")
    args = parser.parse_args()

    try:
        asyncio.run(run_workflow(args))
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
