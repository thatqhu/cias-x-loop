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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config not found: {config_path}")
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Parse environment variables
    if 'llm' in config and 'api_key' in config['llm']:
        api_key = config['llm']['api_key']
        if api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            config['llm']['api_key'] = os.environ.get(env_var, '')

    return config


async def run_workflow(args):
    """Run the CIAS-X workflow."""
    # Load config
    config = load_config(args.config) if args.config else {}

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

    # Initialize LLM client
    llm_client = None
    if llm_config.get('api_key'):
        try:
            llm_client = LLMClient(llm_config)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")

    # Database path and top-k
    db_path = args.db or config.get('database', {}).get('path', 'cias_x.db')
    top_k = args.top_k or config.get('pareto', {}).get('top_k', 10)

    # planner config
    max_configs_per_cycle = config.get('planner', {}).get('max_configs_per_cycle', 3)
    design_id = config.get('planner', {}).get('design_id', 0)

    # Budget and execution mode
    budget = args.budget or config.get('experiment', {}).get('budget_max', 10)
    execution_mode = "remote" if args.remote else config.get('experiment', {}).get("mock_mode", "mock")
    service_url = args.service_url or config.get('experiment', {}).get('service_url', 'http://localhost:8000')

    # Initialize components
    logger.info(f"Initializing CIAS-X with database: {db_path}, top_k={top_k}")
    world_model = CIASWorldModel(db_path, top_k=top_k)

    planner = CIASPlannerAgent(llm_client, world_model, max_configs_per_cycle)
    executor = CIASExecutorAgent(
        llm_client,
        world_model,
        execution_mode=execution_mode,
        service_url=service_url if execution_mode == "remote" else None
    )
    analyst = CIASAnalystAgent(llm_client, world_model)

    # Create workflow
    app = create_cias_workflow(planner, executor, analyst)

    # Initial state
    initial_state: AgentState = {
        "executed_experiment_count": 0,
        "design_space": design_space,
        "budget_remaining": budget,
        "design_id": design_id,  # Planner will initialize
        "configs": [],
        "experiments": [],
        "pareto_frontiers": [],
        "global_summary": "",
        "status": "planning"
    }

    mode_desc = f"mode={execution_mode}, service={service_url}" if execution_mode == "remote" else f"mode={execution_mode}"
    logger.info(f"Starting CIAS-X workflow (budget={budget}, {mode_desc}, top_k={top_k})")
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
