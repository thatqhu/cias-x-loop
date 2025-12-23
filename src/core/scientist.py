"""
AI Scientist Main Loop

Orchestrates Planner, Executor, and Analyzer for automated scientific exploration.
Supports async parallel execution of experiments.
"""

import asyncio
from typing import List, Dict, Any, Tuple
from loguru import logger
import time

from ..agents.base import BaseAgent
from .bus import MessageBus, Event
from .world_model_base import WorldModelBase

class AIScientist:
    """
    AI Scientist Main Loop (Research Director)

    Acts as the central orchestrator (Stage 1 Director) that coords:
    Planner -> Reviewer -> Executor -> Analyzer
    via Message Bus.
    """

    def __init__(
        self,
        world_model: WorldModelBase,
        planner: BaseAgent,
        executor: BaseAgent,
        analyzer: BaseAgent,
        reviewer: BaseAgent,
        bus: MessageBus,
        design_space: Dict[str, Any],
        budget_max: int
    ):
        """
        Initialize AI Scientist

        Args:
            world_model: World model for storing experiment history
            planner: Planner agent
            executor: Executor agent
            analyzer: Analysis agent
            reviewer: Reviewer agent
            bus: Message Bus
            design_space: Design space
            budget_max: Maximum experiment budget
        """
        self.world_model = world_model
        self.planner = planner
        self.executor = executor
        self.analyzer = analyzer
        self.reviewer = reviewer
        self.bus = bus
        self.design_space = design_space
        self.budget_max = budget_max

        # Subscribe to Events
        self.bus.subscribe("EXPERIMENT_COMPLETED", self._on_experiment_completed)
        self.bus.subscribe("PLAN_APPROVED", self._on_plan_approved)
        self.bus.subscribe("PLAN_REJECTED", self._on_plan_rejected)
        self.bus.subscribe("INSIGHT_GENERATED", self._on_insight_generated)

        logger.info(f"AI Scientist (Director) initialized: budget={budget_max}")

        # State tracking
        self.current_cycle = 0
        self.expected_experiments = 0
        self.completed_experiments_cycle = 0
        self.max_cycles = 5
        self.plan_retries = 0

        self.stop_signal = asyncio.Event()
        self.final_pareto = []
        self.final_insights = {}

    async def _on_plan_approved(self, event: Event):
        """Track approved plans to know when cycle ends"""
        configs = event.payload.get('configs', [])
        count = len(configs)
        self.expected_experiments += count
        logger.info(f"Director: Expecting {count} new experiments (Total pending: {self.expected_experiments - self.completed_experiments_cycle})")

    async def _on_plan_rejected(self, event: Event):
        """Handle rejected plans - Retry logic"""
        logger.warning(f"Director: Plan rejected. Feedback: {event.payload.get('feedback')}")

        if self.plan_retries < 2:
            self.plan_retries += 1
            logger.info(f"Retrying planning (Attempt {self.plan_retries + 1})...")
            # Trigger replanning
            budget_remaining = self.budget_max - self.world_model.count_experiments
            await self.bus.publish(Event(
                "PLAN_REQUESTED",
                {"budget": min(3, budget_remaining), "design_space": self.design_space, "cycle": self.current_cycle},
                sender="Director"
            ))
        else:
            logger.error("Max plan retries reached. Skipping cycle.")
            # Triggers analysis which will eventually trigger insight and next cycle
            await self.bus.publish(Event(
                "ANALYSIS_REQUESTED",
                {"trigger_analysis": True, "cycle": self.current_cycle},
                sender="Director"
            ))

    async def _on_experiment_completed(self, event: Event):
        """Handle experiment completion and check for cycle end"""
        result = event.payload.get("result")
        if result:
            self.world_model.add_experiment(result)
            self.completed_experiments_cycle += 1

            total_done = self.world_model.count_experiments
            logger.info(f"Progress: {self.completed_experiments_cycle}/{self.expected_experiments} in cycle (Total: {total_done})")

            # Check if current batch is done
            if self.completed_experiments_cycle >= self.expected_experiments and self.expected_experiments > 0:
                logger.info("All planned experiments completed. Triggering Analysis.")
                await self.bus.publish(Event(
                    "ANALYSIS_REQUESTED",
                    {"trigger_analysis": True, "cycle": self.current_cycle},
                    sender="Director"
                ))
            elif total_done >= self.budget_max:
                logger.info("Budget exhausted. Forcing Analysis.")
                await self.bus.publish(Event(
                    "ANALYSIS_REQUESTED",
                    {"trigger_analysis": True, "cycle": self.current_cycle},
                    sender="Director"
                ))

    async def _on_insight_generated(self, event: Event):
        """Handle analysis results, save to WorldModel, and decide next step"""
        insights = event.payload.get("insights", {})
        pareto = event.payload.get("pareto_ids", [])
        cycle = event.payload.get("cycle", self.current_cycle)

        self.final_insights = insights
        self.final_pareto = pareto

        # Save Analysis Data to World Model
        try:
            if hasattr(self.world_model, 'update_with_insights'):
                self.world_model.update_with_insights(insights)
            else:
                logger.warning("WorldModel does not support update_with_insights")
        except Exception as e:
            logger.error(f"Error updating WorldModel with insights: {e}")

        budget_used = self.world_model.count_experiments
        logger.info(f"Cycle {self.current_cycle} Analysis Complete. Budget: {budget_used}/{self.budget_max}")

        if budget_used < self.budget_max and self.current_cycle < self.max_cycles:
            self._start_next_cycle(budget_used)
        else:
            logger.info("Research Goal Met or Budget Exhausted. Stopping.")
            self.stop_signal.set()

    def _start_next_cycle(self, budget_used: int):
        """Start a new research cycle"""
        self.current_cycle += 1
        self.plan_retries = 0
        # Reset cycle counters
        self.expected_experiments = 0
        self.completed_experiments_cycle = 0

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Cycle {self.current_cycle}/{self.max_cycles}")
        logger.info(f"{'='*60}")

        budget_remaining = self.budget_max - budget_used

        # Trigger Planning asynchronously
        asyncio.create_task(self.bus.publish(Event(
            "PLAN_REQUESTED",
            {"budget": min(3, budget_remaining), "design_space": self.design_space, "cycle": self.current_cycle},
            sender="Director"
        )))

    async def run_async(
        self,
        initial_configs: List[Any],
        max_cycles: int = 5
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Run AI Scientist main loop with async parallel execution (Event-Driven)
        """
        logger.info(f"AI Scientist starting (async): budget={self.budget_max}, cycles={max_cycles}")

        self.max_cycles = max_cycles
        self.current_cycle = 0

        # Initialization
        self.expected_experiments = 0

        logger.info(f"Running {len(initial_configs)} initial experiments...")
        await self.bus.publish(Event(
            "PLAN_APPROVED",
            {"configs": initial_configs},
            sender="Director"
        ))

        # Wait until finished
        await self.stop_signal.wait()

        logger.info(f"\n{'='*60}")
        logger.info(f"Run Complete!")

        return self.final_pareto, self.final_insights
