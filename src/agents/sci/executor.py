"""
Executor Agent - Experiment Execution Agent

Executes experiments by calling the SCI service API.
Supports both mock mode (local simulation) and real mode (remote API).
Supports both sync and async execution.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

import aiohttp
import requests
import numpy as np
from loguru import logger
from ...llm.client import LLMClient
from .world_model import WorldModel

from .structures import (
    SCIConfiguration,
    ExperimentResult,
    Metrics,
    Artifacts,
)


class TaskStatus(str, Enum):
    """Task status enum matching the service"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


from ..base import BaseAgent


class ExecutorAgent(BaseAgent):
    """Experiment execution agent with async and remote service support"""

    def __init__(self, config: Dict[str, Any], llm_client: LLMClient, world_model: WorldModel):
        """
        Initialize executor agent

        Args:
            llm_client: LLM client
            world_model: World Model instance (for context pulling)
        """
        super().__init__("ExecutorAgent", llm_client, world_model)

        self.mock_mode = config.get('mock', False)
        self.api_base_url = config.get('api_base_url', 'http://localhost:8000')
        self.timeout = config.get('timeout', 30)
        self.poll_interval = config.get('poll_interval', 1.0)
        self.max_poll_attempts = config.get('max_poll_attempts', 300)

        logger.info(f"Executor Agent initialized (mock={self.mock_mode}, api={self.api_base_url})")

    # ==================== Sync Methods ====================

    def run_experiment(self, config: SCIConfiguration) -> ExperimentResult:
        """
        Run a single experiment (sync version)

        Args:
            config: Experiment configuration

        Returns:
            Experiment result
        """
        if self.mock_mode:
            return self._run_local_mock(config)
        else:
            return self._run_remote(config)

    def _run_local_mock(self, config: SCIConfiguration) -> ExperimentResult:
        """Run experiment in local mock mode (sync)"""
        time.sleep(0.2)
        return self._generate_mock_result(config)

    def _run_remote(self, config: SCIConfiguration) -> ExperimentResult:
        """Run experiment via remote SCI service API (sync)"""
        started_at = datetime.now().isoformat()

        try:
            task_id = self._submit_task(config)
            logger.info(f"Task submitted: {task_id} for {config.experiment_id}")

            final_status = self._poll_task_status(task_id)

            if final_status == TaskStatus.COMPLETED:
                result = self._get_task_result(task_id)
                return self._parse_result(config, result, started_at)
            else:
                result = self._get_task_result(task_id)
                return self._create_failed_result(
                    config, task_id, result.get("error_message", "Unknown error"), started_at
                )

        except Exception as e:
            logger.error(f"Remote execution failed: {e}")
            return self._create_failed_result(config, None, str(e), started_at)

    # ==================== Async Methods ====================

    async def run_experiment_async(self, config: SCIConfiguration) -> ExperimentResult:
        """
        Run a single experiment (async version)

        Args:
            config: Experiment configuration

        Returns:
            Experiment result
        """
        if self.mock_mode:
            return await self._run_local_mock_async(config)
        else:
            return await self._run_remote_async(config)

    async def run_experiments_async(self, configs: List[SCIConfiguration]) -> List[ExperimentResult]:
        """
        Run multiple experiments in parallel (async)

        Args:
            configs: List of experiment configurations

        Returns:
            List of experiment results
        """
        tasks = [self.run_experiment_async(config) for config in configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Experiment {configs[i].experiment_id} failed: {result}")
                final_results.append(self._create_failed_result(
                    configs[i], None, str(result), datetime.now().isoformat()
                ))
            else:
                final_results.append(result)

        return final_results

    async def _run_local_mock_async(self, config: SCIConfiguration) -> ExperimentResult:
        """Run experiment in local mock mode (async)"""
        await asyncio.sleep(0.2)
        return self._generate_mock_result(config)

    async def _run_remote_async(self, config: SCIConfiguration) -> ExperimentResult:
        """Run experiment via remote SCI service API (async)"""
        started_at = datetime.now().isoformat()

        try:
            async with aiohttp.ClientSession() as session:
                # 1. Submit training task
                task_id = await self._submit_task_async(session, config)
                logger.info(f"Task submitted: {task_id} for {config.experiment_id}")

                # 2. Poll for completion
                final_status = await self._poll_task_status_async(session, task_id)

                # 3. Get result
                if final_status == TaskStatus.COMPLETED:
                    result = await self._get_task_result_async(session, task_id)
                    return self._parse_result(config, result, started_at)
                else:
                    result = await self._get_task_result_async(session, task_id)
                    return self._create_failed_result(
                        config, task_id, result.get("error_message", "Unknown error"), started_at
                    )

        except Exception as e:
            logger.error(f"Remote execution failed: {e}")
            return self._create_failed_result(config, None, str(e), started_at)

    async def _submit_task_async(self, session: aiohttp.ClientSession, config: SCIConfiguration) -> str:
        """Submit training task (async)"""
        url = f"{self.api_base_url}/train"
        payload = config.to_api_format()

        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
            response.raise_for_status()
            data = await response.json()
            return data["task_id"]

    async def _poll_task_status_async(self, session: aiohttp.ClientSession, task_id: str) -> TaskStatus:
        """Poll task status until completion (async)"""
        url = f"{self.api_base_url}/tasks/{task_id}/status"

        for attempt in range(self.max_poll_attempts):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                    response.raise_for_status()
                    data = await response.json()

                    status = TaskStatus(data["status"])
                    progress = data.get("progress", 0)
                    message = data.get("message", "")

                    if attempt % 10 == 0:
                        logger.debug(f"Task {task_id}: {status.value} ({progress:.0%}) - {message}")

                    if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        return status

                    await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.warning(f"Poll attempt {attempt} failed: {e}")
                await asyncio.sleep(self.poll_interval)

        raise TimeoutError(f"Task {task_id} did not complete within {self.max_poll_attempts} attempts")

    async def _get_task_result_async(self, session: aiohttp.ClientSession, task_id: str) -> Dict[str, Any]:
        """Get task result (async)"""
        url = f"{self.api_base_url}/tasks/{task_id}/result"

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
            response.raise_for_status()
            return await response.json()

    # ==================== Sync Helper Methods (for backward compatibility) ====================

    def _submit_task(self, config: SCIConfiguration) -> str:
        """Submit training task (sync)"""
        url = f"{self.api_base_url}/train"
        payload = config.to_api_format()

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        return response.json()["task_id"]

    def _poll_task_status(self, task_id: str) -> TaskStatus:
        """Poll task status (sync)"""
        url = f"{self.api_base_url}/tasks/{task_id}/status"

        for attempt in range(self.max_poll_attempts):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()

                data = response.json()
                status = TaskStatus(data["status"])
                progress = data.get("progress", 0)
                message = data.get("message", "")

                if attempt % 10 == 0:
                    logger.debug(f"Task {task_id}: {status.value} ({progress:.0%}) - {message}")

                if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    return status

                time.sleep(self.poll_interval)

            except Exception as e:
                logger.warning(f"Poll attempt {attempt} failed: {e}")
                time.sleep(self.poll_interval)

        raise TimeoutError(f"Task {task_id} did not complete within {self.max_poll_attempts} attempts")

    def _get_task_result(self, task_id: str) -> Dict[str, Any]:
        """Get task result (sync)"""
        url = f"{self.api_base_url}/tasks/{task_id}/result"

        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()

        return response.json()

    # ==================== Shared Helper Methods ====================

    def _generate_mock_result(self, config: SCIConfiguration) -> ExperimentResult:
        """Generate mock experiment result"""
        base_psnr = 26.0 + np.random.randn() * 2
        metrics = Metrics(
            psnr=base_psnr,
            ssim=max(0.7, min(0.99, 0.85 + np.random.randn() * 0.05)),
            coverage=0.90,
            latency=max(10, 50 + np.random.randn() * 10),
            memory=2048,
            training_time=1.5,
            convergence_epoch=35
        )

        artifacts = Artifacts(
            checkpoint_path="",
            training_log_path="",
            sample_reconstructions=[],
            figure_scripts=[],
            metrics_history={}
        )

        return ExperimentResult(
            experiment_id=config.experiment_id,
            config=config,
            metrics=metrics,
            artifacts=artifacts,
            status="success",
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat()
        )

    def _create_failed_result(
        self,
        config: SCIConfiguration,
        task_id: Optional[str],
        error_message: str,
        started_at: str
    ) -> ExperimentResult:
        """Create a failed experiment result"""
        return ExperimentResult(
            experiment_id=config.experiment_id,
            config=config,
            metrics=Metrics(0, 0, 0, 0, 0, 0, 0),
            artifacts=Artifacts("", "", [], [], {}),
            status="failed",
            error_message=error_message,
            api_task_id=task_id,
            started_at=started_at,
            completed_at=datetime.now().isoformat()
        )

    def _parse_result(
        self,
        config: SCIConfiguration,
        result: Dict[str, Any],
        started_at: str
    ) -> ExperimentResult:
        """Parse remote service result into ExperimentResult"""
        metrics_data = result.get("metrics", {})
        metrics = Metrics(
            psnr=metrics_data.get("psnr", 0),
            ssim=metrics_data.get("ssim", 0),
            coverage=metrics_data.get("coverage", 0),
            latency=metrics_data.get("latency", 0),
            memory=metrics_data.get("memory", 0),
            training_time=metrics_data.get("training_time", 0),
            convergence_epoch=metrics_data.get("convergence_epoch", 0)
        )

        artifacts_data = result.get("artifacts", {})
        artifacts = Artifacts(
            checkpoint_path=artifacts_data.get("checkpoint_path", ""),
            training_log_path=artifacts_data.get("training_log_path", ""),
            sample_reconstructions=artifacts_data.get("sample_reconstructions", []),
            figure_scripts=artifacts_data.get("figure_scripts", []),
            metrics_history=artifacts_data.get("metrics_history", {})
        )

        return ExperimentResult(
            experiment_id=config.experiment_id,
            config=config,
            metrics=metrics,
            artifacts=artifacts,
            status="success",
            api_task_id=result.get("task_id"),
            started_at=started_at,
            completed_at=result.get("completed_at", datetime.now().isoformat())
        )

    def check_service_health(self) -> bool:
        """Check if the remote service is healthy"""
        try:
            response = requests.get(
                f"{self.api_base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
