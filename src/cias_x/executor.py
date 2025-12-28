"""
CIAS-X Executor Agent

Executes experiments and saves results to the database.
Per design doc:
1. Create a new record in plans table
2. Save results to experiments table (configs, metrics, artifacts)
3. Update experiments in graph state

Supports both mock and remote execution modes.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from datetime import datetime
import numpy as np

try:
    import httpx
except ImportError:
    httpx = None

from src.llm.client import LLMClient
from src.cias_x.world_model import CIASWorldModel
from src.cias_x.state import AgentState
from src.cias_x.structures import (
    SCIConfiguration,
    ExperimentResult,
    Metrics,
    Artifacts,
)

logger = logging.getLogger(__name__)


class CIASExecutorAgent:
    """
    Executor Agent for CIAS-X system.

    Responsibilities:
    1. Create a new plan record in the database
    2. Execute experiments (mock or remote via FastAPI service)
    3. Save results to experiments table
    4. Update experiments in graph state
    """

    def __init__(
        self,
        llm_client: LLMClient,
        world_model: CIASWorldModel,
        execution_mode: str = "mock",
        service_url: Optional[str] = None,
        poll_interval: int = 2,
        max_wait_time: int = 300
    ):
        """
        Initialize executor agent.

        Args:
            llm_client: LLM client instance
            world_model: World model instance
            execution_mode: "mock" or "remote"
            service_url: URL of the FastAPI training service (required for remote mode)
            poll_interval: Interval in seconds for polling task status
            max_wait_time: Maximum wait time in seconds for task completion
        """
        self.llm_client = llm_client
        self.world_model = world_model
        self.name = "Executor"
        self.execution_mode = execution_mode
        self.service_url = service_url
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time

        if execution_mode == "remote":
            if not service_url:
                raise ValueError("service_url is required for remote execution mode")
            if httpx is None:
                raise ImportError("httpx is required for remote execution. Install with: pip install httpx")

        logger.info(f"CIASExecutorAgent initialized (mode={execution_mode}, service={service_url or 'N/A'})")

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node entry point."""
        return self.execute(state)

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Execute experiments for the current plan cycle."""
        logger.info(f"Executor Agent: Starting execution phase (mode={self.execution_mode})")

        design_id = state.get("design_id")
        configs = state.get("configs", [])

        # 1. Create plan record
        plan_id = self.world_model.create_plan(design_id)
        logger.info(f"Created Plan ID: {plan_id} for design {design_id}")

        # 2. Execute experiments
        experiments = []
        for config in configs:
            result = self._run_experiment(config)
            experiments.append(result)

            # 3. Save to database
            config_dict = self._to_serializable(asdict(result.config))
            metrics_dict = self._to_serializable(asdict(result.metrics))
            artifacts_dict = self._to_serializable(asdict(result.artifacts))

            exp_id = self.world_model.save_experiment(
                plan_id=plan_id,
                config=config_dict,
                metrics=metrics_dict,
                artifacts=artifacts_dict,
                status=result.status
            )
            logger.debug(f"Saved experiment {result.experiment_id} with DB ID {exp_id}")

        logger.info(f"Executor Agent: Completed {len(experiments)} experiments")
        logger.info(f"Executor Agent: Total executed experiments: {state.get('executed_experiment_count')}")

        # 4. Update state
        return {
            "experiments": experiments,
            "status": "analyzing"
        }

    def _run_experiment(self, config: SCIConfiguration) -> ExperimentResult:
        """Run a single experiment based on execution mode."""
        if self.execution_mode == "mock":
            return self._run_mock_experiment(config)
        elif self.execution_mode == "remote":
            return self._run_remote_experiment(config)
        else:
            raise ValueError(f"Unknown execution mode: {self.execution_mode}")

    def _run_remote_experiment(self, config: SCIConfiguration) -> ExperimentResult:
        """
        Execute experiment via remote FastAPI service.

        Workflow:
        1. Convert config to API format
        2. Submit training request to /train
        3. Poll /tasks/{task_id}/status until complete
        4. Retrieve results from /tasks/{task_id}/result
        """
        started_at = datetime.now().isoformat()

        try:
            # Prepare API request matching TrainRequest model
            api_request = {
                "experiment_id": config.experiment_id,
                "forward_model": {
                    "compression_ratio": config.forward_config.compression_ratio,
                    "mask_type": config.forward_config.mask_type,
                    "sensor_noise": config.forward_config.sensor_noise,
                    "resolution": list(config.forward_config.resolution),
                    "frame_rate": config.forward_config.frame_rate
                },
                "reconstruction": {
                    "family": config.recon_family.value if hasattr(config.recon_family, 'value') else str(config.recon_family),
                    "num_stages": config.recon_params.num_stages,
                    "num_features": config.recon_params.num_features,
                    "num_blocks": config.recon_params.num_blocks,
                    "learning_rate": config.recon_params.learning_rate,
                    "use_physics_prior": config.recon_params.use_physics_prior,
                    "activation": config.recon_params.activation
                },
                "training": {
                    "batch_size": config.train_config.batch_size,
                    "num_epochs": config.train_config.num_epochs,
                    "optimizer": config.train_config.optimizer,
                    "scheduler": config.train_config.scheduler,
                    "early_stopping": config.train_config.early_stopping
                },
                "uncertainty_quantification": {
                    "scheme": config.uq_scheme.value if hasattr(config.uq_scheme, 'value') else str(config.uq_scheme),
                    "params": config.uq_params
                }
            }

            # Submit training task
            with httpx.Client(timeout=30.0) as client:
                logger.info(f"Submitting experiment {config.experiment_id} to {self.service_url}/train")
                response = client.post(
                    f"{self.service_url}/train",
                    json=api_request
                )
                response.raise_for_status()
                task_data = response.json()
                task_id = task_data["task_id"]
                logger.info(f"Task submitted: {task_id}")

                # Poll for completion
                elapsed_time = 0
                while elapsed_time < self.max_wait_time:
                    time.sleep(self.poll_interval)
                    elapsed_time += self.poll_interval

                    # Check status
                    status_response = client.get(f"{self.service_url}/tasks/{task_id}/status")
                    status_response.raise_for_status()
                    status_data = status_response.json()

                    logger.debug(
                        f"Task {task_id} status: {status_data['status']} "
                        f"({status_data['progress']*100:.0f}%) - {status_data['message']}"
                    )

                    if status_data["status"] in ["completed", "failed"]:
                        break
                else:
                    # Timeout
                    logger.warning(f"Task {task_id} timed out after {self.max_wait_time}s")
                    return self._create_failed_result(
                        config, started_at, f"Task timed out after {self.max_wait_time}s", task_id
                    )

                # Get final result
                result_response = client.get(f"{self.service_url}/tasks/{task_id}/result")
                result_response.raise_for_status()
                result_data = result_response.json()

                if result_data["status"] == "failed":
                    logger.error(f"Task {task_id} failed: {result_data.get('error_message')}")
                    return self._create_failed_result(
                        config, started_at, result_data.get("error_message", "Unknown error"), task_id
                    )

                # Parse metrics
                metrics_data = result_data.get("metrics", {})
                metrics = Metrics(
                    psnr=metrics_data.get("psnr", 0),
                    ssim=metrics_data.get("ssim", 0),
                    coverage=metrics_data.get("coverage", 0),
                    latency=metrics_data.get("latency", 0),
                    memory=metrics_data.get("memory", 0),
                    training_time=metrics_data.get("training_time", 0),
                    convergence_epoch=metrics_data.get("convergence_epoch", 0)
                )

                # Parse artifacts
                artifacts_data = result_data.get("artifacts", {})
                artifacts = Artifacts(
                    checkpoint_path=artifacts_data.get("checkpoint_path", ""),
                    training_log_path=artifacts_data.get("training_log_path", ""),
                    sample_reconstructions=artifacts_data.get("sample_reconstructions", []),
                    figure_scripts=artifacts_data.get("figure_scripts", []),
                    metrics_history=artifacts_data.get("metrics_history", {})
                )

                logger.info(
                    f"Task {task_id} completed successfully. "
                    f"PSNR: {metrics.psnr:.2f}dB, SSIM: {metrics.ssim:.4f}"
                )

                return ExperimentResult(
                    experiment_id=config.experiment_id,
                    config=config,
                    metrics=metrics,
                    artifacts=artifacts,
                    status="success",
                    started_at=started_at,
                    completed_at=result_data.get("completed_at", datetime.now().isoformat())
                )

        except Exception as e:
            logger.error(f"Remote execution failed for {config.experiment_id}: {e}")
            return self._create_failed_result(config, started_at, str(e))

    def _create_failed_result(
        self,
        config: SCIConfiguration,
        started_at: str,
        error_message: str,
        task_id: Optional[str] = None
    ) -> ExperimentResult:
        """Create a failed experiment result."""
        return ExperimentResult(
            experiment_id=config.experiment_id,
            config=config,
            metrics=Metrics(
                psnr=0, ssim=0, coverage=0, latency=999,
                memory=0, training_time=0, convergence_epoch=0
            ),
            artifacts=Artifacts(
                checkpoint_path="", training_log_path="",
                sample_reconstructions=[], figure_scripts=[], metrics_history={}
            ),
            status="failed",
            error_message=error_message,
            started_at=started_at,
            completed_at=datetime.now().isoformat()
        )

    def _run_mock_experiment(self, config: SCIConfiguration) -> ExperimentResult:
        """Generate mock experiment results."""
        cr = config.forward_config.compression_ratio
        stages = config.recon_params.num_stages
        features = config.recon_params.num_features

        # Synthetic metric model
        base_psnr = 35.0 - (cr / 2.0) + (stages / 2.0) + (features / 100.0)
        noise = np.random.randn() * 0.5
        psnr = max(15.0, min(45.0, base_psnr + noise))

        ssim = 0.75 + (psnr / 150.0)
        ssim = max(0.7, min(0.99, ssim + np.random.randn() * 0.02))

        latency = 10 + (stages * 5) + (features / 10.0) + np.random.randn() * 2
        memory = 512 + (stages * 100) + (features * 5)
        coverage = 0.90 + np.random.rand() * 0.08

        metrics = Metrics(
            psnr=round(psnr, 2),
            ssim=round(ssim, 4),
            coverage=round(coverage, 4),
            latency=round(max(5, latency), 1),
            memory=int(memory),
            training_time=round(1.0 + np.random.rand(), 2),
            convergence_epoch=int(10 + np.random.randint(0, 20))
        )

        artifacts = Artifacts(
            checkpoint_path=f"/checkpoints/{config.experiment_id}.pth",
            training_log_path=f"/logs/{config.experiment_id}.log",
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

    def _to_serializable(self, obj: Any) -> Any:
        """Recursively convert to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif hasattr(obj, "value"):  # Enum
            return obj.value
        else:
            return obj
