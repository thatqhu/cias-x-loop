"""
SCI Experiment Data Structures

Defines configuration and result structures for SCI reconstruction experiments.
Refactored to use Pydantic for strong typing and LLM structured output.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

from pydantic import BaseModel, Field


class ReconFamily(str, Enum):
    """Reconstruction algorithm family"""
    CIAS_CORE = "CIAS-Core"
    CIAS_PLUS = "CIAS-Plus"
    TRADITIONAL = "Traditional"
    CIAS_CORE_ELP = "CIAS-Core-ELP"


class UQScheme(str, Enum):
    """Uncertainty Quantification scheme"""
    CONFORMAL = "conformal"
    ENSEMBLE = "ensemble"
    BAYESIAN = "bayesian"
    NONE = "none"


class Status(str, Enum):
    """Experiment status"""
    SUCCESS = "success"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"


class ForwardConfig(BaseModel):
    """Forward model configuration"""
    compression_ratio: int = Field(..., description="Compression ratio (e.g., 8, 16)")
    mask_type: str = Field("random", description="Type of mask used (e.g., 'random', 'optimized')")
    sensor_noise: float = Field(0.01, description="Simulated sensor noise level")
    resolution: Tuple[int, int] = Field((256, 256), description="Image resolution (height, width)")
    frame_rate: int = Field(30, description="Video frame rate")


class ReconParams(BaseModel):
    """Reconstruction model parameters"""
    num_stages: int = Field(..., description="Number of unrolling stages", ge=1, le=20)
    num_features: int = Field(..., description="Number of channel features associated with complexity", ge=16, le=256)
    num_blocks: int = Field(..., description="Number of residual blocks per stage", ge=1, le=10)
    learning_rate: float = Field(..., description="Learning rate for training", ge=1e-6, le=1e-2)
    use_physics_prior: bool = Field(True, description="Whether to include physics-based prior")
    activation: str = Field("ReLU", description="Activation function (e.g., 'ReLU', 'LeakyReLU')")


class TrainConfig(BaseModel):
    """Training configuration"""
    batch_size: int = Field(4, description="Training batch size")
    num_epochs: int = Field(50, description="Maximum training epochs")
    optimizer: str = Field("Adam", description="Optimizer name")
    scheduler: str = Field("CosineAnnealing", description="Learning rate scheduler")
    early_stopping: bool = Field(True, description="Enable early stopping")


class SCIConfiguration(BaseModel):
    """Complete SCI experiment configuration"""
    experiment_id: str = Field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    forward_config: ForwardConfig
    recon_family: ReconFamily = Field(default=ReconFamily.CIAS_CORE_ELP)
    recon_params: ReconParams
    uq_scheme: UQScheme = Field(default=UQScheme.CONFORMAL)
    uq_params: Dict[str, Any] = Field(default_factory=dict)
    train_config: TrainConfig = Field(default_factory=TrainConfig)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class Metrics(BaseModel):
    """Experiment metrics"""
    psnr: float = Field(..., description="Peak Signal-to-Noise Ratio (dB)")
    ssim: float = Field(..., description="Structural Similarity Index")
    coverage: float = Field(..., description="Uncertainty coverage")
    latency: float = Field(..., description="Inference latency in ms")
    memory: float = Field(..., description="Peak memory usage in MB")
    training_time: float = Field(..., description="Total training time in seconds")
    convergence_epoch: int = Field(..., description="Epoch at which training converged")


class Artifacts(BaseModel):
    """Experiment artifacts"""
    checkpoint_path: str = ""
    training_log_path: str = ""
    sample_reconstructions: List[str] = Field(default_factory=list)
    figure_scripts: List[str] = Field(default_factory=list)
    metrics_history: Dict[str, List] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    """Complete experiment result"""
    experiment_id: str
    config: SCIConfiguration
    metrics: Metrics
    artifacts: Artifacts
    status: Status
    started_at: str
    completed_at: str
    error_message: Optional[str] = None


# --- Configuration Structures ---

class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.3
    max_tokens: int = 40960

class DesignSpace(BaseModel):
    compression_ratios: List[int] = Field(default_factory=list)
    mask_types: List[str] = Field(default_factory=list)
    recon_families: List[str] = Field(default_factory=list)
    num_stages: List[int] = Field(default_factory=list)
    num_features: List[int] = Field(default_factory=list)
    num_blocks: List[int] = Field(default_factory=list)
    learning_rates: List[float] = Field(default_factory=list)
    activations: List[str] = Field(default_factory=list)

class ExperimentSettings(BaseModel):
    budget_max: int = 10
    max_tokens: int = 40960
    mock_mode: str = "remote"

class PlannerSettings(BaseModel):
    max_configs_per_plan: int = 1
    design_id: int = 0

class ParetoSettings(BaseModel):
    top_k: int = 10

class ExecutorSettings(BaseModel):
    api_base_url: str = "http://localhost:8000"
    timeout: int = 30
    poll_interval: float = 1.0
    max_poll_attempts: int = 300

class DatabaseSettings(BaseModel):
    path: str = "cias-x.db"

class DesignGoalConstraints(BaseModel):
    latency_max: float = 50.0
    compression_ratio_min: int = 16
    psnr_min: float = 28.0

class DesignGoal(BaseModel):
    description: str = "Maximize PSNR and minimize Latency"
    constraints: DesignGoalConstraints = Field(default_factory=DesignGoalConstraints)

class AppConfig(BaseModel):
    """Global Application Configuration"""
    llm: LLMConfig
    design_space: DesignSpace
    experiment: ExperimentSettings
    planner: PlannerSettings
    pareto: ParetoSettings
    executor: ExecutorSettings
    database: DatabaseSettings
    design_goal: Optional[DesignGoal] = None

class PlanEvaluationReport(BaseModel):
    """Structured evaluation report for a plan."""
    # Basic Stats
    avg_psnr: float = 0.0
    max_psnr: float = 0.0
    avg_latency: float = 0.0

    # Compliance (Hard Constraints)
    is_compliant: bool = True
    violations: List[str] = Field(default_factory=list)

    # Tiers (Soft Benchmarks)
    quality_tier: str = "unknown"
    speed_tier: str = "unknown"

    # Attribution
    best_config_summary: str = ""
    best_config_full: Dict[str, Any] = Field(default_factory=dict)
