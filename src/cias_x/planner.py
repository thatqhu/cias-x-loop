"""
CIAS-X Planner Agent

Proposes new experimental configurations using LLM based on:
- Global summary (gaps/underexplored regions)
- Pareto frontiers
- Design space constraints
"""

import json
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.llm.client import LLMClient
from src.cias_x.world_model import CIASWorldModel
from src.cias_x.state import AgentState
from src.cias_x.structures import (
    ForwardConfig,
    ReconParams,
    ReconFamily,
    UQScheme,
    SCIConfiguration,
    TrainConfig
)

logger = logging.getLogger(__name__)


class CIASPlannerAgent:
    """
    Planner Agent for CIAS-X system.

    Generates new experiment configurations based on:
    1. Global summary for identifying under-explored regions
    2. Pareto frontiers for exploitation
    3. Design space for validation
    """

    def __init__(self, llm_client: Optional[LLMClient], world_model: CIASWorldModel, max_configs_per_cycle: int = 3):
        self.llm_client = llm_client
        self.world_model = world_model
        self.name = "Planner"
        self.max_configs_per_cycle = max_configs_per_cycle
        logger.info("CIASPlannerAgent initialized")

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node entry point."""
        return self.plan(state)

    def plan(self, state: AgentState) -> Dict[str, Any]:
        """
        Plan new experiments based on the current state.

        Algorithm from design doc:
        1. Check design_id, create if needed
        2. If no history (first run), generate baseline configs
        3. Otherwise, use LLM to propose new configs based on gaps and frontiers
        4. Validate and project configs into design space
        """
        logger.info("Planner Agent: Starting planning phase")

        # 1. Get or create design_id
        design_id = state.get("design_id", 0)
        if not design_id:
            design_id = self.world_model.get_or_create_design()
            logger.info(f"Created/Retrieved design ID: {design_id}")

        # Get context
        global_summary = state.get("global_summary", "")
        pareto_frontiers = state.get("pareto_frontiers", [])
        design_space = state.get("design_space", {})

        # 2. Generate configs
        if not global_summary and not pareto_frontiers:
            # First time: Generate baseline configs
            logger.info("No history found. Generating baseline configs.")
            new_configs = self._create_baseline_configs(design_space)
        else:
            # Use LLM to generate configs
            logger.info("Using LLM to generate new configs based on history.")
            new_configs = self._llm_generate_configs(
                global_summary, pareto_frontiers, design_space
            )

            if not new_configs:
                logger.warning("LLM returned no configs, falling back to random generation.")
                new_configs = [self._generate_random_config(design_space) for _ in range(3)]

        logger.info(f"Planner Agent: Generated {len(new_configs)} configs")

        return {
            "design_id": design_id,
            "configs": new_configs,
            "status": "executing",
            "executed_experiment_count": state.get("executed_experiment_count", 0) + self.max_configs_per_cycle
        }

    def _build_planner_prompt(
        self,
        global_summary: str,
        pareto_frontiers: List[Dict],
        design_space: Dict
    ) -> str:
        """Build the LLM prompt for config generation."""
        # Format frontiers with rank
        frontier_text = ""
        if pareto_frontiers:
            frontier_text = "## Current Pareto Frontiers (Best Trade-offs)\n"
            for item in pareto_frontiers[:self.world_model.top_k]:
                cfg = item.get('config', {})
                metrics = item.get('metrics', {})
                strata = item.get('strata', 'unknown')
                rank = item.get('rank', '?')
                frontier_text += f"[Rank {rank}, {strata}] PSNR={metrics.get('psnr', 0):.2f}dB, "
                frontier_text += f"Latency={metrics.get('latency', 0):.1f}ms, "
                fc = cfg.get('forward_config', {})
                rp = cfg.get('recon_params', {})
                frontier_text += f"CR={fc.get('compression_ratio', 'N/A')}, "
                frontier_text += f"Stages={rp.get('num_stages', 'N/A')}\n"

        # Format summary
        summary_text = f"## Global Summary\n{global_summary}\n" if global_summary else "## Global Summary\nNo summary yet (initial exploration phase).\n"

        # Format design space
        ds_text = f"## Design Space (Valid Options)\n```json\n{json.dumps(design_space, indent=2)}\n```\n"

        prompt = f"""You are an AI Scientist optimizing SCI (Snapshot Compressive Imaging) reconstruction.

{summary_text}

{frontier_text}

{ds_text}

## Task
Based on the global summary (identifying under-explored regions) and the current Pareto frontiers,
propose {self.max_configs_per_cycle} NEW experiment configurations that:

1. **Explore gaps**: Target under-explored parameter combinations mentioned in the summary
2. **Exploit frontiers**: Refine promising configurations from the Pareto front
3. **Diversify**: Ensure variety in proposed configs

## Output Format
Return ONLY valid JSON with this structure:
{{
    "configs": [
        {{
            "compression_ratio": <int>,
            "mask_type": "<string>",
            "num_stages": <int>,
            "num_features": <int>,
            "num_blocks": <int>,
            "learning_rate": <float>,
            "activation": "<string>",
            "rationale": "<brief reason for this config>"
        }}
    ]
}}

Ensure all values are from the Design Space options."""

        return prompt

    def _llm_generate_configs(
        self,
        global_summary: str,
        pareto_frontiers: List[Dict],
        design_space: Dict
    ) -> List[SCIConfiguration]:
        """Use LLM to generate experiment configurations."""
        if not self.llm_client:
            logger.warning("No LLM client available")
            return []

        prompt = self._build_planner_prompt(global_summary, pareto_frontiers, design_space)

        messages = [
            {"role": "system", "content": "You are an expert AI scientist. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat(messages, response_format="json")
            content = response['content']

            # Parse JSON
            json_content = self._extract_json(content)
            data = json.loads(json_content)
            raw_configs = data.get("configs", [])

            # Convert to SCIConfiguration objects
            valid_configs = []
            for raw in raw_configs:
                config = self._create_config_from_dict(raw, design_space)
                if config:
                    valid_configs.append(config)

            return valid_configs

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return []

    def _extract_json(self, content: str) -> str:
        """Extract JSON from response, handling markdown code blocks."""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            content = "\n".join(lines)
        return content

    def _create_config_from_dict(self, raw: Dict, design_space: Dict) -> Optional[SCIConfiguration]:
        """Create SCIConfiguration from a dictionary."""
        try:
            # Handle both flat and nested structures
            if "forward_config" in raw:
                fc = raw["forward_config"]
                rp = raw.get("recon_params", {})
                tc = raw.get("train_config", {})
            else:
                fc = raw
                rp = raw
                tc = raw

            # Extract values with validation
            cr = int(fc.get("compression_ratio", 16))
            mask = str(fc.get("mask_type", "random"))
            stages = int(rp.get("num_stages", 7))
            features = int(rp.get("num_features", 64))
            blocks = int(rp.get("num_blocks", 3))
            lr = float(rp.get("learning_rate", 1e-4))
            activation = str(rp.get("activation", "ReLU"))
            batch = int(tc.get("batch_size", 4))
            epochs = int(tc.get("num_epochs", 50))

            # Validate against design space
            cr = self._validate_option(cr, design_space.get("compression_ratios", [16]))
            mask = self._validate_option(mask, design_space.get("mask_types", ["random"]))
            stages = self._validate_option(stages, design_space.get("num_stages", [7]))
            features = self._validate_option(features, design_space.get("num_features", [64]))
            blocks = self._validate_option(blocks, design_space.get("num_blocks", [3]))
            lr = self._validate_lr(lr, design_space.get("learning_rates", [1e-4]))
            activation = self._validate_option(activation, design_space.get("activations", ["ReLU"]))

            return SCIConfiguration(
                experiment_id=f"exp_{uuid.uuid4().hex[:8]}",
                forward_config=ForwardConfig(
                    compression_ratio=cr,
                    mask_type=mask,
                    sensor_noise=0.01,
                    resolution=(256, 256),
                    frame_rate=30
                ),
                recon_family=ReconFamily.CIAS_CORE,
                recon_params=ReconParams(
                    num_stages=stages,
                    num_features=features,
                    num_blocks=blocks,
                    learning_rate=lr,
                    use_physics_prior=True,
                    activation=activation
                ),
                uq_scheme=UQScheme.CONFORMAL,
                uq_params={},
                train_config=TrainConfig(
                    batch_size=batch,
                    num_epochs=epochs,
                    optimizer="Adam",
                    scheduler="CosineAnnealing",
                    early_stopping=True
                ),
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.warning(f"Failed to create config: {e}")
            return None

    def _validate_option(self, value, options):
        """Validate value is in options."""
        if value in options:
            return value
        return options[0] if options else value

    def _validate_lr(self, lr: float, options: List) -> float:
        """Validate learning rate."""
        float_options = [float(o) for o in options]
        if lr in float_options:
            return lr
        return min(float_options, key=lambda x: abs(x - lr))

    def _create_baseline_configs(self, design_space: Dict) -> List[SCIConfiguration]:
        """Create baseline configurations for initial exploration."""
        configs = []
        crs = design_space.get("compression_ratios", [8, 16])

        for cr in crs[:2]:
            config = SCIConfiguration(
                experiment_id=f"baseline_cr{cr}",
                forward_config=ForwardConfig(
                    compression_ratio=cr,
                    mask_type="random",
                    sensor_noise=0.01,
                    resolution=(256, 256),
                    frame_rate=30
                ),
                recon_family=ReconFamily.CIAS_CORE,
                recon_params=ReconParams(
                    num_stages=5,
                    num_features=64,
                    num_blocks=3,
                    learning_rate=1e-4,
                    use_physics_prior=True,
                    activation="ReLU"
                ),
                uq_scheme=UQScheme.CONFORMAL,
                uq_params={},
                train_config=TrainConfig(
                    batch_size=4,
                    num_epochs=50,
                    optimizer="Adam",
                    scheduler="CosineAnnealing",
                    early_stopping=True
                ),
                timestamp=datetime.now().isoformat()
            )
            configs.append(config)

        return configs

    def _generate_random_config(self, design_space: Dict) -> SCIConfiguration:
        """Generate a random configuration as fallback."""
        import random

        return SCIConfiguration(
            experiment_id=f"rnd_{uuid.uuid4().hex[:8]}",
            forward_config=ForwardConfig(
                compression_ratio=random.choice(design_space.get("compression_ratios", [16])),
                mask_type=random.choice(design_space.get("mask_types", ["random"])),
                sensor_noise=0.01,
                resolution=(256, 256),
                frame_rate=30
            ),
            recon_family=ReconFamily.CIAS_CORE,
            recon_params=ReconParams(
                num_stages=random.choice(design_space.get("num_stages", [7])),
                num_features=random.choice(design_space.get("num_features", [64])),
                num_blocks=random.choice(design_space.get("num_blocks", [3])),
                learning_rate=random.choice(design_space.get("learning_rates", [1e-4])),
                use_physics_prior=True,
                activation=random.choice(design_space.get("activations", ["ReLU"]))
            ),
            uq_scheme=UQScheme.CONFORMAL,
            uq_params={},
            train_config=TrainConfig(
                batch_size=4,
                num_epochs=50,
                optimizer="Adam",
                scheduler="CosineAnnealing",
                early_stopping=True
            ),
            timestamp=datetime.now().isoformat()
        )
