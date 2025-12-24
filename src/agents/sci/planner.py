"""
Planner Agent - Experiment Planning Agent

Uses LLM to generate experiment configurations based on current state and design space.
Includes deduplication using JSON hash to avoid running duplicate experiments.
"""

import json
import hashlib
import uuid
from typing import List, Dict, Any, Set, Optional
from datetime import datetime
from dataclasses import asdict
from enum import Enum
from ...llm.client import LLMClient
from .world_model import WorldModel

from loguru import logger

from .structures import (
    SCIConfiguration,
    ForwardConfig,
    ReconParams,
    TrainConfig,
    ReconFamily,
    UQScheme,
    ConfigHasher,
)
from ...llm.client import LLMClient
from ...agents.utils import Utils

from ..base import BaseAgent

class PlannerAgent(BaseAgent):
    """LLM-driven experiment planning agent with deduplication"""

    def __init__(self, config: Dict[str, Any], llm_client: LLMClient, world_model: WorldModel):
        """
        Initialize planner agent

        Args:
            llm_client: LLM client
            world_model: World Model instance (for context pulling)
        """
        super().__init__("PlannerAgent", llm_client, world_model)

        self.max_configs_per_cycle = config.get("max_configs_per_cycle", 3)
        self.use_llm = config.get("use_llm", True) and llm_client is not None
        self.max_dedup_retries = config.get("max_dedup_retries", 5)

        logger.info(f"Planner Agent initialized (max {self.max_configs_per_cycle} per cycle, LLM={self.use_llm})")

    def plan_experiments(
        self,
        world_summary: Dict[str, Any],
        design_space: Dict[str, List[Any]],
        budget: int,
        feedback: str = ""
    ) -> List[SCIConfiguration]:
        """
        Plan new experiments using LLM with rich context

        Args:
            world_summary: World model summary
            design_space: Design space definition
            budget: Remaining budget

        Returns:
            List of new experiment configurations (deduplicated)
        """
        # Update existing hashes if provided


        num_configs = min(self.max_configs_per_cycle, budget)
        configs = []

        # Gather rich context from WorldModel
        context = self._gather_planning_context()

        if self.use_llm and self.llm_client:
            # Use LLM to generate configs with rich context
            llm_configs = self._llm_generate_configs(
                world_summary, design_space, num_configs, context, feedback
            )
            configs.extend(llm_configs)

        # Fill remaining with random generation if needed
        remaining = num_configs - len(configs)
        retries = 0
        while len(configs) < num_configs and retries < self.max_dedup_retries * remaining:
            config = self._generate_random_config(design_space)
            if self._is_unique(config, configs):
                configs.append(config)
            retries += 1

        logger.info(f"Planned {len(configs)} unique experiments")
        return configs

    def _gather_planning_context(self) -> Dict[str, Any]:
        """
        Gather rich context for LLM planning

        Returns:
            Context dictionary with Pareto front, insights, recommendations
        """
        context = {
            'pareto_configs': [],
            'best_experiments': [],
            'historical_insights': [],
            'recommendations': []
        }

        if not self.world_model:
            return context

        try:
            # Get Top 5 experiments for context (Exploitation)
            best_exps = self.world_model.get_top_experiments(limit=5, metric='psnr')
            for exp in best_exps:
                context['best_experiments'].append({
                    'id': exp.experiment_id,
                    'psnr': exp.metrics.psnr,
                    'ssim': exp.metrics.ssim,
                    'config': exp.config
                })

            # Get Pareto front details
            pareto_detail = self.world_model.get_pareto_detail(cycle=1)  # Latest cycle
            for exp in pareto_detail[:5]:  # Top 5
                context['pareto_configs'].append({
                    'id': exp['experiment_id'],
                    'objectives': exp['objectives'],
                    'config': exp['config']
                })

            # Get historical LLM insights
            analyses = self.world_model.get_historical_analyses(limit=3)
            for analysis in analyses:
                if analysis['type'] in ['trend_analysis', 'recommendation']:
                    conclusions = analysis.get('conclusions', {})
                    if analysis['type'] == 'trend_analysis':
                        findings = conclusions.get('key_findings', [])
                        if findings:
                            context['historical_insights'].extend(findings[:3])
                    elif analysis['type'] == 'recommendation':
                        suggestions = conclusions.get('config_suggestions', [])
                        if suggestions:
                            context['recommendations'].extend(suggestions[:3])

        except Exception as e:
            logger.warning(f"Failed to gather planning context: {e}")

        return context

    def _is_unique(self, config: SCIConfiguration, current_configs: List[SCIConfiguration]) -> bool:
        """
        Check if configuration is unique (not in DB and not in current batch)

        Args:
            config: Configuration to check
            current_configs: List of currently planned configurations

        Returns:
            True if unique, False if duplicate
        """
        hash_val = ConfigHasher.compute_hash(config)

        # Check against current batch first (to avoid DB query if already redundant)
        for c in current_configs:
            if ConfigHasher.compute_hash(c) == hash_val:
                return False

        # Check against DB
        if self.world_model and self.world_model.check_config_exists(hash_val):
            return False

        return True

    def _llm_generate_configs(
        self,
        world_summary: Dict[str, Any],
        design_space: Dict[str, List[Any]],
        num_configs: int,
        context: Optional[Dict[str, Any]] = None,
        feedback: str = ""
    ) -> List[SCIConfiguration]:
        """
        Use LLM to generate experiment configurations with rich context

        Args:
            world_summary: Current experiment summary
            design_space: Design space definition
            num_configs: Number of configs to generate
            context: Rich context including Pareto front, insights, recommendations

        Returns:
            List of generated configurations
        """
        prompt = self._build_planning_prompt(world_summary, design_space, num_configs, context, feedback)

        messages = [
            {"role": "system", "content": """You are an expert in computational imaging and SCI (Snapshot Compressive Imaging) reconstruction.
Your task is to suggest experiment configurations that will help explore the design space efficiently.
Always respond with valid JSON."""},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat(messages, response_format="json")
            configs = self._parse_llm_response(response['content'], design_space)

            # Filter out duplicates
            unique_configs = []
            for config in configs:
                # We need to pass previously accepted unique_configs to check against them
                if self._is_unique(config, unique_configs):
                    unique_configs.append(config)
                else:
                    logger.debug(f"Skipping duplicate config from LLM")

            logger.info(f"LLM generated {len(unique_configs)} unique configs (from {len(configs)} total)")
            return unique_configs

        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return []

    def _build_planning_prompt(
        self,
        world_summary: Dict[str, Any],
        design_space: Dict[str, List[Any]],
        num_configs: int,
        context: Optional[Dict[str, Any]] = None,
        feedback: str = ""
    ) -> str:
        """Build the prompt for LLM planning with rich context"""

        # Build Pareto front section
        pareto_section = ""
        if context and context.get('pareto_configs'):
            pareto_lines = []
            for pc in context['pareto_configs'][:5]:
                obj = pc.get('objectives', {})
                cfg = pc.get('config', {})

                # Format objectives safely
                psnr_str = f"{obj['psnr']:.2f}" if obj.get('psnr') else "N/A"
                ssim_str = f"{obj['ssim']:.4f}" if obj.get('ssim') else "N/A"

                pareto_lines.append(
                    f"  - {pc['id']}: PSNR={psnr_str}dB, SSIM={ssim_str}, "
                    f"CR={cfg.get('compression_ratio', 'N/A')}, stages={cfg.get('num_stages', 'N/A')}, features={cfg.get('num_features', 'N/A')}"
                )
            if pareto_lines:
                pareto_section = f"""
## Current Pareto Front (Best Trade-off Configurations)
These configurations represent the current best trade-offs between quality and efficiency:
{chr(10).join(pareto_lines)}
"""

        # Build insights section
        insights_section = ""
        if context and context.get('historical_insights'):
            insights = context['historical_insights'][:5]
            if insights:
                insights_section = f"""
## Key Insights from Previous Analysis
{chr(10).join([f'- {insight}' for insight in insights])}
"""

        # Build recommendations section
        recommendations_section = ""
        if context and context.get('recommendations'):
            recs = context['recommendations'][:3]
            if recs:
                rec_lines = []
                for rec in recs:
                    if isinstance(rec, dict):
                        rec_lines.append(f"  - {rec}")
                    else:
                        rec_lines.append(f"  - {rec}")
                if rec_lines:
                    recommendations_section = f"""
## Previous LLM Recommendations (Consider but don't duplicate)
{chr(10).join(rec_lines)}
"""

        # Build feedback section
        feedback_section = ""
        if feedback:
            feedback_section = f"""
## CRITICAL FEEDBACK FROM REVIEWER (MUST ADDRESS)
The previous plan was rejected. You MUST fix the following issues:
{feedback}
"""

        prompt = f"""Based on the current experiment progress, suggest {num_configs} new experiment configurations.

## Current Progress
- Total completed experiments: {world_summary.get('total_experiments', 0)}
- PSNR stats: avg={world_summary.get('psnr_stats', {}).get('avg', 0):.2f}, max={world_summary.get('psnr_stats', {}).get('max', 0):.2f}, min={world_summary.get('psnr_stats', {}).get('min', 0):.2f} dB
- SSIM stats: avg={world_summary.get('ssim_stats', {}).get('avg', 0):.4f}, max={world_summary.get('ssim_stats', {}).get('max', 0):.4f}
{pareto_section}{insights_section}{recommendations_section}{feedback_section}
## Design Space (choose values from these options ONLY)
- compression_ratios: {design_space.get('compression_ratios', [8, 16, 24])}
- mask_types: {design_space.get('mask_types', ['random', 'optimized'])}
- num_stages: {design_space.get('num_stages', [5, 7, 9])}
- num_features: {design_space.get('num_features', [32, 64, 128])}
- num_blocks: {design_space.get('num_blocks', [2, 3, 4])}
- learning_rates: {design_space.get('learning_rates', [1e-4, 5e-5])}
- activations: {design_space.get('activations', ['ReLU', 'LeakyReLU'])}

## Planning Instructions
1. **Exploitation**: Suggest configs similar to Pareto front configs to refine best regions
2. **Exploration**: Test under-explored parameter combinations
3. **Use insights**: Apply learnings from previous analysis
4. **ADDRESS FEEDBACK**: If feedback is provided above, you MUST correct your plan accordingly.
5. **Avoid duplicates**: Don't repeat tested configurations
5. Each configuration must use values from the design space above ONLY

## Response Format
Return ONLY a valid JSON object (no markdown code blocks) with this structure:
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
            "rationale": "<brief explanation of why this config is suggested>"
        }}
    ]
}}"""

        return prompt

    def _parse_llm_response(
        self,
        response_content: str,
        design_space: Dict[str, List[Any]]
    ) -> List[SCIConfiguration]:
        """
        Parse LLM response and create configuration objects

        Args:
            response_content: Raw LLM response
            design_space: Design space for validation

        Returns:
            List of valid configurations
        """
        try:
            # Extract JSON from response (handles markdown code blocks)
            json_content = Utils.extract_json_from_response(response_content)
            logger.debug(f"Extracted JSON: {json_content[:200]}...")

            data = json.loads(json_content)
            configs_data = data.get('configs', [])

            configs = []
            for cfg in configs_data:
                try:
                    # Validate and create configuration
                    config = self._create_config_from_dict(cfg, design_space)
                    if config:
                        configs.append(config)
                except Exception as e:
                    logger.warning(f"Failed to parse config: {e}")
                    continue

            return configs

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return []

    def _create_config_from_dict(
        self,
        cfg: Dict[str, Any],
        design_space: Dict[str, List[Any]]
    ) -> Optional[SCIConfiguration]:
        """
        Create SCIConfiguration from dictionary, validating against design space

        Args:
            cfg: Configuration dictionary from LLM
            design_space: Design space for validation

        Returns:
            Valid SCIConfiguration or None
        """
        # Validate and get values with fallback
        cr = cfg.get('compression_ratio', 16)
        if cr not in design_space.get('compression_ratios', []):
            cr = design_space.get('compression_ratios', [16])[0]

        mask_type = cfg.get('mask_type', 'random')
        if mask_type not in design_space.get('mask_types', []):
            mask_type = design_space.get('mask_types', ['random'])[0]

        num_stages = cfg.get('num_stages', 7)
        if num_stages not in design_space.get('num_stages', []):
            num_stages = design_space.get('num_stages', [7])[0]

        num_features = cfg.get('num_features', 64)
        if num_features not in design_space.get('num_features', []):
            num_features = design_space.get('num_features', [64])[0]

        num_blocks = cfg.get('num_blocks', 3)
        if num_blocks not in design_space.get('num_blocks', []):
            num_blocks = design_space.get('num_blocks', [3])[0]

        # Handle learning_rate - may be string like "5e-5" or float
        lr = cfg.get('learning_rate', 1e-4)
        if isinstance(lr, str):
            try:
                lr = float(lr)
            except ValueError:
                lr = 1e-4

        # Get valid learning rates and ensure they are floats
        valid_lrs = design_space.get('learning_rates', [1e-4])
        valid_lrs_float = []
        for v in valid_lrs:
            if isinstance(v, str):
                try:
                    valid_lrs_float.append(float(v))
                except ValueError:
                    continue
            else:
                valid_lrs_float.append(float(v))

        if not valid_lrs_float:
            valid_lrs_float = [1e-4]

        # Find closest matching learning rate in design space
        if lr not in valid_lrs_float:
            lr = min(valid_lrs_float, key=lambda x: abs(x - lr))

        activation = cfg.get('activation', 'ReLU')
        if activation not in design_space.get('activations', []):
            activation = design_space.get('activations', ['ReLU'])[0]

        forward_config = ForwardConfig(
            compression_ratio=cr,
            mask_type=mask_type,
            sensor_noise=0.01,
            resolution=(256, 256),
            frame_rate=30
        )

        recon_params = ReconParams(
            num_stages=num_stages,
            num_features=num_features,
            num_blocks=num_blocks,
            learning_rate=lr,
            use_physics_prior=True,
            activation=activation
        )

        train_config = TrainConfig(
            batch_size=4,
            num_epochs=50,
            optimizer="Adam",
            scheduler="CosineAnnealing",
            early_stopping=True
        )

        config = SCIConfiguration(
            experiment_id=f"exp_{uuid.uuid4().hex[:8]}",
            forward_config=forward_config,
            recon_family=ReconFamily.CIAS_CORE,
            recon_params=recon_params,
            uq_scheme=UQScheme.CONFORMAL,
            uq_params={},
            train_config=train_config,
            timestamp=datetime.now().isoformat()
        )

        return config

    def _generate_random_config(self, design_space: Dict[str, List[Any]]) -> SCIConfiguration:
        """
        Generate a random experiment configuration (fallback when LLM unavailable)

        Args:
            design_space: Design space definition

        Returns:
            Experiment configuration object
        """
        import random

        forward_config = ForwardConfig(
            compression_ratio=random.choice(design_space["compression_ratios"]),
            mask_type=random.choice(design_space["mask_types"]),
            sensor_noise=0.01,
            resolution=(256, 256),
            frame_rate=30
        )

        recon_params = ReconParams(
            num_stages=random.choice(design_space["num_stages"]),
            num_features=random.choice(design_space["num_features"]),
            num_blocks=random.choice(design_space["num_blocks"]),
            learning_rate=random.choice(design_space["learning_rates"]),
            use_physics_prior=True,
            activation=random.choice(design_space["activations"])
        )

        train_config = TrainConfig(
            batch_size=4,
            num_epochs=50,
            optimizer="Adam",
            scheduler="CosineAnnealing",
            early_stopping=True
        )

        config = SCIConfiguration(
            experiment_id=f"exp_{uuid.uuid4().hex[:8]}",
            forward_config=forward_config,
            recon_family=ReconFamily.CIAS_CORE,
            recon_params=recon_params,
            uq_scheme=UQScheme.CONFORMAL,
            uq_params={},
            train_config=train_config,
            timestamp=datetime.now().isoformat()
        )

        return config


def create_baseline_configs(design_space: Dict[str, List[Any]]) -> List[SCIConfiguration]:
    """
    Create baseline experiment configurations

    Args:
        design_space: Design space definition

    Returns:
        List of baseline configurations
    """
    configs = []
    for i, cr in enumerate([8, 16]):
        config = SCIConfiguration(
            experiment_id=f"baseline_{i+1}",
            forward_config=ForwardConfig(cr, "random", 0.01, (256, 256)),
            recon_family=ReconFamily.CIAS_CORE,
            recon_params=ReconParams(5, 64, 3, 1e-4, True, "ReLU"),
            uq_scheme=UQScheme.CONFORMAL,
            uq_params={},
            train_config=TrainConfig(4, 50, "Adam", "CosineAnnealing", True),
            timestamp=datetime.now().isoformat()
        )
        configs.append(config)
    return configs
