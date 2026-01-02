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
    TrainConfig,
    DesignGoal,
    DesignSpace
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

    def __init__(self, llm_client: Optional[LLMClient], world_model: CIASWorldModel, max_configs_per_plan: int = 3):
        self.llm_client = llm_client
        self.world_model = world_model
        self.name = "Planner"
        self.max_configs_per_plan = max_configs_per_plan
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
        [design_id, global_summary] = self.world_model.get_or_create_design(design_id)
        logger.info(f"Created/Retrieved design ID: {design_id}")

        # Get context
        design_goal = state.get("design_goal", DesignGoal())
        design_space = state.get("design_space", DesignSpace())
        token_remaining = state.get("token_remaining", 0)
        latest_plan_summary = state.get("latest_plan_summary", "")
        if not latest_plan_summary:
            latest_plan_summary = self.world_model.get_latest_plan_summary(design_id)

        pareto_frontiers = state.get("pareto_frontiers", [])
        if not pareto_frontiers:
            pareto_frontiers = self.world_model.get_pareto_frontiers(design_space.recon_families[0])

        # 2. Generate configs
        token_used = 0
        if not pareto_frontiers and not latest_plan_summary:
            # First time: Generate baseline configs
            logger.info("No history found. Generating baseline configs.")
            new_configs = self._create_baseline_configs(design_space)
        else:
            # Use LLM to generate configs
            logger.info("Using LLM to generate new configs based on history.")

            new_configs, token_used = self._llm_generate_configs(
                global_summary=global_summary,
                latest_plan_summary=latest_plan_summary,
                pareto_frontiers=pareto_frontiers,
                design_space=design_space,
                design_goal=design_goal,
                top_k=state.get("top_k", 10)
            )

            if not new_configs:
                logger.warning("LLM returned no configs, falling back to random generation.")
                new_configs = [self._generate_random_config(design_space) for _ in range(3)]

        plan_id = self.world_model.create_plan(design_id)
        self.world_model.append_plan_token_used(plan_id=plan_id, token_used=token_used, token_type="plan")

        logger.info(f"Planner Agent: Generated {len(new_configs)} configs")

        return {
            "design_id": design_id,
            "configs": new_configs,
            "token_remaining": token_remaining - token_used,
            "status": "executing"
        }

    def _build_planner_prompt(
        self,
        global_summary: str,
        latest_plan_summary: str,
        anchors: List[Dict],
        design_space: DesignSpace = DesignSpace(),
        design_goal: DesignGoal = DesignGoal(),
    ) -> str:
        """Build the hierarchical Planner prompt using Global Map + Plan Directive + Anchors."""

        # 1. Goal
        goal_text = f"## 🎯 Design Goal\n{design_goal.description}\n"
        if design_goal.constraints:
            cons = design_goal.constraints
            goal_text += "**Constraints**:\n"
            goal_text += f"- Latency <= {cons.latency_max}ms\n"
            goal_text += f"- Compression Ratio >= {cons.compression_ratio_min}\n"
            goal_text += f"- PSNR >= {cons.psnr_min}dB\n"

        # 2. Strategic Context (Exploration Map)
        strategy_text = "## 🗺️ Strategic Context (Exploration Map)\n"
        if global_summary:
            strategy_text += f"{global_summary}\n"
        else:
            strategy_text += "No history yet. Baseline exploration phase.\n"

        # 3. Tactical Directive (Latest Plan Summary)
        directive_text = "## 🎯 Tactical Directive (Current Mission)\n"
        if latest_plan_summary:
            directive_text += f"**{latest_plan_summary}**\n"
        else:
            directive_text += "Baseline exploration.\n"

        # 4. Anchors
        anchor_text = "## ⚓ Anchor Configurations (Reference Points)\n"
        if anchors:
            for i, anc in enumerate(anchors, 1):
                cfg = anc.get('config', {})
                m = anc.get('metrics', {})
                fc = cfg.get('forward_config', {})
                rp = cfg.get('recon_params', {})

                # Determine status
                lat = m.get('latency', 0)
                cr = fc.get('compression_ratio', 0)
                psnr = m.get('psnr', 0)
                status = "✅ Compliant"
                if design_goal.constraints:
                    if lat > design_goal.constraints.latency_max or cr < design_goal.constraints.compression_ratio_min:
                        status = "❌ Violation"

                anchor_text += f"**Ref {i}** ({status}): "
                anchor_text += f"PSNR={psnr:.1f}dB, Latency={lat:.0f}ms. "
                anchor_text += f"Params: CR={fc.get('compression_ratio')}, Mask={fc.get('mask_type')}, "
                anchor_text += f"Stages={rp.get('num_stages')}, Channels={rp.get('num_features')}\n"
        else:
            anchor_text += "No Pareto points yet. Generate baseline configs.\n"

        # 5. Design Space
        ds_json = json.dumps(design_space.model_dump(), indent=2)

        prompt = f"""You are an Optimization Specialist for SCI reconstruction.

{goal_text}
{strategy_text}
{directive_text}
{anchor_text}

## 🛠️ Task
Propose {self.max_configs_per_plan} NEW configurations that:

**Priority 1 (MUST)**: Follow the Tactical Directive
**Priority 2**: Explore the Gaps mentioned in Strategic Context
**Priority 3**: Apply discovered Patterns to avoid known failures

**Strategy**:
- If Anchor is ✅ Compliant: Tweak slightly to improve (Exploitation)
- If Anchor is ❌ Violation: Fix the specific parameter mentioned in Directive (Repair)
- If Directive suggests new direction: Try it (Exploration)

## Design Space (Allowed Values)
```json
{ds_json}
```

## Output Format
Return valid JSON only:
{{
    "configs": [
        {{
            "compression_ratio": <int>,
            "mask_type": "<str>",
            "num_stages": <int>,
            "num_features": <int>,
            "num_blocks": <int>,
            "learning_rate": <float>,
            "activation": "<str>",
            "rationale": "Following directive: reduced stages to 6 to fix latency."
        }}
    ]
}}
"""
        return prompt

    def _llm_generate_configs(
        self,
        global_summary: str,
        latest_plan_summary: str,
        pareto_frontiers: List[Dict],
        design_space: DesignSpace = DesignSpace(),
        design_goal: DesignGoal = DesignGoal(),
        top_k: int = 10
    ) -> tuple[List[SCIConfiguration], int]:
        """Use LLM to generate experiment configurations."""
        if not self.llm_client:
            logger.warning("No LLM client available")
            return [], 0

        # Select Anchors
        anchors = self._select_anchor_configs(pareto_frontiers, design_goal)

        prompt = self._build_planner_prompt(
            global_summary=global_summary,
            latest_plan_summary=latest_plan_summary,
            anchors=anchors,
            design_space=design_space,
            design_goal=design_goal
        )

        messages = [
            {"role": "system", "content": "You are an expert AI scientist. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat(messages)
            content = response['content']
            token_used = response['tokens']

            # Parse JSON
            # clean strict ```json ... ``` wrapper if present
            content = content.replace("```json", "").replace("```", "").strip()

            data = json.loads(content)
            configs_json = data.get("configs", [])

            new_configs = []
            for cfg in configs_json:
                # Validate against design space (simple check)
                # Map logic... (Assuming LLM follows instructions)

                # Construct SCIConfiguration object
                sci_config = SCIConfiguration(
                    forward_config=ForwardConfig(
                        compression_ratio=cfg.get('compression_ratio', 16),
                        mask_type=cfg.get('mask_type', 'random')
                    ),
                    recon_params=ReconParams(
                        num_stages=cfg.get('num_stages', 9),
                        num_features=cfg.get('num_features', 64),
                        num_blocks=cfg.get('num_blocks', 1),
                        learning_rate=cfg.get('learning_rate', 1e-4) # simplified
                    )
                )
                new_configs.append(sci_config)

            return new_configs, token_used

        except Exception as e:
            logger.error(f"LLM config generation failed: {e}")
            return [], 0

    def _select_anchor_configs(self, pareto_frontiers: List[Dict], design_goal: DesignGoal, top_k: int = 3) -> List[Dict]:
        """
        从 Pareto 前沿中智能选择 Anchors（三阶段策略）。

        Pareto 前沿保持纯粹的数学定义（不预过滤），约束仅在此处应用。

        策略：
        1. Compliant Anchors: 已达标的最优点（Exploitation）
        2. Repairable Anchors: 接近达标的点（Repair）
        3. Diverse Anchors: 不同策略的边界点（Exploration）

        Args:
            pareto_frontiers: 完整的 Pareto 前沿（所有实验）
            design_goal: 设计目标（包含约束）
            top_k: 返回的 Anchor 数量

        Returns:
            精选的 Anchor 配置列表
        """
        if not pareto_frontiers:
            return []

        cons = design_goal.constraints if design_goal else None
        if not cons:
            # 没有约束，直接返回 Top-K Pareto 点（按 PSNR 排序）
            sorted_pareto = sorted(pareto_frontiers, key=lambda x: x.get('metrics', {}).get('psnr', 0), reverse=True)
            return sorted_pareto[:top_k]

        # === 分类 Pareto 点 ===
        compliant = []      # 完全达标
        repairable = []     # 轻微违规（可修复）
        exploratory = []    # 其他（用于探索）

        for item in pareto_frontiers:
            m = item.get('metrics', {})
            cfg = item.get('config', {})

            lat = m.get('latency', 9999)
            psnr = m.get('psnr', 0)
            cr = cfg.get('forward_config', {}).get('compression_ratio', 0)

            # 检查每个约束
            lat_ok = lat <= cons.latency_max
            psnr_ok = psnr >= cons.psnr_min
            cr_ok = cr >= cons.compression_ratio_min

            if lat_ok and psnr_ok and cr_ok:
                # 完全达标
                compliant.append(item)
            elif self._is_repairable(lat, psnr, cr, cons):
                # 轻微违规（可修复）
                repairable.append(item)
            else:
                # 严重违规或用于探索
                exploratory.append(item)

        # === 选择策略 ===
        anchors = []

        # Stage 1: Compliant Anchors（优先级最高）
        if compliant:
            # 按 PSNR 排序，选最好的
            compliant.sort(key=lambda x: x['metrics'].get('psnr', 0), reverse=True)
            anchors.append(compliant[0])
            logger.info(f"Selected Compliant Anchor: PSNR={compliant[0]['metrics'].get('psnr'):.1f}dB")

            # 如果还有空间，选第二好的（增加多样性）
            if len(compliant) > 1 and len(anchors) < top_k:
                anchors.append(compliant[1])

        # Stage 2: Repairable Anchors（次优）
        if len(anchors) < top_k and repairable:
            # 按"修复难度"排序（违规幅度最小的优先）
            repairable.sort(key=lambda x: self._compute_repair_difficulty(x, cons))
            anchors.append(repairable[0])
            logger.info(f"Selected Repairable Anchor: PSNR={repairable[0]['metrics'].get('psnr'):.1f}dB (needs repair)")

        # Stage 3: Diverse Anchors（保底探索）
        if len(anchors) < top_k and exploratory:
            # 优先选择 Rank 1 的点（真正的 Pareto 边界）
            rank1 = [e for e in exploratory if e.get('rank') == 1]

            if rank1:
                # 选择与已有 Anchors Strata 不同的点
                selected_strata = {a.get('strata') for a in anchors}
                for exp in rank1:
                    if exp.get('strata') not in selected_strata:
                        anchors.append(exp)
                        logger.info(f"Selected Diverse Anchor: Strata={exp.get('strata')} (exploration)")
                        break

            # 如果还不够，随便选一个高质量的
            if len(anchors) < top_k:
                exploratory.sort(key=lambda x: x.get('metrics', {}).get('psnr', 0), reverse=True)
                anchors.append(exploratory[0])

        # 如果实在没有任何点，返回空
        if not anchors:
            logger.warning("No suitable anchors found in Pareto frontier!")

        return anchors[:top_k]

    def _is_repairable(self, lat: float, psnr: float, cr: int, cons) -> bool:
        """判断是否属于"可修复"类别（轻微违规）"""
        # 定义"轻微违规"的阈值（例如超出 20% 以内）
        lat_repairable = lat <= cons.latency_max * 1.2
        psnr_repairable = psnr >= cons.psnr_min * 0.85
        cr_repairable = cr >= cons.compression_ratio_min * 0.9

        # 至少一项违规，但所有项都在可修复范围内
        violations = sum([lat > cons.latency_max, psnr < cons.psnr_min, cr < cons.compression_ratio_min])
        return violations > 0 and lat_repairable and psnr_repairable and cr_repairable

    def _compute_repair_difficulty(self, exp: Dict, cons) -> float:
        """计算修复难度（分数越低越容易修复）"""
        m = exp.get('metrics', {})
        cfg = exp.get('config', {})

        lat = m.get('latency', 9999)
        psnr = m.get('psnr', 0)
        cr = cfg.get('forward_config', {}).get('compression_ratio', 0)

        # 计算各项违规的"幅度"
        lat_gap = max(0, lat - cons.latency_max) / cons.latency_max
        psnr_gap = max(0, cons.psnr_min - psnr) / cons.psnr_min
        cr_gap = max(0, cons.compression_ratio_min - cr) / cons.compression_ratio_min

        # PSNR 更难修（权重更高），Latency 相对容易（可以减 stages）
        return lat_gap * 1.0 + psnr_gap * 2.0 + cr_gap * 1.5


    def _extract_json(self, content: str) -> str:
        """Extract JSON from response, handling markdown code blocks."""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            content = "\n".join(lines)
        return content

    def _create_config_from_dict(self, raw: Dict, design_space: DesignSpace = DesignSpace()) -> Optional[SCIConfiguration]:
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
            cr = self._validate_option(cr, design_space.compression_ratios)
            mask = self._validate_option(mask, design_space.mask_types)
            stages = self._validate_option(stages, design_space.num_stages)
            features = self._validate_option(features, design_space.num_features)
            blocks = self._validate_option(blocks, design_space.num_blocks)
            lr = self._validate_lr(lr, design_space.learning_rates)
            activation = self._validate_option(activation, design_space.activations)

            return SCIConfiguration(
                experiment_id=f"exp_{uuid.uuid4().hex[:8]}",
                forward_config=ForwardConfig(
                    compression_ratio=cr,
                    mask_type=mask,
                    sensor_noise=0.01,
                    resolution=(256, 256),
                    frame_rate=30
                ),
                recon_family=ReconFamily.CIAS_CORE_ELP,
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

    def _create_baseline_configs(self, design_space: DesignSpace = DesignSpace()) -> List[SCIConfiguration]:
        """Create baseline configurations for initial exploration."""
        configs = []
        crs = design_space.compression_ratios

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
                recon_family=ReconFamily.CIAS_CORE_ELP,
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

    def _generate_random_config(self, design_space: DesignSpace = DesignSpace()) -> SCIConfiguration:
        """Generate a random configuration as fallback."""
        import random

        return SCIConfiguration(
            experiment_id=f"rnd_{uuid.uuid4().hex[:8]}",
            forward_config=ForwardConfig(
                compression_ratio=random.choice(design_space.compression_ratios),
                mask_type=random.choice(design_space.mask_types),
                sensor_noise=0.01,
                resolution=(256, 256),
                frame_rate=30
            ),
            recon_family=ReconFamily.CIAS_CORE_ELP,
            recon_params=ReconParams(
                num_stages=random.choice(design_space.num_stages),
                num_features=random.choice(design_space.num_features),
                num_blocks=random.choice(design_space.num_blocks),
                learning_rate=random.choice(design_space.learning_rates),
                use_physics_prior=True,
                activation=random.choice(design_space.activations)
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
