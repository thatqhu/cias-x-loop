"""
CIAS-X Analyst Agent

Analyzes experiment results and updates Pareto frontiers.
Per design doc:
1. Combine current experiments with pareto_frontiers to compute new Pareto frontier
2. Save updated frontiers to pareto_frontiers table (grouped by strata with rank)
3. Use LLM to generate plan summary (3-6 sentences, includes recommendation and trends)
4. Check if 50 plans executed since last_summary_plan_id → update global_summary
"""

import logging
from typing import List, Dict, Any
from dataclasses import asdict
import numpy as np

from src.llm.client import LLMClient
from src.cias_x.world_model import CIASWorldModel
from src.cias_x.state import AgentState
from src.cias_x.structures import ExperimentResult

logger = logging.getLogger(__name__)


from src.cias_x.evaluator import PlanEvaluator

class CIASAnalystAgent:
    """
    Analyst Agent for CIAS-X system.

    Responsibilities:
    1. Compute Pareto frontiers per strata
    2. Update pareto_frontiers table with top-k per strata (with rank)
    3. Generate plan summary (3-6 sentences with recommendation and trends)
    4. Update global summary every 50 plans (using last_summary_plan_id)
    """

    def __init__(self, llm_client: LLMClient, world_model: CIASWorldModel, evaluator: PlanEvaluator):
        self.llm_client = llm_client
        self.world_model = world_model
        self.evaluator = evaluator
        self.name = "Analyst"
        self.global_summary_interval = 10
        logger.info("CIASAnalystAgent initialized")

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node entry point."""
        return self.analyze(state)

    def analyze(self, state: AgentState) -> Dict[str, Any]:
        """Analyze experiments and update frontiers."""
        logger.info("Analyst Agent: Starting analysis phase")

        design_id = state.get("design_id")
        executed_experiment_count = state.get("executed_experiment_count", 0)
        experiments = state.get("experiments", [])
        budget_remaining = state.get("budget_remaining", 0)
        token_remaining = state.get("token_remaining", 0)

        # 1. Convert experiments to dict format
        current_experiments = []
        for exp in experiments:
            if isinstance(exp, ExperimentResult):
                # Determine strata from config
                strata = exp.config.recon_family.value if hasattr(exp.config.recon_family, 'value') else str(exp.config.recon_family)

                current_experiments.append({
                    "experiment_id": exp.experiment_id,
                    "config": self._to_serializable(exp.config.model_dump()),
                    "metrics": self._to_serializable(exp.metrics.model_dump()),
                    "strata": strata
                })
            elif isinstance(exp, dict):
                current_experiments.append(exp)

        # 2. Get existing Pareto frontiers
        existing_frontiers = self.world_model.get_pareto_frontiers()

        # 3. Compute Pareto frontiers per strata with rank
        all_experiments = existing_frontiers + current_experiments
        updated_frontiers = self._compute_stratified_pareto_with_rank(all_experiments, top_k=state.get("top_k", 10))

        # 4. Update pareto_frontiers table
        for strata, top_configs in updated_frontiers.items():
            self.world_model.update_pareto_frontiers(strata, top_configs)
            logger.info(f"Updated {len(top_configs)} Pareto frontiers for strata '{strata}'")

        # Flatten for state
        flat_frontiers = []
        for strata, configs in updated_frontiers.items():
            for c in configs:
                c['strata'] = strata
                flat_frontiers.append(c)

        # 5. Generate LLM plan summary (3-6 sentences with recommendation and trends)
        # First, run automated evaluation
        design_goal = state.get("design_goal")
        report = self.evaluator.evaluate(current_experiments, design_goal)

        # Then generate narrative
        plan_summary, analysis_used = self._generate_plan_summary(current_experiments, flat_frontiers, report)

        # 6. Update plan with summary
        latest_plan_id = self.world_model.get_latest_plan_id(design_id)
        if latest_plan_id:
            self.world_model.update_plan_summary(latest_plan_id, plan_summary)
            logger.info(f"Updated plan {latest_plan_id} with summary")


        # 7. Check if global summary needs update
        # Update global summary if threshold reached
        token_used_design = self._try_update_global_summary(design_id)

        self.world_model.append_plan_token_used(plan_id=latest_plan_id, token_used=analysis_used, token_type="analysis")
        self.world_model.append_plan_token_used(plan_id=latest_plan_id, token_used=token_used_design, token_type="global_summary")

        new_budget = budget_remaining - len(current_experiments)

        token_remaining = token_remaining - analysis_used - token_used_design
        token_remaining = 0 if token_remaining <= 0 else token_remaining
        logger.info(f"Analyst Agent: Analysis complete. Budget remaining: {new_budget}. Token remaining: {token_remaining}")

        # Determine next status
        next_status = "planning" if new_budget > 0 and token_remaining > 0 else "end"

        # Apply top_k filter for display/Planner usage (database has full Pareto)
        top_k = state.get("top_k", 10)
        flat_frontiers_for_display = flat_frontiers[:top_k] if len(flat_frontiers) > top_k else flat_frontiers

        logger.info(f"Returning {len(flat_frontiers_for_display)} Pareto points to Planner (full set: {len(flat_frontiers)})")

        return {
            "pareto_frontiers": flat_frontiers_for_display,  # Only top_k for Planner
            "latest_plan_summary": plan_summary,
            "budget_remaining": new_budget,
            "token_remaining": token_remaining,
            "status": next_status
        }

    def _try_update_global_summary(self, design_id: int) -> int:
        plans_since_last = self.world_model.get_plan_count_since(design_id)
        total_plan_counts = self.world_model.count_plans(design_id)

        init_scope = plans_since_last == 0 or (total_plan_counts < self.global_summary_interval and plans_since_last in [5, 10, 20, 35])
        after_scope = plans_since_last >= self.global_summary_interval
        if init_scope or after_scope:
            logger.info(f"Triggering global summary update ({plans_since_last} plans since last update)")
            _, token_used_design = self._update_global_summary(design_id)
            return token_used_design
        return 0

    def _compute_stratified_pareto_with_rank(self, all_experiments: List[Dict], top_k: int = 10) -> Dict[str, List[Dict]]:
        """
        Compute Pareto frontiers grouped by strata with rank.

        Note: Stores COMPLETE Pareto frontier (no truncation).
        The top_k parameter is only used for display purposes in the return value.

        Returns: Dict[strata, List[{experiment_id, rank, config, metrics}]>
        """
        # Group by strata
        grouped = {}
        for exp in all_experiments:
            strata = exp.get('strata', 'default')
            if strata not in grouped:
                grouped[strata] = []
            grouped[strata].append(exp)

        # Compute Pareto per group with rank
        result = {}
        for strata, exps in grouped.items():
            pareto = self._compute_pareto_front(exps)

            # Sort by PSNR and assign rank to ALL Pareto points
            sorted_pareto = sorted(pareto, key=lambda x: x['metrics'].get('psnr', 0), reverse=True)

            # Assign rank to ALL points (not just top_k)
            ranked_list = []
            for rank, item in enumerate(sorted_pareto, start=1):
                ranked_list.append({
                    "experiment_id": item.get('experiment_id', 0),
                    "rank": rank,
                    "config": item['config'],
                    "metrics": item['metrics']
                })

            result[strata] = ranked_list  # Store ALL Pareto points

        return result

    def _compute_pareto_front(self, items: List[Dict]) -> List[Dict]:
        """
        Compute Pareto front for SCI reconstruction.

        Objectives (3D):
        - Maximize PSNR (reconstruction quality)
        - Maximize Coverage (spatial coverage metric)
        - Minimize Latency (inference speed)

        A point is Pareto-optimal if no other point dominates it:
        - Dominates: better in all objectives AND strictly better in at least one
        """
        if not items:
            return []

        # Extract objective vectors: [PSNR, Coverage, -Latency]
        # Note: Latency is negated so all objectives are "maximize"
        vectors = []
        for item in items:
            m = item.get('metrics', {})
            psnr = m.get('psnr', 0)
            coverage = m.get('coverage', 0)
            latency = m.get('latency', 99999)

            vectors.append([
                psnr,       # Objective 1: Maximize PSNR
                coverage,   # Objective 2: Maximize Coverage
                -latency    # Objective 3: Minimize Latency (negated)
            ])

        vectors = np.array(vectors)
        n = len(vectors)
        is_efficient = np.ones(n, dtype=bool)

        # Check for dominated points
        for i in range(n):
            if is_efficient[i]:
                for j in range(n):
                    if i != j and is_efficient[j]:
                        # j dominates i if: j >= i in all objectives AND j > i in at least one
                        if np.all(vectors[j] >= vectors[i]) and np.any(vectors[j] > vectors[i]):
                            is_efficient[i] = False
                            break

        return [items[i] for i in range(n) if is_efficient[i]]

    def _generate_plan_summary(
        self,
        current_experiments: List[Dict],
        pareto_frontiers: List[Dict],
        report: Any = None
    ) -> tuple[str, int]:
        """
        Generate a ONE-SENTENCE tactical directive for the next Planner.
        """
        if not self.llm_client:
            return "Baseline exploration.", 0

        # Format top 3 experiments
        exp_text = ""
        for i, exp in enumerate(current_experiments[:3], 1):
            m = exp.get('metrics', {})
            c = exp.get('config', {})
            rp = c.get('recon_params', {})
            exp_text += f"{i}. PSNR={m.get('psnr', 0):.1f}dB, Latency={m.get('latency', 0):.0f}ms, Stages={rp.get('num_stages', '?')}\n"

        # Evaluation context
        status = "PASSED" if (report and report.is_compliant) else "FAILED"
        violations = ", ".join(report.violations) if (report and report.violations) else "None"
        best_config = report.best_config_summary if report else "N/A"

        prompt = f"""You are the Lead Scientist. Give ONE tactical directive for the next planner.

## Evaluation Report
- Status: {status}
- Violations: {violations}
- Best Config: {best_config}

## Current Batch (Top 3)
{exp_text if exp_text else "No experiments."}

## Task
Output EXACTLY ONE sentence in this format:
"[Action] [Parameter] to [Goal]. [Optional: Avoid X.]"

Examples (GOOD):
✓ "Reduce num_stages to <8 to meet latency constraint."
✓ "Test mask_type='optimized' to break PSNR plateau at 30dB."
✓ "Increase compression_ratio to 24 while keeping stages<7."

Examples (BAD):
✗ "The experiments show that latency is high..." (Too verbose)
✗ "Consider trying different parameters." (Too vague)

Output:"""

        try:
            response = self.llm_client.chat([{"role": "user", "content": prompt}])
            return response['content'].strip(), response['tokens']
        except Exception as e:
            logger.error(f"Plan summary generation failed: {e}")
            return "Summary generation failed.", 0

    def _update_global_summary(self, design_id: int) -> tuple[str, int]:
        """Generate and save updated global summary as an Exploration Map."""
        if not self.llm_client:
            return "", 0

        # Get all Pareto frontiers for this design
        pareto_configs = self.world_model.get_all_pareto_frontiers(design_id)

        # Get design space from state (we'll need to pass it in analyze method)
        # For now, use a default or retrieve from DB if stored
        design_space_dict = self.world_model.get_design_space(design_id)

        if not pareto_configs:
            return "No experiments completed yet.", 0

        # Format Pareto configs for LLM
        config_text = ""
        for i, item in enumerate(pareto_configs[:20], 1):  # Limit to top 20
            cfg = item.get('config', {})
            m = item.get('metrics', {})
            fc = cfg.get('forward_config', {})
            rp = cfg.get('recon_params', {})
            config_text += f"{i}. CR={fc.get('compression_ratio')}, Mask={fc.get('mask_type')}, "
            config_text += f"Stages={rp.get('num_stages')}, Feat={rp.get('num_features')} "
            config_text += f"→ PSNR={m.get('psnr', 0):.1f}dB, Lat={m.get('latency', 0):.0f}ms\n"

        # Format design space
        ds_text = ""
        if design_space_dict:
            ds_text = f"""
Design Space:
- CRs: {design_space_dict.get('compression_ratios', [])}
- Masks: {design_space_dict.get('mask_types', [])}
- Stages: {design_space_dict.get('num_stages', [])}
- Features: {design_space_dict.get('num_features', [])}
"""

        prompt = f"""You are the Project Director reviewing {len(pareto_configs)} completed experiments.

## Tested Configurations (Pareto Frontier)
{config_text}
{ds_text}

## Task
Create an Exploration Map with 3 sections (2-3 sentences each):

1. **Exploited** (Well-tested regions):
   - Which parameter combinations are thoroughly explored?
   - Example: "CR=16 fully tested with stages 5-12."

2. **Gaps** (Unexplored valid regions):
   - Which valid Design Space combinations are untested?
   - Example: "CR=24 never tried. mask='optimized' unexplored."

3. **Patterns** (Discovered rules):
   - Any reliable cause-effect relationships?
   - Example: "stages>10 always violates latency<50ms."

Format:
Exploited: [2-3 sentences]
Gaps: [2-3 sentences]
Patterns: [1-2 rules]

Output:"""

        try:
            response = self.llm_client.chat([{"role": "user", "content": prompt}])
            new_summary = response['content'].strip()

            # Save to DB
            self.world_model.update_global_summary(design_id, new_summary)

            # Update last summary plan ID
            latest_plan_id = self.world_model.get_latest_plan_id(design_id)
            if latest_plan_id:
                self.world_model.update_last_summary_plan_id(design_id, latest_plan_id)

            logger.info(f"Updated global summary (Exploration Map)")

            return new_summary, response['tokens']
        except Exception as e:
            logger.error(f"Global summary update failed: {e}")
            return "", 0

    def _to_serializable(self, obj: Any) -> Any:
        """Recursively convert to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif hasattr(obj, "value"):
            return obj.value
        else:
            return obj
