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


class CIASAnalystAgent:
    """
    Analyst Agent for CIAS-X system.

    Responsibilities:
    1. Compute Pareto frontiers per strata
    2. Update pareto_frontiers table with top-k per strata (with rank)
    3. Generate plan summary (3-6 sentences with recommendation and trends)
    4. Update global summary every 50 plans (using last_summary_plan_id)
    """

    def __init__(self, llm_client: LLMClient, world_model: CIASWorldModel):
        self.llm_client = llm_client
        self.world_model = world_model
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
                    "config": self._to_serializable(asdict(exp.config)),
                    "metrics": self._to_serializable(asdict(exp.metrics)),
                    "strata": strata
                })
            elif isinstance(exp, dict):
                current_experiments.append(exp)

        # 2. Get existing Pareto frontiers
        existing_frontiers = self.world_model.get_pareto_frontiers()

        # 3. Compute Pareto frontiers per strata with rank
        all_experiments = existing_frontiers + current_experiments
        updated_frontiers = self._compute_stratified_pareto_with_rank(all_experiments)

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
        plan_summary, analysis_used = self._generate_plan_summary(current_experiments, flat_frontiers)

        # 6. Update plan with summary
        latest_plan_id = self.world_model.get_latest_plan_id(design_id)
        if latest_plan_id:
            self.world_model.update_plan_summary(latest_plan_id, plan_summary)
            logger.info(f"Updated plan {latest_plan_id} with summary")

        # 7. Check if global summary needs update
        # Update global summary if threshold reached
        token_used_design = self._try_update_global_summary(design_id)

        # Determine next status
        new_budget = budget_remaining - 1 # each plan - execution - analysis costs 1 budget
        next_status = "planning" if new_budget > 0 else "end"

        self.world_model.append_plan_token_used(plan_id=latest_plan_id, token_used=analysis_used, token_type="analysis")
        self.world_model.append_plan_token_used(plan_id=latest_plan_id, token_used=token_used_design, token_type="global_summary")

        token_remaining = token_remaining - analysis_used - token_used_design
        token_remaining = 0 if token_remaining <= 0 else token_remaining
        logger.info(f"Analyst Agent: Analysis complete. Budget remaining: {new_budget}. Token remaining: {token_remaining}")

        return {
            "pareto_frontiers": flat_frontiers,
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

    def _compute_stratified_pareto_with_rank(self, all_experiments: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Compute Pareto frontiers grouped by strata with rank.

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
            # Sort by PSNR and assign rank
            sorted_pareto = sorted(pareto, key=lambda x: x['metrics'].get('psnr', 0), reverse=True)

            # Take top-k and assign rank
            top_k = []
            for rank, item in enumerate(sorted_pareto[:self.world_model.top_k], start=1):
                top_k.append({
                    "experiment_id": item.get('experiment_id', 0),
                    "rank": rank,
                    "config": item['config'],
                    "metrics": item['metrics']
                })

            result[strata] = top_k

        return result

    def _compute_pareto_front(self, items: List[Dict]) -> List[Dict]:
        """
        Compute Pareto front.
        Objectives: Maximize PSNR, Maximize Coverage, Minimize Latency
        """
        if not items:
            return []

        # Extract vectors: [PSNR, Coverage, -Latency]
        vectors = []
        for item in items:
            m = item.get('metrics', {})
            vectors.append([
                m.get('psnr', 0),
                m.get('coverage', 0),
                -m.get('latency', 99999)
            ])

        vectors = np.array(vectors)
        n = len(vectors)
        is_efficient = np.ones(n, dtype=bool)

        for i in range(n):
            if is_efficient[i]:
                for j in range(n):
                    if i != j and is_efficient[j]:
                        if np.all(vectors[j] >= vectors[i]) and np.any(vectors[j] > vectors[i]):
                            is_efficient[i] = False
                            break

        return [items[i] for i in range(n) if is_efficient[i]]

    def _generate_plan_summary(self, current_experiments: List[Dict], pareto_frontiers: List[Dict]) -> tuple[str, int]:
        """
        Generate plan summary using LLM.
        Summary should be 3-6 sentences and include current experiments summary, recommendations, and trends.
        """
        if not self.llm_client:
            return "LLM unavailable. No summary generated."

        # Format experiments
        exp_text = ""
        for exp in current_experiments[:5]:
            m = exp.get('metrics', {})
            c = exp.get('config', {})
            exp_text += f"- PSNR={m.get('psnr', 0):.2f}dB, Latency={m.get('latency', 0):.1f}ms, "
            fc = c.get('forward_config', {})
            rp = c.get('recon_params', {})
            exp_text += f"CR={fc.get('compression_ratio', 'N/A')}, Stages={rp.get('num_stages', 'N/A')}\n"

        # Format frontiers
        front_text = ""
        for f in pareto_frontiers[:5]:
            m = f.get('metrics', {})
            r = f.get('rank', '?')
            front_text += f"- [Rank {r}, {f.get('strata', '?')}] PSNR={m.get('psnr', 0):.2f}dB, Latency={m.get('latency', 0):.1f}ms\n"

        prompt = f"""Analyze these SCI reconstruction experiment results and provide a comprehensive summary.

## Current Batch Results
{exp_text if exp_text else "No experiments in current batch."}

## Current Pareto Frontier
{front_text if front_text else "No Pareto frontier established yet."}

Write a summary that is 3-6 sentences total and includes:
1. Summary of current experiments in plan scope
2. Recommendations for what configurations to explore next
3. Observed trends or patterns

Output the summary text only (no JSON, no markdown, no bullet points - just a flowing paragraph of 3-6 sentences)."""

        try:
            response = self.llm_client.chat([{"role": "user", "content": prompt}])
            return response['content'].strip(), response['tokens']
        except Exception as e:
            logger.error(f"Plan summary generation failed: {e}")
            return "Summary generation failed.", 0

    def _update_global_summary(self, design_id: int) -> tuple[str, int]:
        """Generate and save updated global summary."""
        plan_id = self.world_model.get_last_summary_plan_id_in_design(design_id)
        plan_id = plan_id if plan_id else 1
        recent_summaries = self.world_model.get_plan_summaries_since(design_id, plan_id)
        old_summary = self.world_model.get_global_summary(design_id)

        if not self.llm_client:
            return "", 0

        # Build prompt
        recent_text = "\n".join([f"- {s}" for s in recent_summaries])

        prompt = f"""Update the global research summary based on recent experiment batches.

## Previous Global Summary
{old_summary if old_summary else "No previous summary (initial phase)."}

## Recent Batch Summaries (last {len(recent_summaries)} batches)
{recent_text if recent_text else "No recent summaries."}

Write a consolidated Global Summary (5-10 sentences) that:
1. Identifies the overall best-performing configurations
2. Notes failed approaches or poor parameter regions
3. Highlights robust trends discovered
4. Suggests promising directions for future exploration

Output the summary text only (no JSON, no markdown)."""

        try:
            response = self.llm_client.chat([{"role": "user", "content": prompt}])
            new_summary = response['content'].strip()

            # Save to DB
            self.world_model.update_global_summary(design_id, new_summary)
            logger.info(f"Updated global summary (new baseline plan_id: {plan_id})")

            return new_summary, response['tokens']
        except Exception as e:
            logger.error(f"Global summary update failed: {e}")
            return old_summary, 0

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
