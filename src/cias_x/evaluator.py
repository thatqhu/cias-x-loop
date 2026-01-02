"""
CIAS-X Plan Evaluator

Responsible for evaluating experiment results against design goals and benchmarks.
Produces structured reports for Analyst to save in Vector DB or report to users.
"""

from typing import List, Dict, Any, Optional
import logging

from src.cias_x.structures import (
    DesignGoal,
    PlanEvaluationReport
)

logger = logging.getLogger(__name__)


class PlanEvaluator:
    """
    Evaluates a batch of experiments (a Plan) against goals and benchmarks.
    """

    def __init__(self):
        logger.info("PlanEvaluator initialized (Stateless)")

    def evaluate(self, experiments: List[Dict], design_goal: Optional[DesignGoal] = None) -> PlanEvaluationReport:
        """
        Evaluate a list of experiment results.

        Args:
            experiments: List of dicts (metrics, config, etc.)
            design_goal: Optional design goal with constraints

        Returns:
            PlanEvaluationReport populated with stats and compliance.
        """
        if not experiments:
            return PlanEvaluationReport()

        # 1. Basic Stats
        psnr_list = [e.get('metrics', {}).get('psnr', 0.0) for e in experiments]
        latency_list = [e.get('metrics', {}).get('latency', 0.0) for e in experiments]

        if not psnr_list:
            return PlanEvaluationReport()

        avg_psnr = float(sum(psnr_list) / len(psnr_list))
        max_psnr = float(max(psnr_list))
        avg_latency = float(sum(latency_list) / len(latency_list))

        # 2. Attribution (Find Best Config)
        # We classify "Best" primarily by PSNR for now
        best_idx = psnr_list.index(max_psnr)
        best_exp = experiments[best_idx]
        best_config_full = best_exp.get('config', {})

        # Create a readable summary of params
        fc = best_config_full.get('forward_config', {})
        rp = best_config_full.get('recon_params', {})
        best_config_summary = (
            f"CR={fc.get('compression_ratio')}, Mask={fc.get('mask_type')}, "
            f"Stages={rp.get('num_stages')}, Feat={rp.get('num_features')}"
        )

        # 3. Compliance Check (Hard Constraints)
        is_compliant = True
        violations = []

        if design_goal and design_goal.constraints:
            cons = design_goal.constraints

            # Check Average Latency against Max
            if avg_latency > cons.latency_max:
                is_compliant = False
                violations.append(f"latency_avg({avg_latency:.1f}) > max({cons.latency_max})")

            # Check Minimum Compression Ratio (using Best Config's CR)
            best_cr = fc.get('compression_ratio', 0)
            if best_cr < cons.compression_ratio_min:
                is_compliant = False
                violations.append(f"cr_best({best_cr}) < min({cons.compression_ratio_min})")

            # Check PSNR Floor
            if max_psnr < cons.psnr_min:
                is_compliant = False
                violations.append(f"psnr_max({max_psnr:.1f}) < min({cons.psnr_min})")

        return PlanEvaluationReport(
            avg_psnr=avg_psnr,
            max_psnr=max_psnr,
            avg_latency=avg_latency,
            is_compliant=is_compliant,
            violations=violations,
            quality_tier="evaluated", # Placeholder
            speed_tier="evaluated",   # Placeholder
            best_config_summary=best_config_summary,
            best_config_full=best_config_full
        )
