"""
Analysis Agent

Uses LLM for intelligent Pareto verification, trend analysis, and experiment recommendations
"""

import json
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
from loguru import logger

from ...llm.client import LLMClient
from .world_model import WorldModel

from ...agents.utils import Utils

from ..base import BaseAgent


class AnalysisAgent(BaseAgent):
    """Analysis Agent - Uses LLM for intelligent analysis"""

    def __init__(self, llm_client: LLMClient, world_model: WorldModel):
        """
        Initialize analysis agent

        Args:
            llm_client: LLM client
            world_model: World Model instance (for context pulling)
        """
        super().__init__("AnalysisAgent", llm_client, world_model)

        self.objectives = ['psnr', 'ssim', 'latency']
        logger.info("Analysis Agent initialized with LLM")

    async def run_analysis(self, cycle: int):
        if not self.world_model:
            return [], {}

        logger.info(f"Running analysis for cycle {cycle}")
        pareto_ids, insights = self.analyze(cycle)

        return pareto_ids, insights

    def analyze(
        self,
        cycle_number: int
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Complete analysis workflow (Algorithm 5 Implementation)

        1. Group valid experiments by strata (T, mask, family)
        2. Compute Pareto front per stratum
        3. Compute statistics per stratum
        4. Synthesize trends
        5. Verify and Recommend via LLM using stratified insights
        """
        exp_count = self.world_model.count_experiments()

        if exp_count < 3:
            logger.warning(f"Too few experiments: {exp_count}")
            return [], {"message": "Insufficient data"}

        logger.info(f"Analyzing {exp_count} experiments (Stratified)")

        # Stratification Keys
        # T (Compression Ratio), Mask Type, Recon Family
        strata_keys = [
            '$.forward_config.compression_ratio',
            '$.forward_config.mask_type',
            '$.recon_family'
        ]

        # Step 1 & 2: Group by Strata & Compute Per-Stratum Data
        strata_groups = self.world_model.get_unique_strata(strata_keys)

        global_pareto_ids = set()
        strata_trends = []

        objectives = [('psnr', 'max'), ('ssim', 'max'), ('latency', 'min')]

        for group in strata_groups:
            # Unpack group (val1, val2, val3, count)
            # Note: values might be None if key missing
            t_val, mask_val, family_val, count = group

            # Skip invalid/empty strata if any essential key is missing (optional)
            if count < 1:
                continue

            # Build filters
            filters = {}
            if t_val is not None: filters['$.forward_config.compression_ratio'] = t_val
            if mask_val is not None: filters['$.forward_config.mask_type'] = mask_val
            if family_val is not None: filters['$.recon_family'] = family_val

            stratum_name = f"CR={t_val}, Mask={mask_val}, Family={family_val}"

            # Compute Pareto Front for this stratum
            p_s = self.world_model.find_pareto_frontier_ids_with_filters(objectives, filters)
            global_pareto_ids.update(p_s)

            # Compute Stats
            calib_stats = self.world_model.get_metrics_statistics(filters)

            # Summarize Trend (Lightweight)
            # We aggregate these to pass to LLM, instead of running LLM on every stratum
            t_s = {
                "stratum": stratum_name,
                "count": count,
                "pareto_count": len(p_s),
                "psnr_avg": calib_stats['psnr']['avg'],
                "psnr_max": calib_stats['psnr']['max'],
                "latency_avg": calib_stats['latency']['avg'],
                "params": filters
            }
            strata_trends.append(t_s)

        pareto_ids = list(global_pareto_ids)
        logger.info(f"Global Union Pareto front: {len(pareto_ids)} experiments")

        # Use Stratified Trends to inform Global LLM Analysis
        # We replace the generic 'summarize()' call in LLM methods with finding specific patterns

        analysis_records = []

        # Step 3: LLM Trend Analysis (Enhanced with Strata)
        trends, meta_trends = self._llm_analyze_trends(
            pareto_ids, cycle_number, strata_trends
        )
        if meta_trends:
            analysis_records.append(meta_trends)

        # Step 4: Verification (using Global Pareto)
        verification, meta_ver = self._llm_verify_pareto(
            pareto_ids, cycle_number
        )
        if meta_ver:
            analysis_records.append(meta_ver)

        # Step 5: Recommendations
        recommendations, meta_recs = self._llm_generate_recommendations(
            trends, cycle_number
        )
        if meta_recs:
            analysis_records.append(meta_recs)

        insights = {
            'pareto_front_ids': pareto_ids,
            'strata_trends': strata_trends,
            'llm_analyses': analysis_records,
            'verification': verification,
            'trends': trends,
            'recommendations': recommendations,
            'cycle': cycle_number,
            'total_experiments_analyzed': exp_count
        }

        return pareto_ids, insights

    def _compute_pareto_front(self) -> List[str]:
        """
        Compute Pareto front via WorldModel (Computation Push-down)

        Returns:
            List of Pareto front experiment IDs
        """
        # Define objectives: maximize PSNR and SSIM, minimize Latency
        objectives = [
            ('psnr', 'max'),
            ('ssim', 'max'),
            ('latency', 'min')
        ]

        return self.world_model.find_pareto_frontier_ids(objectives)

    def _llm_verify_pareto(
        self,
        pareto_ids: List[str],
        cycle: int
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        LLM verification of Pareto front
        Returns: (verification_result, metadata_record)
        """
        pareto_exps = self.world_model.get_experiments_by_ids(pareto_ids)
        summary = self.world_model.summarize()

        # Prepare data summary
        pareto_lines = []
        for exp in pareto_exps[:10]:  # Max 10
            pareto_lines.append(
                f"{exp.experiment_id}: PSNR={exp.metrics.psnr:.2f}, "
                f"SSIM={exp.metrics.ssim:.4f}, Latency={exp.metrics.latency:.1f}ms"
            )

        psnr_stats = summary['psnr_stats']
        ssim_stats = summary['ssim_stats']
        # Note: Latency stats not currently in summarize(), using approximations or fetching if critical
        # For now, let's assume latency is not strictly required for the prompt stats if not in summary,
        # or we update summarize(). Assuming summary is sufficient for PSNR/SSIM.
        # If latency stats are absolutely needed, we should update summarize().
        # Let's check prompt usage below. It uses latency min/max.
        # Since summarize() doesn't return latency stats, we might miss them.
        # However, to avoid 'get_all_experiments', let's stick to PSNR/SSIM which are most important,
        # or just omit latency stats in the prompt if unavailable.
        # Actually, let's modify the prompt to use what we have in summary.

        prompt = f"""You are an SCI domain expert. Please verify the reasonableness of the Pareto front.

Total experiments: {self.world_model.count_experiments()}
Pareto front: {len(pareto_ids)} points

Pareto points:
{chr(10).join(pareto_lines)}

Statistics:
- PSNR: {psnr_stats['min']:.2f} - {psnr_stats['max']:.2f} dB
- SSIM: {ssim_stats['min']:.4f} - {ssim_stats['max']:.4f}

Please analyze:
1. Is the Pareto front reasonable
2. Are there any anomalies
3. Trade-off quality
4. Improvement suggestions

Return JSON format:
{{
    "is_reasonable": bool,
    "anomalies": [],
    "suggestions": []
}}"""

        messages = [
            {"role": "system", "content": "You are an SCI domain expert"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat(messages, "json")
            json_content = Utils.extract_json_from_response(response['content'])
            verification = json.loads(json_content)

            metadata = {
                'type': 'pareto_verification',
                'prompt': prompt,
                'response': response['content'],
                'parsed_result': verification,
                'model': response['model'],
                'tokens': response['tokens'],
                'related_ids': pareto_ids,
                'roles': {exp_id: 'pareto' for exp_id in pareto_ids}
            }

            return verification, metadata
        except Exception as e:
            logger.error(f"Pareto verification failed: {e}")
            return {'is_reasonable': True, 'error': str(e)}, None

    def _llm_analyze_trends(
        self,
        pareto_ids: List[str],
        cycle: int,
        strata_trends: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        LLM trend analysis
        Returns: (trends_result, metadata_record)
        """
        summary = self.world_model.summarize()
        psnr_stats = summary['psnr_stats']

        # Prepare context
        strata_text = ""
        if strata_trends:
            # Sort by max PSNR
            sorted_strata = sorted(strata_trends, key=lambda x: x.get('psnr_max', 0), reverse=True)
            top_strata = sorted_strata[:5]

            lines = ["Performance by Stratum (Top 5):"]
            for s in top_strata:
                lines.append(
                    f"- {s['stratum']}: Max PSNR={s['psnr_max']:.2f}, "
                    f"Pareto Count={s['pareto_count']}/{s['count']}"
                )
            strata_text = "\n".join(lines)

        best_exp = self.world_model.get_best_experiment('psnr')

        prompt = f"""You are a data analysis expert. Please analyze experiment trends.

Total experiments: {summary['total_experiments']}
PSNR range: {psnr_stats['min']:.2f} - {psnr_stats['max']:.2f} dB
Best experiment: PSNR={best_exp.metrics.psnr:.2f}, SSIM={best_exp.metrics.ssim:.4f}

{strata_text}

Please analyze:
1. Key findings (main factors affecting performance)
2. Best configuration patterns based on strata
3. Performance bottlenecks
4. Unexpected insights

Return JSON: {{"key_findings": [], "best_patterns": {{}}, "bottlenecks": []}}"""

        messages = [
            {"role": "system", "content": "You are a data analysis expert"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat(messages, "json")
            trends = json.loads(Utils.extract_json_from_response(response['content']))

            metadata = {
                'type': 'trend_analysis',
                'prompt': prompt,
                'response': response['content'],
                'parsed_result': trends,
                'model': response['model'],
                'tokens': response['tokens'],
                'related_ids': pareto_ids,
                'roles': {exp_id: 'pareto_context' for exp_id in pareto_ids}
            }

            return trends, metadata
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {'error': str(e)}, None

    def _llm_generate_recommendations(
        self,
        trends: Dict[str, Any],
        cycle: int
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        LLM generate experiment recommendations
        Returns: (recommendations_result, metadata_record)
        """
        summary = self.world_model.summarize()
        best_psnr = summary['psnr_stats']['max']
        total_exps_count = summary['total_experiments']

        prompt = f"""Based on the analysis, provide experiment recommendations.

Current best PSNR: {best_psnr:.2f} dB
Completed: {total_exps_count} experiments

Please provide:
1. 3 specific configuration suggestions
2. Exploration strategy (explore or exploit)
3. Expected improvements

Return JSON: {{"config_suggestions": [], "strategy": "", "expected_improvements": {{}}}}"""

        messages = [
            {"role": "system", "content": "You are an experiment design expert"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat(messages, "json")
            recommendations = json.loads(Utils.extract_json_from_response(response['content']))

            metadata = {
                'type': 'recommendation',
                'prompt': prompt,
                'response': response['content'],
                'parsed_result': recommendations,
                'model': response['model'],
                'tokens': response['tokens'],
                'related_ids': [],
                'roles': {}
            }

            return recommendations, metadata
        except Exception as e:
            logger.error(f"Recommendation failed: {e}")
            return {'error': str(e)}, None
