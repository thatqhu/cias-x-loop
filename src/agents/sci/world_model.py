"""
World Model

Manages experiment database, stores experiment results, metrics and LLM analysis records.
Supports linking analyses to specific experiments.
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict
from enum import Enum
from ...core.world_model_base import WorldModelBase

from loguru import logger

from .structures import ExperimentResult, ConfigHasher


class EnumEncoder(json.JSONEncoder):
    """Custom JSON encoder with Enum serialization support"""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class WorldModel(WorldModelBase):
    """World Model - Manages persistent storage of experiment data"""

    def __init__(self, db_path: str = "world_model.db"):
        """
        Initialize world model

        Args:
            db_path: SQLite database file path
        """
        self.db_path = db_path
        self._init_database()
        logger.info(f"World Model initialized: {db_path}")

    def _init_database(self):
        """Initialize database table structure"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                config_hash TEXT,  -- Added for optimization
                status TEXT NOT NULL,
                api_task_id TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        # Migration: Check if config_hash column exists, if not add it
        try:
            cursor.execute("ALTER TABLE experiments ADD COLUMN config_hash TEXT")
            logger.info("Migrated experiments table: added config_hash column")
        except sqlite3.OperationalError:
            # Column already exists
            pass

        # Create index for config_hash
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_config_hash ON experiments(config_hash)")

        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                experiment_id TEXT PRIMARY KEY,
                psnr REAL,
                ssim REAL,
                coverage REAL,
                latency REAL,
                memory REAL,
                training_time REAL,
                convergence_epoch INTEGER,
                FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        # Pareto front table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pareto_front (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                cycle_number INTEGER,
                objectives_json TEXT,
                llm_verification TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        # LLM analysis records table (with experiment association)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_number INTEGER,
                analysis_type TEXT,
                input_summary TEXT,
                llm_response TEXT,
                conclusions_json TEXT,
                model_name TEXT,
                tokens_used INTEGER,
                timestamp TIMESTAMP
            )
        """)

        # Analysis-Experiment association table (many-to-many relationship)
        # One analysis can involve multiple experiments, and one experiment can be part of multiple analyses
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                experiment_id TEXT NOT NULL,
                role TEXT,  -- 'pareto', 'analyzed', 'recommended', etc.
                FOREIGN KEY(analysis_id) REFERENCES llm_analyses(id),
                FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id),
                UNIQUE(analysis_id, experiment_id)
            )
        """)

        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_analysis_experiments_analysis
            ON analysis_experiments(analysis_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_analysis_experiments_experiment
            ON analysis_experiments(experiment_id)
        """)

        conn.commit()
        conn.close()

    def add_experiment(self, result: ExperimentResult):
        """
        Add experiment result

        Args:
            result: Experiment result object
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Compute config hash
        config_hash = ConfigHasher.compute_hash(result.config)

        cursor.execute("""
            INSERT OR REPLACE INTO experiments
            (experiment_id, config_json, config_hash, status, api_task_id, started_at, completed_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.experiment_id,
            json.dumps(asdict(result.config), cls=EnumEncoder),
            config_hash,
            result.status,
            result.api_task_id,
            result.started_at,
            result.completed_at,
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))

        if result.status == "success":
            cursor.execute("""
                INSERT OR REPLACE INTO metrics
                (experiment_id, psnr, ssim, coverage, latency, memory, training_time, convergence_epoch)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.experiment_id,
                result.metrics.psnr,
                result.metrics.ssim,
                result.metrics.coverage,
                result.metrics.latency,
                result.metrics.memory,
                result.metrics.training_time,
                result.metrics.convergence_epoch
            ))

        conn.commit()
        conn.close()
        logger.info(f"Experiment added: {result.experiment_id} ({result.status})")

    def get_experiments_by_ids(self, experiment_ids: List[str]) -> List[Any]:
        """
        Get specific experiments by ID
        """
        if not experiment_ids:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ','.join(['?'] * len(experiment_ids))
        cursor.execute(f"""
            SELECT e.experiment_id, e.config_json, e.status,
                   m.psnr, m.ssim, m.coverage, m.latency, m.memory, m.training_time
            FROM experiments e
            LEFT JOIN metrics m ON e.experiment_id = m.experiment_id
            WHERE e.experiment_id IN ({placeholders})
        """, experiment_ids)

        results = []
        for row in cursor.fetchall():
            if row[2] == 'success' and row[3] is not None:
                exp = type('Exp', (), {
                    'experiment_id': row[0],
                    'status': row[2],
                    'metrics': type('Metrics', (), {
                        'psnr': row[3],
                        'ssim': row[4],
                        'coverage': row[5],
                        'latency': row[6],
                        'memory': row[7],
                        'training_time': row[8]
                    })(),
                    'config': json.loads(row[1])
                })()
                results.append(exp)

        conn.close()
        return results

    def get_best_experiment(self, metric: str = 'psnr') -> Optional[Any]:
        """
        Get the best experiment based on a metric (max value)
        """
        allowed_metrics = {'psnr', 'ssim', 'coverage'}
        if metric not in allowed_metrics:
            metric = 'psnr'

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT e.experiment_id, e.config_json, e.status,
                   m.psnr, m.ssim, m.coverage, m.latency, m.memory, m.training_time
            FROM experiments e
            JOIN metrics m ON e.experiment_id = m.experiment_id
            WHERE e.status = 'success'
            ORDER BY m.{metric} DESC
            LIMIT 1
        """)

        row = cursor.fetchone()
        conn.close()

        if row:
            return type('Exp', (), {
                'experiment_id': row[0],
                'status': row[2],
                'metrics': type('Metrics', (), {
                    'psnr': row[3],
                    'ssim': row[4],
                    'coverage': row[5],
                    'latency': row[6],
                    'memory': row[7],
                    'training_time': row[8]
                })(),
                'config': json.loads(row[1])
            })()
        return None

    def get_top_experiments(self, limit: int = 5, metric: str = 'psnr') -> List[Any]:
        """
        Get top K experiments based on a metric (max value)
        """
        allowed_metrics = {'psnr', 'ssim', 'coverage'}
        if metric not in allowed_metrics:
            metric = 'psnr'

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT e.experiment_id, e.config_json, e.status,
                   m.psnr, m.ssim, m.coverage, m.latency, m.memory, m.training_time
            FROM experiments e
            JOIN metrics m ON e.experiment_id = m.experiment_id
            WHERE e.status = 'success'
            ORDER BY m.{metric} DESC
            LIMIT ?
        """, (limit,))

        results = []
        for row in cursor.fetchall():
            exp = type('Exp', (), {
                'experiment_id': row[0],
                'status': row[2],
                'metrics': type('Metrics', (), {
                    'psnr': row[3],
                    'ssim': row[4],
                    'coverage': row[5],
                    'latency': row[6],
                    'memory': row[7],
                    'training_time': row[8]
                })(),
                'config': json.loads(row[1])
            })()
            results.append(exp)

        conn.close()
        return results

    def check_config_exists(self, config_hash: str) -> bool:
        """
        Check if a configuration hash already exists in the database.

        Args:
            config_hash: SHA256 hash of the configuration

        Returns:
            True if exists, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT 1 FROM experiments WHERE config_hash = ? LIMIT 1", (config_hash,))
        exists = cursor.fetchone() is not None

        conn.close()
        return exists

    def count_experiments(self) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM experiments")
        result = cursor.fetchone()
        conn.close()
        return result[0]

    def find_pareto_frontier_ids(
        self,
        objectives: Optional[List[Tuple[str, str]]] = None
    ) -> List[str]:
        """
        Find Pareto optimal experiment IDs using SQL directly (Computation Push-down).

        Args:
            objectives: List of tuples (metric_name, direction).
                       direction must be 'max' or 'min'.
                       Example: [('psnr', 'max'), ('latency', 'min')]
                       Defaults to [('psnr', 'max'), ('ssim', 'max')] if None.

        Returns:
            List of experiment IDs on the Pareto frontier.
        """
        return self.find_pareto_frontier_ids_with_filters(objectives, filters=None)

    def find_pareto_frontier_ids_with_filters(
        self,
        objectives: Optional[List[Tuple[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Find Pareto optimal experiment IDs using SQL directly (Computation Push-down),
        optionally filtered by configuration.
        """
        if objectives is None:
            objectives = [('psnr', 'max'), ('ssim', 'max')]

        # Validate columns to prevent SQL injection
        allowed_metrics = {'psnr', 'ssim', 'coverage', 'latency', 'memory', 'training_time'}
        for metric, direction in objectives:
            if metric not in allowed_metrics:
                raise ValueError(f"Invalid metric: {metric}")
            if direction not in ['max', 'min']:
                raise ValueError(f"Invalid direction: {direction}")

        # Build SQL predicates
        # Condition: t2 dominates t1
        # 1. t2 is better or equal in all objectives
        strict_better_clauses = []
        better_or_equal_clauses = []

        for metric, direction in objectives:
            if direction == 'max':
                better_or_equal_clauses.append(f"t2.{metric} >= t1.{metric}")
                strict_better_clauses.append(f"t2.{metric} > t1.{metric}")
            else:  # min
                better_or_equal_clauses.append(f"t2.{metric} <= t1.{metric}")
                strict_better_clauses.append(f"t2.{metric} < t1.{metric}")

        # Build filter clause
        filter_clause = ""
        params = []
        if filters:
            conditions = []
            for path, val in filters.items():
                conditions.append(f"json_extract(e.config_json, '{path}') = ?")
                params.append(val)
            if conditions:
                filter_clause = "AND " + " AND ".join(conditions)

        where_better_or_equal = " AND ".join(better_or_equal_clauses)
        where_strict_better = " OR ".join(strict_better_clauses)

        # Combine: t2 dominates t1 if (all better_eq) AND (at least one strict better)
        domination_condition = f"({where_better_or_equal}) AND ({where_strict_better})"

        query = f"""
            SELECT t1.experiment_id
            FROM metrics t1
            JOIN experiments e ON t1.experiment_id = e.experiment_id
            WHERE e.status = 'success' {filter_clause}
            AND NOT EXISTS (
                SELECT 1
                FROM metrics t2
                JOIN experiments e2 ON t2.experiment_id = e2.experiment_id
                WHERE {domination_condition}
                AND e2.status = 'success'
                {filter_clause.replace('e.config_json', 'e2.config_json')}
            )
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            # We need params twice: once for outer query, once for inner query
            cursor.execute(query, params + params)
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def save_llm_analysis(
        self,
        cycle: int,
        analysis_type: str,
        input_summary: Any,
        llm_response: str,
        conclusions: Dict,
        model_name: str,
        tokens_used: int,
        related_experiment_ids: Optional[List[str]] = None,
        experiment_roles: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Save LLM analysis record with experiment associations

        Args:
            cycle: Cycle number
            analysis_type: Analysis type (pareto_verification, trend_analysis, recommendation)
            input_summary: Input summary
            llm_response: Raw LLM response
            conclusions: Conclusions JSON
            model_name: Model name
            tokens_used: Number of tokens used
            related_experiment_ids: List of experiment IDs involved in this analysis
            experiment_roles: Dict mapping experiment_id to role (e.g., 'pareto', 'analyzed')

        Returns:
            The ID of the inserted analysis record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert analysis record
        cursor.execute("""
            INSERT INTO llm_analyses
            (cycle_number, analysis_type, input_summary, llm_response,
             conclusions_json, model_name, tokens_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cycle,
            analysis_type,
            json.dumps(input_summary) if isinstance(input_summary, dict) else str(input_summary),
            llm_response,
            json.dumps(conclusions),
            model_name,
            tokens_used,
            datetime.now().isoformat()
        ))

        analysis_id = cursor.lastrowid

        # Link experiments to this analysis
        if related_experiment_ids:
            roles = experiment_roles or {}
            for exp_id in related_experiment_ids:
                role = roles.get(exp_id, 'analyzed')
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO analysis_experiments
                        (analysis_id, experiment_id, role)
                        VALUES (?, ?, ?)
                    """, (analysis_id, exp_id, role))
                except Exception as e:
                    logger.warning(f"Failed to link experiment {exp_id} to analysis: {e}")

        conn.commit()
        conn.close()

        exp_count = len(related_experiment_ids) if related_experiment_ids else 0
        logger.info(f"LLM analysis saved: {analysis_type} (cycle {cycle}, {tokens_used} tokens, {exp_count} experiments)")

        return analysis_id

    def get_analyses_for_experiment(self, experiment_id: str) -> List[Dict]:
        """
        Get all LLM analyses that involve a specific experiment

        Args:
            experiment_id: Experiment ID

        Returns:
            List of analysis records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT la.id, la.cycle_number, la.analysis_type, la.conclusions_json,
                   la.model_name, la.tokens_used, la.timestamp, ae.role
            FROM llm_analyses la
            JOIN analysis_experiments ae ON la.id = ae.analysis_id
            WHERE ae.experiment_id = ?
            ORDER BY la.timestamp DESC
        """, (experiment_id,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'analysis_id': row[0],
                'cycle': row[1],
                'type': row[2],
                'conclusions': json.loads(row[3]) if row[3] else {},
                'model': row[4],
                'tokens': row[5],
                'timestamp': row[6],
                'role': row[7]
            })

        conn.close()
        return results

    def get_experiments_for_analysis(self, analysis_id: int) -> List[Dict]:
        """
        Get all experiments involved in a specific analysis

        Args:
            analysis_id: Analysis ID

        Returns:
            List of experiment info with roles
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT e.experiment_id, e.status, ae.role, m.psnr, m.ssim
            FROM experiments e
            JOIN analysis_experiments ae ON e.experiment_id = ae.experiment_id
            LEFT JOIN metrics m ON e.experiment_id = m.experiment_id
            WHERE ae.analysis_id = ?
        """, (analysis_id,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'experiment_id': row[0],
                'status': row[1],
                'role': row[2],
                'psnr': row[3],
                'ssim': row[4]
            })

        conn.close()
        return results

    def get_unique_strata(self, json_paths: List[str]) -> List[Tuple]:
        """
        Get unique combinations of configuration values (strata).

        Args:
            json_paths: List of JSON paths to group by (e.g. ['$.forward_config.compression_ratio'])

        Returns:
            List of tuples (value1, value2, ..., count)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build extract clause
        extracts = [f"json_extract(config_json, '{path}')" for path in json_paths]
        extract_clause = ", ".join(extracts)

        query = f"""
            SELECT {extract_clause}, COUNT(*)
            FROM experiments
            WHERE status = 'success'
            GROUP BY {extract_clause}
            HAVING COUNT(*) > 0
        """

        try:
            cursor.execute(query)
            return cursor.fetchall()
        finally:
            conn.close()

    def get_metrics_statistics(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get statistics for metrics, optionally filtered by configuration values.

        Args:
            filters: Dict mapping JSON paths to values for filtering.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        where_clause = "status='success'"
        params = []

        if filters:
            for path, val in filters.items():
                where_clause += f" AND json_extract(config_json, '{path}') = ?"
                params.append(val)

        query = f"""
            SELECT COUNT(*),
                   AVG(psnr), MAX(psnr), MIN(psnr),
                   AVG(ssim), MAX(ssim), MIN(ssim),
                   AVG(latency), MAX(latency), MIN(latency)
            FROM metrics m
            JOIN experiments e ON m.experiment_id = e.experiment_id
            WHERE {where_clause}
        """

        cursor.execute(query, params)
        row = cursor.fetchone()
        conn.close()

        if not row or row[0] == 0:
            return {
                "count": 0,
                "psnr": {"avg": 0, "max": 0, "min": 0},
                "ssim": {"avg": 0, "max": 0, "min": 0},
                "latency": {"avg": 0, "max": 0, "min": 0}
            }

        return {
            "count": row[0],
            "psnr": {"avg": row[1], "max": row[2], "min": row[3]},
            "ssim": {"avg": row[4], "max": row[5], "min": row[6]},
            "latency": {"avg": row[7], "max": row[8], "min": row[9]}
        }
    def summarize(self) -> Dict[str, Any]:
        """
        Get database statistics summary

        Returns:
            Dictionary containing experiment statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = self.get_metrics_statistics()

        # Get analysis count
        cursor.execute("SELECT COUNT(*) FROM llm_analyses")
        analysis_count = cursor.fetchone()[0]

        conn.close()

        return {
            "total_experiments": stats["count"],
            "total_analyses": analysis_count,
            "psnr_stats": stats["psnr"],
            "ssim_stats": stats["ssim"],
            # Preserve old structure for compatibility
            "latency_stats": stats["latency"]
        }

    def get_historical_analyses(self, limit: int = 10) -> List[Dict]:
        """
        Get historical LLM analysis records

        Args:
            limit: Maximum number of records to return

        Returns:
            List of analysis records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT la.id, la.cycle_number, la.analysis_type, la.conclusions_json,
                   la.model_name, la.tokens_used, la.timestamp,
                   (SELECT COUNT(*) FROM analysis_experiments ae WHERE ae.analysis_id = la.id) as exp_count
            FROM llm_analyses la
            ORDER BY la.timestamp DESC
            LIMIT ?
        """, (limit,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'analysis_id': row[0],
                'cycle': row[1],
                'type': row[2],
                'conclusions': json.loads(row[3]) if row[3] else {},
                'model': row[4],
                'tokens': row[5],
                'timestamp': row[6],
                'experiment_count': row[7]
            })

        conn.close()
        return results

    def save_pareto_front(
        self,
        cycle: int,
        pareto_experiment_ids: List[str],
        analysis_id: Optional[int] = None
    ):
        """
        Save Pareto front experiments for a cycle with their objectives

        Args:
            cycle: Cycle number
            pareto_experiment_ids: List of experiment IDs in the Pareto front
            analysis_id: Optional link to the LLM analysis that verified this front
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for exp_id in pareto_experiment_ids:
            # Get metrics for this experiment
            cursor.execute("""
                SELECT psnr, ssim, latency FROM metrics WHERE experiment_id = ?
            """, (exp_id,))
            row = cursor.fetchone()

            objectives = None
            if row:
                objectives = json.dumps({
                    'psnr': row[0],
                    'ssim': row[1],
                    'latency': row[2]
                })

            cursor.execute("""
                INSERT INTO pareto_front
                (experiment_id, cycle_number, objectives_json, llm_verification, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                exp_id,
                cycle,
                objectives,
                str(analysis_id) if analysis_id else None,
                datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()
        logger.info(f"Pareto front saved: {len(pareto_experiment_ids)} experiments for cycle {cycle}")

    def get_pareto_history(self, limit: int = 10) -> List[Dict]:
        """
        Get Pareto front history grouped by cycle

        Args:
            limit: Maximum number of cycles to return

        Returns:
            List of Pareto front records grouped by cycle with summary stats
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT cycle_number, GROUP_CONCAT(experiment_id),
                   GROUP_CONCAT(objectives_json), MAX(timestamp),
                   COUNT(*) as count
            FROM pareto_front
            GROUP BY cycle_number
            ORDER BY cycle_number DESC
            LIMIT ?
        """, (limit,))

        results = []
        for row in cursor.fetchall():
            exp_ids = row[1].split(',') if row[1] else []
            objectives_list = row[2].split(',') if row[2] else []

            # Parse objectives to get stats
            psnrs = []
            ssims = []
            latencies = []

            for obj_str in objectives_list:
                if obj_str and obj_str != 'None':
                    try:
                        obj = json.loads(obj_str)
                        if obj.get('psnr'):
                            psnrs.append(obj['psnr'])
                        if obj.get('ssim'):
                            ssims.append(obj['ssim'])
                        if obj.get('latency'):
                            latencies.append(obj['latency'])
                    except:
                        pass

            results.append({
                'cycle': row[0],
                'experiment_ids': exp_ids,
                'count': row[4],
                'timestamp': row[3],
                'stats': {
                    'psnr_range': (min(psnrs), max(psnrs)) if psnrs else None,
                    'ssim_range': (min(ssims), max(ssims)) if ssims else None,
                    'latency_range': (min(latencies), max(latencies)) if latencies else None
                }
            })

        conn.close()
        return results

    def get_pareto_detail(self, cycle: int) -> List[Dict]:
        """
        Get detailed Pareto front for a specific cycle

        Args:
            cycle: Cycle number

        Returns:
            List of experiment details in the Pareto front
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT pf.experiment_id, pf.objectives_json, pf.timestamp,
                   e.config_json
            FROM pareto_front pf
            JOIN experiments e ON pf.experiment_id = e.experiment_id
            WHERE pf.cycle_number = ?
        """, (cycle,))

        results = []
        for row in cursor.fetchall():
            objectives = json.loads(row[1]) if row[1] else {}
            config = json.loads(row[3]) if row[3] else {}

            results.append({
                'experiment_id': row[0],
                'objectives': objectives,
                'config': {
                    'compression_ratio': config.get('forward_config', {}).get('compression_ratio'),
                    'mask_type': config.get('forward_config', {}).get('mask_type'),
                    'num_stages': config.get('recon_params', {}).get('num_stages'),
                    'num_features': config.get('recon_params', {}).get('num_features')
                },
                'timestamp': row[2]
            })

        conn.close()
        return results

    def get_experiment_pareto_appearances(self, experiment_id: str) -> List[Dict]:
        """
        Get all cycles where an experiment appeared in the Pareto front

        Args:
            experiment_id: Experiment ID

        Returns:
            List of cycle info where this experiment was in the Pareto front
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT cycle_number, objectives_json, timestamp
            FROM pareto_front
            WHERE experiment_id = ?
            ORDER BY cycle_number
        """, (experiment_id,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'cycle': row[0],
                'objectives': json.loads(row[1]) if row[1] else {},
                'timestamp': row[2]
            })

        conn.close()
        return results

    def update_with_insights(self, insights: Dict[str, Any]):
        """
        Update world model with analysis insights.
        This handles domain-specific saving logic.

        Args:
            insights: Insights dictionary from AnalysisAgent
        """
        # Save Pareto Front
        pareto_ids = insights.get('pareto_front_ids', [])
        cycle = insights.get('cycle', 0)

        if pareto_ids:
            self.save_pareto_front(cycle, pareto_ids)

        # Save LLM Analyses
        llm_analyses = insights.get('llm_analyses', [])
        for analysis in llm_analyses:
            self.save_llm_analysis(
                cycle,
                analysis['type'],
                analysis['prompt'],
                analysis['response'],
                analysis['parsed_result'],
                analysis['model'],
                analysis['tokens'],
                related_experiment_ids=analysis.get('related_ids', []),
                experiment_roles=analysis.get('roles', {})
            )
