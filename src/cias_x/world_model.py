"""
CIAS-X World Model

SQLite-based persistent storage for experiments, plans, and Pareto frontiers.
Implements the database schema as specified in the CIAS-X design document.
"""

import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class CIASWorldModel:
    """
    World Model for CIAS-X system.

    Database Schema:
    - designs: Design sessions with global summaries
    - plans: Experiment batches within a design
    - experiments: Individual experiment results
    - pareto_frontiers: Top-k Pareto frontier configs per strata (with rank)
    """

    def __init__(self, db_path: str = "cias_x.db",
        top_k: int = 5):
        self.db_path = db_path
        self.top_k = top_k
        self._init_db()
        logger.info(f"CIASWorldModel initialized with database: {db_path}, top_k={top_k}")

    # ... (other methods) ...

    @contextmanager
    def _get_conn(self):
        """Context manager for database connections with WAL mode."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # Designs Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS designs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    global_summary TEXT,
                    last_summary_plan_id INTEGER,
                    token_used INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Plans Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    design_id INTEGER NOT NULL,
                    summary TEXT,
                    token_total_used INTEGER DEFAULT 0,
                    token_plan_used INTEGER DEFAULT 0,
                    token_analysis_used INTEGER DEFAULT 0,
                    token_global_summary_used INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(design_id) REFERENCES designs(id)
                )
            """)

            # Experiments Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    plan_id INTEGER NOT NULL,
                    config JSON NOT NULL,
                    metrics JSON NOT NULL,
                    artifacts JSON DEFAULT '{}',
                    status TEXT DEFAULT 'completed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(plan_id) REFERENCES plans(id)
                )
            """)

            # Pareto Frontiers Table (renamed from optimal_configs)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pareto_frontiers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    strata TEXT NOT NULL,
                    config JSON NOT NULL,
                    metrics JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                )
            """)

            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_plans_design ON plans(design_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_plan ON experiments(plan_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pareto_strata ON pareto_frontiers(strata)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pareto_rank ON pareto_frontiers(rank)")

            conn.commit()

    # ==================== Design Operations ====================

    def get_or_create_design(self, design_id: int = 0) -> List[Any]:
        """Get the latest design or create a new one."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            if design_id <= 0: _create_design()

            cursor.execute("SELECT id, global_summary FROM designs WHERE id = ?", (design_id,))
            row = cursor.fetchone()
            if row:
                return [row[0], row[1]]
            else:
                return self._create_design()

    def _create_design(self) -> List[Any]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO designs (global_summary) VALUES ('')")
            conn.commit()
            cursor.execute("SELECT id, global_summary FROM designs ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return [row[0], row[1]]

    def get_global_summary(self, design_id: int) -> str:
        """Get the global summary for a design."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT global_summary FROM designs WHERE id = ?", (design_id,))
            row = cursor.fetchone()
            return row[0] if row else ""

    def update_global_summary(self, design_id: int, summary: str):
        """Update the global summary and last_summary_plan_id."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM plans WHERE design_id = ? ORDER BY id DESC LIMIT 1",
                (design_id,)
            )
            row = cursor.fetchone()
            last_plan_id = row[0] if row else 1
            cursor.execute(
                "UPDATE designs SET global_summary = ?, last_summary_plan_id = ?, updated_at = ? WHERE id = ?",
                (summary, last_plan_id, datetime.now().isoformat(), design_id)
            )
            conn.commit()

    def update_last_summary_plan_id(self, design_id: int, plan_id: int):
        """Update the last_summary_plan_id for a design."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE designs SET last_summary_plan_id = ?, updated_at = ? WHERE id = ?",
                (plan_id, datetime.now().isoformat(), design_id)
            )
            conn.commit()

    def get_last_summary_plan_id_in_design(self, design_id: int) -> int:
        """Get number of plans since last global summary."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT last_summary_plan_id FROM designs WHERE id = ?", (design_id,))
            row = cursor.fetchone()
            return row[0] if row else 0

    def get_plan_count_since(self, design_id: int) -> int:
        """Get number of plans since last global summary."""
        plan_id = self.get_last_summary_plan_id_in_design(design_id)
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM plans WHERE design_id = ? AND id > ?",
                (design_id, plan_id)
            )
            row = cursor.fetchone()
            return row[0] if row else 0

    # ==================== Plan Operations ====================

    def create_plan(self, design_id: int) -> int:
        """Create a new plan record."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO plans (design_id) VALUES (?)",
                (design_id,)
            )
            conn.commit()
            return cursor.lastrowid

    def update_plan_summary(self, plan_id: int, summary: str):
        """Update plan with summary (includes recommendation and trends)."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE plans SET summary = ? WHERE id = ?",
                (summary, plan_id)
            )
            conn.commit()

    def get_plan_summaries_since(self, design_id: int, since_plan_id: int) -> List[str]:
        """Get plan summaries since a given plan_id."""
        since_plan_id = since_plan_id if since_plan_id else 1
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT summary FROM plans
                   WHERE design_id = ? AND id >= ?
                   ORDER BY id ASC""",
                (design_id, since_plan_id)
            )
            rows = cursor.fetchall()
            return [r[0] for r in rows if r[0]]

    def get_latest_plan_id(self, design_id: int) -> Optional[int]:
        """Get the most recent plan_id for a design."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM plans WHERE design_id = ? ORDER BY id DESC LIMIT 1",
                (design_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def get_latest_plan_summary(self, design_id: int) -> Optional[int]:
        """Get the most recent plan_id for a design."""
        plan_id = self.get_latest_plan_id(design_id)
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT summary FROM plans WHERE id = ?",
                (plan_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def append_plan_token_used(self, plan_id: int, token_used: int, token_type: str = None):
        """Update the token_used for a design."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT token_total_used, token_plan_used, token_analysis_used, token_global_summary_used FROM plans WHERE id = ?", (plan_id,))
            row = cursor.fetchone()
            token_total_used = row[0]
            token_plan_used = row[1]
            token_analysis_used = row[2]
            token_global_summary_used = row[3]

            if token_type == "plan":
                token_plan_used += token_used
            elif token_type == "analysis":
                token_analysis_used += token_used
            elif token_type == "global_summary":
                token_global_summary_used += token_used

            cursor.execute(
                """
                    UPDATE plans SET
                        token_total_used = ?,
                        token_plan_used = ?,
                        token_analysis_used = ?,
                        token_global_summary_used = ?
                    WHERE id = ?
                """,
                (token_total_used + token_used, token_plan_used, token_analysis_used, token_global_summary_used, plan_id)
            )

            cursor.execute("SELECT design_id FROM plans WHERE id = ?", (plan_id,))
            row = cursor.fetchone()
            design_id = row[0] if row else 0

            cursor.execute("SELECT token_used FROM designs WHERE id = ?", (design_id,))
            row = cursor.fetchone()
            design_token_used = row[0] if row else 0

            cursor.execute(
                "UPDATE designs SET token_used = ? WHERE id = ?",
                (design_token_used + token_total_used + token_used, design_id)
            )

            conn.commit()

    def count_plans(self, design_id: int) -> int:
        """Count the number of plans for a design."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM plans WHERE design_id = ?", (design_id,))
            row = cursor.fetchone()
            return row[0] if row else 0
    # ==================== Experiment Operations ====================

    def save_experiment(self, plan_id: int, config: Any, metrics: Any, artifacts: Any = None, status: str = "completed") -> int:
        """
        Save an experiment result.
        Accepts Pydantic models or Dicts for config, metrics, artifacts.
        """
        # 1. Normalize to Dict for SQLite
        config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
        metrics_dict = metrics.model_dump() if hasattr(metrics, 'model_dump') else metrics
        artifacts_dict = (artifacts.model_dump() if hasattr(artifacts, 'model_dump') else artifacts) or {}

        # 2. Extract experiment_id
        experiment_id = config_dict.get('experiment_id', 'unknown')

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO experiments (plan_id, experiment_id, config, metrics, artifacts, status) VALUES (?, ?, ?, ?, ?, ?)",
                (plan_id, experiment_id, json.dumps(config_dict), json.dumps(metrics_dict), json.dumps(artifacts_dict), status)
            )
            conn.commit()
            exp_id = cursor.lastrowid

        return exp_id

    def get_experiments_by_plan(self, plan_id: int) -> List[Dict]:
        """Get all experiments for a plan."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, config, metrics, artifacts, status FROM experiments WHERE plan_id = ?",
                (plan_id,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "config": json.loads(r[1]),
                    "metrics": json.loads(r[2]),
                    "artifacts": json.loads(r[3]),
                    "status": r[4]
                }
                for r in rows
            ]

    def get_all_experiments_by_design(self, design_id: int) -> List[Dict]:
        """
        Get ALL experiments for a design across all plans.
        Critical for accurate Pareto frontier calculation.
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT e.id, e.experiment_id, e.config, e.metrics
                   FROM experiments e
                   JOIN plans p ON e.plan_id = p.id
                   WHERE p.design_id = ?""",
                (design_id,)
            )
            rows = cursor.fetchall()

            results = []
            for r in rows:
                config = json.loads(r[2])
                metrics = json.loads(r[3])

                # Determine strata (algorithm family)
                strata = config.get('recon_family', 'Unknown')

                results.append({
                    "id": r[0],
                    "experiment_id": r[1],
                    "config": config,
                    "metrics": metrics,
                    "strata": strata
                })

            return results

    # ==================== Pareto Frontiers (with Rank and Strata) ====================

    def get_pareto_frontiers(self, strata: str = None) -> List[Dict]:
        """
        Get Pareto frontiers (with rank).

        Args:
            strata: Filter by strata. If None, returns all.
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            if strata:
                cursor.execute(
                    """SELECT id, experiment_id, rank, strata, config, metrics
                       FROM pareto_frontiers WHERE strata = ? ORDER BY rank ASC""",
                    (strata,)
                )
            else:
                cursor.execute(
                    """SELECT id, experiment_id, rank, strata, config, metrics
                       FROM pareto_frontiers ORDER BY strata ASC, rank ASC"""
                )
            rows = cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "experiment_id": r[1],
                    "rank": r[2],
                    "strata": r[3],
                    "config": json.loads(r[4]),
                    "metrics": json.loads(r[5])
                }
                for r in rows
            ]

    def get_all_pareto_frontiers(self, design_id: int) -> List[Dict]:
        """Get all Pareto frontiers for a specific design (across all plans)."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT pf.id, pf.experiment_id, pf.rank, pf.strata, pf.config, pf.metrics
                   FROM pareto_frontiers pf
                   JOIN experiments e ON pf.experiment_id = e.experiment_id
                   JOIN plans p ON e.plan_id = p.id
                   WHERE p.design_id = ?
                   ORDER BY pf.rank ASC""",
                (design_id,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "experiment_id": r[1],
                    "rank": r[2],
                    "strata": r[3],
                    "config": json.loads(r[4]),
                    "metrics": json.loads(r[5])
                }
                for r in rows
            ]

    def get_design_space(self, design_id: int) -> Dict:
        """Get design space for a specific design. Returns None if not stored."""
        # Currently design space is not stored in DB, return None
        # In the future, we could store it in designs table
        return None


    def update_pareto_frontiers(self, strata: str, new_frontiers: List[Dict]):
        """
        Replace Pareto frontiers for a given strata.

        Args:
            strata: The strata identifier
            new_frontiers: List of dicts with {experiment_id, rank, config, metrics}
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                # Delete existing for this strata
                cursor.execute("DELETE FROM pareto_frontiers WHERE strata = ?", (strata,))

                # Insert new ones with rank
                for item in new_frontiers:
                    cursor.execute(
                        """INSERT INTO pareto_frontiers (experiment_id, rank, strata, config, metrics)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            item.get('experiment_id', 0),
                            item['rank'],
                            strata,
                            json.dumps(item['config']),
                            json.dumps(item['metrics'])
                        )
                    )
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to update Pareto frontiers: {e}")
                raise

    def clear_pareto_frontiers(self, strata: str = None):
        """Clear Pareto frontiers, optionally for a specific strata."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            if strata:
                cursor.execute("DELETE FROM pareto_frontiers WHERE strata = ?", (strata,))
            else:
                cursor.execute("DELETE FROM pareto_frontiers")
            conn.commit()
