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

    def __init__(self, db_path: str = "cias_x.db", top_k: int = 10):
        self.db_path = db_path
        self.top_k = top_k  # Configurable top-k for Pareto frontiers
        self._init_db()
        logger.info(f"CIASWorldModel initialized with database: {db_path}, top_k={top_k}")

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
                    global_summary TEXT DEFAULT '',
                    last_summary_plan_id INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Plans Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    design_id INTEGER NOT NULL,
                    summary TEXT DEFAULT '',
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

    def get_or_create_design(self) -> int:
        """Get the latest design or create a new one."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM designs ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()

            if row:
                return row[0]
            else:
                cursor.execute("INSERT INTO designs (global_summary) VALUES ('')")
                conn.commit()
                return cursor.lastrowid

    def get_global_summary(self, design_id: int) -> str:
        """Get the global summary for a design."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT global_summary FROM designs WHERE id = ?", (design_id,))
            row = cursor.fetchone()
            return row[0] if row else ""

    def update_global_summary(self, design_id: int, summary: str, last_plan_id: int):
        """Update the global summary and last_summary_plan_id."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE designs SET global_summary = ?, last_summary_plan_id = ?, updated_at = ? WHERE id = ?",
                (summary, last_plan_id, datetime.now().isoformat(), design_id)
            )
            conn.commit()

    def get_last_summary_plan_id(self, design_id: int) -> int:
        """Get the plan_id at last global summary update."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT last_summary_plan_id FROM designs WHERE id = ?", (design_id,))
            row = cursor.fetchone()
            return row[0] if row else 0

    def get_plan_count_since(self, design_id: int, since_plan_id: int) -> int:
        """Get number of plans since a given plan_id."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM plans WHERE design_id = ? AND id > ?",
                (design_id, since_plan_id)
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
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT summary FROM plans
                   WHERE design_id = ? AND id > ?
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

    # ==================== Experiment Operations ====================

    def save_experiment(self, plan_id: int, config: Dict, metrics: Dict, artifacts: Dict = None, status: str = "completed") -> int:
        """Save an experiment result."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO experiments (plan_id, experiment_id, config, metrics, artifacts, status) VALUES (?, ?, ?, ?, ?, ?)",
                (plan_id, config['experiment_id'], json.dumps(config), json.dumps(metrics), json.dumps(artifacts or {}), status)
            )
            conn.commit()
            return cursor.lastrowid

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
