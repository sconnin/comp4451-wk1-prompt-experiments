"""
Database module for storing experiment results and metrics.
"""
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class Database:
    """Handles all database operations for experiment results."""
    
    def __init__(self, db_path: str = "data/experiments.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def create_tables(self):
        """Create necessary tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Prompts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                template_type TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                variables TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        """)
        
        # Responses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id INTEGER,
                response_text TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_used INTEGER,
                response_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prompt_id) REFERENCES prompts (id)
            )
        """)
        
        # Evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id INTEGER,
                relevance_score REAL,
                accuracy_score REAL,
                completeness_score REAL,
                consistency_score REAL,
                efficiency_score REAL,
                bias_score REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (response_id) REFERENCES responses (id)
            )
        """)
        
        self.conn.commit()
        logger.info("Database tables created/verified")
    
    def create_experiment(self, name: str, config: Dict) -> int:
        """Create a new experiment record."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO experiments (name, config) VALUES (?, ?)",
            (name, json.dumps(config))
        )
        self.conn.commit()
        experiment_id = cursor.lastrowid
        logger.info(f"Created experiment: {name} (ID: {experiment_id})")
        return experiment_id
    
    def create_prompt(self, experiment_id: int, template_type: str, 
                     prompt_text: str, variables: Optional[Dict] = None) -> int:
        """Create a new prompt record."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO prompts (experiment_id, template_type, prompt_text, variables) VALUES (?, ?, ?, ?)",
            (experiment_id, template_type, prompt_text, json.dumps(variables) if variables else None)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def create_response(self, prompt_id: int, response_text: str, model: str,
                       tokens_used: int, response_time: float) -> int:
        """Create a new response record."""
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO responses 
               (prompt_id, response_text, model, tokens_used, response_time) 
               VALUES (?, ?, ?, ?, ?)""",
            (prompt_id, response_text, model, tokens_used, response_time)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def create_evaluation(self, response_id: int, scores: Dict, notes: str = "") -> int:
        """Create a new evaluation record."""
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO evaluations 
               (response_id, relevance_score, accuracy_score, completeness_score,
                consistency_score, efficiency_score, bias_score, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (response_id, scores.get('relevance'), scores.get('accuracy'),
             scores.get('completeness'), scores.get('consistency'),
             scores.get('efficiency'), scores.get('bias'), notes)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def get_experiment_results(self, experiment_id: int) -> List[Dict]:
        """Retrieve all results for a specific experiment."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                e.name as experiment_name,
                p.template_type,
                p.prompt_text,
                r.response_text,
                r.model,
                r.tokens_used,
                r.response_time,
                ev.relevance_score,
                ev.accuracy_score,
                ev.completeness_score,
                ev.consistency_score,
                ev.efficiency_score,
                ev.bias_score,
                r.created_at
            FROM experiments e
            JOIN prompts p ON e.id = p.experiment_id
            JOIN responses r ON p.id = r.prompt_id
            LEFT JOIN evaluations ev ON r.id = ev.response_id
            WHERE e.id = ?
            ORDER BY r.created_at DESC
        """, (experiment_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_experiments(self) -> List[Dict]:
        """Retrieve all experiments."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]
    
    def get_template_comparison(self) -> List[Dict]:
        """Compare performance across different template types."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                p.template_type,
                COUNT(r.id) as response_count,
                AVG(r.response_time) as avg_response_time,
                AVG(r.tokens_used) as avg_tokens,
                AVG(ev.relevance_score) as avg_relevance,
                AVG(ev.accuracy_score) as avg_accuracy,
                AVG(ev.completeness_score) as avg_completeness,
                AVG(ev.consistency_score) as avg_consistency,
                AVG(ev.efficiency_score) as avg_efficiency,
                AVG(ev.bias_score) as avg_bias
            FROM prompts p
            JOIN responses r ON p.id = r.prompt_id
            LEFT JOIN evaluations ev ON r.id = ev.response_id
            GROUP BY p.template_type
        """)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
