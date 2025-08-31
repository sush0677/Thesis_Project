"""
Database Agent for MARL-GCP
-------------------------

This module implements the specialized Database agent which is responsible for
managing GCP database resources like Cloud SQL, Firestore, BigQuery, etc.
"""

from typing import Dict, List, Tuple, Any, Union
import numpy as np
import torch
import logging

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DatabaseAgent(BaseAgent):
    """
    Database Agent specializing in database resource management.
    
    Handles:
    - Cloud SQL instances and configurations
    - Firestore and BigQuery resource allocation
    - Database connection pooling and scaling
    - Backup and restore scheduling
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the Database Agent."""
        super().__init__(agent_id, config)
        self.agent_type = "database"
        
        # Database-specific state and action dimensions
        state_dim = 8  # Database metrics, connections, storage
        action_dim = 5  # Scale up/down, backup, restore, optimize
        
        self._initialize_agent(state_dim, action_dim)
        
        logger.info(f"Initialized Database Agent with state_dim={state_dim}, action_dim={action_dim}")
    
    def get_state_representation(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """
        Convert environment state to database agent's state representation.
        
        Args:
            environment_state: Complete environment state
            
        Returns:
            np.ndarray: Database agent's state vector
        """
        # Extract database-relevant metrics
        db_metrics = environment_state.get('database_metrics', {})
        
        state = np.array([
            db_metrics.get('cpu_usage', 0.0),
            db_metrics.get('memory_usage', 0.0),
            db_metrics.get('storage_usage', 0.0),
            db_metrics.get('connection_count', 0.0),
            db_metrics.get('query_latency', 0.0),
            db_metrics.get('transaction_rate', 0.0),
            db_metrics.get('backup_status', 0.0),
            db_metrics.get('replication_lag', 0.0)
        ], dtype=np.float32)
        
        return state
    
    def interpret_action(self, action: int) -> Dict[str, Any]:
        """
        Convert action index to database management action.
        
        Args:
            action: Action index from the policy network
            
        Returns:
            Dict[str, Any]: Database management action
        """
        actions = {
            0: {"type": "scale_up", "target": "instances"},
            1: {"type": "scale_down", "target": "instances"},
            2: {"type": "backup", "priority": "high"},
            3: {"type": "optimize", "target": "queries"},
            4: {"type": "maintain", "action": "status_quo"}
        }
        
        return actions.get(action, actions[4])
    
    def get_reward_contribution(self, state: np.ndarray, action: int, 
                              next_state: np.ndarray, environment_info: Dict[str, Any]) -> float:
        """
        Calculate database agent's contribution to the total reward.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            environment_info: Additional environment information
            
        Returns:
            float: Reward contribution
        """
        # Database performance metrics
        db_performance = environment_info.get('database_performance', {})
        
        # Reward based on query performance and resource efficiency
        query_improvement = db_performance.get('query_latency_improvement', 0.0)
        resource_efficiency = db_performance.get('resource_efficiency', 0.0)
        uptime_score = db_performance.get('uptime_score', 0.0)
        
        reward = (
            query_improvement * 0.4 +  # 40% weight on query performance
            resource_efficiency * 0.4 +  # 40% weight on resource efficiency
            uptime_score * 0.2  # 20% weight on availability
        )
        
        return reward