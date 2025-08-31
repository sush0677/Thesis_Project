"""
Storage Agent for MARL-GCP
-------------------------

This module implements the specialized Storage agent which is responsible for
managing GCP storage resources like persistent disks, Cloud Storage buckets, etc.
"""

from typing import Dict, List, Tuple, Any, Union
import numpy as np
import torch
import logging

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class StorageAgent(BaseAgent):
    """
    Storage Agent for managing GCP storage resources.
    
    This agent is responsible for:
    - Persistent disk provisioning
    - Cloud Storage bucket management
    - Storage type optimization (SSD vs HDD)
    - Backup and archival strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Storage Agent.
        
        Args:
            config: Configuration dictionary for the storage agent
        """
        # Storage-specific action space
        # Actions: [disk_action, capacity_action, type_action, backup_action]
        # disk_action: 0=no_change, 1=add_disk, 2=remove_disk
        # capacity_action: 0=no_change, 1=increase, 2=decrease
        # type_action: 0=standard, 1=ssd, 2=archive
        # backup_action: 0=no_change, 1=enable_backup, 2=disable_backup
        action_size = 3 * 3 * 3 * 3  # 81 discrete actions
        
        # State space for storage management
        state_size = 18
        
        super().__init__(config, "storage", state_size, action_size)
        
        # Storage-specific parameters
        self.max_storage_gb = config.get('max_storage_gb', 10000)
        self.storage_types = config.get('storage_types', ['standard', 'ssd', 'archive'])
        
        # Reward weights
        self.utilization_weight = config.get('utilization_weight', 0.4)
        self.cost_efficiency_weight = config.get('cost_efficiency_weight', 0.3)
        self.performance_weight = config.get('performance_weight', 0.3)
        
        logger.info(f"Storage Agent initialized with {action_size} actions")
    
    def decode_action(self, action_index: int) -> Dict[str, int]:
        """
        Decode the discrete action index into storage actions.
        
        Args:
            action_index: Index of the selected action
            
        Returns:
            Dictionary with decoded actions
        """
        disk_action = action_index // (3 * 3 * 3)
        remaining = action_index % (3 * 3 * 3)
        
        capacity_action = remaining // (3 * 3)
        remaining = remaining % (3 * 3)
        
        type_action = remaining // 3
        backup_action = remaining % 3
        
        return {
            'disk_action': disk_action,       # 0=no_change, 1=add, 2=remove
            'capacity_action': capacity_action,  # 0=no_change, 1=increase, 2=decrease
            'type_action': type_action,       # 0=standard, 1=ssd, 2=archive
            'backup_action': backup_action    # 0=no_change, 1=enable, 2=disable
        }
    
    def compute_reward_components(self, state: Dict[str, Any], action: Dict[str, int], 
                                next_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute reward components for storage management.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Dictionary with reward components
        """
        rewards = {}
        
        # Storage utilization efficiency
        storage_used = next_state.get('storage_gb', 0)
        storage_capacity = next_state.get('storage_capacity', 1000)
        utilization = storage_used / max(storage_capacity, 1)
        
        # Optimal utilization is around 80%
        utilization_efficiency = 1.0 - abs(utilization - 0.8)
        rewards['utilization'] = utilization_efficiency * self.utilization_weight
        
        # Cost efficiency
        storage_cost = next_state.get('storage_cost', 0.0)
        budget = next_state.get('budget_remaining', 1000.0)
        cost_efficiency = min(1.0, budget / max(storage_cost, 1.0))
        rewards['cost_efficiency'] = cost_efficiency * self.cost_efficiency_weight
        
        # Performance reward (based on storage type appropriateness)
        workload_io_intensity = state.get('workload_io_intensity', 0.5)
        storage_type = next_state.get('primary_storage_type', 'standard')
        
        if workload_io_intensity > 0.7 and storage_type == 'ssd':
            performance_reward = 1.0
        elif workload_io_intensity < 0.3 and storage_type == 'archive':
            performance_reward = 1.0
        elif 0.3 <= workload_io_intensity <= 0.7 and storage_type == 'standard':
            performance_reward = 1.0
        else:
            performance_reward = 0.5
        
        rewards['performance'] = performance_reward * self.performance_weight
        
        # Constraint penalty
        constraint_penalty = 0.0
        if storage_used > self.max_storage_gb:
            constraint_penalty = (storage_used - self.max_storage_gb) * 0.001
        
        rewards['constraint_penalty'] = -constraint_penalty
        
        return rewards
    
    def get_state_features(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """
        Extract storage-relevant features from environment state.
        
        Args:
            environment_state: Full environment state
            
        Returns:
            Numpy array with state features
        """
        features = []
        
        # Current storage resources
        features.extend([
            environment_state.get('storage_gb', 0) / self.max_storage_gb,
            environment_state.get('storage_utilization', 0.0),
        ])
        
        # Storage type distribution
        features.extend([
            environment_state.get('ssd_ratio', 0.0),
            environment_state.get('standard_ratio', 0.0),
            environment_state.get('archive_ratio', 0.0),
        ])
        
        # Workload characteristics
        features.extend([
            environment_state.get('workload_storage_demand', 0.0),
            environment_state.get('workload_io_intensity', 0.5),
            environment_state.get('data_growth_rate', 0.0),
        ])
        
        # Cost and budget
        features.extend([
            environment_state.get('storage_cost', 0.0) / environment_state.get('max_budget', 1000.0),
            environment_state.get('budget_used', 0.0) / environment_state.get('max_budget', 1000.0),
        ])
        
        # Coordination with other agents
        features.extend([
            environment_state.get('instances', 0) / 100.0,  # Compute influence
            environment_state.get('database_load', 0.0),   # Database influence
            environment_state.get('network_load', 0.0),    # Network influence
        ])
        
        # Backup and reliability
        features.extend([
            environment_state.get('backup_enabled', 0.0),
            environment_state.get('backup_cost', 0.0) / 100.0,
        ])
        
        # Time factors
        features.extend([
            environment_state.get('time_of_day', 0.5),
            environment_state.get('day_of_week', 0.5),
        ])
        
        # Operational metrics
        features.extend([
            environment_state.get('io_operations_per_sec', 0.0) / 1000.0,
        ])
        
        # Pad or truncate to expected state size
        features = features[:self.state_size] + [0.0] * max(0, self.state_size - len(features))
        
        return np.array(features, dtype=np.float32) 