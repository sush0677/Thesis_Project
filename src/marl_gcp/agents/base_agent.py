"""
Base Agent for MARL-GCP
-----------------------

This module defines the base agent class that all specialized agents will inherit from.
The base agent includes common functionality like action selection, policy updates,
and training/evaluation mode switching.
"""

from typing import Dict, List, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base agent class for MARL-GCP.
    
    All specialized agents (Compute, Storage, Network, Database) inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any], name: str):
        """
        Initialize the base agent.
        
        Args:
            config: Agent configuration dictionary
            name: Agent name (e.g., 'compute', 'storage')
        """
        self.config = config
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training parameters
        self.gamma = config.get('gamma', 0.99)  # Discount factor
        self.tau = config.get('tau', 0.005)  # Soft update parameter
        self.batch_size = config.get('batch_size', 64)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Policy and value networks (to be implemented by subclasses)
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        
        # Training mode flag
        self.training = True
        
        logger.info(f"Initialized {name} agent")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state observation
            deterministic: If True, select the best action (no exploration)
            
        Returns:
            Action to take
        """
        raise NotImplementedError("Subclasses must implement select_action")
    
    def update(self, experiences: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent's policy based on experiences.
        
        Args:
            experiences: Dictionary with state, action, reward, next_state, and done tensors
            
        Returns:
            Dictionary with update metrics (e.g., loss values)
        """
        raise NotImplementedError("Subclasses must implement update")
    
    def save(self, path: str) -> None:
        """
        Save the agent's models to disk.
        
        Args:
            path: Directory path to save the models
        """
        if self.policy_net is not None:
            torch.save(self.policy_net.state_dict(), f"{path}/{self.name}_policy.pt")
        if self.target_net is not None:
            torch.save(self.target_net.state_dict(), f"{path}/{self.name}_target.pt")
        logger.info(f"Saved {self.name} agent models to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the agent's models from disk.
        
        Args:
            path: Directory path to load the models from
        """
        if self.policy_net is not None:
            self.policy_net.load_state_dict(torch.load(f"{path}/{self.name}_policy.pt", map_location=self.device))
        if self.target_net is not None:
            self.target_net.load_state_dict(torch.load(f"{path}/{self.name}_target.pt", map_location=self.device))
        logger.info(f"Loaded {self.name} agent models from {path}")
    
    def train(self) -> None:
        """Set the agent to training mode."""
        self.training = True
        if self.policy_net is not None:
            self.policy_net.train()
        if self.target_net is not None:
            self.target_net.train()
    
    def eval(self) -> None:
        """Set the agent to evaluation mode."""
        self.training = False
        if self.policy_net is not None:
            self.policy_net.eval()
        if self.target_net is not None:
            self.target_net.eval()
    
    def _soft_update(self) -> None:
        """
        Soft update of the target network parameters:
        θ_target = τ*θ_policy + (1-τ)*θ_target
        """
        if self.policy_net is None or self.target_net is None:
            return
            
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )