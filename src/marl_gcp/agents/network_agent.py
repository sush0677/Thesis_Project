"""
Network Agent for MARL-GCP
-----        # Bandwidth action: 3 options (decrease, no_change, increase)
        # vpc_action: 0=no_change, 1=add_subnet, 2=optimize_routing (3 options)
        # lb_action: 0=no_change, 1=enable_lb, 2=disable_lb (3 options)
        # cdn_action: 0=no_change, 1=enable_cdn, 2=disable_cdn (3 options)
        action_size = 3 * 3 * 3  # 27 discrete actions to match expected size
        
        # State space for network management - Fixed to match actual state size
        state_size = 7  # Match actual state dimensions from environment
        
        super().__init__(config, "network", state_size, action_size)----------

This module implements the specialized Network agent which is responsible for
managing GCP network resources like VPCs, subnets, load balancers, etc.
"""

from typing import Dict, List, Tuple, Any, Union
import numpy as np
import torch
import logging

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class NetworkAgent(BaseAgent):
    """
    Network Agent for managing GCP network resources.
    
    This agent is responsible for:
    - VPC and subnet management
    - Load balancer configuration
    - Bandwidth allocation
    - CDN and caching strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Network Agent.
        
        Args:
            config: Configuration dictionary for the network agent
        """
        # Network-specific action space
        # Actions: [bandwidth_action, vpc_action, lb_action, cdn_action]
        # bandwidth_action: 0=no_change, 1=increase, 2=decrease (3 options)
        # vpc_action: 0=no_change, 1=add_subnet, 2=optimize_routing (3 options)
        # lb_action: 0=no_change, 1=enable_lb, 2=disable_lb (3 options)
        # cdn_action: 0=no_change, 1=enable_cdn, 2=disable_cdn (3 options)
        action_size = 3 * 3 * 3 * 3  # 81 discrete actions
        
        # State space for network management - Fixed to match actual state size
        state_size = 7  # Match actual state dimensions from environment
        
        super().__init__(config, "network", state_size, action_size)
        
        # Network-specific parameters
        self.max_bandwidth_gbps = config.get('max_bandwidth_gbps', 100)
        self.max_subnets = config.get('max_subnets', 20)
        
        # Reward weights
        self.latency_weight = config.get('latency_weight', 0.4)
        self.bandwidth_efficiency_weight = config.get('bandwidth_efficiency_weight', 0.3)
        self.cost_efficiency_weight = config.get('cost_efficiency_weight', 0.3)
        
        logger.info(f"Network Agent initialized with {action_size} actions")
    
    def decode_action(self, action_index: int) -> Dict[str, int]:
        """
        Decode the discrete action index into network actions.
        
        Args:
            action_index: Index of the selected action
            
        Returns:
            Dictionary with decoded actions
        """
        bandwidth_action = action_index // (3 * 3 * 3)
        remaining = action_index % (3 * 3 * 3)
        
        vpc_action = remaining // (3 * 3)
        remaining = remaining % (3 * 3)
        
        lb_action = remaining // 3
        cdn_action = remaining % 3
        
        return {
            'bandwidth_action': bandwidth_action,  # 0=no_change, 1=increase, 2=decrease
            'vpc_action': vpc_action,             # 0=no_change, 1=add_subnet, 2=optimize
            'lb_action': lb_action,               # 0=no_change, 1=enable, 2=disable
            'cdn_action': cdn_action              # 0=no_change, 1=enable, 2=disable
        }
    
    def compute_reward_components(self, state: Dict[str, Any], action: Dict[str, int], 
                                next_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute reward components for network management.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Dictionary with reward components
        """
        rewards = {}
        
        # Latency performance reward
        current_latency = next_state.get('network_latency_ms', 100.0)
        target_latency = 50.0  # Target 50ms
        latency_performance = max(0.0, 1.0 - (current_latency - target_latency) / target_latency)
        rewards['latency'] = latency_performance * self.latency_weight
        
        # Bandwidth efficiency
        bandwidth_used = next_state.get('bandwidth_used_gbps', 0.0)
        bandwidth_allocated = next_state.get('bandwidth_allocated_gbps', 1.0)
        bandwidth_efficiency = bandwidth_used / max(bandwidth_allocated, 0.1)
        bandwidth_efficiency = min(1.0, bandwidth_efficiency)  # Cap at 1.0
        rewards['bandwidth_efficiency'] = bandwidth_efficiency * self.bandwidth_efficiency_weight
        
        # Cost efficiency
        network_cost = next_state.get('network_cost', 0.0)
        budget = next_state.get('budget_remaining', 1000.0)
        cost_efficiency = min(1.0, budget / max(network_cost, 1.0))
        rewards['cost_efficiency'] = cost_efficiency * self.cost_efficiency_weight
        
        # CDN efficiency bonus
        cdn_enabled = next_state.get('cdn_enabled', False)
        cache_hit_ratio = next_state.get('cache_hit_ratio', 0.0)
        if cdn_enabled and cache_hit_ratio > 0.7:
            rewards['cdn_bonus'] = 0.1
        else:
            rewards['cdn_bonus'] = 0.0
        
        return rewards
    
    def get_state_features(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """
        Extract network-relevant features from environment state.
        
        Args:
            environment_state: Full environment state
            
        Returns:
            Numpy array with state features
        """
        features = []
        
        # Current network resources
        features.extend([
            environment_state.get('network_bandwidth_gbps', 0) / self.max_bandwidth_gbps,
            environment_state.get('network_load', 0.0),
        ])
        
        # Network topology
        features.extend([
            environment_state.get('subnet_count', 0) / self.max_subnets,
            environment_state.get('vpc_complexity', 0.0),
        ])
        
        # Performance metrics
        features.extend([
            environment_state.get('network_latency_ms', 100.0) / 1000.0,  # Normalize to [0,1]
            environment_state.get('bandwidth_utilization', 0.0),
        ])
        
        # Load balancing and CDN
        features.extend([
            1.0 if environment_state.get('load_balancer', 'none') != 'none' else 0.0,
            1.0 if environment_state.get('cdn_enabled', False) else 0.0,
            environment_state.get('cache_hit_ratio', 0.0),
        ])
        
        # Cost and budget
        features.extend([
            environment_state.get('network_cost', 0.0) / environment_state.get('max_budget', 1000.0),
        ])
        
        # Coordination with other agents
        features.extend([
            environment_state.get('instances', 0) / 100.0,  # Compute influence
            environment_state.get('storage_gb', 0) / 10000.0,  # Storage influence
            environment_state.get('database_load', 0.0),   # Database influence
        ])
        
        # Traffic patterns
        features.extend([
            environment_state.get('incoming_requests_per_sec', 0.0) / 1000.0,
            environment_state.get('outgoing_traffic_gbps', 0.0) / 10.0,
        ])
        
        # Time factors
        features.extend([
            environment_state.get('time_of_day', 0.5),
        ])
        
        # Pad or truncate to expected state size
        features = features[:self.state_size] + [0.0] * max(0, self.state_size - len(features))
        
        return np.array(features, dtype=np.float32)

from typing import Dict, List, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class NetworkActorNetwork(nn.Module):
    """
    Actor network for the Network agent.
    Maps states to actions for networking resources.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the actor network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
        """
        super(NetworkActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Action distribution parameters
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Using tanh to bound actions between -1 and 1
        return torch.tanh(self.fc3(x))


class NetworkCriticNetwork(nn.Module):
    """
    Critic network for the Network agent.
    Maps state-action pairs to Q-values.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the critic network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
        """
        super(NetworkCriticNetwork, self).__init__()
        
        # Q1 network
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2 network (for min Q learning)
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Tuple of two Q-values (Q1, Q2)
        """
        sa = torch.cat([state, action], dim=1)
        
        # Q1 value
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        # Q2 value
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2


class NetworkAgent(BaseAgent):
    """
    Network Agent implementing TD3 algorithm for networking resource provisioning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Network agent.
        
        Args:
            config: Configuration dictionary
        """
        # Extract state and action sizes from config
        # State space for network agent should match environment output
        state_size = config.get('state_size', 7)  # Fixed to match actual dimensions
        action_size = config.get('action_size', 27)  # 3^3 actions
        
        super().__init__(config, "network", state_size, action_size)
        
        # Network-specific configuration
        self.state_dim = config.get('state_dim', 4)  # State dimensions for network resources
        self.action_dim = config.get('action_dim', 6)  # Action dimensions for network resources
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # TD3-specific parameters (not used with DQN)
        self.policy_noise = config.get('policy_noise', 0.2)
        self.noise_clip = config.get('noise_clip', 0.5)
        self.policy_freq = config.get('policy_freq', 2)
        self.update_counter = 0
        
        # The DQN networks are already initialized in BaseAgent
        # No need to create additional networks since we're using discrete actions with DQN
        
        logger.info(f"Initialized Network Agent with state_dim={self.state_dim}, action_dim={self.action_dim}")
        
        logger.info(f"Initialized Network Agent with state_dim={self.state_dim}, action_dim={self.action_dim}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select an action based on the current state using DQN.
        
        Args:
            state: Current state observation
            deterministic: If True, select the best action (no exploration)
            
        Returns:
            Discrete action to take (0-26 for 3^3 actions)
        """
        # Use the BaseAgent's select_action method which implements epsilon-greedy DQN
        return super().select_action(state, deterministic)
    
    def update(self, experiences: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent's policy based on experiences using DQN.
        
        Args:
            experiences: Dictionary with state, action, reward, next_state, and done tensors
            
        Returns:
            Dictionary with update metrics
        """
        # Use the BaseAgent's update method which implements DQN learning
        return super().update(experiences) 