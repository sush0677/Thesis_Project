"""
Compute Agent for MARL-GCP
-------------------------

This module implements the specialized Compute agent which is responsible for
managing GCP compute resources like virtual machines, instance groups, etc.
"""

from typing import Dict, List, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from marl_gcp.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ComputeAgent(BaseAgent):
    """
    Compute Agent for managing GCP compute resources.
    
    This agent is responsible for:
    - VM instance provisioning and scaling
    - CPU allocation decisions
    - Memory allocation optimization
    - Instance type selection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Compute Agent.
        
        Args:
            config: Configuration dictionary for the compute agent
        """
        # Compute-specific action space
        # Actions: [instance_action, cpu_action, memory_action, scale_action]
        # instance_action: 0=no_change, 1=add_instance, 2=remove_instance
        # cpu_action: 0=no_change, 1=increase_cpu, 2=decrease_cpu  
        # memory_action: 0=no_change, 1=increase_memory, 2=decrease_memory
        # scale_action: 0=no_change, 1=scale_up, 2=scale_down
        action_size = 3 * 3 * 3 * 3  # 81 discrete actions
        
        # State space includes:
        # - Current instances, CPU, memory
        # - Resource utilization
        # - Workload demand
        # - Budget constraints
        # - Other agents' resource usage
        state_size = 20  # Will be adjusted based on actual environment
        
        super().__init__(config, "compute", state_size, action_size)
        
        # Compute-specific parameters
        self.instance_types = config.get('instance_types', ['n1-standard-1', 'n1-standard-2', 'n1-standard-4'])
        self.max_instances = config.get('max_instances', 100)
        self.max_cpus = config.get('max_cpus', 500)
        self.max_memory = config.get('max_memory', 1000)
        
        # Reward weights for compute decisions
        self.cpu_efficiency_weight = config.get('cpu_efficiency_weight', 0.3)
        self.memory_efficiency_weight = config.get('memory_efficiency_weight', 0.3)
        self.cost_efficiency_weight = config.get('cost_efficiency_weight', 0.4)
        
        logger.info(f"Compute Agent initialized with {action_size} actions and {state_size} state dimensions")
    
    def decode_action(self, action_index: int) -> Dict[str, int]:
        """
        Decode the discrete action index into meaningful compute actions.
        
        Args:
            action_index: Index of the selected action
            
        Returns:
            Dictionary with decoded actions
        """
        # Convert single action index to multi-dimensional action
        instance_action = action_index // (3 * 3 * 3)
        remaining = action_index % (3 * 3 * 3)
        
        cpu_action = remaining // (3 * 3)
        remaining = remaining % (3 * 3)
        
        memory_action = remaining // 3
        scale_action = remaining % 3
        
        return {
            'instance_action': instance_action,  # 0=no_change, 1=add, 2=remove
            'cpu_action': cpu_action,           # 0=no_change, 1=increase, 2=decrease
            'memory_action': memory_action,     # 0=no_change, 1=increase, 2=decrease
            'scale_action': scale_action        # 0=no_change, 1=scale_up, 2=scale_down
        }
    
    def compute_reward_components(self, state: Dict[str, Any], action: Dict[str, int], 
                                next_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute reward components specific to compute resource management.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Dictionary with reward components
        """
        rewards = {}
        
        # CPU efficiency reward
        cpu_utilization = next_state.get('cpu_utilization', 0.0)
        cpu_target = 0.7  # Target 70% utilization
        cpu_efficiency = 1.0 - abs(cpu_utilization - cpu_target)
        rewards['cpu_efficiency'] = cpu_efficiency * self.cpu_efficiency_weight
        
        # Memory efficiency reward
        memory_utilization = next_state.get('memory_utilization', 0.0)
        memory_target = 0.8  # Target 80% utilization
        memory_efficiency = 1.0 - abs(memory_utilization - memory_target)
        rewards['memory_efficiency'] = memory_efficiency * self.memory_efficiency_weight
        
        # Cost efficiency reward (penalize over-provisioning)
        total_cost = next_state.get('compute_cost', 0.0)
        budget = next_state.get('budget_remaining', 1000.0)
        cost_efficiency = min(1.0, budget / max(total_cost, 1.0))
        rewards['cost_efficiency'] = cost_efficiency * self.cost_efficiency_weight
        
        # Penalty for constraint violations
        instances = next_state.get('instances', 0)
        cpus = next_state.get('cpus', 0)
        memory = next_state.get('memory_gb', 0)
        
        constraint_penalty = 0.0
        if instances > self.max_instances:
            constraint_penalty += (instances - self.max_instances) * 0.1
        if cpus > self.max_cpus:
            constraint_penalty += (cpus - self.max_cpus) * 0.01
        if memory > self.max_memory:
            constraint_penalty += (memory - self.max_memory) * 0.01
        
        rewards['constraint_penalty'] = -constraint_penalty
        
        return rewards
    
    def get_state_features(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """
        Extract relevant features for the compute agent from environment state.
        
        Args:
            environment_state: Full environment state
            
        Returns:
            Numpy array with state features
        """
        features = []
        
        # Current compute resources
        features.extend([
            environment_state.get('instances', 0) / self.max_instances,
            environment_state.get('cpus', 0) / self.max_cpus,
            environment_state.get('memory_gb', 0) / self.max_memory,
        ])
        
        # Resource utilization
        features.extend([
            environment_state.get('cpu_utilization', 0.0),
            environment_state.get('memory_utilization', 0.0),
        ])
        
        # Workload characteristics
        features.extend([
            environment_state.get('workload_cpu_demand', 0.0),
            environment_state.get('workload_memory_demand', 0.0),
            environment_state.get('workload_priority', 0.5),
        ])
        
        # Budget and cost
        features.extend([
            environment_state.get('budget_used', 0.0) / environment_state.get('max_budget', 1000.0),
            environment_state.get('compute_cost', 0.0) / environment_state.get('max_budget', 1000.0),
        ])
        
        # Other agents' resource usage (coordination)
        features.extend([
            environment_state.get('storage_gb', 0.0) / 10000.0,  # Normalize storage
            environment_state.get('network_load', 0.0),
            environment_state.get('database_load', 0.0),
        ])
        
        # Time and seasonal factors
        features.extend([
            environment_state.get('time_of_day', 0.5),
            environment_state.get('day_of_week', 0.5),
        ])
        
        # Pending operations
        features.extend([
            len(environment_state.get('pending_operations', [])) / 10.0,  # Normalize
            environment_state.get('provisioning_delay', 0.0) / 10.0,
        ])
        
        # Pad or truncate to expected state size
        features = features[:self.state_size] + [0.0] * max(0, self.state_size - len(features))
        
        return np.array(features, dtype=np.float32)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Using tanh to bound actions between -1 and 1
        # These will be scaled to actual action ranges in the agent
        return torch.tanh(self.fc3(x))


class ComputeCriticNetwork(nn.Module):
    """
    Critic network for the Compute agent.
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
        super(ComputeCriticNetwork, self).__init__()
        
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


class ComputeAgent(BaseAgent):
    """
    Compute Agent implementing TD3 algorithm for compute resource provisioning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Compute agent.
        
        Args:
            config: Configuration dictionary
        """
        super(ComputeAgent, self).__init__(config, "compute")
        
        # Compute-specific configuration
        self.state_dim = config.get('state_dim', 10)  # State dimensions for compute resources
        self.action_dim = config.get('action_dim', 5)  # Action dimensions for compute resources
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # TD3-specific parameters
        self.policy_noise = config.get('policy_noise', 0.2)
        self.noise_clip = config.get('noise_clip', 0.5)
        self.policy_freq = config.get('policy_freq', 2)
        self.update_counter = 0
        
        # Create actor-critic networks
        self.policy_net = ComputeActorNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_net = ComputeActorNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.critic = ComputeCriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_target = ComputeCriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # Action scaling
        self.max_action = config.get('max_action', 1.0)
        
        logger.info(f"Initialized Compute Agent with state_dim={self.state_dim}, action_dim={self.action_dim}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state observation
            deterministic: If True, select the best action (no exploration)
            
        Returns:
            Action to take
        """
        # Convert state to torch tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Set networks to evaluation mode
        self.policy_net.eval()
        
        with torch.no_grad():
            # Get action from policy network
            action = self.policy_net(state_tensor).cpu().numpy()
        
        # Set networks back to training mode if needed
        if self.training:
            self.policy_net.train()
        
        # Add noise for exploration during training
        if not deterministic and self.training:
            noise = np.random.normal(0, self.policy_noise, size=action.shape)
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            action = action + noise
        
        # Clip action to valid range
        action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def update(self, experiences: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent's policy based on experiences.
        
        Args:
            experiences: Dictionary with state, action, reward, next_state, and done tensors
            
        Returns:
            Dictionary with update metrics
        """
        states = experiences['states']
        actions = experiences['actions']
        rewards = experiences['rewards']
        next_states = experiences['next_states']
        dones = experiences['dones']
        
        # --- Update critic ---
        with torch.no_grad():
            # Select action according to target policy and add noise
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_actions = self.target_net(next_states) + noise
            next_actions = next_actions.clamp(-self.max_action, self.max_action)
            
            # Get Q values from target critics
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Compute target Q value
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- Update actor (delayed) ---
        actor_loss = 0.0
        
        if self.update_counter % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic(states, self.policy_net(states))[0].mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            self._soft_update()
        
        self.update_counter += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss if isinstance(actor_loss, float) else actor_loss.item()
        }
    
    def _soft_update(self) -> None:
        """
        Soft update of the target network parameters.
        """
        # Update critic target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        
        # Update actor target
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            ) 