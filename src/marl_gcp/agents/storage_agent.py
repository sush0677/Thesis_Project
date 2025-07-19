"""
Storage Agent for MARL-GCP
-------------------------

This module implements the specialized Storage agent which is responsible for
managing GCP storage resources like persistent disks, Cloud Storage buckets, etc.
"""

from typing import Dict, List, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from marl_gcp.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class StorageActorNetwork(nn.Module):
    """
    Actor network for the Storage agent.
    Maps states to actions for storage resources.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the actor network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
        """
        super(StorageActorNetwork, self).__init__()
        
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


class StorageCriticNetwork(nn.Module):
    """
    Critic network for the Storage agent.
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
        super(StorageCriticNetwork, self).__init__()
        
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


class StorageAgent(BaseAgent):
    """
    Storage Agent implementing TD3 algorithm for storage resource provisioning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Storage agent.
        
        Args:
            config: Configuration dictionary
        """
        super(StorageAgent, self).__init__(config, "storage")
        
        # Storage-specific configuration
        self.state_dim = config.get('state_dim', 4)  # State dimensions for storage resources
        self.action_dim = config.get('action_dim', 4)  # Action dimensions for storage resources
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # TD3-specific parameters
        self.policy_noise = config.get('policy_noise', 0.2)
        self.noise_clip = config.get('noise_clip', 0.5)
        self.policy_freq = config.get('policy_freq', 2)
        self.update_counter = 0
        
        # Create actor-critic networks
        self.policy_net = StorageActorNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_net = StorageActorNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.critic = StorageCriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_target = StorageCriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # Action scaling
        self.max_action = config.get('max_action', 1.0)
        
        logger.info(f"Initialized Storage Agent with state_dim={self.state_dim}, action_dim={self.action_dim}")
    
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
        
        # Compute target Q values
        with torch.no_grad():
            # Select next actions using target policy
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            
            next_actions = self.target_net(next_states) + noise
            next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)
            
            # Compute target Q values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Compute current Q values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
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