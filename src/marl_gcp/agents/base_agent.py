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
import torch.nn.functional as F
import torch.optim as optim
import random
import logging
from collections import deque

logger = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for the MARL-GCP agents.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [256, 256]):
        """
        Initialize the DQN network.
        
        Args:
            state_size: Size of the state space
            action_size: Size of the action space
            hidden_sizes: List of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()
        
        # Build the network layers
        layers = []
        input_size = state_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)

class BaseAgent:
    """
    Base agent class for MARL-GCP with DQN implementation.
    
    All specialized agents (Compute, Storage, Network, Database) inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any], name: str, state_size: int, action_size: int):
        """
        Initialize the base agent.
        
        Args:
            config: Agent configuration dictionary
            name: Agent name (e.g., 'compute', 'storage')
            state_size: Size of the observation space
            action_size: Size of the action space
        """
        self.config = config
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # RL parameters
        self.gamma = config.get('gamma', 0.99)  # Discount factor
        self.tau = config.get('tau', 0.005)  # Soft update parameter
        self.batch_size = config.get('batch_size', 64)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epsilon = config.get('epsilon_start', 1.0)  # Exploration rate
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.update_frequency = config.get('update_frequency', 4)
        self.target_update_frequency = config.get('target_update_frequency', 100)
        
        # Network architecture
        self.state_size = state_size
        self.action_size = action_size
        hidden_sizes = config.get('hidden_sizes', [256, 256])
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Copy weights to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Experience replay buffer
        self.memory = deque(maxlen=config.get('buffer_size', 100000))
        
        # Training tracking
        self.training = True
        self.steps_done = 0
        self.update_count = 0
        self.episode_rewards = []
        self.losses = []
        
        logger.info(f"Initialized {name} agent with DQN - State: {state_size}, Action: {action_size}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            deterministic: If True, select greedy action (no exploration)
            
        Returns:
            Selected action index
        """
        self.steps_done += 1
        
        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if deterministic or random.random() > self.epsilon:
            # Greedy action
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
        else:
            # Random action
            action = random.randrange(self.action_size)
        
        # Decay epsilon
        if self.training and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return action
        
        logger.info(f"Initialized {name} agent")
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state)
        
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self, experiences: Dict[str, torch.Tensor] = None) -> Dict[str, float]:
        """
        Update the agent's policy using DQN learning.
        
        Args:
            experiences: Optional external experiences (for centralized training)
            
        Returns:
            Dictionary with update metrics
        """
        if len(self.memory) < self.batch_size:
            return {}
        
        # Sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        # Convert actions to proper format with robust handling
        if isinstance(actions, (tuple, list)):
            # Handle list/tuple of actions
            action_list = []
            for a in actions:
                try:
                    if isinstance(a, (tuple, list)):
                        # Nested structure - take first element
                        if len(a) > 0:
                            val = a[0]
                            if hasattr(val, 'item'):
                                val = int(val.item())
                            else:
                                val = int(val)
                            # Clamp to valid action range
                            action_list.append(max(0, min(val, self.action_size - 1)))
                        else:
                            action_list.append(0)
                    elif hasattr(a, 'item'):  # numpy scalar or tensor
                        # Check if it's actually a scalar (size 1)
                        if hasattr(a, 'size') and (a.size == 1 or (hasattr(a, 'shape') and a.shape == ())):
                            val = int(a.item())
                        elif hasattr(a, 'flat'):
                            # Multi-element array - take first
                            val = int(a.flat[0])
                        elif hasattr(a, '__len__') and len(a) > 0:
                            val = int(a[0])
                        else:
                            val = int(a.item())
                        # Clamp to valid action range
                        action_list.append(max(0, min(val, self.action_size - 1)))
                    elif hasattr(a, '__len__') and len(a) == 1:  # single-element array
                        val = int(a[0])
                        action_list.append(max(0, min(val, self.action_size - 1)))
                    else:  # regular scalar
                        val = int(a)
                        action_list.append(max(0, min(val, self.action_size - 1)))
                except (ValueError, TypeError, AttributeError):
                    # Fallback to 0 if conversion fails
                    action_list.append(0)
            actions = torch.LongTensor(action_list).to(self.device)
        elif hasattr(actions, 'shape') and len(actions.shape) > 1:
            # Handle multi-dimensional numpy array
            action_list = []
            for a in actions.flatten():
                try:
                    if hasattr(a, 'item'):
                        val = int(a.item())
                    else:
                        val = int(a)
                    # Clamp to valid action range
                    action_list.append(max(0, min(val, self.action_size - 1)))
                except (ValueError, TypeError):
                    action_list.append(0)
            actions = torch.LongTensor(action_list).to(self.device)
        else:
            # Handle single action or array
            if hasattr(actions, '__len__') and not isinstance(actions, str):
                action_list = []
                for a in actions:
                    try:
                        if hasattr(a, 'item'):  # numpy scalar
                            # Check if it's actually a scalar
                            if hasattr(a, 'size') and (a.size == 1 or (hasattr(a, 'shape') and a.shape == ())):
                                val = int(a.item())
                            elif hasattr(a, 'flat'):
                                # Multi-element array - take first
                                val = int(a.flat[0])
                            else:
                                val = int(a.item())
                            # Clamp to valid action range
                            action_list.append(max(0, min(val, self.action_size - 1)))
                        elif hasattr(a, '__len__') and len(a) == 1:  # single-element array
                            val = int(a[0])
                            action_list.append(max(0, min(val, self.action_size - 1)))
                        else:  # regular scalar
                            val = int(a)
                            action_list.append(max(0, min(val, self.action_size - 1)))
                    except (ValueError, TypeError, AttributeError):
                        action_list.append(0)
                actions = torch.LongTensor(action_list).to(self.device)
            else:
                try:
                    if hasattr(actions, 'item'):
                        val = int(actions.item())
                        actions = torch.LongTensor([max(0, min(val, self.action_size - 1))]).to(self.device)
                    else:
                        val = int(actions)
                        actions = torch.LongTensor([max(0, min(val, self.action_size - 1))]).to(self.device)
                except (ValueError, TypeError):
                    actions = torch.LongTensor([0]).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self._soft_update()
        
        # Track metrics
        self.losses.append(loss.item())
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_mean': current_q_values.mean().item(),
            'target_q_mean': target_q_values.mean().item()
        }
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for the current state.
        
        Args:
            state: Current state
            
        Returns:
            Action probabilities
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            probabilities = F.softmax(q_values, dim=1)
        
        return probabilities.cpu().numpy().flatten()
    
    def reset_episode(self):
        """Reset episode-specific variables."""
        pass
    
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