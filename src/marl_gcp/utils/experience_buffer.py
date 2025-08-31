"""
Shared Experience Buffer for MARL-GCP
------------------------------------

This module implements a shared experience buffer for MARL agents.
The buffer stores experiences from all agents and allows for experience sharing
and sampling based on priorities.
"""

from typing import Dict, List, Tuple, Any, Union
import numpy as np
import torch
import random
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)

class SharedExperienceBuffer:
    """
    Shared experience buffer for multi-agent learning.
    
    This buffer stores experiences from all agents and supports:
    1. Per-agent experience storage and retrieval
    2. Shared experience sampling across agents
    3. Prioritized experience replay
    """
    
    def __init__(self, buffer_size: int = 10000, batch_size: int = 64):
        """
        Initialize the shared experience buffer.
        
        Args:
            buffer_size: Maximum size of the buffer
            batch_size: Size of batches to sample
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Per-agent buffers
        self.buffers = defaultdict(lambda: deque(maxlen=buffer_size))
        
        # Shared buffer for common experiences
        self.shared_buffer = deque(maxlen=buffer_size)
        
        # Priorities for experiences
        self.priorities = defaultdict(lambda: deque(maxlen=buffer_size))
        
        logger.info(f"Initialized SharedExperienceBuffer with size {buffer_size}, batch size {batch_size}")
    
    def add(self, agent_name: str, state: np.ndarray, action: np.ndarray, 
            reward: float, next_state: np.ndarray, done: bool,
            priority: float = 1.0, share: bool = False) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            agent_name: Name of the agent that generated this experience
            state: State observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether the episode terminated
            priority: Priority of this experience (higher is more important)
            share: Whether to add this experience to the shared buffer too
        """
        experience = (state, action, reward, next_state, done)
        
        # Add to agent-specific buffer
        self.buffers[agent_name].append(experience)
        self.priorities[agent_name].append(priority)
        
        # Add to shared buffer if specified
        if share:
            self.shared_buffer.append((agent_name, experience))
            
    def sample(self, agent_name: str, use_shared: bool = True, use_priority: bool = True) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of experiences for an agent.
        
        Args:
            agent_name: Name of the agent to sample for
            use_shared: Whether to include samples from the shared buffer
            use_priority: Whether to use prioritized sampling
            
        Returns:
            Dictionary with state, action, reward, next_state, done tensors
        """
        agent_buffer = self.buffers[agent_name]
        agent_priorities = self.priorities[agent_name]
        
        # Determine how many samples to take from agent buffer vs shared buffer
        if use_shared and len(self.shared_buffer) > 0:
            agent_samples = max(1, int(self.batch_size * 0.7))  # At least 70% from agent's own experiences
            shared_samples = self.batch_size - agent_samples
        else:
            agent_samples = self.batch_size
            shared_samples = 0
        
        # Ensure we don't sample more than available
        agent_samples = min(agent_samples, len(agent_buffer))
        shared_samples = min(shared_samples, len(self.shared_buffer))
        
        # Sample from agent's buffer
        if use_priority and len(agent_priorities) > 0:
            # Convert priorities to probabilities
            probs = np.array(agent_priorities) / sum(agent_priorities)
            agent_indices = np.random.choice(
                len(agent_buffer), 
                size=agent_samples, 
                replace=False if agent_samples < len(agent_buffer) else True,
                p=probs
            )
            agent_batch = [agent_buffer[i] for i in agent_indices]
        else:
            agent_batch = random.sample(agent_buffer, agent_samples) if agent_samples > 0 else []
        
        # Sample from shared buffer
        shared_batch = []
        if shared_samples > 0:
            shared_indices = np.random.choice(
                len(self.shared_buffer), 
                size=shared_samples, 
                replace=False if shared_samples < len(self.shared_buffer) else True
            )
            for i in shared_indices:
                source_agent, experience = self.shared_buffer[i]
                # Skip if this is the requesting agent (avoid duplicates)
                if source_agent != agent_name:
                    shared_batch.append(experience)
        
        # Combine batches
        batch = agent_batch + shared_batch
        if len(batch) == 0:
            logger.warning(f"Empty batch sampled for agent {agent_name}")
            return None
        
        # Convert to tensors
        states = torch.FloatTensor([exp[0] for exp in batch]).to(self.device)
        actions = torch.FloatTensor([exp[1] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([exp[3] for exp in batch]).to(self.device)
        dones = torch.FloatTensor([float(exp[4]) for exp in batch]).unsqueeze(1).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def update_priorities(self, agent_name: str, indices: List[int], priorities: List[float]) -> None:
        """
        Update priorities for specific experiences.
        
        Args:
            agent_name: Name of the agent
            indices: Indices of experiences to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities[agent_name]):
                self.priorities[agent_name][idx] = priority
    
    def ready_for_update(self) -> bool:
        """
        Check if buffer has enough samples for an update.
        
        Returns:
            True if buffer has enough samples, False otherwise
        """
        for agent_buffer in self.buffers.values():
            if len(agent_buffer) >= self.batch_size:
                return True
        return False
    
    def __len__(self) -> int:
        """
        Get the total number of experiences in all buffers.
        
        Returns:
            Total number of experiences
        """
        return sum(len(buffer) for buffer in self.buffers.values())