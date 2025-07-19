"""
System Architecture for MARL-GCP
--------------------------------

This module defines the overall system architecture for the Multi-Agent Reinforcement Learning
system for Google Cloud Platform resource provisioning.

The architecture includes:
1. Four specialized agents (Compute, Storage, Network, Database)
2. Communication channels between agents
3. Environment interface to GCP
4. Shared experience buffer
5. Monitoring components

The system implements centralized training with decentralized execution.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MARLSystemArchitecture:
    """
    Main system architecture class that orchestrates the multi-agent system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MARL system architecture.
        
        Args:
            config: Configuration dictionary with system parameters
        """
        self.config = config
        self.agents = {}
        self.environment = None
        self.experience_buffer = None
        self.monitoring = None
        
        logger.info("Initializing MARL System Architecture")
        
        # Initialize system components
        self._init_agents()
        self._init_environment()
        self._init_experience_buffer()
        self._init_monitoring()
        
        logger.info("MARL System Architecture initialized")
    
    def _init_agents(self):
        """Initialize the four specialized agents."""
        from marl_gcp.agents.compute_agent import ComputeAgent
        from marl_gcp.agents.storage_agent import StorageAgent
        from marl_gcp.agents.network_agent import NetworkAgent
        from marl_gcp.agents.database_agent import DatabaseAgent
        
        logger.info("Initializing agents")
        
        # Create agent instances
        self.agents['compute'] = ComputeAgent(self.config.get('agents', {}).get('compute', {}))
        self.agents['storage'] = StorageAgent(self.config.get('agents', {}).get('storage', {}))
        self.agents['network'] = NetworkAgent(self.config.get('agents', {}).get('network', {}))
        self.agents['database'] = DatabaseAgent(self.config.get('agents', {}).get('database', {}))
        
        logger.info(f"Created {len(self.agents)} agents")
    
    def _init_environment(self):
        """Initialize the environment interface to GCP."""
        from marl_gcp.environment.gcp_environment import GCPEnvironment
        
        logger.info("Initializing GCP environment")
        self.environment = GCPEnvironment(self.config.get('environment', {}))
    
    def _init_experience_buffer(self):
        """Initialize the shared experience buffer."""
        from marl_gcp.utils.experience_buffer import SharedExperienceBuffer
        
        logger.info("Initializing shared experience buffer")
        self.experience_buffer = SharedExperienceBuffer(
            buffer_size=self.config.get('buffer_size', 10000),
            batch_size=self.config.get('batch_size', 64)
        )
    
    def _init_monitoring(self):
        """Initialize monitoring components."""
        from marl_gcp.utils.monitoring import SystemMonitoring
        
        logger.info("Initializing monitoring components")
        self.monitoring = SystemMonitoring(self.config.get('monitoring', {}))
    
    def _convert_observation_to_array(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert dictionary observation to numpy array.
        
        Args:
            observation: Dictionary observation from environment
            
        Returns:
            Flattened numpy array
        """
        # Flatten all values in the observation dictionary
        flattened = []
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                flattened.extend(value.flatten())
            else:
                flattened.append(float(value))
        return np.array(flattened, dtype=np.float32)
    
    def train(self, episodes: int) -> Dict[str, Any]:
        """
        Train the MARL system for the specified number of episodes.
        
        Args:
            episodes: Number of training episodes
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Starting training for {episodes} episodes")
        
        metrics = {
            'rewards': [],
            'episode_lengths': [],
            'resource_usage': [],
            'costs': []
        }
        
        for episode in range(episodes):
            # Reset environment and get initial observations
            observations = self.environment.reset()
            
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # Episode loop
            while not done:
                # Collect actions from all agents
                actions = {}
                for agent_name, agent in self.agents.items():
                    # Convert dictionary observation to array
                    obs_array = self._convert_observation_to_array(observations[agent_name])
                    actions[agent_name] = agent.select_action(obs_array)
                
                # Execute actions in environment
                next_observations, rewards, done, info = self.environment.step(actions)
                
                # Store experience in buffer
                for agent_name in self.agents:
                    # Convert observations to arrays for storage
                    obs_array = self._convert_observation_to_array(observations[agent_name])
                    next_obs_array = self._convert_observation_to_array(next_observations[agent_name])
                    
                    self.experience_buffer.add(
                        agent_name,
                        obs_array,
                        actions[agent_name],
                        rewards[agent_name],
                        next_obs_array,
                        done
                    )
                
                # Update current observations
                observations = next_observations
                
                # Calculate total reward
                episode_reward += sum(rewards.values())
                episode_steps += 1
                
                # Update agents if enough samples are collected
                if self.experience_buffer.ready_for_update():
                    for agent_name, agent in self.agents.items():
                        experiences = self.experience_buffer.sample(agent_name)
                        agent.update(experiences)
            
            # Log episode results
            logger.info(f"Episode {episode+1}/{episodes} - " +
                       f"Reward: {episode_reward:.2f}, Steps: {episode_steps}")
            
            # Update metrics
            metrics['rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_steps)
            metrics['resource_usage'].append(info.get('resource_usage', {}))
            metrics['costs'].append(info.get('costs', {}))
            
            # Update monitoring
            self.monitoring.update(episode, metrics)
        
        logger.info("Training completed")
        return metrics

    def evaluate(self, episodes: int) -> Dict[str, Any]:
        """
        Evaluate the trained agents for the specified number of episodes.
        
        Args:
            episodes: Number of evaluation episodes
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Starting evaluation for {episodes} episodes")
        
        # Set agents to evaluation mode
        for agent in self.agents.values():
            agent.eval()
        
        metrics = {
            'rewards': [],
            'episode_lengths': [],
            'resource_usage': [],
            'costs': []
        }
        
        for episode in range(episodes):
            observations = self.environment.reset()
            
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # Episode loop
            while not done:
                # Collect actions from all agents (no exploration)
                actions = {}
                for agent_name, agent in self.agents.items():
                    # Convert dictionary observation to array
                    obs_array = self._convert_observation_to_array(observations[agent_name])
                    actions[agent_name] = agent.select_action(obs_array, deterministic=True)
                
                # Execute actions in environment
                next_observations, rewards, done, info = self.environment.step(actions)
                
                # Update current observations
                observations = next_observations
                
                # Calculate total reward
                episode_reward += sum(rewards.values())
                episode_steps += 1
            
            # Log episode results
            logger.info(f"Eval Episode {episode+1}/{episodes} - " +
                       f"Reward: {episode_reward:.2f}, Steps: {episode_steps}")
            
            # Update metrics
            metrics['rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_steps)
            metrics['resource_usage'].append(info.get('resource_usage', {}))
            metrics['costs'].append(info.get('costs', {}))
            
            # Update monitoring with evaluation results
            self.monitoring.update(episode, metrics)
        
        # Set agents back to training mode
        for agent in self.agents.values():
            agent.train()
            
        logger.info("Evaluation completed")
        return metrics