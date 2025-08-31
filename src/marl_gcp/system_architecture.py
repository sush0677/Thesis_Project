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
        from .agents.compute_agent import ComputeAgent
        from .agents.storage_agent import StorageAgent
        from .agents.network_agent import NetworkAgent
        from .agents.database_agent import DatabaseAgent
        
        logger.info("Initializing agents")
        
        # Create agent instances with updated configurations
        agent_configs = self.config.get('agents', {})
        
        # Add common RL parameters to each agent config
        common_rl_config = {
            'gamma': self.config.get('gamma', 0.99),
            'learning_rate': self.config.get('learning_rate', 0.001),
            'epsilon_start': self.config.get('epsilon_start', 1.0),
            'epsilon_min': self.config.get('epsilon_min', 0.01),
            'epsilon_decay': self.config.get('epsilon_decay', 0.995),
            'batch_size': self.config.get('batch_size', 64),
            'buffer_size': self.config.get('buffer_size', 100000),
            'update_frequency': self.config.get('update_frequency', 4),
            'target_update_frequency': self.config.get('target_update_frequency', 100),
            'hidden_sizes': self.config.get('hidden_sizes', [256, 256])
        }
        
        # Merge common config with agent-specific configs
        compute_config = {**common_rl_config, **agent_configs.get('compute', {})}
        compute_config.update({'state_size': 9, 'action_size': 81})
        
        storage_config = {**common_rl_config, **agent_configs.get('storage', {})}
        storage_config.update({'state_size': 4, 'action_size': 25})
        
        network_config = {**common_rl_config, **agent_configs.get('network', {})}
        network_config.update({'state_size': 7, 'action_size': 27})
        
        database_config = {**common_rl_config, **agent_configs.get('database', {})}
        database_config.update({'state_size': 7, 'action_size': 81})
        
        self.agents['compute'] = ComputeAgent(compute_config)
        self.agents['storage'] = StorageAgent(storage_config)
        self.agents['network'] = NetworkAgent(network_config)
        self.agents['database'] = DatabaseAgent(database_config)
        
        logger.info(f"Created {len(self.agents)} RL agents with DQN implementation")
    
    def _init_environment(self):
        """Initialize the environment interface to GCP."""
        from .environment.gcp_environment import GCPEnvironment
        
        logger.info("Initializing GCP environment")
        self.environment = GCPEnvironment(self.config.get('environment', {}))
    
    def _init_experience_buffer(self):
        """Initialize the shared experience buffer."""
        from .utils.experience_buffer import SharedExperienceBuffer
        
        logger.info("Initializing shared experience buffer")
        self.experience_buffer = SharedExperienceBuffer(
            buffer_size=self.config.get('buffer_size', 10000),
            batch_size=self.config.get('batch_size', 64)
        )
    
    def _init_monitoring(self):
        """Initialize monitoring components."""
        from .utils.monitoring import SystemMonitoring
        
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
            elif isinstance(value, dict):
                # Handle nested dictionaries by flattening their values
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float, np.number)):
                        flattened.append(float(subvalue))
                    elif isinstance(subvalue, np.ndarray):
                        flattened.extend(subvalue.flatten())
            elif isinstance(value, (int, float, np.number)):
                flattened.append(float(value))
        return np.array(flattened, dtype=np.float32)

    def _convert_discrete_to_continuous_action(self, agent_name: str, discrete_action: int) -> np.ndarray:
        """
        Convert discrete action index to continuous action array expected by environment.
        
        Args:
            agent_name: Name of the agent
            discrete_action: Discrete action index
            
        Returns:
            Continuous action array
        """
        if agent_name == 'compute':
            # Compute agent: 81 actions (3^4) -> 5 continuous values
            # Decode the action using the agent's decode_action method
            action_params = self.agents[agent_name].decode_action(discrete_action)
            
            # Map to continuous values (-1 to 1)
            instance_types = ['small', 'medium', 'large']
            cpu_configs = ['low', 'medium', 'high'] 
            memory_configs = ['low', 'medium', 'high']
            storage_configs = ['minimal', 'standard', 'high']
            
            # Convert to continuous scale
            scale_instances = (instance_types.index(action_params['instance_type']) - 1) / 1.0  # -1, 0, 1
            scale_cpu = (cpu_configs.index(action_params['cpu_config']) - 1) / 1.0
            scale_memory = (memory_configs.index(action_params['memory_config']) - 1) / 1.0
            region_preference = (storage_configs.index(action_params['storage_config']) - 1) / 1.0
            instance_type = scale_instances  # Reuse for simplicity
            
            return np.array([scale_instances, scale_cpu, scale_memory, region_preference, instance_type], dtype=np.float32)
            
        elif agent_name == 'storage':
            # Storage agent: similar conversion
            return np.random.uniform(-1, 1, 4).astype(np.float32)  # Placeholder
            
        elif agent_name == 'network':
            # Network agent: similar conversion  
            return np.random.uniform(-1, 1, 6).astype(np.float32)  # Placeholder
            
        elif agent_name == 'database':
            # Database agent: similar conversion
            return np.random.uniform(-1, 1, 4).astype(np.float32)  # Placeholder
            
        else:
            # Default fallback
            return np.random.uniform(-1, 1, 4).astype(np.float32)

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
            'costs': [],
            'agent_losses': {agent_name: [] for agent_name in self.agents.keys()}
        }
        
        for episode in range(episodes):
            # Reset environment and get initial observations
            observations = self.environment.reset()
            
            episode_reward = 0
            episode_steps = 0
            done = False
            episode_experiences = {agent_name: [] for agent_name in self.agents.keys()}
            
            # Episode loop
            while not done:
                # Get state features for each agent
                states = {}
                for agent_name, agent in self.agents.items():
                    agent_observation = observations.get(agent_name, {})
                    if hasattr(agent, 'get_state_features'):
                        states[agent_name] = agent.get_state_features(agent_observation)
                    else:
                        states[agent_name] = self._convert_observation_to_array(agent_observation)
                
                # Collect actions from all agents
                actions = {}
                for agent_name, agent in self.agents.items():
                    action_index = agent.select_action(states[agent_name])
                    # Convert discrete action to continuous action expected by environment
                    actions[agent_name] = self._convert_discrete_to_continuous_action(agent_name, action_index)
                
                # Step environment
                next_observations, rewards, done, info = self.environment.step(actions)
                
                # Get next states
                next_states = {}
                for agent_name, agent in self.agents.items():
                    agent_observation = next_observations.get(agent_name, {})
                    if hasattr(agent, 'get_state_features'):
                        next_states[agent_name] = agent.get_state_features(agent_observation)
                    else:
                        next_states[agent_name] = self._convert_observation_to_array(agent_observation)
                
                # Store experiences for each agent
                for agent_name, agent in self.agents.items():
                    agent_reward = rewards.get(agent_name, 0)
                    
                    # Store in agent's replay buffer
                    agent.remember(
                        states[agent_name],
                        actions[agent_name], 
                        agent_reward,
                        next_states[agent_name],
                        done
                    )
                    
                    # Also store in shared experience buffer
                    experience = {
                        'state': states[agent_name],
                        'action': actions[agent_name],
                        'reward': agent_reward,
                        'next_state': next_states[agent_name],
                        'done': done,
                        'agent': agent_name
                    }
                    self.experience_buffer.add(agent_name, states[agent_name], action_index, 
                                          agent_reward, next_states[agent_name], done)
                
                # Update episode tracking
                episode_reward += sum(rewards.values())
                episode_steps += 1
                observations = next_observations
                
                # Update agents (every few steps)
                if episode_steps % 4 == 0 and len(self.experience_buffer) >= 64:  # Update every 4 steps
                    for agent_name, agent in self.agents.items():
                        # Sample experiences for this agent
                        experiences = self.experience_buffer.sample(agent_name)
                        if experiences:
                            update_metrics = agent.update(experiences)
                            if update_metrics and 'loss' in update_metrics:
                                metrics['agent_losses'][agent_name].append(update_metrics['loss'])
            
            # End of episode processing
            for agent_name, agent in self.agents.items():
                agent.reset_episode()
            
            # Record episode metrics
            metrics['rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_steps)
            
            if hasattr(self.environment, 'get_resource_usage'):
                metrics['resource_usage'].append(self.environment.get_resource_usage())
            if hasattr(self.environment, 'get_total_cost'):
                metrics['costs'].append(self.environment.get_total_cost())
            
            # Update monitoring
            if self.monitoring:
                episode_metrics = {
                    'reward': episode_reward,
                    'steps': episode_steps,
                    'info': info
                }
                self.monitoring.update(episode, episode_metrics)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(metrics['rewards'][-10:])
                avg_length = np.mean(metrics['episode_lengths'][-10:])
                logger.info(f"Episode {episode + 1}/{episodes} - Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")
                
                # Log agent-specific metrics
                for agent_name in self.agents.keys():
                    if metrics['agent_losses'][agent_name]:
                        avg_loss = np.mean(metrics['agent_losses'][agent_name][-10:])
                        epsilon = self.agents[agent_name].epsilon
                        logger.info(f"  {agent_name} - Loss: {avg_loss:.4f}, Epsilon: {epsilon:.3f}")
        
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
                # Get state features for each agent
                states = {}
                for agent_name, agent in self.agents.items():
                    agent_observation = observations.get(agent_name, {})
                    if hasattr(agent, 'get_state_features'):
                        states[agent_name] = agent.get_state_features(agent_observation)
                    else:
                        states[agent_name] = self._convert_observation_to_array(agent_observation)
                
                # Collect actions from all agents (deterministic)
                actions = {}
                for agent_name, agent in self.agents.items():
                    agent_state = states[agent_name]
                    # Convert state dict to array if needed
                    if isinstance(agent_state, dict):
                        agent_state = np.array(list(agent_state.values()), dtype=np.float32)
                    
                    action_index = agent.select_action(agent_state, deterministic=True)
                    # Convert action index to action array
                    action_array = self._convert_action_index_to_array(action_index, agent_name)
                    actions[agent_name] = action_array
                
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
            
            if hasattr(self.environment, 'get_resource_usage'):
                metrics['resource_usage'].append(self.environment.get_resource_usage())
            if hasattr(self.environment, 'get_total_cost'):
                metrics['costs'].append(self.environment.get_total_cost())
            
            # Update monitoring with evaluation results
            if self.monitoring:
                self.monitoring.log_episode(episode, episode_reward, episode_steps, info)
        
        # Set agents back to training mode
        for agent in self.agents.values():
            agent.train()
            
        logger.info("Evaluation completed")
        return metrics
    
    def _convert_action_index_to_array(self, action_idx, agent_name):
        """
        Convert single action index to action array expected by environment.
        
        Args:
            action_idx: Single integer action index from agent
            agent_name: Name of the agent
            
        Returns:
            np.ndarray: Action array for environment
        """
        import numpy as np
        
        # Map action index to action components based on agent type
        if agent_name == 'compute':
            # 5 components: scale_instances, scale_cpu, scale_memory, region_preference, instance_type
            components = []
            temp_idx = action_idx
            for i in range(5):
                component_idx = temp_idx % 9  # 0-8 for 9 discrete values
                component_value = (component_idx / 4.0) - 1.0  # Map to [-1, 1]
                components.append(component_value)
                temp_idx = temp_idx // 9
            return np.array(components)
        
        elif agent_name == 'storage':
            # 5 components for storage actions
            components = []
            temp_idx = action_idx
            for i in range(5):
                component_idx = temp_idx % 9
                component_value = (component_idx / 4.0) - 1.0
                components.append(component_value)
                temp_idx = temp_idx // 9
            return np.array(components)
        
        elif agent_name == 'network':
            # 6 components for network actions
            components = []
            temp_idx = action_idx
            for i in range(6):
                component_idx = temp_idx % 5  # Fewer options for network
                component_value = (component_idx / 2.0) - 1.0  # Map to [-1, 1]
                components.append(component_value)
                temp_idx = temp_idx // 5
            return np.array(components)
        
        elif agent_name == 'database':
            # 5 components for database actions
            components = []
            temp_idx = action_idx
            for i in range(5):
                component_idx = temp_idx % 9
                component_value = (component_idx / 4.0) - 1.0
                components.append(component_value)
                temp_idx = temp_idx // 9
            return np.array(components)
        
        else:
            # Default: 5 zero components
            return np.zeros(5)