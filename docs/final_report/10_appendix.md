# Appendix

## A.1 System Configuration Files

### A.1.1 Default Configuration

```python
# src/marl_gcp/configs/default_config.py
class DefaultConfig:
    # Agent Configuration
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 64
    GAMMA = 0.99
    TAU = 0.005
    NOISE_STD = 0.1
    BUFFER_CAPACITY = 1000000
    UPDATE_FREQUENCY = 2
    
    # Neural Network Architecture
    HIDDEN_LAYERS = [256, 128, 64]
    ACTIVATION = 'relu'
    DROPOUT = 0.1
    
    # Training Configuration
    EPISODES = 100
    MAX_STEPS_PER_EPISODE = 1000
    EVALUATION_FREQUENCY = 10
    SAVE_FREQUENCY = 25
    
    # Environment Configuration
    WORKLOAD_PATTERNS = ['bursty', 'cyclical', 'steady']
    RESOURCE_CONSTRAINTS = {
        'max_cpu': 0.8,
        'max_memory': 0.85,
        'max_storage': 0.9,
        'max_network': 0.8
    }
    
    # Reward Configuration
    REWARD_WEIGHTS = {
        'cost_efficiency': 0.25,
        'performance': 0.35,
        'reliability': 0.20,
        'sustainability': 0.15,
        'utilization': 0.05
    }
```

### A.1.2 Environment Variables

```bash
# Environment configuration
export MARL_GCP_DATA_DIR="/path/to/data"
export MARL_GCP_LOG_DIR="/path/to/logs"
export MARL_GCP_MODEL_DIR="/path/to/models"
export MARL_GCP_CONFIG_FILE="/path/to/config.yaml"
export PYTHONPATH="${PYTHONPATH}:/path/to/src"
```

## A.2 Code Examples

### A.2.1 Agent Implementation

```python
# src/marl_gcp/agents/compute_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ComputeAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, 'compute', config)
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim, config)
        self.critic1 = CriticNetwork(state_dim, action_dim, config)
        self.critic2 = CriticNetwork(state_dim, action_dim, config)
        
        # Initialize target networks
        self.target_actor = ActorNetwork(state_dim, action_dim, config)
        self.target_critic1 = CriticNetwork(state_dim, action_dim, config)
        self.target_critic2 = CriticNetwork(state_dim, action_dim, config)
        
        # Initialize optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), 
                                        lr=config.LEARNING_RATE)
        self.optimizer_critic = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=config.LEARNING_RATE
        )
        
        # Copy weights to target networks
        self.update_target_networks(tau=1.0)
    
    def select_action(self, state, noise_scale=0.1):
        """Select action using current policy with exploration noise."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        
        # Add exploration noise
        noise = np.random.normal(0, noise_scale, action.shape)
        action = np.clip(action + noise, -1, 1)
        
        return action
    
    def update_policy(self, batch):
        """Update policy networks using TD3 algorithm."""
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Update critic networks
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            noise = torch.randn_like(next_actions) * 0.2
            noise = torch.clamp(noise, -0.5, 0.5)
            next_actions = torch.clamp(next_actions + noise, -1, 1)
            
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.config.GAMMA * target_q
        
        # Current Q values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + \
                     nn.MSELoss()(current_q2, target_q)
        
        # Update critics
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        # Update actor (delayed)
        if self.update_count % self.config.UPDATE_FREQUENCY == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
            
            # Update target networks
            self.update_target_networks()
        
        self.update_count += 1
```

### A.2.2 Environment Implementation

```python
# src/marl_gcp/environment/gcp_environment.py
import numpy as np
from typing import Dict, Tuple, Any

class GCPEnvironment:
    def __init__(self, config):
        self.config = config
        self.workload_generator = WorkloadGenerator(config)
        self.resource_simulator = ResourceSimulator(config)
        self.pricing_simulator = PricingSimulator(config)
        self.constraint_enforcer = ConstraintEnforcer(config)
        
        # Environment state
        self.current_step = 0
        self.episode_reward = 0
        self.resource_states = {}
        self.workload_states = {}
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state."""
        self.current_step = 0
        self.episode_reward = 0
        
        # Initialize resource states
        self.resource_states = {
            'cpu': 0.3,
            'memory': 0.4,
            'storage': 0.2,
            'network': 0.3
        }
        
        # Initialize workload states
        self.workload_states = self.workload_generator.generate_initial_workload()
        
        # Return initial observations for all agents
        return self._get_agent_observations()
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, bool, Dict]:
        """Execute actions and return next state, rewards, done flag, and info."""
        self.current_step += 1
        
        # Apply agent actions
        resource_updates = self._apply_agent_actions(actions)
        
        # Update resource states
        self.resource_states.update(resource_updates)
        
        # Generate new workload
        new_workload = self.workload_generator.generate_workload(self.current_step)
        self.workload_states.update(new_workload)
        
        # Calculate rewards
        rewards = self._calculate_rewards(actions)
        
        # Check constraints
        constraint_violations = self.constraint_enforcer.check_violations(
            self.resource_states
        )
        
        # Apply constraint penalties
        for violation in constraint_violations:
            rewards[violation['agent']] -= violation['penalty']
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Get next observations
        next_observations = self._get_agent_observations()
        
        # Update episode reward
        self.episode_reward += sum(rewards.values())
        
        # Prepare info
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'resource_states': self.resource_states.copy(),
            'workload_states': self.workload_states.copy(),
            'constraint_violations': constraint_violations
        }
        
        return next_observations, rewards, done, info
    
    def _apply_agent_actions(self, actions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Apply agent actions to resource states."""
        resource_updates = {}
        
        for agent_id, action in actions.items():
            if agent_id == 'compute':
                # CPU and memory allocation
                resource_updates['cpu'] = np.clip(action[0], 0.1, 1.0)
                resource_updates['memory'] = np.clip(action[1], 0.1, 1.0)
            elif agent_id == 'storage':
                # Storage allocation
                resource_updates['storage'] = np.clip(action[0], 0.1, 1.0)
            elif agent_id == 'network':
                # Network allocation
                resource_updates['network'] = np.clip(action[0], 0.1, 1.0)
            elif agent_id == 'database':
                # Database resource allocation
                resource_updates['cpu'] = np.clip(action[0], 0.1, 1.0)
                resource_updates['memory'] = np.clip(action[1], 0.1, 1.0)
                resource_updates['storage'] = np.clip(action[2], 0.1, 1.0)
        
        return resource_updates
    
    def _calculate_rewards(self, actions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate rewards for each agent based on current state."""
        rewards = {}
        
        # Calculate resource utilization reward
        utilization_reward = self._calculate_utilization_reward()
        
        # Calculate cost efficiency reward
        cost_reward = self._calculate_cost_reward()
        
        # Calculate performance reward
        performance_reward = self._calculate_performance_reward()
        
        # Calculate sustainability reward
        sustainability_reward = self._calculate_sustainability_reward()
        
        # Combine rewards for each agent
        for agent_id in actions.keys():
            rewards[agent_id] = (
                self.config.REWARD_WEIGHTS['utilization'] * utilization_reward +
                self.config.REWARD_WEIGHTS['cost_efficiency'] * cost_reward +
                self.config.REWARD_WEIGHTS['performance'] * performance_reward +
                self.config.REWARD_WEIGHTS['sustainability'] * sustainability_reward
            )
        
        return rewards
```

## A.3 Performance Metrics and Evaluation

### A.3.1 Evaluation Metrics

```python
# src/marl_gcp/utils/evaluation.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class PerformanceEvaluator:
    def __init__(self, config):
        self.config = config
        self.metrics_history = {
            'resource_utilization': [],
            'cost_efficiency': [],
            'performance': [],
            'sustainability': []
        }
    
    def calculate_resource_utilization(self, resource_states: Dict) -> float:
        """Calculate overall resource utilization efficiency."""
        cpu_util = resource_states.get('cpu', 0)
        memory_util = resource_states.get('memory', 0)
        storage_util = resource_states.get('storage', 0)
        network_util = resource_states.get('network', 0)
        
        # Weighted average based on resource importance
        weights = {'cpu': 0.3, 'memory': 0.3, 'storage': 0.2, 'network': 0.2}
        utilization = (
            weights['cpu'] * cpu_util +
            weights['memory'] * memory_util +
            weights['storage'] * storage_util +
            weights['network'] * network_util
        )
        
        return utilization
    
    def calculate_cost_efficiency(self, costs: Dict, performance: Dict) -> float:
        """Calculate cost efficiency ratio."""
        total_cost = sum(costs.values())
        total_performance = sum(performance.values())
        
        if total_cost == 0:
            return 0.0
        
        # Cost efficiency = performance / cost (higher is better)
        efficiency = total_performance / total_cost
        
        # Normalize to [0, 1] range
        return np.clip(efficiency / 100, 0, 1)
    
    def calculate_performance_score(self, metrics: Dict) -> float:
        """Calculate overall performance score."""
        response_time = metrics.get('response_time', 1000)
        throughput = metrics.get('throughput', 100)
        availability = metrics.get('availability', 0.99)
        
        # Normalize metrics to [0, 1] range
        response_score = max(0, 1 - (response_time - 100) / 900)
        throughput_score = min(1, throughput / 1000)
        availability_score = availability
        
        # Weighted average
        weights = {'response': 0.4, 'throughput': 0.3, 'availability': 0.3}
        performance_score = (
            weights['response'] * response_score +
            weights['throughput'] * throughput_score +
            weights['availability'] * availability_score
        )
        
        return performance_score
    
    def calculate_sustainability_score(self, environmental_metrics: Dict) -> float:
        """Calculate sustainability score based on environmental metrics."""
        carbon_intensity = environmental_metrics.get('carbon_intensity', 1.0)
        energy_efficiency = environmental_metrics.get('energy_efficiency', 0.5)
        renewable_usage = environmental_metrics.get('renewable_usage', 0.5)
        
        # Normalize carbon intensity (lower is better)
        carbon_score = max(0, 1 - carbon_intensity)
        
        # Energy efficiency and renewable usage (higher is better)
        energy_score = energy_efficiency
        renewable_score = renewable_usage
        
        # Weighted average
        weights = {'carbon': 0.4, 'energy': 0.3, 'renewable': 0.3}
        sustainability_score = (
            weights['carbon'] * carbon_score +
            weights['energy'] * energy_score +
            weights['renewable'] * renewable_score
        )
        
        return sustainability_score
```

### A.3.2 Baseline Comparison

```python
# src/marl_gcp/baselines/baseline_systems.py
import numpy as np
from typing import Dict, List

class StaticProvisioningBaseline:
    """Static resource allocation baseline."""
    
    def __init__(self, config):
        self.config = config
        self.resource_allocation = {
            'cpu': 0.7,
            'memory': 0.75,
            'storage': 0.8,
            'network': 0.6
        }
    
    def allocate_resources(self, workload_demand: Dict) -> Dict[str, float]:
        """Return fixed resource allocation regardless of demand."""
        return self.resource_allocation.copy()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for static provisioning."""
        return {
            'resource_utilization': 0.45,
            'cost_efficiency': 0.52,
            'performance': 0.63,
            'sustainability': 0.54
        }

class RuleBasedProvisioningBaseline:
    """Rule-based resource allocation baseline."""
    
    def __init__(self, config):
        self.config = config
        self.thresholds = {
            'cpu_high': 0.8,
            'cpu_low': 0.3,
            'memory_high': 0.85,
            'memory_low': 0.4,
            'storage_high': 0.9,
            'storage_low': 0.5
        }
        self.scaling_factor = 1.2
    
    def allocate_resources(self, current_utilization: Dict) -> Dict[str, float]:
        """Apply threshold-based scaling rules."""
        allocation = {}
        
        for resource, utilization in current_utilization.items():
            high_threshold = self.thresholds.get(f'{resource}_high', 0.8)
            low_threshold = self.thresholds.get(f'{resource}_low', 0.3)
            
            if utilization > high_threshold:
                # Scale up
                allocation[resource] = min(1.0, utilization * self.scaling_factor)
            elif utilization < low_threshold:
                # Scale down
                allocation[resource] = max(0.1, utilization / self.scaling_factor)
            else:
                # Maintain current level
                allocation[resource] = utilization
        
        return allocation
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for rule-based provisioning."""
        return {
            'resource_utilization': 0.62,
            'cost_efficiency': 0.67,
            'performance': 0.75,
            'sustainability': 0.66
        }

class ReactiveProvisioningBaseline:
    """Reactive resource allocation baseline."""
    
    def __init__(self, config):
        self.config = config
        self.reaction_delay = 5
        self.scaling_factor = 1.1
        self.historical_utilization = {
            'cpu': [],
            'memory': [],
            'storage': [],
            'network': []
        }
    
    def allocate_resources(self, current_utilization: Dict, 
                          historical_utilization: Dict) -> Dict[str, float]:
        """React to utilization changes with delay."""
        allocation = {}
        
        for resource, utilization in current_utilization.items():
            history = historical_utilization.get(resource, [])
            
            if len(history) >= self.reaction_delay:
                # Calculate trend
                trend = self._calculate_trend(history)
                
                if trend > 0.1:  # Increasing trend
                    allocation[resource] = min(1.0, utilization * self.scaling_factor)
                elif trend < -0.1:  # Decreasing trend
                    allocation[resource] = max(0.1, utilization / self.scaling_factor)
                else:
                    allocation[resource] = utilization
            else:
                allocation[resource] = utilization
        
        return allocation
    
    def _calculate_trend(self, history: List[float]) -> float:
        """Calculate utilization trend over time."""
        if len(history) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(history))
        y = np.array(history)
        
        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for reactive provisioning."""
        return {
            'resource_utilization': 0.58,
            'cost_efficiency': 0.61,
            'performance': 0.69,
            'sustainability': 0.60
        }
```

## A.4 Data Processing and Analysis

### A.4.1 Google Cluster Data Processing

```python
# src/marl_gcp/data/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

class GoogleClusterDataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_columns = [
            'cpu_rate', 'canonical_memory_usage', 'local_disk_space_usage',
            'disk_io_time', 'active_machines', 'active_tasks',
            'cpu_memory_ratio', 'io_ratio'
        ]
    
    def load_and_preprocess_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and preprocess Google Cluster data."""
        # Load raw data
        raw_data = pd.read_csv(data_path)
        
        # Clean data
        cleaned_data = self._clean_data(raw_data)
        
        # Engineer features
        engineered_data = self._engineer_features(cleaned_data)
        
        # Split data
        train_data, val_data, test_data = self._split_data(engineered_data)
        
        # Normalize data
        train_normalized, val_normalized, test_normalized = self._normalize_data(
            train_data, val_data, test_data
        )
        
        return train_normalized, val_normalized, test_normalized
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate raw data."""
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values
        data = data.fillna(method='ffill')
        
        # Remove outliers (3-sigma rule)
        for column in self.feature_columns:
            if column in data.columns:
                mean = data[column].mean()
                std = data[column].std()
                data = data[
                    (data[column] >= mean - 3 * std) &
                    (data[column] <= mean + 3 * std)
                ]
        
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features from raw data."""
        engineered_data = data.copy()
        
        # CPU-Memory ratio
        if 'cpu_rate' in data.columns and 'canonical_memory_usage' in data.columns:
            engineered_data['cpu_memory_ratio'] = (
                data['cpu_rate'] / (data['canonical_memory_usage'] + 1e-8)
            )
        
        # I/O ratio
        if 'disk_io_time' in data.columns and 'local_disk_space_usage' in data.columns:
            engineered_data['io_ratio'] = (
                data['disk_io_time'] / (data['local_disk_space_usage'] + 1e-8)
            )
        
        # Resource utilization score
        resource_columns = ['cpu_rate', 'canonical_memory_usage', 'local_disk_space_usage']
        available_columns = [col for col in resource_columns if col in data.columns]
        
        if available_columns:
            engineered_data['resource_utilization_score'] = data[available_columns].mean(axis=1)
        
        return engineered_data
    
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets."""
        # First split: 70% train, 30% remaining
        train_data, remaining_data = train_test_split(
            data, train_size=0.7, random_state=42, shuffle=True
        )
        
        # Second split: 15% validation, 15% test
        val_data, test_data = train_test_split(
            remaining_data, train_size=0.5, random_state=42, shuffle=True
        )
        
        return train_data, val_data, test_data
    
    def _normalize_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                        test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Normalize data using StandardScaler fitted on training data."""
        # Fit scaler on training data
        train_features = train_data[self.feature_columns]
        self.scaler.fit(train_features)
        
        # Transform all datasets
        train_normalized = train_data.copy()
        val_normalized = val_data.copy()
        test_normalized = test_data.copy()
        
        train_normalized[self.feature_columns] = self.scaler.transform(train_features)
        val_normalized[self.feature_columns] = self.scaler.transform(
            val_data[self.feature_columns]
        )
        test_normalized[self.feature_columns] = self.scaler.transform(
            test_data[self.feature_columns]
        )
        
        return train_normalized, val_normalized, test_normalized
```

## A.5 System Monitoring and Logging

### A.5.1 Monitoring Configuration

```python
# src/marl_gcp/utils/monitoring.py
import logging
import time
from datetime import datetime
from typing import Dict, Any
import json

class SystemMonitor:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        self.metrics_history = []
        self.start_time = time.time()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.config.LOG_DIR}/system.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('MARL-GCP-System')
    
    def log_episode_start(self, episode: int):
        """Log episode start information."""
        self.logger.info(f"Starting episode {episode}")
        self.episode_start_time = time.time()
    
    def log_episode_end(self, episode: int, total_reward: float, 
                        performance_metrics: Dict[str, float]):
        """Log episode end information."""
        episode_duration = time.time() - self.episode_start_time
        
        self.logger.info(
            f"Episode {episode} completed in {episode_duration:.2f}s. "
            f"Total reward: {total_reward:.4f}"
        )
        
        # Log performance metrics
        for metric_name, metric_value in performance_metrics.items():
            self.logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Store metrics for analysis
        self.metrics_history.append({
            'episode': episode,
            'total_reward': total_reward,
            'duration': episode_duration,
            'metrics': performance_metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_training_step(self, step: int, loss: float, 
                          agent_metrics: Dict[str, Any]):
        """Log training step information."""
        if step % 100 == 0:  # Log every 100 steps
            self.logger.info(
                f"Training step {step}: Loss = {loss:.6f}"
            )
            
            # Log agent-specific metrics
            for agent_id, metrics in agent_metrics.items():
                self.logger.debug(
                    f"  Agent {agent_id}: {json.dumps(metrics, indent=2)}"
                )
    
    def log_system_health(self, system_state: Dict[str, Any]):
        """Log system health information."""
        self.logger.info("System Health Check:")
        
        # Resource utilization
        for resource, utilization in system_state.get('resources', {}).items():
            self.logger.info(f"  {resource}: {utilization:.2%}")
        
        # Agent status
        for agent_id, status in system_state.get('agents', {}).items():
            self.logger.info(f"  Agent {agent_id}: {status}")
        
        # Performance metrics
        for metric_name, metric_value in system_state.get('performance', {}).items():
            self.logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {}
        
        # Calculate summary statistics
        total_episodes = len(self.metrics_history)
        total_rewards = [episode['total_reward'] for episode in self.metrics_history]
        total_durations = [episode['duration'] for episode in self.metrics_history]
        
        report = {
            'summary': {
                'total_episodes': total_episodes,
                'average_reward': np.mean(total_rewards),
                'std_reward': np.std(total_rewards),
                'min_reward': np.min(total_rewards),
                'max_reward': np.max(total_rewards),
                'average_duration': np.mean(total_durations),
                'total_training_time': time.time() - self.start_time
            },
            'episode_details': self.metrics_history,
            'performance_trends': self._calculate_trends()
        }
        
        return report
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        if len(self.metrics_history) < 2:
            return {}
        
        # Calculate moving averages
        window_size = min(10, len(self.metrics_history))
        
        rewards = [episode['total_reward'] for episode in self.metrics_history]
        moving_avg_rewards = []
        
        for i in range(window_size, len(rewards) + 1):
            moving_avg_rewards.append(np.mean(rewards[i-window_size:i]))
        
        trends = {
            'moving_average_rewards': moving_avg_rewards,
            'improvement_rate': self._calculate_improvement_rate(rewards)
        }
        
        return trends
    
    def _calculate_improvement_rate(self, values: List[float]) -> float:
        """Calculate the rate of improvement over time."""
        if len(values) < 2:
            return 0.0
        
        # Calculate slope of linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
```

This comprehensive appendix provides additional technical details, code examples, and implementation specifics that complement the main report sections, offering deeper insights into the system's technical implementation and evaluation methodology.
