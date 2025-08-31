"""
Default configuration for the MARL-GCP project.
"""

from typing import Dict, Any
import os
from pathlib import Path

def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration for the MARL-GCP project.
    
    Returns:
        Dictionary with default configuration
    """
    # Get project root directory (src/marl_gcp/configs -> src/marl_gcp -> src -> project_root)
    project_root = Path(os.path.abspath(__file__)).parents[3]
    
    # Base configuration
    config = {
        # General settings
        'project_name': 'MARL-GCP',
        'seed': 42,
        'debug': False,
        'project_root': str(project_root),
        
        # Environment settings
        'simulation_mode': True,  # True for simulation, False for real GCP
        'gcp_project_id': 'your-gcp-project-id',
        'use_terraform': False,
        'reward_scale': 1.0,
        'max_steps_per_episode': 200,
        
        # Resource constraints
        'resource_constraints': {
            'max_instances': 100,
            'max_cpus': 500,
            'max_memory_gb': 1000,
            'max_storage_gb': 10000,
            'max_budget': 1000.0
        },
        
        # Delay simulation
        'delay_simulation': {
            'provision_delay_mean': 5,  # steps
            'provision_delay_std': 2
        },
        
        # Workload generator
        'use_workload_generator': True,
        'workload_generator': {
            'type': 'google_cluster',  # Force google_cluster type
            'use_real_data': True,    # Ensure real data is used
            'data_path': os.path.join(project_root, 'data', 'processed', 'processed_data.parquet'),
            'feature_stats_path': os.path.join(project_root, 'data', 'processed', 'feature_statistics.json'),
            'workload_patterns_path': os.path.join(project_root, 'data', 'processed', 'workload_patterns.json'),
            'cluster_patterns_path': os.path.join(project_root, 'data', 'processed', 'cluster_workload_patterns.json'),
            'pattern_transitions': True,
            'pattern_duration_steps': 20,
            'seasonal_effects': True
        },
        'workload_generator': {
            'data_dir': str(project_root / 'data' / 'processed'),
            'viz_dir': str(project_root / 'visualizations'),
            'default_pattern': 'steady'
        },
        
        # Google Cluster Data
        'cluster_data': {
            'cache_dir': str(project_root / 'data' / 'google_cluster'),
            'data_dir': str(project_root / 'data' / 'processed'),
            'start_time': 0,  # in seconds from trace start
            'end_time': 86400,  # 24 hours
            'machine_subset': None,
            'task_subset': None
        },
        
        # Visualization
        'viz_dir': str(project_root / 'visualizations'),
        
        # RL Training Parameters
        'gamma': 0.99,                    # Discount factor
        'learning_rate': 0.001,           # Learning rate for all agents
        'epsilon_start': 1.0,             # Initial exploration rate
        'epsilon_min': 0.01,              # Minimum exploration rate
        'epsilon_decay': 0.995,           # Exploration decay rate
        'batch_size': 64,                 # Batch size for training
        'buffer_size': 100000,            # Experience replay buffer size
        'update_frequency': 4,            # Steps between agent updates
        'target_update_frequency': 100,   # Steps between target network updates
        'hidden_sizes': [256, 256],       # Neural network hidden layer sizes
        'tau': 0.005,                     # Soft update parameter for target networks
        
        # Agent settings - Updated for DQN implementation
        'agents': {
            'action_space_size': 4,  # Reduced to prevent indexing errors
            'compute': {
                'max_instances': 100,
                'max_cpus': 500,
                'max_memory': 1000,
                'instance_types': ['n1-standard-1', 'n1-standard-2', 'n1-standard-4'],
                'cpu_efficiency_weight': 0.3,
                'memory_efficiency_weight': 0.3,
                'cost_efficiency_weight': 0.4
            },
            'storage': {
                'max_storage_gb': 10000,
                'storage_types': ['standard', 'ssd', 'archive'],
                'utilization_weight': 0.4,
                'cost_efficiency_weight': 0.3,
                'performance_weight': 0.3
            },
            'network': {
                'max_bandwidth_gbps': 100,
                'max_subnets': 20,
                'latency_weight': 0.4,
                'bandwidth_efficiency_weight': 0.3,
                'cost_efficiency_weight': 0.3
            },
            'database': {
                'max_db_instances': 50,
                'max_db_storage_gb': 5000,
                'database_types': ['sql', 'nosql', 'managed'],
                'query_performance_weight': 0.4,
                'resource_efficiency_weight': 0.3,
                'cost_efficiency_weight': 0.3
            }
        },
        
        # Shared experience buffer
        'experience_buffer': {
            'capacity': 1000000,
            'batch_size': 256,
            'alpha': 0.6,  # Prioritization exponent
            'beta': 0.4,  # Importance sampling exponent
            'beta_annealing': 0.001  # Annealing rate for beta
        },
        
        # Training settings
        'training': {
            'num_episodes': 3,  # Reduced for testing
            'eval_frequency': 10,
            'save_frequency': 50,
            'log_frequency': 1,
            'max_steps_per_episode': 50,  # Reduced for faster testing
            'update_frequency': 1,
            'target_update_frequency': 10,
            'checkpoint_dir': str(project_root / 'checkpoints')
        },
        
        # Results and model directories
        'results_dir': str(project_root / 'results'),
        'save_dir': str(project_root / 'models'),
        'experiment_name': 'marl_gcp_real_data',  # Updated experiment name
        'eval_episodes': 2,  # Reduced for testing
        'num_episodes': 3,   # Reduced for testing
        'random_seed': 42,
        'log_level': 'INFO',
        
        # Enhanced Reward Engine Configuration
        'reward_engine': {
            'reward_history_size': 1000,
            'immediate_reward_weight': 0.7,
            'delayed_reward_weight': 0.3,
            'budget_limit': 1000.0,
            'reward_shaping_enabled': True,
            'reward_shaping_decay': 0.99,
            'sustainability_weight': 0.15,
            'performance_weight': 0.35,
            'cost_weight': 0.25,
            'reliability_weight': 0.20,
            'utilization_weight': 0.05
        }
    }
    
    return config 