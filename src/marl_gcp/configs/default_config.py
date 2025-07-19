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
    # Get project root directory
    project_root = Path(os.path.abspath(__file__)).parents[2]
    
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
        
        # Agent settings - Updated for real Google Cluster data features
        'agents': {
            'compute': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'hidden_dim': 256,
                'batch_size': 256,
                'buffer_size': 1000000,
                'state_dim': 9,   # cpu_rate(1) + memory_usage(1) + disk_usage(1) + io_time(1) + active_machines(1) + active_tasks(1) + cpu_memory_ratio(1) + io_ratio(1) + cost(1)
                'action_dim': 5   # Scale instances, CPU, memory, region, instance type
            },
            'storage': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'hidden_dim': 256,
                'batch_size': 256,
                'buffer_size': 1000000,
                'state_dim': 4,   # disk_usage(1) + io_time(1) + io_ratio(1) + cost(1)
                'action_dim': 4   # Scale storage, type, replication, location
            },
            'network': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'hidden_dim': 256,
                'batch_size': 256,
                'buffer_size': 1000000,
                'state_dim': 7,   # cpu_rate(1) + memory_usage(1) + active_machines(1) + active_tasks(1) + cpu_memory_ratio(1) + io_ratio(1) + cost(1)
                'action_dim': 6   # bandwidth_scale, vpc_config, subnet_count, firewall_rules, load_balancer, cdn_enabled
            },
            'database': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'hidden_dim': 256,
                'batch_size': 256,
                'buffer_size': 1000000,
                'state_dim': 7,   # cpu_rate(1) + memory_usage(1) + disk_usage(1) + io_time(1) + active_machines(1) + active_tasks(1) + cost(1)
                'action_dim': 3   # scale_instances, scale_storage, type
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
            'num_episodes': 1000,
            'eval_frequency': 10,
            'save_frequency': 50,
            'log_frequency': 1,
            'max_steps_per_episode': 200,
            'update_frequency': 1,
            'target_update_frequency': 10,
            'checkpoint_dir': str(project_root / 'checkpoints')
        },
        
        # Results and model directories
        'results_dir': str(project_root / 'results'),
        'save_dir': str(project_root / 'models'),
        'experiment_name': 'marl_gcp_default',
        'eval_episodes': 10,
        'num_episodes': 1000,
        'random_seed': 42,
        'log_level': 'INFO'
    }
    
    return config 