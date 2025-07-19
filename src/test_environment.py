"""
Test script for the GCP environment.

This script tests the GCP environment implementation with a simple random agent.
"""

import os
import sys
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marl_gcp.environment.gcp_environment import GCPEnvironment
from marl_gcp.configs.default_config import get_default_config

def setup_logging(log_level):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_environment.log')
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test GCP Environment")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps per episode")
    parser.add_argument("--workload_pattern", type=str, default=None, help="Workload pattern to use (steady, burst, cyclical)")
    parser.add_argument("--visualize", action="store_true", help="Visualize workload patterns")
    
    return parser.parse_args()

def random_agent(observation_space, action_space):
    """
    Simple random agent that selects random actions.
    
    Args:
        observation_space: Observation space
        action_space: Action space
        
    Returns:
        Random action
    """
    return {
        agent_name: action_space[agent_name].sample()
        for agent_name in action_space.spaces
    }

def run_episode(env, steps, random_actions=True):
    """
    Run a single episode in the environment.
    
    Args:
        env: Environment
        steps: Number of steps
        random_actions: Whether to use random actions
        
    Returns:
        Episode metrics
    """
    # Reset environment
    observations = env.reset()
    
    # Initialize metrics
    metrics = {
        'rewards': [],
        'total_rewards': 0,
        'resource_usage': [],
        'costs': [],
        'utilization': []
    }
    
    # Run episode
    for step in range(steps):
        # Select actions
        if random_actions:
            actions = random_agent(env.observation_space, env.action_space)
        else:
            # Use a simple heuristic policy
            actions = {
                'compute': np.array([0.5, 0.5, 0.5, 0, 0]),  # Increase resources moderately
                'storage': np.array([0.3, 0, 0, 0]),  # Increase storage slightly
                'network': np.array([0, 0, 0, 0, 0, 0]),  # No network changes
                'database': np.array([0, 0, 0])  # No database changes
            }
        
        # Take step in environment
        next_observations, rewards, done, info = env.step(actions)
        
        # Update observations
        observations = next_observations
        
        # Update metrics
        metrics['rewards'].append(rewards)
        metrics['total_rewards'] += sum(rewards.values())
        metrics['resource_usage'].append(info['resource_usage'])
        metrics['costs'].append(info['costs'])
        metrics['utilization'].append(info['utilization'])
        
        # Print step information
        print(f"Step {step+1}/{steps} - Rewards: {sum(rewards.values()):.2f}")
        
        # Check if episode is done
        if done:
            print(f"Episode ended early at step {step+1}")
            break
    
    return metrics

def plot_metrics(metrics, output_dir):
    """
    Plot episode metrics.
    
    Args:
        metrics: Episode metrics
        output_dir: Output directory for plots
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    rewards_data = np.array([[r[agent] for agent in r] for r in metrics['rewards']])
    for i, agent in enumerate(metrics['rewards'][0].keys()):
        plt.plot(rewards_data[:, i], label=agent)
    plt.title('Agent Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'rewards.png')
    plt.close()
    
    # Plot resource usage
    plt.figure(figsize=(10, 6))
    steps = range(len(metrics['resource_usage']))
    plt.plot(steps, [r['instances'] for r in metrics['resource_usage']], label='Instances')
    plt.plot(steps, [r['cpus'] for r in metrics['resource_usage']], label='CPUs')
    plt.plot(steps, [r['memory_gb'] for r in metrics['resource_usage']], label='Memory (GB)')
    plt.plot(steps, [r['storage_gb'] for r in metrics['resource_usage']], label='Storage (GB)')
    plt.title('Resource Usage')
    plt.xlabel('Step')
    plt.ylabel('Amount')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'resource_usage.png')
    plt.close()
    
    # Plot costs
    plt.figure(figsize=(10, 6))
    costs_data = np.array([[c.get(agent, 0) for agent in ['compute', 'storage', 'network', 'database']] 
                           for c in metrics['costs']])
    for i, agent in enumerate(['compute', 'storage', 'network', 'database']):
        plt.plot(costs_data[:, i], label=agent)
    plt.title('Costs')
    plt.xlabel('Step')
    plt.ylabel('Cost ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'costs.png')
    plt.close()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = get_default_config()
    
    # Update paths to use the correct directories
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config['cluster_data']['cache_dir'] = str(project_root / 'data' / 'google_cluster')
    config['cluster_data']['data_dir'] = str(project_root / 'data' / 'processed')
    config['viz_dir'] = str(project_root / 'visualizations')
    config['workload_generator']['data_dir'] = str(project_root / 'data' / 'processed')
    config['workload_generator']['viz_dir'] = str(project_root / 'visualizations')
    
    # Override configuration
    config['max_steps_per_episode'] = args.steps
    
    # Initialize environment
    env = GCPEnvironment(config)
    
    # Set workload pattern if specified
    if args.workload_pattern:
        success = env.set_workload_pattern(args.workload_pattern)
        if success:
            print(f"Using workload pattern: {args.workload_pattern}")
        else:
            print(f"Failed to set workload pattern: {args.workload_pattern}")
    
    # Visualize workload patterns if requested
    if args.visualize:
        print("Visualizing workload patterns...")
        env.visualize_workloads(steps=100)
    
    # Run episodes
    for episode in range(args.episodes):
        print(f"\nRunning episode {episode+1}/{args.episodes}")
        metrics = run_episode(env, args.steps)
        
        # Plot metrics
        plot_metrics(metrics, f"visualizations/episode_{episode+1}")
        
        print(f"Episode {episode+1} completed - Total reward: {metrics['total_rewards']:.2f}")
    
    print("\nTest completed")

if __name__ == "__main__":
    main() 