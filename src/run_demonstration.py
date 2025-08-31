#!/usr/bin/env python
"""
Demonstration script for MARL-GCP project.

This script runs the GCP environment with different workload patterns
and generates visualizations for presentation purposes.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marl_gcp.environment.gcp_environment import GCPEnvironment
from marl_gcp.configs.default_config import get_default_config
from test_environment import run_episode, plot_metrics

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demonstration.log')
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MARL-GCP Demonstration")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps per episode")
    parser.add_argument("--visualize_all", action="store_true", help="Visualize all workload patterns")
    
    return parser.parse_args()

def run_demonstration(steps, visualize_all=False):
    """
    Run demonstration with different workload patterns.
    
    Args:
        steps: Number of steps per episode
        visualize_all: Whether to visualize all workload patterns
    """
    print("\n" + "="*50)
    print("MARL-GCP Demonstration")
    print("="*50)
    
    # Load configuration
    config = get_default_config()
    
    # Update paths to use the correct directories
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config['cluster_data']['cache_dir'] = str(project_root / 'data' / 'google_cluster')
    config['cluster_data']['data_dir'] = str(project_root / 'data' / 'processed')
    config['viz_dir'] = str(project_root / 'visualizations')
    config['workload_generator']['data_dir'] = str(project_root / 'data' / 'processed')
    config['workload_generator']['viz_dir'] = str(project_root / 'visualizations')
    
    # Initialize environment
    env = GCPEnvironment(config)
    
    # Visualize all workload patterns if requested
    if visualize_all:
        print("\nVisualizing all workload patterns...")
        env.visualize_workloads(steps=100)
        print(f"Workload patterns visualization saved to {config['viz_dir']}/workload_patterns.png")
    
    # Run episodes with different workload patterns
    workload_patterns = ['steady', 'burst', 'cyclical']
    
    for i, pattern in enumerate(workload_patterns):
        print(f"\n{'-'*50}")
        print(f"Running demonstration with {pattern.upper()} workload pattern")
        print(f"{'-'*50}")
        
        # Set workload pattern
        success = env.set_workload_pattern(pattern)
        if success:
            print(f"Using workload pattern: {pattern}")
        else:
            print(f"Failed to set workload pattern: {pattern}")
            continue
        
        # Run episode
        metrics = run_episode(env, steps)
        
        # Plot metrics
        output_dir = f"visualizations/demo_{pattern}"
        plot_metrics(metrics, output_dir)
        
        print(f"\nEpisode with {pattern} workload completed")
        print(f"Total reward: {metrics['total_rewards']:.2f}")
        print(f"Visualizations saved to {output_dir}")
    
    print("\n" + "="*50)
    print("Demonstration completed")
    print("="*50)

def main():
    """Main function."""
    # Set up logging
    setup_logging()
    
    # Parse arguments
    args = parse_arguments()
    
    # Run demonstration
    run_demonstration(args.steps, args.visualize_all)

if __name__ == "__main__":
    main() 