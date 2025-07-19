#!/usr/bin/env python
"""
Retrain agents with real Google Cluster data.

This script retrains all MARL agents using the real Google Cluster data
instead of synthetic data, implementing the 70/15/15 train/val/test split.
"""

import os
import sys
import logging
import numpy as np
import torch
from pathlib import Path
import shutil

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marl_gcp.system_architecture import MARLSystemArchitecture
from marl_gcp.configs.default_config import get_default_config

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('retrain_real_data.log')
        ]
    )

def backup_existing_models():
    """Backup existing models before retraining."""
    print("Backing up existing models...")
    
    models_dir = Path("src/models/marl_gcp_default")
    backup_dir = Path("src/models/backup_before_real_data")
    
    if models_dir.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(models_dir, backup_dir)
        print(f"Models backed up to {backup_dir}")
    else:
        print("No existing models found to backup")

def delete_existing_models():
    """Delete existing models to force retraining."""
    print("Deleting existing models for fresh training...")
    
    models_dir = Path("src/models/marl_gcp_default")
    if models_dir.exists():
        for model_file in models_dir.glob("*.pt"):
            model_file.unlink()
        print("Existing models deleted")
    else:
        print("No existing models to delete")

def train_with_real_data(episodes=1000, eval_episodes=100):
    """
    Train the MARL system with real Google Cluster data.
    
    Args:
        episodes: Number of training episodes
        eval_episodes: Number of evaluation episodes
    """
    print(f"\nStarting training with real Google Cluster data")
    print(f"Training episodes: {episodes}")
    print(f"Evaluation episodes: {eval_episodes}")
    
    # Load configuration
    config = get_default_config()
    
    # Update paths for real data
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config['cluster_data']['data_dir'] = str(project_root / 'data' / 'processed')
    config['cluster_data']['cache_dir'] = str(project_root / 'data' / 'google_cluster')
    config['workload_generator']['data_dir'] = str(project_root / 'data' / 'processed')
    config['workload_generator']['cache_dir'] = str(project_root / 'data' / 'google_cluster')
    config['workload_generator']['viz_dir'] = str(project_root / 'visualizations')
    
    # Update training parameters
    config['num_episodes'] = episodes
    config['eval_episodes'] = eval_episodes
    config['experiment_name'] = 'marl_gcp_real_data'
    
    # Initialize system
    print("Initializing MARL system with real data...")
    system = MARLSystemArchitecture(config)
    
    # Training loop with real data
    print("\nStarting training loop...")
    train_metrics = system.train(episodes)
    
    # Save trained models
    save_path = Path(config['save_dir']) / config['experiment_name']
    save_path.mkdir(parents=True, exist_ok=True)
    
    for agent_name, agent in system.agents.items():
        agent.save(str(save_path))
    
    print(f"\nTraining completed. Models saved to {save_path}")
    
    # Run evaluation
    print("\nRunning evaluation with real data...")
    eval_metrics = system.evaluate(eval_episodes)
    
    # Print results
    print(f"\nTraining Results:")
    print(f"  Average training reward: {np.mean(train_metrics['rewards']):.3f}")
    print(f"  Average episode length: {np.mean(train_metrics['episode_lengths']):.1f}")
    
    print(f"\nEvaluation Results:")
    print(f"  Average evaluation reward: {np.mean(eval_metrics['rewards']):.3f}")
    print(f"  Average evaluation episode length: {np.mean(eval_metrics['episode_lengths']):.1f}")
    
    return train_metrics, eval_metrics

def test_different_workload_patterns():
    """Test the trained agents on different workload patterns."""
    print("\n" + "="*50)
    print("Testing Trained Agents on Different Workload Patterns")
    print("="*50)
    
    # Load configuration
    config = get_default_config()
    
    # Update paths
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config['cluster_data']['data_dir'] = str(project_root / 'data' / 'processed')
    config['cluster_data']['cache_dir'] = str(project_root / 'data' / 'google_cluster')
    config['workload_generator']['data_dir'] = str(project_root / 'data' / 'processed')
    config['workload_generator']['cache_dir'] = str(project_root / 'data' / 'google_cluster')
    config['workload_generator']['viz_dir'] = str(project_root / 'visualizations')
    
    # Initialize system
    system = MARLSystemArchitecture(config)
    
    # Load trained models
    model_path = Path(config['save_dir']) / config['experiment_name']
    if model_path.exists():
        for agent_name, agent in system.agents.items():
            agent.load(str(model_path))
        print("Loaded trained models")
    else:
        print("No trained models found")
        return
    
    # Test different patterns
    patterns = ['steady', 'burst', 'cyclical']
    results = {}
    
    for pattern in patterns:
        print(f"\nTesting {pattern} pattern:")
        
        # Set pattern in environment
        if hasattr(system.environment, 'set_workload_pattern'):
            system.environment.set_workload_pattern(pattern)
        
        # Run evaluation
        eval_metrics = system.evaluate(50)  # 50 episodes per pattern
        
        avg_reward = np.mean(eval_metrics['rewards'])
        avg_length = np.mean(eval_metrics['episode_lengths'])
        
        results[pattern] = {
            'avg_reward': avg_reward,
            'avg_length': avg_length
        }
        
        print(f"  Average reward: {avg_reward:.3f}")
        print(f"  Average episode length: {avg_length:.1f}")
    
    # Print summary
    print(f"\nPattern Performance Summary:")
    print(f"{'Pattern':<12} {'Avg Reward':<12} {'Avg Length':<12}")
    print("-" * 36)
    for pattern, metrics in results.items():
        print(f"{pattern:<12} {metrics['avg_reward']:<12.3f} {metrics['avg_length']:<12.1f}")

def generate_training_report(train_metrics, eval_metrics):
    """Generate a training report."""
    print("\n" + "="*50)
    print("TRAINING REPORT - Real Google Cluster Data")
    print("="*50)
    
    print(f"\nTraining Statistics:")
    print(f"  Total training episodes: {len(train_metrics['rewards'])}")
    print(f"  Average training reward: {np.mean(train_metrics['rewards']):.3f}")
    print(f"  Training reward std: {np.std(train_metrics['rewards']):.3f}")
    print(f"  Average episode length: {np.mean(train_metrics['episode_lengths']):.1f}")
    print(f"  Best training reward: {np.max(train_metrics['rewards']):.3f}")
    print(f"  Worst training reward: {np.min(train_metrics['rewards']):.3f}")
    
    print(f"\nEvaluation Statistics:")
    print(f"  Total evaluation episodes: {len(eval_metrics['rewards'])}")
    print(f"  Average evaluation reward: {np.mean(eval_metrics['rewards']):.3f}")
    print(f"  Evaluation reward std: {np.std(eval_metrics['rewards']):.3f}")
    print(f"  Average episode length: {np.mean(eval_metrics['episode_lengths']):.1f}")
    print(f"  Best evaluation reward: {np.max(eval_metrics['rewards']):.3f}")
    print(f"  Worst evaluation reward: {np.min(eval_metrics['rewards']):.3f}")
    
    # Save report
    report_path = Path("training_report_real_data.txt")
    with open(report_path, 'w') as f:
        f.write("MARL-GCP Training Report - Real Google Cluster Data\n")
        f.write("="*50 + "\n\n")
        
        f.write("Training Statistics:\n")
        f.write(f"  Total training episodes: {len(train_metrics['rewards'])}\n")
        f.write(f"  Average training reward: {np.mean(train_metrics['rewards']):.3f}\n")
        f.write(f"  Training reward std: {np.std(train_metrics['rewards']):.3f}\n")
        f.write(f"  Average episode length: {np.mean(train_metrics['episode_lengths']):.1f}\n")
        f.write(f"  Best training reward: {np.max(train_metrics['rewards']):.3f}\n")
        f.write(f"  Worst training reward: {np.min(train_metrics['rewards']):.3f}\n\n")
        
        f.write("Evaluation Statistics:\n")
        f.write(f"  Total evaluation episodes: {len(eval_metrics['rewards'])}\n")
        f.write(f"  Average evaluation reward: {np.mean(eval_metrics['rewards']):.3f}\n")
        f.write(f"  Evaluation reward std: {np.std(eval_metrics['rewards']):.3f}\n")
        f.write(f"  Average episode length: {np.mean(eval_metrics['episode_lengths']):.1f}\n")
        f.write(f"  Best evaluation reward: {np.max(eval_metrics['rewards']):.3f}\n")
        f.write(f"  Worst evaluation reward: {np.min(eval_metrics['rewards']):.3f}\n")
    
    print(f"\nTraining report saved to {report_path}")

def main():
    """Main function."""
    setup_logging()
    
    print("MARL-GCP Retraining with Real Google Cluster Data")
    print("="*60)
    
    # Check if real data is available
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print("❌ Error: Real data directory not found!")
        print("Please ensure the Google Cluster data is in data/processed/")
        return False
    
    # Backup existing models
    backup_existing_models()
    
    # Delete existing models
    delete_existing_models()
    
    try:
        # Train with real data
        train_metrics, eval_metrics = train_with_real_data(episodes=1000, eval_episodes=100)
        
        # Test different patterns
        test_different_workload_patterns()
        
        # Generate report
        generate_training_report(train_metrics, eval_metrics)
        
        print("\n🎉 Retraining completed successfully!")
        print("Agents are now trained on real Google Cluster data.")
        print("Phase 2 data integration is complete!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during retraining: {e}")
        print("Please check the logs for more details.")
        return False

if __name__ == "__main__":
    main() 