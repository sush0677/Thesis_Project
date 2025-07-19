"""
Main script for MARL-GCP System
------------------------------

This script demonstrates the MARL system architecture for Google Cloud provisioning.
"""

import os
import sys
import logging
import argparse
import numpy as np
import random
import torch

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marl_gcp.system_architecture import MARLSystemArchitecture
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
            logging.FileHandler('marl_gcp.log')
        ]
    )

def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MARL-GCP System")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes to train")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    parser.add_argument("--model_path", type=str, default=None, help="Path to load model from")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = get_default_config()
    
    # Override configuration with command line arguments
    if args.seed is not None:
        config['random_seed'] = args.seed
    if args.episodes is not None:
        config['num_episodes'] = args.episodes
    if args.log_level:
        config['log_level'] = args.log_level
    
    # Setup logging
    setup_logging(config['log_level'])
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_random_seed(config['random_seed'])
    
    # Create results directory if it doesn't exist
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Initialize system
    logger.info("Initializing MARL-GCP System")
    system = MARLSystemArchitecture(config)
    
    # Load model if specified
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        for agent_name, agent in system.agents.items():
            agent.load(args.model_path)
    
    # Run training or evaluation
    if args.eval:
        logger.info("Running evaluation")
        eval_metrics = system.evaluate(config['eval_episodes'])
        logger.info(f"Evaluation results: Average reward: {np.mean(eval_metrics['rewards']):.2f}")
    else:
        logger.info(f"Starting training for {config['num_episodes']} episodes")
        train_metrics = system.train(config['num_episodes'])
        
        # Save trained models
        save_path = os.path.join(config['save_dir'], config['experiment_name'])
        os.makedirs(save_path, exist_ok=True)
        
        for agent_name, agent in system.agents.items():
            agent.save(save_path)
        
        logger.info(f"Training completed. Models saved to {save_path}")
        
        # Run final evaluation
        logger.info("Running final evaluation")
        eval_metrics = system.evaluate(config['eval_episodes'])
        logger.info(f"Final evaluation results: Average reward: {np.mean(eval_metrics['rewards']):.2f}")

if __name__ == "__main__":
    main()