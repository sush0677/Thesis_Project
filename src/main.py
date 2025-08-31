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

from src.marl_gcp.system_architecture import MARLSystemArchitecture
from src.marl_gcp.configs.default_config import get_default_config

def setup_logging(log_level):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create custom formatter for cleaner output
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure file handler
    file_handler = logging.FileHandler('marl_gcp.log')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Reduce verbosity for specific modules to clean up output
    if log_level.upper() == 'INFO':
        logging.getLogger('src.marl_gcp.data.workload_generator').setLevel(logging.WARNING)
        logging.getLogger('src.marl_gcp.environment.gcp_environment').setLevel(logging.WARNING)
        logging.getLogger('src.marl_gcp.agents').setLevel(logging.WARNING)
        logging.getLogger('src.marl_gcp.utils').setLevel(logging.WARNING)

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
    logger.info("🚀 Initializing MARL-GCP System with Real Google Cluster Data")
    logger.info("=" * 80)
    system = MARLSystemArchitecture(config)
    
    # Verify real data integration
    logger.info("🔍 Verifying Real Data Integration:")
    try:
        from src.marl_gcp.data.workload_generator import WorkloadGenerator
        wg = WorkloadGenerator(config)
        if hasattr(wg, 'processed_data') and wg.processed_data is not None:
            logger.info(f"✅ Real Google Cluster Data: {len(wg.processed_data)} records loaded")
        else:
            logger.warning("⚠️  Real data verification: processed_data not found")
        
        feature_stats = wg._load_feature_statistics()
        if feature_stats and 'cpu_usage' in feature_stats:
            cpu_mean = feature_stats['cpu_usage']['mean']
            logger.info(f"✅ Real Feature Statistics: CPU mean = {cpu_mean:.3f}")
        else:
            logger.warning("⚠️  Feature statistics not loaded properly")
    except Exception as e:
        logger.warning(f"⚠️  Data verification error: {e}")
    
    logger.info("=" * 80)
    
    # Load model if specified
    if args.model_path:
        logger.info(f"📂 Loading model from {args.model_path}")
        for agent_name, agent in system.agents.items():
            agent.load(args.model_path)
    
    # Run training or evaluation
    if args.eval:
        logger.info("🧪 Running evaluation")
        eval_metrics = system.evaluate(config['eval_episodes'])
        logger.info(f"📊 Evaluation results: Average reward: {np.mean(eval_metrics['rewards']):.2f}")
    else:
        logger.info(f"🏋️ Starting training for {config['num_episodes']} episodes")
        logger.info("=" * 80)
        
        # Enhanced training with detailed logging
        train_metrics = enhanced_training_with_logging(system, config, logger)
        
        # Save trained models
        save_path = os.path.join(config['save_dir'], config['experiment_name'])
        os.makedirs(save_path, exist_ok=True)
        
        logger.info("💾 Saving trained models...")
        for agent_name, agent in system.agents.items():
            agent.save(save_path)
        
        logger.info(f"✅ Training completed. Models saved to {save_path}")
        
        # Run final evaluation
        logger.info("🔬 Running final evaluation")
        eval_metrics = system.evaluate(config['eval_episodes'])
        logger.info(f"🎯 Final evaluation results: Average reward: {np.mean(eval_metrics['rewards']):.2f}")

def enhanced_training_with_logging(system, config, logger):
    """Enhanced training function with detailed logging."""
    logger.info("🎮 Starting Enhanced Training with Real-Time Monitoring")
    logger.info("-" * 80)
    
    episode_rewards = []
    
    for episode in range(config['num_episodes']):
        logger.info(f"🔄 Episode {episode + 1}/{config['num_episodes']}")
        
        # Reset environment
        state = system.environment.reset()
        logger.info("   📍 Environment reset - Starting new episode")
        
        episode_reward = 0
        step_count = 0
        
        # Log initial state
        log_resource_allocations(system.environment, logger, "Initial")
        
        while step_count < config.get('max_steps_per_episode', 100):
            step_count += 1
            
            # Get actions from agents
            actions = {}
            for agent_name, agent in system.agents.items():
                if hasattr(agent, 'select_action'):
                    agent_state = state[agent_name]
                    # Convert state dict to array if needed
                    if isinstance(agent_state, dict):
                        # Convert dict to flattened array
                        agent_state = np.array(list(agent_state.values()), dtype=np.float32)
                    
                    action_idx = agent.select_action(agent_state)
                    # Convert single action index to action array
                    action_array = convert_action_index_to_array(action_idx, agent_name)
                    actions[agent_name] = action_array
                    if step_count % 20 == 0:  # Log every 20 steps
                        logger.info(f"   🤖 {agent_name} agent action: {action_idx} -> {action_array}")
                else:
                    # Fallback for testing
                    actions[agent_name] = np.zeros(5)  # Default action
            
            # Take environment step
            try:
                next_state, rewards, done, info = system.environment.step(actions)
                
                # Log rewards in detail
                if step_count % 20 == 0:
                    reward_details = [f"{agent}={reward:.3f}" for agent, reward in rewards.items()]
                    logger.info(f"   💰 Step {step_count} Rewards: {', '.join(reward_details)}")
                    
                    # Log resource allocations
                    log_resource_allocations(system.environment, logger, f"Step {step_count}")
                
                # Update episode reward
                total_step_reward = sum(rewards.values())
                episode_reward += total_step_reward
                
                # Store experience in agents
                for agent_name, agent in system.agents.items():
                    if hasattr(agent, 'store_experience'):
                        agent.store_experience(
                            state[agent_name], 
                            actions[agent_name], 
                            rewards[agent_name], 
                            next_state[agent_name], 
                            done
                        )
                
                # Update agents (training step)
                if step_count % config.get('update_frequency', 4) == 0:
                    for agent_name, agent in system.agents.items():
                        if hasattr(agent, 'update'):
                            try:
                                loss = agent.update()
                                if loss and step_count % 20 == 0:
                                    logger.info(f"   🧠 {agent_name} learning loss: {loss:.4f}")
                            except Exception as e:
                                if step_count % 20 == 0:
                                    logger.warning(f"   ⚠️  {agent_name} update failed: {e}")
                                continue
                
                state = next_state
                
                if done:
                    logger.info(f"   ✅ Episode completed at step {step_count}")
                    break
                    
            except Exception as e:
                logger.error(f"   ❌ Step {step_count} failed: {e}")
                break
        
        episode_rewards.append(episode_reward)
        
        # Episode summary
        logger.info(f"   📊 Episode {episode + 1} Summary:")
        logger.info(f"      - Total Reward: {episode_reward:.3f}")
        logger.info(f"      - Steps: {step_count}")
        logger.info(f"      - Average Reward: {episode_reward/step_count:.3f}")
        
        # Log final allocations
        log_resource_allocations(system.environment, logger, "Final")
        
        logger.info("-" * 80)
    
    return {'rewards': episode_rewards}

def convert_action_index_to_array(action_idx, agent_name):
    """
    Convert single action index to action array expected by environment.
    
    Args:
        action_idx: Single integer action index from agent
        agent_name: Name of the agent
        
    Returns:
        np.ndarray: Action array for environment
    """
    # Map action index to action components based on agent type
    if agent_name == 'compute':
        # 5 components: scale_instances, scale_cpu, scale_memory, region_preference, instance_type
        # Map action_idx (0-80) to 5 components each in range [-1, 1]
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

def log_resource_allocations(environment, logger, phase):
    """Log current resource allocations."""
    resources = environment.current_resources
    logger.info(f"   🏗️  {phase} Resource Allocations:")
    logger.info(f"      - Compute: {resources.get('instances', 0)} instances, "
               f"{resources.get('cpus', 0)} CPUs, {resources.get('memory_gb', 0)} GB RAM")
    logger.info(f"      - Storage: {resources.get('storage_gb', 0)} GB")
    logger.info(f"      - Network: {resources.get('network_bandwidth_gbps', 0)} Gbps bandwidth")
    logger.info(f"      - Database: {resources.get('database_instances', 0)} instances, "
               f"{resources.get('database_storage_gb', 0)} GB storage")
    logger.info(f"      - Budget Used: ${resources.get('budget_used', 0):.2f}")

if __name__ == "__main__":
    main()