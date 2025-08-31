"""
System Monitoring for MARL-GCP
-----------------------------

This module provides monitoring capabilities for the MARL system, including
training metrics tracking, performance visualization, and system health monitoring.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

class SystemMonitoring:
    """
    System monitoring class for tracking MARL training metrics and system performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the system monitoring.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        self.agent_metrics = defaultdict(lambda: defaultdict(list))
        
        # Performance tracking
        self.start_time = time.time()
        self.episode_start_time = None
        
        # Monitoring configuration
        self.log_frequency = config.get('log_frequency', 1)
        self.save_frequency = config.get('save_frequency', 50)
        self.viz_frequency = config.get('viz_frequency', 10)
        
        logger.info("Initialized System Monitoring")
    
    def update(self, episode: int, metrics: Dict[str, Any]) -> None:
        """
        Update monitoring with new episode metrics.
        
        Args:
            episode: Current episode number
            metrics: Dictionary containing episode metrics
        """
        # Store episode metrics
        self.episode_metrics['episode'].append(episode)
        self.episode_metrics['rewards'].append(metrics.get('rewards', []))
        self.episode_metrics['episode_lengths'].append(metrics.get('episode_lengths', []))
        self.episode_metrics['resource_usage'].append(metrics.get('resource_usage', []))
        self.episode_metrics['costs'].append(metrics.get('costs', []))
        
        # Calculate summary statistics
        if metrics.get('rewards'):
            total_reward = np.sum(metrics['rewards'])
            avg_reward = np.mean(metrics['rewards'])
            self.metrics['total_rewards'].append(total_reward)
            self.metrics['avg_rewards'].append(avg_reward)
        
        if metrics.get('episode_lengths'):
            avg_length = np.mean(metrics['episode_lengths'])
            self.metrics['avg_episode_lengths'].append(avg_length)
        
        if metrics.get('costs'):
            total_cost = np.sum([sum(cost.values()) if isinstance(cost, dict) else cost for cost in metrics['costs']])
            self.metrics['total_costs'].append(total_cost)
        
        # Log metrics if frequency matches
        if episode % self.log_frequency == 0:
            self._log_metrics(episode, metrics)
    
    def update_agent_metrics(self, agent_name: str, metrics: Dict[str, float]) -> None:
        """
        Update metrics for a specific agent.
        
        Args:
            agent_name: Name of the agent
            metrics: Dictionary containing agent-specific metrics
        """
        for metric_name, value in metrics.items():
            self.agent_metrics[agent_name][metric_name].append(value)
    
    def _log_metrics(self, episode: int, metrics: Dict[str, Any]) -> None:
        """
        Log current metrics.
        
        Args:
            episode: Current episode number
            metrics: Current episode metrics
        """
        # Calculate summary statistics
        total_reward = np.sum(metrics.get('rewards', []))
        avg_reward = np.mean(metrics.get('rewards', [])) if metrics.get('rewards') else 0.0
        avg_length = np.mean(metrics.get('episode_lengths', [])) if metrics.get('episode_lengths') else 0.0
        
        # Calculate resource utilization
        resource_usage = metrics.get('resource_usage', [])
        if resource_usage:
            avg_cpu_util = np.mean([usage.get('cpu_utilization', 0.0) for usage in resource_usage])
            avg_memory_util = np.mean([usage.get('memory_utilization', 0.0) for usage in resource_usage])
        else:
            avg_cpu_util = 0.0
            avg_memory_util = 0.0
        
        # Calculate costs
        costs = metrics.get('costs', [])
        if costs:
            total_cost = np.sum([sum(cost.values()) if isinstance(cost, dict) else cost for cost in costs])
        else:
            total_cost = 0.0
        
        # Log summary
        logger.info(f"Episode {episode} - "
                   f"Total Reward: {total_reward:.2f}, "
                   f"Avg Reward: {avg_reward:.2f}, "
                   f"Avg Length: {avg_length:.1f}, "
                   f"CPU Util: {avg_cpu_util:.2f}, "
                   f"Memory Util: {avg_memory_util:.2f}, "
                   f"Total Cost: {total_cost:.2f}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary containing summary statistics
        """
        summary = {}
        
        # Episode metrics
        if self.metrics['total_rewards']:
            summary['total_rewards'] = {
                'mean': np.mean(self.metrics['total_rewards']),
                'std': np.std(self.metrics['total_rewards']),
                'min': np.min(self.metrics['total_rewards']),
                'max': np.max(self.metrics['total_rewards'])
            }
        
        if self.metrics['avg_rewards']:
            summary['avg_rewards'] = {
                'mean': np.mean(self.metrics['avg_rewards']),
                'std': np.std(self.metrics['avg_rewards']),
                'min': np.min(self.metrics['avg_rewards']),
                'max': np.max(self.metrics['avg_rewards'])
            }
        
        if self.metrics['avg_episode_lengths']:
            summary['avg_episode_lengths'] = {
                'mean': np.mean(self.metrics['avg_episode_lengths']),
                'std': np.std(self.metrics['avg_episode_lengths'])
            }
        
        if self.metrics['total_costs']:
            summary['total_costs'] = {
                'mean': np.mean(self.metrics['total_costs']),
                'std': np.std(self.metrics['total_costs']),
                'total': np.sum(self.metrics['total_costs'])
            }
        
        # Agent-specific metrics
        for agent_name, agent_metrics in self.agent_metrics.items():
            summary[f'agent_{agent_name}'] = {}
            for metric_name, values in agent_metrics.items():
                if values:
                    summary[f'agent_{agent_name}'][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        # Training time
        summary['training_time'] = time.time() - self.start_time
        
        return summary
    
    def save_metrics(self, filepath: str) -> None:
        """
        Save metrics to file.
        
        Args:
            filepath: Path to save metrics
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_dict = {}
        for key, values in self.metrics.items():
            if isinstance(values, list):
                metrics_dict[key] = [float(v) if isinstance(v, np.number) else v for v in values]
        
        # Add episode metrics
        for key, values in self.episode_metrics.items():
            if isinstance(values, list):
                metrics_dict[f'episode_{key}'] = [float(v) if isinstance(v, np.number) else v for v in values]
        
        # Add agent metrics
        for agent_name, agent_metrics in self.agent_metrics.items():
            for metric_name, values in agent_metrics.items():
                key = f'agent_{agent_name}_{metric_name}'
                metrics_dict[key] = [float(v) if isinstance(v, np.number) else v for v in values]
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Saved metrics to {filepath}")
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.episode_metrics.clear()
        self.agent_metrics.clear()
        self.start_time = time.time()
        logger.info("Reset monitoring metrics")
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """
        Get the latest metrics.
        
        Returns:
            Dictionary containing the latest metrics
        """
        latest = {}
        
        for key, values in self.metrics.items():
            if values:
                latest[key] = values[-1]
        
        return latest 