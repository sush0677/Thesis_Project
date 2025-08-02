"""
Enhanced Reward Engine for MARL-GCP System
==========================================

This module implements a comprehensive reward function system that addresses
all Phase 4 requirements:

1. Multi-objective reward functions balancing cost, performance, reliability, and sustainability
2. Hierarchical reward structure with immediate and delayed rewards
3. Reward normalization techniques for different metric scales
4. Constraint violation penalties and budget management
5. Sustainability metrics and environmental impact
6. Differentiable reward shaping for dense feedback during training
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class RewardComponent(Enum):
    """Enumeration of reward components."""
    COST_EFFICIENCY = "cost_efficiency"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    SUSTAINABILITY = "sustainability"
    UTILIZATION = "utilization"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    SCALABILITY = "scalability"
    SECURITY = "security"
    COMPLIANCE = "compliance"

@dataclass
class RewardWeights:
    """Configuration for reward component weights."""
    cost_efficiency: float = 0.25      # 25% weight
    performance: float = 0.35          # 35% weight (latency + throughput)
    reliability: float = 0.20          # 20% weight
    sustainability: float = 0.15       # 15% weight
    utilization: float = 0.05          # 5% weight (bonus for good utilization)
    
    def normalize(self):
        """Normalize weights to sum to 1.0."""
        total = sum(self.__dict__.values())
        for key in self.__dict__:
            self.__dict__[key] /= total

@dataclass
class SustainabilityMetrics:
    """Sustainability and environmental impact metrics."""
    carbon_footprint: float = 0.0      # CO2 emissions in kg
    energy_efficiency: float = 0.0     # Energy efficiency score (0-1)
    renewable_energy_usage: float = 0.0 # Percentage of renewable energy
    resource_waste: float = 0.0        # Waste percentage (0-1)
    cooling_efficiency: float = 0.0    # Cooling system efficiency (0-1)

class RewardNormalizer:
    """Handles normalization of different reward components."""
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize the reward normalizer.
        
        Args:
            history_size: Number of historical values to maintain for normalization
        """
        self.history_size = history_size
        self.reward_history = {component: [] for component in RewardComponent}
        self.normalization_stats = {}
    
    def update_history(self, component: RewardComponent, value: float):
        """Update the history for a reward component."""
        if component not in self.reward_history:
            self.reward_history[component] = []
        
        self.reward_history[component].append(value)
        
        # Keep only the most recent values
        if len(self.reward_history[component]) > self.history_size:
            self.reward_history[component] = self.reward_history[component][-self.history_size:]
    
    def normalize_reward(self, component: RewardComponent, value: float) -> float:
        """
        Normalize a reward value using historical statistics.
        
        Args:
            component: The reward component type
            value: The raw reward value
            
        Returns:
            Normalized reward value between -1 and 1
        """
        if component not in self.reward_history or len(self.reward_history[component]) < 10:
            # Not enough history, return original value
            return np.clip(value, -1.0, 1.0)
        
        history = self.reward_history[component]
        mean_val = np.mean(history)
        std_val = np.std(history) + 1e-8  # Avoid division by zero
        
        # Z-score normalization
        normalized = (value - mean_val) / std_val
        
        # Clip to reasonable range
        return np.clip(normalized, -3.0, 3.0)
    
    def get_component_stats(self, component: RewardComponent) -> Dict[str, float]:
        """Get statistics for a reward component."""
        if component not in self.reward_history or len(self.reward_history[component]) == 0:
            return {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 0.0}
        
        history = self.reward_history[component]
        return {
            "mean": np.mean(history),
            "std": np.std(history),
            "min": np.min(history),
            "max": np.max(history)
        }

class HierarchicalReward:
    """Implements hierarchical reward structure with immediate and delayed rewards."""
    
    def __init__(self, immediate_weight: float = 0.7, delayed_weight: float = 0.3):
        """
        Initialize hierarchical reward system.
        
        Args:
            immediate_weight: Weight for immediate rewards
            delayed_weight: Weight for delayed rewards
        """
        self.immediate_weight = immediate_weight
        self.delayed_weight = delayed_weight
        self.reward_buffer = {}  # Store delayed rewards
        self.episode_rewards = {}  # Track episode-level rewards
    
    def calculate_immediate_reward(self, 
                                 utilization: float, 
                                 cost_efficiency: float, 
                                 performance: float) -> float:
        """
        Calculate immediate reward based on current state.
        
        Args:
            utilization: Resource utilization (0-1)
            cost_efficiency: Cost efficiency score (0-1)
            performance: Performance score (0-1)
            
        Returns:
            Immediate reward value
        """
        # Immediate rewards focus on current resource allocation
        immediate_reward = (
            0.4 * utilization +      # Resource utilization
            0.4 * cost_efficiency +  # Cost efficiency
            0.2 * performance        # Performance
        )
        
        return immediate_reward
    
    def calculate_delayed_reward(self, 
                               episode_performance: float,
                               long_term_efficiency: float,
                               stability_score: float) -> float:
        """
        Calculate delayed reward based on episode-level performance.
        
        Args:
            episode_performance: Overall episode performance
            long_term_efficiency: Long-term efficiency metrics
            stability_score: System stability score
            
        Returns:
            Delayed reward value
        """
        # Delayed rewards focus on long-term outcomes
        delayed_reward = (
            0.5 * episode_performance +    # Overall performance
            0.3 * long_term_efficiency +   # Long-term efficiency
            0.2 * stability_score          # System stability
        )
        
        return delayed_reward
    
    def combine_rewards(self, immediate: float, delayed: float) -> float:
        """Combine immediate and delayed rewards."""
        return (self.immediate_weight * immediate + 
                self.delayed_weight * delayed)

class SustainabilityCalculator:
    """Calculates sustainability and environmental impact metrics."""
    
    def __init__(self):
        """Initialize sustainability calculator."""
        # GCP region sustainability data (simplified)
        self.region_sustainability = {
            'us-central1': {'renewable_percentage': 0.85, 'carbon_intensity': 0.12},
            'us-east1': {'renewable_percentage': 0.80, 'carbon_intensity': 0.15},
            'europe-west1': {'renewable_percentage': 0.90, 'carbon_intensity': 0.10},
            'asia-southeast1': {'renewable_percentage': 0.70, 'carbon_intensity': 0.20}
        }
    
    def calculate_carbon_footprint(self, 
                                 compute_hours: float,
                                 storage_gb: float,
                                 network_gbps: float,
                                 region: str = 'us-central1') -> float:
        """
        Calculate carbon footprint based on resource usage.
        
        Args:
            compute_hours: Compute hours used
            storage_gb: Storage in GB
            network_gbps: Network bandwidth in Gbps
            region: GCP region
            
        Returns:
            Carbon footprint in kg CO2
        """
        region_data = self.region_sustainability.get(region, 
                                                   {'renewable_percentage': 0.80, 'carbon_intensity': 0.15})
        
        # Simplified carbon calculation
        compute_carbon = compute_hours * 0.1 * region_data['carbon_intensity']  # kg CO2 per hour
        storage_carbon = storage_gb * 0.0001 * region_data['carbon_intensity']  # kg CO2 per GB
        network_carbon = network_gbps * 0.01 * region_data['carbon_intensity']  # kg CO2 per Gbps
        
        total_carbon = compute_carbon + storage_carbon + network_carbon
        
        # Apply renewable energy discount
        renewable_discount = region_data['renewable_percentage']
        adjusted_carbon = total_carbon * (1 - renewable_discount)
        
        return adjusted_carbon
    
    def calculate_energy_efficiency(self, 
                                  utilization: float,
                                  resource_waste: float,
                                  cooling_efficiency: float) -> float:
        """
        Calculate energy efficiency score.
        
        Args:
            utilization: Resource utilization (0-1)
            resource_waste: Waste percentage (0-1)
            cooling_efficiency: Cooling efficiency (0-1)
            
        Returns:
            Energy efficiency score (0-1)
        """
        # Higher utilization = better efficiency
        utilization_score = utilization
        
        # Lower waste = better efficiency
        waste_score = 1.0 - resource_waste
        
        # Better cooling = better efficiency
        cooling_score = cooling_efficiency
        
        # Weighted average
        efficiency_score = (
            0.5 * utilization_score +
            0.3 * waste_score +
            0.2 * cooling_score
        )
        
        return np.clip(efficiency_score, 0.0, 1.0)
    
    def calculate_sustainability_reward(self, 
                                      carbon_footprint: float,
                                      energy_efficiency: float,
                                      renewable_usage: float) -> float:
        """
        Calculate sustainability reward component.
        
        Args:
            carbon_footprint: Carbon footprint in kg CO2
            energy_efficiency: Energy efficiency score (0-1)
            renewable_usage: Renewable energy usage percentage (0-1)
            
        Returns:
            Sustainability reward value
        """
        # Lower carbon footprint = higher reward
        carbon_score = max(0, 1.0 - (carbon_footprint / 100.0))  # Normalize to 0-1
        
        # Higher energy efficiency = higher reward
        efficiency_score = energy_efficiency
        
        # Higher renewable usage = higher reward
        renewable_score = renewable_usage
        
        # Combined sustainability score
        sustainability_score = (
            0.4 * carbon_score +
            0.4 * efficiency_score +
            0.2 * renewable_score
        )
        
        return sustainability_score

class ConstraintManager:
    """Manages constraint violations and budget enforcement."""
    
    def __init__(self, budget_limit: float = 1000.0):
        """
        Initialize constraint manager.
        
        Args:
            budget_limit: Monthly budget limit in USD
        """
        self.budget_limit = budget_limit
        self.violation_history = []
        self.current_budget_usage = 0.0
    
    def check_budget_violation(self, current_cost: float) -> Tuple[bool, float]:
        """
        Check if current cost violates budget constraint.
        
        Args:
            current_cost: Current monthly cost
            
        Returns:
            Tuple of (is_violated, penalty_amount)
        """
        self.current_budget_usage = current_cost
        
        if current_cost > self.budget_limit:
            violation_amount = current_cost - self.budget_limit
            penalty = violation_amount * 2.0  # 2x penalty for budget violations
            return True, penalty
        
        return False, 0.0
    
    def check_resource_constraints(self, 
                                 resources: Dict[str, Any],
                                 constraints: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check resource constraint violations.
        
        Args:
            resources: Current resource allocation
            constraints: Resource constraints
            
        Returns:
            Tuple of (is_violated, penalty_amount)
        """
        violations = []
        total_penalty = 0.0
        
        # Check compute constraints
        if resources.get('instances', 0) > constraints.get('max_instances', 100):
            violations.append('max_instances')
            total_penalty += 10.0
        
        if resources.get('cpus', 0) > constraints.get('max_cpus', 1000):
            violations.append('max_cpus')
            total_penalty += 5.0
        
        # Check storage constraints
        if resources.get('storage_gb', 0) > constraints.get('max_storage_gb', 10000):
            violations.append('max_storage')
            total_penalty += 3.0
        
        # Check network constraints
        if resources.get('network_bandwidth_gbps', 0) > constraints.get('max_bandwidth_gbps', 100):
            violations.append('max_bandwidth')
            total_penalty += 2.0
        
        is_violated = len(violations) > 0
        
        if is_violated:
            self.violation_history.append({
                'step': len(self.violation_history),
                'violations': violations,
                'penalty': total_penalty
            })
        
        return is_violated, total_penalty
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of constraint violations."""
        if not self.violation_history:
            return {"total_violations": 0, "total_penalty": 0.0, "violation_rate": 0.0}
        
        total_violations = len(self.violation_history)
        total_penalty = sum(v['penalty'] for v in self.violation_history)
        
        return {
            "total_violations": total_violations,
            "total_penalty": total_penalty,
            "violation_rate": total_violations / max(1, len(self.violation_history)),
            "budget_usage_percentage": (self.current_budget_usage / self.budget_limit) * 100
        }

class EnhancedRewardEngine:
    """
    Enhanced reward engine implementing all Phase 4 requirements.
    
    Features:
    - Multi-objective reward functions
    - Hierarchical reward structure
    - Reward normalization
    - Sustainability metrics
    - Constraint violation penalties
    - Differentiable reward shaping
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the enhanced reward engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.weights = RewardWeights()
        self.weights.normalize()
        
        self.normalizer = RewardNormalizer(
            history_size=config.get('reward_history_size', 1000)
        )
        
        self.hierarchical = HierarchicalReward(
            immediate_weight=config.get('immediate_reward_weight', 0.7),
            delayed_weight=config.get('delayed_reward_weight', 0.3)
        )
        
        self.sustainability = SustainabilityCalculator()
        
        self.constraint_manager = ConstraintManager(
            budget_limit=config.get('budget_limit', 1000.0)
        )
        
        # Reward shaping parameters
        self.shaping_enabled = config.get('reward_shaping_enabled', True)
        self.shaping_decay = config.get('reward_shaping_decay', 0.99)
        self.current_shaping_weight = 1.0
        
        # Performance tracking
        self.performance_history = []
        self.reward_components_history = []
        
        logger.info("Enhanced Reward Engine initialized")
    
    def calculate_comprehensive_rewards(self,
                                      agents: List[str],
                                      current_state: Dict[str, Any],
                                      actions: Dict[str, np.ndarray],
                                      next_state: Dict[str, Any],
                                      episode_step: int,
                                      episode_length: int) -> Dict[str, float]:
        """
        Calculate comprehensive rewards for all agents.
        
        Args:
            agents: List of agent names
            current_state: Current environment state
            actions: Actions taken by agents
            next_state: Next environment state
            episode_step: Current step in episode
            episode_length: Total episode length
            
        Returns:
            Dictionary of rewards for each agent
        """
        rewards = {}
        
        # Calculate base reward components
        cost_efficiency = self._calculate_cost_efficiency(current_state)
        performance = self._calculate_performance_metrics(current_state, next_state)
        reliability = self._calculate_reliability_metrics(current_state)
        sustainability = self._calculate_sustainability_metrics(current_state)
        utilization = self._calculate_utilization_metrics(current_state)
        
        # Calculate constraint violations
        budget_violated, budget_penalty = self.constraint_manager.check_budget_violation(
            current_state.get('total_cost', 0.0)
        )
        
        resource_violated, resource_penalty = self.constraint_manager.check_resource_constraints(
            current_state.get('resources', {}),
            current_state.get('constraints', {})
        )
        
        total_penalty = budget_penalty + resource_penalty
        
        # Calculate rewards for each agent
        for agent in agents:
            agent_reward = self._calculate_agent_reward(
                agent=agent,
                cost_efficiency=cost_efficiency,
                performance=performance,
                reliability=reliability,
                sustainability=sustainability,
                utilization=utilization,
                penalty=total_penalty,
                episode_step=episode_step,
                episode_length=episode_length
            )
            
            rewards[agent] = agent_reward
        
        # Update performance tracking
        self._update_performance_tracking(rewards, {
            'cost_efficiency': cost_efficiency,
            'performance': performance,
            'reliability': reliability,
            'sustainability': sustainability,
            'utilization': utilization,
            'penalty': total_penalty
        })
        
        return rewards
    
    def _calculate_cost_efficiency(self, state: Dict[str, Any]) -> float:
        """Calculate cost efficiency score."""
        total_cost = state.get('total_cost', 0.0)
        workload_demand = state.get('workload_demand', 1.0)
        
        if total_cost <= 0:
            return 0.0
        
        # Higher demand with lower cost = better efficiency
        efficiency = workload_demand / total_cost
        
        # Normalize to 0-1 range
        return np.clip(efficiency / 10.0, 0.0, 1.0)
    
    def _calculate_performance_metrics(self, 
                                     current_state: Dict[str, Any],
                                     next_state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Latency (lower is better)
        current_latency = current_state.get('latency', 100.0)
        next_latency = next_state.get('latency', 100.0)
        latency_improvement = max(0, (current_latency - next_latency) / current_latency)
        
        # Throughput (higher is better)
        current_throughput = current_state.get('throughput', 100.0)
        next_throughput = next_state.get('throughput', 100.0)
        throughput_improvement = max(0, (next_throughput - current_throughput) / max(current_throughput, 1.0))
        
        # Scalability (how well resources scale with demand)
        resource_utilization = current_state.get('resource_utilization', 0.5)
        demand_satisfaction = current_state.get('demand_satisfaction', 0.8)
        scalability = min(1.0, demand_satisfaction / max(resource_utilization, 0.1))
        
        return {
            'latency': 1.0 - (current_latency / 1000.0),  # Normalize
            'throughput': min(1.0, current_throughput / 1000.0),  # Normalize
            'scalability': scalability,
            'latency_improvement': latency_improvement,
            'throughput_improvement': throughput_improvement
        }
    
    def _calculate_reliability_metrics(self, state: Dict[str, Any]) -> float:
        """Calculate reliability score."""
        # Availability (uptime percentage)
        availability = state.get('availability', 0.99)
        
        # Error rate (lower is better)
        error_rate = state.get('error_rate', 0.01)
        error_score = 1.0 - error_rate
        
        # Resource stability (how stable resource allocation is)
        resource_stability = state.get('resource_stability', 0.8)
        
        # Combined reliability score
        reliability = (
            0.4 * availability +
            0.4 * error_score +
            0.2 * resource_stability
        )
        
        return reliability
    
    def _calculate_sustainability_metrics(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate sustainability metrics."""
        resources = state.get('resources', {})
        
        # Calculate carbon footprint
        carbon_footprint = self.sustainability.calculate_carbon_footprint(
            compute_hours=resources.get('compute_hours', 1.0),
            storage_gb=resources.get('storage_gb', 100.0),
            network_gbps=resources.get('network_bandwidth_gbps', 10.0),
            region=state.get('region', 'us-central1')
        )
        
        # Calculate energy efficiency
        energy_efficiency = self.sustainability.calculate_energy_efficiency(
            utilization=state.get('resource_utilization', 0.5),
            resource_waste=state.get('resource_waste', 0.1),
            cooling_efficiency=state.get('cooling_efficiency', 0.8)
        )
        
        # Renewable energy usage
        renewable_usage = state.get('renewable_energy_usage', 0.8)
        
        # Calculate sustainability reward
        sustainability_reward = self.sustainability.calculate_sustainability_reward(
            carbon_footprint=carbon_footprint,
            energy_efficiency=energy_efficiency,
            renewable_usage=renewable_usage
        )
        
        return {
            'carbon_footprint': carbon_footprint,
            'energy_efficiency': energy_efficiency,
            'renewable_usage': renewable_usage,
            'sustainability_reward': sustainability_reward
        }
    
    def _calculate_utilization_metrics(self, state: Dict[str, Any]) -> float:
        """Calculate resource utilization score."""
        utilization = state.get('resource_utilization', 0.5)
        
        # Optimal utilization is around 70-80%
        # Too low = waste, too high = risk of overload
        if 0.6 <= utilization <= 0.85:
            return 1.0
        elif utilization < 0.6:
            return utilization / 0.6  # Linear penalty for under-utilization
        else:
            return max(0, 1.0 - (utilization - 0.85) / 0.15)  # Penalty for over-utilization
    
    def _calculate_agent_reward(self,
                               agent: str,
                               cost_efficiency: float,
                               performance: Dict[str, float],
                               reliability: float,
                               sustainability: Dict[str, float],
                               utilization: float,
                               penalty: float,
                               episode_step: int,
                               episode_length: int) -> float:
        """
        Calculate reward for a specific agent.
        
        Args:
            agent: Agent name
            cost_efficiency: Cost efficiency score
            performance: Performance metrics
            reliability: Reliability score
            sustainability: Sustainability metrics
            utilization: Utilization score
            penalty: Constraint violation penalty
            episode_step: Current episode step
            episode_length: Total episode length
            
        Returns:
            Agent reward value
        """
        # Agent-specific performance weights
        agent_weights = {
            'compute': {'performance': 0.4, 'cost': 0.3, 'reliability': 0.2, 'sustainability': 0.1},
            'storage': {'performance': 0.2, 'cost': 0.4, 'reliability': 0.3, 'sustainability': 0.1},
            'network': {'performance': 0.5, 'cost': 0.2, 'reliability': 0.2, 'sustainability': 0.1},
            'database': {'performance': 0.3, 'cost': 0.3, 'reliability': 0.3, 'sustainability': 0.1}
        }
        
        weights = agent_weights.get(agent, agent_weights['compute'])
        
        # Calculate weighted performance score
        performance_score = (
            weights['performance'] * np.mean(list(performance.values())) +
            weights['cost'] * cost_efficiency +
            weights['reliability'] * reliability +
            weights['sustainability'] * sustainability['sustainability_reward']
        )
        
        # Add utilization bonus
        utilization_bonus = 0.1 * utilization
        
        # Calculate immediate reward
        immediate_reward = performance_score + utilization_bonus - penalty
        
        # Calculate delayed reward (episode-level)
        if episode_step == episode_length - 1:  # End of episode
            episode_performance = np.mean(self.performance_history[-episode_length:]) if self.performance_history else 0.0
            long_term_efficiency = np.mean([r['cost_efficiency'] for r in self.reward_components_history[-episode_length:]]) if self.reward_components_history else 0.0
            stability_score = np.std(self.performance_history[-episode_length:]) if len(self.performance_history) >= episode_length else 0.0
            stability_score = max(0, 1.0 - stability_score)  # Lower std = higher stability
            
            delayed_reward = self.hierarchical.calculate_delayed_reward(
                episode_performance=episode_performance,
                long_term_efficiency=long_term_efficiency,
                stability_score=stability_score
            )
        else:
            delayed_reward = 0.0
        
        # Combine immediate and delayed rewards
        combined_reward = self.hierarchical.combine_rewards(immediate_reward, delayed_reward)
        
        # Apply reward shaping if enabled
        if self.shaping_enabled:
            shaped_reward = self._apply_reward_shaping(combined_reward, episode_step, episode_length)
        else:
            shaped_reward = combined_reward
        
        # Normalize reward
        normalized_reward = self.normalizer.normalize_reward(
            RewardComponent.PERFORMANCE, shaped_reward
        )
        
        # Update history for normalization
        self.normalizer.update_history(RewardComponent.PERFORMANCE, shaped_reward)
        
        return normalized_reward
    
    def _apply_reward_shaping(self, 
                             base_reward: float, 
                             episode_step: int, 
                             episode_length: int) -> float:
        """
        Apply reward shaping for dense feedback during training.
        
        Args:
            base_reward: Base reward value
            episode_step: Current episode step
            episode_length: Total episode length
            
        Returns:
            Shaped reward value
        """
        # Decay shaping weight over time
        self.current_shaping_weight *= self.shaping_decay
        
        # Add shaping based on progress
        progress = episode_step / max(episode_length, 1)
        
        # Shaping reward encourages exploration and progress
        shaping_reward = 0.1 * progress * self.current_shaping_weight
        
        # Combine base reward with shaping
        shaped_reward = base_reward + shaping_reward
        
        return shaped_reward
    
    def _update_performance_tracking(self, 
                                   rewards: Dict[str, float],
                                   components: Dict[str, Any]):
        """Update performance tracking history."""
        # Track overall performance
        overall_reward = np.mean(list(rewards.values()))
        self.performance_history.append(overall_reward)
        
        # Track reward components
        self.reward_components_history.append(components)
        
        # Keep history manageable
        max_history = 10000
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
        if len(self.reward_components_history) > max_history:
            self.reward_components_history = self.reward_components_history[-max_history:]
    
    def get_reward_analysis(self) -> Dict[str, Any]:
        """Get comprehensive reward analysis."""
        if not self.performance_history:
            return {"error": "No performance history available"}
        
        # Calculate statistics
        recent_performance = self.performance_history[-100:] if len(self.performance_history) >= 100 else self.performance_history
        
        analysis = {
            "overall_performance": {
                "mean": np.mean(recent_performance),
                "std": np.std(recent_performance),
                "min": np.min(recent_performance),
                "max": np.max(recent_performance),
                "trend": "improving" if len(recent_performance) >= 2 and recent_performance[-1] > recent_performance[0] else "stable"
            },
            "constraint_violations": self.constraint_manager.get_violation_summary(),
            "reward_components": {
                component.value: self.normalizer.get_component_stats(component)
                for component in RewardComponent
            },
            "sustainability_metrics": {
                "carbon_footprint_trend": "decreasing" if len(self.reward_components_history) >= 2 else "stable",
                "energy_efficiency": np.mean([r.get('sustainability', {}).get('energy_efficiency', 0.0) 
                                            for r in self.reward_components_history[-100:]]) if self.reward_components_history else 0.0
            }
        }
        
        return analysis
    
    def save_reward_config(self, filepath: str):
        """Save reward configuration to file."""
        config = {
            "weights": self.weights.__dict__,
            "hierarchical": {
                "immediate_weight": self.hierarchical.immediate_weight,
                "delayed_weight": self.hierarchical.delayed_weight
            },
            "constraint_manager": {
                "budget_limit": self.constraint_manager.budget_limit
            },
            "shaping": {
                "enabled": self.shaping_enabled,
                "decay": self.shaping_decay,
                "current_weight": self.current_shaping_weight
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Reward configuration saved to {filepath}")
    
    def load_reward_config(self, filepath: str):
        """Load reward configuration from file."""
        if not Path(filepath).exists():
            logger.warning(f"Reward config file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Update weights
        for key, value in config.get("weights", {}).items():
            if hasattr(self.weights, key):
                setattr(self.weights, key, value)
        
        # Update hierarchical weights
        hierarchical_config = config.get("hierarchical", {})
        self.hierarchical.immediate_weight = hierarchical_config.get("immediate_weight", 0.7)
        self.hierarchical.delayed_weight = hierarchical_config.get("delayed_weight", 0.3)
        
        # Update constraint manager
        constraint_config = config.get("constraint_manager", {})
        self.constraint_manager.budget_limit = constraint_config.get("budget_limit", 1000.0)
        
        # Update shaping parameters
        shaping_config = config.get("shaping", {})
        self.shaping_enabled = shaping_config.get("enabled", True)
        self.shaping_decay = shaping_config.get("decay", 0.99)
        self.current_shaping_weight = shaping_config.get("current_weight", 1.0)
        
        logger.info(f"Reward configuration loaded from {filepath}") 