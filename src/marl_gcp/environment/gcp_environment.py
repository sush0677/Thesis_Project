"""
GCP Environment for MARL-GCP
---------------------------

This module implements a simulation environment for Google Cloud Platform resource provisioning.
It follows the OpenAI Gym interface and models:
1. Resource availability constraints
2. Pricing dynamics
3. Provisioning delays
4. Interdependencies between services
"""

import numpy as np
import random
import time
import logging
from typing import Dict, List, Tuple, Any, Union
from collections import defaultdict
from pathlib import Path

# Try to import gym, fall back to gymnasium if not available
try:
    import gym
    from gym import spaces
except ImportError:
    import gymnasium as gym
    from gymnasium import spaces

logger = logging.getLogger(__name__)

class GCPEnvironment(gym.Env):
    """
    Simulation environment for Google Cloud Platform resource provisioning.
    
    This environment follows the OpenAI Gym interface and provides a simulation
    of the GCP environment for training MARL agents.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GCP environment.
        
        Args:
            config: Configuration dictionary with environment parameters
        """
        super(GCPEnvironment, self).__init__()
        
        self.config = config
        self.simulation_mode = config.get('simulation_mode', True)
        self.gcp_project_id = config.get('gcp_project_id', 'your-gcp-project')
        self.use_terraform = config.get('use_terraform', False)
        self.reward_scale = config.get('reward_scale', 1.0)
        
        # Resource constraints
        self.resource_constraints = config.get('resource_constraints', {
            'max_instances': 100,
            'max_cpus': 500,
            'max_memory_gb': 1000,
            'max_storage_gb': 10000,
            'max_budget': 1000.0
        })
        
        # Delay simulation parameters
        self.delay_params = config.get('delay_simulation', {
            'provision_delay_mean': 5,  # steps
            'provision_delay_std': 2
        })
        
        # Current state of resources with all resource types
        self.current_resources = {
            'instances': 0,
            'cpus': 0,
            'memory_gb': 0,
            'storage_gb': 0,
            'budget_used': 0.0,
            'network_load': 0.0,
            'database_load': 0.0,
            # Network resources
            'network_bandwidth_gbps': 10,
            'vpc_complexity': 0.5,
            'subnet_count': 3,
            'firewall_rules': 10,
            'load_balancer': 'none',
            'cdn_enabled': False,
            # Database resources
            'database_instances': 1,
            'database_storage_gb': 100,
            'database_type': 'sql',
            'database_complexity': 1.0
        }
        
        # Pending operations (to simulate delays)
        self.pending_operations = []
        
        # Pricing model
        self.pricing = self._initialize_pricing()
        
        # Initialize workload generator with real Google Cluster data
        self.workload_generator = None
        try:
            from marl_gcp.data.workload_generator import WorkloadGenerator
            workload_config = config.get('workload_generator', {})
            # Ensure proper data paths
            workload_config['data_dir'] = config.get('cluster_data', {}).get('data_dir', 'data/processed')
            workload_config['cache_dir'] = config.get('cluster_data', {}).get('cache_dir', 'data/google_cluster')
            workload_config['viz_dir'] = config.get('viz_dir', 'visualizations')
            
            self.workload_generator = WorkloadGenerator(workload_config)
            logger.info("Initialized workload generator with real Google Cluster data")
            
            # Get initial workload from real data
            self.current_workload = self.workload_generator.generate_workload()
            logger.info("Loaded initial workload from real Google Cluster data")
            
        except Exception as e:
            logger.error(f"Failed to initialize workload generator: {e}")
            # Fallback to synthetic workload
            self.current_workload = {
                'compute_demand': np.array([0.3, 0.3, 0.3]),  # CPU, memory, network
                'storage_demand': 0.3,
                'network_demand': 0.3,
                'database_demand': np.array([0.3, 0.3])  # CPU, storage
            }
            logger.warning("Using synthetic workload as fallback")
        
        # Step counter
        self.steps = 0
        self.max_steps = config.get('max_steps_per_episode', 200)
        
        # Define action and observation spaces for each agent
        self._define_spaces()
        
        # Visualization directory
        self.viz_dir = Path(config.get('viz_dir', "visualizations"))
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("GCP Environment initialized")
    
    def _define_spaces(self):
        """Define the action and observation spaces for each agent based on real Google Cluster data."""
        # Observation space (Dict type with different spaces for each agent)
        # Updated to match real Google Cluster data features
        self.observation_space = spaces.Dict({
            'compute': spaces.Dict({
                'cpu_rate': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Real Google Cluster feature
                'canonical_memory_usage': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Real feature
                'local_disk_space_usage': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Real feature
                'disk_io_time': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Real feature
                'active_machines': spaces.Box(low=30, high=70, shape=(1,), dtype=np.float32),  # Real feature
                'active_tasks': spaces.Box(low=60, high=140, shape=(1,), dtype=np.float32),  # Real feature
                'cpu_memory_ratio': spaces.Box(low=0, high=5, shape=(1,), dtype=np.float32),  # Real feature
                'io_ratio': spaces.Box(low=0, high=3, shape=(1,), dtype=np.float32),  # Real feature
                'cost': spaces.Box(low=0, high=self.resource_constraints['max_budget'], shape=(1,), dtype=np.float32),
            }),
            'storage': spaces.Dict({
                'local_disk_space_usage': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Real feature
                'disk_io_time': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Real feature
                'io_ratio': spaces.Box(low=0, high=3, shape=(1,), dtype=np.float32),  # Real feature
                'cost': spaces.Box(low=0, high=self.resource_constraints['max_budget'], shape=(1,), dtype=np.float32),
            }),
            'network': spaces.Dict({
                'cpu_rate': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Network correlates with CPU
                'canonical_memory_usage': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Memory usage
                'active_machines': spaces.Box(low=30, high=70, shape=(1,), dtype=np.float32),  # Network load
                'active_tasks': spaces.Box(low=60, high=140, shape=(1,), dtype=np.float32),  # Task load
                'cpu_memory_ratio': spaces.Box(low=0, high=5, shape=(1,), dtype=np.float32),  # Load pattern
                'io_ratio': spaces.Box(low=0, high=3, shape=(1,), dtype=np.float32),  # I/O load
                'cost': spaces.Box(low=0, high=self.resource_constraints['max_budget'], shape=(1,), dtype=np.float32),
            }),
            'database': spaces.Dict({
                'cpu_rate': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Database CPU usage
                'canonical_memory_usage': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Database memory
                'local_disk_space_usage': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Database storage
                'disk_io_time': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Database I/O
                'active_machines': spaces.Box(low=30, high=70, shape=(1,), dtype=np.float32),  # Database load
                'active_tasks': spaces.Box(low=60, high=140, shape=(1,), dtype=np.float32),  # Query load
                'cost': spaces.Box(low=0, high=self.resource_constraints['max_budget'], shape=(1,), dtype=np.float32),
            }),
        })
        
        # Action space for each agent
        self.action_space = spaces.Dict({
            'compute': spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32),  # Scale instances, CPU, memory, region, instance type
            'storage': spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),  # Scale storage, type, replication, location
            'network': spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),  # VPC config, subnets, firewalls, load balancers
            'database': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),  # Scale DB instances, storage, type
        })
    
    def _initialize_pricing(self) -> Dict[str, Any]:
        """
        Initialize the pricing model for GCP resources.
        
        Returns:
            Dictionary with pricing information
        """
        # Simplified pricing model (actual GCP pricing is more complex)
        return {
            'compute': {
                'instance_base': 0.05,  # $ per hour
                'cpu': 0.03,  # $ per CPU per hour
                'memory': 0.004,  # $ per GB per hour
                'region_multipliers': {
                    'us-central1': 1.0,
                    'us-east1': 1.0,
                    'us-west1': 1.05,
                    'europe-west1': 1.1,
                    'asia-east1': 1.15
                }
            },
            'storage': {
                'standard': 0.02,  # $ per GB per month
                'ssd': 0.17,  # $ per GB per month
                'replication_multiplier': 1.5  # Cost multiplier for replication
            },
            'network': {
                'ingress': 0.0,  # $ per GB (free)
                'egress': 0.12,  # $ per GB
                'load_balancer': 0.025  # $ per hour
            },
            'database': {
                'instance_base': 0.0543,  # $ per hour
                'storage': 0.17  # $ per GB per month
            }
        }
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment to an initial state.
        
        Returns:
            Initial observation for each agent
        """
        logger.info("Resetting GCP environment")
        
        # Reset step counter
        self.steps = 0
        
        # Reset resources with all resource types
        self.current_resources = {
            'instances': 0,
            'cpus': 0,
            'memory_gb': 0,
            'storage_gb': 0,
            'budget_used': 0.0,
            'network_load': 0.0,
            'database_load': 0.0,
            # Network resources
            'network_bandwidth_gbps': 10,
            'vpc_complexity': 0.5,
            'subnet_count': 3,
            'firewall_rules': 10,
            'load_balancer': 'none',
            'cdn_enabled': False,
            # Database resources
            'database_instances': 1,
            'database_storage_gb': 100,
            'database_type': 'sql',
            'database_complexity': 1.0
        }
        
        # Clear pending operations
        self.pending_operations = []
        
        # Generate new workload
        self._update_workload(initial=True)
        
        # Create initial observation
        observation = self._get_observation()
        
        return observation
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            actions: Dictionary of actions for each agent
            
        Returns:
            Tuple of (observations, rewards, done, info)
        """
        # Increment step counter
        self.steps += 1
        
        # Process pending operations (simulate delays)
        self._process_pending_operations()
        
        # Process actions for each agent
        self._process_actions(actions)
        
        # Update workload and resource utilization
        self._update_workload()
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Get new observations
        observations = self._get_observation()
        
        # Check if episode is done
        done = self.steps >= self.max_steps or self._check_constraints_violated()
        
        # Collect info
        info = {
            'resource_usage': self.current_resources.copy(),
            'costs': self._calculate_costs(),
            'utilization': self._calculate_utilization(),
            'constraints_violated': self._check_constraints_violated(),
            'step': self.steps
        }
        
        return observations, rewards, done, info
    
    def _process_actions(self, actions: Dict[str, np.ndarray]) -> None:
        """
        Process the actions from all agents.
        
        Args:
            actions: Dictionary of actions for each agent
        """
        # Process compute agent actions
        if 'compute' in actions:
            self._process_compute_actions(actions['compute'])
        
        # Process storage agent actions
        if 'storage' in actions:
            self._process_storage_actions(actions['storage'])
        
        # Process network agent actions
        if 'network' in actions:
            self._process_network_actions(actions['network'])
        
        # Process database agent actions
        if 'database' in actions:
            self._process_database_actions(actions['database'])
    
    def _process_compute_actions(self, action: np.ndarray) -> None:
        """
        Process actions from the compute agent.
        
        Args:
            action: Action array from compute agent
        """
        # Extract action components
        scale_instances = action[0]  # -1 to 1 (decrease to increase)
        scale_cpu = action[1]  # -1 to 1 (decrease to increase)
        scale_memory = action[2]  # -1 to 1 (decrease to increase)
        region_preference = action[3]  # -1 to 1 (mapped to region index)
        instance_type = action[4]  # -1 to 1 (mapped to instance type)
        
        # Convert scale factors to actual changes
        instance_change = int(scale_instances * 10)  # -10 to +10 instances
        cpu_per_instance_change = max(1, int((scale_cpu + 1) * 4))  # 1 to 8 CPUs per instance
        memory_per_instance_change = max(1, int((scale_memory + 1) * 8))  # 1 to 16 GB per instance
        
        # Calculate new resource values
        new_instances = max(0, self.current_resources['instances'] + instance_change)
        new_cpus = new_instances * cpu_per_instance_change
        new_memory = new_instances * memory_per_instance_change
        
        # Add pending operation with delay
        delay = int(np.random.normal(
            self.delay_params['provision_delay_mean'],
            self.delay_params['provision_delay_std']
        ))
        delay = max(1, delay)  # Ensure delay is at least 1 step
        
        self.pending_operations.append({
            'type': 'compute',
            'steps_remaining': delay,
            'target': {
                'instances': new_instances,
                'cpus': new_cpus,
                'memory_gb': new_memory
            }
        })
        
        logger.info(f"Scheduled compute change: instances={new_instances}, cpus={new_cpus}, memory={new_memory}GB, delay={delay} steps")
    
    def _process_storage_actions(self, action: np.ndarray) -> None:
        """
        Process actions from the storage agent.
        
        Args:
            action: Action array from storage agent
        """
        # Extract action components
        scale_storage = action[0]  # -1 to 1 (decrease to increase)
        storage_type = action[1]  # -1 to 1 (standard to SSD)
        replication = action[2]  # -1 to 1 (none to multi-region)
        location = action[3]  # -1 to 1 (mapped to location index)
        
        # Improved action scaling for realistic storage allocation
        # Map -1 to 1 range to 0 to 2000 GB for positive scaling
        if scale_storage > 0:
            # Positive scaling: 0 to 2000 GB
            storage_change = int(scale_storage * 2000)
        else:
            # Negative scaling: reduce current storage by up to 50%
            storage_change = int(scale_storage * self.current_resources['storage_gb'] * 0.5)
        
        # Calculate new storage value with realistic constraints
        new_storage = max(0, self.current_resources['storage_gb'] + storage_change)
        new_storage = min(new_storage, self.resource_constraints['max_storage_gb'])  # Respect max constraint
        
        # Add pending operation with delay
        delay = int(np.random.normal(
            self.delay_params['provision_delay_mean'] / 2,  # Storage is faster to provision
            self.delay_params['provision_delay_std'] / 2
        ))
        delay = max(1, delay)  # Ensure delay is at least 1 step
        
        self.pending_operations.append({
            'type': 'storage',
            'steps_remaining': delay,
            'target': {
                'storage_gb': new_storage
            }
        })
        
        logger.info(f"Scheduled storage change: storage={new_storage}GB, delay={delay} steps")
    
    def _process_network_actions(self, action: np.ndarray) -> None:
        """
        Process actions from the network agent.
        
        Args:
            action: Action array from network agent
        """
        # Extract action components
        bandwidth_scale = action[0]  # -1 to 1 (decrease to increase bandwidth)
        vpc_config = action[1]  # -1 to 1 (simple to complex VPC)
        subnet_count = action[2]  # -1 to 1 (few to many subnets)
        firewall_rules = action[3]  # -1 to 1 (basic to advanced security)
        load_balancer = action[4]  # -1 to 1 (none to advanced LB)
        cdn_enabled = action[5]  # -1 to 1 (disabled to enabled)
        
        # Calculate network resources based on actions
        # Bandwidth allocation (Gbps)
        if bandwidth_scale > 0:
            new_bandwidth = int(bandwidth_scale * 100)  # 0 to 100 Gbps
        else:
            new_bandwidth = max(1, int(self.current_resources.get('network_bandwidth_gbps', 10) * (1 + bandwidth_scale * 0.5)))
        
        # VPC complexity affects costs and performance
        vpc_complexity = (vpc_config + 1) / 2  # 0 to 1
        
        # Subnet count affects network segmentation
        subnet_count_val = max(1, int((subnet_count + 1) * 5))  # 1 to 10 subnets
        
        # Firewall rules affect security and costs
        firewall_count = max(5, int((firewall_rules + 1) * 20))  # 5 to 40 rules
        
        # Load balancer configuration
        lb_enabled = load_balancer > 0
        lb_type = "advanced" if load_balancer > 0.5 else "basic" if load_balancer > 0 else "none"
        
        # CDN configuration
        cdn_enabled_val = cdn_enabled > 0
        
        # Update network resources
        self.current_resources['network_bandwidth_gbps'] = new_bandwidth
        self.current_resources['vpc_complexity'] = vpc_complexity
        self.current_resources['subnet_count'] = subnet_count_val
        self.current_resources['firewall_rules'] = firewall_count
        self.current_resources['load_balancer'] = lb_type
        self.current_resources['cdn_enabled'] = cdn_enabled_val
        
        logger.info(f"Network actions: bandwidth={new_bandwidth}Gbps, VPC={vpc_complexity:.2f}, "
                   f"subnets={subnet_count_val}, firewall={firewall_count}, LB={lb_type}, CDN={cdn_enabled_val}")
    
    def _process_database_actions(self, action: np.ndarray) -> None:
        """
        Process actions from the database agent.
        
        Args:
            action: Action array from database agent
        """
        # Extract action components
        db_scale = action[0]  # -1 to 1 (decrease to increase database instances)
        db_storage = action[1]  # -1 to 1 (decrease to increase database storage)
        db_type = action[2]  # -1 to 1 (SQL to NoSQL to managed)
        
        # Calculate database resources based on actions
        # Database instances scaling
        if db_scale > 0:
            new_db_instances = int(db_scale * 20)  # 0 to 20 instances
        else:
            current_instances = self.current_resources.get('database_instances', 1)
            new_db_instances = max(1, int(current_instances * (1 + db_scale * 0.5)))
        
        # Database storage scaling (GB)
        if db_storage > 0:
            new_db_storage = int(db_storage * 5000)  # 0 to 5000 GB
        else:
            current_storage = self.current_resources.get('database_storage_gb', 100)
            new_db_storage = max(10, int(current_storage * (1 + db_storage * 0.5)))
        
        # Database type selection
        if db_type < -0.33:
            db_type_name = "sql"
            db_complexity = 1.0
        elif db_type < 0.33:
            db_type_name = "nosql"
            db_complexity = 1.5
        else:
            db_type_name = "managed"
            db_complexity = 2.0
        
        # Update database resources
        self.current_resources['database_instances'] = new_db_instances
        self.current_resources['database_storage_gb'] = new_db_storage
        self.current_resources['database_type'] = db_type_name
        self.current_resources['database_complexity'] = db_complexity
        
        logger.info(f"Database actions: instances={new_db_instances}, storage={new_db_storage}GB, "
                   f"type={db_type_name}, complexity={db_complexity:.2f}")
    
    def _process_pending_operations(self) -> None:
        """Process pending operations and apply completed ones."""
        completed_operations = []
        
        for operation in self.pending_operations:
            # Decrement remaining steps
            operation['steps_remaining'] -= 1
            
            # Check if operation is complete
            if operation['steps_remaining'] <= 0:
                # Apply the operation
                if operation['type'] == 'compute':
                    self.current_resources['instances'] = operation['target']['instances']
                    self.current_resources['cpus'] = operation['target']['cpus']
                    self.current_resources['memory_gb'] = operation['target']['memory_gb']
                    logger.info(f"Applied compute change: instances={operation['target']['instances']}, " +
                               f"cpus={operation['target']['cpus']}, memory={operation['target']['memory_gb']}GB")
                
                elif operation['type'] == 'storage':
                    self.current_resources['storage_gb'] = operation['target']['storage_gb']
                    logger.info(f"Applied storage change: storage={operation['target']['storage_gb']}GB")
                
                # Mark operation for removal
                completed_operations.append(operation)
        
        # Remove completed operations
        for operation in completed_operations:
            self.pending_operations.remove(operation)
    
    def _update_workload(self, initial: bool = False) -> None:
        """
        Update the current workload based on the workload generator.
        
        Args:
            initial: Whether this is the initial workload
        """
        # Use workload generator if available
        if self.workload_generator is not None:
            new_workload = self.workload_generator.generate_workload()
            if new_workload:  # Check if the generator returned a valid workload
                self.current_workload = new_workload
                logger.debug(f"Generated new workload from generator")
        else:
            # Simple default workload generation
            if initial or self.current_workload is None:
                # Create initial workload
                self.current_workload = {
                    'compute_demand': np.random.uniform(0.2, 0.5, size=3),  # CPU, memory, network
                    'storage_demand': np.random.uniform(0.2, 0.5),
                    'network_demand': np.random.uniform(0.2, 0.5),
                    'database_demand': np.random.uniform(0.2, 0.5, size=2),  # CPU, storage
                }
            else:
                # Add some random fluctuation to the workload
                self.current_workload['compute_demand'] += np.random.normal(0, 0.05, size=3)
                self.current_workload['compute_demand'] = np.clip(self.current_workload['compute_demand'], 0.1, 1.0)
                
                self.current_workload['storage_demand'] += np.random.normal(0, 0.05)
                self.current_workload['storage_demand'] = np.clip(self.current_workload['storage_demand'], 0.1, 1.0)
                
                self.current_workload['network_demand'] += np.random.normal(0, 0.05)
                self.current_workload['network_demand'] = np.clip(self.current_workload['network_demand'], 0.1, 1.0)
                
                self.current_workload['database_demand'] += np.random.normal(0, 0.05, size=2)
                self.current_workload['database_demand'] = np.clip(self.current_workload['database_demand'], 0.1, 1.0)
    
    def _calculate_rewards(self) -> Dict[str, float]:
        """
        Calculate rewards for each agent.
        
        Returns:
            Dictionary of rewards for each agent
        """
        # Calculate utilization and cost efficiency
        utilization = self._calculate_utilization()
        costs = self._calculate_costs()
        
        # Calculate rewards based on utilization (higher is better) and costs (lower is better)
        rewards = {}
        
        # Compute agent reward
        if self.current_resources['instances'] > 0:
            compute_util = np.mean(utilization['compute'])
            compute_cost_efficiency = min(1.0, self.current_workload['compute_demand'].mean() * 100 / (costs['compute'] + 1e-6))
            rewards['compute'] = (0.7 * compute_util - 0.3 * (1 - compute_cost_efficiency)) * self.reward_scale
        else:
            rewards['compute'] = -0.1 * self.reward_scale  # Penalty for having no resources
        
        # Storage agent reward
        if self.current_resources['storage_gb'] > 0:
            storage_util = utilization['storage']
            storage_cost_efficiency = min(1.0, self.current_workload['storage_demand'] * 1000 / (costs['storage'] + 1e-6))
            rewards['storage'] = (0.7 * storage_util - 0.3 * (1 - storage_cost_efficiency)) * self.reward_scale
        else:
            rewards['storage'] = -0.1 * self.reward_scale  # Penalty for having no resources
        
        # Network agent reward (realistic)
        network_bandwidth = self.current_resources.get('network_bandwidth_gbps', 10)
        if network_bandwidth > 0:
            network_util = np.mean(utilization['network'])
            network_cost_efficiency = min(1.0, self.current_workload['network_demand'] * 10 / (costs['network'] + 1e-6))
            rewards['network'] = (0.6 * network_util - 0.4 * (1 - network_cost_efficiency)) * self.reward_scale
        else:
            rewards['network'] = -0.1 * self.reward_scale
        
        # Database agent reward (realistic)
        db_instances = self.current_resources.get('database_instances', 1)
        if db_instances > 0:
            db_util = np.mean(utilization['database'])
            db_cost_efficiency = min(1.0, self.current_workload['database_demand'].mean() * 10 / (costs['database'] + 1e-6))
            rewards['database'] = (0.6 * db_util - 0.4 * (1 - db_cost_efficiency)) * self.reward_scale
        else:
            rewards['database'] = -0.1 * self.reward_scale
        
        # Add constraint violation penalties
        if self._check_constraints_violated():
            for agent in rewards:
                rewards[agent] -= 1.0 * self.reward_scale
        
        return rewards
    
    def _calculate_utilization(self) -> Dict[str, Any]:
        """
        Calculate resource utilization based on workload and provisioned resources.
        
        Returns:
            Dictionary with utilization metrics
        """
        utilization = {}
        
        # Compute utilization
        if self.current_resources['instances'] > 0 and self.current_resources['cpus'] > 0:
            # Calculate how well the provisioned resources match the demand
            cpu_util = min(1.0, self.current_workload['compute_demand'][0] * 100 / self.current_resources['cpus'])
            memory_util = min(1.0, self.current_workload['compute_demand'][1] * 1000 / (self.current_resources['memory_gb'] + 1e-6))
            network_util = min(1.0, self.current_workload['compute_demand'][2])
            
            utilization['compute'] = np.array([cpu_util, memory_util, network_util])
        else:
            utilization['compute'] = np.zeros(3)
        
        # Storage utilization
        if self.current_resources['storage_gb'] > 0:
            storage_util = min(1.0, self.current_workload['storage_demand'] * 1000 / self.current_resources['storage_gb'])
            utilization['storage'] = storage_util
        else:
            utilization['storage'] = 0.0
        
        # Network utilization (realistic)
        bandwidth_util = min(1.0, self.current_workload['network_demand'] * 10 / (self.current_resources.get('network_bandwidth_gbps', 10) + 1e-6))
        vpc_util = self.current_resources.get('vpc_complexity', 0.5)
        utilization['network'] = np.array([bandwidth_util, vpc_util])
        
        # Database utilization (realistic)
        db_instances = self.current_resources.get('database_instances', 1)
        db_storage = self.current_resources.get('database_storage_gb', 100)
        
        db_cpu_util = min(1.0, self.current_workload['database_demand'][0] * 10 / (db_instances + 1e-6))
        db_storage_util = min(1.0, self.current_workload['database_demand'][1] * 1000 / (db_storage + 1e-6))
        utilization['database'] = np.array([db_cpu_util, db_storage_util])
        
        return utilization
    
    def _calculate_costs(self) -> Dict[str, float]:
        """
        Calculate the costs of current resource allocation.
        
        Returns:
            Dictionary with cost breakdown
        """
        costs = {}
        
        # Compute costs (per hour)
        instance_cost = self.current_resources['instances'] * self.pricing['compute']['instance_base']
        cpu_cost = self.current_resources['cpus'] * self.pricing['compute']['cpu']
        memory_cost = self.current_resources['memory_gb'] * self.pricing['compute']['memory']
        costs['compute'] = instance_cost + cpu_cost + memory_cost
        
        # Storage costs (per hour, converted from monthly)
        storage_cost = self.current_resources['storage_gb'] * self.pricing['storage']['standard'] / 730  # ~hours in a month
        costs['storage'] = storage_cost
        
        # Network costs (realistic)
        bandwidth_cost = self.current_resources.get('network_bandwidth_gbps', 10) * 0.01  # $0.01 per Gbps per hour
        vpc_cost = self.current_resources.get('vpc_complexity', 0.5) * 0.02  # $0.02 per complexity unit per hour
        firewall_cost = self.current_resources.get('firewall_rules', 10) * 0.001  # $0.001 per rule per hour
        lb_cost = 0.025 if self.current_resources.get('load_balancer', 'none') != 'none' else 0.0
        cdn_cost = 0.05 if self.current_resources.get('cdn_enabled', False) else 0.0
        costs['network'] = bandwidth_cost + vpc_cost + firewall_cost + lb_cost + cdn_cost
        
        # Database costs (realistic)
        db_instances = self.current_resources.get('database_instances', 1)
        db_storage = self.current_resources.get('database_storage_gb', 100)
        db_complexity = self.current_resources.get('database_complexity', 1.0)
        
        db_instance_cost = db_instances * self.pricing['database']['instance_base'] * db_complexity
        db_storage_cost = db_storage * self.pricing['database']['storage'] / 730  # Convert monthly to hourly
        costs['database'] = db_instance_cost + db_storage_cost
        
        # Total cost
        costs['total'] = sum(costs.values())
        
        # Update budget used
        self.current_resources['budget_used'] += costs['total']
        
        return costs
    
    def _check_constraints_violated(self) -> bool:
        """
        Check if any resource constraints are violated.
        
        Returns:
            True if constraints are violated, False otherwise
        """
        # Check each constraint
        if self.current_resources['instances'] > self.resource_constraints['max_instances']:
            return True
        
        if self.current_resources['cpus'] > self.resource_constraints['max_cpus']:
            return True
        
        if self.current_resources['memory_gb'] > self.resource_constraints['max_memory_gb']:
            return True
        
        if self.current_resources['storage_gb'] > self.resource_constraints['max_storage_gb']:
            return True
        
        if self.current_resources['budget_used'] > self.resource_constraints['max_budget']:
            return True
        
        return False
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation for each agent based on real Google Cluster data features.
        
        Returns:
            Dictionary with observations for each agent
        """
        # Get real data point from workload generator if available
        real_data_point = None
        if self.workload_generator is not None:
            real_data_point = self.workload_generator._get_real_data_point()
        
        # Calculate costs
        costs = self._calculate_costs()
        
        # Create observations based on real Google Cluster features
        observations = {}
        
        # Compute agent observation - using real Google Cluster features
        if real_data_point:
            observations['compute'] = {
                'cpu_rate': np.array([real_data_point.get('cpu_rate', 0.4)], dtype=np.float32),
                'canonical_memory_usage': np.array([real_data_point.get('canonical_memory_usage', 0.5)], dtype=np.float32),
                'local_disk_space_usage': np.array([real_data_point.get('local_disk_space_usage', 0.3)], dtype=np.float32),
                'disk_io_time': np.array([real_data_point.get('disk_io_time', 0.2)], dtype=np.float32),
                'active_machines': np.array([real_data_point.get('active_machines', 50)], dtype=np.float32),
                'active_tasks': np.array([real_data_point.get('active_tasks', 100)], dtype=np.float32),
                'cpu_memory_ratio': np.array([real_data_point.get('cpu_memory_ratio', 0.8)], dtype=np.float32),
                'io_ratio': np.array([real_data_point.get('io_ratio', 0.7)], dtype=np.float32),
                'cost': np.array([costs['compute']], dtype=np.float32),
            }
        else:
            # Fallback to synthetic data
            observations['compute'] = {
                'cpu_rate': np.array([self.current_workload['compute_demand'][0]], dtype=np.float32),
                'canonical_memory_usage': np.array([self.current_workload['compute_demand'][1]], dtype=np.float32),
                'local_disk_space_usage': np.array([self.current_workload['storage_demand']], dtype=np.float32),
                'disk_io_time': np.array([0.2], dtype=np.float32),
                'active_machines': np.array([50], dtype=np.float32),
                'active_tasks': np.array([100], dtype=np.float32),
                'cpu_memory_ratio': np.array([0.8], dtype=np.float32),
                'io_ratio': np.array([0.7], dtype=np.float32),
                'cost': np.array([costs['compute']], dtype=np.float32),
            }
        
        # Storage agent observation - using real Google Cluster features
        if real_data_point:
            observations['storage'] = {
                'local_disk_space_usage': np.array([real_data_point.get('local_disk_space_usage', 0.3)], dtype=np.float32),
                'disk_io_time': np.array([real_data_point.get('disk_io_time', 0.2)], dtype=np.float32),
                'io_ratio': np.array([real_data_point.get('io_ratio', 0.7)], dtype=np.float32),
                'cost': np.array([costs['storage']], dtype=np.float32),
            }
        else:
            observations['storage'] = {
                'local_disk_space_usage': np.array([self.current_workload['storage_demand']], dtype=np.float32),
                'disk_io_time': np.array([0.2], dtype=np.float32),
                'io_ratio': np.array([0.7], dtype=np.float32),
                'cost': np.array([costs['storage']], dtype=np.float32),
            }
        
        # Network agent observation - using real Google Cluster features
        if real_data_point:
            observations['network'] = {
                'cpu_rate': np.array([real_data_point.get('cpu_rate', 0.4)], dtype=np.float32),
                'canonical_memory_usage': np.array([real_data_point.get('canonical_memory_usage', 0.5)], dtype=np.float32),
                'active_machines': np.array([real_data_point.get('active_machines', 50)], dtype=np.float32),
                'active_tasks': np.array([real_data_point.get('active_tasks', 100)], dtype=np.float32),
                'cpu_memory_ratio': np.array([real_data_point.get('cpu_memory_ratio', 0.8)], dtype=np.float32),
                'io_ratio': np.array([real_data_point.get('io_ratio', 0.7)], dtype=np.float32),
                'cost': np.array([costs['network']], dtype=np.float32),
            }
        else:
            observations['network'] = {
                'cpu_rate': np.array([self.current_workload['compute_demand'][0]], dtype=np.float32),
                'canonical_memory_usage': np.array([self.current_workload['compute_demand'][1]], dtype=np.float32),
                'active_machines': np.array([50], dtype=np.float32),
                'active_tasks': np.array([100], dtype=np.float32),
                'cpu_memory_ratio': np.array([0.8], dtype=np.float32),
                'io_ratio': np.array([0.7], dtype=np.float32),
                'cost': np.array([costs['network']], dtype=np.float32),
            }
        
        # Database agent observation - using real Google Cluster features
        if real_data_point:
            observations['database'] = {
                'cpu_rate': np.array([real_data_point.get('cpu_rate', 0.4)], dtype=np.float32),
                'canonical_memory_usage': np.array([real_data_point.get('canonical_memory_usage', 0.5)], dtype=np.float32),
                'local_disk_space_usage': np.array([real_data_point.get('local_disk_space_usage', 0.3)], dtype=np.float32),
                'disk_io_time': np.array([real_data_point.get('disk_io_time', 0.2)], dtype=np.float32),
                'active_machines': np.array([real_data_point.get('active_machines', 50)], dtype=np.float32),
                'active_tasks': np.array([real_data_point.get('active_tasks', 100)], dtype=np.float32),
                'cost': np.array([costs['database']], dtype=np.float32),
            }
        else:
            observations['database'] = {
                'cpu_rate': np.array([self.current_workload['database_demand'][0]], dtype=np.float32),
                'canonical_memory_usage': np.array([self.current_workload['database_demand'][1]], dtype=np.float32),
                'local_disk_space_usage': np.array([self.current_workload['storage_demand']], dtype=np.float32),
                'disk_io_time': np.array([0.2], dtype=np.float32),
                'active_machines': np.array([50], dtype=np.float32),
                'active_tasks': np.array([100], dtype=np.float32),
                'cost': np.array([costs['database']], dtype=np.float32),
            }
        
        return observations
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            Rendered image if mode is 'rgb_array', otherwise None
        """
        # Simple rendering for now
        if mode == 'human':
            print(f"Step: {self.steps}")
            print(f"Resources: {self.current_resources}")
            print(f"Workload: {self.current_workload}")
            print(f"Pending operations: {len(self.pending_operations)}")
            return None
        
        elif mode == 'rgb_array':
            # Return a simple placeholder image
            return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources."""
        pass
    
    def seed(self, seed=None):
        """Set random seed."""
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
    
    def set_workload_pattern(self, pattern_name: str) -> bool:
        """
        Set the current workload pattern.
        
        Args:
            pattern_name: Name of the pattern to use
            
        Returns:
            True if pattern was set successfully, False otherwise
        """
        if self.workload_generator is not None:
            return self.workload_generator.set_pattern(pattern_name)
        else:
            logger.warning("No workload generator available")
            return False
    
    def visualize_workloads(self, steps: int = 100) -> None:
        """
        Visualize available workload patterns.
        
        Args:
            steps: Number of time steps to visualize
        """
        if self.workload_generator is not None:
            self.workload_generator.visualize_patterns(steps)
        else:
            logger.warning("No workload generator available for visualization")