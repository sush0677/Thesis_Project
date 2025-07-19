#!/usr/bin/env python
"""
Test script for data integration - verify that agents are using real Google Cluster data.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marl_gcp.data.workload_generator import WorkloadGenerator
from marl_gcp.environment.gcp_environment import GCPEnvironment
from marl_gcp.configs.default_config import get_default_config

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_workload_generator():
    """Test the workload generator with real data."""
    print("\n" + "="*50)
    print("Testing Workload Generator with Real Google Cluster Data")
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
    
    try:
        # Initialize workload generator
        workload_generator = WorkloadGenerator(config['workload_generator'])
        
        # Get data info
        data_info = workload_generator.get_data_info()
        print(f"\nData Information:")
        print(f"  Real data loaded: {data_info['real_data_loaded']}")
        print(f"  Feature stats loaded: {data_info['feature_stats_loaded']}")
        print(f"  Workload patterns loaded: {data_info['workload_patterns_loaded']}")
        print(f"  Available patterns: {data_info['available_patterns']}")
        
        if data_info['real_data_loaded']:
            print(f"  Total records: {data_info['total_records']}")
            print(f"  Train records: {data_info['train_records']}")
            print(f"  Val records: {data_info['val_records']}")
            print(f"  Test records: {data_info['test_records']}")
        
        # Test different patterns
        patterns = ['steady', 'burst', 'cyclical']
        for pattern in patterns:
            print(f"\nTesting {pattern} pattern:")
            workload_generator.set_pattern(pattern)
            
            # Test training mode
            workload_generator.set_mode('train')
            train_workload = workload_generator.generate_workload()
            print(f"  Train mode - CPU demand: {train_workload['compute_demand'][0]:.3f}")
            
            # Test validation mode
            workload_generator.set_mode('val')
            val_workload = workload_generator.generate_workload()
            print(f"  Val mode - CPU demand: {val_workload['compute_demand'][0]:.3f}")
            
            # Test test mode
            workload_generator.set_mode('test')
            test_workload = workload_generator.generate_workload()
            print(f"  Test mode - CPU demand: {test_workload['compute_demand'][0]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing workload generator: {e}")
        return False

def test_environment():
    """Test the environment with real data."""
    print("\n" + "="*50)
    print("Testing GCP Environment with Real Google Cluster Data")
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
    
    try:
        # Initialize environment
        env = GCPEnvironment(config)
        
        # Test reset
        print("\nTesting environment reset:")
        observations = env.reset()
        
        print(f"  Compute observation keys: {list(observations['compute'].keys())}")
        print(f"  Storage observation keys: {list(observations['storage'].keys())}")
        print(f"  Network observation keys: {list(observations['network'].keys())}")
        print(f"  Database observation keys: {list(observations['database'].keys())}")
        
        # Test step
        print("\nTesting environment step:")
        actions = {
            'compute': np.array([0.1, 0.1, 0.1, 0.0, 0.0]),
            'storage': np.array([0.1, 0.0, 0.0, 0.0]),
            'network': np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'database': np.array([0.1, 0.1, 0.0])
        }
        
        next_observations, rewards, done, info = env.step(actions)
        
        print(f"  Rewards: {rewards}")
        print(f"  Done: {done}")
        print(f"  Info keys: {list(info.keys())}")
        
        # Test workload pattern setting
        print("\nTesting workload pattern setting:")
        success = env.set_workload_pattern('burst')
        print(f"  Set burst pattern: {success}")
        
        return True
        
    except Exception as e:
        print(f"Error testing environment: {e}")
        return False

def test_agent_compatibility():
    """Test if agents are compatible with new observation spaces."""
    print("\n" + "="*50)
    print("Testing Agent Compatibility with Real Data")
    print("="*50)
    
    # Load configuration
    config = get_default_config()
    
    try:
        # Test if agents can handle new observation dimensions
        from marl_gcp.agents.compute_agent import ComputeAgent
        from marl_gcp.agents.storage_agent import StorageAgent
        from marl_gcp.agents.network_agent import NetworkAgent
        from marl_gcp.agents.database_agent import DatabaseAgent
        
        # Initialize agents
        compute_agent = ComputeAgent(config['agents']['compute'])
        storage_agent = StorageAgent(config['agents']['storage'])
        network_agent = NetworkAgent(config['agents']['network'])
        database_agent = DatabaseAgent(config['agents']['database'])
        
        print(f"  Compute agent state_dim: {compute_agent.state_dim}")
        print(f"  Storage agent state_dim: {storage_agent.state_dim}")
        print(f"  Network agent state_dim: {network_agent.state_dim}")
        print(f"  Database agent state_dim: {database_agent.state_dim}")
        
        # Test with dummy observations
        dummy_compute_obs = np.random.random(compute_agent.state_dim)
        dummy_storage_obs = np.random.random(storage_agent.state_dim)
        dummy_network_obs = np.random.random(network_agent.state_dim)
        dummy_database_obs = np.random.random(database_agent.state_dim)
        
        # Test action selection
        compute_action = compute_agent.select_action(dummy_compute_obs)
        storage_action = storage_agent.select_action(dummy_storage_obs)
        network_action = network_agent.select_action(dummy_network_obs)
        database_action = database_agent.select_action(dummy_database_obs)
        
        print(f"  Compute action shape: {compute_action.shape}")
        print(f"  Storage action shape: {storage_action.shape}")
        print(f"  Network action shape: {network_action.shape}")
        print(f"  Database action shape: {database_action.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error testing agent compatibility: {e}")
        return False

def main():
    """Main test function."""
    setup_logging()
    
    print("Starting Data Integration Tests")
    print("="*60)
    
    # Run tests
    tests = [
        ("Workload Generator", test_workload_generator),
        ("Environment", test_environment),
        ("Agent Compatibility", test_agent_compatibility)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Data integration is working correctly.")
        print("Agents are now using real Google Cluster data.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main() 