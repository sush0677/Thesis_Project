from src.marl_gcp.data.workload_generator import WorkloadGenerator
from src.marl_gcp.configs.default_config import get_default_config

# Test the exact same config that main.py uses
config = get_default_config()

# Create WorkloadGenerator with the same config as main.py
from src.marl_gcp.environment.gcp_environment import GCPEnvironment

print("Testing with GCP Environment config processing...")

# Simulate the exact same config processing as in gcp_environment.py
workload_config = config.get('workload_generator', {})

# Debug: Print the config that would be passed to WorkloadGenerator
print(f"Original workload_config: {workload_config}")

# Simulate the path override check
if 'data_dir' not in workload_config:
    workload_config['data_dir'] = config.get('cluster_data', {}).get('data_dir', 'data/processed')
if 'cache_dir' not in workload_config:
    workload_config['cache_dir'] = config.get('cluster_data', {}).get('cache_dir', 'data/google_cluster')
if 'viz_dir' not in workload_config:
    workload_config['viz_dir'] = config.get('viz_dir', 'visualizations')

print(f"Final workload_config: {workload_config}")

# Test with this config
try:
    wg = WorkloadGenerator(workload_config)
    print("WorkloadGenerator initialized successfully!")
    
    # Check the data status
    print(f"cluster_data is None: {wg.cluster_data is None}")
    print(f"train_data length: {len(wg.train_data) if hasattr(wg, 'train_data') else 'No train_data'}")
    print(f"val_data length: {len(wg.val_data) if hasattr(wg, 'val_data') else 'No val_data'}")
    print(f"test_data length: {len(wg.test_data) if hasattr(wg, 'test_data') else 'No test_data'}")
    
    # Test generating a workload
    workload = wg.generate_workload()
    print(f"Generated workload: {workload}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
