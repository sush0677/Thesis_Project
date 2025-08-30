from marl_gcp.environment.gcp_environment import GCPEnvironment
from marl_gcp.configs.default_config import get_default_config

print("Testing GCP Environment initialization from src directory...")

config = get_default_config()

try:
    # Create GCP environment
    env = GCPEnvironment(config)
    print("GCP Environment created successfully!")
    
    # Check workload generator status
    if env.workload_generator:
        print(f"Workload generator exists: True")
        print(f"cluster_data is None: {env.workload_generator.cluster_data is None}")
        if hasattr(env.workload_generator, 'train_data'):
            print(f"train_data length: {len(env.workload_generator.train_data)}")
        if hasattr(env.workload_generator, 'val_data'):
            print(f"val_data length: {len(env.workload_generator.val_data)}")
        if hasattr(env.workload_generator, 'test_data'):
            print(f"test_data length: {len(env.workload_generator.test_data)}")
            
        # Test generating a workload directly
        print("\nTesting workload generation...")
        workload = env.workload_generator.generate_workload()
        print(f"Generated workload: {workload}")
    else:
        print("No workload generator!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
