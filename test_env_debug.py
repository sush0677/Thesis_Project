from src.marl_gcp.environment.gcp_environment import GCPEnvironment
from src.marl_gcp.configs.default_config import get_default_config

print("Testing GCP Environment initialization...")

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
    else:
        print("No workload generator!")
    
    # Test a single step to see if workload generation works
    print("\nTesting environment step...")
    actions = {
        'compute': 0,
        'storage': 0, 
        'network': 0,
        'database': 0
    }
    
    # Get initial state
    states = env.get_state()
    print("Got initial states successfully")
    
    # Try a step
    states, rewards, done, info = env.step(actions)
    print(f"Step completed successfully!")
    print(f"Rewards: {rewards}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
