from src.marl_gcp.data.workload_generator import WorkloadGenerator
from src.marl_gcp.configs.default_config import get_default_config

config = get_default_config()
workload_config = config.get('workload_generator', {})

print(f"Workload config: {workload_config}")

# Test the WorkloadGenerator directly
try:
    wg = WorkloadGenerator(workload_config)
    print("WorkloadGenerator initialized successfully!")
    
    # Test generating a workload
    workload = wg.generate_workload()
    print(f"Generated workload: {workload}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
