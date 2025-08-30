from src.marl_gcp.configs.default_config import get_default_config
import os

config = get_default_config()
data_dir = config["workload_generator"]["data_dir"]
print(f'Data dir: {data_dir}')
print(f'Exists: {os.path.exists(data_dir + "/processed_data.parquet")}')
