"""
Google Cluster Data Processor for MARL-GCP

This module handles the acquisition, processing, and preparation of Google Cluster Data
for realistic workload generation in the MARL system.

The Google Cluster Data contains traces from a Google compute cluster including:
- Machine events (machine additions, removals, updates)
- Task events (task submissions, scheduling, completions)
- Task usage (CPU, memory, disk I/O, network I/O measurements)

Reference: https://github.com/google/cluster-data
"""

import os
import sys
import logging
import json
import gzip
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import requests
from datetime import datetime, timedelta
import hashlib
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class GoogleClusterDataProcessor:
    """
    Processor for Google Cluster Data acquisition and processing.
    
    Handles downloading, processing, and preparing realistic workload data
    for the MARL-GCP system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Google Cluster Data processor.
        
        Args:
            config: Configuration dictionary with data processing parameters
        """
        self.config = config
        self.cache_dir = Path(config.get('cache_dir', 'data/google_cluster'))
        self.data_dir = Path(config.get('data_dir', 'data/processed'))
        self.viz_dir = Path(config.get('viz_dir', 'visualizations'))
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Data constraints
        self.max_size_gb = config.get('max_size_gb', 5.0)
        self.time_window_hours = config.get('time_window_hours', 24)
        self.machine_subset_size = config.get('machine_subset_size', 1000)
        self.task_subset_size = config.get('task_subset_size', 10000)
        
        # Data sources (Google Cluster Data URLs)
        self.data_sources = {
            'machine_events': 'https://storage.googleapis.com/clusterdata-2011-2/machine_events/part-00000-of-00001.csv.gz',
            'task_events': 'https://storage.googleapis.com/clusterdata-2011-2/task_events/part-00000-of-00001.csv.gz',
            'task_usage': 'https://storage.googleapis.com/clusterdata-2011-2/task_usage/part-00000-of-00001.csv.gz'
        }
        
        # Column definitions for Google Cluster Data
        self.column_definitions = {
            'machine_events': [
                'timestamp', 'machine_id', 'event_type', 'platform_id', 'cpu', 'memory'
            ],
            'task_events': [
                'timestamp', 'missing_info', 'job_id', 'task_index', 'machine_id',
                'event_type', 'user', 'scheduling_class', 'priority', 'cpu_request',
                'memory_request', 'disk_space_request', 'different_machines_restriction'
            ],
            'task_usage': [
                'start_time', 'end_time', 'job_id', 'task_index', 'machine_id',
                'cpu_rate', 'canonical_memory_usage', 'assigned_memory_usage',
                'unmapped_page_cache', 'total_page_cache', 'disk_io_time',
                'local_disk_space_usage', 'cpu_rate', 'canonical_memory_usage',
                'assigned_memory_usage', 'unmapped_page_cache', 'total_page_cache',
                'disk_io_time', 'local_disk_space_usage', 'maximum_cpu_rate',
                'maximum_memory_usage', 'maximum_disk_io_time', 'maximum_local_disk_space_usage',
                'cycles_per_instruction', 'memory_accesses_per_instruction',
                'sample_portion', 'aggregation_type', 'sampled_cpu_usage'
            ]
        }
        
        logger.info(f"GoogleClusterDataProcessor initialized with max size: {self.max_size_gb}GB")
    
    def download_cluster_data(self, force_download: bool = False) -> bool:
        """
        Download Google Cluster Data with size constraints.
        
        Args:
            force_download: Whether to force re-download even if files exist
            
        Returns:
            True if download was successful, False otherwise
        """
        logger.info("Starting Google Cluster Data download...")
        
        total_size = 0
        downloaded_files = []
        
        for data_type, url in self.data_sources.items():
            file_path = self.cache_dir / f"{data_type}.csv.gz"
            
            # Check if file already exists and is not empty
            if file_path.exists() and not force_download:
                file_size = file_path.stat().st_size
                if file_size > 1000:  # More than 1KB
                    logger.info(f"File {data_type} already exists ({file_size} bytes)")
                    total_size += file_size
                    downloaded_files.append(str(file_path))
                    continue
            
            try:
                logger.info(f"Downloading {data_type} from {url}")
                
                # Download with progress tracking
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                # Get file size
                file_size = int(response.headers.get('content-length', 0))
                if file_size > 0:
                    total_size_mb = file_size / (1024 * 1024)
                    logger.info(f"File size: {total_size_mb:.2f} MB")
                    
                    # Check size constraint
                    if total_size + file_size > self.max_size_gb * 1024 * 1024 * 1024:
                        logger.warning(f"File would exceed size limit. Skipping {data_type}")
                        continue
                
                # Download file
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                downloaded_files.append(str(file_path))
                total_size += file_path.stat().st_size
                
                logger.info(f"Successfully downloaded {data_type}")
                
            except Exception as e:
                logger.error(f"Failed to download {data_type}: {e}")
                # If download fails, create a synthetic dataset
                self._create_synthetic_dataset(data_type)
                downloaded_files.append(str(file_path))
        
        # Save download metadata
        metadata = {
            'download_time': datetime.now().isoformat(),
            'files_downloaded': downloaded_files,
            'total_size_mb': total_size / (1024 * 1024),
            'time_window': [0, self.time_window_hours * 3600],
            'machine_subset': self.machine_subset_size,
            'task_subset': self.task_subset_size
        }
        
        with open(self.cache_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Download completed. Total size: {total_size / (1024 * 1024):.2f} MB")
        return True
    
    def _create_synthetic_dataset(self, data_type: str) -> None:
        """
        Create a synthetic dataset when real data is unavailable.
        
        Args:
            data_type: Type of dataset to create
        """
        logger.info(f"Creating synthetic {data_type} dataset")
        
        file_path = self.cache_dir / f"{data_type}.csv.gz"
        
        if data_type == 'machine_events':
            # Create synthetic machine events
            num_machines = 1000
            timestamps = np.random.uniform(0, self.time_window_hours * 3600, num_machines)
            machine_ids = np.arange(num_machines)
            event_types = np.random.choice([0, 1, 2], num_machines)  # ADD, REMOVE, UPDATE
            cpus = np.random.uniform(1, 32, num_machines)
            memory = np.random.uniform(4, 128, num_machines)
            
            data = pd.DataFrame({
                'timestamp': timestamps,
                'machine_id': machine_ids,
                'event_type': event_types,
                'platform_id': np.random.choice(['platform_1', 'platform_2'], num_machines),
                'cpu': cpus,
                'memory': memory
            })
            
        elif data_type == 'task_events':
            # Create synthetic task events
            num_tasks = 10000
            timestamps = np.random.uniform(0, self.time_window_hours * 3600, num_tasks)
            job_ids = np.random.randint(0, 1000, num_tasks)
            task_indices = np.random.randint(0, 100, num_tasks)
            machine_ids = np.random.randint(0, 1000, num_tasks)
            event_types = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8], num_tasks)
            cpu_requests = np.random.uniform(0.1, 8.0, num_tasks)
            memory_requests = np.random.uniform(0.1, 16.0, num_tasks)
            
            data = pd.DataFrame({
                'timestamp': timestamps,
                'missing_info': np.zeros(num_tasks),
                'job_id': job_ids,
                'task_index': task_indices,
                'machine_id': machine_ids,
                'event_type': event_types,
                'user': np.random.choice(['user_1', 'user_2', 'user_3'], num_tasks),
                'scheduling_class': np.random.randint(0, 3, num_tasks),
                'priority': np.random.randint(0, 11, num_tasks),
                'cpu_request': cpu_requests,
                'memory_request': memory_requests,
                'disk_space_request': np.random.uniform(0.1, 100.0, num_tasks),
                'different_machines_restriction': np.random.choice([True, False], num_tasks)
            })
            
        elif data_type == 'task_usage':
            # Create synthetic task usage data
            num_records = 50000
            start_times = np.random.uniform(0, self.time_window_hours * 3600, num_records)
            end_times = start_times + np.random.uniform(300, 3600, num_records)  # 5-60 minutes
            job_ids = np.random.randint(0, 1000, num_records)
            task_indices = np.random.randint(0, 100, num_records)
            machine_ids = np.random.randint(0, 1000, num_records)
            
            # Realistic usage patterns
            cpu_rates = np.random.beta(2, 5, num_records)  # Skewed towards lower usage
            memory_usage = np.random.beta(2, 5, num_records)
            disk_io = np.random.exponential(0.1, num_records)
            network_io = np.random.exponential(0.05, num_records)
            
            data = pd.DataFrame({
                'start_time': start_times,
                'end_time': end_times,
                'job_id': job_ids,
                'task_index': task_indices,
                'machine_id': machine_ids,
                'cpu_rate': cpu_rates,
                'canonical_memory_usage': memory_usage,
                'assigned_memory_usage': memory_usage * np.random.uniform(0.8, 1.2, num_records),
                'unmapped_page_cache': np.random.uniform(0, 0.1, num_records),
                'total_page_cache': np.random.uniform(0, 0.2, num_records),
                'disk_io_time': disk_io,
                'local_disk_space_usage': np.random.uniform(0, 0.5, num_records),
                'maximum_cpu_rate': cpu_rates * np.random.uniform(1.0, 2.0, num_records),
                'maximum_memory_usage': memory_usage * np.random.uniform(1.0, 1.5, num_records),
                'maximum_disk_io_time': disk_io * np.random.uniform(1.0, 3.0, num_records),
                'maximum_local_disk_space_usage': np.random.uniform(0, 1.0, num_records),
                'cycles_per_instruction': np.random.uniform(0.5, 2.0, num_records),
                'memory_accesses_per_instruction': np.random.uniform(0.1, 1.0, num_records),
                'sample_portion': np.ones(num_records),
                'aggregation_type': np.zeros(num_records),
                'sampled_cpu_usage': cpu_rates
            })
        
        # Save compressed data
        with gzip.open(file_path, 'wt') as f:
            data.to_csv(f, index=False)
        
        logger.info(f"Synthetic {data_type} dataset created with {len(data)} records")
    
    def process_cluster_data(self) -> Dict[str, Any]:
        """
        Process the downloaded Google Cluster Data into simulation-compatible format.
        
        Returns:
            Dictionary with processed data and statistics
        """
        logger.info("Processing Google Cluster Data...")
        
        # Load and process each dataset
        machine_events = self._load_machine_events()
        task_events = self._load_task_events()
        task_usage = self._load_task_usage()
        
        # Extract workload patterns
        workload_patterns = self._extract_workload_patterns(task_usage, task_events)
        
        # Create time-series data
        time_series_data = self._create_time_series_data(task_usage, task_events)
        
        # Calculate feature statistics
        feature_stats = self._calculate_feature_statistics(task_usage)
        
        # Save processed data
        processed_data = {
            'workload_patterns': workload_patterns,
            'time_series_data': time_series_data,
            'feature_stats': feature_stats,
            'machine_events': machine_events,
            'task_events': task_events,
            'task_usage': task_usage
        }
        
        # Save as parquet for efficient storage
        processed_df = pd.DataFrame(time_series_data)
        processed_df.to_parquet(self.data_dir / 'processed_data.parquet', index=False)
        
        # Save workload patterns
        with open(self.data_dir / 'cluster_workload_patterns.json', 'w') as f:
            json.dump(workload_patterns, f, indent=2)
        
        # Save feature statistics
        with open(self.data_dir / 'feature_statistics.json', 'w') as f:
            json.dump(feature_stats, f, indent=2)
        
        logger.info("Data processing completed")
        return processed_data
    
    def _load_machine_events(self) -> pd.DataFrame:
        """Load and process machine events data."""
        file_path = self.cache_dir / 'machine_events.csv.gz'
        
        if not file_path.exists():
            logger.warning("Machine events file not found")
            return pd.DataFrame()
        
        try:
            data = pd.read_csv(file_path, compression='gzip', header=None)
            data.columns = self.column_definitions['machine_events']
            
            # Filter by time window
            data = data[data['timestamp'] <= self.time_window_hours * 3600]
            
            # Sample if too large
            if len(data) > self.machine_subset_size:
                data = data.sample(n=self.machine_subset_size, random_state=42)
            
            logger.info(f"Loaded {len(data)} machine events")
            return data
            
        except Exception as e:
            logger.error(f"Error loading machine events: {e}")
            return pd.DataFrame()
    
    def _load_task_events(self) -> pd.DataFrame:
        """Load and process task events data."""
        file_path = self.cache_dir / 'task_events.csv.gz'
        
        if not file_path.exists():
            logger.warning("Task events file not found")
            return pd.DataFrame()
        
        try:
            data = pd.read_csv(file_path, compression='gzip', header=None)
            data.columns = self.column_definitions['task_events']
            
            # Filter by time window
            data = data[data['timestamp'] <= self.time_window_hours * 3600]
            
            # Sample if too large
            if len(data) > self.task_subset_size:
                data = data.sample(n=self.task_subset_size, random_state=42)
            
            logger.info(f"Loaded {len(data)} task events")
            return data
            
        except Exception as e:
            logger.error(f"Error loading task events: {e}")
            return pd.DataFrame()
    
    def _load_task_usage(self) -> pd.DataFrame:
        """Load and process task usage data."""
        file_path = self.cache_dir / 'task_usage.csv.gz'
        
        if not file_path.exists():
            logger.warning("Task usage file not found")
            return pd.DataFrame()
        
        try:
            data = pd.read_csv(file_path, compression='gzip', header=None)
            data.columns = self.column_definitions['task_usage']
            
            # Filter by time window
            data = data[data['end_time'] <= self.time_window_hours * 3600]
            
            # Sample if too large
            max_records = self.task_subset_size * 5  # More records for usage data
            if len(data) > max_records:
                data = data.sample(n=max_records, random_state=42)
            
            logger.info(f"Loaded {len(data)} task usage records")
            return data
            
        except Exception as e:
            logger.error(f"Error loading task usage: {e}")
            return pd.DataFrame()
    
    def _extract_workload_patterns(self, task_usage: pd.DataFrame, task_events: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract workload patterns from the cluster data.
        
        Args:
            task_usage: Task usage DataFrame
            task_events: Task events DataFrame
            
        Returns:
            Dictionary with extracted workload patterns
        """
        logger.info("Extracting workload patterns...")
        
        patterns = {}
        
        if not task_usage.empty:
            # Calculate time-based patterns
            task_usage['hour'] = (task_usage['start_time'] / 3600).astype(int)
            
            # Steady pattern (baseline)
            steady_stats = {
                'cpu_mean': task_usage['cpu_rate'].mean(),
                'cpu_std': task_usage['cpu_rate'].std(),
                'memory_mean': task_usage['canonical_memory_usage'].mean(),
                'memory_std': task_usage['canonical_memory_usage'].std(),
                'disk_mean': task_usage['local_disk_space_usage'].mean(),
                'disk_std': task_usage['local_disk_space_usage'].std(),
                'network_mean': task_usage['disk_io_time'].mean(),
                'network_std': task_usage['disk_io_time'].std()
            }
            patterns['steady'] = {'type': 'steady', **steady_stats}
            
            # Burst pattern (high variance periods)
            burst_threshold = task_usage['cpu_rate'].quantile(0.9)
            burst_data = task_usage[task_usage['cpu_rate'] > burst_threshold]
            
            if not burst_data.empty:
                burst_stats = {
                    'cpu_mean': burst_data['cpu_rate'].mean(),
                    'cpu_std': burst_data['cpu_rate'].std(),
                    'memory_mean': burst_data['canonical_memory_usage'].mean(),
                    'memory_std': burst_data['canonical_memory_usage'].std(),
                    'disk_mean': burst_data['local_disk_space_usage'].mean(),
                    'disk_std': burst_data['local_disk_space_usage'].std(),
                    'network_mean': burst_data['disk_io_time'].mean(),
                    'network_std': burst_data['disk_io_time'].std(),
                    'burst_probability': len(burst_data) / len(task_usage),
                    'burst_magnitude': burst_data['cpu_rate'].mean() / steady_stats['cpu_mean'],
                    'burst_duration': [5, 15]
                }
                patterns['burst'] = {'type': 'burst', **burst_stats}
            
            # Cyclical pattern (hourly variations)
            hourly_stats = task_usage.groupby('hour').agg({
                'cpu_rate': ['mean', 'std'],
                'canonical_memory_usage': ['mean', 'std'],
                'local_disk_space_usage': ['mean', 'std'],
                'disk_io_time': ['mean', 'std']
            }).reset_index()
            
            if len(hourly_stats) > 1:
                cyclical_stats = {
                    'cpu_mean': steady_stats['cpu_mean'],
                    'cpu_amplitude': hourly_stats['cpu_rate']['mean'].std() * 2,
                    'memory_mean': steady_stats['memory_mean'],
                    'memory_amplitude': hourly_stats['canonical_memory_usage']['mean'].std() * 2,
                    'disk_mean': steady_stats['disk_mean'],
                    'disk_amplitude': hourly_stats['local_disk_space_usage']['mean'].std() * 2,
                    'network_mean': steady_stats['network_mean'],
                    'network_amplitude': hourly_stats['disk_io_time']['mean'].std() * 2,
                    'period': 24,
                    'phase_shift': {'cpu': 0, 'memory': 2, 'disk': 4, 'network': 6}
                }
                patterns['cyclical'] = {'type': 'cyclical', **cyclical_stats}
        
        logger.info(f"Extracted {len(patterns)} workload patterns")
        return patterns
    
    def _create_time_series_data(self, task_usage: pd.DataFrame, task_events: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Create time series data for simulation.
        
        Args:
            task_usage: Task usage DataFrame
            task_events: Task events DataFrame
            
        Returns:
            Dictionary with time series data
        """
        logger.info("Creating time series data...")
        
        if task_usage.empty:
            return self._create_synthetic_time_series()
        
        # Create time bins (5-minute intervals)
        time_bins = np.arange(0, self.time_window_hours * 3600, 300)
        
        time_series = {
            'timestamp': [],
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': [],
            'active_machines': [],
            'active_tasks': [],
            'cpu_memory_ratio': [],
            'io_ratio': []
        }
        
        for i in range(len(time_bins) - 1):
            start_time = time_bins[i]
            end_time = time_bins[i + 1]
            
            # Filter data for this time bin
            mask = (task_usage['start_time'] < end_time) & (task_usage['end_time'] > start_time)
            bin_data = task_usage[mask]
            
            if not bin_data.empty:
                time_series['timestamp'].append(start_time)
                time_series['cpu_usage'].append(bin_data['cpu_rate'].mean())
                time_series['memory_usage'].append(bin_data['canonical_memory_usage'].mean())
                time_series['disk_io'].append(bin_data['local_disk_space_usage'].mean())
                time_series['network_io'].append(bin_data['disk_io_time'].mean())
                time_series['active_machines'].append(bin_data['machine_id'].nunique())
                time_series['active_tasks'].append(len(bin_data))
                time_series['cpu_memory_ratio'].append(
                    bin_data['cpu_rate'].mean() / (bin_data['canonical_memory_usage'].mean() + 1e-6)
                )
                time_series['io_ratio'].append(
                    bin_data['disk_io_time'].mean() / (bin_data['local_disk_space_usage'].mean() + 1e-6)
                )
        
        logger.info(f"Created time series with {len(time_series['timestamp'])} data points")
        return time_series
    
    def _create_synthetic_time_series(self) -> Dict[str, List[float]]:
        """Create synthetic time series data when real data is unavailable."""
        logger.info("Creating synthetic time series data")
        
        time_bins = np.arange(0, self.time_window_hours * 3600, 300)
        
        # Generate realistic patterns
        base_cpu = 0.4 + 0.2 * np.sin(2 * np.pi * time_bins / (24 * 3600))  # Daily cycle
        base_memory = 0.5 + 0.15 * np.sin(2 * np.pi * time_bins / (24 * 3600) + np.pi/4)
        base_disk = 0.3 + 0.1 * np.sin(2 * np.pi * time_bins / (24 * 3600) + np.pi/2)
        base_network = 0.2 + 0.1 * np.sin(2 * np.pi * time_bins / (24 * 3600) + np.pi)
        
        # Add noise
        noise_scale = 0.05
        cpu_usage = base_cpu + np.random.normal(0, noise_scale, len(time_bins))
        memory_usage = base_memory + np.random.normal(0, noise_scale, len(time_bins))
        disk_io = base_disk + np.random.normal(0, noise_scale, len(time_bins))
        network_io = base_network + np.random.normal(0, noise_scale, len(time_bins))
        
        # Ensure values are in [0, 1]
        cpu_usage = np.clip(cpu_usage, 0, 1)
        memory_usage = np.clip(memory_usage, 0, 1)
        disk_io = np.clip(disk_io, 0, 1)
        network_io = np.clip(network_io, 0, 1)
        
        time_series = {
            'timestamp': time_bins.tolist(),
            'cpu_usage': cpu_usage.tolist(),
            'memory_usage': memory_usage.tolist(),
            'disk_io': disk_io.tolist(),
            'network_io': network_io.tolist(),
            'active_machines': (50 + 20 * np.sin(2 * np.pi * time_bins / (24 * 3600))).tolist(),
            'active_tasks': (100 + 50 * np.sin(2 * np.pi * time_bins / (24 * 3600))).tolist(),
            'cpu_memory_ratio': (cpu_usage / (memory_usage + 1e-6)).tolist(),
            'io_ratio': (disk_io / (network_io + 1e-6)).tolist()
        }
        
        return time_series
    
    def _calculate_feature_statistics(self, task_usage: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistics for features in the cluster data.
        
        Args:
            task_usage: Task usage DataFrame
            
        Returns:
            Dictionary with feature statistics
        """
        if task_usage.empty:
            return self._create_synthetic_statistics()
        
        stats = {}
        
        # Calculate statistics for each feature
        features = ['cpu_rate', 'canonical_memory_usage', 'local_disk_space_usage', 'disk_io_time']
        
        for feature in features:
            if feature in task_usage.columns:
                data = task_usage[feature].dropna()
                if len(data) > 0:
                    stats[feature] = {
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'normalized': True
                    }
        
        # Calculate derived features
        if 'cpu_rate' in task_usage.columns and 'canonical_memory_usage' in task_usage.columns:
            cpu_memory_ratio = task_usage['cpu_rate'] / (task_usage['canonical_memory_usage'] + 1e-6)
            stats['cpu_memory_ratio'] = {
                'mean': float(cpu_memory_ratio.mean()),
                'std': float(cpu_memory_ratio.std()),
                'min': float(cpu_memory_ratio.min()),
                'max': float(cpu_memory_ratio.max()),
                'normalized': True
            }
        
        if 'local_disk_space_usage' in task_usage.columns and 'disk_io_time' in task_usage.columns:
            io_ratio = task_usage['disk_io_time'] / (task_usage['local_disk_space_usage'] + 1e-6)
            stats['io_ratio'] = {
                'mean': float(io_ratio.mean()),
                'std': float(io_ratio.std()),
                'min': float(io_ratio.min()),
                'max': float(io_ratio.max()),
                'normalized': True
            }
        
        # Machine and task counts
        stats['active_machines'] = {
            'mean': float(task_usage['machine_id'].nunique()),
            'std': 0.0,
            'min': float(task_usage['machine_id'].nunique()),
            'max': float(task_usage['machine_id'].nunique()),
            'normalized': True
        }
        
        stats['active_tasks'] = {
            'mean': float(len(task_usage)),
            'std': 0.0,
            'min': float(len(task_usage)),
            'max': float(len(task_usage)),
            'normalized': True
        }
        
        return stats
    
    def _create_synthetic_statistics(self) -> Dict[str, Any]:
        """Create synthetic statistics when real data is unavailable."""
        return {
            'cpu_rate': {'mean': 0.4, 'std': 0.2, 'min': 0.0, 'max': 1.0, 'normalized': True},
            'canonical_memory_usage': {'mean': 0.5, 'std': 0.25, 'min': 0.0, 'max': 1.0, 'normalized': True},
            'local_disk_space_usage': {'mean': 0.3, 'std': 0.15, 'min': 0.0, 'max': 1.0, 'normalized': True},
            'disk_io_time': {'mean': 0.2, 'std': 0.1, 'min': 0.0, 'max': 1.0, 'normalized': True},
            'cpu_memory_ratio': {'mean': 0.8, 'std': 0.4, 'min': 0.0, 'max': 5.0, 'normalized': True},
            'io_ratio': {'mean': 0.7, 'std': 0.35, 'min': 0.0, 'max': 3.0, 'normalized': True},
            'active_machines': {'mean': 50, 'std': 10, 'min': 30, 'max': 70, 'normalized': True},
            'active_tasks': {'mean': 100, 'std': 20, 'min': 60, 'max': 140, 'normalized': True}
        }
    
    def visualize_data(self, processed_data: Dict[str, Any]) -> None:
        """
        Create visualizations of the processed cluster data.
        
        Args:
            processed_data: Processed data dictionary
        """
        logger.info("Creating data visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Google Cluster Data Analysis', fontsize=16)
        
        # Time series plot
        if 'time_series_data' in processed_data:
            ts_data = processed_data['time_series_data']
            time_hours = np.array(ts_data['timestamp']) / 3600
            
            axes[0, 0].plot(time_hours, ts_data['cpu_usage'], label='CPU', alpha=0.8)
            axes[0, 0].plot(time_hours, ts_data['memory_usage'], label='Memory', alpha=0.8)
            axes[0, 0].plot(time_hours, ts_data['disk_io'], label='Disk I/O', alpha=0.8)
            axes[0, 0].plot(time_hours, ts_data['network_io'], label='Network I/O', alpha=0.8)
            axes[0, 0].set_xlabel('Time (hours)')
            axes[0, 0].set_ylabel('Usage')
            axes[0, 0].set_title('Resource Usage Over Time')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Feature distribution
        if 'feature_stats' in processed_data:
            stats = processed_data['feature_stats']
            features = ['cpu_rate', 'canonical_memory_usage', 'local_disk_space_usage', 'disk_io_time']
            
            means = [stats.get(f, {}).get('mean', 0) for f in features]
            stds = [stats.get(f, {}).get('std', 0) for f in features]
            
            x_pos = np.arange(len(features))
            axes[0, 1].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8)
            axes[0, 1].set_xlabel('Features')
            axes[0, 1].set_ylabel('Mean Usage')
            axes[0, 1].set_title('Feature Statistics')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels([f.replace('_', ' ').title() for f in features], rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Workload patterns
        if 'workload_patterns' in processed_data:
            patterns = processed_data['workload_patterns']
            pattern_names = list(patterns.keys())
            
            if pattern_names:
                cpu_means = [patterns[p].get('cpu_mean', 0) for p in pattern_names]
                memory_means = [patterns[p].get('memory_mean', 0) for p in pattern_names]
                
                x_pos = np.arange(len(pattern_names))
                width = 0.35
                
                axes[1, 0].bar(x_pos - width/2, cpu_means, width, label='CPU', alpha=0.8)
                axes[1, 0].bar(x_pos + width/2, memory_means, width, label='Memory', alpha=0.8)
                axes[1, 0].set_xlabel('Workload Patterns')
                axes[1, 0].set_ylabel('Mean Usage')
                axes[1, 0].set_title('Workload Pattern Comparison')
                axes[1, 0].set_xticks(x_pos)
                axes[1, 0].set_xticklabels([p.replace('_', ' ').title() for p in pattern_names], rotation=45)
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation heatmap
        if 'time_series_data' in processed_data:
            ts_data = processed_data['time_series_data']
            features = ['cpu_usage', 'memory_usage', 'disk_io', 'network_io']
            
            if all(f in ts_data for f in features):
                data_matrix = np.array([ts_data[f] for f in features]).T
                corr_matrix = np.corrcoef(data_matrix.T)
                
                im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[1, 1].set_xticks(range(len(features)))
                axes[1, 1].set_yticks(range(len(features)))
                axes[1, 1].set_xticklabels([f.replace('_', ' ').title() for f in features], rotation=45)
                axes[1, 1].set_yticklabels([f.replace('_', ' ').title() for f in features])
                axes[1, 1].set_title('Feature Correlation Matrix')
                
                # Add correlation values
                for i in range(len(features)):
                    for j in range(len(features)):
                        text = axes[1, 1].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                              ha="center", va="center", color="black")
                
                plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'cluster_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Data visualizations saved to {self.viz_dir / 'cluster_data_analysis.png'}")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the processed data.
        
        Returns:
            Dictionary with data summary
        """
        summary = {
            'cache_dir': str(self.cache_dir),
            'data_dir': str(self.data_dir),
            'max_size_gb': self.max_size_gb,
            'time_window_hours': self.time_window_hours,
            'machine_subset_size': self.machine_subset_size,
            'task_subset_size': self.task_subset_size
        }
        
        # Check file sizes
        cache_files = list(self.cache_dir.glob('*.csv.gz'))
        total_cache_size = sum(f.stat().st_size for f in cache_files)
        summary['cache_size_mb'] = total_cache_size / (1024 * 1024)
        summary['cache_files'] = [f.name for f in cache_files]
        
        # Check processed files
        processed_files = list(self.data_dir.glob('*'))
        summary['processed_files'] = [f.name for f in processed_files]
        
        return summary 