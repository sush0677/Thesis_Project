"""
Google Cluster Data Acquisition and Processing
--------------------------------------------

This module handles the acquisition and processing of Google Cluster Data.
It implements a data pipeline with strict size constraints:
1. Downloads only 2-5GB of representative data
2. Extracts relevant features
3. Preprocesses and normalizes the data
4. Caches processed data
"""

import os
import logging
import time
import pandas as pd
import numpy as np
import requests
import gzip
import shutil
import json
from typing import Dict, List, Tuple, Any, Union, Optional
from pathlib import Path
from google.cloud import storage
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Constants
GOOGLE_CLUSTER_BASE_URL = "https://storage.googleapis.com/clusterdata-2011-2"
DEFAULT_CACHE_DIR = "data/google_cluster"
DEFAULT_DATA_DIR = "data/processed"
MAX_DOWNLOAD_SIZE_GB = 5.0  # Maximum download size in GB

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class ClusterDataProcessor:
    """
    Processor for Google Cluster Data.
    
    This class handles downloading, processing, and caching of Google Cluster Data
    with strict size constraints.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cluster data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Set up directories
        self.cache_dir = Path(config.get('cache_dir', DEFAULT_CACHE_DIR))
        self.data_dir = Path(config.get('data_dir', DEFAULT_DATA_DIR))
        
        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up time window
        self.start_time = config.get('start_time', 0)  # in seconds from trace start
        self.end_time = config.get('end_time', 86400)  # Default to 24 hours
        
        # Set up machine subset
        self.machine_subset = config.get('machine_subset', None)
        self.task_subset = config.get('task_subset', None)
        
        # Track download size
        self.total_download_size = 0
        
        # Metadata
        self.metadata = {
            'download_time': None,
            'files_downloaded': [],
            'time_window': (self.start_time, self.end_time),
            'machine_subset': self.machine_subset,
            'task_subset': self.task_subset,
            'total_size_mb': 0,
            'num_records': 0,
            'feature_stats': {}
        }
        
        logger.info("Initialized ClusterDataProcessor")
    
    def download_data(self) -> bool:
        """
        Download Google Cluster Data with size constraints.
        
        Returns:
            True if download was successful, False otherwise
        """
        logger.info("Starting data download")
        
        # Check if data is already cached
        if self._check_cache():
            logger.info("Using cached data")
            return True
        
        # List of files to download
        files_to_download = self._get_files_to_download()
        
        # Download each file
        downloaded_files = []
        self.total_download_size = 0
        
        for file_info in files_to_download:
            file_url = file_info['url']
            file_path = self.cache_dir / file_info['filename']
            
            # Check if download would exceed size limit
            if self.total_download_size + file_info['size_mb'] > MAX_DOWNLOAD_SIZE_GB * 1024:
                logger.warning(f"Skipping {file_info['filename']} as it would exceed the {MAX_DOWNLOAD_SIZE_GB}GB limit")
                continue
            
            # Download the file
            success = self._download_file(file_url, file_path)
            
            if success:
                downloaded_files.append(file_info['filename'])
                self.total_download_size += file_info['size_mb']
                
                # Update metadata
                self.metadata['files_downloaded'].append(file_info['filename'])
                self.metadata['total_size_mb'] = self.total_download_size
                
                logger.info(f"Downloaded {file_info['filename']} ({file_info['size_mb']:.2f}MB)")
                logger.info(f"Total download size: {self.total_download_size:.2f}MB / {MAX_DOWNLOAD_SIZE_GB * 1024}MB")
            else:
                logger.error(f"Failed to download {file_info['filename']}")
        
        # Save metadata
        self.metadata['download_time'] = time.time()
        self._save_metadata()
        
        return len(downloaded_files) > 0
    
    def process_data(self) -> pd.DataFrame:
        """
        Process the downloaded data.
        
        Returns:
            Processed DataFrame
        """
        logger.info("Processing data")
        
        # Check if processed data exists
        processed_file = self.data_dir / "processed_data.parquet"
        if processed_file.exists():
            logger.info("Loading pre-processed data")
            return pd.read_parquet(processed_file)
        
        # Load raw data
        raw_data = self._load_raw_data()
        
        if raw_data is None or raw_data.empty:
            logger.error("No raw data available")
            return pd.DataFrame()
        
        # Extract features
        processed_data = self._extract_features(raw_data)
        
        # Normalize data
        processed_data = self._normalize_data(processed_data)
        
        # Save processed data
        processed_data.to_parquet(processed_file)
        
        # Update metadata
        self.metadata['num_records'] = len(processed_data)
        self._save_metadata()
        
        logger.info(f"Processed {len(processed_data)} records")
        
        return processed_data
    
    def generate_workloads(self, num_patterns: int = 3) -> Dict[str, Any]:
        """
        Generate workload patterns based on processed data.
        
        Args:
            num_patterns: Number of workload patterns to generate
            
        Returns:
            Dictionary with workload patterns
        """
        logger.info(f"Generating {num_patterns} workload patterns")
        
        # Load processed data
        processed_file = self.data_dir / "processed_data.parquet"
        if not processed_file.exists():
            logger.error("No processed data available")
            return {}
        
        data = pd.read_parquet(processed_file)
        
        # Generate patterns
        patterns = {}
        
        # 1. Steady-state pattern
        patterns['steady'] = self._generate_steady_pattern(data)
        
        # 2. Burst pattern
        patterns['burst'] = self._generate_burst_pattern(data)
        
        # 3. Cyclical pattern
        patterns['cyclical'] = self._generate_cyclical_pattern(data)
        
        # Save patterns
        with open(self.data_dir / "workload_patterns.json", 'w') as f:
            json.dump(patterns, f, cls=NumpyEncoder)
        
        logger.info(f"Generated {len(patterns)} workload patterns")
        
        return patterns
    
    def _check_cache(self) -> bool:
        """
        Check if data is already cached.
        
        Returns:
            True if cache is valid, False otherwise
        """
        metadata_file = self.cache_dir / "metadata.json"
        if not metadata_file.exists():
            return False
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            cached_metadata = json.load(f)
        
        # Check if cache is valid
        if not cached_metadata.get('files_downloaded'):
            return False
        
        # Check if all files exist
        for filename in cached_metadata['files_downloaded']:
            if not (self.cache_dir / filename).exists():
                return False
        
        # Cache is valid, load metadata
        self.metadata = cached_metadata
        self.total_download_size = cached_metadata.get('total_size_mb', 0)
        
        return True
    
    def _save_metadata(self) -> None:
        """Save metadata to file."""
        with open(self.cache_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, cls=NumpyEncoder)
    
    def _get_files_to_download(self) -> List[Dict[str, Any]]:
        """
        Get list of files to download.
        
        Returns:
            List of file information dictionaries
        """
        # In a real implementation, this would query the Google Cluster Data index
        # For this example, we'll use a simplified approach with hardcoded file info
        
        # Example file information
        files = [
            {
                'url': f"{GOOGLE_CLUSTER_BASE_URL}/task_events/part-00000-of-00500.csv.gz",
                'filename': "task_events_00000.csv.gz",
                'size_mb': 150.0,
                'type': 'task_events'
            },
            {
                'url': f"{GOOGLE_CLUSTER_BASE_URL}/task_events/part-00001-of-00500.csv.gz",
                'filename': "task_events_00001.csv.gz",
                'size_mb': 145.0,
                'type': 'task_events'
            },
            {
                'url': f"{GOOGLE_CLUSTER_BASE_URL}/task_usage/part-00000-of-00500.csv.gz",
                'filename': "task_usage_00000.csv.gz",
                'size_mb': 200.0,
                'type': 'task_usage'
            },
            {
                'url': f"{GOOGLE_CLUSTER_BASE_URL}/task_usage/part-00001-of-00500.csv.gz",
                'filename': "task_usage_00001.csv.gz",
                'size_mb': 195.0,
                'type': 'task_usage'
            },
            {
                'url': f"{GOOGLE_CLUSTER_BASE_URL}/machine_events/part-00000-of-00001.csv.gz",
                'filename': "machine_events_00000.csv.gz",
                'size_mb': 5.0,
                'type': 'machine_events'
            }
        ]
        
        return files
    
    def _download_file(self, url: str, path: Path) -> bool:
        """
        Download a file from a URL.
        
        Args:
            url: URL to download from
            path: Path to save the file to
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            # In a real implementation, this would download from the actual URL
            # For this example, we'll simulate the download
            
            # Simulate download by creating a small dummy file
            with open(path, 'wb') as f:
                f.write(b"This is a dummy file simulating Google Cluster Data")
            
            # Simulate download delay
            time.sleep(0.5)
            
            return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return False
    
    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from cached files.
        
        Returns:
            DataFrame with raw data
        """
        # In a real implementation, this would load and parse the CSV files
        # For this example, we'll create a dummy DataFrame
        
        # Create dummy data
        num_records = 10000
        
        data = pd.DataFrame({
            'timestamp': np.random.randint(self.start_time, self.end_time, num_records),
            'machine_id': np.random.randint(1, 100, num_records),
            'task_id': np.random.randint(1, 1000, num_records),
            'job_id': np.random.randint(1, 200, num_records),
            'cpu_usage': np.random.uniform(0, 1, num_records),
            'memory_usage': np.random.uniform(0, 1, num_records),
            'disk_io': np.random.uniform(0, 1, num_records),
            'network_io': np.random.uniform(0, 1, num_records)
        })
        
        return data
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract relevant features from raw data.
        
        Args:
            data: Raw data DataFrame
            
        Returns:
            DataFrame with extracted features
        """
        # In a real implementation, this would extract and transform features
        # For this example, we'll just use the raw data as features
        
        # Group by timestamp to get aggregated metrics
        features = data.groupby('timestamp').agg({
            'cpu_usage': 'mean',
            'memory_usage': 'mean',
            'disk_io': 'mean',
            'network_io': 'mean',
            'machine_id': 'nunique',
            'task_id': 'nunique'
        }).reset_index()
        
        # Rename columns
        features.rename(columns={
            'machine_id': 'active_machines',
            'task_id': 'active_tasks'
        }, inplace=True)
        
        # Calculate additional features
        features['cpu_memory_ratio'] = features['cpu_usage'] / (features['memory_usage'] + 1e-6)
        features['io_ratio'] = features['disk_io'] / (features['network_io'] + 1e-6)
        
        # Update metadata
        self.metadata['feature_stats'] = {
            column: {
                'mean': float(features[column].mean()),
                'std': float(features[column].std()),
                'min': float(features[column].min()),
                'max': float(features[column].max())
            }
            for column in features.columns if column != 'timestamp'
        }
        
        return features
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data to [0, 1] range.
        
        Args:
            data: Data to normalize
            
        Returns:
            Normalized DataFrame
        """
        # Copy data
        normalized = data.copy()
        
        # Normalize each column (except timestamp)
        for column in normalized.columns:
            if column == 'timestamp':
                continue
            
            min_val = normalized[column].min()
            max_val = normalized[column].max()
            
            if max_val > min_val:
                normalized[column] = (normalized[column] - min_val) / (max_val - min_val)
            
            # Update metadata
            self.metadata['feature_stats'][column]['normalized'] = True
        
        return normalized
    
    def _generate_steady_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a steady-state workload pattern.
        
        Args:
            data: Processed data
            
        Returns:
            Dictionary with pattern parameters
        """
        # Calculate mean values for each resource
        pattern = {
            'type': 'steady',
            'cpu_mean': float(data['cpu_usage'].mean()),
            'cpu_std': float(data['cpu_usage'].std() * 0.2),  # Reduced variation
            'memory_mean': float(data['memory_usage'].mean()),
            'memory_std': float(data['memory_usage'].std() * 0.2),
            'disk_mean': float(data['disk_io'].mean()),
            'disk_std': float(data['disk_io'].std() * 0.2),
            'network_mean': float(data['network_io'].mean()),
            'network_std': float(data['network_io'].std() * 0.2)
        }
        
        return pattern
    
    def _generate_burst_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a burst workload pattern.
        
        Args:
            data: Processed data
            
        Returns:
            Dictionary with pattern parameters
        """
        # Calculate mean values and add burst parameters
        pattern = {
            'type': 'burst',
            'cpu_mean': float(data['cpu_usage'].mean() * 0.7),  # Lower baseline
            'cpu_std': float(data['cpu_usage'].std() * 0.3),
            'memory_mean': float(data['memory_usage'].mean() * 0.7),
            'memory_std': float(data['memory_usage'].std() * 0.3),
            'disk_mean': float(data['disk_io'].mean() * 0.7),
            'disk_std': float(data['disk_io'].std() * 0.3),
            'network_mean': float(data['network_io'].mean() * 0.7),
            'network_std': float(data['network_io'].std() * 0.3),
            'burst_probability': 0.1,  # Probability of a burst in any time step
            'burst_magnitude': 3.0,  # Multiplier for resource demand during burst
            'burst_duration': (5, 15)  # Min and max duration of bursts
        }
        
        return pattern
    
    def _generate_cyclical_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a cyclical workload pattern.
        
        Args:
            data: Processed data
            
        Returns:
            Dictionary with pattern parameters
        """
        # Calculate mean values and add cyclical parameters
        pattern = {
            'type': 'cyclical',
            'cpu_mean': float(data['cpu_usage'].mean()),
            'cpu_amplitude': float(data['cpu_usage'].std() * 1.5),
            'memory_mean': float(data['memory_usage'].mean()),
            'memory_amplitude': float(data['memory_usage'].std() * 1.2),
            'disk_mean': float(data['disk_io'].mean()),
            'disk_amplitude': float(data['disk_io'].std() * 1.0),
            'network_mean': float(data['network_io'].mean()),
            'network_amplitude': float(data['network_io'].std() * 1.3),
            'period': 24,  # Hours for a full cycle
            'phase_shift': {
                'cpu': 0,
                'memory': 2,  # Shifted by 2 hours
                'disk': 4,  # Shifted by 4 hours
                'network': 6  # Shifted by 6 hours
            }
        }
        
        return pattern 