"""
Realistic workload generator for GCP resource provisioning simulation.

This module generates realistic workload patterns based on real Google Cluster Data
and common application types (web services, batch processing, ML workloads, etc.).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import pickle

logger = logging.getLogger(__name__)

class WorkloadGenerator:
    """
    Generates realistic workload patterns for GCP resource provisioning simulation.
    
    Based on real Google Cluster Data and common application patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the workload generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = Path(config.get('data_dir', 'data/processed'))
        self.cache_dir = Path(config.get('cache_dir', 'data/google_cluster'))
        self.viz_dir = Path(config.get('viz_dir', 'visualizations'))
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Data split ratios
        self.train_ratio = 0.70  # 70% for training
        self.val_ratio = 0.15    # 15% for validation
        self.test_ratio = 0.15   # 15% for testing
        
        # Load real Google Cluster Data
        self.cluster_data = self._load_real_cluster_data()
        self.feature_stats = self._load_feature_statistics()
        self.workload_patterns = self._load_workload_patterns()
        
        # Data splits
        self.train_data, self.val_data, self.test_data = self._split_data()
        
        # Current pattern and mode
        self.current_pattern = config.get('default_pattern', 'steady')
        self.current_mode = 'train'  # train, val, test
        self.step_counter = 0
        self.data_index = 0
        
        # Workload patterns (now based on real data)
        self.patterns = {
            'steady': self._generate_real_steady_workload,
            'burst': self._generate_real_burst_workload,
            'cyclical': self._generate_real_cyclical_workload,
            'web_service': self._generate_real_web_service_workload,
            'batch_processing': self._generate_real_batch_processing_workload,
            'ml_training': self._generate_real_ml_training_workload,
            'data_analytics': self._generate_real_data_analytics_workload
        }
        
        logger.info(f"WorkloadGenerator initialized with real Google Cluster data")
        logger.info(f"Data split: Train={len(self.train_data)}, Val={len(self.val_data)}, Test={len(self.test_data)}")
    
    def _load_real_cluster_data(self) -> Optional[pd.DataFrame]:
        """
        Load real Google Cluster data from raw files.
        
        Returns:
            DataFrame with real cluster data or None if not available
        """
        try:
            # First try to load from processed data (if available)
            processed_file = self.data_dir / 'processed_data.parquet'
            if processed_file.exists():
                data = pd.read_parquet(processed_file)
                logger.info(f"✅ Loaded processed Google Cluster data: {len(data)} records")
                logger.info(f"✅ Data columns: {list(data.columns)}")
                return data
            
            # If processed data not available, try to load from raw google cluster data
            logger.info("⚠️ Processed data not found, attempting to load from raw Google Cluster data...")
            
            # Look for parquet files in the cache directory
            parquet_files = list(self.cache_dir.glob('*.parquet.gz'))
            if not parquet_files:
                logger.error(f"❌ No parquet files found in {self.cache_dir}")
                return None
            
            # Load the first available parquet file
            data_file = parquet_files[0]
            logger.info(f"📁 Loading data from: {data_file}")
            
            # Read the compressed parquet file
            data = pd.read_parquet(data_file)
            logger.info(f"✅ Loaded raw Google Cluster data: {len(data)} records")
            logger.info(f"✅ Data columns: {list(data.columns)}")
            
            # Basic data preprocessing
            if 'timestamp' not in data.columns:
                # Add timestamp if not present
                data['timestamp'] = range(len(data))
            
            # Ensure we have the required columns for workload generation
            required_columns = ['cpu_usage', 'memory_usage', 'disk_io', 'network_io']
            available_columns = list(data.columns)
            
            # Map available columns to required columns if needed
            column_mapping = {}
            for col in required_columns:
                if col in available_columns:
                    column_mapping[col] = col
                elif col.replace('_', '') in available_columns:
                    column_mapping[col] = col.replace('_', '')
                else:
                    # Use default values if column not found
                    data[col] = 0.5  # Default value
                    column_mapping[col] = col
            
            # Add missing columns with default values
            if 'active_machines' not in data.columns:
                data['active_machines'] = 50
            if 'active_tasks' not in data.columns:
                data['active_tasks'] = 100
            if 'cpu_memory_ratio' not in data.columns:
                data['cpu_memory_ratio'] = data['cpu_usage'] / (data['memory_usage'] + 1e-6)
            if 'io_ratio' not in data.columns:
                data['io_ratio'] = data['disk_io'] / (data['network_io'] + 1e-6)
            
            logger.info(f"✅ Preprocessed Google Cluster data with {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error loading cluster data: {e}")
            return None
    
    def _load_feature_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Load feature statistics from real Google Cluster data.
        
        Returns:
            Dictionary with feature statistics or None if not available
        """
        try:
            # First try to load from processed directory
            stats_file = self.data_dir / 'feature_statistics.json'
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                # Check if file is not empty
                if stats and len(stats) > 0:
                    logger.info(f"✅ Loaded feature statistics: {list(stats.keys())}")
                    return stats
                else:
                    logger.warning("⚠️ Feature statistics file is empty")
            
            # If not found or empty, try to load from cache directory
            cache_stats_file = self.cache_dir / 'feature_statistics.json'
            if cache_stats_file.exists():
                with open(cache_stats_file, 'r') as f:
                    stats = json.load(f)
                if stats and len(stats) > 0:
                    logger.info(f"✅ Loaded feature statistics from cache: {list(stats.keys())}")
                    return stats
            
            # Generate statistics from loaded data
            if self.cluster_data is not None:
                logger.info("📊 Generating feature statistics from real cluster data...")
                stats = self._generate_feature_statistics()
                if stats and len(stats) > 0:
                    # Save the generated statistics
                    with open(stats_file, 'w') as f:
                        json.dump(stats, f, indent=2)
                    logger.info(f"✅ Generated and saved feature statistics: {list(stats.keys())}")
                    return stats
            else:
                logger.error(f"❌ Feature statistics not found and no data available to generate from")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading feature statistics: {e}")
            return None
    
    def _load_workload_patterns(self) -> Optional[Dict[str, Any]]:
        """
        Load workload patterns from real Google Cluster data.
        
        Returns:
            Dictionary with workload patterns or None if not available
        """
        try:
            # First try to load from processed directory
            patterns_file = self.data_dir / 'workload_patterns.json'
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns = json.load(f)
                # Check if file is not empty
                if patterns and len(patterns) > 0:
                    logger.info(f"✅ Loaded workload patterns: {list(patterns.keys())}")
                    return patterns
                else:
                    logger.warning("⚠️ Workload patterns file is empty")
            
            # If not found or empty, try to load from cache directory
            cache_patterns_file = self.cache_dir / 'workload_patterns.json'
            if cache_patterns_file.exists():
                with open(cache_patterns_file, 'r') as f:
                    patterns = json.load(f)
                if patterns and len(patterns) > 0:
                    logger.info(f"✅ Loaded workload patterns from cache: {list(patterns.keys())}")
                    return patterns
            
            # Generate patterns from loaded data
            if self.cluster_data is not None:
                logger.info("📊 Generating workload patterns from real cluster data...")
                patterns = self._generate_workload_patterns()
                if patterns and len(patterns) > 0:
                    # Save the generated patterns
                    with open(patterns_file, 'w') as f:
                        json.dump(patterns, f, indent=2)
                    logger.info(f"✅ Generated and saved workload patterns: {list(patterns.keys())}")
                    return patterns
            else:
                logger.error(f"❌ Workload patterns not found and no data available to generate from")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading workload patterns: {e}")
            return None
    
    def _split_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split the real Google Cluster data into train/validation/test sets.
        
        Returns:
            Tuple of (train_data, val_data, test_data) as lists of dictionaries
        """
        if self.cluster_data is None:
            logger.warning("❌ No cluster data available for splitting")
            return [], [], []
        
        try:
            # Convert DataFrame to list of dictionaries
            data_list = self.cluster_data.to_dict('records')
        
            # Calculate split indices
            total_size = len(data_list)
            train_size = int(total_size * self.train_ratio)
            val_size = int(total_size * self.val_ratio)
        
            # Split the data
            train_data = data_list[:train_size]
            val_data = data_list[train_size:train_size + val_size]
            test_data = data_list[train_size + val_size:]
        
            logger.info(f"✅ Data split completed:")
            logger.info(f"   - Training: {len(train_data)} records ({len(train_data)/total_size*100:.1f}%)")
            logger.info(f"   - Validation: {len(val_data)} records ({len(val_data)/total_size*100:.1f}%)")
            logger.info(f"   - Test: {len(test_data)} records ({len(test_data)/total_size*100:.1f}%)")
        
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"❌ Error splitting data: {e}")
            return [], [], []
    
    def set_mode(self, mode: str) -> bool:
        """
        Set the data mode (train, val, test).
        
        Args:
            mode: Data mode ('train', 'val', 'test')
            
        Returns:
            True if mode was set successfully, False otherwise
        """
        if mode in ['train', 'val', 'test']:
            self.current_mode = mode
            self.data_index = 0
            logger.info(f"Switched to {mode} mode")
            return True
        else:
            logger.warning(f"Unknown mode: {mode}")
            return False
    
    def set_pattern(self, pattern_name: str) -> bool:
        """
        Set the workload pattern.
        
        Args:
            pattern_name: Name of the pattern to use
            
        Returns:
            True if pattern was set successfully, False otherwise
        """
        if pattern_name in self.patterns:
            self.current_pattern = pattern_name
            self.step_counter = 0
            self.data_index = 0
            logger.info(f"Switched to workload pattern: {pattern_name}")
            return True
        else:
            logger.warning(f"Unknown workload pattern: {pattern_name}")
            return False
    
    def generate_workload(self) -> Dict[str, np.ndarray]:
        """
        Generate the next workload based on real Google Cluster data.
        
        Returns:
            Dictionary with workload demands for each resource type
        """
        self.step_counter += 1
        
        # Check if we have real data available
        if self.cluster_data is not None and len(self.train_data) > 0:
            logger.debug(f"🟢 Using REAL Google Cluster data for workload generation")
            workload = self.patterns[self.current_pattern]()
            
            # Check if we got a valid workload
            if workload and isinstance(workload, dict):
                logger.debug(f"✅ Generated real workload for pattern: {self.current_pattern}")
                return workload
            else:
                logger.warning(f"⚠️ Pattern {self.current_pattern} returned invalid workload, using synthetic")
                return self._generate_synthetic_workload()
                
        else:
            # Fallback to synthetic patterns
            logger.warning("⚠️ Using synthetic patterns - real data not available")
            logger.warning(f"   - cluster_data: {'✅' if self.cluster_data is not None else '❌'}")
            logger.warning(f"   - train_data: {'✅' if len(self.train_data) > 0 else '❌'}")
            return self._generate_synthetic_workload()
    
    def _get_real_data_point(self) -> Optional[Dict[str, float]]:
        """
        Get a real data point from the current mode.
        
        Returns:
            Dictionary with real data features or None if not available
        """
        # Determine which dataset to use based on current mode
        if self.current_mode == 'train':
            data = self.train_data
            mode_name = "training"
        elif self.current_mode == 'val':
            data = self.val_data
            mode_name = "validation"
        else:  # test
            data = self.test_data
            mode_name = "test"
        
        if not data:
            logger.warning(f"❌ No {mode_name} data available")
            return None
        
        # Get data point and advance index
        data_point = data[self.data_index % len(data)]
        self.data_index += 1
        
        # Log the real data point for debugging
        logger.debug(f"📊 Real data point from {mode_name} set: {list(data_point.keys())}")
        
        return data_point
    
    def _generate_real_steady_workload(self) -> Dict[str, np.ndarray]:
        """
        Generate steady workload based on real Google Cluster data.
        
        Returns:
            Dictionary with steady workload demands
        """
        data_point = self._get_real_data_point()
        
        if data_point:
            # Use real data point directly
            cpu_demand = data_point.get('cpu_usage', 0.4)
            memory_demand = data_point.get('memory_usage', 0.5)
            disk_demand = data_point.get('disk_io', 0.3)
            network_demand = data_point.get('network_io', 0.4)
            
            # Add small noise for variation
            cpu_demand += np.random.normal(0, 0.02)
            memory_demand += np.random.normal(0, 0.02)
            disk_demand += np.random.normal(0, 0.02)
            network_demand += np.random.normal(0, 0.02)
            
            # Clip to valid range
            cpu_demand = np.clip(cpu_demand, 0.1, 0.9)
            memory_demand = np.clip(memory_demand, 0.1, 0.9)
            disk_demand = np.clip(disk_demand, 0.1, 0.9)
            network_demand = np.clip(network_demand, 0.1, 0.9)
            
            logger.debug(f"✅ Generated real steady workload from data point")
            
            return {
                'compute_demand': np.array([cpu_demand, memory_demand, network_demand]),
                'storage_demand': disk_demand,
                'network_demand': network_demand,
                'database_demand': np.array([cpu_demand * 0.8, disk_demand * 0.9])
            }
        else:
            # Fallback to synthetic
            logger.warning("⚠️ No real data point available, using synthetic steady workload")
            return self._generate_synthetic_steady_workload()
    
    def _generate_real_burst_workload(self) -> Dict[str, np.ndarray]:
        """
        Generate burst workload based on real Google Cluster data.
        
        Returns:
            Dictionary with burst workload demands
        """
        data_point = self._get_real_data_point()
        
        if data_point:
            # Use real data point as base
            cpu_base = data_point.get('cpu_usage', 0.4)
            memory_base = data_point.get('memory_usage', 0.5)
            disk_base = data_point.get('disk_io', 0.3)
            network_base = data_point.get('network_io', 0.4)
            
            # Determine if we're in a burst period
            burst_prob = 0.1
            burst_magnitude = 2.5
            
            # Check if current step is in burst period
            burst_cycle = 50
            in_burst = (self.step_counter % burst_cycle) < np.random.randint(5, 15)
            
            if in_burst and np.random.random() < burst_prob:
                # Burst period - high demand
                multiplier = burst_magnitude
                logger.debug(f"🔥 Burst period detected, multiplier: {multiplier}")
            else:
                # Normal period - low demand
                multiplier = 1.0
            
            # Apply burst multiplier
            cpu_demand = np.clip(cpu_base * multiplier, 0.1, 0.9)
            memory_demand = np.clip(memory_base * multiplier, 0.1, 0.9)
            disk_demand = np.clip(disk_base * multiplier, 0.1, 0.9)
            network_demand = np.clip(network_base * multiplier, 0.1, 0.9)
            
            logger.debug(f"✅ Generated real burst workload from data point")
            
            return {
                'compute_demand': np.array([cpu_demand, memory_demand, network_demand]),
                'storage_demand': disk_demand,
                'network_demand': network_demand,
                'database_demand': np.array([cpu_demand * 0.8, disk_demand * 0.9])
            }
        else:
            # Fallback to synthetic
            logger.warning("⚠️ No real data point available, using synthetic burst workload")
            return self._generate_synthetic_burst_workload()
    
    def _generate_real_cyclical_workload(self) -> Dict[str, np.ndarray]:
        """
        Generate cyclical workload based on real Google Cluster data.
        
        Returns:
            Dictionary with cyclical workload demands
        """
        data_point = self._get_real_data_point()
        
        if data_point:
            # Use real data point as base
            cpu_base = data_point.get('cpu_usage', 0.4)
            memory_base = data_point.get('memory_usage', 0.5)
            disk_base = data_point.get('disk_io', 0.3)
            network_base = data_point.get('network_io', 0.4)
            
            # Add cyclical variation
            period = 24  # 24 steps per cycle
            phase = (self.step_counter % period) / period * 2 * np.pi
            
            # Different phases for different resources
            cpu_phase = phase
            memory_phase = phase + np.pi/2
            disk_phase = phase + np.pi
            network_phase = phase + 3*np.pi/2
            
            # Cyclical variation (amplitude 0.2)
            cpu_variation = 0.2 * np.sin(cpu_phase)
            memory_variation = 0.2 * np.sin(memory_phase)
            disk_variation = 0.2 * np.sin(disk_phase)
            network_variation = 0.2 * np.sin(network_phase)
            
            # Apply cyclical variation
            cpu_demand = np.clip(cpu_base + cpu_variation, 0.1, 0.9)
            memory_demand = np.clip(memory_base + memory_variation, 0.1, 0.9)
            disk_demand = np.clip(disk_base + disk_variation, 0.1, 0.9)
            network_demand = np.clip(network_base + network_variation, 0.1, 0.9)
            
            logger.debug(f"✅ Generated real cyclical workload from data point")
            
            return {
                'compute_demand': np.array([cpu_demand, memory_demand, network_demand]),
                'storage_demand': disk_demand,
                'network_demand': network_demand,
                'database_demand': np.array([cpu_demand * 0.8, disk_demand * 0.9])
            }
        else:
            # Fallback to synthetic
            logger.warning("⚠️ No real data point available, using synthetic cyclical workload")
            return self._generate_synthetic_cyclical_workload()
    
    def _generate_real_web_service_workload(self) -> Dict[str, np.ndarray]:
        """
        Generate web service workload based on real Google Cluster data.
        
        Returns:
            Dictionary with web service workload demands
        """
        data_point = self._get_real_data_point()
        
        if data_point and self.feature_stats:
            # Use real feature statistics for web service patterns
            # Web services typically have higher CPU and network usage
            cpu_base = self.feature_stats.get('cpu_rate', {}).get('mean', 0.4)
            memory_base = self.feature_stats.get('canonical_memory_usage', {}).get('mean', 0.5)
            disk_base = self.feature_stats.get('local_disk_space_usage', {}).get('mean', 0.3)
            
            # Add web service specific patterns (higher during peak hours)
            hour_of_day = (self.step_counter // 8) % 24
            morning_peak = 8 <= hour_of_day <= 10
            evening_peak = 18 <= hour_of_day <= 20
            
            if morning_peak or evening_peak:
                multiplier = 1.5
            else:
                multiplier = 0.8
            
            # Add realistic noise
            cpu_demand = np.clip(cpu_base * multiplier + np.random.normal(0, 0.1), 0.1, 0.9)
            memory_demand = np.clip(memory_base * multiplier + np.random.normal(0, 0.1), 0.1, 0.9)
            disk_demand = np.clip(disk_base * multiplier + np.random.normal(0, 0.05), 0.1, 0.9)
            network_demand = np.clip(cpu_demand * 1.2 + np.random.normal(0, 0.1), 0.1, 0.9)
            
            return {
                'compute_demand': np.array([cpu_demand, memory_demand, network_demand]),
                'storage_demand': disk_demand,
                'network_demand': network_demand,
                'database_demand': np.array([cpu_demand * 0.6, disk_demand * 0.8])
            }
        else:
            # Fallback to synthetic
            return self._generate_synthetic_web_service_workload()
    
    def _generate_real_batch_processing_workload(self) -> Dict[str, np.ndarray]:
        """
        Generate batch processing workload based on real Google Cluster data.
        
        Returns:
            Dictionary with batch processing workload demands
        """
        data_point = self._get_real_data_point()
        
        if data_point and self.feature_stats:
            # Use real feature statistics for batch processing patterns
            # Batch processing typically has high CPU and disk usage
            cpu_base = self.feature_stats.get('cpu_rate', {}).get('mean', 0.4)
            memory_base = self.feature_stats.get('canonical_memory_usage', {}).get('mean', 0.5)
            disk_base = self.feature_stats.get('local_disk_space_usage', {}).get('mean', 0.3)
            io_base = self.feature_stats.get('disk_io_time', {}).get('mean', 0.2)
            
            # Batch processing patterns (high CPU, high disk I/O)
            cpu_demand = np.clip(cpu_base * 1.3 + np.random.normal(0, 0.15), 0.1, 0.95)
            memory_demand = np.clip(memory_base * 1.1 + np.random.normal(0, 0.1), 0.1, 0.9)
            disk_demand = np.clip(disk_base * 1.4 + np.random.normal(0, 0.2), 0.1, 0.95)
            network_demand = np.clip(io_base * 2.0 + np.random.normal(0, 0.1), 0.1, 0.9)
            
            return {
                'compute_demand': np.array([cpu_demand, memory_demand, network_demand]),
                'storage_demand': disk_demand,
                'network_demand': network_demand,
                'database_demand': np.array([cpu_demand * 0.9, disk_demand * 1.2])
            }
        else:
            # Fallback to synthetic
            return self._generate_synthetic_batch_processing_workload()
    
    def _generate_real_ml_training_workload(self) -> Dict[str, np.ndarray]:
        """
        Generate ML training workload based on real Google Cluster data.
        
        Returns:
            Dictionary with ML training workload demands
        """
        data_point = self._get_real_data_point()
        
        if data_point and self.feature_stats:
            # Use real feature statistics for ML training patterns
            # ML training typically has high CPU, memory, and GPU usage
            cpu_base = self.feature_stats.get('cpu_rate', {}).get('mean', 0.4)
            memory_base = self.feature_stats.get('canonical_memory_usage', {}).get('mean', 0.5)
            disk_base = self.feature_stats.get('local_disk_space_usage', {}).get('mean', 0.3)
            
            # ML training patterns (very high CPU and memory usage)
            cpu_demand = np.clip(cpu_base * 1.8 + np.random.normal(0, 0.2), 0.2, 0.98)
            memory_demand = np.clip(memory_base * 1.6 + np.random.normal(0, 0.15), 0.2, 0.95)
            disk_demand = np.clip(disk_base * 1.2 + np.random.normal(0, 0.1), 0.1, 0.9)
            network_demand = np.clip(cpu_demand * 0.8 + np.random.normal(0, 0.1), 0.1, 0.9)
            
            return {
                'compute_demand': np.array([cpu_demand, memory_demand, network_demand]),
                'storage_demand': disk_demand,
                'network_demand': network_demand,
                'database_demand': np.array([cpu_demand * 0.7, disk_demand * 0.9])
            }
        else:
            # Fallback to synthetic
            return self._generate_synthetic_ml_training_workload()
    
    def _generate_real_data_analytics_workload(self) -> Dict[str, np.ndarray]:
        """
        Generate data analytics workload based on real Google Cluster data.
        
        Returns:
            Dictionary with data analytics workload demands
        """
        data_point = self._get_real_data_point()
        
        if data_point and self.feature_stats:
            # Use real feature statistics for data analytics patterns
            # Data analytics typically has high disk I/O and moderate CPU
            cpu_base = self.feature_stats.get('cpu_rate', {}).get('mean', 0.4)
            memory_base = self.feature_stats.get('canonical_memory_usage', {}).get('mean', 0.5)
            disk_base = self.feature_stats.get('local_disk_space_usage', {}).get('mean', 0.3)
            io_base = self.feature_stats.get('disk_io_time', {}).get('mean', 0.2)
            
            # Data analytics patterns (moderate CPU, high disk I/O)
            cpu_demand = np.clip(cpu_base * 1.1 + np.random.normal(0, 0.1), 0.1, 0.9)
            memory_demand = np.clip(memory_base * 1.2 + np.random.normal(0, 0.1), 0.1, 0.9)
            disk_demand = np.clip(disk_base * 1.5 + np.random.normal(0, 0.2), 0.1, 0.95)
            network_demand = np.clip(io_base * 1.8 + np.random.normal(0, 0.15), 0.1, 0.9)
            
            return {
                'compute_demand': np.array([cpu_demand, memory_demand, network_demand]),
                'storage_demand': disk_demand,
                'network_demand': network_demand,
                'database_demand': np.array([cpu_demand * 0.8, disk_demand * 1.1])
            }
        else:
            # Fallback to synthetic
            return self._generate_synthetic_data_analytics_workload()

    # Fallback synthetic methods (keeping original implementations)
    def _generate_synthetic_workload(self) -> Dict[str, np.ndarray]:
        """Fallback to synthetic workload generation."""
        return self._generate_synthetic_steady_workload()
    
    def _generate_synthetic_steady_workload(self) -> Dict[str, np.ndarray]:
        """Generate synthetic steady-state workload."""
        base_demand = 0.4
        return {
            'compute_demand': np.array([base_demand, base_demand, base_demand]),
            'storage_demand': base_demand,
            'network_demand': base_demand,
            'database_demand': np.array([base_demand, base_demand])
        }
    
    def _generate_synthetic_burst_workload(self) -> Dict[str, np.ndarray]:
        """Generate synthetic burst workload."""
        burst_cycle = 50
        burst_duration = 10
        
        if (self.step_counter % burst_cycle) < burst_duration:
            base_demand = 0.8 + 0.2 * np.random.random()
        else:
            base_demand = 0.2 + 0.1 * np.random.random()
        
        return {
            'compute_demand': np.array([base_demand, base_demand * 0.8, base_demand * 1.2]),
            'storage_demand': base_demand * 0.6,
            'network_demand': base_demand * 1.5,
            'database_demand': np.array([base_demand * 0.7, base_demand * 0.9])
        }
    
    def _generate_synthetic_cyclical_workload(self) -> Dict[str, np.ndarray]:
        """Generate synthetic cyclical workload."""
        cycle_length = 200
        time_in_cycle = self.step_counter % cycle_length
        cycle_progress = 2 * np.pi * time_in_cycle / cycle_length
        base_demand = 0.3 + 0.4 * np.sin(cycle_progress) + 0.1 * np.random.random()
        base_demand = np.clip(base_demand, 0.1, 0.9)
        
        return {
            'compute_demand': np.array([base_demand, base_demand * 0.9, base_demand * 1.1]),
            'storage_demand': base_demand * 0.7,
            'network_demand': base_demand * 1.3,
            'database_demand': np.array([base_demand * 0.8, base_demand * 0.6])
        }
    
    def _generate_synthetic_web_service_workload(self) -> Dict[str, np.ndarray]:
        """Generate synthetic web service workload."""
        morning_peak = 40 <= self.step_counter % 200 <= 60
        evening_peak = 120 <= self.step_counter % 200 <= 140
        
        if morning_peak or evening_peak:
            base_demand = 0.7 + 0.3 * np.random.random()
        else:
            base_demand = 0.3 + 0.2 * np.random.random()
        
        return {
            'compute_demand': np.array([base_demand, base_demand * 0.9, base_demand * 1.3]),
            'storage_demand': base_demand * 0.5,
            'network_demand': base_demand * 1.4,
            'database_demand': np.array([base_demand * 0.6, base_demand * 0.7])
        }
    
    def _generate_synthetic_batch_processing_workload(self) -> Dict[str, np.ndarray]:
        """Generate synthetic batch processing workload."""
        # Batch processing: high CPU, moderate memory, high disk I/O
        cpu_demand = 0.7 + 0.3 * np.random.random()
        memory_demand = 0.5 + 0.3 * np.random.random()
        disk_demand = 0.6 + 0.4 * np.random.random()
        network_demand = 0.4 + 0.2 * np.random.random()
        
        return {
            'compute_demand': np.array([cpu_demand, memory_demand, network_demand]),
            'storage_demand': disk_demand,
            'network_demand': network_demand,
            'database_demand': np.array([cpu_demand * 0.8, disk_demand * 1.1])
        }
    
    def _generate_synthetic_ml_training_workload(self) -> Dict[str, np.ndarray]:
        """Generate synthetic ML training workload."""
        # ML training: very high CPU and memory usage
        cpu_demand = 0.8 + 0.2 * np.random.random()
        memory_demand = 0.7 + 0.3 * np.random.random()
        disk_demand = 0.4 + 0.3 * np.random.random()
        network_demand = 0.3 + 0.2 * np.random.random()
        
        return {
            'compute_demand': np.array([cpu_demand, memory_demand, network_demand]),
            'storage_demand': disk_demand,
            'network_demand': network_demand,
            'database_demand': np.array([cpu_demand * 0.7, disk_demand * 0.8])
        }
    
    def _generate_synthetic_data_analytics_workload(self) -> Dict[str, np.ndarray]:
        """Generate synthetic data analytics workload."""
        # Data analytics: moderate CPU, high disk I/O
        cpu_demand = 0.5 + 0.3 * np.random.random()
        memory_demand = 0.6 + 0.3 * np.random.random()
        disk_demand = 0.7 + 0.3 * np.random.random()
        network_demand = 0.4 + 0.3 * np.random.random()
        
        return {
            'compute_demand': np.array([cpu_demand, memory_demand, network_demand]),
            'storage_demand': disk_demand,
            'network_demand': network_demand,
            'database_demand': np.array([cpu_demand * 0.8, disk_demand * 1.0])
        }

    def _add_realistic_variations(self, workload: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Add realistic variations to the workload based on real data patterns.
        
        Args:
            workload: Base workload dictionary
            
        Returns:
            Workload with added variations
        """
        # Add small random variations to simulate real-world fluctuations
        for key, value in workload.items():
            if isinstance(value, np.ndarray):
                # Add 5-10% random variation
                variation = np.random.normal(0, 0.05, value.shape)
                workload[key] = np.clip(value + variation, 0.01, 0.99)
            else:
                # Add 5-10% random variation for scalar values
                variation = np.random.normal(0, 0.05)
                workload[key] = np.clip(value + variation, 0.01, 0.99)
        
        return workload

    def visualize_workloads(self, steps: int = 200) -> None:
        """
        Visualize different workload patterns.
        
        Args:
            steps: Number of steps to visualize
        """
        patterns = ['steady', 'burst', 'cyclical']
        fig, axes = plt.subplots(len(patterns), 1, figsize=(12, 8))
        
        if len(patterns) == 1:
            axes = [axes]
        
        for i, pattern in enumerate(patterns):
            self.set_pattern(pattern)
            cpu_demands = []
            memory_demands = []
            storage_demands = []
            network_demands = []
            
            for step in range(steps):
                workload = self.generate_workload()
                cpu_demands.append(workload['compute_demand'][0])
                memory_demands.append(workload['compute_demand'][1])
                storage_demands.append(workload['storage_demand'])
                network_demands.append(workload['network_demand'])
            
            axes[i].plot(cpu_demands, label='CPU', alpha=0.8)
            axes[i].plot(memory_demands, label='Memory', alpha=0.8)
            axes[i].plot(storage_demands, label='Storage', alpha=0.8)
            axes[i].plot(network_demands, label='Network', alpha=0.8)
            axes[i].set_title(f'{pattern.capitalize()} Workload Pattern')
            axes[i].set_ylabel('Resource Demand')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'workload_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Workload patterns visualization saved to {self.viz_dir}/workload_patterns.png")

    def get_available_patterns(self) -> List[str]:
        """
        Get list of available workload patterns.
        
        Returns:
            List of pattern names
        """
        return list(self.patterns.keys())
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data.
        
        Returns:
            Dictionary with data information
        """
        info = {
            'real_data_loaded': self.cluster_data is not None,
            'feature_stats_loaded': self.feature_stats is not None,
            'workload_patterns_loaded': self.workload_patterns is not None,
            'current_mode': self.current_mode,
            'current_pattern': self.current_pattern,
            'available_patterns': self.get_available_patterns()
        }
        
        if self.cluster_data is not None:
            info['total_records'] = len(self.cluster_data)
            info['train_records'] = len(self.train_data)
            info['val_records'] = len(self.val_data)
            info['test_records'] = len(self.test_data)
        
        return info 

    def _generate_feature_statistics(self) -> Dict[str, Any]:
        """
        Generate feature statistics from loaded cluster data.
        
        Returns:
            Dictionary with feature statistics
        """
        if self.cluster_data is None:
            return {}
        
        stats = {}
        
        # Calculate statistics for each feature
        numeric_columns = self.cluster_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in self.cluster_data.columns:
                values = self.cluster_data[col].dropna()
                if len(values) > 0:
                    stats[col] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'normalized': True
                    }
        
        # Ensure we have the required features
        required_features = ['cpu_usage', 'memory_usage', 'disk_io', 'network_io', 
                           'active_machines', 'active_tasks', 'cpu_memory_ratio', 'io_ratio']
        
        for feature in required_features:
            if feature not in stats:
                # Use default statistics if feature not found
                stats[feature] = {
                    'mean': 0.5,
                    'std': 0.2,
                    'min': 0.0,
                    'max': 1.0,
                    'normalized': True
                }
        
        return stats
    
    def _generate_workload_patterns(self) -> Dict[str, Any]:
        """
        Generate workload patterns from loaded cluster data.
        
        Returns:
            Dictionary with workload patterns
        """
        if self.cluster_data is None:
            return {}
        
        patterns = {}
        
        # Calculate base statistics from the data
        cpu_mean = self.cluster_data['cpu_usage'].mean() if 'cpu_usage' in self.cluster_data.columns else 0.5
        memory_mean = self.cluster_data['memory_usage'].mean() if 'memory_usage' in self.cluster_data.columns else 0.5
        disk_mean = self.cluster_data['disk_io'].mean() if 'disk_io' in self.cluster_data.columns else 0.5
        network_mean = self.cluster_data['network_io'].mean() if 'network_io' in self.cluster_data.columns else 0.5
        
        cpu_std = self.cluster_data['cpu_usage'].std() if 'cpu_usage' in self.cluster_data.columns else 0.1
        memory_std = self.cluster_data['memory_usage'].std() if 'memory_usage' in self.cluster_data.columns else 0.1
        disk_std = self.cluster_data['disk_io'].std() if 'disk_io' in self.cluster_data.columns else 0.1
        network_std = self.cluster_data['network_io'].std() if 'network_io' in self.cluster_data.columns else 0.1
        
        # Steady pattern
        patterns['steady'] = {
            'type': 'steady',
            'cpu_mean': cpu_mean,
            'cpu_std': cpu_std * 0.5,  # Lower variation for steady
            'memory_mean': memory_mean,
            'memory_std': memory_std * 0.5,
            'disk_mean': disk_mean,
            'disk_std': disk_std * 0.5,
            'network_mean': network_mean,
            'network_std': network_std * 0.5
        }
        
        # Burst pattern
        patterns['burst'] = {
            'type': 'burst',
            'cpu_mean': cpu_mean * 0.7,  # Lower base for burst
            'cpu_std': cpu_std * 1.5,    # Higher variation
            'memory_mean': memory_mean * 0.7,
            'memory_std': memory_std * 1.5,
            'disk_mean': disk_mean * 0.7,
            'disk_std': disk_std * 1.5,
            'network_mean': network_mean * 0.7,
            'network_std': network_std * 1.5,
            'burst_probability': 0.1,
            'burst_magnitude': 3.0,
            'burst_duration': [5, 15]
        }
        
        # Cyclical pattern
        patterns['cyclical'] = {
            'type': 'cyclical',
            'cpu_mean': cpu_mean,
            'cpu_amplitude': cpu_std * 2.0,
            'memory_mean': memory_mean,
            'memory_amplitude': memory_std * 1.8,
            'disk_mean': disk_mean,
            'disk_amplitude': disk_std * 1.5,
            'network_mean': network_mean,
            'network_amplitude': network_std * 1.8,
            'period': 24,
            'phase_shift': {'cpu': 0, 'memory': 2, 'disk': 4, 'network': 6}
        }
        
        return patterns 