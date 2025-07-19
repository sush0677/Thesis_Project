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
        Load real Google Cluster data from processed files.
        
        Returns:
            DataFrame with real cluster data or None if not available
        """
        try:
            # Try to load processed data
            processed_file = self.data_dir / 'processed_data.parquet'
            if processed_file.exists():
                data = pd.read_parquet(processed_file)
                logger.info(f"Loaded real Google Cluster data: {len(data)} records")
                return data
            else:
                logger.warning("Processed Google Cluster data not found")
                return None
        except Exception as e:
            logger.error(f"Error loading cluster data: {e}")
            return None
    
    def _load_feature_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Load feature statistics from real Google Cluster data.
        
        Returns:
            Dictionary with feature statistics or None if not available
        """
        try:
            stats_file = self.data_dir / 'feature_statistics.json'
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                logger.info("Loaded real Google Cluster feature statistics")
                return stats
            else:
                logger.warning("Feature statistics not found")
                return None
        except Exception as e:
            logger.error(f"Error loading feature statistics: {e}")
            return None
    
    def _load_workload_patterns(self) -> Optional[Dict[str, Any]]:
        """
        Load real workload patterns from Google Cluster data.
        
        Returns:
            Dictionary with workload patterns or None if not available
        """
        try:
            patterns_file = self.data_dir / 'workload_patterns.json'
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns = json.load(f)
                logger.info("Loaded real Google Cluster workload patterns")
                return patterns
            else:
                logger.warning("Workload patterns not found")
                return None
        except Exception as e:
            logger.error(f"Error loading workload patterns: {e}")
            return None
    
    def _split_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split the real data into training, validation, and test sets.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if self.cluster_data is None:
            logger.warning("No real data available, using synthetic data")
            return [], [], []
        
        # Convert DataFrame to list of dictionaries
        data_list = self.cluster_data.to_dict('records')
        
        # Calculate split indices
        n_total = len(data_list)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        # Split the data
        train_data = data_list[:n_train]
        val_data = data_list[n_train:n_train + n_val]
        test_data = data_list[n_train + n_val:]
        
        logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
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
        
        # Use real data if available
        if self.cluster_data is not None and self.workload_patterns is not None:
            workload = self.patterns[self.current_pattern]()
        else:
            # Fallback to synthetic patterns
            logger.warning("Using synthetic patterns - real data not available")
            workload = self._generate_synthetic_workload()
        
        return workload
    
    def _get_real_data_point(self) -> Optional[Dict[str, float]]:
        """
        Get a real data point from the current mode.
        
        Returns:
            Dictionary with real data features or None if not available
        """
        if self.current_mode == 'train':
            data = self.train_data
        elif self.current_mode == 'val':
            data = self.val_data
        else:  # test
            data = self.test_data
        
        if not data:
            return None
        
        # Get data point and advance index
        data_point = data[self.data_index % len(data)]
        self.data_index += 1
        
        return data_point
    
    def _generate_real_steady_workload(self) -> Dict[str, np.ndarray]:
        """
        Generate steady workload based on real Google Cluster data.
        
        Returns:
            Dictionary with steady workload demands
        """
        data_point = self._get_real_data_point()
        
        if data_point and self.workload_patterns and 'steady' in self.workload_patterns:
            # Use real steady pattern
            pattern = self.workload_patterns['steady']
            
            # Add realistic noise based on real data
            cpu_noise = np.random.normal(0, pattern.get('cpu_std', 0.05))
            memory_noise = np.random.normal(0, pattern.get('memory_std', 0.05))
            disk_noise = np.random.normal(0, pattern.get('disk_std', 0.05))
            network_noise = np.random.normal(0, pattern.get('network_std', 0.05))
            
            cpu_demand = np.clip(pattern.get('cpu_mean', 0.5) + cpu_noise, 0.1, 0.9)
            memory_demand = np.clip(pattern.get('memory_mean', 0.5) + memory_noise, 0.1, 0.9)
            disk_demand = np.clip(pattern.get('disk_mean', 0.5) + disk_noise, 0.1, 0.9)
            network_demand = np.clip(pattern.get('network_mean', 0.5) + network_noise, 0.1, 0.9)
            
            return {
                'compute_demand': np.array([cpu_demand, memory_demand, network_demand]),
                'storage_demand': disk_demand,
                'network_demand': network_demand,
                'database_demand': np.array([cpu_demand * 0.8, disk_demand * 0.9])
            }
        else:
            # Fallback to synthetic
            return self._generate_synthetic_steady_workload()
    
    def _generate_real_burst_workload(self) -> Dict[str, np.ndarray]:
        """
        Generate burst workload based on real Google Cluster data.
        
        Returns:
            Dictionary with burst workload demands
        """
        data_point = self._get_real_data_point()
        
        if data_point and self.workload_patterns and 'burst' in self.workload_patterns:
            # Use real burst pattern
            pattern = self.workload_patterns['burst']
            
            # Determine if we're in a burst period
            burst_prob = pattern.get('burst_probability', 0.1)
            burst_magnitude = pattern.get('burst_magnitude', 3.0)
            burst_duration = pattern.get('burst_duration', [5, 15])
            
            # Check if current step is in burst period
            burst_cycle = 50
            in_burst = (self.step_counter % burst_cycle) < np.random.randint(*burst_duration)
            
            if in_burst and np.random.random() < burst_prob:
                # Burst period - high demand
                multiplier = burst_magnitude
            else:
                # Normal period - low demand
                multiplier = 1.0
            
            # Base demands from real pattern
            cpu_base = pattern.get('cpu_mean', 0.35) * multiplier
            memory_base = pattern.get('memory_mean', 0.35) * multiplier
            disk_base = pattern.get('disk_mean', 0.35) * multiplier
            network_base = pattern.get('network_mean', 0.35) * multiplier
            
            # Add realistic noise
            cpu_demand = np.clip(cpu_base + np.random.normal(0, pattern.get('cpu_std', 0.08)), 0.1, 0.95)
            memory_demand = np.clip(memory_base + np.random.normal(0, pattern.get('memory_std', 0.08)), 0.1, 0.95)
            disk_demand = np.clip(disk_base + np.random.normal(0, pattern.get('disk_std', 0.08)), 0.1, 0.95)
            network_demand = np.clip(network_base + np.random.normal(0, pattern.get('network_std', 0.08)), 0.1, 0.95)
            
            return {
                'compute_demand': np.array([cpu_demand, memory_demand, network_demand]),
                'storage_demand': disk_demand,
                'network_demand': network_demand,
                'database_demand': np.array([cpu_demand * 0.7, disk_demand * 0.9])
            }
        else:
            # Fallback to synthetic
            return self._generate_synthetic_burst_workload()
    
    def _generate_real_cyclical_workload(self) -> Dict[str, np.ndarray]:
        """
        Generate cyclical workload based on real Google Cluster data.
        
        Returns:
            Dictionary with cyclical workload demands
        """
        data_point = self._get_real_data_point()
        
        if data_point and self.workload_patterns and 'cyclical' in self.workload_patterns:
            # Use real cyclical pattern
            pattern = self.workload_patterns['cyclical']
            
            # Get cyclical parameters
            period = pattern.get('period', 24)  # hours
            cpu_amplitude = pattern.get('cpu_amplitude', 0.4)
            memory_amplitude = pattern.get('memory_amplitude', 0.3)
            disk_amplitude = pattern.get('disk_amplitude', 0.3)
            network_amplitude = pattern.get('network_amplitude', 0.4)
            
            # Calculate cyclical component
            cycle_progress = 2 * np.pi * (self.step_counter % (period * 8)) / (period * 8)
            
            # Base demands with cyclical variation
            cpu_base = pattern.get('cpu_mean', 0.5)
            memory_base = pattern.get('memory_mean', 0.5)
            disk_base = pattern.get('disk_mean', 0.5)
            network_base = pattern.get('network_mean', 0.5)
            
            # Add cyclical variation
            cpu_demand = np.clip(cpu_base + cpu_amplitude * np.sin(cycle_progress), 0.1, 0.9)
            memory_demand = np.clip(memory_base + memory_amplitude * np.sin(cycle_progress + np.pi/6), 0.1, 0.9)
            disk_demand = np.clip(disk_base + disk_amplitude * np.sin(cycle_progress + np.pi/3), 0.1, 0.9)
            network_demand = np.clip(network_base + network_amplitude * np.sin(cycle_progress + np.pi/2), 0.1, 0.9)
            
            return {
                'compute_demand': np.array([cpu_demand, memory_demand, network_demand]),
                'storage_demand': disk_demand,
                'network_demand': network_demand,
                'database_demand': np.array([cpu_demand * 0.8, disk_demand * 0.6])
            }
        else:
            # Fallback to synthetic
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