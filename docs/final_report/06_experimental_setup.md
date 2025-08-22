# Experimental Setup

## 6.1 Environment Configuration

### 6.1.1 Development Environment Setup

The experimental environment was configured to provide a realistic simulation of GCP resource management scenarios while maintaining reproducibility and control over experimental variables.

**Local Development Environment**
- **Operating System**: Windows 10/11 with WSL2 support
- **Python Environment**: Python 3.8+ with virtual environment isolation
- **Hardware Specifications**:
  - CPU: Intel i7-10700K or equivalent (8 cores, 16 threads)
  - RAM: 32GB DDR4 for large dataset handling
  - GPU: NVIDIA RTX 3070 or equivalent for accelerated training
  - Storage: 1TB NVMe SSD for fast data access

**Cloud Development Environment**
- **Google Cloud Platform**: Primary cloud environment
- **Compute Engine**: Custom machine types for training
  - Machine Type: n1-standard-8 (8 vCPUs, 30GB memory)
  - GPU: NVIDIA T4 or V100 for accelerated training
  - Storage: 100GB boot disk + 500GB data disk
  - Network: Premium tier for low-latency connections

### 6.1.2 Software Environment Configuration

**Python Environment Setup**
```bash
# Create virtual environment
python -m venv marl_gcp_env
source marl_gcp_env/bin/activate  # Linux/Mac
marl_gcp_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
```

**Core Dependencies Installation**
```python
# requirements.txt contents
torch>=1.9.0          # Deep learning framework
numpy>=1.21.0         # Numerical computing
pandas>=1.3.0         # Data manipulation
matplotlib>=3.4.0     # Static plotting
plotly>=5.0.0         # Interactive visualization
seaborn>=0.11.0       # Statistical visualization
pyyaml>=5.4.0         # Configuration files
tqdm>=4.62.0          # Progress bars
scikit-learn>=1.0.0   # Machine learning utilities
```

**Environment Variables Configuration**
```bash
# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export MARL_GCP_DATA_DIR="$(pwd)/data"
export MARL_GCP_LOG_DIR="$(pwd)/logs"
export MARL_GCP_MODEL_DIR="$(pwd)/models"
```

### 6.1.3 Data Environment Configuration

**Google Cluster Data Integration**
- **Data Source**: Authentic Google Cluster trace data
- **Data Format**: Processed CSV files with standardized features
- **Data Location**: `data/` directory with organized subdirectories
- **Data Validation**: Automated data integrity checks

**Data Directory Structure**
```
data/
├── raw/                    # Raw Google Cluster data
├── processed/              # Preprocessed and feature-engineered data
├── train/                  # Training dataset (70%)
├── validation/             # Validation dataset (15%)
├── test/                   # Test dataset (15%)
└── metadata/               # Data schema and statistics
```

**Data Preprocessing Pipeline**
```python
# Data preprocessing workflow
def preprocess_google_cluster_data():
    # 1. Load raw data
    raw_data = load_raw_google_cluster_data()
    
    # 2. Clean and validate
    cleaned_data = clean_and_validate_data(raw_data)
    
    # 3. Feature engineering
    engineered_data = engineer_features(cleaned_data)
    
    # 4. Data splitting
    train_data, val_data, test_data = split_data(engineered_data)
    
    # 5. Normalization
    normalized_data = normalize_data(train_data, val_data, test_data)
    
    return normalized_data
```

## 6.2 Experiment Configuration

### 6.2.1 Experiment Design Framework

The experimental design implements a comprehensive framework for evaluating the MARL system's performance across multiple dimensions and scenarios.

**Experimental Variables**
- **Independent Variables**:
  - Workload patterns (bursty, cyclical, steady)
  - Resource constraints (CPU, memory, storage, network limits)
  - Training episodes (100 episodes per scenario)
  - Agent configurations (learning rates, network architectures)

- **Dependent Variables**:
  - Resource utilization efficiency
  - Cost optimization performance
  - System response time and throughput
  - Sustainability metrics and environmental impact

**Control Variables**
- **Environment Configuration**: Consistent GCP simulation parameters
- **Data Sources**: Same Google Cluster dataset across all experiments
- **Hardware Resources**: Consistent computational resources for training
- **Random Seeds**: Fixed seeds for reproducible results

### 6.2.2 Scenario Configuration

**Scenario 1: Resource Efficiency Testing**
```python
# Resource efficiency scenario configuration
resource_efficiency_config = {
    'objective': 'maximize_resource_utilization',
    'workload_pattern': 'steady',
    'episodes': 100,
    'metrics': ['cpu_utilization', 'memory_efficiency', 'storage_optimization'],
    'constraints': {
        'max_cpu': 0.8,
        'max_memory': 0.85,
        'max_storage': 0.9
    }
}
```

**Scenario 2: Cost Optimization Testing**
```python
# Cost optimization scenario configuration
cost_optimization_config = {
    'objective': 'minimize_operational_costs',
    'workload_pattern': 'cyclical',
    'episodes': 100,
    'metrics': ['cost_per_transaction', 'resource_cost_efficiency', 'budget_adherence'],
    'constraints': {
        'monthly_budget': 1000,  # USD
        'cost_performance_ratio': 0.1
    }
}
```

**Scenario 3: Performance Testing**
```python
# Performance testing scenario configuration
performance_config = {
    'objective': 'maximize_system_performance',
    'workload_pattern': 'bursty',
    'episodes': 100,
    'metrics': ['response_time', 'throughput', 'scalability'],
    'constraints': {
        'min_response_time': 100,  # ms
        'min_throughput': 1000,   # requests/second
        'max_latency': 500        # ms
    }
}
```

**Scenario 4: Sustainability Testing**
```python
# Sustainability testing scenario configuration
sustainability_config = {
    'objective': 'optimize_environmental_impact',
    'workload_pattern': 'mixed',
    'episodes': 100,
    'metrics': ['carbon_footprint', 'energy_efficiency', 'renewable_usage'],
    'constraints': {
        'max_carbon_intensity': 0.5,  # kg CO2/kWh
        'min_renewable_percentage': 0.7
    }
}
```

### 6.2.3 Baseline System Configuration

**Traditional Resource Management Baselines**

**Static Provisioning Baseline**
```python
class StaticProvisioningBaseline:
    def __init__(self, config):
        self.resource_allocation = {
            'cpu': 0.7,      # 70% CPU allocation
            'memory': 0.75,  # 75% memory allocation
            'storage': 0.8,  # 80% storage allocation
            'network': 0.6   # 60% network allocation
        }
    
    def allocate_resources(self, workload_demand):
        # Fixed allocation regardless of demand
        return self.resource_allocation
```

**Rule-Based Provisioning Baseline**
```python
class RuleBasedProvisioningBaseline:
    def __init__(self, config):
        self.thresholds = {
            'cpu_high': 0.8,    # Scale up when CPU > 80%
            'cpu_low': 0.3,     # Scale down when CPU < 30%
            'memory_high': 0.85,
            'memory_low': 0.4
        }
    
    def allocate_resources(self, current_utilization):
        # Simple threshold-based scaling rules
        allocation = {}
        for resource, utilization in current_utilization.items():
            if utilization > self.thresholds[f'{resource}_high']:
                allocation[resource] = min(1.0, utilization * 1.2)
            elif utilization < self.thresholds[f'{resource}_low']:
                allocation[resource] = max(0.1, utilization * 0.8)
            else:
                allocation[resource] = utilization
        return allocation
```

**Reactive Provisioning Baseline**
```python
class ReactiveProvisioningBaseline:
    def __init__(self, config):
        self.reaction_delay = 5  # 5 time steps delay
        self.scaling_factor = 1.1
    
    def allocate_resources(self, current_utilization, historical_utilization):
        # React to current utilization with delay
        allocation = {}
        for resource, utilization in current_utilization.items():
            if len(historical_utilization[resource]) >= self.reaction_delay:
                trend = self.calculate_trend(historical_utilization[resource])
                if trend > 0.1:  # Increasing trend
                    allocation[resource] = min(1.0, utilization * self.scaling_factor)
                elif trend < -0.1:  # Decreasing trend
                    allocation[resource] = max(0.1, utilization / self.scaling_factor)
                else:
                    allocation[resource] = utilization
            else:
                allocation[resource] = utilization
        return allocation
```

### 6.2.4 MARL System Configuration

**Agent Configuration Parameters**
```python
# Agent configuration for all agent types
agent_config = {
    'learning_rate': 3e-4,
    'batch_size': 64,
    'gamma': 0.99,           # Discount factor
    'tau': 0.005,            # Soft update rate
    'noise_std': 0.1,        # Action noise for exploration
    'buffer_capacity': 1000000,  # Experience buffer size
    'update_frequency': 2,   # Policy update frequency
    'target_update_frequency': 2  # Target network update frequency
}
```

**Neural Network Architecture Configuration**
```python
# Neural network architecture configuration
network_config = {
    'actor': {
        'hidden_layers': [256, 128, 64],
        'activation': 'relu',
        'output_activation': 'tanh',
        'dropout': 0.1
    },
    'critic': {
        'hidden_layers': [256, 128, 64],
        'activation': 'relu',
        'output_activation': 'linear',
        'dropout': 0.1
    }
}
```

**Training Configuration**
```python
# Training configuration parameters
training_config = {
    'episodes': 100,
    'max_steps_per_episode': 1000,
    'evaluation_frequency': 10,
    'save_frequency': 25,
    'early_stopping_patience': 20,
    'learning_rate_schedule': 'constant',
    'exploration_schedule': 'linear_decay'
}
```

## 6.3 Tools and Technologies

### 6.3.1 Development and Testing Tools

**Code Development Tools**
- **IDE**: Visual Studio Code with Python extensions
- **Version Control**: Git with GitHub for code management
- **Code Quality**: Pylint, Black, and isort for code formatting
- **Testing Framework**: pytest for unit and integration testing

**Development Workflow Tools**
```bash
# Code quality checks
pylint src/marl_gcp/
black src/marl_gcp/
isort src/marl_gcp/

# Running tests
pytest tests/ -v
pytest tests/ --cov=src/marl_gcp/ --cov-report=html

# Type checking
mypy src/marl_gcp/
```

**Performance Profiling Tools**
- **CPU Profiling**: cProfile and line_profiler
- **Memory Profiling**: memory_profiler and tracemalloc
- **GPU Profiling**: PyTorch profiler for CUDA operations
- **Network Profiling**: Custom network latency measurement tools

### 6.3.2 Simulation and Environment Tools

**GCP Environment Simulation**
```python
# GCP environment simulation configuration
class GCPEnvironmentSimulator:
    def __init__(self, config):
        self.workload_generator = WorkloadGenerator(config)
        self.resource_simulator = ResourceSimulator(config)
        self.pricing_simulator = PricingSimulator(config)
        self.constraint_enforcer = ConstraintEnforcer(config)
        self.monitoring = EnvironmentMonitoring(config)
    
    def simulate_episode(self, agent_actions):
        # Simulate resource allocation and workload execution
        resource_states = self.resource_simulator.update(agent_actions)
        workload_performance = self.workload_generator.execute(resource_states)
        costs = self.pricing_simulator.calculate_costs(resource_states)
        constraints = self.constraint_enforcer.check_violations(resource_states)
        
        return resource_states, workload_performance, costs, constraints
```

**Workload Generation Tools**
```python
# Workload pattern generation
class WorkloadPatternGenerator:
    def __init__(self, config):
        self.patterns = {
            'bursty': BurstyWorkloadGenerator(config),
            'cyclical': CyclicalWorkloadGenerator(config),
            'steady': SteadyWorkloadGenerator(config)
        }
    
    def generate_workload(self, pattern_type, duration, intensity):
        return self.patterns[pattern_type].generate(duration, intensity)
```

### 6.3.3 Monitoring and Visualization Tools

**Real-time Monitoring System**
```python
# Real-time monitoring configuration
class MonitoringSystem:
    def __init__(self, config):
        self.metrics_collector = MetricsCollector(config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        self.alert_system = AlertSystem(config)
        self.dashboard_updater = DashboardUpdater(config)
    
    def monitor_system(self, system_state):
        # Collect and analyze system metrics
        metrics = self.metrics_collector.collect(system_state)
        analysis = self.performance_analyzer.analyze(metrics)
        
        # Update dashboard and check for alerts
        self.dashboard_updater.update(analysis)
        self.alert_system.check_alerts(analysis)
        
        return analysis
```

**Visualization and Dashboard Tools**
```python
# Dashboard configuration
dashboard_config = {
    'update_frequency': 1.0,  # Update every second
    'charts': {
        'resource_utilization': True,
        'cost_tracking': True,
        'performance_metrics': True,
        'training_progress': True,
        'sustainability_metrics': True
    },
    'export_formats': ['png', 'pdf', 'html'],
    'interactive_features': True
}
```

**Data Analysis and Reporting Tools**
- **Statistical Analysis**: scipy.stats for statistical testing
- **Data Visualization**: matplotlib, seaborn, and plotly for charts
- **Report Generation**: Custom report generators for experiment results
- **Performance Benchmarking**: Automated benchmarking tools

### 6.3.4 Deployment and Production Tools

**Containerization and Orchestration**
```dockerfile
# Docker configuration for deployment
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

EXPOSE 8000
CMD ["python", "src/main.py"]
```

**Cloud Deployment Tools**
```yaml
# Kubernetes deployment configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: marl-gcp-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: marl-gcp
  template:
    metadata:
      labels:
        app: marl-gcp
    spec:
      containers:
      - name: marl-gcp
        image: marl-gcp:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

**CI/CD Pipeline Configuration**
```yaml
# GitHub Actions workflow
name: MARL-GCP CI/CD
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest tests/ -v
    - name: Run linting
      run: |
        pylint src/marl_gcp/
```

This comprehensive experimental setup provides a robust foundation for evaluating the MARL system's performance across multiple dimensions, ensuring reliable and reproducible results for cloud resource management optimization.
