# Methodology

## 4.1 Data Collection and Preprocessing

### 4.1.1 Google Cluster Data Integration

The project leverages authentic Google Cluster data to ensure realistic evaluation of the MARL system. The data collection process involved:

**Data Sources**
- **Google Cluster Trace Data**: Real workload traces from Google's production clusters
- **Resource Utilization Metrics**: CPU, memory, disk, and network usage patterns
- **Task Scheduling Information**: Job submission, execution, and completion data
- **Machine Configuration Data**: Hardware specifications and resource capacities

**Data Preprocessing Pipeline**
```
Raw Google Cluster Data → Data Cleaning → Feature Engineering → Data Splitting → Normalization
```

**Key Features Extracted**
- `cpu_rate`: CPU utilization patterns (mean: 0.4, std: 0.2)
- `canonical_memory_usage`: Memory usage patterns (mean: 0.5, std: 0.25)
- `local_disk_space_usage`: Disk usage patterns (mean: 0.3, std: 0.15)
- `disk_io_time`: I/O patterns (mean: 0.2, std: 0.1)
- `active_machines`: Machine count patterns (30-70 range)
- `active_tasks`: Task count patterns (60-140 range)
- `cpu_memory_ratio`: CPU/memory ratios (0-5 range)
- `io_ratio`: I/O ratios (0-3 range)

**Data Split Strategy**
- **Training Set**: 70% (103 records) - Used for agent training
- **Validation Set**: 15% (22 records) - Used for hyperparameter tuning
- **Test Set**: 15% (23 records) - Used for final evaluation

### 4.1.2 Workload Pattern Generation

Three distinct workload patterns were implemented to test system adaptability:

**Bursty Workloads**
- Sudden spikes in resource demand
- Tests system's ability to scale up rapidly
- Simulates real-world traffic spikes and viral content scenarios

**Cyclical Workloads**
- Periodic variations in resource requirements
- Tests system's ability to predict and prepare for regular patterns
- Simulates business hour variations and seasonal trends

**Steady Workloads**
- Consistent resource demand over time
- Tests system's ability to maintain optimal resource allocation
- Simulates background services and batch processing jobs

## 4.2 Experimental Design

### 4.2.1 Multi-Agent System Architecture

The experimental design implements a sophisticated multi-agent system with specialized agents for different resource types:

**Agent Specialization**
- **Compute Agent**: Manages CPU and memory allocation decisions
- **Storage Agent**: Handles disk space and I/O optimization
- **Network Agent**: Controls bandwidth allocation and network routing
- **Database Agent**: Manages database resource provisioning

**Centralized Training, Decentralized Execution**
- Training phase: Centralized coordination for optimal learning
- Execution phase: Independent decision-making for scalability
- Experience sharing through centralized buffer for coordinated learning

### 4.2.2 Baseline Comparison Framework

Traditional resource management methods were implemented as benchmarks:

**Static Provisioning**
- Fixed resource allocation based on peak demand estimates
- No adaptation to changing workload patterns
- Serves as lower-bound performance baseline

**Rule-Based Provisioning**
- Simple heuristics for resource scaling
- Basic threshold-based scaling rules
- Represents current industry practices

**Reactive Provisioning**
- Resource adjustment based on current utilization
- Limited predictive capabilities
- Common in many cloud environments

### 4.2.3 Experimental Scenarios

**Scenario 1: Resource Efficiency Testing**
- Objective: Maximize resource utilization while minimizing waste
- Metrics: CPU utilization, memory efficiency, storage optimization
- Duration: 100 episodes per scenario

**Scenario 2: Cost Optimization Testing**
- Objective: Minimize operational costs while maintaining performance
- Metrics: Cost per transaction, resource cost efficiency, budget adherence
- Duration: 100 episodes per scenario

**Scenario 3: Performance Testing**
- Objective: Maximize system performance under varying loads
- Metrics: Response time, throughput, scalability
- Duration: 100 episodes per scenario

**Scenario 4: Sustainability Testing**
- Objective: Optimize environmental impact while maintaining efficiency
- Metrics: Carbon footprint, energy efficiency, renewable energy usage
- Duration: 100 episodes per scenario

## 4.3 Implementation Details

### 4.3.1 MARL Algorithm Selection

**TD3 (Twin Delayed Deep Deterministic Policy Gradient)**
- **Rationale**: Selected for its superior performance in continuous action spaces
- **Advantages**: 
  - Twin critics reduce overestimation bias
  - Delayed policy updates improve training stability
  - Noise injection enhances exploration
- **Implementation**: Custom PyTorch implementation with GCP-specific adaptations

**Algorithm Architecture**
```python
class TD3Agent:
    def __init__(self, state_dim, action_dim, agent_type):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic1 = CriticNetwork(state_dim, action_dim)
        self.critic2 = CriticNetwork(state_dim, action_dim)
        self.target_actor = ActorNetwork(state_dim, action_dim)
        self.target_critic1 = CriticNetwork(state_dim, action_dim)
        self.target_critic2 = CriticNetwork(state_dim, action_dim)
```

### 4.3.2 Neural Network Architecture

**Actor Networks (Policy)**
- **Input Layer**: State dimensions specific to each agent type
- **Hidden Layers**: 3 fully connected layers (256, 128, 64 neurons)
- **Activation**: ReLU for hidden layers, Tanh for output
- **Output**: Continuous actions in [-1, 1] range

**Critic Networks (Value Function)**
- **Input Layer**: State and action concatenation
- **Hidden Layers**: 3 fully connected layers (256, 128, 64 neurons)
- **Activation**: ReLU for hidden layers, linear for output
- **Output**: Q-value estimation for state-action pairs

### 4.3.3 Training Configuration

**Hyperparameters**
- **Learning Rate**: 3e-4 for both actor and critic networks
- **Batch Size**: 64 experiences per training step
- **Discount Factor**: 0.99 for future reward consideration
- **Soft Update Rate**: 0.005 for target network updates
- **Noise Standard Deviation**: 0.1 for action exploration

**Training Process**
1. **Episode Initialization**: Reset environment and agent states
2. **Action Selection**: Agents select actions using current policies
3. **Environment Step**: Execute actions and observe rewards
4. **Experience Storage**: Store transitions in shared experience buffer
5. **Network Updates**: Update agent networks using sampled batches
6. **Target Updates**: Soft update of target networks
7. **Iteration**: Repeat until episode completion

### 4.3.4 Reward Engineering

**Multi-Objective Reward Structure**
- **Cost Efficiency (25% weight)**: Resource cost optimization
- **Performance (35% weight)**: Latency, throughput, and scalability
- **Reliability (20% weight)**: System availability and stability
- **Sustainability (15% weight)**: Environmental impact and energy efficiency
- **Utilization (5% weight)**: Optimal resource utilization

**Reward Normalization**
- **Z-Score Normalization**: Standardizes rewards across different scales
- **Historical Statistics**: Maintains running statistics for normalization
- **Adaptive Updates**: Statistics update as training progresses
- **Component-Specific**: Different normalization for each reward type

**Constraint Management**
- **Budget Violations**: 2x penalty for exceeding budget limits
- **Resource Violations**: Fixed penalties for exceeding resource limits
- **Violation Tracking**: Comprehensive history for analysis
- **Penalty Calculation**: Proportional penalties based on violation severity

### 4.3.5 System Integration

**Environment Simulation**
- **GCP Environment**: Realistic cloud resource simulation
- **Resource Constraints**: Hardware limitations and service quotas
- **Pricing Models**: Actual GCP pricing for cost calculations
- **Provisioning Delays**: Realistic deployment and scaling times

**Monitoring and Logging**
- **Performance Metrics**: Real-time resource utilization tracking
- **Cost Tracking**: Detailed cost analysis and budget monitoring
- **Training Progress**: Learning curves and convergence metrics
- **System Health**: Agent performance and coordination status

**Visualization and Analysis**
- **Real-time Dashboards**: Live performance monitoring
- **Training Visualization**: Learning progress and agent behavior
- **Performance Comparison**: Baseline vs. MARL system analysis
- **Cost Analysis**: Detailed cost breakdown and optimization insights

This comprehensive methodology ensures rigorous evaluation of the MARL system's effectiveness in real-world cloud resource management scenarios.
