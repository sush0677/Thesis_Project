# Workload Patterns in MARL-GCP System

## Overview

Your MARL-GCP system supports **7 different workload patterns**, each representing different types of real-world cloud computing scenarios. Changing the workload pattern significantly affects how the system behaves and how the agents learn to optimize resource allocation.

## Available Workload Patterns

### 1. **Steady Pattern** (`steady`)
**What it does:**
- Generates consistent, predictable workload demands
- Uses real Google Cluster data points with minimal variation (±2% noise)
- Resource demands stay relatively constant over time

**Characteristics:**
- CPU: 0.1-0.9 (based on real data + small noise)
- Memory: 0.1-0.9 (based on real data + small noise)
- Disk: 0.1-0.9 (based on real data + small noise)
- Network: 0.1-0.9 (based on real data + small noise)

**Best for:**
- Testing baseline performance
- Stable production environments
- Learning basic resource allocation strategies

---

### 2. **Burst Pattern** (`burst`)
**What it does:**
- Creates sudden spikes in resource demand
- Burst periods occur every 50 steps with 10% probability
- Burst magnitude: 2.5x normal demand
- Burst duration: 5-15 steps

**Characteristics:**
- Normal periods: Low demand (base real data)
- Burst periods: High demand (2.5x multiplier)
- Unpredictable timing and duration

**Best for:**
- Testing scalability under pressure
- Learning to handle traffic spikes
- E-commerce or event-driven applications

---

### 3. **Cyclical Pattern** (`cyclical`)
**What it does:**
- Creates periodic variations in resource demand
- 24-step cycle with different phases for each resource
- CPU, memory, disk, and network have offset phases

**Characteristics:**
- Period: 24 steps per complete cycle
- Amplitude: ±20% variation from base demand
- Phase offsets: CPU (0°), Memory (90°), Disk (180°), Network (270°)

**Best for:**
- Simulating daily/weekly business cycles
- Learning time-based optimization
- Applications with predictable peak hours

---

### 4. **Web Service Pattern** (`web_service`)
**What it does:**
- Simulates typical web application workloads
- Higher demand during peak hours (8-10 AM, 6-8 PM)
- Emphasizes CPU and network usage

**Characteristics:**
- Peak hours: 1.5x multiplier
- Off-peak hours: 0.8x multiplier
- High network demand relative to CPU
- Moderate disk usage

**Best for:**
- Web applications and APIs
- E-commerce platforms
- Social media applications

---

### 5. **Batch Processing Pattern** (`batch_processing`)
**What it does:**
- Simulates data processing jobs
- High CPU and memory usage
- Lower network requirements
- Consistent high demand during job execution

**Characteristics:**
- High CPU utilization (1.3-1.7x base)
- High memory usage (1.2-1.6x base)
- Moderate disk I/O
- Lower network requirements

**Best for:**
- Data processing pipelines
- ETL jobs
- Scientific computing
- Background task processing

---

### 6. **ML Training Pattern** (`ml_training`)
**What it does:**
- Simulates machine learning training workloads
- Very high CPU and memory demands
- GPU-like resource patterns
- Sustained high usage

**Characteristics:**
- CPU: 1.8x base demand (very high)
- Memory: 1.6x base demand (very high)
- Disk: 1.2x base demand (moderate)
- Network: 0.8x CPU demand (lower)

**Best for:**
- Machine learning model training
- Deep learning workloads
- AI/ML research environments

---

### 7. **Data Analytics Pattern** (`data_analytics`)
**What it does:**
- Simulates data analytics and reporting workloads
- High disk I/O and moderate CPU
- Periodic high demand for data processing

**Characteristics:**
- CPU: 1.1x base demand (moderate)
- Memory: 1.2x base demand (moderate)
- Disk: 1.5x base demand (high I/O)
- Network: 1.8x I/O base (high)

**Best for:**
- Business intelligence applications
- Data warehousing
- Reporting systems
- Big data processing

## Impact of Changing Workload Patterns

### **On Agent Learning:**
1. **Different Strategies**: Each pattern requires different optimization strategies
2. **Adaptation Speed**: Agents must learn to adapt to changing patterns
3. **Resource Prioritization**: Different patterns prioritize different resources

### **On System Performance:**
1. **Resource Utilization**: Different patterns have different resource bottlenecks
2. **Cost Optimization**: Some patterns are more cost-sensitive than others
3. **Scalability Requirements**: Burst patterns require more aggressive scaling

### **On Training Effectiveness:**
1. **Learning Diversity**: Multiple patterns help agents generalize better
2. **Robustness**: Agents trained on varied patterns are more robust
3. **Real-world Applicability**: Different patterns simulate real-world scenarios

## How to Change Workload Patterns

### **In the Frontend Dashboard:**
1. Navigate to the "Demonstrations" section
2. Select "Workload Patterns" demonstration
3. Choose different patterns from the dropdown
4. Observe how resource demands change

### **In Code:**
```python
# Change pattern programmatically
workload_generator.set_pattern('burst')  # or any other pattern
```

### **In Configuration:**
```python
config['workload_generator']['default_pattern'] = 'cyclical'
```

## Real Data Integration

All patterns are now based on **real Google Cluster data**:
- **Base values**: Derived from actual cluster usage statistics
- **Variations**: Applied on top of real data points
- **Fallback**: Synthetic patterns if real data unavailable

## Recommendations

### **For Testing:**
- Start with `steady` pattern for baseline
- Use `burst` pattern to test scalability
- Try `cyclical` pattern for time-based optimization

### **For Production Simulation:**
- Use `web_service` for web applications
- Use `ml_training` for AI/ML workloads
- Use `data_analytics` for BI applications

### **For Training:**
- Cycle through different patterns during training
- Use `burst` and `cyclical` for robustness
- Combine patterns for comprehensive learning

## Monitoring Impact

When you change patterns, monitor:
1. **Resource utilization** changes
2. **Cost variations** across patterns
3. **Agent decision-making** adaptations
4. **System performance** under different loads

This helps understand how well your MARL system adapts to different real-world scenarios! 