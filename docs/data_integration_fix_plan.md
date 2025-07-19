# Data Integration Fix Plan

## 🚨 CRITICAL ISSUE: Agents Not Using Real Google Cluster Data

### Problem Summary
The MARL agents are currently training on synthetic/simulated data instead of the real Google Cluster data you've uploaded. This is a **critical issue** that makes the entire thesis invalid.

### Current State
- ✅ **Real Data Available**: 200MB+ of Google Cluster data in `data/google_cluster/`
- ✅ **Processed Data**: Real statistics in `data/processed/feature_statistics.json`
- ✅ **Workload Patterns**: Real patterns in `data/processed/workload_patterns.json`
- ❌ **Agents Using Fake Data**: Environment generates synthetic workloads instead of using real data

---

## 🔧 STEP-BY-STEP FIX PLAN

### Step 1: Fix Workload Generator
**File**: `src/marl_gcp/data/workload_generator.py`

**Current Problem**: The workload generator creates synthetic patterns instead of loading real Google Cluster data.

**Required Changes**:
```python
# Instead of generating synthetic data:
def generate_workload(self, pattern_name):
    # CURRENT: Creates fake patterns
    return synthetic_pattern

# Should load real data:
def load_real_workload(self, pattern_name):
    # Load from data/processed/workload_patterns.json
    # Use real CPU, memory, disk patterns from Google Cluster
    return real_google_cluster_pattern
```

### Step 2: Update Environment to Use Real Data
**File**: `src/marl_gcp/environment/gcp_environment.py`

**Current Problem**: Environment uses hardcoded synthetic workloads.

**Required Changes**:
```python
# In __init__ method:
def __init__(self, config):
    # CURRENT: Uses synthetic workload
    self.current_workload = {
        'compute_demand': np.array([0.3, 0.3, 0.3]),  # FAKE DATA
        'storage_demand': 0.3,  # FAKE DATA
        # ...
    }
    
    # SHOULD BE: Load real Google Cluster data
    self.workload_generator = WorkloadGenerator(config)
    self.current_workload = self.workload_generator.load_real_workload('steady')
```

### Step 3: Update Agent Observation Spaces
**Files**: All agent files in `src/marl_gcp/agents/`

**Current Problem**: Agents expect synthetic observation spaces that don't match real Google Cluster data features.

**Required Changes**:
```python
# Current observation space (synthetic):
'compute': spaces.Dict({
    'instances': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
    'cpus': spaces.Box(low=0, high=500, shape=(1,), dtype=np.float32),
    # ...
})

# Should match real Google Cluster features:
'compute': spaces.Dict({
    'cpu_rate': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Real feature
    'canonical_memory_usage': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Real feature
    'local_disk_space_usage': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Real feature
    'disk_io_time': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Real feature
    'active_machines': spaces.Box(low=30, high=70, shape=(1,), dtype=np.float32),  # Real feature
    'active_tasks': spaces.Box(low=60, high=140, shape=(1,), dtype=np.float32),  # Real feature
})
```

### Step 4: Update Configuration
**File**: `src/marl_gcp/configs/default_config.py`

**Required Changes**:
```python
# Update agent configurations to match real data dimensions:
'agents': {
    'compute': {
        'state_dim': 8,  # Match real Google Cluster features
        'action_dim': 5,
        # ...
    },
    'storage': {
        'state_dim': 4,  # Match real storage features
        'action_dim': 4,
        # ...
    },
    # ... update other agents
}
```

### Step 5: Retrain All Agents
**Files**: All model files in `src/models/marl_gcp_default/`

**Required Action**: Delete existing models and retrain on real data.

```bash
# Delete existing models
rm src/models/marl_gcp_default/*.pt

# Retrain with real data
python src/main.py --episodes 1000
```

---

## 📊 REAL DATA FEATURES TO USE

Based on `data/processed/feature_statistics.json`, here are the real Google Cluster features:

### Compute Features:
- `cpu_rate`: Mean=0.4, Std=0.2 (CPU utilization)
- `canonical_memory_usage`: Mean=0.5, Std=0.25 (Memory usage)
- `local_disk_space_usage`: Mean=0.3, Std=0.15 (Disk usage)
- `disk_io_time`: Mean=0.2, Std=0.1 (I/O time)
- `cpu_memory_ratio`: Mean=0.8, Std=0.4 (CPU/Memory ratio)
- `io_ratio`: Mean=0.7, Std=0.35 (I/O ratio)
- `active_machines`: Mean=50, Std=10 (Active machines)
- `active_tasks`: Mean=100, Std=20 (Active tasks)

### Real Workload Patterns:
From `data/processed/workload_patterns.json`:
- **Steady**: CPU mean=0.496, Memory mean=0.497
- **Burst**: CPU mean=0.347, with burst probability=0.1, magnitude=3.0
- **Cyclical**: CPU mean=0.496, amplitude=0.427, period=24 hours

---

## 🎯 IMPLEMENTATION PRIORITY

### High Priority (Must Fix):
1. **Fix Workload Generator** - Load real data patterns
2. **Update Environment** - Use real workload data
3. **Retrain Agents** - Train on real data

### Medium Priority:
4. **Update Agent Architectures** - Match real feature dimensions
5. **Update Configuration** - Align with real data

### Low Priority:
6. **Optimize Performance** - Fine-tune for real data patterns

---

## ⏱️ ESTIMATED TIME TO FIX

- **Step 1-2**: 2-3 hours (fix data loading)
- **Step 3-4**: 1-2 hours (update configurations)
- **Step 5**: 4-6 hours (retrain agents)
- **Testing**: 2-3 hours

**Total**: 9-14 hours of focused work

---

## 🚀 QUICK START COMMANDS

```bash
# 1. Backup current models
mkdir src/models/backup
cp src/models/marl_gcp_default/*.pt src/models/backup/

# 2. Delete current models
rm src/models/marl_gcp_default/*.pt

# 3. Test data loading
python -c "
from marl_gcp.data.workload_generator import WorkloadGenerator
wg = WorkloadGenerator({'data_dir': 'data/processed'})
print('Real workload patterns:', wg.get_available_patterns())
"

# 4. Retrain with real data
python src/main.py --episodes 1000 --log_level DEBUG
```

---

## ✅ SUCCESS CRITERIA

After fixing, verify:
1. **Environment loads real data**: Check logs show "Loading real Google Cluster patterns"
2. **Agents use real features**: Observation spaces match real data dimensions
3. **Training uses real patterns**: Loss curves show learning from real workload patterns
4. **Results are realistic**: Performance metrics match Google's published findings

---

## ⚠️ WARNING

**Do not proceed with thesis presentation until this data integration issue is fixed!**

The current system is essentially a "toy example" using synthetic data. For a valid thesis, you need to demonstrate that your MARL agents can learn from and adapt to real Google Cluster infrastructure patterns. 