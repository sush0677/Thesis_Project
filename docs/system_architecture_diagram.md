# Detailed System Architecture Diagram

## MARL-GCP System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MARL-GCP SYSTEM ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              CENTRAL COORDINATOR                            │
│                         (MARLSystemArchitecture)                           │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Training      │    │   Experience    │    │   Monitoring    │         │
│  │   Controller    │    │   Buffer        │    │   System        │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Centralized Training
                                    │ Decentralized Execution
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FOUR SPECIALIZED AGENTS                        │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   COMPUTE       │    │   STORAGE       │    │   NETWORK       │         │
│  │   AGENT         │    │   AGENT         │    │   AGENT         │         │
│  │                 │    │                 │    │                 │         │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │         │
│  │ │   Policy    │ │    │ │   Policy    │ │    │ │   Policy    │ │         │
│  │ │   Network   │ │    │ │   Network   │ │    │ │   Network   │ │         │
│  │ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │         │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │         │
│  │ │   Critic    │ │    │ │   Critic    │ │    │ │   Critic    │ │         │
│  │ │   Network   │ │    │ │   Network   │ │    │ │   Network   │ │         │
│  │ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │   DATABASE      │                                                        │
│  │   AGENT         │                                                        │
│  │                 │                                                        │
│  │ ┌─────────────┐ │                                                        │
│  │ │   Policy    │ │                                                        │
│  │ │   Network   │ │                                                        │
│  │ └─────────────┘ │                                                        │
│  │ ┌─────────────┐ │                                                        │
│  │ │   Critic    │ │                                                        │
│  │ │   Network   │ │                                                        │
│  │ └─────────────┘ │                                                        │
│  └─────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Agent Actions & Observations
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GCP ENVIRONMENT                                │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Resource      │    │   Workload      │    │   Pricing       │         │
│  │   Constraints   │    │   Generator     │    │   Model         │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Google        │    │   Provisioning  │    │   Cost          │         │
│  │   Cluster Data  │    │   Delays        │    │   Calculator    │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Architecture

### 1. Central Coordinator (MARLSystemArchitecture)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CENTRAL COORDINATOR                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           Training Loop                                 │ │
│  │                                                                         │ │
│  │  1. Reset Environment                                                   │ │
│  │  2. Collect Observations from all agents                               │ │
│  │  3. Get actions from all agents                                         │ │
│  │  4. Execute actions in environment                                      │ │
│  │  5. Store experiences in shared buffer                                  │ │
│  │  6. Update agent policies                                               │ │
│  │  7. Repeat until episode ends                                           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        Experience Buffer                                │ │
│  │                                                                         │ │
│  │  • Shared memory for all agents                                         │ │
│  │  • Prioritized experience replay                                        │ │
│  │  • Batch sampling for training                                          │ │
│  │  • Capacity: 1,000,000 experiences                                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         Monitoring System                               │ │
│  │                                                                         │ │
│  │  • Track training progress                                               │ │
│  │  • Monitor agent performance                                             │ │
│  │  • Log resource usage and costs                                          │ │
│  │  • Generate visualizations                                               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Agent Architecture (Each Agent)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AGENT ARCHITECTURE                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           Observation Space                             │ │
│  │                                                                         │ │
│  │  • Resource utilization (CPU, Memory, Storage, Network)                │ │
│  │  • Current demand patterns                                              │ │
│  │  • Cost metrics                                                         │ │
│  │  • Service-specific metrics                                             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           Policy Network                                │ │
│  │                                                                         │ │
│  │  • Input: State observation                                             │ │
│  │  • Hidden layers: 256 neurons each                                      │ │
│  │  • Output: Action distribution parameters                               │ │
│  │  • Activation: ReLU + Tanh (bounded actions)                           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           Action Space                                  │ │
│  │                                                                         │ │
│  │  • Continuous actions (scaled -1 to 1)                                  │ │
│  │  • Service-specific actions                                             │ │
│  │  • Validation and clipping                                              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           Critic Network                                │ │
│  │                                                                         │ │
│  │  • Input: State + Action                                                │ │
│  │  • Twin Q-networks (Q1, Q2)                                             │ │
│  │  • Output: Q-values for state-action pairs                              │ │
│  │  • Used for policy improvement                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Environment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GCP ENVIRONMENT                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        Resource Management                              │ │
│  │                                                                         │ │
│  │  • Current resource allocation                                           │ │
│  │  • Resource constraints and limits                                       │ │
│  │  • Service interdependencies                                             │ │
│  │  • Provisioning delays simulation                                        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        Workload Generator                               │ │
│  │                                                                         │ │
│  │  • Real Google Cluster data integration                                 │ │
│  │  • Workload patterns (steady, burst, cyclical)                          │ │
│  │  • Demand forecasting                                                    │ │
│  │  • Temporal patterns                                                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          Reward System                                  │ │
│  │                                                                         │ │
│  │  • Multi-objective reward function                                      │ │
│  │  • Cost efficiency (20-30% weight)                                      │ │
│  │  • Performance metrics (30-40% weight)                                  │ │
│  │  • Reliability guarantees (15-20% weight)                               │ │
│  │  • Sustainability metrics (10-15% weight)                               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Communication Flow

### Training Phase (Centralized)
```
1. Environment → All Agents: Send observations
2. All Agents → Central Coordinator: Send actions
3. Central Coordinator → Environment: Execute actions
4. Environment → Central Coordinator: Return rewards & new state
5. Central Coordinator → Experience Buffer: Store experience
6. Central Coordinator → All Agents: Update policies
```

### Execution Phase (Decentralized)
```
1. Environment → Individual Agent: Send local observation
2. Individual Agent → Environment: Send action
3. Environment → Individual Agent: Return reward & new state
4. Individual Agent: Update local policy
```

## Data Flow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Google        │    │   Data          │    │   Workload      │
│   Cluster       │───▶│   Processor     │───▶│   Generator     │
│   Data          │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Feature       │    │   Environment   │    │   Agent         │
│   Statistics    │◀───│   State         │◀───│   Observations  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Reward        │    │   Experience    │    │   Policy        │
│   Calculator    │───▶│   Buffer        │───▶│   Updates       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## System Properties

### Scalability
- **Horizontal**: Add more agents for different services
- **Vertical**: Scale individual agent capacity
- **Distributed**: Support for distributed training

### Reliability
- **Fault Tolerance**: Individual agent failures don't crash system
- **Redundancy**: Multiple agents can handle same service
- **Recovery**: Automatic recovery from failures

### Performance
- **Parallel Processing**: Agents work simultaneously
- **Efficient Communication**: Minimal overhead between agents
- **Optimized Training**: Shared experience reduces training time

### Security
- **Isolation**: Agents operate in isolated environments
- **Validation**: All actions validated before execution
- **Audit Trail**: Complete logging of all decisions 