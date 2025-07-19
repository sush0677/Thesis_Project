# MARL System Architecture

## Overview

The Multi-Agent Reinforcement Learning (MARL) system for Google Cloud Platform resource provisioning follows a centralized training with decentralized execution approach. This document describes the key components and their interactions.

## Components

### 1. Customer Request
- Entry point for workload requirements
- Includes specifications for compute, storage, network, and database resources
- Translated into environment states for the agents

### 2. Environment Interface
- Translates customer requests into agent-specific observations
- Manages the state representation of the cloud environment
- Processes agent actions and returns rewards
- Provides a simulation layer during training and interfaces with GCP during deployment

### 3. Specialized Agents
The system employs four specialized agents, each responsible for a different aspect of cloud resource provisioning:

#### a. Compute Agent
- Manages VM instances, instance groups, and compute resources
- Makes decisions about instance types, sizes, regions, and autoscaling policies
- Implements TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm

#### b. Storage Agent
- Handles persistent disks, Cloud Storage buckets, and other storage solutions
- Determines storage types, sizes, replication policies, and access patterns

#### c. Network Agent
- Controls VPC networks, subnets, firewalls, and load balancers
- Configures network topology, routing, and security policies

#### d. Database Agent
- Provisions database services (Cloud SQL, Spanner, Bigtable)
- Manages database configurations, scaling, and replication

### 4. Shared Experience Buffer
- Stores experiences from all agents
- Enables experience sharing between agents
- Implements prioritized experience replay for more efficient learning
- Facilitates centralized training while maintaining agent specialization

### 5. Centralized Trainer
- Updates agent policies based on collected experiences
- Coordinates learning across all agents
- Implements algorithms for multi-agent cooperation
- Manages training hyperparameters and learning schedules

### 6. GCP API/Terraform Interface
- Translates agent actions into actual GCP resource provisioning commands
- Uses Terraform for infrastructure-as-code deployment
- Provides a consistent interface for both simulated and real environments

### 7. Monitoring Component
- Tracks resource utilization, costs, and performance metrics
- Generates rewards based on multi-objective optimization criteria
- Provides feedback to agents for learning and improvement
- Visualizes system behavior and agent decisions

## Information Flow

1. **Customer Request → Environment Interface**:
   - Workload requirements are translated into environment states

2. **Environment Interface → Agents**:
   - Each agent receives a specialized view of the environment state

3. **Agents → Shared Experience Buffer**:
   - Agents take actions and store their experiences in the buffer

4. **Shared Experience Buffer → Agents**:
   - Agents sample experiences for learning, including those from other agents

5. **Agents → GCP API/Terraform**:
   - Agent actions are translated into resource provisioning commands

6. **GCP API/Terraform → Monitoring**:
   - Resource provisioning results are tracked and measured

7. **Monitoring → Environment Interface**:
   - Performance metrics are used to calculate rewards

8. **Shared Experience Buffer → Centralized Trainer**:
   - Experiences are used for policy updates

9. **Centralized Trainer → Agents**:
   - Updated policies are distributed to the agents

## Training vs. Deployment

### Centralized Training
- All agents share experiences through the buffer
- The centralized trainer coordinates policy updates
- Agents learn from both their own experiences and those of other agents
- Simulation environment provides fast feedback and safe exploration

### Decentralized Execution
- Each agent operates independently using its learned policy
- No communication required between agents during execution
- Agents can be deployed separately or together as needed
- Direct interface with GCP for real-world resource provisioning

## Reward Structure

The system uses a multi-objective reward function that balances:
- Cost efficiency (minimizing resource costs)
- Performance (meeting latency and throughput requirements)
- Reliability (ensuring high availability and fault tolerance)
- Sustainability (optimizing for energy efficiency and carbon footprint)

These objectives are weighted according to customer priorities and business requirements. 