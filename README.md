# MARL for Google Cloud Provisioning

This project implements a Multi-Agent Reinforcement Learning (MARL) system for automating cloud resource provisioning on Google Cloud Platform.

## Overview

The system consists of four specialized agents:
- **Compute Agent**: Manages VM instances and compute resources
- **Storage Agent**: Handles storage solutions (disks, buckets)
- **Network Agent**: Controls networking resources (VPCs, subnets)
- **Database Agent**: Provisions database services

These agents learn to work together to efficiently provision cloud resources based on customer requests.

## System Architecture

The system uses a centralized training with decentralized execution approach:
- Each agent has its own policy network that maps states to actions
- A shared experience buffer allows agents to learn from each other
- During training, agents share information to coordinate
- During execution/deployment, agents can operate independently

### System Architecture Diagram

![MARL System Architecture](images/system_architecture.png)

For a detailed description of the system architecture, see [System Architecture](system_architecture.md).

## Project Structure

```
├── src/
│   ├── marl_gcp/
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── base_agent.py
│   │   │   ├── compute_agent.py
│   │   │   ├── storage_agent.py 
│   │   │   ├── network_agent.py
│   │   │   └── database_agent.py
│   │   ├── environment/
│   │   │   ├── __init__.py
│   │   │   └── gcp_environment.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── experience_buffer.py
│   │   │   └── monitoring.py
│   │   ├── configs/
│   │   │   ├── __init__.py
│   │   │   └── default_config.py
│   │   ├── __init__.py
│   │   └── system_architecture.py
│   └── main.py
├── models/
├── results/
├── images/
│   └── system_architecture.png
├── MARL_GCP_Project_Phases.md
├── system_architecture.md
├── README.md
└── requirements.txt
```

## Installation

1. Clone this repository
2. Install dependencies:
```
pip install -r requirements.txt
```
3. (Optional) Set up GCP credentials if connecting to real GCP services

## Usage

### Training the system:
```
python src/main.py --episodes 1000 --seed 42
```

### Evaluating a trained model:
```
python src/main.py --eval --model_path models/marl_gcp_default
```

### Additional options:
- `--config`: Path to custom configuration file
- `--log_level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--seed`: Set random seed for reproducibility
- `--episodes`: Override number of training episodes

## System Components

### Agent Implementation
Each agent uses the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm, which includes:
- Actor network: Maps states to deterministic actions
- Critic networks: Two Q-value estimators to reduce overestimation
- Target networks: Stabilize training

### Experience Buffer
The shared experience buffer facilitates:
- Per-agent experience storage
- Experience sharing between agents
- Prioritized sampling based on importance

### Configuration
Default settings are defined in `src/marl_gcp/configs/default_config.py`, including:
- Agent network architectures
- Training hyperparameters
- Environment settings
- Resource constraints

## Development Roadmap

The full development plan is documented in `MARL_GCP_Project_Phases.md`, outlining the following phases:
1. System Architecture Design (current phase)
2. Environment Simulation Development
3. MARL Agent Implementation
4. Reward Function Engineering
5. Training Infrastructure Setup
6. Agent Coordination Mechanism
7. GCP Integration Layer
8. Evaluation Framework
9. Deployment & Monitoring System
10. Thesis Documentation & Analysis

## License

This project is part of a Master's thesis in Artificial Intelligence and Machine Learning.

## Contact

For questions or feedback, please contact the project author.