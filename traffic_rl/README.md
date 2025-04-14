# Traffic Signal Control using Deep Reinforcement Learning

This project implements a Deep Q-Network (DQN) agent to optimize traffic signal control in a simulated urban environment. The agent learns to minimize waiting times and queue lengths at intersections by controlling traffic light phases.

## Features

- Deep Q-Network (DQN) agent for traffic signal control
- Real-time visualization of agent performance
- Training progress tracking and visualization
- Support for SUMO traffic simulation
- Configurable traffic scenarios

## Installation

1. Install SUMO:
```bash
# On macOS
brew install sumo

# On Ubuntu
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
traffic_rl/
├── agents/           # DQN agent implementation
├── DC_downtown/      # SUMO simulation files
├── models/           # Saved model checkpoints
├── plots/            # Training progress plots
├── train.py          # Training script
├── visualize.py      # Visualization script
└── requirements.txt  # Python dependencies
```

## Usage

1. Train the agent:
```bash
python train.py DC_downtown/key_intersection.sumocfg
```

2. Visualize the trained agent:
```bash
python visualize.py DC_downtown/key_intersection.sumocfg models/dqn_agent_[timestamp].pt
```

## Key Metrics

The agent optimizes for:
- Queue lengths at intersections
- Vehicle waiting times
- Traffic flow efficiency

## Visualization

The visualization shows:
- Real-time queue lengths
- Waiting times
- Agent rewards
- Traffic light phase changes

## Presentation Demo

For the presentation, you can:
1. Show the training process with real-time visualization
2. Compare baseline vs. DQN-controlled traffic flow
3. Demonstrate the agent's decision-making process
4. Display performance metrics and improvements

## Future Improvements

- Multi-agent coordination
- Advanced reward shaping
- Transfer learning between intersections
- Real-world deployment considerations

## Project Structure

```
traffic_rl/
├── env/
│   └── traffic_env.py      # Traffic environment implementation
├── agents/
│   └── dqn_agent.py        # DQN agent implementation
├── configs/
│   ├── intersection.json   # Training configuration
│   ├── intersection.netccfg # Network generation config
│   ├── intersection.nod.xml # Node definitions
│   ├── intersection.edg.xml # Edge definitions
│   ├── intersection.tll.xml # Traffic light logic
│   ├── intersection.sumocfg # SUMO configuration
│   └── intersection.settings.xml # GUI settings
├── utils/                  # Utility functions
├── train.py               # Training script
└── requirements.txt       # Project dependencies
```

## Prerequisites

- Python 3.7+
- SUMO (Simulation of Urban MObility)
- PyTorch
- Other dependencies listed in requirements.txt

## Configuration

The project uses several configuration files:

- `intersection.json`: Training parameters for the DQN agent
- `intersection.netccfg`: Network generation parameters
- `intersection.nod.xml`: Intersection node definitions
- `intersection.edg.xml`: Road edge definitions
- `intersection.tll.xml`: Traffic light logic
- `intersection.sumocfg`: SUMO simulation parameters
- `intersection.settings.xml`: GUI visualization settings

## Reward Function

The reward function considers:
- Throughput (vehicles passing through)
- Queue lengths (penalty)
- Waiting times (penalty)

## Results

Training results are saved in the output directory:
- Model checkpoints (`model_*.pt`)
- Training history (`rewards_history.npy`)

## References

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [Deep Q-Learning Paper](https://www.nature.com/articles/nature14236)
- [Flow Project](https://github.com/flow-project/flow) 