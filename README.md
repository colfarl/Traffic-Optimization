# Traffic Signal Control with Deep Reinforcement Learning

This project implements a Deep Reinforcement Learning (DRL) approach for optimizing traffic signal control in urban environments using SUMO (Simulation of Urban MObility).

## Setup

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

3. Set up SUMO_HOME environment variable:
   ```bash
   # On macOS
   export SUMO_HOME=/usr/local/opt/sumo/share/sumo
   
   # On Ubuntu
   export SUMO_HOME=/usr/share/sumo
   ```

## Usage

### Collecting Baseline Metrics

To collect baseline metrics for a traffic simulation:

```bash
python baseline/collect_baseline.py <sumocfg_path> <output_dir>
```

Example:
```bash
python baseline/collect_baseline.py simple/simple_grid.sumocfg baseline_results/
```

The script will:
- Run the simulation for the specified number of steps
- Collect global, edge-specific, and junction-specific metrics
- Generate CSV files with detailed metrics
- Create summary plots of key performance indicators

## Features

- Dynamic network detection and metric collection
- Configurable metrics collection
- Support for multiple network types
- Comprehensive visualization of traffic patterns
- Detailed performance analysis

## Dependencies

- Python 3.7+
- SUMO
- numpy
- pandas
- matplotlib
- traci (SUMO Python interface)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 