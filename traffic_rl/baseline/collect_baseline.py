import traci
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt

class SimpleTrafficMetrics:
    def __init__(self, sumo_cfg, max_steps=3600, output_dir="metrics_output_eval"):
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.output_dir = output_dir
        
        self.metrics = {
            'step': [],
            'vehicle_count': [],
            'avg_waiting_time': [],
            'avg_speed': [],
            'total_stopped': []
        }
        self.max_queue_lengths = {'North': 0, 'South': 0, 'East': 0, 'West': 0}
        
    def run(self):
        traci.start(["sumo", "-c", self.sumo_cfg])
        step = 0
        
        while step < self.max_steps and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            self.collect_metrics(step)
            step += 1

        self.save_results()
        self.plot_results()
        self.print_summary(step)
        traci.close()  # Close after printing summary

    def collect_metrics(self, step):
        vehicles = traci.vehicle.getIDList()
        num_vehicles = len(vehicles)

        waiting_times = [traci.vehicle.getWaitingTime(v) for v in vehicles]
        avg_waiting_time = np.mean(waiting_times) if waiting_times else 0

        speeds = [traci.vehicle.getSpeed(v) for v in vehicles]
        avg_speed = np.mean(speeds) if speeds else 0

        total_stopped = sum(1 for v in vehicles if traci.vehicle.getSpeed(v) < 0.1)

        self.metrics['step'].append(step)
        self.metrics['vehicle_count'].append(num_vehicles)
        self.metrics['avg_waiting_time'].append(avg_waiting_time)
        self.metrics['avg_speed'].append(avg_speed)
        self.metrics['total_stopped'].append(total_stopped)

        # Assume edges are named conventionally ("north", "south", "east", "west")
        directions = {'North': 'north', 'South': 'south', 'East': 'east', 'West': 'west'}
        for dir_name, edge_id in directions.items():
            if edge_id in traci.edge.getIDList():
                queue_length = traci.edge.getLastStepHaltingNumber(edge_id)
                if queue_length > self.max_queue_lengths[dir_name]:
                    self.max_queue_lengths[dir_name] = queue_length

    def save_results(self):
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(self.metrics)
        df.to_csv(os.path.join(self.output_dir, f"metrics_summary_{timestamp}.csv"), index=False)

    def plot_results(self):
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(self.metrics['step'], self.metrics['vehicle_count'], label="Vehicle Count")
        plt.ylabel("Vehicles")
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(self.metrics['step'], self.metrics['avg_waiting_time'], label="Avg Waiting Time (s)", color='orange')
        plt.ylabel("Seconds")
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.metrics['step'], self.metrics['avg_speed'], label="Avg Speed (m/s)", color='green')
        plt.ylabel("m/s")
        plt.xlabel("Simulation Step")
        plt.legend()

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.output_dir, f"metrics_plot_{timestamp}.png"))
        plt.close()

    def print_summary(self, steps_completed):
        summary = {
            'Total Steps': steps_completed,
            'Max Vehicles': max(self.metrics['vehicle_count']),
            'Total Teleported': traci.simulation.getStartingTeleportNumber(),
            'Avg Waiting Time': np.mean(self.metrics['avg_waiting_time']),
            'Max Waiting Time': max(self.metrics['avg_waiting_time']),
            'Avg Speed': np.mean(self.metrics['avg_speed']),
        }

        print("\nSimulation Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")

        for direction, queue_length in self.max_queue_lengths.items():
            print(f"Max Queue Length {direction}: {queue_length}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simplified_metrics.py <sumocfg_path>")
        sys.exit(1)

    sumocfg_path = sys.argv[1]
    metrics_collector = SimpleTrafficMetrics(sumocfg_path)
    metrics_collector.run()