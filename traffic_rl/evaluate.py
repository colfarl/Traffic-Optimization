# evaluate.py

import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traci

from env.traffic_env import TrafficEnv
from agents.dqn_agent import DQNAgent

def evaluate(args):
    config = {
        "sumo_config": args.sumo_config,
        "max_steps": args.max_steps,
        "yellow_time": 3,
        "min_green": 10,
        "max_green": 60,
        "use_gui": args.gui
    }

    env = TrafficEnv(config)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    agent.load(args.model_path)
    agent.epsilon = 0.0  # Evaluation = pure exploitation

    metrics = {
        "step": [],
        "reward": [],
        "vehicle_count": [],
        "avg_waiting_time": [],
        "avg_speed": []
    }

    for ep in range(args.episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            # --- Collect real-time stats using traci ---
            lane_ids = []
            for tl in env.tl_ids:
                lane_ids.extend(traci.trafficlight.getControlledLanes(tl))
            lane_ids = list(set(lane_ids))

            vehicle_ids = traci.vehicle.getIDList()
            vehicle_count = len(vehicle_ids)

            waiting_times = [traci.lane.getWaitingTime(lane) for lane in lane_ids]
            avg_wait = np.mean(waiting_times) if waiting_times else 0

            speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in lane_ids]
            avg_speed = np.mean(speeds) if speeds else 0

            # --- Log it ---
            metrics["step"].append(step)
            metrics["reward"].append(reward)
            metrics["vehicle_count"].append(vehicle_count)
            metrics["avg_waiting_time"].append(avg_wait)
            metrics["avg_speed"].append(avg_speed)

            state = next_state
            total_reward += reward
            step += 1

        print(f"[Eval] Episode {ep+1}/{args.episodes}, Total Reward: {total_reward:.2f}")

    env.close()

    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(metrics)
    csv_path = os.path.join(args.output_dir, "evaluation_metrics.csv")
    df.to_csv(csv_path, index=False)

    # Plot summary
    plot_path = os.path.join(args.output_dir, "evaluation_plot.png")
    plot_metrics(df, plot_path)

    # Print summary
    print("\n--- Evaluation Summary ---")
    print(f"Avg Reward: {np.mean(metrics['reward']):.2f}")
    print(f"Avg Waiting Time: {np.mean(metrics['avg_waiting_time']):.2f}")
    print(f"Avg Speed: {np.mean(metrics['avg_speed']):.2f}")
    print(f"Max Vehicles: {max(metrics['vehicle_count'])}")
    print(f"Metrics saved to: {csv_path}")


def plot_metrics(df, save_path):
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(df["step"], df["vehicle_count"])
    plt.ylabel("Vehicle Count")

    plt.subplot(3, 1, 2)
    plt.plot(df["step"], df["avg_waiting_time"], color="orange")
    plt.ylabel("Avg Waiting Time")

    plt.subplot(3, 1, 3)
    plt.plot(df["step"], df["avg_speed"], color="green")
    plt.ylabel("Avg Speed")
    plt.xlabel("Step")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sumo_config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="eval_output")
    parser.add_argument("--gui", action="store_true", help="Enable GUI mode")

    args = parser.parse_args()
    evaluate(args)
