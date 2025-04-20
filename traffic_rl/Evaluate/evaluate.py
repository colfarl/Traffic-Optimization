# traffic_rl/evaluate.py
# Author: Colin Farley • Last edit: 2025‑04‑20
"""
Evaluate a trained DQN model or a fixed-time baseline controller and export CSV +
PNG for the paper.
"""

from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traci

from traffic_rl.env.traffic_env import TrafficEnv
from traffic_rl.agents.dqn_agent import DQNAgent


def run_fixed_time(env: TrafficEnv, phase: int = 0) -> Dict[str, List[float]]:
    """Cycle a static phase for the whole episode and log KPIs."""
    metrics = defaultdict_lists()
    state = env.reset()
    done, step = False, 0

    while not done:
        _, reward, done, _ = env.step(phase)  # keep light on selected phase
        log_step(metrics, env, reward, step)
        step += 1
    return metrics


def run_dqn(env: TrafficEnv, agent: DQNAgent) -> Dict[str, List[float]]:
    metrics = defaultdict_lists()
    state = env.reset()
    done, step = False, 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        log_step(metrics, env, reward, step)
        state, step = next_state, step + 1
    return metrics


def log_step(store: Dict[str, List[float]], env: TrafficEnv, reward: float, step: int):
    lane_ids = {
        lane for tl in env.tl_ids for lane in traci.trafficlight.getControlledLanes(tl)
    }
    vehicle_count = len(traci.vehicle.getIDList())
    waiting_times = [traci.lane.getWaitingTime(l) for l in lane_ids]
    speeds = [traci.lane.getLastStepMeanSpeed(l) for l in lane_ids]

    store["step"].append(step)
    store["reward"].append(reward)
    store["vehicle_count"].append(vehicle_count)
    store["avg_waiting_time"].append(np.mean(waiting_times) if waiting_times else 0)
    store["avg_speed"].append(np.mean(speeds) if speeds else 0)


def defaultdict_lists() -> Dict[str, List[float]]:
    return {
        "step": [],
        "reward": [],
        "vehicle_count": [],
        "avg_waiting_time": [],
        "avg_speed": [],
    }

def plot_metrics(df: pd.DataFrame, save_path: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL or baseline controller")
    parser.add_argument("--sumo_config", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--output_dir", default="eval_output")
    parser.add_argument("--gui", action="store_true")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model_path", help="Path to .pth checkpoint")
    mode.add_argument("--baseline", action="store_true", help="Run fixed phase 0")

    args = parser.parse_args()

    all_metrics = defaultdict_lists()
    for ep in range(args.episodes):
        env = TrafficEnv(
            {
                "sumo_config": args.sumo_config,
                "max_steps": args.max_steps,
                "yellow_time": 3,
            }
        )

        if args.baseline:
            ep_metrics = run_fixed_time(env, phase=0)
        else:
            agent = DQNAgent(state_size=4, action_size=4, device="cpu")
            agent.load(args.model_path)
            agent.epsilon = 0.0
            ep_metrics = run_dqn(env, agent)

        env.close()
        for k in all_metrics:
            all_metrics[k].extend(ep_metrics[k])

        print(
            f"[Eval] Episode {ep+1}/{args.episodes} "
            f"Avg wait={np.mean(ep_metrics['avg_waiting_time']):.2f}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(all_metrics)
    csv_path = Path(args.output_dir, "evaluation_metrics.csv")
    df.to_csv(csv_path, index=False)
    plot_metrics(df, Path(args.output_dir, "evaluation_plot.png"))

    print("\n--- Summary ---")
    print(f"Avg Reward: {np.mean(all_metrics['reward']):.2f}")
    print(f"Avg Waiting Time: {np.mean(all_metrics['avg_waiting_time']):.2f}")
    print(f"Avg Speed: {np.mean(all_metrics['avg_speed']):.2f}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
