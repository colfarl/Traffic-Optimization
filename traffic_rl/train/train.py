# Author: Colin Farley • Last edit: 2025‑04‑20
# Train a Double‑DQN agent for traffic‑signal control (single or multi‑env).

from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import List

from torch.utils.tensorboard import SummaryWriter

from traffic_rl.env.traffic_env import TrafficEnv
from traffic_rl.agents.dqn_agent import DQNAgent


def build_env(file_path: Path, max_steps: int, yellow: int) -> TrafficEnv:
    """Instantiate TrafficEnv with standard config keys."""
    return TrafficEnv(
        {
            "sumo_config": str(file_path),
            "max_steps": max_steps,
            "yellow_time": yellow,
        }
    )


def parse_env_list(env_arg: str) -> List[Path]:
    """
    Accept either:
      • a comma separated list of .sumocfg paths
      • a directory → we will glob **/*.sumocfg
    """
    p = Path(env_arg)
    if p.is_dir():
        return sorted(p.rglob("*.sumocfg"))
    return [Path(x.strip()) for x in env_arg.split(",")]


def train_single(
    env_file: Path,
    episodes: int,
    max_steps: int,
    agent: DQNAgent,
    yellow: int,
    writer: SummaryWriter,
) -> None:
    env = build_env(env_file, max_steps, yellow)
    for ep in range(episodes):
        state = env.reset()
        total_reward, done = 0.0, False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
        agent._soft_update()  # explicit call in case user sets tau==1e‑2
        writer.add_scalar("single_env/episode_reward", total_reward, ep)
    env.close()


def train_multi_env(
    env_files: List[Path],
    episodes: int,
    max_steps: int,
    agent: DQNAgent,
    yellow: int,
    writer: SummaryWriter,
    update_target_every: int = 5,
) -> None:
    for ep in range(episodes):
        total_reward_ep = 0.0
        for env_path in env_files:
            env = build_env(env_path, max_steps, yellow)
            state, done = env.reset(), False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                state = next_state
                total_reward_ep += reward
            env.close()
        writer.add_scalar("multi_env/episode_reward", total_reward_ep, ep)
        if (ep + 1) % update_target_every == 0:
            agent._soft_update()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN for SUMO traffic control")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--single", help="Path to one .sumocfg file", metavar="PATH")
    mode.add_argument(
        "--multi",
        help="Dir with .sumocfg files or comma‑separated list",
        metavar="DIR|LIST",
    )

    # RL / training hyper‑params
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--yellow_time", type=int, default=3)

    parser.add_argument("--save_to", default="dqn_multienv.pth")
    parser.add_argument("--logdir", default="runs")

    args = parser.parse_args()

    agent = DQNAgent(
        state_size=4,
        action_size=4,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch,
        tau=args.tau,
        device=args.device,
    )

    writer = SummaryWriter(log_dir=f"{args.logdir}/{int(time.time())}")

    if args.single:
        train_single(
            env_file=Path(args.single),
            episodes=args.episodes,
            max_steps=args.max_steps,
            agent=agent,
            yellow=args.yellow_time,
            writer=writer,
        )
    else: 
        env_files = parse_env_list(args.multi)
        if not env_files:
            raise ValueError("No .sumocfg files found for multi‑env training.")
        train_multi_env(
            env_files=env_files,
            episodes=args.episodes,
            max_steps=args.max_steps,
            agent=agent,
            yellow=args.yellow_time,
            writer=writer,
        )

    writer.close()
    agent.save(args.save_to)
    print(f"✔  Training complete. Model saved to {args.save_to}")


if __name__ == "__main__":
    main()