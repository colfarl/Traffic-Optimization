# train_multi_agent.py

import numpy as np
import torch
import traci

from env.multiagent_traffic_env import MultiAgentTrafficEnv
from agents.dqn_agent import DQNAgent

def train_multi_agent():
    """
    Train a separate DQN agent for each traffic light in the network.
    """

    config = {
        "sumo_config": "DC_downtown/dayuan.sumocfg",
        "max_steps": 12000,
        "yellow_time": 3,
        "min_green": 10,
        "max_green": 60
    }
    env = MultiAgentTrafficEnv(config)

    # 1) Initialize environment & get the initial dictionary of observations
    obs_dict = env.reset()  # => { tl_id: obs }
    
    # We'll discover the traffic lights from env.tl_ids
    tl_ids = env.tl_ids

    # 2) Create one agent per traffic light
    agents = {}
    for tl_id in tl_ids:
        # Each agent uses the same dimension, e.g. 4-state, 4-action
        state_size = env.agent_observation_space.shape[0]  # 4
        action_size = env.agent_action_space.n             # 4

        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=32,
            device="cpu"
        )
        agents[tl_id] = agent

    num_episodes = 10
    update_target_interval = 5

    for episode in range(num_episodes):
        obs_dict = env.reset()  # reset environment => all TLs
        done = False
        total_rewards = {tl_id: 0.0 for tl_id in tl_ids}

        while not done:
            # 3) Each agent chooses an action
            actions_dict = {}
            for tl_id in tl_ids:
                state = obs_dict[tl_id]
                action = agents[tl_id].act(state)
                actions_dict[tl_id] = action

            # 4) Step environment with the dict of actions
            next_obs_dict, rewards_dict, done, info_dict = env.step(actions_dict)

            # 5) Store experiences & replay
            for tl_id in tl_ids:
                state = obs_dict[tl_id]
                action = actions_dict[tl_id]
                reward = rewards_dict[tl_id]
                next_state = next_obs_dict[tl_id]

                agents[tl_id].remember(state, action, reward, next_state, done)
                agents[tl_id].replay()

                total_rewards[tl_id] += reward

            obs_dict = next_obs_dict

        # 6) After each episode, optionally update target networks
        if (episode + 1) % update_target_interval == 0:
            for tl_id in tl_ids:
                agents[tl_id].update_target_network()

        # Print some stats
        avg_reward = np.mean(list(total_rewards.values()))
        print(f"Episode {episode+1}/{num_episodes}, mean reward = {avg_reward:.2f}")

    # 7) Save each agent
    for tl_id in tl_ids:
        agents[tl_id].save(f"dqn_model_{tl_id}.pth")

    env.close()
    print("Training finished for all traffic lights!")

if __name__ == "__main__":
    train_multi_agent()
