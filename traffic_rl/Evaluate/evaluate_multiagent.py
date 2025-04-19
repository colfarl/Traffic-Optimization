# evaluate_multi_agent.py

import numpy as np
import torch
import traci

from multiagent_traffic_env import MultiAgentTrafficEnv
from dqn_agent import DQNAgent

def evaluate_multi_agent(load_prefix="dqn_model_", episodes=3):
    """
    Loads each traffic light's agent from disk (e.g., dqn_model_{tl_id}.pth),
    sets epsilon=0, and runs multiple evaluation episodes in MultiAgentTrafficEnv.
    
    Args:
        load_prefix: path/prefix for loading each agent model, 
            e.g. "dqn_model_" means we expect "dqn_model_{tl_id}.pth"
        episodes: how many evaluation runs to perform
    """
    config = {
        "sumo_config": "DC_downtown/dayuan.sumocfg",
        "max_steps": 12000,
        "yellow_time": 3,
        "min_green": 10,
        "max_green": 60
    }
    env = MultiAgentTrafficEnv(config)

    # Step 1: Start environment just to get tl_ids
    obs_dict = env.reset()
    tl_ids = env.tl_ids
    env.close()  # We’ll re-reset in the evaluation loop below

    # Step 2: Load each agent
    agents = {}
    for tl_id in tl_ids:
        # We assume each intersection uses the same 4-state, 4-action dimension
        state_size = env.agent_observation_space.shape[0]
        action_size = env.agent_action_space.n
        
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.001,  # not so relevant during evaluation
            gamma=0.95
        )
        
        # Load the model for this TL
        model_path = f"{load_prefix}{tl_id}.pth"
        print(f"Loading agent for TL={tl_id} from {model_path}")
        agent.load(model_path)
        
        # Turn off exploration
        agent.epsilon = 0.0

        agents[tl_id] = agent

    # Step 3: Run multiple evaluation episodes
    for ep in range(episodes):
        obs_dict = env.reset()
        done = False
        
        # Track cumulative rewards for logging
        total_rewards = {tl_id: 0.0 for tl_id in tl_ids}

        while not done:
            # Each agent picks an action
            actions_dict = {}
            for tl_id in tl_ids:
                state = obs_dict[tl_id]
                action = agents[tl_id].act(state)  # pure exploitation
                actions_dict[tl_id] = action

            # Step environment
            next_obs_dict, rewards_dict, done, info_dict = env.step(actions_dict)

            # Accumulate the local reward for each agent
            for tl_id in tl_ids:
                total_rewards[tl_id] += rewards_dict[tl_id]

            obs_dict = next_obs_dict

        # After one episode, print or log results
        # e.g. average reward across all lights
        mean_ep_reward = np.mean(list(total_rewards.values()))
        print(f"[Eval] Episode {ep+1}/{episodes}, Mean Reward = {mean_ep_reward:.2f}")

    env.close()
    print("Evaluation completed for all agents.")
    
if __name__ == "__main__":
    evaluate_multi_agent(load_prefix="dqn_model_", episodes=3)
