# train.py

from env.traffic_env import TrafficEnv
from agents.dqn_agent import DQNAgent
import torch

def train():
    config = {
        "sumo_config": "simple/simple_grid.sumocfg",
        "max_steps": 1000
    }

    env = TrafficEnv(config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, device="cpu")

    num_episodes = 50
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

        agent.update_target()
        print(f"Episode {ep+1} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    agent.save("dqn_traffic_model.pth")
    print("Model saved.")
    env.close()

def train_multi_env():
    env_files = [
        "simple/simple_grid.sumocfg",
        "simple2/random.sumocfg",
        "simple3/random.sumocfg"
    ]

    num_episodes = 150
    update_target_interval = 5

    agent = DQNAgent(
        state_size=4,
        action_size=4,
        lr=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
        device="cpu"
    )

    for episode in range(num_episodes):
        total_reward = 0
        for env_file in env_files:
            config = {
                "sumo_config": env_file,
                "max_steps": 1000
            }
            env = TrafficEnv(config)
            state = env.reset()
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                state = next_state
                total_reward += reward

            env.close()

        if (episode + 1) % update_target_interval == 0:
            agent.update_target_network()

        print(f"Episode {episode+1} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    agent.save("dqn_multienv.pth")
    print("Model saved.")

if __name__ == "__main__":
    #train()
    train_multi_env()