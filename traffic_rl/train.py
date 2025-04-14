# train.py

import os
import sys
import traci
import numpy as np
from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt
from datetime import datetime
import time

from env.traffic_env import TrafficEnv

class TrafficEnvironment:
    def __init__(self, sumo_config_path):
        self.sumo_config_path = sumo_config_path
        self.state_size = 8  # Queue lengths for 4 approaches, 2 lanes each
        self.action_size = 4  # 4 possible signal phases
        self.edges = ['gneE0', 'gneE1', 'gneE2', 'gneE3']  # Correct edge IDs
        
    def get_state(self):
        """Get current state of the intersection"""
        state = []
        # Get queue lengths for each approach
        for edge in self.edges:
            for lane in range(2):  # 2 lanes per approach
                queue_length = len(traci.edge.getLastStepHaltingVehicles(f"{edge}_{lane}"))
                state.append(queue_length)
        return np.array(state)
    
    def get_reward(self):
        """Calculate reward based on waiting time and queue lengths"""
        total_waiting_time = 0
        total_queue_length = 0
        
        for edge in self.edges:
            for lane in range(2):
                waiting_time = traci.edge.getWaitingTime(f"{edge}_{lane}")
                queue_length = len(traci.edge.getLastStepHaltingVehicles(f"{edge}_{lane}"))
                total_waiting_time += waiting_time
                total_queue_length += queue_length
        
        # Negative reward for waiting time and queue length
        reward = -0.1 * total_waiting_time - 0.01 * total_queue_length
        return reward
    
    def step(self, action):
        """Execute one step in the environment"""
        # Set traffic light phase based on action
        traci.trafficlight.setPhase("intersection", action)
        traci.simulationStep()
        
        next_state = self.get_state()
        reward = self.get_reward()
        done = traci.simulation.getTime() >= 3600  # End after 1 hour
        
        return next_state, reward, done

def train_agent(env, agent, episodes=100, batch_size=32, target_update=10):
    """Train the DQN agent"""
    for episode in range(episodes):
        try:
            # Use sumo instead of sumo-gui for training
            traci.start(["sumo", "-c", env.sumo_config_path])
            
            state = env.get_state()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                agent.replay(batch_size)
                
                state = next_state
                total_reward += reward
                
                if traci.simulation.getTime() % 100 == 0:
                    print(f"Time: {traci.simulation.getTime()}, Reward: {reward:.2f}")
            
            agent.reward_history.append(total_reward)
            
            if episode % target_update == 0:
                agent.update_target_model()
            
            traci.close()
            
            print(f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            
            # Save model and plot progress every 10 episodes
            if (episode + 1) % 10 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                agent.save(f"models/dqn_agent_{timestamp}.pt")
                agent.plot_training_progress(f"plots/training_progress_{timestamp}.png")
                
        except Exception as e:
            print(f"Error in episode {episode + 1}: {str(e)}")
            try:
                traci.close()
            except:
                pass
            time.sleep(1)  # Wait before retrying

def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py <sumo_config_path>")
        sys.exit(1)
    
    sumo_config_path = sys.argv[1]
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Initialize environment and agent
    env = TrafficEnvironment(sumo_config_path)
    agent = DQNAgent(env.state_size, env.action_size)
    
    # Train the agent
    train_agent(env, agent, episodes=50, batch_size=32, target_update=10)

if __name__ == "__main__":
    main()

def evaluate_dqn_agent(load_model_path="dqn_model.pth", episodes=1):
    """
    Optional function to load a trained model and run a few evaluation episodes.
    This is helpful to check performance without exploration (epsilon=0).
    """
    # Recreate the same environment config used during training
    config = {
        "sumo_config": "DC_downtown/dayuan.sumocfg",
        "max_steps": 12000,
        "yellow_time": 3,
        "min_green": 10,
        "max_green": 60
    }
    env = TrafficEnv(config)

    # Match agent's expected dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create agent and load weights
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size
    )
    agent.load(load_model_path)

    # Set epsilon to 0 for pure exploitation
    agent.epsilon = 0.0

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward

        print(f"[Eval] Episode {ep+1}/{episodes} finished. Reward = {total_reward:.2f}")

    env.close()
