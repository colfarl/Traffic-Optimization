import sys
import traci
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from agents.dqn_agent import DQNAgent
from train import TrafficEnvironment
import os
import time

def visualize_agent(sumo_config_path, model_path):
    try:
        # Initialize environment and load agent
        env = TrafficEnvironment(sumo_config_path)
        agent = DQNAgent(env.state_size, env.action_size)
        agent.load(model_path)
        
        # Set up visualization
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle('Traffic Signal Control Visualization')
        
        # Initialize data arrays
        times = []
        queue_lengths = []
        waiting_times = []
        rewards = []
        
        def update(frame):
            try:
                # Clear previous plots
                ax1.clear()
                ax2.clear()
                
                # Get current state and take action
                state = env.get_state()
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                # Update data arrays
                times.append(traci.simulation.getTime())
                queue_lengths.append(np.sum(state))
                waiting_times.append(env.get_reward())
                rewards.append(reward)
                
                # Plot queue lengths
                ax1.plot(times, queue_lengths, 'b-', label='Total Queue Length')
                ax1.set_title('Queue Length Over Time')
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Queue Length')
                ax1.legend()
                
                # Plot waiting times and rewards
                ax2.plot(times, waiting_times, 'r-', label='Waiting Time')
                ax2.plot(times, rewards, 'g-', label='Reward')
                ax2.set_title('Waiting Time and Reward')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Value')
                ax2.legend()
                
                plt.tight_layout()
                
                if done:
                    ani.event_source.stop()
                    plt.close()
                    traci.close()
            except Exception as e:
                print(f"Error in visualization update: {str(e)}")
                ani.event_source.stop()
                plt.close()
                traci.close()
        
        # Start SUMO simulation with GUI
        traci.start(["sumo-gui", "-c", sumo_config_path])
        
        # Create animation
        ani = FuncAnimation(fig, update, interval=100)
        plt.show()
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        try:
            traci.close()
        except:
            pass
    finally:
        try:
            traci.close()
        except:
            pass

def main():
    if len(sys.argv) != 3:
        print("Usage: python visualize.py <sumo_config_path> <model_path>")
        sys.exit(1)
    
    sumo_config_path = sys.argv[1]
    model_path = sys.argv[2]
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        sys.exit(1)
    
    visualize_agent(sumo_config_path, model_path)

if __name__ == "__main__":
    main() 