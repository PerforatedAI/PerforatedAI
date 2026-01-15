#!/usr/bin/env python3
import rclpy
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import datetime
from torch.utils.tensorboard import SummaryWriter

# --- IMPORTS FROM YOUR ENV FILE ---
# Ensure dqn_environment.py is in the same folder
from dqn_environment import RLEnvironment 

# ==============================================================================
# CONFIGURATION (MATCHING PAI FOR FAIRNESS)
# ==============================================================================
state_size = 26
action_size = 5
batch_size = 64
learning_rate = 0.00025
discount_factor = 0.99
epsilon_init = 1.0
epsilon_decay = 0.99
epsilon_min = 0.05
memory_size = 100000
train_start = 1000

# ==============================================================================
# STANDARD NEURAL NETWORK (FIXED ARCHITECTURE)
# ==============================================================================
class StandardDQN(nn.Module):
    def __init__(self):
        super(StandardDQN, self).__init__()
        # Standard Fixed Architecture (Simulating a "Static" Brain)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ==============================================================================
# STANDARD DQN AGENT
# ==============================================================================
class DQNAgentStandard:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚠️  Standard DQN Agent running on: {self.device}")

        self.model = StandardDQN().to(self.device)
        self.target_model = StandardDQN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() 

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon_init

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < train_start:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # -----------------------------------------------------------
        # STANDARD DQN LOGIC (The "Flawed" Teacher)
        # -----------------------------------------------------------
        # In Standard DQN, we take the max Q-value directly from the Target Network.
        # This causes "Overestimation Bias" because it assumes the best possible 
        # outcome is guaranteed, which isn't true in stochastic environments.
        
        next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (discount_factor * next_q * (1 - dones))
        
        # -----------------------------------------------------------

        # Get Current Q (Predicted)
        current_q = self.model(states).gather(1, actions)
        
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay Epsilon
        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

# ==============================================================================
# MAIN LOOP
# ==============================================================================
def main():
    rclpy.init(args=None)
    
    # DISTINCT LOG NAME FOR TENSORBOARD
    log_dir = "runs/STANDARD_DQN_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)
    print(f"📊 Logging Standard DQN to: {log_dir}")

    env = RLEnvironment()
    agent = DQNAgentStandard()
    
    if not os.path.exists("./save_model"):
        os.makedirs("./save_model")

    scores = deque(maxlen=10)
    
    print("Starting Standard DQN Training...")

    try:
        # Run for 500 episodes (same as PAI and Double)
        for e in range(500):
            state = env.reset()
            done = False
            score = 0
            
            while not done:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                
                agent.append_sample(state, action, reward, next_state, done)
                
                if len(agent.memory) >= train_start:
                    agent.train_model()

                score += reward
                state = next_state

            # End of Episode
            agent.update_target_model()
            
            scores.append(score)
            avg_score = np.mean(scores)

            writer.add_scalar('Reward/Score', score, e)
            writer.add_scalar('Reward/Average', avg_score, e)
            writer.add_scalar('Epsilon', agent.epsilon, e)
            writer.flush()

            print(f"STD DQN | Ep: {e+1} | Score: {score:.2f} | Avg: {avg_score:.2f} | Eps: {agent.epsilon:.2f}")

            if (e + 1) % 50 == 0:
                agent.save_model(f"./save_model/standard_dqn_agent_{e+1}.pth")

    except KeyboardInterrupt:
        print("\n\n[!] Ctrl+C Detected! Saving Model...")
        agent.save_model("./save_model/standard_dqn_interrupted.pth")
        writer.close()
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()