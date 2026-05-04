#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import datetime
import csv
from torch.utils.tensorboard import SummaryWriter

# --- ENV IMPORT ---
from dqn_environment import RLEnvironment 

# ==============================================================================
# ⚙️ CONFIGURATION (MATCHING PAI FOR FAIRNESS)
# ==============================================================================
state_size = 26
action_size = 5
batch_size = 64
learning_rate = 0.00025
discount_factor = 0.99
epsilon_init = 1.0
epsilon_decay = 0.995   # Matching PAI's slow decay
epsilon_min = 0.05
memory_size = 100000
train_start = 1000      # Matching PAI's warmup
NUM_EPOCHS = 1000       # Target 1000 Epochs

# ==============================================================================
# NETWORK (STATIC - NO PAI)
# ==============================================================================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ==============================================================================
# DOUBLE DQN AGENT
# ==============================================================================
class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.train_start = train_start
        self.last_loss = 0.0

        self.memory = deque(maxlen=memory_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Double DQN running on: {self.device}")
        
        # Two distinct networks
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state)
            return q_values.argmax().item()

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.train_start:
            return

        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        actions = torch.LongTensor(np.array([x[1] for x in batch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([x[2] for x in batch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([x[4] for x in batch])).unsqueeze(1).to(self.device)

        # --- DOUBLE DQN LOGIC ---
        # 1. Select best action using ONLINE model
        curr_q = self.model(states).gather(1, actions)
        next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
        
        # 2. Evaluate that action using TARGET model
        with torch.no_grad():
            next_q = self.target_model(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.discount_factor * next_q

        loss = F.mse_loss(curr_q, target_q)
        self.last_loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

# ==============================================================================
# MAIN LOOP
# ==============================================================================
def main():
    rclpy.init(args=None)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "runs/dqn_double_" + timestamp
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard: {log_dir}")
    
    # CSV Logger
    csv_filename = f"results/DOUBLE_DQN_{timestamp}.csv"
    if not os.path.exists("results"): os.makedirs("results")
    with open(csv_filename, 'w', newline='') as f:
        csv.writer(f).writerow(['Wall time', 'Step', 'Value'])

    env = RLEnvironment()
    agent = DoubleDQNAgent(state_size, action_size)
    
    if not os.path.exists("./save_model"): os.makedirs("./save_model")
    
    score_history = []
    
    print(f"🚀 Starting Double DQN: {NUM_EPOCHS} Episodes")
    
    try:
        for e in range(NUM_EPOCHS):
            done = False
            score = 0
            state = env.reset()
            
            while not done:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.append_sample(state, action, reward, next_state, done)
                agent.train_model()
                score += reward
                state = next_state
            
            agent.update_target_model()
            score_history.append(score)
            avg_score = np.mean(score_history[-10:])
            
            # Logs
            writer.add_scalar('Reward/Score', score, e)
            writer.add_scalar('Reward/Average', avg_score, e)
            writer.add_scalar('Epsilon', agent.epsilon, e)
            writer.flush()
            
            with open(csv_filename, 'a', newline='') as f:
                csv.writer(f).writerow([datetime.datetime.now().timestamp(), e, score])
            
            print(f"Double DQN | Ep: {e+1} | Score: {score:.2f} | Avg: {avg_score:.2f} | Eps: {agent.epsilon:.2f}")

            if (e + 1) % 50 == 0:
                agent.save_model(f"./save_model/double_checkpoint_{e+1}.pth")

    except KeyboardInterrupt:
        print("\nStopping Double DQN...")
        agent.save_model("./save_model/double_final.pth")
        writer.close()
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()