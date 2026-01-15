#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import os
import sys
import random
import copy
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

# --- PAI IMPORTS ---
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# ==============================================================================
# CONFIGURATION
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
train_start = 100  # Start training after 100 steps of random exploration

# --- PAI SETTINGS ---
# We use try/except to prevent crashes if local PAI version differs slightly
try:
    GPA.pc.set_dendrite_improvement_threshold(0.15) 
    GPA.pc.set_dendrite_improvement_thresholdRaw(1e-4)
except:
    pass

GPA.pc.set_max_dendrites(5)
GPA.pc.set_testing_dendrite_capacity(False)
GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)

# ==============================================================================
# NEURAL NETWORK
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
# DQN AGENT
# ==============================================================================
class DQNAgent:
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
        self.last_loss = 0.0 # Track loss for TensorBoard

        self.memory = deque(maxlen=memory_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Agent running on: {self.device}")
        
        # --- MODEL SETUP ---
        self.model = DQN(state_size, action_size).to(self.device)
        
        # 1. Initialize PAI
        self.model = UPA.initialize_pai(self.model)
        
        # 2. Warmup: Force PAI to initialize buffers
        # This prevents the "Unexpected key" error by creating shapes now
        print("🔥 Warming up PAI layers...")
        with torch.no_grad():
            dummy_input = torch.zeros(1, state_size).to(self.device)
            self.model(dummy_input)

        # 3. Create Target Network 
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        
        # 4. Setup Optimizer
        GPA.pai_tracker.set_optimizer(optim.Adam)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        GPA.pai_tracker.set_optimizer_instance(self.optimizer)

    def update_target_model(self):
        # CRITICAL FIX: strict=False ignores PAI metadata keys that cause crashes
        self.target_model.load_state_dict(self.model.state_dict(), strict=False)

    def full_target_sync(self):
        # Called when PAI adds dendrites. Hard copy the new structure.
        print("⚡ Structure Changed: Hard Syncing Target Network...")
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()

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

        # Get Q values
        curr_q = self.model(states).gather(1, actions)
        
        # Get Target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.discount_factor * next_q

        # Loss and Optimize
        loss = F.mse_loss(curr_q, target_q)
        self.last_loss = loss.item() # Save for TensorBoard

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
    
    # 1. Setup TensorBoard
    log_dir = "runs/dqn_pai_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard logging to: {log_dir}")
    print("   To view: tensorboard --logdir=runs")

    # 2. Setup Environment and Agent
    env = RLEnvironment()
    state_size = 26 
    action_size = 5
    agent = DQNAgent(state_size, action_size)
    
    # Ensure save directory exists
    if not os.path.exists("./save_model"):
        os.makedirs("./save_model")
    
    episodes = 5000
    score_history = []
    pai_check_interval = 20 
    
    print("Starting Training Loop (Dendritic DQN)...")
    
    try:
        for e in range(episodes):
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
            
            # End of Episode
            agent.update_target_model()
            
            score_history.append(score)
            avg_score = np.mean(score_history[-10:])
            
            # --- TENSORBOARD LOGGING ---
            writer.add_scalar('Reward/Score', score, e)
            writer.add_scalar('Reward/Average', avg_score, e)
            writer.add_scalar('Epsilon', agent.epsilon, e)
            writer.add_scalar('Loss', agent.last_loss, e)
            writer.flush()
            
            print(f"Episode: {e+1} | Score: {score:.2f} | Avg: {avg_score:.2f} | Epsilon: {agent.epsilon:.2f}")

            # --- PAI LOGIC ---
            if (e + 1) % pai_check_interval == 0 and len(agent.memory) >= agent.train_start:
                print(f"🔍 PAI Check at Episode {e+1}...")
                
                agent.model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
                    avg_score, 
                    agent.model,
                    "turtlebot_dqn_pai"
                )
                
                if restructured:
                    print("🧠 Dendrites Added! Restructuring Optimizer and Target Net.")
                    # 1. New Optimizer
                    agent.optimizer = optim.Adam(agent.model.parameters(), lr=agent.learning_rate)
                    GPA.pai_tracker.set_optimizer_instance(agent.optimizer)
                    
                    # 2. Hard Sync Target Net (New Architecture)
                    agent.full_target_sync()
                    
                    # 3. Ensure Device Placement
                    agent.model.to(agent.device)
                    agent.target_model.to(agent.device)
                    
            elif (e + 1) % pai_check_interval == 0:
                 print(f"⚠️ Skipping PAI Check: Not enough data yet.")

            # Save Checkpoint every 50 episodes
            if (e + 1) % 50 == 0:
                agent.save_model(f"./save_model/agent_checkpoint_{e+1}.pth")

    except KeyboardInterrupt:
        print("\n\n[!] Ctrl+C Detected! Saving Model before exiting...")
        agent.save_model("./save_model/final_model_interrupted.pth")
        print("[+] Model saved to ./save_model/final_model_interrupted.pth")
        
        writer.close()
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()