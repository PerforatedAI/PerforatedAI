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
import csv  # Added for detailed logging
from torch.utils.tensorboard import SummaryWriter

# --- IMPORTS FROM YOUR ENV FILE ---
from dqn_environment import RLEnvironment 

# --- PAI IMPORTS ---
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# ==============================================================================
# ⚙️ CONFIGURATION FOR LONG RUN (1000 EPOCHS)
# ==============================================================================
state_size = 26
action_size = 5
batch_size = 64
learning_rate = 0.00025
discount_factor = 0.99
epsilon_init = 1.0
epsilon_decay = 0.995  # 
epsilon_min = 0.05
memory_size = 100000
train_start = 1000     # Increased warmup for stability
NUM_EPOCHS = 1000      # <--- TARGET: 1000 EPOCHS

# --- PAI SETTINGS ---
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
        self.last_loss = 0.0

        self.memory = deque(maxlen=memory_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Agent running on: {self.device}")
        
        # --- MODEL SETUP ---
        self.model = DQN(state_size, action_size).to(self.device)
        self.model = UPA.initialize_pai(self.model)
        
        # Warmup
        print("🔥 Warming up PAI layers...")
        with torch.no_grad():
            dummy_input = torch.zeros(1, state_size).to(self.device)
            self.model(dummy_input)

        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        
        GPA.pai_tracker.set_optimizer(optim.Adam)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        GPA.pai_tracker.set_optimizer_instance(self.optimizer)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict(), strict=False)

    def full_target_sync(self):
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

        curr_q = self.model(states).gather(1, actions)
        
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
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
# MAIN LOOP (MODIFIED FOR 1000 EPOCHS)
# ==============================================================================
def main():
    rclpy.init(args=None)
    
    # 1. Setup Logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "runs/dqn_pai_" + timestamp
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard logging to: {log_dir}")
    
    # Custom CSV Logger for "Step-Level" Detail
    csv_filename = f"results/dqn_pai_{timestamp}.csv"
    if not os.path.exists("results"): os.makedirs("results")
    
    with open(csv_filename, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Wall time', 'Step', 'Value']) # Header

    env = RLEnvironment()
    agent = DQNAgent(state_size, action_size)
    
    if not os.path.exists("./save_model"):
        os.makedirs("./save_model")
    
    score_history = []
    pai_check_interval = 20
    global_step_counter = 0  # Track total steps for CSV logging
    
    print(f"🚀 Starting Long Training Run: {NUM_EPOCHS} Episodes")
    
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
                global_step_counter += 1
            
            # --- End of Episode ---
            agent.update_target_model()
            score_history.append(score)
            avg_score = np.mean(score_history[-10:])
            
            # 1. Log to TensorBoard
            writer.add_scalar('Reward/Score', score, e)
            writer.add_scalar('Reward/Average', avg_score, e)
            writer.add_scalar('Epsilon', agent.epsilon, e)
            writer.add_scalar('Loss', agent.last_loss, e)
            writer.flush()

            # 2. Log to CSV (Append Mode) for Professional Plots
            with open(csv_filename, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                # Logging Reward per Episode (using 'e' as Step for this specific csv)
                csv_writer.writerow([datetime.datetime.now().timestamp(), e, score])
            
            print(f"Episode: {e+1}/{NUM_EPOCHS} | Score: {score:.2f} | Avg: {avg_score:.2f} | Epsilon: {agent.epsilon:.2f}")

            # --- PAI LOGIC (Growth Check) ---
            if (e + 1) % pai_check_interval == 0 and len(agent.memory) >= agent.train_start:
                print(f"🔍 PAI Check at Episode {e+1}...")
                
                agent.model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
                    avg_score, 
                    agent.model,
                    "turtlebot_dqn_pai"
                )
                
                if restructured:
                    print(f"🧠 Dendrites Added! Active Params: {sum(p.numel() for p in agent.model.parameters())}")
                    agent.optimizer = optim.Adam(agent.model.parameters(), lr=agent.learning_rate)
                    GPA.pai_tracker.set_optimizer_instance(agent.optimizer)
                    agent.full_target_sync()
                    agent.model.to(agent.device)
                    agent.target_model.to(agent.device)

            # Checkpoint
            if (e + 1) % 50 == 0:
                agent.save_model(f"./save_model/agent_checkpoint_{e+1}.pth")

    except KeyboardInterrupt:
        print("\n[!] Interrupted! Saving Emergency Model...")
        agent.save_model("./save_model/final_model_interrupted.pth")
        
        writer.close()
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()