################################################################################################################
# Author:                                                                                                      #
# Nicholas Mesa-Cucalon (nmesacuc@andrew.cmu.edu)                                                              #
#                                                                                                              #
# Vanilla Twin Delayed Deep Deterministic Policy Gradient (TD3) Implementation                                 #
################################################################################################################

#
"""
Imports
"""
import os
import copy
import torch
import wandb
import imageio
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dm_control import suite

#
"""
Buffer
"""
class Buffer:
    def __init__(self, obs_dim, act_dim, capacity, device):
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim
        self.capacity = capacity
        self.device   = device
        self.reset()

    def reset(self):
        self.ptr  = 0
        self.size = 0

        # preallocate tensors on device
        self.obs        = torch.zeros((self.capacity, self.obs_dim), dtype=torch.float32, device=self.device)
        self.next_obs   = torch.zeros((self.capacity, self.obs_dim), dtype=torch.float32, device=self.device)
        self.actions    = torch.zeros((self.capacity, self.act_dim), dtype=torch.float32, device=self.device)
        self.rewards    = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
        self.dones      = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)

    def add(self, obs, action, next_obs, reward, done):
        i = self.ptr
        self.obs[i]        = obs
        self.next_obs[i]   = next_obs
        self.actions[i]    = action
        self.rewards[i]    = reward
        self.dones[i]      = done

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("The buffer is empty")

        # Randomly sample from buffer
        idx  = torch.arange(self.size, device=self.device)
        perm = torch.randperm(idx.numel(), device=self.device)[:batch_size]
        idx  = idx[perm]

        return dict(
            obs=self.obs[idx],
            next_obs=self.next_obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            dones=self.dones[idx],
        )

#
"""
Twin Delayed DDPG (TD3)
"""

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, act_high):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.act_high = act_high

    def forward(self, state):
        a = self.network(state)
        return self.act_high * torch.tanh(a)

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        critic_factory = lambda : nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.critic1 = critic_factory()
        self.critic2 = critic_factory()

    def forward(self, state_action):
        q1 = self.critic1(state_action)
        q2 = self.critic2(state_action)
        return q1, q2

    def q1(self, state_action):
        return self.critic1(state_action)

class TD3:
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim,
        act_low,
        act_high,
        device,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        delay=2,
        lr=3e-4,
        buffer_size=10000,
        exploration_noise=0.1,
        batch_size=128,
        warmup_steps=5000,
        dendrites=False,
    ):
        self.device = device

        # TD3 Hyperparameters
        self.act_low = act_low
        self.act_high = act_high
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.delay = delay
        self.exploration_noise = exploration_noise
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.lr = lr

        # Training State
        self.total_steps = 0
        self.update_count = 0
        self.total_it = 0

        # Factories
        actor_factory  = lambda: Actor(obs_dim, act_dim, hidden_dim, act_high).to(self.device)
        critic_factory = lambda: Critic(obs_dim, act_dim, hidden_dim).to(self.device)

        # Actor + Critics
        self.actor   = actor_factory()
        self.critic  = critic_factory()

        # Target Actor + Critics
        self.target_actor   = actor_factory()
        self.target_critic  = critic_factory()

        # Initialize target networks with weights from online networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr = lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr = lr)

        # Buffer
        self.buffer = Buffer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            capacity=buffer_size,
            device=device
        )

    def save(self, filename):
        # Critics
        torch.save(self.critic.state_dict(), filename + "_critic.pt")
        torch.save(self.critic_opt.state_dict(), filename + "_critic_opt.pt")

        # Actor
        torch.save(self.actor.state_dict(), filename + "_actor.pt")
        torch.save(self.actor_opt.state_dict(), filename + "_actor_opt.pt")

    def load(self, filename):
        # Critics
        self.critic.load_state_dict(torch.load(filename + "_critic.pt"))
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_opt.load_state_dict(torch.load(filename + "_critic_opt.pt"))

        # Actor
        self.actor.load_state_dict(torch.load(filename + "_actor.pt"))
        self.actor_opt.load_state_dict(torch.load(filename + "_actor_opt.pt"))
        self.target_actor = copy.deepcopy(self.actor)

    @torch.no_grad()
    def act(self, obs):
        return self.actor(obs)

    @torch.no_grad()
    def noisy_act(self, obs):
        # The TD3 Actor is deterministic, so we use Gaussian Noise to help the policy explore
        action = self.actor(obs)
        noise = self.exploration_noise * torch.randn_like(action)
        return torch.clamp(action + noise, self.act_low, self.act_high)

    @torch.no_grad()
    def soft_update(self, local_model, target_model):
        for param, t_param in zip(local_model.parameters(), target_model.parameters()):
            t_param.data = (self.tau * param.data) + ((1 - self.tau) * t_param.data)

    def step(self, transition):
        # Add transition to the buffer
        obs_t = torch.as_tensor(transition["obs"], dtype=torch.float32, device = self.device)
        act_t = torch.as_tensor(transition["act"], dtype=torch.float32, device = self.device)
        next_obs_t = torch.as_tensor(transition["next_obs"], dtype=torch.float32, device = self.device)

        self.buffer.add(obs_t,act_t,next_obs_t,float(transition["reward"]),float(transition["done"]))

        # Update training state
        self.total_steps += 1

        # Check if we can sample from the buffer
        curr_buf_size = self.buffer.size

        if (self.batch_size > curr_buf_size) or (self.warmup_steps > curr_buf_size):
            # Buffer is too small so return no update
            return {}

        #
        """
        TD3 Update
        """
        batch = self.buffer.sample(self.batch_size)
        self.update_count += 1
        do_actor_update = ((self.update_count % self.delay) == 0)

        obs = batch["obs"]
        acts = batch["actions"]
        next_obs = batch["next_obs"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        sa = torch.cat((obs, acts), dim=1)
        current_q1, current_q2 = self.critic(sa)

        # TD3 Target with Policy Smoothing
        with torch.no_grad():
            # Select actions according to target actor and add policy noise for smoothing
            tgt_action = self.target_actor(next_obs)
            noise = torch.clamp(self.policy_noise * torch.randn_like(tgt_action), -self.noise_clip, self.noise_clip)
            tgt_action = torch.clamp(tgt_action + noise, self.act_low, self.act_high)

            # Compute the target Q value
            next_sa  = torch.cat((next_obs, tgt_action), dim=1)
            q1, q2   = self.target_critic(next_sa)
            min_qval = torch.min(q1, q2)
            target_q = rewards + (self.gamma * (1 - dones) * min_qval)

        # Critic Update
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        q_loss = critic1_loss + critic2_loss

        self.critic_opt.zero_grad()
        q_loss.backward()
        self.critic_opt.step()

        # (Delayed) Actor Update
        if do_actor_update:
            actor_sa   = torch.cat((obs, self.actor(obs)), dim=1)
            actor_loss = -1 * self.critic.q1(actor_sa).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Perform soft updates w/ Polyak Averaging
            self.soft_update(self.actor,self.target_actor)
            self.soft_update(self.critic,self.target_critic)
        else:
            with torch.no_grad():
                actor_sa   = torch.cat((obs, self.actor(obs)), dim=1)
                actor_loss = -1 * self.critic.q1(actor_sa).mean()

        # Return stats object for logging
        stats = {
            "actor_loss": float(actor_loss.item()),
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "q1": float(current_q1.mean().item()),
            "q2": float(current_q2.mean().item()),
        }
        return stats

#
"""
Helper Functions
"""
def flatten_obs(obs_dict):
    return np.concatenate([obs.flatten() for obs in obs_dict.values()])

def log_training_stats(stats, log, total_steps, updates_performed):
    if not stats:
        # No update occurred
        return
    avg_return = np.mean(log['episodic_return'][-10:] if log['episodic_return'] else [0])
    print(
        f"[Step {total_steps:>8d}] "
        f"Updates: {updates_performed:>4d} | "
        f"Actor: {log['actor_loss'][-1]:>7.4f} | "
        f"C1: {log['critic1_loss'][-1]:>7.4f} | "
        f"C2: {log['critic2_loss'][-1]:>7.4f} | "
        f"Q1: {log['q1'][-1]:>7.2f} | "
        f"Q2: {log['q2'][-1]:>7.2f} | "
        f"Ret: {avg_return:>7.1f}"
    )

def get_env_action(agent, obs, action_spec, eval):
    obs = torch.from_numpy(obs).to(agent.device, dtype=torch.float32)
    if eval:
        # We only add exploration noise while training
        action = agent.act(obs)
    else:
        action = agent.noisy_act(obs)
    action = np.asarray(action.cpu(), dtype=np.float32).reshape(action_spec.shape)
    a_env = np.asarray(action, dtype=np.float32).reshape(action_spec.shape)
    a_env = np.clip(a_env, action_spec.minimum, action_spec.maximum)
    return action, a_env

def evaluate_policy(agent, domain_name, task_name, episodes, seed):
    scores = []
    for ep in range(episodes):
        env = suite.load(domain_name = domain_name, task_name = task_name, task_kwargs = {'random':seed + ep})
        action_spec = env.action_spec()
        time_step = env.reset()
        
        obs = flatten_obs(time_step.observation)
        ep_ret = 0.0

        while not time_step.last():
            _, a_env = get_env_action(agent, obs, action_spec, True)
            time_step = env.step(a_env)
            ep_ret += time_step.reward
            obs = flatten_obs(time_step.observation)
        scores.append(ep_ret)
        env.close()
    return float(np.mean(scores)), float(np.std(scores))

def record_eval_video(agent, video_dir, video_name, domain_name, task_name, seed, ep_idx):
    os.makedirs(video_dir, exist_ok=True)

    env = suite.load(domain_name = domain_name, task_name = task_name, task_kwargs = {'random':seed + ep_idx})
    action_spec = env.action_spec()
    time_step = env.reset()
    
    video_path = os.path.join(video_dir, f"{video_name}.mp4")
    frames = []

    obs = flatten_obs(time_step.observation)

    while not time_step.last():
        # Render frame
        frame = env.physics.render(height=480, width=640, camera_id=0)
        frames.append(frame)

        # Get action from agent
        _, a_env = get_env_action(agent, obs, action_spec, True)

        # Step environment
        time_step = env.step(a_env)
        obs = flatten_obs(time_step.observation)

    # Save video
    imageio.mimsave(video_path, frames, fps=30)

    return video_path

#
"""
Runner
"""
def run(args):
    #
    """
    Setup
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = suite.load(domain_name = args.domain_name, task_name = args.task_name, task_kwargs = {'random':args.seed})
    action_spec = env.action_spec()

    act_low = torch.from_numpy(copy.deepcopy(action_spec.minimum)).to(device, dtype=torch.float32)
    act_high = torch.from_numpy(copy.deepcopy(action_spec.maximum)).to(device, dtype=torch.float32)

    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "algorithm": "TD3",
            "domain_name": args.domain_name,
            "task_name": args.task_name,
            "total_steps": args.total_steps,
            "seed": args.seed,
            "lr": args.lr,
            "gamma": args.gamma,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "warmup_steps": args.warmup_steps,
            "hidden_dim": args.hidden_dim,
            "tau": args.tau,
            "policy_noise": args.policy_noise,
            "delay": args.delay,
            "noise_clip": args.noise_clip,
            "exploration_noise": args.exploration_noise,
        }
    )


    agent = TD3(
            obs_dim=sum(np.prod(spec.shape) for spec in env.observation_spec().values()),
            act_dim=env.action_spec().shape[0],
            hidden_dim=args.hidden_dim,
            act_low=act_low,
            act_high=act_high,
            device=device,
            gamma=args.gamma,
            tau=args.tau,
            policy_noise=args.policy_noise, # Policy noise helps prevent the policy from overfitting to Q-fn peaks
            noise_clip=args.noise_clip,
            delay=args.delay,
            lr=args.lr,
            buffer_size=args.buffer_size,
            exploration_noise=args.exploration_noise, # Noise to help policy explore
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)

    log = {
        "steps": [],
        "episodic_return": [],
        "eval_mean": [],
        "eval_std": [],
        "eval_steps": [],
        "actor_loss": [],
        "critic1_loss": [],
        "critic2_loss": [],
        "q1": [],
        "q2": [],
    }

    #
    """
    Training Loop
    """
    time_step = env.reset()

    obs = flatten_obs(time_step.observation)
    ep_ret = 0.0
    total_steps = 0
    best_eval = -1e9
    ep_idx = 0
    updates_performed = 0

    print(
        f"Twin Delayed DDPG (TD3) Training: \n"
        f"Target steps:  {args.total_steps:>10d} | "
        f"Batch size:    {args.batch_size:>6d} | "
        f"Buffer size:   {args.buffer_size:>10d} | "
        f"Warmup steps:  {args.warmup_steps:>10d}\n"
        f"{'-' * 80}"
    )

    last_stat_update = 0
    while total_steps < args.total_steps:
        # Get environment ready action
        action, a_env = get_env_action(agent, obs, action_spec, False)

        # Step Environment
        time_step = env.step(a_env)
        next_obs = flatten_obs(time_step.observation)
        stop = time_step.last()
        
        # Create transition dictionary
        transition = {
            "obs": obs.copy(),
            "act": action.copy(),
            "reward": float(time_step.reward),
            "next_obs": next_obs.copy(),
            "done": stop
        }

        # Step the agent
        stats = agent.step(transition)

        # Update state
        ep_ret += time_step.reward
        obs = next_obs
        total_steps += 1

        # Handle episode boundary
        if stop:
            log["steps"].append(total_steps)
            log["episodic_return"].append(ep_ret)

            wandb_log = {
                "episode/return": ep_ret,
                "episode/steps": total_steps,
            }

            # Moving Avg (N=10)
            if len(log["episodic_return"]) >= 10:
                window = min(50, len(log["episodic_return"]) // 10)
                ma = np.mean(log["episodic_return"][-window:])
                wandb_log["episode/return_ma"] = ma
            
            wandb.log(wandb_log, step=total_steps)
            
            # Reset environment + global state
            env.close()
            seed = args.seed + (50 * ep_idx)
            env = suite.load(domain_name = args.domain_name, task_name = args.task_name, task_kwargs = {'random':seed})

            time_step = env.reset()
            obs = flatten_obs(time_step.observation)

            ep_idx += 1
            ep_ret = 0.0

        # Collect optimization stats when update occurred
        if stats:  
            # stats is only populated when an update occurred
            for k, v in stats.items():
                log[k].append(v)

            wandb_log = {
                "train/actor_loss": stats["actor_loss"],
                "train/critic1_loss": stats["critic1_loss"],
                "train/critic2_loss": stats["critic2_loss"],
                "train/q1": stats["q1"],
                "train/q2": stats["q2"],
            }
            
            # Moving Avg (N=20)
            if len(log["actor_loss"]) >= 20:
                window = min(100, len(log["actor_loss"]) // 20)
                wandb_log["train/actor_loss_ma"] = np.mean(log["actor_loss"][-window:])
                wandb_log["train/critic1_loss_ma"] = np.mean(log["critic1_loss"][-window:])
                wandb_log["train/critic2_loss_ma"] = np.mean(log["critic2_loss"][-window:])
                wandb_log["train/q1_ma"] = np.mean(log["q1"][-window:])
                wandb_log["train/q2_ma"] = np.mean(log["q2"][-window:])
        
            wandb.log(wandb_log, step=total_steps)

            if total_steps - last_stat_update >= args.log_every:
                log_training_stats(stats, log, total_steps, updates_performed)
                last_stat_update = total_steps

            updates_performed += 1

        # Periodic evaluation
        if args.eval_every > 0 and total_steps % args.eval_every == 0:
            print(f"\n--- Evaluation at step {total_steps} ---")
            with torch.no_grad():
                mean_r, std_r = evaluate_policy(
                    agent, args.domain_name, args.task_name, args.eval_episodes, 6502
                )
            log["eval_mean"].append(mean_r)
            log["eval_std"].append(std_r)
            log["eval_steps"].append(total_steps)

            wandb.log({
                "eval/mean_return": mean_r,
                "eval/std_return": std_r,
            }, step=total_steps)
            
            print(f"Eval: {mean_r:.1f} ± {std_r:.1f}")
            
            # Save best model
            if mean_r > best_eval:
                best_eval = mean_r
                model_path = os.path.join(args.out_dir, "best")
                # Save model
                agent.save(model_path)

                wandb.run.summary["best_eval_return"] = mean_r
                wandb.run.summary["best_eval_step"] = total_steps

                print(f"New best model saved! Return: {mean_r:.1f}")

            print("--- End Evaluation ---\n")

        # Periodic video recording
        if args.video_every > 0 and total_steps % args.video_every == 0:
            print(f"Recording video at step {total_steps}...")
            name = f"{args.video_prefix}_t{total_steps}"
            try:
                with torch.no_grad():
                    path = record_eval_video(
                        agent, 
                        video_dir=args.video_dir, 
                        video_name=name,
                        domain_name=args.domain_name, 
                        task_name=args.task_name, 
                        seed=4004,
                        ep_idx=ep_idx
                    )
                print(f"Video saved: {path}")
                wandb.log({
                    "video": wandb.Video(path, fps=30, format="mp4")
                }, step=total_steps)
            except Exception as e:
                print(f"Video recording failed: {e}")
                import traceback
                traceback.print_exc()

    # Final evaluation
    if args.eval_episodes > 0:
        if best_eval > -1e9:
            model_path = os.path.join(args.out_dir, "best")
            try:
                agent.load(model_path)
                print(f"Loaded best model from {model_path} for final evaluation")
            except:
                print(f"Best model file not found at {model_path}, using current model for final eval")
        
        print("Final evaluation...")
        with torch.no_grad():
            mean_r, std_r = evaluate_policy(
                    agent, args.domain_name, args.task_name, args.eval_episodes, 68000
                )
        print(f"Final performance: {mean_r:.1f} ± {std_r:.1f}")
        log["eval_mean"].append(mean_r)
        log["eval_std"].append(std_r)
        log["eval_steps"].append(total_steps)

        wandb.log({
            "final_eval/mean_return": mean_r,
            "final_eval/std_return": std_r,
        }, step=total_steps)
        
        wandb.run.summary["final_mean_return"] = mean_r
        wandb.run.summary["final_std_return"] = std_r

    # Final cleanup and saving
    print(f"\nTraining completed! Total steps: {total_steps}")
    print(f"Total updates: {updates_performed - 1}")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Twin Delayed DDPG (TD3)")

    # Environment
    parser.add_argument("--domain_name", type=str, default='cheetah',
                        help="DMControl Environment Name")
    parser.add_argument("--task_name", type=str, default='run',
                        help="Environment Task Name")

    # Training
    parser.add_argument("--total_steps", type=int, default=500000,
                   help="Total environment steps")
    parser.add_argument("--seed", type=int, default=8008,
                   help="Random seed")

    # General Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=128,
                   help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=100000,
                   help="Replay buffer size")
    parser.add_argument("--warmup_steps", type=int, default=5000,
                   help="Steps before starting updates")
    parser.add_argument("--hidden_dim", type=int, default=256,
                   help="Hidden Dimension for Actors+Critics")

    # TD3 Hyperparameters
    parser.add_argument("--tau", type=float, default=0.005,
                   help="Target network soft update rate")
    parser.add_argument("--policy_noise", type=float, default=0.2,
                   help="TD3 target policy smoothing noise std")
    parser.add_argument("--delay", type=float, default=2,
                   help="TD3 target policy delay")
    parser.add_argument("--noise_clip", type=float, default=0.5,
                   help="TD3 target policy noise clip")
    parser.add_argument("--exploration_noise", type=float, default=0.1,
                   help="TD3 exploration noise std during env interaction")

    # Logging
    parser.add_argument("--log_every", type=int, default=10000,
                   help="Print training stats every N steps")

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="vanilla-td3-dmcontrol",
                   help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                   help="Wandb run name (default: auto-generated)")

    # Evaluation
    parser.add_argument("--eval_every", type=int, default=50000,
                   help="Run evaluation every N steps (0 to disable)")
    parser.add_argument("--eval_episodes", type=int, default=100,
                   help="Episodes per evaluation")

    # Video
    parser.add_argument("--video_every", type=int, default=100000,
                   help="Record video every N steps (0 to disable)")
    parser.add_argument("--video_dir", type=str, default="videos",
                   help="Video output directory")
    parser.add_argument("--video_prefix", type=str, default="eval",
                   help="Video filename prefix")

    # Output
    parser.add_argument("--out_dir", type=str, default="runs/rl_training",
                   help="Output directory for models and logs")

    # Debug
    parser.add_argument('--debug', action='store_true', help="Toggles Wandb")


    args = parser.parse_args()
    run(args)