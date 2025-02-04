"""Training script for robotic manipulation tasks using RL algorithms (current: SAC)."""

import os
import argparse
from datetime import datetime

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import HerReplayBuffer, DDPG, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

# configuration
CONFIG = {
    "env_id": "FetchPickAndPlace-v4",
    "seed": 42,
    "model_class": "SAC",
    "total_timesteps": 2_000_000,
    "log_dir": "./logs",
    # model hyperparameters
    "buffer_size": 1_000_000,
    "batch_size": 1024,
    "gamma": 0.95,
    "tau": 0.05,
    "learning_rate": 0.001,
    "n_sampled_goal": 4,
    "goal_selection_strategy": "future",
    "action_noise_sigma": 0.1,
    "n_critics": 2,
    "policy_net_arch": [512, 512, 512],
    "eval_freq": 50_000,
    "checkpoint_freq": 200_000,
}

# argument parser for flexibility
def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent for FetchPush environment.")
    parser.add_argument("--model", type=str, default=CONFIG["model_class"], choices=["DDPG", "TD3", "SAC"], help="RL model to use (DDPG, TD3, SAC)")
    parser.add_argument("--env", type=str, default=CONFIG["env_id"], help="Gymnasium environment ID")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="Seed for reproducibility")
    parser.add_argument("--timesteps", type=int, default=CONFIG["total_timesteps"], help="Total training timesteps")
    parser.add_argument("--log_dir", type=str, default=CONFIG["log_dir"], help="Base directory for logs")
    return parser.parse_args()

args = parse_args()

# update config with args
CONFIG.update({
    "model_class": args.model,
    "env_id": args.env,
    "seed": args.seed,
    "total_timesteps": args.timesteps,
    "log_dir": args.log_dir,
})

# organizing logs
env_name = CONFIG["env_id"]
timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

# create base directories
base_dir = os.path.join(CONFIG["log_dir"], env_name)
run_dir = os.path.join(base_dir, f"{CONFIG['model_class']}_{timestamp}")

# set up directory structure
CONFIG.update({
    "checkpoint_dir": os.path.join(run_dir, "checkpoints"),
    "tensorboard_log_dir": os.path.join(base_dir, "tensorboard"),
})

# setup logging directories
for dir_path in [CONFIG["checkpoint_dir"], 
                CONFIG["tensorboard_log_dir"]]:
    os.makedirs(dir_path, exist_ok=True)

# environment setup
gym.register_envs(gymnasium_robotics)
env = gym.make(CONFIG["env_id"])
env.reset(seed=CONFIG["seed"])
env.action_space.seed(CONFIG["seed"]) # Seed action space for more deterministic behavior

# callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=CONFIG["checkpoint_freq"], save_path=CONFIG["checkpoint_dir"]
)
eval_callback = EvalCallback(
    env,
    best_model_save_path=run_dir,
    log_path=run_dir,
    eval_freq=CONFIG["eval_freq"],
)
callback = CallbackList([checkpoint_callback, eval_callback])

# model setup
model_class = {
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": SAC,
}[CONFIG["model_class"]]

# action noise
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=CONFIG["action_noise_sigma"] * np.ones(n_actions)
)

# model initialization
model = model_class(
    "MultiInputPolicy",
    env=env,
    buffer_size=CONFIG["buffer_size"],
    batch_size=CONFIG["batch_size"],
    gamma=CONFIG["gamma"],
    tau=CONFIG["tau"],
    learning_rate=CONFIG["learning_rate"],
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=CONFIG["n_sampled_goal"],
        goal_selection_strategy=CONFIG["goal_selection_strategy"],
    ),
    verbose=1,
    action_noise=action_noise,
    tensorboard_log=CONFIG["tensorboard_log_dir"],
    policy_kwargs=dict(n_critics=CONFIG["n_critics"], net_arch=CONFIG["policy_net_arch"]),
    seed=CONFIG["seed"]
)

# training loop
try:
    model.learn(total_timesteps=CONFIG["total_timesteps"], callback=callback)
    print("\nTraining completed. Saving model...")

except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving model...")

finally:
    # saving model and replay buffer
    model_name = f"{env_name}"
    model_path = os.path.join(run_dir, model_name)
    model.save(model_path)
    model.save_replay_buffer(f"{model_path}_buffer")
    print(f"Model and replay buffer saved to: {run_dir}")

    # cleanup
    env.close()
