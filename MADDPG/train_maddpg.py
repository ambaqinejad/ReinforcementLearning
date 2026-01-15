import numpy as np
import torch

from maddpg_env import MultiAgentQuadEnv
from maddpg import MADDPG

import os
os.makedirs("models", exist_ok=True)

# ======================================================
# Hyperparameters (عمداً واضح و قابل تنظیم)
# ======================================================
N_AGENTS = 3
OBS_DIM = 18
ACTION_DIM = 3

MAX_EPISODES = 2000
MAX_STEPS = 300
BATCH_SIZE = 256

LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
GAMMA = 0.95
TAU = 0.01

NOISE_START = 0.3
NOISE_END = 0.05
NOISE_DECAY = 0.9995

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RENDER_TRAIN = True
RENDER_EVERY = 50      # هر 50 اپیزود
RENDER_STEPS = 200    # حداکثر چند step نمایش داده شود

# ======================================================
# Environment
# ======================================================
env = MultiAgentQuadEnv(
    render=True,
    n_agents=N_AGENTS,
    num_obs=6,
    max_steps=MAX_STEPS
)

STATE_DIM = env.get_state().shape[0]

# ======================================================
# MADDPG
# ======================================================
maddpg = MADDPG(
    n_agents=N_AGENTS,
    obs_dim=OBS_DIM,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    lr_actor=LR_ACTOR,
    lr_critic=LR_CRITIC,
    gamma=GAMMA,
    tau=TAU,
    batch_size=BATCH_SIZE,
    device=DEVICE
)

# ======================================================
# Training Loop
# ======================================================
noise = NOISE_START
episode_rewards = []

for episode in range(MAX_EPISODES):

    render_episode = RENDER_TRAIN and (episode % RENDER_EVERY == 0)

    obs, _ = env.reset()
    state = env.get_state()

    total_reward = np.zeros(N_AGENTS)

    for step in range(MAX_STEPS):

        actions = maddpg.select_actions(obs, noise=noise)

        next_obs, rewards, terminated, truncated, _ = env.step(actions)
        next_state = env.get_state()

        done = terminated or truncated

        maddpg.replay_buffer.push(
            state, obs, actions, rewards,
            next_state, next_obs, done
        )

        maddpg.update()

        obs = next_obs
        state = next_state
        total_reward += rewards

        # -------------------------------
        # Render (فقط بعضی اپیزودها)
        # -------------------------------
        if render_episode and step < RENDER_STEPS:
            env.render()

        if done:
            break
    if episode % 100 == 0:
        maddpg.save(f"models/maddpg_ep{episode}.pth")


    # ------------------------------------------------------
    # Noise decay (خیلی مهم)
    # ------------------------------------------------------
    noise = max(NOISE_END, noise * NOISE_DECAY)

    episode_rewards.append(total_reward.mean())

    # ------------------------------------------------------
    # Logging
    # ------------------------------------------------------
    if episode % 10 == 0:
        print(
            f"Episode {episode:04d} | "
            f"Avg Reward: {total_reward.mean():.2f} | "
            f"Noise: {noise:.3f}"
        )

maddpg.save("models/maddpg_final.pth")
print("Training finished.")
