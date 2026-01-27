# train_attention_quad.py

import os
import numpy as np
import torch

from environment import MultiAgentQuadEnv  # محیط جدید با موانع و local obs
from maddpg_attention import MADDPGAttention
from roles import ROLE_SCOUT, ROLE_SURROUND, ROLE_ATTACKER

# ======================================================
# Directories
# ======================================================
os.makedirs("models", exist_ok=True)

# ======================================================
# Agents configuration
# ======================================================
N_AGENTS = 9  # 4 شناسایی + 4 محاصره + 1 مهاجم
roles = [
    ROLE_SCOUT, ROLE_SCOUT, ROLE_SCOUT, ROLE_SCOUT,  # scouts
    ROLE_SURROUND, ROLE_SURROUND, ROLE_SURROUND, ROLE_SURROUND,  # surrounders
    ROLE_ATTACKER  # attacker
]

# ======================================================
# Training hyperparameters
# ======================================================
MAX_EPISODES = 2000
MAX_STEPS = 300
BATCH_SIZE = 256

LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
GAMMA = 0.95
TAU = 0.005

NOISE_START = 0.3
NOISE_END = 0.05
NOISE_DECAY = 0.9995

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# Environment
# ======================================================
env = MultiAgentQuadEnv(
    # n_agents=N_AGENTS,
    max_steps=MAX_STEPS,
    render=True
)

obs, _ = env.reset()

OBS_DIM = obs.shape[1]
STATE_DIM = env.get_state().shape[0]

# تعداد ویژگی‌های موانع در observation (مثلاً نزدیک‌ترین 5 مانع × 5 مختصات)
OBSTACLE_FEAT_DIM = env.max_obstacles * 5

# تعداد ویژگی‌های همسایه ها در observation (مثلاً نزدیک‌ترین 4 مانع × 3+6)
NEIGHBOR_FEAT_DIM = env.max_neighbors * (6 + 3)

ACTION_DIM = 3

print(f"OBS_DIM={OBS_DIM}, STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}, N_AGENTS={N_AGENTS}")

# ======================================================
# MADDPGAttention
# ======================================================
maddpg = MADDPGAttention(
    n_agents=N_AGENTS,
    obs_dim=OBS_DIM,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    obstacle_feat_dim=OBSTACLE_FEAT_DIM,  # اضافه شده
    neighbor_feat_dim=NEIGHBOR_FEAT_DIM,
    roles=roles,
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

for episode in range(MAX_EPISODES):

    render_episode = episode % 50 == 0
    obs, _ = env.reset()
    state = env.get_state()

    episode_reward = np.zeros(N_AGENTS)

    for step in range(MAX_STEPS):
        actions, messages = maddpg.select_actions(obs, noise=noise)

        next_obs, rewards, terminated, truncated, _ = env.step(actions)
        next_state = env.get_state()
        done = terminated or truncated

        maddpg.replay_buffer.push(
            state, obs, actions, messages, rewards, next_state, next_obs, done
        )

        maddpg.update()

        obs = next_obs
        state = next_state
        episode_reward += rewards

        if render_episode:
            env.render()

        if done:
            break

    # ---------------- Noise decay ----------------
    noise = max(NOISE_END, noise * NOISE_DECAY)

    # ---------------- Logging ----------------
    if episode % 10 == 0:
        print(f"Ep {episode:04d} | Reward: {episode_reward} | Noise: {noise:.3f}")

    # ---------------- Checkpoint ----------------
    if episode % 100 == 0:
        maddpg.save(f"models/maddpg_attention_ep{episode}.pth")

# ======================================================
# Save final model
# ======================================================
maddpg.save("models/maddpg_attention_final.pth")
print("Training finished.")
