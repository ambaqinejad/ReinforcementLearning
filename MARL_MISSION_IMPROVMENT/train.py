# train_attention_quad.py

import os
import numpy as np
import torch

from environment import MultiAgentQuadEnv  # محیط جدید با موانع و local obs
from simple_maddpg import MADDPG
from roles import ROLE_SCOUT, ROLE_SURROUND, ROLE_ATTACKER, ROLE_GOAL

# ======================================================
# Directories
# ======================================================
os.makedirs("models", exist_ok=True)

# ======================================================
# Agents configuration
# ======================================================
N_SCOUT = 4
N_SURROUND = 4
N_ATTACKER = 1
N_GOAL = 1

RENDER_TRAIN = True
RENDER_EVERY = 50
RENDER_STEPS = 200

roles = [ROLE_SCOUT for _ in range(N_SCOUT)]
roles.extend([ROLE_SURROUND for _ in range(N_SURROUND)])
roles.extend([ROLE_ATTACKER for _ in range(N_ATTACKER)])
roles.extend([ROLE_GOAL for _ in range(N_GOAL)])
N_AGENTS = len(roles)

# ======================================================
# Training hyperparameters
# ======================================================
MAX_EPISODES = 2000
MAX_STEPS = 500
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
env = MultiAgentQuadEnv(render=True,
                        roles=roles,
                        n_scout=N_SCOUT,
                        n_surround=N_SURROUND,
                        n_attacker=N_ATTACKER,
                        max_steps=MAX_STEPS)


obs, _ = env.reset()

OBS_DIM = obs.shape[1]
STATE_DIM = env.get_state().shape[0]

ACTION_DIM = 3

print(f"OBS_DIM={OBS_DIM}, STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}, N_AGENTS={N_AGENTS}")

# ======================================================
# MADDPGAttention
# ======================================================
maddpg = MADDPG(
    n_agents=N_AGENTS,
    obs_dim=OBS_DIM,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
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
    print(f"EPISODE={episode}")

    render_episode = RENDER_TRAIN and (episode % RENDER_EVERY == 0)
    obs, _ = env.reset()
    state = env.get_state()

    episode_reward = np.zeros(N_AGENTS)

    for step in range(MAX_STEPS):
        # print(f"STEP={step}")

        stop_surrounder = True
        stop_attacker = True

        actions = maddpg.select_actions(
            obs,
            noise_pursuer=noise,
            noise_evader=noise * 0.5,
            stop_surrounder=stop_surrounder, stop_attacker=stop_attacker
        )

        next_obs, rewards, terminated, truncated, _ = env.step(actions)
        next_state = env.get_state()

        done = terminated or truncated

        maddpg.replay_buffer.push(
            state,
            obs,
            actions,
            rewards,
            next_state,
            next_obs,
            done
        )

        maddpg.update()

        obs = next_obs
        state = next_state
        episode_reward += rewards

        if render_episode and step < RENDER_STEPS:
            env.render()

        if done:
            break

    # ---------------- Noise decay ----------------
    noise = max(NOISE_END, noise * NOISE_DECAY)

    # ---------------- Logging ----------------
    if episode % 10 == 0:
        print(f"Ep {episode:04d} | Reward: {episode_reward} | Noise: {noise:.3f}")

    # ---------------- Checkpoint ----------------
    if episode % 50 == 0:
        maddpg.save(f"models/maddpg_attention_ep{episode}.pth")

# ======================================================
# Save final model
# ======================================================
maddpg.save("models/maddpg_attention_final.pth")
print("Training finished.")
