import numpy as np
import torch
import os

from env_adversarial_quad import MultiAgentQuadEnv
from maddpg import MADDPG

# ======================================================
# Directories
# ======================================================
os.makedirs("models", exist_ok=True)

# ======================================================
# Agents configuration
# ======================================================
N_PURSUERS = 3
N_EVADERS = 1
N_AGENTS = N_PURSUERS + N_EVADERS

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

# Rendering
RENDER_TRAIN = True
RENDER_EVERY = 50
RENDER_STEPS = 200

# ======================================================
# Environment
# ======================================================
env = MultiAgentQuadEnv(
    n_pursuers=N_PURSUERS,
    n_evaders=N_EVADERS,
    max_steps=MAX_STEPS,
    render=True
)

obs, _ = env.reset()

OBS_DIM = obs.shape[1]
STATE_DIM = env.get_state().shape[0]
ACTION_DIM = 3

print(f"OBS_DIM={OBS_DIM}, STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}")

# ======================================================
# MADDPG
# ======================================================
maddpg = MADDPG(
    n_pursuers=N_PURSUERS,
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

for episode in range(MAX_EPISODES):

    render_episode = RENDER_TRAIN and (episode % RENDER_EVERY == 0)

    obs, _ = env.reset()
    state = env.get_state()

    episode_reward = np.zeros(N_AGENTS)

    for step in range(MAX_STEPS):

        actions = maddpg.select_actions(
            obs,
            noise_pursuer=noise,
            noise_evader=noise * 0.5
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

    # --------------------------------------------------
    # Noise decay
    # --------------------------------------------------
    noise = max(NOISE_END, noise * NOISE_DECAY)

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    if episode % 10 == 0:
        pursuer_reward = episode_reward[:N_PURSUERS].mean()
        evader_reward = episode_reward[-1]

        print(
            f"Ep {episode:04d} | "
            f"Pursuers: {pursuer_reward:7.2f} | "
            f"Evader: {evader_reward:7.2f} | "
            f"Noise: {noise:.3f}"
        )

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------
    if episode % 100 == 0:
        maddpg.save(f"models/maddpg_ep{episode}.pth")

# ======================================================
# Save final model
# ======================================================
maddpg.save("models/maddpg_final.pth")
print("Training finished.")
