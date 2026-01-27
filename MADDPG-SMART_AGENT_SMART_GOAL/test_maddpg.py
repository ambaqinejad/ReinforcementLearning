import numpy as np
import torch

from env_adversarial_quad import MultiAgentQuadEnv
from maddpg import MADDPG

# ======================================================
# Device
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# Agent configuration
# ======================================================
N_PURSUERS = 3
N_EVADERS = 1
N_AGENTS = N_PURSUERS + N_EVADERS

MAX_STEPS = 400

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
# Load MADDPG
# ======================================================
maddpg = MADDPG(
    n_pursuers=N_PURSUERS,
    obs_dim=OBS_DIM,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    device=DEVICE
)

maddpg.load("models/maddpg_final.pth")
print("Loaded pretrained MADDPG model.")

# ======================================================
# Run test loop
# ======================================================
while True:
    obs, _ = env.reset()
    state = env.get_state()

    for step in range(MAX_STEPS):
        # بدون نویز در تست
        actions = maddpg.select_actions(obs, noise_pursuer=0.0,noise_evader=0.0)

        next_obs, rewards, terminated, truncated, _ = env.step(actions)
        next_state = env.get_state()

        print(f"Step {step:03d} | Rewards: {rewards}")

        obs = next_obs
        state = next_state

        env.render()

        if terminated or truncated:
            print("Episode finished.")
            break
