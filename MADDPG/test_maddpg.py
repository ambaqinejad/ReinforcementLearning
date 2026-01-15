import numpy as np
import torch

from maddpg_env import MultiAgentQuadEnv
from maddpg import MADDPG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OBS_DIM = 18
ACTION_DIM = 3

env = MultiAgentQuadEnv(
    render=True,
    n_agents=3,
    num_obs=6,
    max_steps=400
)

STATE_DIM = env.get_state().shape[0]

maddpg = MADDPG(
    n_agents=3,
    obs_dim=OBS_DIM,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    device=DEVICE
)

maddpg.load("models/maddpg_final.pth")

while True:
    obs, _ = env.reset()
    state = env.get_state()

    for step in range(500):
        actions = maddpg.select_actions(obs, noise=0.0)  # بدون نویز
        obs, rewards, terminated, truncated, _ = env.step(actions)

        env.render()

        if terminated or truncated:
            break
