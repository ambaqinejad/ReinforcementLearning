import numpy as np
import random


class ReplayBuffer:
    """
    Replay Buffer for MADDPG
    Stores full joint transitions
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(
        self,
        state,        # (state_dim,)
        obs,          # (n_agents, obs_dim)
        actions,      # (n_agents, action_dim)
        rewards,      # (n_agents,)
        next_state,   # (state_dim,)
        next_obs,     # (n_agents, obs_dim)
        done          # scalar
    ):
        transition = (
            np.array(state, dtype=np.float32),
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            float(done)
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        state, obs, actions, rewards, next_state, next_obs, done = zip(*batch)

        return (
            np.stack(state),        # (B, state_dim)
            np.stack(obs),          # (B, n_agents, obs_dim)
            np.stack(actions),      # (B, n_agents, action_dim)
            np.stack(rewards),      # (B, n_agents)
            np.stack(next_state),   # (B, state_dim)
            np.stack(next_obs),     # (B, n_agents, obs_dim)
            np.array(done).reshape(-1, 1)  # (B, 1)
        )

    def __len__(self):
        return len(self.buffer)
