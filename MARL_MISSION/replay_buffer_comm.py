# replay_buffer_comm.py

import random
import numpy as np
from collections import deque


class ReplayBufferComm:
    """
    Replay Buffer for MADDPG with Communication

    Stores:
        state        : (state_dim,)
        obs          : (N, obs_dim)
        actions      : (N, action_dim)
        messages     : (N, message_dim)
        rewards      : (N,)
        next_state   : (state_dim,)
        next_obs     : (N, obs_dim)
        done         : scalar or (N,)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    # --------------------------------------------------
    def __len__(self):
        return len(self.buffer)

    # --------------------------------------------------
    def push(
        self,
        state,
        obs,
        actions,
        messages,
        rewards,
        next_state,
        next_obs,
        done
    ):
        """
        All inputs are expected as numpy arrays
        """

        transition = (
            np.array(state, copy=False),
            np.array(obs, copy=False),
            np.array(actions, copy=False),
            np.array(messages.detach().cpu().numpy(), copy=False),
            np.array(rewards, copy=False),
            np.array(next_state, copy=False),
            np.array(next_obs, copy=False),
            np.array(done, copy=False),
        )

        self.buffer.append(transition)

    # --------------------------------------------------
    def sample(self, batch_size: int):
        """
        Returns batch with shapes:

        state       : (B, state_dim)
        obs         : (B, N, obs_dim)
        actions     : (B, N, action_dim)
        messages    : (B, N, message_dim)
        rewards     : (B, N)
        next_state  : (B, state_dim)
        next_obs    : (B, N, obs_dim)
        done        : (B, 1) or (B, N)
        """

        batch = random.sample(self.buffer, batch_size)

        (
            state,
            obs,
            actions,
            messages,
            rewards,
            next_state,
            next_obs,
            done
        ) = zip(*batch)

        return (
            np.stack(state, axis=0),
            np.stack(obs, axis=0),
            np.stack(actions, axis=0),
            np.stack(messages, axis=0),
            np.stack(rewards, axis=0),
            np.stack(next_state, axis=0),
            np.stack(next_obs, axis=0),
            np.stack(done, axis=0),
        )
