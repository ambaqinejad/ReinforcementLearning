import numpy as np
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(
        self,
        state,
        obs,
        actions,
        rewards,
        next_state,
        next_obs,
        done
    ):
        data = (state, obs, actions, rewards, next_state, next_obs, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        state, obs, actions, rewards, next_state, next_obs, done = map(
            np.array, zip(*batch)
        )

        return state, obs, actions, rewards, next_state, next_obs, done

    def __len__(self):
        return len(self.buffer)
