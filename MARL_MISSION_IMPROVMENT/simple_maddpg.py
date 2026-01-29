# simple_maddpg.py

import torch
import torch.nn.functional as F
import numpy as np

from MARL_MISSION.environment import ROLE_ATTACKER
from MARL_MISSION.roles import ROLE_SURROUND
from simple_actor import Actor
from simple_critic import Critic
from simple_replay_buffer import ReplayBuffer
from utils import soft_update


class MADDPG:
    """
    MADDPG with Attention-based Communication
    - Centralized Critic (CTDE)
    - Decentralized Actors
    - Learned Message Passing
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        roles: list,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.95,
        tau=0.01,
        buffer_size=1_000_000,
        batch_size=256,
        device="cpu"
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.roles = roles

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        # ==================================================
        # Actors
        # ==================================================
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []

        self.actor_opts = []
        self.critic_opts = []

        for _ in range(n_agents):
            actor = Actor(
                obs_dim=obs_dim,
                action_dim=action_dim
            ).to(device)

            critic = Critic(
                state_dim,
                self.n_agents * action_dim
            ).to(device)

            target_actor = Actor(
                obs_dim=obs_dim,
                action_dim=action_dim
            ).to(device)

            target_critic = Critic(
                state_dim,
                self.n_agents * action_dim
            ).to(device)

            target_actor.load_state_dict(actor.state_dict())
            target_critic.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)

            self.actor_opts.append(
                torch.optim.Adam(actor.parameters(), lr=lr_actor)
            )
            self.critic_opts.append(
                torch.optim.Adam(critic.parameters(), lr=lr_critic)
            )

        # ==================================================
        # Replay Buffer
        # ==================================================
        self.replay_buffer = ReplayBuffer(buffer_size)

    # ======================================================
    # Action & Message Selection
    # ======================================================
    def select_actions(self, obs, noise_pursuer, noise_evader, stop_surrounder, stop_attacker):
        """
        obs: (n_agents, obs_dim)
        """
        actions = []

        for i, actor in enumerate(self.actors):
            a = []
            if i < len(self.roles) and ((self.roles[i] == ROLE_ATTACKER and stop_attacker) or self.roles[i] == ROLE_SURROUND and stop_surrounder):
                a = np.array([0.0, 0.0, 0.0])

            else:
                o = torch.tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    a = actor(o).cpu().numpy()[0]

            if i == self.n_agents - 1:  # Evader
                a += noise_evader * np.random.randn(*a.shape)
            else:  # Pursuers
                a += noise_pursuer * np.random.randn(*a.shape)

            actions.append(np.clip(a, -1.0, 1.0))

        return np.array(actions)

    # ======================================================
    # Learning Step
    # ======================================================
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, obs, actions, rewards, next_state, next_obs, done = \
            self.replay_buffer.sample(self.batch_size)

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        # ==================================================
        # Target actions
        # ==================================================
        next_actions = []
        for i in range(self.n_agents):
            o = torch.tensor(
                next_obs[:, i, :],
                dtype=torch.float32
            ).to(self.device)
            next_actions.append(self.target_actors[i](o))

        next_actions = torch.cat(next_actions, dim=1)

        # ==================================================
        # Update critics
        # ==================================================
        for i in range(self.n_agents):
            with torch.no_grad():
                target_q = self.target_critics[i](
                    next_state,
                    next_actions
                )
                y = rewards[:, i:i + 1] + self.gamma * (1 - done) * target_q

            current_q = self.critics[i](
                state,
                actions.view(self.batch_size, -1)
            )

            critic_loss = F.mse_loss(current_q, y)

            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()

        # ==================================================
        # Update actors
        # ==================================================
        for i in range(self.n_agents):
            current_actions = []

            for j in range(self.n_agents):
                o = torch.tensor(
                    obs[:, j, :],
                    dtype=torch.float32
                ).to(self.device)

                if j == i:
                    current_actions.append(self.actors[j](o))
                else:
                    current_actions.append(self.actors[j](o).detach())

            current_actions = torch.cat(current_actions, dim=1)

            actor_loss = -self.critics[i](
                state,
                current_actions
            ).mean()

            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()

    def save(self, path):
        torch.save({
            "actors": [a.state_dict() for a in self.actors],
            "critics": [c.state_dict() for c in self.critics]
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(data["actors"][i])
            self.critics[i].load_state_dict(data["critics"][i])

