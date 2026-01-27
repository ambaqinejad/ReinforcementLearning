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
        self.target_actors = []
        self.actor_opts = []

        for _ in range(n_agents):
            actor = Actor(
                obs_dim=obs_dim,
                action_dim=action_dim
            ).to(device)

            target_actor = Actor(
                obs_dim=obs_dim,
                action_dim=action_dim
            ).to(device)

            target_actor.load_state_dict(actor.state_dict())

            self.actors.append(actor)
            self.target_actors.append(target_actor)

            self.actor_opts.append(torch.optim.Adam(actor.parameters(), lr=lr_actor))

        # ==================================================
        # Critic
        # ==================================================
        self.critic = Critic(
            state_dim=state_dim,
            action_dim_total=self.n_agents * action_dim
        ).to(device)

        self.target_critic = Critic(
            state_dim=state_dim,
            action_dim_total=self.n_agents * action_dim
        ).to(device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

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

        (
            state,
            obs,
            actions,
            messages,
            rewards,
            next_state,
            next_obs,
            done
        ) = self.replay_buffer.sample(self.batch_size)

        # ---------------- Tensors ----------------
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        messages = torch.tensor(messages, dtype=torch.float32, device=self.device)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        # ==================================================
        # Target actions & messages
        # ==================================================
        next_actions = []
        next_messages = []

        zero_messages = torch.zeros(
            (self.batch_size, self.n_agents, MESSAGE_DIM),
            device=self.device
        )

        for i in range(self.n_agents):
            a, m = self.target_actors[i](
                next_obs[:, i, :],
                self.roles[i].repeat(self.batch_size),
                zero_messages
            )
            next_actions.append(a)
            next_messages.append(m)

        next_actions = torch.stack(next_actions, dim=1)
        next_messages = torch.stack(next_messages, dim=1)

        # ==================================================
        # Critic update
        # ==================================================
        with torch.no_grad():
            target_q = self.target_critic(next_state, next_actions, next_messages)
            y = rewards.mean(dim=1, keepdim=True) + self.gamma * (1 - done) * target_q

        current_q = self.critic(state, actions, messages)
        critic_loss = F.mse_loss(current_q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ==================================================
        # Actor update
        # ==================================================
        for i in range(self.n_agents):
            cur_actions = []
            cur_messages = []

            zero_messages = torch.zeros(
                (self.batch_size, self.n_agents, MESSAGE_DIM), device=self.device
            )

            for j in range(self.n_agents):
                if j == i:
                    a, m = self.actors[j](obs[:, j, :], self.roles[j].repeat(self.batch_size), zero_messages)
                else:
                    with torch.no_grad():
                        a, m = self.actors[j](obs[:, j, :], self.roles[j].repeat(self.batch_size), zero_messages)

                cur_actions.append(a)
                cur_messages.append(m)

            cur_actions = torch.stack(cur_actions, dim=1)
            cur_messages = torch.stack(cur_messages, dim=1)

            actor_loss = -self.critic(state, cur_actions, cur_messages).mean()

            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()

        # ==================================================
        # Soft update
        # ==================================================
        for i in range(self.n_agents):
            soft_update(self.target_actors[i], self.actors[i], self.tau)

        soft_update(self.target_critic, self.critic, self.tau)

    def save(self, path):
        checkpoint = {
            "actors": [actor.state_dict() for actor in self.actors],
            "target_actors": [actor.state_dict() for actor in self.target_actors],
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_opts": [opt.state_dict() for opt in self.actor_opts],
            "critic_opt": self.critic_opt.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint["actors"][i])
            self.target_actors[i].load_state_dict(checkpoint["target_actors"][i])
            self.actor_opts[i].load_state_dict(checkpoint["actor_opts"][i])

        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt"])

