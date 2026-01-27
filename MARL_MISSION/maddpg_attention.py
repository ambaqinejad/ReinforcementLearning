# simple_maddpg.py

import torch
import torch.nn.functional as F
import numpy as np

from actor_attention import ActorAttention, MESSAGE_DIM
from critic_attention import CriticAttention
from replay_buffer_comm import ReplayBufferComm
from utils import soft_update


class MADDPGAttention:
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
        obstacle_feat_dim: int,
        neighbor_feat_dim: int,
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
        self.obstacle_feat_dim = obstacle_feat_dim
        self.neighbor_feat_dim = neighbor_feat_dim

        self.roles = torch.tensor(roles, dtype=torch.long, device=device)

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
            actor = ActorAttention(
                self_obs_dim=obs_dim,
                action_dim=action_dim,
                obstacle_feat_dim=obstacle_feat_dim,
                neighbor_feat_dim=neighbor_feat_dim
            ).to(device)

            target_actor = ActorAttention(
                self_obs_dim=obs_dim,
                action_dim=action_dim,
                obstacle_feat_dim=obstacle_feat_dim,
                neighbor_feat_dim=neighbor_feat_dim
            ).to(device)

            target_actor.load_state_dict(actor.state_dict())

            self.actors.append(actor)
            self.target_actors.append(target_actor)

            self.actor_opts.append(torch.optim.Adam(actor.parameters(), lr=lr_actor))

        # ==================================================
        # Critic
        # ==================================================
        self.critic = CriticAttention(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents
        ).to(device)

        self.target_critic = CriticAttention(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents
        ).to(device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # ==================================================
        # Replay Buffer
        # ==================================================
        self.replay_buffer = ReplayBufferComm(buffer_size)

    # ======================================================
    # Action & Message Selection
    # ======================================================
    def select_actions(self, obs, noise=0.0):
        """
        obs: (n_agents, obs_dim)
        returns:
            actions  : (n_agents, action_dim)
            messages : (n_agents, MESSAGE_DIM)
        """

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

        actions = []
        messages = []

        # Placeholder messages for first pass
        all_messages = torch.zeros(
            (1, self.n_agents, MESSAGE_DIM), device=self.device
        )

        for i in range(self.n_agents):
            # Example split — adjust indices to match your env
            self_obs = obs_t[i, :self.obs_dim].unsqueeze(0)

            neighbor_feats = self_obs[0][6: 4*(6+3)+6]
            neighbor_feats = neighbor_feats.view(1, 1, 36)

            obstacle_feats = self_obs[0][4*(6+3)+6:]
            obstacle_feats = obstacle_feats.view(1, 1, 30)

            a, m = self.actors[i](
                self_obs,
                self.roles[i].unsqueeze(0),
                neighbor_feats,
                obstacle_feats
            )

            a = a.squeeze(0)
            m = m.squeeze(0)

            # Add exploration noise
            a = a + noise * torch.randn_like(a)
            a = torch.clamp(a, -1.0, 1.0)

            actions.append(a)
            messages.append(m)

            all_messages[0, i] = m.detach()

        actions = torch.stack(actions).detach().cpu().numpy()

        messages = torch.stack(messages).detach()

        return actions, messages

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

