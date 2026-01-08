import torch
import torch.nn.functional as F
import numpy as np

from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer
from utils import soft_update

class MADDPG:
    def __init__(
        self,
        n_agents,
        obs_dim,
        state_dim,
        action_dim,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.95,
        tau=0.01,
        buffer_size=1_000_000,
        batch_size=256,
        device="cpu"
    ):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []

        self.actor_opts = []
        self.critic_opts = []

        for _ in range(n_agents):
            actor = Actor(obs_dim, action_dim).to(device)
            critic = Critic(state_dim, n_agents * action_dim).to(device)

            target_actor = Actor(obs_dim, action_dim).to(device)
            target_critic = Critic(state_dim, n_agents * action_dim).to(device)

            target_actor.load_state_dict(actor.state_dict())
            target_critic.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)

            self.actor_opts.append(torch.optim.Adam(actor.parameters(), lr=lr_actor))
            self.critic_opts.append(torch.optim.Adam(critic.parameters(), lr=lr_critic))

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_actions(self, obs, noise=0.1):
        actions = []

        for i, actor in enumerate(self.actors):
            o = torch.tensor(obs[i], dtype=torch.float32).to(self.device)
            a = actor(o).detach().cpu().numpy()
            a += noise * np.random.randn(*a.shape)
            actions.append(np.clip(a, -1, 1))

        return np.array(actions)


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
        next_actions = []
        for i in range(self.n_agents):
            o = torch.tensor(next_obs[:, i, :], dtype=torch.float32).to(self.device)
            next_actions.append(self.target_actors[i](o))

        next_actions = torch.cat(next_actions, dim=1)
        for i in range(self.n_agents):
            with torch.no_grad():
                target_q = self.target_critics[i](
                    next_state,
                    next_actions
                )
                y = rewards[:, i:i+1] + self.gamma * (1 - done) * target_q

            current_q = self.critics[i](state, actions.view(self.batch_size, -1))
            critic_loss = F.mse_loss(current_q, y)

            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()
        for i in range(self.n_agents):
            current_actions = []

            for j in range(self.n_agents):
                o = torch.tensor(obs[:, j, :], dtype=torch.float32).to(self.device)
                if j == i:
                    current_actions.append(self.actors[j](o))
                else:
                    current_actions.append(self.actors[j](o).detach())

            current_actions = torch.cat(current_actions, dim=1)

            actor_loss = -self.critics[i](state, current_actions).mean()

            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()
            
        for i in range(self.n_agents):
            soft_update(self.target_actors[i], self.actors[i], self.tau)
            soft_update(self.target_critics[i], self.critics[i], self.tau)
