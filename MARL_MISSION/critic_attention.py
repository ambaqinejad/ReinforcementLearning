# critic_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# Hyperparameters (MUST match Actor)
# =====================================================
MESSAGE_DIM = 32
NUM_HEADS = 8
HEAD_DIM = MESSAGE_DIM // NUM_HEADS


# =====================================================
# Multi-Head Attention for Agents
# =====================================================
class MultiHeadAgentAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query_vec, key_value_vecs):
        """
        query_vec: (B, embed_dim)
        key_value_vecs: (B, N, embed_dim)
        """

        B, N, D = key_value_vecs.shape

        Q = self.query(query_vec).view(B, NUM_HEADS, D // NUM_HEADS)
        K = self.key(key_value_vecs).view(B, N, NUM_HEADS, D // NUM_HEADS)
        V = self.value(key_value_vecs).view(B, N, NUM_HEADS, D // NUM_HEADS)

        scores = torch.einsum("bhd,bnhd->bhn", Q, K) / (HEAD_DIM ** 0.5)
        attn = F.softmax(scores, dim=-1)

        context = torch.einsum("bhn,bnhd->bhd", attn, V)
        context = context.reshape(B, D)

        return self.out_proj(context)


# =====================================================
# Centralized Critic with Agent-Level Attention
# =====================================================
class CriticAttention(nn.Module):
    """
    Centralized Critic for MARL with Communication Awareness

    Inputs:
        - global_state  : (B, state_dim)
        - actions       : (B, n_agents, action_dim)
        - messages      : (B, n_agents, MESSAGE_DIM)

    Output:
        - Q-value       : (B, 1)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 512
    ):
        super().__init__()

        self.n_agents = n_agents
        self.action_dim = action_dim

        # ---------------- Agent encoder ----------------
        self.agent_encoder = nn.Sequential(
            nn.Linear(action_dim + MESSAGE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # ---------------- Attention ----------------
        self.attention = MultiHeadAgentAttention(hidden_dim)

        # ---------------- State encoder ----------------
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # ---------------- Q head ----------------
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    # -------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # -------------------------------------------------
    def forward(self, state, actions, messages):
        """
        state    : (B, state_dim)
        actions  : (B, n_agents, action_dim)
        messages : (B, n_agents, MESSAGE_DIM)
        """

        B, N, _ = actions.shape

        # -------- Encode agents --------
        agent_inputs = torch.cat([actions, messages], dim=-1)
        agent_embeds = self.agent_encoder(
            agent_inputs.view(B * N, -1)
        ).view(B, N, -1)

        # -------- Attention over agents --------
        global_agent_context = self.attention(
            agent_embeds.mean(dim=1),
            agent_embeds
        )

        # -------- Encode state --------
        state_embed = self.state_encoder(state)

        # -------- Final Q --------
        q_input = torch.cat([state_embed, global_agent_context], dim=-1)
        q_value = self.q_head(q_input)

        return q_value
