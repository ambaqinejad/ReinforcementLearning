# actor_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from roles import RoleEmbedding

# =====================================================
# Hyperparameters (LOCKED)
# =====================================================
MESSAGE_DIM = 32
NUM_HEADS = 8
HEAD_DIM = MESSAGE_DIM // NUM_HEADS


# =====================================================
# Multi-Head Attention Module
# =====================================================
class MultiHeadCommAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.query = nn.Linear(MESSAGE_DIM, MESSAGE_DIM)
        self.key = nn.Linear(MESSAGE_DIM, MESSAGE_DIM)
        self.value = nn.Linear(MESSAGE_DIM, MESSAGE_DIM)

        self.out_proj = nn.Linear(MESSAGE_DIM, MESSAGE_DIM)

    def forward(self, query_vec, key_value_vecs, mask=None):
        """
        query_vec: (B, MESSAGE_DIM)
        key_value_vecs: (B, N, MESSAGE_DIM)
        mask: (B, N)
        """

        B, N, _ = key_value_vecs.shape

        Q = self.query(query_vec).view(B, NUM_HEADS, HEAD_DIM)
        K = self.key(key_value_vecs).view(B, N, NUM_HEADS, HEAD_DIM)
        V = self.value(key_value_vecs).view(B, N, NUM_HEADS, HEAD_DIM)

        scores = torch.einsum("bhd,bnhd->bhn", Q, K) / (HEAD_DIM ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)

        context = torch.einsum("bhn,bnhd->bhd", attn, V)
        context = context.reshape(B, MESSAGE_DIM)

        return self.out_proj(context)


# =====================================================
# Actor with Obstacle + Neighbor Attention
# =====================================================
class ActorAttention(nn.Module):
    def __init__(
        self,
        self_obs_dim: int,
        neighbor_feat_dim: int,
        obstacle_feat_dim: int,
        action_dim: int,
        role_embed_dim: int = 8
    ):
        super().__init__()

        # ---------------- Self encoder ----------------
        self.self_encoder = nn.Sequential(
            nn.Linear(self_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, MESSAGE_DIM),
            nn.ReLU()
        )

        # ---------------- Neighbor encoder ----------------
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(neighbor_feat_dim, MESSAGE_DIM),
            nn.ReLU()
        )

        # ---------------- Obstacle encoder ----------------
        self.obstacle_encoder = nn.Sequential(
            nn.Linear(obstacle_feat_dim, MESSAGE_DIM),
            nn.ReLU()
        )

        # ---------------- Role ----------------
        self.role_embedding = RoleEmbedding(role_embed_dim)

        # ---------------- Message ----------------
        self.message_head = nn.Sequential(
            nn.Linear(MESSAGE_DIM + role_embed_dim, MESSAGE_DIM),
            nn.ReLU()
        )

        # ---------------- Attention ----------------
        self.neighbor_attention = MultiHeadCommAttention()
        self.obstacle_attention = MultiHeadCommAttention()

        # ---------------- Action head ----------------
        self.action_head = nn.Sequential(
            nn.Linear(MESSAGE_DIM * 3, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(
        self,
        self_obs,
        role_id,
        neighbor_feats,
        obstacle_feats,
        neighbor_mask=None,
        obstacle_mask=None
    ):
        """
        self_obs: (B, self_obs_dim)
        neighbor_feats: (B, K, neighbor_feat_dim)
        obstacle_feats: (B, M, obstacle_feat_dim)
        """

        # Encode self
        self_embed = self.self_encoder(self_obs)

        # Role
        role_embed = self.role_embedding(role_id)

        # Message
        msg_input = torch.cat([self_embed, role_embed], dim=-1)
        my_message = self.message_head(msg_input)

        # Encode neighbors & obstacles
        neigh_enc = self.neighbor_encoder(neighbor_feats)
        obs_enc = self.obstacle_encoder(obstacle_feats)

        # Attention
        neigh_ctx = self.neighbor_attention(self_embed, neigh_enc, neighbor_mask)
        obs_ctx = self.obstacle_attention(self_embed, obs_enc, obstacle_mask)

        # Final action
        final_input = torch.cat([self_embed, neigh_ctx, obs_ctx], dim=-1)
        action = self.action_head(final_input)

        return action, my_message
