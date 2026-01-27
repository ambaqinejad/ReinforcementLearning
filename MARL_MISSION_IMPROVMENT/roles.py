# roles.py

import torch
import torch.nn as nn

# -------------------------------
# Role IDs
# -------------------------------
ROLE_SCOUT = 0
ROLE_SURROUND = 1
ROLE_ATTACKER = 2
ROLE_GOAL = 3

ROLE_NAMES = {
    ROLE_SCOUT: "scout",
    ROLE_SURROUND: "surround",
    ROLE_ATTACKER: "attacker",
    ROLE_GOAL: "goal",
}

N_ROLES = 4


# -------------------------------
# Role Embedding Module
# -------------------------------
class RoleEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(N_ROLES, embed_dim)

        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, role_ids: torch.Tensor):
        """
        role_ids: (B,) or (B,1)
        return: (B, embed_dim)
        """
        return self.embedding(role_ids)
