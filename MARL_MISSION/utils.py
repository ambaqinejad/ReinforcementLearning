# utils.py

import torch


def soft_update(target: torch.nn.Module,
                source: torch.nn.Module,
                tau: float):
    """
    Soft update of target network parameters.

    target = tau * source + (1 - tau) * target

    Args:
        target (nn.Module): target network
        source (nn.Module): online network
        tau (float): update rate (0 < tau <= 1)
    """
    for target_param, source_param in zip(
        target.parameters(), source.parameters()
    ):
        target_param.data.copy_(
            tau * source_param.data +
            (1.0 - tau) * target_param.data
        )
