"""Device-side folded-window view; no fitting, metadata, or inferred pairings."""
import torch


def build_pair_grid(x: torch.Tensor, partner_indices: torch.Tensor) -> torch.Tensor:
    """Return [B,9,100,100]; row r=i, column c=199-j.

    x is the unmodified schema-v3 [B,13,200] tensor. Channel 8 uses
    the original bracket mask, never the self-index sentinel alone. Call
    per minibatch on the inference/training device, before latent mixup.
    """
    if x.ndim != 3 or x.shape[1:] != (13, 200):
        raise ValueError('Expected original features [B,13,200]')
    if partner_indices.shape != (x.shape[0], 200) or partner_indices.dtype != torch.int64:
        raise ValueError('Expected int64 partner indices [B,200]')
    if partner_indices.device != x.device or not x.is_floating_point():
        raise ValueError('Features must be floating point and share the indices device')
    b = x.shape[0]
    left = x[:, :4, :100].unsqueeze(-1).expand(b, 4, 100, 100)
    right = x[:, :4, 100:].flip(-1).unsqueeze(-2).expand(b, 4, 100, 100)
    right_positions = torch.arange(199, 99, -1, device=x.device)
    paired = (x[:, 4, :100] + x[:, 5, :100]) > 0
    contacts = ((partner_indices[:, :100, None] == right_positions[None, None, :])
                & paired[:, :, None]).unsqueeze(1).to(x.dtype)
    return torch.cat((left, right, contacts), dim=1)
