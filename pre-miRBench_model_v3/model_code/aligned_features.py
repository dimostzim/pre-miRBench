"""Input-only lossless terminal-loop alignment, applied to individual minibatches."""
import math
import numpy as np
import torch
from features import CHANNELS

ALIGNED_CHANNELS = CHANNELS + ['valid_position']

def alignment_info(structure):
    """Parse immediate-parent tree; select a terminal pair without labels/metadata."""
    if len(structure) != 200:
        raise ValueError('Expected 200 positions')
    stack, terminals = [], []
    for i, char in enumerate(structure):
        if char == '(':
            if stack:
                stack[-1][1] = True
            stack.append([i, False])
        elif char == ')':
            if not stack:
                raise ValueError('Unbalanced structure')
            opening, has_child = stack.pop()
            if not has_child:
                terminals.append((opening, i))
        elif char != '.':
            raise ValueError('Invalid structure alphabet')
    if stack:
        raise ValueError('Unbalanced structure')
    eligible = [(i,j) for i,j in terminals if 159 <= i+j <= 239]
    selected = min(eligible, key=lambda p: (abs(p[0]+p[1]-199), p[0])) if eligible else None
    midpoint = sum(selected)/2 if selected else None
    offset = math.floor(119.5-midpoint+0.5) if selected else 20
    return dict(eligible_loop_count=len(eligible), fallback=selected is None,
                selected_pair=selected, selected_midpoint=midpoint, insertion_offset=offset)


def build_aligned_features(x, partner_indices, structures):
    """Return float32 [B,14,240], int64 [B,240]; MFE passes separately unchanged.

    Accept numpy arrays or torch tensors; outputs preserve backend/device. Original
    features must be unmodified schema-v3 features corresponding to structures.
    Only a minibatch is allocated. Pair scales remain relative to original length.
    """
    b = len(structures)
    if x.shape != (b,13,200) or partner_indices.shape != (b,200):
        raise ValueError('Expected original x [B,13,200] and indices [B,200]')
    is_torch = isinstance(x, torch.Tensor)
    if is_torch:
        if x.dtype != torch.float32 or partner_indices.dtype != torch.int64 or x.device != partner_indices.device:
            raise ValueError('Expected float32 features and int64 indices on same device')
        out = x.new_zeros((b,14,240))
        indices = torch.arange(240, device=x.device).expand(b,-1).clone()
    else:
        if x.dtype != np.float32 or partner_indices.dtype != np.int64:
            raise ValueError('Expected float32 features and int64 indices')
        out = np.zeros((b,14,240), dtype=np.float32)
        indices = np.broadcast_to(np.arange(240, dtype=np.int64), (b,240)).copy()
    for k, structure in enumerate(structures):
        offset = alignment_info(structure)['insertion_offset']
        out[k,:13,offset:offset+200] = x[k]
        out[k,13,offset:offset+200] = 1
        indices[k,offset:offset+200] = partner_indices[k] + offset
    return out, indices
