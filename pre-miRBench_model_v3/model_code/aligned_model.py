"""Lossless aligned one-message CNN; frozen source model.py remains untouched."""
import torch
from torch import nn
from model import PreMiRNACNN

class AlignedPairCNN(PreMiRNACNN):
    def __init__(self):
        super().__init__()
        self.stem[0] = nn.Conv1d(14,64,7,padding=3,bias=False)

    def encode_pairs(self, sequence, mfe, partner_indices):
        b = sequence.shape[0]
        if sequence.shape != (b,14,240) or mfe.shape != (b,1):
            raise ValueError('Expected [B,14,240] and [B,1]')
        if partner_indices.shape != (b,240) or partner_indices.dtype != torch.int64:
            raise ValueError('Expected int64 [B,240] partners')
        paired_mask = ((sequence[:,4] + sequence[:,5]) > 0) & (sequence[:,13] > 0)
        return self.pair_message(self.stem(sequence), partner_indices, paired_mask)
    # Inherited forward_train gathers original-example partners, then mixes
    # hidden sequence, MFE, targets together before encoder's first max pool.
