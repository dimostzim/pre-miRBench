#!/usr/bin/env python3
"""Structure-aware pair-message CNN for precursor-miRNA classification."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 12
    sequence_length: int = 200
    conv_channels: int = 64
    stem_kernel_size: int = 7
    block_kernel_size: int = 7
    dilations: tuple[int, ...] = (1, 2, 4)
    block_dropout: float = 0.15
    avg_pool_bins: int = 10
    mfe_features: int = 1
    hidden_dim: int = 64
    classifier_dropout: float = 0.35

    @property
    def pooled_dim(self) -> int:
        return self.conv_channels * (self.avg_pool_bins + 1) + self.mfe_features

    def to_dict(self) -> Dict[str, Any]:
        output = asdict(self)
        output["dilations"] = list(self.dilations)
        output["pooled_dim"] = self.pooled_dim
        return output


class PairMessageBlock(nn.Module):
    """Fuse local convolutional context with exact base-pair messages."""
    def __init__(self, channels: int = 64, kernel_size: int = 7,
                 dilation: int = 1, dropout: float = 0.15) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd to preserve sequence length")
        padding = (kernel_size // 2) * dilation
        self.local = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(4 * channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def gather_partner_states(x: torch.Tensor, partner_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = x.shape[-1]
        gather_index = partner_indices.unsqueeze(1).expand(-1, x.shape[1], -1)
        gathered = torch.gather(x, dim=2, index=gather_index)
        positions = torch.arange(length, device=x.device).view(1, 1, length)
        paired_mask = partner_indices.unsqueeze(1).ne(positions)
        return gathered * paired_mask.to(x.dtype), paired_mask

    def forward(self, x: torch.Tensor, partner_indices: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        partner, paired_mask = self.gather_partner_states(x, partner_indices)
        difference = (x - partner).abs() * paired_mask.to(x.dtype)
        fused = self.fuse(torch.cat((x, local, partner, difference), dim=1))
        return x + fused


class PairMessageCNN(nn.Module):
    """One-logit CNN using local and RNA base-pair graph information."""
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.input_channels != 12 or c.sequence_length != 200:
            raise ValueError("This architecture requires the fixed [12, 200] representation")
        if len(c.dilations) != 3:
            raise ValueError("Exactly three pair-message blocks are required")
        self.stem = nn.Sequential(
            nn.Conv1d(c.input_channels, c.conv_channels, c.stem_kernel_size,
                      padding=c.stem_kernel_size // 2),
            nn.BatchNorm1d(c.conv_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            PairMessageBlock(c.conv_channels, c.block_kernel_size, dilation, c.block_dropout)
            for dilation in c.dilations
        ])
        self.avg_pool = nn.AdaptiveAvgPool1d(c.avg_pool_bins)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(c.pooled_dim, c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.classifier_dropout),
            nn.Linear(c.hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_inputs(self, channels: torch.Tensor, partner_indices: torch.Tensor,
                         mfe: torch.Tensor) -> torch.Tensor:
        c = self.config
        if channels.ndim != 3 or tuple(channels.shape[1:]) != (c.input_channels, c.sequence_length):
            raise ValueError(f"channels must have shape [B,{c.input_channels},{c.sequence_length}], got {tuple(channels.shape)}")
        if partner_indices.ndim != 2 or tuple(partner_indices.shape) != (channels.shape[0], c.sequence_length):
            raise ValueError(f"partner_indices must have shape [B,{c.sequence_length}], got {tuple(partner_indices.shape)}")
        if partner_indices.dtype != torch.long:
            raise TypeError(f"partner_indices must be torch.long, got {partner_indices.dtype}")
        if partner_indices.device != channels.device or mfe.device != channels.device:
            raise ValueError("all model inputs must be on the same device")
        if partner_indices.numel() and (partner_indices.min() < 0 or partner_indices.max() >= c.sequence_length):
            raise ValueError("partner_indices contains an out-of-range index")
        if mfe.ndim == 1:
            mfe = mfe.unsqueeze(1)
        if mfe.ndim != 2 or tuple(mfe.shape) != (channels.shape[0], c.mfe_features):
            raise ValueError(f"mfe must have shape [B,{c.mfe_features}], got {tuple(mfe.shape)}")
        return mfe

    def forward(self, channels: torch.Tensor, partner_indices: torch.Tensor,
                mfe: torch.Tensor) -> torch.Tensor:
        mfe = self._validate_inputs(channels, partner_indices, mfe)
        encoded = self.stem(channels)
        for block in self.blocks:
            encoded = block(encoded, partner_indices)
        features = torch.cat((self.avg_pool(encoded).flatten(1),
                              self.max_pool(encoded).flatten(1), mfe), dim=1)
        return self.classifier(features).squeeze(1)


def build_model(config_dict: Dict[str, Any] | None = None) -> PairMessageCNN:
    if config_dict is None:
        return PairMessageCNN()
    values = dict(config_dict)
    values.pop("pooled_dim", None)
    if "dilations" in values:
        values["dilations"] = tuple(values["dilations"])
    return PairMessageCNN(ModelConfig(**values))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
