#!/usr/bin/env python3
"""Species-conditioned graph-context BiGRU for precursor-miRNA classification."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 12
    sequence_length: int = 200
    stem_width: int = 96
    stem_kernel_size: int = 7
    graph_dilations: tuple[int, ...] = (1, 2, 4)
    local_kernel_short: int = 5
    local_kernel_long: int = 15
    graph_dropout: float = 0.15
    num_species: int = 69
    species_embedding_dim: int = 16
    species_dropout: float = 0.30
    global_features: int = 13
    gru_hidden_size: int = 64
    avg_pool_bins: int = 10
    classifier_hidden: int = 128
    classifier_dropout: float = 0.35

    @property
    def gru_output_width(self) -> int:
        return 2 * self.gru_hidden_size

    @property
    def classifier_input_dim(self) -> int:
        width = self.gru_output_width
        return width * self.avg_pool_bins + width + width + self.global_features + self.species_embedding_dim

    def to_dict(self) -> Dict[str, Any]:
        output = asdict(self)
        output["graph_dilations"] = list(self.graph_dilations)
        output["gru_output_width"] = self.gru_output_width
        output["classifier_input_dim"] = self.classifier_input_dim
        return output


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = (kernel_size // 2) * dilation
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=padding,
                                   dilation=dilation, groups=channels, bias=False)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels independently at each sequence position."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class GraphContextBlock(nn.Module):
    """Fuse two local scales with exact, masked RNA base-pair messages."""
    def __init__(self, channels: int = 96, dilation: int = 1,
                 short_kernel: int = 5, long_kernel: int = 15,
                 dropout: float = 0.15) -> None:
        super().__init__()
        self.local_short = DepthwiseSeparableConv1d(channels, short_kernel, dilation)
        self.local_long = DepthwiseSeparableConv1d(channels, long_kernel, dilation)
        self.fuse = nn.Sequential(
            nn.Conv1d(5 * channels, channels, kernel_size=1),
            ChannelLayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def gather_partner_states(x: torch.Tensor, partner_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or partner_indices.ndim != 2:
            raise ValueError("x and partner_indices must have shapes [B,C,L] and [B,L]")
        if partner_indices.dtype != torch.long:
            raise TypeError("partner_indices must be torch.int64")
        batch, channels, length = x.shape
        if tuple(partner_indices.shape) != (batch, length):
            raise ValueError("partner_indices shape does not match x")
        if partner_indices.device != x.device:
            raise ValueError("partner_indices and x must be on the same device")
        if partner_indices.numel() and (partner_indices.min() < 0 or partner_indices.max() >= length):
            raise ValueError("partner_indices are out of bounds")
        gathered = torch.gather(x, 2, partner_indices.unsqueeze(1).expand(-1, channels, -1))
        positions = torch.arange(length, device=x.device).view(1, length)
        paired_mask = partner_indices.ne(positions).unsqueeze(1)
        return gathered * paired_mask.to(dtype=x.dtype), paired_mask

    def forward(self, x: torch.Tensor, partner_indices: torch.Tensor) -> torch.Tensor:
        partner, paired_mask = self.gather_partner_states(x, partner_indices)
        # Mask the difference too: an unpaired self-index must contribute no pair message.
        difference = (x - partner).abs() * paired_mask.to(dtype=x.dtype)
        fused = torch.cat((x, self.local_short(x), self.local_long(x), partner, difference), dim=1)
        return x + self.fuse(fused)


class AttentionPool1d(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.score = nn.Linear(width, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: [B,L,C], with no padding in this fixed-length task.
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class SpeciesGraphGRU(nn.Module):
    """Graph-context CNN + BiGRU with species-conditioned FiLM."""
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.num_species < 1:
            raise ValueError("num_species is the count of known species and must be positive")
        self.stem = nn.Sequential(
            nn.Conv1d(c.input_channels, c.stem_width, c.stem_kernel_size,
                      padding=c.stem_kernel_size // 2),
            nn.BatchNorm1d(c.stem_width),
            nn.GELU(),
        )
        self.positional_embedding = nn.Parameter(torch.empty(1, c.sequence_length, c.stem_width))
        # Index zero is unknown; known species occupy 1..num_species.
        self.species_embedding = nn.Embedding(c.num_species + 1, c.species_embedding_dim)
        self.species_embedding_dropout = nn.Dropout(c.species_dropout)
        self.species_film = nn.Linear(c.species_embedding_dim, 2 * c.stem_width)
        self.graph_blocks = nn.ModuleList([
            GraphContextBlock(c.stem_width, dilation=d,
                              short_kernel=c.local_kernel_short,
                              long_kernel=c.local_kernel_long,
                              dropout=c.graph_dropout)
            for d in c.graph_dilations
        ])
        self.bigru = nn.GRU(c.stem_width, c.gru_hidden_size, num_layers=1,
                           bidirectional=True, batch_first=True)
        self.attention_pool = AttentionPool1d(c.gru_output_width)
        self.classifier = nn.Sequential(
            nn.Linear(c.classifier_input_dim, c.classifier_hidden),
            nn.GELU(),
            nn.Dropout(c.classifier_dropout),
            nn.Linear(c.classifier_hidden, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Preserve PyTorch's standard GRU initialization by excluding GRU modules.
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.species_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Start exactly species-neutral: scale=0 and bias=0 after tanh.
        nn.init.zeros_(self.species_film.weight)
        nn.init.zeros_(self.species_film.bias)

    def _validate_inputs(self, channels: torch.Tensor, partner_indices: torch.Tensor,
                         global_features: torch.Tensor, species_indices: torch.Tensor) -> None:
        c = self.config
        if channels.ndim != 3 or tuple(channels.shape[1:]) != (c.input_channels, c.sequence_length):
            raise ValueError(f"channels must have shape [B,{c.input_channels},{c.sequence_length}]")
        batch = channels.shape[0]
        if tuple(partner_indices.shape) != (batch, c.sequence_length):
            raise ValueError(f"partner_indices must have shape [B,{c.sequence_length}]")
        if partner_indices.dtype != torch.long:
            raise TypeError("partner_indices must be torch.int64")
        if tuple(global_features.shape) != (batch, c.global_features):
            raise ValueError(f"global_features must have shape [B,{c.global_features}]")
        if tuple(species_indices.shape) != (batch,) or species_indices.dtype != torch.long:
            raise TypeError("species_indices must be int64 with shape [B]")
        devices = {channels.device, partner_indices.device, global_features.device, species_indices.device}
        if len(devices) != 1:
            raise ValueError("all model inputs must be on the same device")
        if not channels.is_floating_point() or not global_features.is_floating_point():
            raise TypeError("channels and global_features must be floating point")
        if species_indices.numel() and (species_indices.min() < 0 or species_indices.max() > c.num_species):
            raise ValueError("species index is outside [0, num_species]")
        if not torch.isfinite(channels).all() or not torch.isfinite(global_features).all():
            raise ValueError("floating-point inputs must be finite")

    def forward(self, channels: torch.Tensor, partner_indices: torch.Tensor,
                global_features: torch.Tensor, species_indices: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(channels, partner_indices, global_features, species_indices)
        species = self.species_embedding_dropout(self.species_embedding(species_indices))
        x = self.stem(channels).transpose(1, 2)
        x = x + self.positional_embedding
        film_scale, film_bias = self.species_film(species).chunk(2, dim=-1)
        x = x * (1.0 + torch.tanh(film_scale).unsqueeze(1)) + torch.tanh(film_bias).unsqueeze(1)
        x = x.transpose(1, 2)
        for block in self.graph_blocks:
            x = block(x, partner_indices)
        sequence, _ = self.bigru(x.transpose(1, 2))
        channel_first = sequence.transpose(1, 2)
        avg = F.adaptive_avg_pool1d(channel_first, self.config.avg_pool_bins).flatten(1)
        maximum = F.adaptive_max_pool1d(channel_first, 1).flatten(1)
        attention = self.attention_pool(sequence)
        pooled = torch.cat((avg, maximum, attention, global_features, species), dim=1)
        return self.classifier(pooled).squeeze(-1)


def build_model(config: Dict[str, Any] | ModelConfig | None = None) -> SpeciesGraphGRU:
    if config is None or isinstance(config, ModelConfig):
        return SpeciesGraphGRU(config)
    values = dict(config)
    values.pop("gru_output_width", None)
    values.pop("classifier_input_dim", None)
    if "graph_dilations" in values:
        values["graph_dilations"] = tuple(values["graph_dilations"])
    return SpeciesGraphGRU(ModelConfig(**values))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
