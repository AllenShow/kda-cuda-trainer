"""Small decoder-only language model with KDA in every sequence-mixing block."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .kda import KDAAttention, RMSNorm


@dataclass
class ModelConfig:
    vocab_size: int = 256
    dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: int = 4
    max_seq_len: int = 512
    dropout: float = 0.0
    kda_backend: str = "auto"


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.attn = KDAAttention(
            config.dim, config.num_heads, backend=config.kda_backend
        )
        self.mlp_norm = RMSNorm(config.dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, config.mlp_ratio * config.dim),
            nn.GELU(),
            nn.Linear(config.mlp_ratio * config.dim, config.dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class KDAForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        for layer in self.layers:
            nn.init.normal_(
                layer.attn.o_proj.weight,
                mean=0.0,
                std=0.02 / (2 * config.num_layers) ** 0.5,
            )
            nn.init.normal_(
                layer.mlp[2].weight,
                mean=0.0,
                std=0.02 / (2 * config.num_layers) ** 0.5,
            )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {input_ids.shape[1]} exceeds max_seq_len "
                f"{self.config.max_seq_len}"
            )
        x = self.dropout(self.token_embedding(input_ids))
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)

