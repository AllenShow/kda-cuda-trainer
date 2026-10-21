"""Tiny byte-level data pipeline with no tokenizer dependency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ByteDataset:
    train: torch.Tensor
    validation: torch.Tensor

    @classmethod
    def from_text(cls, path: str | Path, validation_fraction: float = 0.1) -> "ByteDataset":
        raw = Path(path).read_bytes()
        if len(raw) < 4:
            raise ValueError(f"Dataset {path} must contain at least 4 bytes of text")
        tokens = torch.tensor(list(raw), dtype=torch.long)
        split = int(len(tokens) * (1.0 - validation_fraction))
        split = min(max(split, 2), len(tokens) - 2)
        return cls(train=tokens[:split], validation=tokens[split:])

    def get_batch(
        self, split: str, batch_size: int, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.train if split == "train" else self.validation
        max_start = source.numel() - seq_len - 1
        if max_start < 0:
            raise ValueError(
                f"The {split} split has {source.numel()} bytes; it needs at least "
                f"seq_len + 1 = {seq_len + 1}. Use a shorter sequence or more text."
            )
        starts = torch.randint(max_start + 1, (batch_size,))
        offsets = torch.arange(seq_len)
        positions = starts[:, None] + offsets[None, :]
        x = source[positions]
        y = source[positions + 1]
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

