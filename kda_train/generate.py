"""Sample bytes from a checkpoint produced by kda_train.train."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import KDAForCausalLM, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config_values = dict(checkpoint["model_config"])
    config_values["kda_backend"] = "auto"
    config = ModelConfig(**config_values)
    model = KDAForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt = torch.tensor(list(args.prompt.encode("utf-8")), dtype=torch.long, device=device)
    if prompt.numel() == 0:
        prompt = torch.zeros(1, dtype=torch.long, device=device)
    ids = prompt[None, :]
    with torch.inference_mode():
        for _ in range(args.tokens):
            context = ids[:, -config.max_seq_len :]
            logits, _ = model(context)
            next_logits = logits[:, -1, :] / max(args.temperature, 1e-5)
            if args.top_k > 0:
                values, _ = torch.topk(next_logits, min(args.top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, [-1]]] = -float("inf")
            next_id = torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1)
            ids = torch.cat((ids, next_id), dim=1)

    print(bytes(ids[0].tolist()).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()

