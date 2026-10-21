"""Train a compact byte-level causal LM using Kimi Delta Attention."""

from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

from .data import ByteDataset
from .model import KDAForCausalLM, ModelConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/tiny.txt")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/kda.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument(
        "--kda-backend", choices=("auto", "fla", "reference"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-every", type=int, default=500)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def lr_at_step(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(args.warmup_steps, 1)
    progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + cosine * (args.lr - args.min_lr)


@torch.no_grad()
def estimate_loss(
    model: KDAForCausalLM,
    dataset: ByteDataset,
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    result: dict[str, float] = {}
    for split in ("train", "validation"):
        losses = []
        for _ in range(args.eval_batches):
            x, y = dataset.get_batch(split, args.batch_size, args.seq_len, device)
            with autocast_context(device, amp_dtype):
                _, loss = model(x, y)
            assert loss is not None
            losses.append(loss.float().item())
        result[split] = sum(losses) / len(losses)
    model.train(was_training)
    return result


def main() -> None:
    args = parse_args()
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be at least 1")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    device = choose_device(args.device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]
    if device.type != "cuda":
        amp_dtype = None
    elif args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16; choose --amp fp16 or off")

    dataset = ByteDataset.from_text(args.dataset)
    config = ModelConfig(
        vocab_size=256,
        dim=args.dim,
        num_heads=args.heads,
        num_layers=args.layers,
        mlp_ratio=args.mlp_ratio,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        kda_backend=args.kda_backend,
    )
    model = KDAForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16
    )

    print(
        f"device={device} backend={args.kda_backend} params={sum(p.numel() for p in model.parameters()):,} "
        f"train_bytes={dataset.train.numel():,} val_bytes={dataset.validation.numel():,}"
    )
    model.train()
    start = time.time()
    for step in range(args.steps):
        lr = lr_at_step(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = dataset.get_batch("train", args.batch_size, args.seq_len, device)
            with autocast_context(device, amp_dtype):
                _, loss = model(x, y)
                assert loss is not None
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            running_loss += loss.detach().float().item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        completed_step = step + 1
        if completed_step % args.eval_interval == 0 or completed_step == 1:
            metrics = estimate_loss(model, dataset, args, device, amp_dtype)
            elapsed = time.time() - start
            print(
                f"step={completed_step:>6} lr={lr:.2e} "
                f"train_loss={metrics['train']:.4f} val_loss={metrics['validation']:.4f} "
                f"iter_loss={running_loss:.4f} elapsed={elapsed:.1f}s"
            )

        if completed_step % args.save_every == 0 or completed_step == args.steps:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": completed_step,
                    "model": model.state_dict(),
                    "model_config": asdict(config),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                args.out,
            )
            print(f"saved {args.out}")


if __name__ == "__main__":
    main()

