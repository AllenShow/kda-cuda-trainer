"""Exercise the public training and generation commands on CPU."""

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cpu_training_checkpoint_can_generate(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "cpu-smoke.pt"
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "kda_train.train",
            "--device", "cpu",
            "--kda-backend", "reference",
            "--amp", "off",
            "--steps", "2",
            "--batch-size", "1",
            "--seq-len", "16",
            "--dim", "32",
            "--heads", "2",
            "--layers", "1",
            "--eval-interval", "1",
            "--eval-batches", "1",
            "--out", str(checkpoint),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        timeout=120,
    )

    assert checkpoint.is_file()
    assert checkpoint.stat().st_size > 0

    generated = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "kda_train.generate",
            str(checkpoint),
            "--device", "cpu",
            "--prompt", "Hello",
            "--tokens", "8",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        timeout=120,
    )

    assert generated.stdout.startswith(b"Hello")
    assert len(generated.stdout.removesuffix(b"\n")) > len(b"Hello")
