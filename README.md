# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Languages: [English](README.md) | [中文](docs/readme/README.zh-CN.md) | [हिन्दी](docs/readme/README.hi.md) | [Español](docs/readme/README.es.md) | [العربية](docs/readme/README.ar.md) | [Français](docs/readme/README.fr.md) | [Português](docs/readme/README.pt.md) | [Русский](docs/readme/README.ru.md) | [Bahasa Indonesia](docs/readme/README.id.md) | [Deutsch](docs/readme/README.de.md) | [日本語](docs/readme/README.ja.md) | [한국어](docs/readme/README.ko.md) | [Italiano](docs/readme/README.it.md) | [Nederlands](docs/readme/README.nl.md)

KDA CUDA Trainer is a compact, trainable causal language model example built from scratch. Its sequence-mixing module uses Kimi Delta Attention (KDA): CUDA training uses the chunked Triton kernel from FLA, and a differentiable token-by-token PyTorch reference implementation is included for readability and CPU experiments.

This repository implements a pure-KDA teaching model (not full Kimi Linear). It uses UTF-8 bytes as tokens to avoid tokenizer dependencies, and includes next-byte language modeling, AdamW, warmup/cosine LR schedule, gradient accumulation, mixed precision, validation loss, checkpoints, and sampling.

## Terminology

For consistency across all language versions, core technical terms are kept in English: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate`, and `delta rule`.

## Requirements

- Python 3.10+
- NVIDIA GPU and CUDA-enabled PyTorch
- BF16-capable GPUs are recommended; older GPUs can use FP16

Install CUDA-matched PyTorch first, then install the project and FLA kernel:

    pip install -e '.[cuda]'

## Quick Start

Train on the included tiny text corpus:

    python -m kda_train.train --device cuda --steps 1000

Use your own UTF-8 text corpus:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

Generate text:

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Backends

- `auto`: Uses `fla.ops.kda.chunk_kda` on CUDA when `fla-core` is installed; otherwise falls back to the differentiable PyTorch reference path.
- `fla`: Forces FLA CUDA kernel.
- `reference`: Forces token-by-token PyTorch scan. This keeps a full graph and is not suitable for long-sequence training.

Use `fla-core >= 0.5.2` to ensure KDA beta sigmoid support in CUDA path.

## Algorithm Summary

Each head maintains a K×V state matrix. For each token, a per-key forget gate decays the state; then key-prediction error is computed and written back to value via delta rule. Query vectors read from the updated state. Each layer also includes input projection, beta and per-channel gate, output gating, and MLP.

This implementation prioritizes a readable training pipeline. Full Kimi Linear reproduction requires additional components such as hybrid MLA layers, short convolution, larger scale, and full training recipe.

## References

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Kimi-Linear official repository](https://github.com/MoonshotAI/Kimi-Linear)
- [Flash Linear Attention (FLA) KDA implementation](https://github.com/fla-org/flash-linear-attention)

## Development and Contribution

Install test dependencies:

    pip install -e '.[test]'
    pytest -q

Tests force the PyTorch reference backend so they can run on CPU. For CUDA kernel training path, use the `.[cuda]` extra. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

## Security

Please do not disclose potential vulnerabilities in public issues. Reporting guidance is in [SECURITY.md](SECURITY.md).

## Traffic Analytics

- Total visits:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Region/country distribution (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Visitor map (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)

  > To enable ClustrMaps, create a counter on their website and replace `YOUR_CLUSTRMAPS_TOKEN` and `YOUR_CLUSTRMAPS_SITE_ID`.
