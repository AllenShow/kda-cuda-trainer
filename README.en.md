# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

语言 / Languages: [中文](README.zh-CN.md) | [English](README.en.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Italiano](README.it.md)

## Traffic Analytics

- Total visits:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Region/country distribution (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Visitor map (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)

  > To enable ClustrMaps, create a counter on their website and replace `YOUR_CLUSTRMAPS_TOKEN` and `YOUR_CLUSTRMAPS_SITE_ID`.

KDA CUDA Trainer is a compact, trainable causal language model example built from scratch. Its sequence-mixing module uses Kimi Delta Attention (KDA): CUDA training uses the chunked Triton kernel from FLA, and a differentiable token-by-token PyTorch reference implementation is also included for readability and CPU experiments.

This repository implements a pure-KDA teaching model (not full Kimi Linear). It uses UTF-8 bytes as tokens to avoid tokenizer dependencies, and includes next-byte language modeling, AdamW, warmup/cosine LR schedule, gradient accumulation, mixed precision, validation loss, checkpoints, and sampling.

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

## Development and Contribution

Install test dependencies:

    pip install -e '.[test]'
    pytest -q

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR. For security reports, see [SECURITY.md](SECURITY.md).
