# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer is a compact, trainable causal language model example built from scratch. Its sequence-mixing module uses Kimi Delta Attention (KDA): CUDA training uses the chunked Triton kernel from FLA, and a differentiable token-by-token PyTorch reference implementation is included for readability and CPU experiments.

This repository implements a pure-KDA teaching model (not full Kimi Linear). It uses UTF-8 bytes as tokens to avoid tokenizer dependencies, and includes next-byte language modeling, AdamW, warmup/cosine LR schedule, gradient accumulation, mixed precision, validation loss, checkpoints, and sampling.

## Project Website

Official project homepage: [kda-cuda-trainer.pages.dev](https://kda-cuda-trainer.pages.dev/)

The website is the primary public landing page for demos and analytics-backed visitor geography, while GitHub remains the canonical source for code and releases.

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

Tests force the PyTorch reference backend so they can run on CPU. For CUDA kernel training path, use the `.[cuda]` extra. Please read [CONTRIBUTING.md](../../CONTRIBUTING.md) before opening a PR.

## Security

Please do not disclose potential vulnerabilities in public issues. Reporting guidance is in [SECURITY.md](../../SECURITY.md).

## Website Analytics (Cloudflare Free)

The existing Cloudflare Pages project serves the website and a read-only `/stats.svg`
Pages Function. The function queries Cloudflare Web Analytics, filters the project hostname
and site tag, and caches the statistics card for five minutes. Loading the card does not
increment a visitor counter. These website figures are independent of the
VisitorBadge and FlagCounter images in the GitHub Repository Traffic section below.

See [deployment and analytics configuration](../website-analytics.md) for the three
required Pages bindings. Until they are configured, or when the upstream API fails,
the card explicitly reports that statistics are unavailable rather than showing fake zeros.

## Official Website Traffic — Cloudflare

[![Website traffic: last 30 days, page views, visits, and top countries / regions](https://kda-cuda-trainer.pages.dev/stats.svg)](https://kda-cuda-trainer.pages.dev/#traffic-heading)

Cloudflare Web Analytics statistics for the project website over the last 30 days.
The website and this README share the same statistics card, refreshed every five minutes
(GitHub's image cache may delay updates). Visits are not unique visitors (UV).
GitHub repository views are excluded. Country / region percentages use all website page views.

## GitHub Repository Traffic — VisitorBadge & FlagCounter

The counters below track this repository's README image requests, separately from
the official website's Cloudflare statistics. The existing counter identifiers are
retained and shared across the README language versions.

**README views (VisitorBadge):**

![GitHub README views — VisitorBadge](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

**README visitor countries / regions (FlagCounter):**

[![GitHub README visitor countries / regions — FlagCounter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

These are approximate image-based counters: GitHub's image proxy and cache can
affect both counts and apparent regions. They are not GitHub's official Traffic
metrics. Website and README counts are reported separately and are not added together.
