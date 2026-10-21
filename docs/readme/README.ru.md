# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Лицензия: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer — компактный обучаемый пример причинно-следственной языковой модели. Модуль смешивания последовательностей использует Kimi Delta Attention (KDA): на CUDA применяется chunked Triton kernel из FLA, также включена дифференцируемая эталонная реализация на PyTorch по токенам.

## Требования

- Python 3.10+
- NVIDIA GPU и PyTorch с поддержкой CUDA
- Рекомендуется GPU с BF16; на старых GPU можно использовать FP16

    pip install -e '.[cuda]'

## Быстрый старт

    python -m kda_train.train --device cuda --steps 1000

Use your own UTF-8 text corpus:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8
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

## Разработка

    pip install -e '.[test]'
    pytest -q

Tests force the PyTorch reference backend so they can run on CPU. For CUDA kernel training path, use the `.[cuda]` extra. Please read [CONTRIBUTING.md](../../CONTRIBUTING.md) before opening a PR.

## Security

Please do not disclose potential vulnerabilities in public issues. Reporting guidance is in [SECURITY.md](../../SECURITY.md).

## Статистика посещений

- Общее число посещений:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Распределение по странам/регионам (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Карта посетителей (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
