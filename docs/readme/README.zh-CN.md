# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

语言 / Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer 是一个从零搭建的、可训练的小型因果语言模型示例。序列混合模块使用 Kimi Delta Attention（KDA）：CUDA 路径采用 FLA 的分块 Triton kernel，同时包含可微分的 PyTorch 逐 token 参考实现，便于阅读与 CPU 实验。

## 环境要求

- Python 3.10+
- NVIDIA GPU 与 CUDA 版 PyTorch
- 推荐 BF16 GPU；较旧 GPU 可使用 FP16

    pip install -e '.[cuda]'

## 快速开始

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

## 开发

    pip install -e '.[test]'
    pytest -q

Tests force the PyTorch reference backend so they can run on CPU. For CUDA kernel training path, use the `.[cuda]` extra. Please read [CONTRIBUTING.md](../../CONTRIBUTING.md) before opening a PR.

## Security

Please do not disclose potential vulnerabilities in public issues. Reporting guidance is in [SECURITY.md](../../SECURITY.md).

## 访问统计

- 总访问量：

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- 国家/地区分布（FlagCounter）：

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- 访问地图（ClustrMaps）：

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
