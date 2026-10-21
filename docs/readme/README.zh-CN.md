# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

语言 / Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer 是一个从零搭建的、可训练的小型因果语言模型示例。序列混合模块使用 Kimi Delta Attention（KDA）：CUDA 训练使用 FLA 提供的分块 Triton kernel，同时提供可微的 PyTorch 逐 token 参考实现，便于阅读与 CPU 小实验。

本仓库实现的是纯 KDA 教学模型（不是完整 Kimi Linear）。它使用 UTF-8 字节作为 token，避免 tokenizer 依赖；训练流程包含 next-byte 语言建模、AdamW、warmup/cosine 学习率、梯度累积、混合精度、验证损失、checkpoint 与采样。

## 环境要求

- Python 3.10+
- NVIDIA GPU 与 CUDA 版 PyTorch
- 推荐支持 BF16 的 GPU；旧卡可使用 FP16

先安装与 CUDA 版本匹配的 PyTorch，再安装项目和 FLA kernel：

    pip install -e '.[cuda]'

## 快速开始

使用仓库内置小语料训练：

    python -m kda_train.train --device cuda --steps 1000

使用你自己的 UTF-8 文本语料：

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

生成文本：

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## 后端

- `auto`：若在 CUDA 环境且安装了 `fla-core`，则调用 `fla.ops.kda.chunk_kda`；否则回退到可微 PyTorch 参考实现。
- `fla`：强制使用 FLA CUDA kernel。
- `reference`：强制逐 token PyTorch scan。该路径会保留完整计算图，不适合长序列训练。

CUDA 路径建议使用 `fla-core >= 0.5.2`，以确保 KDA beta sigmoid 选项可用。

## 算法概要

每个 head 维护一个 K×V 状态矩阵。对每个 token，先用逐 key 通道遗忘门衰减状态，再计算 key 对当前状态的预测误差，并通过 delta rule 写回 value。query 向量从更新后的状态读取输出。每层还包括输入投影、beta 与逐通道 gate、输出门控和 MLP。

本实现优先保证训练流程可读性。若要复现完整 Kimi Linear，还需要混合 MLA 层、短卷积、更大模型规模与完整训练 recipe。

## 参考

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Kimi-Linear 官方仓库](https://github.com/MoonshotAI/Kimi-Linear)
- [Flash Linear Attention（FLA）KDA 实现](https://github.com/fla-org/flash-linear-attention)

## 开发与贡献

安装测试依赖：

    pip install -e '.[test]'
    pytest -q

测试强制使用 PyTorch reference 后端，因此可在 CPU 上运行。若要训练 CUDA kernel 路径，请使用 `.[cuda]` extra。提交前请阅读 [CONTRIBUTING.md](../../CONTRIBUTING.md)。

## 安全

请不要在公开 issue 中披露潜在漏洞。报告方式见 [SECURITY.md](../../SECURITY.md)。

## 访问统计

- 总访问量：

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- 国家/地区分布（FlagCounter）：

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- 访问地图（ClustrMaps）：

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
