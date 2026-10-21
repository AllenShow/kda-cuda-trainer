# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

语言 / Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer 是一个从零搭建的、可训练的小型因果语言模型示例。序列混合模块使用 Kimi Delta Attention（KDA）：CUDA 训练使用 FLA 提供的分块 Triton kernel，同时提供可微的 PyTorch 逐 token 参考实现，便于阅读与 CPU 实验。

本仓库实现的是纯 KDA 教学模型，并非完整的 Kimi Linear。它使用 UTF-8 字节作为 token，避免 tokenizer 依赖；包含 next-byte 语言建模、AdamW、warmup/cosine 学习率调度、梯度累积、混合精度、验证损失、checkpoint 和采样。

## 项目官网

官方项目首页：[kda-cuda-trainer.pages.dev](https://kda-cuda-trainer.pages.dev/)

官网是展示演示和基于统计数据的访客地区分布的主要公共入口；GitHub 是代码与发布版本的权威来源。

## 术语约定

为保证多语言版本一致性，核心技术术语统一保留英文：`token`、`backend`、`kernel`、`head`、`layer`、`query`、`key`、`value`、`gate`、`delta rule`。

## 环境要求

- Python 3.10+
- NVIDIA GPU 与 CUDA 版 PyTorch
- 推荐支持 BF16 的 GPU；旧卡可使用 FP16

先安装与 CUDA 匹配的 PyTorch，再安装项目和 FLA kernel：

    pip install -e '.[cuda]'

## 快速开始

使用内置的小型文本语料训练：

    python -m kda_train.train --device cuda --steps 1000

使用自己的 UTF-8 文本语料：

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

生成文本：

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## 后端

- `auto`：若在 CUDA 环境且安装了 `fla-core`，则调用 `fla.ops.kda.chunk_kda`；否则回退到可微 PyTorch 参考实现。
- `fla`：强制使用 FLA CUDA kernel。
- `reference`：强制逐 token PyTorch scan。该路径会保留完整计算图，不适合长序列训练。

使用 `fla-core >= 0.5.2`，以确保 CUDA 路径支持 KDA beta sigmoid。

## 算法概要

每个 head 维护一个 K×V 状态矩阵。对于每个 token，逐 key 的遗忘 gate 先使状态衰减，随后计算 key 预测误差，并通过 delta rule 写回 value。query 向量从更新后的状态读取输出。每个 layer 还包含输入投影、beta 与逐通道 gate、输出门控和 MLP。

本实现优先保证训练流程清晰易读。完整复现 Kimi Linear 还需要混合 MLA layer、短卷积、更大规模和完整训练方案等组件。

## 参考

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Kimi-Linear 官方仓库](https://github.com/MoonshotAI/Kimi-Linear)
- [Flash Linear Attention（FLA）KDA 实现](https://github.com/fla-org/flash-linear-attention)

## 开发与贡献

安装测试依赖：

    pip install -e '.[test]'
    pytest -q

测试强制使用 PyTorch reference backend，因此可在 CPU 上运行。CUDA kernel 训练路径使用 `.[cuda]` 扩展。提交 PR 前请阅读 [CONTRIBUTING.md](../../CONTRIBUTING.md)。

## 安全

请不要在公开 issue 中披露潜在漏洞。报告方式见 [SECURITY.md](../../SECURITY.md)。

## 网站统计配置（Cloudflare 免费版）

现有 Cloudflare Pages 项目提供官网和只读的 `/stats.svg` Pages Function。该函数查询 Cloudflare Web Analytics，按项目域名和 site tag 筛选数据，并将统计卡片缓存五分钟。加载卡片不会增加访问计数。这些官网数据独立于下方 GitHub 仓库访问统计中的 VisitorBadge 和 FlagCounter 图片。

三个必需 Pages 绑定的配置方法见[部署与统计配置](../website-analytics.md)。配置未完成或上游 API 失败时，卡片会明确显示统计不可用，不会伪造零值。

## 官网访问统计 — Cloudflare

[![官网访问统计 — Cloudflare](https://kda-cuda-trainer.pages.dev/stats.svg)](https://kda-cuda-trainer.pages.dev/#traffic-heading)

这是项目官网近 30 天的 Cloudflare Web Analytics 统计。官网与本 README 共用同一张统计卡片，每五分钟刷新一次（GitHub 的图片缓存可能延迟更新）。Visits 不等于去重访客数（UV）。数据不包含 GitHub 仓库浏览量；国家/地区占比以官网全部页面浏览量为分母。

## GitHub 仓库访问统计 — VisitorBadge 与 FlagCounter

下方计数器统计本仓库 README 的图片请求，与官网的 Cloudflare 统计分开展示。沿用原有计数器标识，所有 README 语言版本共用。

**README 浏览计数（VisitorBadge）：**

![README 浏览计数（VisitorBadge）](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

**README 访客国家/地区（FlagCounter）：**

[![README 访客国家/地区（FlagCounter）](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

这些是基于图片的近似计数：GitHub 图片代理与缓存可能影响计数和显示的地区。它们并非 GitHub 官方 Traffic 指标。官网与 README 的计数分开展示，不相加。
