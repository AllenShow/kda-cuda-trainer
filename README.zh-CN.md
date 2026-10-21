# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

语言 / Languages: [中文](README.zh-CN.md) | [English](README.en.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Italiano](README.it.md)

## 访问统计

- 总访问量（Visitor Count）：

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- 访问地区统计（FlagCounter）：

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- 访问地图统计（ClustrMaps）：

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)

  > 首次启用 ClustrMaps：到官网创建计数器后，将 `YOUR_CLUSTRMAPS_TOKEN` 和 `YOUR_CLUSTRMAPS_SITE_ID` 替换为你的值。

一个从空项目搭起的、可训练的小型因果语言模型示例。序列混合模块使用 Kimi Delta Attention（KDA）：CUDA 训练通过 FLA 提供的分块 Triton kernel 执行；项目内另有一个可微的 PyTorch 逐 token 参考实现，方便阅读算法和在 CPU 上跑小型实验。

这个项目实现的是一个纯 KDA 的教学模型，不是完整的 Kimi Linear。它使用 UTF-8 字节作为 token，避免额外 tokenizer 依赖；训练代码包括 next-byte 语言建模、AdamW、warmup/cosine 学习率、梯度累积、混合精度、验证 loss、checkpoint 和采样。

## 环境

- Python 3.10 或更新版本
- NVIDIA GPU 和 CUDA 版 PyTorch
- 建议使用支持 BF16 的 GPU；较旧 GPU 可以选择 FP16

先按 PyTorch 官方安装说明安装与你的 CUDA 版本匹配的 PyTorch，再安装项目和 FLA CUDA kernel：

    pip install -e '.[cuda]'

## 快速开始

在 NVIDIA CUDA 机器上用仓库内的小文本训练：

    python -m kda_train.train --device cuda --steps 1000

训练默认读取 data/tiny.txt，checkpoint 写入 outputs/kda.pt。指定自己的 UTF-8 文本：

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

采样生成：

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## 开发与贡献

先安装测试依赖：

    pip install -e '.[test]'
    pytest -q

提交改动前请阅读 [贡献指南](CONTRIBUTING.md)。安全问题报告方式见 [SECURITY.md](SECURITY.md)。
