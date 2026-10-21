# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer は、ゼロから構築した小規模な学習可能因果言語モデルです。系列混合には Kimi Delta Attention (KDA) を使用し、CUDA 学習では FLA のチャンク化 Triton カーネルを使います。可読性と CPU 実験向けに、PyTorch の逐次参照実装も含まれます。

## 動作環境

- Python 3.10+
- NVIDIA GPU と CUDA 対応 PyTorch
- BF16 対応 GPU 推奨（古い GPU は FP16 を使用可能）

    pip install -e '.[cuda]'

## クイックスタート

    python -m kda_train.train --device cuda --steps 1000
    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## 開発

    pip install -e '.[test]'
    pytest -q

[CONTRIBUTING.md](../../CONTRIBUTING.md) と [SECURITY.md](../../SECURITY.md) を参照してください。

## アクセス統計

- 総アクセス数:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- 地域/国別分布 (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- 訪問マップ (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
