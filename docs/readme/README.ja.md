# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer は、ゼロから構築した小規模な学習可能因果言語モデルです。系列混合には Kimi Delta Attention (KDA) を使用し、CUDA 学習では FLA の分割 Triton カーネルを使います。可読性と CPU 実験向けに、PyTorch の逐次参照実装も含まれます。

## 用語

多言語版での一貫性を保つため、主要な技術用語は英語のまま使用します：`token`、`backend`、`kernel`、`head`、`layer`、`query`、`key`、`value`、`gate`、`delta rule`。

## 動作環境

- Python 3.10+
- NVIDIA GPU と CUDA 対応 PyTorch
- BF16 対応 GPU 推奨（古い GPU は FP16 利用可）

    pip install -e '.[cuda]'

## クイックスタート

    python -m kda_train.train --device cuda --steps 1000

独自の UTF-8 テキストコーパスを使う場合：

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## バックエンド

- `auto`: CUDA 環境で `fla-core` があれば `fla.ops.kda.chunk_kda` を使用し、なければ微分可能な PyTorch 参照実装にフォールバックします。
- `fla`: FLA CUDA カーネルを強制します。
- `reference`: トークン単位の PyTorch スキャンを強制します。完全な計算グラフを保持するため、長系列には不向きです。

CUDA 経路では `fla-core >= 0.5.2` を推奨します。

## アルゴリズム概要

各ヘッドは K×V 状態行列を保持します。各トークンで key チャネルごとの忘却ゲートにより状態を減衰させ、予測誤差を計算し、delta rule で value へ書き戻します。query ベクトルは更新後の状態から出力を読み取ります。各層には入力投影、beta とチャネルゲート、出力ゲート、MLP も含まれます。

## 参考資料

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Kimi-Linear 公式リポジトリ](https://github.com/MoonshotAI/Kimi-Linear)
- [Flash Linear Attention（FLA）KDA 実装](https://github.com/fla-org/flash-linear-attention)

## 開発と貢献

    pip install -e '.[test]'
    pytest -q

テストは PyTorch reference バックエンドを強制するため CPU で実行できます。CUDA カーネル学習には `.[cuda]` を使ってください。[CONTRIBUTING.md](../../CONTRIBUTING.md) を参照してください。

## セキュリティ

潜在的な脆弱性を公開 issue に投稿しないでください。報告方法は [SECURITY.md](../../SECURITY.md) を参照してください。

## アクセス統計

- 総アクセス数：

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- 国/地域分布（FlagCounter）：

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- 訪問マップ（ClustrMaps）：

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
