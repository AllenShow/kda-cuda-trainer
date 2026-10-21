# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer は、ゼロから構築した小型の学習可能な因果言語モデルの例です。系列混合モジュールは Kimi Delta Attention（KDA）を使用します。CUDA 学習では FLA のチャンク化された Triton kernel を使い、可読性と CPU 実験のために、token ごとに処理する微分可能な PyTorch 参照実装も含まれています。

このリポジトリは純粋な KDA の教材モデルであり、完全な Kimi Linear ではありません。tokenizer への依存を避けるため UTF-8 の各バイトを token として使用し、次バイト言語モデリング、AdamW、warmup/cosine 学習率スケジュール、勾配累積、混合精度、検証損失、checkpoint、サンプリングを備えています。

## プロジェクト公式サイト

プロジェクトの公式ホームページ: [kda-cuda-trainer.pages.dev](https://kda-cuda-trainer.pages.dev/)

公式サイトはデモや分析データに基づく訪問者の地域分布を公開する主な窓口です。コードとリリースの正規の配布元は GitHub です。

## 用語

多言語版での一貫性を保つため、主要な技術用語は英語のまま使用します：`token`、`backend`、`kernel`、`head`、`layer`、`query`、`key`、`value`、`gate`、`delta rule`。

## 動作環境

- Python 3.10+
- NVIDIA GPU と CUDA 対応 PyTorch
- BF16 対応 GPU 推奨（古い GPU は FP16 利用可）

まず CUDA に対応する PyTorch をインストールしてから、プロジェクトと FLA kernel をインストールしてください：

    pip install -e '.[cuda]'

## クイックスタート

付属の小規模テキストコーパスで学習：

    python -m kda_train.train --device cuda --steps 1000

独自の UTF-8 テキストコーパスを使用：

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

テキストを生成：

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## バックエンド

- `auto`: CUDA 環境で `fla-core` があれば `fla.ops.kda.chunk_kda` を使用し、なければ微分可能な PyTorch 参照実装にフォールバックします。
- `fla`: FLA CUDA カーネルを強制します。
- `reference`: トークン単位の PyTorch スキャンを強制します。完全な計算グラフを保持するため、長系列には不向きです。

CUDA 経路で KDA beta sigmoid のサポートを確保するため、`fla-core >= 0.5.2` を使用してください。

## アルゴリズム概要

各 head は K×V 状態行列を保持します。各 token に対し、key ごとの忘却 gate が状態を減衰させ、続いて key の予測誤差を計算し、delta rule で value に書き戻します。query ベクトルは更新後の状態を読み取ります。各 layer は入力投影、beta とチャネルごとの gate、出力ゲーティング、MLP も備えます。

この実装は学習処理の読みやすさを重視しています。完全な Kimi Linear の再現には、ハイブリッド MLA layer、短い畳み込み、大規模化、完全な学習レシピなどの追加要素が必要です。

## 参考資料

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Kimi-Linear 公式リポジトリ](https://github.com/MoonshotAI/Kimi-Linear)
- [Flash Linear Attention（FLA）KDA 実装](https://github.com/fla-org/flash-linear-attention)

## 開発と貢献

テスト用依存関係をインストール：

    pip install -e '.[test]'
    pytest -q

テストは PyTorch reference backend を強制するため CPU で実行できます。CUDA kernel による学習では `.[cuda]` extra を使用してください。PR を作成する前に [CONTRIBUTING.md](../../CONTRIBUTING.md) をお読みください。

## セキュリティ

潜在的な脆弱性を公開 issue に投稿しないでください。報告方法は [SECURITY.md](../../SECURITY.md) を参照してください。

## サイト統計の設定（Cloudflare Free）

既存の Cloudflare Pages プロジェクトが公式サイトと読み取り専用の `/stats.svg` Pages Function を提供します。この関数は Cloudflare Web Analytics に問い合わせ、プロジェクトの hostname と site tag で絞り込み、統計カードを 5 分間キャッシュします。カードの読み込みでは訪問カウンターは増えません。公式サイトの数値は、下の GitHub リポジトリ統計にある VisitorBadge と FlagCounter の画像とは独立しています。

必要な 3 つの Pages バインディングについては[デプロイと統計の設定](../website-analytics.md)を参照してください。設定が不足している場合や上流 API が失敗した場合、カードは架空のゼロを表示せず、統計が利用できないことを明示します。

## 公式サイトのアクセス統計 — Cloudflare

[![公式サイトのアクセス統計 — Cloudflare](https://kda-cuda-trainer.pages.dev/stats.svg)](https://kda-cuda-trainer.pages.dev/#traffic-heading)

プロジェクト公式サイトの過去 30 日間の Cloudflare Web Analytics 統計です。サイトとこの README は同じカードを表示し、5 分ごとに更新します（GitHub の画像キャッシュで遅れる場合があります）。Visits はユニーク訪問者数（UV）ではありません。GitHub リポジトリの閲覧数は含みません。国・地域の割合は公式サイトの全ページビューを分母にします。

## GitHub リポジトリのアクセス統計 — VisitorBadge と FlagCounter

以下のカウンターは、このリポジトリの README 画像へのリクエストを計測し、公式サイトの Cloudflare 統計とは分けて表示します。既存のカウンター識別子を引き継ぎ、README の全言語版で共有します。

**README 閲覧カウンター（VisitorBadge）:**

![README 閲覧カウンター（VisitorBadge）](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

**README 訪問者の国・地域（FlagCounter）:**

[![README 訪問者の国・地域（FlagCounter）](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

これらは画像に基づく概算のカウンターです。GitHub の画像プロキシとキャッシュは、数値や表示地域に影響する場合があります。GitHub 公式の Traffic 指標ではありません。公式サイトと README の数値は別々に表示し、合算しません。
