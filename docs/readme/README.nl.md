# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licentie: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Talen: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer is een compact, trainbaar voorbeeld van een causaal taalmodel. De sequence-mixing gebruikt Kimi Delta Attention (KDA): CUDA-training gebruikt de chunked Triton-kernel van FLA, en er is ook een differentieerbare PyTorch-referentie-implementatie per token.

## Terminologie

Voor consistentie tussen alle taalversies blijven de belangrijkste technische termen in het Engels: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate` en `delta rule`.

## Vereisten

- Python 3.10+
- NVIDIA GPU en CUDA-geschikte PyTorch
- BF16-geschikte GPU aanbevolen; oudere GPU's kunnen FP16 gebruiken

    pip install -e '.[cuda]'

## Snelstart

    python -m kda_train.train --device cuda --steps 1000

Met je eigen UTF-8-tekstcorpus:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Backends

- `auto`: gebruikt op CUDA `fla.ops.kda.chunk_kda` als `fla-core` is geïnstalleerd; anders valt het terug op de differentieerbare PyTorch-referentie.
- `fla`: forceert de FLA CUDA-kernel.
- `reference`: forceert token-by-token PyTorch-scan. Dit bewaart de volledige grafiek en is niet geschikt voor lange sequenties.

Voor het CUDA-pad wordt `fla-core >= 0.5.2` aanbevolen.

## Algoritme-overzicht

Elke head houdt een K×V-statusmatrix bij. Voor elk token dempt een per-key-kanaal forget gate de status, daarna wordt de voorspellingsfout berekend en via de delta-regel teruggeschreven naar value. Query-vectoren lezen uit de bijgewerkte status. Elke laag bevat ook inputprojectie, beta en per-kanaal gate, output-gating en MLP.

## Referenties

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Officiële Kimi-Linear-repository](https://github.com/MoonshotAI/Kimi-Linear)
- [Flash Linear Attention (FLA) KDA-implementatie](https://github.com/fla-org/flash-linear-attention)

## Ontwikkeling en bijdragen

    pip install -e '.[test]'
    pytest -q

Tests forceren de PyTorch-reference-backend en kunnen daarom op CPU draaien. Gebruik voor CUDA-training `.[cuda]`. Zie [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Beveiliging

Publiceer mogelijke kwetsbaarheden niet in openbare issues. Zie [SECURITY.md](../../SECURITY.md).

## Verkeersstatistieken

- Totaal aantal bezoeken:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Land-/regioverdeling (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)
