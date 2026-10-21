# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licença: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

语言 / Languages: [中文](README.zh-CN.md) | [English](README.en.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Italiano](README.it.md)

## Estatísticas de acesso

- Total de visitas:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Distribuição por país/região (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Mapa de visitantes (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)

  > Para ativar o ClustrMaps, crie um contador e substitua `YOUR_CLUSTRMAPS_TOKEN` e `YOUR_CLUSTRMAPS_SITE_ID`.

KDA CUDA Trainer é um exemplo compacto de modelo de linguagem causal treinável do zero. O módulo de mistura de sequência usa Kimi Delta Attention (KDA): o treinamento em CUDA usa o kernel Triton em blocos do FLA, e também há uma implementação de referência em PyTorch token a token.

Este repositório implementa um modelo didático puramente KDA (não o Kimi Linear completo). Ele usa bytes UTF-8 como tokens e inclui next-byte LM, AdamW, warmup/cosine, acumulação de gradiente, precisão mista, validação, checkpoints e geração.

## Requisitos

- Python 3.10+
- GPU NVIDIA e PyTorch com CUDA
- Recomenda-se GPU com BF16; GPUs antigas podem usar FP16

    pip install -e '.[cuda]'

## Início rápido

    python -m kda_train.train --device cuda --steps 1000
    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Desenvolvimento e contribuição

    pip install -e '.[test]'
    pytest -q

Leia [CONTRIBUTING.md](CONTRIBUTING.md) antes de abrir PRs. Para segurança: [SECURITY.md](SECURITY.md).
