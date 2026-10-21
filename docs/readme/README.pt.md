# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licença: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer é um exemplo compacto de modelo de linguagem causal treinável. O módulo de sequence-mixing usa Kimi Delta Attention (KDA): o treino em CUDA utiliza o chunked Triton kernel do FLA, com implementação de referência diferenciável em PyTorch token por token.

## Requisitos

- Python 3.10+
- GPU NVIDIA e PyTorch com CUDA
- Recomenda-se GPU com BF16; GPUs antigas podem usar FP16

    pip install -e '.[cuda]'

## Início rápido

    python -m kda_train.train --device cuda --steps 1000
    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Desenvolvimento

    pip install -e '.[test]'
    pytest -q

Veja [CONTRIBUTING.md](../../CONTRIBUTING.md) e [SECURITY.md](../../SECURITY.md).

## Estatísticas de acesso

- Total de visitas:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Distribuição por país/região (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Mapa de visitantes (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
