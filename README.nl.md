# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licentie: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Talen: [English](README.md) | [中文](README.zh-CN.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Italiano](README.it.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [العربية](README.ar.md) | [Nederlands](README.nl.md) | [हिन्दी](README.hi.md) | [Bahasa Indonesia](README.id.md)

KDA CUDA Trainer is een compact, trainbaar voorbeeld van een causaal taalmodel. De sequence-mixing gebruikt Kimi Delta Attention (KDA): CUDA-training gebruikt de chunked Triton-kernel van FLA, en er is ook een differentieerbare PyTorch-referentie-implementatie per token.

## Vereisten

- Python 3.10+
- NVIDIA GPU en CUDA-geschikte PyTorch
- BF16-geschikte GPU aanbevolen; oudere GPU's kunnen FP16 gebruiken

    pip install -e '.[cuda]'

## Snelstart

    python -m kda_train.train --device cuda --steps 1000
    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Ontwikkeling

    pip install -e '.[test]'
    pytest -q

Zie [CONTRIBUTING.md](CONTRIBUTING.md) en [SECURITY.md](SECURITY.md).

## Verkeersstatistieken

- Totaal aantal bezoeken:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Land-/regioverdeling (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Bezoekerskaart (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
