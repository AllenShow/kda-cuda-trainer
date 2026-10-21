# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Lizenz: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sprachen: [English](README.md) | [中文](README.zh-CN.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Italiano](README.it.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [العربية](README.ar.md) | [Nederlands](README.nl.md) | [हिन्दी](README.hi.md) | [Bahasa Indonesia](README.id.md)

KDA CUDA Trainer ist ein kompaktes, trainierbares Beispiel für ein kausales Sprachmodell. Das Sequenz-Mixing nutzt Kimi Delta Attention (KDA): Auf CUDA wird der chunked Triton-Kernel aus FLA verwendet; zusätzlich gibt es eine differenzierbare PyTorch-Referenzimplementierung (tokenweise) für Lesbarkeit und CPU-Experimente.

## Voraussetzungen

- Python 3.10+
- NVIDIA GPU und CUDA-fähiges PyTorch
- BF16-fähige GPUs werden empfohlen; ältere GPUs können FP16 nutzen

    pip install -e '.[cuda]'

## Schnellstart

    python -m kda_train.train --device cuda --steps 1000
    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Entwicklung

    pip install -e '.[test]'
    pytest -q

Siehe [CONTRIBUTING.md](CONTRIBUTING.md) und [SECURITY.md](SECURITY.md).

## Zugriffsstatistiken

- Gesamtaufrufe:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Länder-/Regionsverteilung (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Besucherkarte (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
