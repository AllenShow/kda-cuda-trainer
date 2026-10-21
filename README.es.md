# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licencia: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Languages: [English](README.md) | [中文](README.zh-CN.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Italiano](README.it.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [العربية](README.ar.md) | [Nederlands](README.nl.md) | [हिन्दी](README.hi.md) | [Bahasa Indonesia](README.id.md)

KDA CUDA Trainer es un ejemplo compacto de modelo de lenguaje causal entrenable. El módulo de mezcla de secuencia utiliza Kimi Delta Attention (KDA): el entrenamiento en CUDA usa el chunked Triton kernel de FLA, y también incluye una implementación de referencia diferenciable en PyTorch token por token.

## Requisitos

- Python 3.10+
- GPU NVIDIA y PyTorch con CUDA
- Se recomienda GPU con BF16; en GPUs antiguas se puede usar FP16

    pip install -e '.[cuda]'

## Inicio rápido

    python -m kda_train.train --device cuda --steps 1000
    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Desarrollo

    pip install -e '.[test]'
    pytest -q

Consulta [CONTRIBUTING.md](CONTRIBUTING.md) y [SECURITY.md](SECURITY.md).

## Estadísticas de visitas

- Visitas totales:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Distribución por país/región (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Mapa de visitantes (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
