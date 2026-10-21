# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licencia: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

语言 / Languages: [中文](README.zh-CN.md) | [English](README.en.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Italiano](README.it.md)

## Analíticas de visitas

- Visitas totales:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Distribución por país/región (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Mapa de visitantes (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)

  > Para activar ClustrMaps, crea un contador y reemplaza `YOUR_CLUSTRMAPS_TOKEN` y `YOUR_CLUSTRMAPS_SITE_ID`.

KDA CUDA Trainer es un ejemplo compacto de modelo de lenguaje causal entrenable desde cero. El mezclador de secuencias usa Kimi Delta Attention (KDA): en CUDA se ejecuta con el kernel Triton por bloques de FLA, y también se incluye una implementación de referencia en PyTorch token por token.

Este repositorio implementa un modelo educativo puro KDA (no Kimi Linear completo). Usa bytes UTF-8 como tokens e incluye next-byte LM, AdamW, warmup/cosine LR, acumulación de gradientes, precisión mixta, validación, checkpoints y muestreo.

## Requisitos

- Python 3.10+
- GPU NVIDIA y PyTorch con CUDA
- Se recomienda GPU con BF16; en GPUs antiguas puedes usar FP16

    pip install -e '.[cuda]'

## Inicio rápido

    python -m kda_train.train --device cuda --steps 1000
    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Desarrollo y contribución

    pip install -e '.[test]'
    pytest -q

Revisa [CONTRIBUTING.md](CONTRIBUTING.md) antes de abrir un PR. Para reportes de seguridad: [SECURITY.md](SECURITY.md).
