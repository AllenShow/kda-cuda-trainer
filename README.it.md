# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licenza: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

语言 / Languages: [中文](README.zh-CN.md) | [English](README.en.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Italiano](README.it.md)

## Statistiche visite

- Visite totali:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Distribuzione per paese/regione (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Mappa visitatori (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)

  > Per attivare ClustrMaps, crea un contatore e sostituisci `YOUR_CLUSTRMAPS_TOKEN` e `YOUR_CLUSTRMAPS_SITE_ID`.

KDA CUDA Trainer è un esempio compatto di modello linguistico causale addestrabile da zero. Il modulo di sequence mixing usa Kimi Delta Attention (KDA): l'addestramento CUDA sfrutta il kernel Triton a blocchi di FLA; è inclusa anche una reference implementation PyTorch token-by-token.

Questo repository implementa un modello didattico puro KDA (non il Kimi Linear completo). Usa byte UTF-8 come token e include next-byte LM, AdamW, warmup/cosine LR, gradient accumulation, mixed precision, validation loss, checkpoint e sampling.

## Requisiti

- Python 3.10+
- GPU NVIDIA e PyTorch con CUDA
- Consigliata una GPU con BF16; su GPU più vecchie si può usare FP16

    pip install -e '.[cuda]'

## Avvio rapido

    python -m kda_train.train --device cuda --steps 1000
    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Sviluppo e contributi

    pip install -e '.[test]'
    pytest -q

Leggi [CONTRIBUTING.md](CONTRIBUTING.md) prima di aprire una PR. Per la sicurezza: [SECURITY.md](SECURITY.md).
