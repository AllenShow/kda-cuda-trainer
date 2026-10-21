# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licenza: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer è un esempio compatto di modello linguistico causale addestrabile. Il modulo di sequence-mixing usa Kimi Delta Attention (KDA): il training CUDA sfrutta il chunked Triton kernel di FLA, insieme a una reference implementation differenziabile in PyTorch token-by-token.

## Terminologia

Per mantenere coerenza tra tutte le versioni linguistiche, i principali termini tecnici restano in inglese: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate` e `delta rule`.

## Requisiti

- Python 3.10+
- GPU NVIDIA e PyTorch con CUDA
- Consigliata GPU con BF16; su GPU più vecchie è possibile usare FP16

    pip install -e '.[cuda]'

## Avvio rapido

    python -m kda_train.train --device cuda --steps 1000

Con il tuo corpus UTF-8:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Backend

- `auto`: usa `fla.ops.kda.chunk_kda` su CUDA quando `fla-core` è installato; altrimenti usa la reference PyTorch differenziabile.
- `fla`: forza il kernel CUDA FLA.
- `reference`: forza lo scan token-by-token in PyTorch. Mantiene il grafo completo e non è adatto a sequenze lunghe.

Per il percorso CUDA è consigliato `fla-core >= 0.5.2`.

## Sintesi dell’algoritmo

Ogni head mantiene una matrice di stato K×V. Per ogni token, una forget gate per canale key attenua lo stato; poi l’errore di previsione viene calcolato e riscritto in value con la regola delta. I vettori query leggono dallo stato aggiornato. Ogni layer include anche proiezione in ingresso, beta e gate per canale, gate di uscita e MLP.

## Riferimenti

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Repository ufficiale Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Implementazione KDA di Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)

## Sviluppo e contributi

    pip install -e '.[test]'
    pytest -q

I test forzano il backend di riferimento PyTorch e possono girare su CPU. Per il training CUDA usa `.[cuda]`. Leggi [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Sicurezza

Non pubblicare potenziali vulnerabilità in issue pubbliche. Consulta [SECURITY.md](../../SECURITY.md).

## Statistiche visite

- Visite totali:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Distribuzione per paese/regione (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Mappa visitatori (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
