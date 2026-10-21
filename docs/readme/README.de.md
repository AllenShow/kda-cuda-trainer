# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Lizenz: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Sprachen: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer ist ein kompaktes, trainierbares Beispiel für ein kausales Sprachmodell. Das Sequence-Mixing nutzt Kimi Delta Attention (KDA): Für CUDA-Training wird der chunked Triton-Kernel aus FLA verwendet. Zusätzlich gibt es eine differenzierbare PyTorch-Referenzimplementierung pro Token.

## Voraussetzungen

- Python 3.10+
- NVIDIA-GPU und CUDA-fähiges PyTorch
- BF16-fähige GPUs werden empfohlen; ältere GPUs können FP16 nutzen

    pip install -e '.[cuda]'

## Schnellstart

    python -m kda_train.train --device cuda --steps 1000

Mit eigenem UTF-8-Textkorpus:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Backends

- `auto`: Verwendet auf CUDA mit installiertem `fla-core` `fla.ops.kda.chunk_kda`, sonst die differenzierbare PyTorch-Referenz.
- `fla`: Erzwingt den FLA-CUDA-Kernel.
- `reference`: Erzwingt tokenweises PyTorch-Scanning. Diese Variante hält den vollständigen Graphen und ist für lange Sequenzen ungeeignet.

Für den CUDA-Pfad wird `fla-core >= 0.5.2` empfohlen.

## Algorithmusüberblick

Jeder Head hält eine K×V-Zustandsmatrix. Pro Token wird der Zustand zunächst mit einem key-kanalweisen Forget-Gate gedämpft, anschließend der Vorhersagefehler berechnet und per Delta-Regel in Value zurückgeschrieben. Query liest aus dem aktualisierten Zustand. Jede Schicht enthält zusätzlich Input-Projektion, Beta und kanalweises Gate, Output-Gating sowie MLP.

## Referenzen

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Kimi-Linear offizielles Repository](https://github.com/MoonshotAI/Kimi-Linear)
- [Flash Linear Attention (FLA) KDA-Implementierung](https://github.com/fla-org/flash-linear-attention)

## Entwicklung und Beitrag

    pip install -e '.[test]'
    pytest -q

Tests erzwingen den PyTorch-Reference-Backend und laufen daher auf CPU. Für CUDA-Training nutze `.[cuda]`. Siehe [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Sicherheit

Bitte veröffentliche potenzielle Sicherheitslücken nicht in öffentlichen Issues. Siehe [SECURITY.md](../../SECURITY.md).

## Zugriffsstatistiken

- Gesamtbesuche:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Länder-/Regionsverteilung (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Besucherkarte (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
