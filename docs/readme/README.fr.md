# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer est un exemple compact de modèle de langage causal entraînable. Le module de mélange de séquences utilise Kimi Delta Attention (KDA) : l'entraînement CUDA s'appuie sur le kernel Triton en blocs de FLA, avec une implémentation de référence PyTorch différentiable token par token.

## Prérequis

- Python 3.10+
- GPU NVIDIA et PyTorch avec CUDA
- GPU BF16 recommandé ; FP16 possible sur matériel plus ancien

    pip install -e '.[cuda]'

## Démarrage rapide

    python -m kda_train.train --device cuda --steps 1000
    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Développement

    pip install -e '.[test]'
    pytest -q

Voir [CONTRIBUTING.md](../../CONTRIBUTING.md) et [SECURITY.md](../../SECURITY.md).

## Statistiques de visites

- Nombre total de visites :

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Répartition pays/régions (FlagCounter) :

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Carte des visiteurs (ClustrMaps) :

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
