# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

语言 / Languages: [中文](README.zh-CN.md) | [English](README.en.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Italiano](README.it.md)

## Statistiques de visites

- Nombre total de visites :

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Répartition par pays/région (FlagCounter) :

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Carte des visiteurs (ClustrMaps) :

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)

  > Pour activer ClustrMaps, créez un compteur puis remplacez `YOUR_CLUSTRMAPS_TOKEN` et `YOUR_CLUSTRMAPS_SITE_ID`.

KDA CUDA Trainer est un exemple compact de modèle de langage causal entraînable. Le module de mélange de séquence utilise Kimi Delta Attention (KDA) : l'entraînement CUDA passe par le kernel Triton en blocs de FLA, et une implémentation de référence PyTorch différentiable token par token est aussi incluse.

Ce dépôt fournit un modèle pédagogique purement KDA (pas Kimi Linear complet). Les tokens sont des octets UTF-8, et le code inclut next-byte LM, AdamW, warmup/cosine, accumulation de gradients, précision mixte, validation, checkpoints et génération.

## Prérequis

- Python 3.10+
- GPU NVIDIA et PyTorch avec CUDA
- Un GPU compatible BF16 est recommandé ; FP16 fonctionne sur du matériel plus ancien

Installez d'abord la version de PyTorch adaptée à votre CUDA, puis :

    pip install -e '.[cuda]'

## Démarrage rapide

    python -m kda_train.train --device cuda --steps 1000
    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Développement et contribution

    pip install -e '.[test]'
    pytest -q

Consultez [CONTRIBUTING.md](CONTRIBUTING.md) avant toute PR. Pour la sécurité : [SECURITY.md](SECURITY.md).
