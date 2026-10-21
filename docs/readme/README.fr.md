# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer est un exemple compact de modèle de langage causal entraînable. Le module de mélange de séquences utilise Kimi Delta Attention (KDA) : l’entraînement CUDA s’appuie sur le kernel Triton en blocs de FLA, avec une implémentation de référence PyTorch différentiable token par token.

## Terminologie

Pour assurer la cohérence entre toutes les versions linguistiques, les termes techniques clés restent en anglais : `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate` et `delta rule`.

## Prérequis

- Python 3.10+
- GPU NVIDIA et PyTorch avec CUDA
- GPU BF16 recommandé ; FP16 possible sur matériel plus ancien

    pip install -e '.[cuda]'

## Démarrage rapide

    python -m kda_train.train --device cuda --steps 1000

Avec votre propre corpus UTF-8 :

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Backends

- `auto` : utilise `fla.ops.kda.chunk_kda` sur CUDA quand `fla-core` est installé ; sinon bascule vers la référence PyTorch différentiable.
- `fla` : force le kernel CUDA FLA.
- `reference` : force un scan PyTorch token par token. Ce mode conserve tout le graphe et n’est pas adapté aux longues séquences.

Utilisez `fla-core >= 0.5.2` pour le chemin CUDA.

## Résumé de l’algorithme

Chaque tête maintient une matrice d’état K×V. Pour chaque token, une porte d’oubli par canal key réduit l’état, puis l’erreur de prédiction est calculée et réécrite dans value via la règle delta. Les requêtes lisent l’état mis à jour. Chaque couche inclut aussi projection d’entrée, beta et porte par canal, porte de sortie et MLP.

## Références

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Dépôt officiel Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Implémentation KDA de Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)

## Développement et contribution

    pip install -e '.[test]'
    pytest -q

Les tests forcent le backend de référence PyTorch et peuvent donc tourner sur CPU. Pour le chemin CUDA, utilisez `.[cuda]`. Voir [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Sécurité

Merci de ne pas divulguer publiquement les vulnérabilités potentielles. Voir [SECURITY.md](../../SECURITY.md).

## Statistiques de visites

- Nombre total de visites :

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Répartition pays/régions (FlagCounter) :

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)
