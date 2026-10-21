# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer est un exemple compact de modèle de langage causal entraînable construit de zéro. Son module de mélange de séquences utilise Kimi Delta Attention (KDA) : l’entraînement CUDA emploie le kernel Triton par blocs de FLA, et une implémentation de référence PyTorch différentiable token par token facilite la lecture et les expériences sur CPU.

Ce dépôt implémente un modèle pédagogique purement KDA, pas Kimi Linear complet. Il utilise les octets UTF-8 comme token pour éviter les dépendances de tokenizer et comprend la modélisation du prochain octet, AdamW, un calendrier de LR warmup/cosinus, l’accumulation de gradients, la précision mixte, la perte de validation, les checkpoints et l’échantillonnage.

## Site web du projet

Page officielle du projet: [kda-cuda-trainer.pages.dev](https://kda-cuda-trainer.pages.dev/)

Le site web est le principal point d’entrée public pour les démonstrations et la répartition géographique des visiteurs fondée sur les statistiques ; GitHub reste la source de référence du code et des versions.

## Terminologie

Pour assurer la cohérence entre toutes les versions linguistiques, les termes techniques clés restent en anglais : `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate` et `delta rule`.

## Prérequis

- Python 3.10+
- GPU NVIDIA et PyTorch avec CUDA
- GPU BF16 recommandé ; FP16 possible sur matériel plus ancien

Installez d’abord PyTorch adapté à votre CUDA, puis le projet et le kernel FLA :

    pip install -e '.[cuda]'

## Démarrage rapide

Entraînez le modèle sur le petit corpus de texte fourni :

    python -m kda_train.train --device cuda --steps 1000

Utilisez votre propre corpus de texte UTF-8 :

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

Générez du texte :

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Backends

- `auto` : utilise `fla.ops.kda.chunk_kda` sur CUDA quand `fla-core` est installé ; sinon bascule vers la référence PyTorch différentiable.
- `fla` : force le kernel CUDA FLA.
- `reference` : force un scan PyTorch token par token. Ce mode conserve tout le graphe et n’est pas adapté aux longues séquences.

Utilisez `fla-core >= 0.5.2` pour garantir la prise en charge de KDA beta sigmoid dans le chemin CUDA.

## Résumé de l’algorithme

Chaque head maintient une matrice d’état K×V. Pour chaque token, un gate d’oubli par key atténue l’état ; l’erreur de prédiction de key est ensuite calculée et réécrite dans value via delta rule. Les vecteurs query lisent l’état mis à jour. Chaque layer comprend aussi une projection d’entrée, beta et un gate par canal, une modulation de sortie et un MLP.

Cette implémentation privilégie un entraînement facile à lire. Reproduire Kimi Linear complet nécessite notamment des layers hybrides MLA, une convolution courte, une échelle supérieure et une recette d’entraînement complète.

## Références

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Dépôt officiel Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Implémentation KDA de Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)

## Développement et contribution

Installez les dépendances de test :

    pip install -e '.[test]'
    pytest -q

Les tests imposent le backend de référence PyTorch et peuvent donc tourner sur CPU. Pour l’entraînement avec le kernel CUDA, utilisez l’extra `.[cuda]`. Lisez [CONTRIBUTING.md](../../CONTRIBUTING.md) avant d’ouvrir une PR.

## Sécurité

Merci de ne pas divulguer publiquement les vulnérabilités potentielles. Voir [SECURITY.md](../../SECURITY.md).

## Configuration des statistiques web (Cloudflare Free)

Le projet Cloudflare Pages existant sert le site et une Pages Function en lecture seule à l’adresse `/stats.svg`. La fonction interroge Cloudflare Web Analytics, filtre le hostname et le site tag du projet et met la carte en cache pendant cinq minutes. Charger la carte n’incrémente aucun compteur de visites. Ces chiffres du site sont indépendants des images VisitorBadge et FlagCounter de la section consacrée au trafic GitHub ci-dessous.

Consultez la [configuration du déploiement et des statistiques](../website-analytics.md) pour les trois bindings Pages requis. En cas de configuration manquante ou d’échec de l’API en amont, la carte indique explicitement que les statistiques sont indisponibles, sans afficher de faux zéros.

## Trafic du site officiel — Cloudflare

[![Trafic du site officiel — Cloudflare](https://kda-cuda-trainer.pages.dev/stats.svg)](https://kda-cuda-trainer.pages.dev/#traffic-heading)

Statistiques Cloudflare Web Analytics du site du projet sur les 30 derniers jours. Le site et ce README partagent la même carte, actualisée toutes les cinq minutes (le cache d’images de GitHub peut retarder la mise à jour). Visits ne désigne pas les visiteurs uniques (UV). Les vues du dépôt GitHub sont exclues. Les pourcentages par pays/région utilisent toutes les pages vues du site.

## Trafic du dépôt GitHub — VisitorBadge et FlagCounter

Les compteurs ci-dessous suivent les requêtes des images du README de ce dépôt, séparément des statistiques Cloudflare du site officiel. Les identifiants existants sont conservés et partagés entre toutes les langues du README.

**Vues du README (VisitorBadge):**

![Vues du README (VisitorBadge)](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

**Pays/régions des visiteurs du README (FlagCounter):**

[![Pays/régions des visiteurs du README (FlagCounter)](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

Ces compteurs d’images sont approximatifs : le proxy et le cache d’images de GitHub peuvent affecter les chiffres et les régions apparentes. Ce ne sont pas les métriques officielles GitHub Traffic. Les chiffres du site et du README sont présentés séparément et ne sont pas additionnés.
