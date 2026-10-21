# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licentie: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Talen: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer is een compact, vanaf nul opgebouwd voorbeeld van een trainbaar causaal taalmodel. De module voor het mengen van sequenties gebruikt Kimi Delta Attention (KDA): CUDA-training gebruikt de gechunkte Triton-kernel van FLA. Voor leesbaarheid en CPU-experimenten is ook een differentieerbare PyTorch-referentie opgenomen die token voor token werkt.

Deze repository implementeert een leermodel met uitsluitend KDA, niet het volledige Kimi Linear. UTF-8-bytes dienen als token om tokenizer-afhankelijkheden te vermijden. Het model bevat voorspelling van de volgende byte, AdamW, een warmup/cosinus-LR-schema, gradiëntaccumulatie, gemengde precisie, validatieverlies, checkpoints en sampling.

## Projectwebsite

Officiële projectpagina: [kda-cuda-trainer.pages.dev](https://kda-cuda-trainer.pages.dev/)

De website is de voornaamste openbare ingang voor demo’s en bezoekersgeografie op basis van statistieken; GitHub blijft de gezaghebbende bron voor code en releases.

## Terminologie

Voor consistentie tussen alle taalversies blijven de belangrijkste technische termen in het Engels: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate` en `delta rule`.

## Vereisten

- Python 3.10+
- NVIDIA GPU en CUDA-geschikte PyTorch
- BF16-geschikte GPU aanbevolen; oudere GPU's kunnen FP16 gebruiken

Installeer eerst de bij CUDA passende PyTorch-versie, daarna het project en de FLA-kernel:

    pip install -e '.[cuda]'

## Snelstart

Train op het meegeleverde kleine tekstcorpus:

    python -m kda_train.train --device cuda --steps 1000

Gebruik je eigen UTF-8-tekstcorpus:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

Genereer tekst:

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Backends

- `auto`: gebruikt op CUDA `fla.ops.kda.chunk_kda` als `fla-core` is geïnstalleerd; anders valt het terug op de differentieerbare PyTorch-referentie.
- `fla`: forceert de FLA CUDA-kernel.
- `reference`: forceert token-by-token PyTorch-scan. Dit bewaart de volledige grafiek en is niet geschikt voor lange sequenties.

Gebruik `fla-core >= 0.5.2` om KDA beta sigmoid-ondersteuning in het CUDA-pad te garanderen.

## Algoritme-overzicht

Elke head houdt een K×V-toestandsmatrix bij. Voor elk token zwakt een forget gate per key eerst de toestand af; daarna wordt de key-voorspellingsfout berekend en via delta rule teruggeschreven naar value. query-vectoren lezen de bijgewerkte toestand. Elke layer bevat ook invoerprojectie, beta en een gate per kanaal, uitvoermodulatie en MLP.

Deze implementatie geeft voorrang aan een leesbaar trainingsproces. Volledige reproductie van Kimi Linear vereist extra onderdelen, zoals hybride MLA-layers, korte convolutie, een grotere schaal en het volledige trainingsrecept.

## Referenties

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Officiële Kimi-Linear-repository](https://github.com/MoonshotAI/Kimi-Linear)
- [Flash Linear Attention (FLA) KDA-implementatie](https://github.com/fla-org/flash-linear-attention)

## Ontwikkeling en bijdragen

Installeer de testafhankelijkheden:

    pip install -e '.[test]'
    pytest -q

Tests forceren het PyTorch-referentie-backend en kunnen daarom op de CPU draaien. Gebruik voor training met de CUDA-kernel de extra `.[cuda]`. Lees [CONTRIBUTING.md](../../CONTRIBUTING.md) voordat je een PR opent.

## Beveiliging

Publiceer mogelijke kwetsbaarheden niet in openbare issues. Zie [SECURITY.md](../../SECURITY.md).

## Websitestatistieken configureren (Cloudflare Free)

Het bestaande Cloudflare Pages-project levert de website en een alleen-lezen Pages Function op `/stats.svg`. De functie vraagt Cloudflare Web Analytics op, filtert op de hostname en site tag van het project en bewaart de statistiekkaart vijf minuten in de cache. Het laden van de kaart verhoogt geen bezoekenteller. Deze websitecijfers staan los van de VisitorBadge- en FlagCounter-afbeeldingen in het onderstaande gedeelte over GitHub-verkeer.

Zie de [configuratie voor deployment en statistieken](../website-analytics.md) voor de drie vereiste Pages-bindings. Bij ontbrekende configuratie of een fout van de bron-API meldt de kaart expliciet dat statistieken niet beschikbaar zijn in plaats van fictieve nullen te tonen.

## Bezoeken aan de officiële website — Cloudflare

[![Bezoeken aan de officiële website — Cloudflare](https://kda-cuda-trainer.pages.dev/stats.svg)](https://kda-cuda-trainer.pages.dev/#traffic-heading)

Cloudflare Web Analytics-statistieken van de projectwebsite over de afgelopen 30 dagen. De website en deze README tonen dezelfde kaart, die elke vijf minuten wordt bijgewerkt; GitHubs afbeeldingscache kan updates vertragen. Visits zijn geen unieke bezoekers (UV). Weergaven van de GitHub-repository zijn uitgesloten. Percentages per land/regio gebruiken alle paginaweergaven van de website.

## GitHub-repositoryverkeer — VisitorBadge en FlagCounter

De onderstaande tellers registreren afbeeldingsverzoeken voor de README van deze repository, los van de Cloudflare-statistieken van de officiële website. De bestaande teller-ID’s blijven behouden en worden gedeeld door alle README-talen.

**README-weergaven (VisitorBadge):**

![README-weergaven (VisitorBadge)](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

**Landen/regio’s van README-bezoekers (FlagCounter):**

[![Landen/regio’s van README-bezoekers (FlagCounter)](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

Dit zijn benaderende tellers op basis van afbeeldingen: GitHubs afbeeldingsproxy en cache kunnen de aantallen en weergegeven regio’s beïnvloeden. Dit zijn niet de officiële GitHub Traffic-statistieken. Website- en README-aantallen worden afzonderlijk weergegeven en niet opgeteld.
