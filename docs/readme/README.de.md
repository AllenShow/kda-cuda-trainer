# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Lizenz: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Sprachen: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer ist ein kompaktes, von Grund auf entwickeltes Beispiel für ein trainierbares kausales Sprachmodell. Das Modul zum Mischen von Sequenzen nutzt Kimi Delta Attention (KDA): CUDA-Training verwendet den blockweisen Triton-Kernel von FLA. Eine differenzierbare PyTorch-Referenzimplementierung, die token für token arbeitet, dient der Lesbarkeit und CPU-Experimenten.

Dieses Repository implementiert ein reines KDA-Lehrmodell, nicht das vollständige Kimi Linear. Es verwendet UTF-8-Bytes als token, um tokenizer-Abhängigkeiten zu vermeiden. Enthalten sind Next-Byte-Sprachmodellierung, AdamW, Warmup/Kosinus-LR-Zeitplan, Gradientenakkumulation, gemischte Präzision, Validierungsverlust, Checkpoints und Sampling.

## Projektwebsite

Offizielle Projektseite: [kda-cuda-trainer.pages.dev](https://kda-cuda-trainer.pages.dev/)

Die Website ist die zentrale öffentliche Anlaufstelle für Demos und analytikbasierte Besucherregionen. GitHub bleibt die maßgebliche Quelle für Code und Releases.

## Terminologie

Zur Konsistenz zwischen allen Sprachversionen bleiben zentrale Fachbegriffe auf Englisch: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate` und `delta rule`.

## Voraussetzungen

- Python 3.10+
- NVIDIA-GPU und CUDA-fähiges PyTorch
- BF16-fähige GPUs werden empfohlen; ältere GPUs können FP16 nutzen

Installiere zuerst die zu CUDA passende PyTorch-Version, danach das Projekt und den FLA-Kernel:

    pip install -e '.[cuda]'

## Schnellstart

Mit dem enthaltenen kleinen Textkorpus trainieren:

    python -m kda_train.train --device cuda --steps 1000

Eigenen UTF-8-Textkorpus verwenden:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

Text generieren:

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Backends

- `auto`: Verwendet auf CUDA mit installiertem `fla-core` `fla.ops.kda.chunk_kda`, sonst die differenzierbare PyTorch-Referenz.
- `fla`: Erzwingt den FLA-CUDA-Kernel.
- `reference`: Erzwingt tokenweises PyTorch-Scanning. Diese Variante hält den vollständigen Graphen und ist für lange Sequenzen ungeeignet.

Verwende `fla-core >= 0.5.2`, damit KDA beta sigmoid im CUDA-Pfad unterstützt wird.

## Algorithmusüberblick

Jeder head hält eine K×V-Zustandsmatrix. Für jeden token schwächt zunächst ein Forget-gate je key den Zustand ab; anschließend wird der key-Vorhersagefehler berechnet und per delta rule in value zurückgeschrieben. query-Vektoren lesen den aktualisierten Zustand. Jeder layer enthält außerdem Eingabeprojektion, beta und kanalweises gate, Ausgabemodulation und MLP.

Diese Implementierung legt Wert auf einen lesbaren Trainingsablauf. Die vollständige Reproduktion von Kimi Linear erfordert zusätzliche Komponenten wie hybride MLA-layers, kurze Faltung, einen größeren Maßstab und die vollständige Trainingskonfiguration.

## Referenzen

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Kimi-Linear offizielles Repository](https://github.com/MoonshotAI/Kimi-Linear)
- [Flash Linear Attention (FLA) KDA-Implementierung](https://github.com/fla-org/flash-linear-attention)

## Entwicklung und Beitrag

Testabhängigkeiten installieren:

    pip install -e '.[test]'
    pytest -q

Tests erzwingen das PyTorch-Referenz-backend und laufen daher auf der CPU. Für das Training mit dem CUDA-Kernel verwende das Extra `.[cuda]`. Lies vor dem Öffnen eines PR [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Sicherheit

Bitte veröffentliche potenzielle Sicherheitslücken nicht in öffentlichen Issues. Siehe [SECURITY.md](../../SECURITY.md).

## Website-Analytik konfigurieren (Cloudflare Free)

Das bestehende Cloudflare-Pages-Projekt stellt die Website und eine schreibgeschützte Pages Function unter `/stats.svg` bereit. Die Funktion fragt Cloudflare Web Analytics ab, filtert nach Hostname und Site-Tag des Projekts und speichert die Statistikgrafik fünf Minuten im Cache. Das Laden der Grafik erhöht keinen Besuchszähler. Diese Website-Zahlen sind unabhängig von den VisitorBadge- und FlagCounter-Bildern im nachfolgenden Abschnitt zum GitHub-Repository.

Die drei erforderlichen Pages-Bindings sind in der [Deployment- und Analytik-Konfiguration](../website-analytics.md) beschrieben. Bei fehlender Konfiguration oder einem Fehler der vorgeschalteten API meldet die Grafik ausdrücklich, dass Statistiken nicht verfügbar sind, statt erfundene Nullen zu zeigen.

## Besuche der offiziellen Website — Cloudflare

[![Besuche der offiziellen Website — Cloudflare](https://kda-cuda-trainer.pages.dev/stats.svg)](https://kda-cuda-trainer.pages.dev/#traffic-heading)

Cloudflare-Web-Analytics-Statistik der Projektwebsite für die letzten 30 Tage. Website und README zeigen dieselbe Grafik, die alle fünf Minuten aktualisiert wird; GitHubs Bild-Cache kann Aktualisierungen verzögern. Visits sind keine eindeutigen Besucher (UV). Aufrufe des GitHub-Repositorys sind ausgeschlossen. Länder-/Regionsanteile beziehen sich auf sämtliche Seitenaufrufe der Website.

## GitHub-Repository-Besuche — VisitorBadge und FlagCounter

Die folgenden Zähler erfassen Bildanfragen für die README dieses Repositorys getrennt von der Cloudflare-Statistik der offiziellen Website. Die vorhandenen Zählerkennungen bleiben erhalten und werden in allen README-Sprachen gemeinsam verwendet.

**README-Aufrufe (VisitorBadge):**

![README-Aufrufe (VisitorBadge)](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

**Länder/Regionen der README-Besucher (FlagCounter):**

[![Länder/Regionen der README-Besucher (FlagCounter)](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

Dies sind ungefähre bildbasierte Zähler: GitHubs Bild-Proxy und Cache können Zahlen und angezeigte Regionen beeinflussen. Sie sind nicht die offiziellen GitHub-Traffic-Metriken. Website- und README-Zahlen werden getrennt ausgewiesen und nicht addiert.
