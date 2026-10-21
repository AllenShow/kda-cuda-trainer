# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licenza: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer è un esempio compatto di modello linguistico causale addestrabile costruito da zero. Il modulo di mescolamento delle sequenze usa Kimi Delta Attention (KDA): l’addestramento CUDA utilizza il kernel Triton a blocchi di FLA, e un’implementazione di riferimento PyTorch differenziabile token per token facilita la lettura e gli esperimenti su CPU.

Questo repository implementa un modello didattico basato solo su KDA, non Kimi Linear completo. Utilizza byte UTF-8 come token per evitare dipendenze da tokenizer e include modellazione del byte successivo, AdamW, pianificazione del LR warmup/coseno, accumulo dei gradienti, precisione mista, perdita di validazione, checkpoint e campionamento.

## Sito del progetto

Pagina ufficiale del progetto: [kda-cuda-trainer.pages.dev](https://kda-cuda-trainer.pages.dev/)

Il sito è il principale punto di accesso pubblico per demo e distribuzione geografica dei visitatori basata sulle statistiche; GitHub rimane la fonte canonica per codice e release.

## Terminologia

Per mantenere coerenza tra tutte le versioni linguistiche, i principali termini tecnici restano in inglese: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate` e `delta rule`.

## Requisiti

- Python 3.10+
- GPU NVIDIA e PyTorch con CUDA
- Consigliata GPU con BF16; su GPU più vecchie è possibile usare FP16

Installa prima PyTorch compatibile con CUDA, poi il progetto e il kernel FLA:

    pip install -e '.[cuda]'

## Avvio rapido

Addestra sul piccolo corpus di testo incluso:

    python -m kda_train.train --device cuda --steps 1000

Usa un tuo corpus di testo UTF-8:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

Genera testo:

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Backend

- `auto`: usa `fla.ops.kda.chunk_kda` su CUDA quando `fla-core` è installato; altrimenti usa la reference PyTorch differenziabile.
- `fla`: forza il kernel CUDA FLA.
- `reference`: forza lo scan token-by-token in PyTorch. Mantiene il grafo completo e non è adatto a sequenze lunghe.

Usa `fla-core >= 0.5.2` per garantire il supporto a KDA beta sigmoid nel percorso CUDA.

## Sintesi dell’algoritmo

Ogni head mantiene una matrice di stato K×V. Per ogni token, un gate di dimenticanza per key attenua lo stato; viene quindi calcolato l’errore di previsione di key e riscritto in value tramite delta rule. I vettori query leggono lo stato aggiornato. Ogni layer include anche proiezione in ingresso, beta e un gate per canale, modulazione in uscita e MLP.

Questa implementazione privilegia un flusso di addestramento leggibile. Riprodurre Kimi Linear completo richiede componenti aggiuntivi, tra cui layer ibridi MLA, convoluzione corta, scala maggiore e la ricetta completa di addestramento.

## Riferimenti

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Repository ufficiale Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Implementazione KDA di Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)

## Sviluppo e contributi

Installa le dipendenze dei test:

    pip install -e '.[test]'
    pytest -q

I test forzano il backend di riferimento PyTorch e possono quindi essere eseguiti su CPU. Per l’addestramento con kernel CUDA usa l’extra `.[cuda]`. Leggi [CONTRIBUTING.md](../../CONTRIBUTING.md) prima di aprire una PR.

## Sicurezza

Non pubblicare potenziali vulnerabilità in issue pubbliche. Consulta [SECURITY.md](../../SECURITY.md).

## Configurazione delle statistiche del sito (Cloudflare Free)

Il progetto Cloudflare Pages esistente serve il sito e una Pages Function di sola lettura su `/stats.svg`. La funzione interroga Cloudflare Web Analytics, filtra hostname e site tag del progetto e memorizza la scheda nella cache per cinque minuti. Caricare la scheda non incrementa un contatore di visite. Questi dati del sito sono indipendenti dalle immagini VisitorBadge e FlagCounter nella sezione sul traffico GitHub sottostante.

Vedi la [configurazione di deployment e statistiche](../website-analytics.md) per i tre binding Pages richiesti. Se mancano o l’API di origine non risponde correttamente, la scheda segnala che le statistiche non sono disponibili, senza mostrare zeri fittizi.

## Traffico del sito ufficiale — Cloudflare

[![Traffico del sito ufficiale — Cloudflare](https://kda-cuda-trainer.pages.dev/stats.svg)](https://kda-cuda-trainer.pages.dev/#traffic-heading)

Statistiche Cloudflare Web Analytics del sito del progetto negli ultimi 30 giorni. Il sito e questo README condividono la stessa scheda, aggiornata ogni cinque minuti; la cache delle immagini GitHub può ritardare gli aggiornamenti. Visits non indica i visitatori unici (UV). Le visualizzazioni del repository GitHub sono escluse. Le percentuali per paese/regione si basano su tutte le visualizzazioni delle pagine del sito.

## Traffico del repository GitHub — VisitorBadge e FlagCounter

I contatori seguenti registrano le richieste delle immagini del README di questo repository, separatamente dalle statistiche Cloudflare del sito ufficiale. Gli identificatori esistenti sono conservati e condivisi tra tutte le lingue del README.

**Visualizzazioni del README (VisitorBadge):**

![Visualizzazioni del README (VisitorBadge)](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

**Paesi/regioni dei visitatori del README (FlagCounter):**

[![Paesi/regioni dei visitatori del README (FlagCounter)](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

Sono contatori approssimativi basati su immagini: il proxy e la cache delle immagini di GitHub possono influenzare sia i numeri sia le regioni apparenti. Non sono le metriche ufficiali di GitHub Traffic. I conteggi del sito e del README sono mostrati separatamente e non vengono sommati.
