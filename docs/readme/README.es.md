# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licencia: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer es un ejemplo compacto de modelo de lenguaje causal entrenable. El módulo de mezcla de secuencia usa Kimi Delta Attention (KDA): el entrenamiento en CUDA utiliza el chunked Triton kernel de FLA, y también incluye una implementación de referencia diferenciable en PyTorch token por token.

## Terminología

Para mantener la coherencia entre todas las versiones, los términos técnicos clave se conservan en inglés: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate` y `delta rule`.

## Requisitos

- Python 3.10+
- GPU NVIDIA y PyTorch con CUDA
- Se recomienda GPU con BF16; en GPUs antiguas se puede usar FP16

    pip install -e '.[cuda]'

## Inicio rápido

    python -m kda_train.train --device cuda --steps 1000

Con tu propio corpus UTF-8:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Backends

- `auto`: usa `fla.ops.kda.chunk_kda` en CUDA cuando `fla-core` está instalado; en caso contrario, vuelve a la referencia diferenciable de PyTorch.
- `fla`: fuerza el kernel CUDA de FLA.
- `reference`: fuerza el escaneo token a token en PyTorch. Mantiene el grafo completo y no es adecuado para secuencias largas.

Para CUDA se recomienda `fla-core >= 0.5.2`.

## Resumen del algoritmo

Cada cabeza mantiene una matriz de estado K×V. Para cada token, una compuerta de olvido por canal de key atenúa el estado; después se calcula el error de predicción y se escribe en value mediante la regla delta. Los vectores query leen del estado actualizado. Cada capa también incluye proyección de entrada, beta y compuerta por canal, compuerta de salida y MLP.

## Referencias

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Repositorio oficial de Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Implementación KDA de Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)

## Desarrollo y contribución

    pip install -e '.[test]'
    pytest -q

Las pruebas fuerzan el backend de referencia de PyTorch y pueden ejecutarse en CPU. Para el camino CUDA usa `.[cuda]`. Revisa [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Seguridad

No publiques vulnerabilidades potenciales en issues públicos. Consulta [SECURITY.md](../../SECURITY.md).

## Estadísticas de visitas

- Visitas totales:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Distribución por país/región (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Mapa de visitantes (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
