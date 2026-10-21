# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licencia: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer es un ejemplo compacto de modelo de lenguaje causal entrenable construido desde cero. Su módulo de mezcla de secuencias utiliza Kimi Delta Attention (KDA): el entrenamiento CUDA usa el kernel Triton por bloques de FLA, y se incluye una implementación de referencia PyTorch diferenciable token a token para facilitar la lectura y los experimentos en CPU.

Este repositorio implementa un modelo didáctico basado solo en KDA, no Kimi Linear completo. Utiliza bytes UTF-8 como token para evitar dependencias de tokenizer e incluye modelado del siguiente byte, AdamW, planificación de LR con warmup/coseno, acumulación de gradientes, precisión mixta, pérdida de validación, checkpoints y muestreo.

## Sitio web del proyecto

Página oficial del proyecto: [kda-cuda-trainer.pages.dev](https://kda-cuda-trainer.pages.dev/)

El sitio web es el punto de entrada público principal para las demostraciones y la distribución geográfica de visitantes basada en estadísticas; GitHub es la fuente canónica del código y las versiones.

## Terminología

Para mantener la coherencia entre todas las versiones, los términos técnicos clave se conservan en inglés: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate` y `delta rule`.

## Requisitos

- Python 3.10+
- GPU NVIDIA y PyTorch con CUDA
- Se recomienda GPU con BF16; en GPUs antiguas se puede usar FP16

Instala primero PyTorch compatible con tu CUDA y después el proyecto y el kernel de FLA:

    pip install -e '.[cuda]'

## Inicio rápido

Entrena con el pequeño corpus de texto incluido:

    python -m kda_train.train --device cuda --steps 1000

Usa tu propio corpus de texto UTF-8:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

Genera texto:

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Backends

- `auto`: usa `fla.ops.kda.chunk_kda` en CUDA cuando `fla-core` está instalado; en caso contrario, vuelve a la referencia diferenciable de PyTorch.
- `fla`: fuerza el kernel CUDA de FLA.
- `reference`: fuerza el escaneo token a token en PyTorch. Mantiene el grafo completo y no es adecuado para secuencias largas.

Usa `fla-core >= 0.5.2` para garantizar el soporte de KDA beta sigmoid en la ruta CUDA.

## Resumen del algoritmo

Cada head mantiene una matriz de estado K×V. Para cada token, un gate de olvido por key atenúa el estado; después se calcula el error de predicción de key y se escribe en value mediante delta rule. Los vectores query leen del estado actualizado. Cada layer incluye también proyección de entrada, beta y un gate por canal, modulación de salida y MLP.

Esta implementación prioriza un flujo de entrenamiento legible. Reproducir Kimi Linear completo requiere componentes adicionales, como layers híbridos MLA, convolución corta, mayor escala y la receta completa de entrenamiento.

## Referencias

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Repositorio oficial de Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Implementación KDA de Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)

## Desarrollo y contribución

Instala las dependencias de pruebas:

    pip install -e '.[test]'
    pytest -q

Las pruebas fuerzan el backend de referencia de PyTorch y pueden ejecutarse en CPU. Para entrenar con el kernel CUDA, usa el extra `.[cuda]`. Lee [CONTRIBUTING.md](../../CONTRIBUTING.md) antes de abrir un PR.

## Seguridad

No publiques vulnerabilidades potenciales en issues públicos. Consulta [SECURITY.md](../../SECURITY.md).

## Configuración de estadísticas web (Cloudflare Free)

El proyecto existente de Cloudflare Pages sirve el sitio web y una Pages Function de solo lectura en `/stats.svg`. La función consulta Cloudflare Web Analytics, filtra por hostname y site tag del proyecto y almacena la tarjeta en caché durante cinco minutos. Cargar la tarjeta no incrementa un contador de visitas. Estos datos del sitio son independientes de las imágenes VisitorBadge y FlagCounter de la sección de tráfico de GitHub situada más abajo.

Consulta [despliegue y configuración de estadísticas](../website-analytics.md) para los tres bindings de Pages necesarios. Si faltan o falla la API de origen, la tarjeta indica que las estadísticas no están disponibles en lugar de mostrar ceros ficticios.

## Tráfico del sitio oficial — Cloudflare

[![Tráfico del sitio oficial — Cloudflare](https://kda-cuda-trainer.pages.dev/stats.svg)](https://kda-cuda-trainer.pages.dev/#traffic-heading)

Estadísticas de Cloudflare Web Analytics del sitio del proyecto durante los últimos 30 días. El sitio y este README comparten la misma tarjeta, actualizada cada cinco minutos (la caché de imágenes de GitHub puede retrasarla). Visits no equivale a visitantes únicos (UV). Se excluyen las vistas del repositorio GitHub. Los porcentajes por país/región se calculan sobre todas las páginas vistas del sitio.

## Tráfico del repositorio GitHub — VisitorBadge y FlagCounter

Los contadores siguientes registran las solicitudes de imágenes del README de este repositorio, por separado de las estadísticas Cloudflare del sitio oficial. Se conservan los identificadores originales, compartidos entre los idiomas del README.

**Vistas del README (VisitorBadge):**

![Vistas del README (VisitorBadge)](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

**Países/regiones de los visitantes del README (FlagCounter):**

[![Países/regiones de los visitantes del README (FlagCounter)](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

Son contadores aproximados basados en imágenes: el proxy y la caché de imágenes de GitHub pueden afectar tanto a las cifras como a las regiones aparentes. No son las métricas oficiales de GitHub Traffic. Los recuentos del sitio y del README se muestran por separado y no se suman.
