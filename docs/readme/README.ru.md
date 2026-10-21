# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Лицензия: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer — компактный обучаемый пример причинно-следственной языковой модели. Модуль смешивания последовательностей использует Kimi Delta Attention (KDA): в CUDA используется chunked Triton kernel из FLA, также включена дифференцируемая эталонная реализация на PyTorch по токенам.

## Требования

- Python 3.10+
- NVIDIA GPU и PyTorch с поддержкой CUDA
- Рекомендуется GPU с BF16; на старых GPU можно использовать FP16

    pip install -e '.[cuda]'

## Быстрый старт

    python -m kda_train.train --device cuda --steps 1000

Собственный UTF-8 корпус:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Бэкенды

- `auto`: использует `fla.ops.kda.chunk_kda` на CUDA при установленном `fla-core`, иначе переключается на дифференцируемый PyTorch reference.
- `fla`: принудительно использует CUDA kernel из FLA.
- `reference`: принудительный token-by-token scan в PyTorch. Сохраняет полный граф и не подходит для длинных последовательностей.

Для CUDA пути рекомендуется `fla-core >= 0.5.2`.

## Кратко об алгоритме

Каждая голова хранит матрицу состояния K×V. Для каждого токена forget gate по каналам key затухает состояние, затем вычисляется ошибка предсказания и записывается обратно в value по delta rule. Векторы query читают из обновлённого состояния. В каждом слое также есть входная проекция, beta и поканальный gate, выходной gate и MLP.

## Ссылки

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Официальный репозиторий Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Реализация KDA в Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)

## Разработка и вклад

    pip install -e '.[test]'
    pytest -q

Тесты принудительно используют reference-бэкенд PyTorch, поэтому запускаются на CPU. Для CUDA-пути используйте `.[cuda]`. См. [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Безопасность

Пожалуйста, не публикуйте потенциальные уязвимости в публичных issues. Инструкции: [SECURITY.md](../../SECURITY.md).

## Статистика посещений

- Общее число посещений:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Распределение по странам/регионам (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Карта посетителей (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
