# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Лицензия: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer — компактный обучаемый пример авторегрессионной языковой модели, созданный с нуля. Модуль смешивания последовательностей использует Kimi Delta Attention (KDA): обучение на CUDA выполняется блочным Triton kernel из FLA, а для удобства чтения и экспериментов на CPU включена дифференцируемая потактовая реализация на PyTorch, обрабатывающая token за token.

Репозиторий реализует учебную модель только с KDA, а не полный Kimi Linear. В качестве token используются байты UTF-8, что исключает зависимость от tokenizer. Включены предсказание следующего байта, AdamW, расписание LR с warmup и косинусным спадом, накопление градиентов, смешанная точность, валидационная функция потерь, checkpoints и сэмплирование.

## Сайт проекта

Официальная главная страница проекта: [kda-cuda-trainer.pages.dev](https://kda-cuda-trainer.pages.dev/)

Сайт — основная публичная страница с демонстрациями и географией посетителей по данным аналитики; GitHub остаётся основным источником кода и релизов.

## Терминология

Для единообразия между всеми языковыми версиями ключевые технические термины сохраняются на английском: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate`, `delta rule`.

## Требования

- Python 3.10+
- NVIDIA GPU и PyTorch с поддержкой CUDA
- Рекомендуется GPU с BF16; на старых GPU можно использовать FP16

Сначала установите PyTorch, соответствующий вашей версии CUDA, затем проект и kernel FLA:

    pip install -e '.[cuda]'

## Быстрый старт

Обучение на небольшом текстовом корпусе из репозитория:

    python -m kda_train.train --device cuda --steps 1000

Использование собственного текстового корпуса UTF-8:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

Генерация текста:

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Бэкенды

- `auto`: использует `fla.ops.kda.chunk_kda` на CUDA при установленном `fla-core`, иначе переключается на дифференцируемый PyTorch reference.
- `fla`: принудительно использует CUDA kernel из FLA.
- `reference`: принудительный token-by-token scan в PyTorch. Сохраняет полный граф и не подходит для длинных последовательностей.

Используйте `fla-core >= 0.5.2`, чтобы обеспечить поддержку KDA beta sigmoid в CUDA-пути.

## Кратко об алгоритме

Каждый head хранит матрицу состояния K×V. Для каждого token забывающий gate для каждого key сначала ослабляет состояние; затем вычисляется ошибка предсказания key и записывается в value по delta rule. Векторы query читают обновлённое состояние. Каждый layer также содержит входную проекцию, beta и поканальный gate, выходной gate и MLP.

Эта реализация ставит на первое место понятный процесс обучения. Полное воспроизведение Kimi Linear требует дополнительных компонентов: гибридных MLA layers, короткой свёртки, большего масштаба и полного рецепта обучения.

## Ссылки

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Официальный репозиторий Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Реализация KDA в Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)

## Разработка и вклад

Установите зависимости для тестов:

    pip install -e '.[test]'
    pytest -q

Тесты принудительно используют эталонный backend PyTorch, поэтому работают на CPU. Для обучения с CUDA kernel установите extra `.[cuda]`. Перед открытием PR прочитайте [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Безопасность

Пожалуйста, не публикуйте потенциальные уязвимости в публичных issues. Инструкции: [SECURITY.md](../../SECURITY.md).

## Настройка аналитики сайта (Cloudflare Free)

Существующий проект Cloudflare Pages обслуживает сайт и Pages Function только для чтения по адресу `/stats.svg`. Функция запрашивает Cloudflare Web Analytics, фильтрует данные по hostname и site tag проекта и кэширует карточку на пять минут. Загрузка карточки не увеличивает счётчик посещений. Данные сайта независимы от изображений VisitorBadge и FlagCounter в разделе статистики репозитория GitHub ниже.

Три обязательных Pages bindings описаны в [настройке развёртывания и аналитики](../website-analytics.md). При отсутствии конфигурации или сбое API источника карточка явно сообщает о недоступности статистики вместо отображения вымышленных нулей.

## Посещения официального сайта — Cloudflare

[![Посещения официального сайта — Cloudflare](https://kda-cuda-trainer.pages.dev/stats.svg)](https://kda-cuda-trainer.pages.dev/#traffic-heading)

Статистика Cloudflare Web Analytics сайта проекта за последние 30 дней. Сайт и этот README показывают одну карточку, обновляемую каждые пять минут; кэш изображений GitHub может задерживать обновления. Visits — не уникальные посетители (UV). Просмотры репозитория GitHub исключены. Доли стран и регионов рассчитаны от всех просмотров страниц сайта.

## Посещения репозитория GitHub — VisitorBadge и FlagCounter

Счётчики ниже учитывают запросы изображений README этого репозитория отдельно от Cloudflare-аналитики официального сайта. Существующие идентификаторы сохранены и используются во всех языковых версиях README.

**Просмотры README (VisitorBadge):**

![Просмотры README (VisitorBadge)](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

**Страны и регионы посетителей README (FlagCounter):**

[![Страны и регионы посетителей README (FlagCounter)](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

Это приблизительные счётчики на основе изображений: прокси и кэш изображений GitHub могут влиять на значения и отображаемые регионы. Это не официальные метрики GitHub Traffic. Данные сайта и README показываются раздельно и не суммируются.
