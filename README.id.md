# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Lisensi: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Bahasa: [English](README.md) | [中文](README.zh-CN.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Italiano](README.it.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [العربية](README.ar.md) | [Nederlands](README.nl.md) | [हिन्दी](README.hi.md) | [Bahasa Indonesia](README.id.md)

KDA CUDA Trainer adalah contoh ringkas model bahasa kausal yang dapat dilatih dari nol. Modul sequence-mixing menggunakan Kimi Delta Attention (KDA): pelatihan CUDA memakai chunked Triton kernel dari FLA, dan tersedia juga implementasi referensi PyTorch token-by-token untuk keterbacaan dan eksperimen CPU.

## Persyaratan

- Python 3.10+
- NVIDIA GPU dan PyTorch dengan dukungan CUDA
- Disarankan GPU dengan BF16; GPU lama dapat memakai FP16

    pip install -e '.[cuda]'

## Mulai Cepat

    python -m kda_train.train --device cuda --steps 1000
    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Pengembangan

    pip install -e '.[test]'
    pytest -q

Silakan baca [CONTRIBUTING.md](CONTRIBUTING.md) dan [SECURITY.md](SECURITY.md).

## Statistik Kunjungan

- Total kunjungan:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Distribusi negara/wilayah (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Peta pengunjung (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
