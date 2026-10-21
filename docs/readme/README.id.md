# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Lisensi: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Bahasa: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer adalah contoh ringkas model bahasa kausal yang dapat dilatih. Modul sequence-mixing menggunakan Kimi Delta Attention (KDA): pelatihan CUDA memakai chunked Triton kernel dari FLA, dan tersedia juga implementasi referensi PyTorch token-by-token yang dapat didiferensiasi.

## Persyaratan

- Python 3.10+
- NVIDIA GPU dan PyTorch dengan dukungan CUDA
- Disarankan GPU dengan BF16; GPU lama dapat memakai FP16

    pip install -e '.[cuda]'

## Mulai Cepat

    python -m kda_train.train --device cuda --steps 1000

Dengan corpus UTF-8 milikmu:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Backend

- `auto`: memakai `fla.ops.kda.chunk_kda` di CUDA bila `fla-core` terpasang; jika tidak, fallback ke referensi PyTorch yang dapat didiferensiasi.
- `fla`: memaksa kernel CUDA FLA.
- `reference`: memaksa scan PyTorch token-by-token. Mode ini menyimpan graph penuh dan tidak cocok untuk urutan panjang.

Untuk jalur CUDA, disarankan `fla-core >= 0.5.2`.

## Ringkasan algoritme

Setiap head menyimpan matriks state K×V. Untuk tiap token, forget gate per kanal key meluruhkan state, lalu error prediksi dihitung dan ditulis kembali ke value dengan aturan delta. Vektor query membaca dari state yang sudah diperbarui. Tiap layer juga mencakup proyeksi input, beta dan gate per kanal, output gate, serta MLP.

## Referensi

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Repositori resmi Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Implementasi KDA Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)

## Pengembangan dan kontribusi

    pip install -e '.[test]'
    pytest -q

Pengujian memaksa backend referensi PyTorch, sehingga dapat dijalankan di CPU. Untuk jalur pelatihan kernel CUDA, gunakan `.[cuda]`. Lihat [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Keamanan

Jangan ungkapkan kerentanan potensial di issue publik. Panduan pelaporan ada di [SECURITY.md](../../SECURITY.md).

## Statistik Kunjungan

- Total kunjungan:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- Distribusi negara/wilayah (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- Peta pengunjung (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
