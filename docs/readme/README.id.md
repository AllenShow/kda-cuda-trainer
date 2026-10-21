# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Lisensi: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Bahasa: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer adalah contoh ringkas model bahasa kausal yang dapat dilatih dan dibangun dari awal. Modul pencampuran urutannya memakai Kimi Delta Attention (KDA): pelatihan CUDA menggunakan kernel Triton berbasis chunk dari FLA, dan implementasi referensi PyTorch yang dapat didiferensiasi token demi token disertakan untuk memudahkan pembacaan dan eksperimen CPU.

Repositori ini mengimplementasikan model pengajaran KDA murni, bukan Kimi Linear lengkap. Byte UTF-8 digunakan sebagai token untuk menghindari ketergantungan tokenizer. Fitur yang tersedia meliputi pemodelan byte berikutnya, AdamW, penjadwalan LR warmup/cosine, akumulasi gradien, presisi campuran, loss validasi, checkpoint, dan sampling.

## Situs proyek

Halaman resmi proyek: [kda-cuda-trainer.pages.dev](https://kda-cuda-trainer.pages.dev/)

Situs ini merupakan halaman publik utama untuk demo dan sebaran geografis pengunjung berbasis analitik, sedangkan GitHub tetap menjadi sumber utama kode dan rilis.

## Terminologi

Agar konsisten di semua versi bahasa, istilah teknis inti tetap menggunakan bahasa Inggris: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate`, `delta rule`.

## Persyaratan

- Python 3.10+
- NVIDIA GPU dan PyTorch dengan dukungan CUDA
- Disarankan GPU dengan BF16; GPU lama dapat memakai FP16

Pasang terlebih dahulu PyTorch yang sesuai dengan CUDA, lalu proyek dan kernel FLA:

    pip install -e '.[cuda]'

## Mulai Cepat

Latih menggunakan korpus teks kecil yang disertakan:

    python -m kda_train.train --device cuda --steps 1000

Gunakan korpus teks UTF-8 sendiri:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

Hasilkan teks:

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Backend

- `auto`: memakai `fla.ops.kda.chunk_kda` di CUDA bila `fla-core` terpasang; jika tidak, fallback ke referensi PyTorch yang dapat didiferensiasi.
- `fla`: memaksa kernel CUDA FLA.
- `reference`: memaksa scan PyTorch token-by-token. Mode ini menyimpan graph penuh dan tidak cocok untuk urutan panjang.

Gunakan `fla-core >= 0.5.2` untuk memastikan dukungan KDA beta sigmoid pada jalur CUDA.

## Ringkasan algoritme

Setiap head menyimpan matriks state K×V. Untuk setiap token, forget gate per key terlebih dahulu meluruhkan state; kemudian galat prediksi key dihitung dan ditulis kembali ke value melalui delta rule. Vektor query membaca state yang diperbarui. Setiap layer juga mencakup proyeksi input, beta dan gate per kanal, output gating, serta MLP.

Implementasi ini mengutamakan alur pelatihan yang mudah dibaca. Reproduksi Kimi Linear lengkap memerlukan komponen tambahan seperti layer MLA hibrida, konvolusi pendek, skala lebih besar, dan resep pelatihan lengkap.

## Referensi

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Repositori resmi Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Implementasi KDA Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)

## Pengembangan dan kontribusi

Pasang dependensi pengujian:

    pip install -e '.[test]'
    pytest -q

Pengujian memaksa backend referensi PyTorch sehingga dapat dijalankan pada CPU. Untuk pelatihan dengan kernel CUDA, gunakan extra `.[cuda]`. Baca [CONTRIBUTING.md](../../CONTRIBUTING.md) sebelum membuka PR.

## Keamanan

Jangan ungkapkan kerentanan potensial di issue publik. Panduan pelaporan ada di [SECURITY.md](../../SECURITY.md).

## Konfigurasi analitik situs (Cloudflare Free)

Proyek Cloudflare Pages yang ada menyajikan situs dan Pages Function hanya-baca di `/stats.svg`. Fungsi ini meminta data Cloudflare Web Analytics, memfilter hostname dan site tag proyek, serta menyimpan kartu dalam cache selama lima menit. Memuat kartu tidak menambah penghitung kunjungan. Angka situs ini terpisah dari gambar VisitorBadge dan FlagCounter pada bagian lalu lintas repositori GitHub di bawah.

Lihat [konfigurasi deployment dan analitik](../website-analytics.md) untuk tiga binding Pages yang diperlukan. Jika konfigurasi belum lengkap atau API sumber gagal, kartu menyatakan statistik tidak tersedia alih-alih menampilkan angka nol palsu.

## Lalu lintas situs resmi — Cloudflare

[![Lalu lintas situs resmi — Cloudflare](https://kda-cuda-trainer.pages.dev/stats.svg)](https://kda-cuda-trainer.pages.dev/#traffic-heading)

Statistik Cloudflare Web Analytics untuk situs proyek selama 30 hari terakhir. Situs dan README ini berbagi kartu yang sama, diperbarui setiap lima menit; cache gambar GitHub dapat menunda pembaruan. Visits bukan pengunjung unik (UV). Tayangan repositori GitHub dikecualikan. Persentase negara/wilayah memakai seluruh tayangan halaman situs.

## Lalu lintas repositori GitHub — VisitorBadge dan FlagCounter

Penghitung di bawah mencatat permintaan gambar README repositori ini, terpisah dari statistik Cloudflare situs resmi. Identitas penghitung asli dipertahankan dan digunakan bersama oleh semua versi bahasa README.

**Tayangan README (VisitorBadge):**

![Tayangan README (VisitorBadge)](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

**Negara/wilayah pengunjung README (FlagCounter):**

[![Negara/wilayah pengunjung README (FlagCounter)](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

Ini adalah penghitung perkiraan berbasis gambar: proxy dan cache gambar GitHub dapat memengaruhi jumlah maupun wilayah yang terlihat. Angka ini bukan metrik resmi GitHub Traffic. Jumlah situs dan README ditampilkan terpisah dan tidak dijumlahkan.
