# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer는 처음부터 구축한 소형 학습 가능한 인과 언어 모델 예제입니다. 시퀀스 혼합은 Kimi Delta Attention(KDA)을 사용하며, CUDA 학습은 FLA의 chunked Triton kernel을 사용합니다. 가독성과 CPU 실험을 위해 PyTorch 토큰 단위 참조 구현도 포함되어 있습니다.

## 요구 사항

- Python 3.10+
- NVIDIA GPU 및 CUDA 지원 PyTorch
- BF16 지원 GPU 권장(구형 GPU는 FP16 사용 가능)

    pip install -e '.[cuda]'

## 빠른 시작

    python -m kda_train.train --device cuda --steps 1000
    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## 개발

    pip install -e '.[test]'
    pytest -q

[CONTRIBUTING.md](../../CONTRIBUTING.md), [SECURITY.md](../../SECURITY.md)를 참고하세요.

## 방문 통계

- 총 방문 수:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- 국가/지역 분포 (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- 방문 지도 (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
