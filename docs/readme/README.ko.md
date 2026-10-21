# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer는 처음부터 구축한 소형 학습 가능한 인과 언어 모델 예제입니다. 시퀀스 혼합은 Kimi Delta Attention(KDA)을 사용하며, CUDA 학습은 FLA의 분할 Triton 커널을 사용합니다. 가독성과 CPU 실험을 위해 PyTorch 토큰 단위 참조 구현도 포함되어 있습니다.

## 용어

다국어 문서 간 일관성을 위해 핵심 기술 용어는 영어로 유지합니다: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate`, `delta rule`.

## 요구 사항

- Python 3.10+
- NVIDIA GPU 및 CUDA 지원 PyTorch
- BF16 지원 GPU 권장(구형 GPU는 FP16 사용 가능)

    pip install -e '.[cuda]'

## 빠른 시작

    python -m kda_train.train --device cuda --steps 1000

사용자 UTF-8 말뭉치 사용:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## 백엔드

- `auto`: CUDA 환경에서 `fla-core`가 설치되어 있으면 `fla.ops.kda.chunk_kda`를 사용하고, 아니면 미분 가능한 PyTorch 참조 구현으로 폴백합니다.
- `fla`: FLA CUDA 커널을 강제합니다.
- `reference`: 토큰 단위 PyTorch 스캔을 강제합니다. 전체 그래프를 보존하므로 긴 시퀀스 학습에는 부적합합니다.

CUDA 경로에서는 `fla-core >= 0.5.2`를 권장합니다.

## 알고리즘 요약

각 head는 K×V 상태 행렬을 유지합니다. 각 토큰에서 key 채널별 forget gate가 상태를 감쇠시키고, 예측 오차를 계산해 delta rule로 value에 다시 기록합니다. query 벡터는 업데이트된 상태에서 출력을 읽습니다. 각 레이어에는 입력 투영, beta 및 채널별 gate, 출력 게이트, MLP가 포함됩니다.

## 참고자료

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Kimi-Linear 공식 저장소](https://github.com/MoonshotAI/Kimi-Linear)
- [Flash Linear Attention(FLA) KDA 구현](https://github.com/fla-org/flash-linear-attention)

## 개발 및 기여

    pip install -e '.[test]'
    pytest -q

테스트는 PyTorch reference 백엔드를 강제하므로 CPU에서 실행할 수 있습니다. CUDA 커널 학습 경로는 `.[cuda]`를 사용하세요. [CONTRIBUTING.md](../../CONTRIBUTING.md)를 참고하세요.

## 보안

잠재적 취약점은 공개 이슈에 올리지 마세요. 보고 방법은 [SECURITY.md](../../SECURITY.md)를 참고하세요.

## 방문 통계

- 총 방문 수:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- 국가/지역 분포(FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)
