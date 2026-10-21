# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer는 처음부터 구축한 소형 학습 가능한 인과 언어 모델 예제입니다. 시퀀스 혼합 모듈은 Kimi Delta Attention(KDA)을 사용합니다. CUDA 학습에는 FLA의 청크 기반 Triton kernel을 사용하며, 가독성과 CPU 실험을 위해 token 단위로 처리하는 미분 가능한 PyTorch 참조 구현도 포함합니다.

이 저장소는 순수 KDA 교육용 모델이며 전체 Kimi Linear 구현은 아닙니다. tokenizer 의존성을 피하려고 UTF-8 바이트를 token으로 사용합니다. 다음 바이트 언어 모델링, AdamW, warmup/cosine 학습률 스케줄, 그래디언트 누적, 혼합 정밀도, 검증 손실, checkpoint, 샘플링을 포함합니다.

## 프로젝트 공식 웹사이트

프로젝트 공식 홈페이지: [kda-cuda-trainer.pages.dev](https://kda-cuda-trainer.pages.dev/)

웹사이트는 데모와 분석 데이터에 기반한 방문자 지역 분포를 공개하는 주된 페이지입니다. 코드와 릴리스의 공식 원본은 GitHub에 있습니다.

## 용어

다국어 문서 간 일관성을 위해 핵심 기술 용어는 영어로 유지합니다: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate`, `delta rule`.

## 요구 사항

- Python 3.10+
- NVIDIA GPU 및 CUDA 지원 PyTorch
- BF16 지원 GPU 권장(구형 GPU는 FP16 사용 가능)

먼저 CUDA에 맞는 PyTorch를 설치한 뒤 프로젝트와 FLA kernel을 설치하세요:

    pip install -e '.[cuda]'

## 빠른 시작

포함된 작은 텍스트 말뭉치로 학습:

    python -m kda_train.train --device cuda --steps 1000

직접 준비한 UTF-8 텍스트 말뭉치 사용:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

텍스트 생성:

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## 백엔드

- `auto`: CUDA 환경에서 `fla-core`가 설치되어 있으면 `fla.ops.kda.chunk_kda`를 사용하고, 아니면 미분 가능한 PyTorch 참조 구현으로 폴백합니다.
- `fla`: FLA CUDA 커널을 강제합니다.
- `reference`: 토큰 단위 PyTorch 스캔을 강제합니다. 전체 그래프를 보존하므로 긴 시퀀스 학습에는 부적합합니다.

CUDA 경로에서 KDA beta sigmoid 지원을 보장하려면 `fla-core >= 0.5.2`를 사용하세요.

## 알고리즘 요약

각 head는 K×V 상태 행렬을 유지합니다. 각 token에서 key별 망각 gate가 상태를 감쇠시킨 다음, key 예측 오차를 계산하여 delta rule로 value에 다시 기록합니다. query 벡터는 갱신된 상태를 읽습니다. 각 layer에는 입력 투영, beta와 채널별 gate, 출력 게이팅, MLP도 포함됩니다.

이 구현은 읽기 쉬운 학습 과정을 우선합니다. 전체 Kimi Linear를 재현하려면 하이브리드 MLA layer, 짧은 합성곱, 더 큰 규모, 완전한 학습 방법 등 추가 구성 요소가 필요합니다.

## 참고자료

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Kimi-Linear 공식 저장소](https://github.com/MoonshotAI/Kimi-Linear)
- [Flash Linear Attention(FLA) KDA 구현](https://github.com/fla-org/flash-linear-attention)

## 개발 및 기여

테스트 의존성 설치:

    pip install -e '.[test]'
    pytest -q

테스트는 PyTorch reference backend를 강제로 사용하므로 CPU에서 실행할 수 있습니다. CUDA kernel 학습 경로에서는 `.[cuda]` extra를 사용하세요. PR을 열기 전에 [CONTRIBUTING.md](../../CONTRIBUTING.md)를 읽어 주세요.

## 보안

잠재적 취약점은 공개 이슈에 올리지 마세요. 보고 방법은 [SECURITY.md](../../SECURITY.md)를 참고하세요.

## 웹사이트 통계 설정(Cloudflare Free)

기존 Cloudflare Pages 프로젝트가 웹사이트와 읽기 전용 `/stats.svg` Pages Function을 제공합니다. 함수는 Cloudflare Web Analytics를 조회하고 프로젝트 hostname과 site tag로 필터링한 뒤 통계 카드를 5분 동안 캐시합니다. 카드를 불러오는 것만으로 방문 카운터가 증가하지 않습니다. 이 웹사이트 수치는 아래 GitHub 저장소 통계의 VisitorBadge 및 FlagCounter 이미지와 독립적입니다.

필수 Pages 바인딩 3개는 [배포 및 통계 설정](../website-analytics.md)을 참고하세요. 설정이 누락되거나 원본 API가 실패하면 카드에 가짜 0을 표시하지 않고 통계를 사용할 수 없다고 명시합니다.

## 공식 웹사이트 방문 통계 — Cloudflare

[![공식 웹사이트 방문 통계 — Cloudflare](https://kda-cuda-trainer.pages.dev/stats.svg)](https://kda-cuda-trainer.pages.dev/#traffic-heading)

프로젝트 웹사이트의 최근 30일 Cloudflare Web Analytics 통계입니다. 웹사이트와 이 README는 같은 카드를 표시하며 5분마다 갱신됩니다. GitHub 이미지 캐시로 갱신이 지연될 수 있습니다. Visits는 순 방문자 수(UV)가 아닙니다. GitHub 저장소 조회 수는 제외됩니다. 국가/지역 비율은 웹사이트의 전체 페이지 조회 수를 기준으로 계산합니다.

## GitHub 저장소 방문 통계 — VisitorBadge 및 FlagCounter

아래 카운터는 이 저장소의 README 이미지 요청을 집계하며 공식 웹사이트의 Cloudflare 통계와 분리하여 표시합니다. 기존 카운터 식별자를 유지하고 모든 README 언어 버전이 공유합니다.

**README 조회 카운터(VisitorBadge):**

![README 조회 카운터(VisitorBadge)](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

**README 방문자 국가/지역(FlagCounter):**

[![README 방문자 국가/지역(FlagCounter)](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

이미지 기반의 대략적인 카운터입니다. GitHub 이미지 프록시와 캐시가 집계 수치와 표시 지역에 영향을 줄 수 있습니다. GitHub 공식 Traffic 지표가 아닙니다. 웹사이트와 README 수치는 따로 표시하며 합산하지 않습니다.
