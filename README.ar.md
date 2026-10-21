# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![الترخيص: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

اللغات: [English](README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

مشروع KDA CUDA Trainer هو مثال مدمج لنموذج لغوي سببي قابل للتدريب. يستخدم Kimi Delta Attention (KDA) لخلط التسلسل: التدريب على CUDA يعتمد على chunked Triton kernel من FLA، مع وجود تنفيذ مرجعي تفاضلي في PyTorch على مستوى كل token.

## المتطلبات

- Python 3.10+
- بطاقة NVIDIA ونسخة PyTorch الداعمة لـ CUDA
- يوصى ببطاقات تدعم BF16؛ ويمكن استخدام FP16 على البطاقات الأقدم

    pip install -e '.[cuda]'

## بداية سريعة

    python -m kda_train.train --device cuda --steps 1000
    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## التطوير

    pip install -e '.[test]'
    pytest -q

راجع [CONTRIBUTING.md](CONTRIBUTING.md) و [SECURITY.md](SECURITY.md).

## إحصاءات الزيارات

- إجمالي الزيارات:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- التوزيع حسب الدول/المناطق (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- خريطة الزوار (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
