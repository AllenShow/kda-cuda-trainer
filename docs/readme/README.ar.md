# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![الترخيص: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

اللغات: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer هو مثال مدمج لنموذج لغوي سببي قابل للتدريب. يستخدم Kimi Delta Attention (KDA) لخلط التسلسل: مسار CUDA يعتمد على chunked Triton kernel من FLA، كما يتوفر تنفيذ مرجعي قابل للاشتقاق في PyTorch على مستوى كل token.

## المصطلحات

لضمان الاتساق بين جميع النسخ اللغوية، نُبقي المصطلحات التقنية الأساسية باللغة الإنجليزية: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate`, `delta rule`.

## المتطلبات

- Python 3.10+
- بطاقة NVIDIA ونسخة PyTorch الداعمة لـ CUDA
- يُنصح ببطاقات تدعم BF16؛ ويمكن استخدام FP16 على البطاقات الأقدم

    pip install -e '.[cuda]'

## بداية سريعة

    python -m kda_train.train --device cuda --steps 1000

باستخدام نص UTF-8 خاص بك:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## الواجهات الخلفية

- `auto`: يستخدم `fla.ops.kda.chunk_kda` على CUDA عند تثبيت `fla-core`، وإلا يعود إلى تنفيذ PyTorch المرجعي القابل للاشتقاق.
- `fla`: يفرض استخدام FLA CUDA kernel.
- `reference`: يفرض المسح token-by-token في PyTorch. هذا المسار يحتفظ بالمخطط الكامل وغير مناسب للتسلسلات الطويلة.

في مسار CUDA يُوصى باستخدام `fla-core >= 0.5.2`.

## ملخص الخوارزمية

يحافظ كل head على مصفوفة حالة K×V. لكل token، تقوم بوابة نسيان لكل قناة key بتوهين الحالة، ثم يُحسب خطأ التنبؤ ويُكتب مرة أخرى إلى value عبر قاعدة delta. تقرأ متجهات query من الحالة بعد التحديث. تتضمن كل طبقة أيضًا إسقاط الإدخال و beta وبوابة لكل قناة وبوابة إخراج و MLP.

## المراجع

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [المستودع الرسمي لـ Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [تنفيذ KDA في Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)

## التطوير والمساهمة

    pip install -e '.[test]'
    pytest -q

تفرض الاختبارات استخدام backend المرجعي في PyTorch، لذلك يمكن تشغيلها على CPU. لمسار تدريب CUDA استخدم `.[cuda]`. راجع [CONTRIBUTING.md](../../CONTRIBUTING.md).

## الأمان

يرجى عدم نشر الثغرات المحتملة في القضايا العامة. إرشادات الإبلاغ موجودة في [SECURITY.md](../../SECURITY.md).

## إحصاءات الزيارات

- إجمالي الزيارات:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- التوزيع حسب الدول/المناطق (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)
