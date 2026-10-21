# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![الترخيص: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

اللغات: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer مثال صغير لنموذج لغوي سببي قابل للتدريب، مبني من الصفر. تستخدم وحدة مزج التسلسل Kimi Delta Attention (KDA): يعتمد التدريب على CUDA على kernel ‏Triton المقسّم إلى كتل من FLA، ويتضمن المشروع تنفيذًا مرجعيًا قابلًا للاشتقاق في PyTorch، يعمل token بعد token، لتسهيل القراءة والتجارب على CPU.

ينفذ هذا المستودع نموذجًا تعليميًا يعتمد على KDA وحده، وليس Kimi Linear الكامل. يستخدم بايتات UTF-8 بوصفها token لتجنب الاعتماد على tokenizer، ويشمل نمذجة البايت التالي وAdamW وجدولة معدل التعلم warmup/cosine وتراكم التدرجات والدقة المختلطة وخسارة التحقق وcheckpoints وأخذ العينات.

## موقع المشروع

الصفحة الرئيسية الرسمية للمشروع: [kda-cuda-trainer.pages.dev](https://kda-cuda-trainer.pages.dev/)

الموقع هو المدخل العام الرئيسي للعروض والتوزيع الجغرافي للزوار المبني على بيانات الإحصاءات، بينما يبقى GitHub المصدر المعتمد للكود والإصدارات.

## المصطلحات

لضمان الاتساق بين جميع النسخ اللغوية، نُبقي المصطلحات التقنية الأساسية باللغة الإنجليزية: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate`, `delta rule`.

## المتطلبات

- Python 3.10+
- بطاقة NVIDIA ونسخة PyTorch الداعمة لـ CUDA
- يُنصح ببطاقات تدعم BF16؛ ويمكن استخدام FP16 على البطاقات الأقدم

ثبّت أولًا إصدار PyTorch المتوافق مع CUDA لديك، ثم ثبّت المشروع وkernel الخاص بـ FLA:

    pip install -e '.[cuda]'

## بداية سريعة

درّب النموذج باستخدام مجموعة النصوص الصغيرة المرفقة:

    python -m kda_train.train --device cuda --steps 1000

استخدم مجموعة نصوص UTF-8 خاصة بك:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

ولّد نصًا:

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## الواجهات الخلفية

- `auto`: يستخدم `fla.ops.kda.chunk_kda` على CUDA عند تثبيت `fla-core`، وإلا يعود إلى تنفيذ PyTorch المرجعي القابل للاشتقاق.
- `fla`: يفرض استخدام FLA CUDA kernel.
- `reference`: يفرض المسح token-by-token في PyTorch. هذا المسار يحتفظ بالمخطط الكامل وغير مناسب للتسلسلات الطويلة.

استخدم `fla-core >= 0.5.2` لضمان دعم KDA beta sigmoid في مسار CUDA.

## ملخص الخوارزمية

يحتفظ كل head بمصفوفة حالة K×V. لكل token، يخفّض gate النسيان الخاص بكل key الحالة أولًا؛ ثم يُحسب خطأ التنبؤ بـ key ويُكتب في value باستخدام delta rule. تقرأ متجهات query من الحالة المحدّثة. يتضمن كل layer أيضًا إسقاط الإدخال وbeta وgate لكل قناة والتحكم في الإخراج وMLP.

يعطي هذا التنفيذ الأولوية لوضوح مسار التدريب. تتطلب إعادة إنتاج Kimi Linear بالكامل مكونات إضافية، منها layers هجينة من MLA والتفاف قصير وحجم أكبر ووصفة تدريب كاملة.

## المراجع

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [المستودع الرسمي لـ Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [تنفيذ KDA في Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)

## التطوير والمساهمة

ثبّت متطلبات الاختبار:

    pip install -e '.[test]'
    pytest -q

تفرض الاختبارات استخدام backend المرجعي في PyTorch، لذا تعمل على CPU. لمسار التدريب باستخدام kernel CUDA، استخدم الإضافة `.[cuda]`. اقرأ [CONTRIBUTING.md](../../CONTRIBUTING.md) قبل فتح PR.

## الأمان

يرجى عدم نشر الثغرات المحتملة في القضايا العامة. إرشادات الإبلاغ موجودة في [SECURITY.md](../../SECURITY.md).

## إعداد إحصاءات الموقع (Cloudflare Free)

يقدّم مشروع Cloudflare Pages الحالي الموقع وPages Function للقراءة فقط على `/stats.svg`. تستعلم الوظيفة من Cloudflare Web Analytics، وترشّح حسب hostname وsite tag الخاصين بالمشروع، وتخزّن بطاقة الإحصاءات مؤقتًا لخمس دقائق. تحميل البطاقة لا يزيد عدّاد الزيارات. بيانات الموقع هذه مستقلة عن صور VisitorBadge وFlagCounter في قسم زيارات مستودع GitHub أدناه.

راجع [إعداد النشر والإحصاءات](../website-analytics.md) للـ bindings الثلاثة المطلوبة في Pages. إذا لم تُضبط أو فشلت API المصدر، تعرض البطاقة بوضوح أن الإحصاءات غير متاحة بدل عرض أصفار وهمية.

## زيارات الموقع الرسمي — Cloudflare

[![زيارات الموقع الرسمي — Cloudflare](https://kda-cuda-trainer.pages.dev/stats.svg)](https://kda-cuda-trainer.pages.dev/#traffic-heading)

هذه إحصاءات Cloudflare Web Analytics لموقع المشروع خلال آخر 30 يومًا. يشترك الموقع وREADME هذا في البطاقة نفسها، وتتحدث كل خمس دقائق، وقد تؤخر ذاكرة صور GitHub المؤقتة التحديث. لا تعني Visits عدد الزوار الفريدين (UV). لا تشمل الأرقام مشاهدات مستودع GitHub، وتُحسب نسب البلدان والمناطق من جميع مشاهدات صفحات الموقع.

## زيارات مستودع GitHub — VisitorBadge وFlagCounter

تتبع العدادات أدناه طلبات صور README لهذا المستودع بصورة منفصلة عن إحصاءات Cloudflare للموقع الرسمي. تُحفظ معرّفات العدادات الأصلية، وتشترك فيها جميع لغات README.

**مشاهدات README ‏(VisitorBadge):**

![مشاهدات README ‏(VisitorBadge)](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

**بلدان ومناطق زوار README ‏(FlagCounter):**

[![بلدان ومناطق زوار README ‏(FlagCounter)](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

هذه عدادات تقريبية تعتمد على الصور: قد يؤثر وكيل صور GitHub وذاكرتها المؤقتة في الأعداد والمناطق الظاهرة. ليست هذه مقاييس GitHub Traffic الرسمية. تُعرض أعداد الموقع وREADME منفصلة ولا تُجمع.
