# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

भाषाएँ: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer शुरू से बनाया गया एक छोटा, प्रशिक्षण योग्य causal language model उदाहरण है। इसका sequence-mixing मॉड्यूल Kimi Delta Attention (KDA) उपयोग करता है: CUDA प्रशिक्षण में FLA का chunked Triton kernel चलता है, और पठनीयता तथा CPU प्रयोगों के लिए differentiable token-by-token PyTorch reference implementation भी शामिल है।

यह रिपॉज़िटरी केवल KDA पर आधारित शिक्षण मॉडल है, पूरा Kimi Linear नहीं। यह tokenizer निर्भरता से बचने के लिए UTF-8 bytes को token बनाता है। इसमें next-byte language modeling, AdamW, warmup/cosine LR schedule, gradient accumulation, mixed precision, validation loss, checkpoints और sampling शामिल हैं।

## परियोजना वेबसाइट

परियोजना का आधिकारिक मुखपृष्ठ: [kda-cuda-trainer.pages.dev](https://kda-cuda-trainer.pages.dev/)

डेमो और आँकड़ों पर आधारित आगंतुकों के भौगोलिक वितरण के लिए वेबसाइट मुख्य सार्वजनिक प्रवेश बिंदु है; कोड और रिलीज़ का प्रामाणिक स्रोत GitHub है।

## शब्दावली

सभी भाषा संस्करणों में एकरूपता बनाए रखने के लिए प्रमुख तकनीकी शब्द अंग्रेज़ी में ही रखे गए हैं: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate`, `delta rule`।

## आवश्यकताएँ

- Python 3.10+
- NVIDIA GPU और CUDA-enabled PyTorch
- BF16 समर्थित GPU बेहतर है; पुराने GPU पर FP16 उपयोग किया जा सकता है

पहले CUDA से मेल खाने वाला PyTorch इंस्टॉल करें, फिर परियोजना और FLA kernel इंस्टॉल करें:

    pip install -e '.[cuda]'

## त्वरित शुरुआत

शामिल छोटे टेक्स्ट कॉर्पस पर प्रशिक्षण:

    python -m kda_train.train --device cuda --steps 1000

अपना UTF-8 टेक्स्ट कॉर्पस उपयोग करें:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

टेक्स्ट उत्पन्न करें:

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## बैकएंड

- `auto`: CUDA पर `fla-core` इंस्टॉल होने पर `fla.ops.kda.chunk_kda` उपयोग करता है, अन्यथा differentiable PyTorch reference path पर जाता है।
- `fla`: FLA CUDA kernel को force करता है।
- `reference`: token-by-token PyTorch scan को force करता है। यह पूरा graph रखता है, इसलिए लंबी sequence training के लिए उपयुक्त नहीं है।

CUDA पथ में KDA beta sigmoid समर्थन सुनिश्चित करने के लिए `fla-core >= 0.5.2` उपयोग करें।

## एल्गोरिदम सारांश

हर head एक K×V state matrix रखता है। हर token पर, प्रति-key forget gate पहले state को क्षय करता है; फिर key prediction error की गणना होती है और delta rule से value वापस लिखी जाती है। query vectors अद्यतन state से पढ़ते हैं। हर layer में input projection, beta और प्रति-channel gate, output gating तथा MLP भी होते हैं।

इस कार्यान्वयन में प्रशिक्षण प्रक्रिया की पठनीयता को प्राथमिकता दी गई है। पूरे Kimi Linear को पुनरुत्पादित करने के लिए hybrid MLA layers, short convolution, बड़े पैमाने और पूरी training recipe सहित अतिरिक्त घटक चाहिए।

## संदर्भ

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Kimi-Linear आधिकारिक रिपॉज़िटरी](https://github.com/MoonshotAI/Kimi-Linear)
- [Flash Linear Attention (FLA) KDA implementation](https://github.com/fla-org/flash-linear-attention)

## विकास और योगदान

परीक्षण निर्भरताएँ इंस्टॉल करें:

    pip install -e '.[test]'
    pytest -q

परीक्षण PyTorch reference backend को बाध्य करते हैं, इसलिए CPU पर चल सकते हैं। CUDA kernel प्रशिक्षण के लिए `.[cuda]` extra उपयोग करें। PR खोलने से पहले [CONTRIBUTING.md](../../CONTRIBUTING.md) पढ़ें।

## सुरक्षा

कृपया संभावित सुरक्षा कमजोरियों को सार्वजनिक issue में प्रकाशित न करें। रिपोर्टिंग निर्देश [SECURITY.md](../../SECURITY.md) में हैं।

## वेबसाइट आँकड़ों का विन्यास (Cloudflare Free)

मौजूदा Cloudflare Pages परियोजना वेबसाइट और केवल पढ़ने वाला `/stats.svg` Pages Function प्रदान करती है। Function, Cloudflare Web Analytics से डेटा लेता है, परियोजना hostname और site tag के अनुसार फ़िल्टर करता है तथा कार्ड को पाँच मिनट cache करता है। कार्ड लोड करने से विज़िट काउंटर नहीं बढ़ता। वेबसाइट के ये आँकड़े नीचे GitHub रिपॉज़िटरी वाले VisitorBadge और FlagCounter चित्रों से स्वतंत्र हैं।

तीन आवश्यक Pages bindings के लिए [deployment और analytics विन्यास](../website-analytics.md) देखें। विन्यास अधूरा होने या upstream API विफल होने पर कार्ड आँकड़े अनुपलब्ध बताता है, नकली शून्य नहीं दिखाता।

## आधिकारिक वेबसाइट का ट्रैफ़िक — Cloudflare

[![आधिकारिक वेबसाइट का ट्रैफ़िक — Cloudflare](https://kda-cuda-trainer.pages.dev/stats.svg)](https://kda-cuda-trainer.pages.dev/#traffic-heading)

ये परियोजना वेबसाइट के पिछले 30 दिनों के Cloudflare Web Analytics आँकड़े हैं। वेबसाइट और यह README एक ही कार्ड दिखाते हैं, जो हर पाँच मिनट में ताज़ा होता है (GitHub image cache से देर हो सकती है)। Visits, unique visitors (UV) नहीं हैं। GitHub रिपॉज़िटरी views शामिल नहीं हैं। देश/क्षेत्र के प्रतिशत वेबसाइट के सभी page views पर आधारित हैं।

## GitHub रिपॉज़िटरी ट्रैफ़िक — VisitorBadge और FlagCounter

नीचे के काउंटर इस रिपॉज़िटरी के README image requests गिनते हैं और वेबसाइट के Cloudflare आँकड़ों से अलग हैं। पुराने काउंटर पहचानकर्ता बरकरार हैं और सभी README भाषाओं में साझा हैं।

**README views (VisitorBadge):**

![README views (VisitorBadge)](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

**README आगंतुकों के देश/क्षेत्र (FlagCounter):**

[![README आगंतुकों के देश/क्षेत्र (FlagCounter)](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

ये चित्र-आधारित अनुमानित काउंटर हैं: GitHub image proxy और cache गिनती तथा दिखने वाले क्षेत्रों को प्रभावित कर सकते हैं। ये GitHub के आधिकारिक Traffic metrics नहीं हैं। वेबसाइट और README की गिनती अलग दिखाई जाती है, जोड़ी नहीं जाती।
