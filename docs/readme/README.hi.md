# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

भाषाएँ: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer एक कॉम्पैक्ट और trainable causal language model उदाहरण है। इसका sequence-mixing मॉड्यूल Kimi Delta Attention (KDA) का उपयोग करता है: CUDA प्रशिक्षण FLA के chunked Triton kernel से चलता है, और readability व CPU प्रयोगों के लिए token-by-token PyTorch reference implementation भी शामिल है।

## शब्दावली

सभी भाषा संस्करणों में एकरूपता बनाए रखने के लिए प्रमुख तकनीकी शब्द अंग्रेज़ी में ही रखे गए हैं: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate`, `delta rule`।

## आवश्यकताएँ

- Python 3.10+
- NVIDIA GPU और CUDA-enabled PyTorch
- BF16 समर्थित GPU बेहतर है; पुराने GPU पर FP16 उपयोग किया जा सकता है

    pip install -e '.[cuda]'

## त्वरित शुरुआत

    python -m kda_train.train --device cuda --steps 1000

अपने UTF-8 टेक्स्ट कॉर्पस के साथ:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## बैकएंड

- `auto`: CUDA पर `fla-core` इंस्टॉल होने पर `fla.ops.kda.chunk_kda` उपयोग करता है, अन्यथा differentiable PyTorch reference path पर जाता है।
- `fla`: FLA CUDA kernel को force करता है।
- `reference`: token-by-token PyTorch scan को force करता है। यह पूरा graph रखता है, इसलिए लंबी sequence training के लिए उपयुक्त नहीं है।

CUDA path के लिए `fla-core >= 0.5.2` अनुशंसित है।

## एल्गोरिदम सारांश

हर head एक K×V state matrix बनाए रखता है। हर token पर per-key-channel forget gate state को decay करता है, फिर prediction error निकालकर delta rule से value में लिखता है। Query vectors updated state से output पढ़ते हैं। हर layer में input projection, beta और per-channel gate, output gating और MLP भी शामिल हैं।

## संदर्भ

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Kimi-Linear आधिकारिक रिपॉज़िटरी](https://github.com/MoonshotAI/Kimi-Linear)
- [Flash Linear Attention (FLA) KDA implementation](https://github.com/fla-org/flash-linear-attention)

## विकास और योगदान

    pip install -e '.[test]'
    pytest -q

टेस्ट PyTorch reference backend को force करते हैं, इसलिए CPU पर चल सकते हैं। CUDA kernel training path के लिए `.[cuda]` उपयोग करें। [CONTRIBUTING.md](../../CONTRIBUTING.md) देखें।

## सुरक्षा

कृपया संभावित सुरक्षा कमजोरियों को सार्वजनिक issue में प्रकाशित न करें। रिपोर्टिंग निर्देश [SECURITY.md](../../SECURITY.md) में हैं।

## विज़िट आँकड़े

- कुल विज़िट:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- देश/क्षेत्र वितरण (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)
