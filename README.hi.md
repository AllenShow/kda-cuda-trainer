# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

भाषाएँ: [English](README.md) | [中文](README.zh-CN.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Italiano](README.it.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [العربية](README.ar.md) | [Nederlands](README.nl.md) | [हिन्दी](README.hi.md) | [Bahasa Indonesia](README.id.md)

KDA CUDA Trainer एक कॉम्पैक्ट और trainable causal language model उदाहरण है। इसका sequence-mixing मॉड्यूल Kimi Delta Attention (KDA) का उपयोग करता है: CUDA प्रशिक्षण FLA के chunked Triton kernel से चलता है, और readability व CPU प्रयोगों के लिए token-by-token PyTorch reference implementation भी शामिल है।

## आवश्यकताएँ

- Python 3.10+
- NVIDIA GPU और CUDA-enabled PyTorch
- BF16 समर्थित GPU बेहतर है; पुराने GPU पर FP16 उपयोग किया जा सकता है

    pip install -e '.[cuda]'

## त्वरित शुरुआत

    python -m kda_train.train --device cuda --steps 1000
    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## विकास

    pip install -e '.[test]'
    pytest -q

कृपया [CONTRIBUTING.md](CONTRIBUTING.md) और [SECURITY.md](SECURITY.md) देखें।

## विज़िट आँकड़े

- कुल विज़िट:

  ![Visitor Count](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

- देश/क्षेत्र वितरण (FlagCounter):

  [![Flag Counter](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

- विज़िटर मैप (ClustrMaps):

  [![ClustrMaps](https://www.clustrmaps.com/map_v2.png?d=YOUR_CLUSTRMAPS_TOKEN&cl=ffffff&w=a)](https://clustrmaps.com/site/YOUR_CLUSTRMAPS_SITE_ID)
