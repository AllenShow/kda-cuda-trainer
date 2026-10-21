# KDA CUDA Trainer

[![CI](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/AllenShow/kda-cuda-trainer/actions/workflows/ci.yml)
[![Licença: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

Languages: [English](../../README.md) | [中文](README.zh-CN.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Português](README.pt.md) | [Русский](README.ru.md) | [Bahasa Indonesia](README.id.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md)

KDA CUDA Trainer é um exemplo compacto de modelo de linguagem causal treinável construído do zero. Seu módulo de mistura de sequências usa Kimi Delta Attention (KDA): o treino CUDA utiliza o kernel Triton em blocos do FLA, e uma implementação de referência PyTorch diferenciável token por token facilita a leitura e os experimentos em CPU.

Este repositório implementa um modelo didático puramente KDA, não o Kimi Linear completo. Usa bytes UTF-8 como token para evitar dependências de tokenizer e inclui modelagem do próximo byte, AdamW, agendamento de LR warmup/cosseno, acumulação de gradientes, precisão mista, perda de validação, checkpoints e amostragem.

## Site do projeto

Página oficial do projeto: [kda-cuda-trainer.pages.dev](https://kda-cuda-trainer.pages.dev/)

O site é a principal página pública para demonstrações e distribuição geográfica dos visitantes baseada em estatísticas; o GitHub permanece a fonte canônica do código e das versões.

## Terminologia

Para manter consistência entre todas as versões, os principais termos técnicos permanecem em inglês: `token`, `backend`, `kernel`, `head`, `layer`, `query`, `key`, `value`, `gate` e `delta rule`.

## Requisitos

- Python 3.10+
- GPU NVIDIA e PyTorch com CUDA
- Recomenda-se GPU com BF16; GPUs antigas podem usar FP16

Instale primeiro o PyTorch compatível com seu CUDA e, em seguida, o projeto e o kernel FLA:

    pip install -e '.[cuda]'

## Início rápido

Treine com o pequeno corpus de texto incluído:

    python -m kda_train.train --device cuda --steps 1000

Use seu próprio corpus de texto UTF-8:

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

Gere texto:

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

## Backends

- `auto`: usa `fla.ops.kda.chunk_kda` no CUDA quando `fla-core` está instalado; caso contrário, cai para a referência diferenciável em PyTorch.
- `fla`: força o kernel CUDA do FLA.
- `reference`: força varredura token a token em PyTorch. Mantém o grafo completo e não é adequado para sequências longas.

Use `fla-core >= 0.5.2` para garantir suporte a KDA beta sigmoid no caminho CUDA.

## Resumo do algoritmo

Cada head mantém uma matriz de estado K×V. Para cada token, um gate de esquecimento por key atenua o estado; em seguida, calcula-se o erro de predição de key e escreve-se de volta em value pela delta rule. Os vetores query leem o estado atualizado. Cada layer inclui também projeção de entrada, beta e um gate por canal, modulação de saída e MLP.

Esta implementação prioriza um fluxo de treinamento legível. A reprodução completa do Kimi Linear exige componentes adicionais, como layers híbridos MLA, convolução curta, maior escala e a receita completa de treinamento.

## Referências

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Repositório oficial Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Implementação KDA do Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)

## Desenvolvimento e contribuição

Instale as dependências de teste:

    pip install -e '.[test]'
    pytest -q

Os testes forçam o backend de referência PyTorch e podem rodar em CPU. Para treino com o kernel CUDA, use o extra `.[cuda]`. Leia [CONTRIBUTING.md](../../CONTRIBUTING.md) antes de abrir um PR.

## Segurança

Não divulgue possíveis vulnerabilidades em issues públicos. Consulte [SECURITY.md](../../SECURITY.md).

## Configuração de estatísticas do site (Cloudflare Free)

O projeto Cloudflare Pages existente serve o site e uma Pages Function somente de leitura em `/stats.svg`. A função consulta o Cloudflare Web Analytics, filtra pelo hostname e pelo site tag do projeto e mantém o cartão em cache por cinco minutos. Carregar o cartão não incrementa um contador de visitas. Esses números do site são independentes das imagens VisitorBadge e FlagCounter na seção de tráfego GitHub abaixo.

Veja a [configuração de implantação e estatísticas](../website-analytics.md) para os três bindings Pages necessários. Sem configuração ou se a API de origem falhar, o cartão informa explicitamente que as estatísticas estão indisponíveis, em vez de mostrar zeros fictícios.

## Tráfego do site oficial — Cloudflare

[![Tráfego do site oficial — Cloudflare](https://kda-cuda-trainer.pages.dev/stats.svg)](https://kda-cuda-trainer.pages.dev/#traffic-heading)

Estatísticas do Cloudflare Web Analytics para o site do projeto nos últimos 30 dias. O site e este README compartilham o mesmo cartão, atualizado a cada cinco minutos (o cache de imagens do GitHub pode atrasar a atualização). Visits não equivale a visitantes únicos (UV). As visualizações do repositório GitHub são excluídas. As porcentagens por país/região usam todas as visualizações de páginas do site.

## Tráfego do repositório GitHub — VisitorBadge e FlagCounter

Os contadores abaixo registram solicitações das imagens do README deste repositório, separadamente das estatísticas Cloudflare do site oficial. Os identificadores originais são mantidos e compartilhados entre os idiomas do README.

**Visualizações do README (VisitorBadge):**

![Visualizações do README (VisitorBadge)](https://api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&label=Repository%20Views&labelColor=%23697689&countColor=%23263759)

**Países/regiões dos visitantes do README (FlagCounter):**

[![Países/regiões dos visitantes do README (FlagCounter)](https://s05.flagcounter.com/count/WGbp/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_20/viewers_1/labels_1/pageviews_1/flags_0/percent_1/)](https://info.flagcounter.com/WGbp)

São contadores aproximados baseados em imagens: o proxy e o cache de imagens do GitHub podem afetar os números e as regiões aparentes. Eles não são as métricas oficiais do GitHub Traffic. Os números do site e do README são exibidos separadamente e não são somados.
