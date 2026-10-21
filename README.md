# KDA CUDA Trainer

一个从空项目搭起的、可训练的小型因果语言模型示例。序列混合模块使用 Kimi Delta Attention（KDA）：CUDA 训练通过 FLA 提供的分块 Triton kernel 执行；项目内另有一个可微的 PyTorch 逐 token 参考实现，方便阅读算法和在 CPU 上跑小型实验。

这个项目实现的是一个纯 KDA 的教学模型，不是完整的 Kimi Linear。它使用 UTF-8 字节作为 token，避免额外 tokenizer 依赖；训练代码包括 next-byte 语言建模、AdamW、warmup/cosine 学习率、梯度累积、混合精度、验证 loss、checkpoint 和采样。

## 环境

- Python 3.10 或更新版本
- NVIDIA GPU 和 CUDA 版 PyTorch
- 建议使用支持 BF16 的 GPU；较旧 GPU 可以选择 FP16

先按 PyTorch 官方安装说明安装与你的 CUDA 版本匹配的 PyTorch，再安装项目和 FLA CUDA kernel：

    pip install -e '.[cuda]'

## 快速开始

在 NVIDIA CUDA 机器上用仓库内的小文本训练：

    python -m kda_train.train --device cuda --steps 1000

训练默认读取 data/tiny.txt，checkpoint 写入 outputs/kda.pt。指定自己的 UTF-8 文本：

    python -m kda_train.train --dataset /path/to/corpus.txt --device cuda --seq-len 512 --batch-size 8

采样生成：

    python -m kda_train.generate outputs/kda.pt --prompt 'Once upon a time' --tokens 200

训练参数可用 --help 查看。模型默认 4 层、256 hidden size、4 个 KDA heads，适合先验证训练流程；实际训练建议增加数据、模型宽度和步数。

## 后端

- auto：CUDA 且安装了 fla-core 时调用 fla.ops.kda.chunk_kda；其他情况使用可微 PyTorch 参考实现。
- fla：强制使用 FLA CUDA kernel。
- reference：强制逐 token PyTorch scan。它会保留整条计算图，内存和速度都不适合长序列训练。

CUDA 安装应使用 fla-core 0.5.2 或更新版本，以确保 KDA 的 beta sigmoid 选项可用。这个项目使用 FLA 的公开 kernel，不复制其 Triton kernel 源码。

## 算法概要

每个 head 维护一个 K×V 状态矩阵。每个 token 先用逐 key 通道的遗忘门衰减状态，再计算 key 对当前状态的预测误差，并用 delta rule 写入 value。查询向量从更新后的状态读取输出。每层还包括输入投影、beta 和逐通道 gate、输出门控及 MLP。

本仓库的实现以可读训练流程为目标。若要复现完整 Kimi Linear，还需要其混合 MLA 层、短卷积、模型规模和训练 recipe。

## 参考

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Moonshot AI, [Kimi-Linear 官方仓库](https://github.com/MoonshotAI/Kimi-Linear)
- [Flash Linear Attention（FLA）KDA 实现](https://github.com/fla-org/flash-linear-attention)

