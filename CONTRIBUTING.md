# Contributing

感谢你对 KDA CUDA Trainer 的关注。

## 开始前

1. 先查看现有的 Issue，避免重复工作。
2. 较大的改动请先创建 Issue，说明动机、方案和验证方式。
3. 每个 Pull Request 聚焦一件事，并说明影响范围。

## 本地开发

项目需要 Python 3.10 或更高版本。安装测试依赖并运行测试：

    pip install -e '.[test]'
    pytest -q

测试包含 CPU reference backend 的两步训练、checkpoint 保存和加载生成流程。
该流程在临时目录保存 checkpoint，无需 GPU 或 FLA；CI 会在 Python 3.10 和 3.12 上运行。

CUDA 路径还需要与本机 CUDA 匹配的 PyTorch 和 FLA：

    pip install -e '.[cuda]'

## Pull Request 要求

- 保持公开 API、参数名称和 README 示例一致。
- 为行为变更加测试，尤其是张量形状、dtype 与数值稳定性。
- 不提交 checkpoint、训练语料或其他大文件。
- 在 PR 描述中写明测试命令及结果。

## 行为准则

请保持讨论专注、尊重且可复现。欢迎报告问题、改进文档和提交代码。

