# GSM8K SFT

使用原始 GSM8K 题目和标准解答，对 Qwen3 做全参数监督微调。训练机默认使用 Qwen3-1.7B，开发机默认使用 Qwen3-0.6B。训练使用 PyTorch、Transformers、Datasets 和 Accelerate，配置通过 PyYAML 读取。

## 数据约定

- `question` 和 `answer` 原样使用，保留解题过程、`<<表达式=结果>>` 计算器注释和 `#### 数值`。
- 使用 Qwen3 的非思考对话模板，将解题过程放在普通助手回答中；模板自带的空 `<think>` 前缀不计损失。这仍然包含逐步解题过程。
- `input_ids` 包含题目和标准解答；题目和对话前缀的 `labels` 为 `-100`，助手解答及结束标记接受监督。标签保持与输入等长，由后续因果语言模型在计算损失时移位。
- 样本不截断，批次内按最长样本补齐，补齐位置的标签为 `-100`。

## 训练

配置集中放在 `configs/`：

```text
configs/
├── common.yaml              # 两台机器共用的精度、数据和模板设置
├── development/
│   └── sft.yaml             # Qwen3-0.6B，本机开发环境（8 GB 显存）
└── training/
    └── sft.yaml             # Qwen3-1.7B，租用训练环境（32 GB 显存，默认）
```

修改对应 YAML 文件即可调整模型和训练参数，无须改训练代码。未传 `--config` 时，训练程序始终读取训练机配置，不自动根据本机显存切换。后续按机器准备配置的约定见 [默认训练环境](findings.md#默认训练环境)。

在租用 GPU 上进入仓库目录，使用装有 `requirements.txt` 中依赖的 Python 环境运行：

```bash
python train_sft.py
```

显式选择训练机配置或开发机配置：

```bash
python train_sft.py --config configs/training/sft.yaml
python train_sft.py --config configs/development/sft.yaml
```

开发机配置单独使用每批 1 条、梯度累积 16 次、Adafactor 和梯度检查点，输出到 `outputs/development/gsm8k-sft/`；训练机配置保持下面的 32 GB 设置。

首次运行会下载模型和数据；已有完整缓存时，可在命令前加 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` 离线运行。

训练配置面向用户租用的 **32 GB 显存 GPU**。默认训练 3 轮，学习率 `2e-5`，每批 2 条、梯度累积 8 次，单卡有效批量为 16 条。1.7B 模型的权重、梯度和 AdamW 状态占用更多显存，因此启用非重入梯度检查点，并使用融合 AdamW，避免默认 foreach 实现额外的整组参数中间量。CUDA 支持 BF16 时自动使用 BF16 混合精度，模型权重保持 FP32。具体 GPU 型号尚未确定，实际训练显存仍需在训练机上验证。

训练损失每 10 个优化器更新步骤显示一次。每轮保存检查点，保留最近两个；完整训练结束后，模型和分词器保存在 `outputs/training/qwen3-1.7b-gsm8k-sft/final/`，训练汇总保存在同一运行目录下的 `train_results.json`。新输出目录与原有 0.6B 检查点分开。程序只使用训练集，不自动评测。

命令行中的 `--epochs`、`--learning-rate`、`--batch-size`、`--gradient-accumulation-steps` 和 `--output-dir` 可临时覆盖所选配置。例如只训练一轮并使用新的输出目录：

```bash
python train_sft.py --epochs 1 --output-dir outputs/training/qwen3-1.7b-gsm8k-sft-1epoch
```

已有非空输出目录时，程序要求另选目录或明确续训。中断后可以指定现有检查点恢复训练；将下面的路径替换为实际目录：

```bash
python train_sft.py --resume-from-checkpoint outputs/training/qwen3-1.7b-gsm8k-sft/checkpoint-STEP
```

续训时使用初次训练的模型、配置文件及参数；如果初次使用了自定义 `--output-dir`，续训时也指定同一路径。开发机续训也需传入 `--config configs/development/sft.yaml`。

## 预览训练样本

在已有依赖的环境中，预览一条训练样本及其词元和标签：

```bash
/home/yunli/miniconda3/bin/python prepare_sft_data.py
```

样本预览默认使用开发机配置，预览训练集第一条样本，打印模型名称、完整输入、参与损失的文本、词元 ID 和监督标签。预览训练机模型时显式选择同一配置：

```bash
python prepare_sft_data.py --config configs/training/sft.yaml
```

本机已有完整缓存，可离线运行：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 /home/yunli/miniconda3/bin/python prepare_sft_data.py
```

`build_sft_example()` 返回 `input_ids`、`attention_mask` 和 `labels`，供后续训练代码复用。推理时需使用同样的非思考前缀及答案格式约定。

保留计算器注释只提供训练文本，不会自动执行计算。

## 来源

- [GSM8K 数据格式与计算器说明](https://github.com/openai/grade-school-math#dataset-details)
- [Qwen3-0.6B 对话模板与模式说明](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen3-1.7B 对话模板与模式说明](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
