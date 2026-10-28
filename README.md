# Qwen3 Base + LoRA SFT

使用 GSM8K 完整标准解答训练 Qwen3 Base，只训练 PEFT LoRA 适配器。训练与抽样使用同一份简洁的 user / assistant 聊天模板，约定见 [Base 训练约定](docs/qwen3-base.md)。

## 数据

```text
<|im_start|>user
题目原文<|im_end|>
<|im_start|>assistant
标准解答的推导原文
#### 最终数值<|im_end|>
```

解答原文（含计算器注释和 `####`）及助手结束标记参与损失。题目、回答前缀和 padding 标签为 `-100`。不截断样本，不拆分推导与结果；聊天模板随分词器保存。

检查一条真实样本及其监督范围：

```bash
python prepare_sft_data.py --config configs/development/sft.yaml
```

## 训练

在运行训练的 Python 环境中安装依赖，然后选择对应配置：

```bash
python -m pip install -r requirements.txt
python train_sft.py --config configs/training/sft.yaml
```

| 配置 | 起始模型 | 默认输出目录 |
| --- | --- | --- |
| `configs/development/sft.yaml` | Qwen/Qwen3-0.6B-Base | `outputs/development/qwen3-0.6b-base-gsm8k-lora` |
| `configs/training/sft.yaml` | Qwen/Qwen3-4B-Base | `/root/autodl-tmp/outputs/training/qwen3-4b-base-gsm8k-lora` |
| `configs/training/sft-8b.yaml` | Qwen/Qwen3-8B-Base | `/root/autodl-tmp/outputs/training/qwen3-8b-base-gsm8k-lora` |

配置只包含模型、输出目录、轮次、学习率、批量和 LoRA 参数。模型固定 FP32，使用 Trainer 默认训练行为；每批直接更新，不配置混合精度、梯度累积或额外显存优化。LoRA 应用于注意力和 MLP 线性层，默认 r=16、alpha=32、dropout=0.05。这些是起始实验参数，实际显存和准确率需在相应机器验证。

Trainer 输出基础训练日志，每轮保存 checkpoint，最多保留两个。训练结束的 `final/` 包含适配器、配置和分词器；checkpoint 另含优化器、调度器等原生续训状态。适配器需要配合其对应 Base 权重使用。

每次新实验指定空目录，已有非空目录会报错。使用 `--output-dir` 选择其他路径：

```bash
python train_sft.py --config configs/training/sft.yaml --output-dir /root/autodl-tmp/outputs/training/base-lora-run-2
```

使用同一份配置，从新流程产生的 `checkpoint-*` 续训；路径以实际保留的 checkpoint 为准。适配器参数取自 checkpoint，`final/` 不用于恢复优化器状态：

```bash
python train_sft.py --config configs/training/sft.yaml --resume-from-checkpoint /root/autodl-tmp/outputs/training/qwen3-4b-base-gsm8k-lora/checkpoint-14946
```

## 抽样查看

查看原始 Base，或查看本流程保存的 LoRA 产物：

```bash
python sample_checkpoint.py --model-name Qwen/Qwen3-4B-Base --num-samples 10
python sample_checkpoint.py --checkpoint /root/autodl-tmp/outputs/training/qwen3-4b-base-gsm8k-lora/final --num-samples 10
```

默认从 test 抽 5 道题，种子 42，采用贪心生成，输出预算为 1024 个新词元；这是本项目设置。可用 `--split`、`--seed`、`--max-new-tokens` 调整。报告包含题目、完整回答、是否截断和标准解答，默认保存到 `outputs/samples/`，终端输出文件路径；`--output` 可指定报告位置。重复使用同名报告会覆盖该报告。

原始模型可下载缺失文件，也可加 `--local-files-only` 限制为已有缓存。checkpoint 抽样仅从本地文件及缓存读取，需准备其对应 Base 权重。数据也已缓存时，设置 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` 可完全离线运行。

推理自动选择 CUDA 或 CPU，均为 FP32；也可用 `--device cpu` 显式选择。4B / 8B 在训练机上抽样，开发机使用 0.6B。参考解答只写入报告，不输入模型；少量问答不等于准确率评测。

## 本地验证

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python tools/test_base_sft.py -v
```

测试使用临时随机微型 Qwen3 与合成问题，不下载模型权重或进行真实模型训练。
