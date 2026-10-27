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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_sft.py
```

显式选择训练机配置或开发机配置：

```bash
python train_sft.py --config configs/training/sft.yaml
python train_sft.py --config configs/development/sft.yaml
```

开发机配置单独使用每批 1 条、梯度累积 16 次、Adafactor 和梯度检查点，输出到 `outputs/development/gsm8k-sft/`；训练机配置保持下面的 32 GB 设置。

首次运行会下载模型和数据；已有完整缓存时，可在命令前加 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` 离线运行。

训练配置面向用户租用的 **32 GB 显存 GPU**。默认训练 3 轮，学习率 `2e-5`，每批 1 条、梯度累积 16 次，单卡有效批量为 16 条。1.7B 模型的权重、梯度和 AdamW 状态占用更多显存，因此启用非重入梯度检查点，并使用融合 AdamW，避免默认 foreach 实现额外的整组参数中间量。CUDA 支持 BF16 时自动使用 BF16 混合精度，模型权重保持 FP32。启动时会打印实际批量、累积次数、BF16 和梯度检查点开关，便于核对机器环境。

AutoDL 的 NVIDIA vGPU-32GB 实测显示此前每批 2 条会在反向传播时显存不足，因此将微批量降为 1，并增加累积次数。上面的分配器环境变量用于缓解变长样本带来的显存碎片，需在启动 Python 前设置。新配置的峰值显存仍需在训练机上实测。

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

若沿用旧配置运行后 OOM，可直接用命令行覆盖重试，并使用新目录保留失败日志：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_sft.py \
  --batch-size 1 --gradient-accumulation-steps 16 \
  --output-dir outputs/training/qwen3-1.7b-gsm8k-sft-bs1
```

按当前每轮保存策略，第一轮结束前失败的训练尚未产生检查点，需从头重试。若重试目录已有内容，再选择新目录；从实际存在的检查点恢复时才使用 `--resume-from-checkpoint`。

## TensorBoard 训练指标

两台机器的配置均已启用 TensorBoard。先在训练用的 Python 环境中安装项目依赖，然后按上面的命令训练：

```bash
python -m pip install -r requirements.txt
```

Trainer 在第 1 次参数更新和之后每 10 次参数更新记录一次曲线；横轴为优化器更新次数，梯度累积中的单个微批次不会单独记为一步。

| TensorBoard 指标 | 含义 |
| --- | --- |
| `train/loss` | 上次记录以来的平均训练损失 |
| `train/learning_rate` | 学习率 |
| `train/grad_norm` | 当前更新在梯度裁剪前的梯度范数 |
| `train/epoch` | 当前训练轮次，含轮内进度 |
| `train/train_loss` | 训练结束时的整体平均损失 |
| `train/train_runtime` | 本次训练耗时，单位为秒，结束时记录 |
| `train/train_samples_per_second` | 本次训练的平均样本吞吐量，结束时记录 |
| `train/train_steps_per_second` | 本次训练每秒平均参数更新次数，结束时记录 |

日志默认写入所选 `output_dir` 下的 `runs/<时间戳_主机名>/`，训练机和开发机随各自输出目录分开；`--output-dir` 覆盖后日志也随之移动。恢复检查点时会新建日志目录，横轴沿用恢复的更新次数。日志已包含在 Git 忽略的 `outputs/` 中。

在项目目录启动查看服务：

```bash
python -m tensorboard.main --logdir outputs --host 127.0.0.1 --port 6006
```

在浏览器打开 `http://127.0.0.1:6006`，在 Scalars 或 Time Series 中选择对应运行。命令会同时发现两台机器保存在此目录中的日志；也可将 `--logdir` 指向某次训练的输出目录。若训练运行在远端，通过 SSH 转发该端口后在本机查看，例如 `ssh -L 6006:127.0.0.1:6006 用户名@训练机地址`。

当前训练脚本只使用训练集；验证损失和 GSM8K 答案正确率需后续接入评测后记录。

## 下载分词器时的 Brotli 报错

若 `AutoTokenizer.from_pretrained()` 下载文件时在 `httpx2/_decoders.py` 报 `TypeError: process() takes no keyword arguments`，先修复训练所用解释器的 Brotli 依赖。HTTPX2 传入了 `output_buffer_limit`，旧版 Brotli 的 `process()` 不接受这个参数；项目依赖已固定兼容版本。

在原来的训练环境中执行：

```bash
python -m pip install --upgrade "Brotli==1.2.1"
python -c "import brotli; assert brotli.Decompressor().process(brotli.compress(b'ok'), output_buffer_limit=1024) == b'ok'; print('Brotli OK:', brotli.__version__)"
```

看到 `Brotli OK: 1.2.1` 后重新运行原训练命令。该错误发生在下载阶段，无须删除模型缓存或调整训练参数。完整新环境直接安装 `requirements.txt` 即可包含此依赖。

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
- [Transformers TensorBoard 回调](https://huggingface.co/docs/transformers/main_classes/callback#transformers.integrations.TensorBoardCallback)
- [TensorBoard 查看日志](https://www.tensorflow.org/tensorboard/get_started)
- [Brotli 1.2.0 的解压接口变更](https://github.com/google/brotli/releases/tag/v1.2.0)
- [PyTorch CUDA 显存分配设置](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
