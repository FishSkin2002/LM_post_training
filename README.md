# 大语言模型数学推理后训练

## 项目目录

```text
.
├── prepare_sft_data.py          # 检查普通文本提示、答案去注释与损失标签
├── train_sft.py                 # 新训练、断点续训、完成后加训
├── evaluate_checkpoint.py       # 手动或训练结束后的生成指标评测
├── sample_checkpoint.py         # 原始 Base 或 LoRA 产物抽样问答
├── training_workflow.py         # 数据划分、运行状态与终端展示
├── configs/
│   ├── development/sft.yaml     # 开发机：Qwen3-0.6B-Base
│   └── training/
│       ├── sft.yaml             # 默认训练机：Qwen3-4B-Base
│       └── sft-8b.yaml          # 可选：Qwen3-8B-Base
├── outputs/                    # 本地运行后生成的产物
│   ├── development/<run>/
│   │   ├── run.json            # 配置、文本格式、数据划分、加训来源
│   │   ├── checkpoint-<step>/  # 完整训练断点，最多保留最近两个
│   │   ├── final/              # 适配器、分词器与运行信息
│   │   ├── runs/               # 本次运行的 TensorBoard 事件
│   │   ├── evaluations/        # 生成指标及逐题结果 JSON 报告
│   │   └── trainer_state.json  # Trainer 训练进度与日志
│   └── samples/                # 抽样问答 Markdown 报告
├── docs/                       # 项目目标、训练与协作约定
├── tools/
│   ├── test_base_sft.py         # 随机微型模型集成验证
│   └── git_workflow.py          # 受控 Git 工具及相应测试
├── requirements.txt            # Python 依赖
├── AGENTS.md                   # 协作规则与文档入口
├── CONTEXT.md                  # 领域术语
├── task_plan.md                # 当前计划
├── progress.md                 # 进展记录
└── findings.md                 # 环境与实现发现
```

训练机配置默认把产物写入 `/root/autodl-tmp/outputs/training/<run>/`，内部结构与上面的开发机运行目录相同。模型和数据的 Hugging Face 缓存位于仓库之外；适配器需要配合对应的 Base 权重使用。

新训练和加训遇到同名目录或文件时，自动尝试 `原名-1`、`原名-2`，启动时显示实际目录；断点续训沿用原运行目录。`checkpoint-*` 包含优化器、调度器和随机状态，可恢复训练；`final/` 用于抽样或开启加训阶段。

## 主要脚本使用说明

以下指令在项目根目录执行。先在相应 Python 环境运行 `python -m pip install -r requirements.txt`；示例中的 `/path/to/...` 替换为实际路径。

### prepare_sft_data.py

#### 检查一条训练样本及监督范围

```bash
python prepare_sft_data.py
```

输出完整的普通文本数学提示、参与损失的解答和 token 标签。输入结构为简短英文说明、`Question: 问题` 和 `Answer:`；说明用“最终答案为 4 时写成 `#### 4`”的例子约束答案位置，不要求公式标记。

`answer` 删除整段 `<<...>>` 计算器注释，例如 `2+3=<<2+3=5>>5` 变为 `2+3=5`，其余推导与 `####` 原样保留。清理后的解答和一个原生 `<|endoftext|>` 接受监督；说明、题目、回答前缀和 padding 不计损失。训练和验证 loss 使用相同处理，原始数据文件保留。

可选参数：

| 参数 | 可填项／默认行为 | 用途 |
| --- | --- | --- |
| `--config` | YAML 路径；默认 `configs/development/sft.yaml` | 选择模型对应的分词器，也可填写两份训练机配置或自定义配置。 |

### train_sft.py

#### 发起新训练

```bash
python train_sft.py
```

默认使用训练机 4B-Base 配置。配置中的模型、轮次、学习率、批量、LoRA 参数与保存/验证频率均可直接修改；模型保持 FP32，只训练 LoRA。

本版采用普通文本格式，从原始 Base 开始新训练。恢复、加训、抽样和独立评测的来源须为当前格式保存的产物；旧聊天格式 checkpoint 不属于本版来源范围。`run.json` 记录并校验提示、答案清理规则和 EOS；训练产物不依赖聊天模板文件。

训练过程只计算验证 loss。正常结束后先保存 `final/` 和训练状态，再自动计算一次完整验证集的生成指标，报告写入本次运行的 `evaluations/final-n748.json`（题数随验证集大小变化）。评测复用已有模型；若评测失败或中断，可用下面的独立脚本重新评测已保存的最终模型。

终端阶段说明使用英文，包括 `Training`、`Validation loss`、`Generate validation answers`。指标采用 `Training | loss=... | grad_norm=... | learning_rate=... | epoch=...`，TensorBoard 中仍保存原始数值。

可选参数：

| 参数 | 可填项／默认行为 | 用途 |
| --- | --- | --- |
| `--config` | 默认 `configs/training/sft.yaml`；也可填 `configs/development/sft.yaml`、`configs/training/sft-8b.yaml` 或自定义 YAML | 选择 0.6B / 4B / 8B 及训练设置。 |
| `--output-dir` | 目录路径；默认使用配置中的 `output_dir` | 指定本次实验目录，重名时自动添加数字。 |

#### 从断点恢复未完成的训练

```bash
python train_sft.py --resume-from-checkpoint /path/to/run/checkpoint-500
```

使用实际保留的、尚未完成原计划的 `checkpoint-*`。恢复模型、优化器、调度器、随机状态和进度，沿用原数据划分、训练计划及 TensorBoard 曲线；必要状态缺失会在加载模型前报错。

Ctrl+C 不会额外保存中断时刻的状态；续训从最近完整断点开始。已完成的训练使用独立评测脚本计算指标，或使用下一用途开启加训。

可选参数：

| 参数 | 可填项／默认行为 | 用途 |
| --- | --- | --- |
| `--config` | YAML 路径；默认 `configs/training/sft.yaml` | 必须与该阶段原配置一致；加训阶段也使用启动该阶段时的配置。 |
| `--output-dir` | 仅可填写 checkpoint 所属运行目录；默认自动使用该目录 | 通常不用填写，续训不分配新数字后缀。 |

#### 训练完成后再加训

```bash
python train_sft.py --continue-from /path/to/previous/final --additional-epochs 1
```

从已有适配器权重开始新的阶段，再训练一轮；新建优化器、学习率调度、计步状态、输出目录和 TensorBoard 记录。来源需包含当前流程保存的 `run.json`，沿用其验证划分，不叠加第二套 LoRA。加训阶段中断后，也用上一用途的断点恢复命令。

可选参数：

| 参数 | 可填项／默认行为 | 用途 |
| --- | --- | --- |
| `--config` | YAML 路径；默认 `configs/training/sft.yaml` | Base 和 LoRA 结构需匹配来源；新阶段学习率等设置取自这份配置，不自动降低学习率。 |
| `--output-dir` | 目录路径；默认使用配置的输出目录并自动避重 | 选择新阶段的独立目录。 |

`--additional-epochs` 为该用途必填的正数，可填小数；它表示新阶段的追加轮次，不是累计总轮次。`--continue-from` 与 `--resume-from-checkpoint` 互斥。

### evaluate_checkpoint.py

#### 评测一个 checkpoint 或 final 的完整验证集

```bash
python evaluate_checkpoint.py --checkpoint /path/to/run/final
```

也可填写训练中保存的 `checkpoint-500` 等目录。读取该产物的 `run.json`，校验数据并沿用原来的验证划分；不更新权重、优化器或训练步数。未指定题数时评测完整验证集，默认划分约 748 题，不使用 test。

可选参数：

| 参数 | 可填项／默认行为 | 用途 |
| --- | --- | --- |
| `--num-samples` | `1` 到验证集题数的整数；默认全量 | 限制评测题数，超出范围会报错。 |
| `--device` | `cpu` 或 `cuda`；默认自动选择 | 指定 FP32 推理设备。 |
| `--output` | JSON 文件路径；默认 `<run>/evaluations/<checkpoint>-n<N>.json` | 指定报告位置；相同路径的已有报告会被覆盖。 |

#### 用相同题目子集比较训练断点

```bash
python evaluate_checkpoint.py --checkpoint /path/to/run/checkpoint-1000 --num-samples 32
```

替换 checkpoint 路径并保持相同题数，可比较不同训练阶段。使用验证集固定顺序的前 N 题，`32` 可替换为合法题数；分母和实际索引随报告保存。小样本结果用于观察趋势。

可选参数：

| 参数 | 可填项／默认行为 | 用途 |
| --- | --- | --- |
| `--device` | `cpu` 或 `cuda`；默认自动选择 | 指定 FP32 推理设备。 |
| `--output` | JSON 文件路径；默认 `<run>/evaluations/<checkpoint>-n<N>.json` | 保存本次比较报告；相同路径会被覆盖。 |

两种用途均只从本地 checkpoint 和模型缓存加载，需保留分词器与对应 Base 权重。生成使用贪心解码、1024-token 上限和与训练共用的普通文本提示；输入止于 `Answer:` 后的换行，不含标准解答，输出原生 EOS 时结束。

报告包含生成正确率、格式有效率、截断率、题数、耗时、吞吐、来源、文本格式、题目索引和逐题生成结果。恰有一个 `####` 且其后仅为最终数值才算格式有效；合法千位逗号和小数按精确数值比较，`5.0` 与 `5` 相等。格式不合格或截断计错，分母包括所有评测题；不提取或验算公式。详细口径见 [训练运行约定](docs/training-workflow.md)。

### sample_checkpoint.py

#### 查看未接受本项目训练的 Base 回答

```bash
python sample_checkpoint.py --model-name Qwen/Qwen3-4B-Base
```

原始 Base 与 SFT 使用相同的任务说明和 Question/Answer 前缀；报告中的标准解答同样去除计算器注释。

可选参数：

| 参数 | 可填项／默认行为 | 用途 |
| --- | --- | --- |
| `--num-samples` | 正整数；默认 `5` | 抽题数量。 |
| `--split` | `train` 或 `test`；默认 `test` | 选择抽样数据来源。 |
| `--seed` | 整数；默认 `42` | 固定抽题，便于对照。 |
| `--max-new-tokens` | 正整数；默认 `1024` | 贪心生成的最大输出长度。 |
| `--local-files-only` | 无值开关；默认允许下载缺失模型文件 | 仅使用已有模型缓存。 |
| `--device` | `cpu` 或 `cuda`；不填时自动选择 | 指定推理设备，精度均为 FP32。 |
| `--output` | Markdown 文件路径；默认写入 `outputs/samples/` | 指定报告位置，同名报告会被覆盖。 |

#### 查看训练产物的回答

```bash
python sample_checkpoint.py --checkpoint /path/to/run/final
```

也可填写本流程保存的 `checkpoint-*`。仅从本地目录和缓存加载，需保留分词器、run.json 及对应 Base 权重；加载前校验当前文本格式。标准解答只写入报告，不输入模型。

可选参数：

| 参数 | 可填项／默认行为 | 用途 |
| --- | --- | --- |
| `--num-samples` | 正整数；默认 `5` | 抽题数量。 |
| `--split` | `train` 或 `test`；默认 `test` | 选择抽样数据来源。 |
| `--seed` | 整数；默认 `42` | 与 Base 抽样使用相同题目。 |
| `--max-new-tokens` | 正整数；默认 `1024` | 设置输出预算，达到上限时报告截断。 |
| `--local-files-only` | 无值开关；本用途始终仅缓存加载 | 可省略，填写不会改变加载方式。 |
| `--device` | `cpu` 或 `cuda`；不填时自动选择 | 指定 FP32 推理设备。 |
| `--output` | Markdown 文件路径；默认写入 `outputs/samples/` | 指定报告位置，同名报告会被覆盖。 |

这个用途已固定仅缓存加载，无需填写 `--local-files-only`。少量问答用于观察输出，不等于准确率评测。模型与数据均已缓存时，可在命令前设置 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` 离线运行。

### TensorBoard

#### 查看训练与验证曲线

```bash
tensorboard --logdir /root/autodl-tmp/outputs/training
```

填训练产物的根目录可比较多个实验；开发机可改填 `outputs/development`，也可填某次运行的 `runs/` 只查看该次记录。

常用可选参数：

| 参数 | 可填项／默认行为 | 用途 |
| --- | --- | --- |
| `--port` | 可用端口号；默认 `6006` | 选择 TensorBoard 服务端口。 |
| `--host` | 监听地址，如 `127.0.0.1`、`0.0.0.0`；默认由 TensorBoard 选择本地地址 | 指定服务监听地址。 |

训练指标默认每 10 步记录，验证 loss 和保存默认每 500 步进行。train 固定划出 10% 作验证（seed=42），test 留作最终评测；验证 loss 始终全量计算。训练期间不监控生成正确率，生成结果写入独立评测报告。

| 曲线 | 含义 |
| --- | --- |
| `train/loss`、`train/grad_norm`、`train/learning_rate`、`train/epoch` | 区间平均训练损失、记录步的裁剪前梯度范数、学习率、训练轮次。 |
| `eval/loss` | 标准答案条件下的验证损失，不是数学答案正确率。 |
| `eval/runtime`、`eval/samples_per_second` | 验证 loss 的耗时与吞吐记录，不包含结束后独立生成评测。 |

### tools/test_base_sft.py

#### 离线验证训练运行功能

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python tools/test_base_sft.py
```

使用临时随机微型模型和合成样本，核对普通文本提示、去注释、EOS 监督、续训、加训、日志、目录、独立评测及结束调用；不加载真实 Base 权重。

可选参数：

| 参数 | 可填项／默认行为 | 用途 |
| --- | --- | --- |
| `-v` | 无值开关 | 显示逐项测试结果。 |
| `-q` | 无值开关 | 使用简略输出。 |
| `-k` | 测试名称子串或模式；默认运行全部 | 筛选测试。 |
