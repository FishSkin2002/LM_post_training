# 大语言模型数学推理后训练

## 项目目录

```text
.
├── prepare_sft_data.py          # 检查 GSM8K 样本、聊天模板与损失标签
├── train_sft.py                 # 新训练、断点续训、完成后加训
├── sample_checkpoint.py         # 原始 Base 或 LoRA 产物抽样问答
├── training_workflow.py         # 数据划分、运行状态与生成验证
├── configs/
│   ├── development/sft.yaml     # 开发机：Qwen3-0.6B-Base
│   └── training/
│       ├── sft.yaml             # 默认训练机：Qwen3-4B-Base
│       └── sft-8b.yaml          # 可选：Qwen3-8B-Base
├── outputs/                    # 本地运行后生成的产物
│   ├── development/<run>/
│   │   ├── run.json            # 配置、数据划分、加训来源
│   │   ├── checkpoint-<step>/  # 完整训练断点，最多保留最近两个
│   │   ├── final/              # 适配器、分词器与运行信息
│   │   ├── runs/               # 本次运行的 TensorBoard 事件
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

输出完整聊天序列、参与损失的解答和 token 标签。`answer` 原文（含计算器注释与 `####`）及助手结束标记接受监督；题目、回答前缀和 padding 不计损失。

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

### sample_checkpoint.py

#### 查看未接受本项目训练的 Base 回答

```bash
python sample_checkpoint.py --model-name Qwen/Qwen3-4B-Base
```

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

也可填写本流程保存的 `checkpoint-*`。仅从本地目录和缓存加载，需保留分词器及对应 Base 权重；标准解答只写入报告，不输入模型。

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

训练指标默认每 10 步记录，验证和保存默认每 500 步进行。train 固定划出 10% 作验证（seed=42），test 留作最终评测；验证 loss 始终全量计算，定期生成固定 128 条，结束时生成完整验证集。

验证先显示完整验证集的 loss 进度，随后显示 `Generate validation answers (step ...)` 的生成进度。生成进度按完成的题目更新，并显示耗时和预计剩余时间；loss 的 `748/748` 不表示整次验证已结束。

| 曲线 | 含义 |
| --- | --- |
| `train/loss`、`train/grad_norm`、`train/learning_rate`、`train/epoch` | 区间平均训练损失、记录步的裁剪前梯度范数、学习率、训练轮次。 |
| `eval/loss` | 标准答案条件下的验证损失，不是数学答案正确率。 |
| `eval/answer_accuracy`、`eval/answer_format_rate`、`eval/generation_truncation_rate` | 固定生成子集的正确率、格式有效率、截断率。 |
| `eval/generation_samples`、`eval/generation_runtime`、`eval/generation_samples_per_second` | 生成评测实际题数、耗时、吞吐。 |
| `eval/full_*` | 训练结束时完整验证集的对应生成指标，与固定子集分开记录。 |
| `eval/runtime`、`eval/samples_per_second` | 验证阶段总耗时与原生吞吐记录。 |

生成正确率要求恰有一个 `####`、其后仅为最终数值；合法千位逗号和小数按精确数值比较，`5.0` 与 `5` 相等。格式不合格或截断计错，分母包括所有评测题。详细口径见 [训练运行约定](docs/training-workflow.md)。

### tools/test_base_sft.py

#### 离线验证训练运行功能

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python tools/test_base_sft.py
```

使用临时随机微型模型和合成样本，核对模板、续训、加训、日志、目录和生成判定；不加载真实 Base 权重。

可选参数：

| 参数 | 可填项／默认行为 | 用途 |
| --- | --- | --- |
| `-v` | 无值开关 | 显示逐项测试结果。 |
| `-q` | 无值开关 | 使用简略输出。 |
| `-k` | 测试名称子串或模式；默认运行全部 | 筛选测试。 |
