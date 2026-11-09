# 大语言模型数学推理后训练

以学习为目的的大语言模型数学推理后训练项目，主要学习 LoRA SFT、DPO、GRPO。

> **实验报告：[查看目录](reports/)**
>
> [实验一：GSM8K 上的 Qwen3-4B-Base LoRA SFT](reports/exp1/report.md)

## 项目目录

```text
.
├── train.py                     # 训练入口，选择 SFT、DPO 或 GRPO
├── evaluate_checkpoint.py       # 生成验证集回答，完成一二级判分
├── summarize_evaluation.py     # 可选 API 三级判定、正确率汇总与阅读报告
├── configs/                     # 训练、开发、大模型判分和公共生成配置
├── outputs/                     # 运行生成的训练与评测产物
├── utils/                       # 训练、数据准备、生成与评分的实现
├── tools/                       # 开发工具与测试脚本
├── legacy/                      # 读取旧 SFT 数据所需的兼容程序
├── docs/                        # 项目目标、设计约定与研究记录
├── reports/                     # 实验报告及其结果文件
├── requirements.txt             # Python 环境依赖
├── README.md                    # 项目目录与脚本用法
├── .gitignore                   # 不纳入版本管理的文件规则
├── AGENTS.md                    # Codex 在本项目中的协作规则
├── CONTEXT.md                   # 项目术语及定义
├── task_plan.md                 # 当前任务的步骤与完成状态
├── progress.md                  # 每轮工作与验证结果
└── findings.md                  # 已确认的环境信息与技术发现
```

### configs 目录

```text
configs/                         # 训练、开发、大模型判分和公共生成配置
├── training/                    # 正式训练配置
├── development/                 # 开发与数据检查配置
└── evaluation/                  # 大模型判分配置
```

### outputs 目录

以下按当前保存逻辑，以训练机上的一次 SFT 运行为例；输出位置由配置决定。

```text
/root/autodl-tmp/outputs/         # 训练机产物目录
└── training/                    # 正式训练产物
    └── qwen3-4b-instruct-openr1-lora/  # 一次 SFT 运行
        ├── run.json                     # 模型、配置、数据划分与运行来源
        ├── trainer_state.json           # 训练结束时的进度与日志记录
        ├── checkpoint-<step>/           # 训练中保存的完整断点
        ├── final/                       # 正常训练结束后保存的模型产物
        │   ├── README.md                # 适配器模型说明
        │   ├── adapter_config.json      # LoRA 结构与底座信息
        │   ├── adapter_model.safetensors # LoRA 权重
        │   ├── tokenizer.json           # 完整分词器
        │   ├── tokenizer_config.json    # 分词器设置
        │   ├── special_tokens_map.json  # 特殊词元映射
        │   ├── added_tokens.json        # 新增词元及其 ID（按需保存）
        │   ├── vocab.json               # 词表（部分版本保存）
        │   ├── merges.txt               # 分词合并规则（部分版本保存）
        │   ├── chat_template.jinja      # 聊天模板
        │   ├── training_args.bin        # 本次训练参数
        │   └── run.json                 # 最终产物对应的运行信息
        ├── runs/                        # TensorBoard 日志
        │   └── events.out.tfevents.<标识> # 训练与验证曲线的事件文件
        ├── evaluations/                 # 生成及一二级评测
        │   └── <评测名>/
        │       ├── metadata.json         # 来源、共享快照引用、配置与环境
        │       ├── answers.jsonl         # 模型回答与每题生成数据
        │       ├── judgments.jsonl       # 各级判决与 API 调用记录，用于续跑
        │       ├── scorings/             # 各次独立评分
        │       │   └── scoring/          # 重名时加 -1、-2 等后缀
        │       │       └── metadata.json # 回答来源与本次裁判设置
        │       ├── answers-0001.md       # 分卷问答及三阶段结果，汇总后生成
        │       ├── api-decided.md        # API 判对/判错、原因与原题链接
        │       └── api-pending.md        # API 待复核原因与原题链接
        ├── prepared-sft/                # 读取旧 SFT 产物时自动准备的数据缓存
        │   ├── manifest.json            # 数据来源、准备配置与校验信息
        │   └── dataset/                 # 已准备的数据集
        │       ├── data-00000-of-00001.arrow # 数据内容，较大时分为多个文件
        │       ├── dataset_info.json    # 数据集字段与元信息
        │       └── state.json           # 数据集文件清单与读取状态
        └── .sft-data-migration.lock      # 准备旧 SFT 数据时使用的并发锁
```

`<step>` 是断点步数，默认最多保留最近两个断点。分词器文件随保存版本有所不同。`<评测名>` 默认是 `final-OpenR1-Math-220k-default-train-n<N>`，`<N>` 是评测题数。`scorings/` 与 Markdown 在训练结束自动汇总或单独运行汇总入口后生成；问答默认每卷最多 100 题，总指标只在第一卷。`prepared-sft/` 和锁文件仅在需要准备旧 SFT 数据时生成。

原始模型的评测默认保存在仓库的 `outputs/evaluations/`；题目与参考解答统一保存在仓库的 `outputs/evaluation-datasets/<快照标识>.json`，相同评测样本供各模型共用。开发输出由开发配置指定。新训练和加训遇到重名运行目录时自动加 `-1`、`-2` 等后缀，断点续训沿用原目录。

## 聊天格式

```text
<|im_start|>system
Solve the following math problem step by step.
End your response with the final answer in \boxed{...}, for example \boxed{4}.<|im_end|>
<|im_start|>user
{题目原文}<|im_end|>
<|im_start|>assistant
{完整解答}<|im_end|>
```

## 回答正确性判定

按以下三个层次判断模型回答是否正确。前一级未确认正确的回答才进入下一级；生成截断直接判错，不进入后两级。

1. **一级：参考答案核验。** 提取模型回答中最后一个 `\boxed{...}`，与题目提供的参考最终答案比较数学等价性。
2. **二级：R1 解答核验。** 比较模型回答与所选 R1 参考解答在正式作答部分的单个 `\boxed{...}`，判断答案是否数学等价。
3. **三级：大模型判分。** 由大模型结合题目、参考信息和模型完整回答，判断最终答案是否完整且正确；无法明确判断则标记为待复核。

## 环境准备

在支持 BF16 的 CUDA 设备上创建环境并安装依赖，将两处 `<环境名>` 替换为相同的自选名称。

```bash
conda create -n <环境名> python=3.12 pip -y
conda activate <环境名>
python -m pip install -r requirements.txt
```

默认生成引擎为 vLLM，当前依赖使用 CUDA 12.8，宿主 NVIDIA 驱动需至少 570.26。

## 主要脚本使用说明

以下按用途给出填好必需参数的示例；将 `/path/to/...` 替换为实际路径。全部参数的默认值、可填内容及作用见 [脚本参数参考](docs/script-arguments.md)。

### train.py

执行 SFT、DPO 或 GRPO 训练；正常结束保存模型，并自动完成验证集生成评测、大模型判分及报告汇总。

#### SFT：发起新训练

从原始 Instruct 模型开始训练。

```bash
python train.py sft
```

#### SFT：恢复未完成的训练

从完整断点恢复原训练进度，沿用原运行目录。

```bash
python train.py sft --resume-from-checkpoint /path/to/sft-run/checkpoint-5000
```

#### SFT：完成后加训

从已有 SFT 权重开启新运行，示例追加训练 1 轮。

```bash
python train.py sft --continue-from /path/to/sft-run/final --additional-epochs 1
```

#### DPO：从 SFT 开始训练

使用已完成的 SFT `final/` 开始独立的 DPO 训练。

```bash
python train.py dpo --sft-source /path/to/sft-run/final
```

#### DPO：恢复未完成的训练

恢复 DPO 完整断点；来源 SFT 产物需保留在原路径。

```bash
python train.py dpo --resume-from-checkpoint /path/to/dpo-run/checkpoint-100
```

#### GRPO：从 SFT 开始训练

使用已完成的 SFT `final/` 开始独立的 GRPO 训练。

```bash
python train.py grpo --sft-source /path/to/sft-run/final
```

#### GRPO：恢复未完成的训练

恢复 GRPO 完整断点；来源 SFT 产物需保留在原路径。

```bash
python train.py grpo --resume-from-checkpoint /path/to/grpo-run/checkpoint-100
```

### evaluate_checkpoint.py

生成验证集回答并完成一二级判分；中断后重跑相同命令会继续已有进度。

#### 评测原始模型

使用默认评测配置，评测原始 Instruct 模型。

```bash
python evaluate_checkpoint.py --model-name Qwen/Qwen3-4B-Instruct-2507
```

#### 评测训练产物

评测 SFT、DPO 或 GRPO 的 `final/` 或 `checkpoint-<step>/`，自动读取保存的评测配置与划分。

```bash
python evaluate_checkpoint.py --checkpoint /path/to/run/final
```

### summarize_evaluation.py

汇总已有回答，默认通过 API 补做三级判定，正常结束后生成问答分卷与核查清单。

#### 新建评分并汇总

选择一二级评测目录，创建独立评分；终端打印本次评分目录。

```bash
python summarize_evaluation.py --evaluation /path/to/evaluation
```

#### 继续评分

使用已有评分目录，沿用原裁判设置和已保存判决，继续处理未完成的题。

```bash
python summarize_evaluation.py --resume-from /path/to/evaluation/scorings/scoring
```

#### 重试待复核题

继续已有评分，同时对待复核题重新请求 API。

```bash
python summarize_evaluation.py --resume-from /path/to/evaluation/scorings/scoring --retry-pending
```

#### 只查看已有结果

汇总所选评分已有的完整或部分结果，不请求 API。

```bash
python summarize_evaluation.py --resume-from /path/to/evaluation/scorings/scoring --no-api
```
