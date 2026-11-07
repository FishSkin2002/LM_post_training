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
├── rescore_evaluation.py        # 调用在线大模型，为已保存的回答判分
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
        ├── evaluations/                 # 生成评测与大模型判分结果
        │   ├── <评测名>.json            # 评测信息、完整回答与一二级判决
        │   ├── <评测名>.progress.jsonl  # 逐题回答与一级判决，用于续跑
        │   ├── <评测名>.md              # 题目、参考解答与模型回答的阅读版
        │   ├── <评测名>.stage2.progress.jsonl # 二级判决，用于续跑
        │   ├── <评测名>.stage2.md       # 二级判决的阅读版
        │   ├── <评测名>-rescored.json   # 大模型判分结果与统计
        │   ├── <评测名>-rescored.progress.jsonl # 大模型判分进度
        │   ├── <评测名>-rescored.md     # 大模型判分结果的阅读版
        │   └── <评测名>-rescored.api-review.md # 判分依据与待复核题目
        ├── prepared-sft/                # 读取旧 SFT 产物时自动准备的数据缓存
        │   ├── manifest.json            # 数据来源、准备配置与校验信息
        │   └── dataset/                 # 已准备的数据集
        │       ├── data-00000-of-00001.arrow # 数据内容，较大时分为多个文件
        │       ├── dataset_info.json    # 数据集字段与元信息
        │       └── state.json           # 数据集文件清单与读取状态
        └── .sft-data-migration.lock      # 准备旧 SFT 数据时使用的并发锁
```

`<step>` 是断点步数，默认最多保留最近两个断点。分词器文件随保存版本有所不同。`<评测名>` 默认是 `final-OpenR1-Math-220k-default-train-n<N>`，`<N>` 是评测题数；`-rescored` 文件在运行大模型判分后生成，`prepared-sft/` 和锁文件仅在需要准备旧 SFT 数据时生成。

原始模型的评测默认保存在仓库的 `outputs/evaluations/`；开发输出由开发配置指定。新训练和加训遇到重名运行目录时自动加 `-1`、`-2` 等后缀，断点续训沿用原目录。

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

### train.py

执行 SFT、DPO 或 GRPO 训练，支持各自的断点恢复。

首次读取旧 SFT 产物且没有可用数据缓存时，会自动创建 CPU 兼容环境并准备数据，后续复用缓存。各方法正常结束后保存 `final/` 并自动执行全量验证集一二级评测。

#### SFT：发起新训练

```bash
python train.py sft
```

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/training/sft.yaml` | 选择模型、数据与训练设置。 |
| `--output-dir` | 配置中的输出目录 | 指定新运行目录，重名自动加后缀。 |
| `--generation-config` | `configs/generation.yaml` | 设置训练结束后的生成评测。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可填 `vllm` 或 `transformers`。 |
| `--prepared-sft-data` | 按来源自动准备或复用 | 指定已有数据缓存目录。 |

#### SFT：恢复未完成的训练

```bash
python train.py sft --resume-from-checkpoint /path/to/sft-run/checkpoint-5000
```

将路径替换为完整断点，恢复原训练进度；Ctrl+C 后从最近保存的断点继续。

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/training/sft.yaml` | 选择与断点一致的训练配置。 |
| `--output-dir` | 断点所属运行目录 | 显式填写时须为同一目录。 |
| `--generation-config` | `configs/generation.yaml` | 设置训练结束后的生成评测。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可填 `vllm` 或 `transformers`。 |
| `--prepared-sft-data` | 按来源自动准备或复用 | 指定来源匹配的数据缓存。 |

#### SFT：完成后加训

```bash
python train.py sft --continue-from /path/to/sft-run/final --additional-epochs 1
```

将路径替换为已完成 SFT 的 `final/`；`--additional-epochs` 为追加轮数，可填正小数。加训创建新运行。

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/training/sft.yaml` | 设置本次加训，保持来源所需配置一致。 |
| `--output-dir` | 配置中的输出目录，重名加后缀 | 指定新的加训目录。 |
| `--generation-config` | `configs/generation.yaml` | 设置加训结束后的生成评测。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可填 `vllm` 或 `transformers`。 |
| `--prepared-sft-data` | 按来源自动准备或复用 | 指定来源匹配的数据缓存。 |

#### DPO：从 SFT 开始训练

```bash
python train.py dpo --sft-source /path/to/sft-run/final
```

将路径替换为已完成 SFT 的 `final/`，新建一次 DPO 训练。

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/training/dpo.yaml` | 选择偏好数据与 DPO 训练设置。 |
| `--output-dir` | DPO 配置中的输出目录 | 指定新运行目录，重名自动加后缀。 |
| `--generation-config` | `configs/generation.yaml` | 设置 DPO 结束后的生成评测。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可填 `vllm` 或 `transformers`。 |
| `--prepared-sft-data` | 按 SFT 来源自动准备或复用 | 指定源 SFT 的数据缓存。 |

#### DPO：恢复未完成的训练

```bash
python train.py dpo --resume-from-checkpoint /path/to/dpo-run/checkpoint-100
```

将路径替换为 DPO 的完整断点；原 SFT `final/` 需保留在保存时的路径。

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/training/dpo.yaml` | 选择与 DPO 断点一致的配置。 |
| `--output-dir` | 断点所属运行目录 | 显式填写时须为同一目录。 |
| `--generation-config` | `configs/generation.yaml` | 设置 DPO 结束后的生成评测。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可填 `vllm` 或 `transformers`。 |
| `--prepared-sft-data` | 按 SFT 来源自动准备或复用 | 指定源 SFT 的数据缓存。 |

#### GRPO：从 SFT 开始训练

```bash
python train.py grpo --sft-source /path/to/sft-run/final
```

将路径替换为已完成 SFT 的 `final/`，新建一次 GRPO 训练。

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/training/grpo.yaml` | 选择数学题数据与 GRPO 训练设置。 |
| `--output-dir` | GRPO 配置中的输出目录 | 指定新运行目录，重名自动加后缀。 |
| `--generation-config` | `configs/generation.yaml` | 设置 GRPO 结束后的生成评测。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可填 `vllm` 或 `transformers`。 |
| `--prepared-sft-data` | 按 SFT 来源自动准备或复用 | 指定源 SFT 的数据缓存。 |

#### GRPO：恢复未完成的训练

```bash
python train.py grpo --resume-from-checkpoint /path/to/grpo-run/checkpoint-100
```

将路径替换为 GRPO 的完整断点；原 SFT `final/` 需保留在保存时的路径。

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/training/grpo.yaml` | 选择与 GRPO 断点一致的配置。 |
| `--output-dir` | 断点所属运行目录 | 显式填写时须为同一目录。 |
| `--generation-config` | `configs/generation.yaml` | 设置 GRPO 结束后的生成评测。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可填 `vllm` 或 `transformers`。 |
| `--prepared-sft-data` | 按 SFT 来源自动准备或复用 | 指定源 SFT 的数据缓存。 |

### evaluate_checkpoint.py

生成原始模型或训练产物的验证集回答，并完成一二级判分。

首次读取旧 SFT 产物且缺少数据缓存时，会自动创建 CPU 兼容环境并准备数据。中断后重跑相同命令会继续已有进度；需要从头评测时用 `--output` 指定新路径。

#### 评测原始模型

```bash
python evaluate_checkpoint.py --model-name Qwen/Qwen3-4B-Instruct-2507
```

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/training/sft.yaml` | 设置评测数据与生成预算。 |
| `--sft-source` | 使用配置确定的划分 | 指向 SFT `final/`，使用其保存的评测配置与划分；与 `--config` 互斥。 |
| `--num-samples` | 完整验证集 | 评测固定顺序的前 N 题，须为不超过验证集大小的正整数。 |
| `--batch-size` | vLLM：`64`；Transformers：`4` | 每次提交的最大题数，须为正整数；Transformers 自适应模式下为起始题数。 |
| `--auto-batch-size` | 固定批大小 | 仅限 Transformers：成功时增大批次，生成失败后缩小重试；与 vLLM 同用会报错。 |
| `--device` | 自动选择 CUDA 或 CPU | 可填 `cuda` 或 `cpu`，须满足配置的精度要求。 |
| `--output` | `outputs/evaluations/` 下自动命名 | 指定报告 JSON 路径，同名保存逐题进度和阅读版。 |
| `--generation-config` | `configs/generation.yaml` | 选择生成配置。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可填 `vllm` 或 `transformers`。 |
| `--prepared-sft-data` | 按来源自动准备或复用 | 指定来源匹配的数据缓存。 |

#### 评测训练产物

```bash
python evaluate_checkpoint.py --checkpoint /path/to/run/final
```

将路径替换为 SFT、DPO 或 GRPO 的 `final/` 或 `checkpoint-<step>/`。省略题数时评测完整验证集。

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--num-samples` | 完整验证集 | 评测固定顺序的前 N 题，须为不超过验证集大小的正整数。 |
| `--batch-size` | vLLM：`64`；Transformers：`4` | 每次提交的最大题数，须为正整数；Transformers 自适应模式下为起始题数。 |
| `--auto-batch-size` | 固定批大小 | 仅限 Transformers：成功时增大批次，生成失败后缩小重试；与 vLLM 同用会报错。 |
| `--device` | 自动选择 CUDA 或 CPU | 可填 `cuda` 或 `cpu`，须满足来源精度要求。 |
| `--output` | 运行目录的 `evaluations/` 下自动命名 | 指定报告 JSON 路径，同名保存逐题进度和阅读版。 |
| `--generation-config` | `configs/generation.yaml` | 选择生成配置。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可填 `vllm` 或 `transformers`。 |
| `--prepared-sft-data` | 按来源自动准备或复用 | 指定来源匹配的数据缓存。 |

### rescore_evaluation.py

通过 API 调用在线大模型，对已保存的 OpenR1 评测回答进行判分。

#### 准备 API 密钥

在运行判分命令的终端中输入模型服务所需的密钥：

```bash
read -rsp 'DeepSeek API Key: ' DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
```

#### 使用大模型判分

```bash
python rescore_evaluation.py --input /path/to/evaluation.json --output /path/to/evaluation-rescored.json
```

将 `--input` 替换为已完成一二级评测的 JSON 报告，将 `--output` 替换为新的报告路径。

默认使用 DeepSeek Flash，关闭思考模式，仅核对最终答案；完整提示词见 [JUDGE_PROMPT](utils/answer_judge.py)。更换裁判模型或提示词后，使用新的 `--output` 路径开始判分。

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/evaluation/deepseek.yaml` | 选择模型服务地址、模型与请求设置。 |
| `--retry-pending` | 复用已保存的待复核判决 | 重新尝试待复核题。 |

#### 继续判分或重试待复核题

```bash
python rescore_evaluation.py --input /path/to/evaluation.json --output /path/to/evaluation-rescored.json --retry-pending
```

沿用原来的输入和输出路径；不加 `--retry-pending` 时仅继续未处理题，加上后也重试待复核题。缺少密钥时，需要大模型判分的题会记为待复核。

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/evaluation/deepseek.yaml` | 沿用首次判分的模型服务配置。 |
