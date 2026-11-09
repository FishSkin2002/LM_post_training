# 脚本参数参考

命令示例见 [README 的主要脚本使用说明](../README.md#主要脚本使用说明)。以下按脚本和用途列出必需参数及全部适用的可选参数；已经由用途确定的互斥来源在各自用途下说明。

“默认值”说明省略参数时的实际取值或行为；“无；本用途必填”表示需要显式填写。开关参数直接写参数名，不再填写值。表中的配置文件路径相对于仓库展示，自选配置可改变由配置决定的默认值。

所有脚本及 `train.py` 的子命令都支持：

| 参数 | 默认值 | 可填内容及作用 |
| --- | --- | --- |
| `-h, --help` | 不启用 | 开关；显示当前脚本或训练子命令的参数帮助后退出。 |

## train.py

先选择 `sft`、`dpo` 或 `grpo` 子命令。训练来源参数互斥；各用途所需来源已在对应表中列出。

正常结束先保存 `final/` 和训练状态，再自动执行完整验证集的一二级评测、API 三级判定及报告汇总。需要 API 时读取 `DEEPSEEK_API_KEY`，未设置则在终端隐藏输入。训练中断不触发结束评测；后续评测失败不影响已保存的训练产物。

首次读取需要兼容处理的旧 SFT 来源且没有可用缓存时，当前程序会自动创建 CPU 兼容环境并准备数据；后续复用缓存。`--prepared-sft-data` 指定的是已筛选、分词的 SFT 数据；DPO、GRPO 用它恢复共同的 SFT 验证数据，不用它替代各自的训练数据。

### SFT：发起新训练

从原始 Instruct 模型开始；没有额外必填选项。

| 参数 | 默认值 | 可填内容及作用 |
| --- | --- | --- |
| `--config` | `configs/training/sft.yaml` | 训练配置文件路径；选择模型、数据和训练设置。 |
| `--output-dir` | 配置中的 `output_dir` | 目录路径；指定本次 SFT 的新运行目录，重名自动加 `-1`、`-2` 等后缀。 |
| `--generation-config` | `configs/generation.yaml` | 生成配置文件路径；设置训练结束后的生成评测。 |
| `--generation-backend` | `vllm`（由生成配置决定） | `vllm` 或 `transformers`；选择训练结束后评测使用的生成引擎。 |
| `--prepared-sft-data` | 未指定；使用当前环境的数据准备与缓存 | 已准备的 SFT 数据目录，包含 `manifest.json` 和 `dataset/`；显式指定时须与本次数据及分词设置一致。 |

### SFT：恢复未完成的训练

恢复原来的训练状态与计划，沿用运行目录和日志；Ctrl+C 后可选择最近保存的完整断点。

| 参数 | 默认值 | 可填内容及作用 |
| --- | --- | --- |
| `--resume-from-checkpoint` | 无；本用途必填 | SFT 的完整 `checkpoint-<step>/` 目录；恢复模型、优化器、调度器及训练进度。 |
| `--config` | `configs/training/sft.yaml` | 训练配置文件路径；必须与断点保存的训练配置一致。 |
| `--output-dir` | 断点所属运行目录 | 目录路径；显式填写时必须仍为断点所属运行目录。 |
| `--generation-config` | `configs/generation.yaml` | 生成配置文件路径；设置训练结束后的生成评测。 |
| `--generation-backend` | `vllm`（由生成配置决定） | `vllm` 或 `transformers`；选择训练结束后评测使用的生成引擎。 |
| `--prepared-sft-data` | 未指定；按来源自动定位缓存，必要时准备 | 已准备的 SFT 数据目录，包含 `manifest.json` 和 `dataset/`；显式指定时须与来源数据及分词设置一致。 |

### SFT：完成后加训

从已有 SFT 权重开始新运行，重新计算本阶段步数；学习率等使用本次配置。

| 参数 | 默认值 | 可填内容及作用 |
| --- | --- | --- |
| `--continue-from` | 无；本用途必填 | 已保存的 SFT 适配器目录，通常为 `final/`；选择本次加训的初始权重。 |
| `--additional-epochs` | 无；本用途必填 | 有限正数，可填 `1`、`0.5` 等；指定本次新阶段追加的训练轮数，须与 `--continue-from` 同时填写。 |
| `--config` | `configs/training/sft.yaml` | 训练配置文件路径；设置本次加训，模型、数据、分词和 LoRA 等须与来源一致。 |
| `--output-dir` | 配置中的 `output_dir` | 目录路径；指定本次 SFT 加训的新运行目录，重名自动加 `-1`、`-2` 等后缀。 |
| `--generation-config` | `configs/generation.yaml` | 生成配置文件路径；设置训练结束后的生成评测。 |
| `--generation-backend` | `vllm`（由生成配置决定） | `vllm` 或 `transformers`；选择训练结束后评测使用的生成引擎。 |
| `--prepared-sft-data` | 未指定；按来源自动定位缓存，必要时准备 | 已准备的 SFT 数据目录，包含 `manifest.json` 和 `dataset/`；显式指定时须与来源数据及分词设置一致。 |

### DPO：从 SFT 开始训练

从已完成的 SFT `final/` 开始独立的 DPO 训练。

| 参数 | 默认值 | 可填内容及作用 |
| --- | --- | --- |
| `--sft-source` | 无；本用途必填 | 已完成 SFT 的 `final/` 目录；读取初始模型、分词器及共同评测来源。 |
| `--config` | `configs/training/dpo.yaml` | 训练配置文件路径；选择偏好数据与 DPO 训练设置。 |
| `--output-dir` | 配置中的 `output_dir` | 目录路径；指定本次 DPO 的新运行目录，重名自动加 `-1`、`-2` 等后缀。 |
| `--generation-config` | `configs/generation.yaml` | 生成配置文件路径；设置训练结束后的生成评测。 |
| `--generation-backend` | `vllm`（由生成配置决定） | `vllm` 或 `transformers`；选择训练结束后评测使用的生成引擎。 |
| `--prepared-sft-data` | 未指定；按来源自动定位缓存，必要时准备 | 已准备的 SFT 数据目录，包含 `manifest.json` 和 `dataset/`；显式指定时须与来源数据及分词设置一致。 |

### DPO：恢复未完成的训练

恢复 DPO 断点；来源 SFT 的 `final/` 仍需保留在运行记录保存的路径。

| 参数 | 默认值 | 可填内容及作用 |
| --- | --- | --- |
| `--resume-from-checkpoint` | 无；本用途必填 | DPO 的完整 `checkpoint-<step>/` 目录；恢复该阶段的训练状态与进度。 |
| `--config` | `configs/training/dpo.yaml` | 训练配置文件路径；必须与 DPO 断点保存的配置一致。 |
| `--output-dir` | 断点所属运行目录 | 目录路径；显式填写时必须仍为断点所属运行目录。 |
| `--generation-config` | `configs/generation.yaml` | 生成配置文件路径；设置训练结束后的生成评测。 |
| `--generation-backend` | `vllm`（由生成配置决定） | `vllm` 或 `transformers`；选择训练结束后评测使用的生成引擎。 |
| `--prepared-sft-data` | 未指定；按来源自动定位缓存，必要时准备 | 已准备的 SFT 数据目录，包含 `manifest.json` 和 `dataset/`；显式指定时须与来源数据及分词设置一致。 |

### GRPO：从 SFT 开始训练

从已完成的 SFT `final/` 开始独立的 GRPO 训练。

| 参数 | 默认值 | 可填内容及作用 |
| --- | --- | --- |
| `--sft-source` | 无；本用途必填 | 已完成 SFT 的 `final/` 目录；读取初始模型、分词器及共同评测来源。 |
| `--config` | `configs/training/grpo.yaml` | 训练配置文件路径；选择数学题数据与 GRPO 训练设置。 |
| `--output-dir` | 配置中的 `output_dir` | 目录路径；指定本次 GRPO 的新运行目录，重名自动加 `-1`、`-2` 等后缀。 |
| `--generation-config` | `configs/generation.yaml` | 生成配置文件路径；设置训练结束后的生成评测。 |
| `--generation-backend` | `vllm`（由生成配置决定） | `vllm` 或 `transformers`；选择训练结束后评测使用的生成引擎。 |
| `--prepared-sft-data` | 未指定；按来源自动定位缓存，必要时准备 | 已准备的 SFT 数据目录，包含 `manifest.json` 和 `dataset/`；显式指定时须与来源数据及分词设置一致。 |

### GRPO：恢复未完成的训练

恢复 GRPO 断点；来源 SFT 的 `final/` 仍需保留在运行记录保存的路径。

| 参数 | 默认值 | 可填内容及作用 |
| --- | --- | --- |
| `--resume-from-checkpoint` | 无；本用途必填 | GRPO 的完整 `checkpoint-<step>/` 目录；恢复该阶段的训练状态与进度。 |
| `--config` | `configs/training/grpo.yaml` | 训练配置文件路径；必须与 GRPO 断点保存的配置一致。 |
| `--output-dir` | 断点所属运行目录 | 目录路径；显式填写时必须仍为断点所属运行目录。 |
| `--generation-config` | `configs/generation.yaml` | 生成配置文件路径；设置训练结束后的生成评测。 |
| `--generation-backend` | `vllm`（由生成配置决定） | `vllm` 或 `transformers`；选择训练结束后评测使用的生成引擎。 |
| `--prepared-sft-data` | 未指定；按来源自动定位缓存，必要时准备 | 已准备的 SFT 数据目录，包含 `manifest.json` 和 `dataset/`；显式指定时须与来源数据及分词设置一致。 |

## evaluate_checkpoint.py

生成验证集回答并完成一二级判分，不自动调用 API 汇总。使用 `--checkpoint` 或 `--model-name` 二选一。

中断后重跑相同命令会继续已有回答和判分进度；从头评测使用新的 `--output`。题目和参考答案保存在共享快照中，元数据、回答和判决保存在评测目录，此阶段不生成 Markdown。已有旧格式报告或进度占用同名位置时，需选择新的输出目录。

读取旧 SFT 来源且无可用数据缓存时，当前程序可能自动创建 CPU 兼容环境并准备数据。

### 评测原始模型

评测尚未接受本项目训练的原始 Instruct 模型。`--config` 与 `--sft-source` 互斥。

| 参数 | 默认值 | 可填内容及作用 |
| --- | --- | --- |
| `--model-name` | 无；本用途必填 | `Qwen/Qwen3-4B-Instruct-2507`；选择原始模型，不加载本项目 LoRA。 |
| `--config` | `configs/training/sft.yaml` | SFT 配置文件路径；确定评测数据、长度和生成预算。 |
| `--sft-source` | 未指定；按配置确定评测数据及划分 | SFT 的 `final/` 或 `checkpoint-<step>/` 目录；使用其保存的评测配置和验证集划分，不加载它的模型权重。 |
| `--num-samples` | 完整验证集 | 不超过验证集大小的正整数；评测固定顺序的前 N 题。 |
| `--batch-size` | vLLM：`64`；Transformers：`4` | 正整数；每次生成提交的最大题数，在自适应模式下为起始题数。 |
| `--auto-batch-size` | 关闭（固定批大小） | 开关；仅限 Transformers，成功时增大批次，首次生成异常后缩小重试；与 vLLM 不能同时使用。 |
| `--device` | 有 CUDA 时 `cuda`，否则 `cpu` | `cuda` 或 `cpu`；选择评测设备，须满足配置的精度要求。 |
| `--output` | 仓库 `outputs/evaluations/` 下自动命名 | 评测目录路径；指定元数据、回答和判决的保存位置，使用新目录可从头评测。 |
| `--generation-config` | `configs/generation.yaml` | 生成配置文件路径；设置本次生成评测。 |
| `--generation-backend` | `vllm`（由生成配置决定） | `vllm` 或 `transformers`；选择生成引擎。 |
| `--prepared-sft-data` | 未指定；有 `--sft-source` 时按来源定位，否则按配置准备 | 已准备的 SFT 数据目录，包含 `manifest.json` 和 `dataset/`；覆盖自动定位结果，须与评测数据及分词设置一致。 |

### 评测训练产物

从产物记录自动读取模型设置、分词协议、评测数据及验证集划分；DPO、GRPO 使用其记录的 SFT 评测来源。

| 参数 | 默认值 | 可填内容及作用 |
| --- | --- | --- |
| `--checkpoint` | 无；本用途必填 | SFT、DPO 或 GRPO 的 `final/` 或 `checkpoint-<step>/` 目录；选择被评测的模型权重。 |
| `--num-samples` | 完整验证集 | 不超过验证集大小的正整数；评测固定顺序的前 N 题。 |
| `--batch-size` | vLLM：`64`；Transformers：`4` | 正整数；每次生成提交的最大题数，在自适应模式下为起始题数。 |
| `--auto-batch-size` | 关闭（固定批大小） | 开关；仅限 Transformers，成功时增大批次，首次生成异常后缩小重试；与 vLLM 不能同时使用。 |
| `--device` | 有 CUDA 时 `cuda`，否则 `cpu` | `cuda` 或 `cpu`；选择评测设备，须满足配置的精度要求。 |
| `--output` | 运行目录的 `evaluations/` 下自动命名 | 评测目录路径；指定元数据、回答和判决的保存位置，使用新目录可从头评测。 |
| `--generation-config` | `configs/generation.yaml` | 生成配置文件路径；设置本次生成评测。 |
| `--generation-backend` | `vllm`（由生成配置决定） | `vllm` 或 `transformers`；选择生成引擎。 |
| `--prepared-sft-data` | 未指定；按来源自动定位缓存，必要时准备 | 已准备的 SFT 数据目录，包含 `manifest.json` 和 `dataset/`；显式指定时须与来源数据及分词设置一致。 |

## summarize_evaluation.py

读取已完成的一二级结果，默认通过 API 补做尚未处理的三级判定，再汇总并生成阅读报告。`--evaluation` 与 `--resume-from` 二选一。

当前默认裁判为 DeepSeek Flash，关闭思考模式，完整提示见 [JUDGE_PROMPT](../utils/answer_judge.py)。需要 API 时先读取 `DEEPSEEK_API_KEY`，缺少则在终端隐藏输入；密钥仅供本次运行使用。取消输入或中断会保留机器进度，不刷新阅读报告。

正常结束后，阅读文件保存在原评测目录：`answers-0001.md` 等问答分卷、`api-decided.md` 和 `api-pending.md` 两类核查清单。再次成功汇总会替换同名阅读版，API 原始响应及用量保留在判决记录中。

### 新建评分并汇总

每次创建独立评分，复用已有题目、回答和一二级判决；终端打印实际评分目录。更换裁判设置时使用这一用途。

| 参数 | 默认值 | 可填内容及作用 |
| --- | --- | --- |
| `--evaluation` | 无；本用途必填 | 包含 `metadata.json`、`answers.jsonl`、`judgments.jsonl` 的一二级评测目录；选择已有回答并新建评分。 |
| `--config` | `configs/evaluation/deepseek.yaml` | 裁判配置文件路径；首次启用 API 时选择服务、模型和请求设置。 |
| `--no-api` | 关闭 | 开关；仅汇总已保存结果，不请求 API，也不读取裁判配置或索取密钥。 |
| `--retry-pending` | 关闭 | 开关；重试所选评分已有的待复核题，新评分没有历史待复核判决，通常无需填写。 |
| `--questions-per-file` | `100` | 正整数；每份问答 Markdown 的题数上限。 |

### 继续评分

继续已有评分，只处理未完成的题；已保存的待复核判决默认直接复用。

| 参数 | 默认值 | 可填内容及作用 |
| --- | --- | --- |
| `--resume-from` | 无；本用途必填 | 已有评分目录，如 `<评测目录>/scorings/scoring/`；沿用保存的回答来源、裁判设置和判决。 |
| `--config` | `configs/evaluation/deepseek.yaml`（仅首次启用 API 时使用） | 裁判配置文件路径；已有裁判设置时沿用已保存配置和提示，尚未启用 API 时才读取该文件。 |
| `--no-api` | 关闭 | 开关；仅汇总已保存结果，不请求 API，也不读取裁判配置或索取密钥。 |
| `--retry-pending` | 关闭 | 开关；在处理未完成题目之外，再请求所选评分中已保存的待复核题，追加新判决。 |
| `--questions-per-file` | `100` | 正整数；每份问答 Markdown 的题数上限。 |

### 重试待复核题

继续已有评分，额外重试已保存的待复核题。

| 参数 | 默认值 | 可填内容及作用 |
| --- | --- | --- |
| `--resume-from` | 无；本用途必填 | 已有评分目录，如 `<评测目录>/scorings/scoring/`；沿用保存的回答来源、裁判设置和判决。 |
| `--retry-pending` | 关闭；本用途须开启 | 开关；同时处理未完成的题，并对所选评分的待复核题重新请求 API，追加新判决。 |
| `--config` | `configs/evaluation/deepseek.yaml`（仅首次启用 API 时使用） | 裁判配置文件路径；已有裁判设置时沿用已保存配置和提示，尚未启用 API 时才读取该文件。 |
| `--no-api` | 关闭 | 开关；仅汇总已保存结果，不请求 API，也不读取裁判配置或索取密钥。 |
| `--questions-per-file` | `100` | 正整数；每份问答 Markdown 的题数上限。 |

### 只查看已有结果

可查看中断后的部分结果，未处理、待复核与明确错误分别统计。此用途不读取 `--config`，也不执行 `--retry-pending`。

| 参数 | 默认值 | 可填内容及作用 |
| --- | --- | --- |
| `--resume-from` | 无；本用途必填 | 已有评分目录，如 `<评测目录>/scorings/scoring/`；沿用保存的回答来源、裁判设置和判决。 |
| `--no-api` | 关闭；本用途须开启 | 开关；跳过 API 请求、密钥输入及待复核重试，只汇总现有判决并生成阅读报告。 |
| `--questions-per-file` | `100` | 正整数；每份问答 Markdown 的题数上限。 |
