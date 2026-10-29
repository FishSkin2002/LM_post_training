# 大语言模型数学推理后训练

## 项目目录

```text
.
├── math_data.py                 # OpenR1/GSM8K统一读取、R1筛选与来源索引
├── math_scoring.py              # boxed提取及Math-Verify数学等价判分
├── prepare_sft_data.py          # 原生聊天包装、长度统计和assistant-only标签
├── train_sft.py                 # 新训练、断点续训、完成后加训
├── evaluate_checkpoint.py       # 原始Instruct或LoRA的同题生成评测
├── sample_checkpoint.py         # 原始Instruct或LoRA抽样问答
├── training_workflow.py         # 题目组划分、运行状态与终端展示
├── configs/
│   ├── development/sft.yaml     # Instruct数据检查及本地输出路径
│   └── training/
│       ├── sft.yaml             # 主线Qwen3-4B-Instruct-2507/OpenR1配置
│       └── sft-8b.yaml          # 历史Base配置，新版入口拒绝使用
├── outputs/
│   ├── development/<run>/
│   │   ├── run.json            # 模型、模板、数据、预算、精度和阶段来源
│   │   ├── checkpoint-<step>/  # 完整断点，最多保留最近两个
│   │   ├── final/              # LoRA、分词器、原生模板与运行信息
│   │   ├── runs/               # TensorBoard事件
│   │   ├── evaluations/        # 逐题JSONL、Markdown阅读版和汇总JSON
│   │   └── trainer_state.json
│   ├── evaluations/            # 原始Instruct的评测报告
│   └── samples/                # 抽样问答Markdown
├── docs/                       # 目标、训练和协作约定
├── tools/
│   ├── test_instruct_sft.py     # 随机微型模型离线集成检查
│   ├── fixtures/               # 测试用固定版本官方模板
│   └── git_workflow.py          # 受控Git入口及相应测试
├── requirements.txt
├── AGENTS.md
├── CONTEXT.md
├── task_plan.md
├── progress.md
└── findings.md
```

训练机默认产物目录为 `/root/autodl-tmp/outputs/training/qwen3-4b-instruct-openr1-lora/`；模型和数据缓存位于仓库之外。新训练和加训遇到重名目录或文件会自动加 `-1`、`-2`；续训沿用原目录。适配器必须与同一官方底座配合使用，旧Base运行产物保留但不进入新版恢复或评测。

## 主要脚本使用说明

在项目根目录先运行 `python -m pip install -r requirements.txt`。真实主线固定使用 `Qwen/Qwen3-4B-Instruct-2507`；配置默认BF16底座/计算、FP32 LoRA及非重入激活检查点，训练时关闭KV cache。真实训练需要支持BF16的CUDA设备；CPU/FP32只用于显式配置的随机微型接口检查。

### prepare_sft_data.py

#### 检查数据、长度分布和监督标签

```bash
python prepare_sft_data.py
```

首次按配置读取数据和分词器，之后可复用Hugging Face缓存。默认OpenR1的 `default/train`：逐条对齐正确性及完整性标记，每题取原列表首条合格R1轨迹；不训练上游 `solution`。正确性字段中的 `null` 视为未提供判定，至少一个评判器必须明确通过。无法解析参考终值、无合格轨迹的记录和超长记录分别统计。超长不截断；`max_length` 初始检查预算为8192，正式训练前先查看实际长度和保留数量，再决定是否调整配置。

原生聊天结构为system固定指令、user原题、assistant完整R1解答：

```text
<|im_start|>system
Solve the following math problem step by step.
End your response with the final answer in \boxed{...}, for example \boxed{4}.<|im_end|>
<|im_start|>user
{problem原文}<|im_end|>
<|im_start|>assistant
{选中的R1完整解答}<|im_end|>
```

保留R1的LaTeX和think段；只监督assistant正文与一个im_end EOS，system/user/角色头/padding不计loss。生成只输入同一system/user和assistant起始头，不提供参考答案、不预填think。末尾格式换行不作额外监督目标。

可选参数：

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/development/sft.yaml` | 选择数据来源、长度预算及模型分词器。 |

GSM8K读取接口保留在 `math_data.py`；需测试时在自定义YAML的 `data` 中选择 `openai/gsm8k`、`config: main` 和对应split。保留原题、去掉整个 `<<...>>` 注释的完整参考解答与独立最终数值。新版主SFT入口仍限定OpenR1，不把GSM8K原解答伪造成R1轨迹。

### train_sft.py

#### 发起新训练

```bash
python train_sft.py
```

默认使用 `configs/training/sft.yaml`。先完成数据筛选、长度统计和按题目原文分组的固定划分，再加载模型；seed=42，10%题目组用于验证，同题不跨划分。配置显式保存长度和生成预算，原始模型与SFT比较时使用同一份配置和题目。

训练只验证loss，默认每500步验证和保存、每10步记录，保留最近两个checkpoint。正常结束先保存final和状态，再复用已有模型做一次完整验证集生成评测；失败可独立重跑，不需加训。`run.json` 绑定模型、原生模板、system、数据指纹/索引、筛选规则、预算、精度和判分版本。

可选参数：

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/training/sft.yaml` | 轮次、学习率、LoRA、数据、预算及频率设置。 |
| `--output-dir` | 配置中的 `output_dir` | 新运行目录，重名自动避让。 |

#### 从完整断点恢复未完成的训练

```bash
python train_sft.py --resume-from-checkpoint /path/to/run/checkpoint-500
```

恢复优化器、调度器、随机状态和进度，沿用原数据与训练计划、目录及TensorBoard步数。缺少必要状态或来源不符时在加载权重前拒绝。Ctrl+C不会额外保存中断状态，从最近完整断点继续。

可选参数：

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/training/sft.yaml` | 必须与该阶段配置一致。 |
| `--output-dir` | checkpoint所属运行目录 | 仅允许填写同一目录，不新建后缀。 |

#### 完成后开启加训阶段

```bash
python train_sft.py --continue-from /path/to/previous/final --additional-epochs 1
```

`/path/to/...` 换成实际产物路径。只继承适配器权重，重新创建优化器、调度器、步数和日志目录；`--additional-epochs` 是新阶段追加轮数，可为正小数。来源仍需匹配模型、模板、数据、预算、精度和LoRA结构，不叠加第二套适配器。该阶段也可从自己的完整断点恢复。

可选参数：

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/training/sft.yaml` | 新阶段学习率等训练设置；来源协议保持。 |
| `--output-dir` | 配置路径并自动避重 | 指定新阶段目录。 |

### evaluate_checkpoint.py

#### 评测未经本项目训练的原始Instruct

```bash
python evaluate_checkpoint.py --model-name Qwen/Qwen3-4B-Instruct-2507
```

不需要训练目录或适配器。只从缓存加载模型和分词器；数据准备与训练共用同一来源、筛选、分词长度规则及固定划分。省略题数时评测完整验证集。原始模型也接收同一system格式指令，方便提取答案。

可选参数：

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/training/sft.yaml` | 指定与待比较SFT相同的数据、精度和预算。 |
| `--num-samples` | 完整验证集 | 选固定顺序前N题，须为合法正整数。 |
| `--device` | 自动选择CUDA或CPU | 指定设备；BF16配置要求支持的CUDA。 |
| `--output` | `outputs/evaluations/original-<model>-<dataset>-<config>-<split>-n<N>.json` | 三种报告的共同基础名，相同命令可续跑。 |

#### 评测checkpoint或final

```bash
python evaluate_checkpoint.py --checkpoint /path/to/run/final
```

读取保存的模型/数据/预算等配置，沿用该运行划分；评测不改变权重或训练状态，不创建优化器。

可选参数：

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--num-samples` | 完整验证集 | 固定顺序前N题，超范围报错。 |
| `--device` | 自动选择 | 设备需满足来源精度要求。 |
| `--output` | `<run>/evaluations/<checkpoint>-<dataset>-<config>-<split>-n<N>.json` | 自定义报告位置。 |
| `--config` | 本用途忽略，使用产物保存的配置 | 仅原始模型评测需要选择外部配置。 |

两种模型来源互斥。生成使用贪心和保存的 `max_new_tokens`，初始检查配置为8192；前后比较固定预算，不提供标准解答，遇im_end停止。数据身份进入默认路径，避免复用旧Base或另一数据集的进度文件。

每题完成后立即追加并flush `.progress.jsonl`，再同步 `.md` 阅读版；最终写 `.json` 汇总。记录question、完整 `reference_answer`、独立 `reference_final_answer`、生成answer、原始记录/轨迹索引、token数、判分、截断与耗时，标准答案不进入生成输入。

中断后原样重跑同一命令，保持来源、数据、配置、题数和output一致。已有N条就从第N+1题继续；错误/格式无效/截断回答同样复用，全部完成不再生成。Markdown丢失会从JSONL重建。汇总包含全部记录，耗时累计实际生成时间；从头重评换新的output路径。不增加任务签名、任务头或日志修复。

评分取最后一个 `\boxed{...}`，处理嵌套花括号，再用固定版本Math-Verify比较参考终值，支持分数/小数等数学等价；A–E选项使用字符串解析，忽略大小写。当前不会根据题干把选项字母换算成数值答案，参考值为数字而输出字母时会判错。最后一个boxed缺失、未闭合、不可解析或生成截断均计错，不回退从推导末尾猜数字；不检查过程是否正确。报告同时给格式有效率和截断率。数学等价只用于生成评分，SFT仍按参考词元交叉熵训练。完整协议见[训练运行约定](docs/training-workflow.md)。

### sample_checkpoint.py

#### 查看原始Instruct的问答

```bash
python sample_checkpoint.py --model-name Qwen/Qwen3-4B-Instruct-2507 --local-files-only
```

从配置的数据集合按种子抽样，保存Markdown；少量样例不等于正确率评测。

可选参数：

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/training/sft.yaml` | 原始模型的数据、精度和默认预算。 |
| `--num-samples` | `5` | 抽题数，须不超过数据大小。 |
| `--seed` | `42` | 固定抽题。 |
| `--max-new-tokens` | 配置的生成预算 | 显式覆盖本次抽样预算。 |
| `--local-files-only` | 默认允许获取缺少的模型文件 | 本例明确只用缓存。 |
| `--device` | 自动选择 | 设备需满足配置精度要求。 |
| `--output` | `outputs/samples/`下含来源/数据名/种子的文件 | 保存Markdown，同名覆盖。 |

#### 查看训练产物问答

```bash
python sample_checkpoint.py --checkpoint /path/to/run/final
```

本用途始终只从本地和缓存加载，沿用产物保存的配置，校验来源；也可使用checkpoint目录。完整参考解答和终值仅写报告，不进入模型输入。

可选参数：

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--num-samples` | `5` | 抽题数。 |
| `--seed` | `42` | 固定抽题以便对照。 |
| `--max-new-tokens` | 来源保存的预算 | 覆盖本次抽样上限。 |
| `--device` | 自动选择 | 设备需满足来源精度。 |
| `--output` | `outputs/samples/` | 自定义Markdown路径。 |
| `--local-files-only` | 本用途已固定开启 | 可省略。 |
| `--config` | 本用途忽略 | 使用来源保存的配置。 |

### TensorBoard

#### 查看训练与验证曲线

```bash
tensorboard --logdir /root/autodl-tmp/outputs/training
```

开发输出可改填 `outputs/development`，单次运行可填其 `runs/`。

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--port` | `6006` | 指定端口。 |
| `--host` | TensorBoard默认值 | 指定监听地址。 |

`train/loss`、`train/grad_norm`、`train/learning_rate`、`train/epoch`记录训练状态；`eval/loss`、`eval/runtime`、`eval/samples_per_second`来自完整验证集loss。生成正确率单独写评测报告，不能把验证loss解释成生成正确率。

### tools/test_instruct_sft.py

#### 离线检查数据、训练、恢复及评测

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python tools/test_instruct_sft.py
```

使用随机微型Qwen/LoRA、固定版本官方模板、合成OpenR1记录和真实Math-Verify。验证筛选/分组/长度、掩码/EOS、参数更新、恢复/加训、TensorBoard、生成前后对比、逐题续跑及缓存恢复；不加载真实4B权重或测量其显存与准确率。

20条真实记录与官方完整分词器的检查结果见[样本检查记录](docs/research/openr1-data-smoke.md)，包含长度范围和数据标签与本地判分不一致的具体边界。

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `-v` | 关闭详细逐项输出 | 显示测试名称。 |
| `-q` | 默认unittest输出 | 简略显示。 |
| `-k` | 全部 | 按名称筛选。 |
