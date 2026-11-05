# 大语言模型数学推理后训练

以学习为目的的大语言模型数学推理后训练项目，主要学习LoRA SFT、DPO、GRPO等。[实验报告](reports/)

## 项目目录

```text
.
├── math_data.py                 # OpenR1/GSM8K统一读取、R1筛选与来源索引
├── math_scoring.py              # boxed提取及Math-Verify数学等价判分
├── answer_judge.py              # 可配置的 API 裁判、重试及判决解析
├── prepare_sft_data.py          # 原生聊天包装、长度统计和assistant-only标签
├── prepared_sft_data.py         # 导出原环境已校验的 SFT 准备数据，供新环境复用
├── train_sft.py                 # 新训练、断点续训、完成后加训
├── train_post_sft.py            # 从同一 SFT final 分别训练 DPO 和 GRPO
├── post_training_data.py        # PRM800K 偏好对、Skywork 数学题和答案奖励
├── evaluate_checkpoint.py       # 原始Instruct或LoRA的同题生成评测
├── rescore_evaluation.py        # 已保存 OpenR1 回答的三级重评与人工核查报告
├── sample_checkpoint.py         # 原始Instruct或LoRA抽样问答
├── generation.py                # vLLM/Transformers 公共生成与训练后资源交接
├── check_vllm_environment.py     # 不加载模型的 GPU/驱动/依赖检查
├── training_workflow.py         # 题目组划分、运行状态与终端展示
├── configs/
│   ├── development/sft.yaml     # Instruct数据检查及本地输出路径
│   ├── evaluation/glm-5.3.yaml  # 官方 GLM-5.3 裁判与请求限制
│   ├── generation.yaml          # 公共生成引擎，默认 vLLM
│   └── training/
│       ├── sft.yaml             # 主线Qwen3-4B-Instruct-2507/OpenR1配置
│       ├── dpo.yaml             # PRM800K 完整解答偏好对
│       ├── grpo.yaml            # Skywork 数学题与 GRPO 采样设置
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
│   ├── test_post_sft.py         # DPO、GRPO 与共用评测离线检查
│   ├── test_rescore_evaluation.py # 三级重评与模拟 API 离线检查
│   ├── fixtures/               # 测试用固定版本官方模板
│   └── git_workflow.py          # 受控Git入口及相应测试
├── reports/                    # 实验报告
├── requirements.txt
├── requirements-vllm-cu128.txt   # 独立 vLLM 环境的固定依赖
├── AGENTS.md
├── CONTEXT.md
├── task_plan.md
├── progress.md
└── findings.md
```

训练机默认产物目录为 `/root/autodl-tmp/outputs/training/qwen3-4b-instruct-openr1-lora/`；模型和数据缓存位于仓库之外。新训练和加训遇到重名目录或文件会自动加 `-1`、`-2`；续训沿用原目录。适配器必须与同一官方底座配合使用，旧Base运行产物保留但不进入新版恢复或评测。

## 主要脚本使用说明

默认生成引擎为 vLLM。推荐使用独立 Conda 环境，由 Conda 创建 Python，再用 pip 安装依赖：

```bash
conda create -n qwen-vllm --override-channels -c conda-forge python=3.12 pip -y
conda activate qwen-vllm
python -m pip install -r requirements-vllm-cu128.txt
python check_vllm_environment.py
```

固定组合为 PyTorch 2.8.0/cu128、vLLM 0.10.2、TRL 0.27.2 和 Transformers 4.57.6。宿主 NVIDIA 驱动需至少 570.26；容器里的 CUDA Toolkit 不会升级宿主驱动。已有 CUDA 12.4/R550 训练机须先选择支持 CUDA 12.8 的驱动/镜像，再使用该环境。系统不必单独安装同版 Toolkit，预编译包自带运行时；真实 GPU 短测仍需通过。

原环境继续使用 `requirements.txt`；运行生成入口时显式加 `--generation-backend transformers`。新 GRPO 配置默认 `trainer.use_vllm: true`，原生训练需在阶段 YAML 显式设为 `false`，该设置与训练结束后的评测引擎分开。旧 GRPO 断点使用原环境及保存时的配置恢复，不将更换引擎或训练库版本当作同一次续训。

真实主线固定使用 `Qwen/Qwen3-4B-Instruct-2507`；配置默认BF16底座/计算、FP32 LoRA及非重入激活检查点，训练时关闭KV cache。真实训练需要支持BF16的CUDA设备；CPU/FP32只用于显式配置的随机微型接口检查。

`train_sft.py`、`train_post_sft.py`、`evaluate_checkpoint.py` 和 `sample_checkpoint.py` 均支持 `--generation-config PATH`（默认 `configs/generation.yaml`）和 `--generation-backend vllm|transformers`。前者控制公共评测/抽样，后者覆盖其中的引擎设置；失败时明确报错，不自动更换引擎。四个模型的对比应统一引擎，切换引擎使用新的评测输出路径。

训练正常结束先保存 `final/` 和状态。vLLM 评测通过替换训练进程启动，先释放训练模型、优化器和 GRPO 推理引擎，再加载保存的产物。可随后独立评测 `final/`，无需重新训练。GRPO 的训练采样保持当前策略同步、固定 SFT 参考、KL 和 0/1 数学奖励；具体要求见[vLLM 生成约定](docs/vllm-generation.md)。

旧环境训练的 SFT 产物可能因新旧 `datasets` 的指纹算法不同而无法直接重新准备出相同身份的数据。先在原环境运行下文的 `prepared_sft_data.py` 导出一次，再为新环境的 DPO/GRPO、独立评测和抽样指定 `--prepared-sft-data /path/to/prepared-sft`。导出保留原划分与指纹，不修改 SFT 产物。

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

训练只验证loss，默认每5000步验证和保存、每10步记录，保留最近两个checkpoint。正常结束先保存final和状态，再复用已有模型做一次完整验证集生成评测；失败可独立重跑，不需加训。`run.json` 绑定模型、原生模板、system、数据指纹/索引、筛选规则、预算、精度和判分版本。

可选参数：

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--generation-config` | `configs/generation.yaml` | 公共评测/抽样的生成配置。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可选 `vllm` 或 `transformers`，显式覆盖。 |
| `--config` | `configs/training/sft.yaml` | 轮次、学习率、LoRA、数据、预算及频率设置。 |
| `--output-dir` | 配置中的 `output_dir` | 新运行目录，重名自动避让。 |

#### 从完整断点恢复未完成的训练

```bash
python train_sft.py --resume-from-checkpoint /path/to/run/checkpoint-5000
```

恢复优化器、调度器、随机状态和进度，沿用原数据与训练计划、目录及TensorBoard步数。缺少必要状态或来源不符时在加载权重前拒绝。Ctrl+C不会额外保存中断状态，从最近完整断点继续。

可选参数：

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--generation-config` | `configs/generation.yaml` | 公共评测/抽样的生成配置。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可选 `vllm` 或 `transformers`，显式覆盖。 |
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
| `--generation-config` | `configs/generation.yaml` | 公共评测/抽样的生成配置。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可选 `vllm` 或 `transformers`，显式覆盖。 |
| `--config` | `configs/training/sft.yaml` | 新阶段学习率等训练设置；来源协议保持。 |
| `--output-dir` | 配置路径并自动避重 | 指定新阶段目录。 |

### prepared_sft_data.py

#### 在原 SFT 环境导出准备数据

激活训练该 SFT 时的原环境，在同一份数据缓存仍可用时运行：

```bash
python prepared_sft_data.py --sft-source /path/to/sft-run/final --output /path/to/prepared-sft
```

`--output` 必须是新目录。导出前按原 SFT 产物核对完整数据及划分，输出准备数据和身份清单；会保存一份准备数据副本。然后切换到 `qwen-vllm`，在 DPO/GRPO、评测或抽样命令中添加 `--prepared-sft-data /path/to/prepared-sft`。同一份导出可供多个阶段共用，原始模型与训练后模型继续使用同一批验证题。

使用导出的准备数据开始 DPO/GRPO 后，运行信息会记录该路径，后续阶段续训、自动评测和产物抽样默认沿用它；请保留该目录。

### train_post_sft.py

#### 从已完成的 SFT 开始 DPO 或 GRPO

```bash
python train_post_sft.py dpo --sft-source /path/to/sft-run/final
python train_post_sft.py grpo --sft-source /path/to/sft-run/final
```

将 `/path/to/sft-run/final` 换成 `configs/training/sft.yaml` 对应运行的 `final/`，两条命令使用同一路径。两个阶段各自更新一份从 SFT 加载的 LoRA，各有优化器和输出目录。DPO 共用 SFT 提示，按 boxed 格式和可解析性筛选 PRM800K 的 `chosen`。`rejected` 按标题包裹原内容，末尾标题为空或残缺时仅去掉标题。GRPO 在 Skywork 数学题上采样，按最后一个 boxed 答案的数学等价性和原生 EOS 计 0/1 奖励。

两个阶段都排除与原 SFT 验证题仅有空白差异的训练题。正常结束后先保存 `final/`，再对原 SFT 验证集生成评测。默认配置面向单 GPU。训练机短测可复制对应 YAML，在 `trainer` 下设置 `max_steps` 并指定单独的 `--output-dir`。默认长度和采样组大小是短测的起始值。短测正常结束后同样进入全量评测，可在看到适配器保存完成后单独用 `evaluate_checkpoint.py --num-samples` 做抽样评测。

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--prepared-sft-data` | 使用产物已记录的路径，或重新准备 | 读取在原 SFT 环境导出的准备数据，保留原身份与划分。 |
| `--generation-config` | `configs/generation.yaml` | 公共评测/抽样的生成配置。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可选 `vllm` 或 `transformers`，显式覆盖。 |
| `--config` | `configs/training/dpo.yaml` 或 `grpo.yaml` | 指定数据版本、长度、训练步数和算法设置。 |
| `--output-dir` | 阶段 YAML 的 `output_dir` | 指定新运行目录，重名自动避让。 |

#### 恢复未完成的 DPO 或 GRPO

```bash
python train_post_sft.py dpo --resume-from-checkpoint /path/to/dpo-run/checkpoint-100
```

恢复 GRPO 时改用 `grpo` 和相应阶段的断点。续训从断点恢复策略、优化器、调度器和进度，按记录的 SFT 产物重建固定参考。原始 SFT `final/` 需保留在保存时的路径。切换 DPO 答案格式规则时，从 SFT `final/` 开启新运行。

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--prepared-sft-data` | 使用产物已记录的路径，或重新准备 | 读取在原 SFT 环境导出的准备数据，保留原身份与划分。 |
| `--generation-config` | `configs/generation.yaml` | 公共评测/抽样的生成配置。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可选 `vllm` 或 `transformers`，显式覆盖。 |
| `--config` | 对应阶段的训练 YAML | 指定与断点一致的阶段配置。 |
| `--output-dir` | checkpoint 所属运行目录 | 显式填写时须指向同一目录。 |

完整要求见[后续训练约定](docs/post-training-workflow.md)。

### evaluate_checkpoint.py

#### 评测未经本项目训练的原始Instruct

```bash
python evaluate_checkpoint.py --model-name Qwen/Qwen3-4B-Instruct-2507
```

不需要训练目录或适配器。只从缓存加载模型和分词器；数据准备与训练共用同一来源、筛选、分词长度规则及固定划分。省略题数时评测完整验证集。原始模型也接收同一system格式指令，方便提取答案。

可选参数：

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--prepared-sft-data` | 使用产物已记录的路径，或重新准备 | 读取在原 SFT 环境导出的准备数据，保留原身份与划分。 |
| `--generation-config` | `configs/generation.yaml` | 公共评测/抽样的生成配置。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可选 `vllm` 或 `transformers`，显式覆盖。 |
| `--config` | `configs/training/sft.yaml` | 指定与待比较SFT相同的数据、精度和预算。 |
| `--sft-source` | 按配置生成固定划分 | 指向待比较的 SFT `final/`，用产物保存的准确划分、模板和预算。与 `--config` 互斥。 |
| `--num-samples` | 完整验证集 | 选固定顺序前N题，须为合法正整数。 |
| `--batch-size` | `4` | 固定模式的每批最大题数，或自适应模式的起始题数，须为正整数。 |
| `--auto-batch-size` | 固定批大小 | 启用成功翻倍、失败后缩小批次的自适应模式。 |
| `--device` | 自动选择CUDA或CPU | 指定设备；BF16配置要求支持的CUDA。 |
| `--output` | `outputs/evaluations/` 下按模型和数据命名 | 三种报告的共同基础名。指定 `--sft-source` 时文件名包含该评测划分和解码设置的标识。 |

#### 评测checkpoint或final

```bash
python evaluate_checkpoint.py --checkpoint /path/to/run/final
```

SFT、DPO 和 GRPO 均可通过该命令评测。报告区分训练阶段，共用源 SFT 运行的评测题、模板、精度和生成预算。

可选参数：

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--prepared-sft-data` | 使用产物已记录的路径，或重新准备 | 读取在原 SFT 环境导出的准备数据，保留原身份与划分。 |
| `--generation-config` | `configs/generation.yaml` | 公共评测/抽样的生成配置。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可选 `vllm` 或 `transformers`，显式覆盖。 |
| `--num-samples` | 完整验证集 | 固定顺序前N题，超范围报错。 |
| `--batch-size` | `4` | 固定模式的每批最大题数，或自适应模式的起始题数，须为正整数。 |
| `--auto-batch-size` | 固定批大小 | 启用成功翻倍、失败后缩小批次的自适应模式。 |
| `--device` | 自动选择 | 设备需满足来源精度要求。 |
| `--output` | `<run>/evaluations/<checkpoint>-<dataset>-<config>-<split>-n<N>.json` | 自定义报告位置。 |
| `--config` | 使用产物保存的评测配置 | 仅原始模型评测使用外部配置。 |

两种模型来源互斥。生成使用贪心和保存的 `max_new_tokens`，初始检查配置为8192；前后比较固定预算，不提供标准解答，遇im_end停止。数据身份进入默认路径，避免复用旧Base或另一数据集的进度文件。

自动生成评测也默认每批最多 4 题。每批返回后按题序逐题评分、追加并 flush `.progress.jsonl`，再向 `.md` 追加新题，最后写 `.json` 汇总。逐题记录题目、完整参考解答、参考终值、模型回答、原始索引、token 数、判分、截断、实际批大小和分摊耗时。

启用 `--auto-batch-size` 后，每批生成成功就翻倍，直到首次生成异常。首次异常优先回到最后成功的较小批次，首批即异常或尾批已经更小时则直接减半。此后遇到异常就减半，从同一题重试。单题仍异常时报告原错误。

中断后使用相同来源、数据、题数和输出路径重跑，可调整批大小和模式。已有 N 条就从第 N+1 题继续，自适应模式从本次指定的起始大小重新试探。错误、格式无效及截断记录同样复用，不重新核验；Markdown 只追加新题，不重建旧正文。批生成耗时在该批题目间均分，每题另计评分耗时。汇总将保存的耗时相加，旧记录按单题批次读入。从头重评使用新的 `--output`。

评分取最后一个 `\boxed{...}`，处理嵌套花括号，再用固定版本Math-Verify比较参考终值，支持分数/小数等数学等价；A–E选项使用字符串解析，忽略大小写。当前不会根据题干把选项字母换算成数值答案，参考值为数字而输出字母时会判错。最后一个boxed缺失、未闭合、不可解析或生成截断均计错，不回退从推导末尾猜数字；不检查过程是否正确。报告同时给格式有效率和截断率。数学等价只用于生成评分，SFT仍按参考词元交叉熵训练。完整协议见[训练运行约定](docs/training-workflow.md)。

### rescore_evaluation.py

#### 准备 API 裁判凭据

在运行重评的 Bash 终端中输入密钥：

```bash
read -rsp 'GLM API Key: ' ZAI_API_KEY
export ZAI_API_KEY
```

默认裁判配置为 `configs/evaluation/glm-5.3.yaml`，使用官方 `glm-5.3` 接口，密钥从 `ZAI_API_KEY` 读取。配置中可调整基础地址、模型名、凭据环境变量、超时、最大尝试次数、重试间隔、输出预算及模型支持的 `request_parameters`。默认每次请求超时 180 秒，最多尝试 3 次，输出预算 4096 tokens；不会在配置或报告中保存密钥。

切换到 DeepSeek 时，复制 YAML 并填写官方基础地址、实际模型名及对应凭据环境变量，再通过 `--config` 选择。`request_parameters` 直接对应请求体中的附加参数，不能覆盖模型、消息、流式模式及输出预算。

#### 对完整生成报告补充后续判分

```bash
python rescore_evaluation.py --input /path/to/evaluation.json --output /path/to/evaluation-rescored.json
```

把路径替换为生成完成的 OpenR1 JSON 报告及新的输出路径。原始 Instruct、SFT、DPO、GRPO 的报告均可使用；不接受生成过程中的 JSONL，不重新生成答案或加载 Qwen 权重。现有生成入口和 GRPO 奖励继续使用原有评分。

第一级已在生成阶段完成，直接沿用保存的 `correct`、`format_valid` 和 `truncated`，不重新数学核验。截断或一级判正确的题直接结束；其余结果进入第二级。第二级只比较 R1 和模型正式回答里的单个完整非空 boxed，通过后停止，否则调用 API。API 无法确定、返回内容不符合要求或请求失败均记为待复核。明确的思考结束边界为 `</think>`，没有边界时检查全文。

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/evaluation/glm-5.3.yaml` | 选择服务地址、实际裁判模型和请求设置。 |
| `--output` | 输入文件同目录的 `原文件名-rescored.json` | 指定新报告的基础名，不能覆盖输入。 |
| `--retry-pending` | 复用已有待复核判决 | 本次重新尝试待复核题；已明确判决的题继续复用。 |

以 `evaluation-rescored.json` 为输出时，同时保存：

- `.progress.jsonl`：本次独立重评的逐题判决，随判分立即保存，用于继续运行。
- `.json`：整体正确率和完整逐题结果，保留原报告的指标与原判分。
- `.md`：逐题追加的阅读版，最新正确比例、正确题数/总题数、待复核数量及题号在文末。
- `.api-review.md`：追加所有进入第三级的题目，标明 API 判对、判错或待复核，含前两级失败原因、原题、参考、回答及 API 判分依据。

JSON 在完成或中断时更新汇总，Markdown 每题追加，并在文末追加本次汇总。已有正文不读取或重写；重试待复核题时追加新的判决，同一索引以较后的判决为准。正确率始终为已确认正确数除以全部输入题数，待复核题保留在分母里；API 新增正确会自动计入，不重复累加。截断题在第一级终止，不进入 API 或待复核列表。中断时另报尚未处理的题数。完整规则见[独立重评约定](docs/evaluation-rescoring.md)。

#### 继续重评或重试待复核题

```bash
python rescore_evaluation.py --input /path/to/evaluation.json --output /path/to/evaluation-rescored.json --retry-pending
```

继续时只按已保存的索引跳过旧记录，不核查历史题目、判决内容或配置身份，只追加未处理的题。读取保存的判决供正确率汇总，不重新计算旧分数。加上 `--retry-pending` 才会再次处理已有待复核题，之前的尝试记录一并保留。输入、模型、提示或配置变化时自行选择新的输出路径。缺少凭据时，仅需 API 的题记为待复核，设置凭据后可用该命令继续。

| 可选参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--config` | `configs/evaluation/glm-5.3.yaml` | 沿用首次重评的裁判配置。 |
| `--retry-pending` | 省略时复用待复核记录 | 本次重新尝试待复核题。 |

### sample_checkpoint.py

#### 查看原始Instruct的问答

```bash
python sample_checkpoint.py --model-name Qwen/Qwen3-4B-Instruct-2507 --local-files-only
```

从配置的数据集合按种子抽样，保存Markdown；少量样例不等于正确率评测。

可选参数：

| 参数 | 默认行为 | 用途 |
| --- | --- | --- |
| `--prepared-sft-data` | 使用产物已记录的路径，或重新准备 | 读取在原 SFT 环境导出的准备数据，保留原身份与划分。 |
| `--generation-config` | `configs/generation.yaml` | 公共评测/抽样的生成配置。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可选 `vllm` 或 `transformers`，显式覆盖。 |
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
| `--prepared-sft-data` | 使用产物已记录的路径，或重新准备 | 读取在原 SFT 环境导出的准备数据，保留原身份与划分。 |
| `--generation-config` | `configs/generation.yaml` | 公共评测/抽样的生成配置。 |
| `--generation-backend` | 配置中的引擎，默认 `vllm` | 可选 `vllm` 或 `transformers`，显式覆盖。 |
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

### tools/test_rescore_evaluation.py

#### 离线检查三级重评

```bash
python tools/test_rescore_evaluation.py
```

使用真实 Math-Verify 和模拟 HTTP 响应检查三级路由、截断拦截、API 协议及重试、待复核、正确率汇总、人工核查明细、输入保护和继续处理。不调用真实付费 API；裁判的实际判分质量需人工核查。可用 `-v` 查看逐项结果。

### tools/test_post_sft.py

#### 离线检查 DPO、GRPO 和共用评测

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python tools/test_post_sft.py
```

使用随机微型模型核对偏好对、奖励、参考冻结、参数更新、保存恢复和评测一致性。可用 `-v` 查看逐项结果。
