# 环境发现

## GSM8K 公开实现核查

- 2026-10-27 研究范围为代表性一手实现，不据此声称所有项目都采用同一方案。
- [GSM8K 原论文](https://arxiv.org/pdf/2110.14168)附录 E、图 12 展示的联合训练目标中，语言模型目标屏蔽问题词元、监督完整 solution；该 solution 包括推导和最终答案。此处是附录的目标说明，不将其混同于 GRPO 或仅最终答案交叉熵。
- [MetaMath train_math.py](https://github.com/meta-math/MetaMath/blob/main/train_math.py)：targets 使用完整 output/response 加 EOS；preprocess 只将 source 前缀 labels 设为 -100，collator 将 padding 设为 -100。未找到针对 response 内推导段的单独损失屏蔽。
- [HuggingFaceTB 的 Qwen3/GSM8K 数据脚本](https://huggingface.co/HuggingFaceTB/qwen3-1.7b-gsm8k-sft/blob/main/scripts/prepare_combined_data.py)：移除原解答中的计算器注释和末尾 #### 行，将步骤放进非空 think 块，最终答案另放在块外；还混入 MetaMathQA 的 GSM 衍生样本。这是社区公开示例，不是 Qwen 官方配方。
- 对应 [train_improved.py](https://huggingface.co/HuggingFaceTB/qwen3-1.7b-gsm8k-sft/blob/main/scripts/train_improved.py) 和 train_continued.py 将包含完整 assistant 内容的 messages 数据交给 SFTTrainer，没有设置非空思考段排除项，也没有显式开启 assistant_only_loss。依照 [TRL 文档](https://huggingface.co/docs/trl/sft_trainer)的语言建模数据默认行为推断，思考段保留为监督目标；默认甚至包含问题部分，不能笼统说所有框架都默认屏蔽题目。历史实际依赖版本未在这两份脚本中锁定，不能声称已复现其逐词元标签。
- [verl GSM8K 数据处理](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/gsm8k.py)：从原始 answer 的 #### 后提取最终数值供 reward_model.ground_truth 使用，题目附带逐步推导要求；[奖励函数](https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/gsm8k.py) 按最终数值匹配打分。
- 已继续追踪 verl 单轮路径：[ray_trainer.compute_response_mask](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/ray_trainer.py) 从响应位置的注意力掩码构建 response_mask；[core_algos.compute_grpo_outcome_advantage](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py) 将结果优势广播到 response_mask，策略损失也按该 mask 聚合。因此如果响应包含思考词元，它们同样参与策略训练，不能解释为只训练最后答案词元。
- [ms-swift 默认 SFT 损失](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md)：assistant 响应计算交叉熵，默认不监督用户/系统部分；ignore_empty_think 只去掉空块，ignore_think_prefix 只去掉开头标记，后续推导和闭合标记仍计算损失。这是 Qwen 官方推荐框架的通用机制，并非 GSM8K 专属实验结果。
- 核查结论：上述普通/显式思考 SFT 都支持完整推导作为训练目标，未找到整段非空思考默认屏蔽的设置；结果奖励 RL 则只核对最终数值，但通常更新完整生成响应。不能把“只按最终答案打分”解释成“只对最终答案词元训练”。
- 对本项目的建议是以“推导与最终输出都监督，问题与 padding 屏蔽”作为思考 SFT 的基线；这是基于实现的建议，不是已经选定或验证效果的实验决定。本轮未改代码或运行配置。

## Qwen3 思考模式与损失范围

- 用户于 2026-10-27 确定后续各规模 Qwen3 遵循官方思考模式最佳实践；长期约定见 [思考模式约定](docs/qwen3-thinking.md)。当前 Python/YAML 尚未适配该新约定。
- Qwen 官方 ms-swift 指南中的 ignore_empty_think 指屏蔽空思考块，不是屏蔽非空推导。用户提出的非空思考段损失掩码尚未成为训练决定。
- 已核对当前 Transformers 的因果语言模型损失：先构造完整 logits，再由交叉熵忽略 labels=-100 的位置；它没有将这些输入位置移出注意力上下文，也没有按标签掩码切断此前隐藏状态到最终输出的计算依赖。
- 文档站训练页面无法访问，改用 QwenLM/Qwen3 官方仓库的 [ms-swift 训练说明](https://github.com/QwenLM/Qwen3/blob/main/docs/source/training/ms_swift.md) 交叉核对。

## 默认训练环境

- 用户于 2026-10-27 确定长期模型默认：**训练机使用 Qwen3-1.7B，开发机使用 Qwen3-0.6B**。后续为某台机器准备训练或验证配置时沿用其默认；用户显式指定其他模型时使用指定值。可执行模型名称保存在对应机器的 YAML 中。
- 用户于 2026-10-26 明确：实际训练使用租用的 **32 GB 显存 GPU**，后续默认训练配置均按这一环境准备。2026-10-27 日志显示设备名为 NVIDIA vGPU-32GB，nvidia-smi 总显存 32760 MiB、驱动 550.120、驱动支持 CUDA 12.4；CUDA OOM 报告可用总容量 31.60 GiB。底层物理 GPU 型号和 BF16 支持状态仍未确认，不能由 vGPU 名称推断。
- 本机环境只用于当前代码和数据验证，不能用本机显存容量替代上述训练环境。
- 配置统一放在 `configs/`：`common.yaml` 存放共享精度、数据和模板设置；`development/sft.yaml` 与 `training/sft.yaml` 分别存放各机器的模型名称和训练参数。选择或调整模型时读取对应机器的 YAML，参数值以该文件为准。
- `train_sft.py` 默认读取训练机配置；`prepare_sft_data.py` 默认读取开发机配置。两者都可用 `--config` 显式选择机器配置，并使用所选模型的分词器；训练入口还用同一模型名称加载权重。两套输出目录分开，训练机的新目录还与原有 0.6B 检查点分开。
- 两个入口的共享设置通过 `yaml.safe_load` 读取 `common.yaml`，不依赖执行时的当前目录。原有训练命令行参数可临时覆盖文件中的对应项；`bf16: auto` 在构造 TrainingArguments 时按 GPU 能力解析。后续调整机器配置后，需确认模型字段不会传给 TrainingArguments，且分词器与模型使用同一名称。
- 本机已有 PyYAML 6.0.3，现已作为直接依赖列入 requirements.txt。YAML 中的学习率写为 `0.00002`；读取时也支持 `2e-5` 这种可能被 PyYAML 识别为字符串的科学计数法。

## 本机验证环境

- 仓库为 `/home/yunli/code/qwen/codex`，远程 origin 为 `git@github.com:FishSkin2002/qwen.git`。
- `/home/yunli/miniconda3/bin/python` 为 Python 3.12.15，已有 torch 2.6.0+cu124、transformers 5.20.2、datasets 5.1.0、accelerate 1.15.0。
- 本机 GPU 为约 8 GB 显存的 NVIDIA RTX 4060，仅作为本地环境记录；实际训练默认使用上述 32 GB 租用 GPU。Codex 沙箱内访问本机 GPU 受限，需要正式宿主授权。
- Hugging Face 缓存已有 Qwen3-0.6B 和 GSM8K，可离线加载，已在此前真实运行中核实。

## 当前训练结果与推理输入

- 用户于 2026-10-27 在第二次 1.7B OOM 后，将当前实验改为 0.6B 并完成新训练；现提供下载后的 `outputs/development/gsm8k-sft/checkpoint-1404` 供本机抽样问答。推理使用该 checkpoint 的模型和分词器，不从训练机 YAML 的模型默认值加载起始模型。
- 该目录 config.json 的 hidden_size=1024、num_hidden_layers=28、vocab_size=151936、dtype=float32；model.safetensors 大小 2384234968 字节。分词器的聊天模板单独保存在 chat_template.jinja，tokenizer_config.json 内没有内联 chat_template 字段并不表示模板缺失。
- 样本观察默认使用 GSM8K test，固定种子抽取 5 道不同题目，沿用训练中的非思考模板。原始题目作为唯一用户输入，数据集参考解答在模型生成后展示。此用途是定性查看回答，不记录少量样本为完整答案正确率。
- 已在本机用 CUDA/BF16 离线加载该 checkpoint，对 test 索引 `[1309, 228, 51, 563, 501]` 生成原始问答报告，均正常生成结束标记。用户要求自行查看答案，报告只保留问题、模型回答和数据集参考解答，不加入人工判分或点评。
- 用户随后要求问答仅保存到文件，终端不展示问答。抽样入口默认保存到项目 `outputs/samples/<checkpoint目录名>-<split>-seed<seed>.md`，可用 --output 指定其他路径；同名报告更新。常规加载进度和问答正文不输出到终端，完成时仅提示保存路径。

## GSM8K 的解答格式与计算器

- 官方说明：`answer` 包含自然语言解题过程，最后一行的最终数值前有 `####`。这一标记用于提取答案，无须转成自然语言才能训练。
- `<<表达式=结果>>` 是计算器注释。原论文训练时将它们视为普通词元；推理时由外部程序检测表达式并填入结果，随后继续生成。仅在 SFT 数据中保留这些注释不会自动接通计算器。
- 官方给出的去注释方法是删除 `<<` 到 `>>` 的整个子串。例如 `2/2=<<2/2=1>>1` 清理为 `2/2=1`，外面的结果仍保留。
- 来源：[官方仓库数据说明](https://github.com/openai/grade-school-math#dataset-details)、[论文附录 C，第 17 页](https://arxiv.org/pdf/2110.14168#page=17)。
- 2026-10-26 离线检查本机 main/train：7473 条解答中，7378 条包含计算器注释，共 23716 处；所有解答均恰有一个 `####`；移除完整注释后没有残留的 `<<` 或 `>>`。
- 2026-10-26 用户最新决定题目与解答原样用于 SFT，保留计算器注释和 `#### 数值`，覆盖此前改成英文答案短语的决定。
- 沙箱内 `load_dataset` 首次失败：读取缓存也会在缓存目录创建锁文件，报只读文件系统。通过正式宿主授权后，同一离线统计命令运行成功；无须下载或 GPU。

## 代表性 SFT 实践与本机能力

- [HuggingFaceTB 的 Qwen3/GSM8K SFT 数据脚本](https://huggingface.co/HuggingFaceTB/qwen3-1.7b-gsm8k-sft/blob/main/scripts/prepare_combined_data.py) 的 `format_reasoning_gsm8k` 删除计算器注释和原始答案行，`create_gsm8k_example` 构造完整助手解答，并以英文 `The answer is ...` 给出最终答案。这是一个具体实例，不能据此宣称所有项目都采用同一格式。
- [MetaMath 论文附录 B](https://arxiv.org/pdf/2309.12284#page=18) 明确在训练解答末尾追加 `The answer is: {gold answer}`，评测时提取该标记之后的答案。
- [Qwen2.5-Math 官方实现](https://github.com/QwenLM/Qwen2.5-Math#quick-start) 区分 CoT 和 TIR，最终答案使用 `\boxed{...}`；CoT 例子可以直接通过 Transformers 生成，真正运行程序的 TIR 示例另接 Qwen-Agent。原始 GSM8K 则使用自己的计算器采样循环，因此外部计算器并非所有数学推理 SFT 实验的必需环节。
- [原始 GSM8K 计算器采样代码](https://github.com/openai/grade-school-math/blob/master/grade_school_math/calculator.py) 在生成过程中检查未闭合注释里的表达式，计算后填入结果再继续生成。该示例硬编码了旧 tokenizer 的部分 token ID，不能直接作为 Qwen3 的调用程序使用。
- 当前仓库 Python 文件没有接入计算器、工具调用或执行推理代码的控制流程；`main.py` 直接调用 `model.generate()`。已有终端对照分支也是直接生成。
- 2026-10-26 在 `/home/yunli/miniconda3/bin/python` 核对安装包：torch 2.6.0+cu124、transformers 5.20.2、datasets 5.1.0、accelerate 1.15.0、trl 1.16.0、sympy 1.13.1；该解释器未安装 peft、vllm、qwen-agent 或 math-verify。包清单说明当前解释器的安装状态，不代表其他环境或整台机器。
- 现有 Python 与 Transformers 已具备实现简单计算器控制循环的基础，但计算器接入仍需另写程序。SymPy 已安装不等于模型会自动调用它。
- 训练目标采用原始 `answer`，不追加提示词或转换文本；`train_sft.py` 直接复用样本构造，不接入计算器推理循环。
- 当前 Transformers 的 `apply_chat_template()` 在本机返回 BatchEncoding；不能把返回对象本身当作 token 列表。代码显式设置 `return_dict=True` 并读取 `input_ids`。
- 本机 Qwen3 模板将普通助手回答放在空 `<think>` 前缀之后。样本使用 `enable_thinking=False` 生成同样的提示前缀，并将整个前缀设为不计损失；原始解答及 `<|im_end|>` 参与监督。后续训练和推理沿用这一模板。
- 已实际预览第一条训练样本，解题过程、两处计算器注释和最后的 `#### 72` 原样出现在监督文本中。

## SFT 训练代码验证

- 2026-10-27 训练机首次实际运行：1.7B 分词器处理 7473 条 GSM8K 训练样本，最长 550 tokens；预训练权重下载并载入成功。原微批量 2、累积 8 配置完成 10 次更新后在反向传播 OOM，不能继续把该配置视为已适配此训练机。
- OOM 日志显示进程占用 31.20 GiB，PyTorch 分配 27.33 GiB、保留未分配 3.56 GiB，申请 464 MiB 时只剩 408.12 MiB。显存余量不足，同时存在较大的保留空间；保留量本身不足以证明全部问题都来自碎片。退出后 GPU 没有运行进程，显存已释放。
- 因该实测结果，训练机默认改为微批量 1、累积 16，保持单卡有效批量 16、FP32 权重和融合 AdamW。启动命令设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，用于缓解变长样本反复分配造成的碎片；参数适用性已在本机 PyTorch 2.6 的头文件和 [官方 CUDA 分配器说明](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf) 核对。此选项不会缩减权重或优化器状态，最终能否完成训练仍需训练机实测。
- 已验证本机 PyTorch 2.6 分配器接受该选项；新机器配置构造 TrainingArguments 后，CPU 随机微型 Qwen3 对长度 8～550 的变长合成样本完成 20 次更新，参数及 TensorBoard 指标均有限。开发机配置文件未改。上述检查验证配置和代码集成，不能作为实际训练机不再 OOM 的证据。
- 默认使用 Transformers Trainer 做全参数 SFT。2026-10-27 将训练机切换至 1.7B 后，减小微批量、增加梯度累积以保持单卡有效批量 16；启用非重入梯度检查点，优化器改为融合 AdamW。权重保持 FP32，GPU 支持时使用 BF16 混合精度，学习率和训练轮数保持此前设置，尚无评测结果支持另行调优。
- 本机 PyTorch 2.6 的 AdamW 在 CUDA 上默认可能使用 foreach，源码文档说明这会增加约一整组参数大小的峰值中间量；融合实现直接调用 `_fused_adamw_`，避免这项开销。约 1.7B 参数的 FP32 权重、梯度和两组 Adam 状态本身就需约 27 GB（十进制），还需留出混合精度副本、激活和运行时空间。上述设置是面向 32 GB 的起始配置，不能替代训练机实测。
- 模型与非思考模板核对来源：[Qwen 官方 Qwen3-1.7B 模型卡](https://huggingface.co/Qwen/Qwen3-1.7B)。优化器实现核对本机固定依赖的 `torch/optim/adamw.py` 与 `torch/optim/optimizer.py`；官方模型卡没有承诺本项目训练配置的显存用量。
- 2026-10-26 使用 0.6B 分词器离线核对全部 7473 条训练样本：监督部分解码后等于原始 `answer` 加对话结束标记，最长为 550 tokens。未截断、过滤或改写解答；这不是 1.7B 分词器的实测结果，训练入口会用所选模型的分词器重新统计长度。
- DataCollatorForSeq2Seq 在未传入编码器模型时可用于本脚本的动态补齐；实测保留已有题目掩码，并将补齐标签设为 `-100`，不产生 decoder 输入。
- 2026-10-26 使用 CPU 上的随机微型 Qwen3 验证 Trainer 集成：损失与手工移位后的交叉熵一致，反向传播梯度均有限，AdamW 初始化和保存/重新加载通过。这不验证真实模型在租用 GPU 上的运行显存；未启动真实训练。
- 2026-10-27 验证两台机器配置能构造 TrainingArguments，模型字段已剥离；用替代加载器确认训练入口对分词器和权重传入同一模型名称，预览入口默认 0.6B 且可显式选择 1.7B。CPU 随机微型 Qwen3 配合非重入梯度检查点完成反向传播，融合 AdamW 和 Adafactor 均完成一次有限值参数更新。命令行覆盖、无效 YAML/模型字段及非空输出目录保护通过；未加载预训练权重或数据集，未验证真实 GPU 峰值显存。
- 参考：[Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)。接口与功能验证以本机固定版本为准。

## TensorBoard 日志

- 2026-10-27 本机已安装 TensorBoard 2.21.0，现作为固定直接依赖加入 requirements.txt。训练机和开发机 YAML 使用 `report_to: tensorboard`，由 Transformers 内置回调记录 Trainer 产生的标量；当前训练曲线包含损失、学习率、梯度范数和轮次，结束时还有总体损失、耗时和吞吐量。
- 本机 Transformers 5.20.2 的 TensorBoardCallback 默认将日志写入 `output_dir/runs/<时间戳_主机名>/`，可随 `--output-dir` 自动隔离。该版本使用环境变量 `TENSORBOARD_LOGGING_DIR` 覆盖默认路径，而非 TrainingArguments 的 `logging_dir` 字段；本项目沿用默认路径。使用外部环境变量时需自行保持不同实验目录分开。
- 接口以固定版本源码 `transformers/integrations/integration_utils.py` 为准；参考 [官方回调说明](https://huggingface.co/docs/transformers/main_classes/callback#transformers.integrations.TensorBoardCallback) 和 [TensorBoard 查看方法](https://www.tensorflow.org/tensorboard/get_started)。TensorBoard 记录训练过程；答案正确率仍需独立评测。
- 已用两套 YAML 分别完成 CPU 随机微型 Qwen3 的 11 次更新，从真实事件文件读回 8 项主要指标且数值有限，核对曲线步数 1、10 和结束汇总步数 11；同时确认训练参数/模型配置文本写入、`--output-dir` 覆盖及两台机器的日志隔离。没有加载预训练权重或 GSM8K，也没有验证正式训练指标的数值表现。

## 训练机下载依赖

- 2026-10-27 用户提供 AutoDL 日志：`/root/miniconda3` 的 Python 3.12/base 环境在下载 Qwen3-1.7B 分词器文件时，HTTPX2 的 BrotliDecoder 调用 `process(..., output_buffer_limit=...)` 报参数错误。训练尚未开始；从堆栈推断，实际导入的 Brotli 实现不支持下载层要求的接口。训练机具体包版本尚未直接查询，不能将开发机版本作为训练机版本。
- 本机 HTTPX2 2.16.0 源码同样调用该参数；其发行元数据声明 CPython 的 Brotli 可选依赖为 `brotli>=1.2.0`。本机目前没有 Google Brotli，使用的是 brotlicffi 回退，因此此前离线模型缓存验证没有覆盖训练机这次下载错误。
- [Brotli 1.2.0 官方发布说明](https://github.com/google/brotli/releases/tag/v1.2.0) 确认 `Decompressor.process` 新增 `output_buffer_limit`；[Brotli 1.2.1 官方版本](https://github.com/google/brotli/releases/tag/v1.2.1) 也已发布。修复方向是在训练所用解释器中安装兼容 Brotli，无须改变模型或删除缓存。
- 项目依赖现固定 Brotli 1.2.1。仅在 /tmp 安装并验证其与本机 HTTPX2 2.16.0 的关键字接口、BrotliDecoder 和响应 `iter_bytes()`；小型 JSON 与 5.12 MB 数据的分块压缩/解压往返均通过。该验证覆盖报错的解压链路，不代表训练机实际安装结果或完整模型下载已经成功。
- 随后用户重试日志确认分词器、数据集和 1.7B 权重均已下载并进入训练，说明该次下载阻塞已解除；新的故障为上面的 GPU OOM。

## Git 权限核查

- GitHub 插件在本轮返回的账号为 FishSkin2002，但对 `FishSkin2002/qwen` 仓库查询返回 404。SSH 访问仓库正常，不能把插件的仓库访问失败理解为远端仓库不存在。
- 用户已安装本机 GitHub CLI 2.103.2 并完成登录，CLI API 核对账号为 FishSkin2002，可访问本仓库并创建 PR。后续仓库 PR 操作可直接使用这一已验证的 CLI 登录态，无须重新登录或使用浏览器。

- 原始 push、commit、switch 等命令前缀不能区分尾随危险参数，宽泛规则需撤销，不能仅靠文档声明保留历史。
- 新日常入口固定仓库、分支类型和参数，原始受限写命令不再自动放行；实际部署以用户规则层文件和受保护入口的校验结果为准。
- 来源：[OpenAI 命令规则说明](https://learn.chatgpt.com/docs/agent-configuration/rules)。
