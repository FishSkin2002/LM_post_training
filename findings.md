# 环境发现

## 原始模型加载错误（2026-10-28）

- 用户将仓库名写成 Qwen3-4B，导致 https://huggingface.co/Qwen3-4B/ 下的文件查找 404；本项目官方起点为 Qwen/Qwen3-4B，缓存同样依赖完整仓库名。
- 固定版本 Transformers 5.20.2 的 models/auto/auto_factory.py 先提取顶层 hub_kwargs，再从独立 adapter_kwargs 调用 find_adapter_config_file；只补充 revision，未继承 local_files_only。后者在 utils/peft_utils.py 中默认为 False，所以仅传顶层标志不足以保证探测离线。修复应显式传入 adapter_kwargs.local_files_only，不改第三方包或全局环境。
- 回归通过真实 AutoModel 调用到适配器探测，修复前确认该标志为 False；修复后原始模型与 LoRA 底座调用均为 True。实际模型权重加载仍以临时随机微型模型验证，没有网络请求或预训练权重下载。完整官方仓库名仍是立即修正用户 404 的关键，缓存模式不自动补全名称。

## 原始 Qwen3 抽样入口（2026-10-28）

- 需求是使用本项目尚未微调的起始模型生成回答，例如 Qwen/Qwen3-4B；原始模型仍按 CONTEXT.md 定义，不自动替换为名称中带 Base 的其他变体。
- 既有 sample_checkpoint.py 只接受 --checkpoint 本地目录，验证 config.json / adapter_config.json；完整模型或底座 + 适配器均通过离线加载。新入口保留此路径，另用 --model-name 读取模型仓库与分词器，默认允许下载，--local-files-only 限制到已有缓存，--revision 可固定原始模型版本。
- 从最新主线创建独立功能分支，PR #11 的功能已合并。本机 PEFT 0.21.0 现已安装在实际解释器中，验证无需继续依赖此前的 /tmp 隔离包路径。
- --model-name 已实现普通完整模型加载，不附加本项目适配器；模型与分词器支持缓存下载或仅缓存，并共用入口解析的版本。原始报告记录来源和版本信息，默认文件名保留仓库命名空间，避免与 checkpoint 的报告重名。
- 核对本机 Transformers 5.20.2 源码发现 config._commit_hash 已停止使用，当前通过 resolve_revision 得到 ResolvedRevision，实际提交在 .resolved 中。新增入口据此固定模型和分词器版本，不依赖旧私有字段；仅缓存标志同样传入版本解析。旧适配器的 revision 可能为空，对照时只有非空值才能直接用来固定原始版本。
- 种子在模型加载完成后设置。随机微型模型的原始加载、完整模型 checkpoint、零增量 LoRA 适配器三条路径实际生成一致，说明各加载过程对随机数的不同消耗不会改变对照的初始生成状态；这不是微调效果评测。

## LoRA 实施（2026-10-27）

- 用户已确认全部方案并授权实施：两机默认 PEFT LoRA，开发机 0.6B、训练机默认 4B / 可选 8B，保留全参数选项；适配器保存、同模式续训和抽样加载一并接入。参数与运行路径以对应 YAML 为准，量化训练与完整模型合并导出留待后续。
- PEFT 0.21.0 已通过正式网络授权下载至 /tmp/qwen-lora-deps，并与本机固定的 Torch 2.6.0 / Transformers 5.20.2 / Accelerate 1.15.0 完成导入及 CPU 随机微型模型集成验证；没有修改本机解释器已有包。requirements.txt 已固定 PEFT 版本，运行环境安装该依赖后即可使用。
- 已验证 LoRA 的 BF16 微型底座保持冻结、FP32 适配器梯度有限并更新，真实 Trainer 断点恢复后与不中断的适配器参数逐项完全一致。完整模型保存/加载、适配器保存/本地底座加载、模式/底座/参数不匹配拒绝、输出冲突避重及实际 TensorBoard 路径也已覆盖。
- 正式宿主授权下，随机微型模型在本机 RTX 4060 上分别完成 BF16 自动路径和 FP16 回退路径的两次更新；适配器保存/加载后参数有限，memory_results.json 的权重精度、FP16开关和 CUDA峰值字段均通过断言。该 GPU 检查仍是随机微型模型验证，不是 4B / 8B 资源测量。
- --memory-smoke-test 重复最长真实训练样本，保留完整梯度累积，默认两次更新且每次保存 checkpoint。memory_results.json 记录实际模式、精度、最长样本、可训练参数和 CUDA 峰值；CPU 验证不报告 GPU 峰值。真实 4B / 8B 短测仍由用户在训练机执行，本机验证不保证其实际显存或训练效果。

### 方案评估依据（历史记录）

- 当前训练入口为全参数 SFT：共享配置明确使用 FP32 权重；训练机启用 BF16 自动混合精度、融合 AdamW、非重入梯度检查点，微批量 1、累积 16。BF16 混合精度不等于权重、梯度与 Adam 状态全部采用 BF16；应结合固定依赖实现计算显存。
- 约 1.7B 参数按 FP32 权重、FP32 梯度及两组 FP32 Adam 状态估算，仅这四项就约 27.2 GB / 25.3 GiB；激活、混合精度副本与运行时空间另外计算。这是内存预算估算，不是训练机测量。
- 已有记录包含微批量 2 时的明确 OOM 数值，以及用户随后再次 OOM 的事实；当前微批量 1 的完整报错和 BF16 开关实值没有可核查记录。已知训练机为 NVIDIA vGPU-32GB，底层架构未知；本机 RTX 4060 不能替代训练机兼容性核对。
- LoRA 冻结底座参数、训练低秩适配器；QLoRA 进一步把冻结底座的主要线性层量化存储，反向传播仍经过底座计算。二者均不消除激活与词表输出内存，QLoRA 也不代表整个训练过程都以 4 bit 计算。
- 本机 Torch 2.6.0 的 Adam 状态以 zeros_like(p) 创建；Accelerate 1.15.0 用 autocast 包装 forward，支持上述 FP32 固定状态预算。现有 SDPA、use_cache=False、微批量 1 与梯度检查点已经启用，不能作为新增改动重复计入收益。
- 当前 requirements.txt 与本机解释器均无 PEFT / bitsandbytes；普通 LoRA 需增加前者，QLoRA 另需量化后端、k-bit preparation 与对应 GPU 兼容验证。固定依赖尚未与候选新依赖做运行验证。
- 本机 Transformers 5.20.2 Trainer 包含 PEFT 保存与恢复分支，但须先构造匹配的底座和适配器再恢复。现有 sample_checkpoint.py 要求完整 config.json 且按完整模型加载；原生 adapter 产物需配套修改加载路径，或明确提供合并后的完整导出。原全参数优化器 checkpoint 不能等价续接为 LoRA 断点。
- [PEFT checkpoint 文档](https://huggingface.co/docs/peft/developer_guides/checkpoint)确认适配器权重不包含底座；本项目的本地离线抽样还须确保对应底座已经缓存。[PEFT 精度说明](https://huggingface.co/docs/peft/developer_guides/troubleshooting#selecting-the-dtype-of-the-adapter)说明适配器默认通常保留 FP32，可与 BF16 冻结底座区分。
- 普通 LoRA 建议优先：先去除冻结底座的全模型梯度与优化器状态，若训练机支持 BF16，再把冻结底座按 BF16 加载。QLoRA 在此基础上进一步压缩主要线性层的权重，引入量化误差与后端验证；没有实测证据保证本项目数学正确率不变或训练更快。底座按 BF16 保存时，约 1.7B 参数的纯权重预算约 3.4 GB，另加适配器、激活、logits 与运行时空间；不将该数字当作训练峰值。
- 后续显存验收应覆盖实际最长样本、梯度累积、优化器更新和保存/恢复，并记录 peak allocated / peak reserved；只用短样本成功启动不足以验收。正式训练和数学正确率评测仍需按后续小步授权。
- 用户接受普通 LoRA 优先、QLoRA 后备，并保留全参数训练为显式配置选项；该轮之后的参数与产物决定现已收口，见本节实施状态。
- 一手参考：[PEFT LoRA 指南](https://huggingface.co/docs/peft/developer_guides/lora)、[Transformers bitsandbytes 指南](https://huggingface.co/docs/transformers/quantization/bitsandbytes)、[QLoRA 原论文](https://arxiv.org/abs/2305.14314)。适用接口仍以本项目固定依赖和候选 PEFT / bitsandbytes 版本核对为准。

### 软件包分工与当前接入建议

- PEFT 负责参数高效微调方法，提供 LoRA 配置、适配器注入以及适配器保存和加载；它可与不同训练器组合使用。[官方说明](https://huggingface.co/docs/peft/index)
- Transformers 提供当前项目所用的模型、分词器和 Trainer。建议保留现有样本与 labels 路径，用 PEFT 包装模型后交给当前 Trainer，减少为解决显存问题引入的迁移。
- TRL 提供 SFT、GRPO、DPO 等后训练训练器，并集成 PEFT；采用 TRL 与采用 PEFT 可以同时发生。当前 SFT 不必为增加 LoRA 而迁移训练器；后续 GRPO 再按用户授权评估 TRL。[官方说明](https://huggingface.co/docs/trl/index)、[PEFT 集成](https://huggingface.co/docs/trl/peft_integration)
- verl 侧重强化学习训练流程与训练、生成后端的协作，支持分布式扩展。对当前单卡 SFT 的 LoRA 接入，没有必要先引入其整套编排与依赖。[官方文档](https://verl.readthedocs.io/en/latest/)
- 用户已认可 Transformers Trainer + PEFT 的组合；后续若启用 QLoRA，按已评估的方案再增加 bitsandbytes 量化后端。尚未锁定新增依赖版本或完成兼容性验证。

### 第二轮候选方案（已被后续两机与大模型需求修订）

- 训练配方建议：训练机 Qwen3-1.7B 的注意力和 MLP 线性层使用 LoRA，冻结 embedding / 输出层等底座权重；r=16、alpha=32、dropout=0.05、学习率 1e-4，作为起始实验配置，不宣称是最佳参数。沿用微批量 1、累积 16、3 轮、检查点及既定思考段监督；本轮不新增长度截断或调整数据。覆盖范围可按 PEFT all-linear 机制核对。
- 精度建议：训练机支持 BF16 时采用 BF16 冻结底座和 FP32 适配器；不支持时明确使用 FP32 LoRA 路径并记录实际精度。FP32 全参数模式保留原配置。开发机维持原训练默认，仅适配共同的配置和抽样入口。
- 产物建议：checkpoint 保存适配器及必要优化器/调度器状态以支持同模式续训，final 保存适配器、分词器与对应底座标识；抽样入口同时支持完整模型和底座 + 适配器。底座必须可取得，仅下载适配器不能独立推理；本轮不额外提供合并完整模型的导出工具。新的 LoRA 实验从对应官方底座开始，不混用旧全参数优化器断点。
- 验证分工建议：先由 Codex 完成随机微型模型的梯度、底座冻结、保存恢复与配置集成验证，并提供训练机短测入口记录显存峰值；实际 1.7B / 32 GB 短测由用户在训练机执行。短测应覆盖实际最长样本、完整累积与优化器更新，并核对保存恢复，不直接启动 3 轮正式训练。短测通过不代替完整训练的稳定性验证。

### 两机 LoRA 与 4B / 8B 补充评估

- 用户已明确两台机器均默认 LoRA，并接受 Codex 做实现与微型模型验证、用户在训练机执行真实模型短测；随后已确认训练机默认 4B / 可选 8B，以及适配器产物。
- 2026-10-27 正式宿主授权下重新查询开发机：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB 总显存、581 MiB 已用、7376 MiB 空闲；空闲值只代表查询时刻。训练机预算仍为用户给定的 32 GB，不能用本机空闲值代替。
- 官方混合思考模型 [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) 为约 4.0B 参数，[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) 为约 8.2B 参数；两者均支持 enable_thinking，不能将其他 Instruct / Thinking 变体混为同一模型。BF16 纯权重预算分别约 8.0 GB 与 16.4 GB，不包含适配器、梯度、优化器、激活或运行时空间。
- 当前约 550 tokens 的 SFT 样本、微批量 1 与梯度检查点下，32 GB 的 8B 普通 LoRA 具有可评估的内存空间，不能仅按全参数 OOM 或模型名认定其必然勉强；长序列、批量扩大及后续 GRPO 需另外预算。这仍是预算判断，须训练机短测，不能承诺总峰值或速度。
- 适配器产物更小属于磁盘收益；训练显存收益来自冻结底座及只为可训练参数保留梯度/优化器状态。模型完整导出会增加磁盘和导出时的内存开销，但不会倒推改变已经发生的训练显存；底座 + 适配器与完整模型是两种产物表示。
- 已核对官方 [4B config](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json) 与 [8B config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)：分别为 36 层、hidden_size 2560 / 4096，4B 共享 embedding 与输出权重，8B 不共享；LoRA 参数量需按各线性层实际形状计算。
- 原先“开发机保留全参数默认”的候选建议已被用户两机 LoRA 决定替代；原先“不支持 BF16 时退回 FP32 底座”的候选策略不适合直接推广到 8B，因其纯 FP32 权重已约 32.8 GB。大型模型应核对实际 GPU 低精度支持并明确精度路径。
- 子代理依据官方配置，用本机 Transformers 5.20.2 在 meta 设备构造空模型计数，未下载权重或训练：0.6B / 4B / 8B 分别为 596049920 / 4022468096 / 8190735360 参数，BF16 底座约 1.110 / 7.492 / 15.256 GiB。
- 七类注意力与 MLP 线性层、r=16 时，适配器分别为 10092544 / 33030144 / 43646976 参数；适配器权重、梯度和两组 Adam 状态均 FP32 时，合计约 0.150 / 0.492 / 0.650 GiB。计算为每个线性层 r × (输入维度 + 输出维度)，再按层求和；上述固定项不包括激活与运行时空间。
- 统一参数建议沿用 r=16、alpha=32、dropout=0.05、学习率 1e-4、微批量 1、累积 16、3 轮和非重入检查点。相同 rank 便于两机检查同一套路径；8B 改为 r=8 只削减约 0.325 GiB 适配器训练状态，不能将其视为解决多 GiB 缺口的主要手段。具体数值为起始实验设置，不构成效果或峰值承诺。
- 用户已接受开发机仍用 0.6B，训练机先选 4B 为默认起点，8B 保留为可选规模；4B 给激活与后续配置调整留下更多空间。
- 以 4B / r16 为例，纯 FP32 适配器权重约 126 MiB，而 BF16 底座约 7.492 GiB；训练 checkpoint 还包含优化器与其他状态。每次实验保存适配器可以复用缓存中的同一底座，若导出合并模型则需要另存完整权重。
- 0.6B / 4B / 8B 适配器依附不同底座，不能互相套用。4B / 8B 的 BF16 推理也不适合当前 8 GB 开发机直接执行；建议先在训练机抽样并下载 Markdown 报告。本机对大型模型的量化推理可作为另一小步，不因启用 LoRA 自动获得。

## 当前思考 SFT 与运行目录适配

- 2026-10-27 用户选定当前 SFT 格式及监督范围，规范见 [思考模式约定](docs/qwen3-thinking.md#sft-样本与损失掩码)。用户进一步说明忽略 #### 指不指导模型生成它；当前通过分词前去除分隔符，使其没有对应的词元损失项，原始参考字段保留。
- 已用本地 0.6B checkpoint 分词器离线核对 GSM8K train 全部 7473 条：监督文本精确等于格式化的推导、最终结果及助手结束标记，最长 548 tokens。推导、计算器注释与最终结果未改写，题目/前缀和 padding 标签均排除。没有本地 1.7B 分词器缓存，未下载验证该规模。
- 对两条真实样本进行词表的一一映射后，用随机微型 Qwen3 在 CPU 上核对移位交叉熵与手工计算一致，反向梯度有限；这只验证标签与损失集成，不验证真实模型训练效果或显存。
- 新训练目录由 train_sft.reserve_output_dir 原子创建。已验证已有运行目录、空目录、同名文件、八次并发分配、续训默认 checkpoint 父目录及显式输出覆盖；旧目录标记文件内容保持不变。
- 训练入口使用替代数据/模型加载器及 Trainer，验证实际加后缀目录贯穿 TrainingArguments、final 模型路径和训练汇总；真实 TensorBoard 回调事件也写入该目录的 runs/，未加载预训练权重。
- 抽样入口同步已合并的官方思考模式约定。替代生成器验证采样参数、思考/最终输出分段、缺少闭合标记时保留内容、EOS 与长度边界、上下文预算检查和终端仅提示文件路径；未运行真实问答。

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
- 研究时建议以“推导与最终输出都监督，问题与 padding 屏蔽”作为思考 SFT 基线；用户随后采纳并进入上面的适配步骤。没有实验结果证明其相对效果。

## Qwen3 思考模式与损失范围

- 用户于 2026-10-27 确定后续各规模 Qwen3 遵循官方思考模式最佳实践；长期约定见 [思考模式约定](docs/qwen3-thinking.md)。当前 SFT 与抽样入口已适配，已有 checkpoint 未重新训练。
- Qwen 官方 ms-swift 指南中的 ignore_empty_think 指屏蔽空思考块，不是屏蔽非空推导。用户随后选定同时监督推导与最终结果，没有采用屏蔽非空思考段方案。
- 已核对当前 Transformers 的因果语言模型损失：先构造完整 logits，再由交叉熵忽略 labels=-100 的位置；它没有将这些输入位置移出注意力上下文，也没有按标签掩码切断此前隐藏状态到最终输出的计算依赖。
- 文档站训练页面无法访问，改用 QwenLM/Qwen3 官方仓库的 [ms-swift 训练说明](https://github.com/QwenLM/Qwen3/blob/main/docs/source/training/ms_swift.md) 交叉核对。

## 默认训练环境

- 用户于 2026-10-27 进一步确定当前长期默认：**训练机 Qwen3-4B LoRA、开发机 Qwen3-0.6B LoRA，训练机可选 Qwen3-8B LoRA**，替代此前的 1.7B 全参数训练默认。用户显式指定其他模型或 full 模式时使用指定值；可执行模型名称与训练参数保存在对应 YAML 中。
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
- 最初的训练目标采用原始 `answer`，不追加提示词或转换文本；本轮已改为上述思考格式。`train_sft.py` 继续复用样本构造，不接入计算器推理循环。
- 当前 Transformers 的 `apply_chat_template()` 在本机返回 BatchEncoding；不能把返回对象本身当作 token 列表。代码显式设置 `return_dict=True` 并读取 `input_ids`。
- 旧非思考实现使用 `enable_thinking=False` 生成空 think 前缀，整个前缀不计损失；本轮改为思考模式，非空推导和最终结果参与监督。旧 checkpoint 的训练格式没有因此改变。
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
