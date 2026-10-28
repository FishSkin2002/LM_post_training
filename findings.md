# 环境发现

## 验证耗时实测与提示缺口（2026-10-28）

- 训练机本次 checkpoint-500：eval_generation_runtime=7785.0046844 秒，eval_runtime=7851.8661 秒，128 题平均约 60.8 秒/题，生成占总验证时间约 99.1%。截断率 0.9375 表示 120 题完整使用 1024-token 预算；这解释了长时间生成，而非 loss 计算缓慢。
- 原 evaluation_loop 在原生 loss 进度条结束后执行普通 Python for 循环串行生成，期间没有进度输出。最小复现和正式回归在生成首题前检查可见提示，原实现确定失败，加入 tqdm 后通过；这是可见性修复，不是生成速度优化。
- Trainer 的 _maybe_log_save_evaluate 先完成 evaluate 再执行 _save_checkpoint。停在某个更新步的生成阶段时，该步 checkpoint 尚未完成保存；不能把已经显示 step1000 当成已有 checkpoint-1000。
- loss 输入为题目与标准 answer 的完整序列，用因果掩码在一次前向中并行计算各位置预测误差；自由生成只能按生成前文逐 token 推进。低 teacher-forcing loss 与早期自由生成格式/正确率为零可以同时出现。
- 评测工作量候选修改仍未授权：定期固定 32 题/每 2000 步，保持每 500 步 loss/保存；结束全量已确认保留。当前修复保留 128 题/每 500 步、1024-token 上限、完整终评及原来的恢复协议。

## 训练运行实施（2026-10-28）

- PR #16 已合入 35dee1c，功能分支 codex/training-workflow。以原生 Trainer 的 evaluation_loop / _save 扩展少量记录逻辑，没有引入训练框架或额外显存策略。
- run.json 随运行根目录、checkpoint 和 final 保存，包含配置、有效阶段轮次、固定划分索引、数据指纹、模板、生成设置及来源。续训使用保存的有效轮次并校验原配置；加训沿用来源数据，只继承一套适配器，重新建立优化器与调度。
- 验证全部 loss 后计算生成指标，一并进入该步 Trainer 日志与 checkpoint 状态。末步全量生成时仍单独记录固定子集结果，避免不同题数混入同一准确率曲线。
- 固定 runs/ 下使用原生 SummaryWriter 与 TensorBoardCallback；恢复步为 N 时 purge_step=N+1，保留断点步和此前有效数据，排除旧尾部。微型测试额外写入 step777 的旧事件，恢复后训练曲线严格为 step1..18，旧事件被排除。
- 加训微型测试确认初始 global_step0、优化器状态为空、学习率取新配置、适配器只有一套且初始权重等于来源；随后可从该阶段自己的断点恢复到追加轮次终点。
- 最初十项测试仅旧断点文件读取断言失败：save_total_limit=2 已正确删除 checkpoint-2，改为验证保留的 checkpoint-18 后十项通过。该错误属于测试对保存策略的适配，实际恢复逐参数一致比较已通过。
- 最终扩展为 11 项集成检查并全部通过。加训 0.5 轮在 step9 正常结束，虽然保存周期为 2，仍生成全量验证指标并保存 checkpoint-9；从加训自身断点恢复后 TensorBoard 步数为 1..9，未继承前阶段步数。
- 真实缓存 Arrow 只读核验：7473 条参考答案均可严格解析，固定划分为 train6725 / validation748，索引无交集、完整覆盖且可复现；固定生成子集取验证顺序的前 128 条。三个入口的实际帮助、README 参数与链接、TensorBoard 帮助以及三套配置默认值均已核对。

## 训练运行功能事实审计（2026-10-28）

- PR #15 已合入 a9731f8；当前要求分支 codex/training-workflow-policy。用户本轮明确需要续训、TensorBoard 训练/验证日志、目录自动后缀、训练后加训和 README 重组；Base + 简洁 ChatML + FP32 LoRA 保持。
- 现有 tools/test_base_sft.py 的恢复测试是 callback 正常提前停止并保存后从 step1 续到 step2，适配器逐元素等同连续训练。没有模拟进程强杀或保存中断；当前每 epoch 保存一次，中途故障只能回到已保存断点。
- 本机 Transformers 5.20.2 对缺 optimizer.pt / scheduler.pt 的 checkpoint 会跳过相应状态恢复，当前入口只检查 Base ID，因此需增加完整状态预检。final 只有适配器与分词器，不是完整恢复状态。
- 当前恢复输出目录仍取 YAML；从带数字后缀的运行恢复应沿用 checkpoint 父目录。新训练数字避重和恢复旧目录是不同路径，应分别验收。
- 本机 TensorBoardCallback 已提供 train/loss、train/grad_norm、train/learning_rate、train/epoch 和 eval/loss、耗时/吞吐标量，使用 state.global_step。grad_norm 来自 Trainer._clip_grad_norm，语义为裁剪前范数，不是区间平均梯度。
- 固定版 TrainingArguments 已没有 logging_dir 参数。默认 TensorBoardCallback 每次开始建立 output_dir/runs/时间戳主机名，续训会成为另一个子 run；可直接向原生 TensorBoardCallback(tb_writer=...) 提供固定路径的 SummaryWriter，并通过 purge_step 处理较早断点之后的失效事件，同时避免重复注册回调。
- DefaultFlowCallback 支持 steps 验证/保存同频，正常到 max_steps 的最后一步即使不是周期倍数也会触发验证及保存；日志顺序是训练日志、验证、保存。原生 evaluate 不传 compute_metrics 时只计算 loss，不保留全量 logits。
- GSM8K 本地 dataset_info.json 只有 train=7473、test=1319，没有 validation。候选方案为从 train 固定划出 10%、seed42，test 留作最终评测；验证损失是标准解答条件下的预测损失，不是生成答案的数学正确率。数据与指标、周期尚待用户确认。
- PEFT 原生支持 PeftModel.from_pretrained(..., is_trainable=True) 从 final/适配器权重开启新训练；新建 Trainer 且 train 不传 resume 会新建优化器、调度器和 TrainerState。无需叠加另一套 LoRA 或引入框架。候选接口 --continue-from 与 --additional-epochs，阶段语义等待用户确认。
- 直接把完成 checkpoint 的总 epoch 增大再恢复会混用旧优化器学习率与按新总步数建立的调度函数。当前线性调度在完成时到0，下一次更新可能先用旧0学习率再按新计划变化；不能把它与清晰的新阶段加训视为等价。新阶段应保留来源验证划分，避免以前已用于训练的样本变为验证样本。
- README 已明确标题和章节顺序、每用途一条命令及紧邻可选参数说明，已独立记录规则；实际 README 在功能实现后按真实接口重组，不先宣称待实现功能已可用。
- 用户已选定每 500 步验证保存、训练每 10 步记录、保留最近两个断点，以及重置优化器/调度/步数的新阶段加训。验证必须包含生成答案正确率；已选定每次固定 128 条、结束全量生成，loss 始终全量。沿用现有贪心与 1024-token 上限；全量和子集结果需分开标识。
- 只读 Arrow 核验 GSM8K 标签：train 7473 条中 7394 普通整数、79 带千位逗号，负数 3 条；test 1319 条中 1305 普通整数、14 带逗号，负数 2 条。全部解答只有一个 ####，没有小数/货币/单位结尾。未写缓存锁或加载模型。
- 用户已选严格格式与数值匹配，允许生成的 5.0 与参考 5 相等；缺标记、多标记、额外尾部文字、非法数字和截断均计错，正确率分母包含全部评测题。同时记录格式有效率和截断率。此规则是项目协议，不是复刻官方字符串匹配。
- [OpenAI GSM8K evaluator](https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py) 使用正则取首个 #### 后数字、去逗号后字符串比较；[lm-eval GSM8K 配置](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml) 区分 strict-match / flexible-extract，并有不同提示和停止配置。本项目不直接把内部验证分数称为榜单可比成绩。
- sample_checkpoint.generate_answer 已返回文本、生成 token 数和截断状态，可复用其题目输入、停止与预算协议；它不负责 model.eval()，验证调用方需负责评估模式并在结束后恢复训练状态。

## 最小 Base + LoRA 实施（2026-10-28）

- PR #14 已合入 e0dc862，功能分支 codex/minimal-base-lora。旧实现已按约定文件边界撤下，独立撤销提交 48e05c8 保留历史；未对已有 outputs 或数据执行修改。
- 新业务仅 prepare_sft_data.py、train_sft.py、sample_checkpoint.py，共 245 行。配置只含 Base 名称、输出目录、基础训练参数和 LoRA 参数；固定 FP32，Trainer 原生保存恢复，无旧 full 模式、精度分支、额外优化、显存统计或 TensorBoard。main.py / sft_runtime.py / common.yaml 不再存在。
- 本机 Transformers 5.20.2 的 apply_chat_template 默认 return_dict=True，与此前常见的默认列表行为不同。新数据构造显式 return_dict=False，并检查题目 token 是完整训练样本的精确前缀，避免掩码错位。
- 新模板复用 im_start / im_end，末条消息后无额外换行。答案保持原文与 ####，监督直到 im_end；生成停止包含 im_end 与 Base 原生 EOS，正常结束及长度截断分别记录。默认贪心、输出预算 1024，是项目设置而非官方最优声明。
- 五组 CPU 随机微型模型测试已通过，真实 Trainer 训练/保存/恢复后的 LoRA 参数与连续训练逐项完全一致，底座权重不变；零增量适配器与原始随机模型抽样一致，助手结束标记能触发真实生成停止。测试将官方仓库读取替换为临时随机权重与合成数据，未下载或加载预训练权重。
- 实际缓存分词器的一条完整解答回译已核实；全量 GSM8K 缓存读取需写锁文件，首次沙箱拒绝后已按正式审批重试。这里使用已有普通0.6B缓存分词器核查新格式的真实词元边界，不将其称作 Base 权重测试。
- 7473 条真实缓存训练样本全部通过原文回译与掩码长度核验，最长 545 token。实际 Base 模型训练、GPU 峰值和数学正确率仍未验证；相应运行由用户在后续小步开展。

## 最新决定：简单的 Base + LoRA 路线（2026-10-28）

- 用户已确认简洁 user / assistant 模板、原样标准解答及结束标记监督；同时要求当前只用 LoRA 节省显存，优先整个项目简单。此前的混合精度、激活检查点、累积等保留建议已被本轮指示覆盖。
- 已确定要求集中到 docs/qwen3-base.md。以固定 FP32 和常规 Trainer 训练为基线，不加入额外显存策略或自动精度选择；只保留 LoRA 路径、必要数据/保存/抽样能力，现有业务实现尚未变化。
- 角色模板使用已有 im_start / im_end 标记，answer 全文含 ####；用户、回答前缀和 padding 掩码为 -100，生成停止与助手结束标记一致。旧的原生 Base EOS 纯文本候选及 think 目标均不再是当前约定。
- 本轮删除讨论草案与旧思考规则，更新 AGENTS 指针、术语与项目一目标。研究记录保留历史选择与依据，顶部本节及新约定表示当前决定；未对 outputs 或 Git 历史执行删除。

## Base 角色模板与当前显存措施（2026-10-28）

- [Transformers 聊天模板说明](https://huggingface.co/docs/transformers/chat_templating)明确：聊天与非聊天因果模型都续写词元序列；在 Base 上以角色标记格式做 SFT 是合理路线，模板是数据序列化规则，不要求起点已经过官方聊天后训练。它不是所有数学实验的唯一标准。
- [Qwen3-4B-Base 实际 tokenizer 配置](https://huggingface.co/Qwen/Qwen3-4B-Base/blob/main/tokenizer_config.json)中的 assistant 分支会包装 think 内容，即使原 content 只含标准解答，也会产生空思考块；enable_thinking=False 的生成前缀同样带空思考块。不能把“用官方模板”直接描述为只添加 user / assistant 边界。
- 当前推荐候选是简洁 ChatML 角色模板，使用已有 im_start / im_end 标记，不新增词表项，不加入 think 包装；assistant 内容为原始 answer（含推导、计算器注释、#### 和最终结果），只监督其内容及结束标记。训练/抽样同模板，生成停止须包含 im_end；本建议仍待确认，已替换纯文本候选草案。
- 当前实际训练代码（尚未 Base 重写）：train_sft.make_training_args 在 CUDA LoRA 下按支持情况选择 BF16 底座与 BF16 AMP，或 FP16 底座与 FP16 AMP；sft_runtime.attach_lora 保留 FP32 适配器，因此不是所有训练张量都采用半精度。CPU 验证采用 FP32。
- configs 三套训练参数均为 microbatch=1、gradient_accumulation_steps=16，gradient_checkpointing=True / use_reentrant=False；单卡有效批量为 16，省显存来自较小微批量，累积不消除模型状态占用。检查点以反向重算换较少的激活保存，不把 use_reentrant=False 本身当作独立显存优化。
- COMMON_CONFIG 指定 attention_implementation=sdpa；是否调度到 Flash 或其他内核受设备、形状、精度和 PyTorch 版本影响，项目没有显式启用 flash_attention_2。训练时 model.config.use_cache=False，推理时才重新启用缓存。
- DataCollatorForSeq2Seq(padding=True) 只补到当前批最长序列，避免预先补到全局固定长；当前批量 1 时基本没有批内 padding。prompt / padding 的 labels=-100 是监督范围，不代表这些位置免于前向和注意力计算。
- README 推荐 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 以缓解碎片，训练脚本没有自动设置；未核实用户远程进程是否实际导出该环境变量。它不减少模型理论张量需求。optim=adamw_torch_fused 主要用于执行效率，适配器 Adam 状态依旧 FP32。
- 当前 requirements 和训练路径没有 QLoRA / bitsandbytes、8-bit 优化器、CPU/NVMe offload 或 ZeRO / FSDP 分片。Base 重写尚未执行，不能把这些现有机制描述成新的 Base 实测结果。

## Base 全面重写前的边界核查（2026-10-28）

- 用户已选择 Base 唯一主线并要求先追加撤销提交、再重写旧代码；该选择覆盖此前“起点待定”和“保留旧普通版兼容”的候选方案。历史产物未获删除要求。
- 当前 HEAD / origin/main 为 3f09a13（PR #13 merge）。最后功能提交 d7bb775 只修复仓库名称提示与 local_files_only 传递；撤销该修复不会消除聊天模型依赖。受控 revert 入口仅支持非合并提交，不能直接对 merge commit 指定 mainline；本轮未执行撤销。
- 根提交 daab1ee 的 main.py 已写死 Qwen/Qwen3-0.6B 并调用 apply_chat_template；早期 SFT 也使用普通版模型。因此不存在一个历史版本可直接用作 Base 项目。
- 41ad57a 同时引入 think 格式与输出目录避重，33a81f1 同时引入 LoRA、低精度冻结权重、保存恢复、短测及普通版配置。通用工程能力与旧模型假设混在这些历史提交中。建议按旧业务文件边界形成一次明确撤销提交，再另提交 Base 重写；既有 Git 约定和独立工具不回退。
- [Base 官方 tokenizer_config](https://huggingface.co/Qwen/Qwen3-4B-Base/blob/main/tokenizer_config.json) 当前含 chat_template，eos_token 为 endoftext。更准确的区别是预训练权重尚未学会该聊天格式；是否调用模板是项目选择，不是“Base 只能输出 token、聊天版不是 token”的区别。两者都是因果语言模型。
- [Transformers 因果语言模型文档](https://huggingface.co/docs/transformers/tasks/language_modeling)说明训练预测下一个 token。纯文本 SFT 应将题目提示和标准解答拼接，屏蔽提示与 padding，只在标准解答及 EOS 上计算词元交叉熵；标签保持与输入等长，由模型移位。预先 generate 再比较离散 token / 字符串不属于当前要实现的标准 SFT。
- 当前建议的 Question/Answer 前缀、保留原 answer 中 #### 与计算器注释、追加 EOS 已写入可评审草案。保留 #### 与此前“移除 ####”的约定不同，等待用户确认后更新权威要求文档；没有直接修改业务格式。
- 第一次通过网页工具读取 Base 原始 tokenizer_config / generation_config 链接不可访问，随后以官方 blob 页面核实模板和 EOS；未下载权重或执行模型。

## GSM8K 实验起点调查（2026-10-28，已完成代表性核查）

- 调查口径：区分实验直接起点与训练后成品；RL 起点与整个流水线最初的 Base 也分别记录。GSM8K 仅作评测的论文不能据此视为在 GSM8K 上微调。论文中的 base model 可能只表示当前对照或起点，分类需看实际 ID 和模型来源。
- 结论：仅预训练起点与已有后训练起点均有明确的一手实例，SFT 并不要求起点未经官方后训练，RL 的直接起点也不能由整条流水线的起点推断。此前推荐 Base 应限定为服务完整阶段研究的候选路线，不能将现有普通版实验称为模型选错，或声称改用 Base 必然更容易提升。本调查没有修改选型决定。
- 调查是目的性选择的代表性来源，覆盖论文与官方框架示例，不用于估计全领域的数量占比。框架提供脚本不等于一篇独立效果论文；不同训练数据、提示、输出预算、答案提取和采样次数的分数不直接横比。
- 已核实：[MetaMath 作者仓库](https://github.com/meta-math/MetaMath) 的训练命令是 meta-llama/Llama-2-7b-hf，而非 chat；MetaMathQA 是 GSM8K / MATH 训练题增强数据，不是只使用原始 7473 条。
- 已核实：[DeepSeekMath 论文](https://arxiv.org/html/2402.03300v3) 第 3 节从 DeepSeekMath-Base 7B 做数学 SFT，数据含 GSM8K、MATH 等；第 4.2 节 GRPO 的直接起点为 DeepSeekMath-Instruct 7B，RL 使用 SFT 数据中的约 144K 条 GSM8K / MATH 相关问题。整条流程起点为 Base，不能把 GRPO 直接起点也归成未经 SFT 的 Base。
- 已核实：[verl 官方算法基线](https://github.com/verl-project/verl/blob/main/docs/algo/baseline.md) 明确记录 google/gemma-2-2b-it 的原模型 / SFT / SFT+PPO，以及 Qwen/Qwen2.5-0.5B-Instruct、1.5B-Instruct、3B-Instruct 等 GRPO-LoRA 实验；已有后训练版本也被用于完整的 SFT 再 RL 比较。
- 已核实：[SkyRL 官方 GSM8K 快速开始](https://docs.skyrl.ai/docs/getting-started/quickstart) 将策略初始化为 Qwen/Qwen2.5-1.5B-Instruct，直接进行 GRPO；这是框架示例，不视为一篇独立效果论文。
- 已核实：[verl 当前 Qwen3 GRPO 脚本](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_8b_fsdp.sh) 默认 MODEL_PATH 为 Qwen/Qwen3-8B，训练数据为 gsm8k/train.parquet + math/train.parquet，adv_estimator=grpo。这是普通混合思考后训练版，不是 Qwen3-8B-Base；脚本可说明公开用法，不能当作本项目单卡 32 GB 配方或已复现收益。
- 已核实：[ReFT 论文](https://arxiv.org/html/2401.08967v1) 第 4.3 节使用 Galactica-6.7B 和 CodeLlama-7B，原始起点为预训练模型；[官方 GSM8K SFT 脚本](https://github.com/lqtrung1998/mwp_ReFT/blob/main/exps/paper_exps/SFT/gsm8k.sh) 明确 CodeLlama-7b-hf / galactica-6.7b。ReFT 先做 SFT 预热，再对同一训练题做 PPO。解题标注包含 GPT-3.5-turbo 生成的自然语言及程序推理，不能误认为教师 GPT-3.5 就是训练起点，也不能直接等同当前原始 GSM8K 解答。另有 125M 模型实验，此处主表只列主要模型。
- 已核实：[CoT Prompting 论文](https://arxiv.org/html/2201.11903v6) 第 3.1 节列出 PaLM 8B / 62B / 540B 及其他模型，通过提示提供少量示例，不做任务梯度更新；GSM8K 在这里用于推理方法评测。论文还涉及作者认为对应 InstructGPT 的 API 版本，因此不能把所有列出的 GPT 模型笼统归成纯预训练模型。主表中的训练路线与此类提示实验分开说明。
- [Qwen3-8B 官方模型卡](https://huggingface.co/Qwen/Qwen3-8B/raw/main/README.md) 明确标注 Pretraining & Post-training，支持对 verl Qwen3 默认脚本的归类；[Gemma 2 官方模型卡](https://huggingface.co/google/gemma-2-2b-it) 区分 pre-trained 与 instruction-tuned 版本，verl 的 it 起点属于后者。
- 调查中 MetaMath 原始 README.md 请求 404，作者仓库实际文件名为 README.MD，已通过仓库正文获得原始命令。所有研究仅访问文本来源，未加载模型、执行训练或修改运行配置。

### 代表性训练路线对照

| 来源 | 训练起点与类别 | GSM8K 用途与后续阶段 |
| --- | --- | --- |
| MetaMath（2023） | Llama 2 预训练版本；官方示例为 Llama-2-7b-hf | GSM8K / MATH 训练题增强后组成 MetaMathQA，进行 SFT。 |
| ReFT（2024） | Galactica-6.7B / CodeLlama-7b-hf，预训练版本 | GSM8K 训练题及生成解题标注，SFT 预热 → PPO；PPO 直接起点已经过自己的 SFT。 |
| DeepSeekMath SFT（2024） | DeepSeekMath-Base 7B，经过数学继续预训练的 Base | 包含 GSM8K / MATH 等的数学 SFT，得到 Instruct。 |
| DeepSeekMath GRPO（2024） | DeepSeekMath-Instruct 7B，已有数学 SFT | GSM8K / MATH 相关训练问题进行 GRPO，得到 RL；论文报告无工具 Top1 GSM8K 从 82.9% 到 88.2%，是其自身设置下结果。 |
| verl Gemma 基线 | google/gemma-2-2b-it，已有官方指令后训练 | 在 GSM8K 上再做 SFT → PPO，记录原版、SFT、SFT+PPO。 |
| verl GRPO-LoRA 基线 | Qwen2.5-0.5B / 1.5B / 3B / 7B-Instruct，已有官方后训练 | GSM8K 上的 GRPO-LoRA；与本项目小模型、LoRA、后续 RL 的工程条件较接近。 |
| SkyRL 官方入门 | Qwen2.5-1.5B-Instruct，已有官方后训练 | 直接做 GSM8K GRPO；属于框架示例。 |
| verl 当前 Qwen3 示例 | Qwen3-8B，已有官方混合思考后训练 | GSM8K + MATH 上 GRPO；属于官方脚本，不能据此声称已复现或适合单卡。 |

### 对本项目的判断

- 若主要研究自己从预训练权重开始的 SFT 与后续 RL 各自效果，Base → 自己的 SFT → 自己的 GRPO 是有文献依据的方案；已有后训练模型 → 自己的 SFT → 自己的 RL 也有官方基线实例，仍然能完成阶段比较。
- 若主要研究 RL 目标、奖励、采样或 LoRA 更新，在 Instruct / 混合思考版上继续训练同样有公开先例，不能仅因其经过官方后训练而排除。是否已接近该评测上限应实测基线，不能从模型标签推断。
- 当前仍待用户选定主线；本轮将此前“推荐 Base”表述限定为项目目标下的取舍，而非 GSM8K 的统一最佳实践。更换起点与否，都应固定评测并区分格式遵循、答案正确、推理预算和数据增强的影响。

## 训练起点：Base 与官方后训练版本（2026-10-28）

- 用户指出当前起点已经过官方后训练，希望为实习项目评估未经官方后训练的版本。当前 configs 中分别是 Qwen/Qwen3-0.6B、Qwen/Qwen3-4B、Qwen/Qwen3-8B；“原始模型”一直表示本项目训练前的起点，不能据此推断它没有经过官方后训练。CONTEXT.md 已补充 Base、官方后训练模型与适配器底座的区别，未改变起点选择。
- [普通 4B 官方模型卡](https://huggingface.co/Qwen/Qwen3-4B/raw/main/README.md) 标注 Training Stage: Pretraining & Post-training；[4B-Base 官方模型卡](https://huggingface.co/Qwen/Qwen3-4B-Base/raw/main/README.md) 标注 Pretraining。[官方概念说明](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/concepts.md) 区分 Base 与混合思考版本，Base 权重尚未学会预定义聊天格式。Base 已有预训练能力，预训练也含数学、推理和合成数据，不是从随机权重开始，不能假定评测数据从未被见过。
- 项目目标明确侧重可复现的 SFT / GRPO 实验、理解算法与验证结论，不预设性能提升。评估建议：训练机以 Qwen/Qwen3-4B-Base 做正式主线，开发机后续考虑对应 0.6B-Base；旧普通版结果保留为参照。该建议基于学习完整阶段的目标，不是认定已后训练版本不能再改善专项表现；从 Base 出发也不保证提升或超过官方版本。用户尚未确认改选。
- 实验候选结构为 Base → GSM8K SFT → 后续 GRPO，统一题目、答案提取、评测预算与随机设置；官方后训练版只作独立参照。Base 的聊天格式遵循与数学正确性应分别核对；如果需要 few-shot 诊断 Base 的已有解题能力，应另列条件，避免把提示设置变化误记为训练收益。调参使用从训练集固定划出的验证集，官方测试集保留作最终评测；当前项目尚无此完整评测入口。
- 只读审计核对 [Base tokenizer_config](https://huggingface.co/Qwen/Qwen3-4B-Base/blob/main/tokenizer_config.json)：文件当前确实带 chat_template、enable_thinking 分支及 think 标记。不能声称 Base 缺少模板或 apply_chat_template 必然报错；模板文件存在不等于模型已学会聊天和思考开关。若保留既定思考段监督，应将格式视为本项目 SFT 要教的目标，不把普通版官方思考开关与采样建议直接当作 Base 已有能力或最佳参数。
- [Base config](https://huggingface.co/Qwen/Qwen3-4B-Base/raw/main/config.json) 的 max_position_embeddings=32768。sample_checkpoint.py 默认 max_new_tokens=32768，并在生成前检查题目与输出预算总和；因此非空题目下只替换成 Base ID 会在该检查处失败，需要明确 Base 的输出预算并记录为实验设置。
- [Base generation_config](https://huggingface.co/Qwen/Qwen3-4B-Base/blob/main/generation_config.json) 默认结束符为 endoftext（151643），但模板助手结尾为 im_end（151645），prepare_sft_data.py 当前监督后者，sample_checkpoint.py 未覆盖生成停止条件。后续适配须统一训练结尾、生成停止与截断判定，并确保适配器抽样能恢复相同设置；当前未实施。
- Base 与普通版 4B 的主要架构和投影尺寸一致，已有 PEFT LoRA 目标层与显存方案可复用，Base 不因省去官方后训练而显著减少显存。r16 / alpha32 / dropout0.05 / lr1e-4 / 三轮可暂作为候选起始配方，需由验证集决定，不是对 Base 的最佳值承诺。正式规模仍推荐 4B，8B 待有明确实验需要再评估，尤其后续 GRPO 的采样成本。
- 旧 LoRA checkpoint 绑定普通版底座，现有 sft_runtime.py 会检查续训的底座标识。切换 Base 应建立新的适配器与独立实验目录，不能修改底座名后接续旧优化器状态。已有工程与实验仍可用于加载验证和参照，不因改选而删除。
- 本轮依据官方小型文本文件和本地代码给出评估；未下载模型权重、运行真实 Base 问答、启动训练或改配置。默认预算越界是配置与代码推导，实际收敛、格式遵循、显存峰值、正确率及 GRPO 收益均未验证。

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
