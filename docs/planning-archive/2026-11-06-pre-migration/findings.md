# 环境发现

## 自动迁移要求核查（2026-11-06）

- 用户已确认新主线环境名称任意、系统Toolkit仅12.8；允许程序自动创建兼容Conda环境。旧数据准备只用CPU，官方PyTorch提供2.6.0 CPU wheel，无需为兼容环境另装系统Toolkit12.4。参考：https://docs.pytorch.org/get-started/previous-versions/ 。
- Conda支持--prefix，环境位置可由程序管理；使用已有Conda创建Python/pip，再由pip安装固定依赖，可避免依赖Conda环境名称或用户手动activate。参考：https://docs.conda.io/projects/conda/en/latest/commands/create.html 。
- 自动重建的范围限项目支持的旧SFT依赖配置；旧run元数据并非任意环境的完整锁文件，原始数据也需缓存或记录来源可取得。核对不一致时明确失败，不把“自动配置”误说成“任意旧产物必然迁移成功”。

## vLLM 接入核查（2026-11-05）

- 功能提交b53f9cf已完成；交付遇GitHub `fatal error in commit_refs_rules`。SSH已独立认证成功，origin经用户正式批准改为同仓库新地址 `git@github.com:FishSkin2002/LM_post_training.git`，正常受控push仍远端拒绝；不能将PR未创建描述为已交付。PR正文 `/tmp/qwen-vllm-generation-feature-pr.md` 可在远端恢复后复用。

- 已完成真实验证：当前 qwen-vllm 依赖自洽，开发机621.01支持cu128；原/新环境各45项检查通过。GPU微型推理及两步GRPO均通过。真正4B模型的显存、吞吐仍需训练机短测，不将合成奖励或微型性能冒充真实训练结果。
- 分词器跨版本实际问题包括 TokenizersBackend 类名（并非所有Qwen产物都改名）、extra_special_tokens从列表到旧版映射字段的差别，以及generic fast默认token_type_ids。已仅在加载内存中适配，实际151669词表与所有后端结构、三条中英数学提示token IDs跨环境相同。
- datasets5.1新增ArrowTable/ChunkedArray哈希归约，4.4.2没有，故同内容原始指纹不同。使用原环境校验导出完整准备数据及清单；新环境load_from_disk会因格式重设再次算hash，因此先验证保存state的原指纹与清单一致，再恢复该已导出的身份，保留现有data_plan严格比较。
- 一次性旧适配器GPU评测验证通过；实际vLLM worker退出仍可能打印上游NCCL销毁提示，公共引擎已关闭worker并清理本进程分布式资源，未把该提示推断为实际泄漏。

- 新增约束已收敛并由用户确认：CUDA12.8、宿主驱动R570+；开发机实际621.01/RTX4060Laptop8GB。用户指定Conda建环境、pip装依赖，已建立 `/home/yunli/miniconda3/envs/qwen-vllm`，大型依赖安装日志 `/tmp/qwen-vllm-pip-install.log`，会话59077。
- 配套选择 torch2.8.0+cu128 / vllm0.10.2 / TRL0.27.2 / Transformers4.57.6。0.10.2发布资产没有cu126，不照抄旧文档中的缺失构建。TRL0.27.2的GRPO能自动复制default至ref，并保留MIS；DPO仍需在项目中显式load固定SFT ref并设置ref_adapter_name，已适配。DPO已分词列名也需从*_ids映射到*_input_ids。
- 新增 generation.py、公共configs/generation.yaml、环境预检和随机微型GPU smoke。旧环境40项针对性测试已通过，日志 `/tmp/qwen-vllm-native-tests.log`；未声称GPU/新环境验证已完成。已在 `/tmp/qwen-vllm-portability/baseline.json` 保存旧环境合成OpenR1指纹，需核对新栈是否能沿用历史SFT划分。

- 用户追加 CUDA 12.4 约束，仅可适当升级。vLLM 0.8.5 官方默认 cu124、torch2.6，支持 Qwen3；当前 TRL1.16 不能与其直接混用。旧 TRL0.19/0.20 默认 PEFT 参考是底座，且缺少当前概率修正，不能在不适配时当作当前 GRPO 的等价替换。参考：https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html 。

- `sample_checkpoint.generate_answers` 是当前原生批量生成入口，独立评测和三个训练阶段结束后的评测均通过它生成；抽样入口通过 `generate_answer` 调用它。GRPO 训练另由 TRL 内部生成，当前明确 `use_vllm=False`。
- 现行训练约定要求训练结束评测复用训练模型且避免同时加载第二份权重。vLLM 接入需要明确释放训练资源后从已保存 final/ 加载，或有经过验证的复用路径，不能在训练模型仍驻留时直接再加载一份推理模型。
- 官方集成文档提供 colocate 与 server 模式；colocate 共用训练 GPU，存在显存竞争。固定版本的实际参数、LoRA 同步和依赖范围仍在核对。参考：https://huggingface.co/docs/trl/vllm_integration 。
- 本机 TRL 1.16.0 元数据限定 `vllm>=0.22.0,<=0.31.0`。官方对应版本文档 https://huggingface.co/docs/trl/v1.16.0/vllm_integration 一致；vLLM 0.22.0 的 CUDA 依赖固定 torch 2.11.0，0.31.0 固定 2.13.0，均不匹配现有 torch 2.6.0。仅拆服务端环境不能消除客户端依赖：本机 TRL 在选择 colocate/server 前检查 vLLM 已安装。未逐个验证中间版本或真实 GPU 二进制兼容性。
- 固定 TRL 的 colocate 创建推理模型，PEFT 同步临时 merge/unmerge 后推送有效权重，并重置前缀缓存；首次、恢复后首次和更新后的下一轮生成会同步。当前 HF 专用 `use_cache`/`suppress_tokens` 不能直接传入 vLLM SamplingParams，需要明确 EOS/PAD 的等价转换。该版本已有训练/推理概率差异修正；睡眠可降低更新期间显存但涉及重载开销。
- 用户确认默认 vLLM、保留原生设置，不静默回退；采用单 GPU colocate，新建配套训练/评测环境并保留原环境。新文档覆盖运行上下文、资源交接、旧断点原生恢复、只追加输出和分层验收。未安装依赖、运行真实生成或干预训练。

## GRPO 耗时与零奖励核查（2026-11-05）

- 用户提供训练日志：进度 3/99727，约 103.55 秒/步；该条日志的回答长度 min/mean/max 均 2048、clipped_ratio=1、正常结束长度均 0、reward 和 reward_std 均 0、frac_reward_zero_std=1、loss/grad_norm/kl 均 0。只能据此说明该日志窗口缺少有效奖励差异，不能推断整个后续数据均如此。
- 用户提供训练机 nvidia-smi：设备名 NVIDIA vGPU-32GB，12836/32760 MiB，GPU-Util=54%，功耗 127/320W；用户猜为 4080，设备输出不足以核实物理 GPU 型号。单个瞬时利用率不足以证明 GPU/CPU 或虚拟化瓶颈。
- 当前单卡默认 per-device=1、梯度累积=4、每题 4 条输出，TRL 的 steps_per_generation 默认继承梯度累积，生成批为 4，因此一组满长回答约生成 8192 token，再计算策略/参考概率及反向更新。当前使用原生 generate（use_vllm=False），生成 kwargs 已开启 use_cache。LoRA 不消除底座逐 token 前向成本。
- 已核对本地 TRL 1.16.0：clipped_ratio 由输出末 token 是否 EOS/PAD 判定；组内优势按奖励减组均值再除标准差计算。项目 boxed_reward 遇非 EOS 结尾直接返回 0，故该批奖励为 0 不依赖 Math-Verify 是否能解析答案。初始策略与固定 SFT 参考相同、KL 为 0 时，全零奖励差异可解释零 loss/梯度。
- 按当前 103.55 秒/步及剩余 99724 步粗算约 119.5 天，只有几步的初始 ETA 不代表稳定吞吐。建议先用少量实际输出区分正常推理尚未结束、重复生成或结束条件问题，确认能正常结束且出现奖励差异后，再评估生成引擎和并行度；未修改配置或控制用户训练进程。
- 官方依据：https://huggingface.co/docs/trl/main/en/grpo_trainer 对 clipped_ratio、frac_reward_zero_std 与生成瓶颈、vLLM 加速有说明。当前没有实际训练机 profiling，不能把 103 秒/步定性为硬件异常。
- 用户追加利用率 100%、257/320W 的快照，显存仍为 12836/32760 MiB，认为补全生成是瓶颈。已核实本地 TRL 原生路径将生成批的提示左填充后一次传入 generate；默认这 4 份是同批并行生成，仍需逐 token 推进，不能按四次完全串行生成理解。
- 生成是首要待验证的瓶颈假设，尚无真实阶段耗时可确认占比。当前 TRL 的 ProfilingContext 只发往其配置的 W&B/MLflow/Trackio 后端，项目 report_to=none 且自行使用 TensorBoardCallback，现有终端 step_time 没有生成/参考前向/反向的分段耗时。建议下一步增加最少的生成及其余训练耗时来验证，不直接改批量或推理引擎。
- NVIDIA 官方说明 GPU-Util 是采样窗口中至少一个 GPU kernel 执行的时间占比；100% 不能单独说明理论算力饱和或具体瓶颈。依据 https://docs.nvidia.com/deploy/nvidia-smi/ 。本轮仍只读检查，无代码/参数变更。

## 独立重评实施核查（2026-11-05）

- 最新用户纠正实现：其愿意重跑完整流程，已经写入的记录只按 index 恢复并追加后续，禁止历史内容复验、重新判分或重建 Markdown。本次据此纠正旧实现偏差，直接沿用生成报告的一级 `correct`/`format_valid`/`truncated`；R1 与 API 仅处理剩余记录。删除输入哈希、配置/提示一致性和历史记录模式校验，已存在的判决直接用于计数。
- 已将 `evaluate_checkpoint.py` 的 Markdown 从覆盖重建改为追加，旧 JSONL 原先就未重判；后续判分也逐题追加正文及 API 核查记录，运行末尾追加最新汇总，空续跑不写旧产物。只有显式 `--retry-pending` 才再次处理已有待复核题。
- 对应测试证明保存的一级结果即使与当前文本不一致也直接沿用，历史索引不再评分或渲染，旧 JSONL/Markdown 字节保留，新题追加；元数据/字段差异不再触发复验。45 项针对性检查在 8.480 秒通过，日志 `/tmp/qwen-append-resume-final-tests.log`；语法、差异与 13 个文档链接/锚点检查通过。
- 本轮已按约定 fetch main，并一次性仅查询 open PR 编号及分支，确认 #46 仍 open（同批还列出 #33）。直接在现有功能分支追加修正提交，不改写历史、不新建重复 PR。之前要求文档中我额外写入的复验与 Markdown 重建条款也同步纠正，遵守用户本轮明确指示。
- 修正已以 `908464b` 正常提交并推送至 PR #46，包含两个脚本、两个测试、README 和两份直接相关运行文档，旧规划/研究改动未混入提交。GitHub 插件 `update_pull_request` 尝试仅同步新版描述时返回 403 Resource not accessible by integration；未修改 PR 状态或评审，未绕过项目限制使用 gh 额外操作。代码与文档更新成功；描述稿保存在 `/tmp/qwen-layered-rescoring-feature-pr.md`，已向用户说明以最新提交为准。

- 用户已合入 PR #45 并授权实现；受控入口从最新主线创建 `codex/layered-evaluation`。基础数学评分仅依赖 Math-Verify，不需要导入训练入口；新重评应独立实现，避免改变 GRPO 奖励。
- 已核对 GLM 官方兼容接口。使用 JSON 的 Chat Completions 请求，`stream=false`，模型 `glm-5.3`，基础地址 `https://open.bigmodel.cn/api/paas/v4/`；凭据读取 `ZAI_API_KEY`。现有依赖已有 PyYAML，标准库 HTTP 足够，不额外依赖 SDK。裁判参数由 YAML 显式记录，不强制不兼容的采样值。
- 初始读取时误用了不存在的 `training_config.py` 路径；实际配置加载在 `prepare_sft_data.py`，独立入口直接使用 PyYAML，避免加载训练依赖。
- 已实现 `answer_judge.py`、`rescore_evaluation.py`、`configs/evaluation/glm-5.3.yaml` 和 17 项新测试，补充 README。依次短路第一级和 R1 单框比较，截断第一层终止，API 未形成判决记待复核；新输出保存原判分、更新正确率和所有 API 核查记录，并以独立进度支持继续及显式重试待复核。未改 `math_scoring.py`、生成入口或训练奖励。
- 最终 61 项离线测试在 12.493 秒内通过（17 项新重评测试及原有 44 项）。覆盖真实 Math-Verify、模拟 GLM/DeepSeek HTTP、截断与通过时不调用后续、正确率、API 失败/不确定/答复截断、秘密脱敏、输入/配置/提示不匹配、进度校验、中断和报告重建；实际 CLI 在项目外目录用默认配置执行和继续均通过，无训练模型导入。日志 `/tmp/qwen-rescoring-final-tests.log`。未加载用户真实 4B checkpoint、生成完整验证集或调用付费 API。
- 编写补丁时一次 JS 原始字符串内的 Markdown 反引号导致解析失败，未执行修改；改用明确字符替换构造 Markdown。另一次 README 补丁尾部上下文与实际行不一致，校验拒绝后按现有标题定位成功应用。首批 15 项测试通过后补充 HTTP 不完整响应与进度损坏、实际 CLI 检查，最终全套 61 项通过。
- 用户随后质疑完整回归，称当前只测试 `evaluate_checkpoint.py` 能否加载 checkpoint 生成答案。已解释完整回归为约 12 秒的离线微型测试，不是真实 checkpoint 或全验证集生成；已发出范围问题，选择先只检查生成或继续交付三级重评。当前所有新增代码仍未提交，未创建功能 PR。
- 用户明确继续交付三级重评，前述问题只是好奇原有 SFT/DPO/GRPO 测试的用途，不改变任务范围。新代码和配置语法、空白及 README 的 5 个本地链接/锚点检查通过。最后补齐 HTTP 错误体自身被截断时的异常处理，17 项重评测试再次通过（`/tmp/qwen-rescoring-verified-tests.log`）；原始生成/训练模块与依赖文件均未修改。准备仅交付 5 个功能文件。
- 功能已交付 `cd3b0dc`，PR #46：https://github.com/FishSkin2002/LM_post_training/pull/46 。包含 README、两个独立脚本、GLM-5.3 YAML 及 17 项测试共 5 个文件；规划/发现/进展和旧研究文件未混入功能提交。运行命令为 `python rescore_evaluation.py --input /path/to/evaluation.json --output /path/to/evaluation-rescored.json`，显式 `--retry-pending` 重试已有待复核；默认配置从 `ZAI_API_KEY` 读取密钥。真实裁判连通性、费用和语义判分质量尚待用户实际调用与人工核查。

## 选择题字母与数值的评分边界（2026-11-05）

- 用户提供的索引 59962 评测记录中，题干 B 对应 `\dfrac1{12}`，独立参考为 `\frac{1}{12}`，模型末个 boxed 为 B，当前判错。当前 `score_answer` 未接收题目，不能建立选项映射；用户口述的 1/2 是抛硬币这一步的概率，整题参考为 1/12。
- 只读检查现有 `/tmp/qwen-openr1-smoke/rows.json` 的 20 条预览，发现逐行 `(A)`、逐行 `A.`，以及末选项后跟考试出处的不同结构。附件另含行内 `\textbf{(A) }` 与 `\qquad`。这些例子只证明格式多样，不构成全量覆盖统计。
- 提案为在评分前按题目中的明确选项建立字母到内容映射，保留原文及映射依据；只清理编号的已支持排版包装，结合连续编号、唯一性、选项边界和内容可解析性检查。完整格式核验也不能证明覆盖任意题干。未识别或有歧义的样本需显式记录，不猜测、不静默丢题。
- 本轮仅回答识别可靠性问题，尚未确定新评分协议，未修改业务代码或重新判分。后续若实施，先审计实际评测集的格式与失败案例，并按独立要求 PR 的流程澄清边界。
- 后续用户提出复用已标正确 R1 的末个 boxed。当前 `_openr1` 确实依来源顺序返回首个满足两个正确标记之一为 True、完整标记为 True、末个 boxed 可解析的轨迹；未实现最短选择。对附件用现有函数实际检查：原参考 `\frac{1}{12}`、R1 boxed B、生成 boxed B；B 对原参考为 False、对 R1 为 True，数值 `\frac{1}{12}` 对两者的结果相反。因此直接替换参考会产生反向的字母/数值错配，保留原参考并另加 R1 参考可修复这一样本。
- 该提案仍有边界：选中 R1 若也只给数值，不能从中获得字母别名；标记正确的轨迹可能包含评判误差，数值参考冲突不应未经决定自动全部接受。这里只验证具体样本和讨论方案，尚未授权新的参考接受规则。首次只读验证脚本多写一个右括号而语法失败，修正后验证通过；未改项目代码。
- 用户接受增加 R1 参考仍可能漏判的边界，随后询问 dataset index 20045 的多个 boxed。当前工作区没有该条评测记录；`index` 为准备后数据索引，不能视为 OpenR1 原始 `source_index`。已请求用户提供两处框及附近文本，尚未推测其具体内容。
- 现有 `boxed_answer` 仅解析最后一次出现的 boxed；最后一处残缺时返回 None，不回退到先前完整框。只读示例验证：重复 boxed 取末值；`x=boxed{1}, y=boxed{2}` 只取 2；使用该末值当完整额外参考时，`x=boxed{999}, y=boxed{2}` 会误判为正确。因此多个框若代表答案不同部分，不能把其末框无条件加入完整答案参考。当前正式评分仍使用独立终值，未实施 R1 额外参考。
- 用户随后提供 index 20045 的题干，但未提供参考/生成解答。题目为反复将一片撕成三片，询问是否可得到 100 或 2017 片；由初始 1 片、每次总数增加 2 可推得两个结论依次为不可、可以。不能仅据题干确定其两个 boxed 的实际内容或本次判错原因。
- 尝试按题干检索公开原始记录，普通搜索未找到目标，Web 无法访问 datasets-server search URL；沙箱 urllib 请求 DNS 失败后依正式授权重试，接口返回 HTTP 500。未取得原始记录，不把原始索引或框内容作猜测。仍需其 Reference final answer 和两份解答的 boxed 附近片段。
- 用户最终提供该条最终答案片段：R1 按 100/2017 分别框 No/Yes；独立参考为 1008；模型也分别框 No/Yes。数学结论正确，1008 对应达到 2017 片的操作次数。当前末框 Yes 对独立数值参考不能通过，此例属于已确认的评分漏判。用户又指出对子问逆序回答时不能仅比较框序列，应保留子问与答案对应关系；任意自然语言对应不能靠末框提取解决。
- 用户询问是否能调用大模型 API 判分。已核对 Open R1 官方 Update #2（2025-02-10）：https://huggingface.co/blog/open-r1/update-2 ，团队用 Llama-3.3-70B-Instruct 在部分规则未通过的问题上比较最终答案，提示显式考虑格式、选择题表示及等价表达，并允许 EQUIVALENT/DIFFERENT/AMBIGUOUS。可作为本项目离线 API 复判的实践依据；不能由此推断任意裁判模型在本项目的准确率。
- 研究 https://arxiv.org/abs/2306.05685 记录 LLM 裁判的偏差和推理能力限制。建议方向仍待用户决定：向裁判提供题目、原终值、已选 R1 解答和待评模型回答；只评最终结果的完整正确性，允许不确定；固定裁判设置和保存原始判决，并在人审小样本上核对漏判与误收。尚未选择 API、调用付费服务或修改正式评测要求及代码。
- 用户提出三级流程：先按原参考数学验证；未通过再比较选中 R1 boxed 与模型 boxed，单框以外视为本级无效/false；两级均未通过时由 API 裁判，人工重点检查仅 API 判对的结果。流程可以限制 API 调用及新增正确结果的复核范围，但只查 API 判对项不能排除前两级误收或 API 漏判。
- 已用 grill-with-docs 针对两处未定边界发问：第二级任一侧无框、多框、残缺是否跳过本级继续 API，或直接整题 false；框数量在明确 `</think>` 后的正式回答统计还是全文统计。附件选择题在思考和正式回答均写 boxed，全文单框要求会跳过这类回答；用户尚未回答。当前正式约定允许多个中间 boxed 且仅取末框，新的单框要求不能未澄清便套用到整个第一层评分。
- 用户已明确：前两级均为布尔判定，只要都为 false 就进入 API；第二级无框、多框、残缺或无法解析可直接返回本级 false，不能被当作整题的最终 false。助手此前将本级未通过和最终判错混淆，已向用户纠正。
- 第二级检查范围已确定：对所选 R1 解答及模型回答，检查明确 `</think>` 后的正式回答；无该边界时检查全文，要求恰好一个完整且非空的 boxed，再进行数学等价比较。思考中的重复框不参与这一数量检查。保留第一级使用原参考的现有判断，单框要求目前仅针对第二级。
- 用户已确认 API 不确定、输出解析失败或重试耗尽均保留待复核。用户询问“独立离线重评”的含义，已改用“生成后单独判分”解释：读取现有回答执行三级评分，API 仍需联网，不重新生成回答。
- 最新决定为生成后单独重评。用户强调第一级通过后不执行第二级，任何一级通过后停止；重评需要更新整体正确率，并重点提供所有进入第三级的题目及 API 判断，供人工核查。复核范围从先前重点关注仅 API 判正确项，扩展为所有进入 API 的题目。
- 正确率设计按每题最终状态重算，已通过各级的题目互不重复，分母保留原评测全部题数；不能在旧百分比上累加，也不能在续跑时重复计算 API 新增正确。某级未执行需与该级 false 区分，以准确记录进入第三级的原因。
- Q5–Q7 已回复：始终显示已确认正确比例及正确数/总题数，另列待复核题；截断始终计错；API 计划用 GLM 或 DeepSeek，具体模型通过运行配置选择。已核实两家官方兼容接口：智谱 https://docs.bigmodel.cn/cn/guide/develop/openai/introduction ，DeepSeek https://api-docs.deepseek.com/ 。新要求草稿和术语、训练约定入口位于从 `9342088` 创建的独立要求分支，尚未提交或调用裁判 API。
- 用户随后明确更正：截断题不送第二、第三级，当前全部代码判分手段均归第一级；第一级能确定结束的不进入第二级，第二级判正确的不进入第三级。已删除草稿中截断仍调用 API 的错误安排：截断在第一级直接计错并终止，后两级未执行，无 API 或待复核记录；其余未被第一级终止的失败结果继续 R1 和 API。此决定取代先前将截断送 API 仅供核查的解释。
- 首次同步补丁把文档中的一个 boxed 反斜杠表示写错，校验失败且未应用；去掉不必要的该项修改后成功更新正文，随后单独同步规划记录。
- 要求已以 `9bceda9` 正常提交、推送并交付 PR #45：https://github.com/FishSkin2002/LM_post_training/pull/45 。仅包含 `CONTEXT.md`、`docs/training-workflow.md` 和新增 `docs/evaluation-rescoring.md`；3 份文档的 8 个本地链接/锚点、空白和代码围栏、暂存差异检查通过。未实施评分代码或运行真实裁判 API；既有本地研究/规划差异保持未提交。
- 用户确认不增加旧 `.progress.jsonl` 的第二级追加判分，之后从头执行独立重评、复用完整 JSON 中的已生成回答。随后明确使用 GLM-5.3 官方 API。已再次核实 https://docs.bigmodel.cn/cn/guide/develop/openai/introduction 的官方示例，模型标识 `glm-5.3`、基础地址 `https://open.bigmodel.cn/api/paas/v4/`，示例凭据环境变量 `ZAI_API_KEY`。仅确认并记录配置，没有获取密钥、创建实现配置文件或调用真实 API。

## 自适应生成评测批大小核查（2026-11-05）

- 正式入口从 `7d2d210` 接入。44 项离线测试通过，覆盖翻倍、模拟 OOM 和一般生成异常、首次失败、尾批、持续减半、原异常终止、继续读取旧进度以及固定、评分、中断路径。真实 CLI 和 README 检查通过。
- 首轮有一项测试把上一个成功批和下一失败批误断言为同一批。改为比较失败批和其缩小重试，并核对已保存题目对应全部成功批次，最终全量通过。
- 主线 `3299c4d` 已包含固定批大小 PR #42。生成调用发生于评分和逐题持久化之前，生成异常时可以保留现有题目游标，缩小同一段剩余题目后重试。
- 临时状态模拟通过五种情形，包含 `4 → 8 → 16 失败 → 8 → 8 失败 → 4`、首批失败逐步缩至 1、尾批实际大小小于上次成功批、单题失败终止、带旧进度的奇数起始值。未运行实际 GPU OOM 试验。

## 生成评测批大小核查（2026-11-05）

- 正式实现共用默认 4，生成助手左填充且按行取首个 EOS。旧进度记录按单题批次读入，已保存字节保留。续跑测试把 1 条旧记录与新批次 4+1 题合并，题序和统计正确。假时钟核对两批生成及逐题评分合计为 26 秒。
- 40 项离线测试通过，真实 Qwen 分词器核对左填充前后每道提示的有效 token 完全一致，实际 CLI 帮助显示默认 4。微型测试覆盖分批行为，真实 4B 显存和吞吐由训练机验证。
- `evaluate_checkpoint` 是 SFT、DPO、GRPO 自动评测与独立评测的共用入口。当前逐题循环持久化 `.progress.jsonl` 并按行数恢复，批大小属于执行设置，可独立于保存的 SFT 训练配置及输出路径。
- 临时 CPU 微型 Qwen 以原生模板左填充四种长度的提示，验证逐行注意力掩码、首个 EOS 裁剪、预算最后一步 EOS 与无 EOS 截断。受控生成的四条词元序列和串行一致，覆盖 2、3、4、4 个生成词元。未测真实 4B 的显存和吞吐。

## DPO rejected 格式放宽核查（2026-11-04）

- 在正式预处理代码上全量验证 4,120 对，保留 4,084 对，`invalid_chosen_format=27`、`unsupported_chosen_answer=9`。用官方分词器核对保留对的原始推导与答案内容、原生 EOS 和 SFT 提示。相对 `4045b56` 的双边格式筛选新增 17 个来源索引。64 条 Skywork 的输出和指纹逐项一致。
- 按 `chosen` 严格转换及解析、`rejected` 只按标题包裹非空尾部的候选规则，全量试算固定 PRM800K。4,120 对中筛除 27 对 `chosen` 格式问题和 9 对 `chosen` 不可解析，保留 4,084 对。恢复了上一规则因 rejected 格式筛除的 17 对。用官方分词器核对，本次试算的有效对均在当前长度预算内且两侧编码有区别。实际 SFT 验证题重合筛除单独计数。

## DPO 最终答案格式核查（2026-11-04）

- 在已合入要求的 `16eebf6` 实际运行新预处理。4,120 对中筛除 44 对格式问题和 9 对不可解析的 chosen，保留 4,067 对。用官方分词器逐条解码核对保留对的两侧推导前缀、答案内容、原生 EOS 和 SFT 提示。计数在排除真实 SFT 验证题之前。
- GRPO 对照首次用不同 Python 模块名加载旧函数，Datasets 因函数模块身份改变生成了不同的过滤指纹。用相同模块名和源码路径重比 64 条真实 Skywork 记录后，旧新输入、输出、指纹和元数据均一致。
- 固定版本 PRM800K `solution/train` 的 4,120 对，两侧均有独立 `# Answer` 标题。4 对至少一侧是多行答案，另有 3 对 rejected 最终答案花括号残缺。10 条 rejected 有两个同名标题，最后一个标记提供最终答案位置。
- 按最后一个标题取一行答案、移除恰好覆盖整个答案的外层数学分隔符并核验盒子边界。进一步筛除剩余嵌套或未配对的数学分隔符，避免把混合的数学模式直接塞进 `\boxed`。4,076 对通过结构检查。再要求 chosen 可由项目 Math-Verify 解析则保留 4,067 对，两侧均可解析则保留 4,008 对。计数在 SFT 验证题重合排除前，筛除策略尚待选择。
- Skywork 固定 `math` 分片 105,055 条，每条 `ground_truth` 为 JSON 列表并含一项。105,012 项没有 `\boxed` 字样，43 项含有该字样，11 项以 `\boxed{` 起始。训练提示对模型生成要求 boxed，奖励函数单独解析数据集参考。
- 远端 `main` 为 `1f091c6`，包含原训练功能提交 `0858fc0`。格式改变影响已合入的数据与提示约定，按 Git 约定先交付独立要求。

## DPO 与 GRPO 接入核查（2026-11-04）

- 功能分支从已合入要求的 `fe78b6a` 创建。PRM800K 元数据第一次读取遇到沙箱 DNS 限制，宿主授权后取得提交 `33c3d8ed2f3a3375cc9ebbfd05ff6d11b7ee6ba6`。Skywork 使用已核对的 `1cdedc52e0e2db85fdf252f9be682e63a5a38c33`。
- 两个固定版本的原始 Parquet 均用官方分词器做全量预处理。PRM800K 有 4,120 对，测试用空白重合排除 1 对，余下 4,119 对全部保持正文和单个回答 EOS。Skywork 有 105,055 条数学题，排除测试用空白重合 1 条、超长 8 条、参考均不可解析 1,891 条，保留 103,155 条。测试重合题为人为选定，用实际 SFT 验证集运行时重合计数取决于该产物。
- 现有 Conda 环境可运行固定 TRL；其发布包要求 Python 3.11 或更高。默认阶段配置面向单 GPU，DPO 序列上限 4,096，GRPO 提示与回答上限各 2,048、每题采样 4 条，均为训练机短测起点。
- `configs/training/sft.yaml` 的生产底座为 `Qwen/Qwen3-4B-Instruct-2507`。SFT 正常结束先在运行目录保存 `final/` 的适配器、分词器和 `run.json`，随后执行生成评测。`evaluate_checkpoint.py` 当前直接将产物的 `config` 和数据划分当作 SFT 评测来源。
- [tasksource/prm800k_dpo](https://huggingface.co/datasets/tasksource/prm800k_dpo) 的 `solution/train` 是同题完整解答偏好对，4,120 对，数据卡说明按原始 MATH 划分。`step` 是下一步偏好。[M4-ai/prm_dpo_pairs](https://huggingface.co/datasets/M4-ai/prm_dpo_pairs) 同样派生自 PRM800K，但标签和划分不同。
- [Skywork-OR1-RL-Data](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data) 的 `default/math` 有 105,055 条记录，`prompt` 是消息列表，`reward_model.ground_truth` 为 JSON 编码的答案列表。[官方实际评分](https://github.com/SkyworkAI/Skywork-OR1/blob/64e96afa213ae89d0ad21932106d3b8aafe9ace2/verl/utils/reward_score/livecodebench/compute_score.py#L95-L124) 接受任一参考。难度字段针对三种 DeepSeek-R1-Distill-Qwen 模型测量。原发布的去重和去污染范围不能直接覆盖项目的 OpenR1 验证题。
- 本地 Conda 已安装 TRL 1.16.0，项目 `requirements.txt` 尚未固定 TRL。生产依赖已固定 Transformers 5.20.2、Datasets 5.1.0、PEFT 0.21.0 和 PyTorch 2.6.0。
- TRL DPO 可在同一份冻结底座上复制初始 SFT LoRA 为冻结参考。GRPO 的 `beta>0` 也可共用底座；当前 TRL 的默认 `loss_type` 是 `dapo`，经典 `grpo` 需显式设置。标准 Transformers rollout 给奖励函数传入实际 `completion_ids`，可用原生 EOS 判结束。
- TRL 1.16 文本 DPO 默认对完整对话重新分词，原生模板的 EOS 后换行会被包含在回答区间。实现需用微型测试明确验证有效的 prompt、completion 与 EOS。保存默认会附带 `ref/`，而当前 Transformers 在找到适配器子目录后只加载子目录，恢复测试需要覆盖训练适配器和固定参考分别加载。
- GitHub `main` 已核实为 `5316935`，仍使用首条合格 R1 轨迹的实现。首次在沙箱内读取远端失败，宿主正式授权后读取并获取了最新主线。后续比较应沿用实际 SFT 产物保存的来源和划分。
- 离线随机微型 Qwen 探针验证了 TRL 1.16.0 的两种 Trainer 可各运行一步。DPO 渲染字符串的完整输入与现有 SFT 对齐且只包含一个回答 EOS，策略参数更新，初始参考保持逐参数一致，只保存策略时无 `ref/`。GRPO 使用真实 Transformers rollout，回调收到生成词元及对应参考，初始参考保持不变。本探针的 GRPO 奖励是合成的，尚未测试生产奖励和完整断点恢复。
- 追加离线探针验证断点机制：DPO 和 GRPO 各在 step 1 保存仅含策略的断点，以原始 SFT 初始化冻结参考，再从断点恢复到 step 2。两种 Trainer 的策略都与不间断运行逐参数一致，冻结参考与输入适配器逐参数一致。使用合成数据及奖励，结果只覆盖微型 CPU 情形。

## 验证/保存间隔调整的微型影响试验（2026-11-01）

- /tmp/qwen-interval-assessment-20261101.py为临时原型：仅在内存中允许eval_steps/save_steps变化，加载TrainerState后应用新间隔并同步元数据；未改正式文件。对照每2步验证/保存，实验从step2恢复后改每4步，二者均结束于step18、学习率轨迹相同，新间隔及末步事件正确保存。
- /tmp/qwen-interval-assessment-20261101.json显示两间隔同时变化时权重非逐位相同；/tmp/qwen-interval-assessment-save-only-20261101.json及/tmp/qwen-interval-assessment-unchanged-20261101.json显示仅保存频次变化或间隔不变时权重逐位一致。改变验证时机可改变数值轨迹，不能把相同LR/进度解读为权重必然一致；微型参数差值不能换算成真实模型准确率影响。
- PyTorch 2.6的DataLoader迭代初始化会生成base_seed并消耗其generator状态；验证频次变化可能影响随机数消耗。当前无自动选最佳或早停，验证loss不直接用于优化器更新。
- Trainer._maybe_log_save_evaluate先_evaluate再_save_checkpoint；当前步显示正在验证，不代表该步断点已完成保存。实施后的中断/恢复应基于最后完整checkpoint，保持原数据及原轨迹选择策略。

## 验证间隔的恢复状态（2026-11-01）

- Transformers 5.20.2的Trainer._init_training_state先state.compute_steps(args,max_steps)，随后从checkpoint的trainer_state.json替换整个state；init_training_references只更新总步数、轮次等引用，不重算eval_steps。验证回调读取state.eval_steps，配置变化仅传入TrainingArguments不足以确保新频次生效。
- 用户暂缓修改epoch的功能；当前关注仅调整验证频次，其他训练配置、数据来源与学习率曲线保持。此时尚未获得进一步实施指令，业务代码仍拒绝改变eval_steps的原状态恢复。

## epoch目标与默认学习率调度（2026-10-30）

- 当前配置仅显式设置learning_rate=0.0001、num_train_epochs=3；固定依赖Transformers 5.20.2的默认lr_scheduler_type=linear、warmup_steps=0。总更新步数按实际训练DataLoader与梯度累积计算每epoch更新数，再取ceil(epoch目标×每epoch更新数)。
- 未改epoch时的加训阶段可能metadata.epochs与config.num_train_epochs不同（例如追加0.5而配置仍为3）；新功能需保留这种既有恢复语义。仅改变epoch时应使用新有效目标，比较当前global_step与新目标步数，不能继续用旧trainer_state.max_steps作为完成判断。
- 学习率随新目标重算还是保留原曲线尚待用户选择；恢复时的调度计数、当前学习率与首个更新需一并核验。达到目标的路径应避免多走一步，也不能通过缩短epoch绕过其他配置/数据来源校验。

## 最短合格R1轨迹的计量与来源边界（2026-10-30）

- 用户确认“最短”按token数而非字符数；比较对象为完整、未改写的R1正文，包含think/推导/最终答案，不含题干、聊天模板、额外EOS或padding。同长保留原索引，质量判定仍为既有三个条件，不增加本地最终答案重判门槛。
- 当前math_data仍在首条合格时返回；功能实现需比较全部合格候选，并保留来源记录与确定性。分词计量放在格式/分词层或通过明确计量接口提供，数据入口不硬编码聊天模板；分词器实际词表/规则参与来源身份，避免加载路径或临时padding状态影响选择与缓存。
- 选择策略会影响超长筛除和最终题目组，旧训练来源校验需继续拒绝新策略；原始模型默认评测路径目前仅含模型/数据身份与题数，后续需纳入策略标识，避免JSONL按条数误复用旧题目。自定义output仍由使用者换新路径，不新增任务签名或日志机制。
- /tmp/qwen-openr1-smoke/shortest-policy-audit.json保存20条旧预览的只读比较：6/19选择变化，正文token总量减少8022（约7.1%），长度上限保留数均为15。字符/token排序在此小样本未出现差异，不推断一般情形；最短规则不保证峰值显存降低或推理质量提高。

## 新训练验证与保存频次（2026-10-29）

- 用户最终选择每5000个优化器更新步验证并保存，适用于当前training/development两份Instruct配置；两者采用steps策略。训练结束的最后一步验证/保存和保存后一次全量生成评测沿用既有实现。
- 本轮按重新训练处理，不改变read_source的续训校验或Trainer状态恢复。旧Base配置继续作为历史资料；实际新训练读取启动时选择的YAML。

## Instruct/OpenR1实现与验证（2026-10-29）

- 当前功能实现与新运行来源为Qwen/Qwen3-4B-Instruct-2507、OpenR1 default/train和原生三消息模板；具体行为以math_data/prepare_sft_data及配置为准，已交付PR #31。旧Base产物继续留作历史资料，不用于新协议恢复。
- 保存后的分词器加载路径会改变默认datasets函数指纹，padding/truncation运行状态也不应改变数据身份。编码阶段显式使用原始数据指纹、实际模板和移除临时padding/truncation状态后的底层分词器内容构造指纹；原始与保存后分词器的划分一致性已实际验证。
- 当前Transformers 5.20.2的apply_chat_template分词默认可返回BatchEncoding，构建标签时必须使用return_dict=False取词元列表。真实数据允许某个正确性评判器整列为null；该值不算通过，字段不存在或非空数组错位仍报错。
- 新数学依赖仅为测试安装到/tmp/qwen-openr1-test-deps；仓库requirements固定版本，现有环境包集保持。CUDA不可用，微型FP32实际训练及CPU BF16梯度验证通过，不据此声明4B显存、吞吐或数学准确率。
- 官方原生模板、20条真实记录、长度统计及R1通过标签与本地评分不一致的边界，统一保存在[样本检查](docs/research/openr1-data-smoke.md)。8192仍是初始显式预算，正式训练前需要全量检查；原始Instruct与SFT对比保持同一预算和评分协议。

## system放置的项目证据与开始交付（2026-10-29）

- 用户授权：先核查其他项目的system用法，合理即开始。已查到直接匹配的[Fast-Math-R1实际SFT代码](https://github.com/analokmaus/kaggle-aimo2-fast-math-r1/blob/09bf8ff6c71928da3ca4cca35d0643fc68f1b5e7/experiments/train_first_stage.py#L93)：system保存逐步推理与boxed要求，user直接取problem，assistant直接取generation，随后交SFTTrainer训练。它确实使用OpenR1的派生SFT集，支持当前三角色分工。
- [OpenR1固定版本蒸馏配方](https://github.com/huggingface/open-r1/blob/5b6ff22b3fb7aa069c54866e517f39dfc3160e09/recipes/OpenR1-Distill-7B/sft/config_distill.yaml)在模板中使用system指令要求Thought/Solution。该快照已采用Mixture-of-Thoughts，作为OpenR1项目后续system实践，不能冒充2025年2月原始OpenR1-Math配方；src/open_r1/utils/data.py本身并不添加system，不作此误报。
- Qwen官方模板明确支持显式system，前轮实际渲染已核对题干原样、前后输入对齐、R1保留与原生EOS。以上证据支持system放置合理，没有同题对照证明system优于user；本项目的理由是固定指令和原题分工清晰、训练前后协议一致，不增加更长提示或预填思考。
- 按用户授权开始实际交付；项目Git约定要求独立要求PR先合入，随后再从最新main实现功能。本轮先交付Instruct/OpenR1/原生模板及已确认优化的正式要求，不绕过用户合并权。原有4份跟踪/术语差异及研究目录已备份到/tmp/qwen-openr1-instruct-policy-backup-20261029。
- 正式要求的起始选择为每题按原顺序首条合格R1轨迹，记录索引；该规则是确定性的实现起点，不宣称优于其他选取策略。完整渲染后超长样本筛除不截断，具体预算在实际数据准备时确定并保存；不把初始规则或预算误报为已证明的最优设置。
- 数学等价评测采用Math-Verify，原始answer作为独立参考；官方README明确支持LaTeX、表达式与选项字符串等不同解析配置。boxed边界先按嵌套括号正确提取，生成失败不改变分母，不回退从过程数字猜答案；该等价判断不进入SFT词元loss。

## 已确认system指令与原题user（2026-10-29）

- 用户接受Instruct模板方案的其余部分，并确定将两句固定解题/boxed格式要求移到system，使每道题的user内容直接保留problem原文。最终结构为system固定指令、user原题、assistant选中的R1完整轨迹；生成时只提供前两条并由模板添加assistant起始头。
- 已使用同一固定版本官方模板重新渲染，核对system包装、user未改动、完整R1示例保留、训练/生成前缀对齐及无think预填。训练前的官方Instruct和训练后模型共用相同system/user；loss屏蔽system/user及assistant头，只监督assistant正文和原生EOS。
- [格式记录](docs/research/qwen3-instruct-chat-format.md)已按最终职责更新，并替代前一版把指令与题目合放user的建议。当前只确认与记录协议，正式要求及业务代码尚未迁移。

## Instruct原生模板的具体包装评估（2026-10-29）

- 用户接受Instruct起点，并要求评估普通QA要如何加入聊天模板；当前模型方向确定为Qwen/Qwen3-4B-Instruct-2507，不再将原无后缀模型当作已推荐主线。具体模板方案见[格式评估](docs/research/qwen3-instruct-chat-format.md)。
- 已核实官方固定版本cdbee75f17c01a7cc42f958dc650907174af0554的完整tokenizer_config：建议只使用user/assistant，格式提示和题目置于user，选定R1全文置于assistant，省去普通QA的字面Question/Answer前缀。原生模板保留assistant中的think，不预填think；EOS为im_end、PAD为endoftext。
- 已用实际官方模板渲染自写示例，核对训练/生成前缀完全对齐、思考内容原样保留及无默认system、无自动思考前缀。只读取9377字节配置，未下载模型或完整分词器；沙箱内首次DNS失败后经正式审批读取，未绕过权限。
- 评估建议保持assistant-only交叉熵，监督R1全文和原生EOS；用户消息、assistant头及padding掩码。训练前后使用同一原生模板和boxed提示，只给user输入。正式要求、代码、数据筛选和评分实现仍待下一步，不把渲染检查当作训练实现验收。

## 以Instruct学习长推理的实验取舍（2026-10-29）

- 用户补充选择Instruct的核心目的：复用聊天/指令能力，同时研究新增数学推理训练的作用，而非优先复用现成的长思考行为。此前推荐原Qwen3-4B侧重格式现成、思考模式匹配，未充分权衡该实验目标；据此调整建议，Qwen/Qwen3-4B-Instruct-2507是合理且更贴近该目标的主线候选。
- 必须修正事实前提：[官方Instruct-2507卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)明确模型已经后训练，并报告逻辑推理、数学等能力改进。non-thinking只描述发布模型不输出显式think段，公开说明不足以证明它完全没有接受推理相关训练，不能作为推理未经强化的纯净对照。
- Instruct不是架构上禁止长CoT。SFT可以让模型学习R1的完整思考轨迹及新输出行为；采用它就是有意研究这一步适配，不能把需要学习的新行为当成排除该起点的理由。[Light-R1论文](https://arxiv.org/html/2503.10460v1#S3)明确从没有long-CoT行为的Qwen2.5-32B-Instruct，经课程SFT和DPO建立长推理；[OpenR1原始实验](https://huggingface.co/blog/open-r1/update-2#performance-comparison-with-deepseek-distill-qwen-7b)也使用Qwen2.5-Math-Instruct。二者支持路线可行，不代表已验证本项目Qwen3-4B-Instruct-2507的收益。
- “无后缀模型已强所以增益小、Instruct增益会大”仍是假设，不能由名称或思考开关推出。建议按同一Instruct官方权重的训练前、OpenR1 R1轨迹SFT后、未来GRPO后三阶段比较；原生聊天模板和boxed提示保持一致，固定评测题、解码及输出上限，同时看正确率、格式有效率、输出长度与耗时。只增加think块或回答长度不算数学能力提升。
- 本轮更新的是候选建议与论证，用户尚未明确要求把该具体模型作为最终起点实施。已接受的R1监督、GSM8K接口、数据入口、boxed约定及两项优化保持；未切换权重/模板或修改正式要求，不重开此前已确认事项。

## 后训练模型与聊天模板候选评估（2026-10-29）

- 用户提出OpenR1包含聊天记录/思考标记，是否改用qwen3-4b-instruct并采用聊天模板。本轮属于起点与格式方案评估，用户的“请继续”承接评估，不当作已选定具体模型或替换此前Base约定的实施授权。
- 数据记录的messages角色字段、模板渲染后的消息边界词元与解答内的think标记是三个层次。messages并不强制选择某种模型；同一R1轨迹可用于普通QA或聊天SFT，Base也可通过SFT学习对话格式。选择已后训练模型的合理理由是复用其已学习的指令/对话行为，不能仅由数据含messages推出必须切换。
- [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)：官方已完成预训练及后训练，支持thinking/non-thinking切换，默认enable_thinking=True，并提供apply_chat_template用法。模型名称没有Instruct后缀也不代表它是Base。
- [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)：官方明确仅non-thinking，不生成think块；它仍能进行数学推理，不能把non-thinking误解为无解题能力。继续SFT当然可改变其输出行为，但当前已选择保留完整R1思考轨迹，直接采用该版本不是格式上最自然的默认选择。
- [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)：仅thinking，官方提示思考长度增加；默认模板已加入开头think，因此后续实现要保证与R1目标边界对齐，避免重复开头或丢失思考内容。它是另一个合适候选，不据模型名直接断言本项目SFT增益更大。
- 评估建议：若用户希望复用现成对话/思考协议，可改用Qwen/Qwen3-4B及其原生聊天模板，保留完整R1轨迹；相比Instruct-2507更贴近当前目标。若继续Base路线，普通QA仍可用，数据并未迫使切换。具体模型和格式尚未由用户确定，现有业务代码及正式约定保持。
- 改用后训练模型后，实验仍可研究本项目继续SFT与GRPO的增益。训练前参照必须是同一官方模型在未接受本项目训练时的表现；不能把不同起点的差异算作本项目SFT收益。boxed要求仍放在用户题干前，训练前后共用模板、模式、提示、评测题目与评分规则，不预设一定提升。
- 原Qwen3-4B官方建议thinking模式采用采样并避免贪心；当前贪心/1024-token预算是原Base流程的配置，若改模型需重新确定适用预算及解码设置，随后固定用于训练前后比较。本轮仅形成候选建议和概念记录，未下载权重、切换模型、恢复聊天模板或运行训练。

## 已确定R1监督与共用最终答案提示（2026-10-29）

- 用户明确接受使用R1生成解答作SFT目标，Q5已解决。使用problem与筛选后的generations构造普通文本QA，允许保留think；answer独立用于最终答案评测，上游solution不作为本轮主监督目标。继续常规词元交叉熵，不按生成答案数学等价清零loss。
- 用户强调：必须在问题前添加输出格式提示，让未经本项目SFT的原始Base也按相同最终答案格式回答，便于提取及与SFT后比较。沿用OpenR1已有的LaTeX boxed作为OpenR1主线的统一标记，支持分数和其他数学表达式。
- 共用提示约定如下；{problem}是题目替换位，boxed中的省略号与4只是固定格式示例，不能放入当前题目的参考答案：

```text
Solve the following math problem step by step.
End your response with the final answer in \boxed{...}, for example \boxed{4}.

Question: {problem}
Answer:
```

- 训练与loss验证在该前缀后接选定R1完整解答及原生EOS，提示和问题不计loss；独立生成输入止于Answer后的换行，不包含参考解答或answer。原始Base与SFT共用同一份提示构造、评测题目/顺序、精度、解码预算与提取评分口径；格式无效与截断单独记录，不能只为SFT模型添加提示。
- 监督解答与提示也须遵循相同最终答案约定；下一步要求稿需明确缺失/不完整boxed的筛选处理、具体提取评分与长度预算，不能只换提示而沿用只解析####/Decimal的旧计分器。本轮仅记录已选方向及具体提示，不新增聊天模板或启动训练，原始下载记录保留。

## OpenR1使用者的SFT解答来源（2026-10-29）

- 用户已确认沿用参考词元交叉熵，不按最终答案数学等价清零SFT loss。用户明确本轮只调查实际使用OpenR1-Math-220k的代表项目；此前把范围扩到MetaMath、ReFT等独立数学数据路线不符合请求，不纳入本轮结论。
- [OpenR1原始发布实验](https://huggingface.co/blog/open-r1/update-2#data-filtering)：实际SFT监督来自筛选后的DeepSeek-R1生成；比较奖励模型选最高分正确生成与随机正确生成，不是上游solution与R1轨迹对照。发布结果属于原始OpenR1-Math配方，不能拿后来改用Mixture-of-Thoughts的配置替代证据。
- [Fast-Math-R1-14B官方说明](https://github.com/analokmaus/kaggle-aimo2-fast-math-r1/blob/09bf8ff6c71928da3ca4cca35d0643fc68f1b5e7/README.md#L182)：从OpenR1-Math-220k抽样，并混合openr1_hard及Light-R1数据，去重后每题选择最短的正确R1生成，构成7900条派生SFT样本。不是7900条全来自OpenR1。
- [Fast-Math实际训练代码](https://github.com/analokmaus/kaggle-aimo2-fast-math-r1/blob/09bf8ff6c71928da3ca4cca35d0643fc68f1b5e7/experiments/train_first_stage.py#L93)把problem放入user、派生集的单数generation放入assistant，随后送SFTTrainer训练。监督的是选中R1轨迹，不是原始solution；完整抽选脚本未在该训练文件中公开，原始轨迹索引不作逐条复现声明。
- [Light-R1论文数据收集](https://arxiv.org/html/2503.10460v1#S3.SS1.SSS1)列OpenR1-Math-220k为种子来源之一；[数据生成流程](https://arxiv.org/html/2503.10460v1#S3.SS1.SSS3)明确筛选难题后重新查询DeepSeek-R1，只保留最终答案正确的长CoT，多条正确时随机选一条。因而也训练R1生成，但不能说直接复用OpenR1原generations/messages字段。
- [Light-R1公开SFT集](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData)的conversations assistant value保存R1推理及解答；[Light-R1-7B-DS模型卡](https://huggingface.co/qihoo360/Light-R1-7B-DS)确认使用3k stage2继续SFT。公开脚本注明与内部执行代码有少量差异，目标来源结论依据论文、发布数据和模型卡共同支持，不声称全流程精确复现。
- 以上三个已核实案例都以R1生成解答作SFT目标，未发现其中以OpenR1保留的上游solution作主要监督者。这个有限样本支持“沿用该数据集的公开蒸馏实践通常选择R1轨迹”的建议，不证明所有使用者都这样做，也不证明同题同预算下solution必然更差。Q5尚未替用户选定，聊天模板与目标解答来源仍是独立选择。
- 排除项：TRL的LoRA Without Regret文档确实用OpenR1派生集，但对应GRPO示例，不当作SFT解答来源证据；ServiceNow独立数据集未证实来自OpenR1，亦不列入。未继续展开其他项目综述。

## 数学等价与SFT损失调研（2026-10-29）

- 用户提出参考1/4、模型回答0.25时是否应免除loss，要求调查代表性数学推理实现。沿用只读事实审计，GitHub插件读取官方源码，网页只查一级来源；研究不授予损失改写或训练权限。
- 固定快照MetaMath fe667b14f9a4c51bde9809adafb82b6b5d255800、ReFT bdc844e02e5c783d4361d7777f4ad486e7e116da、Open R1 5b6ff22b3fb7aa069c54866e517f39dfc3160e09和Skywork-OR1 64e96afa213ae89d0ad21932106d3b8aafe9ace2。前三者SFT使用固定监督文本的词元交叉熵，没有先生成完整答案再按数学等价把样本loss清零的分支；独立生成评测与RL奖励另行判定答案。
- Open R1的依赖固定TRL0.18.0/Transformers4.52.3；SFTTrainer默认委托模型loss，实际recipe可用Liger融合交叉熵。messages形式纯LM数据不应泛称只监督回答，completion_only_loss的默认取决于数据是否含prompt列。当前公开distill配置已改用Mixture-of-Thoughts，不能冒充2025年OpenR1-Math原始recipe。
- MetaMath旧MATH评分器主要规范化后比字符串，仅特判0.5为1/2；ReFT自然语言评分先float再按容差比较，不支持直接float解析1/4。它们证明评分与SFT损失分开，但不能据此声称所有数学项目都支持通用分数/小数等价。
- Skywork实际7b_8k脚本使用GRPO，reward_manager=yr，经main_ppo传入livecodebench.compute_score；数学分支调用math_verify_reward_function，对生成与参考parse/verify给0/1奖励。core_algos先按同题组归一化奖励再算策略目标；答对或奖励相同不表示RL总loss必为0。未把仓库未启用的math.py旧规范化路径当作实际训练判分器。
- NVIDIA旧NeMo-Skills路径经插件未返回可读README，API指向迁移；未用该失败路径推断损失行为，本轮选择已追通的四项目。首次打印Skywork树响应输出过大，随后只筛选路径/相关函数并保存工具结果，不写入代码或数据目录。
- 详细代码定位与解释整理到docs/research/math-equivalence-and-sft-loss.md；项目建议以常规SFT交叉熵建立基线，生成评测/RL奖励按数学等价处理，是否采用该建议仍与需求收口区分。

## 上游solution质量证据核查（2026-10-29）

- 用户追问作者是否评估过上游solution。重新核对上游官方资料，不能将“没有R1这组正确性标签”扩大为“上游从未检查质量”。
- [NuminaMath-1.5官方数据卡](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5/blob/main/README.md)明确说明olympiads_ref取自多个国家奥赛官网，对题目和解答进行人工解析与核查；cn_contest、inequalities、number_theory由数据合作方人工整理。该说明只覆盖对应来源，不能扩展为所有solution均经过人工逐步验证。
- [上游官方预览](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)包含problem_is_valid与solution_is_valid，预览可见Yes/Incomplete等值。OpenR1记录未保留这两列，不能把它们与generations对应的correctness_*混为一谈。已查官方README没有解释这些标签由人工还是模型产生、提示规范、是否逐步检查或标签准确率，不能把字段名当作数学过程正确性证明。
- [作者网站的Numina原始技术报告](https://faculty.bicmr.pku.edu.cn/~dongbin/Publications/numina_dataset.pdf)发表于2024-07-22，讲初代NuminaMath-CoT：3.1/3.2节说明OCR、参考解答配对、翻译及GPT-4/GPT-4o整理；4节有训练模型的基准结果；5节承认合成数据正确性难验证、复杂题解答细节不足。报告支持上游也做过数据处理与整体效果评估，不代表1.5全量solution的逐步正确性审计，更不能直接解释1.5有效性标签的生成方式。
- [OpenR1官方过滤流程](https://huggingface.co/blog/open-r1/update-2#data-filtering)明确新增的Math-Verify/Llama比较针对R1生成最终答案。在已查Numina官方卡、报告及OpenR1数据卡、博客中，未找到固定同批题目和预算、只替换solution与generations监督目标的直接SFT对照；不能由两边各自的模型成绩得出本项目哪种解答更优。
- 访问project-numina地址曾失败，经旧AI-MO官方地址可正常跳转读取同一仓库；GitHub raw PDF返回application/octet-stream被网页工具拒绝，改用作者站点同名原始报告，未绕过审批下载文件。两次只读资料审计结论一致。上游并非完全未质控，R1标签也并非过程验证，Q5仍为用户待定选择。

## OpenR1 数据与训练方案（2026-10-29，第二轮）

### 用户已确定

- OpenR1-Math-220k成为后续SFT主数据源，保留GSM8K读取/使用接口，后续可用于测试等用途。后续GRPO考虑Skywork-OR1-RL-Data，不再把从OpenR1筛RL题视为已接受路线；本轮不实现GRPO或Skywork接入。
- 继续普通文本Question/Answer，可以保留解答中的think段；不恢复聊天模板。用户随后已明确选择筛选后的R1 generations，solution保留作来源资料；原始Base与SFT共用前置格式提示，具体文本见本文件顶部。
- 用户已确认Q4先单文件math_data.py统一入口，并确认Q6列出的BF16底座加载/训练计算、LoRA保持FP32与非重入激活检查点两项，包含配套训练关闭KV缓存。后续新增优化仍须先确认。普通QA选择视为简化训练协议的决定，不推断Base预训练语料完全不含聊天格式。

### 数据代码边界与候选结构

- 当前load_dataset写死在prepare_sft_data、train_sft、sample_checkpoint、evaluate_checkpoint四个入口；data_plan按行划分，运行来源绑定全局TEXT_FORMAT，评测的Markdown标题、数据源与评分描述仍写死GSM8K。整理应统一字段语义与来源，不仅移动下载调用。
- 用户已接受当前先单文件math_data.py，统一加载入口配合GSM8K/OpenR1各自的处理函数。各来源解析/过滤明显变长、独立变化时再拆成包，公共入口保持不变。数据模块返回自然文本，不负责QA包装、分词、EOS、labels或模型输出评分；后续Skywork有实际任务时再接入。
- 候选公共记录区分question、sft_target（完整监督解答）、final_answers（参考最终答案列表）和group_id/sample_id；来源清单另外保存仓库、config、split、revision、筛选/清理规则及fingerprint。该结构是内部归一化候选，不改写下载的原始记录。
- OpenR1可从answer取参考终值；GSM8K可从原answer的####末行取终值，同时保留去注释完整解答作为sft_target。Skywork将来从prompt取题目，并解码reward_model.ground_truth内的JSON列表；没有示范解答时sft_target为空，不能把短终值伪装成完整解答。
- 题目先分组划分，再选择/展开轨迹；不同数据集的默认报告文件应带数据身份，避免按条数续跑误复用另一数据集的JSONL。沿用用户选择的最简续跑语义，不扩展任务签名、日志修复等新机制。完整解答与最终答案在报告中分开，避免无声改变既有reference_answer的含义。

### solution与R1目标的比较

- [官方数据卡](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)和[过滤说明](https://huggingface.co/blog/open-r1/update-2#data-filtering)区分上游solution、独立answer与R1的generations；correctness_*校验的是生成结果，不能当作原始solution的质量标签。boxed是生成约定，并非所有样本都保证正确或可解析，见[维护者说明](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/discussions/5)。
- 缺少统一最终答案标记不妨碍用solution：可直接使用answer作为判分参考，也可在确认参考终值与解答一致后规范监督文本末尾。补标记只解决格式，不会修正错误、缺失或有噪声的solution；参考答案不进入生成输入。
- 选择R1轨迹的实质是选择学习教师推理过程，可利用逐条生成校验与完整性信号，但同时引入长序列预算。原始solution没有相同的逐条生成校验标签，二者孰优不能仅由有没有boxed判断。现有资料不足以保证任一来源在本项目上效果更好，尚未统计真实长度与筛选后规模。
- Q5本轮澄清：correctness_math_verify主要核验生成的最终答案是否与answer数学等价；correctness_llama是模型判定的补充，不是逐步过程标签。即使最终答案通过，推导仍可能存在错误或跳步；不能据此声称R1推理过程已验证，也不能由此认定其必然优于solution。用户仅提问该含义，尚未选择监督解答来源。

### 用户已确认的首批优化

- 第一项为BF16底座加载与BF16训练计算，在目标GPU支持且最小验证正常时启用，LoRA可训练参数保持FP32。这里明确包含底座加载精度变更；单独开启autocast不会自动把FP32权重减半。原始Base与SFT生成比较需统一评测精度，精度与来源记录一起确定。依据：[PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)、[PEFT参数精度](https://huggingface.co/docs/peft/developer_guides/troubleshooting#selecting-the-dtype-of-the-adapter)。
- 第二项为非重入激活检查点，use_reentrant=False，以重算减少激活保存；配套在训练时关闭KV缓存，生成时另用生成缓存设置。最小验证需确认LoRA梯度和参数更新正常。依据：[PyTorch检查点](https://docs.pytorch.org/docs/stable/checkpoint.html)、[Transformers缓存](https://huggingface.co/docs/transformers/cache_explanation)。
- 用户已确认上述两项，无需再次索取同一范围的批准；目前没有新增依赖或修改模型/训练设置，目标vGPU的BF16支持、显存与吞吐收益未实测。解答来源现已选R1；数据长度阈值、样本预算、答案提取判定细节与评测规模继续收口，之后按要求/功能分支边界实施。

## 原始Base生成评测实现与验证（2026-10-29）

- 功能分支起于已合入要求的main 4329a47。evaluate_checkpoint的checkpoint参数改为可选并增加互斥model_name，Base路径metadata为None，直接data_plan(dataset, None)获得相同固定验证划分；不伪造训练run，也不引入第二套评测循环。
- Base直接缓存FP32加载且不调用PeftModel，分词器取模型名；checkpoint继续使用保存来源并加载PEFT，训练结束内存模型复用调用保持兼容。report新增source_type，原始Base的checkpoint为null；三种产物及按条数续跑共用原循环。
- 21项离线微型集成检查一次通过；实际随机Base与零B矩阵LoRA的逐题生成文本、设置、题目和评分一致，Base加载未改变原权重文件。Base已有记录续跑仅生成剩余题，全部完成再跑不重复生成，JSONL/Markdown/汇总文件内容一致。
- 实际eval--help的5项参数与README命令/表格一致；1个本地链接、编译及差异检查通过。本轮无真实模型权重运行，无法据此报告4B的速度或准确率。

## 原始Base生成评测入口（2026-10-29）

- 当前evaluate_checkpoint.py只接受必填--checkpoint，函数首先read_source并在加载模型时调用PeftModel.from_pretrained；不能以原始Base仓库直接完成指标评测。sample_checkpoint原始模型抽样入口可复用模型名校验逻辑，生成函数本身已共用数学提示与EOS。
- data_plan(dataset, source=None)固定random.Random(42)打乱、ceil(10%)划验证集，来源存在时校验全计划一致。原始Base用无source调用即可在同一份数据下获得与训练/SFT评测相同的索引与顺序，无需伪造run.json或增加参考训练目录参数。
- 已确定最小要求：互斥--model-name/--checkpoint，Base直接加载且无适配器，两来源均仅缓存FP32；保留num-samples/device/output。默认Base独立outputs/evaluations路径，checkpoint仍用运行目录；共用原有保存/恢复/指标，主线不支持非Base系列。
- 要求分支来自main cf889d8；本轮仅需求文档变更，尚无新功能代码或真实模型运行。

## 最少PR核对的权限收口（2026-10-29）

- 用户明确允许只核对open PR，不深入内容；替代此前gh只create的限制。一次列表只含编号、源分支和目标分支即可决定现有PR是否仍接收更新；本轮复用列表，不轮询或逐项查看。
- open列表缺少某项只说明其未列为open，不能区分已合并和已关闭；是否进入主线仍基于Git提交关系。协作规则与Git约定均已统一该边界，未扩展评审、合并、关闭等写权限。
- 本轮受控start获得main 75948cc；一次gh pr list返回[]。不把该空列表单独解释为特定PR已合并，不读取其他PR字段或内容。

## 逐题文件换行约束（2026-10-29）

- 当前JSONL每题一行，长question/reference_answer/answer使用JSON字符串且正文换行转义为\\n；仅为json.dumps增加indent会破坏逐行json.loads读取及标准JSONL结构。
- JSON缩进可让字段分行，但不能让字符串中的换行成为未转义的物理换行；如需正文也按原始行展示，可考虑YAML块字符串。用户最初表示暂不做Markdown，随后明确改选Markdown阅读版；最终保持JSONL和JSON格式，新增同名.md作为阅读输出。
- Markdown为JSONL记录的展示：启动用已有记录重建，生成一题后在JSONL先落盘，再追加Markdown并flush。恢复仍只看JSONL条数，Markdown丢失不影响续跑；原始问题/解答作为缩进代码块显示，避免####或模型自带Markdown改变报告层级。
- 18项离线集成检查通过；已有续跑测试同时验证第二题生成期间首题阅读版已可读、缺少Markdown时从JSONL重建、完整记录重跑不生成或重复段落，以及训练结束自动评测也保存同名阅读版。无新依赖或CLI。8d8cd91实际仅推送旧分支，不能把它当成已更新合入的PR #25。
- 用户说明#25已合入后，受控start获得main 757c43d并建立codex/evaluation-markdown-report，Markdown差异以d87dded新提交。获取最新主线与判断Git提交关系属于既有Git管理权限，不要求gh查看/检查PR，更不需要合并或关闭权限。

## 最简评测追加与续跑实现（2026-10-29）

- 当前功能起于main fb02a2a。evaluate_checkpoint先按原逻辑校验来源/数据，再从output同名基础名.progress.jsonl读取预测列表；以len(predictions)作为下一题位置，不增加任务头或进度校验。每题一行json.dumps并flush，进度条initial为已有条数。
- 保存question、去注释完整reference_answer及生成answer，时间为每题generation_runtime。汇总复用所有记录中的原判分标记并累加题目耗时；已全部完成时仍按原逻辑加载模型，但不重复生成，不新增优化/状态分支。
- 18项离线微型检查通过，包括在第二题生成途中读取第一题落盘记录、KeyboardInterrupt续跑仅补一题、已有数据前缀不被覆盖、完成后全部记录复用、四种得分状态复用和最终报告失败后重建。生产改动仅评测脚本，其他变更为测试和README。
- 测试辅助evaluate默认分配独立输出路径，避免原有“多次改变模拟回答”的用例误复用缓存；续跑用例显式传相同output，符合用户同命令按条数继续的约定。CLI仍19项，无新增开关。

## 评测记录可读性补充（2026-10-29）

- 用户要求逐题记录包含数据集题目和标准答案。当前生成文本字段answer保持，新增question与reference_answer；后者展示与训练/验证一致的去注释完整推导及####最终数值，避免只存索引或最后数值而无法对照。两项新增字段也保留在最终汇总的predictions中。
- 此补充属于PR #24的同一要求范围；不改变逐题一行追加、按已有条数续跑或标准答案不进入模型输入的约定。

## 评测落盘与续跑现状（2026-10-29）

- 当前 evaluate_checkpoint.py 在生成循环中只向 results/predictions 内存列表追加；循环结束后才解析输出路径、生成汇总并 write_text。KeyboardInterrupt、生成异常或进程退出均不会写出已完成回答，也没有进度读取逻辑。
- 正常结束自动评测和手动入口共用 evaluate_checkpoint 函数，接入一处即可保持同一运行语义。原来的最终 JSON 可以保留，逐题 JSONL 作为续跑依据；一题生成结束后保存，不涉及模型训练断点或 token 级恢复。
- 默认文件名包含 checkpoint 名称与题数，当前 --output 允许指定报告位置。用户已明确选择固定顺序下直接按已有记录数N继续第N+1题；使用相同命令与输出路径，不增加权重/任务/顺序校验、状态管理或修复逻辑。错误或截断但已完整生成的回答也复用，避免自动重试改变既定评分口径。
- 当前存在未提交的 OpenR1 研究差异（CONTEXT、findings、progress、task_plan），已经完整备份并保留。本轮只暂存独立运行约定，不把这些差异混入新要求 PR；本地跟踪记录继续维护。

## OpenR1 主线可行性评估（2026-10-28，首轮）

### 当前项目事实

- 用户希望后续以 OpenR1-Math-220k 为主线，认为 GSM8K 可能偏小；本轮授权为方案评估，尚未确认迁移后的格式、评分、训练规模或资源优化规则。
- 正式目标仍是 Qwen3 Base → LoRA SFT → GRPO 的数学推理实验。训练默认 4B-Base，开发默认 0.6B-Base；历史已确认训练环境为租用的约32GB vGPU，物理型号与BF16能力未确认。当前约定为FP32 LoRA，未开放混合精度、激活检查点或梯度累积等优化。
- 当前生产代码完整编码 GSM8K question 与清理后的 answer，没有训练长度上限或长样本过滤；评分只接受唯一 #### 后的有限十进制数。独立生成预算固定1024 token，按行索引划10%验证集。直接更换仓库名不能适配 OpenR1 的字段、答案空间和轨迹长度。
- 原GSM8K train有7473题，实际训练6725题、验证748题。旧格式训练约一轮后验证loss上升，支持那次训练出现过拟合；缺少step0与对应生成正确率，不能证明数据规模是唯一原因。当前普通文本版尚无新的真实训练效果证据。

### 官方数据证据

- [原始数据卡](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/raw/main/README.md) 当前声明 default/train 93,733条记录、extended/train 131,396条、all/train 225,129条。每条围绕一道题组织，多条推理轨迹另存于 generations；网页各配置汇总的450,258行不能解释为45万独立题。尚未全量审计去重，因此记录数也不直接当作已核验的独立题数。
- [官方数据说明](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) 报告 default 的SFT表现优于extended。每题至少一份解答通过校验，不代表所有generations都适合正样本监督；应逐条检查正确性与推理完整标记。数据含think段与boxed答案，原生成上限16k token、约75%题目可在8k内解出；这不是本项目Qwen3分词器下的长度分布。
- [官方发布实验](https://huggingface.co/blog/open-r1/update-2#performance-comparison-with-deepseek-distill-qwen-7b) 使用Qwen2.5-Math-7B-Instruct、3轮训练并扩展到32k上下文，支持该数据源具有蒸馏价值，不能直接外推本项目4B Base、FP32 LoRA与当前预算的效果。每题随机选一份正确解答可作基线，不应默认把全部轨迹都展开训练。
- [维护者说明](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/discussions/5) 区分Math-Verify与Llama补充判定；Llama检查参考答案与think段之后的回答。正确性标签主要针对最终答案，不保证逐步推理正确。维护者提到过解析误判与版本修正，实际实验应固定数据版本和判分版本。
- [上游NuminaMath-1.5](https://huggingface.co/datasets/project-numina/NuminaMath-1.5#problem-metadata) 含选择、证明与其他数学题，参考答案可能为表达式、多问答案或proof等。未从已读官方说明中找到针对本项目GSM8K/MATH评测集的完整无污染保证；这既不证明存在污染，也不能免除交集审计。

### 初步建议与待确认边界

- 建议把default视为后续SFT候选主数据源，先以完整、通过校验且长度可承受的轨迹建立受控子集；先按题目身份分组划分，再选择或展开轨迹，避免同题跨训练与验证。记录题目数、轨迹数和监督token数，不能只比较轮次或记录条数。
- SFT可用不代表直接适合GRPO。后续RL题库需要另外筛选可自动核验的题目与答案；不能按生成proof字面内容给正确奖励。外部评测应独立于训练来源，并审计题目交集。
- 分数、根式、区间等需要数学答案提取与等价判定；这与已取消的正文公式验算是不同任务。原1024-token预算和每500步全量验证不能机械搬到大规模长轨迹训练，需重新测量预算与评测成本。
- 第一轮待确认：OpenR1在SFT、GRPO与评测中的用途；是否保留完整R1推理及think文本；在既有4B/约32GB/LoRA资源边界下，是否允许为长序列重新评估BF16和激活检查点等常规优化。以上均为建议，尚未写成正式要求。
- 尚未下载全量OpenR1、运行Qwen3长度统计、测试训练显存/速度或计算评测交集。可行性结论为有条件候选，不是已实测的训练效果或容量保证。

## 普通文本主线实现与验证（2026-10-28）

- 当前实现从 main bcd07b0 的独立功能分支开始。prepare_sft_data 维护一个 PROMPT、clean_answer 和 TEXT_FORMAT；训练与验证共用build_sft_example，抽样/评测共用build_prompt。前缀与答案分别编码再拼接，确保训练输入前缀与生成输入词元一致，答案末尾显式追加一个原生EOS。
- load_tokenizer 将随Base分发的 chat_template 元数据设为None，框架保存时可能留下null键，但不会生成chat_template.jinja。无需额外删除词表、改token ID或引入分词器保存兼容层；微型分词器的词表完整保持。
- 来源在run.json的text_format里记录精确提示、去注释正则和EOS名称；read_source统一校验，抽样入口复用它。不再读取旧template字段或要求模板文件，旧/不匹配来源通过缺少或不相等的当前格式信息被拒绝。
- 16项离线微型模型检查通过；没有预训练权重下载。生产清理函数在7473条真实GSM8K缓存上核验23716处删除、最终答案及非注释文本保持；标签/前缀完整性使用字符分词器检查，不据此报告真实Qwen序列长度或性能。
- CLI帮助可能在参数描述中把长参数换行，例如--continue-from拆成--continue-与from；从帮助文本提取参数时需只识别完整选项。最终19项脚本选项与README一致；此次只改说明内容，不新增CLI。

## 去计算器注释方案收口与验证（2026-10-28）

- 用户已明确采用：取消公式评分与提示引导，同时从监督答案删除完整计算器注释子串；原始 Base/SFT 共用的提示只约束最终答案格式。此前公式验算请求被该决定替代，不再需要确认其语法、分母或数值容差。
- 在原始缓存上只读验证 <<[^<>]*>> 的整段删除规则：7473条解答移除23716个注释，7378条改变、95条不变；逐条比较原始/清理后 #### 尾部字符串和 Decimal 数值一致，拼接所有非注释区段所得文本与清理结果完全一致，未残留 << 或 >>。
- 普通文本提示已从要求稿删除公式示例，只保留逐步推导和 #### 最终答案示例。数据变化仅是候选清理的验证结果；没有就地改写原始数据或实现生产代码。

## GSM8K 官方标记定义（2026-10-28 核实）

- [官方 README 的 Dataset Details / Calculation Annotations / Solution Extracting](https://github.com/openai/grade-school-math#dataset-details) 明确：最终数值在最后一行并以 #### 开头；<<...>> 是为调用计算器注入的计算注释。官方允许删除从 << 到 >> 的整个子串，留下外部推导与结果。
- [原论文附录 C，第17页](https://arxiv.org/pdf/2110.14168#page=17) 说明计算器注释由硬编码逻辑和微调模型共同生成，自动逻辑不完美，可能漏掉可标注的行。因此“并非每题都有/并非所有公式都标注”不能推导出“非 GSM8K 官方约定”。
- [官方 dataset.py:is_correct](https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py) 仅使用 #### 后提取的答案比较，不检查计算注释或逐步公式。它使用正则搜索及字符串比较，不能将项目现有的唯一标记/尾部严格校验与 Decimal 归一化全部归为官方实现。
- 原始解答中的注释即便不在提示中要求，仍会作为 SFT 标签被学习。用户已据此选择同时对监督文本采用官方整段去注释方法。

## 公式评测事实审计（2026-10-28）

- 现有 evaluate_checkpoint.py:25 的 answer_number 仅解析唯一 #### 后十进制数；:74 的评判只看最终数值匹配且未截断，不检查 <<>>。用户新提出的公式验算尚无实现。
- 直接 Dataset.from_file 读取 GSM8K train Arrow 缓存，以 <<([^<>]*?)>> 检查注释，7473条中95条无注释；按照原 data_plan 的 random.Random(42).shuffle 和 ceil(10%)，固定验证748条中6条无注释，实际训练6725条中89条。验证索引为618、2517、3562、4121、4358、6281；主代理另逐条核实连 << 和 >> 也不存在。
- 618/4121为概率百分比运算，2517为房价利润佣金计算，4358为切比萨整除判断；3562/6281含字母方程。它们都含计算过程，不可将“无注释”解释为“不需要计算”。原始标准答案在每题强制至少一注释的门槛下，仅742/748≈99.20%满足。
- 审计提取23716个注释，所有边界可配对；内部字符限数字、空白、+ - * / ( ) . , =，194个含圆括号，4632个含除法。此结构统计不证明算式成立，也不能据此识别正文里所有未标注公式；注释外另有百分比、变量与自然语言。
- 用户已确认要验算公式，而不只检查标记。缺注释数据如何处理尚未选择，要求先看6个样本；不能把推荐“公式单列”写成用户已接受，也不能先改写原始监督目标。

## 普通文本主线的清理范围（2026-10-28）

- 用户最新决定普通文本主线，取代上轮“建议先做格式对照”的候选；当前不增加 A/B 格式开关或保留聊天兼容。提示需用原写法/目标写法直接对比；GSM8K 标注示例保留外侧等号，不把用户简写解释为改写数据集的要求。
- 用户明确提示的主要用途是统一原始 Base/SFT 的可提取输出协议；固定说明不应只加在 SFT 一侧。最终行仅含 #### 与数值、标记只出现一次、无单位/尾文，与现有严格计分匹配。公式注释方便查看，但不扩展成公式正确率或工具执行；格式有效率继续单独报告。
- 当前业务中的清理点：prepare_sft_data 的 CHAT_TEMPLATE / apply_chat_template，sample_checkpoint 的角色消息和 im_end 双停止，training_workflow 的模板字段校验及 chat_template.jinja 必需文件；对应微型测试、数据预览和 README 需要跟随。原生词表中的登记标记不属于应删数据。
- 新来源仍需记录/校验普通文本提示与 EOS，防止不同样本格式混入恢复/加训。旧聊天产物按来源不匹配拒绝，无加载或迁移兼容。不能只删模板字段校验而丢失当前提示一致性保护。
- Base 的 EOS 与 PAD 同为 endoftext；构造/补齐标签须按位置与 attention_mask 区分，不能将所有相同 token ID 一律设为 -100，否则会误屏蔽真实结束监督。
- 本轮是独立要求交付，业务和 README 尚未实施。当前分支 codex/plain-text-gsm8k-policy，未提交研究记录完整备份 /tmp/qwen-plain-text-policy-notes-20261028。

## 普通文本数学 SFT 的证据与比较边界（2026-10-28）

- [GSM8K 官方数据构造](https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py)：get_examples 为 question 加换行、answer 加 endoftext；GSMDataset 拼接题目与答案，可配置前缀是否计损失。
- [MetaMath 训练实现](https://github.com/meta-math/MetaMath/blob/main/train_math.py)：PROMPT_DICT 使用普通文本 Instruction/Input/Response，targets 为 output 加 tokenizer.eos_token，preprocess 将 source 前缀标签设为 -100。说明普通文本前缀加答案监督有实践依据，不能作为 Qwen3 上的优越性证据。
- 用户方案同时更换消息边界并增加说明，两者可能分别改变条件分布和生成格式；想判定边界影响需只改变一个因素，或给两组相同的说明。不同格式的 loss 受上下文、分词边界及结束目标影响，生成正确率应在相同题目、解码预算与计分下比较。
- loss 验证仍根据标准解答前文预测；纯文本生成评测输入应止于 Answer 前缀，不含参考答案。自然文本仍需明确结束监督；保留原始 answer 并追加 Base 原生 EOS 是候选方案，尚未成为项目要求。

## Base 聊天边界与词元登记（2026-10-28）

- 本项目模板不新增词表项。build_sft_example 的监督后缀为 answer 原文加 `<|im_end|>`；题目、user 边界、assistant 起始边界全部掩码为 -100，但仍作为上下文。训练中的验证 loss 使用相同构造，不执行自由生成。
- sample_checkpoint.generate_answer 先提供 `<|im_start|>assistant\n`，再生成正文；停止 ID 包含 im_end 和 tokenizer.eos_token_id。没有强行补结束标记，也不要求生成第二个 im_start。解码跳过 special=true 标记，报告仅存解码文字和 token 数/截断状态，不存原始生成 IDs。
- 官方 [4B 分词器配置](https://huggingface.co/Qwen/Qwen3-4B-Base/blob/main/tokenizer_config.json) 登记 26 项 added tokens，ID 151643～151668；[0.6B](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/tokenizer_config.json) 与 [8B](https://huggingface.co/Qwen/Qwen3-8B-Base/blob/main/tokenizer_config.json) 的这组登记一致。
- special=true 共 14 项：endoftext、im_start/im_end、object_ref_start/end、box_start/end、quad_start/end、vision_start/end/pad、image_pad、video_pad。special=false 共 12 项：tool_call/闭合、tool_response/闭合、think/闭合、fim_prefix/middle/suffix/pad、repo_name、file_sep。独立词元不等于被 tokenizer 标为特殊词元。
- 官方 tokenizer_config 的 EOS/PAD 同为 `<|endoftext|>`，bos_token=null、add_bos_token=false；[生成配置](https://huggingface.co/Qwen/Qwen3-4B-Base/blob/main/generation_config.json) 的 eos_token_id=151643。Base 文件中确实有含思考/工具分支的 chat_template；项目 load_tokenizer 会覆盖为自定义简洁模板。
- 以上证据说明分词器登记与项目行为，不能据此证明预训练时每类标记的使用频率、Base 已学会对话协议或文本 Base 具备视觉能力。user/assistant 角色文字及 GSM8K 的 #### 不是本项目新增特殊词元。

## 实际训练曲线分析（2026-10-28）

- 用户事件文件共2018个train/loss点、41个eval/loss点，完整覆盖3轮20175步。事件参数确认batch1、累积1、学习率1e-4线性下降、warmup_steps0、max_grad_norm1、FP32、save_total_limit2且load_best_model_at_end=False。
- 分轮训练日志均值0.339064/0.217819/0.126117、标准差0.048546/0.036247/0.020804；验证点均值0.338622/0.364019/0.443707。训练平滑线下降且绝对波动缩小，不支持“震荡越来越失控”的判断；后两轮验证误差上升，符合过拟合信号。
- 最低已记录验证loss为step6000、epoch0.892193、0.334727；最终0.451424，较最低高34.8633%。6500→7000及13000→13500的验证loss分别上跳0.035112和0.069539，跨过epoch边界；学习率在边界连续，未重启。
- 梯度全部有限，中位数1.9758，p95=4.9344，最大11.8129；这是裁剪前的记录值，约84.8%的已记录点大于裁剪阈值1，不能当作所有更新步的精确比例。没有梯度NaN/Inf证据。
- 事件没有step0的Base验证基线和生成正确率；按记录最低验证loss选权重不保证最高解题正确率。仅保留最近两份且无最佳保留时早期权重可能已清理，TensorBoard事件本身不能恢复权重。
- 分轮训练统计由区间平均日志再汇总，属于近似描述；直接把总epoch从3改1也会改变线性调度轨迹。下轮预算可围绕第一轮对照，但不把一次日志分析当作已证明某个最佳学习率或rank。

## 独立生成评测实现（2026-10-28）

- 用户最新职责边界立即生效：Git 的分支、提交、获取和正常推送按受控入口管理；gh 只用 pr create，PR 重复/冲突/状态与接受、拒绝由用户审查。规则已单独交付 PR #20，功能分支从已合入 PR #19 的 main e0ee7d3 开始，不含未合入的规则源文件。
- 新独立评测入口共享训练结束调用的同一函数；手动时加载 Base/适配器，自动时复用已有模型。验证题目来自 run.json 中固定顺序的索引，题数非法或数据变化先报错，省略题数为全量。
- 生成指标写到运行目录 evaluations/<checkpoint>-n<N>.json，保留原始数值、题目索引与逐题结果；训练 TensorBoard 仅新增 train/eval-loss 标量，不伪造未执行的生成分数。来源 checkpoint 文件及训练状态不由评测修改。
- 最小终端回调沿用原生进度生命周期，创建时即有英文阶段名；只格式化输出副本，原始日志和 TensorBoard 数值保持。数据预览标题及采样/评测进度也采用英文，原始数据和中文报告不改写。
- 首轮微型检查的唯一失败来自同一秒事件文件名 UID 排序，不是恢复参数或缺失事件。测试将关闭后的历史事件文件按逻辑时间命名，保留对 purge_step 排除 step777 的真实验收；14 项检查随后全部通过。

## 最终独立评测方案（2026-10-28）

- 用户最终选择取消全部训练中生成监控，仅在正常完成后自动全量生成评测一次；独立脚本支持 checkpoint/final 和自选题数。此选择覆盖此前 32/2000、跳过前两次及开关方案。项目终端提示统一英文，原始数据不改写。
- 独立要求分支 codex/separate-generation-evaluation-policy 基于 main cd37e57；旧 PR #18 尚未合入。此前未提交的研究/设计记录已完整转移到新分支，未丢失内容或修改历史。
- 训练进程结束参数更新时仍持有最终模型；若另起进程加载一份同样的 FP32 权重，会增加同时驻留的权重开销。自动调用复用独立脚本内相同评测函数与已有模型即可保持单份模型，手动 CLI 则独立加载来源权重。
- 独立评测沿用保存的验证索引，省略 --num-samples 为全量，指定 N 为固定前 N 题；实际索引、分母、来源和解码设置写报告。先保存 final/训练状态后评测，评测失败只需重跑评测，不必改变已完成的训练计划。

## GSM8K 训练监控与生成正确率调查（2026-10-28）

- 调查区分三件事：SFT 参数更新的 token 交叉熵、训练期间验证安排、最终独立解题评分。已核查的代表性来源不是随机样本，不据此声称全领域使用比例。
- [MetaMath 官方训练/评测说明](https://github.com/meta-math/MetaMath/blob/main/README.MD)：公开 SFT 命令 evaluation_strategy=no；[训练入口](https://github.com/meta-math/MetaMath/blob/main/train_math.py) 的 eval_dataset=None。[独立 GSM8K 评测](https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py) 调用生成、提取最终数值并汇总正确率。这是训练中不做验证生成、另行生成评测的明确实例，不表示该项目只用 loss 报告数学效果。
- [ReFT 的 SFT 入口](https://github.com/lqtrung1998/mwp_ReFT/blob/main/train_sft_model.py)：参数更新记录 loss；按可配置步数/epoch 调用 evaluate_generation，贪心生成后处理数值，记录 value_accuracy 并用于最佳模型判断。此例说明训练期间也可以测生成正确率，但频率是实验设置。[GSM8K 实验入口](https://github.com/lqtrung1998/mwp_ReFT/blob/main/exps/paper_exps/SFT/gsm8k.sh) 同时含自然语言/程序推理版本，不能把其计算器执行与本项目纯文本混为一谈。
- [lm-evaluation-harness GSM8K 配置](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml) 使用 generate_until、贪心生成和 exact_match，有严格/灵活提取过滤器；它是评测工具，不是训练项目。用于佐证 GSM8K 解题成绩来自生成答案，而非训练 loss。
- [GSM8K 原作者答案判定](https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py) 接收生成 completion，抽取 #### 后数值并与标准答案比较。不同项目的提示、停止、抽取及计分有差别，生成正确率数值不能仅因名称相同而直接横比。
- [verl 官方 GSM8K 示例](https://verl.readthedocs.io/en/latest/examples/gsm8k_example.html) 链接到当前 [SFT Trainer](https://github.com/verl-project/verl/blob/main/verl/trainer/sft_trainer.py)：训练路径记录 loss 等标量，验证将 infer_batch 的 loss 汇总为 val/loss，该验证循环不自由生成。文档中的后续强化学习阶段根据生成最终答案匹配给奖励，不能将 RL 阶段生成需求直接套为 SFT 验证的强制条件。
- 结论是对上述公开代码的归纳：SFT 更新使用 loss；训练中生成正确率可选、频率随实验预算；独立解题效果评测依赖生成答案并判定最终结果。当前默认 loss 监控、可选小样本生成、结束后独立全量评测的方案有直接实践依据；此前每 500 步绑定生成属于本项目选择，不是 GSM8K 的统一要求。
- 对照建议不构成已实施变更：保持 Base/SFT 相同生成与计分协议，32 题用于过程观察；项目内约 748 题是验证集，最终公开 GSM8K 成绩应区分保留的官方 test。MetaMath 使用增强训练数据、ReFT 有程序/自然语言版本，不能复制分数或所有实验习惯到当前纯文本 SFT。

## 终端与续训评测开关的事实核查（2026-10-28）

- 固定版 Transformers 的 ProgressCallback.on_log 复制日志、把浮点格式化为 4 位有效数字后 str(dict) 输出；PrinterCallback 也打印字典。定制终端回调可改为 key=value 行，不改变传给 TensorBoard 的数值。训练/验证条创建于 ProgressCallback，分词为 train_sft.py 内两个 Dataset.map；现有生成条位于 WorkflowTrainer.evaluation_loop，sample_checkpoint.py 的抽样生成循环尚无进度条。
- 现有 train_sft.py 不捕获 KeyboardInterrupt，finally 仅关闭 writer；WorkflowTrainer 的 finally 仅恢复模型模式。Ctrl+C 不会自动写入当前状态，训练机只能恢复最后完整保存的断点。
- 只读审计确认优化器更新、调度器更新、清梯度、global_step/epoch 更新并非一个原子操作；在任意异常位置直接补保存可能写入不一致状态。安全暂停需单独处理安全边界及生成中断，不能用简单 catch-and-save 宣称实现。
- 当前 read_source 对 generation 的 samples/max_new_tokens 完全相等校验，128→64 会被拒绝；新增字段若混入训练配置也会触发全配置一致检查。设计需将允许变更的评测运行选项与训练计划校验分开，保留模型/LoRA/数据/优化器计划的恢复保护，并记录切换步数及实际评测题数。
- 用户最初要求 64 题/1000 步，最终改为固定 32 题/每 2000 步，沿用跳过前两次。按阶段 global_step 即跳过 2000/4000，首次 6000；恢复不重置这个门槛。按上次速度，32 题一次约 32 分钟；定期生成工作量约为旧 128 题/500 步的 1/16，不等于整体训练耗时减少 16 倍。
- 已选开关语义：新训练默认关闭；续训未显式指定时继承来源，显式开启/关闭只改变评测执行，不改变训练计划；开关控制定期和末次全部生成，关闭时仍执行完整验证 loss/保存。未执行的生成不应填成正确率 0。暂停只恢复已有完整断点，不新增安全保存机制。
- 用户只要求训练结束关闭生成时也有一次全量测评方式，不限定具体接口。独立 evaluate_checkpoint.py 可直接复用同一固定验证划分、题目输入及严格计分，读取 final 或 checkpoint 即可，无需优化器/调度器，也不绕过已完成训练拒绝续训的保护。当前为设计选择，入口尚未实现。

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
