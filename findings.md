# 当前实现发现

## README 参数核对（2026-11-09）

- 当前公开名称仍为 evaluate_checkpoint.py、resume-from-checkpoint、continue-from、sft-source、prepared-sft-data 和汇总的 resume-from；文档必须使用实际名称。additional-epochs 仅用于 SFT 加训，与 continue-from 成对要求正数。
- 默认公共生成配置 backend 为 vllm；显式 generation-backend 可选 vllm/transformers。独立生成评测 batch-size 的有效默认值分别为 64/4；auto-batch-size 只适用 Transformers。
- 原始模型评测的 config 与 sft-source 互斥；训练产物评测配置来自 run.json。汇总继续运行时沿用已冻结裁判配置，仅首次启用 API 时读取 config；no-api 时 config 与 retry-pending 不生效。
- 当前 README 末尾有一张脱离用途的重复 config 表，参数文档将按用途消除歧义。继续评分与重试待复核应分为独立用途，后者示例显式包含 retry-pending。
- 保存讨论快照时，附加导出 patch 的临时脚本误用了 check_output(check=False) 并报错；三个规划快照已完整保存、工作文件已恢复已提交基线。已核对备份三文件存在并继续，不涉及 Git 历史修改。
- 最终文档按 13 个用途覆盖全部适用参数；未把用途已选定的互斥来源列为可选来源。help 在参考顶部共用说明；no-api 用途中 config/retry-pending 不生效，在正文明确，参数表仅列适用项。原始模型有无 sft-source 的 prepared 数据默认行为分别说明，避免混淆。
- 文档检查脚本位于 /tmp/qwen-check-script-docs.py，仅提取解析器定义并解析占位路径，不执行训练/评测；5 组实际帮助输出均成功，临时输出位于 /tmp/qwen-readme-*-help.txt。

## 训练结束编排调查（2026-11-09）

- 三种训练方法都在 save_final 后调用 train.evaluate_after_training；原生路径此时复用模型完成一二级。vLLM 或训练期 GRPO 使用 vLLM 时，方法返回后由 train.main 执行 os.execv 释放原进程资源。
- 当前 execv 目标是 evaluate_checkpoint.py，因此进程替换后只完成一二级。目标应改为 train.py 内部的训练后评测流程，由新进程依次完成一二级与汇总，公开一二级命令保持原有边界。
- 汇总从评测返回的实际目录读取已保存数据，避免重新计算来源/路径或重复生成。API 失败时 final/ 和一二级数据已完整保存，可按汇总打印的评分目录继续。
- 已实现同一 train.py 的内部续程：进程替换后不再跳到仅做一二级的公开评测入口，从而保留后续 API 与报告。单独入口行为不受训练编排影响；生成器关闭后才开始汇总。
- 112 项检查确认 API 请求时模型和 trainer_state 已保存、全部本地判决已落盘；原生模型不重复加载，GRPO 训练使用 vLLM 时即便最终选 Transformers 也走新进程。交接采用模拟 execv/生成器验证，不冒充真实 GPU 显存测量。

## 汇总与阅读报告实现调查（2026-11-09）

- PR64 的 EvaluationStore 已提供共享快照、独立回答和逐级判决。汇总应直接读这些文件，不导入 evaluate_checkpoint 或训练模块，不重做本地规则。
- 旧 rescore 的逐题复制、追加 Markdown 和 finally 中渲染与新要求冲突；本轮替换为 summarize_evaluation.py。现有 APIJudge 直接从环境读密钥，需允许本次隐藏输入的密钥只传给实例，并沿用保存的提示原文与版本。
- 独立评分只新增元数据和带评分标识的三级记录，所有级别仍在一份 judgments.jsonl；只保留一套当前选定评分的阅读版，避免各评分目录复制完整问答。
- 评分来源以相对于 scoring 目录的位置保存；重名评分自动加数字。API 设置、提示原文和版本首次启用时登记，续跑直接使用已保存内容；显式重试仅追加本次调用详情，汇总累计历史调用次数而不重复复制旧 attempts。
- 阅读生成先在临时目录准备所有分卷，再替换公共阅读版；替换失败恢复旧文件。API 清单按本次最新结果互斥分组，原文链接随分卷上限重算，旧多余分卷移除，其他文件保持原样。
- 模拟检查覆盖 100 题分母中正确 70、错误 8、待复核 2、未处理 20，以及 101 题分卷/重分页链接、密钥取消与脱敏、不同评分互不混用、发布失败回滚。完整 107 项检查通过，不以模拟 API 结果代表真实裁判语义质量。

## 评测机器产物实现调查（2026-11-09）

- `evaluate_checkpoint.py` 当前每题复制题干、参考和引擎到 progress JSONL，同时写 Markdown，结束写完整 JSON。`utils/rule_scoring.py` 另存 stage2 进度并重复 stages；新产物需把回答与每级判决分开，不重执行已保存的级别。
- 现有训练结束调用复用 evaluate_checkpoint；旧 rescore 只接受历史完整 JSON。本轮保留该入口的历史用途，新的汇总能力按下一步交付。
- 快照中的 `index` 是固定题序内的编号，原始 `source_index` 与 `generation_index` 一起保留在快照中，避免处理后数据集行号影响同内容快照复用。旧迁移测试改为精确比较保留题的来源、轨迹、题干、完整参考与终值，而非将快照编号等同旧 Arrow 行号。
- 初始化耗时单独保存在元数据的测量字段，生成配置只保留静态设置；回答记录只存生成耗时，一级/二级耗时分别在判决中，不把本地判分耗时加入生成吞吐。
- 新 `EvaluationStore` 将快照引用保存为项目相对路径，按内容标识复用公共 JSON，并用完整临时文件加硬链接发布避免并发读取半份快照。两个 JSONL 逐条追加；缺失一级判决恢复时使用快照与已存回答。
- evaluate_checkpoint 的返回值保留临时内存视图供现有调用者使用，完整问答及累计指标不再另存一份 JSON。默认完整评测通过已有元数据定位；恢复后不读取或重新准备原数据集、不重新加载生成模型。
- 最终完整回归 100 项通过，vLLM 生成使用模拟引擎，训练/保存使用随机微型模型，未下载真实模型或调用付费 API。旧 rescore 处理历史 JSON，新产物的汇总和阅读输出仍为下一实现步骤。

## 评测产物精简调查（2026-11-09）

- 只读调查的报告位于另一工作树 `/home/yunli/code/qwen/FishSkin2002/outputs`，本任务修改 Codex 工作树中的要求，不修改该目录。现存五份评估 JSON 各自 6447 条 `generation_engine` 完全相同，内嵌的 `runtime` 是环境对象；每题实际耗时为另一个变化字段 `generation_runtime`。
- 原始与三级 Markdown 重复保存整套问答，API 核查 Markdown 再复制子集；三级 JSON 同一条记录同时含 `rule_scoring.stages` 与 `stages`。现有 `data.indices` 纵向列出 6447 项。
- `index` 是处理后数据集行号；`source_index` 和 `generation_index` 才能从原始 OpenR1 找回当时的题目及选定 R1 解答。旧报告保存浮动 `revision: main` 与训练机 prepared 路径，本机没有对应样本缓存。用户据此选择导出冻结的完整评测样本快照，不再依赖后续重读原数据集。
- 项目尚无公共 data/datasets 目录；生成数据集中在已被 Git 忽略的 `outputs/`。已确认公共快照使用 `outputs/evaluation-datasets/<快照标识>.json`，相对于项目根目录解析，与各训练运行目录分离；不另存 indices.json。
- 最新设计替代旧约定中的独立三级/独立汇总两个入口、生成阶段写 Markdown、Markdown 只追加。目标为 `summarize_evaluation.py` 默认补做三级再汇总，`--no-api` 只汇总已保存判决；API 中断仅保存机器进度，人工判决回写留后续。
- 相关旧表述分布于 evaluation-rescoring、project-layout、training-workflow、post-training-workflow、vllm-generation、automatic-sft-migration；已统一改为引用新的 evaluation-artifacts，历史规划归档及反映当前实现的 README 保留原状。
- 文档验证覆盖 8 份正式 Markdown 的 57 个本地链接与锚点；只读复核未发现漏项或冲突。本轮仅是要求交付，后续仍须验证真实新产物，不将文档检查当作实现验收。

## DeepSeek 最终答案裁判调查（2026-11-07）

- utils/answer_judge.py 已使用兼容 Chat Completions 的标准库 HTTP 请求；现有模拟测试覆盖 GLM 和 DeepSeek，但默认配置、帮助和 README 仍为 GLM。
- 原提示未禁止独立解题，撕纸示例还给出推导。将其替换为最终结论比较规则及纯比较示例；完整题目、R1 解答和候选回答依既有约定保留，只限定其用途，不自行裁剪输入。
- DeepSeek 官方当前首调文档列出 deepseek-flash 和 deepseek-v4-pro；采用 deepseek-flash，基础地址 https://api.deepseek.com，凭据 DEEPSEEK_API_KEY。来源：[首次调用](https://api-docs.deepseek.com/)。
- 官方思考模式默认开启，需在直接 HTTP JSON 中显式配置 thinking: {type: disabled}；结构化输出配置 response_format: {type: json_object}，提示包含 JSON 和有效对象示例。来源：[思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)、[JSON 输出](https://api-docs.deepseek.com/guides/json_mode/)。关闭思考和提示约束不能构成模型内部行为的绝对保证，真实语义质量仍需人工核查。
- 既有重评不会核查旧输出配置并会复用历史判决。本轮不改变已确认的续跑规则，README 要提醒更换服务或提示时选新输出路径。
- 实现已替换旧 GLM 配置为 configs/evaluation/deepseek.yaml，默认值与 CLI 帮助同步；JUDGE_PROMPT 为 1344 字符的中文原文，PROMPT_VERSION 升为 math-final-answer-v2。两个测试模块共 23 项通过，覆盖真实 Math-Verify、模拟两家 API、DeepSeek 参数和版本审计、短路、脱敏、失败/不确定/截断及续跑。未调用真实 API，不能将离线请求检查当作语义遵循保证。

## 旧兼容环境 SOCKS 依赖（2026-11-06）

- HTTPX 的 SOCKS 支持是可选依赖，官方安装方式为 httpx[socks]：[HTTPX 代理说明](https://www.python-httpx.org/advanced/proxies/#socks)。当前主环境 HTTPX 为 0.28.1 且缺少 socksio，可用不联网的客户端构造复现相同报错。
- 兼容环境依赖与版本检查分别位于 legacy/sft_data/requirements.txt 和 profile.json；compatibility_key 同时摘要两份依赖清单及 profile。添加依赖后自动准备会选择新受管环境，已有有效 prepared-sft 缓存仍直接复用。
- HTTPX 0.28.1 的已安装元数据声明 SOCKS extra 依赖 socksio==1.*；本轮固定 socksio==1.0.0，并将 HTTPX 与 socksio 纳入兼容环境版本检查，避免完成标记漏检缺包。
- 无 socksio 时客户端构造复现原始 ImportError。将实际 socksio wheel 安装到隔离临时目录后，socks5 和 socks5h 环境变量下的同步、异步 HTTPX 客户端全部初始化成功，未发网络请求；不存在完整代理连通性或训练通过的实测结论。

## 兼容环境渠道与 CUDA 数据处理（2026-11-06）

- 用户已明确要求自动创建兼容环境沿用其现有 Conda 渠道配置；这覆盖此前内部创建命令强制使用 conda-forge 的实现选择。
- legacy/sft_data/migration.py 在 Conda 创建阶段指定 --override-channels -c conda-forge，创建后才使用兼容环境的 python -m pip install --isolated 安装固定 CPU 数据依赖。主训练环境与该兼容环境分开。
- worker.export_data 调用 load_tokenizer、prepare_dataset、data_plan 和 save_prepared_sft_data。prepare_dataset 的结果是 Python 列表及 Arrow 数据集，分词和筛选中没有 CUDA 张量转移或模型前向；继续核对各阶段的具体实现和官方设备语义。
- 已核对完整数据链：math_data 使用 Python 字符串、哈希和 Math-Verify 解析；prepare_sft_data 使用 fast tokenizer 返回 Python IDs 列表，随后由 datasets/Arrow map、filter、排序及计数；data_plan 用 Python 按题分组和固定种子划分，prepared_sft_data 保存 Arrow 与 JSON。该链没有需要 PyTorch CUDA 执行的数值计算。
- 官方依据：PyTorch 2.8 CUDA 语义要求在 CUDA 设备上分配或转移张量后才执行对应 GPU 运算（https://docs.pytorch.org/docs/2.8/notes/cuda.html）；Hugging Face Tokenizers 说明其 Rust 实现及服务器 CPU 分词（https://huggingface.co/docs/tokenizers/index）；Datasets 的 GPU map 示例需用户显式将模型/张量放到设备（https://huggingface.co/docs/datasets/process）。仅安装 CUDA 版 PyTorch 不会把当前文本处理代码迁到 GPU。
- 执行核查：/tmp/qwen-legacy-channel-cuda-audit.py 在 torch 2.8.0+cu128 中对 20 条合成记录执行相同 prepare_dataset、data_plan 和缓存保存/读取；TorchDispatch 运算数为 0，CUDA 初始化和转移调用均为 0，完整记录和划分一致。另通过实际 ensure_compatibility_environment 函数核实新创建命令不覆盖 Conda 渠道、后续仍调用 pip。没有在用户 A800 上做 CPU/CUDA 版本计时。
- 结论：当前兼容数据准备保留 CPU PyTorch；主训练和 vLLM 的 CUDA 环境独立使用。若后续优化准备速度，应先测文本解析、逐条 map、批量分词和磁盘读写，而非更换 CUDA wheel；此轮不实施这些后续优化。

## 批次策略实现调查（2026-11-06）

- PR58 已合入 main/a87d28a；本轮从该点实现，前一节的默认 4/共用自适应描述为此前代码事实。
- 训练结束原生路径直接调用 evaluate_checkpoint，vLLM 路径执行其 CLI；两条路径均不显式传批大小。函数与 CLI 统一在生成配置解析后选择有效默认值即可同时覆盖，不需要给 train.py 增加重复默认值。
- vLLM 固定生成失败现已直接退出，只有 auto 模式进入 recover；提前拒绝 vLLM+auto 后，固定分块失败与保存边界可保留原实现。报告和终端应使用解析后的有效批大小。
- 本轮已实现：函数和 CLI 以未指定批大小为默认，统一解析后端并选择 vLLM 64 / Transformers 4；显式值覆盖，报告和终端使用有效值。auto 仅允许 Transformers，模型/来源加载前拒绝冲突。train.py 无需改动即可覆盖直接调用与进程替换路径。
- 新增五项检查与既有回归共 93 项通过，包含 129 题验证集上的 64/64/1 固定提交和改批大小续跑；vLLM 使用模拟引擎，未将该检查作为真实吞吐证据。

## 按生成引擎区分评测批次（2026-11-06）

- 目前 evaluate_checkpoint 的两个引擎共用默认 4 和成功翻倍/失败缩小逻辑；VLLMGenerator 将一个外部批次完整交给 LLM.generate，返回后评测才逐题保存。外部批次大小不等于引擎内部并发数。
- 用户最新目标：vLLM 固定提交，默认 64，可显式调整；自适应仅供 Transformers。64 是起始设置，未做真实模型吞吐测量，不宣称最优。参数冲突提前报错，vLLM 失败后从已保存条数继续的边界保持。
- vLLM 0.10.2 官方离线生成说明：引擎按内存限制安排给定提示的批处理。来源：https://docs.vllm.ai/en/v0.10.2/api/vllm/entrypoints/llm.html 。这说明无需借外层生成错误探测其并发上限，不等于任意外部块大小都达到相同吞吐。
- 直接冲突的要求位于 docs/training-workflow.md 和 docs/vllm-generation.md；README 两处参数表在功能阶段同步，当前不提前描述未实现行为。GRPO 训练采样不属于本次评测批次调整。
- PR57 已合入 main/1305f0f；本轮从该基点只交付独立要求。代码中的默认值与自适应行为仍是旧实现，不能将本轮文档目标当作已生效。

## PR57 代码整理范围（2026-11-06）

- 用户明确 train.py 保留 CLI 与独立方法调度，evaluate_checkpoint.py 和 rescore_evaluation.py 整体包含自身流程；当前整理集中更新 PR57。
- DPO/GRPO 分别由 utils/training/dpo.py、grpo.py 实现，共用模型加载、适配器保存、训练器基础行为和运行依赖核验按用户新要求放 train.py；共用数据读取模块保留历史缓存身份。
- 根评测函数仍供训练结束流程调用，导入根脚本不执行命令行；utils 中旧的 evaluation/rescoring 业务壳不保留。
- 训练后原生评测和 vLLM 进程替换调用已集中到 train.py；用户确认本轮不接入自动大模型判分，独立三级/汇总新功能仍留后续。

## 训练 CLI 入口整理（2026-11-06）

- 用户最新决定是保留统一 train.py，将 CLI 本身放入根入口；各方法的训练实现继续位于 utils/training，DPO/GRPO 的独立实现拆分属于下一步。
- train.py 已直接定义 main(argv=None)，保留 __main__ 保护；三个测试模块直接导入根入口。原 utils/training/cli.py 已删除，无兼容转发模块或遗留 Python 导入。

## PR56 冲突处理（2026-11-06）

- PR55 已合入 main/2b9a636，其中 README 最新标题为“回答正确性判定”；PR56 保留相关约定与术语同步。
- 两分支在共同基点 7d45fa1 后都修改了三份规划文件的开头；正式约定和 README 本身没有内容冲突。双方记录分别保留，当前执行安排只以 task_plan 顶部为准。
- DPO/GRPO 代码拆分、独立三级、汇总与训练结束编排尚未实施；本轮只处理 Git 合并与规划记录。

## README 相关文档审计（2026-11-06）

- README 展示板块最新名称为“回答正确性判定”，用于介绍判断模型回答是否正确的方法；内部三级判定规则沿用既有约定。
- README 的最新展示要求应集中在 docs/readme-writing.md；AGENTS 已有准确入口，不需要重复细则。规则需覆盖目录深度、脚本范围、独立方法用法、简短展示、环境占位符和纯文字报告入口。
- docs/project-layout.md 原先允许算法共处模块，已在 PR56 同步为 DPO/GRPO 独立训练实现目标，当前代码仍共处 utils/training/post_sft.py。
- docs/automatic-sft-migration.md 原先的“唯一的正常安装入口”和具名环境示例已在 PR56 修正。程序内部 CPU 兼容环境的明确安装来源是实际约定，用户安装示例不指定渠道不改变该行为。
- DPO/GRPO 共用同一次 SFT 起点和固定评测题是实验对照条件，保留；方法运行互不依赖。判分方式称“大模型判分”，API 仍用于描述通信、密钥和请求细节；单独启动与训练后自动启动的真实边界保留。
- 模型、训练和评分详细约定承担技术规范职责，README 删掉的长说明不转存成新文档。历史报告、研究记录和规划归档保留其历史语境。
- 当前根部业务入口为 train.py、evaluate_checkpoint.py、rescore_evaluation.py；DPO/GRPO 仍共处 post_sft.py。四入口和独立算法模块属于正式目标，待后续实施；下方旧调查中出现的旧入口与分支状态保留为历史记录。

| 关联文档 | 本轮处理 |
| --- | --- |
| docs/readme-writing.md | 集中记录本轮全部展示要求，供以后修改 README 时读取。 |
| docs/project-layout.md、docs/post-training-workflow.md、docs/qwen3-base.md | 明确两种独立方法及独立训练实现目标，保留实验对照条件。 |
| docs/training-workflow.md、docs/evaluation-rescoring.md、docs/vllm-generation.md | 同步大模型判分用语，保留评分规则、生成配置及调用边界。 |
| docs/automatic-sft-migration.md | 同步环境占位符、主环境依赖和旧数据自动准备表述，保留内部兼容环境规则。 |
| CONTEXT.md | 新增大模型判分定义，删除 GRPO 定义中重复的跨方法实验条件。 |
| AGENTS.md、docs/project-goals.md | 已有入口和独立方法目标一致，无需重复新增规则。 |
| docs/research/、docs/planning-archive/、reports/ | 核查其历史语境；既有记录保留，未跟踪研究文件排除。 |

## README 核查（2026-11-06）

- 用户明确 README 当前不使用 emoji；需要醒目标识时使用纯文字、加粗与普通引用块。
- “回答正确性判定”与聊天格式使用同级标题，介绍判断模型回答是否正确的方法；一级对原参考终值，二级对所选 R1 正式作答的单框，第三级统称大模型判分。截断直接判错、无法明确判定时待复核。
- 输出示例最新展示规则：checkpoint-<step>/ 收起为一行用途注释，final/ 展开到文件。
- 环境示例使用 `<环境名>` 占位符，由读者替换；基础安装命令不强制指定 Conda 渠道。
- 最新用语：README 简写“大模型判分”，在首次介绍中说明通过 API 调用在线大模型。目录和文件注释描述判分用途，命令用法单独说明如何启动或继续。
- 根部入口为 train.py、evaluate_checkpoint.py、rescore_evaluation.py。DPO/GRPO 是互斥子命令，各自从 SFT 来源加载并独立运行；post_sft.py 仍通过 stage 分支组织两种算法，代码拆分留待下一步。
- configs 的子目录只有 training、development、evaluation；公共 generation.yaml 位于 configs 根部。根目录总览及 configs 分树均按用户指定只介绍文件夹层级。
- 本地没有 outputs/training；训练机路径来自 configs/training/sft.yaml，README 的 training 树按保存代码构造示例。训练结束保存 final 与根级 trainer_state；checkpoint 另含优化器、调度器和随机状态；TensorBoard 事件直接位于 runs。
- prepared-sft 仅在不兼容旧 SFT 来源缺少缓存时自动准备；数据缓存含 manifest.json 及 dataset 的 Arrow/元数据。相关入口首次读取时可能自动创建 CPU 兼容 Conda 环境，新 SFT 正常准备不建该环境。
- evaluate 现保存五类文件：JSON、一级 JSONL/Markdown、二级 JSONL/Markdown；手动 rescore 另保存四类文件。当前训练结束只自动执行一二级，README 不应提前声称独立三级及汇总已实现。
- 主环境 Transformers 4.57.6 的分词器保存逻辑还包含 special_tokens_map.json，Qwen2 fast tokenizer 默认可保存 vocab.json/merges.txt；不能照旧 TF5 产物漏列这些文件。

## 一二级评测接通（2026-11-06）

- 基点 main/33367b2，分支 codex/two-stage-checkpoint-evaluation。当前 evaluate 已直接依赖公共生成；原 rescoring 中 single_formal_box/grade_rules 是不调用 API 的完整本地规则，可直接提取共享。
- 一级记录中的 correct 保留基础规则判决。二级结果需独立记录，不能覆盖后让旧重评把二级通过误记为一级通过；计划通过 rule_scoring 附入最终 JSON。
- 数据集 index 非连续；二级按已有索引映射跳过已保存题目，不按最大数值续跑、不复核历史答案。一级生成异常时不开始二级。
- 输出 `.stage2.progress.jsonl` 和 `.stage2.md`；最终 JSON 的顶层 rule_scoring 记录完成情况，每题同名字段记录本地判决，原 correct/metrics 保留一级含义。二级未确认正确表现为 correct=null 且 stage3=not_run，属于尚未评分，不能计为 API 待复核。
- 本地结果在 JSONL flush 后即算已保存，Markdown 失败后续跑也跳过旧题，不补写旧 Markdown。二级中断时 JSON 保留部分本地结果；旧 API 重评入口对含未完成二级的报告明确拒绝。
- 第二步已交付 3949c92/PR54，分支 codex/two-stage-checkpoint-evaluation；完整 87 项与后补定向/边界检查通过。下一步独立三级应读取完成的 rule_scoring，继续复用各题本地判决。

## 目录整理实施（2026-11-06）

- PR52 已合入 main/3e3b108，功能分支 codex/organize-training-entrypoints。
- 当前训练入口只有 train_sft 与 train_post_sft，共用模型加载目前从入口反向导入；计划移入 utils/training/common.py。DPO/GRPO 继续共用一份阶段实现，统一 CLI 只在解析后加载所选阶段。
- 模块搬迁需修正 prepare_sft_data.CONFIG_DIR、answer_judge.DEFAULT_CONFIG、evaluate 默认输出及 generation 的根部评测脚本定位；使用轻量路径模块统一仓库根。
- legacy worker 保留独立执行与依赖边界，只更新 utils 导入；API 重评模块保持不导入训练依赖。随机抽样专属测试随功能删除，共享生成边界检查保留。
- datasets 的 raw.map/filter 指纹包含可调用对象的模块身份，单纯搬 math_data/post_training_data 会影响旧来源恢复。数据转换保留原序列化名称，代码与公共导入仍放 utils/；不改变指纹检查或跳过旧来源校验。
- 主环境与旧 CPU 环境已各自对照 main/3e3b108：OpenR1、GSM8K、DPO、GRPO 的完整记录、词元、来源、指纹和已有划分不变。保留历史序列化模块名是兼容旧 run 的必要行为，不是供用户执行的旧入口；后续整理时不能简单移除此处兼容。
- 目录/统一训练第一步已交付 ab900f0/PR53，分支 codex/organize-training-entrypoints。根部当前为 train.py、evaluate_checkpoint.py 和暂保留的 rescore_evaluation.py；评分三级/汇总新入口待下一步实现。

## 项目整理调查（2026-11-06）

- 最新决定覆盖前两轮的一体化调用：一二级共一脚本，三级和汇总独立；训练结束的编排负责自动调用，手动一二级不启动三级。三级新建与续跑分开，不选旧运行总是新建，重名加数字。
- 独立三级/汇总所需题干、参考、模型回答已在一级记录中保存；完整一二级产物还应包含固定全集索引/分母、二级判决与来源。现一级中断 JSONL 没有完整任务元信息，不能拿其已生成条数当全部题数。
- generation.exec_final_evaluation 当前 os.execv 直接替换训练进程，调用后不返回；后续三级/汇总需由替换后的内部编排层或保留的外层流程负责。原生路径可继续复用模型生成。当前没有“重跑已完成 train 命令只接续评测”的用途，本轮不自动新增。
- 用户已确认三级新建 `--evaluation`、续跑 `--resume-from`，自动沿用保存的一二级来源与裁判配置。汇总用 `--scoring` 选定一次评分，支持三级中断后的部分结果，固定总分母并区分未处理与待复核；训练结束自动选本次新建评分。
- 最新要求已推送到 PR52（34fcd8a）；标题/说明已同步为独立一二级、三级及汇总的方案。GitHub 连接器更新元数据返回 403，已获用户一次性明确许可后使用 gh pr edit 成功；现有 gh 使用边界不作长期变更。

- 前两轮已确认且仍有效：业务目录 `utils/`，全量一级 → 全量二级 → 统一三级的判分顺序，训练结束自动评测也三级，缺密钥在三级开始前终端隐藏输入；一二级无 `--input`，原命令自动找进度追加。
- 自动进度发现本来就在 evaluate 中：按来源/数据/题数或 `--output` 算报告路径，再读同名 `.progress.jsonl`。只有独立 rescore 要求 `--input`；整合不应将其变为续跑前提。
- 输入只应发生在恢复进度后仍有待请求的三级题时。当前 APIJudge 每题读取环境变量、没有交互输入；源码无 GUI 框架。os.execv 保留既有标准输入连接，不保证后台启动具备终端。

- 代码事实和需求澄清统一记录在 [项目整理需求](docs/project-layout.md)。`evaluate_checkpoint.py:17` 间接导入 `sample_checkpoint.generate_answers`，实际函数位于 `generation.py:109`；不依赖随机抽样 CLI。
- `training_workflow.py` 无 CLI。SFT 与 DPO/GRPO 两个入口之间存在共用模型加载函数的反向依赖，可在统一入口实施时下沉到业务共用模块。
- 生成 JSONL 以已保存条数定位；重评 JSONL 以 index 读取判决；index 是验证集顺序中的原数据集索引，不能按数值最大值推进。现有独立重评逐题串联二三级，目标改为各级全量结束后进入下一级。
- 旧实现缺 API 密钥会计为已处理的待复核。目标改成三级前输入；取消/空输入/无终端保留进度并退出，未请求题不写为待复核。训练结束调用的同一评测也执行三级。
- 根部模块使用 `__file__` 定位配置、默认输出与评测入口；迁入目录时需保持仓库根定位。legacy worker 及测试导入要随重构同步调整，旧数据依赖边界不应因普通移文件被打破。
- 远程 main/1cf99db 已合入 PR51；需求分支为 `codex/project-layout-requirements`，本次要求已交付 PR52，待用户合入。初次查询仅 #33 仍 open，后续复用结果，不重复查询或读取 PR 内容。

## 原入口与数据流程

prepared_sft_data.ensure_prepared_sft_data(source,metadata,explicit,saved)先复用缓存；新SFT/同数据依赖来源走现有准备流程；识别旧来源后才lazy调用legacy/sft_data/migration.py。原四个入口不要求新开关或独立命令。CPU worker不递归调用helper，只准备并核对数据。

SFT checkpoint与final都可准备，缓存放在来源运行的prepared-sft/。DPO/GRPO读取源SFT，原始模型指定同一--sft-source时复用它。显式缓存仍优先，不完整或来源不符的缓存拒绝使用。数据身份读取保留经原栈核对的fingerprint，继续比较完整data_plan，不新增历史JSONL核验。

## Conda管理

- 标准conda create -n qwen-sft-data-<profile-key> --override-channels -c conda-forge python=3.12 pip -y，再用conda run查询实际Python，直接python -m pip安装固定CPU依赖。
- 环境目录来自conda info的envs_dirs，应用不覆盖它、不使用--prefix。Conda返回的sys.prefix/解释器需与实际命名环境一致。
- XDG用户缓存只保存按Conda root区分的锁和创建/安装/ready状态；环境内有匹配key/name/root的ready标记。不覆盖未受管同名环境。安装失败可重跑原命令，Conda环境本身不完整时提示修复后重跑。
- 有有效准备数据时普通路径不调用Conda或旧解释器。原主环境名任意，Conda定位用CONDA_EXE或PATH。

## 实际验收

环境名qwen-sft-data-instruct-datasets51-cpu-v1-1e87c5bc6256，实际路径/home/yunli/miniconda3/envs/qwen-sft-data-instruct-datasets51-cpu-v1-1e87c5bc6256。PyTorch2.6.0+cpu、transformers5.20.2、datasets5.1.0，CUDA runtime=None；真实Conda创建、pip安装与CPU准备通过。

主Python/home/yunli/miniconda3/envs/qwen-vllm/bin/python；主栈torch2.8.0+cu128/vllm0.10.2/transformers4.57.6/trl0.27.2/datasets4.4.2。真实原evaluate入口用旧LoRA/随机微型底座生成通过；65项针对性微型检查通过。测试数据/日志在/tmp/qwen-named-migration-fixture与/tmp/qwen-named-migration-smoke.log，环境本身在标准Conda目录。

未重复GPU vLLM/GRPO短测，因为此次只更改数据准备控制；承接代码的GPU微型证据保存在归档中，不冒充4B性能或显存结果。

## 旧 SFT 自动准备的交付记录

旧功能分支 codex/implicit-sft-migration 从 PR50/main 基点 ebbe9ca 开始，普通合并 7215c05 完整承接 PR49；其 PR51 已由用户合入 main/1cf99db，结束开发。本任务使用前述独立需求分支；日常写操作走受控入口，历史只追加。

未提交研究docs/research/math-equivalence-and-sft-loss.md保留、不混入。旧历史见Git既有提交及docs/planning-archive/2026-11-06-pre-migration/。
