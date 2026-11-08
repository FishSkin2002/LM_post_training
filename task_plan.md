# 当前计划：接通训练结束自动评测与汇总

用户于 2026-11-09 在 PR65 交付后说“继续”。已确认 773663a 合入 main/8f7d342，受控入口创建 `codex/training-final-summary`。本轮接通 SFT、DPO、GRPO 正常保存后的完整流程：一二级成功后调用默认启用 API 的汇总，保留原生模型复用与 vLLM 新进程资源交接；单独评测仍只做一二级。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 恢复与核对 | complete | PR65 已合入，训练入口/三种方法与 vLLM 交接路径已核对，用户研究文件保留。 |
| 自动编排 | complete | 原生复用和新进程两条路径均在一二级完成后默认汇总；训练中断不触发，失败保留 final/ 与评测记录并提示继续方法。 |
| 验证与交付 | in_progress | 八模块 112 项回归、语法、README 链接、差异检查通过，确认裁判配置数值未变；准备提交推送并交付 PR。 |

本轮一次 open 查询仅见其他任务 PR33，不读取 PR 内容或评审。测试仅使用随机微型模型、合成数据与模拟 API，不启动真实训练或付费请求，不改历史 outputs。

主流程仍在 train.py：原生评测后直接汇总，vLLM 或训练期 GRPO 使用 vLLM 时以 execv 进入同一文件的内部评测续程。公开 train.py --help 不展示内部续程标记；公开 evaluate_checkpoint.py 仍只完成一二级。日志为 `/tmp/qwen-training-final-summary-tests.log`，112 项检查在 27.139s 通过。

## 已完成：实现可选 API 汇总与分卷阅读报告

用户于 2026-11-09 在 PR64 交付后说“继续”。已确认 adac06e 进入远程 main/a9b1b9a，受控入口创建 `codex/evaluation-summary-reports`。本轮更名并实现独立汇总入口，默认补做三级、`--no-api` 仅汇总，正常结束后生成分卷问答和两类 API 核查清单；训练结束自动串联留下一步。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 恢复与核对 | complete | PR64 已合入，正式约定及现有机器产物/裁判接口已核对；用户研究文件保留。 |
| 汇总与阅读实现 | complete | 已更名为 summarize_evaluation.py，默认 API/只汇总、独立评分、隐藏密钥、100 题分卷和两类核查链接已实现；中断不刷新阅读版。 |
| 验证与交付 | complete | 32 项定向及完整 107 项回归和必要检查通过，426329e 已正常推送并创建功能 PR65。 |

每份评测共用原有快照、answers.jsonl 和 judgments.jsonl；三级记录带评分运行标识。评分目录只保存本次来源与裁判元数据，公共阅读版位于评测目录，反映本次显式选定的评分。中断不修改上一份成功阅读版。历史报告不迁移，不调用真实付费 API。本轮一次 open 查询仅见其他任务 PR33。

验证日志为 `/tmp/qwen-summary-tests.log`，最终 107 项在 29.258s 通过。README 已同步新建、续跑、待复核重试与只看现有结果的命令；本轮不修改 train.py 的训练后编排。

本轮已交付 [PR65](https://github.com/FishSkin2002/LM_post_training/pull/65)。待用户审查合入并说“继续”后，从最新 main 接通训练结束自动调用汇总；本轮不自动推进该步骤。

## 已完成：实现评测机器产物与共享快照

用户于 2026-11-09 在要求 PR63 交付后说“继续”。已确认 a91ef7b 进入远程 main/697a8b4，并通过受控入口创建 `codex/evaluation-machine-artifacts`。本轮只实现生成及一二级的共享快照、元数据、回答/判决 JSONL 和续跑，不实施汇总入口、API 评分或 Markdown 阅读报告。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 恢复与核对 | complete | 读取正式要求、规划和现有评测/测试，确认 PR63 合入，保留用户研究文件。 |
| 机器产物实现 | complete | 已接通公共快照、一次性元数据、回答和逐级判决 JSONL，支持缺失判决补齐及无原始数据集续跑；一二级不生成 Markdown。 |
| 验证与交付 | complete | 100 项离线/微型回归及必要检查通过；b062e3f 已正常推送并创建功能 PR64。 |

本轮不修改 `/home/yunli/code/qwen/FishSkin2002/outputs` 中的旧报告，不运行真实权重或付费 API。现有 rescore 入口继续处理历史完整 JSON；新产物的汇总入口和阅读报告留待下一步，不把本轮称为全部报告功能完成。本轮一次 open 查询仅见其他任务 PR33。

新评测目录只有 metadata.json、answers.jsonl、judgments.jsonl；题干与参考保存在项目级公共快照。旧格式报告或进度占用同名位置时要求选择新输出目录，保持旧文件并避免隐式重生成。README 已同步当前保存行为；验证日志为 `/tmp/qwen-machine-artifacts-tests.log`，最终 100 项检查在 27.303s 通过。

本轮已交付 [PR64](https://github.com/FishSkin2002/LM_post_training/pull/64)。等待用户审查合入并说“继续”后，再从最新 main 推进汇总/API 与阅读报告，不在本轮扩展后续步骤。

## 已完成：精简评测产物要求

用户于 2026-11-08 至 2026-11-09 通过 grill-with-docs 确认：共享冻结样本快照，回答与三级判决分别保存 JSONL；汇总入口默认补做 API 三级判定，可用 `--no-api` 只汇总。所有 Markdown 在汇总正常结束后生成，问答分卷与两类 API 核查清单通过链接关联。本轮用户说“继续”，只同步项目要求并交付独立 PR，不修改业务代码或历史产物。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 恢复与核对 | complete | 已读取约定和规划，通过受控入口从最新远程 main 创建独立要求分支。 |
| 同步要求 | complete | 新增产物约定并同步六份相关约定及术语，替换逐题 Markdown 追加与独立三级入口目标。 |
| 验证与交付 | complete | 差异、57 个本地链接/锚点及只读一致性复核通过；d61704f 已正常推送并创建要求 PR63。 |

共享快照位置为项目根目录下 `outputs/evaluation-datasets/<快照标识>.json`；同样本、题序与参考内容共用一份，发生变化创建新快照。每份评测只引用快照，两个 JSONL 各存自身变化数据；不再重复保存题干、参考、配置或前级判决。功能实现等待用户合入要求并说“继续”后另开分支；本轮不运行模型、真实 API，不改已有报告和用户研究文件。

当前远程 main/13f203a 已包含 PR62 的 21823ad；本轮一次 open 列表仅见其他任务 PR33。使用 `codex/evaluation-report-storage-requirements` 独立交付要求，不复用已结束的功能分支。

本轮已交付 [PR63](https://github.com/FishSkin2002/LM_post_training/pull/63)。等待用户审查合入并说“继续”后，再从最新远程 main 开展实现；未修改业务代码、README 使用说明或历史报告。

## 已完成：切换 DeepSeek 裁判并限定最终答案比较

用户于 2026-11-07 要求第三级改用 DeepSeek，并查看完整提示词；裁判应专注最终答案正确性，不自行解题。本轮只调整现有裁判配置、提示和配套使用说明，不推进独立三级入口或训练后编排。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 配置与提示 | complete | 默认 DeepSeek Flash，关闭思考并要求 JSON；中文提示明确禁止重解题和补造参考结论，版本更新为 v2。 |
| 验证与交付 | complete | 23 项模拟/本地检查、CLI 帮助、语法与链接检查通过；21823ad 已推送并创建 PR62，完整提示随本轮回复展示。 |

PR61 已通过提交关系确认合入 main/0f1380b；从最新远程 main 创建 codex/deepseek-answer-judge。原有完整题目、参考和候选回答保留，提示限定其用途；不调用真实付费 API。更换模型或提示后使用新的输出路径，避免混合旧判决。

本轮已交付 [PR62](https://github.com/FishSkin2002/LM_post_training/pull/62)，等待用户审查提示和后续指令。真实语义判分仍待抽样核查；不自动推进原计划的独立三级、汇总或训练后编排。

## 已完成：补齐旧兼容环境的 SOCKS 代理依赖

用户于 2026-11-06 报告旧兼容环境因缺少 socksio 导致 HTTPX SOCKS 代理初始化失败，要求将该依赖加入旧环境。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 依赖修复 | complete | 已添加 httpx[socks]==0.28.1、socksio==1.0.0，并将两包纳入固定配置检查。 |
| 验证与交付 | complete | SOCKS 同步/异步构造及 19 项迁移检查通过，9bbb015 已推送并创建 PR61。 |

PR60 已通过提交关系确认合入 main/56e55d3；本轮从该点创建 codex/legacy-data-socks-proxy。既有数据、训练设置及用户研究文件保持各自原状。

本轮已交付 [PR61](https://github.com/FishSkin2002/LM_post_training/pull/61)。服务器现有旧兼容环境可用 pip 单独补装 HTTPX SOCKS extra；等待用户审查及后续指令。

## 已完成：兼容环境沿用 Conda 渠道并评估 CUDA 数据处理

用户于 2026-11-06 要求删除自动创建旧 SFT 兼容环境时强制指定的 Conda 渠道，并评估 CUDA 是否能加速该数据处理流程；仅在有加速依据时改用 CUDA 版 PyTorch。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 渠道修复 | complete | 已删除强制渠道参数，命名环境创建沿用现有配置，pip 安装流程保留。 |
| CUDA 评估 | complete | 代码链和设备核查均无 CUDA 数值运算，保留 CPU 兼容环境；不把微型检查当 A800 计时。 |
| 验证与交付 | complete | 19 项迁移检查及渠道/设备核查通过，cd3b57a 已推送并创建 PR60。 |

PR59 已通过提交关系确认合入 main/ef9c99e；本轮从该点创建 codex/legacy-data-conda-channels。保留未跟踪研究文件，本轮不修改 GRPO 长度或补全数量，不启动真实训练。

本轮已交付 [PR60](https://github.com/FishSkin2002/LM_post_training/pull/60)。当前兼容流程没有 CUDA 运算路径，保留 CPU 版 PyTorch；待用户审查及后续指令。

## 已完成：实现按生成引擎区分评测批次

用户确认 PR58 已合入并要求继续。受控入口从最新 main/a87d28a 创建 codex/backend-evaluation-batches；本轮只实现已确认的批次策略及 README 同步。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 评测实现 | complete | vLLM 默认固定 64，Transformers 默认 4；显式覆盖有效，vLLM 与 auto 冲突在加载前拒绝，报告记录有效值。 |
| 使用说明与验证 | complete | README 两处参数表同步；完整 93 项离线/微型检查、帮助、语法及文档链接检查通过。 |
| 交付 | complete | 实现提交 69f9571 已推送并创建 PR59。 |

保留用户研究文件，不启动真实模型训练或付费 API；64 的真实吞吐仍待实测。独立三级、汇总和训练后自动大模型判分保持后续步骤。

本轮已交付 [PR59](https://github.com/FishSkin2002/LM_post_training/pull/59)，等待用户审查及后续指令；不继续推进其他评分功能。

## 已完成：按生成引擎区分评测批次要求

用户于 2026-11-06 确认 vLLM 评测使用固定分块，默认每次提交 64 题，可用 --batch-size 调整；自动增减仅用于 Transformers。本轮按 Git 工作约定只交付要求，用户合入后再实现并同步 README。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 核对与决定 | complete | 两份只读调查核对代码和文档，用户已确认 vLLM 默认 64。 |
| 同步要求 | complete | training-workflow 与 vllm-generation 已区分两种引擎，明确默认值、参数冲突及失败/续跑行为。 |
| 验证与交付 | complete | 差异与 17 个本地链接检查通过，要求已提交 1fe5409 并交付 PR58；本轮未改运行代码。 |

PR57 已由用户合入 origin/main/1305f0f；从该基点创建 codex/backend-evaluation-batch-policy，旧代码分支结束开发。保留用户未跟踪的研究文件。后续功能步骤覆盖独立评测与训练结束评测；独立三级、汇总和自动大模型判分仍留原计划后续。

本轮已交付 [PR58](https://github.com/FishSkin2002/LM_post_training/pull/58)。按 Git 工作约定，待用户合入要求并说“继续”后，从最新 main 开始功能实现及 README 同步。

## 已完成：PR57 代码整理

用户于 2026-11-06 明确保留统一的 train.py，将参数解析和方法选择放进根入口；随后要求 evaluate_checkpoint.py、rescore_evaluation.py 整体包含各自流程，并将当前代码整理统一提交到仍 open 的 PR57。本轮同时完成已确定的 DPO/GRPO 独立训练实现拆分，公共基础操作继续复用。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 入口迁移 | complete | train.py 直接包含 main/参数解析和调度，原 cli 模块已删除，三个现有测试模块直接导入根入口。 |
| 评测与判分整体迁回 | complete | 根目录包含完整实现与 CLI，训练和测试引用同步，utils 中旧业务模块已移除。 |
| DPO/GRPO 独立实现 | complete | 独立模块负责各自配置、Trainer 和训练流程；共用函数与现有结束评测调度位于 train.py。 |
| 验证与交付 | complete | 完整 88 项微型/离线检查、入口帮助及 62 个本地链接检查通过；e057cf2 已推送到 PR57，标题与说明已匹配完整整理范围。 |

受控 fetch 确认 PR56 已由用户合入 origin/main/7bb1232；从该基点创建 codex/train-entrypoint-cli。用户研究文件保留，不运行真实模型训练或付费 API。

沿用 [PR57](https://github.com/FishSkin2002/LM_post_training/pull/57) 交付本轮整理；用户明确当前代码整理及配套说明一起交付，不另分 PR。独立三级、汇总和训练结束自动编排的新功能仍留后续。

用户随后明确将 common.py 共用函数也放入 train.py，并确认本轮只归拢已有训练结束评测调用，大模型判分的自动接入仍留后续。根入口通过延迟导入依赖和 __main__ 保护支持方法复用，保留独立命令的行为。

## 已完成：PR56 文档同步与冲突处理

用户于 2026-11-06 要求在 PR56 追加提交，解决与 main 的三份规划记录冲突。PR55 已由用户合入 main/2b9a636；本轮在 codex/sync-readme-documentation-requirements 同步该主线，保留双方记录并追加普通合并提交。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 合并记录 | complete | 保留 PR56 文档审计与 PR55 README 整理记录；仅保留一个当前计划，旧计划归入已完成历史。 |
| 核对与交付 | complete | 三份记录、文档链接及差异检查通过，冲突解决提交随 PR56 推送交付。 |

PR56 的正式约定保留，README 使用已合入 PR55 的版本。当前目标是交付这次冲突解决；之后等待用户审查 PR56 并说“继续”，再从最新 main 推进 DPO/GRPO 独立实现。独立三级、汇总及训练后编排仍为后续步骤。

## 已完成：同步 README 相关文档约定

用户于 2026-11-06 要求检查与刚完成的 README 整理相关的项目文档，确保以后读取的约定符合本轮已确认的要求；仅未决含义才询问。继续使用 grill-with-docs（grilling/domain-modeling）、planning-with-files-zh 和 writing-for-agents。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 相关文档审计 | complete | 两个只读调查确认 README 编写约定、算法组织和环境措辞存在冲突，领域规则无待裁定分歧。 |
| 同步正式约定 | complete | 9 份约定/术语已同步；展示规则集中进入 readme-writing，算法独立性、环境示例与大模型判分用语保持一致。 |
| 一致性验证与交付 | complete | 两份只读复核、16 份文档/参考文件的 61 个本地链接及差异检查通过；b74e7fe 已推送并创建独立要求 PR56。 |

本轮从最新 origin/main/7d45fa1 创建 `codex/sync-readme-documentation-requirements`。README 展示改动已通过 [PR55](https://github.com/FishSkin2002/LM_post_training/pull/55) 合入 main，本轮按 Git 约定单独交付要求文档。对照副本为 `/tmp/qwen-approved-readme-reference.md`。未跟踪研究文件、历史报告与规划归档保留。

已确认范围：根部目录逐项注释且不展开文件夹；configs 仅文件夹；training 产物示例展开文件但 checkpoint 收起、final 展开；仅根部脚本一句简介及用途命令和必要参数；DPO/GRPO 分别说明；聊天格式仅展示；同级“回答正确性判定”简介不指定判分模型；醒目的报告入口使用纯文字；Conda 使用 `<环境名>` 且不强制渠道；多余说明直接从 README 删除，不另建文档承接。

本轮不实施 DPO/GRPO 代码拆分；将其目标写入正式约定，待用户继续后推进。独立三级、汇总和训练结束编排也保持原有后续步骤。

本轮已交付 [PR56](https://github.com/FishSkin2002/LM_post_training/pull/56)，README 展示仍由 PR55 交付。后续实现先由用户合入相关要求，再按用户继续指令从最新 main 开始。

## 已完成：README 整理

用户于 2026-11-06 确认本轮只整理 README：根目录树逐项注释、文件夹不展开；单列 configs 文件夹介绍和 training 完整产物示例；聊天格式独立展示；突出实验报告入口。根目录脚本各用一句话介绍，保留用途命令和必要参数；多余原理、内部实现和大段环境说明直接删除，不另建文档。DPO 与 GRPO 分别说明，代码拆分留到下一步。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 事实核对与范围确认 | complete | 已核对三个根部入口、目录与保存逻辑；用户已确认删除多余说明、本轮不拆代码。 |
| README 重组 | complete | 根目录逐项注释；configs 只列文件夹；training 示例展开到文件；仅三个根部脚本，DPO/GRPO 独立说明；聊天格式与报告入口已整理。 |
| 验证与交付 | complete | 五组 CLI 帮助、配置路径、两处链接、目录注释与聊天内容检查通过；6297e28 已推送并创建 PR55。 |

最新远程 main/7d45fa1 已包含 PR54；本轮从该基点创建 `codex/readme-layout-and-script-guide`。未跟踪研究文件保留并排除提交。后续代码拆分和既有独立三级/汇总步骤，均等待用户继续。

本轮已交付 [PR55](https://github.com/FishSkin2002/LM_post_training/pull/55)，现已由用户合入 main。用户已选择下一步拆分 DPO/GRPO 实现，本轮不提前实施。

2026-11-06 补充修正：用户要求准确说明在线大模型 API 判分，并润色措辞。README 统一简写为“大模型判分”，脚本介绍写明“通过 API 调用在线大模型”。相关目录注释、文件说明、用途标题与参数说明已同步，命令示例和链接检查通过；沿用 PR55 交付（现已合入）。

环境示例使用 `<环境名>` 占位符，提示创建与激活命令填写相同的自选名称；按用户最新要求移除强制指定渠道的参数，沿用 PR55 交付。

输出目录示例按用户最新要求折叠 `checkpoint-<step>/`，仅保留用途注释，`final/` 继续展开；相应删除“每个断点结构如上”的表述。

README 与“聊天格式”同级的板块按最新要求命名为“回答正确性判定”，介绍判断模型回答是否正确的三个层次；第三级仅称大模型，沿用 PR55 交付。

README 按用户明确偏好使用纯文字标识，不使用 emoji；实验报告入口以引用块和加粗文字突出显示。

## 已完成第二步：接通一二级评测

以下为已交付任务的历史记录，当前工作和后续步骤以上方计划为准。

用户确认 PR53 已合入并要求继续。受控 fetch 确认远程 main/33367b2 包含该 PR；从最新 main 创建 `codex/two-stage-checkpoint-evaluation`。本轮只将第二级规则接入评测，保留各级进度追加与旧结果复用，手动评测不调用 API 或汇总。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 提取本地规则 | complete | single_formal_box/grade_rules 提取到 utils/rule_scoring.py，旧重评直接复用保存规则或兼容原始报告。 |
| 评测接通二级 | complete | 全量生成一级后才二级，独立 stage2 JSONL/Markdown，按索引追加；中断也保存 JSON 进度。 |
| 针对性验证 | complete | 完整微型/离线回归 87 项通过；补充二级未完成的 API 拦截及跨生成批次阶段屏障后，定向 23 项和六项边界复测通过。 |
| 文档与交付 | complete | README/规划、语法及八个链接检查完成；3949c92 已推送并创建 PR54。 |

第二步已交付 [PR54](https://github.com/FishSkin2002/LM_post_training/pull/54)。等待用户审查并说“继续”后推进独立三级评分，不自动跨步骤执行。

后续独立步骤：三级评分新建/续跑、独立结果汇总、训练结束自动编排。已有 rescore 薄入口暂保留，不在本轮启动真实大模型训练或付费 API。研究文件继续保留并排除提交。

## 已完成第一步：目录与统一训练入口

用户于 2026-11-06 要求开始改动。受控 fetch 确认 PR52 已由用户合入远程 main/3e3b108；从该基点创建 `codex/organize-training-entrypoints`。按项目每轮一个小步骤，本轮只实施业务模块归拢、统一训练入口及删除随机抽样。

| 本轮工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 目录与路径迁移 | complete | 共享业务代码进入 utils/，根部薄入口、配置、默认输出、legacy worker 与测试导入已同步。 |
| 统一训练入口 | complete | train.py sft/dpo/grpo，共享 CLI、模型加载和保存，微型 SFT/DPO/GRPO 已经统一入口验证。 |
| 删除随机抽样 | complete | sample_checkpoint.py 移除，共享生成直接导入，固定前 N 题评测保留。 |
| 验证与交付 | complete | 80 项现有检查+新增指纹回归通过，回归也在旧 CPU 环境通过；八条数据流程精确对比与 CLI/语法/链接检查完成；ab900f0 已推送并创建 PR53。 |

第一步已交付 [PR53](https://github.com/FishSkin2002/LM_post_training/pull/53)。按每轮一个小步骤的约定，等待用户审查并说“继续”后推进评分流程；不把第一步当作全部整理完成。

后续步骤（本轮不实施）：一二级评测整合、独立三级新建/续跑、独立汇总及训练结束自动编排。原重评命令在拆分完成前暂时保留，避免本轮中间状态丢失既有能力。不启动真实大模型训练或付费 API。

未提交研究文件 `docs/research/math-equivalence-and-sft-loss.md` 保留并排除提交。

## 已完成要求交付

2026-11-06，用户调用 grill-with-docs，要求归拢业务脚本、统一训练入口、删除抽样问答入口，并将三级判定整合到评测，按各级进度追加。

- 用户最新调整替代此前“一二级自动三级”决定：一二级共用脚本，三级与汇总分别独立；只有训练结束自动依次串联。三级选已有运行才续跑，否则新建，重名加数字。`utils/`、统一训练、删除抽样入口、隐藏输入密钥及按进度追加仍保留。
- 用户已确认增补问题：三级新建 `--evaluation`、续跑 `--resume-from`；汇总选定一次评分，支持三级中断后的部分结果，固定总题数并区分待复核与未处理。
- 当前阶段：最新要求已推送并更新 [PR52](https://github.com/FishSkin2002/LM_post_training/pull/52) 的标题与说明，等待用户审查合入。入口以[项目整理约定](docs/project-layout.md)为准；尚未修改业务代码。
- Git：远程 main 仍为 1cf99db；上轮一次性 open 查询确认 #52 仍 open，继续使用 `codex/project-layout-requirements`，无需新 PR。本轮复用该查询，不读取 PR 正文或评审。
- 后续：用户合入并要求继续后，再从最新远程 main 新建功能分支，按小步骤落实入口与模块重构。本轮不启动重构、真实训练或付费 API。
- 继续保留未提交研究文件 `docs/research/math-equivalence-and-sft-loss.md`，不混入本任务提交。

| 步骤 | 状态 | 完成条件 |
| --- | --- | --- |
| 第一版要求交付 | complete | 2fb7724/PR52 已交付；本轮新要求取代其调用边界。 |
| 最新边界调查与澄清 | complete | 已核实产物与训练交接；用户已确认来源/续跑参数、单次评分汇总及部分结果。 |
| 最新要求同步与检查 | complete | git diff --check 通过；11 份文档的 38 个本地链接有效，无未决问题或旧调用边界冲突。 |
| 更新需求 PR52 | complete | 要求提交 34fcd8a 已普通推送；用户明确允许一次 gh pr edit，标题/说明已同步。用户合入后再实现。 |

GitHub 连接器更新 PR 的参数字段名修正后返回 403。用户已明确允许本次仅用 gh pr edit 更新 PR52 标题/说明，操作成功；该次例外不更改后续 gh 使用边界。

本轮遇到一次文档补丁上下文未匹配；去掉多余的末尾上下文后已重新应用，无业务代码受影响。

## 前次已完成计划：现有入口自动准备旧 SFT 数据

## 用户最新决定

用户于2026-11-06明确纠正：兼容环境按标准 Conda 命名环境流程创建，再用其 Python/pip 安装依赖；用户继续运行 train_sft.py、train_post_sft.py、evaluate_checkpoint.py，不额外运行 migrate_sft.py，也不增加启用迁移的开关。sample_checkpoint.py使用同一内部逻辑。

- 读取需要兼容的旧来源且没有可用缓存时，现有入口显示准备进度，自动创建CPU兼容环境、处理并核对旧数据，成功后继续当前命令。
- 命名环境交由Conda环境目录配置决定，不将环境放入自建cache前缀或/tmp，不改写用户环境目录配置。
- 新SFT、兼容的主环境产物和已有有效缓存直接正常工作；有效缓存路径不查询Conda或调用旧Python。
- 完整来源/划分/分词校验、权重和run记录保留、JSONL/Markdown按条数追加不变；旧配置、安装与数据失败仍明确报错。

## 当前实施

PR #50已合入main/ebbe9ca。已从最新远程main创建codex/implicit-sft-migration，通过普通合并7215c05完整承接PR49/c406fc3，保留双方历史；三份规划冲突保留PR50最新决定。

- prepared_sft_data.ensure_prepared_sft_data供原有四个入口调用，识别需要兼容的旧来源并自动准备，主环境新SFT/兼容来源/已有缓存均跳过旧环境。
- 控制逻辑移入legacy/sft_data/migration.py，根migrate_sft.py删除。兼容环境标准conda create -n，再用Conda查询的Python/pip安装，位置服从envs_dirs；cache只放锁与状态。
- worker接受旧SFT checkpoint和final，不依赖final存在。完整data_plan与缓存读取校验、原权重和run信息、JSONL/Markdown追加规则保留。
- README去掉独立迁移命令和compat-root，原命令无需增加启用开关。

| 步骤 | 状态 | 完成条件 |
| --- | --- | --- |
| 承接历史 | complete | 最新main新分支，普通合并PR49代码和历史。 |
| 原入口自动调用 | complete | 新SFT/兼容来源/缓存正常路径，旧来源首次准备后继续原任务。 |
| 标准Conda环境 | complete | -n创建、Conda查询解释器、pip安装，标记/锁/恢复/用户环境保护。 |
| 针对性验证 | complete | 20迁移边界检查+45微型检查；实际原评测入口创建全新命名环境、准备旧数据并生成。 |
| 文档与交付 | complete | README和规划更新、差异检查，提交推送新功能PR。 |

未提交docs/research/math-equivalence-and-sft-loss.md继续保留并排除提交。本轮不跑真实4B训练、全量数据处理或付费API。

## 历史

此前详细规划在既有Git历史及PR49的docs/planning-archive/2026-11-06-pre-migration/中保留；当前文件只描述本次纠正。按用户最新决定执行，不再沿用独立迁移入口方案。

## 本轮交付

功能与验收已完成，提交435f3d6并推送，PR #51：https://github.com/FishSkin2002/LM_post_training/pull/51。该PR包含PR49完整vLLM实现及按PR50修正后的流程，无需再合入旧PR49；合并/关闭由用户操作。
