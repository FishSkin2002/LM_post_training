# 当前进展

## 2026-11-06：修正并润色大模型判分表述

- 用户指出相关操作是在调用在线大模型 API 判分，并要求措辞自然。README 统一使用“大模型判分”，在脚本介绍中说明“通过 API 调用在线大模型，对已保存的 OpenR1 评测回答进行判分”。
- 同步修正目录、产物注释、生成条件、用途标题、密钥与配置说明；不再用手动操作描述判分方式。命令、文件名和代码保持原样。
- 本轮受控 fetch 确认 main 仍为 7d45fa1，一次 open 列表确认 PR55 仍 open，继续原分支。README 措辞检索、全部命令示例一致性、链接及 git diff --check 通过；本次更新沿用 PR55。

## 2026-11-06：README 整理范围已确认

- 按用户调用的 grill-with-docs 读取 grilling/domain-modeling，并委派两个只读核查，确认根目录当前仅 train/evaluate/rescore 三个业务入口；DPO/GRPO 运行独立，但实现共处 post_sft。
- 用户确认：每个根部脚本一句介绍，保留用途命令与必要参数；删除原理、内部实现及大段环境说明，不转存新文档；本轮只整理文档，下个代码步骤再拆 DPO/GRPO。
- 目录总览不展开文件夹，全部文件/文件夹有注释；另列 configs 文件夹层级和 training 的完整文件示例，聊天格式独立成节，实验报告使用醒目入口。
- 受控 fetch 和本轮一次 open PR 查询确认 PR54 已合入 main/7d45fa1；受控 start 首次因沙箱 .git 只读失败，正式宿主审批后创建 codex/readme-layout-and-script-guide。
- 使用 planning-with-files-zh 和 writing-for-agents 维护本轮记录；研究文件未改，不运行真实训练或付费 API。
- README 已重组：总览文件夹不展开、所有条目有注释；configs 单列文件夹，training 示例展开 checkpoint/final/日志/评测/条件性数据缓存；聊天格式单独展示，标题下突出实验报告目录及现有实验一。
- 主要脚本仅保留根部三个入口，各一句介绍；SFT/DPO/GRPO 的用途命令与参数就近列出，DPO/GRPO 的新训练和恢复分别展示。原理、实现、内部工具和测试说明直接删除，未另建文档。脚本部分由 13568 减为 6090 字符（约减少 55%）。
- 验证 train 的 sft/dpo/grpo、evaluate、rescore 五组实际 --help；两处报告链接、配置路径、目录结构与注释、原生聊天 system 与代码的一致性均通过。输出示例已对照保存逻辑，文件名中的 OpenR1 大小写已修正。
- 核查时误读不存在的 legacy/sft_data/environment.py，改用 rg --files 后定位 migration.py 并确认锁文件保留；未影响变更。仅文档改动，不运行训练回归或付费 API。
- README 与本轮记录已提交 6297e28，正常推送并创建 [PR55](https://github.com/FishSkin2002/LM_post_training/pull/55)。本轮整理完成，下一步 DPO/GRPO 代码拆分等待用户继续；既有独立三级、汇总与训练后编排仍未实施。

## 2026-11-06：开始一二级评测整合

- 用户确认 PR53 已合入；远程 main/33367b2 已包含该 PR，新建功能分支 `codex/two-stage-checkpoint-evaluation`。
- 本轮把旧重评中的本地规则提取为共用模块，evaluate 先完成全量生成/一级，再按独立进度完成二级；不调用 API 或汇总。
- 保留原一级 JSONL/Markdown 与顶层一级字段；二级结果另存并附入 JSON 的 rule_scoring，供后续三级直接使用。旧重评对含已保存规则结果的输入直接沿用，避免重复二级。
- 已接通公共本地规则与二级进度，API 与训练依赖不进入本地规则模块；一级完成且生成器关闭后才执行二级。二级结果使用 stage2.progress.jsonl/stage2.md，各题本地判决和全局完成状态附入 JSON。
- 新增六项实际评测集成检查，覆盖补判与截断、阶段顺序、生成中断、二级中断和完成续跑、Markdown 失败后复用已 flush 判决、旧重评复用规则及 GSM8K；连同原重评 23 项通过。README 已按当前功能更新，累计各级汇总留待后续独立脚本。
- README 补丁的 boxed 反斜杠上下文未匹配，改为按段落定位后应用；代码检查无报错。
- 主环境完整微型/离线回归 87 项通过（新增六项 + 原 81 项），覆盖统一训练、生成、分级评分、旧数据自动准备与指纹恢复。随后补充：部分二级报告禁止旧 API 入口提前评分；定向复测及边界检查继续执行。
- 补充的 API 前置条件与跨批次阶段屏障检查通过：23 项定向复测及六项边界复测均通过；完成续跑不执行旧评分或 Markdown 渲染，首级和二级的历史字节不变。本地规则模块导入不带 torch、API 模块或重评入口。
- 语法、差异及八个本地文档链接检查通过。日志 /tmp/qwen-two-stage-tests.log、/tmp/qwen-two-stage-targeted.log、/tmp/qwen-two-stage-boundaries.log。未运行真实大模型或付费 API；本轮准备交付一二级整合。
- 第二步功能提交 3949c92 已正常推送，创建 [PR54](https://github.com/FishSkin2002/LM_post_training/pull/54)。等待用户继续后实现独立三级的新建/续跑；独立汇总及训练后编排仍未实施。未提交研究文件未改、未纳入。

## 2026-11-06：开始目录与统一训练入口

- 用户授权开始改动；远程 main/3e3b108 已合入 PR52，从最新 main 创建 `codex/organize-training-entrypoints`。
- 本轮推进一个结构整理步骤：业务实现迁入 utils/，train.py 统一 SFT/DPO/GRPO，移除随机抽样入口，同步 legacy worker、测试、README 和路径定位。
- 评分拆分与自动编排留作后续步骤；原 rescore 命令在拆分前暂保留薄入口。验证沿用随机微型模型与离线测试，不运行真实训练或付费 API。
- 已将业务模块移入 utils/、接通 train.py 三阶段及共用模型/保存逻辑，移除随机抽样；配置、默认输出、评测进程替换和 legacy 导入路径已同步，README 按当前实现更新。
- 首轮现有微型检查 80 项通过。额外对照发现：数据内容与来源完全相同，但迁移模块名改变 datasets 对 callable 的序列化指纹（ac6f258cf7c0f02e → 7df3d492776f694a）；已为数据转换保留历史序列化模块身份，接下来核对真实迁移前后指纹及内容。
- 静态检查中已修正批量导入替换误改函数名、SFT 续训输出路径比较及一处补丁上下文不匹配；未影响既有产物。
- 主环境微型检查再次通过 80 项，SFT、DPO、GRPO 的实际微型训练均通过统一 CLI 调用；训练恢复、策略更新、生成续跑、重评与旧数据自动准备检查保持通过。
- 与 main/3e3b108 的原代码在隔离解释器中比较：主环境与已安装旧 CPU 环境各自的 OpenR1/GSM8K/DPO/GRPO 共八种路径，全部文本、词元、来源信息、数据指纹及 SFT 划分完全相等。主环境 OpenR1 指纹 ac6f258cf7c0f02e，旧环境 4e255131893486f9。比较脚本首次合成 GRPO 记录缺少 ability，补齐后通过。
- 证据：/tmp/qwen-layout-tests.log、/tmp/qwen-layout-identity.log、/tmp/qwen-layout-data-identity.json；额外增加已保存指纹恢复的回归检查。CLI 帮助及互斥参数在仓库外使用标准库 Python 验证通过，无需加载训练依赖。
- 新增指纹回归在主环境和旧 CPU 环境各通过 1 项（主环境累计 81 项）；评测、重评的根部薄入口在仓库外 `--help` 通过，数据检查模块 CLI、全部源码语法及 8 个本地 Markdown 链接检查通过。
- 暂存后差异检查发现三个新文件多余 EOF 空行，已清理；本轮只归拢代码、统一训练和删除随机抽样，评分拆分尚未实施。README 明确现阶段可用命令，原重评暂保留根部薄入口。
- 第一阶段功能提交 ab900f0 已正常推送，创建 [PR53](https://github.com/FishSkin2002/LM_post_training/pull/53)。第一步已完成；等待用户继续后，实施一二级整合、独立三级及汇总，不自动跨步骤执行。研究文件未纳入提交。

## 2026-11-06：独立三级与汇总参数已确认

- 用户确认新建三级用 `--evaluation`、续跑用 `--resume-from`；汇总选定一次评分，三级未完成也可手动汇总，未处理/待复核分开且总分母不变。
- 将项目整理与评分约定恢复为正式目标，明确四个根部入口、一二级手动不触发后续、训练结束自动编排、三级新建/续跑互斥及保存来源/配置、汇总 `--scoring` 选择本次运行。
- 补充首个 API 请求前保存运行信息、取消后续跑/部分汇总、累计各级正确率与固定分母样例等验收。既有训练算法、模型与研究文件未改。
- 本轮仍使用 PR52 的需求分支；上轮 open 查询结果复用，本轮受控 fetch 确认 main 仍为 1cf99db。接下来检查并推送最新要求及同步 PR 说明。
- 文档验证通过：git diff --check；11 份 Markdown 的 38 个本地文件/章节链接全部有效；当前约定无待确认标记或旧调用边界冲突。变更范围仅需求、术语与规划，未运行训练或付费 API。
- 交付：最新要求提交 34fcd8a 已推送到 PR52，包含此前 9e51159 的调用边界调整。GitHub 连接器首次字段校验未通过，修正为 repository_full_name/pr_number 后 API 返回 403（Resource not accessible by integration），未改动远端元数据。
- 用户通过明确回复允许本次用 gh pr edit 仅更新 PR52 标题/说明；已按准备好的完整文本执行成功，PR 标题为 Define separate evaluation, API scoring, and summary requirements。未修改 PR 状态、评审或合并设置；一次性例外不扩大未来授权。
- 本轮要求交付完成，等待用户合入并要求继续后，从最新 main 新建功能分支。业务代码与未提交研究文件保持原样。

## 2026-11-06：改为独立三级评分与汇总

- 用户明确改变调用边界：一二级放在一起，三级单独脚本，汇总再单独脚本；训练结束自动依次一二级、三级、汇总。手动一二级不调用三级，三级和汇总可手动运行。
- 三级选择已有评分运行才继续；不传选择参数就创建新评分，重名加数字。原有 utils/、统一训练、删除随机抽样、逐级追加和隐藏输入密钥的决定保留。
- 按 grilling 的事实调查要求委派只读复核：保存文本足够三级/汇总独立使用；一级中断JSONL没有固定全集分母；训练末 vLLM 的 os.execv 不会返回，需要内部编排层承接后续步骤，不能直接在 execv 后追加代码。
- 远程 main 仍为 1cf99db，本轮一次性 open 查询确认 PR52 仍 open。继续同一需求分支，本地先同步已定边界与待决问题，暂不推送未确认的设计。
- 待用户决定新评分来源与旧运行选择参数、单次评分汇总选择、三级中断时是否可手动部分汇总。业务代码与未提交研究文件未改。

## 2026-11-06：项目整理需求确认与文档同步

- 用户确认全量一级 → 全量二级 → 统一三级，不保留 `--input`，各级和一级一样自动找进度、向末尾追加；缺密钥按推荐使用终端隐藏输入。结合上一轮答复，设计问题全部确定。
- 将 `docs/project-layout.md` 从澄清草稿改为正式目标约定，明确 `utils/`、统一训练、删除旧入口、保留配置/训练产物与旧数据兼容、各级进度及功能验收。
- 同步三级判定、训练、后续训练、vLLM、旧数据自动准备及模型样本文档，消除独立重评、抽样入口与统一流程的冲突；CONTEXT 只新增三种判定的领域定义。
- 终端输入仅在恢复后尚有三级工作且缺密钥时发生；取消、空输入或无终端保留进度，不把未请求题写成待复核。历史 JSONL/Markdown 只追加，完成重跑不生成、不重判、不请求密钥或 API。
- 一次文档补丁的无关末行上下文不匹配，去掉该上下文后成功应用。业务代码、已有产物和用户研究文件未改。
- 验证：git diff --check 通过；11 份变更 Markdown 的 39 个本地文件/锚点链接全部有效；无指向已改名旧章节的引用，旧独立重评/抽样要求已同步。仅文档变更，不运行训练或 API；接下来提交并独立交付要求 PR。
- 交付：要求提交 2fb7724 正常推送，已创建 [PR #52](https://github.com/FishSkin2002/LM_post_training/pull/52)，目标 main。等待用户合入后再开始功能；不通过本地 main 或代用户合并。用户未提交研究文件未纳入。

## 2026-11-06：项目整理第一轮答复已记录

- 用户将业务目录名改为 `utils/`，确认训练结束自动评测执行完整三级；缺密钥应在三级开始前交互输入。
- 已解释旧流程：原评测先全量生成及一级，独立重评再逐题二级及必要的三级。原评测自动找同名进度追加；`--input` 只是独立重评的输入参数，不是自动续跑的条件。
- 继续只读核实：当前没有 GUI 输入实现；APIJudge 每题读环境变量；训练结束的 os.execv 没有主动断开原终端。待用户确认终端隐藏输入形式及新阶段顺序。
- 更新 `docs/project-layout.md` 和规划；未修改业务代码、未索取真实密钥、未运行 API。文档差异检查通过。

## 2026-11-06：项目整理第一轮澄清

- 已读取 grill-with-docs 及其组合的 grilling/domain-modeling；使用 planning-with-files-zh 和 writing-for-agents 维护本轮资料。按 grilling 要求，委派两个只读调查分别核对评测与训练。
- 远程 main/1cf99db 已包含 PR51；受控入口创建独立需求分支 `codex/project-layout-requirements`。open PR 一次性查询仅有 #33。
- 核实评测对 sample_checkpoint 只有共享生成函数的间接依赖；实际训练只有两个可执行入口；记录 index 非连续数值，当前一二三级没有分开的执行进度。
- 新增 `docs/project-layout.md`，区分用户目标、代码事实和待决问题。待用户回答第一轮问题再继续；尚未修改业务代码或既有训练/评分要求。
- 本轮仅检查代码与文档，未运行训练、评测生成或付费 API。未提交研究文件保持原样。

## 2026-11-06：现有入口自动准备与标准Conda环境已实现

用户确认PR50已合入。受控入口从最新main/ebbe9ca创建codex/implicit-sft-migration，普通合并7215c05承接PR49/c406fc3完整历史；规划文件冲突保留最新要求，受控入口不能完成冲突后的merge，已获一次性正式宿主审批完成普通merge提交。

- 原train_sft、train_post_sft、evaluate_checkpoint、sample_checkpoint入口通过统一helper自动准备旧SFT数据，无需额外命令或开关。新SFT、当前数据环境兼容的来源与已有缓存跳过旧环境；DPO/GRPO及原始模型对照追溯原SFT。
- 根migrate_sft.py已删除，控制模块在legacy/sft_data/migration.py。Conda标准命名环境创建，再用Conda查询的实际Python/pip安装CPU依赖；envs_dirs不被覆盖。用户缓存仅存状态/锁，不安装环境。并发锁、ready标记、安装中断重试和未受管环境保护保留。
- 旧SFT checkpoint直接准备，不要求final存在。数据完整来源/指纹/token/划分核对后才原子发布缓存；原权重/run和历史JSONL/Markdown保持既有行为。
- README已改为主安装与原命令使用说明，去掉独立迁移与--compat-root步骤。

验证：
- tools.test_sft_migration：20项通过，覆盖所有原入口首次准备、新SFT/当前来源跳过、缓存复用、旧checkpoint、来源不符、子进程失败、命名环境安装中断、双层并发与未受管环境拒改。
- tools.test_generation + tools.test_instruct_sft + tools.test_post_sft：45项微型检查通过，未运行真实4B训练。
- 真实首次验收由evaluate_checkpoint.main正常命令触发：自动conda create -n建立此前不存在的qwen-sft-data-instruct-datasets51-cpu-v1-1e87c5bc6256，实际位于/home/yunli/miniconda3/envs/下；pip安装与check通过，torch2.6.0+cpu、CUDA runtime=None。
- 随后CPU worker实际处理合成20题旧数据并核对，普通评测入口继续加载随机微型底座与旧LoRA并生成1条回答。仅原数据读取和底座规模使用测试替代，Conda/pip/处理/校验/生成真实执行。
- 指纹仍为4e255131893486f9，验证索引[19,5]；全部题目、监督/参考、来源/组、token IDs/labels逐条相等；SFT产物字节未变。完成续跑在禁止Conda/旧子进程/生成器/评分器的检查下通过，JSONL与Markdown字节未变。
- 日志/tmp/qwen-named-migration-smoke.log；结果/tmp/qwen-named-migration-fixture/migration-result.json；/tmp仅存合成测试数据和日志，兼容环境不在/tmp。

功能已提交435f3d6并推送，PR #51：https://github.com/FishSkin2002/LM_post_training/pull/51；等待用户审查合入。未更改主环境、系统Toolkit或真实SFT产物；未调用付费API。此前独立入口方案仅为PR49旧实现，当前已被本轮流程替换。

## 历史

PR48确立主vLLM依赖和CPU兼容处理；PR49完成基础实现与GPU微型验证但入口和环境位置需修正；PR50已合入这次明确的新要求。旧详细记录保存在已有Git历史与docs/planning-archive/2026-11-06-pre-migration/。
