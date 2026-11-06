# 当前进展

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
