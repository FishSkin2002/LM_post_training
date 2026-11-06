# 当前进展

## 2026-11-06：整理上下文并开始自动迁移功能

用户确认 PR #48 已合入，要求整理上下文后开工。旧 task_plan/progress/findings 共2080行，已逐字归档到 docs/planning-archive/2026-11-06-pre-migration/，当前文件仅保留活跃任务信息。

通过受控入口从最新远程 main 创建 codex/automatic-sft-migration；已通过普通合并6316a74承接 b53f9cf 原 vLLM 功能，主线基点22fd7a5。主环境和兼容环境的全部产品决定已确认，见 task_plan.md 与 docs/automatic-sft-migration.md。

## 已有功能与证据

- codex/vllm-generation / b53f9cf：17个文件，公共生成 backend、eval/sample、GRPO colocate、SFT/DPO/GRPO后评测进程交接、cu128依赖预检、手动准备数据导出及跨 Transformers 版本兼容。
- 原环境与新环境各45项检查通过；GPU微型底座/LoRA生成与两步GRPO（合成奖励）通过；未跑真实4B训练。
- 实际Qwen分词器151669词表、完整后端和中英数学提示IDs跨环境相同。
- 旧环境微型SFT适配器配导出数据，在新环境真实evaluate_checkpoint生成通过；完成续跑不加载引擎、不重判、不改写JSONL/MD。
- 旧代码原先被GitHub远端commit_refs_rules拒推；随后PR48要求分支已成功推送创建，旧代码尚未单独发布。现已承接并统一交付功能PR #49。

## 自动迁移实现与验收

- 根 requirements.txt 已切为主 vLLM/cu128 栈，独立 requirements-vllm-cu128.txt 已收敛；历史直接依赖、CPU 覆盖、配置和 worker 置于 legacy/sft_data/。
- migrate_sft.py 提供主环境独立迁移入口；按配置摘要管理 Conda 前缀，明确 conda-forge 渠道和 pip 来源，CPU-only 版本检查、依赖检查成功后发布 ready 标记。环境与输出各有进程锁；安装中断可重试，不完整 Conda 创建明确提示改用新 compat-root，不覆盖用户已有环境。
- worker 不加载权重，按原完整 data_plan 核对来源、筛选、token、组与划分后导出；主进程再次核对后原子发布 final/ 同级 prepared-sft/。原权重和 run.json 不变。
- eval/sample/SFT 加训/DPO/GRPO 共享缓存发现，原始模型 --sft-source 对照复用同一缓存；阶段和报告记录实际路径。新 SFT 记录数据依赖版本，无需旧栈迁移。缺缓存时提示迁移，不从日常命令自动安装。
- 真实首次验收：仅从主 Python 启动，在此前不存在的 /tmp/qwen-sft-migration-end-to-end 前缀通过 Conda/pip 完成 CPU 环境创建、旧数据处理、原划分核对和输出发布；没有借用 base 原环境。原始输入来自合成20题微型旧SFT；唯一替换是私有 worker 中 Hugging Face 原数据读取，环境和处理均为实际运行。
- CPU环境 Python3.12.16 / torch2.6.0+cpu，torch.version.cuda=None，pip check通过；指纹4e255131893486f9、验证索引[19,5]、完整题目/解答/参考/组/源索引/输入IDs/labels逐条相等。结果含全部实际包版本，位于 /tmp/qwen-migration-fixture/migration-result.json；完整日志 /tmp/qwen-sft-migration-end-to-end.log。
- 重复迁移在阻止 Conda/子进程的检查下通过；额外运行真实公共CLI并把CONDA_EXE指向不存在路径，仍直接复用。未重新判分或改写历史生成 JSONL/Markdown。
- 主环境针对性检查：生成接口13项、迁移14项、DPO/GRPO微型检查4项通过；此前本轮同时执行的Instruct微型28项通过。覆盖旧缓存自动发现、来源不符拒绝、安装失败/重试、两个层次的并发、未受管环境拒改与未完成缓存。主环境 pip check通过。
- README 提供唯一主安装入口、一次迁移和日常使用完整命令。未跑真实4B训练、全量数据处理或付费API。功能已提交412710e并推送，PR #49：https://github.com/FishSkin2002/LM_post_training/pull/49；等待用户审查合入。

合并文档冲突已解决；受控Git入口无法完成冲突后的merge提交，已获一次性宿主审批完成普通merge，不改写历史。

## 历史

详见 [完整原进展](docs/planning-archive/2026-11-06-pre-migration/progress.md)。
