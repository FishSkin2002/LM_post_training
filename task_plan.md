# 当前计划：现有入口自动准备旧 SFT 数据

## 用户最新决定

用户于2026-11-06明确纠正：兼容环境按标准 Conda 命名环境流程创建，再用其 Python/pip 安装依赖；用户继续运行 train_sft.py、train_post_sft.py、evaluate_checkpoint.py，不额外运行 migrate_sft.py，也不增加启用迁移的开关。sample_checkpoint.py使用同一内部逻辑。

- 读取需要兼容的旧来源且没有可用缓存时，现有入口显示准备进度，自动创建CPU兼容环境、处理并核对旧数据，成功后继续当前命令。
- 命名环境交由Conda环境目录配置决定，不将环境放入自建cache前缀或/tmp，不改写用户环境目录配置。
- 新SFT、兼容的主环境产物和已有有效缓存直接正常工作；有效缓存路径不查询Conda或调用旧Python。
- 完整来源/划分/分词校验、权重和run记录保留、JSONL/Markdown按条数追加不变；旧配置、安装与数据失败仍明确报错。

## 本轮边界

按docs/git-workflow.md，修改已合入的项目要求必须先交付独立要求PR，用户合入后再实施。此轮只修改docs/automatic-sft-migration.md及docs/vllm-generation.md与本任务规划记录，不修改功能实现或README当前命令。

## Git与交付

- 当前分支codex/implicit-sft-migration-requirements，从最新远程main（22fd7a5）创建。
- 一次open列表确认PR49（codex/automatic-sft-migration）与PR33仍open；本轮不再查询，PR49暂缓合入，旧实现保持完整。
- 之前vLLM和自动迁移代码在PR49，提交b53f9cf/6316a74/412710e/c406fc3；它目前仍要求独立迁移并管理自建前缀。
- 保留未提交的docs/research/math-equivalence-and-sft-loss.md，排除本次提交。

| 步骤 | 状态 | 完成条件 |
| --- | --- | --- |
| 核对现状 | complete | 区分/tmp仅验收路径、默认cache前缀与用户新要求；核对PR49仍open。 |
| 修订要求 | complete | 原有脚本自动触发、标准Conda命名环境、缓存直接复用；保留数据与生成协议。 |
| 验证与提交 | complete | 文档无矛盾、diff检查通过，提交推送独立要求PR。 |
| 功能修订 | pending | 用户合入要求后，从最新远程main新建功能分支，承接PR49历史后修改并验证。 |

## 历史

此前详细规划在既有Git历史及PR49的docs/planning-archive/2026-11-06-pre-migration/中保留；当前文件只描述本次纠正。按用户最新决定执行，不再沿用独立迁移入口方案。

## 本轮交付

要求PR #50：https://github.com/FishSkin2002/LM_post_training/pull/50。文档检查通过，代码尚未修订；等待用户合入要求，再承接PR49实施新方案。
