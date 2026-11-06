# 当前计划：现有入口自动准备旧 SFT 数据

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
| 文档与交付 | in_progress | README和规划更新、差异检查，提交推送新功能PR。 |

未提交docs/research/math-equivalence-and-sft-loss.md继续保留并排除提交。本轮不跑真实4B训练、全量数据处理或付费API。

## 历史

此前详细规划在既有Git历史及PR49的docs/planning-archive/2026-11-06-pre-migration/中保留；当前文件只描述本次纠正。按用户最新决定执行，不再沿用独立迁移入口方案。

## 本轮交付

功能开发和验收完成，待提交推送新功能PR。PR49的代码和历史已经承接，新PR将包含完整vLLM功能及按PR50修正后的入口流程；PR的合并或关闭仍由用户操作。
