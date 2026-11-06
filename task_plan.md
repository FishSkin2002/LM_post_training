# 当前计划：独立三级与汇总要求交付

2026-11-06，用户调用 grill-with-docs，要求归拢业务脚本、统一训练入口、删除抽样问答入口，并将三级判定整合到评测，按各级进度追加。

- 用户最新调整替代此前“一二级自动三级”决定：一二级共用脚本，三级与汇总分别独立；只有训练结束自动依次串联。三级选已有运行才续跑，否则新建，重名加数字。`utils/`、统一训练、删除抽样入口、隐藏输入密钥及按进度追加仍保留。
- 用户已确认增补问题：三级新建 `--evaluation`、续跑 `--resume-from`；汇总选定一次评分，支持三级中断后的部分结果，固定总题数并区分待复核与未处理。
- 当前阶段：最新要求、验收与文档检查已完成，更新 PR52。入口以[项目整理约定](docs/project-layout.md)为准；尚未修改业务代码。
- Git：远程 main 仍为 1cf99db；上轮一次性 open 查询确认 #52 仍 open，继续使用 `codex/project-layout-requirements`，无需新 PR。本轮复用该查询，不读取 PR 正文或评审。
- 后续：用户合入并要求继续后，再从最新远程 main 新建功能分支，按小步骤落实入口与模块重构。本轮不启动重构、真实训练或付费 API。
- 继续保留未提交研究文件 `docs/research/math-equivalence-and-sft-loss.md`，不混入本任务提交。

| 步骤 | 状态 | 完成条件 |
| --- | --- | --- |
| 第一版要求交付 | complete | 2fb7724/PR52 已交付；本轮新要求取代其调用边界。 |
| 最新边界调查与澄清 | complete | 已核实产物与训练交接；用户已确认来源/续跑参数、单次评分汇总及部分结果。 |
| 最新要求同步与检查 | complete | git diff --check 通过；11 份文档的 38 个本地链接有效，无未决问题或旧调用边界冲突。 |
| 更新需求 PR52 | in_progress | 最终要求普通提交推送，保持已有历史；用户审查合入后再实现。 |

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
