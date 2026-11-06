# 当前计划：项目整理要求交付

2026-11-06，用户调用 grill-with-docs，要求归拢业务脚本、统一训练入口、删除抽样问答入口，并将三级判定整合到评测，按各级进度追加。

- 用户已确认全部问题：目录为 `utils/`；训练统一 `train.py`；全量一级 → 全量二级 → 统一三级；训练结束自动评测也三级；不保留 `--input`，原命令自动发现各级进度并追加；缺密钥在三级前终端隐藏输入。
- 当前阶段：要求已交付 [PR #52](https://github.com/FishSkin2002/LM_post_training/pull/52)，等待用户审查合入。入口以[项目整理约定](docs/project-layout.md)为准，评分与恢复以[三级判定约定](docs/evaluation-rescoring.md)为准。尚未修改业务代码。
- Git：PR51 已在远程 main/1cf99db 合入；从最新远程 main 创建 `codex/project-layout-requirements`。本轮 open PR 一次性查询仅有 #33，无本任务可续用 PR。
- 后续：用户合入并要求继续后，再从最新远程 main 新建功能分支，按小步骤落实入口与模块重构。本轮不启动重构、真实训练或付费 API。
- 继续保留未提交研究文件 `docs/research/math-equivalence-and-sft-loss.md`，不混入本任务提交。

| 步骤 | 状态 | 完成条件 |
| --- | --- | --- |
| 事实调查与逐轮澄清 | complete | 训练入口、抽样依赖及旧判分核实；用户已回答全部待决问题。 |
| 要求与术语同步 | complete | 统一入口、三级顺序、追加与密钥输入写入约定；旧文档冲突同步消除。 |
| 文档检查 | complete | git diff --check 通过；11 份变更文档的 39 个本地链接有效，旧流程冲突已同步。 |
| 独立需求 PR | complete | 要求提交 2fb7724 已正常推送，PR #52 指向 main；评审与合入由用户操作。 |

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
