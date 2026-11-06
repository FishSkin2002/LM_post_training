# 当前计划：统一 vLLM 主环境与旧 SFT 自动迁移

## 目标与授权

用户已合入 PR #48，并要求整理上下文后继续实现。要求见 [自动迁移约定](docs/automatic-sft-migration.md) 和 [vLLM 生成约定](docs/vllm-generation.md)。所有主要决定均已确认，不再反复询问。

- 根目录 requirements.txt 是唯一主环境安装入口，采用已验证 vLLM/cu128 栈；环境名任意。
- 旧依赖、兼容配置、旧数据处理 worker 集中到 legacy/sft_data/，仅由程序调用。
- 独立迁移命令从主环境启动；仅有主环境、无旧环境时，自动 Conda 建 Python/pip，再 pip 安装固定 CPU 兼容依赖。
- 使用 CPU 准备旧 SFT 数据，核对原数据身份和训练/验证划分后发布缓存；后续普通训练/评测/抽样自动定位复用。
- 系统只装 Toolkit12.8 即可，不要求12.4；主环境需要相应驱动和 Python 依赖。
- 兼容环境与迁移缓存支持复用、并发/中断恢复；未知旧配置或原数据不一致明确失败，不绕过来源校验。
- 不重新核查生成 JSONL 历史内容，不重判或重写 Markdown；保留原有按条数追加。

## 当前代码与 Git

新分支 codex/automatic-sft-migration 已通过受控入口从最新 origin/main 创建，PR48 合入基点为22fd7a5；已用普通合并6316a74承接 codex/vllm-generation 的 b53f9cf，保留双方历史。

b53f9cf 已完成公共 vLLM 生成、GRPO、环境预检及手动旧数据导出，原/新环境各45项和 GPU微型验证通过；它从未成功创建功能PR，不能丢弃或重写。通过正常 merge 保留该提交历史；如 docs/vllm-generation.md 冲突，以新要求为准并保留有用事实。

日常 Git 写入口：/usr/bin/python3 -I -S /mnt/c/Users/yunli/.codex/bin/qwen-git-workflow.py。禁止 reset/rebase/amend/force，main 只由用户 GitHub 合入。gh 仅 PR create 与一次 open 编号/分支列表。origin 已获用户正式批准改为 git@github.com:FishSkin2002/LM_post_training.git。

## 步骤

| 阶段 | 状态 | 完成条件 |
| --- | --- | --- |
| 整理与恢复 | complete | 旧规划原样归档、当前文件压缩，创建新功能分支，承接 b53f9cf。 |
| 主依赖与兼容 worker | complete | 主 requirements、legacy 私有 CPU 配置和 worker，主环境名称无关。 |
| 迁移与自动复用 | complete | 自动 Conda/pip、锁/完成标记、缓存原子发布、来源核对和普通入口发现。 |
| 针对性验证 | complete | 单元边界、真实全新 CPU 前缀首迁、原身份/划分/token不变、有效缓存不调用旧Python。 |
| 文档与交付 | complete | README 改为主环境安装/一次迁移/正常命令；必要检查通过后提交推送并提一个完整功能PR。 |

## 范围

仅随机微型模型与合成数据验证，不跑真实4B完整训练或付费API。首次真实兼容环境重建是已授权验收，不能借用已有base旧栈冒充。

## 历史

原2080行规划记录已逐字归档在 [2026-11-06归档](docs/planning-archive/2026-11-06-pre-migration/)。未提交研究 docs/research/math-equivalence-and-sft-loss.md 保留，不混入功能变更。

## 交付

功能已提交并推送，PR #49：https://github.com/FishSkin2002/LM_post_training/pull/49。待用户审查合入；本轮不推进后续正式训练或数据运行。
