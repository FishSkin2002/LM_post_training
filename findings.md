# 当前实现发现

## 项目整理调查（2026-11-06）

- 用户两轮答复已确认：业务目录 `utils/`，全量一级 → 全量二级 → 统一三级，训练结束自动评测也三级，缺密钥在三级开始前终端隐藏输入；不保留 `--input`，原命令自动找各级进度追加。
- 自动进度发现本来就在 evaluate 中：按来源/数据/题数或 `--output` 算报告路径，再读同名 `.progress.jsonl`。只有独立 rescore 要求 `--input`；整合不应将其变为续跑前提。
- 输入只应发生在恢复进度后仍有待请求的三级题时。当前 APIJudge 每题读取环境变量、没有交互输入；源码无 GUI 框架。os.execv 保留既有标准输入连接，不保证后台启动具备终端。

- 代码事实和需求澄清统一记录在 [项目整理需求](docs/project-layout.md)。`evaluate_checkpoint.py:17` 间接导入 `sample_checkpoint.generate_answers`，实际函数位于 `generation.py:109`；不依赖随机抽样 CLI。
- `training_workflow.py` 无 CLI。SFT 与 DPO/GRPO 两个入口之间存在共用模型加载函数的反向依赖，可在统一入口实施时下沉到业务共用模块。
- 生成 JSONL 以已保存条数定位；重评 JSONL 以 index 读取判决；index 是验证集顺序中的原数据集索引，不能按数值最大值推进。现有独立重评逐题串联二三级，目标改为各级全量结束后进入下一级。
- 旧实现缺 API 密钥会计为已处理的待复核。目标改成三级前输入；取消/空输入/无终端保留进度并退出，未请求题不写为待复核。训练结束调用的同一评测也执行三级。
- 根部模块使用 `__file__` 定位配置、默认输出与评测入口；迁入目录时需保持仓库根定位。legacy worker 及测试导入要随重构同步调整，旧数据依赖边界不应因普通移文件被打破。
- 远程 main/1cf99db 已合入 PR51；新需求分支为 `codex/project-layout-requirements`。本次查询仅 #33 仍 open，后续不重复查询。

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
