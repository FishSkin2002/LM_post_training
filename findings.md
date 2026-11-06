# 当前实现发现

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

## Git

main基点ebbe9ca（用户合入PR50），分支codex/implicit-sft-migration，普通合并7215c05完整承接PR49。日常写操作走受控入口，历史只追加。PR49未由Codex合并或关闭；新功能PR将包含其完整实现。

未提交研究docs/research/math-equivalence-and-sft-loss.md保留、不混入。旧历史见Git既有提交及docs/planning-archive/2026-11-06-pre-migration/。
