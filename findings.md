# 当前实现发现

## 最新产品边界

- 用户要保留原命令：train_sft.py、train_post_sft.py、evaluate_checkpoint.py；sample_checkpoint.py复用同一内部逻辑。
- 旧数据准备自动触发，不再要求独立迁移脚本或启用开关。主环境、已有缓存不需要旧环境处理。
- 标准conda create -n命名环境，再用该环境的python -m pip安装；环境位置服从Conda配置，不写死/tmp或用户目录，不使用自建prefix。

## 现有代码与验证的适用范围

- PR49仍open，分支codex/automatic-sft-migration，最后提交c406fc3，包含vLLM基础b53f9cf及自动迁移412710e。
- 其migrate_sft.py默认XDG用户缓存下的自建prefix，并用--compat-root显式指定验收位置/tmp；并非生产写死/tmp，但也不是用户现在要的标准命名环境。
- 当前主main为22fd7a5，只有PR48要求；本轮分支codex/implicit-sft-migration-requirements只修订要求。
- 新实现需将控制逻辑收为普通入口调用的内部模块，保持主环境读取缓存与旧环境CPU worker分离；旧checkpoint来源不额外要求用户提供final或改命令。
- 缓存核对与既有data_plan来源/划分检查继续保留；load_from_disk重新hash的兼容处理仍是经核对导出身份的读取，不是新增文件防篡改签名。JSONL历史不额外复查。

## 本机环境事实

- 主Python：/home/yunli/miniconda3/envs/qwen-vllm/bin/python，仅本机示例名。主栈torch2.8.0+cu128/vllm0.10.2/transformers4.57.6/trl0.27.2/datasets4.4.2。
- Conda：/home/yunli/miniconda3/bin/conda。用户授权由程序自动创建CPU兼容环境，不要求另装系统Toolkit12.4。
- 旧兼容栈torch2.6.0+cpu/transformers5.20.2/datasets5.1.0，指纹相关传递依赖已固定在PR49的legacy/sft_data/。
- 原开发GPU为RTX4060Laptop8GB；真实CUDA检查需正式沙箱外权限。此次只做要求文档，不运行安装或GPU测试。

## Git约束

日常Git写操作走固定qwen-git-workflow.py入口，历史只追加；PR合并/关闭由用户操作。新要求合入前不实施功能。PR49基础代码和历史须保留，后续从最新main新分支承接。

旧详细记录保留于既有Git历史和PR49归档。未提交研究docs/research/math-equivalence-and-sft-loss.md不属于本轮。
