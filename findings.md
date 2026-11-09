# 当前维护发现

## 评测 EOS 告警（2026-11-09）

- 固定版本 vLLM 0.10.2 的 vllm.inputs.preprocess.get_eos_token_id 在内部 tokenizer 为空时逐次输出 `Using None for EOS token id because tokenizer is not initialized`；请求处理器逐题调用它，导致每次提交一批题目时刷屏。
- 公共 VLLMGenerator 使用项目分词器先构造 prompt_token_ids，显式传入 ignore_eos=True、stop_token_ids=[tokenizer.eos_token_id]，并通过项目分词器解码；跳过的只有 vLLM 内部分词器初始化，不包括模型和推理引擎。
- 来源：[vLLM 预处理器](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/inputs/preprocess.py#L55-L63)、[逐请求调用](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/engine/processor.py#L346)。最终方案只匹配该 logger、WARNING 级别和完整文本，由独立模块供公共评测和 GRPO 训练复用，不提高模块级别或修改第三方源码。
- Python 官方 logging.addFilter 接受可调用函数，重复添加同一函数不会叠加过滤器；在 vLLM 日志初始化后注册。模块只依赖标准库，导入时不加载 vLLM 或 CUDA，不依赖 Conda 或额外配置文件。

## 精简盘点（2026-11-09）

- 当前维护入口为 train.py、evaluate.py、summarize_evaluation.py；SFT、DPO、GRPO、默认 current 与显式 legacy 均仍在维护。不能把旧环境兼容当作废弃功能。
- tools/test_rescore_evaluation.py 已不测试旧 rescore 入口，实际覆盖 utils.answer_judge 和当前汇总入口；旧名字需要纠正，API 行为覆盖应保留。
- tools/smoke_sft_migration.py 仍读取旧 .progress.jsonl 与 .md，和当前 EvaluationStore 产物不符；这是一份未随流程维护的手工验收脚本。
- docs/planning-archive/2026-11-06-pre-migration 的三份长记录及根目录反复叠加的交付日志属于历史，不是当前操作依据；有效规则与环境事实需先核对再保留。
- 回归中仍有有效的历史格式拒绝覆盖（防止覆盖旧报告）和旧分词器/指纹读取覆盖；这些检查对应当前代码分支，继续保留。删去的两项缓存检查与 MigrationTests 的导出/自动定位覆盖重叠，独有断言已移入保留测试。
- 唯一仍引用的研究资料是聊天模板依据；一次性 OpenR1 抽检和旧规划归档无活跃文档引用。项目正式约定仍承担当前功能的维护边界，不因已完成实现而删除。

## 当前环境与兼容事实

- 主环境检查使用 /home/yunli/miniconda3/bin/python；数据处理版本和旧环境配置从实际依赖与 legacy/sft_data/profile.json 读取。
- 新运行引用 HF 原生 Arrow 数据与磁盘筛选索引，保留列投影、指纹和实际数据环境；历史 prepared-sft 读取仍在使用。旧序列化模块名参与数据指纹，不能因清理旧名称而去掉兼容别名。
- 缓存缺失重建、失效绝对路径回退及完整运行跨机器恢复不在已实现范围。
- Git 写操作使用已安装受控入口。该入口已获批修复为允许提交完全解决并暂存的合并；仓库 tools/git_workflow.py 尚未同步此修复。候选及 16 项验证在 /tmp/qwen-merge-entry-fix-vysvc2r4/，后续维护勿覆盖安装版本。
