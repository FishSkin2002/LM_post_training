# 当前进展

## 2026-11-09：隐藏评测 EOS 告警

- 初步添加局部过滤后，用户要求暂停以讨论方案；期间未提交或推送。项目已经构造输入 token、显式指定原生 EOS 并自行解码，vLLM 的默认 EOS 查询不承担当前评测的停止条件。
- 用户随后确认精确过滤，并要求独立模块供多个入口复用。已再次获取 main/ecb4628，继续 codex/silence-evaluation-eos-warning；此前局部过滤已移除。
- 新增 utils/vllm_logging.py，使用标准 logging.addFilter，完整匹配 logger、WARNING 和 EOS 消息；公共 VLLMGenerator 在 vLLM 导入后配置，MathGRPOTrainer 在启用 vLLM 的初始化结束后配置。重复初始化复用同一过滤函数，其他日志级别、handlers 和传播设置不变。
- 精确日志与两个入口的临时检查已通过。首轮现有回归未设置离线开关，依赖探测 Hugging Face 配置时发生 DNS 失败并重复重试，已中断；改用 HF_HUB_OFFLINE=1、HF_DATASETS_OFFLINE=1 重跑，禁止下载模型或数据。
- 离线重跑 tools.test_generation 和 tools.test_post_sft 共 24 项通过（16.040s）。已核对 64 条重复告警及格式化文本匹配、同模块其他 INFO/WARNING、同文本 ERROR、其他模块同文本日志、幂等注册、模块导入不加载 vLLM，以及评测/GRPO 启用和原生 GRPO 不启用的接入行为；编译和差异检查通过。未运行真实 GPU/vLLM 生成。
- 实现提交 cde5c5e 已正常推送，创建 [PR72](https://github.com/FishSkin2002/LM_post_training/pull/72)。本轮一次 open PR 查询仅见其他任务 PR33，未读取 PR 内容或评审；用户原有未跟踪研究文件保留。当前小步骤完成，等待用户审查及下一指令。

## 2026-11-09：精简维护材料

- 用户要求检查并精简不再维护的回归检查与文档。已获取 main/90814cf，确认 PR70 已合入，使用受控入口新建 codex/prune-obsolete-maintenance。
- 一次 open PR 查询仅见其他任务 PR33；不读取 PR 内容或评审。用户未跟踪研究文件与历史实验报告保留。
- 已盘点文档和检查入口：三份根目录规划反复叠加旧进度，旧规划归档另有 2080 行；旧迁移 smoke 仍检查已删除的评测输出格式。其余测试需要按当前代码职责判断，不能按文件名含 legacy/rescore 就删除。
- 正在核对重复测试与文档引用，先将当前规划缩为本轮所需状态；历史过程可从 Git 历史查阅。
- 已删除三份旧规划归档、一次性 OpenR1 前 20 条抽检记录与失效的迁移 smoke。仍被模型约定引用的聊天格式资料保留，并清除“生产尚未迁移”等过时状态。
- 两项生成模块中的 prepared 缓存检查已合并到迁移检查：保留内容、指纹、统计、划分、配置不匹配及被改动身份的断言，自动评测缓存定位由原有集成测试覆盖。API 裁判检查更名为 test_answer_judge 并同步引用；GSM8K 检查改为核对当前判决记录，移除对废弃文件名的空检查。
- 本轮不修改业务实现、正式项目约定、依赖或运行产物；测试和研究/规划资料属于这次维护精简，不新增功能要求。
- 自动发现 131 项检查全部通过（119 项业务检查、12 项 Git 检查，38.088s，/tmp/qwen-maintenance-pruning-tests.log）；编译、104 个本地路径链接、旧引用搜索及差异检查通过。维护中的根目录/docs Markdown 从 24 份、4564 行降到 20 份、约 1550 行，完整历史仍可从 Git 查阅。
- 精简提交 1e03688 已通过受控入口正常推送，并创建 [PR71](https://github.com/FishSkin2002/LM_post_training/pull/71)。只剩用户原有研究文件未跟踪；本轮完成，等待审查合入与下一指令。

## 已完成的最近一步

PR70 已合入：公开 resume/source/checkpoint 与训练 env 已统一，121 项离线/微型回归通过。旧环境安装和 GPU/vLLM 未在该轮实测。
