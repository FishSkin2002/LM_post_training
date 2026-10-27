# 当前计划

## 当前小步

加载已下载的 0.6B checkpoint，从 GSM8K 抽取少量题目生成回答并展示参考解答，便于定性查看训练结果。

## 阶段

| 阶段 | 状态 | 完成条件 |
| --- | --- | --- |
| 核实 checkpoint 与主线 | complete | 找到 outputs/development/gsm8k-sft/checkpoint-1404，确认 0.6B 完整权重、分词器及 chat_template.jinja；PR #6 已合并，从最新 origin/main（83c45ba）创建 codex/checkpoint-sample-qa。 |
| 实现抽样问答入口 | complete | 新增 sample_checkpoint.py，显式加载本地 checkpoint；默认 test、5 题、seed 42，沿用训练模板；输出回答、参考解答和长度状态，支持保存 Markdown。 |
| 验证并实际生成 | complete | 帮助、编译和模板检查通过；已离线加载用户 checkpoint 生成 5 题并保存原始报告。用户要求自行看答案，已移除人工点评，不追加运行。 |
| 交付 PR | complete | 功能提交已推送，main 目标 PR #7 已创建并附加到当前聊天；本轮不合并。 |

## 交付

本轮交付一个小型命令行抽样入口和实际问答报告。只将题目放入模型输入，参考解答仅用于生成后的展示；模型和分词器均来自用户指定的本地目录。采样报告保存在 outputs/ 下，不提交模型或生成文件。

用户最新要求答案自行查看；最终仅交付脚本、运行命令和原始问答，保持报告无人工判分或点评。

[PR #7](https://github.com/FishSkin2002/qwen/pull/7) 已创建。本轮完成，后续运行和答案查看由用户自行安排。

## 边界

本轮不合并 PR，不改变 main，不训练或修改 checkpoint，不接入计算器，不做完整评测或扩大到多模型对比。允许离线读取已有数据缓存和加载真实 checkpoint 做用户要求的少量推理。

项目要求、Git 历史保护与其他分支前缀边界仍按 docs/git-workflow.md 执行。

## 遇到的问题

- 暂无。
