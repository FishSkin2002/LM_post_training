# 当前计划

## 当前小步

抽样问答运行后默认保存到 Markdown 文件，终端仅提示保存路径；问题、回答和参考解答不再打印到终端。

## 阶段

| 阶段 | 状态 | 完成条件 |
| --- | --- | --- |
| 核实 checkpoint 与主线 | complete | 找到 outputs/development/gsm8k-sft/checkpoint-1404，确认 0.6B 完整权重、分词器及 chat_template.jinja；PR #6 已合并，从最新 origin/main（83c45ba）创建 codex/checkpoint-sample-qa。 |
| 实现抽样问答入口 | complete | 新增 sample_checkpoint.py，显式加载本地 checkpoint；默认 test、5 题、seed 42，沿用训练模板；输出回答、参考解答和长度状态，支持保存 Markdown。 |
| 验证并实际生成 | complete | 帮助、编译和模板检查通过；已离线加载用户 checkpoint 生成 5 题并保存原始报告。用户要求自行看答案，已移除人工点评，不追加运行。 |
| 交付 PR | complete | 功能提交已推送，main 目标 PR #7 已创建并附加到当前聊天；本轮不合并。 |
| 默认保存及终端输出调整 | complete | 默认保存到项目 outputs/samples/<checkpoint>-<split>-seed<seed>.md；正文和加载进度不再打印，完成时只提示路径。 |
| 验证并更新 PR | complete | 替代加载器验证默认/自定义文件写入、文件更新和终端仅输出路径；提交已推送并更新 PR #7 说明，没有新增真实推理。 |

## 交付

本轮交付一个小型命令行抽样入口和实际问答报告。只将题目放入模型输入，参考解答仅用于生成后的展示；模型和分词器均来自用户指定的本地目录。采样报告保存在 outputs/ 下，不提交模型或生成文件。

用户最新要求答案自行查看；最终仅交付脚本、运行命令和原始问答，保持报告无人工判分或点评。

[PR #7](https://github.com/FishSkin2002/qwen/pull/7) 尚未合并，本轮沿用当前分支完善同一抽样入口。后续真实运行和答案查看由用户自行安排。

默认保存和终端输出调整已交付到同一 PR，本轮完成；用户使用原命令即可生成文件。

## 边界

本轮不合并 PR，不改变 main，不运行真实模型推理，不修改 checkpoint，不接入计算器或判分。验证使用替代加载器和临时输出目录。

项目要求、Git 历史保护与其他分支前缀边界仍按 docs/git-workflow.md 执行。

## 遇到的问题

- 暂无。
