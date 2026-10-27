# 当前计划

## 当前小步

核实 GSM8K 代表性公开实现的数据格式与训练损失范围，区分普通 SFT、显式思考 SFT 和结果奖励训练；暂不改训练代码或选择掩码方案。

## 阶段

| 阶段 | 状态 | 完成条件 |
| --- | --- | --- |
| 核对来源与当前实现 | complete | 核对官方模型卡、训练说明和本机损失实现；从最新 origin/main（9fca1be）创建 codex/qwen3-thinking-practice-policy。 |
| 记录已确定约定 | complete | 已增加 AGENTS 条件指针及集中约定文档，区分官方推理参数、待讨论掩码和当前代码状态；CONTEXT 补齐领域定义。 |
| 校对文档 | complete | 已核对官方模型卡、训练指南和损失语义；仅修改 Markdown，文档入口有效，git diff --check 通过，未运行模型。 |
| 交付要求文档 PR | complete | 规则文档提交已推送，main 目标 PR #9 已创建并附加到当前聊天；本轮不合并。 |
| 追踪公开实现 | complete | 已核对 GSM8K 原论文、MetaMath、Qwen3/GSM8K 示例、ms-swift 与 verl 的数据、损失或奖励路径，并标明框架默认值推断的限制。 |
| 整理研究结论 | complete | findings.md 已整理完整解答 SFT、显式思考格式、结果奖励及全响应策略更新的证据和一手链接；未改变损失方案。 |

## 交付

本轮属于项目要求文档维护类，独立交付思考模式约定。规则进入主线后，运行配置和样本构造再按后续已明确的小步骤适配。

[PR #9](https://github.com/FishSkin2002/qwen/pull/9) 尚未合并，本轮在其分支补充同一议题的研究证据。是否采用非空思考段损失掩码仍留作后续实验选择。

## 边界

本轮不合并 PR，不改变 main，不修改 Python/YAML 或 checkpoint，不运行模型训练/推理，不实现用户仍在询问的损失掩码选项。

项目要求、Git 历史保护与其他分支前缀边界仍按 docs/git-workflow.md 执行。

## 遇到的问题

- 暂无。
