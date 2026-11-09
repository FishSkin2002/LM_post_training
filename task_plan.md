# 当前计划：精简维护材料

用户要求检查并精简不再维护的回归检查与文档。PR70 已合入 main/90814cf；当前分支 codex/prune-obsolete-maintenance。只清理已被替代、重复或纯历史内容，保持现有训练、评测、旧环境兼容能力。

| 工作 | 状态 | 完成条件 |
| --- | --- | --- |
| 盘点 | complete | 已核对所有文档和 10 个回归模块、两个 smoke；正式约定和现有兼容流程仍有效。 |
| 精简 | complete | 删除三份旧归档、一次性抽检记录和失效 smoke；压缩当前规划，合并两项重复缓存检查，纠正旧测试名称。 |
| 验证 | complete | 自动发现的 131 项回归通过；编译、104 个本地路径链接、旧引用搜索和差异检查通过。 |
| 交付 | complete | 1e03688 已正常推送并创建 [PR71](https://github.com/FishSkin2002/LM_post_training/pull/71)，等待用户审查合入及下一指令。 |

用户未跟踪的 docs/research/math-equivalence-and-sft-loss.md 和 reports/exp1 历史实验产物保留。已安装 Git 入口的修复与仓库源码尚未同步，留独立维护步骤，不混入本轮。
