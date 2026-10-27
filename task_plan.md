# 当前计划

## 当前小步

修复训练机下载分词器时的 Brotli 接口不兼容，补齐下载依赖并提供训练机修复命令。

## 阶段

| 阶段 | 状态 | 完成条件 |
| --- | --- | --- |
| 核实报错与依赖 | complete | 已核对下载层传入 output_buffer_limit，Brotli 旧 process 接口不接受该参数；PR #4 已合并，从最新 origin/main（cf34633）建立 codex/fix-brotli-download-compat。 |
| 补齐依赖和说明 | complete | 固定 Brotli 1.2.1；README 包含同一解释器下的安装和解压 API 检查命令。 |
| 验证下载解压链路 | complete | 仅向 /tmp 安装 Brotli 1.2.1；解压关键字接口、HTTPX2 解码器及响应流对小型 JSON 和 5.12 MB 内容往返校验通过，未修改开发机现有环境。 |
| 交付 PR | complete | 功能提交已推送，main 目标 PR #5 已创建并附加到当前聊天；本轮不合并。 |

## 交付

本轮只修复下载依赖兼容性，不更改模型、训练参数或 TensorBoard 设置。训练机没有可用的远程连接，实际安装及联网分词器检查交由用户执行。

[PR #5](https://github.com/FishSkin2002/qwen/pull/5) 已创建。本轮修复完成，下一步由用户在训练机安装 Brotli 1.2.1、执行接口检查并重试原命令。

## 边界

本轮不合并 PR，不改变 main，不启动模型训练，不删除模型缓存。开发完成允许创建 PR，不等于获得 PR 合并授权。

项目要求、Git 历史保护与其他分支前缀边界仍按 docs/git-workflow.md 执行。

## 遇到的问题

- 沙箱内 pip 无法找到候选发行包，按网络限制使用正式审批重试；验证包仅安装到 /tmp。
- 本机 pip 缓存目录不可写，不能使用 pip cache；改用不写缓存的临时目标目录。
