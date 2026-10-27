# 进展记录

## 2026-10-27：查看训练后 checkpoint 的回答

- 用户在 1.7B 再次 OOM 后改用 0.6B 进行新训练，现已下载 checkpoint，要求抽取几道数据集问题查看问答表现。
- 已找到 `outputs/development/gsm8k-sft/checkpoint-1404`；配置 hidden_size=1024、28 层，与 Qwen3-0.6B 一致，完整模型权重约 2.38 GB。目录包含 tokenizer.json、tokenizer_config.json 和 chat_template.jinja，可直接本地加载。
- 已确认 PR #6 合并，通过受控入口从最新 origin/main（83c45ba）创建 `codex/checkpoint-sample-qa`。本轮只实现并运行少量问答，不启动训练或完整评测。
- 已新增 sample_checkpoint.py：显式选择本地目录，默认 test/5 题/seed 42，使用与训练一致的非思考模板和逐步取最高概率词元的生成方式；支持 train、样本数量、种子、长度、设备和可选 Markdown 报告。报告同时显示参考解答和是否触及生成长度上限。
- 本地分词器已验证能读取独立的 chat_template.jinja，并生成与训练一致的空 think 前缀；帮助、编译与差异检查通过。真实模型推理已通过正式宿主授权启动，离线读取已有缓存并使用 CUDA，不训练模型。
- 已用 CUDA/BF16 加载实际 checkpoint，对 test 索引 `[1309, 228, 51, 563, 501]` 完成问答，均生成结束标记且未达到长度上限。原始问答报告保存为 `outputs/samples/checkpoint-1404-test-seed42.md`，模型及报告均保留在忽略的 outputs/ 下。
- 用户最新要求自行看答案；已删除临时加入的人工点评，仅保留问题、模型回答和数据集参考解答。本轮不追加运行，最终提供脚本、运行命令和原始报告。
- 聚焦检查通过：输入只包含问题、只解码新增词元、推理时关闭梯度、结束标记恰在长度上限时不误报截断，达到上限但无结束标记时提示截断。
- 功能提交 `6fa05b4` 已正常推送，已创建并附加 [PR #7](https://github.com/FishSkin2002/qwen/pull/7)。未合并 PR；后续运行和答案查看由用户自行安排。

## 2026-10-27：按 AutoDL 实测降低训练显存

- 用户提供完整重试日志：Brotli 问题已排除，1.7B 权重下载和 7473 条训练样本处理成功，最长 550 tokens；第 10 次更新后在反向传播中 OOM。
- CUDA 报告总容量 31.60 GiB、进程使用 31.20 GiB，其中 PyTorch 已分配 27.33 GiB、保留未分配 3.56 GiB；申请 464 MiB 时仅余 408.12 MiB。退出后 nvidia-smi 显示显存释放、没有运行进程。
- 已确认 PR #5 合并，通过受控入口从最新 origin/main（4402cba）创建 `codex/reduce-sft-training-memory`。本轮缩小微批量并补充日志所建议的分配器选项，保持有效批量和训练算法。
- 训练机默认改为每批 1 条、累积 16 次；启动日志增加实际批量、累积、BF16 与梯度检查点开关。开发机配置、模型、FP32 权重及融合 AdamW 保持原有设置。
- README 在标准启动命令中添加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，并提供无需更新代码即可覆盖批量的重试命令；使用新输出目录保留旧日志。第 10 步失败尚未到第一轮保存检查点，重试需从头开始。
- 本机 PyTorch 2.6 接受该分配器选项。CPU 随机微型 Qwen3 配合 8～550 tokens 的合成变长样本完成 20 次更新，参数和 TensorBoard 第 1、10、20 步指标均有限；有效批量、开发机配置保留、编译及差异检查通过。该验证不测量真实 1.7B 的 GPU 峰值显存，仍需用户在训练机重试。
- 功能提交 `8063a00` 已正常推送，已创建并附加 [PR #6](https://github.com/FishSkin2002/qwen/pull/6)。未合并 PR；已提供可在现有训练代码上直接使用的覆盖命令，实际训练机重试结果待用户确认。

## 2026-10-27：修复训练机下载解压依赖

- 用户在 AutoDL 的 Python 3.12/base 环境运行训练，下载 Qwen3-1.7B 的 tokenizer_config.json 时由 HTTPX2 Brotli 解压触发 `TypeError: process() takes no keyword arguments`；尚未进入训练。
- 已核对本机 HTTPX2 2.16.0：BrotliDecoder 调用 `process(..., output_buffer_limit=...)`，其 Brotli 可选依赖声明要求 CPython 使用 Brotli >=1.2.0。官方 Brotli 1.2.0 发布说明确认新增该参数；拟固定至已发布的 1.2.1。
- 已确认 PR #4 合并，通过受控入口从最新 origin/main（cf34633）新建 `codex/fix-brotli-download-compat`。候选解压包通过正式网络审批安装到 /tmp 验证，不修改开发机现有包。
- requirements.txt 已固定 `Brotli==1.2.1`，README 补充原环境安装和解压 API 检查命令；训练代码、模型及超参数保持原有设置。
- 在临时目录验证 Brotli 1.2.1 与 HTTPX2 2.16.0：带 `output_buffer_limit` 的接口调用、BrotliDecoder 分块解码和 HTTP 响应 `iter_bytes()` 均通过。小型分词器配置 JSON 和 5.12 MB 内容都能完整往返，差异检查通过。没有连接训练机、下载真实模型或启动训练。
- 已向用户提供训练机可立即执行的单包更新命令。实际训练机修复结果尚待其执行后确认。
- 功能提交 `016bb88` 已正常推送，已创建并附加 [PR #5](https://github.com/FishSkin2002/qwen/pull/5)。未合并 PR；训练机安装与重试由用户执行，本轮不启动训练。

## 2026-10-27：记录 TensorBoard 训练指标

- 用户要求用 TensorBoard 记录主要训练指标；本轮只接入训练日志和使用说明，不启动真实模型训练。
- 已确认 PR #3 合入主线；通过受控入口从最新 origin/main（90348df）新建 `codex/tensorboard-sft-metrics`。
- 本机已有 TensorBoard 2.21.0。固定版本 Transformers 的内置回调可记录损失、学习率、梯度范数、轮次和训练结束汇总，默认在输出目录的 `runs/` 下创建独立日志目录；无须新增自定义回调。
- 两套 YAML 已改为 `report_to: tensorboard`，requirements.txt 固定 TensorBoard 2.21.0，保持现有模型、训练参数及日志频率。README 新增指标表、默认日志位置、启动命令和远端 SSH 端口转发示例。
- 两套机器配置各完成 11 次 CPU 随机微型 Qwen3 更新；从实际事件文件读回损失、学习率、梯度范数、轮次、总体损失、耗时和两项吞吐量，全部为有限值。曲线在第 1、10 步记录，结束汇总在第 11 步记录；训练参数和模型配置文本也已写入。
- 验证仅覆盖日志接入，使用合成样本和随机微型模型；未加载 Qwen3 预训练权重或 GSM8K。自定义输出目录下的两套日志相互隔离；TensorBoard 模块启动、训练帮助和差异检查通过。
- 功能提交 `c356a63` 已正常推送，已创建并附加 [PR #4](https://github.com/FishSkin2002/qwen/pull/4)。未合并 PR，未修改 main；下次训练可直接记录指标，后续步骤等待用户继续。

## 2026-10-27：按机器选择模型

- 用户要求训练机改用 Qwen3-1.7B，并适配 32 GB 显存；长期默认训练机 1.7B、开发机 0.6B，使用 writing-for-agents 记录。
- 已核对现有共享模型配置及两台机器参数，通过受控入口从最新 origin/main 新建 `codex/qwen3-per-machine-models`。
- 本轮范围为配置、入口适配、文档和必要验证；不启动真实模型训练。
- 已将模型名称从 common.yaml 移入各机器 YAML；训练入口使用所选配置的模型名称加载分词器和权重，样本预览默认开发机并支持 `--config`。共享读取函数会校验模型字段，训练入口将其从 TrainingArguments 参数中剥离。
- 训练机调整为每批 2 条、梯度累积 8 次，保持单卡有效批量 16；启用非重入梯度检查点和融合 AdamW，保留 FP32 权重、自动 BF16 混合精度、原有学习率与轮数。新输出目录为 `outputs/training/qwen3-1.7b-gsm8k-sft`，与旧模型分开。
- 按 writing-for-agents 将用户的长期默认放入 findings.md 的环境入口，并由 README 指向；运行参数以对应 YAML 为准。README 已同步训练及恢复命令，没有改写导入的项目目标原文。
- 两套 TrainingArguments、加载入口模型路由、预览默认/切换、BF16 自动解析、命令行覆盖、无效 YAML/模型字段及非空目录保护通过；随机微型 Qwen3 在 CPU 上用梯度检查点完成反向传播，融合 AdamW 和 Adafactor 均完成有限值更新。帮助、编译与差异检查通过。未加载预训练权重或数据集；真实 1.7B 显存仍待训练机实测。
- GitHub CLI 的沙箱联网失败后已按正式宿主审批重试，确认本分支没有现存 PR，继续通过已有 CLI 登录态交付。
- 功能提交 `c638b9a` 已正常推送，已创建并附加 [PR #3](https://github.com/FishSkin2002/qwen/pull/3)。未合并 PR，未修改 main；本轮完成，下一步可在训练机短跑验证峰值显存，等待用户继续。

## 2026-10-26：通过 PR 合入主线

- 用户要求从最新 main 更新合并规则：开发完成后代理可自主发起 PR，本地禁止合并到 main，其他允许操作的非 main 分支可直接合并。
- 已通过受控入口获取远程，确认 main 与 origin/main 都为 d28258f，从该提交新建 `codex/pr-merge-workflow`，本轮仅更新 Git 协作规则和受控入口。
- GitHub 插件访问仓库仍返回 404；正式网络授权后 GitHub CLI 已核对账号为 FishSkin2002，沿用已登录 CLI 交付 PR。
- AGENTS 和 Git 约定已区分创建 PR、授权合并 PR、非 main 本地合并。受控入口拒绝本地向 main 合并或直接推送 main；非 main 合并不自动推送。
- 新分支改为先获取远程 main，再从 origin/main 创建；本地 main 可保持原位，无须通过本地合并同步主线。
- 12 项隔离仓库测试通过，覆盖 main 拒绝路径、codex/ 与其他无前缀目标分支的正常合并、main 合入功能分支、前缀限制和已有历史保护。
- 新版受控入口已通过正式审批安装，逐字节核对与工作区源文件一致。未修改本地或远程 main，也未启动模型训练。

## 2026-10-26：准备 SFT Pull Request

- 用户授权使用其 GitHub 账号和本地 codex Git 身份创建 PR，目标 main，仅准备合并，不执行 merge。
- 本地工作区干净，功能分支上的 4 个提交作者均为 `codex <codex@local.invalid>`；已通过受控入口推送，远端功能分支为 a91fba4，main 仍为 6b232f6。
- GitHub 插件连接账号确认为 FishSkin2002，但访问本仓库返回 404，无法用该连接创建 PR。用户随后安装 GitHub CLI 2.103.2 并完成登录；已通过 CLI API 确认账号为 FishSkin2002。
- 已创建 [PR #1](https://github.com/FishSkin2002/qwen/pull/1)，标题为 `Add GSM8K SFT training with per-machine YAML configs`，来源 `codex/gsm8k-sft-data`，目标 main；PR 已附加至当前聊天。未执行合并或真实训练。

## 2026-10-26：按机器分开保存配置

- 用户要求开发机与训练机配置各放一个子目录，统一收进一个配置目录。
- 用户指定使用 YAML；共享模型、数据和模板设置放在 `configs/common.yaml`，`configs/development/sft.yaml` 和 `configs/training/sft.yaml` 分别存放两台机器的训练参数。默认保持 32 GB 训练机配置。
- 本轮只完成配置整理、入口适配和必要验证，不启动训练。
- 已将共享模型/数据/模板设置与两台机器的训练参数从代码迁入 YAML；默认训练配置与此前 32 GB 设置一致，原有命令行参数仍可覆盖。
- 两套 YAML 到 TrainingArguments 的加载、命令行覆盖、异目录默认加载、学习率科学计数法和无效 YAML 报错均通过；帮助、编译与差异检查通过，未加载模型或启动训练。

## 2026-10-26：编写 SFT 训练脚本

- 用户要求训练代码可以开始编写，写完由用户自行训练。本轮不启动真实模型训练。
- 沿用现有 `codex/gsm8k-sft-data` 分支和 `build_sft_example()`，只使用原始 GSM8K train，不转换 `answer`。
- 已核对本机 Transformers 的 Trainer、TrainingArguments 和补齐器接口，使用现有依赖完成全参数 SFT。
- 用户明确实际训练使用租用的 32 GB 显存 GPU，要求以后均按 32 GB 准备。已将该事实记录在 findings.md；训练默认改为 AdamW、每批 4 条、梯度累积 4 次，移除原先为本机 8 GB 考虑的 Adafactor 和梯度检查点。
- 已完成 `train_sft.py`：固定原始训练集，动态补齐，按轮保存检查点并支持恢复；结束后保存完整模型、分词器和训练指标。输出目录已加入忽略规则。
- 帮助、编译、默认参数与差异检查通过。经正式宿主授权离线验证全部 7473 条原始解答（最长 550 tokens）、变长批次掩码，以及随机微型 Qwen3 的 CPU 损失、反向传播、优化器初始化和保存加载，全部通过。
- README 已提供在租用 GPU 上的运行和恢复命令。未加载真实模型权重或启动正式训练，交由用户执行。

## 2026-10-26：构造 SFT 样本

- 用户最新决定直接使用 GSM8K 原始输出，覆盖此前改成 `The answer is: 数值` 的决定。题目和解答均原样使用，已移除答案转换和额外提示词。
- 在已有 `codex/gsm8k-sft-data` 分支实现这一个小步：构造并展示训练输入及 labels，检查监督范围；不启动训练或接入推理计算器。
- 新增 `prepare_sft_data.py`，直接将原始题目和解答放入 Qwen3 模板，返回输入、注意力掩码和 labels；删除了本轮未提交的格式转换与额外选项。
- 首次运行遇到模板返回 BatchEncoding 的兼容问题，已显式读取 `input_ids`。正式授权后离线运行真实训练集第一条样本通过，原始 `<<...>>` 和 `#### 72` 均保留，题目及前缀标签为 `-100`。
- 使用说明、依赖和忽略规则已补齐；本步不加载模型权重、不启动训练。完成后等待用户继续接训练循环。
- 暂存时沙箱拒绝创建 `.git/index.lock`，未产生提交；改用正式宿主授权完成常规暂存与受控提交。

## 2026-10-26：开始确定 SFT 样本格式

- 用户同意以 GSM8K 标准解答上的普通 SFT 开始实践。本轮范围限定为训练样本构造，不启动完整训练。
- 使用受控入口，从本地 main 新建 `codex/gsm8k-sft-data`。已有终端对照分支保留，未合并或推送。
- 核对官方计算器说明与论文附录 C，确认训练文本注释与推理程序执行计算是两个环节；`####` 是最终数值的提取标记。
- 已通过正式宿主授权完成训练集离线检查：7473 条数据的最终答案标记和计算器注释结构均符合本次检查条件，详细计数见 findings.md。
- 用户要求先核实常见项目的 SFT 格式、所需推理程序与本机能力，暂未选择计算器和最终答案格式。尚未编写样本处理代码或启动训练。
- 补充核查公开 Qwen3/GSM8K 数据处理脚本、MetaMath 论文、Qwen2.5-Math 与原始 GSM8K 的推理实现。已确认独立解题和调用工具均有公开实践，答案格式随项目而异；本项目已有普通生成代码，尚无计算器控制流程。依赖版本与来源记录在 findings.md。
- 本轮只更新术语与事实/进度文档，`git diff --check` 通过。后续数据处理等待约定确定。

## 2026-10-25：建立保留历史的协作基础

- 用户以 daab1ee2d2efb33934df04bd7218257919c91de2 为保留起点，一次性授权本轮整理此前产生的分支历史。
- 原历史及命令授权已在仓库外备份；本轮历史整理不是后续改写历史的长期授权。
- 项目要求明确：每轮只做一个小步骤，主要由 Codex 编写代码，按需答疑，完成后等待用户继续。
- 项目要求与功能分支分别交付；用户说 merge 后才合并并推送。GSM8K 保持未合并。
- 所有分支改为历史只追加，回退使用新的 revert 提交。日常写操作通过固定参数的受控入口执行，原始宽泛允许与一次性历史重建授权需清理。
- 受控入口通过 6 项临时仓库测试：普通建分支、提交、推送、合并、revert 和安全 fetch 可用，危险参数、替换远程历史及后退的跟踪引用被拒绝。
- 拟安装规则通过 14 项匹配检查，常规入口保留 allow，危险原始写命令为 forbidden，裸 push/commit 的宽泛允许已从候选配置移除。
