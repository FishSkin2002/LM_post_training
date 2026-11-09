# 当前进展

## 2026-11-09：实现脚本接口精简

- 用户确认 PR69 完成并明确要求改代码。受控入口从最新 main/099768a 创建 codex/simplify-script-interfaces，保留未跟踪研究文件，不处理独立 Git 工具源码维护。
- 已读取已合入接口要求、README 编写约定、模型样本及训练约定；本轮实现公开入口与数据环境选择，不扩展缓存缺失重建、绝对路径回退或完整目录搬移。
- 已更名 evaluate.py、统一公开 resume/source/checkpoint、删除公开 prepared 参数及内部 CLI 传递。训练 current 路径不调用旧环境，legacy 任务通过已有 CPU worker 处理三类数据，并以 HF 原生 Arrow 文件/筛选索引的小份引用交接，不 save_to_disk 复制整份数据。
- 已保留历史 prepared-sft 数据读取，并在新运行记录实际数据环境及原生缓存引用；评测复用记录中的 SFT 缓存，DPO/GRPO 自身训练缓存可供本阶段恢复。README 与参数参考已同步新接口及 env。
- 首轮回归因公共参数的 Path 导入误删导致 27 个同源错误，已恢复导入；新增分派检查中 GRPO 的 execv 未模拟导致测试进程被替换并在假路径前结束，已模拟交接，不涉及真实模型或 API。随后 119 项离线/微型回归通过（27.722s），继续核查新增缓存评测场景。
- 最终补充三类 checkpoint 的原生缓存读取、旧 worker 失败不回退和内存索引不伪装为持久引用；后训练产物显式拒绝原始模型专用 config。对应旧测试夹具已改为只在原始模型时传 config。九模块共 121 项回归通过（28.028s，/tmp/qwen-script-interface-tests-final.log）。
- 13 个 README 示例已按真实解析器解析，13 张用途参数表完整覆盖适用选项，三列、默认值和 5 个本地链接检查通过；5 组公开帮助、Python 编译及差异检查通过。旧环境子进程采用合成数据与模拟安装/调度验证，未真实安装环境、完整训练、运行 GPU/vLLM 或调用付费 API。

## 2026-11-09：解决 PR69 与 PR68 合入后的冲突

- 用户确认 PR68 已合入并要求更新 PR69。受控入口获取 main/dcb9f68，核对其内容与 PR68 源分支 f3d6204 完全相同；在 PR69 分支合入这一允许的本地分支，保留双方全部提交。
- 冲突涉及 docs/automatic-sft-migration.md 和三份规划。已同时保留 PR69 的 current/legacy 规则、PR68 的独立参数参考分工，以及双方规划/进展/发现；当前计划单独记录本次冲突解决。
- 未读取 PR 内容或评审，未修改运行代码或用户研究文件；正在验证合并结果后完成普通合并提交与正常推送。
- 四处冲突均已消除，双方规则和记录完整，17 个本地链接/锚点、单一当前计划与未修改运行代码检查通过。普通暂存遭遇 Git 索引只读；受控入口也因 MERGE_HEAD 拒绝冲突中的提交，需通过正式宿主审批执行本次暂存及普通合并提交，再回到受控入口推送。
- 直接 Git 暂存/提交申请被自动审批明确拒绝，要求使用安装的受控入口。已在 /tmp/qwen-merge-entry-fix-vysvc2r4/ 准备入口最小修复与隔离测试；16 项检查通过，仅允许 commit 完成已全部暂存解决的合并，其余历史和分支保护保留。拟按 Git 约定通过正式审批安装，工具修复不混入 PR69。
- 最小入口修复已正式获批安装，通过固定入口生成普通合并提交 db6bd3e 并正常推送更新 PR69。两父提交及原分支历史完整，本地 merge-tree 确认与 main/dcb9f68 合并无差异；不查询或代替用户判断 PR 状态。项目运行代码与使用文档未改，工具修复源码/测试暂存本机，独立于 PR69。

## 2026-11-09：脚本接口精简要求

- 用户确认原始 Qwen 才允许在评测中使用 source 对齐训练配置，后训练产物自动读取自身记录，并要求开始修改。按 Git 约定本轮先交付独立要求，不把此前已撤回的缺失恢复/搬移能力纳入范围。
- 沿用 writing-for-agents；受控入口从最新 main/1271647 创建 codex/script-interface-simplification-policy。一次 open 查询只见 PR68 和其他任务 PR33，不读取 PR 内容或评审；PR68 保持独立，用户研究文件保留。
- 已核对当前公开接口及数据准备入口：训练环境选择会改变既有“识别旧来源后自动准备”的训练触发条件；评测只改接口与保留自动配置/缓存定位，不扩展缺失场景处理。
- 新增 docs/script-interface.md 集中记录已定接口，七份相关约定同步新评测入口、汇总 resume、训练环境显式选择和参数删除边界。README 与参数参考继续描述实际入口，PR68 内容未混入。
- 8 份要求文档共 67 个本地链接/锚点、旧名称残留检查、最小范围要点与 git diff --check 均通过。运行代码、配置、使用文档及术语定义未改；没有新增测试文件或执行真实模型/环境安装。
- 要求提交 c0a2132 已通过受控入口正常推送并创建 [PR69](https://github.com/FishSkin2002/LM_post_training/pull/69)。本轮只完成已确认范围的要求交付；待用户合入后再从最新 main 实施代码，不扩大第五项的数据处理范围。

## 2026-11-09：同步使用文档编写约定

- 用户已确认 PR67 合入并要求继续。本轮沿用 writing-for-agents，只同步 README 与独立参数参考的展示要求，不实施脚本接口更名或数据处理改造。
- 受控入口从最新 main/1271647 创建 codex/readme-parameter-documentation-policy；不重复查询用户已确认的 PR 状态，不读取 PR 内容或评审，用户研究文件保留。
- 已核对相关约定，旧规则仍要求示例下列常用可选参数，并将兼容准备提示放在 README。现改为 README 必需参数示例、独立参考完整三列表格，并同步旧数据准备文档中的展示位置。
- 已明确条件必填参数须同时进入示例与参考表，默认值按实际取值或配置/运行条件表达，可填枚举合并说明用途。13 个本地链接/锚点及与现有 13 个用途的一致性检查通过，git diff --check 通过；差异仅含两份要求与三份规划。
- 要求提交 dcb7538 已正常推送并创建 [PR68](https://github.com/FishSkin2002/LM_post_training/pull/68)。本轮规则同步完成，等待用户审查合入；未改 README、参数参考正文或运行代码，未继续脚本接口改造。

## 2026-11-09：README 简明示例与参数参考

- 用户要求先整理当前使用文档：README 仅简短用途说明与填好必需参数的命令，链接独立的三列参数参考。参数参考覆盖必需和适用可选参数，合并说明可填项及用途。
- 已核对实际 train/evaluate_checkpoint/summarize_evaluation 与公共参数、生成配置；本轮保留当前脚本名和参数名，不实施接口讨论中的目标。
- 已通过受控入口从最新 main 创建 codex/readme-script-parameter-reference。此前未提交的讨论规划原文已保存在 /tmp/qwen-interface-discussion-u0vrbgxk/，本轮规划从已提交基线续写，不混入未决设计。用户研究文件保留。
- README 脚本说明已精简为 13 个用途的简短说明及必要命令，新增 docs/script-arguments.md 承载三列参数参考；补齐 additional-epochs，并将继续评分和重试待复核拆成两个准确示例。其余 README 章节逐字保持。
- 从当前脚本参数构造代码提取实际解析器，13 个示例全部可解析；13 个用途表覆盖全部适用 CLI 选项，必需项均在示例中。实际配置默认值、三列表格与 5 个本地链接/锚点核对通过；SFT/DPO/GRPO、独立评测与汇总的 5 组 --help 均正常退出。
- git diff --check 发现 README 末尾多余空行，已修正并复核。一次 open 查询仅见其他任务 PR33；未读取 PR 内容或评审，准备单独交付使用文档。
- 文档提交 ac474f9 已通过受控入口正常推送并创建 [PR67](https://github.com/FishSkin2002/LM_post_training/pull/67)。本轮 README 与参数参考完成，等待用户审查；未推进脚本接口实现或独立项目要求同步。

## 2026-11-09：训练结束自动评测与汇总

- 用户要求继续，提交关系确认 PR65/773663a 已合入 main/8f7d342，受控入口创建 `codex/training-final-summary`。一次 open 查询仅见其他任务 PR33，未读取 PR 内容或评审。
- 本轮沿用 planning-with-files-zh 与 writing-for-agents 维护规划，只接通训练结束编排；用户研究文件与历史报告保留。
- 已在 train.py 统一串联保存后的一二级与默认汇总。原生路径复用训练模型；vLLM 或训练期 GRPO 引擎需要释放时，execv 进入 train.py 内部续程，在新进程中完成完整后续链路。单独 evaluate_checkpoint.py 不启动汇总。
- 评测仅在返回的内存视图增加实际目录，未重复保存元数据或问答。失败消息给出 final/、本地评测位置与独立继续方法；汇总失败后可直接选择已打印评分运行继续，不重新训练。
- 八模块 112 项离线/微型检查一次通过（27.139s，`/tmp/qwen-training-final-summary-tests.log`）。新增核查覆盖实际保存后才请求模拟 API、取消密钥后用 --no-api 恢复、GRPO 训练 vLLM/原生评测交接、execv 失败无原进程回退；既有 SFT/DPO/GRPO 微型训练均生成阅读报告。
- README 已说明训练结束默认 API 汇总与失败恢复；裁判配置注释同步为运行时提供凭据。未启动真实大模型训练或付费 API，未修改历史 reports/outputs 与用户研究文件。
- Python 语法、git diff --check、README 三个本地链接与裁判 YAML 语义一致性检查通过，配置数值与此前版本相同；准备交付本轮自动编排功能。
- 功能提交 92225fe 已通过受控入口正常推送并创建 [PR66](https://github.com/FishSkin2002/LM_post_training/pull/66)。本轮训练结束自动链路完成，等待用户审查合入；评测产物精简的已确认实现步骤已全部交付，不跨到其他训练或研究任务。

## 2026-11-09：可选 API 汇总与阅读报告实现

- 用户要求继续，提交关系确认 PR64/adac06e 已合入 main/a9b1b9a；受控入口创建 `codex/evaluation-summary-reports`。一次 open 查询仅见其他任务 PR33，未读取 PR 内容或评审。
- 本轮实现独立汇总与阅读报告，训练结束自动串联留后续；继续使用 planning-with-files-zh 与 writing-for-agents 维护规划，保留用户研究文件及历史 outputs。
- 已将 rescore_evaluation.py 替换为 summarize_evaluation.py，直接读取共享快照与一二级机器记录。三级按评分标识追加同一 judgments.jsonl，不复制前级；各评分目录只保存自己的来源、裁判配置与冻结提示。
- 默认补做尚未处理的三级；--no-api 完全跳过配置读取、请求和密钥，支持部分结果。缺密钥时仅在需要请求前终端隐藏输入，密钥只传本次 APIJudge 实例，不改环境、不落盘，取消保留未处理状态。
- 成功汇总统一替换评测目录内一套阅读版：默认 100 题分卷、仅首卷总指标、逐题三阶段结果，以及判对/判错与待复核两类 API 清单和原文链接。渲染/发布失败保留机器判决并回滚旧阅读版；无新工作时不再请求 API。
- 32 项定向检查和完整八模块 107 项离线/微型回归均一次通过（29.258s，`/tmp/qwen-summary-tests.log`）；CLI 帮助及不导入 torch、Python 语法、差异检查与 README 三个本地链接通过。未调用真实付费 API，未下载真实权重。
- README 已同步实际入口、参数与产物；训练结束自动串联尚未接入，留下一独立步骤。本轮保留历史报告与用户研究文件，准备交付独立汇总功能。
- 功能提交 426329e 已通过受控入口正常推送，创建 [PR65](https://github.com/FishSkin2002/LM_post_training/pull/65)。本轮独立汇总与阅读报告完成，等待用户审查合入及后续“继续”，再推进训练结束编排。

## 2026-11-09：评测机器产物实现

- 用户要求继续，提交关系确认 PR63/a91ef7b 已进入远程 main/697a8b4；受控入口创建 `codex/evaluation-machine-artifacts`。本轮一次 open 查询仅见其他任务 PR33，未读取 PR 内容或评审。
- 继续使用 planning-with-files-zh 记录进度、writing-for-agents 维护规划。本轮只接通共享快照与一二级机器数据，汇总/API/Markdown 留后续；未跟踪研究文件保留。
- 已接通公共快照与两份 JSONL，新增共享发布与保存边界检查；首轮 29 项定向检查通过。一次规则测试补丁因旧上下文不同未应用，核对函数范围后已重新写入。
- 首轮完整 100 项回归发现 prepared 数据来源字段遗漏，以及将初始化耗时放入生成设置后影响跨模型配置相等校验；已补回来源字段，将初始化耗时移出生成配置，并优化默认运行查找只读取元数据，准备复测。
- 复测仅剩迁移测试仍把快照编号当旧数据集行号；已改为核对原始记录/轨迹、题干及两种参考的精确一致性，继续验证同一保留评测集。快照编号不再依赖预处理行号。
- 最终七模块 100 项离线/微型检查通过（27.303s，`/tmp/qwen-machine-artifacts-tests.log`），覆盖快照共享/并发发布/位置、默认与显式目录无原始数据集恢复、回答先落盘后补一级、二级中断、旧判决复用及训练后调用；旧 rescore 的模拟 API 检查继续通过。
- CLI 帮助、Python 语法、`git diff --check` 和 README 的 3 个本地链接检查通过。README 已改为公共快照加三份运行文件，旧 rescore 明确用于历史完整 JSON；旧文件或进度占用同名位置时拒绝静默从头生成。
- 本轮尚未实施 summarize_evaluation.py、真实 API 三级接入或 Markdown 阅读报告，原报告与用户研究文件未改动；准备交付当前保存/续跑步骤。
- 功能提交 b062e3f 已通过受控入口正常推送并创建 [PR64](https://github.com/FishSkin2002/LM_post_training/pull/64)。当前保存与续跑步骤完成，等待用户审查合入及后续“继续”指令，再从最新 main 开始下一实现步骤。

## 2026-11-09：精简评测产物要求

- 用户确认设计后要求继续；本轮只同步项目要求，使用 planning-with-files-zh、writing-for-agents 维护文档，不提前实施代码。
- 已通过受控入口获取最新远程 main 并创建 `codex/evaluation-report-storage-requirements`，保留未跟踪研究文件 `docs/research/math-equivalence-and-sft-loss.md`。
- 只读调查确认每套现存报告有 6447 条记录，原始问答 Markdown 约 121–176 MB；引擎与环境逐题重复，三级复制整套问答及前级判决。用户最终选择共享样本快照、拆分 JSONL、汇总后才生成分卷和 API 核查链接。
- 提交关系确认 PR62 的 21823ad 已进入远程 main/13f203a；本轮一次 open 列表仅见其他任务 PR33，未读取任何 PR 内容或评审。
- 新增 `docs/evaluation-artifacts.md` 作为产物要求入口；同步分级判定、项目入口、训练、DPO/GRPO、vLLM、旧数据准备六份约定，术语表增加评测样本快照与评测汇总的领域定义。README 留待功能实现时同步真实行为。
- `git diff --check`、8 份正式 Markdown 的 57 个本地文件/锚点检查通过；只读复核确认所有已定设计已覆盖，正式约定无旧独立三级入口或逐题 Markdown 追加要求残留。差异仅包含 Markdown；未运行模型、训练或付费 API，用户研究文件未修改。
- 要求提交 d61704f 已通过受控入口正常推送，创建 [PR63](https://github.com/FishSkin2002/LM_post_training/pull/63)；本轮只交付要求。待用户审查合入并说“继续”后，再从最新 main 分步骤实现，不提前修改运行脚本或已有产物。

## 2026-11-07：DeepSeek 最终答案裁判

- 已恢复规划并核对分级评测、训练、README 和 Git 约定。PR61 已合入 main/0f1380b；通过受控入口新建 codex/deepseek-answer-judge，一次性 open 查询只见其他任务 PR33。未跟踪研究文件保留。
- 本轮按用户明确要求调整现有第三级配置和提示，采用 DeepSeek Flash、显式关闭思考、要求 JSON；保留完整输入并禁止裁判独立解题。独立三级/汇总新入口仍留后续。
- 规划文件首次补丁因 progress.md 标题与预期不同而整批未应用；核对实际标题后重新应用。
- 已替换默认 YAML 并同步 CLI/README，完整中文提示保留最终答案定位、子问对应、数学等价及待复核，禁止重解题、补造参考和以中间推导替代最终答案；移除原有推导示例，增加参考不足的纯比较案例。提示版本更新为 math-final-answer-v2。
- tools.test_rescore_evaluation 与 tools.test_rule_evaluation 共 23 项在 qwen-vllm 环境通过（1.672s）；默认配置、CLI 帮助、三份 Python 语法、四份 Markdown 的本地链接及 git diff --check 均通过。本轮没有真实 API 调用或模型训练。
- 功能提交 21823ad 已通过受控入口正常推送，已创建 [PR62](https://github.com/FishSkin2002/LM_post_training/pull/62)。本轮完成现有裁判的切换及提示修订，完整提示在回复中展示；等待用户审查，不跨步骤继续。

## 2026-11-06：补齐旧兼容环境的 SOCKS 代理依赖

- 用户报告 HTTPX 使用 SOCKS 代理时缺少 socksio，要求补充旧兼容环境依赖。
- 受控 fetch、一次 open 列表查询及提交关系确认 PR60 已合入 main/56e55d3；从最新主线创建 codex/legacy-data-socks-proxy，继续使用既有规划与文档记录技能。
- 已确认 requirements.txt 没有 HTTPX SOCKS extra；兼容环境身份摘要包含 profile 与依赖清单字节，新增依赖后会使用新的受管环境身份，不直接改写旧环境或已发布数据缓存。
- 先用本地 HTTPX 0.28.1、缺少 socksio 的环境复现用户原始 ImportError；随后在 legacy/sft_data/requirements.txt 添加 httpx[socks]==0.28.1、socksio==1.0.0，profile.json 同步检查两包版本，原历史依赖记录保持原样。
- 验证下载首次因沙箱 DNS 限制失败，正式审批后仅将 socksio 安装到 /tmp/qwen-legacy-socks-validation，不修改主环境。真实 HTTPX 同步和异步客户端在 socks5、socks5h 代理变量下均能初始化，未发送网络请求。
- 现有 19 项迁移检查通过（1.236s），日志 /tmp/qwen-legacy-socks-migration-tests.log；git diff --check 通过。当前服务器已创建环境可单独执行 pip 安装 HTTPX SOCKS extra 后重跑原命令，未声称已远程安装。
- 依赖修复提交 9bbb015 已正常推送，并创建 [PR61](https://github.com/FishSkin2002/LM_post_training/pull/61)。本轮完成，未更改主训练环境或训练参数。

## 2026-11-06：兼容环境渠道修复与 CUDA 数据处理评估

- 用户要求删除自动创建兼容环境时的强制 Conda 渠道，并在评估有加速依据时才采用 CUDA 版 PyTorch。
- 使用 planning-with-files-zh 和 writing-for-agents 维护本轮计划及证据。受控 fetch 与一次 open 列表查询确认 PR59 不再 open；提交关系确认其历史已合入 main/ef9c99e，从最新主线创建 codex/legacy-data-conda-channels。
- 已定位强制渠道来自 legacy/sft_data/migration.py 的创建命令；后续项目依赖仍由兼容环境的 Python 调用 pip 安装。
- 初步核对 worker 只检查环境并调用现有数据准备和缓存导出，不加载模型权重；继续检查分词、数学解析、Arrow 处理及设备调用。
- 删除创建命令中的 --override-channels -c conda-forge；Conda 继续创建命名环境并安装 Python/pip，兼容依赖继续由 pip 安装。此次不改变已有兼容配置、旧数据身份或训练参数。
- 已完成代码链及官方设备语义核对：当前文本解析、分词、筛选、划分和缓存读写没有 CUDA 数值计算，单纯换 CUDA wheel 没有 GPU 加速路径。因此保留 CPU 版兼容 PyTorch；主环境的 CUDA 训练依赖保持其既有用途。
- 验证：tools.test_sft_migration 的 19 项离线检查通过（1.293s），日志 /tmp/qwen-legacy-conda-migration-tests.log。临时审计脚本还通过实际创建函数捕获命令，确认沿用渠道且继续调用 pip。
- 在主环境 torch 2.8.0+cu128 中运行相同数据准备、划分及导出/读取代码，20 条合成记录的 TorchDispatch 张量运算、CUDA 初始化及 CUDA 转移均为 0，输出记录及划分一致。日志 /tmp/qwen-legacy-channel-cuda-audit.log；这属于执行路径核查，未比较 A800 上 CPU/CUDA 版本耗时，不声称测出加速倍数。
- git diff --check 通过；本轮只改渠道创建命令和三份规划记录，不新增实现镜像测试、不安装真实环境、不启动训练。
- 实现提交 cd3b57a 已经受控入口正常提交和推送，并创建 [PR60](https://github.com/FishSkin2002/LM_post_training/pull/60)。本轮渠道修复与 CUDA 评估完成；后续优化或训练参数改动等待用户继续。

## 2026-11-06：实现按生成引擎区分评测批次

- 用户表示要求已处理并继续。受控 start 获取 main/a87d28a，提交关系确认 PR58 完整历史已合入；创建 codex/backend-evaluation-batches。
- 本轮按已确定目标实现：vLLM 固定默认 64，Transformers 默认 4 且可选自适应；两种后端可显式覆盖，vLLM 与 auto 冲突提前拒绝。独立与训练结束评测共用有效默认值。
- 同步 README 当前用法并补充必要边界验证；不修改训练采样或判分流程，不启动真实训练或付费 API。用户研究文件保持未跟踪。
- 根评测函数与 CLI 在解析实际后端后选择默认值；vLLM 与 auto 的冲突在来源读取、分词器和引擎加载前拒绝。既有固定生成、失败退出、逐题保存和续跑实现保留，训练调用无需重复默认值。
- 新增五项行为检查，覆盖 vLLM 64/64/1 固定分块、配置与显式后端覆盖、无效批大小、auto 参数提前拒绝、分块失败后更换大小续跑，以及原生函数和进程替换 CLI 的训练后默认值。
- 完整六模块 93 项离线/微型检查一次通过，日志 /tmp/qwen-backend-batch-tests.log（26.030s）。CLI 帮助、语法、git diff --check、四份变更 Markdown 的 4 个本地文件/章节链接及单一当前计划检查通过。README 两处参数表已同步。
- 本轮未测真实模型的吞吐或最优提交量；未运行完整真实训练、付费 API 或更改 GRPO 训练生成配置。
- 实现提交 69f9571 已正常推送并创建 [PR59](https://github.com/FishSkin2002/LM_post_training/pull/59)。本轮功能完成，等待用户审查；独立三级、汇总和自动大模型判分继续留后续。
## 2026-11-06：按生成引擎区分评测批次要求

- 用户确认自动增减对 vLLM 内部调度帮助有限，选择默认每次固定提交 64 题，可用 --batch-size 调整；Transformers 保留原默认 4 和可选自适应。
- 继续使用 grill-with-docs、planning-with-files-zh 和 writing-for-agents；两个只读调查核对调用边界及相关文档。vLLM 搭配自适应开关明确报参数不兼容，固定分块生成失败退出并保留已保存进度。
- 受控 fetch 与一次 open 列表查询确认 PR57 不再 open，提交关系确认其完整历史已合入 main/1305f0f；从该基点创建独立要求分支 codex/backend-evaluation-batch-policy。
- 修改 training-workflow 和 vllm-generation 的目标要求及验收；本轮保留运行代码和 README 当前用法，待要求合入后另做功能实现。未跟踪研究文件和历史归档保持。
- 五份变更文档的 17 个本地文件/章节链接、单一当前计划和 git diff --check 检查通过；代码与 README 无差异。本轮只验证文档，不运行模型训练、生成或付费 API。
- 要求提交 1fe5409 已正常推送并创建 [PR58](https://github.com/FishSkin2002/LM_post_training/pull/58)。本轮要求整理完成，用户合入后再做功能步骤；PR57 的已合入分支不继续追加提交。

## 2026-11-06：扩大 PR57 的代码整理范围

- 用户要求评测和判分两个根目录脚本整体保留实现，当前代码整理统一更新 PR57；同时完成此前已确定的 DPO/GRPO 独立实现。
- 本轮受控 fetch 确认 main 仍为 7bb1232，一次 open 查询确认 PR57 可接收提交，继续原功能分支。
- 评测完整业务函数会被训练结束流程复用；迁回根目录后保留 __main__ 保护并同步调用。共用数据读取与缓存身份保持原模块，避免改变已有数据指纹和断点来源。
- 两个根目录评测/判分脚本已整体迁回，utils 中原模块已删除；DPO/GRPO 分别使用独立训练模块，原 post_sft.py 已移除，调用方和测试同步更新。
- 用户进一步要求 common.py 及共用训练后调用放入 train.py；共用训练函数与原生评测、进程替换调度已集中到根入口，common.py 已删除。用户明确本轮只整理已有调用，不新增自动大模型判分。
- 第一轮完整微型/离线检查 88 项通过，覆盖完整脚本迁移、独立方法及共用函数迁回；随后归拢结束评测调用，继续做针对性复测。整理时捕获并修正一处 DPO 缩进错误，未进入提交。
- 结束评测归拢后的一处旧测试仍 mock 已移出的函数名称，已改为新的调用点，针对性复测通过；完整 88 项再次通过，日志 /tmp/qwen-pr57-verified-tests.log。
- 全部根入口帮助及缺参行为保持一致；语法和旧模块引用检查通过，62 个本地文件/章节链接有效；README、配置与数据准备实现保持原样。代码格式调整通过 AST 等价检查，git diff --check 正常。
- 整理提交 e057cf2 已正常推送到既有 PR57，没有新建代码 PR。GitHub 连接器更新元数据返回 403；本次经正式宿主审批使用 gh pr edit，仅更新 PR57 标题和说明，未修改状态或合并设置，该次例外不扩大后续 gh 使用范围。
- 本轮代码整理完成。DPO/GRPO 已独立实现，共用训练与现有结束评测调度位于 train.py；独立三级/汇总的新功能及自动大模型判分仍按用户确认留后续。

## 2026-11-06：训练入口直接包含 CLI

- 用户明确保留 train.py，要求把 cli.py 的解析与方法选择移入根目录入口；这轮不拆成三个根目录训练脚本。
- 受控 fetch 确认 PR56 已合入 main/7bb1232，从该基点创建 codex/train-entrypoint-cli。本轮只推进 CLI 迁移，DPO/GRPO 独立训练实现留下一步。
- 已核对三个现有测试模块导入旧 cli，生产入口仅 train.py 直接转发；迁移将更新这些导入并删除旧 cli。已保存当前四组帮助输出和缺少 DPO 来源时的行为作对照。
- 已将 CLI 解析和方法调度原样放入 train.py，保留 main(argv=None) 及 __main__ 保护，删除 utils/training/cli.py；三个测试模块改为直接导入 train。
- 验证通过：SFT/DPO/GRPO 训练、恢复、加训、评测及旧数据准备的现有 51 项微型/离线检查；四组帮助输出和缺少 DPO 来源时的退出行为逐字一致，CLI 主体 AST 一致，无遗留 Python 导入，git diff --check 正常。日志 /tmp/qwen-train-cli-tests.log。
- 模型、数据、参数语义和训练实现未改；DPO/GRPO 独立实现拆分仍为下一步，未启动真实训练或付费 API。
- 功能已提交 11318c8 并正常推送，创建 [PR57](https://github.com/FishSkin2002/LM_post_training/pull/57)。本轮训练入口调整完成，等待用户继续后推进 DPO/GRPO 独立实现。

## 2026-11-06：解决 PR56 的规划记录冲突

- 用户明确要求为 PR56 追加提交以便合入；本轮一次 open 查询确认 PR56 仍 open，PR55 已合入 main/2b9a636。
- 合并 main 与 PR56 在 task_plan、progress、findings 的修改：当前计划单独保留，双方已完成计划与工作记录均保留，当前状态中的 PR55 更新为已合入。
- 正式约定保持 PR56 版本，README 保留 main 版本；研究文件不纳入提交。本次验证检查记录完整性、链接、差异与主线合并关系。

## 2026-11-06：润色回答正确性判定板块名称

- 用户希望标题突出“判断模型回答是否正确”的用途；板块名称改为“回答正确性判定”，保留三级判断流程。
- 本轮一次 open 查询确认 PR55、PR56 均 open；展示与约定分别更新原分支，README 编写约定先同步名称和用途描述。

## 2026-11-06：同步 README 相关文档约定

- 用户要求检查相关约定，避免未来修改重新违背已确认的 README 要求。复用 grill-with-docs，委派两个只读调查；未发现需要重新决定的领域含义。
- 主要冲突：readme-writing 仍沿用旧结构、全部可选参数与 TensorBoard 分组；project-layout 允许 DPO/GRPO 共处模块；旧环境约定仍有固定示例名和“唯一入口”。相关训练、评分和生成文档需要统一“大模型判分”用词。
- 本轮一次 open 查询确认 PR55 仍 open；从最新 main/7d45fa1 新建独立要求分支 codex/sync-readme-documentation-requirements。受控 start 首次因 .git 只读失败，正式宿主审批后成功。
- 使用 planning-with-files-zh 和 writing-for-agents 维护本轮资料；展示规则集中写入已有 README 约定，算法与环境事实在各自原文档同步，AGENTS 的现有入口继续使用。
- 参考本轮已完成 README 的临时副本 /tmp/qwen-approved-readme-reference.md；不将 PR55 的展示改动混入要求 PR。未跟踪研究文件和历史记录保持原样。
- 已重写 README 编写约定，统一目录深度、根部脚本范围、方法独立用法、简短展示、环境占位符及纯文字标识；算法组织目标和旧数据准备文档已同步，术语表新增“大模型判分”的纯领域定义。
- 一次训练约定补丁因旧行上下文不符失败，重新定位原文后按明确措辞逐项更新；未修改算法规则或业务代码。当前继续核对判分用词、链接及各文档职责。
- 两个只读复核确认正式约定已覆盖 README 决定，算法与实验语义一致；根据复核补齐“大模型判决/判分明细”的用词并清理一处多余空格。网络请求、密钥和服务配置仍按实际含义说明。
- 验证通过：16 份活动文档及 README 参考中的 61 个本地文件/章节链接有效；旧口径检索无遗漏，git diff --check 正常。README 参考与 PR55 分支内容逐字相等；本轮仅 12 份 Markdown 变更（9 份约定/术语和 3 份规划），内部兼容环境来源和服务配置表保持原样。
- 无需修改 AGENTS 入口、项目目标、研究/历史报告或规划归档；未改业务代码、运行配置或原始产物，不运行训练或付费 API。
- 文档同步提交 b74e7fe 已正常推送，并创建独立要求 [PR56](https://github.com/FishSkin2002/LM_post_training/pull/56)。README 展示仍在 PR55；本轮文档核对完成，DPO/GRPO 代码拆分及其余训练编排后续步骤等待用户继续。

## 2026-11-06：润色回答正确性判定标题

- 用户希望板块突出判断模型回答是否正确的方法；标题改为“回答正确性判定”，开头点明用途，原有三级判断内容保持原样。
- 本轮一次 open 查询确认 PR55、PR56 仍 open；README 编写约定已在 PR56 以 3ddf039 同步名称，本分支更新 PR55 展示。
- 已核对两分支的板块名称一致，README 仅修改标题和开头用途句，三级判断内容及其余板块不变；git diff --check 通过。

## 2026-11-06：移除实验报告入口的表情符号

- 用户明确 README 当前不允许使用 emoji；报告入口改为加粗“实验报告：查看目录”，保留普通引用块，移除 emoji、箭头及会显示图标的提示块标记。
- 本轮受控 fetch 与一次 open 查询后继续更新 PR55，仅修改入口展示与本轮记录。
- README 全文表情符号扫描、实验报告链接与 git diff --check 通过，其余板块保持原样。

## 2026-11-06：修正 Conda 占位符与渠道参数

- 用户澄清不指定环境名应写为 `<环境名>` 占位符；已替换创建、激活命令中的空引号，并同步替换说明。
- 删除 README 安装示例中的 --override-channels -c conda-forge，使用读者现有的渠道配置；本次仅修正文档，不执行环境创建或安装。
- 本轮受控 fetch 和一次 open 查询确认可继续更新 PR55。
- 已检查两处占位符及替换后的命令参数，强制渠道参数已移除；其余 README 板块保持原样，git diff --check 通过。

## 2026-11-06：补充三级判分简介

- 按用户要求，在 README 的聊天格式之后新增同级标题“三级判分逻辑”，简述一级参考终值核验、二级 R1 正式作答单框核验、三级大模型判分。
- 简介保留未确认正确再进入下一级、截断直接终止、无法明确判断时待复核的规则；第三级不写具体模型名称。
- 已对照分级判定约定、术语表及现有规则代码；本轮受控 fetch 和一次 open 查询后，继续 PR55 分支更新。
- 检查通过：新板块与聊天格式均为二级标题、包含三级简介且无具体判分模型名称；命令示例保持原样，链接及 git diff --check 正常。

## 2026-11-06：收起断点目录的文件说明

- 按用户要求，README 的 checkpoint-<step>/ 仅保留一行断点用途注释；final/ 继续展示内部文件，删除树后已不适用的展开说明。
- 本轮受控 fetch 及一次 open 查询后，沿用 PR55 分支更新；本次只调整目录展示与记录。
- 检查确认断点目录下无展开条目，final 文件树与全部命令示例保持原样，git diff --check 通过。

## 2026-11-06：环境名示例留空

- 用户要求 README 不推荐固定环境名；创建与激活命令均改为空引号，说明执行前填入相同的自选环境名。
- 本轮一次 open 查询确认 PR55 仍 open，继续原分支更新；仅修改示例与本轮记录，不创建或修改环境。
- 已检查 README 无固定环境名，两条 Conda 命令均保留正确的空引号参数，git diff --check 通过。

## 2026-11-06：修正并润色大模型判分表述

- 用户指出相关操作是在调用在线大模型 API 判分，并要求措辞自然。README 统一使用“大模型判分”，在脚本介绍中说明“通过 API 调用在线大模型，对已保存的 OpenR1 评测回答进行判分”。
- 同步修正目录、产物注释、生成条件、用途标题、密钥与配置说明；不再用手动操作描述判分方式。命令、文件名和代码保持原样。
- 本轮受控 fetch 确认 main 仍为 7d45fa1，一次 open 列表确认 PR55 仍 open，继续原分支。README 措辞检索、全部命令示例一致性、链接及 git diff --check 通过；本次更新沿用 PR55。

## 2026-11-06：README 整理范围已确认

- 按用户调用的 grill-with-docs 读取 grilling/domain-modeling，并委派两个只读核查，确认根目录当前仅 train/evaluate/rescore 三个业务入口；DPO/GRPO 运行独立，但实现共处 post_sft。
- 用户确认：每个根部脚本一句介绍，保留用途命令与必要参数；删除原理、内部实现及大段环境说明，不转存新文档；本轮只整理文档，下个代码步骤再拆 DPO/GRPO。
- 目录总览不展开文件夹，全部文件/文件夹有注释；另列 configs 文件夹层级和 training 的完整文件示例，聊天格式独立成节，实验报告使用醒目入口。
- 受控 fetch 和本轮一次 open PR 查询确认 PR54 已合入 main/7d45fa1；受控 start 首次因沙箱 .git 只读失败，正式宿主审批后创建 codex/readme-layout-and-script-guide。
- 使用 planning-with-files-zh 和 writing-for-agents 维护本轮记录；研究文件未改，不运行真实训练或付费 API。
- README 已重组：总览文件夹不展开、所有条目有注释；configs 单列文件夹，training 示例展开 checkpoint/final/日志/评测/条件性数据缓存；聊天格式单独展示，标题下突出实验报告目录及现有实验一。
- 主要脚本仅保留根部三个入口，各一句介绍；SFT/DPO/GRPO 的用途命令与参数就近列出，DPO/GRPO 的新训练和恢复分别展示。原理、实现、内部工具和测试说明直接删除，未另建文档。脚本部分由 13568 减为 6090 字符（约减少 55%）。
- 验证 train 的 sft/dpo/grpo、evaluate、rescore 五组实际 --help；两处报告链接、配置路径、目录结构与注释、原生聊天 system 与代码的一致性均通过。输出示例已对照保存逻辑，文件名中的 OpenR1 大小写已修正。
- 核查时误读不存在的 legacy/sft_data/environment.py，改用 rg --files 后定位 migration.py 并确认锁文件保留；未影响变更。仅文档改动，不运行训练回归或付费 API。
- README 与本轮记录已提交 6297e28，正常推送并创建 [PR55](https://github.com/FishSkin2002/LM_post_training/pull/55)。本轮整理完成，下一步 DPO/GRPO 代码拆分等待用户继续；既有独立三级、汇总与训练后编排仍未实施。

## 2026-11-06：开始一二级评测整合

- 用户确认 PR53 已合入；远程 main/33367b2 已包含该 PR，新建功能分支 `codex/two-stage-checkpoint-evaluation`。
- 本轮把旧重评中的本地规则提取为共用模块，evaluate 先完成全量生成/一级，再按独立进度完成二级；不调用 API 或汇总。
- 保留原一级 JSONL/Markdown 与顶层一级字段；二级结果另存并附入 JSON 的 rule_scoring，供后续三级直接使用。旧重评对含已保存规则结果的输入直接沿用，避免重复二级。
- 已接通公共本地规则与二级进度，API 与训练依赖不进入本地规则模块；一级完成且生成器关闭后才执行二级。二级结果使用 stage2.progress.jsonl/stage2.md，各题本地判决和全局完成状态附入 JSON。
- 新增六项实际评测集成检查，覆盖补判与截断、阶段顺序、生成中断、二级中断和完成续跑、Markdown 失败后复用已 flush 判决、旧重评复用规则及 GSM8K；连同原重评 23 项通过。README 已按当前功能更新，累计各级汇总留待后续独立脚本。
- README 补丁的 boxed 反斜杠上下文未匹配，改为按段落定位后应用；代码检查无报错。
- 主环境完整微型/离线回归 87 项通过（新增六项 + 原 81 项），覆盖统一训练、生成、分级评分、旧数据自动准备与指纹恢复。随后补充：部分二级报告禁止旧 API 入口提前评分；定向复测及边界检查继续执行。
- 补充的 API 前置条件与跨批次阶段屏障检查通过：23 项定向复测及六项边界复测均通过；完成续跑不执行旧评分或 Markdown 渲染，首级和二级的历史字节不变。本地规则模块导入不带 torch、API 模块或重评入口。
- 语法、差异及八个本地文档链接检查通过。日志 /tmp/qwen-two-stage-tests.log、/tmp/qwen-two-stage-targeted.log、/tmp/qwen-two-stage-boundaries.log。未运行真实大模型或付费 API；本轮准备交付一二级整合。
- 第二步功能提交 3949c92 已正常推送，创建 [PR54](https://github.com/FishSkin2002/LM_post_training/pull/54)。等待用户继续后实现独立三级的新建/续跑；独立汇总及训练后编排仍未实施。未提交研究文件未改、未纳入。

## 2026-11-06：开始目录与统一训练入口

- 用户授权开始改动；远程 main/3e3b108 已合入 PR52，从最新 main 创建 `codex/organize-training-entrypoints`。
- 本轮推进一个结构整理步骤：业务实现迁入 utils/，train.py 统一 SFT/DPO/GRPO，移除随机抽样入口，同步 legacy worker、测试、README 和路径定位。
- 评分拆分与自动编排留作后续步骤；原 rescore 命令在拆分前暂保留薄入口。验证沿用随机微型模型与离线测试，不运行真实训练或付费 API。
- 已将业务模块移入 utils/、接通 train.py 三阶段及共用模型/保存逻辑，移除随机抽样；配置、默认输出、评测进程替换和 legacy 导入路径已同步，README 按当前实现更新。
- 首轮现有微型检查 80 项通过。额外对照发现：数据内容与来源完全相同，但迁移模块名改变 datasets 对 callable 的序列化指纹（ac6f258cf7c0f02e → 7df3d492776f694a）；已为数据转换保留历史序列化模块身份，接下来核对真实迁移前后指纹及内容。
- 静态检查中已修正批量导入替换误改函数名、SFT 续训输出路径比较及一处补丁上下文不匹配；未影响既有产物。
- 主环境微型检查再次通过 80 项，SFT、DPO、GRPO 的实际微型训练均通过统一 CLI 调用；训练恢复、策略更新、生成续跑、重评与旧数据自动准备检查保持通过。
- 与 main/3e3b108 的原代码在隔离解释器中比较：主环境与已安装旧 CPU 环境各自的 OpenR1/GSM8K/DPO/GRPO 共八种路径，全部文本、词元、来源信息、数据指纹及 SFT 划分完全相等。主环境 OpenR1 指纹 ac6f258cf7c0f02e，旧环境 4e255131893486f9。比较脚本首次合成 GRPO 记录缺少 ability，补齐后通过。
- 证据：/tmp/qwen-layout-tests.log、/tmp/qwen-layout-identity.log、/tmp/qwen-layout-data-identity.json；额外增加已保存指纹恢复的回归检查。CLI 帮助及互斥参数在仓库外使用标准库 Python 验证通过，无需加载训练依赖。
- 新增指纹回归在主环境和旧 CPU 环境各通过 1 项（主环境累计 81 项）；评测、重评的根部薄入口在仓库外 `--help` 通过，数据检查模块 CLI、全部源码语法及 8 个本地 Markdown 链接检查通过。
- 暂存后差异检查发现三个新文件多余 EOF 空行，已清理；本轮只归拢代码、统一训练和删除随机抽样，评分拆分尚未实施。README 明确现阶段可用命令，原重评暂保留根部薄入口。
- 第一阶段功能提交 ab900f0 已正常推送，创建 [PR53](https://github.com/FishSkin2002/LM_post_training/pull/53)。第一步已完成；等待用户继续后，实施一二级整合、独立三级及汇总，不自动跨步骤执行。研究文件未纳入提交。

## 2026-11-06：独立三级与汇总参数已确认

- 用户确认新建三级用 `--evaluation`、续跑用 `--resume-from`；汇总选定一次评分，三级未完成也可手动汇总，未处理/待复核分开且总分母不变。
- 将项目整理与评分约定恢复为正式目标，明确四个根部入口、一二级手动不触发后续、训练结束自动编排、三级新建/续跑互斥及保存来源/配置、汇总 `--scoring` 选择本次运行。
- 补充首个 API 请求前保存运行信息、取消后续跑/部分汇总、累计各级正确率与固定分母样例等验收。既有训练算法、模型与研究文件未改。
- 本轮仍使用 PR52 的需求分支；上轮 open 查询结果复用，本轮受控 fetch 确认 main 仍为 1cf99db。接下来检查并推送最新要求及同步 PR 说明。
- 文档验证通过：git diff --check；11 份 Markdown 的 38 个本地文件/章节链接全部有效；当前约定无待确认标记或旧调用边界冲突。变更范围仅需求、术语与规划，未运行训练或付费 API。
- 交付：最新要求提交 34fcd8a 已推送到 PR52，包含此前 9e51159 的调用边界调整。GitHub 连接器首次字段校验未通过，修正为 repository_full_name/pr_number 后 API 返回 403（Resource not accessible by integration），未改动远端元数据。
- 用户通过明确回复允许本次用 gh pr edit 仅更新 PR52 标题/说明；已按准备好的完整文本执行成功，PR 标题为 Define separate evaluation, API scoring, and summary requirements。未修改 PR 状态、评审或合并设置；一次性例外不扩大未来授权。
- 本轮要求交付完成，等待用户合入并要求继续后，从最新 main 新建功能分支。业务代码与未提交研究文件保持原样。

## 2026-11-06：改为独立三级评分与汇总

- 用户明确改变调用边界：一二级放在一起，三级单独脚本，汇总再单独脚本；训练结束自动依次一二级、三级、汇总。手动一二级不调用三级，三级和汇总可手动运行。
- 三级选择已有评分运行才继续；不传选择参数就创建新评分，重名加数字。原有 utils/、统一训练、删除随机抽样、逐级追加和隐藏输入密钥的决定保留。
- 按 grilling 的事实调查要求委派只读复核：保存文本足够三级/汇总独立使用；一级中断JSONL没有固定全集分母；训练末 vLLM 的 os.execv 不会返回，需要内部编排层承接后续步骤，不能直接在 execv 后追加代码。
- 远程 main 仍为 1cf99db，本轮一次性 open 查询确认 PR52 仍 open。继续同一需求分支，本地先同步已定边界与待决问题，暂不推送未确认的设计。
- 待用户决定新评分来源与旧运行选择参数、单次评分汇总选择、三级中断时是否可手动部分汇总。业务代码与未提交研究文件未改。

## 2026-11-06：项目整理需求确认与文档同步

- 用户确认全量一级 → 全量二级 → 统一三级，不保留 `--input`，各级和一级一样自动找进度、向末尾追加；缺密钥按推荐使用终端隐藏输入。结合上一轮答复，设计问题全部确定。
- 将 `docs/project-layout.md` 从澄清草稿改为正式目标约定，明确 `utils/`、统一训练、删除旧入口、保留配置/训练产物与旧数据兼容、各级进度及功能验收。
- 同步三级判定、训练、后续训练、vLLM、旧数据自动准备及模型样本文档，消除独立重评、抽样入口与统一流程的冲突；CONTEXT 只新增三种判定的领域定义。
- 终端输入仅在恢复后尚有三级工作且缺密钥时发生；取消、空输入或无终端保留进度，不把未请求题写成待复核。历史 JSONL/Markdown 只追加，完成重跑不生成、不重判、不请求密钥或 API。
- 一次文档补丁的无关末行上下文不匹配，去掉该上下文后成功应用。业务代码、已有产物和用户研究文件未改。
- 验证：git diff --check 通过；11 份变更 Markdown 的 39 个本地文件/锚点链接全部有效；无指向已改名旧章节的引用，旧独立重评/抽样要求已同步。仅文档变更，不运行训练或 API；接下来提交并独立交付要求 PR。
- 交付：要求提交 2fb7724 正常推送，已创建 [PR #52](https://github.com/FishSkin2002/LM_post_training/pull/52)，目标 main。等待用户合入后再开始功能；不通过本地 main 或代用户合并。用户未提交研究文件未纳入。

## 2026-11-06：项目整理第一轮答复已记录

- 用户将业务目录名改为 `utils/`，确认训练结束自动评测执行完整三级；缺密钥应在三级开始前交互输入。
- 已解释旧流程：原评测先全量生成及一级，独立重评再逐题二级及必要的三级。原评测自动找同名进度追加；`--input` 只是独立重评的输入参数，不是自动续跑的条件。
- 继续只读核实：当前没有 GUI 输入实现；APIJudge 每题读环境变量；训练结束的 os.execv 没有主动断开原终端。待用户确认终端隐藏输入形式及新阶段顺序。
- 更新 `docs/project-layout.md` 和规划；未修改业务代码、未索取真实密钥、未运行 API。文档差异检查通过。

## 2026-11-06：项目整理第一轮澄清

- 已读取 grill-with-docs 及其组合的 grilling/domain-modeling；使用 planning-with-files-zh 和 writing-for-agents 维护本轮资料。按 grilling 要求，委派两个只读调查分别核对评测与训练。
- 远程 main/1cf99db 已包含 PR51；受控入口创建独立需求分支 `codex/project-layout-requirements`。open PR 一次性查询仅有 #33。
- 核实评测对 sample_checkpoint 只有共享生成函数的间接依赖；实际训练只有两个可执行入口；记录 index 非连续数值，当前一二三级没有分开的执行进度。
- 新增 `docs/project-layout.md`，区分用户目标、代码事实和待决问题。待用户回答第一轮问题再继续；尚未修改业务代码或既有训练/评分要求。
- 本轮仅检查代码与文档，未运行训练、评测生成或付费 API。未提交研究文件保持原样。

## 2026-11-06：现有入口自动准备与标准Conda环境已实现

用户确认PR50已合入。受控入口从最新main/ebbe9ca创建codex/implicit-sft-migration，普通合并7215c05承接PR49/c406fc3完整历史；规划文件冲突保留最新要求，受控入口不能完成冲突后的merge，已获一次性正式宿主审批完成普通merge提交。

- 原train_sft、train_post_sft、evaluate_checkpoint、sample_checkpoint入口通过统一helper自动准备旧SFT数据，无需额外命令或开关。新SFT、当前数据环境兼容的来源与已有缓存跳过旧环境；DPO/GRPO及原始模型对照追溯原SFT。
- 根migrate_sft.py已删除，控制模块在legacy/sft_data/migration.py。Conda标准命名环境创建，再用Conda查询的实际Python/pip安装CPU依赖；envs_dirs不被覆盖。用户缓存仅存状态/锁，不安装环境。并发锁、ready标记、安装中断重试和未受管环境保护保留。
- 旧SFT checkpoint直接准备，不要求final存在。数据完整来源/指纹/token/划分核对后才原子发布缓存；原权重/run和历史JSONL/Markdown保持既有行为。
- README已改为主安装与原命令使用说明，去掉独立迁移与--compat-root步骤。

验证：
- tools.test_sft_migration：20项通过，覆盖所有原入口首次准备、新SFT/当前来源跳过、缓存复用、旧checkpoint、来源不符、子进程失败、命名环境安装中断、双层并发与未受管环境拒改。
- tools.test_generation + tools.test_instruct_sft + tools.test_post_sft：45项微型检查通过，未运行真实4B训练。
- 真实首次验收由evaluate_checkpoint.main正常命令触发：自动conda create -n建立此前不存在的qwen-sft-data-instruct-datasets51-cpu-v1-1e87c5bc6256，实际位于/home/yunli/miniconda3/envs/下；pip安装与check通过，torch2.6.0+cpu、CUDA runtime=None。
- 随后CPU worker实际处理合成20题旧数据并核对，普通评测入口继续加载随机微型底座与旧LoRA并生成1条回答。仅原数据读取和底座规模使用测试替代，Conda/pip/处理/校验/生成真实执行。
- 指纹仍为4e255131893486f9，验证索引[19,5]；全部题目、监督/参考、来源/组、token IDs/labels逐条相等；SFT产物字节未变。完成续跑在禁止Conda/旧子进程/生成器/评分器的检查下通过，JSONL与Markdown字节未变。
- 日志/tmp/qwen-named-migration-smoke.log；结果/tmp/qwen-named-migration-fixture/migration-result.json；/tmp仅存合成测试数据和日志，兼容环境不在/tmp。

功能已提交435f3d6并推送，PR #51：https://github.com/FishSkin2002/LM_post_training/pull/51；等待用户审查合入。未更改主环境、系统Toolkit或真实SFT产物；未调用付费API。此前独立入口方案仅为PR49旧实现，当前已被本轮流程替换。

## 历史

PR48确立主vLLM依赖和CPU兼容处理；PR49完成基础实现与GPU微型验证但入口和环境位置需修正；PR50已合入这次明确的新要求。旧详细记录保存在已有Git历史与docs/planning-archive/2026-11-06-pre-migration/。
