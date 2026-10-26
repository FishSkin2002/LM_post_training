# 环境发现

## 默认训练环境

- 用户于 2026-10-26 明确：实际训练使用租用的 **32 GB 显存 GPU**，后续默认训练配置均按这一环境准备。GPU 型号和远端软件版本尚未提供，不作假定。
- 本机环境只用于当前代码和数据验证，不能用本机显存容量替代上述训练环境。
- 随后用户要求开发机与训练机配置分目录保存，并指定使用 YAML。统一入口为 `configs/`：`common.yaml` 存放共享模型/数据/模板配置，`development/sft.yaml` 保存本机参数，`training/sft.yaml` 保存默认 32 GB 训练机参数。
- `train_sft.py` 默认读取训练机配置；只有显式传入开发机配置时才采用本机的每批 1 条、梯度累积 16 次、Adafactor 与梯度检查点。两套输出目录分开；这是用户要求的两套配置，不改变默认训练环境。
- 两个入口的共享设置均通过 `yaml.safe_load` 读取 `common.yaml`，不依赖执行时的当前目录。训练参数使用 `--config` 选择，原有命令行参数可临时覆盖文件中的对应项；`bf16: auto` 在构造 TrainingArguments 时按 GPU 能力解析。
- 本机已有 PyYAML 6.0.3，现已作为直接依赖列入 requirements.txt。YAML 中的学习率写为 `0.00002`；读取时也支持 `2e-5` 这种可能被 PyYAML 识别为字符串的科学计数法。

## 本机验证环境

- 仓库为 `/home/yunli/code/qwen/codex`，远程 origin 为 `git@github.com:FishSkin2002/qwen.git`。
- `/home/yunli/miniconda3/bin/python` 为 Python 3.12.15，已有 torch 2.6.0+cu124、transformers 5.20.2、datasets 5.1.0、accelerate 1.15.0。
- 本机 GPU 为约 8 GB 显存的 NVIDIA RTX 4060，仅作为本地环境记录；实际训练默认使用上述 32 GB 租用 GPU。Codex 沙箱内访问本机 GPU 受限，需要正式宿主授权。
- Hugging Face 缓存已有 Qwen3-0.6B 和 GSM8K，可离线加载，已在此前真实运行中核实。

## GSM8K 的解答格式与计算器

- 官方说明：`answer` 包含自然语言解题过程，最后一行的最终数值前有 `####`。这一标记用于提取答案，无须转成自然语言才能训练。
- `<<表达式=结果>>` 是计算器注释。原论文训练时将它们视为普通词元；推理时由外部程序检测表达式并填入结果，随后继续生成。仅在 SFT 数据中保留这些注释不会自动接通计算器。
- 官方给出的去注释方法是删除 `<<` 到 `>>` 的整个子串。例如 `2/2=<<2/2=1>>1` 清理为 `2/2=1`，外面的结果仍保留。
- 来源：[官方仓库数据说明](https://github.com/openai/grade-school-math#dataset-details)、[论文附录 C，第 17 页](https://arxiv.org/pdf/2110.14168#page=17)。
- 2026-10-26 离线检查本机 main/train：7473 条解答中，7378 条包含计算器注释，共 23716 处；所有解答均恰有一个 `####`；移除完整注释后没有残留的 `<<` 或 `>>`。
- 2026-10-26 用户最新决定题目与解答原样用于 SFT，保留计算器注释和 `#### 数值`，覆盖此前改成英文答案短语的决定。
- 沙箱内 `load_dataset` 首次失败：读取缓存也会在缓存目录创建锁文件，报只读文件系统。通过正式宿主授权后，同一离线统计命令运行成功；无须下载或 GPU。

## 代表性 SFT 实践与本机能力

- [HuggingFaceTB 的 Qwen3/GSM8K SFT 数据脚本](https://huggingface.co/HuggingFaceTB/qwen3-1.7b-gsm8k-sft/blob/main/scripts/prepare_combined_data.py) 的 `format_reasoning_gsm8k` 删除计算器注释和原始答案行，`create_gsm8k_example` 构造完整助手解答，并以英文 `The answer is ...` 给出最终答案。这是一个具体实例，不能据此宣称所有项目都采用同一格式。
- [MetaMath 论文附录 B](https://arxiv.org/pdf/2309.12284#page=18) 明确在训练解答末尾追加 `The answer is: {gold answer}`，评测时提取该标记之后的答案。
- [Qwen2.5-Math 官方实现](https://github.com/QwenLM/Qwen2.5-Math#quick-start) 区分 CoT 和 TIR，最终答案使用 `\boxed{...}`；CoT 例子可以直接通过 Transformers 生成，真正运行程序的 TIR 示例另接 Qwen-Agent。原始 GSM8K 则使用自己的计算器采样循环，因此外部计算器并非所有数学推理 SFT 实验的必需环节。
- [原始 GSM8K 计算器采样代码](https://github.com/openai/grade-school-math/blob/master/grade_school_math/calculator.py) 在生成过程中检查未闭合注释里的表达式，计算后填入结果再继续生成。该示例硬编码了旧 tokenizer 的部分 token ID，不能直接作为 Qwen3 的调用程序使用。
- 当前仓库 Python 文件没有接入计算器、工具调用或执行推理代码的控制流程；`main.py` 直接调用 `model.generate()`。已有终端对照分支也是直接生成。
- 2026-10-26 在 `/home/yunli/miniconda3/bin/python` 核对安装包：torch 2.6.0+cu124、transformers 5.20.2、datasets 5.1.0、accelerate 1.15.0、trl 1.16.0、sympy 1.13.1；该解释器未安装 peft、vllm、qwen-agent 或 math-verify。包清单说明当前解释器的安装状态，不代表其他环境或整台机器。
- 现有 Python 与 Transformers 已具备实现简单计算器控制循环的基础，但计算器接入仍需另写程序。SymPy 已安装不等于模型会自动调用它。
- 训练目标采用原始 `answer`，不追加提示词或转换文本；`train_sft.py` 直接复用样本构造，不接入计算器推理循环。
- 当前 Transformers 的 `apply_chat_template()` 在本机返回 BatchEncoding；不能把返回对象本身当作 token 列表。代码显式设置 `return_dict=True` 并读取 `input_ids`。
- 本机 Qwen3 模板将普通助手回答放在空 `<think>` 前缀之后。样本使用 `enable_thinking=False` 生成同样的提示前缀，并将整个前缀设为不计损失；原始解答及 `<|im_end|>` 参与监督。后续训练和推理沿用这一模板。
- 已实际预览第一条训练样本，解题过程、两处计算器注释和最后的 `#### 72` 原样出现在监督文本中。

## SFT 训练代码验证

- 默认使用 Transformers Trainer 做全参数 SFT；面向用户指定的 32 GB 显存，采用 AdamW、每批 4 条和梯度累积 4 次，关闭梯度检查点。权重保持 FP32，GPU 支持时使用 BF16 混合精度。
- 已离线核对全部 7473 条训练样本：监督部分解码后等于原始 `answer` 加对话结束标记，最长为 550 tokens。未截断、过滤或改写解答。
- DataCollatorForSeq2Seq 在未传入编码器模型时可用于本脚本的动态补齐；实测保留已有题目掩码，并将补齐标签设为 `-100`，不产生 decoder 输入。
- 使用 CPU 上的随机微型 Qwen3 验证 Trainer 集成：损失与手工移位后的交叉熵一致，反向传播梯度均有限，AdamW 初始化和保存/重新加载通过。这不验证实际 0.6B 模型在租用 GPU 上的运行显存；未启动真实训练。
- 参考：[Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)。接口与功能验证以本机固定版本为准。

## Git 权限核查

- GitHub 插件在本轮返回的账号为 FishSkin2002，但对 `FishSkin2002/qwen` 仓库查询返回 404。SSH 访问仓库正常，不能把插件的仓库访问失败理解为远端仓库不存在。
- 用户已安装本机 GitHub CLI 2.103.2 并完成登录，CLI API 核对账号为 FishSkin2002，可访问本仓库并创建 PR。后续仓库 PR 操作可直接使用这一已验证的 CLI 登录态，无须重新登录或使用浏览器。

- 原始 push、commit、switch 等命令前缀不能区分尾随危险参数，宽泛规则需撤销，不能仅靠文档声明保留历史。
- 新日常入口固定仓库、分支类型和参数，原始受限写命令不再自动放行；实际部署以用户规则层文件和受保护入口的校验结果为准。
- 来源：[OpenAI 命令规则说明](https://learn.chatgpt.com/docs/agent-configuration/rules)。
