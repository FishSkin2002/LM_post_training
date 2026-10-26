# 环境发现

## 可用运行环境

- 仓库为 `/home/yunli/code/qwen/codex`，远程 origin 为 `git@github.com:FishSkin2002/qwen.git`。
- `/home/yunli/miniconda3/bin/python` 为 Python 3.12.15，已有 torch 2.6.0+cu124、transformers 5.20.2、datasets 5.1.0、accelerate 1.15.0。
- 可用 GPU 为约 8 GB 显存的 NVIDIA RTX 4060；Codex 沙箱内访问 GPU 受限，需要正式宿主授权运行推理。
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
- 训练目标采用原始 `answer`，不追加提示词或转换文本。本步只构造训练样本，不接入计算器推理循环。
- 当前 Transformers 的 `apply_chat_template()` 在本机返回 BatchEncoding；不能把返回对象本身当作 token 列表。代码显式设置 `return_dict=True` 并读取 `input_ids`。
- 本机 Qwen3 模板将普通助手回答放在空 `<think>` 前缀之后。样本使用 `enable_thinking=False` 生成同样的提示前缀，并将整个前缀设为不计损失；原始解答及 `<|im_end|>` 参与监督。后续训练和推理沿用这一模板。
- 已实际预览第一条训练样本，解题过程、两处计算器注释和最后的 `#### 72` 原样出现在监督文本中。

## Git 权限核查

- 原始 push、commit、switch 等命令前缀不能区分尾随危险参数，宽泛规则需撤销，不能仅靠文档声明保留历史。
- 新日常入口固定仓库、分支类型和参数，原始受限写命令不再自动放行；实际部署以用户规则层文件和受保护入口的校验结果为准。
- 来源：[OpenAI 命令规则说明](https://learn.chatgpt.com/docs/agent-configuration/rules)。
