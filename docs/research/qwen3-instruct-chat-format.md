# Qwen3-4B-Instruct-2507 的数学聊天样本格式

本项目使用 Instruct 和原生聊天模板：固定解题要求放入 system，user 原样保存 problem，assistant 保存筛选后的完整 R1 解答。此文保留选型依据和实际渲染示例；当前规则以[模型与样本约定](../qwen3-base.md)为准。

## system 放置的项目证据

- [Fast-Math-R1 实际 SFT 代码](https://github.com/analokmaus/kaggle-aimo2-fast-math-r1/blob/09bf8ff6c71928da3ca4cca35d0643fc68f1b5e7/experiments/train_first_stage.py#L93)把逐步推理和 boxed 要求放在 system，user 直接使用 problem，assistant 使用 generation。它是使用 OpenR1 派生集的直接对应案例。
- [OpenR1 后续蒸馏配方](https://github.com/huggingface/open-r1/blob/5b6ff22b3fb7aa069c54866e517f39dfc3160e09/recipes/OpenR1-Distill-7B/sft/config_distill.yaml)通过模板的 system 指令约定思考和正式解答。该快照的数据已改为 Mixture-of-Thoughts，只作为同项目后续实践，不冒充原始 OpenR1-Math 实验。
- 官方模型模板支持 system，且下述完整模板渲染已验证本项目分工。结论是这种包装合理、可直接实施；这些资料没有证明 system 相比相同指令放 user 一定提高准确率。保持本项目短提示，不照搬其他项目的长身份或思考指令。

## 官方格式依据

核实版本为 `cdbee75f17c01a7cc42f958dc650907174af0554` 的 [tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer_config.json)。

- 一条普通消息渲染为 `<|im_start|>角色\n正文<|im_end|>\n`。
- 模板支持显式 system 消息；本项目提供一条固定 system 指令。未提供 system 且没有 tools 时，模板也不会隐式补入 system。
- assistant 字符串正文原样插入，R1 的 `<think>...</think>` 因此得以保留；模板本身不自动插入或删除思考段。
- 生成前缀由 `add_generation_prompt=True` 添加 `<|im_start|>assistant\n`，没有预填的 `<think>`。
- 原生 EOS 为 `<|im_end|>`，原生 PAD 为 `<|endoftext|>`；模板结束后不再补 EOS。
- 模板已存在，使用 `tokenizer.apply_chat_template` 构造输入，不另写自定义模板或改动词表。

## 具体样本

下面是自写短例，用来演示包装；不声称是 OpenR1 原版题目或 R1 原始输出。生产样本的 assistant 替换为选中的真实 R1 轨迹，格式提示中的 4 只是固定例子。

消息结构：

```python
messages = [
    {
        "role": "system",
        "content": (
            "Solve the following math problem step by step.\n"
            "End your response with the final answer in \\boxed{...}, "
            "for example \\boxed{4}."
        ),
    },
    {"role": "user", "content": "What is 1/2 divided by 2?"},
    {
        "role": "assistant",
        "content": (
            "<think>\n"
            "Dividing by 2 is multiplying by 1/2, so the result is 1/4.\n"
            "</think>\n"
            "\\boxed{\\frac{1}{4}}"
        ),
    },
]
```

完整训练文本由上述 messages 使用 `add_generation_prompt=False` 渲染：

```text
<|im_start|>system
Solve the following math problem step by step.
End your response with the final answer in \boxed{...}, for example \boxed{4}.<|im_end|>
<|im_start|>user
What is 1/2 divided by 2?<|im_end|>
<|im_start|>assistant
<think>
Dividing by 2 is multiplying by 1/2, so the result is 1/4.
</think>
\boxed{\frac{1}{4}}<|im_end|>
```

实际渲染在最后的 im_end 后还有一个换行。消息边界由模板提供，think 与 boxed 属于 assistant 的解答内容。

生成时提供相同 system 和 user 消息，以 `add_generation_prompt=True` 渲染：

```text
<|im_start|>system
Solve the following math problem step by step.
End your response with the final answer in \boxed{...}, for example \boxed{4}.<|im_end|>
<|im_start|>user
What is 1/2 divided by 2?<|im_end|>
<|im_start|>assistant
```

模型从此处开始续写。未接受本项目 SFT 的 Instruct 与本项目 SFT 结果都用此输入，不给参考解答，不预填 think 或空思考段；是否生成思考段属于它们各自的输出行为。无需启用该版本没有提供的 thinking 切换功能。

## 后续实现边界

- 继续 assistant-only 的常规词元交叉熵：system、user、assistant 起始头和 padding 不计 loss；选定 R1 轨迹的思考、正式解答、最终答案及 assistant 结束 EOS 参与监督。模板末尾格式换行不作为额外解答目标。不假定框架会自动给出所需掩码。
- 训练、loss 验证、数据预览、训练前后生成评测共用消息构造与原生模板。固定任务和格式要求放在 system；user.content 直接使用 problem 原文，不拼提示、不添加 Question 标签或清理题干。assistant 内 LaTeX 和思考内容保持。
- 生成以该模型的原生 EOS 停止；保存来源时记录实际模型、模板及结束设置。当前只支持 Base、显式清空 chat_template、追加 endoftext 的入口需要一并调整，已有运行产物保留。
- 最终答案按 boxed 约定提取；思考段的出现、长度或内容不充当答案正确性的替代指标。具体筛选、数学等价判定及长度预算仍在后续要求稿收口。

## 本次核验

仅获取官方 9377 字节的分词器配置，以本地 Transformers 的模板渲染函数执行完整官方 Jinja 模板；未下载词表文件、模型权重或运行生成。沙箱内首次域名解析失败，随后经正式审批读取同一固定版本小文件。

已核对训练与生成前缀逐字符对齐、assistant 正文和一对 think 原样保留、不自动补 think、不隐式补 system、显式 system 分支以及 EOS/PAD 配置。按用户选定的 system/user/assistant 结构重新渲染，核对 user 题目保持原文。临时渲染产物位于 `/tmp/qwen3-instruct-template-eval-20261029/`；它们不是生产训练数据，也不表示业务代码已实现该协议。
