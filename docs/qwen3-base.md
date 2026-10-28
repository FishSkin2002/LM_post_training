# Qwen3 Base 训练约定

用户于 2026-10-28 确定：项目一仅使用 Qwen3 Base，以普通文本说明和 Question/Answer 前缀构造数学题续写样本，显存方面只采用 LoRA。该决定替代此前的聊天模板方案；当前优先建立简单、清楚的数学训练基线。

## 模型与范围

- 主线为原始 Base → 本项目 LoRA SFT → 后续另行开展的 GRPO。开发机使用 Qwen/Qwen3-0.6B-Base，训练机默认 Qwen/Qwen3-4B-Base，8B-Base 仍可显式选择。
- 业务代码只服务这条 Base 路线，不保留普通版 Qwen3、旧思考格式或旧 checkpoint 的兼容分支。
- 提供数据检查、LoRA 训练、训练产物保存、原始 Base / 新 LoRA 产物抽样，以及 [训练运行约定](training-workflow.md) 中的 loss 验证记录、独立生成评测、续训、加训和目录避重。沿用 Trainer 和 PEFT，不额外建立训练框架。
- GRPO 实现、量化、适配器合并导出、显存报告和独立短测入口留待后续明确需求。

## 普通文本格式与监督目标

提示的目的，是让原始 Base 与 SFT 模型遵循同一输出协议，使最终数值容易提取、计算过程便于定位，从而比较训练前后的效果。使用以下固定英文提示，通过原写法与目标写法的对比说明格式；公式外侧保留等号，与 GSM8K 解答一致：

```text
Solve the following math problem step by step.
For arithmetic calculations, follow this example: write 2+3=5 as 2+3=<<2+3=5>>5.
On the last line, write only #### followed by the final numeric answer, without units or other text. For example, if the final answer is 4, write #### 4.
Use #### only once in your response. Do not write anything after the final answer.

Question: {question}
Answer:
{answer}<|endoftext|>
```

- question 与 answer 使用 GSM8K 原文。answer 中的推导、计算器注释、#### 及最终数值均保留，不拆分、不重写、不包装思考块。
- 固定说明、Question/Answer 前缀、问题及 padding 的 labels 为 -100；原始 answer 全文和末尾一个原生 EOS 接受同一因果语言模型交叉熵监督。标签与输入等长，由模型执行一次移位；EOS 按词元 ID 追加，不将它与 padding 一起屏蔽。
- 训练和训练过程中的验证 loss 使用标准解答前文预测下一个标准词元，不先 generate 再对生成字符串计算损失。普通批次 padding 与掩码只用于正确构造数据。
- 训练、loss 验证、抽样与独立生成评测共享同一提示构造。生成输入止于 `Answer:` 后的换行，不含参考答案；模型续写解答，输出原生 `<|endoftext|>` 时停止。计分规则沿用 [训练运行约定](training-workflow.md)。
- 原始 Base 的生成也使用同一提示，不只为 SFT 模型提供格式说明。比较训练前后效果时保持问题、提示、解码设置与计分口径一致；格式说明中只含固定的无关示例，不加入待评测题的参考解答。公式注释用于定位和阅读，生成正确率仍仅判定最终数值，同时报告格式有效率与截断率。
- 移除自定义聊天模板、角色消息构造、消息边界停止条件和对应的元数据/文件依赖；不保留旧聊天格式兼容或格式切换分支。运行与训练产物不使用聊天模板，不生成 chat_template.jinja；原生词表保持不变，不添加、删除或重编号词元。
- 运行信息保存普通文本提示与结束设置，并在恢复、加训和独立评测时校验一致；校验服务于当前格式，不保留旧模板读取分支。新格式从原始 Base 发起新训练，旧聊天格式 checkpoint 不作为当前格式的续训、加训或评测来源；已有文件保留。
- 不加入思考开关、think 包装或解析，也不接入执行计算器的工具。提示中的计算注释仅是要学习的输出文本。

## 最小 LoRA 训练

- 使用 PEFT LoRA 与 Transformers Trainer，只训练适配器。当前只保留 LoRA 一条训练路径，移除全参数模式选择及其产物加载兼容。
- 使用固定 FP32 加载和常规训练，不加入自动半精度或混合精度选择；每个批次直接执行更新，不配置梯度累积。
- 不额外配置激活检查点、专用注意力后端、训练 KV cache 覆盖、分配器调优、量化优化器或 offload。框架自身默认行为不另加控制层。
- 批量、学习率、训练轮次和 LoRA 参数作为少量直接配置，保持易读；不宣称这些起始值是最佳参数。
- 保持普通 LoRA 训练，不增加显存短测或峰值统计分支；训练/验证曲线按运行约定记录，必要的功能验收在测试中完成。

## 重写与交付

最小 Base 实现已按“新增撤销提交 → 独立重写提交”通过 PR #15 交付。后续功能在此基础上分小步迭代，要求与功能的交付边界遵循 [Git 工作约定](git-workflow.md)。已有历史、数据及 outputs 产物保留，旧模型产物不属于兼容目标。

## 当前状态

普通文本方案已由用户确认，当前处于独立要求交付阶段；代码实施在要求合入后进行。实现与验证状态见 [README](../README.md) 和 [进展记录](../progress.md)；本文不表示普通文本格式的训练或准确率已经验证。
