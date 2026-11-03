# 实验一 使用GSM8K数据集对Qwen3-4B-Base进行LoRA SFT

**实验目的：** 使用小数据集快速跑通LoRA SFT流程，判定微调是否真正提升了模型的推理能力。

**测试集来源：** 在训练前从全部数据中划出748题，未参与任何训练和验证流程。

**测试结果原始数据：** [Tensorboard记录](events.out.tfevents.1793201305.autodl-container-b027488635-81279389.72516.0) | [Qwen3-4B-Base测试结果原始数据](base-Qwen3-4B-Base-n748.json) | [训练后模型测试结果原始数据](final-n748.json)

**实验设计：** 
1. 使用LoRA，设置 $r=16$ 、 $\alpha=32$
2. 对模型输出与标准答案计算逐token交叉熵损失
3. 为排除训练时模型过度关注文本格式，且为方便提取原模型的答案，在两个模型的提示词中都加入以下prompt： `Solve the following math problem step by step.\nOn the last line, write only #### followed by the final numeric answer, without units or other text. For example, if the final answer is 4, write #### 4.\nUse #### only once in your response. Do not write anything after the final answer.\n\n`
4. 考虑到Qwen Base系列模型没有经过有聊天模板的后训练，本次实验为简化实验在prompt后加入纯文本问答模板 `Question: {question}\nAnswer:\n` ，模型在其后输出答案

**实验过程：** 为排除Qwen3-4B-Base原模型的准确率

**衡量指标：** 最终答案正确率 `metrics.answer_accuracy` ，不检查任何中间过程的准确率。

**实验结果：** 
训练过程中，每训练完1epoch，训练损失都会有一次较大的下降，而验证损失同步有一次较大的升高，但并未影响最终测试的答题准确率，推测验证损失高低跟推理正误并无直接关系，因验证损失主要关注“模型与标准答案逐token差异”，这与推理是否正确还是有区别的。最终训练后模型准确率为0.8206，原模型为0.6403