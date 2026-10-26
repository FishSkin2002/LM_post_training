# GSM8K SFT 训练样本构造

使用 Qwen3-0.6B 的分词器，将 GSM8K 训练集中的题目和标准解答构造成 SFT 输入。本步只预览数据，不加载模型权重或启动训练。

## 数据约定

- `question` 和 `answer` 原样使用，保留解题过程、`<<表达式=结果>>` 计算器注释和 `#### 数值`。
- 使用 Qwen3 的非思考对话模板，将解题过程放在普通助手回答中；模板自带的空 `<think>` 前缀不计损失。这仍然包含逐步解题过程。
- `input_ids` 包含题目和标准解答；题目和对话前缀的 `labels` 为 `-100`，助手解答及结束标记接受监督。标签保持与输入等长，由后续因果语言模型在计算损失时移位。
- 当前每条样本不补齐、不截断。接入批训练时再处理补齐，并将补齐位置的标签设为 `-100`。

## 运行

在已有依赖的环境中，预览一条训练样本及其词元和标签：

```bash
/home/yunli/miniconda3/bin/python prepare_sft_data.py
```

程序预览训练集第一条样本，打印完整输入、参与损失的文本、词元 ID 和监督标签。

本机已有完整缓存，可离线运行：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 /home/yunli/miniconda3/bin/python prepare_sft_data.py
```

`build_sft_example()` 返回 `input_ids`、`attention_mask` 和 `labels`，供后续训练代码复用。推理时需使用同样的非思考前缀及答案格式约定。

保留计算器注释只提供训练文本，不会自动执行计算。

## 来源

- [GSM8K 数据格式与计算器说明](https://github.com/openai/grade-school-math#dataset-details)
- [Qwen3-0.6B 对话模板与模式说明](https://huggingface.co/Qwen/Qwen3-0.6B)
