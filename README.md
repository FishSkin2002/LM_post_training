# GSM8K SFT

使用 GSM8K 题目和标准解答，对 Qwen3 做思考模式 LoRA 监督微调。开发机（8 GB）默认 Qwen3-0.6B，训练机（32 GB）默认 Qwen3-4B，可选 Qwen3-8B。PEFT 负责适配器，Transformers Trainer 负责训练，保留全参数训练选项。

## 数据约定

- `question` 原样使用。预处理用 `answer` 中唯一的 `####` 拆出推导和最终结果；原始数据文件不改动，推导中的 `<<表达式=结果>>` 计算器注释保留。
- 使用 Qwen3 思考模式，将推导放入 `<think>...</think>`，块外只放数据集的最终结果。`####` 在分词前去掉，不参与损失，也不作为要求模型生成的目标；原始参考解答中仍保留它。
- `input_ids` 包含题目和格式化后的标准解答；题目和对话前缀的 `labels` 为 `-100`，思考内容、最终结果、格式标记及助手结束标记接受监督。使用同一个因果语言模型交叉熵，标签保持与输入等长，由模型计算损失时移位。
- 样本不截断，批次内按最长样本补齐，补齐位置的标签为 `-100`。

助手训练目标的形式为：

```text
<think>
数据集中的推导过程
</think>

数据集中的最终结果
```

## 训练

先在运行训练所用的 Python 环境中安装依赖：

```bash
python -m pip install -r requirements.txt
```

| 配置文件 | 模型 | 环境 |
| --- | --- | --- |
| `configs/development/sft.yaml` | Qwen3-0.6B | 8 GB 开发机 |
| `configs/training/sft.yaml` | Qwen3-4B | 32 GB 训练机，默认 |
| `configs/training/sft-8b.yaml` | Qwen3-8B | 32 GB 训练机，可选，先短测 |

`configs/common.yaml` 存放数据、思考模板和生成设置，以及全参数模式的加载精度。未传 `--config` 时始终使用训练机配置，不根据本机显存切换。模型名称与训练方式以所选 YAML 为准；环境约定见 [默认训练环境](findings.md#默认训练环境)。

在对应机器上运行：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_sft.py --config configs/training/sft.yaml
python train_sft.py --config configs/development/sft.yaml
# 可选 8B：
python train_sft.py --config configs/training/sft-8b.yaml
```

三套配置均默认 LoRA：在注意力的 q/k/v/o 和 MLP 的 gate/up/down 线性层添加适配器，`r=16`、`lora_alpha=32`、`lora_dropout=0.05`，学习率 `1e-4`，每批 1 条、累积 16 次、训练 3 轮。冻结 embedding、输出层及其他底座参数，只训练适配器；思考与最终结果仍共同监督。这些是起始实验参数，未通过答案正确率评测证明优劣。

CUDA 支持 BF16 时，LoRA 底座以 BF16 存储并使用 BF16 混合精度；否则使用 FP16 底座与 FP16 混合精度。适配器参数、梯度和 Adam 状态保持 FP32，CPU 验证采用 FP32。大型模型不静默退回 FP32 底座。三套配置保留 SDPA、非重入梯度检查点，训练时关闭 KV cache，使用融合 AdamW。启动日志显示模式、实际权重精度、批量、累积及混合精度开关。

`--epochs`、`--learning-rate`、`--batch-size`、`--gradient-accumulation-steps`、`--max-steps` 和 `--output-dir` 可临时覆盖配置。保留全参数模式，例如在资源足够的机器上运行：

```bash
python train_sft.py --training-mode full --learning-rate 0.00002 \
  --output-dir /root/autodl-tmp/outputs/training/qwen3-4b-gsm8k-full
```

全参数模式按 common.yaml 的 FP32 精度加载全部权重，显存预算与 LoRA 不同；这个命令不代表 4B 全参数训练适合 32 GB。切换模式时学习率仍由配置或命令行决定，不会自动重新调参。

首次训练会下载模型和数据；已有完整缓存时可加 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`。程序只使用训练集，不自动评测。分配器选项用于缓解变长样本的显存碎片，不能替代对真实峰值的检查。

## 最长样本显存短测

在 32 GB 训练机先运行以下命令；8B 使用对应的配置文件替换。短测重复分词后最长的真实训练样本，每次更新仍完成配置中的全部梯度累积，默认更新 2 次并在每次更新后保存 checkpoint。短测是显存与保存恢复检查，重复单题的损失不能作为效果结论。

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_sft.py \
  --config configs/training/sft.yaml --memory-smoke-test \
  --output-dir /root/autodl-tmp/outputs/training/qwen3-4b-lora-smoke
```

短测结束检查实际运行目录中的 `memory_results.json`：记录模型、底座精度、批量、累积次数、可训练参数、最长样本长度，以及 CUDA 的 `peak_allocated_gib`、`peak_reserved_gib` 和总显存。峰值包括训练中的更新与 checkpoint 保存；不包含其他进程的占用，也不代表全部显存都可用。CPU 验证不写 GPU 峰值。该报告用于正式训练的初步检查，成功短测不保证长时间训练、长文本生成或 GRPO 的显存用量。

再从该次运行的 `checkpoint-1` 恢复到第 2 次更新，显式放到另一个目录，检查优化器与随机状态恢复。路径替换为启动日志打印的实际目录，冲突时可能带数字后缀：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_sft.py \
  --config configs/training/sft.yaml --memory-smoke-test --max-steps 2 \
  --resume-from-checkpoint /root/autodl-tmp/outputs/training/qwen3-4b-lora-smoke/checkpoint-1 \
  --output-dir /root/autodl-tmp/outputs/training/qwen3-4b-lora-smoke-resume
```

确认没有 OOM、损失及梯度范数有限、显存仍有余量后，再使用不带 `--memory-smoke-test` 的普通训练命令。开发机可用其配置进行同样的检查。真实 4B / 8B 峰值由训练机实测，本机微型模型测试不能代替。

## 保存与续训

正常训练每轮保存 checkpoint，保留最近两个；结束时在实际运行目录的 `final/` 保存模型产物和分词器，并保存 `train_results.json`、`trainer_state.json` 与 `memory_results.json`。默认输出路径见对应 YAML。新训练通过数字后缀避开已有目录、空目录和同名文件；启动时打印最终路径，旧结果保留。

LoRA 的 `final/` 保存 `adapter_model.safetensors`、`adapter_config.json` 和分词器，适配器配置记录对应底座及远程 revision；没有重复保存完整底座权重。每个训练 checkpoint 还包含优化器、调度器和随机状态等续训文件。`final/` 用于推理，续训使用完整的 `checkpoint-STEP/`。全参数模式继续保存完整模型。

```bash
python train_sft.py --config configs/training/sft.yaml \
  --resume-from-checkpoint /root/autodl-tmp/outputs/training/qwen3-4b-gsm8k-lora/checkpoint-STEP
```

续训使用相同模型、LoRA 参数和训练设置；短测续训也要带 `--memory-smoke-test`。默认沿用 checkpoint 父目录，显式 `--output-dir` 优先。全参数与 LoRA 的优化器状态不混用；0.6B、4B、8B 的适配器也不通用。原有 checkpoint 保留，新 LoRA 实验从对应官方底座开始，不续接旧全参数优化器状态。

## TensorBoard 训练指标

两台机器的配置均已启用 TensorBoard。先在训练用的 Python 环境中安装项目依赖，然后按上面的命令训练：

```bash
python -m pip install -r requirements.txt
```

Trainer 在第 1 次参数更新和之后每 10 次参数更新记录一次曲线；横轴为优化器更新次数，梯度累积中的单个微批次不会单独记为一步。

| TensorBoard 指标 | 含义 |
| --- | --- |
| `train/loss` | 上次记录以来的平均训练损失 |
| `train/learning_rate` | 学习率 |
| `train/grad_norm` | 当前更新在梯度裁剪前的梯度范数 |
| `train/epoch` | 当前训练轮次，含轮内进度 |
| `train/train_loss` | 训练结束时的整体平均损失 |
| `train/train_runtime` | 本次训练耗时，单位为秒，结束时记录 |
| `train/train_samples_per_second` | 本次训练的平均样本吞吐量，结束时记录 |
| `train/train_steps_per_second` | 本次训练每秒平均参数更新次数，结束时记录 |

日志默认写入所选 `output_dir` 下的 `runs/<时间戳_主机名>/`，训练机和开发机随各自输出目录分开；`--output-dir` 覆盖后日志也随之移动。恢复检查点时会新建日志目录，横轴沿用恢复的更新次数。日志已包含在 Git 忽略的 `outputs/` 中。

在训练机启动查看服务，覆盖基础目录及所有数字后缀目录：

```bash
python -m tensorboard.main --logdir /root/autodl-tmp/outputs/training --host 127.0.0.1 --port 6006
```

在浏览器打开 `http://127.0.0.1:6006`，在 Scalars 或 Time Series 中选择对应运行。开发机使用 `--logdir outputs`；自定义了输出位置时，将 `--logdir` 指向实际运行目录或其父目录。若训练运行在远端，通过 SSH 转发该端口后在本机查看，例如 `ssh -L 6006:127.0.0.1:6006 用户名@训练机地址`。

当前训练脚本只使用训练集；验证损失和 GSM8K 答案正确率需后续接入评测后记录。

## 下载分词器时的 Brotli 报错

若 `AutoTokenizer.from_pretrained()` 下载文件时在 `httpx2/_decoders.py` 报 `TypeError: process() takes no keyword arguments`，先修复训练所用解释器的 Brotli 依赖。HTTPX2 传入了 `output_buffer_limit`，旧版 Brotli 的 `process()` 不接受这个参数；项目依赖已固定兼容版本。

在原来的训练环境中执行：

```bash
python -m pip install --upgrade "Brotli==1.2.1"
python -c "import brotli; assert brotli.Decompressor().process(brotli.compress(b'ok'), output_buffer_limit=1024) == b'ok'; print('Brotli OK:', brotli.__version__)"
```

看到 `Brotli OK: 1.2.1` 后重新运行原训练命令。该错误发生在下载阶段，无须删除模型缓存或调整训练参数。完整新环境直接安装 `requirements.txt` 即可包含此依赖。

## 预览训练样本

在已有依赖的环境中，预览一条训练样本及其词元和标签：

```bash
/home/yunli/miniconda3/bin/python prepare_sft_data.py
```

样本预览默认使用开发机配置，预览训练集第一条样本，打印模型名称、完整输入、参与损失的文本、词元 ID 和监督标签。预览训练机模型时显式选择同一配置：

```bash
python prepare_sft_data.py --config configs/training/sft.yaml
```

本机已有完整缓存，可离线运行：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 /home/yunli/miniconda3/bin/python prepare_sft_data.py
```

`build_sft_example()` 返回 `input_ids`、`attention_mask` 和 `labels`，供训练代码复用。缺少唯一 `####`、推导为空或最终结果为空的解答会报错，不静默改成其他监督目标。

保留计算器注释只提供训练文本，不会自动执行计算。

## 抽题查看原始模型或 checkpoint 的回答

抽样入口支持微调前的原始 Qwen3、本地完整模型和 LoRA 适配器目录。默认从 GSM8K **test** 固定抽取 5 道不同题目，启用思考模式，将问题、模型思考、最终输出和数据集原始参考解答保存到 Markdown 文件。没有生成 `</think>` 时保留整段生成内容并标明尚未识别最终输出。终端仅提示保存位置。

通过模型名称抽样微调前的起始模型，与 --checkpoint 二选一：

```bash
# 32 GB 训练机：
python sample_checkpoint.py --model-name Qwen/Qwen3-4B
# 8 GB 开发机：
python sample_checkpoint.py --model-name Qwen/Qwen3-0.6B
```

这里的“原始”指尚未接受本项目微调，例如使用训练起点 Qwen/Qwen3-4B，不自动替换为其他 Base 变体。原始模型入口默认允许下载缺失的模型和分词器文件，已有缓存会复用；加 --local-files-only 可只使用缓存。使用 --revision 可指定模型分支、标签或提交，报告记录模型名称和解析出的版本，模型和分词器共用同一版本；比较某个 LoRA 结果时，如其 adapter_config.json 中记录了非空的底座 revision，可使用该提交。

```bash
python sample_checkpoint.py --model-name Qwen/Qwen3-4B --local-files-only
```

原始模型默认报告名为 `outputs/samples/original-Qwen--Qwen3-4B-test-seed42.md`，与 checkpoint 报告分开。保持相同的 --split、--seed、--num-samples 和 --max-new-tokens，可在同一模型规模下观察微调前后的回答；生成种子在模型加载后设置，使加载过程不影响生成随机状态。少量抽样不代替完整评测。

例如查看当前下载的 0.6B checkpoint：

```bash
python sample_checkpoint.py \
  --checkpoint outputs/development/gsm8k-sft/checkpoint-1404 \
  --num-samples 5
```

这条命令默认保存到项目内的 `outputs/samples/checkpoint-1404-test-seed42.md`；重复运行会更新同名报告。使用 `--output 路径.md` 可以自定义保存位置。

使用 --checkpoint 时，分词器从该本地目录加载，单独的 `chat_template.jinja` 也需一起保留。完整模型目录包含 `config.json` 和权重；LoRA 目录包含 `adapter_config.json` 和适配器权重，程序会另外从本地路径或 Hugging Face 缓存加载配置中记录的同一底座及 revision。checkpoint 路径继续保持 `local_files_only=True`，底座缓存缺失时先在该机器准备对应版本的底座。推理启用缓存并关闭梯度，不加载优化器状态。

例如，在已有对应底座缓存的训练机上查看 LoRA 的最终产物：

```bash
python sample_checkpoint.py \
  --checkpoint /root/autodl-tmp/outputs/training/qwen3-4b-gsm8k-lora/final
```

适配器文件小不等于推理模型小。4B / 8B 的 BF16 底座不适合当前 8 GB 开发机直接加载，先在训练机抽样并下载 Markdown 报告；本轮不加入量化推理或完整模型合并导出。

数据集已有缓存时，可在命令前加 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` 离线运行。默认自动选择 CUDA，按设备能力使用 BF16 或 FP16；没有 CUDA 时使用 CPU/FP32，也可用 `--device cpu` 显式选择。

`--split train` 可查看训练集题目，`--seed` 控制抽题及生成随机种子。按 Qwen 官方思考模式建议使用采样：`temperature=0.6`、`top_p=0.95`、`top_k=20`、`min_p=0`，设置集中在 `configs/common.yaml`。`--max-new-tokens` 默认 32768，包含思考与最终输出；可按资源预算调小，实际预算记录在报告中。输入加输出预算超过模型配置的上下文长度时会报错，触及生成上限且未生成结束标记时提示截断。

只将原始问题输入模型，参考解答用于事后对照。少量样本用于观察回答格式和解题过程，完整答案正确率仍需另行评测。

## 来源

- [GSM8K 数据格式与计算器说明](https://github.com/openai/grade-school-math#dataset-details)
- [Qwen3-0.6B 对话模板与模式说明](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen3-4B 对话模板与模式说明](https://huggingface.co/Qwen/Qwen3-4B)
- [Qwen3-8B 对话模板与模式说明](https://huggingface.co/Qwen/Qwen3-8B)
- [PEFT LoRA](https://huggingface.co/docs/peft/developer_guides/lora)
- [PEFT 保存格式](https://huggingface.co/docs/peft/developer_guides/checkpoint)
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Transformers TensorBoard 回调](https://huggingface.co/docs/transformers/main_classes/callback#transformers.integrations.TensorBoardCallback)
- [TensorBoard 查看日志](https://www.tensorflow.org/tensorboard/get_started)
- [Brotli 1.2.0 的解压接口变更](https://github.com/google/brotli/releases/tag/v1.2.0)
- [PyTorch CUDA 显存分配设置](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
