# 活跃任务环境与实现发现

## 已验证环境

- 主环境：/home/yunli/miniconda3/envs/qwen-vllm/bin/python。名称仅本机示例，不写死应用逻辑。
- 主栈：Python3.12，torch2.8.0+cu128、vllm0.10.2、transformers4.57.6、trl0.27.2、datasets4.4.2、huggingface-hub0.36.0、tokenizers0.22.2、peft0.21.0、accelerate1.15.0。
- 原栈：/home/yunli/miniconda3/bin/python，Python3.12.15、torch2.6.0+cu124、transformers5.20.2、trl1.16.0、datasets5.1.0、hub2.3.0、tokenizers0.23.3。
- 原栈指纹相关实际版本：dill0.4.1、pyarrow26.0.0、xxhash4.0.2、numpy2.5.2、sympy1.13.1、fsspec2026.7.0、multiprocess0.70.19。
- 开发GPU：RTX4060 Laptop 8GB、驱动621.01；CUDA真实检查需沙箱外正式exec权限，沙箱内NVML失败不是实际驱动失败。
- 用户训练机此前vGPU-32GB、驱动550.120/CUDA12.4，已同意升级到CUDA12.8配套R570+。不自行操作远程训练或驱动。
- 已有Conda：/home/yunli/miniconda3/bin/conda。conda env create曾被默认渠道TOS阻挡，成功方式是 conda create --override-channels -c conda-forge --prefix/--name ... python=3.12 pip -y，再pip装依赖。

## 必须保留的兼容处理（b53f9cf）

- datasets5.1增加ArrowTable/ChunkedArray哈希归约，4.4.2没有；同一内容原始数据集指纹不同。不能直接跳过data_plan比较。
- prepared_sft_data保存原环境验证过的数据、manifest与state身份；load_from_disk会因重设格式重新hash，当前loader核对磁盘state与manifest后恢复已导出的原身份，并核对条数、完整tokenizer后端与配置。
- Transformers5保存的TokenizersBackend类名在4.x不可识别；extra_special_tokens的列表需映射到旧版additional_special_tokens，且不止generic类受影响。prepare_sft_data.load_tokenizer已兼容并限定model_input_names为input_ids/attention_mask。
- 4.x加载model时local_files_only不能同时出现在adapter_kwargs；generation工厂已有版本适配。
- 公共vLLM只接收项目构造的prompt IDs，skip_tokenizer_init=True，detokenize=False，手动按原EOS解码；避免引擎再次套模板/加载不兼容分词器。原生与vLLM保持贪心预算和EOS统计，不额外改变公共PAD采样策略。
- TRL0.27.2 GRPO会复制SFT default到固定ref并支持importance correction；DPO需显式load SFT ref、设ref_adapter_name/model_adapter_name，偏好数据列从*_ids映射到*_input_ids。
- GRPO vLLM参数须用ignore_eos/stop_token_ids/bad_words，不能传HF专用use_cache/suppress_tokens。训练默认4条、2048补全、KL beta0.001不变。
- SFT/DPO loss和validation loss不做生成；正常保存final后vLLM评测用os.execv替换训练进程，先释放GPU。普通原生eval可复用模型。
- 续训须保持保存的TRL/vLLM版本；旧无版本元数据的post-SFT断点限原TRL1.16环境。

## 主要代码接口（承接后检查实际版本）

- generation.py：load_generation_config/create_generator/exec_final_evaluation。
- prepared_sft_data.py：save_prepared_sft_data/load_prepared_sft_data；旧公开main需移入legacy/sft_data/私有worker。
- evaluate_checkpoint、sample_checkpoint、train_post_sft支持显式prepared数据并记录来源；需新增约定位置自动发现，不静默自动联网建旧环境。
- 迁移仅独立入口可自动Conda/pip。当前CPU兼容环境不存在，必须真实创建全新前缀验收。不要误用已有base解释器执行数据worker后宣称重建成功。
- 建议CPU profile用torch2.6.0+cpu与原旧固定版本，补齐关键传递依赖；主requirements改vLLM，所有旧依赖文件和worker归legacy/sft_data/。

## 实验文件与日志

- /tmp/qwen-vllm-portability/：baseline.json、prepared/、real-tokenizer/compatibility.json、legacy-base/、legacy-run/final/及实际eval结果。baseline原fingerprint 4e255131893486f9、验证索引[19,5]，原始raw_fingerprint443992961ec20dff。
- 合成原始20条存于baseline.json的rows；真数据worker首迁测试需在新CPU前缀中重新处理这些rows并核对原SFT元数据，不能直接复用已导出prepared冒充准备成功。
- /tmp/qwen-vllm-native-tests.log、/tmp/qwen-vllm-new-stack-tests.log：各45项passed。
- /tmp/qwen-vllm-gpu-smoke/inference.json、grpo.json：真实GPU微型结果。GRPO仅合成奖励，用于验证两步更新/ref不变/step后sync，峰值采样约4.70GiB，不能外推4B显存或吞吐。
- /tmp/qwen-vllm-checkpoint-smoke.py与.log：旧适配器到真实GPU评测及完成续跑的检查；脚本必须有__main__保护。

## Git与上下文注意

- 控制脚本的ensure_idle会拒绝存在MERGE_HEAD时的一切动作，包括commit；如合并冲突，先检查/解决，再通过正式审批申请必要的一次性完成合并命令，不绕过历史保护或改工具。受控merge本身只允许非main并保留历史。
- 旧笔记逐字归档：[2026-11-06历史](docs/planning-archive/2026-11-06-pre-migration/)。它们含旧方案，当前授权以task_plan和已合入PR48要求为准。

## 自动迁移当前实现

- 公共入口 migrate_sft.py；仅调用CPU worker的兼容目录 legacy/sft_data/。普通缓存读取保留 prepared_sft_data.py，普通脚本不导入worker，不自动安装。
- 兼容配置键 instruct-datasets51-cpu-v1-1e87c5bc6256，前缀为 <compat-root>/environments/<key>；锁与状态在外部 locks/states 下，避免移位Conda前缀破坏shebang。
- 真实首迁环境 /tmp/qwen-sft-migration-end-to-end/environments/instruct-datasets51-cpu-v1-1e87c5bc6256。数据 /tmp/qwen-migration-fixture/sft/prepared-sft，权重和run.json字节未变。
- 首迁验收工具 tools/smoke_sft_migration.py：旧栈 --make-fixture DIR 生成微型数据/适配器，主栈 --fixture DIR --compat-root NEW_DIR --conda-exe PATH 实际创建新环境并迁移。仅原数据读取使用合成20题，不使用真实模型或付费API。
- 无旧栈或Conda时，只要有效缓存已存在，普通读取与重复迁移均无需旧解释器。CONDA_EXE=/nonexistent-conda 的真实 migrate_sft CLI 复用通过，日志 /tmp/qwen-sft-migration-reuse-cli.log。
- 重要边界：数据身份来自经旧栈核对的导出，不是新增加的文件防篡改签名。完整 source/划分校验照旧执行；不扩展历史JSONL内容核查。
