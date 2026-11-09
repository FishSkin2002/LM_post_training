"""Train a DPO LoRA policy from a completed SFT adapter."""

import hashlib
import math
import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import set_seed
from transformers.integrations import TensorBoardCallback
from trl import DPOConfig, DPOTrainer
from packaging.version import Version
import trl
import yaml

from utils.post_training_data import DPO_ANSWER_FORMAT, SOURCES, prepare_post_training_data
from utils.prepare_sft_data import load_tokenizer, prepare_dataset, validate_precision
from train import AdapterTrainerMixin, evaluate_after_training, load_training_model, require_resume_engine, save_final
from utils.training_workflow import data_plan, read_source, reserve_directory, save_run
from utils.generation import load_generation_config, package_version
from utils.data_preparation import prepare_sft, prepare_stage


class PreferenceTrainer(AdapterTrainerMixin, DPOTrainer):
    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        # The pinned TRL text collator accepts these IDs without re-rendering an EOS suffix.
        if Version(trl.__version__) < Version("1.0"):
            return dataset.rename_columns({key + "_ids": key + "_input_ids"
                                            for key in ("prompt", "chosen", "rejected")})
        return dataset


def load_config(path):
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    source = config['data']
    if tuple((source[key] for key in ('name', 'config', 'split'))) != SOURCES['dpo']:
        raise ValueError(f"DPO uses {'/'.join(SOURCES['dpo'])}.")
    if not isinstance(source['revision'], str) or not source['revision']:
        raise ValueError('Data revision must be a nonempty string.')
    length_key = 'max_length'
    if type(config[length_key]) is not int or config[length_key] <= 0:
        raise ValueError(f'{length_key} must be a positive integer.')
    beta = config['trainer']['beta']
    if not math.isfinite(beta) or beta < 0 or beta == 0:
        raise ValueError(f"Invalid beta for DPO.")
    return (config, config[length_key])


def training_arguments(settings, output_dir, bf16, tokenizer):
    settings = dict(settings)
    common = dict(
        output_dir=str(output_dir),
        bf16=bf16,
        logging_first_step=True,
        report_to='none',
        eval_strategy='no',
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}
    )
    return DPOConfig(**settings, **common, loss_type='sigmoid', max_length=None)


def train(
    config_path,
    *,
    sft_source=None,
    resume=None,
    output_dir=None,
    generation_config=None,
    generation_backend=None,
    prepared_sft_data=None,
    env="current"
):
    (config, length_limit) = load_config(config_path)
    settings = config['trainer']
    generation_settings = load_generation_config(generation_config, generation_backend)
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Post-SFT training uses one process.')
    saved = step = None
    if resume is not None:
        (resume, saved, step) = read_source(resume, resume=True)
        require_resume_engine(saved)
        prepared_sft_data = prepared_sft_data or saved.get('prepared_sft_data')
        if saved.get('stage') != 'dpo':
            raise ValueError(f"Resume requires a DPO checkpoint.")
        if saved['data'].get('answer_format') != DPO_ANSWER_FORMAT:
            raise ValueError('DPO checkpoint uses a different answer format. Start from the SFT final/ with --source.')
        if output_dir is not None and output_dir.expanduser().resolve() != resume.parent:
            raise ValueError("Resume uses the checkpoint's run directory.")
        for key in ('data', 'trainer', 'max_length', 'max_prompt_length'):
            if config.get(key) != saved['config'].get(key):
                raise ValueError('Resume must keep the saved stage configuration.')
        sft_source = saved['source_adapter']
    (sft_source, sft, _) = read_source(sft_source)
    if sft.get('stage', 'sft') != 'sft' or sft_source.name != 'final':
        raise ValueError('Use the final/ adapter from a finished SFT run.')
    evaluation = {key: sft[key] for key in ('config', 'data', 'text_format')}
    if 'data_environment' in sft:
        evaluation['data_environment'] = sft['data_environment']
    with (sft_source / 'adapter_model.safetensors').open('rb') as weights:
        source_hash = hashlib.file_digest(weights, 'sha256').hexdigest()
    if saved is not None and (
            any(evaluation[key] != saved['evaluation'][key] for key in ('config', 'data', 'text_format'))
            or source_hash != saved['source_adapter_sha256']):
        raise ValueError('The source SFT artifact has changed since this stage started.')
    if resume is not None and (resume.parent / 'final' / 'run.json').is_file():
        raise ValueError('This stage already has a saved final adapter.')
    sft_config = sft['config']
    bf16 = sft_config.get('bf16', False)
    device = 'cpu' if settings.get('use_cpu') or not torch.cuda.is_available() else 'cuda'
    validate_precision(sft_config, device)
    tokenizer = load_tokenizer(sft_source, local_files_only=True)
    run_path = resume.parent if resume is not None else output_dir or config['output_dir']
    args = training_arguments(settings, run_path, bf16, tokenizer)
    sft_data, sft_info, sft_environment, sft_cache, prepared_sft_data = prepare_sft(
        sft_config, tokenizer, source=sft_source, metadata=saved['evaluation'] if saved else sft,
        env=env, prepared=prepared_sft_data, prepare=prepare_dataset)
    plan = data_plan(sft_data, sft, sft_info)
    held_out = sft_data.select(plan['validation_indices'])['question']
    train_data, train_info, environment, cache = prepare_stage(
        'dpo', config['data'], tokenizer, held_out, length_limit,
        env=env, metadata=saved, prepare=prepare_post_training_data)
    evaluation['data_environment'] = sft_environment
    if sft_cache:
        evaluation['sft_data_cache'] = sft_cache
    metadata = {
        'stage': 'dpo',
        'config': config,
        'data': train_info,
        'evaluation': evaluation,
        'scoring': sft['scoring'],
        'source_adapter': str(sft_source),
        'source_adapter_sha256': source_hash
    }
    metadata['data_environment'] = environment
    if cache:
        metadata['training_data_cache'] = cache
    if prepared_sft_data:
        metadata['prepared_sft_data'] = str(Path(prepared_sft_data).expanduser().resolve())
    metadata["generation_engine"] = {
        'backend': 'transformers',
        'trl_version': package_version('trl'),
        'vllm_version': package_version('vllm'),
        'generation_kwargs': None,
        'effective_settings': {key: value for (key, value) in args.to_dict().items() if key.startswith('vllm_')},
    }
    if saved is not None and train_info != saved['data']:
        raise ValueError('Training data or filtering differs from the saved checkpoint.')
    set_seed(args.seed)
    model = load_training_model(sft_config, sft_source)
    if Version(trl.__version__) < Version('1.0'):
        # The reference keeps the original SFT adapter rather than the raw base.
        model.load_adapter(sft_source, adapter_name='ref', is_trainable=False, local_files_only=True)
        model.set_adapter('default')
        (args.ref_adapter_name, args.model_adapter_name) = ('ref', 'default')
    context_limit = length_limit
    if context_limit > model.config.max_position_embeddings:
        raise ValueError("The stage's token budget exceeds the model context.")
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    run_path = resume.parent if resume is not None else reserve_directory(run_path)
    args.output_dir = str(run_path)
    save_run(run_path, metadata)
    print(f"Stage: DPO\nSFT source: {sft_source}\nOutput directory: {run_path}", flush=True)
    model.print_trainable_parameters()
    with SummaryWriter(log_dir=str(run_path / 'runs'), purge_step=step + 1 if step is not None else None) as writer:
        trainer = PreferenceTrainer(
            model=model,
            args=args,
            train_dataset=train_data,
            processing_class=tokenizer,
            callbacks=[TensorBoardCallback(tb_writer=writer)],
            run_metadata=metadata
        )
        result = trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)
        if not math.isfinite(result.training_loss):
            raise RuntimeError('Training loss is not finite.')
        final_dir = save_final(trainer, tokenizer, run_path)
        evaluate_after_training(
            final_dir, generation_settings, model=trainer.model, tokenizer=tokenizer, dataset=sft_data,
            data_info=sft_info, disable_tqdm=args.disable_tqdm, generation_config=generation_config,
        )
    return final_dir
