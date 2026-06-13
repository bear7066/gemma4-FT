"""
Gemma 4 Stage 1 Multimodal Fine-tuning
- Model: google/gemma-4-e4b-it (Gemma4ForConditionalGeneration)
- Training: SFT with LoRA via GemmaSFTTrainer + DeepSpeed
- Monitoring: Trackio (real-time metrics)
- Supports: image + video inputs

Usage:
    uv run deepspeed src/train.py --deepspeed deepspeed_config/stage1.json [args...]
"""

import os
import pathlib
import sys
import torch
from transformers import (
    AutoProcessor,
    Gemma4ForConditionalGeneration,
    HfArgumentParser,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from arguments import DataArguments, GemmaSFTTrainingArguments, ModelArguments
from ds_wrapper import make_data_module
from sft import GemmaSFTTrainer
from training_modes import apply_training_mode, configure_gradient_checkpointing
from utils import _log


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, GemmaSFTTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = torch.bfloat16

    _log(f"Loading model: {model_args.model_id}")

    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_args.model_id,
        dtype=compute_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation="sdpa",
    )

    model = apply_training_mode(model, model_args, training_args, compute_dtype)
    configure_gradient_checkpointing(model, training_args)

    model.config.use_cache = False
    model.config.image_encoder_lr = training_args.image_encoder_lr
    model.config.projector_lr = training_args.projector_lr

    # Processor handles image + video tokenization for Gemma 4
    processor = AutoProcessor.from_pretrained(model_args.model_id)

    data_module = make_data_module(
        processor=processor,
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        max_seq_length=training_args.max_seq_length,
    )

    _log(f"data_args.data_path: {data_args.data_path}")
    _log(f"data_args.eval_data_path: {data_args.eval_data_path}")
    _log(f"data_module keys: {list(data_module.keys())}")

    train_dataset = data_module["train_dataset"]
    data_collator = data_module.get("data_collator", None)

    eval_dataset = None

    if data_args.eval_data_path:
        if not os.path.exists(data_args.eval_data_path):
            raise FileNotFoundError(f"eval_data_path not found: {data_args.eval_data_path}")

        eval_data_module = make_data_module(
            processor=processor,
            data_path=data_args.eval_data_path,
            image_folder=data_args.image_folder,
            max_seq_length=training_args.max_seq_length,
        )

        _log(f"eval_data_module keys: {list(eval_data_module.keys())}")

        # make_data_module 通常只會回傳 train_dataset，
        # 即使用的是 val.json，也還是叫 train_dataset
        eval_dataset = eval_data_module.get("eval_dataset", None)

        if eval_dataset is None:
            eval_dataset = eval_data_module.get("train_dataset", None)

    # Debug：確認真的有載入
    _log(f"train_dataset size: {len(train_dataset)}")
    _log(f"eval_dataset size: {len(eval_dataset) if eval_dataset is not None else None}")

    if training_args.eval_strategy != "no" and eval_dataset is None:
        raise ValueError(
            f"eval_strategy is enabled, but eval_dataset is None. "
            f"data_args.eval_data_path={data_args.eval_data_path}. "
            f"Please check make_data_module return keys."
        )

    trainer = GemmaSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    output_dir = pathlib.Path(training_args.output_dir)
    resume = bool(list(output_dir.glob("checkpoint-*")))
    _log(f"{'Resuming' if resume else 'New training'} → {training_args.output_dir}")
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_state()
    model.config.use_cache = True

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(training_args.output_dir)
    else:
        from peft import PeftModel
        if isinstance(trainer.model, PeftModel):
            # Save only LoRA adapter weights (~32 MB) instead of full model (~8 GB)
            trainer.model.save_pretrained(training_args.output_dir)
        else:
            state_dict = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
            trainer._save(training_args.output_dir, state_dict=state_dict)
            trainer.model.config.save_pretrained(training_args.output_dir)

    _log("Training complete, model saved to", training_args.output_dir)


if __name__ == "__main__":
    train()
