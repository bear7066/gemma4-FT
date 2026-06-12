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
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import (
    AutoProcessor,
    Gemma4ForConditionalGeneration,
    HfArgumentParser,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config.common import COMMON_TRAINING_DEFAULTS
from ds_wrapper import make_data_module
from sft import GemmaSFTTrainer
from utils import _log


@dataclass
class ModelArguments:
    model_id: str = field(
        default=COMMON_TRAINING_DEFAULTS["model_name"],
        metadata={"help": "HuggingFace model ID or local path"},
    )
    use_lora: bool = field(
        default=COMMON_TRAINING_DEFAULTS["use_lora"] == "True",
        metadata={"help": "Use LoRA for LLM backbone (recommended for stage 1)"},
    )
    lora_r: int = field(default=COMMON_TRAINING_DEFAULTS["lora_r"], metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=COMMON_TRAINING_DEFAULTS["lora_alpha"], metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=COMMON_TRAINING_DEFAULTS["lora_dropout"], metadata={"help": "LoRA dropout"})


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Training data JSON file path"})

    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Validation data JSON file path"},
    )

    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Image/video root directory"},
    )


@dataclass
class Stage1TrainingArguments(TrainingArguments):
    image_encoder_lr: Optional[float] = field(
        default=COMMON_TRAINING_DEFAULTS["image_encoder_lr"],
        metadata={"help": "Vision tower learning rate (0 = frozen)"},
    )
    projector_lr: Optional[float] = field(
        default=COMMON_TRAINING_DEFAULTS["projector_lr"],
        metadata={"help": "embed_vision (projector) learning rate"},
    )
    max_seq_length: int = field(
        default=COMMON_TRAINING_DEFAULTS["max_seq_length"],
        metadata={"help": "Max sequence length"},
    )
    cache_dir: Optional[str] = field(default=None)


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, Stage1TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = torch.bfloat16

    _log(f"Loading model: {model_args.model_id}")

    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_args.model_id,
        dtype=compute_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation="sdpa",
    )

    # Full fine-tuning: train all parameters
    for param in model.parameters():
        param.requires_grad = True

    _log("Full fine-tuning enabled: all parameters trainable")

    # Optionally add LoRA to LLM backbone (keeps it frozen but adds trainable adapters)
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            layers_to_transform=list(range(42)),  # gemma4-e4b has 42 text layers
            layers_pattern="language_model.layers",
        )
        model = get_peft_model(model, lora_config)
        _log("LoRA applied to LLM backbone")

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

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
