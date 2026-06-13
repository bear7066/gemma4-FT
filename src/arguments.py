from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

from config.common import COMMON_TRAINING_DEFAULTS


@dataclass
class ModelArguments:
    model_id: str = field(
        default=COMMON_TRAINING_DEFAULTS["model_name"],
        metadata={"help": "HuggingFace model ID or local path"},
    )
    lora_r: int = field(default=COMMON_TRAINING_DEFAULTS["lora_r"], metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=COMMON_TRAINING_DEFAULTS["lora_alpha"], metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=COMMON_TRAINING_DEFAULTS["lora_dropout"], metadata={"help": "LoRA dropout"})
    training_mode: str = field(
        default=COMMON_TRAINING_DEFAULTS["training_mode"],
        metadata={"help": "Training mode: full, lora, or projector_only"},
    )


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
class GemmaSFTTrainingArguments(TrainingArguments):
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
