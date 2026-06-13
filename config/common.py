COMMON_TRAINING_DEFAULTS = {
    "model_name": "google/gemma-4-e4b-it",
    "deepspeed_config": "deepspeed_config/stage1.json",
    "master_port": "29500",
    "bf16": "True",
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "gradient_checkpointing": "True",
    "logging_steps": 10,
    "dataloader_num_workers": 4,
    "max_seq_length": 2304,
    "training_mode": "full",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "image_encoder_lr": 0.0,
    "projector_lr": 2e-5,
    "weight_decay": 0.01,
    "num_train_epochs": 1,
    "save_total_limit": 2,
}


SHELL_ENV_NAMES = {
    "model_name": "MODEL_NAME",
    "data_path": "DATA_PATH",
    "eval_data_path": "EVAL_DATA_PATH",
    "image_folder": "IMAGE_FOLDER",
    "output_dir": "OUTPUT_DIR",
    "run_name": "RUN_NAME",
    "deepspeed_config": "DEEPSPEED_CONFIG",
    "num_gpus": "NUM_GPUS",
    "master_port": "MASTER_PORT",
    "bf16": "BF16",
    "lora_r": "LORA_R",
    "lora_alpha": "LORA_ALPHA",
    "lora_dropout": "LORA_DROPOUT",
    "num_train_epochs": "NUM_TRAIN_EPOCHS",
    "per_device_train_batch_size": "PER_DEVICE_TRAIN_BATCH_SIZE",
    "per_device_eval_batch_size": "PER_DEVICE_EVAL_BATCH_SIZE",
    "gradient_accumulation_steps": "GRADIENT_ACCUMULATION_STEPS",
    "optim": "OPTIM",
    "learning_rate": "LEARNING_RATE",
    "image_encoder_lr": "IMAGE_ENCODER_LR",
    "projector_lr": "PROJECTOR_LR",
    "weight_decay": "WEIGHT_DECAY",
    "warmup_ratio": "WARMUP_RATIO",
    "lr_scheduler_type": "LR_SCHEDULER_TYPE",
    "eval_strategy": "EVAL_STRATEGY",
    "eval_steps": "EVAL_STEPS",
    "save_strategy": "SAVE_STRATEGY",
    "save_steps": "SAVE_STEPS",
    "save_total_limit": "SAVE_TOTAL_LIMIT",
    "gradient_checkpointing": "GRADIENT_CHECKPOINTING",
    "logging_steps": "LOGGING_STEPS",
    "dataloader_num_workers": "DATALOADER_NUM_WORKERS",
    "report_to": "REPORT_TO",
    "max_seq_length": "MAX_SEQ_LENGTH",
    "training_mode": "TRAINING_MODE",
}


def to_shell_defaults(training_profile: dict) -> str:
    lines = []
    for key, env_name in SHELL_ENV_NAMES.items():
        if key not in training_profile:
            continue
        value = str(training_profile[key]).replace('"', '\\"')
        lines.append(f': "${{{env_name}:={value}}}"')
    return "\n".join(lines)
