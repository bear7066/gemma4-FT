from config.common import COMMON_TRAINING_DEFAULTS


TRAINING_PROFILE = {
    **COMMON_TRAINING_DEFAULTS,
    "data_path": "./dataset/kinetics_3k/kinetic_3K.json",
    "image_folder": "./dataset/kinetics_3k",
    "output_dir": "./output/gemma4_e4b_lora_only",
    "run_name": "gemma4-e4b-lora_only",
    "training_mode": "lora",
    "num_gpus": 1,
    "optim": "paged_adamw_8bit",
    "learning_rate": 2e-4,
    "projector_lr": 0.0,
    "weight_decay": 0.0,
    "num_train_epochs": 3,
    "eval_strategy": "no",
    "save_strategy": "steps",
    "save_steps": 200,
    "report_to": "wandb",
}
