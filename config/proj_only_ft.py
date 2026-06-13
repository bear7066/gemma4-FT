from config.common import COMMON_TRAINING_DEFAULTS


TRAINING_PROFILE = {
    **COMMON_TRAINING_DEFAULTS,
    "data_path": "./kinetic_3K.json",
    "image_folder": "$HOME/data",
    "output_dir": "./output/gemma4_e4b_caption",
    "run_name": "gemma4-e4b-caption",
    "training_mode": "projector_only",
    "num_gpus": 1,
    "optim": "paged_adamw_8bit",
    "learning_rate": 5e-5,
    "projector_lr": 5e-5,
    "weight_decay": 0.0,
    "num_train_epochs": 3,
    "eval_strategy": "no",
    "save_strategy": "steps",
    "save_steps": 200,
    "report_to": "wandb",
}
