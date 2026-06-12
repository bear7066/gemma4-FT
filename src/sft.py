"""
Optimizer with separate learning rates for Gemma 4 E4B action-recognition fine-tuning:
- embed_vision (projector)
- vision_tower (image encoder; frozen by default)
- LLM backbone (LoRA adapters)
"""

import torch.nn as nn
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names

# transformers 5.x removed ALL_LAYERNORM_LAYERS from trainer; define locally
ALL_LAYERNORM_LAYERS = [nn.LayerNorm]


"""
1. override create_optimizer
2. 替 embed_vision / vision_tower 設不同 learning rate
3. log optimizer param group LR
"""

class GemmaSFTTrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model
        if self.optimizer is not None:
            return self.optimizer

        decay_params = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_params = [n for n in decay_params if "bias" not in n]

        lr_mapper: dict[str, float] = {}
        if self.args.projector_lr is not None:
            lr_mapper["embed_vision"] = self.args.projector_lr
        if self.args.image_encoder_lr is not None:
            lr_mapper["vision_tower"] = self.args.image_encoder_lr

        special_names: set[str] = set()
        for keyword in lr_mapper:
            for n, _ in opt_model.named_parameters():
                if keyword in n:
                    special_names.add(n)

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if n in decay_params and n not in special_names and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if n not in decay_params and n not in special_names and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        for keyword, lr in lr_mapper.items():
            module_names = {n for n, _ in opt_model.named_parameters() if keyword in n}
            optimizer_grouped_parameters.extend([
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n in decay_params and n in module_names and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": lr,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n not in decay_params and n in module_names and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ])

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch,
        ignore_keys_for_eval, start_time, learning_rate=None,
    ):
        super()._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch,
            ignore_keys_for_eval, start_time, learning_rate=learning_rate,
        )
        if self.control.should_log and self.optimizer is not None:
            logs = {}
            for i, pg in enumerate(self.optimizer.param_groups):
                name = pg.get("param_group_name", f"group_{i}")
                logs[f"lr_{name}"] = pg["lr"]
            self.log(logs)
