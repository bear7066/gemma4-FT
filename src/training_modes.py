from peft import LoraConfig, get_peft_model
from arguments import GemmaSFTTrainingArguments, ModelArguments
from utils import _freeze_llm, _log, _unfreeze_image_encoder


def apply_training_mode(model, model_args: ModelArguments, training_args: GemmaSFTTrainingArguments, compute_dtype):
    device = training_args.device

    if model_args.training_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        _log("Training mode: full fine-tuning")
        return model

    if model_args.training_mode == "projector_only":
        _freeze_llm(model)
        _unfreeze_image_encoder(model, compute_dtype, device)
        _log("Training mode: projector-only fine-tuning")
        return model

    if model_args.training_mode == "lora":
        _freeze_llm(model)
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
        _log("Training mode: LoRA fine-tuning")
        return model

    raise ValueError(f"Unsupported training_mode: {model_args.training_mode}")


def configure_gradient_checkpointing(model, training_args: GemmaSFTTrainingArguments):
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
