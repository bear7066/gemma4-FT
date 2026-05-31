#!/bin/bash
# Gemma 4 E4B — Action Recognition Fine-tuning (Stage 1)
# Target model: google/gemma-4-e4b-it
# Task: video action recognition (K710 / VideoChat2-IT / WebVid action)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLASH_ATTENTION_FORCE_DISABLE=1
export TRANSFORMERS_NO_FLASH_ATTENTION=1

set -e

MODEL_NAME="${MODEL_NAME:-google/gemma-4-e4b-it}"

# Recommended action-recognition datasets produced by the prepare_*.py scripts:
#   ~/data/videochat2_action/videochat2_action.json      (prepare_videochat2.py)
#   ~/data/k710_action.json                              (prepare_k710_gemma4.py)
#   ~/data/webvid_openai_rewritten/webvid_action.json    (prepare_webvid_openai.py)
DATA_PATH="${DATA_PATH:-./dataset/gemma-4-e4b-kinetics_54K/annotations/splits/train.json}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-./dataset/gemma-4-e4b-kinetics_54K/annotations/splits/val.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-./dataset/gemma-4-e4b-kinetics_54K}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/gemma4_e4b_action_stage1}"
RUN_NAME="${RUN_NAME:-gemma4-e4b-action-stage1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-deepspeed_config/stage1.json}"

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

uv run deepspeed \
    --num_gpus "$NUM_GPUS" \
    --master_port "$MASTER_PORT" \
    stage1/train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    \
    --bf16 True \
    \
    --use_lora False \
    --lora_r 16 \
    --lora_alpha 32 \
    \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --optim "adamw_torch" \
    \
    --learning_rate 5e-6 \
    --image_encoder_lr 0.0 \
    --projector_lr 5e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --report_to "none"
    # To enable Trackio monitoring: --report_to "trackio"
