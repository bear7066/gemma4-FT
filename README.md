# GemmaFT — Gemma 4 E4B Action Recognition Fine-tuning

Video action-recognition SFT pipeline for `google/gemma-4-e4b-it`
(`Gemma4ForConditionalGeneration`). LoRA on the LLM backbone, trainable
`embed_vision` projector, frozen `vision_tower`, DeepSpeed ZeRO-2, PyAV
video decoding (bypasses system FFmpeg).

## Layout

```
src/
  train.py        entrypoint (Gemma4ForConditionalGeneration + LoRA)
  sft.py          GemmaSFTTrainer with per-group LRs
  ds_wrapper.py   SupervisedDataset (messages format, PyAV video I/O)
utils/
  utils.py        freeze/unfreeze helpers (model.model.language_model etc.)
config/
  common.py       shared default values
  full_ft.py      full fine-tuning profile
  lora_ft.py      LoRA fine-tuning profile
  proj_only_ft.py projector-only fine-tuning profile
  entry.py        small CLI entry for scripts
deepspeed_config/stage1.json   ZeRO-2 config
scripts/full_ft.sh             full fine-tuning launcher
scripts/lora_ft.sh             LoRA fine-tuning launcher
scripts/proj_only_ft.sh        projector-only fine-tuning launcher
scripts/train.sh               shared launcher used by profile wrappers
```

## Data format (messages JSON)

```json
[
  {
    "video_metadata": {"fps": 25.0, "duration_sec": 8.3},
    "messages": [
      {"role": "user", "content": [
        {"type": "video", "video": "clips/xxx.mp4"},
        {"type": "text",  "text": "What action is performed?"}
      ]},
      {"role": "assistant", "content": [{"type": "text", "text": "riding a bicycle"}]}
    ]
  }
]
```

## Quickstart

```bash
# Train
cd ~/test/GemmaFT
DATA_PATH=~/data/videochat2_action/videochat2_action.json \
OUTPUT_DIR=./output/gemma4_e4b_action_stage1 \
bash scripts/full_ft.sh
```

## Gemma 4 E4B notes (design choices baked in)

- `Gemma4ForConditionalGeneration` wraps `Gemma4Model` at `.model`; LLM is
  at `model.model.language_model`, vision encoder at
  `model.model.vision_tower`, projector at `model.model.embed_vision`.
- LoRA uses `layers_to_transform=list(range(42))` +
  `layers_pattern="language_model.layers"` to skip
  `Gemma4ClippableLinear` inside the vision encoder. E4B has 42 text
  layers — do not change this unless you move to another Gemma 4 size.
- Video I/O uses PyAV (bundled FFmpeg). `SupervisedDataset._load_video_as_array`
  decodes to numpy `[T,H,W,C]`; the Gemma 4 video processor skips its
  internal FFmpeg path when it sees a 4-D array. Do not switch to
  torchcodec / torchvision video.
- `transformers 5.x` no longer exports `ALL_LAYERNORM_LAYERS`; `sft.py`
  defines it locally as `[nn.LayerNorm]`.
- `scripts/full_ft.sh` defaults target 4 GPUs; scale `NUM_GPUS`,
  `gradient_accumulation_steps`, and ZeRO stage as needed.

## Evaluation

Handled in a separate repo (not included here).
