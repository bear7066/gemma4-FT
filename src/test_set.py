#!/usr/bin/env python3
"""
test_set.py

Evaluate a Hugging Face Gemma 4 / multimodal action-recognition model on test.json.

This script is designed to live at:
    ./src/test_set.py

It intentionally follows ./src/train.py style:
    - sys.path inserts the project root
    - imports make_data_module from ./src/ds_wrapper.py
    - uses the same ds_wrapper preprocessing path for video/image/text samples

Default test set paths:
    data_path    = ./dataset/gemma-4-e4b-kinetics_54K/annotations/splits/test.json
    image_folder = ./dataset/gemma-4-e4b-kinetics_54K
    output_dir   = ./test_results

Output folder rule:
    Results are automatically saved under one extra folder named after the model.
    Example:
        --model_id THChou1220/gemma-4-e4b-kinetics54K_FFT
        --output_dir ./test_results

    Actual files will be saved to:
        ./test_results/gemma-4-e4b-kinetics54K_FFT/

Outputs:
    test_results/
    └── <model_name>/
        ├── metrics.json
        ├── caption_metrics.json
        ├── predictions.jsonl
        ├── predictions_pretty.json
        ├── per_class_metrics.csv
        └── confusion_matrix.csv

Metrics:
    Classification/action-label metrics:
        - test_loss
        - Top-1 accuracy
        - Macro precision / recall / F1
        - Per-class accuracy
        - Confusion matrix

    Caption/text-generation metrics:
        - CIDEr, implemented as a lightweight CIDEr-style TF-IDF cosine score
        - ROUGE-L F1
        - BLEU-1
        - BLEU-4
        - BERTScore precision / recall / F1, if bert_score is installed

Recommended smoke test:
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 src/test_set.py \
      --model_id THChou1220/gemma-4-e4b-kinetics54K_FFT \
      --limit 10 \
      --skip_bertscore \
      --device_map balanced \
      --num_beams 1 \
      --max_new_tokens 64 \
      --output_dir ./test_results

Recommended full test:
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 src/test_set.py \
      --model_id THChou1220/gemma-4-e4b-kinetics54K_FFT \
      --device_map balanced \
      --num_beams 1 \
      --max_new_tokens 64 \
      --output_dir ./test_results

Optional 20-sample full-metric check:
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 src/test_set.py \
      --model_id THChou1220/gemma-4-e4b-kinetics54K_FFT \
      --limit 20 \
      --device_map balanced \
      --num_beams 1 \
      --max_new_tokens 64 \
      --output_dir ./test_results

Optional dependencies:
    pip install tqdm bert-score

Notes:
    - BERTScore can be slow and may need to download the BERTScore backbone model.
      Use --skip_bertscore if you only need the classification/action metrics.
    - CIDEr is most meaningful for natural-language captions with multiple references.
      For single action labels, treat CIDEr/BLEU/ROUGE/BERTScore as auxiliary only.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
import pathlib
import re
import sys
import gc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Gemma4ForConditionalGeneration
import transformers

# Match train.py import behavior.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ds_wrapper import (  # noqa: E402
    IGNORE_INDEX,
    DataCollatorForSupervisedDataset,
    SupervisedDataset,
    make_data_module,
)


ASSISTANT_ROLES = {"assistant", "model"}

DEFAULT_DATA_PATH = "./dataset/gemma-4-e4b-kinetics_54K/annotations/splits/test.json"
DEFAULT_IMAGE_FOLDER = "./dataset/gemma-4-e4b-kinetics_54K"
DEFAULT_OUTPUT_DIR = "./test_results"


def get_model_result_dir_name(model_id: str) -> str:
    """Return a safe folder name for a model id/path.

    Examples:
        THChou1220/gemma-4-e4b-kinetics54K_FFT -> gemma-4-e4b-kinetics54K_FFT
        ./outputs/final_model                  -> final_model
    """
    cleaned = str(model_id).strip().rstrip("/\\")
    if not cleaned:
        return "unknown_model"

    # HF repo ids use "/", local paths may use "/" or "\".
    name = re.split(r"[/\\]+", cleaned)[-1].strip()
    if not name:
        name = "unknown_model"

    # Keep the name readable while preventing accidental nested folders.
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._-")
    return name or "unknown_model"

PREFIX_PATTERNS = [
    r"^the\s+(action|activity)\s+(shown\s+)?(is|being performed is)\s*[:\-]?\s*",
    r"^(action|activity)\s*[:\-]\s*",
    r"^this\s+video\s+shows\s+(a\s+person\s+|someone\s+)?",
    r"^the\s+video\s+shows\s+(a\s+person\s+|someone\s+)?",
    r"^(a\s+person|someone|the\s+person)\s+(is\s+|are\s+)?",
    r"^it\s+is\s+",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a HF Gemma 4 action-recognition model on test set."
    )

    parser.add_argument(
        "--model_id",
        required=True,
        help="Hugging Face model repo id or local model path, e.g. thchou1220/my-gemma-action-model",
    )
    parser.add_argument(
        "--data_path",
        default=DEFAULT_DATA_PATH,
        help=f"Path to test.json. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--image_folder",
        default=DEFAULT_IMAGE_FOLDER,
        help=f"Dataset root folder for relative video/image paths. Default: {DEFAULT_IMAGE_FOLDER}",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save test results. Default: {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2304,
        help="Use the same max_seq_length as training.",
    )
    parser.add_argument(
        "--max_decode_frames",
        type=int,
        default=8,
        help="Use the same max_decode_frames as training.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For quick smoke test; evaluate only first N samples.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum generated tokens for action answer.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Beam count for generation. The script always returns only one best answer; use 1 for memory-safe greedy decoding.",
    )
    parser.add_argument(
        "--candidate_labels",
        default=None,
        help=(
            "Optional labels.txt, one label per line. "
            "If not provided, labels are inferred from assistant answers in the evaluated test set."
        ),
    )
    parser.add_argument(
        "--fuzzy_threshold",
        type=float,
        default=0.86,
        help="Fuzzy matching threshold for mapping generated text to known labels. Set 1.0 to disable.",
    )
    parser.add_argument(
        "--unknown_label",
        default="__unknown__",
        help="Label used when generated output cannot be mapped to any known class.",
    )

    parser.add_argument(
        "--skip_loss",
        action="store_true",
        help="Skip teacher-forced test_loss computation.",
    )
    parser.add_argument(
        "--loss_batch_size",
        type=int,
        default=1,
        help="Keep 1 for multimodal/video eval unless you are sure batching works.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers for loss computation.",
    )

    parser.add_argument(
        "--skip_bertscore",
        action="store_true",
        help="Skip BERTScore computation. Recommended for quick tests.",
    )
    parser.add_argument(
        "--bertscore_model_type",
        default="distilbert-base-uncased",
        help=(
            "Backbone for BERTScore. distilbert-base-uncased is lighter; "
            "roberta-large is common but much heavier."
        ),
    )
    parser.add_argument(
        "--bertscore_batch_size",
        type=int,
        default=32,
        help="BERTScore batch size.",
    )

    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Model dtype for inference.",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help='Usually "auto" for large models. Use "cuda", "cpu", or empty string if needed.',
    )
    parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        help='Attention implementation, e.g. "sdpa", "eager", "flash_attention_2". Default: sdpa.',
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to from_pretrained.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision/branch/commit.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. If unset, uses login/cache or HF_TOKEN env var.",
    )

    return parser.parse_args()


def get_torch_dtype(name: str) -> torch.dtype:
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model_and_processor(args: argparse.Namespace):
    token = args.token or os.environ.get("HF_TOKEN")
    dtype = get_torch_dtype(args.dtype)

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        token=token,
        trust_remote_code=args.trust_remote_code,
    )

    if getattr(processor, "tokenizer", None) is None:
        raise RuntimeError(
            "Processor has no tokenizer. Please check whether the HF repo contains tokenizer/processor files."
        )

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model_kwargs = dict(
        revision=args.revision,
        token=token,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    )

    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    # First try the exact model class used in train.py.
    try:
        model = Gemma4ForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
        model.eval()
        return model, processor
    except Exception as gemma_err:
        print(f"[WARN] Gemma4ForConditionalGeneration failed: {gemma_err}")
        print("[WARN] Falling back to AutoModel classes...")

    tried: List[str] = []
    last_err: Optional[Exception] = None
    for class_name in [
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ]:
        cls = getattr(transformers, class_name, None)
        if cls is None:
            continue
        tried.append(class_name)
        try:
            model = cls.from_pretrained(args.model_id, **model_kwargs)
            model.eval()
            return model, processor
        except Exception as err:
            last_err = err

    raise RuntimeError(f"Could not load model with any of {tried}. Last error: {last_err}")


def get_input_device(model: torch.nn.Module) -> torch.device:
    """Return the fixed input device used for evaluation.

    IMPORTANT:
    Do not inspect model.hf_device_map here. Accelerate/Transformers may store
    GPU ids as values like 0, 1, "0", or "1". Calling torch.device("1")
    is invalid and caused repeated RuntimeError: Invalid device string: '1'.

    For this script we intentionally keep the model sharded with
    --device_map balanced/auto, but always place input tensors on cuda:0.
    Accelerate will route tensors through the sharded model correctly.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        return next(p for p in model.parameters() if p.is_floating_point()).dtype
    except StopIteration:
        return torch.float32


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    model_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device=device, dtype=model_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return " ".join(part.strip() for part in parts if part and part.strip()).strip()

    if isinstance(content, dict) and content.get("type") == "text":
        return str(content.get("text", "")).strip()

    return ""


def get_ground_truth_text(sample: Dict[str, Any]) -> str:
    for msg in sample.get("messages", []):
        if msg.get("role") in ASSISTANT_ROLES:
            text = extract_text_from_content(msg.get("content", ""))
            if text:
                return text
    raise ValueError("No assistant/model text found in sample.")


def get_sample_label(sample: Dict[str, Any]) -> str:
    """Return the action class label from the JSON sample.

    Preferred source is the top-level `label`, e.g. "land sailing".
    If missing, fall back to the class folder inside the video path:
      kinetic600/land sailing/QszpApNTHuQ_000038_000048 -> land sailing
    If both are missing, fall back to assistant text so caption-only datasets still work.
    """
    label = sample.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()

    video_path = get_sample_video_path(sample)
    if isinstance(video_path, str) and video_path.strip():
        parts = video_path.strip().split("/")
        if len(parts) >= 3:
            return parts[-2].strip()

    return get_ground_truth_text(sample)


def extract_media_path_from_content(content: Any) -> Optional[str]:
    """Return the first video/image path stored in a message content field.

    This preserves the JSON path string, e.g.
    kinetic600/land sailing/QszpApNTHuQ_000038_000048
    """
    if isinstance(content, dict):
        content_items = [content]
    elif isinstance(content, list):
        content_items = content
    else:
        return None

    # Prefer video over image because this test set is video action recognition.
    for wanted_type in ("video", "image"):
        for item in content_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != wanted_type:
                continue
            for key in (wanted_type, "path", "url", "image", "video"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def get_sample_video_path(sample: Dict[str, Any]) -> Optional[str]:
    for msg in sample.get("messages", []):
        media_path = extract_media_path_from_content(msg.get("content", ""))
        if media_path:
            return media_path
    return None


def get_prompt_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt: List[Dict[str, Any]] = []
    for msg in sample.get("messages", []):
        if msg.get("role") in ASSISTANT_ROLES:
            break
        prompt.append(msg)

    if not prompt:
        raise ValueError("No user prompt found before assistant answer.")
    return prompt


def normalize_text(text: str) -> str:
    s = str(text).lower().strip()
    s = s.replace("\n", " ")
    s = re.sub(r"[`*_#>\[\]{}()\"']", " ", s)
    s = re.sub(r"[。．.!?,;:，、；：]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    changed = True
    while changed:
        changed = False
        for pattern in PREFIX_PATTERNS:
            new_s = re.sub(pattern, "", s).strip()
            if new_s != s:
                s = new_s
                changed = True

    return re.sub(r"\s+", " ", s).strip()


def simple_tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    return text.split()


def build_label_maps(labels: Iterable[str]) -> Tuple[List[str], Dict[str, str]]:
    norm_to_display: Dict[str, str] = {}
    for label in labels:
        norm = normalize_text(label)
        if norm and norm not in norm_to_display:
            norm_to_display[norm] = str(label).strip()
    return sorted(norm_to_display.keys()), norm_to_display


def read_candidate_labels(
    path: Optional[str],
    fallback_samples: Sequence[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, str]]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
    else:
        labels = [get_sample_label(sample) for sample in fallback_samples]
    return build_label_maps(labels)


def map_prediction_to_label(
    text: str,
    known_labels: Sequence[str],
    unknown_label: str,
    fuzzy_threshold: float,
) -> str:
    norm = normalize_text(text)
    if not norm:
        return unknown_label

    label_set = set(known_labels)
    if norm in label_set:
        return norm

    # Substring match: prefer longest known label to avoid matching overly short labels.
    for label in sorted(known_labels, key=len, reverse=True):
        if re.search(rf"(^|\s){re.escape(label)}($|\s)", norm):
            return label

    if fuzzy_threshold < 1.0 and known_labels:
        match = difflib.get_close_matches(norm, list(known_labels), n=1, cutoff=fuzzy_threshold)
        if match:
            return match[0]

    return unknown_label


def make_video_metadata(
    sample: Dict[str, Any],
    detected_fps: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    meta = sample.get("video_metadata") or {}
    fps_override = meta.get("fps") or sample.get("fps")
    if not detected_fps:
        return None

    video_metadata: List[Dict[str, Any]] = []
    for item in detected_fps:
        fps = fps_override if fps_override is not None else item["fps"]
        video_metadata.append(
            {
                "fps": fps,
                "total_num_frames": item["total_num_frames"],
            }
        )
    return video_metadata


def prepare_encoded_like_ds_wrapper(
    encoded: Dict[str, torch.Tensor],
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    """Mirror ds_wrapper._build_sample tensor-key handling, but for prompt-only generation."""
    output: Dict[str, torch.Tensor] = {}
    length = max_seq_length

    output["input_ids"] = encoded["input_ids"][:, :length].long()
    output["attention_mask"] = encoded["attention_mask"][:, :length].long()

    if "pixel_values" in encoded:
        output["pixel_values"] = encoded["pixel_values"]

    if "pixel_values_videos" in encoded:
        pixel_values = encoded["pixel_values_videos"]
        if pixel_values.dim() > 3:
            num_patches, hidden_dim = pixel_values.shape[-2:]
            pixel_values = pixel_values.reshape(-1, num_patches, hidden_dim)
        output["pixel_values"] = pixel_values

    if "image_position_ids" in encoded:
        output["image_position_ids"] = encoded["image_position_ids"]

    if "video_position_ids" in encoded:
        video_pos = encoded["video_position_ids"]
        if video_pos.dim() > 3:
            num_patches, two = video_pos.shape[-2:]
            video_pos = video_pos.reshape(-1, num_patches, two)
        output["image_position_ids"] = video_pos

    if "mm_token_type_ids" in encoded:
        output["mm_token_type_ids"] = encoded["mm_token_type_ids"][:, :length].long()

    return output


def build_prompt_inputs(
    dataset: SupervisedDataset,
    processor: Any,
    sample: Dict[str, Any],
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    prompt_messages = get_prompt_messages(sample)

    # Reuse the exact image/video loading logic in ds_wrapper.py.
    normalized_messages, detected_fps = dataset._normalize_messages(prompt_messages)
    video_metadata = make_video_metadata(sample, detected_fps)
    processor_kwargs = (
        {"videos_kwargs": {"video_metadata": video_metadata}} if video_metadata else None
    )

    encoded = processor.apply_chat_template(
        normalized_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
        processor_kwargs=processor_kwargs,
    )

    return prepare_encoded_like_ds_wrapper(encoded, max_seq_length=max_seq_length)


def decode_generated_texts(tokenizer: Any, sequences: torch.Tensor, prompt_len: int) -> List[str]:
    generated_only = sequences[:, prompt_len:]
    return tokenizer.batch_decode(generated_only, skip_special_tokens=True)


def generate_for_sample(
    model: torch.nn.Module,
    processor: Any,
    dataset: SupervisedDataset,
    sample: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    model_dtype: torch.dtype,
    known_labels: Sequence[str],
) -> Tuple[str, List[str], str]:
    prompt_inputs = build_prompt_inputs(
        dataset=dataset,
        processor=processor,
        sample=sample,
        max_seq_length=args.max_seq_length,
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]
    prompt_inputs = move_batch_to_device(prompt_inputs, device=device, model_dtype=model_dtype)

    num_beams = max(1, int(args.num_beams))

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": num_beams,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    if num_beams > 1:
        generation_kwargs["early_stopping"] = True

    with torch.no_grad():
        sequences = model.generate(
            **prompt_inputs,
            **generation_kwargs,
        )

    raw_outputs = decode_generated_texts(
        processor.tokenizer,
        sequences=sequences,
        prompt_len=prompt_len,
    )

    raw_top1 = raw_outputs[0] if raw_outputs else ""
    pred_top1 = map_prediction_to_label(
        text=raw_top1,
        known_labels=known_labels,
        unknown_label=args.unknown_label,
        fuzzy_threshold=args.fuzzy_threshold,
    )
    caption_prediction = normalize_text(raw_top1) if raw_top1 else ""

    return pred_top1, raw_outputs, caption_prediction


def compute_test_loss(
    model: torch.nn.Module,
    dataset: SupervisedDataset,
    processor: Any,
    args: argparse.Namespace,
    device: torch.device,
    model_dtype: torch.dtype,
) -> Optional[float]:
    if args.skip_loss:
        return None

    sample_count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    subset = torch.utils.data.Subset(dataset, list(range(sample_count)))
    collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id
    )
    loader = DataLoader(
        subset,
        batch_size=args.loss_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    total_loss_times_tokens = 0.0
    total_tokens = 0

    for batch in tqdm(loader, desc="Computing test_loss"):
        batch = move_batch_to_device(batch, device=device, model_dtype=model_dtype)
        token_count = int((batch["labels"] != IGNORE_INDEX).sum().item())
        if token_count == 0:
            continue

        with torch.no_grad():
            outputs = model(**batch)

        if outputs.loss is None or not torch.isfinite(outputs.loss):
            print("[WARN] Non-finite or missing loss detected; skipping this batch.")
            continue

        total_loss_times_tokens += float(outputs.loss.item()) * token_count
        total_tokens += token_count

    if total_tokens == 0:
        return None
    return total_loss_times_tokens / total_tokens


def compute_classification_metrics(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    total = len(records)
    top1_correct = sum(1 for record in records if record["gt"] == record["pred_top1"])

    gt_classes = sorted(set(record["gt"] for record in records))
    true_positive = Counter()
    false_positive = Counter()
    false_negative = Counter()
    support = Counter()

    for record in records:
        gt = record["gt"]
        pred = record["pred_top1"]
        support[gt] += 1

        if pred == gt:
            true_positive[gt] += 1
        else:
            false_negative[gt] += 1
            if pred in gt_classes:
                false_positive[pred] += 1

    per_class: Dict[str, Dict[str, Any]] = {}
    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []

    for label in gt_classes:
        tp = true_positive[label]
        fp = false_positive[label]
        fn = false_negative[label]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / support[label] if support[label] > 0 else 0.0

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

        per_class[label] = {
            "support": int(support[label]),
            "correct": int(tp),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "num_samples": total,
        "top1_accuracy": top1_correct / total if total else 0.0,
        "macro_precision": sum(precision_values) / len(precision_values) if precision_values else 0.0,
        "macro_recall": sum(recall_values) / len(recall_values) if recall_values else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
    }


def ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1)))


def compute_bleu(
    predictions: Sequence[str],
    references: Sequence[str],
    max_order: int,
    smooth: bool = True,
) -> float:
    """Corpus BLEU with simple add-one smoothing for n>=2.

    For short action labels, unsmoothed BLEU-4 is often zero. The smoothing here
    makes BLEU-4 less brittle but should still be treated as an auxiliary metric.
    """
    matches_by_order = [0 for _ in range(max_order)]
    possible_matches_by_order = [0 for _ in range(max_order)]
    pred_length = 0
    ref_length = 0

    for pred_text, ref_text in zip(predictions, references):
        pred_tokens = simple_tokenize(pred_text)
        ref_tokens = simple_tokenize(ref_text)
        pred_length += len(pred_tokens)
        ref_length += len(ref_tokens)

        for order in range(1, max_order + 1):
            pred_ngram_counts = ngram_counts(pred_tokens, order)
            ref_ngram_counts = ngram_counts(ref_tokens, order)
            overlap = pred_ngram_counts & ref_ngram_counts
            matches_by_order[order - 1] += sum(overlap.values())
            possible_matches_by_order[order - 1] += max(0, len(pred_tokens) - order + 1)

    precisions: List[float] = []
    for i in range(max_order):
        if possible_matches_by_order[i] == 0:
            precisions.append(0.0)
        elif smooth and i > 0:
            precisions.append((matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0))
        else:
            precisions.append(matches_by_order[i] / possible_matches_by_order[i])

    if min(precisions) <= 0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)

    if pred_length == 0:
        return 0.0

    ratio = pred_length / ref_length if ref_length > 0 else 0.0
    brevity_penalty = 1.0 if ratio > 1.0 else math.exp(1.0 - 1.0 / ratio) if ratio > 0 else 0.0
    return brevity_penalty * geo_mean


def lcs_length(x: Sequence[str], y: Sequence[str]) -> int:
    if not x or not y:
        return 0

    # Dynamic programming with two rows to save memory.
    prev = [0] * (len(y) + 1)
    curr = [0] * (len(y) + 1)

    for x_token in x:
        for j, y_token in enumerate(y, start=1):
            if x_token == y_token:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (len(y) + 1)

    return prev[-1]


def compute_rouge_l_f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    scores: List[float] = []
    for pred_text, ref_text in zip(predictions, references):
        pred_tokens = simple_tokenize(pred_text)
        ref_tokens = simple_tokenize(ref_text)
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        lcs = lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)

    return sum(scores) / len(scores) if scores else 0.0


def compute_cider_style(predictions: Sequence[str], references: Sequence[str], max_order: int = 4) -> float:
    """Lightweight CIDEr-style metric using TF-IDF n-gram cosine similarity.

    Official CIDEr usually expects multiple references per prediction and uses
    implementation details from COCO caption evaluation. For action labels with
    one reference each, this simplified version is more practical and dependency-free.

    Score range is roughly 0 to 10, where higher is better.
    """
    tokenized_preds = [simple_tokenize(text) for text in predictions]
    tokenized_refs = [simple_tokenize(text) for text in references]
    num_docs = len(tokenized_refs)
    if num_docs == 0:
        return 0.0

    total_scores: List[float] = []

    for pred_tokens, ref_tokens in zip(tokenized_preds, tokenized_refs):
        order_scores: List[float] = []

        for order in range(1, max_order + 1):
            # Document frequency from all references.
            document_frequency: Counter = Counter()
            for ref in tokenized_refs:
                document_frequency.update(set(ngram_counts(ref, order).keys()))

            pred_counts = ngram_counts(pred_tokens, order)
            ref_counts = ngram_counts(ref_tokens, order)

            if not pred_counts or not ref_counts:
                order_scores.append(0.0)
                continue

            def tfidf_vector(counts: Counter) -> Dict[Tuple[str, ...], float]:
                total = sum(counts.values())
                vector: Dict[Tuple[str, ...], float] = {}
                for ngram, count in counts.items():
                    tf = count / total if total > 0 else 0.0
                    # Smooth IDF to avoid zeroing n-grams that appear in every reference.
                    df = document_frequency.get(ngram, 0)
                    idf = math.log((num_docs + 1.0) / (df + 1.0)) + 1.0
                    vector[ngram] = tf * idf
                return vector

            pred_vec = tfidf_vector(pred_counts)
            ref_vec = tfidf_vector(ref_counts)
            keys = set(pred_vec) | set(ref_vec)

            dot = sum(pred_vec.get(key, 0.0) * ref_vec.get(key, 0.0) for key in keys)
            pred_norm = math.sqrt(sum(value * value for value in pred_vec.values()))
            ref_norm = math.sqrt(sum(value * value for value in ref_vec.values()))

            if pred_norm == 0.0 or ref_norm == 0.0:
                order_scores.append(0.0)
            else:
                order_scores.append(dot / (pred_norm * ref_norm))

        total_scores.append(10.0 * sum(order_scores) / max_order)

    return sum(total_scores) / len(total_scores) if total_scores else 0.0


def compute_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Optional[float]]:
    if args.skip_bertscore:
        return {
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
            "bertscore_model_type": args.bertscore_model_type,
            "bertscore_status": "skipped",
        }

    try:
        from bert_score import score as bertscore_score
    except Exception as err:
        return {
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
            "bertscore_model_type": args.bertscore_model_type,
            "bertscore_status": f"bert_score import failed: {err}",
        }

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        precision, recall, f1 = bertscore_score(
            list(predictions),
            list(references),
            model_type=args.bertscore_model_type,
            lang="en",
            device=device,
            batch_size=args.bertscore_batch_size,
            verbose=True,
        )
        return {
            "bertscore_precision": float(precision.mean().item()),
            "bertscore_recall": float(recall.mean().item()),
            "bertscore_f1": float(f1.mean().item()),
            "bertscore_model_type": args.bertscore_model_type,
            "bertscore_status": "ok",
        }
    except Exception as err:
        return {
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
            "bertscore_model_type": args.bertscore_model_type,
            "bertscore_status": f"failed: {err}",
        }


def compute_caption_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "bleu1": compute_bleu(predictions, references, max_order=1, smooth=False),
        "bleu4": compute_bleu(predictions, references, max_order=4, smooth=True),
        "rouge_l": compute_rouge_l_f1(predictions, references),
        "cider": compute_cider_style(predictions, references),
    }
    metrics.update(compute_bertscore(predictions, references, args))
    return metrics


def save_predictions(records: Sequence[Dict[str, Any]], path: Path) -> None:
    # Keep JSONL compact: one JSON object per line, convenient for programmatic reading.
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_predictions_pretty(records: Sequence[Dict[str, Any]], path: Path) -> None:
    # Human-readable version: each field is on its own line.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(records), f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_per_class_csv(per_class: Dict[str, Dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "support", "correct", "accuracy", "precision", "recall", "f1"])
        for label, metrics in sorted(per_class.items(), key=lambda item: (item[1]["accuracy"], item[0])):
            writer.writerow(
                [
                    label,
                    metrics["support"],
                    metrics["correct"],
                    f"{metrics['accuracy']:.8f}",
                    f"{metrics['precision']:.8f}",
                    f"{metrics['recall']:.8f}",
                    f"{metrics['f1']:.8f}",
                ]
            )


def save_confusion_matrix(
    records: Sequence[Dict[str, Any]],
    path: Path,
    known_labels: Sequence[str],
    unknown_label: str,
) -> None:
    gt_labels = sorted(set(record["gt"] for record in records))
    pred_labels = sorted(set(record["pred_top1"] for record in records))
    labels = sorted(set(gt_labels) | set(pred_labels) | set(known_labels))

    if unknown_label in labels:
        labels = [label for label in labels if label != unknown_label] + [unknown_label]

    matrix: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        matrix[record["gt"]][record["pred_top1"]] += 1

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt/pred"] + labels)
        for gt in labels:
            writer.writerow([gt] + [matrix[gt][pred] for pred in labels])


def dump_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_dir)
    model_result_dir = get_model_result_dir_name(args.model_id)
    output_dir = output_root / model_result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model from: {args.model_id}")
    print(f"[INFO] Output root : {output_root}")
    print(f"[INFO] Output dir  : {output_dir}")
    model, processor = load_model_and_processor(args)

    device = get_input_device(model)
    model_dtype = get_model_dtype(model)
    print(f"[INFO] Input device: {device}")
    print(f"[INFO] Model dtype : {model_dtype}")

    print(f"[INFO] Loading test dataset: {args.data_path}")
    data_module = make_data_module(
        processor=processor,
        data_path=args.data_path,
        image_folder=args.image_folder,
        max_seq_length=args.max_seq_length,
        max_decode_frames=args.max_decode_frames,
    )

    # Your current ds_wrapper.make_data_module returns train_dataset=dataset, eval_dataset=None.
    dataset = data_module.get("eval_dataset") or data_module.get("train_dataset")
    if dataset is None:
        raise RuntimeError("make_data_module returned neither eval_dataset nor train_dataset.")
    if not isinstance(dataset, SupervisedDataset):
        print(f"[WARN] Dataset type is {type(dataset)}, expected SupervisedDataset.")

    eval_count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    raw_samples = dataset.samples[:eval_count]

    known_labels, norm_to_display = read_candidate_labels(args.candidate_labels, raw_samples)
    if not known_labels:
        raise RuntimeError(
            "No labels found. Check assistant answers in test.json or provide --candidate_labels labels.txt."
        )

    print(f"[INFO] Samples evaluated: {eval_count}")
    print(f"[INFO] Known labels     : {len(known_labels)}")

    test_loss = compute_test_loss(
        model=model,
        dataset=dataset,
        processor=processor,
        args=args,
        device=device,
        model_dtype=model_dtype,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    records: List[Dict[str, Any]] = []
    caption_predictions: List[str] = []
    caption_references: List[str] = []

    for index, sample in enumerate(tqdm(raw_samples, desc="Generating predictions")):
        gt_raw = get_ground_truth_text(sample)
        video_path = get_sample_video_path(sample)
        label = get_sample_label(sample)
        gt = map_prediction_to_label(
            text=label,
            known_labels=known_labels,
            unknown_label=args.unknown_label,
            fuzzy_threshold=args.fuzzy_threshold,
        )

        pred_top1, raw_outputs, caption_prediction = generate_for_sample(
            model=model,
            processor=processor,
            dataset=dataset,
            sample=sample,
            args=args,
            device=device,
            model_dtype=model_dtype,
            known_labels=known_labels,
        )

        caption_reference = normalize_text(gt_raw)
        caption_predictions.append(caption_prediction)
        caption_references.append(caption_reference)

        record = {
            "index": index,
            "video": video_path,
            "label": label,
            "label_normalized": gt,
            "gt_raw": gt_raw,
            "gt": gt,
            "raw_outputs": raw_outputs,
            "caption_reference": caption_reference,
            "caption_prediction": caption_prediction,
            "pred_top1": pred_top1,
            "pred_top1_display": norm_to_display.get(pred_top1, pred_top1),
            "top1_correct": bool(gt == pred_top1),
        }
        records.append(record)

    classification_metrics = compute_classification_metrics(records)
    caption_metrics = compute_caption_metrics(caption_predictions, caption_references, args)

    metrics: Dict[str, Any] = {
        "model_id": args.model_id,
        "data_path": args.data_path,
        "image_folder": args.image_folder,
        "output_root": str(output_root),
        "output_dir": str(output_dir),
        "model_result_dir": model_result_dir,
        "max_seq_length": args.max_seq_length,
        "max_decode_frames": args.max_decode_frames,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "test_loss": test_loss,
        **classification_metrics,
        "caption_metrics": caption_metrics,
    }

    predictions_path = output_dir / "predictions.jsonl"
    predictions_pretty_path = output_dir / "predictions_pretty.json"
    metrics_path = output_dir / "metrics.json"
    caption_metrics_path = output_dir / "caption_metrics.json"
    per_class_path = output_dir / "per_class_metrics.csv"
    confusion_matrix_path = output_dir / "confusion_matrix.csv"

    save_predictions(records, predictions_path)
    save_predictions_pretty(records, predictions_pretty_path)
    dump_json(metrics, metrics_path)
    dump_json(caption_metrics, caption_metrics_path)
    save_per_class_csv(classification_metrics["per_class"], per_class_path)
    save_confusion_matrix(records, confusion_matrix_path, known_labels, args.unknown_label)

    print("\n========== TEST RESULTS ==========")
    print(f"num_samples      : {metrics['num_samples']}")
    print(f"test_loss        : {metrics['test_loss']}")
    print(f"top1_accuracy    : {metrics['top1_accuracy']:.6f}")
    print(f"macro_precision  : {metrics['macro_precision']:.6f}")
    print(f"macro_recall     : {metrics['macro_recall']:.6f}")
    print(f"macro_f1         : {metrics['macro_f1']:.6f}")
    print("---------- Caption metrics --------")
    print(f"CIDEr            : {caption_metrics['cider']:.6f}")
    print(f"ROUGE-L          : {caption_metrics['rouge_l']:.6f}")
    print(f"BLEU-1           : {caption_metrics['bleu1']:.6f}")
    print(f"BLEU-4           : {caption_metrics['bleu4']:.6f}")
    print(f"BERTScore P      : {caption_metrics['bertscore_precision']}")
    print(f"BERTScore R      : {caption_metrics['bertscore_recall']}")
    print(f"BERTScore F1     : {caption_metrics['bertscore_f1']}")
    print(f"BERTScore status : {caption_metrics['bertscore_status']}")
    print("==================================")
    print(f"Saved metrics           -> {metrics_path}")
    print(f"Saved caption metrics   -> {caption_metrics_path}")
    print(f"Saved predictions       -> {predictions_path}")
    print(f"Saved pretty predictions-> {predictions_pretty_path}")
    print(f"Saved per-class metrics -> {per_class_path}")
    print(f"Saved confusion matrix  -> {confusion_matrix_path}")


if __name__ == "__main__":
    main()