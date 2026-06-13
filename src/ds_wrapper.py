"""
Dataset wrapper for Gemma 4 E4B action-recognition SFT.

Data format (messages JSON):
[
  {"video_metadata": {"fps": 25.0, "duration_sec": 8.3},
   "messages": [
    {"role": "user", "content": [
      {"type": "video", "video": "path/to/clip.mp4"},
      {"type": "text",  "text": "What action is performed?"}
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "riding a bicycle"}]}
  ]}
]
Legacy top-level fps/duration still accepted. Images also supported.
"""
import copy
import os
import pathlib
import sys
from typing import Dict, List, Optional
import torch
import ujson as json
import transformers
from torch.utils.data import Dataset
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from utils import _pad_sequence

IGNORE_INDEX = -100

class SupervisedDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            processor: transformers.ProcessorMixin,
            image_folder: str | None = None,
            max_seq_length: int = 2304,
            max_decode_frames: int = 8,
            ) -> None:
        self.processor = processor
        self.image_folder = image_folder
        self.max_seq_length = max_seq_length
        # Must be >= processor.video_processor.num_frames (default 32).
        # The processor re-samples from this pre-decoded array; passing fewer
        # frames than it expects raises ValueError.
        self.max_decode_frames = max_decode_frames
        with open(data_path, "r") as f:
            self.samples: List[dict] = json.load(f)

    def _resolve_path(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path

        # 原路徑存在
        if os.path.exists(path):
            return path

        # 原路徑不存在，但加 .mp4 後存在
        if os.path.exists(path + ".mp4"):
            return path + ".mp4"

        if self.image_folder:
            candidate = os.path.join(self.image_folder, path)

            # image_folder + 原路徑存在
            if os.path.exists(candidate):
                return candidate

            # image_folder + 原路徑 + .mp4 存在
            if os.path.exists(candidate + ".mp4"):
                return candidate + ".mp4"

        return path

    def _load_image(self, src) -> Image.Image:
        if isinstance(src, Image.Image):
            return src.convert("RGB")
        if isinstance(src, str):
            path = self._resolve_path(src)
            if path.startswith(("http://", "https://")):
                import requests
                from io import BytesIO
                resp = requests.get(path, timeout=15)
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content)).convert("RGB")
            return Image.open(path).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(src)}")

    def _load_video_as_array(self, src, num_frames: int = 32):
        """Decode with PyAV (bundled FFmpeg) -> (numpy [T,H,W,C] uint8, fps float, total frames int)."""
        import av, numpy as np

        path = self._resolve_path(src) if isinstance(src, str) else src

        # 如果 JSON 裡的影片路徑沒有 .mp4，但實際檔案有 .mp4，就自動補上
        if isinstance(path, str) and not os.path.exists(path):
            if os.path.exists(path + ".mp4"):
                path = path + ".mp4"

        container = av.open(path)
        stream = container.streams.video[0]

        fps = float(stream.average_rate) if stream.average_rate else 25.0

        all_frames = []
        for frame in container.decode(video=0):
            all_frames.append(frame.to_ndarray(format="rgb24"))

        container.close()

        if not all_frames:
            raise ValueError(f"Video has no frames: {path}")

        original_total_num_frames = len(all_frames)

        indices = torch.linspace(
                0,
                original_total_num_frames - 1,
                steps=min(num_frames, original_total_num_frames),
                ).long().tolist()

        sampled = np.stack([all_frames[i] for i in indices], axis=0)
        sampled_total_num_frames = sampled.shape[0]

        return sampled, fps, sampled_total_num_frames

    def _normalize_messages(self, messages: List[dict]):
        """Returns (normalized_messages, video_meta_list).
        video_meta_list has one entry per video content item encountered.
        """
        messages = copy.deepcopy(messages)
        fps_list: List[float] = []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]
            new_content = []

            for item in content:
                # 如果 content 裡面是純文字字串，轉成 Gemma processor 要的格式
                if isinstance(item, str):
                    new_content.append({
                        "type": "text",
                        "text": item,
                    })
                    continue

                if isinstance(item, dict):
                    if item.get("type") == "image":
                        for key in ("image", "path", "url"):
                            if key in item:
                                item = {**item, key: self._load_image(item[key])}
                                break
                    elif item.get("type") == "video":
                        converted = False
                        for key in ("video", "path", "url"):
                            if key in item:
                                frames, fps, total_num_frames = self._load_video_as_array(
                                    item[key], num_frames=self.max_decode_frames
                                )

                                # 把 video 拆成多張 image，讓 template 產生 image tokens
                                for frame in frames:
                                    new_content.append({
                                        "type": "image",
                                        "image": Image.fromarray(frame).convert("RGB"),
                                    })
                                
                                fps_list.append({
                                    "fps": fps,
                                    "total_num_frames": total_num_frames,
                                })
                                converted = True
                                break
                        if converted:
                            continue

                new_content.append(item)
            msg["content"] = new_content
        return messages, fps_list

    def _build_sample(
            self, messages: List[dict], fps_override: Optional[float] = None
            ) -> Dict[str, torch.Tensor]:
        processor = self.processor
        normalized, detected_fps = self._normalize_messages(messages)

        # Build video_metadata for Gemma 4 frame-timestamp computation.
        # fps_override (stored in JSON during preprocessing) takes priority over
        # the value detected live from the video stream.
        if detected_fps:
            video_metadata = []
            for meta in detected_fps:
                fps = fps_override if fps_override is not None else meta["fps"]
                video_metadata.append({
                    "fps": fps,
                    "total_num_frames": meta["total_num_frames"],
                })
        else:
            video_metadata = None

        processor_kwargs = (
            {"videos_kwargs": {"video_metadata": video_metadata}}
            if video_metadata
            else None
        )

        encoded = processor.apply_chat_template(
                normalized,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=False,
                enable_thinking=False,
                processor_kwargs=processor_kwargs,
        )
        input_ids = encoded["input_ids"].squeeze(0).long()
        attention_mask = encoded["attention_mask"].squeeze(0).long()
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        assistant_roles = {"assistant", "model"}
        for idx, msg in enumerate(normalized):
            if msg["role"] not in assistant_roles:
                continue
            if idx == 0:
                start_len = 0
            else:
                prefix = processor.apply_chat_template(
                        normalized[:idx],
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        add_generation_prompt=True,
                        enable_thinking=False,
                        processor_kwargs=processor_kwargs,
                        )
                start_len = prefix["input_ids"].size(1)
                # Guard: verify prefix tokens align with full input_ids.
                # Misalignment means the tokenizer is not prefix-stable across
                # add_generation_prompt, which would silently corrupt labels.
                prefix_ids = prefix["input_ids"].squeeze(0)
                if start_len > input_ids.size(0) or not torch.equal(
                        input_ids[:start_len], prefix_ids
                        ):
                    from utils import _log
                    _log(
                            f"WARNING: label span misalignment at turn {idx} "
                            f"(prefix_len={start_len}, seq_len={input_ids.size(0)}) "
                            "— labels skipped for this turn"
                            )
                    continue

            prefix_with_answer = processor.apply_chat_template(
                    normalized[:idx + 1],
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=False,
                    enable_thinking=False,
                    processor_kwargs=processor_kwargs,
                    )
            end_len = prefix_with_answer["input_ids"].size(1)
            labels[start_len:end_len] = input_ids[start_len:end_len]

        if labels.numel() > 0 and labels[0].item() != IGNORE_INDEX:
            labels[0] = IGNORE_INDEX

        # Enforce max_seq_length — truncate text tensors only.
        # Vision tensors (pixel_values*) are kept intact; their placeholder
        # tokens are inserted near the start of input_ids (user turn), so
        # truncating from the tail is safe.
        L = self.max_seq_length
        if input_ids.size(0) > L:
            input_ids = input_ids[:L]
            attention_mask = attention_mask[:L]
            labels = labels[:L]

        data = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                )
        if "pixel_values" in encoded:
            data["pixel_values"] = encoded["pixel_values"]

        if "pixel_values_videos" in encoded:
            pv = encoded["pixel_values_videos"]

            # Gemma4 processor output is usually:
            # [batch, frames, num_patches, hidden_dim]
            # Example from your debug: [1, 32, 630, 768]
            if pv.dim() > 3:
                num_patches, hidden_dim = pv.shape[-2:]
                pv = pv.reshape(-1, num_patches, hidden_dim)

            data["pixel_values"] = pv

        if "image_position_ids" in encoded:
            data["image_position_ids"] = encoded["image_position_ids"]

        if "video_position_ids" in encoded:
            vpos = encoded["video_position_ids"]
            
            # [batch, frames, num_patches, 2] -> [batch * frames, num_patches, 2]
            if vpos.dim() > 3:
                num_patches, two = vpos.shape[-2:]
                vpos = vpos.reshape(-1, num_patches, two)

            data["image_position_ids"] = vpos

        if "mm_token_type_ids" in encoded:
            mm = encoded["mm_token_type_ids"].squeeze(0).long()
            data["mm_token_type_ids"] = mm[:L]

        if not hasattr(self, "_printed_debug_shapes"):
            self._printed_debug_shapes = True
            print("[DEBUG] encoded keys:", list(encoded.keys()))
            for k, v in encoded.items():
                if hasattr(v, "shape"):
                    print(f"[DEBUG] encoded {k}: {tuple(v.shape)}")
            for k, v in data.items():
                if hasattr(v, "shape"):
                    print(f"[DEBUG] data {k}: {tuple(v.shape)}")

        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[i]
        # fps from new-format JSON (video_metadata.fps) with fallback to legacy top-level fps
        meta = sample.get("video_metadata") or {}
        fps = meta.get("fps") or sample.get("fps")
        return self._build_sample(sample["messages"], fps_override=fps)


class DataCollatorForSupervisedDataset:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [ex["input_ids"] for ex in examples]
        labels_list = [ex["labels"] for ex in examples]
        has_image = any("pixel_values" in ex for ex in examples)
        has_image_pos = any("image_position_ids" in ex for ex in examples)
        has_video = any("pixel_values_videos" in ex for ex in examples)
        has_video_pos = any("video_position_ids" in ex for ex in examples)

        input_ids = _pad_sequence(input_ids_list, padding_value=self.pad_token_id)
        labels = _pad_sequence(labels_list, padding_value=IGNORE_INDEX)
        attention_mask = (input_ids != self.pad_token_id).long()

        batch = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                )

        if has_image:
            batch["pixel_values"] = torch.cat(
                    [ex["pixel_values"] for ex in examples if "pixel_values" in ex], dim=0
                    )

        """if has_video:
            batch["pixel_values_videos"] = torch.cat(
                    [ex["pixel_values_videos"] for ex in examples if "pixel_values_videos" in ex], dim=0
                    )
        """

        if has_image_pos:
            batch["image_position_ids"] = torch.cat(
                    [ex["image_position_ids"] for ex in examples if "image_position_ids" in ex],
                    dim=0
                    )

        """if has_video_pos:
            # shape per sample: (num_videos, num_frames, max_patches, 2) → cat on dim=0
            batch["video_position_ids"] = torch.cat(
                    [ex["video_position_ids"] for ex in examples if "video_position_ids" in ex], dim=0
                    )
        """

        mm_token_type_ids_list = [
                ex.get("mm_token_type_ids", torch.zeros_like(ex["input_ids"])) for ex in examples
                ]
        batch["mm_token_type_ids"] = _pad_sequence(mm_token_type_ids_list, padding_value=0)

        """if "pixel_values_videos" in batch and "pixel_values" not in batch:
            batch["pixel_values"] = batch["pixel_values_videos"]
        """

        """if "video_position_ids" in batch and "image_position_ids" not in batch:
            batch["image_position_ids"] = batch["video_position_ids"]
        """

        return batch


def make_data_module(
        processor: transformers.ProcessorMixin,
        data_path: str,
        image_folder: str | None = None,
        max_seq_length: int = 2304,
        max_decode_frames: int = 8,
        ) -> dict:
    dataset = SupervisedDataset(
            data_path=data_path,
            processor=processor,
            image_folder=image_folder,
            max_seq_length=max_seq_length,
            max_decode_frames=max_decode_frames,
            )
    collator = DataCollatorForSupervisedDataset(
            pad_token_id=processor.tokenizer.pad_token_id
            )
    return dict(
            train_dataset=dataset,
            eval_dataset=None,
            data_collator=collator,
            )
