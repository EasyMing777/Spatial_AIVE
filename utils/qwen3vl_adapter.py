"""Qwen3-VL model adapter for local inference.

Provides a HuggingFace-transformers-based adapter for Qwen3-VL models, including
a compatibility patch for PyTorch 2.6 + transformers >= 4.57.
"""

import os
from typing import Any, Dict, List

import torch
from PIL import Image

from utils.ModelAdapter import BaseModelAdapter


class QwenVLLocalModelAdapter(BaseModelAdapter):
    """Adapter for locally-hosted Qwen3-VL models via HuggingFace transformers.

    Parameters
    ----------
    model_config : dict
        Must contain ``model_path`` pointing to a local Qwen3-VL checkpoint.
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        super().__init__(model_config)
        self.model_path: str = self.config.get("model_path", "")
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.model_path:
            raise ValueError("model_path must be provided for QwenVLLocalModelAdapter")

    # ------------------------------------------------------------------
    # Position-embedding compatibility patch
    # ------------------------------------------------------------------

    def _patch_fast_pos_embed(self) -> None:
        """Patch ``fast_pos_embed_interpolate`` and ``rot_pos_emb`` on the visual
        encoder to fix an incompatibility between transformers >= 4.57 and
        PyTorch 2.6 where tensor scalars are passed to functions expecting
        plain Python ints."""

        try:
            visual = self.model.visual

            def _patched_fast(grid_thw: torch.Tensor) -> torch.Tensor:
                # --- handle both tensor and list-of-lists input ---
                if isinstance(grid_thw, torch.Tensor):
                    g = grid_thw.int().tolist()
                    ts = [int(r[0]) for r in g]
                    hs = [int(r[1]) for r in g]
                    ws = [int(r[2]) for r in g]
                else:
                    ts = [int(r[0]) for r in grid_thw]
                    hs = [int(r[1]) for r in grid_thw]
                    ws = [int(r[2]) for r in grid_thw]

                idx_list: List[List[int]] = [[] for _ in range(4)]
                weight_list: List[List[float]] = [[] for _ in range(4)]

                for t, h, w in zip(ts, hs, ws):
                    h_idxs = torch.linspace(0, visual.num_grid_per_side - 1, h)
                    w_idxs = torch.linspace(0, visual.num_grid_per_side - 1, w)
                    hf = h_idxs.int()
                    wf = w_idxs.int()
                    hc = (hf + 1).clip(max=visual.num_grid_per_side - 1)
                    wc = (wf + 1).clip(max=visual.num_grid_per_side - 1)
                    dh = h_idxs - hf
                    dw = w_idxs - wf
                    bh = hf * visual.num_grid_per_side
                    bh_c = hc * visual.num_grid_per_side
                    indices = [
                        (bh[None].T + wf[None]).flatten(),
                        (bh[None].T + wc[None]).flatten(),
                        (bh_c[None].T + wf[None]).flatten(),
                        (bh_c[None].T + wc[None]).flatten(),
                    ]
                    weights = [
                        ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                        ((1 - dh)[None].T * dw[None]).flatten(),
                        (dh[None].T * (1 - dw)[None]).flatten(),
                        (dh[None].T * dw[None]).flatten(),
                    ]
                    for i in range(4):
                        idx_list[i].extend(indices[i].tolist())
                        weight_list[i].extend(weights[i].tolist())

                device = visual.pos_embed.weight.device
                dtype = visual.pos_embed.weight.dtype
                idx_t = torch.tensor(idx_list, dtype=torch.long, device=device)
                wt_t = torch.tensor(weight_list, dtype=dtype, device=device)
                pe = visual.pos_embed(idx_t) * wt_t[:, :, None]
                pp = (pe[0] + pe[1] + pe[2] + pe[3]).split(
                    [h * w for h, w in zip(hs, ws)]
                )

                merge_size = visual.config.spatial_merge_size
                out: List[torch.Tensor] = []
                for pe_i, t, h, w in zip(pp, ts, hs, ws):
                    pe_i = pe_i.repeat(t, 1)
                    pe_i = pe_i.view(
                        t, h // merge_size, merge_size,
                        w // merge_size, merge_size, -1,
                    )
                    pe_i = pe_i.permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
                    out.append(pe_i)
                return torch.cat(out)

            visual.fast_pos_embed_interpolate = _patched_fast

            def _patched_rot(grid_thw: torch.Tensor) -> torch.Tensor:
                merge_size = visual.config.spatial_merge_size
                max_hw = int(grid_thw[:, 1:].max().item())
                freq_table = visual.rotary_pos_emb(max_hw)
                device = freq_table.device
                total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
                pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

                offset = 0
                for num_frames, height, width in grid_thw.int().tolist():
                    merged_h = height // merge_size
                    merged_w = width // merge_size
                    block_rows = torch.arange(merged_h, device=device)
                    block_cols = torch.arange(merged_w, device=device)
                    intra_row = torch.arange(merge_size, device=device)
                    intra_col = torch.arange(merge_size, device=device)
                    row_idx = (
                        block_rows[:, None, None, None] * merge_size
                        + intra_row[None, None, :, None]
                    )
                    col_idx = (
                        block_cols[None, :, None, None] * merge_size
                        + intra_col[None, None, None, :]
                    )
                    row_idx = row_idx.expand(
                        merged_h, merged_w, merge_size, merge_size
                    ).reshape(-1)
                    col_idx = col_idx.expand(
                        merged_h, merged_w, merge_size, merge_size
                    ).reshape(-1)
                    coords = torch.stack((row_idx, col_idx), dim=-1)
                    if num_frames > 1:
                        coords = coords.repeat(num_frames, 1)
                    n = coords.shape[0]
                    pos_ids[offset: offset + n] = coords
                    offset += n

                return freq_table[pos_ids].flatten(1)

            visual.rot_pos_emb = _patched_rot
            print("[INFO] Patched Qwen3-VL visual encoder for transformers/PyTorch compatibility.")

        except AttributeError:
            print("[WARN] self.model.visual not found — patch skipped.")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_model(self, **kwargs: Any) -> None:
        """Load model + processor from a local checkpoint."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")

        print(f"[INFO] Loading Qwen3-VL from {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        ).eval()

        self._patch_fast_pos_embed()
        self.generation_config: Dict[str, Any] = dict(max_new_tokens=1024, do_sample=False)
        print(f"[INFO] Qwen3-VL loaded on {self.model.device}")

    # ------------------------------------------------------------------
    # Prepare inputs
    # ------------------------------------------------------------------

    def prepare_inputs(self, sys_prompt: str, content: List[tuple]) -> Dict[str, Any]:
        """Build Qwen3-VL chat-template inputs."""
        user_content: List[Dict[str, Any]] = []
        for item in content:
            if not isinstance(item, tuple):
                continue
            text = item[0] if item[0] else ""
            if len(item) > 1 and isinstance(item[1], str) and os.path.exists(item[1]):
                user_content.append({"type": "image", "image": Image.open(item[1]).convert("RGB")})
            if text and text.strip():
                user_content.append({"type": "text", "text": text})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": user_content},
        ]
        return {"messages": messages}

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, inputs: Dict[str, Any], **kwargs: Any) -> str:
        """Run inference with the Qwen3-VL model."""
        try:
            messages = inputs["messages"]
            max_new_tokens = kwargs.get("max_new_tokens", 1024)
            do_sample = kwargs.get("do_sample", False)
            temperature = kwargs.get("temperature", 0.0)

            pil_images = [
                block["image"]
                for msg in messages
                for block in msg.get("content", [])
                if isinstance(block, dict) and block.get("type") == "image"
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            proc_kwargs: Dict[str, Any] = dict(text=[text], return_tensors="pt", padding=True)
            if pil_images:
                proc_kwargs["images"] = pil_images
            processed = self.processor(**proc_kwargs)

            device = getattr(self.model, "device", torch.device("cpu"))
            for k, v in processed.items():
                if k in ("input_ids", "attention_mask", "image_grid_thw"):
                    processed[k] = v.to(device)
                else:
                    processed[k] = v.to(device, dtype=torch.bfloat16)

            with torch.no_grad():
                output = self.model.generate(
                    **processed,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                )

            input_len = processed["input_ids"].shape[1]
            return self.processor.decode(
                output[0][input_len:], skip_special_tokens=True
            ).strip()

        except Exception as e:
            print(f"[ERROR] Qwen3-VL local generation failed: {e}")
            return ""
