"""InternVL3 model adapter for local inference.

Supports both the native InternVL3 format (``model.chat``) and the HuggingFace
transformers format with optional PEFT/LoRA adapter merging.
"""

import os
from typing import Any

import torch
from PIL import Image

from utils.ModelAdapter import BaseModelAdapter


class InternVL3ModelAdapter(BaseModelAdapter):
    """Adapter for locally-hosted InternVL3 models.

    Parameters
    ----------
    model_config : dict
        Must contain ``model_path``.  For the HuggingFace variant also include
        ``adapter_path`` pointing to a PEFT/LoRA checkpoint.
    """

    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__(model_config)
        self.model_path: str = self.config.get("model_path", "")
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name: str = model_config["model_name"]
        if not self.model_path:
            raise ValueError("model_path must be provided for InternVL3ModelAdapter")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_model(self, **kwargs: Any) -> None:
        """Load model weights, tokenizer, and (for HF) processor + PEFT adapter."""
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")

        is_hf_version = "hf" in self.model_path.lower()
        print(f"[INFO] Loading InternVL3 from {self.model_path} (hf={is_hf_version})")

        if is_hf_version:
            self._load_hf_variant()
        else:
            self._load_native_variant()

        self.generation_config: dict[str, Any] = dict(max_new_tokens=1024, do_sample=False)
        print(f"[INFO] InternVL3 loaded on {self.model.device}")

    def _load_hf_variant(self) -> None:
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        adapter_path = self.config.get("adapter_path")
        if not adapter_path:
            raise ValueError("adapter_path must be set in config for HF-format InternVL3")

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        self.tokenizer = self.processor.tokenizer

        base_model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path).merge_and_unload().eval()

    def _load_native_variant(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        ).eval()

        if torch.cuda.is_available():
            self.model = self.model.to("cuda:0")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False, local_files_only=True
        )
        self.processor = None

    # ------------------------------------------------------------------
    # Prepare inputs
    # ------------------------------------------------------------------

    def prepare_inputs(self, sys_prompt: str, content: list[tuple]) -> dict[str, Any]:
        """Build model-specific inputs from (text, image_path) tuples."""
        image_paths: list[str] = []
        full_text = ""

        for item in content:
            if isinstance(item, tuple):
                text = item[0] if item[0] else ""
                full_text += text
                if len(item) > 1 and isinstance(item[1], str) and os.path.exists(item[1]):
                    image_paths.append(item[1])

        if not image_paths:
            raise ValueError("No valid images found in content")

        if hasattr(self.model, "chat"):
            return self._prepare_for_native_chat(image_paths, sys_prompt, full_text)
        else:
            return self._prepare_for_hf(image_paths, sys_prompt, full_text)

    def _prepare_for_native_chat(
        self, image_paths: list[str], sys_prompt: str, full_text: str
    ) -> dict[str, Any]:
        from utils.InternVL3 import build_transform, dynamic_preprocess

        input_size = 448
        transform = build_transform(input_size=input_size)
        all_pixel_values: list[torch.Tensor] = []
        num_patches_list: list[int] = []

        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            blocks = dynamic_preprocess(
                image, image_size=input_size, use_thumbnail=True, max_num=12
            )
            pv = torch.stack([transform(b) for b in blocks])
            all_pixel_values.append(pv)
            num_patches_list.append(pv.size(0))

        pixel_values = torch.cat(all_pixel_values, dim=0).to(torch.bfloat16).to(self.device)
        image_tokens = "\n".join("<image>" * n for n in num_patches_list)
        full_prompt = f"{sys_prompt}\n{image_tokens}\n{full_text}"
        return {
            "pixel_values": pixel_values,
            "full_prompt": full_prompt,
            "num_patches_list": num_patches_list,
        }

    @staticmethod
    def _prepare_for_hf(image_paths: list[str], sys_prompt: str, full_text: str) -> dict[str, Any]:
        user_content: list[dict[str, Any]] = [
            {"type": "image", "image": Image.open(p).convert("RGB")} for p in image_paths
        ]
        user_content.append({"type": "text", "text": full_text})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": user_content},
        ]
        return {"messages": messages}

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, inputs: dict[str, Any], **kwargs: Any) -> str:
        """Run inference, dispatching to the appropriate internal pipeline."""
        try:
            if hasattr(self.model, "chat"):
                return self._generate_with_chat(inputs, **kwargs)
            return self._generate_without_chat(inputs, **kwargs)
        except Exception as e:
            print(f"[ERROR] InternVL3 generation failed: {e}")
            return ""

    def _generate_with_chat(self, inputs: dict[str, Any], **kwargs: Any) -> str:
        for key in ("pixel_values", "full_prompt", "num_patches_list"):
            if key not in inputs:
                raise ValueError(f"Missing required key '{key}' for InternVL3 chat generation")

        response, _ = self.model.chat(
            self.tokenizer,
            inputs["pixel_values"],
            inputs["full_prompt"],
            kwargs.get("generation_config", self.generation_config),
            num_patches_list=inputs["num_patches_list"],
            history=kwargs.get("history"),
            return_history=True,
        )
        return response

    def _generate_without_chat(self, inputs: dict[str, Any], **kwargs: Any) -> str:
        if "messages" not in inputs:
            raise ValueError("Missing required key 'messages' for InternVL3 standard generation")
        if not hasattr(self, "processor") or self.processor is None:
            raise AttributeError("Processor is required but not initialized")

        max_new_tokens = kwargs.get("max_new_tokens", 200)
        do_sample = kwargs.get("do_sample", False)
        temperature = kwargs.get("temperature", 0.0)

        processed = self.processor.apply_chat_template(
            inputs["messages"],
            padding=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        device = getattr(self.model, "device", torch.device("cpu"))
        for k, v in processed.items():
            if k in ("input_ids", "attention_mask"):
                processed[k] = v.to(device)
            else:
                processed[k] = v.to(device, dtype=torch.float16)

        with torch.no_grad():
            output = self.model.generate(
                **processed,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("max_new_tokens", "do_sample", "temperature")
                },
            )

        input_len = processed["input_ids"].shape[1]
        return self.processor.decode(output[0][input_len:], skip_special_tokens=True).strip()
