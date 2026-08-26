"""Model abstraction layer for VLM-based spatial reasoning.

Provides a unified interface (BaseModelAdapter) for API-based models (OpenAI,
Google Gemini) and local models (see internvl3_adapter / qwen3vl_adapter).
The Agent class wires model adapters to prompt formatters for the three
pipeline roles: question-answering, action-planning, and discrimination.
"""

import base64
import os
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from utils.config import ALL_MODEL_CONFIGS, API_MODEL_CONFIGS
from utils.prompt import (
    format_action_planing,
    format_answer_question,
    format_answer_question_exploration,
    format_discriminator,
)

# ===========================================================================
# Base adapter
# ===========================================================================


class BaseModelAdapter(ABC):
    """Abstract interface for a VLM model adapter."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        self.config = model_config
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None

    @abstractmethod
    def load_model(self, **kwargs: Any) -> None:
        """Load model weights and processors into memory."""
        ...

    @abstractmethod
    def prepare_inputs(self, sys_prompt: str, content: list[tuple]) -> Any:
        """Convert (sys_prompt, content) into model-specific inputs."""
        ...

    @abstractmethod
    def generate(self, inputs: Any, **kwargs: Any) -> str:
        """Run inference and return decoded text."""
        ...

    @staticmethod
    def _load_images(images: list[str]) -> list[Image.Image]:
        pil_images: list[Image.Image] = []
        for img_path in images:
            if os.path.exists(img_path):
                try:
                    pil_images.append(Image.open(img_path).convert("RGB"))
                except Exception as e:
                    print(f"[WARN] Cannot load image {img_path}: {e}")
                    raise
        return pil_images


# ===========================================================================
# Shared helpers
# ===========================================================================


def _build_openai_messages(sys_prompt: str, content: list[tuple]) -> list[dict[str, Any]]:
    """Convert (sys_prompt, content) tuples into OpenAI-format chat messages."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}]
    user_content: list[dict[str, Any]] = []

    for item in content:
        if len(item) == 2 and item[1] is not None:
            text_part, img_path = item
            if text_part and text_part.strip():
                user_content.append({"type": "text", "text": text_part})
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                    }
                )
        else:
            text_content = item[0] if item else ""
            if text_content and text_content.strip():
                user_content.append({"type": "text", "text": text_content})

    if user_content:
        messages.append({"role": "user", "content": user_content})
    return messages


# ===========================================================================
# API adapters (OpenAI / Google)
# ===========================================================================


class OpenAIModelAdapter(BaseModelAdapter):
    """Adapter for OpenAI-compatible chat-completion APIs."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__(model_config)
        try:
            from openai import OpenAI

            api_key = os.getenv("_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Neither _OPENAI_API_KEY nor OPENAI_API_KEY is set")

            base_url = self.config.get("base_url") or os.getenv("OPENAI_BASE_URL")
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model_name: str = self.config.get("model_name", "gpt-4o")
        except ImportError as exc:
            raise ImportError("openai package is required: pip install openai") from exc

    def load_model(self, **kwargs: Any) -> None:
        pass

    def prepare_inputs(self, sys_prompt: str, content: list[tuple]) -> list[dict[str, Any]]:
        return _build_openai_messages(sys_prompt, content)

    def generate(self, inputs: list[dict[str, Any]], **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=inputs,  # type: ignore[arg-type]  # OpenAI message types are overly strict
            max_tokens=kwargs.get("max_new_tokens", 1024),
            temperature=kwargs.get("temperature", 0.0),
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


class GoogleModelAdapter(BaseModelAdapter):
    """Adapter for Google Gemini API."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__(model_config)
        try:
            import google.generativeai as genai

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not set")
            genai.configure(api_key=api_key)
            self.genai = genai
            self.model_name: str = self.config.get("model_name", "gemini-1.5-flash")
        except ImportError as exc:
            raise ImportError(
                "google-generativeai package is required: pip install google-generativeai"
            ) from exc

    def load_model(self, **kwargs: Any) -> None:
        self.model = self.genai.GenerativeModel(self.model_name)

    def prepare_inputs(self, sys_prompt: str, content: list[tuple]) -> list[Any]:
        parts: list[Any] = [sys_prompt] if sys_prompt else []
        image_paths: list[str] = []
        texts: list[str] = []

        for item in content:
            text = item[0] if item else ""
            if text and text.strip():
                texts.append(text)
            if len(item) == 2 and item[1] is not None and os.path.exists(item[1]):
                image_paths.append(item[1])

        parts.extend(texts)
        parts.extend(self._load_images(image_paths))
        return parts

    def generate(self, inputs: list[Any], **kwargs: Any) -> str:
        return self.model.generate_content(inputs).text


# ===========================================================================
# Model registry
# ===========================================================================


class ModelRegistry:
    """Factory that resolves model-name strings to adapter instances."""

    @classmethod
    def get_adapter(cls, model_name: str) -> BaseModelAdapter:
        config: dict[str, Any] | None = None
        for key, cfg in ALL_MODEL_CONFIGS.items():
            if key.lower() == model_name.lower() or key.lower() in model_name.lower():
                config = cfg.copy()
                config["model_name"] = model_name
                break

        if config is None:
            name_lower = model_name.lower()
            if any(k in name_lower for k in ("gpt", "o1", "o3", "o4")):
                config = {"type": "openai", "model_name": model_name, "vision": True}
            elif "gemini" in name_lower:
                config = {"type": "google", "model_name": model_name, "vision": True}
            elif "qwen3" in name_lower:
                config = {"type": "openai", "model_name": model_name, "vision": True}
            elif "internvl3" in name_lower:
                config = {"type": "local", "class": "InternVL3Model", "vision": True}
            else:
                raise ValueError(f"Unknown model: {model_name}")

        if config["type"] == "relay":
            from utils.relay_adapter import RelayModelAdapter

            return RelayModelAdapter(config)
        elif config["type"] == "openai":
            return OpenAIModelAdapter(config)
        elif config["type"] == "google":
            return GoogleModelAdapter(config)
        elif config["type"] == "local":
            cls_name = config.get("class")
            if cls_name == "QwenVLLocalModel":
                from utils.qwen3vl_adapter import QwenVLLocalModelAdapter

                return QwenVLLocalModelAdapter(config)
            elif cls_name == "InternVL3Model":
                from utils.internvl3_adapter import InternVL3ModelAdapter

                return InternVL3ModelAdapter(config)
            else:
                raise ValueError(f"Unknown local model class: {cls_name}")
        else:
            raise ValueError(f"Unsupported model type: {config['type']}")


# ===========================================================================
# Public API
# ===========================================================================


def create_model_adapter(model_name: str, **kwargs: Any) -> BaseModelAdapter:
    """Create and load a model adapter by name."""
    adapter = ModelRegistry.get_adapter(model_name)
    adapter.load_model(**kwargs)
    return adapter


def list_available_models() -> dict[str, list[str]]:
    """Return available API and local model names."""
    from utils.config import LOCAL_MODEL_CONFIGS

    return {
        "api_models": list(API_MODEL_CONFIGS.keys()),
        "local_models": list(LOCAL_MODEL_CONFIGS.keys()),
    }


# ===========================================================================
# Agent
# ===========================================================================


class Agent:
    """Orchestrates VLM calls for the three pipeline roles.

    Roles:
        qa (question-answering): produces the final answer.
        ap (action-planning): proposes the next camera actions.
        d  (discrimination): gates whether to explore or stop.
    """

    def __init__(
        self,
        qa_model_name: str | None = None,
        qa_model_adapter: BaseModelAdapter | None = None,
        ap_model_name: str | None = None,
        ap_model_adapter: BaseModelAdapter | None = None,
        d_model_name: str | None = None,
        d_model_adapter: BaseModelAdapter | None = None,
        generation_config: dict[str, Any] | None = None,
        **adapter_kwargs: Any,
    ) -> None:
        self.qa_model_adapter = self._resolve_adapter(
            model_name=qa_model_name,
            model_adapter=qa_model_adapter,
            role="qa",
            **adapter_kwargs,
        )
        self.ap_model_adapter = self._resolve_adapter(
            model_name=ap_model_name,
            model_adapter=ap_model_adapter,
            fallback=self.qa_model_adapter,
            role="ap",
            **adapter_kwargs,
        )
        self.d_model_adapter = self._resolve_adapter(
            model_name=d_model_name,
            model_adapter=d_model_adapter,
            fallback=self.qa_model_adapter,
            role="d",
            **adapter_kwargs,
        )
        self.curr_prompt: Any = None

    @staticmethod
    def _resolve_adapter(
        model_name: str | None = None,
        model_adapter: BaseModelAdapter | None = None,
        fallback: BaseModelAdapter | None = None,
        role: str = "",
        **adapter_kwargs: Any,
    ) -> BaseModelAdapter:
        """Resolve a single model adapter from the given args, with fallback."""
        if model_name is not None:
            adapter = ModelRegistry.get_adapter(model_name)
            adapter.load_model(**adapter_kwargs)
            return adapter
        if model_adapter is not None:
            return model_adapter
        if fallback is not None:
            return fallback
        raise ValueError(f"Either model_name or model_adapter must be provided for role '{role}'")

    def _get_adapter(self, model_type: str) -> BaseModelAdapter:
        adapter = {
            "ap": self.ap_model_adapter,
            "qa": self.qa_model_adapter,
            "d": self.d_model_adapter,
        }.get(model_type)
        if adapter is None:
            raise ValueError(f"Invalid model_type '{model_type}'. Must be 'ap', 'qa', or 'd'.")
        return adapter

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_prompt(
        self,
        prompt_type: str,
        question: str,
        answer_choices: list[str],
        images: list[str],
        step_idx: int | None = None,
        exploration_history: list[dict[str, Any]] | None = None,
        sys_prompt: str | None = None,
    ) -> tuple[str, list[tuple]]:
        """Route to the appropriate prompt formatter."""
        kwargs: dict[str, Any] = dict(
            question=question,
            answer_choices=answer_choices,
            images=images,
            sys_prompt=sys_prompt,
        )
        if prompt_type == "answer_question":
            return format_answer_question(**kwargs)
        elif prompt_type == "action_planing":
            return format_action_planing(
                **kwargs,
                step_idx=step_idx or 0,
                exploration_history=exploration_history or [],
            )
        elif prompt_type == "discriminator":
            return format_discriminator(
                **kwargs,
                exploration_history=exploration_history,
                step_idx=step_idx,
            )
        elif prompt_type == "answer_question_exploration":
            return format_answer_question_exploration(
                **kwargs,
                exploration_history=exploration_history or [],
            )
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")

    # ------------------------------------------------------------------
    # Model runners
    # ------------------------------------------------------------------

    def run_model(
        self,
        prompt_type: str,
        question: str,
        answer_choices: list[str],
        images: list[str],
        model_type: str,
        step_idx: int | None = None,
        exploration_history: list[dict[str, Any]] | None = None,
        sys_prompt: str | None = None,
        **generation_kwargs: Any,
    ) -> str:
        """Run a single VLM call end-to-end (format → prepare → generate)."""
        adapter = self._get_adapter(model_type)
        sys_prompt_out, content = self.format_prompt(
            prompt_type=prompt_type,
            question=question,
            answer_choices=answer_choices,
            images=images,
            step_idx=step_idx,
            exploration_history=exploration_history,
            sys_prompt=sys_prompt,
        )
        try:
            inputs = adapter.prepare_inputs(sys_prompt=sys_prompt_out, content=content)
            return adapter.generate(inputs, **generation_kwargs)
        except Exception as e:
            print(f"[ERROR] {model_type} model generation failed: {e}")
            return ""

    def run_ap_model(
        self,
        prompt_type: str,
        question: str,
        answer_choices: list[str],
        images: list[str],
        step_idx: int,
        exploration_history: list[dict[str, Any]],
        sys_prompt: str | None = None,
        **generation_kwargs: Any,
    ) -> str:
        """Run action-planning VLM."""
        return self.run_model(
            prompt_type,
            question=question,
            answer_choices=answer_choices,
            images=images,
            model_type="ap",
            step_idx=step_idx,
            exploration_history=exploration_history,
            sys_prompt=sys_prompt,
            **generation_kwargs,
        )

    def run_qa_model(
        self,
        prompt_type: str,
        question: str,
        answer_choices: list[str],
        images: list[str],
        exploration_history: list[dict[str, Any]] | None = None,
        sys_prompt: str | None = None,
        **generation_kwargs: Any,
    ) -> str:
        """Run question-answering VLM."""
        return self.run_model(
            prompt_type,
            question=question,
            answer_choices=answer_choices,
            images=images,
            model_type="qa",
            exploration_history=exploration_history,
            sys_prompt=sys_prompt,
            **generation_kwargs,
        )

    def run_d_model(
        self,
        prompt_type: str,
        question: str,
        answer_choices: list[str],
        images: list[str],
        exploration_history: list[dict[str, Any]] | None = None,
        sys_prompt: str | None = None,
        step_idx: int | None = None,
        **generation_kwargs: Any,
    ) -> str:
        """Run discriminator VLM."""
        return self.run_model(
            prompt_type,
            question=question,
            answer_choices=answer_choices,
            images=images,
            model_type="d",
            exploration_history=exploration_history,
            sys_prompt=sys_prompt,
            step_idx=step_idx,
            **generation_kwargs,
        )
