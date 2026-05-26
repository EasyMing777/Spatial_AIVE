
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ===========================================================================
# Environment variable keys
# ===========================================================================

class Env:
    """Canonical environment-variable names used throughout the project."""

    # OpenAI-compatible API
    OPENAI_API_KEY = "OPENAI_API_KEY"
    OPENAI_BASE_URL = "OPENAI_BASE_URL"
    _OPENAI_API_KEY = "_OPENAI_API_KEY"  # fallback key

    # Google Gemini API
    GOOGLE_API_KEY = "GOOGLE_API_KEY"

    # Azure (legacy)
    AZURE_OPENAI_API_KEY = "AZURE_OPENAI_API_KEY"


# ===========================================================================
# Model registries
# ===========================================================================

API_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gpt-4o": {
        "type": "openai",
        "model_name": "gpt-4o",
        "vision": True,
    },
    "gpt-4.1": {
        "type": "openai",
        "model_name": "gpt-4.1",
        "vision": True,
    },
    "gpt-5": {
        "type": "openai",
        "model_name": "gpt-5",
        "vision": True,
    },
    "gpt-5-mini": {
        "type": "openai",
        "model_name": "gpt-5-mini",
        "vision": True,
    },
    "gpt-5.5": {
        "type": "relay",
        "model_name": "gpt-5.5",
        "vision": True,
    },
    "gpt-5.5-turbo": {
        "type": "relay",
        "model_name": "gpt-5.5-turbo",
        "vision": True,
    },
    "o1": {
        "type": "openai",
        "model_name": "o1",
        "vision": False,
    },
    "o3": {
        "type": "openai",
        "model_name": "o3",
        "vision": True,
    },
    "o3-mini": {
        "type": "openai",
        "model_name": "o3-mini",
        "vision": False,
    },
    "o4-mini": {
        "type": "openai",
        "model_name": "o4-mini",
        "vision": True,
    },
    "gemini-1.5-flash": {
        "type": "google",
        "model_name": "gemini-1.5-flash",
        "vision": True,
    },
    "gemini-1.5-pro": {
        "type": "google",
        "model_name": "gemini-1.5-pro",
        "vision": True,
    },
    "qwen3-vl-30b-a3b-instruct": {
        "type": "openai",
        "model_name": "Qwen3-VL-30B-A3B-Instruct",
        "vision": True,
    },
    "qwen3-vl-30b-a3b-thinking": {
        "type": "openai",
        "model_name": "Qwen3-VL-30B-A3B-Thinking",
        "vision": True,
    },
    "qwen3-vl-235b-a22b-instruct": {
        "type": "openai",
        "model_name": "Qwen3-VL-235B-A22B-Instruct",
        "vision": True,
    },
    "qwen3-vl-235b-a22b-thinking": {
        "type": "openai",
        "model_name": "Qwen3-VL-235B-A22B-Thinking",
        "vision": True,
    },
}

LOCAL_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "internvl3-8b-hf": {
        "type": "local",
        "class": "InternVL3Model",
        "model_path": "<set_internvl3_8b_hf_path>",
        "adapter_path": "<set_internvl3_8b_sft_adapter_path>",
        "model_name": "InternVL3-8B-hf",
        "vision": True,
    },
    "internvl3-8b": {
        "type": "local",
        "class": "InternVL3Model",
        "model_path": "<set_internvl3_8b_path>",
        "model_name": "InternVL3-8B",
        "vision": True,
    },
    "internvl3-14b": {
        "type": "local",
        "class": "InternVL3Model",
        "model_path": "<set_internvl3_14b_path>",
        "model_name": "InternVL3-14B",
        "vision": True,
    },
    "qwen3-vl-8b-instruct": {
        "type": "local",
        "class": "QwenVLLocalModel",
        "model_path": "<set_qwen3_vl_8b_instruct_path>",
        "model_name": "Qwen3-VL-8B-Instruct",
        "vision": True,
    },
}

ALL_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {**API_MODEL_CONFIGS, **LOCAL_MODEL_CONFIGS}


# ===========================================================================
# Pipeline defaults
# ===========================================================================

@dataclass
class PipelineDefaults:

    # ---- paths ----
    output_dir: str = "./output"
    input_dir: str = "./data"

    # ---- action discretisation ----
    sampling_interval_meter: float = 0.25
    sampling_interval_angle: int = 9
    max_forward_distance: float = 1.0
    max_turn_angle: int = 90

    # ---- exploration budget ----
    max_steps_per_question: int = 5
    max_tries_gpt: int = 3
    max_images: int = 2

    # ---- VLM ----
    vlm_qa_model_name: str = "gpt-4o"
    vlm_ap_model_name: Optional[str] = None
    vlm_d_model_name: Optional[str] = None

    # ---- miscellaneous ----
    seed: Optional[int] = 42
    question_type: str = "None"
    force_explore: bool = False

    # ---- question filtering ----
    question_types: List[str] = field(default_factory=lambda: [
        "action_conseq",
        "ego_movement",
        "goal_aim",
        "obj_movement",
        "action_sequence",
        "action_consequence",
    ])
