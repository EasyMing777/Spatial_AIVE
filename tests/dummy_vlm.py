"""A scripted VLM adapter for offline pipeline testing.

Returns canned responses based on the active system prompt, so the full AIVE
loop can be exercised without any API access.
"""

from typing import Any

from utils.ModelAdapter import BaseModelAdapter


class DummyVLMAdapter(BaseModelAdapter):
    """Deterministic VLM that never hits the network."""

    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        super().__init__(model_config or {})
        self._sys_prompt = ""

    def load_model(self, **kwargs: Any) -> None:
        pass

    def prepare_inputs(self, sys_prompt: str, content: list[tuple]) -> list[tuple]:
        self._sys_prompt = sys_prompt
        return content

    def generate(self, inputs: Any, **kwargs: Any) -> str:
        sp = self._sys_prompt
        if "plan the next sequence" in sp:  # Planner
            return "turn-left 9\nmove-forward 0.5"
        if "continue exploring" in sp:  # Checker (in-loop)
            return "STOP"
        if "determine if you need to move" in sp:  # Checker (initial)
            return "EXPLORE"
        if "answer the question" in sp:  # Answerer
            return "A. Two"
        return "ANSWER"
