"""End-to-end test of the AIVE exploration loop (offline, no API access).

Uses a synthetic SAT-format dataset, a scripted DummyVLMAdapter, and the
MockDreamer so the full Checker -> Planner -> Dreamer -> Answerer loop is
exercised deterministically.
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest

from pipelines.AIVE import SpatialVQAPipelineAIVE
from pipelines.AIVE_baseline import PipelineBase
from tests.dummy_vlm import DummyVLMAdapter
from utils.ModelAdapter import Agent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sat_dataset(tmp_path):
    """A small SAT-format dataset with three questions."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    questions = []
    for i in range(3):
        img_path = f"image_{i}_0.png"
        cv2.imwrite(str(data_dir / img_path), np.full((64, 64, 3), 40 * (i + 1), dtype=np.uint8))
        questions.append(
            {
                "database_idx": str(i),
                "question_type": "ego_movement" if i % 2 == 0 else "action_conseq",
                "question": f"How many objects are ahead? (q{i})",
                "answer_choices": ["A. Two", "B. Four", "C. Six"],
                "correct_answer": "A. Two",
                "img_paths": [img_path],
            }
        )

    (data_dir / "val.json").write_text(json.dumps(questions, indent=2))
    return data_dir, tmp_path / "output"


def _run_pipeline(data_dir, output_dir, monkeypatch):
    """Construct and run the pipeline with a dummy VLM injected."""
    monkeypatch.setattr(
        PipelineBase,
        "_build_agent",
        lambda self: Agent(qa_model_adapter=DummyVLMAdapter()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aive",
            "--input_dir",
            str(data_dir),
            "--output_dir",
            str(output_dir),
            "--split",
            "val",
            "--num_questions",
            "3",
            "--max_steps_per_question",
            "3",
            "--dreamer_type",
            "mock",
            "--mock_dreamer_mode",
            "identity",
            "--seed",
            "42",
        ],
    )
    pipeline = SpatialVQAPipelineAIVE()
    pipeline.run()
    return pipeline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_answers_all_correctly(sat_dataset, monkeypatch):
    data_dir, output_dir = sat_dataset
    pipeline = _run_pipeline(data_dir, output_dir, monkeypatch)

    assert pipeline.results["accuracy"]["all"] == 1.0
    assert len(pipeline.results["progress"]["ego_movement"]["correct"]) == 2
    assert len(pipeline.results["progress"]["action_conseq"]["correct"]) == 1


def test_pipeline_writes_results_and_traces(sat_dataset, monkeypatch):
    data_dir, output_dir = sat_dataset
    _run_pipeline(data_dir, output_dir, monkeypatch)

    assert (output_dir / "results.json").exists()
    assert (output_dir / "final_log.json").exists()

    # Dreamer output for the first question
    imagined = output_dir / "0" / "step_1" / "imagined_step_0.png"
    assert imagined.exists(), f"expected Dreamer output at {imagined}"


def test_pipeline_skips_out_of_control_answers(sat_dataset, monkeypatch):
    data_dir, output_dir = sat_dataset

    class FailingAnswerer(DummyVLMAdapter):
        def generate(self, inputs, **kwargs):
            if "answer the question" in self._sys_prompt:
                return "gibberish"
            return super().generate(inputs, **kwargs)

    monkeypatch.setattr(
        PipelineBase,
        "_build_agent",
        lambda self: Agent(qa_model_adapter=FailingAnswerer()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aive",
            "--input_dir",
            str(data_dir),
            "--output_dir",
            str(output_dir),
            "--split",
            "val",
            "--num_questions",
            "3",
            "--seed",
            "42",
        ],
    )
    pipeline = SpatialVQAPipelineAIVE()
    pipeline.run()

    # retries fall back to 'wrong', never crash
    assert pipeline.results["accuracy"]["all"] == 0.0
    assert sum(len(v["wrong"]) for v in pipeline.results["progress"].values()) == 3


def test_continue_checker_sees_initial_view(sat_dataset, monkeypatch):
    """In-loop Checker must condition on the initial view i0 (Algorithm 1, L11)."""
    data_dir, output_dir = sat_dataset
    pipeline = _run_pipeline(data_dir, output_dir, monkeypatch)

    adapter = pipeline.vlm.qa_model_adapter
    continue_calls = [imgs for sp, imgs in adapter.calls if "continue exploring" in sp]
    assert continue_calls, "expected at least one in-loop Checker call"

    for imgs in continue_calls:
        # the initial view of each question is stored at <output>/<qid>/step_0/img_0.png
        assert any(p.endswith(os.path.join("step_0", "img_0.png")) for p in imgs), (
            f"in-loop Checker must see the initial view i0, got {imgs}"
        )


def test_missing_dataset_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aive",
            "--input_dir",
            str(tmp_path / "no_such_dir"),
            "--output_dir",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(FileNotFoundError):
        SpatialVQAPipelineAIVE()
