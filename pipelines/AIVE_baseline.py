"""Shared scaffolding for AIVE inference pipelines.

``PipelineBase`` wires together the four ingredients every AIVE run needs:

1. **CLI arguments** — parsed by :mod:`utils.args` and exposed as ``model_args``.
2. **Dataset** — SAT / SpaThor-1K questions, normalised into one internal format.
3. **Models** — the :class:`utils.ModelAdapter.Agent` (Checker / Planner /
   Answerer roles) and the :class:`dreamer.base.BaseDreamer` world model.
4. **Result bookkeeping** — accuracy tracking and ``results.json`` persistence.

Concrete pipelines (e.g. :class:`pipelines.AIVE.SpatialVQAPipelineAIVE`) subclass
this class and implement the exploration loop.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
from typing import Any

from dreamer import create_dreamer
from utils.args import parse_args
from utils.metrics import init_results
from utils.metrics import save_results as persist_results
from utils.ModelAdapter import Agent

# ---------------------------------------------------------------------------
# Dataset normalisation
# ---------------------------------------------------------------------------

#: Canonical internal question keys (independent of the raw dataset format).
_CANONICAL_QID = ("database_idx", "id", "qid")
_CANONICAL_QTYPE = ("question_type", "type", "category")
_CANONICAL_CHOICES = ("answer_choices", "choices", "answers")
_CANONICAL_ANSWER = ("correct_answer", "answer", "label")
_CANONICAL_IMAGES = ("img_paths", "images", "image_paths", "image_path", "image")


def _first_mapping(item: dict[str, Any], keys) -> str | None:
    """Return the value of the first key present in ``item`` (case-insensitive)."""
    lowered = {k.lower(): v for k, v in item.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None:
            return value
    return None


def _normalise_question(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a raw dataset record into the canonical internal format.

    Handles both the SAT layout used by ``utils/data_process.py`` and the
    upcoming SpaThor-1K layout by matching on several accepted field names.

    Parameters
    ----------
    raw : dict
        A raw QA record.

    Returns
    -------
    dict
        Normalised record with keys ``database_idx``, ``question_type``,
        ``question``, ``answer_choices``, ``correct_answer``, ``img_paths``.
    """
    qid = _first_mapping(raw, _CANONICAL_QID)
    if qid is None:
        raise ValueError(f"Question record missing an id field: {raw}")

    qtype = _first_mapping(raw, _CANONICAL_QTYPE) or "other"
    raw_choices = _first_mapping(raw, _CANONICAL_CHOICES)
    raw_images = _first_mapping(raw, _CANONICAL_IMAGES)
    answer = _first_mapping(raw, _CANONICAL_ANSWER)

    choices: list[str] = [raw_choices] if isinstance(raw_choices, str) else (raw_choices or [])
    images: list[str] = [raw_images] if isinstance(raw_images, str) else (raw_images or [])

    return {
        "database_idx": str(qid),
        "question_type": str(qtype),
        "question": str(raw.get("question", "")),
        "answer_choices": list(choices),
        "correct_answer": str(answer or ""),
        "img_paths": list(images),
    }


# ---------------------------------------------------------------------------
# Pipeline base
# ---------------------------------------------------------------------------


class PipelineBase:
    """Base class for AIVE inference pipelines.

    Subclasses must implement :meth:`run`. Helper methods for logging and
    result persistence are provided here.
    """

    #: Name used in logs and output file names.
    pipeline_name: str = "aive"

    def __init__(self) -> None:
        # ---- arguments -------------------------------------------------
        self.model_args: argparse.Namespace = parse_args()
        self.model_args.seed = int(self.model_args.seed or 42)

        if self.model_args.num_questions is not None and self.model_args.num_questions < -1:
            raise ValueError("--num_questions must be -1 or a non-negative integer.")

        # ---- reproducibility -------------------------------------------
        random.seed(self.model_args.seed)

        # ---- output directory -------------------------------------------
        os.makedirs(self.model_args.output_dir, exist_ok=True)

        # ---- dataset -----------------------------------------------------
        self.questions: list[dict[str, Any]] = self._load_dataset()

        # ---- models ------------------------------------------------------
        self.vlm: Agent = self._build_agent()
        self.dreamer = create_dreamer(
            dreamer_type=self.model_args.dreamer_type,
            mode=getattr(self.model_args, "mock_dreamer_mode", "identity"),
        )

        # ---- results -----------------------------------------------------
        question_types = list(self.model_args.question_types)
        for q in self.questions:
            if q["question_type"] not in question_types:
                question_types.append(q["question_type"])
        self.results: dict[str, Any] = init_results(question_types)

        # ---- logging -----------------------------------------------------
        self._log_file_handle: Any | None = None
        self._log_file_path: str | None = None

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def _dataset_path(self) -> str:
        """Path to the ``{split}.json`` question file."""
        return os.path.join(self.model_args.input_dir, f"{self.model_args.split}.json")

    def _load_dataset(self) -> list[dict[str, Any]]:
        """Load and normalise the question dataset.

        The raw file is a list of records in either SAT or SpaThor-1K layout;
        each is normalised via :func:`_normalise_question`. An optional
        ``--num_questions`` cap and ``--question_type`` filter are applied here.

        Returns
        -------
        list of dict
            Normalised question records.
        """
        dataset_path = self._dataset_path()
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_path}. Run "
                f"`python utils/data_process.py --split {self.model_args.split}` "
                f"to download the SAT dataset first."
            )

        with open(dataset_path, encoding="utf-8") as f:
            raw_records = json.load(f)

        questions: list[dict[str, Any]] = []
        for raw in raw_records:
            try:
                question = _normalise_question(raw)
            except (ValueError, TypeError) as exc:
                print(f"[WARN] Skipping malformed record: {exc}")
                continue

            if (
                self.model_args.question_type != "None"
                and question["question_type"] != self.model_args.question_type
            ):
                continue

            questions.append(question)
            if (
                self.model_args.num_questions != -1
                and len(questions) >= self.model_args.num_questions
            ):
                break

        print(f"[INFO] Loaded {len(questions)} question(s) from {dataset_path}")
        return questions

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def _build_agent(self) -> Agent:
        """Build the three-role VLM agent (Answerer / Planner / Checker).

        The Planner and Checker fall back to the Answerer model unless a
        dedicated model is configured via ``--vlm_ap_model_name`` /
        ``--vlm_d_model_name``.
        """
        return Agent(
            qa_model_name=self.model_args.vlm_qa_model_name,
            ap_model_name=self.model_args.vlm_ap_model_name,
            d_model_name=self.model_args.vlm_d_model_name,
        )

    # ------------------------------------------------------------------
    # Result persistence
    # ------------------------------------------------------------------

    def save_results(self) -> None:
        """Persist the current ``results`` dict to ``output_dir/results.json``."""
        out_path = os.path.join(self.model_args.output_dir, "results.json")
        persist_results(self.results, out_path)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_print(self, message: str, level: str = "INFO") -> None:
        """Print a timestamped log message and persist it as JSONL."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = {"timestamp": timestamp, "level": level, "message": message}
        print(f"[{timestamp}] [{level}] {message}")

        if self._log_file_handle is None:
            log_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_file_path = os.path.join(
                self.model_args.output_dir, f"run_log_{log_start_time}.jsonl"
            )
            os.makedirs(os.path.dirname(self._log_file_path), exist_ok=True)
            # Handle is intentionally kept open across calls (not context-managed).
            assert self._log_file_path is not None
            self._log_file_handle = open(self._log_file_path, "a", encoding="utf-8")  # noqa: SIM115
        elif self._log_file_handle.closed:
            assert self._log_file_path is not None
            self._log_file_handle = open(self._log_file_path, "a", encoding="utf-8")  # noqa: SIM115

        try:
            self._log_file_handle.write(json.dumps(log_entry) + "\n")
            self._log_file_handle.flush()
        except ValueError:
            assert self._log_file_path is not None
            self._log_file_handle = open(self._log_file_path, "a", encoding="utf-8")  # noqa: SIM115
            self._log_file_handle.write(json.dumps(log_entry) + "\n")
            self._log_file_handle.flush()

    def _close_log(self) -> None:
        """Close the JSONL log file handle if open."""
        if self._log_file_handle and not self._log_file_handle.closed:
            self._log_file_handle.close()
            self._log_file_handle = None

    # ------------------------------------------------------------------
    # Abstract entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the pipeline. Must be implemented by subclasses."""
        raise NotImplementedError


if __name__ == "__main__":
    # `python pipelines/AIVE_baseline.py --help` prints the full CLI.
    from utils.args import build_parser

    parser = build_parser()
    parser.parse_args()
    parser.print_help()
