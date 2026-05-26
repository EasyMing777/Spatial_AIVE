
import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Answer processing
# ---------------------------------------------------------------------------

def classify_answer(response: str, question: Dict[str, Any]) -> str:
    """Classify a VLM response as ``"correct"``, ``"wrong"``, or
    ``"out of control"``.

    Parameters
    ----------
    response : str
        Raw VLM response text.
    question : dict
        SAT-format question dict with keys ``answer_choices`` and
        ``correct_answer``.

    Returns
    -------
    str
        One of ``"correct"``, ``"wrong"``, ``"out of control"``.
    """
    response_l = response.lower()
    try:
        if any(c.lower() in response_l for c in question["answer_choices"]):
            return (
                "correct"
                if question["correct_answer"].lower() in response_l.split("\n")[-1]
                else "wrong"
            )
    except Exception:
        pass
    return "out of control"


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

def compute_accuracy(
    progress: Dict[str, Dict[str, List[str]]],
    skip_indices: Optional[List[str]] = None,
    total_questions: int = 0,
) -> Dict[str, Any]:
    """Compute per-type and overall accuracy from a progress dictionary.

    Parameters
    ----------
    progress : dict
        Mapping ``{question_type: {"correct": [...], "wrong": [...]}}``.
    skip_indices : list, optional
        List of skipped question IDs.
    total_questions : int
        Total number of questions in the dataset.

    Returns
    -------
    dict
        With keys ``all`` (float), ``types`` (dict), and ``current`` (str).
    """
    correct_total = sum(len(progress[t]["correct"]) for t in progress)
    wrong_total = sum(len(progress[t]["wrong"]) for t in progress)
    total = correct_total + wrong_total

    accuracy: Dict[str, Any] = {
        "all": correct_total / total if total > 0 else 0.0,
        "types": {},
    }

    for t in progress:
        n = len(progress[t]["correct"]) + len(progress[t]["wrong"])
        accuracy["types"][t] = (
            len(progress[t]["correct"]) / n if n > 0 else None
        )

    n_skipped = len(skip_indices) if skip_indices else 0
    accuracy["current"] = f"{total + n_skipped} / {total_questions}"

    return accuracy


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def init_results(question_types: List[str]) -> Dict[str, Any]:
    """Create a fresh results dictionary.

    Parameters
    ----------
    question_types : list
        All expected question-type strings.

    Returns
    -------
    dict
        Initialised results with ``progress``, ``accuracy``, ``skip_indices``,
        and ``parsing_err_stats``.
    """
    return {
        "progress": {
            t: {"correct": [], "wrong": []} for t in question_types
        },
        "accuracy": {"all": 0.0, "types": {}, "current": "0 / 0"},
        "skip_indices": [],
        "parsing_err_stats": {
            "answer": 0,
            "answer_qid": [],
        },
    }


def load_results(path: str) -> Dict[str, Any]:
    """Load results from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_results(results: Dict[str, Any], path: str) -> None:
    """Persist results to a JSON file."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
