import json
import os
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Colour palette (colourblind-friendly)
# ---------------------------------------------------------------------------

PALETTE = {
    "blue": "#377eb8",
    "orange": "#ff7f00",
    "green": "#4daf4a",
    "pink": "#f781bf",
    "purple": "#984ea3",
    "grey": "#999999",
}

QUESTION_TYPE_LABELS = {
    "action_conseq": "Action\nConsequence",
    "ego_movement": "Ego\nMovement",
    "goal_aim": "Goal\nAim",
    "obj_movement": "Object\nMovement",
    "action_sequence": "Action\nSequence",
    "action_consequence": "Action\nConseq.",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(results_path: str) -> dict[str, Any]:
    """Load a ``results.json`` file produced by the pipeline."""
    with open(results_path) as f:
        return json.load(f)


def load_run_logs(output_dir: str) -> list[dict[str, Any]]:
    """Load all JSONL log entries from a pipeline run."""
    logs: list[dict[str, Any]] = []
    for fname in sorted(os.listdir(output_dir)):
        if fname.startswith("run_log_") and fname.endswith(".jsonl"):
            with open(os.path.join(output_dir, fname)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        logs.append(json.loads(line))
    return logs


# ---------------------------------------------------------------------------
# Accuracy plotting
# ---------------------------------------------------------------------------


def plot_per_type_accuracy(
    results: dict[str, Any],
    save_path: str | None = None,
    title: str = "Per-Type Accuracy",
) -> Any:
    """Bar chart of accuracy broken down by question type.

    Parameters
    ----------
    results : dict
        Pipeline results dict (from ``results.json``).
    save_path : str, optional
        If given, save the figure to this path.
    title : str
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    type_accuracy = results["accuracy"].get("types", {})
    types = [t for t in type_accuracy if type_accuracy[t] is not None]
    accs = [type_accuracy[t] * 100 for t in types]
    labels = [QUESTION_TYPE_LABELS.get(t, t) for t in types]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [PALETTE["blue"]] * len(types)
    bars = ax.bar(range(len(types)), accs, color=colors, edgecolor="white", linewidth=0.8)

    # Annotate bars
    for bar, acc in zip(bars, accs, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(accs) * 1.15 if accs else 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    overall = results["accuracy"].get("all", 0) * 100
    ax.axhline(
        y=overall,
        color=PALETTE["orange"],
        linestyle="--",
        linewidth=1.2,
        label=f"Overall: {overall:.1f}%",
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Exploration statistics
# ---------------------------------------------------------------------------


def plot_exploration_steps_histogram(
    output_dir: str,
    save_path: str | None = None,
    max_steps: int = 5,
) -> Any:
    """Histogram of how many exploration steps were taken per question.

    Parses the pipeline logs to count step usage across all questions.
    """
    import matplotlib.pyplot as plt

    logs = load_run_logs(output_dir)
    step_counts: list[int] = []
    current_qid: str | None = None
    step_for_qid = 0

    for entry in logs:
        msg = entry.get("message", "")
        if msg.startswith("===== QID"):
            if current_qid is not None:
                step_counts.append(step_for_qid)
            current_qid = None
            step_for_qid = 0
        elif current_qid is not None and msg.startswith("--- Step"):
            step_for_qid = max(step_for_qid, int(msg.split("Step")[1].strip().split()[0]) + 1)

    if current_qid is not None:
        step_counts.append(step_for_qid)

    if not step_counts:
        print("[WARN] No step data found in logs.")
        return None

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = (np.arange(0, max_steps + 2) - 0.5).tolist()
    ax.hist(
        step_counts, bins=bins, color=PALETTE["blue"], edgecolor="white", linewidth=0.8, alpha=0.85
    )
    ax.set_xlabel("Exploration steps used", fontsize=11)
    ax.set_ylabel("Number of questions", fontsize=11)
    ax.set_title("Exploration Budget Usage", fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, max_steps + 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Action distribution
# ---------------------------------------------------------------------------


def plot_action_distribution(
    output_dir: str,
    save_path: str | None = None,
) -> Any:
    """Pie chart showing the distribution of planned action types."""
    import matplotlib.pyplot as plt

    logs = load_run_logs(output_dir)
    action_counts: dict[str, int] = {"move-forward": 0, "turn-left": 0, "turn-right": 0}

    for entry in logs:
        msg = entry.get("message", "")
        if msg.startswith("Planned actions:"):
            actions_str = msg.split("Planned actions:")[1].strip()
            for action in actions_str.strip("[]").split(","):
                action = action.strip().strip("'\"")
                for key in action_counts:
                    if key in action:
                        action_counts[key] += 1
                        break

    total = sum(action_counts.values())
    if total == 0:
        print("[WARN] No action data found in logs.")
        return None

    labels = ["Move Forward", "Turn Left", "Turn Right"]
    sizes = [action_counts["move-forward"], action_counts["turn-left"], action_counts["turn-right"]]
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    # autopct is set, so matplotlib returns (wedges, texts, autotexts);
    # the stubs resolve the generic overload to the 2-tuple form.
    wedges, texts, autotexts = ax.pie(  # type: ignore[misc]
        sizes,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.6,
        explode=(0.02, 0.02, 0.02),
    )

    ax.legend(
        wedges,
        labels,
        title="Action Type",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9,
    )
    ax.set_title("Planned Action Distribution", fontsize=13, fontweight="bold")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Trajectory visualisation
# ---------------------------------------------------------------------------


def plot_exploration_trajectory(
    exploration_history: list[dict[str, Any]],
    save_path: str | None = None,
) -> Any:
    """Top-down view of camera trajectory from a single exploration."""
    import matplotlib.pyplot as plt

    if not exploration_history:
        return None

    pos = np.zeros(2, dtype=float)  # (x, z)
    angle = -10.0  # initial pitch-relative heading
    positions: list[np.ndarray] = [pos.copy()]

    for entry in exploration_history:
        action_seq = entry.get("action_sequence", [])
        for action_str in action_seq:
            action_type, value_str = action_str.split()
            value = float(value_str)
            if action_type == "move-forward":
                angle_rad = np.radians(angle)
                pos[0] += -value * np.sin(angle_rad)
                pos[1] += -value * np.cos(angle_rad)
            elif action_type == "turn-left":
                angle += value
            elif action_type == "turn-right":
                angle -= value
        positions.append(pos.copy())

    trajectory: np.ndarray = np.array(positions)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        "o-",
        color=PALETTE["blue"],
        linewidth=2,
        markersize=8,
        markeredgecolor="white",
    )

    # Mark start and end
    start_x, start_y = trajectory[0]
    end_x, end_y = trajectory[-1]
    ax.scatter(
        start_x,
        start_y,
        color="green",
        s=120,
        zorder=5,
        edgecolors="white",
        linewidth=1.5,
        label="Start",
    )
    ax.scatter(
        end_x, end_y, color="red", s=120, zorder=5, edgecolors="white", linewidth=1.5, label="End"
    )

    ax.set_xlabel("X (meters)", fontsize=11)
    ax.set_ylabel("Z (meters)", fontsize=11)
    ax.set_title("Camera Exploration Trajectory", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------


def generate_report(
    results_path: str,
    output_dir: str,
    figure_dir: str,
) -> None:
    """Generate all paper figures from a completed pipeline run."""
    os.makedirs(figure_dir, exist_ok=True)
    results = load_results(results_path)

    plot_per_type_accuracy(
        results,
        save_path=os.path.join(figure_dir, "per_type_accuracy.pdf"),
    )
    plot_exploration_steps_histogram(
        output_dir,
        save_path=os.path.join(figure_dir, "exploration_steps.pdf"),
    )
    plot_action_distribution(
        output_dir,
        save_path=os.path.join(figure_dir, "action_distribution.pdf"),
    )

    # Print summary stats
    overall = results["accuracy"].get("all", 0) * 100
    print(f"Overall accuracy: {overall:.1f}%")
    print("Per-type accuracy:")
    for qtype, acc in results["accuracy"].get("types", {}).items():
        if acc is not None:
            print(f"  {qtype}: {acc * 100:.1f}%")
    print(f"Figures saved to {figure_dir}/")
