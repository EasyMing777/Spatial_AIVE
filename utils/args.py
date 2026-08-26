"""Command-line argument parsing for the AIVE pipeline.

This module is the single source of truth for every CLI flag consumed by the
AIVE inference pipeline. Defaults are seeded from ``config.PipelineDefaults`` so
that dataclass-level defaults and CLI-level defaults never drift apart.
"""

import argparse

from utils.config import PipelineDefaults


def build_parser(defaults: PipelineDefaults | None = None) -> argparse.ArgumentParser:
    """Build the argument parser for the AIVE pipeline.

    Parameters
    ----------
    defaults : PipelineDefaults, optional
        Dataclass holding the default values. If ``None``, a fresh
        ``PipelineDefaults`` instance is used.

    Returns
    -------
    argparse.ArgumentParser
        Fully populated parser.
    """
    d = defaults or PipelineDefaults()

    parser = argparse.ArgumentParser(
        prog="aive",
        description="AIVE: Active Imagined View Exploration for Visual Spatial Reasoning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    parser.add_argument(
        "--input_dir",
        type=str,
        default=d.input_dir,
        help="Directory containing the QA dataset ({split}.json + images).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=d.output_dir,
        help="Directory where per-question traces, logs, and results are written.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )

    # ------------------------------------------------------------------
    # VLM configuration (per-role)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--vlm_qa_model_name",
        type=str,
        default=d.vlm_qa_model_name,
        help="Model for the Answerer role (final question answering).",
    )
    parser.add_argument(
        "--vlm_ap_model_name",
        type=str,
        default=d.vlm_ap_model_name,
        help="Model for the Planner role. Defaults to the QA model.",
    )
    parser.add_argument(
        "--vlm_d_model_name",
        type=str,
        default=d.vlm_d_model_name,
        help="Model for the Checker role. Defaults to the QA model.",
    )

    # ------------------------------------------------------------------
    # Exploration budget
    # ------------------------------------------------------------------
    parser.add_argument(
        "--max_steps_per_question",
        type=int,
        default=d.max_steps_per_question,
        help="Maximum number of action-planning / imagination cycles (T).",
    )
    parser.add_argument(
        "--max_tries_gpt",
        type=int,
        default=d.max_tries_gpt,
        help="VLM retries before giving up on a single parseable call.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=d.max_images,
        help="Maximum number of images per question before skipping.",
    )

    # ------------------------------------------------------------------
    # Action discretisation (paper: 0.25 m increments up to 3 m; 9 deg up to 90 deg)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--sampling_interval_meter",
        type=float,
        default=d.sampling_interval_meter,
        help="Translation quantization step (metres).",
    )
    parser.add_argument(
        "--sampling_interval_angle",
        type=int,
        default=d.sampling_interval_angle,
        help="Rotation quantization step (degrees).",
    )
    parser.add_argument(
        "--max_forward_distance",
        type=float,
        default=d.max_forward_distance,
        help="Maximum cumulative forward distance per step (metres).",
    )
    parser.add_argument(
        "--max_turn_angle",
        type=int,
        default=d.max_turn_angle,
        help="Maximum rotation per action (degrees).",
    )

    # ------------------------------------------------------------------
    # Dreamer (world model)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--dreamer_type",
        type=str,
        default="mock",
        choices=["mock", "wan2_2"],
        help="World-model backend. 'mock' is an identity placeholder; "
        "'wan2_2' requires a trained checkpoint (see dreamer.wan2_2).",
    )
    parser.add_argument(
        "--mock_dreamer_mode",
        type=str,
        default="identity",
        choices=["identity", "shift"],
        help="MockDreamer behaviour: 'identity' returns the input view; "
        "'shift' applies a small affine translation to simulate camera motion.",
    )

    # ------------------------------------------------------------------
    # Question filtering
    # ------------------------------------------------------------------
    parser.add_argument(
        "--num_questions", type=int, default=-1, help="Number of questions to evaluate (-1 = all)."
    )
    parser.add_argument(
        "--question_type",
        type=str,
        default=d.question_type,
        help="Restrict evaluation to one question type ('None' = all).",
    )
    parser.add_argument(
        "--force_explore",
        action="store_true",
        help="Bypass the Checker gate and always enter the exploration loop.",
    )

    # ------------------------------------------------------------------
    # Reproducibility / misc
    # ------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=d.seed, help="Random seed for reproducibility.")
    parser.add_argument(
        "--question_types",
        type=str,
        nargs="+",
        default=d.question_types,
        help="Question types to score in the results report.",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments into a namespace.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments, compatible with ``PipelineDefaults``.
    """
    parser = build_parser()
    return parser.parse_args(argv)


def namespace_to_dict(args: argparse.Namespace) -> dict:
    """Convert a parsed namespace into a plain dictionary."""
    return vars(args)


if __name__ == "__main__":
    import sys

    parser = build_parser()
    parser.parse_args(sys.argv[1:])
    parser.print_help()
