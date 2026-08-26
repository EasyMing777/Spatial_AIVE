"""Unit tests for CLI argument parsing and paper-aligned defaults."""

from utils.args import build_parser


def test_defaults_match_paper():
    args = build_parser().parse_args([])
    # exploration budget T = 3 (paper §5.1)
    assert args.max_steps_per_question == 3
    # action space: forward up to 3 m in 0.25 m steps; rotation up to 90 deg in 9 deg steps
    assert args.max_forward_distance == 3.0
    assert args.sampling_interval_meter == 0.25
    assert args.max_turn_angle == 90
    assert args.sampling_interval_angle == 9
    # default VLM + dreamer backends
    assert args.vlm_qa_model_name == "gpt-4o"
    assert args.dreamer_type == "mock"


def test_role_models_overridable():
    args = build_parser().parse_args(
        ["--vlm_ap_model_name", "internvl3-8b", "--vlm_d_model_name", "gpt-5"]
    )
    assert args.vlm_ap_model_name == "internvl3-8b"
    assert args.vlm_d_model_name == "gpt-5"


def test_force_explore_flag():
    args = build_parser().parse_args(["--force_explore"])
    assert args.force_explore is True
