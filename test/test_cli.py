"""CLI subcommands that need no weights (list, probe)."""
from __future__ import annotations

import pytest

from audiosronnx.cli import main


def test_cli_list(capsys):
    rc = main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "lavasr" in out
    assert "novasr" in out
    assert "48000 Hz" in out


def test_cli_probe(capsys, speech16k_path):
    rc = main(["probe", speech16k_path])
    out = capsys.readouterr().out
    assert rc == 0
    assert "sample_rate: 16000" in out
    assert "duration:" in out


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])


# --------------------------------------------------------------------------- #
# Denoise subcommands
# --------------------------------------------------------------------------- #
def test_denoise_subcommands_are_registered():
    from audiosronnx.cli import build_parser

    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    commands = set()
    for action in actions:
        commands.update(action.choices)
    assert {"denoise", "denoise-dir"} <= commands


def test_denoise_parses_engine_and_model():
    from audiosronnx.cli import build_parser

    args = build_parser().parse_args(
        ["denoise", "in.wav", "out.wav", "--engine", "gtcrn", "--model", "dpdfnet8"])
    assert args.engine == "gtcrn" and args.model == "dpdfnet8"


def test_denoise_defaults_to_the_default_denoiser():
    from audiosronnx.api import DEFAULT_DENOISER
    from audiosronnx.cli import build_parser

    args = build_parser().parse_args(["denoise", "in.wav", "out.wav"])
    assert args.engine == DEFAULT_DENOISER
    assert args.model is None


def test_denoise_kwargs_omits_unset_model():
    """Engines without a `model` parameter would reject model=None."""
    from audiosronnx.cli import _denoise_kwargs, build_parser

    args = build_parser().parse_args(["denoise", "in.wav", "out.wav"])
    assert _denoise_kwargs(args) == {}


def test_list_groups_both_kinds(capsys):
    from audiosronnx.cli import build_parser

    args = build_parser().parse_args(["list"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "super-resolution" in out and "denoising" in out
    assert "dpdfnet" in out and "lavasr" in out
