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
