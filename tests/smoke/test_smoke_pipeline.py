"""Whole pipeline on a tiny SYNTHETIC dataset (never reported as PTB-XL results)."""

from dp_ecg.cli import main


def test_smoke_pipeline(capsys):
    main(["smoke"])
    assert "smoke test passed" in capsys.readouterr().out
