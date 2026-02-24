import subprocess
import sys


def test_train_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "src.train", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--target" in result.stdout
    assert "--limit-files" in result.stdout
    assert "--delta-target" in result.stdout
    assert "--sweep" in result.stdout
    assert "--model" in result.stdout
    assert "--lookback" in result.stdout


def test_eval_cli_runs_demo():
    result = subprocess.run(
        [sys.executable, "-m", "src.eval"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "composite_score" in result.stdout
