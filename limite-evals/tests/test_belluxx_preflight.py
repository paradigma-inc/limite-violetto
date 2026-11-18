"""The benchmark preflight is scheduler-neutral and imports no ML libraries."""

import runpy
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "src/limite_evals/belluxx_benchmark.py"


def invoke(monkeypatch, tmp_path, *options):
    monkeypatch.setenv("HF_TOKEN", "unit-test-placeholder")
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "prepare", "--root", str(tmp_path), "--check", *options]
    )
    runpy.run_path(str(SCRIPT), run_name="__main__")


def test_preflight_does_not_import_ml_or_create_outputs(monkeypatch, tmp_path):
    before = set(sys.modules)
    invoke(monkeypatch, tmp_path)
    assert not list(tmp_path.iterdir())
    assert not ({"torch", "transformers", "datasets", "vllm"} & (set(sys.modules) - before))


@pytest.mark.parametrize(
    "options,message",
    [
        (["--max-new-tokens", "32"], "non-AIME"),
        (["--diagnostic"], "explicit bounded"),
        (["--diagnostic-context-length", "512"], "diagnostic workload"),
        (["--profile"], "bounded diagnostic"),
        (
            ["--diagnostic", "--max-new-tokens", "16", "--diagnostic-context-length", "512"],
            "Lower-context",
        ),
        (["--drafter-profile", "single-stream", "--draft-tokens", "3"], "conflicts"),
        (["--run-id", "../escape"], "path component"),
        (["--concurrency", "0"], "Invalid concurrency"),
        (["--max-batched-tokens", "1"], "complete decode batch"),
    ],
)
def test_invalid_measurements_fail_before_model_import(monkeypatch, tmp_path, options, message):
    with pytest.raises(ValueError, match=message):
        invoke(monkeypatch, tmp_path, *options)


@pytest.mark.parametrize("profile", ["single-stream", "multistream"])
def test_explicit_profiles_pass_nonmutating_preflight(monkeypatch, tmp_path, profile):
    source = tmp_path / "reference/tools/spec"
    source.mkdir(parents=True)
    (source / "drafter.py").touch()
    invoke(
        monkeypatch,
        tmp_path,
        "--drafter-profile",
        profile,
        "--drafter-source",
        str(tmp_path / "reference"),
    )
