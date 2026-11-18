"""Reject incomplete or altered evidence before reporting a speed comparison."""

import json

import pytest

from limite_evals.belluxx_benchmark import digest, write
from limite_evals.belluxx_comparison import load_group


def evidence(tmp_path):
    run = tmp_path / "runs" / "baseline"
    run.mkdir(parents=True)
    write(
        run / "manifest.json",
        {
            "problems": 1,
            "rollouts": 2,
            "expected_samples": 2,
            "sample_keys": [[0, 0], [0, 1]],
            "max_model_len": 4,
            "draft_tokens": 0,
            "model_runner": "V2",
        },
    )
    rows = [
        dict(
            problem_index=0,
            rollout_index=i,
            completion_tokens=2,
            token_ids=[1, 2],
            max_tokens=2,
            prompt_tokens=2,
            finish_reason="length",
            protocols={"exact": {"correct": False}},
        )
        for i in range(2)
    ]
    (run / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write(
        run / "completion.json",
        {
            "complete": True,
            "samples": 2,
            "samples_sha256": digest(run / "samples.jsonl"),
        },
    )
    return run


def test_complete_truncated_samples_remain_in_comparison(tmp_path):
    evidence(tmp_path)
    rows = load_group(tmp_path, ["baseline"])[2]
    assert len(rows) == 2
    assert all(row["finish_reason"] == "length" for row in rows)


@pytest.mark.parametrize("mutation", ["altered_bytes", "duplicate_key", "short_budget"])
def test_invalid_evidence_is_rejected(tmp_path, mutation):
    run = evidence(tmp_path)
    path = run / "samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "duplicate_key":
        rows[1]["rollout_index"] = 0
    elif mutation == "short_budget":
        rows[1]["prompt_tokens"] = 1
    else:
        rows[1]["token_ids"][0] = 3
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if mutation != "altered_bytes":
        receipt = json.loads((run / "completion.json").read_text())
        receipt["samples_sha256"] = digest(path)
        write(run / "completion.json", receipt)
    with pytest.raises(AssertionError):
        load_group(tmp_path, ["baseline"])
