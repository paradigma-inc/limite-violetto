"""Configuration checks require no model, dataset, or accelerator imports."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

API = runpy.run_path(
    str(Path(__file__).parents[1] / "src/limite_vllm/belluxx_config.py")
)


def config():
    return NS(
        model_config=NS(
            dtype="torch.bfloat16",
            hf_config=NS(model_type="limite", num_hidden_layers=48, hidden_size=1280),
        ),
        speculative_config=NS(
            **API["speculative_config"]("single-stream"),
            draft_sample_method="greedy",
            rejection_sample_method="standard",
        ),
        cache_config=NS(enable_prefix_caching=False),
        kv_transfer_config=None,
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
        scheduler_config=NS(async_scheduling=None),
    )


@pytest.mark.parametrize("profile,count", [("single-stream", 5), ("multistream", 3)])
def test_profiles_select_the_same_adapter(profile, count):
    spec = API["speculative_config"](profile)
    assert spec["num_speculative_tokens"] == count
    assert API["is_belluxx"](NS(**spec))
    path = Path(__file__).parents[1] / "configs/belluxx" / f"{profile}.json"
    assert json.loads(path.read_text()) == spec


def test_admission_is_scoped_to_the_exact_adapter():
    assert not API["is_belluxx"](None)
    assert not API["is_belluxx"](NS(method="custom_class", model="other.Speculator"))
    assert not API["is_belluxx"](NS(method="eagle3", model=API["CLASS_PATH"]))
    API["validate_config"](config())


@pytest.mark.parametrize(
    "group,key,value,message",
    [
        ("speculative_config", "num_speculative_tokens", 0, "1..7"),
        ("speculative_config", "num_speculative_tokens", 8, "1..7"),
        ("speculative_config", "draft_sample_method", "probabilistic", "greedy"),
        ("speculative_config", "rejection_sample_method", "synthetic", "standard"),
        ("parallel_config", "tensor_parallel_size", 2, "TP=PP=1"),
        ("parallel_config", "pipeline_parallel_size", 2, "TP=PP=1"),
        ("cache_config", "enable_prefix_caching", True, "prefix caching"),
        ("scheduler_config", "async_scheduling", False, "async scheduling"),
    ],
)
def test_reject_unsupported_execution(group, key, value, message):
    candidate = config()
    setattr(getattr(candidate, group), key, value)
    with pytest.raises(ValueError, match=message):
        API["validate_config"](candidate)


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_type", "other"),
        ("num_hidden_layers", 24),
        ("hidden_size", 2048),
    ],
)
def test_reject_incompatible_target_before_loading(key, value):
    candidate = config()
    setattr(candidate.model_config.hf_config, key, value)
    with pytest.raises(ValueError, match="Limite target"):
        API["validate_config"](candidate)


def test_reject_target_precision_change():
    candidate = config()
    candidate.model_config.dtype = "torch.float16"
    with pytest.raises(ValueError, match="BF16"):
        API["validate_config"](candidate)


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", "kv_both"])
def test_reject_external_kv_even_without_local_prefix_caching(role):
    candidate = config()
    candidate.kv_transfer_config = NS(kv_connector="ExampleConnector", kv_role=role)
    with pytest.raises(ValueError, match="KV transfer.*feature history"):
        API["validate_config"](candidate)


def test_allow_inactive_kv_transfer_config():
    candidate = config()
    candidate.kv_transfer_config = NS(kv_connector=None, kv_role=None)
    API["validate_config"](candidate)
