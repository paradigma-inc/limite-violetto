"""CPU-only configuration boundary for the native V2 Belluxx adapter."""

CLASS_PATH = "limite_vllm.belluxx_v2.BelluxxSpeculator"
PROFILES = {"single-stream": 5, "multistream": 3}


def speculative_config(profile: str) -> dict[str, object]:
    """Return the explicit proposal count for a supported workload profile."""
    return {
        "method": "custom_class",
        "model": CLASS_PATH,
        "num_speculative_tokens": PROFILES[profile],
    }


def is_belluxx(spec) -> bool:
    return (
        spec is not None and spec.method == "custom_class" and spec.model == CLASS_PATH
    )


def validate_config(config) -> None:
    """Reject unsupported configurations before allocating the draft model."""
    target = config.model_config.hf_config
    if (
        target.model_type != "limite"
        or target.num_hidden_layers != 48
        or target.hidden_size != 1280
    ):
        raise ValueError("Belluxx V2 requires the 48-layer, 1280-wide Limite target")
    if str(config.model_config.dtype) not in ("bfloat16", "torch.bfloat16"):
        raise ValueError("Belluxx V2 requires a BF16 target")
    spec = config.speculative_config
    if not 1 <= spec.num_speculative_tokens <= 7:
        raise ValueError("Belluxx supports 1..7 draft tokens")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("Belluxx V2 requires prefix caching disabled")
    transfer = config.kv_transfer_config
    if transfer is not None and transfer.kv_connector is not None:
        raise ValueError(
            "Belluxx V2 requires KV transfer disabled; external KV caches "
            "do not restore drafter feature history"
        )
    parallel = config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("Belluxx V2 requires TP=PP=1")
    if config.scheduler_config.async_scheduling is False:
        raise ValueError("Belluxx V2 requires async scheduling")
    if spec.draft_sample_method != "greedy":
        raise ValueError("Belluxx V2 requires greedy draft proposals")
    if spec.rejection_sample_method != "standard":
        raise ValueError("Belluxx V2 requires standard rejection sampling")
