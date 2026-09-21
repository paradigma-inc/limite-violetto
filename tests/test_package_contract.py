"""Standalone package, configuration, and vLLM graph contracts."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "limite_vllm"

RUNTIME_FILES = {
    "src/limite_vllm/__init__.py",
    "src/limite_vllm/_components.py",
    "src/limite_vllm/configuration.py",
    "src/limite_vllm/contract.py",
    "src/limite_vllm/modeling.py",
    "src/limite_vllm/register.py",
    "src/limite_vllm/validation.py",
}

SOURCE_REVISIONS = {
    "staging": "51dc058457fa53315183fb58ae17b5bbc4f12206",
}


def _config(*, version: int) -> dict[str, object]:
    raw: dict[str, object] = {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 16,
        "mlp_ratio": 2,
        "vocab_size": 32,
        "padded_vocab_size": 32,
        "tokenizer_vocab_size": 32,
        "torch_dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
        "attention_softmax_scale": 0.1,
        "sliding_window": 4,
        "sliding_window_convention": (
            "k >= q - sliding_window, inclusive of the query token "
            "(span = sliding_window + 1 keys)"
        ),
        "global_window": -1,
        "global_layers": [1],
        "global_nope": True,
        "attn_gate_channels": 0,
        "pos_mode": "rope",
        "rope_frac": 0.5,
        "rope_base_local": 10_000.0,
        "rope_base_global": 10_000.0,
        "rope_per_layer": False,
        "rope_n_pairs": 2,
        "rope_style": "interleaved_pairs_odd_lane_sign_flip",
        "rope_cos_sin_dtype": "bfloat16",
        "qk_norm": "rms_pre_rope",
        "rms_norm_has_weight": False,
        "rms_norm_eps_mode": "torch_finfo_default",
        "ve_dim": 4,
        "ve_layers": [0],
        "ve_gate_channels": 2,
        "ve_gate_scale": 2.0,
        "ve_head_slice": "first_num_key_value_heads",
        "xsa": False,
        "xsa_layers": [],
        "xsa_normalize_eps": 1e-6,
        "mudd": False,
        "mudd_layers": [],
        "mudd_taps": 0,
        "mudd_inter": 0,
        "mudd_tap_idx": {},
        "mudd_accumulation": "ordered_left_to_right",
        "mlp_type": "relu2",
        "softcap_logits": {"kind": "sigmoid", "a": 23.0, "b": 5.0, "c": 7.5},
        "final_softcap": 0.0,
        "lm_head_precision_mode": "oracle_exact",
        "tie_word_embeddings": True,
        "source_format": "test_fixture",
        "checkpoint_step": 1,
        "checkpoint_world_size": 1,
    }
    if version == 5:
        raw.update(
            {
                "mlp_type": "swiglu",
                "attn_gate_channels": 2,
                "attn_gate_scale": 2.0,
                "attn_gate_applied": "per_head_before_o_proj",
                "ve_stored_heads": 1,
                "mudd": True,
                "mudd_layers": [1],
                "mudd_taps": 1,
                "mudd_inter": 2,
                "mudd_tap_idx": {"1": [0]},
                "mudd_mlp": True,
                "mudd_r_site": "resid",
            }
        )
    return raw


def _install_vllm_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide only the vLLM import surface needed to define the model graph."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    modules = {
        name: ModuleType(name)
        for name in (
            "vllm",
            "vllm.compilation",
            "vllm.compilation.decorators",
            "vllm.config",
            "vllm.distributed",
            "vllm.model_executor",
            "vllm.model_executor.layers",
            "vllm.model_executor.layers.attention",
            "vllm.model_executor.layers.vocab_parallel_embedding",
            "vllm.model_executor.model_loader",
            "vllm.model_executor.model_loader.weight_utils",
            "vllm.model_executor.models",
            "vllm.model_executor.models.utils",
            "vllm.v1",
            "vllm.v1.attention",
            "vllm.v1.attention.backend",
        )
    }
    for name, module in modules.items():
        if name in {
            "vllm",
            "vllm.compilation",
            "vllm.model_executor",
            "vllm.model_executor.layers",
            "vllm.model_executor.model_loader",
            "vllm.model_executor.models",
            "vllm.v1",
            "vllm.v1.attention",
        }:
            module.__path__ = []  # type: ignore[attr-defined]

    def support_torch_compile(**_kwargs: object):
        return lambda cls: cls

    class PlaceholderAttention(nn.Module):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__()

    class PlaceholderEmbedding(nn.Module):
        def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            **_kwargs: object,
        ) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(num_embeddings, embedding_dim),
                requires_grad=False,
            )

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return F.embedding(input_ids, self.weight)

    modules["vllm.compilation.decorators"].support_torch_compile = (
        support_torch_compile
    )
    modules["vllm.config"].VllmConfig = object
    modules["vllm.distributed"].get_pp_group = lambda: SimpleNamespace(world_size=1)
    modules["vllm.distributed"].get_tensor_model_parallel_world_size = lambda: 1
    modules["vllm.model_executor.layers.attention"].Attention = PlaceholderAttention
    modules[
        "vllm.model_executor.layers.vocab_parallel_embedding"
    ].VocabParallelEmbedding = PlaceholderEmbedding
    modules["vllm.model_executor.model_loader.weight_utils"].default_weight_loader = (
        lambda parameter, loaded: parameter.copy_(loaded)
    )
    modules["vllm.model_executor.models.utils"].maybe_prefix = lambda prefix, name: (
        f"{prefix}.{name}" if prefix else name
    )
    modules["vllm.v1.attention.backend"].AttentionType = SimpleNamespace(
        DECODER="decoder"
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_canonical_runtime_inventory() -> None:
    present = {
        str(path.relative_to(PROJECT_ROOT))
        for path in SOURCE_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert present == RUNTIME_FILES
    assert not list((PROJECT_ROOT / "src" / "limite_model").glob("*.py"))
    assert not list((PROJECT_ROOT / "hub").glob("*"))
    assert not list((PROJECT_ROOT / "scripts").glob("*"))


def test_public_identity_and_entry_point() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "limite-vllm"
    assert metadata["project"]["requires-python"] == "~=3.12.0"
    assert metadata["project"]["dependencies"] == []
    assert "transformers==5.6.2" in metadata["dependency-groups"]["dev"]
    assert metadata["project"]["entry-points"]["vllm.general_plugins"] == {
        "limite": "limite_vllm.register:register"
    }
    assert metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/limite_vllm"
    ]

    from limite_vllm import LimiteConfig
    from limite_vllm import register

    assert LimiteConfig.model_type == "limite"
    assert register.ARCHITECTURE == "LimiteForCausalLM"
    assert register.MODEL_TYPE == "limite"


def test_plugin_registration_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_vllm_import_stubs(monkeypatch)

    class Registry:
        architectures: dict[str, object] = {}
        registrations: list[tuple[str, object]] = []

        @classmethod
        def get_supported_archs(cls) -> list[str]:
            return list(cls.architectures)

        @classmethod
        def register_model(cls, architecture: str, model: object) -> None:
            cls.architectures[architecture] = model
            cls.registrations.append((architecture, model))

    sys.modules["vllm"].ModelRegistry = Registry

    from limite_vllm.register import register

    register()
    register()

    assert len(Registry.registrations) == 1
    architecture, model = Registry.registrations[0]
    assert architecture == "LimiteForCausalLM"
    assert model.__name__ == "LimiteForCausalLM"


def test_vllm_version_guard_accepts_build_suffixes_and_rejects_other_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from limite_vllm import register

    monkeypatch.setattr(register, "package_version", lambda _name: "0.26.0+cu129")
    register._validate_vllm_version()

    monkeypatch.setattr(register, "package_version", lambda _name: "0.27.0")
    with pytest.raises(RuntimeError, match=r"supports vLLM 0\.26\.0; found 0\.27\.0"):
        register._validate_vllm_version()


def test_configuration_is_self_contained_and_uses_limite_identity() -> None:
    from limite_vllm.configuration import LimiteConfig
    from limite_vllm.validation import validate

    config = LimiteConfig(**_config(version=4))
    assert config.model_type == "limite"
    assert config.architectures == ["LimiteForCausalLM"]
    assert config.sliding_window == 5
    assert config.to_dict()["sliding_window"] == 4
    assert config.layer_types == ["sliding_attention", "full_attention"]
    validate(config)


@pytest.mark.parametrize("version", [4, 5])
def test_supported_configurations_are_admitted(version: int) -> None:
    from limite_vllm.configuration import LimiteConfig
    from limite_vllm.validation import validate

    config = LimiteConfig(**_config(version=version))
    validate(config)


def test_lint_configuration_checks_the_complete_runtime() -> None:
    config = tomllib.loads((PROJECT_ROOT / ".ruff.toml").read_text())
    assert "exclude" not in config
    assert config["lint"]["select"] == ["E4", "E7", "E9", "F"]


def test_provenance_records_vllm_only_ownership() -> None:
    provenance = (PROJECT_ROOT / "PROVENANCE.md").read_text()
    for source, revision in SOURCE_REVISIONS.items():
        assert source in provenance
        assert revision in provenance
    assert "canonical" in provenance.lower()
    assert "single optimized execution graph" in provenance.lower()
    assert "transformers graph" in provenance.lower()


def test_repository_contains_no_legacy_or_cross_package_imports() -> None:
    forbidden_names = ("gui" + "do", "gm" + "pt", "gm" + "_pt")
    forbidden_imports = (
        "from limite_" + "model",
        "import limite_" + "model",
    )
    excluded_parts = {
        ".git",
        ".hub-staging",
        ".venv",
        "dist",
        "outputs",
        "__pycache__",
        ".pytest_cache",
    }
    offenders: list[str] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or excluded_parts.intersection(path.parts):
            continue
        try:
            content = path.read_text().lower()
        except UnicodeDecodeError:
            continue
        if any(name in content for name in forbidden_names + forbidden_imports):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []


def _full_source_inventory(
    model: object,
) -> tuple[list[tuple[str, object]], dict[str, object]]:
    """Create one complete checkpoint inventory and its once-folded parameters."""
    import torch

    params = dict(model.named_parameters())
    sources: dict[str, torch.Tensor] = {}
    expected: dict[str, torch.Tensor] = {}

    def values(shape: torch.Size, dtype: torch.dtype, ordinal: int) -> torch.Tensor:
        count = shape.numel()
        raw = torch.arange(count, dtype=torch.float32).reshape(shape)
        return (raw / 128 + ordinal / 16).to(dtype)

    ordinal = 1
    for param_name, param in sorted(params.items()):
        if param_name.endswith(".self_attn.qkv_proj.weight"):
            stem = param_name.removesuffix("qkv_proj.weight")
            hidden = int(model.config.hidden_size)
            kv_size = int(model.config.num_key_value_heads) * int(
                model.config.head_dim
            )
            names_and_rows = (
                (stem + "q_proj.weight", hidden),
                (stem + "k_proj.weight", kv_size),
                (stem + "v_proj.weight", kv_size),
            )
            chunks = []
            for source_name, rows in names_and_rows:
                source = values(
                    torch.Size((rows, param.shape[1])), param.dtype, ordinal
                )
                ordinal += 1
                sources[source_name] = source
                chunks.append(source)
            expected[param_name] = torch.cat(chunks)
        elif param_name.endswith(".mlp.gate_up_proj.weight"):
            stem = param_name.removesuffix("gate_up_proj.weight")
            rows = int(model.config.intermediate_size)
            chunks = []
            for source_name in (stem + "gate_proj.weight", stem + "up_proj.weight"):
                source = values(
                    torch.Size((rows, param.shape[1])), param.dtype, ordinal
                )
                ordinal += 1
                sources[source_name] = source
                chunks.append(source)
            expected[param_name] = torch.cat(chunks)
        else:
            source = values(param.shape, param.dtype, ordinal)
            ordinal += 1
            sources[param_name] = source
            expected[param_name] = source.clone()

    for layer_index in range(int(model.config.num_hidden_layers)):
        stem = f"model.layers.{layer_index}.self_attn."
        qkv_name = stem + "qkv_proj.weight"
        o_name = stem + "o_proj.weight"
        qkv_scale_name = stem + "qkv_scale"
        o_scale_name = stem + "o_scale"
        sources[qkv_scale_name] = torch.tensor(1.25 + layer_index / 8)
        sources[o_scale_name] = torch.tensor(0.75 + layer_index / 8)
        expected[qkv_scale_name] = sources[qkv_scale_name].clone()
        expected[o_scale_name] = sources[o_scale_name].clone()
        expected[qkv_name] = expected[qkv_name] * sources[qkv_scale_name].to(
            expected[qkv_name].dtype
        )
        expected[o_name] = expected[o_name] * sources[o_scale_name].to(
            expected[o_name].dtype
        )

    return sorted(sources.items()), expected


@pytest.mark.parametrize("version", [4, 5])
def test_full_reload_restores_raw_before_folding_once(
    monkeypatch: pytest.MonkeyPatch, version: int
) -> None:
    import torch

    from limite_vllm.configuration import LimiteConfig

    _install_vllm_import_stubs(monkeypatch)
    from limite_vllm.modeling import LimiteForCausalLM

    config = LimiteConfig(**_config(version=version))
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        cache_config=SimpleNamespace(sliding_window=None),
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = LimiteForCausalLM(vllm_config=vllm_config)
    finally:
        torch.set_default_dtype(previous_dtype)

    source_inventory, expected = _full_source_inventory(model)
    expected_names = set(dict(model.named_parameters()))

    loaded_first = model.load_weights(
        [(name, tensor.clone()) for name, tensor in source_inventory]
    )
    after_first = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    assert loaded_first == expected_names
    for name in expected_names:
        assert torch.equal(after_first[name], expected[name]), name

    loaded_second = model.load_weights(
        [(name, tensor.clone()) for name, tensor in source_inventory]
    )
    after_second = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    assert loaded_second == expected_names
    for name in expected_names:
        assert torch.equal(after_second[name], expected[name]), name
        assert torch.equal(after_second[name], after_first[name]), name
