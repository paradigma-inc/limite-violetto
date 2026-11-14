"""Exercise the pinned installer in fresh processes without loading weights."""

import os
import subprocess
import sys
import textwrap

import pytest


def run_check(source):
    env = dict(os.environ, VLLM_PLUGINS="limite", VLLM_USE_V2_MODEL_RUNNER="1")
    env.pop("LIMITE_BELLUXX_WEIGHTS", None)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("hook", ["dynamic", "builtin", "changed_checks"])
def test_reject_incompatible_hooks_without_partial_install(hook):
    run_check(f"""
        import inspect
        from unittest.mock import patch

        import vllm.config.speculative as spec_module
        import vllm.config.vllm as config_module
        import vllm.v1.worker.gpu.model_runner as runner_module
        import vllm.v1.worker.gpu.spec_decode as factory_module
        from limite_vllm.belluxx_v2 import install

        cls = config_module.VllmConfig
        hook = {hook!r}
        if hook == "dynamic":
            namespace = {{}}
            exec("def replacement(self): pass", namespace)
            cls.__post_init__ = namespace["replacement"]
        elif hook == "builtin":
            cls.__post_init__ = object.__init__

        def hooks():
            return (
                cls.__post_init__,
                cls._get_v2_model_runner_unsupported_features,
                spec_module.SpeculativeConfig.use_eagle,
                factory_module.init_speculator,
                runner_module.init_speculator,
                runner_module.GPUModelRunner.__init__,
            )

        original = hooks()
        source = inspect.getsource
        def getsource(obj):
            if hook == "changed_checks" and obj is cls.__post_init__:
                return "def __post_init__(self): pass"
            return source(obj)

        with patch.object(inspect, "getsource", getsource):
            try:
                install()
            except RuntimeError as exc:
                expected = (
                    "async compatibility checks changed"
                    if hook == "changed_checks" else "VLLM_PLUGINS=limite"
                )
                assert expected in str(exc), str(exc)
            else:
                raise AssertionError("Incompatible hook was accepted")
        assert all(a is b for a, b in zip(original, hooks()))
        assert not getattr(cls, "_belluxx_v2_installed", False)
    """)


def test_install_is_idempotent_on_pinned_runtime():
    run_check("""
        import vllm.config.vllm as config_module
        import vllm.v1.worker.gpu.spec_decode as factory_module
        from limite_vllm.belluxx_v2 import install

        cls = config_module.VllmConfig
        original = cls.__post_init__
        install()
        installed = cls.__post_init__
        factory = factory_module.init_speculator
        assert installed is not original
        assert cls._belluxx_v2_installed
        install()
        assert cls.__post_init__ is installed
        assert factory_module.init_speculator is factory
    """)
