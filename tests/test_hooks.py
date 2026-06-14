"""
Structural tests for turboquant_vllm that run without a GPU or vLLM install.

Tests verify:
- Package exports are correct
- _worker_fn is importable at module level (required for pickle serialization)
- install_hooks raises RuntimeError with a descriptive message on a mock engine
  that has no recognised executor path
- _sum_rpc handles all expected input shapes
"""
import importlib
import types
import pytest


def test_package_exports_install_hooks():
    import turboquant_vllm
    assert hasattr(turboquant_vllm, "install_hooks")
    assert callable(turboquant_vllm.install_hooks)


def test_worker_fn_is_module_level():
    """_worker_fn must be importable by name for pickle to serialize it."""
    hooks_mod = importlib.import_module("turboquant_vllm.hooks")
    assert hasattr(hooks_mod, "_worker_fn"), "_worker_fn must be a module-level name"
    fn = hooks_mod._worker_fn
    # pickle requires __module__ and __qualname__ to be set on module-level functions
    assert fn.__module__ == "turboquant_vllm.hooks"
    assert "." not in fn.__qualname__, "_worker_fn must not be a closure or nested function"


def test_sum_rpc_integer():
    from turboquant_vllm.hooks import _sum_rpc
    assert _sum_rpc(24) == 24


def test_sum_rpc_list_mixed():
    from turboquant_vllm.hooks import _sum_rpc
    # collective_rpc may return worker ints mixed with framework metadata dicts
    assert _sum_rpc([12, {"meta": "data"}, 12]) == 24


def test_sum_rpc_non_int():
    from turboquant_vllm.hooks import _sum_rpc
    assert _sum_rpc({"not": "int"}) == 0


def test_install_hooks_raises_on_unrecognised_engine():
    """Ensure RuntimeError is raised when no executor path is found."""
    from turboquant_vllm.hooks import install_hooks

    # Minimal mock LLM with no recognised executor structure
    mock_llm = types.SimpleNamespace(
        llm_engine=types.SimpleNamespace(
            engine_core=None,
            model_executor=None,
        )
    )

    with pytest.raises(RuntimeError, match="supported vLLM executor path"):
        install_hooks(mock_llm)


def test_version():
    import turboquant_vllm
    assert turboquant_vllm.__version__ == "0.1.0"
