"""
Structural tests for turboquant_vllm that run without a GPU or vLLM install.

Tests verify:
- Package exports are correct
- _worker_fn_impl is module-level (required for pickle to serialize it by name
  when used inside a functools.partial passed to collective_rpc)
- install_hooks builds a partial that binds caller-supplied parameters
- install_hooks raises RuntimeError with a descriptive message on a mock engine
  that has no recognised executor path
- _sum_rpc handles all expected input shapes
"""
import functools
import importlib
import types
import pytest


def test_package_exports_install_hooks():
    import turboquant_vllm
    assert hasattr(turboquant_vllm, "install_hooks")
    assert callable(turboquant_vllm.install_hooks)


def test_worker_fn_impl_is_module_level():
    """_worker_fn_impl must be a module-level function.

    install_hooks() wraps it in functools.partial to bind parameters.
    pickle serializes partial objects by embedding the underlying function
    reference by name — so _worker_fn_impl must be importable at module scope.
    """
    hooks_mod = importlib.import_module("turboquant_vllm.hooks")
    assert hasattr(hooks_mod, "_worker_fn_impl"), "_worker_fn_impl must be module-level"
    fn = hooks_mod._worker_fn_impl
    assert fn.__module__ == "turboquant_vllm.hooks"
    assert "." not in fn.__qualname__, "_worker_fn_impl must not be a closure or nested function"


def test_worker_fn_partial_is_picklable():
    """functools.partial(_worker_fn_impl, ...) must survive a pickle round-trip.

    This is the object collective_rpc actually serializes and sends to workers.
    """
    import pickle
    from turboquant_vllm.hooks import _worker_fn_impl
    p = functools.partial(_worker_fn_impl, key_bits=4, value_bits=2, buffer_size=64)
    data = pickle.dumps(p)
    p2 = pickle.loads(data)
    assert p2.keywords == {"key_bits": 4, "value_bits": 2, "buffer_size": 64}


def test_install_hooks_binds_parameters():
    """Parameters passed to install_hooks() must reach the worker function.

    Verifies that install_hooks builds a partial with the caller's values,
    not hardcoded defaults.  Uses a mock engine that captures the worker_fn
    passed to collective_rpc rather than executing it.
    """
    import pickle
    from turboquant_vllm.hooks import install_hooks

    captured = {}

    def fake_collective_rpc(fn):
        captured["fn"] = fn
        return [0]  # pretend 0 layers hooked

    mock_llm = types.SimpleNamespace(
        llm_engine=types.SimpleNamespace(
            engine_core=types.SimpleNamespace(
                collective_rpc=fake_collective_rpc,
            ),
            model_executor=None,
        )
    )

    install_hooks(mock_llm, key_bits=4, value_bits=3, buffer_size=64)

    fn = captured["fn"]
    assert isinstance(fn, functools.partial), "worker_fn must be a functools.partial"
    assert fn.keywords["key_bits"] == 4
    assert fn.keywords["value_bits"] == 3
    assert fn.keywords["buffer_size"] == 64

    # Confirm it survives a pickle round-trip (the actual IPC path)
    fn2 = pickle.loads(pickle.dumps(fn))
    assert fn2.keywords["key_bits"] == 4


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
