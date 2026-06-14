"""
TurboQuant hook installation for vLLM.

Installs TurboQuant attention hooks (Zandieh et al., 2024) on all workers
of a vLLM engine after model load.  Handles the vLLM v1 MP, v1 in-process,
and v0 executor trees automatically.

vLLM executor hierarchy (tried outermost-first):

    v1 MP (Modal / multi-process):
        llm.llm_engine.engine_core          → SyncMPClient
            .collective_rpc(fn)             ← hook installs here

    v1 in-process:
        llm.llm_engine.engine_core.engine_core.model_executor
            .collective_rpc(fn)             ← hook installs here

    v0 (single driver worker, direct access):
        llm.llm_engine.model_executor.driver_worker.model_runner
            install_turboquant_hooks(runner) ← called directly

pickle serialization note:
    collective_rpc transports the worker function via pickle/msgspec.
    install_hooks() builds a functools.partial bound to the caller's
    key_bits/value_bits/buffer_size and passes that to collective_rpc.
    functools.partial is picklable in Python 3 — pickle serializes the
    underlying function by name (turboquant_vllm.hooks._worker_fn_impl,
    which must remain module-level) and embeds the bound args by value.
    Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 in your environment if vLLM
    refuses to serialize the partial (vLLM >= 0.19 defaults to msgspec;
    this env var enables the pickle fallback).
"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM


# ---------------------------------------------------------------------------
# Worker-side implementation — MUST stay module-level so pickle can
# reference it by name as turboquant_vllm.hooks._worker_fn_impl.
# install_hooks() wraps it in functools.partial to bind caller parameters.
# ---------------------------------------------------------------------------

def _worker_fn_impl(worker, *, key_bits: int, value_bits: int, buffer_size: int) -> int:
    """Install TurboQuant hooks on a single vLLM worker with given parameters.

    Called on each GPU worker via collective_rpc (as a functools.partial).
    Returns the number of attention layers hooked.
    """
    from turboquant.vllm_attn_backend import MODE_ACTIVE, install_turboquant_hooks

    hooks = install_turboquant_hooks(
        worker.model_runner,
        key_bits=key_bits,
        value_bits=value_bits,
        buffer_size=buffer_size,
        mode=MODE_ACTIVE,
    )
    return len(hooks) if isinstance(hooks, list) else (hooks or 0)


# ---------------------------------------------------------------------------
# Executor-tree navigation
# ---------------------------------------------------------------------------

def _sum_rpc(results) -> int:
    """Extract integer results from collective_rpc return value.

    collective_rpc returns a list that may mix worker results with framework
    metadata dicts; extract only the integer elements.
    """
    if isinstance(results, list):
        return sum(r for r in results if isinstance(r, int))
    return results if isinstance(results, int) else 0


def install_hooks(
    llm: "LLM",
    key_bits: int = 3,
    value_bits: int = 2,
    buffer_size: int = 128,
) -> int:
    """Install TurboQuant KV-cache compression hooks on all vLLM workers.

    Call this once after ``LLM(...)`` returns and before any generation.
    Disable prefix caching when using TurboQuant (``enable_prefix_caching=False``).

    Args:
        llm:         A ``vllm.LLM`` instance with the model already loaded.
        key_bits:    Bit-width for keys (default 3, from the paper).
        value_bits:  Bit-width for values (default 2, from the paper).
        buffer_size: Recent tokens kept in full FP16 (default 128).

    Returns:
        Number of attention layers hooked (first 4 are skipped by TurboQuant
        design; for a 28-layer model expect 24).

    Raises:
        RuntimeError: If no supported executor path is found.
    """
    worker_fn = functools.partial(
        _worker_fn_impl,
        key_bits=key_bits,
        value_bits=value_bits,
        buffer_size=buffer_size,
    )

    engine = llm.llm_engine

    # ── v1 MP path: SyncMPClient sits at engine_core, exposes collective_rpc ──
    core = getattr(engine, "engine_core", None)
    if core is not None and hasattr(core, "collective_rpc"):
        return _sum_rpc(core.collective_rpc(worker_fn))

    # ── v1 in-process path ──
    if core is not None:
        inner = getattr(core, "engine_core", None)
        if inner is not None:
            executor = getattr(inner, "model_executor", None)
            if executor is not None and hasattr(executor, "collective_rpc"):
                return _sum_rpc(executor.collective_rpc(worker_fn))

    # ── v0 path: single driver worker, direct model_runner access ──
    executor = getattr(engine, "model_executor", None)
    if executor is not None:
        from turboquant.vllm_attn_backend import MODE_ACTIVE, install_turboquant_hooks
        runner = executor.driver_worker.model_runner
        hooks = install_turboquant_hooks(
            runner,
            key_bits=key_bits,
            value_bits=value_bits,
            buffer_size=buffer_size,
            mode=MODE_ACTIVE,
        )
        return len(hooks) if isinstance(hooks, list) else (hooks or 0)

    raise RuntimeError(
        f"Could not locate a supported vLLM executor path on this engine.\n"
        f"  llm.llm_engine type:       {type(engine)}\n"
        f"  engine_core type:          {type(core)}\n"
        f"Supported: vLLM v1 MP (SyncMPClient), v1 in-process, v0 driver-worker."
    )
