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
    _worker_fn must be a module-level function (not a closure or lambda)
    so pickle can reference it by fully-qualified name:
        turboquant_vllm.hooks._worker_fn
    Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 in your environment if vLLM
    refuses to serialize the function (vLLM >= 0.19 defaults to msgspec;
    this env var enables the pickle fallback).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM


# ---------------------------------------------------------------------------
# Worker-side function — MUST stay module-level for pickle by name
# ---------------------------------------------------------------------------

def _worker_fn(worker) -> int:
    """Install TurboQuant hooks on a single vLLM worker.

    Called on each GPU worker via collective_rpc.  Parameters are hardcoded
    to the 3k2v configuration (3-bit keys, 2-bit values, 128-token ring
    buffer) from the TurboQuant paper.  The first 4 attention layers are
    left uncompressed (initial_layers_count=4 in TurboQuant defaults).

    Returns the number of attention layers hooked.
    """
    from turboquant.vllm_attn_backend import MODE_ACTIVE, install_turboquant_hooks

    hooks = install_turboquant_hooks(
        worker.model_runner,
        key_bits=3,
        value_bits=2,
        buffer_size=128,
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

    Note:
        ``key_bits``, ``value_bits``, and ``buffer_size`` are applied via the
        v0 direct path only.  In the v1 MP path the worker function
        ``_worker_fn`` is pickled by name and uses its own hardcoded 3k2v
        defaults.  If you need custom values in the v1 MP path, fork
        ``_worker_fn`` as a module-level function with your desired values.

    Raises:
        RuntimeError: If no supported executor path is found.
    """
    engine = llm.llm_engine

    # ── v1 MP path: SyncMPClient sits at engine_core, exposes collective_rpc ──
    core = getattr(engine, "engine_core", None)
    if core is not None and hasattr(core, "collective_rpc"):
        return _sum_rpc(core.collective_rpc(_worker_fn))

    # ── v1 in-process path ──
    if core is not None:
        inner = getattr(core, "engine_core", None)
        if inner is not None:
            executor = getattr(inner, "model_executor", None)
            if executor is not None and hasattr(executor, "collective_rpc"):
                return _sum_rpc(executor.collective_rpc(_worker_fn))

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
