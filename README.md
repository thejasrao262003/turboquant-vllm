# turboquant-vllm

Install TurboQuant KV-cache compression on a vLLM-served model with one function call, across all vLLM executor architectures.

## Why this exists

[TurboQuant](https://arxiv.org/abs/2407.11110) (Zandieh et al., 2024) compresses the KV cache to 3-bit keys and 2-bit values using random orthogonal rotation and Lloyd-Max quantisation. It ships with a hook API that patches vLLM attention layers at runtime.

The problem is getting those hooks onto the right workers.

vLLM has three executor architectures — v0 (single driver worker), v1 in-process, and v1 multi-process — with completely different internal object hierarchies. In production deployments (Modal, Ray, multi-GPU), vLLM runs the engine in a subprocess behind a `SyncMPClient`. You cannot reach workers directly. You must use `collective_rpc`, pass a picklable function (not a closure, not a lambda — it must be importable by fully-qualified name), handle the fact that `collective_rpc` returns a mixed list of worker results and framework metadata, and bind your quantization parameters across the IPC boundary without losing them.

None of this is documented. You find it by hitting:

```
AttributeError: 'SyncMPClient' object has no attribute 'model_executor'
TypeError: Object of type function is not JSON serializable
# and then: install_hooks(llm, key_bits=4) silently uses 3-bit because
# the closure didn't serialize correctly
```

This package solves all of it:

```python
from turboquant_vllm import install_hooks
install_hooks(llm)
```

## Features

- **Single call** — one function handles all three vLLM executor architectures
- **vLLM v1 multi-process** — `SyncMPClient` / `collective_rpc` path (the default Modal / multi-GPU case)
- **vLLM v1 in-process** — `engine_core.model_executor` path
- **vLLM v0** — direct `driver_worker.model_runner` path
- **Correct parameter propagation** — `key_bits`, `value_bits`, `buffer_size` reach workers on all paths via `functools.partial`
- **Runtime hook installation** — no model code changes, no forking vLLM
- **Model-agnostic** — works with any model vLLM supports that uses standard MHA or GQA attention

## Quick start

```bash
pip install --no-build-isolation git+https://github.com/0xSero/turboquant.git
pip install git+https://github.com/thejasrao262003/turboquant-vllm.git
```

Set this environment variable before running. vLLM ≥ 0.19 defaults to msgspec for IPC; the pickle fallback is required to transport the worker function across the subprocess boundary:

```bash
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
```

```python
from vllm import LLM, SamplingParams
from turboquant_vllm import install_hooks

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="float16",
    max_model_len=32768,
    enable_prefix_caching=False,  # required — incompatible with TurboQuant KV interception
)

n_hooked = install_hooks(llm)
# → 24  (TurboQuant skips the first 4 layers by design; 28-layer model)

outputs = llm.generate(["Explain KV cache quantisation."], SamplingParams(max_tokens=200))
print(outputs[0].outputs[0].text)
```

Custom parameters work on all executor paths:

```python
install_hooks(llm, key_bits=4, value_bits=2, buffer_size=256)
```

## How it works

`install_hooks()` detects which executor architecture vLLM is using, finds the path to GPU workers, and installs TurboQuant's hooks with the caller's parameters.

```
install_hooks(llm, key_bits=3, value_bits=2, buffer_size=128)
        │
        ▼
  Detect executor topology
        │
        ├── v1 MP?   llm.llm_engine.engine_core is SyncMPClient
        │            └─→ collective_rpc(partial(_worker_fn_impl, key_bits=3, ...))
        │
        ├── v1 in-process?   engine_core.engine_core.model_executor exists
        │                    └─→ collective_rpc(partial(_worker_fn_impl, key_bits=3, ...))
        │
        └── v0?   llm.llm_engine.model_executor.driver_worker exists
                  └─→ install_turboquant_hooks(model_runner, key_bits=3, ...)
```

**Executor discovery:** `install_hooks()` walks the engine hierarchy using `getattr`/`hasattr`, trying each path from outermost to innermost. It uses the first path that resolves.

**Parameter propagation:** Parameters are bound via `functools.partial(_worker_fn_impl, key_bits=..., ...)` before being passed to `collective_rpc`. `functools.partial` is picklable — pickle serializes the underlying function by module-level name and embeds the kwargs by value. This is why `install_hooks(llm, key_bits=4)` actually uses 4-bit keys on workers rather than falling back to the default.

**Return value:** `install_hooks()` returns the number of attention layers hooked. For a 28-layer model the expected return is 24 — TurboQuant skips the first 4 layers by design.

## Compatibility

Tested on vLLM 0.19.1. The executor attribute paths are vLLM internals, not a stable public API — minor version upgrades may require updates if the hierarchy changes.

| Executor | How to identify | Status |
|----------|-----------------|--------|
| v1 multi-process | `llm.llm_engine.engine_core` is `SyncMPClient` | Supported |
| v1 in-process | `engine_core.engine_core.model_executor` exists | Supported |
| v0 single driver | `llm.llm_engine.model_executor.driver_worker` exists | Supported |

## Model compatibility

`turboquant_vllm` installs hooks at the vLLM layer, not inside model-specific code. Whether TurboQuant's hooks behave correctly on a given model depends on TurboQuant's own compatibility with that model's attention implementation.

| Attention type | Expected |
|----------------|----------|
| Standard MHA (Llama, Falcon, …) | Should work |
| Grouped-query attention / GQA (Llama 3, Qwen2, Mistral, …) | Should work |
| MLA / absorbed-KV (DeepSeek-V2/V3) | Untested |
| Sliding-window attention | Untested |

Tested: Qwen2.5-7B-Instruct on a single A10G (24 GB) via Modal.

Validate perplexity on your model against the FP16 baseline before production use or benchmarks.

## Technical notes

**Prefix caching must be disabled.** TurboQuant intercepts KV cache updates at the attention layer; prefix caching conflicts with this interception. Set `enable_prefix_caching=False` when constructing `LLM`.

**Hybrid decode latency.** TurboQuant's `MODE_ACTIVE` dequantises the entire compressed history to float32 on every decode step. Latency grows with context length. This is a TurboQuant algorithm design choice, not a limitation of this package.

**v1 MP parameter binding.** If you are on an older vLLM version that does not support `VLLM_ALLOW_INSECURE_SERIALIZATION`, the pickle fallback is unavailable and hook installation will fail. This is a vLLM constraint.

**Hook count sanity check.** `install_hooks()` returns the number of layers hooked. If this is 0 or significantly less than `num_hidden_layers - 4`, something went wrong before generation starts.

## Memory savings

These numbers come from TurboQuant (Zandieh et al., 2024) and reflect the algorithm's compression ratio, not anything specific to this package.

At 28K context on Qwen2.5-7B-Instruct (28 layers, A10G 24 GB):

| Configuration | KV cache memory |
|---------------|-----------------|
| FP16 baseline | ~1.6 GB |
| TurboQuant 3-bit keys / 2-bit values | ~270 MB (~6× reduction) |

Model weights (~14 GB FP16) are unchanged.

## API

### `install_hooks(llm, key_bits=3, value_bits=2, buffer_size=128) → int`

Install TurboQuant KV-cache compression hooks on all GPU workers of a `vllm.LLM` instance.

Call once after `LLM(...)` returns and before any calls to `generate()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `vllm.LLM` | — | A loaded LLM instance |
| `key_bits` | `int` | `3` | Bit-width for key cache |
| `value_bits` | `int` | `2` | Bit-width for value cache |
| `buffer_size` | `int` | `128` | Recent tokens kept in full FP16 |

**Returns** `int` — number of attention layers hooked.

**Raises** `RuntimeError` — if no supported vLLM executor path is found, with a diagnostic message including the engine and engine_core types.

## Citation

If you use TurboQuant in your work, cite the original paper:

```bibtex
@article{zandieh2024turboquant,
  title   = {SubGen: Token Generation in Sublinear Time and Memory},
  author  = {Zandieh, Amir and Han, Insu and Mirrokni, Vahab and Karbasi, Amin},
  journal = {arXiv preprint arXiv:2407.11110},
  year    = {2024},
}
```

## License

MIT.
