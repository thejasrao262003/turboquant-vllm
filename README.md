# turboquant_vllm

vLLM integration layer for [TurboQuant](https://arxiv.org/abs/2407.11110) KV-cache compression (Zandieh et al., 2024).

TurboQuant compresses the KV cache using random orthogonal rotation followed by Lloyd-Max scalar quantisation, spreading outlier energy across dimensions to minimise quantisation error. This package handles installing TurboQuant's attention hooks on a `vllm.LLM` engine after model load, across all vLLM executor architectures (v1 multi-process, v1 in-process, v0).

## What this package does

TurboQuant patches vLLM's attention layers at runtime. The patch intercepts `do_kv_cache_update()` and `forward()` on each attention layer to compress older KV tokens to low-bit representations. This package provides `install_hooks()` — a single function that navigates vLLM's internal executor tree (which changed significantly between v0 and v1) and installs the hooks on all GPU workers.

## Compression parameters (paper defaults)

| Cache | Bit-width | Method |
|-------|-----------|--------|
| Keys | 3-bit | Lloyd-Max scalar quant + random orthogonal rotation |
| Values | 2-bit | Group quantisation (group_size=32) |
| Ring buffer | FP16 | Last 128 tokens kept uncompressed |
| First 4 layers | FP16 | Uncompressed (early layers are information-dense) |

At 28K context on Qwen2.5-7B (28 layers, A10G 24 GB):
- Model weights: ~14 GB (FP16)
- TurboQuant KV cache: ~270 MB (~37.5% of keys, ~25% of values vs FP16)
- FP16 baseline KV cache: ~1.6 GB

## Installation

First install the TurboQuant library, then this package:

```bash
pip install --no-build-isolation git+https://github.com/0xSero/turboquant.git
pip install .           # from this directory
# or
pip install git+https://github.com/<your-fork>/turboquant_vllm.git
```

Also set this environment variable before running (vLLM 0.19+ defaults to msgspec serialisation; the pickle fallback is needed to transport the worker function over IPC):

```bash
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
```

## Quick start

```python
from vllm import LLM, SamplingParams
from turboquant_vllm import install_hooks

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="float16",
    max_model_len=32768,
    enable_prefix_caching=False,   # incompatible with TurboQuant KV interception
)

n_hooked = install_hooks(llm)      # default: 3-bit keys, 2-bit values
print(f"TurboQuant active on {n_hooked} attention layers")
# → TurboQuant active on 24 attention layers  (first 4 skipped by design)

outputs = llm.generate(["Explain KV cache quantisation."], SamplingParams(max_tokens=200))
print(outputs[0].outputs[0].text)
```

## API

### `install_hooks(llm, key_bits=3, value_bits=2, buffer_size=128) → int`

Install TurboQuant hooks on all workers of a `vllm.LLM` engine.

**Args**
- `llm` — a `vllm.LLM` instance (model already loaded)
- `key_bits` — bit-width for keys (default 3)
- `value_bits` — bit-width for values (default 2)
- `buffer_size` — recent tokens kept in FP16 ring buffer (default 128)

**Returns** number of attention layers hooked.

**Raises** `RuntimeError` if no supported executor path is found.

> Parameters are applied across all executor paths. `install_hooks()` builds
> a `functools.partial` bound to the caller's values and passes that to
> `collective_rpc`, so `key_bits=4` in v1 MP mode works as expected.

## vLLM executor compatibility

| Executor | Detection | Hook path |
|----------|-----------|-----------|
| v1 multi-process (Modal, `--tensor-parallel`) | `engine.engine_core` is `SyncMPClient` | `collective_rpc(_worker_fn)` |
| v1 in-process | `engine.engine_core.engine_core.model_executor` | `collective_rpc(_worker_fn)` |
| v0 single driver | `engine.model_executor.driver_worker` | direct `install_turboquant_hooks()` |

## Model compatibility

`turboquant_vllm` is model-agnostic — it installs hooks at the vLLM layer, not the model layer. Any model vLLM supports should work in principle.

| Attention type | Status |
|----------------|--------|
| Standard MHA / GQA (Llama, Qwen, Mistral, Falcon, …) | Expected to work |
| MLA / absorbed-KV (DeepSeek-V2/V3) | Untested — hook interface may differ |
| Sliding-window attention (Mistral long-context variants) | Untested |

Validate perplexity on your specific model against the FP16 baseline before running benchmarks.

## Known limitations

- **Hybrid decode**: TurboQuant's `MODE_ACTIVE` dequantises the entire compressed history to float32 on every decode step. Latency grows with context length but does not affect output quality.
- **No prefix caching**: set `enable_prefix_caching=False` when constructing `LLM`.

## Citation

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
