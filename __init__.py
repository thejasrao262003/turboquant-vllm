"""
turboquant_vllm — vLLM integration layer for TurboQuant KV-cache compression.

Installs TurboQuant attention hooks (Zandieh et al., 2024) on a vLLM engine
after model load, handling the vLLM v1 MP / v1 in-process / v0 executor
hierarchy automatically.

    from turboquant_vllm import install_hooks

    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", ...)
    n_hooked = install_hooks(llm)          # default: 3-bit keys, 2-bit values
    print(f"TurboQuant active on {n_hooked} attention layers")
"""
from turboquant_vllm.hooks import install_hooks  # noqa: F401

__version__ = "0.1.0"
__all__ = ["install_hooks"]
