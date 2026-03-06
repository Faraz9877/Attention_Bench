# Attention Kernel Benchmark

Best-effort benchmark matrix for FlashAttention / SageAttention variants on the same Q/K/V tensors.
The runner attempts many methods and skips methods that fail to import, compile, or execute.

## Install

```bash
make install_dep
```

`install_dep` does the following:
- creates `.venv` with Python 3.11 (required by the local FlashAttention wheel),
- installs FlashAttention from `dist/flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl`,
- installs SageAttention editable from `/workspace/SageAttention` with `TORCH_CUDA_ARCH_LIST=10.0` for B200/SM100.

## Run

```bash
make bench
```

Or run directly:

```bash
.venv/bin/python benchmark_sage_vs_flash.py --seq-lens 1024,2048,4096 --dtype bf16
```

Filter to a subset of methods by substring:

```bash
.venv/bin/python benchmark_sage_vs_flash.py --methods flash,sage.fp16_triton,sage.fp8_cuda
```

Change the baseline used for the speedup column:

```bash
.venv/bin/python benchmark_sage_vs_flash.py --baseline torch.sdpa
```

To include numerical checks against `torch.nn.functional.scaled_dot_product_attention`:

```bash
.venv/bin/python benchmark_sage_vs_flash.py --check
```

## Notes

- The script benchmarks fixed-shape `q/k/v` methods and also includes varlen / KV-cache API wrappers for FA2, FA3, and SageAttention where available.
- Unsupported methods are reported as `SKIP`; OOM cases are reported as `OOM`.
