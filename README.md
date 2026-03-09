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
.venv/bin/python benchmark_sage_vs_flash.py --dtype bf16
```

By default, the runner reads model dimensions from `model_dimensions.json` and runs every selected method across each model entry.
By default, it runs **non-causal** attention only.

Filter to a subset of methods by substring:

```bash
.venv/bin/python benchmark_sage_vs_flash.py --methods flash,sage.fp16_triton,sage.fp8_cuda
```

Filter to a subset of models by name substring:

```bash
.venv/bin/python benchmark_sage_vs_flash.py --models qwen,flux
```

Use a custom model-dimensions file:

```bash
.venv/bin/python benchmark_sage_vs_flash.py --models-config /path/to/model_dimensions.json
```

`model_dimensions.json` can be either:
- a top-level list of model objects, or
- an object with a `models` list.

Each model object supports:
- `name` (string),
- `batch_size` (int),
- `num_heads` (int),
- `head_dim` (int),
- `seq_lens` (list of ints or comma-separated string),
- `num_kv_heads` (optional int, defaults to `num_heads`).

Current default entries include:
- `qwen-image`: `num_heads=24`, `head_dim=128`
- `flux.2-dev`: `num_heads=24`, `head_dim=128`

Change the baseline used for the speedup column:

```bash
.venv/bin/python benchmark_sage_vs_flash.py --baseline torch.sdpa
```

To include numerical checks against `torch.nn.functional.scaled_dot_product_attention`:

```bash
.venv/bin/python benchmark_sage_vs_flash.py --check
```

Each run writes successful (`OK`) method rows to `results.csv` by default for plotting.
You can change the path with:

```bash
.venv/bin/python benchmark_sage_vs_flash.py --results-csv /path/to/results.csv
```

To run both non-causal and causal tables:

```bash
.venv/bin/python benchmark_sage_vs_flash.py --include-causal
```

To run only causal tables:

```bash
.venv/bin/python benchmark_sage_vs_flash.py --causal-only
```

Plot the CSV results (one vertical subplot per model, top-6 methods per sequence length):

```bash
.venv/bin/python plot_results.py --input results.csv --output results_plot.png --top-n 6 --causal false
```

## Notes

- The script benchmarks fixed-shape `q/k/v` methods and also includes varlen / KV-cache API wrappers for FA2, FA3, and SageAttention where available.
- If `flash-attn-4` is installed (e.g. `pip install flash-attn-4`), the FA4-style entrypoint `from flash_attn.cute import flash_attn_func` is benchmarked as `flash.cute_api` next to `flash.cute`.
- Unsupported methods are reported as `SKIP`; OOM cases are reported as `OOM`.
