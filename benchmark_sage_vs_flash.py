#!/usr/bin/env python3
import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F


TensorFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]


@dataclass
class BenchMethod:
    name: str
    group: str
    fn: TensorFn
    note: str = ""


@dataclass
class BenchResult:
    status: str
    ms: Optional[float] = None
    tflops: Optional[float] = None
    max_abs_err: Optional[float] = None
    message: str = ""


@dataclass
class ModelBenchConfig:
    name: str
    batch_size: int
    num_heads: int
    head_dim: int
    seq_lens: list[int]
    num_kv_heads: Optional[int] = None


def parse_seq_lens(seq_lens: str) -> List[int]:
    return [int(x.strip()) for x in seq_lens.split(",") if x.strip()]


def parse_dtype(dtype: str) -> torch.dtype:
    name = dtype.lower()
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def get_arch(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm{major}{minor}"


def unwrap_output(result) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (tuple, list)) and result:
        for item in result:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Unsupported kernel output type: {type(result)}")


def import_optional(module_name: str):
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def short_error(exc: BaseException) -> str:
    msg = str(exc).strip().replace("\n", " ")
    if len(msg) > 120:
        msg = msg[:117] + "..."
    return f"{type(exc).__name__}: {msg}"


def load_model_bench_configs(config_path: Path) -> list[ModelBenchConfig]:
    data = json.loads(config_path.read_text())
    rows = data["models"] if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("Model config must be a list or an object with a 'models' list.")

    models: list[ModelBenchConfig] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Model config entry {idx} must be an object.")
        name = str(row.get("name", f"model_{idx}"))
        if "batch_size" not in row or "num_heads" not in row or "head_dim" not in row or "seq_lens" not in row:
            raise ValueError(
                f"Model '{name}' must define batch_size, num_heads, head_dim, and seq_lens."
            )
        seq_lens_raw = row["seq_lens"]
        if isinstance(seq_lens_raw, str):
            seq_lens = parse_seq_lens(seq_lens_raw)
        elif isinstance(seq_lens_raw, list):
            seq_lens = [int(v) for v in seq_lens_raw]
        else:
            raise ValueError(f"Model '{name}' has invalid seq_lens type: {type(seq_lens_raw)}")
        models.append(
            ModelBenchConfig(
                name=name,
                batch_size=int(row["batch_size"]),
                num_heads=int(row["num_heads"]),
                head_dim=int(row["head_dim"]),
                seq_lens=seq_lens,
                num_kv_heads=int(row["num_kv_heads"]) if row.get("num_kv_heads") is not None else None,
            )
        )
    return models


def should_keep_model(name: str, filter_expr: str) -> bool:
    if not filter_expr or filter_expr.lower() == "all":
        return True
    tokens = [tok.strip().lower() for tok in filter_expr.split(",") if tok.strip()]
    lower_name = name.lower()
    return any(tok in lower_name for tok in tokens)


@torch.inference_mode()
def benchmark_kernel(
    fn: TensorFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float]:
    out = None
    for _ in range(warmup):
        out = unwrap_output(fn(q, k, v, causal))
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = unwrap_output(fn(q, k, v, causal))
    end.record()
    torch.cuda.synchronize()

    assert out is not None
    return out, start.elapsed_time(end) / iters


@torch.inference_mode()
def sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    q_hnd = q.permute(0, 2, 1, 3).contiguous()
    k_hnd = k.permute(0, 2, 1, 3).contiguous()
    v_hnd = v.permute(0, 2, 1, 3).contiguous()
    out_hnd = F.scaled_dot_product_attention(q_hnd, k_hnd, v_hnd, is_causal=causal)
    return out_hnd.permute(0, 2, 1, 3).contiguous()


def compute_tflops(batch: int, heads: int, seq: int, dim: int, causal: bool, ms: float) -> float:
    flops = 4 * batch * heads * seq * seq * dim
    if causal:
        flops //= 2
    return flops / (ms * 1e-3) / 1e12


def add_repo_sources_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    candidate_paths = [
        root / "flash-attention",
        root / "flash-attention" / "hopper",
    ]
    for path in candidate_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def _as_varlen_nhd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    batch, seq, _, _ = q.shape
    cu_seqlens = torch.arange(0, (batch + 1) * seq, step=seq, device=q.device, dtype=torch.int32)
    q_flat = q.reshape(batch * seq, q.shape[2], q.shape[3]).contiguous()
    k_flat = k.reshape(batch * seq, k.shape[2], k.shape[3]).contiguous()
    v_flat = v.reshape(batch * seq, v.shape[2], v.shape[3]).contiguous()
    return q_flat, k_flat, v_flat, cu_seqlens, seq


def _fa_kvcache_wrapper(
    fa_with_kvcache: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    # Use empty cache + (k, v) append so each invocation is self-contained.
    k_cache = torch.empty_like(k)
    v_cache = torch.empty_like(v)
    return fa_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=0, causal=causal)


def build_methods(arch: str) -> tuple[list[BenchMethod], list[str]]:
    methods: list[BenchMethod] = []
    availability_notes: list[str] = []

    methods.append(
        BenchMethod(
            name="torch.sdpa",
            group="torch",
            fn=lambda q, k, v, causal: sdpa_reference(q, k, v, causal),
            note="PyTorch reference",
        )
    )

    flash_mod, flash_err = import_optional("flash_attn")
    if flash_mod is None:
        availability_notes.append(f"skip flash_attn (FA2 package): {flash_err}")
    else:
        if hasattr(flash_mod, "flash_attn_func"):
            methods.append(
                BenchMethod(
                    name="flash.fa2",
                    group="flash",
                    fn=lambda q, k, v, causal: flash_mod.flash_attn_func(q, k, v, causal=causal),
                )
            )
        if hasattr(flash_mod, "flash_attn_qkvpacked_func"):

            def fa2_qkvpacked(q, k, v, causal):
                qkv = torch.stack((q, k, v), dim=2)
                return flash_mod.flash_attn_qkvpacked_func(qkv, causal=causal)

            methods.append(BenchMethod(name="flash.fa2_qkvpacked", group="flash", fn=fa2_qkvpacked))
        if hasattr(flash_mod, "flash_attn_kvpacked_func"):

            def fa2_kvpacked(q, k, v, causal):
                kv = torch.stack((k, v), dim=2)
                return flash_mod.flash_attn_kvpacked_func(q, kv, causal=causal)

            methods.append(BenchMethod(name="flash.fa2_kvpacked", group="flash", fn=fa2_kvpacked))
        if hasattr(flash_mod, "flash_attn_varlen_func"):

            def fa2_varlen(q, k, v, causal):
                q_flat, k_flat, v_flat, cu_seqlens, max_seqlen = _as_varlen_nhd(q, k, v)
                out = flash_mod.flash_attn_varlen_func(
                    q_flat,
                    k_flat,
                    v_flat,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    causal=causal,
                )
                return out.reshape_as(q)

            methods.append(BenchMethod(name="flash.fa2_varlen", group="flash", fn=fa2_varlen))
        if hasattr(flash_mod, "flash_attn_varlen_qkvpacked_func"):

            def fa2_varlen_qkvpacked(q, k, v, causal):
                q_flat, k_flat, v_flat, cu_seqlens, max_seqlen = _as_varlen_nhd(q, k, v)
                qkv = torch.stack((q_flat, k_flat, v_flat), dim=1)
                out = flash_mod.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, causal=causal)
                return out.reshape_as(q)

            methods.append(BenchMethod(name="flash.fa2_varlen_qkvpacked", group="flash", fn=fa2_varlen_qkvpacked))
        if hasattr(flash_mod, "flash_attn_varlen_kvpacked_func"):

            def fa2_varlen_kvpacked(q, k, v, causal):
                q_flat, k_flat, v_flat, cu_seqlens, max_seqlen = _as_varlen_nhd(q, k, v)
                kv = torch.stack((k_flat, v_flat), dim=1)
                out = flash_mod.flash_attn_varlen_kvpacked_func(
                    q_flat,
                    kv,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    causal=causal,
                )
                return out.reshape_as(q)

            methods.append(BenchMethod(name="flash.fa2_varlen_kvpacked", group="flash", fn=fa2_varlen_kvpacked))
        if hasattr(flash_mod, "flash_attn_with_kvcache"):
            methods.append(
                BenchMethod(
                    name="flash.fa2_kvcache",
                    group="flash",
                    fn=lambda q, k, v, causal: _fa_kvcache_wrapper(
                        flash_mod.flash_attn_with_kvcache, q, k, v, causal
                    ),
                )
            )

    fa2_iface, fa2_iface_err = import_optional("flash_attn.flash_attn_interface")
    if fa2_iface is None:
        availability_notes.append(f"skip flash_attn.flash_attn_interface: {fa2_iface_err}")
    else:
        methods.append(
            BenchMethod(
                name="flash.fa2_interface",
                group="flash",
                fn=lambda q, k, v, causal: fa2_iface.flash_attn_func(q, k, v, causal=causal),
            )
        )
        if hasattr(fa2_iface, "flash_attn_varlen_func"):

            def fa2_interface_varlen(q, k, v, causal):
                q_flat, k_flat, v_flat, cu_seqlens, max_seqlen = _as_varlen_nhd(q, k, v)
                out = fa2_iface.flash_attn_varlen_func(
                    q_flat,
                    k_flat,
                    v_flat,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    causal=causal,
                )
                return out.reshape_as(q)

            methods.append(
                BenchMethod(
                    name="flash.fa2_interface_varlen",
                    group="flash",
                    fn=fa2_interface_varlen,
                )
            )
        if hasattr(fa2_iface, "flash_attn_with_kvcache"):
            methods.append(
                BenchMethod(
                    name="flash.fa2_interface_kvcache",
                    group="flash",
                    fn=lambda q, k, v, causal: _fa_kvcache_wrapper(
                        fa2_iface.flash_attn_with_kvcache, q, k, v, causal
                    ),
                )
            )

    fa_cute, fa_cute_err = import_optional("flash_attn.cute.interface")
    if fa_cute is None:
        availability_notes.append(f"skip flash_attn.cute.interface (CuTe/FA4 path): {fa_cute_err}")
    elif hasattr(fa_cute, "flash_attn_func"):
        methods.append(
            BenchMethod(
                name="flash.cute",
                group="flash",
                fn=lambda q, k, v, causal: fa_cute.flash_attn_func(q, k, v, causal=causal),
                note="CuTe interface",
            )
        )
    fa4_api, fa4_api_err = import_optional("flash_attn.cute")
    if fa4_api is None:
        availability_notes.append(f"skip flash_attn.cute (FA4 API path): {fa4_api_err}")
    elif hasattr(fa4_api, "flash_attn_func"):
        methods.append(
            BenchMethod(
                name="flash.cute_api",
                group="flash",
                fn=lambda q, k, v, causal: fa4_api.flash_attn_func(q, k, v, causal=causal),
                note="CuTe package API (from flash_attn.cute import flash_attn_func)",
            )
        )

    fa3_mod, fa3_err = import_optional("flash_attn_interface")
    if fa3_mod is None:
        availability_notes.append(f"skip flash_attn_interface (FA3 package): {fa3_err}")
    else:
        if hasattr(fa3_mod, "flash_attn_func"):
            methods.append(
                BenchMethod(
                    name="flash.fa3",
                    group="flash",
                    fn=lambda q, k, v, causal: fa3_mod.flash_attn_func(q, k, v, causal=causal),
                )
            )
        if hasattr(fa3_mod, "flash_attn_qkvpacked_func"):

            def fa3_qkvpacked(q, k, v, causal):
                qkv = torch.stack((q, k, v), dim=2)
                return fa3_mod.flash_attn_qkvpacked_func(qkv, causal=causal)

            methods.append(BenchMethod(name="flash.fa3_qkvpacked", group="flash", fn=fa3_qkvpacked))
        if hasattr(fa3_mod, "flash_attn_varlen_func"):

            def fa3_varlen(q, k, v, causal):
                q_flat, k_flat, v_flat, cu_seqlens, max_seqlen = _as_varlen_nhd(q, k, v)
                out = fa3_mod.flash_attn_varlen_func(
                    q_flat,
                    k_flat,
                    v_flat,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    causal=causal,
                )
                return out.reshape_as(q)

            methods.append(BenchMethod(name="flash.fa3_varlen", group="flash", fn=fa3_varlen))
        if hasattr(fa3_mod, "flash_attn_with_kvcache"):
            methods.append(
                BenchMethod(
                    name="flash.fa3_kvcache",
                    group="flash",
                    fn=lambda q, k, v, causal: _fa_kvcache_wrapper(
                        fa3_mod.flash_attn_with_kvcache, q, k, v, causal
                    ),
                )
            )
        if hasattr(fa3_mod, "flash_attn_func") and hasattr(torch, "float8_e4m3fn"):

            def fa3_fp8(q, k, v, causal):
                q_fp8 = q.to(torch.float8_e4m3fn)
                k_fp8 = k.to(torch.float8_e4m3fn)
                v_fp8 = v.to(torch.float8_e4m3fn)
                # FA3 expects 2D descale tensors, with Q using query heads and K/V using KV heads.
                q_descale = torch.ones((q.shape[0], q.shape[2]), dtype=torch.float32, device=q.device)
                kv_descale = torch.ones((k.shape[0], k.shape[2]), dtype=torch.float32, device=q.device)
                return fa3_mod.flash_attn_func(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    softmax_scale=q.shape[-1] ** -0.5,
                    causal=causal,
                    q_descale=q_descale,
                    k_descale=kv_descale,
                    v_descale=kv_descale,
                )

            methods.append(BenchMethod(name="flash.fa3_fp8", group="flash", fn=fa3_fp8, note="FA3 FP8 fwd"))

    sage_mod, sage_err = import_optional("sageattention")
    if sage_mod is None:
        availability_notes.append(f"skip sageattention: {sage_err}")
    else:
        if hasattr(sage_mod, "sageattn"):
            methods.append(
                BenchMethod(
                    name="sage.auto",
                    group="sage",
                    fn=lambda q, k, v, causal: sage_mod.sageattn(
                        q, k, v, tensor_layout="NHD", is_causal=causal
                    ),
                )
            )
        if hasattr(sage_mod, "sageattn_varlen"):

            def sage_varlen(q, k, v, causal):
                q_flat, k_flat, v_flat, cu_seqlens, max_seqlen = _as_varlen_nhd(q, k, v)
                out = sage_mod.sageattn_varlen(
                    q_flat,
                    k_flat,
                    v_flat,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    is_causal=causal,
                )
                return out.reshape_as(q)

            methods.append(BenchMethod(name="sage.varlen", group="sage", fn=sage_varlen))
        if hasattr(sage_mod, "sageattn_qk_int8_pv_fp16_triton"):
            methods.append(
                BenchMethod(
                    name="sage.fp16_triton",
                    group="sage",
                    fn=lambda q, k, v, causal: sage_mod.sageattn_qk_int8_pv_fp16_triton(
                        q, k, v, tensor_layout="NHD", is_causal=causal
                    ),
                )
            )
        if hasattr(sage_mod, "sageattn_qk_int8_pv_fp16_cuda"):
            for qgran in ("per_thread", "per_warp"):
                for accum in ("fp16", "fp16+fp32", "fp32"):
                    methods.append(
                        BenchMethod(
                            name=f"sage.fp16_cuda.{qgran}.{accum}",
                            group="sage",
                            fn=lambda q, k, v, causal, qgran=qgran, accum=accum: sage_mod.sageattn_qk_int8_pv_fp16_cuda(
                                q,
                                k,
                                v,
                                tensor_layout="NHD",
                                is_causal=causal,
                                qk_quant_gran=qgran,
                                pv_accum_dtype=accum,
                            ),
                        )
                    )
        if hasattr(sage_mod, "sageattn_qk_int8_pv_fp8_cuda"):
            for qgran in ("per_thread", "per_warp"):
                for accum in ("fp32", "fp32+fp32", "fp32+fp16"):
                    methods.append(
                        BenchMethod(
                            name=f"sage.fp8_cuda.{qgran}.{accum}",
                            group="sage",
                            fn=lambda q, k, v, causal, qgran=qgran, accum=accum: sage_mod.sageattn_qk_int8_pv_fp8_cuda(
                                q,
                                k,
                                v,
                                tensor_layout="NHD",
                                is_causal=causal,
                                qk_quant_gran=qgran,
                                pv_accum_dtype=accum,
                            ),
                        )
                    )
        if hasattr(sage_mod, "sageattn_qk_int8_pv_fp8_cuda_sm90"):
            for qgran in ("per_thread", "per_warp"):
                for accum in ("fp32", "fp32+fp32"):
                    methods.append(
                        BenchMethod(
                            name=f"sage.fp8_cuda_sm90.{qgran}.{accum}",
                            group="sage",
                            fn=lambda q, k, v, causal, qgran=qgran, accum=accum: sage_mod.sageattn_qk_int8_pv_fp8_cuda_sm90(
                                q,
                                k,
                                v,
                                tensor_layout="NHD",
                                is_causal=causal,
                                qk_quant_gran=qgran,
                                pv_accum_dtype=accum,
                            ),
                        )
                    )
    availability_notes.append(f"registered {len(methods)} candidate methods for arch={arch}")
    return methods, availability_notes


def should_keep_method(name: str, filter_expr: str) -> bool:
    if not filter_expr or filter_expr.lower() == "all":
        return True
    tokens = [tok.strip().lower() for tok in filter_expr.split(",") if tok.strip()]
    lower_name = name.lower()
    return any(tok in lower_name for tok in tokens)


def run_single_method(
    method: BenchMethod,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    warmup: int,
    iters: int,
    ref: Optional[torch.Tensor],
) -> BenchResult:
    try:
        out, ms = benchmark_kernel(method.fn, q, k, v, causal, warmup, iters)
        result = BenchResult(status="OK", ms=ms)
        if ref is not None:
            result.max_abs_err = (out.to(torch.float32) - ref).abs().max().item()
        return result
    except Exception as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            torch.cuda.empty_cache()
            return BenchResult(status="OOM", message=short_error(exc))
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return BenchResult(status="SKIP", message=short_error(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Best-effort attention benchmark matrix (FlashAttention / SageAttention variants).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-lens", type=str, default="1024,2048,4096")
    parser.add_argument(
        "--models-config",
        type=str,
        default=str(Path(__file__).resolve().with_name("model_dimensions.json")),
        help="Path to JSON model-dimension config.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated substrings to filter model names from --models-config (default: all).",
    )
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--causal-only",
        action="store_true",
        help="Run only causal attention shapes.",
    )
    parser.add_argument(
        "--include-causal",
        action="store_true",
        help="Run both non-causal and causal attention shapes (default runs non-causal only).",
    )
    parser.add_argument("--check", action="store_true", help="Compare outputs against torch SDPA (max abs error).")
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated substrings to filter method names (default: all).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="flash.fa2",
        help="Method name used for speedup column when available.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.causal_only and args.include_causal:
        raise ValueError("--causal-only and --include-causal cannot be used together.")

    add_repo_sources_to_path()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    arch = get_arch(device)
    dtype = parse_dtype(args.dtype)

    model_cfg_path = Path(args.models_config)
    if model_cfg_path.exists():
        model_configs = load_model_bench_configs(model_cfg_path)
        model_configs = [m for m in model_configs if should_keep_model(m.name, args.models)]
    else:
        cli_seq_lens = parse_seq_lens(args.seq_lens)
        print(f"[warn] model config not found at {model_cfg_path}; using CLI dimensions.")
        model_configs = [
            ModelBenchConfig(
                name="cli",
                batch_size=args.batch_size,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                seq_lens=cli_seq_lens,
                num_kv_heads=args.num_heads,
            )
        ]

    if not model_configs:
        raise RuntimeError("No model configs selected.")

    print(f"GPU arch: {arch}")
    print(
        f"Config: dtype={args.dtype}, methods={args.methods}, models={len(model_configs)}, "
        f"models_config={model_cfg_path}"
    )

    methods, notes = build_methods(arch)
    methods = [m for m in methods if should_keep_method(m.name, args.methods)]
    for note in notes:
        print(f"[info] {note}")
    print(f"[info] active methods: {len(methods)}")

    if not methods:
        raise RuntimeError("No methods selected.")

    if args.causal_only:
        causal_modes = [True]
    elif args.include_causal:
        causal_modes = [False, True]
    else:
        causal_modes = [False]
    for model in model_configs:
        kv_heads = model.num_kv_heads if model.num_kv_heads is not None else model.num_heads
        if model.num_heads % kv_heads != 0:
            raise ValueError(
                f"Model '{model.name}' has incompatible heads: num_heads={model.num_heads}, num_kv_heads={kv_heads}"
            )
        print(
            f"\nmodel={model.name} batch={model.batch_size} heads={model.num_heads} "
            f"kv_heads={kv_heads} head_dim={model.head_dim} seq_lens={model.seq_lens}"
        )

        for causal in causal_modes:
            for seq in model.seq_lens:
                q = torch.randn(model.batch_size, seq, model.num_heads, model.head_dim, device=device, dtype=dtype)
                k = torch.randn(model.batch_size, seq, kv_heads, model.head_dim, device=device, dtype=dtype)
                v = torch.randn_like(k)

                ref = None
                if args.check:
                    try:
                        ref = sdpa_reference(q, k, v, causal=causal).to(torch.float32)
                    except Exception as exc:
                        print(f"[warn] model={model.name} seq={seq} failed to build SDPA reference: {short_error(exc)}")
                        ref = None

                per_method: dict[str, BenchResult] = {}
                for method in methods:
                    res = run_single_method(method, q, k, v, causal, args.warmup, args.iters, ref)
                    if res.ms is not None:
                        res.tflops = compute_tflops(model.batch_size, model.num_heads, seq, model.head_dim, causal, res.ms)
                    per_method[method.name] = res

                baseline_ms = None
                baseline_name = None
                baseline_candidates = [args.baseline, "flash.fa2", "torch.sdpa"]
                for candidate in baseline_candidates:
                    if candidate in per_method and per_method[candidate].status == "OK" and per_method[candidate].ms is not None:
                        baseline_ms = per_method[candidate].ms
                        baseline_name = candidate
                        break

                print(
                    f"\n[table] model={model.name} seq={seq} causal={causal} dtype={args.dtype} "
                    f"batch={model.batch_size} heads={model.num_heads} kv_heads={kv_heads} head_dim={model.head_dim} "
                    f"warmup={args.warmup} iters={args.iters} baseline={baseline_name or 'n/a'} "
                    f"(requested={args.baseline})"
                )
                print("method | status | ms | tflops | speedup_vs_baseline | max_abs_err | note")
                print("-" * 140)
                sorted_methods = sorted(
                    methods,
                    key=lambda method: (
                        0 if per_method[method.name].tflops is not None else 1,
                        -(per_method[method.name].tflops or 0.0),
                        method.name,
                    ),
                )
                for method in sorted_methods:
                    res = per_method[method.name]
                    if res.status != "OK":
                        print(f"{method.name:32} | {res.status:5} | {'-':>8} | {'-':>8} | {'-':>17} | {'-':>11} | {res.message}")
                        continue
                    speedup = baseline_ms / res.ms if (baseline_ms is not None and res.ms and res.ms > 0) else None
                    ms_str = f"{res.ms:.3f}" if res.ms is not None else "-"
                    tflops_str = f"{res.tflops:.2f}" if res.tflops is not None else "-"
                    speedup_str = f"{speedup:.3f}" if speedup is not None else "-"
                    err_str = f"{res.max_abs_err:.4e}" if res.max_abs_err is not None else "n/a"
                    print(
                        f"{method.name:32} | {res.status:5} | {ms_str:>8} | {tflops_str:>8} | "
                        f"{speedup_str:>17} | {err_str:>11} | {method.note}"
                    )


if __name__ == "__main__":
    main()
