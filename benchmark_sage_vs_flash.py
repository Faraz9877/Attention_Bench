#!/usr/bin/env python3
import argparse
import importlib
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
    ]
    for path in candidate_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


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

    sage_mod, sage_err = import_optional("sageattention")
    if sage_mod is None:
        availability_notes.append(f"skip sageattention: {sage_err}")
    else:
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
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--causal-only", action="store_true")
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

    add_repo_sources_to_path()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    arch = get_arch(device)
    dtype = parse_dtype(args.dtype)
    seq_lens = parse_seq_lens(args.seq_lens)

    print(f"GPU arch: {arch}")
    print(
        f"Config: batch={args.batch_size}, heads={args.num_heads}, head_dim={args.head_dim}, "
        f"dtype={args.dtype}, seq_lens={seq_lens}, methods={args.methods}"
    )

    methods, notes = build_methods(arch)
    methods = [m for m in methods if should_keep_method(m.name, args.methods)]
    for note in notes:
        print(f"[info] {note}")
    print(f"[info] active methods: {len(methods)}")

    if not methods:
        raise RuntimeError("No methods selected.")

    causal_modes = [True] if args.causal_only else [False, True]
    for causal in causal_modes:
        print(f"\ncausal={causal}")
        for seq in seq_lens:
            q = torch.randn(args.batch_size, seq, args.num_heads, args.head_dim, device=device, dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            ref = None
            if args.check:
                try:
                    ref = sdpa_reference(q, k, v, causal=causal).to(torch.float32)
                except Exception as exc:
                    print(f"[warn] seq={seq} failed to build SDPA reference: {short_error(exc)}")
                    ref = None

            per_method: dict[str, BenchResult] = {}
            for method in methods:
                res = run_single_method(method, q, k, v, causal, args.warmup, args.iters, ref)
                if res.ms is not None:
                    res.tflops = compute_tflops(args.batch_size, args.num_heads, seq, args.head_dim, causal, res.ms)
                per_method[method.name] = res

            baseline_ms = None
            baseline_candidates = [args.baseline, "flash.fa2", "torch.sdpa"]
            for candidate in baseline_candidates:
                if candidate in per_method and per_method[candidate].status == "OK" and per_method[candidate].ms is not None:
                    baseline_ms = per_method[candidate].ms
                    break

            print(f"\nseq={seq}")
            print("method | status | ms | tflops | speedup_vs_baseline | max_abs_err | note")
            print("-" * 140)
            for method in methods:
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
