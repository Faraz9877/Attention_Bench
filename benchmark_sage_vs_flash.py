#!/usr/bin/env python3
import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F


TensorFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]


@dataclass
class BenchMethod:
    name: str
    group: str
    fn: TensorFn
    input_format: str = "NHD[B,S,H,D]"
    output_format: str = "NHD"
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
    # Always run one untimed setup call so one-time input preparation is never charged to kernel runtime.
    out = unwrap_output(fn(q, k, v, causal))
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


def _make_cache_key(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[Any, ...]:
    return (
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        q.shape,
        k.shape,
        v.shape,
        q.dtype,
        k.dtype,
        v.dtype,
        q.device,
        k.device,
        v.device,
    )


def _cached_prepare(
    prepare: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any],
    run_prepared: Callable[[Any, bool], torch.Tensor],
) -> TensorFn:
    cache_key: Optional[tuple[Any, ...]] = None
    cache_value: Any = None

    def wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
        nonlocal cache_key, cache_value
        key = _make_cache_key(q, k, v)
        if key != cache_key:
            cache_value = prepare(q, k, v)
            cache_key = key
        return run_prepared(cache_value, causal)

    return wrapper


def _to_hnd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    make_contiguous: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_hnd = q.permute(0, 2, 1, 3)
    k_hnd = k.permute(0, 2, 1, 3)
    v_hnd = v.permute(0, 2, 1, 3)
    if make_contiguous:
        q_hnd = q_hnd.contiguous()
        k_hnd = k_hnd.contiguous()
        v_hnd = v_hnd.contiguous()
    return q_hnd, k_hnd, v_hnd


def build_methods(arch: str) -> tuple[list[BenchMethod], list[str]]:
    methods: list[BenchMethod] = []
    availability_notes: list[str] = []

    def _prepare_nhd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return q, k, v

    def _add_sage_layout_variants(
        base_name: str,
        make_call: Callable[[str], Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]],
        note: str = "",
    ) -> None:
        nhd_call = make_call("NHD")
        hnd_call = make_call("HND")
        methods.append(
            BenchMethod(
                name=f"{base_name}.nhd",
                group="sage",
                fn=_cached_prepare(
                    _prepare_nhd,
                    lambda prepared, causal, nhd_call=nhd_call: nhd_call(
                        prepared[0], prepared[1], prepared[2], causal
                    ),
                ),
                input_format="NHD[B,S,H,D]",
                output_format="NHD",
                note=note,
            )
        )
        methods.append(
            BenchMethod(
                name=f"{base_name}.hnd",
                group="sage",
                fn=_cached_prepare(
                    lambda q, k, v: _to_hnd(q, k, v),
                    lambda prepared, causal, hnd_call=hnd_call: hnd_call(
                        prepared[0], prepared[1], prepared[2], causal
                    ),
                ),
                input_format="HND[B,H,S,D]",
                output_format="HND",
                note=note,
            )
        )

    methods.append(
        BenchMethod(
            name="torch.sdpa",
            group="torch",
            fn=_cached_prepare(
                lambda q, k, v: _to_hnd(q, k, v),
                lambda prepared, causal: F.scaled_dot_product_attention(
                    prepared[0], prepared[1], prepared[2], is_causal=causal
                ),
            ),
            input_format="HND[B,H,S,D]",
            output_format="HND",
            note="PyTorch reference",
        )
    )

    tile_mod, tile_err = import_optional("cuTile.cuTile_flash_attn")
    if tile_mod is None:
        availability_notes.append(f"skip cuTile tile_fmha: {tile_err}")
    elif hasattr(tile_mod, "tile_fmha"):
        methods.append(
            BenchMethod(
                name="tile.fmha",
                group="tile",
                fn=_cached_prepare(
                    lambda q, k, v: _to_hnd(q, k, v),
                    lambda prepared, causal: tile_mod.tile_fmha(
                        prepared[0], prepared[1], prepared[2], is_causal=causal
                    ),
                ),
                input_format="HND[B,H,S,D]",
                output_format="HND",
                note="cuTile FlashAttention kernel",
            )
        )
    else:
        availability_notes.append("skip cuTile tile_fmha: missing tile_fmha symbol")

    flash_mod, flash_err = import_optional("flash_attn")
    if flash_mod is None:
        availability_notes.append(f"skip flash_attn (FA2 package): {flash_err}")
    else:
        if hasattr(flash_mod, "flash_attn_func"):
            methods.append(
                BenchMethod(
                    name="flash.fa2",
                    group="flash",
                    fn=_cached_prepare(
                        _prepare_nhd,
                        lambda prepared, causal: flash_mod.flash_attn_func(
                            prepared[0], prepared[1], prepared[2], causal=causal
                        ),
                    ),
                    input_format="NHD[B,S,H,D]",
                )
            )
        if hasattr(flash_mod, "flash_attn_qkvpacked_func"):
            methods.append(
                BenchMethod(
                    name="flash.fa2_qkvpacked",
                    group="flash",
                    fn=_cached_prepare(
                        lambda q, k, v: torch.stack((q, k, v), dim=2).contiguous(),
                        lambda qkv, causal: flash_mod.flash_attn_qkvpacked_func(qkv, causal=causal),
                    ),
                    input_format="NHD_qkv[B,S,3,H,D]",
                )
            )
        if hasattr(flash_mod, "flash_attn_kvpacked_func"):
            methods.append(
                BenchMethod(
                    name="flash.fa2_kvpacked",
                    group="flash",
                    fn=_cached_prepare(
                        lambda q, k, v: (q, torch.stack((k, v), dim=2).contiguous()),
                        lambda prepared, causal: flash_mod.flash_attn_kvpacked_func(
                            prepared[0], prepared[1], causal=causal
                        ),
                    ),
                    input_format="NHD_kv[B,S,2,H,D]",
                )
            )
        if hasattr(flash_mod, "flash_attn_varlen_func"):
            methods.append(
                BenchMethod(
                    name="flash.fa2_varlen",
                    group="flash",
                    fn=_cached_prepare(
                        _as_varlen_nhd,
                        lambda prepared, causal: flash_mod.flash_attn_varlen_func(
                            prepared[0],
                            prepared[1],
                            prepared[2],
                            prepared[3],
                            prepared[3],
                            prepared[4],
                            prepared[4],
                            causal=causal,
                        ),
                    ),
                    input_format="varlen[total,H,D]",
                    output_format="varlen",
                )
            )
        if hasattr(flash_mod, "flash_attn_varlen_qkvpacked_func"):
            def _prepare_fa2_varlen_qkvpacked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
                q_flat, k_flat, v_flat, cu_seqlens, max_seqlen = _as_varlen_nhd(q, k, v)
                qkv = torch.stack((q_flat, k_flat, v_flat), dim=1).contiguous()
                return qkv, cu_seqlens, max_seqlen

            methods.append(
                BenchMethod(
                    name="flash.fa2_varlen_qkvpacked",
                    group="flash",
                    fn=_cached_prepare(
                        _prepare_fa2_varlen_qkvpacked,
                        lambda prepared, causal: flash_mod.flash_attn_varlen_qkvpacked_func(
                            prepared[0],
                            prepared[1],
                            prepared[2],
                            causal=causal,
                        ),
                    ),
                    input_format="varlen_qkv[total,3,H,D]",
                    output_format="varlen",
                )
            )
        if hasattr(flash_mod, "flash_attn_varlen_kvpacked_func"):
            def _prepare_fa2_varlen_kvpacked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
                q_flat, k_flat, v_flat, cu_seqlens, max_seqlen = _as_varlen_nhd(q, k, v)
                kv = torch.stack((k_flat, v_flat), dim=1).contiguous()
                return q_flat, kv, cu_seqlens, max_seqlen

            methods.append(
                BenchMethod(
                    name="flash.fa2_varlen_kvpacked",
                    group="flash",
                    fn=_cached_prepare(
                        _prepare_fa2_varlen_kvpacked,
                        lambda prepared, causal: flash_mod.flash_attn_varlen_kvpacked_func(
                            prepared[0],
                            prepared[1],
                            prepared[2],
                            prepared[2],
                            prepared[3],
                            prepared[3],
                            causal=causal,
                        ),
                    ),
                    input_format="varlen_kv[total,2,H,D]",
                    output_format="varlen",
                )
            )
        if hasattr(flash_mod, "flash_attn_with_kvcache"):
            methods.append(
                BenchMethod(
                    name="flash.fa2_kvcache",
                    group="flash",
                    fn=_cached_prepare(
                        lambda q, k, v: (q, k, v, torch.empty_like(k), torch.empty_like(v)),
                        lambda prepared, causal: flash_mod.flash_attn_with_kvcache(
                            prepared[0],
                            prepared[3],
                            prepared[4],
                            k=prepared[1],
                            v=prepared[2],
                            cache_seqlens=0,
                            causal=causal,
                        ),
                    ),
                    input_format="NHD[B,S,H,D]+cache",
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
                fn=_cached_prepare(
                    _prepare_nhd,
                    lambda prepared, causal: fa2_iface.flash_attn_func(
                        prepared[0], prepared[1], prepared[2], causal=causal
                    ),
                ),
                input_format="NHD[B,S,H,D]",
            )
        )
        if hasattr(fa2_iface, "flash_attn_varlen_func"):
            methods.append(
                BenchMethod(
                    name="flash.fa2_interface_varlen",
                    group="flash",
                    fn=_cached_prepare(
                        _as_varlen_nhd,
                        lambda prepared, causal: fa2_iface.flash_attn_varlen_func(
                            prepared[0],
                            prepared[1],
                            prepared[2],
                            prepared[3],
                            prepared[3],
                            prepared[4],
                            prepared[4],
                            causal=causal,
                        ),
                    ),
                    input_format="varlen[total,H,D]",
                    output_format="varlen",
                )
            )
        if hasattr(fa2_iface, "flash_attn_with_kvcache"):
            methods.append(
                BenchMethod(
                    name="flash.fa2_interface_kvcache",
                    group="flash",
                    fn=_cached_prepare(
                        lambda q, k, v: (q, k, v, torch.empty_like(k), torch.empty_like(v)),
                        lambda prepared, causal: fa2_iface.flash_attn_with_kvcache(
                            prepared[0],
                            prepared[3],
                            prepared[4],
                            k=prepared[1],
                            v=prepared[2],
                            cache_seqlens=0,
                            causal=causal,
                        ),
                    ),
                    input_format="NHD[B,S,H,D]+cache",
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
                fn=_cached_prepare(
                    _prepare_nhd,
                    lambda prepared, causal: fa_cute.flash_attn_func(
                        prepared[0], prepared[1], prepared[2], causal=causal
                    ),
                ),
                input_format="NHD[B,S,H,D]",
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
                fn=_cached_prepare(
                    _prepare_nhd,
                    lambda prepared, causal: fa4_api.flash_attn_func(
                        prepared[0], prepared[1], prepared[2], causal=causal
                    ),
                ),
                input_format="NHD[B,S,H,D]",
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
                    fn=_cached_prepare(
                        _prepare_nhd,
                        lambda prepared, causal: fa3_mod.flash_attn_func(
                            prepared[0], prepared[1], prepared[2], causal=causal
                        ),
                    ),
                    input_format="NHD[B,S,H,D]",
                )
            )
        if hasattr(fa3_mod, "flash_attn_qkvpacked_func"):
            methods.append(
                BenchMethod(
                    name="flash.fa3_qkvpacked",
                    group="flash",
                    fn=_cached_prepare(
                        lambda q, k, v: torch.stack((q, k, v), dim=2).contiguous(),
                        lambda qkv, causal: fa3_mod.flash_attn_qkvpacked_func(qkv, causal=causal),
                    ),
                    input_format="NHD_qkv[B,S,3,H,D]",
                )
            )
        if hasattr(fa3_mod, "flash_attn_varlen_func"):
            methods.append(
                BenchMethod(
                    name="flash.fa3_varlen",
                    group="flash",
                    fn=_cached_prepare(
                        _as_varlen_nhd,
                        lambda prepared, causal: fa3_mod.flash_attn_varlen_func(
                            prepared[0],
                            prepared[1],
                            prepared[2],
                            prepared[3],
                            prepared[3],
                            prepared[4],
                            prepared[4],
                            causal=causal,
                        ),
                    ),
                    input_format="varlen[total,H,D]",
                    output_format="varlen",
                )
            )
        if hasattr(fa3_mod, "flash_attn_with_kvcache"):
            methods.append(
                BenchMethod(
                    name="flash.fa3_kvcache",
                    group="flash",
                    fn=_cached_prepare(
                        lambda q, k, v: (q, k, v, torch.empty_like(k), torch.empty_like(v)),
                        lambda prepared, causal: fa3_mod.flash_attn_with_kvcache(
                            prepared[0],
                            prepared[3],
                            prepared[4],
                            k=prepared[1],
                            v=prepared[2],
                            cache_seqlens=0,
                            causal=causal,
                        ),
                    ),
                    input_format="NHD[B,S,H,D]+cache",
                )
            )
        if hasattr(fa3_mod, "flash_attn_func") and hasattr(torch, "float8_e4m3fn"):
            methods.append(
                BenchMethod(
                    name="flash.fa3_fp8",
                    group="flash",
                    fn=_cached_prepare(
                        lambda q, k, v: (
                            q.to(torch.float8_e4m3fn),
                            k.to(torch.float8_e4m3fn),
                            v.to(torch.float8_e4m3fn),
                            torch.ones((q.shape[0], q.shape[2]), dtype=torch.float32, device=q.device),
                            torch.ones((k.shape[0], k.shape[2]), dtype=torch.float32, device=q.device),
                            q.shape[-1] ** -0.5,
                        ),
                        lambda prepared, causal: fa3_mod.flash_attn_func(
                            prepared[0],
                            prepared[1],
                            prepared[2],
                            softmax_scale=prepared[5],
                            causal=causal,
                            q_descale=prepared[3],
                            k_descale=prepared[4],
                            v_descale=prepared[4],
                        ),
                    ),
                    input_format="NHD_fp8[B,S,H,D]",
                    note="FA3 FP8 fwd",
                )
            )

    sage_mod, sage_err = import_optional("sageattention")
    if sage_mod is None:
        availability_notes.append(f"skip sageattention: {sage_err}")
    else:
        if hasattr(sage_mod, "sageattn"):
            _add_sage_layout_variants(
                "sage.auto",
                lambda layout: lambda q, k, v, causal: sage_mod.sageattn(
                    q, k, v, tensor_layout=layout, is_causal=causal
                ),
            )
        if hasattr(sage_mod, "sageattn_varlen"):
            methods.append(
                BenchMethod(
                    name="sage.varlen",
                    group="sage",
                    fn=_cached_prepare(
                        _as_varlen_nhd,
                        lambda prepared, causal: sage_mod.sageattn_varlen(
                            prepared[0],
                            prepared[1],
                            prepared[2],
                            cu_seqlens_q=prepared[3],
                            cu_seqlens_k=prepared[3],
                            max_seqlen_q=prepared[4],
                            max_seqlen_k=prepared[4],
                            is_causal=causal,
                        ),
                    ),
                    input_format="varlen[total,H,D]",
                    output_format="varlen",
                )
            )
        if hasattr(sage_mod, "sageattn_qk_int8_pv_fp16_triton"):
            _add_sage_layout_variants(
                "sage.fp16_triton",
                lambda layout: lambda q, k, v, causal: sage_mod.sageattn_qk_int8_pv_fp16_triton(
                    q, k, v, tensor_layout=layout, is_causal=causal
                ),
            )
        if hasattr(sage_mod, "sageattn_qk_int8_pv_fp16_cuda"):
            for qgran in ("per_thread", "per_warp"):
                for accum in ("fp16", "fp16+fp32", "fp32"):
                    _add_sage_layout_variants(
                        f"sage.fp16_cuda.{qgran}.{accum}",
                        lambda layout, qgran=qgran, accum=accum: (
                            lambda q, k, v, causal: sage_mod.sageattn_qk_int8_pv_fp16_cuda(
                                q,
                                k,
                                v,
                                tensor_layout=layout,
                                is_causal=causal,
                                qk_quant_gran=qgran,
                                pv_accum_dtype=accum,
                            )
                        ),
                    )
        if hasattr(sage_mod, "sageattn_qk_int8_pv_fp8_cuda"):
            for qgran in ("per_thread", "per_warp"):
                for accum in ("fp32", "fp32+fp32", "fp32+fp16"):
                    _add_sage_layout_variants(
                        f"sage.fp8_cuda.{qgran}.{accum}",
                        lambda layout, qgran=qgran, accum=accum: (
                            lambda q, k, v, causal: sage_mod.sageattn_qk_int8_pv_fp8_cuda(
                                q,
                                k,
                                v,
                                tensor_layout=layout,
                                is_causal=causal,
                                qk_quant_gran=qgran,
                                pv_accum_dtype=accum,
                            )
                        ),
                    )
        if hasattr(sage_mod, "sageattn_qk_int8_pv_fp8_cuda_sm90"):
            for qgran in ("per_thread", "per_warp"):
                for accum in ("fp32", "fp32+fp32"):
                    _add_sage_layout_variants(
                        f"sage.fp8_cuda_sm90.{qgran}.{accum}",
                        lambda layout, qgran=qgran, accum=accum: (
                            lambda q, k, v, causal: sage_mod.sageattn_qk_int8_pv_fp8_cuda_sm90(
                                q,
                                k,
                                v,
                                tensor_layout=layout,
                                is_causal=causal,
                                qk_quant_gran=qgran,
                                pv_accum_dtype=accum,
                            )
                        ),
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
            if method.output_format == "NHD":
                out_ref = out
            elif method.output_format == "HND":
                out_ref = out.permute(0, 2, 1, 3).contiguous()
            elif method.output_format == "varlen":
                out_ref = out.reshape_as(q)
            else:
                raise ValueError(f"Unsupported output format for {method.name}: {method.output_format}")
            result.max_abs_err = (out_ref.to(torch.float32) - ref).abs().max().item()
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
                print("method | input_format | status | ms | tflops | speedup_vs_baseline | max_abs_err | note")
                print("-" * 180)
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
                        print(
                            f"{method.name:32} | {method.input_format:22} | {res.status:5} | {'-':>8} | {'-':>8} | "
                            f"{'-':>17} | {'-':>11} | {res.message}"
                        )
                        continue
                    speedup = baseline_ms / res.ms if (baseline_ms is not None and res.ms and res.ms > 0) else None
                    ms_str = f"{res.ms:.3f}" if res.ms is not None else "-"
                    tflops_str = f"{res.tflops:.2f}" if res.tflops is not None else "-"
                    speedup_str = f"{speedup:.3f}" if speedup is not None else "-"
                    err_str = f"{res.max_abs_err:.4e}" if res.max_abs_err is not None else "n/a"
                    print(
                        f"{method.name:32} | {method.input_format:22} | {res.status:5} | {ms_str:>8} | {tflops_str:>8} | "
                        f"{speedup_str:>17} | {err_str:>11} | {method.note}"
                    )


if __name__ == "__main__":
    main()
