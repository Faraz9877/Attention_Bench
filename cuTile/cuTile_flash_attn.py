import math
import cuda.tile as ct
from cuda.tile import RoundingMode as RMd
from types import SimpleNamespace
 
# Type aliases for compile-time constants
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]
 
# Conversion factor: we use exp2 instead of exp for efficiency
INV_LOG_2 = 1.0 / math.log(2)
 
@ct.kernel()
def fmha_kernel(
    Q, K, V, Out,              # Input/output tensors
    qk_scale: float,           # Scale factor (1/sqrt(d))
    input_pos: int,            # Position offset for causal masking
    TILE_D: ConstInt,          # Head dimension (for example, 128)
    H: ConstInt,               # Number of attention heads
    TILE_M: ConstInt,          # Tile size for Q dimension (for example, 64)
    TILE_N: ConstInt,          # Tile size for K/V dimension (for example, 64)
    QUERY_GROUP_SIZE: ConstInt,# For Grouped Query Attention (GQA)
    CAUSAL: ConstBool,         # Whether to apply causal mask
    EVEN_K: ConstBool,         # Whether K length is divisible by TILE_N
    NUM_M_BLOCKS: ConstInt     # Total number of M blocks (for causal masking logic)
):
    # Get block indices
    # bid_x = ct.bid(0)  # Which tile along the sequence dimension
    if CAUSAL:
        bid_x = NUM_M_BLOCKS - 1 - ct.bid(0)  # Process tiles N, N-1, N-2, ...
    else:
        bid_x = ct.bid(0)
    bid_y = ct.bid(1)  # Which batch-head combination
     
    # Decode batch and head from flattened index
    batch_idx = bid_y // H
    head_idx = bid_y % H
     
    # For Grouped Query Attention: multiple Q heads share one K/V head
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    # Convert scale for base-2 exponential (faster than natural exp)
    qk_scale = qk_scale * INV_LOG_2
     
    # Create position indices for this tile
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m += input_pos
    offs_m = offs_m[:, None]  # Shape: [TILE_M, 1]
     
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
    offs_n_tile = offs_n_tile[None, :]  # Shape: [1, TILE_N]
     
    # Online softmax state (float32 for numerical stability)
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)  # Running max
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)        # Running sum
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)   # Output accumulator

    # Load Q tile: shape [1, 1, TILE_M, TILE_D] -> [TILE_M, TILE_D]
    q = ct.load(
        Q, 
        index=(batch_idx, head_idx, bid_x, 0), 
        shape=(1, 1, TILE_M, TILE_D)
    ).reshape((TILE_M, TILE_D))

    # Calculate loop bounds
    m_end = input_pos + (bid_x + 1) * TILE_M
    k_seqlen = K.shape[2]
    
    # Calculate where masking starts being necessary
    mask_start = (input_pos + bid_x * TILE_M) // TILE_N
    mask_start = min(mask_start, k_seqlen // TILE_N)
    
    # Calculate where to stop (for causal, we exit early)
    if CAUSAL:
        # For causal attention, stop early (future tokens are masked)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
    
    for j in range(0, Tc):
        # --- Step A: Load Key tile and compute QK^T ---
        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),  # Transpose for correct layout
            latency=2            # Hint for memory prefetching
        ).reshape((TILE_D, TILE_N))
        
        # Matrix multiply: Q @ K^T
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)  # Uses Tensor Cores automatically

        # --- Step B: Apply causal masking ---
        # if CAUSAL or not EVEN_K:
        #     offs_n = j * TILE_N + offs_n_tile
        #     mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)
             
        #     # Boundary mask (for non-divisible sequence lengths)
        #     if not EVEN_K:
        #         mask = mask & (offs_n < k_seqlen)
             
        #     # Causal mask: query position >= key position
        #     if CAUSAL:
        #         mask = mask & (offs_m >= offs_n)
             
        #     # Convert to additive mask: True->0, False->-inf
        #     mask = ct.where(mask, 0.0, -math.inf)
        #     qk += mask
        
            # ONLY apply masking when necessary
        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)
            mask = ct.where(mask, 0.0, -math.inf)
            qk += mask

        # --- Step C: Online softmax ---
        # Find max in current tile
        qk_max = ct.max(qk, axis=-1, keepdims=True)
        qk_max_scaled = qk_max * qk_scale
        
        # Update running maximum
        m_ij = max(m_i, qk_max_scaled)
        
        # Scale QK scores
        qk = qk * qk_scale
        qk = qk - m_ij
        
        # Compute attention weights (using exp2 for speed)
        p = ct.exp2(qk, flush_to_zero=True)
        
        # Update running sum
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # Correction factor
        l_i = l_i * alpha
        l_i = l_i + l_ij
        
        # Rescale previous accumulator
        acc = acc * alpha

        # --- Step D: Load V and accumulate ---
        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            latency=4
        ).reshape((TILE_N, TILE_D))
         
        # Cast attention weights back to input dtype for Tensor Core MMA
        p = p.astype(Q.dtype)
         
        # Accumulate: acc += P @ V
        acc = ct.mma(p, v, acc)
         
        # Update max for next iteration
        m_i = m_ij
    
    # --- Final: Normalize and store ---
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


def _fmha_autotune_configs():
    """Search space for autotuning.
 
    The autotuner will benchmark these configurations and cache the best one
    per input shape (sequence length, batch size, etc.).
    """
    gpu_capability = torch.cuda.get_device_capability()
 
    if gpu_capability in [(12, 0), (12, 1)]:
        # RTX 50 series (sm120, sm121)
        yield SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2)
    else:
        # B200/GB200 (sm100) - Try multiple tile sizes
        # Autotuner will discover:
        # - 64x64 is best for short sequences (1024-2048)
        # - 128x128 may be best for medium sequences (4096)
        # - 256x128 is best for long sequences (8192+)
        yield SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_M=256, TILE_N=128, num_ctas=1, occupancy=1)


import cuda.tile_experimental as ct_experimental

def autotune_launch_fmha(
    stream, q, k, v, o, sm_scale, input_pos,
    hidden_size, num_heads, query_group_size, is_causal
):
    batch_size, _, q_len, _ = q.shape
 
    def _grid_fn(cfg):
        return (math.ceil(q_len / cfg.TILE_M), batch_size * num_heads, 1)
 
    def _args_fn(cfg):
        num_m_blocks = math.ceil(q_len / cfg.TILE_M)
        even_k = (k.shape[2] % cfg.TILE_N) == 0
        return (
            q, k, v, o, sm_scale, input_pos,
            hidden_size, num_heads, cfg.TILE_M, cfg.TILE_N,
            query_group_size, is_causal, even_k, num_m_blocks,
        )
 
    ct_experimental.autotune_launch(
        stream,
        grid_fn=_grid_fn,
        kernel=fmha_kernel,
        args_fn=_args_fn,
        hints_fn=lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy},
        search_space=_fmha_autotune_configs,
    )


import torch
from math import ceil
 
def tile_fmha(q, k, v, sm_scale=None, is_causal=False):
    """
    Launch the Flash Attention kernel.
     
    Args:
        q: Query tensor, shape [batch, heads, seq_len, head_dim]
        k: Key tensor, shape [batch, kv_heads, seq_len, head_dim]
        v: Value tensor, shape [batch, kv_heads, seq_len, head_dim]
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))
        is_causal: Whether to apply causal masking
     
    Returns:
        Output tensor, same shape as q
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    
    batch_size, num_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape
    
    # Calculate query group size for GQA
    query_group_size = num_heads // num_kv_heads
    
    # Ensure contiguous memory layout
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Allocate output
    o = torch.empty_like(q)
    
    # Launch kernel with autotuning
    autotune_launch_fmha(
    torch.cuda.current_stream(), q, k, v, o, sm_scale, 0,
    head_dim, num_heads, query_group_size, is_causal
)
     
    return o

if __name__ == "__main__":
    def benchmark_tile_fmha(
        q,
        k,
        v,
        is_causal=False,
        warmup=5,
        iters=20,
    ):
        for _ in range(warmup):
            tile_fmha(q, k, v, is_causal=is_causal)

        torch.cuda.synchronize()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(iters):
            tile_fmha(q, k, v, is_causal=is_causal)
        end_evt.record()

        torch.cuda.synchronize()
        avg_ms = start_evt.elapsed_time(end_evt) / iters
        return avg_ms

    # Simple benchmark case
    batch, heads, head_dim = 4, 32, 128
    seq_lens = [1024] # [1024, 2048, 4096, 8192, 16384]

    # Create MHA inputs with varying sequence lengths without using GQA (num_kv_heads = num_heads)
    for seq_len in seq_lens:
        print(f"\nTesting sequence length: {seq_len}")
        q = torch.randn(batch, heads, seq_len, head_dim, device="cuda")
        k = torch.randn(batch, heads, seq_len, head_dim, device="cuda")  # No GQA: num_kv_heads = num_heads
        v = torch.randn(batch, heads, seq_len, head_dim, device="cuda")

        best_cfg = None
        avg_ms = benchmark_tile_fmha(
            q,
            k,
            v,
            is_causal=False,
        )

        print(f"  Elapsed_time ({avg_ms:.3f} ms), tflops: {2 * batch * heads * seq_len * head_dim * seq_len / (avg_ms * 1e6):.2f} TFLOPS")
