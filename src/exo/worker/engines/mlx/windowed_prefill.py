# SPDX-License-Identifier: MIT
"""Windowed (blocked) prefill attention for DeepSeek-V4.

Problem
-------
At one-shot prefill the layer builds a dense `(L, L)` causal+window mask and runs
a full `L x L` SDPA -- even though `sliding_window` is 128. Almost every score is
computed and then masked away, so a sliding-window model pays quadratic cost.

Fix
---
Block the queries. For query block `[i, j)` only keys within `window` of it can
be unmasked, so each block costs `block x (block + w)` instead of `block x L`:

    total  L * (block + w)   instead of   L * L
    L=8192, block=512, w=128  ->  640 vs 8192 keys  (~12.8x less attention work)

Masks are built per block from index arithmetic, so the dense `(L, L)` mask is
never materialised either.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

# Below this the dense path is already cheap and blocking only adds overhead.
MIN_PREFILL_LEN = 1024
# Measured on M3 Ultra at L=8192/16384: 256 edges out 512 (1.74x/3.25x vs
# 1.72x/3.20x) and clearly beats 1024. Smaller blocks waste less work on the
# window overhang; below 256 the per-block dispatch overhead starts to bite.
DEFAULT_BLOCK = 256

_PATCHED = False


@mx.compile
def _blocked_window_attention_compiled(
    q: mx.array,
    kv: mx.array,
    scale: float,
    window: int,
    q_start: int,
    kv_start: int,
    sinks: Optional[mx.array],
    block: int,
) -> mx.array:
    """Compiled blocked sliding-window attention - fuses the Python loop."""
    from mlx_lm.models.base import scaled_dot_product_attention

    L = q.shape[2]
    n_local = kv.shape[2]

    outs = []
    for i in range(0, L, block):
        j = min(i + block, L)
        lo = max(0, q_start + i - window + 1 - kv_start)
        hi = min(n_local, q_start + j - kv_start)
        if hi <= lo:
            outs.append(mx.zeros_like(q[:, :, i:j, :]))
            continue

        qa = mx.arange(q_start + i, q_start + j)[:, None]
        ka = mx.arange(kv_start + lo, kv_start + hi)[None, :]
        m = (ka <= qa) & (ka > qa - window)
        k_blk = kv[:, :, lo:hi, :]

        outs.append(
            scaled_dot_product_attention(
                q[:, :, i:j, :],
                k_blk,
                k_blk,
                cache=None,
                scale=scale,
                mask=m,
                sinks=sinks,
            )
        )

    return mx.concatenate(outs, axis=2)


def blocked_window_attention(
    q: mx.array,
    kv: mx.array,
    *,
    scale: float,
    window: int,
    q_start: int,
    kv_start: int,
    sinks: Optional[mx.array],
    n_local: int,
    block: int = DEFAULT_BLOCK,
) -> mx.array:
    """Blocked sliding-window attention.

    q         (B, H, L, D), queries at absolute positions `q_start + [0, L)`
    kv        (B, Hkv, n_local, D); local segment only
    kv_start  absolute position of `kv[..., 0, :]`
    """
    # Use compiled version to fuse the Python loop into single kernel dispatch
    return _blocked_window_attention_compiled(
        q, kv, scale, window, q_start, kv_start, sinks, block
    )


def apply(block: int = DEFAULT_BLOCK, min_len: int = MIN_PREFILL_LEN) -> bool:
    """Patch V4Attention with the blocked prefill path for rltakashige fork."""
    global _PATCHED
    if _PATCHED:
        return False

    import mlx_lm.models.deepseek_v4 as dsv4
    from mlx_lm.models.base import scaled_dot_product_attention

    # The rltakashige fork uses a unified V4Attention class, not separate
    # LocalAttention/CompressedAttention classes like ashhart's reference.
    v4_orig = dsv4.V4Attention.__call__

    def v4_call(self, x, cache=None):
        B, S, _ = x.shape
        
        # Skip blocking for short sequences or decode (S=1)
        if S < min_len or S == 1:
            return v4_orig(self, x, cache=cache)
        
        # Only apply blocked attention for non-compressor layers.
        # Compressor layers use compressed pool + visibility masks
        # which require the full SDPA path.
        if self.compress_ratio != 0:
            return v4_orig(self, x, cache=cache)

        # For prefill with long context, use blocked attention
        # We need to intercept after q/kv projection but before SDPA
        
        # Get cache offset
        v4_cache = cache if hasattr(cache, 'local') else None
        win_cache = v4_cache.local if v4_cache is not None else cache
        offset = int(win_cache.offset) if win_cache is not None else 0

        # Re-implement the forward pass with blocked attention
        rd = self.rope_head_dim

        # Fused: wqkv_a matmul + slice + 2 RMSNorms
        wqkv = self.wqkv_a
        if hasattr(wqkv, 'mode') and wqkv.mode == "mxfp4":
            from mlx_lm.models.deepseek_v4 import _attn_wqkv_quant_split_norm
            qr, kv = _attn_wqkv_quant_split_norm(
                x, wqkv.weight, wqkv.scales, wqkv.group_size, wqkv.bits,
                self.q_norm.weight, self.kv_norm.weight,
                self.q_lora_rank, self.eps,
            )
        else:
            from mlx_lm.models.deepseek_v4 import _attn_qkv_split_norm
            qkv_a = wqkv(x)
            qr, kv = _attn_qkv_split_norm(
                qkv_a, self.q_norm.weight, self.kv_norm.weight,
                self.q_lora_rank, self.eps,
            )

        # Fused: wq_b matmul + reshape/transpose + per-head RMSNorm
        wqb = self.wq_b
        if hasattr(wqb, 'mode') and wqb.mode == "mxfp4":
            from mlx_lm.models.deepseek_v4 import _attn_q_proj_quant_norm
            q = _attn_q_proj_quant_norm(
                qr, wqb.weight, wqb.scales, wqb.group_size, wqb.bits,
                self.n_heads, self.head_dim, self.eps,
            )
        else:
            from mlx_lm.models.deepseek_v4 import _attn_q_proj_norm
            q = _attn_q_proj_norm(wqb(qr), self.n_heads, self.head_dim, self.eps)

        # Fused: partial-RoPE on q + kv
        from mlx_lm.models.deepseek_v4 import _attn_qkv_partial_rope
        q, kv = _attn_qkv_partial_rope(q, kv, offset, rd, self.rope.freqs)

        # Get KV from cache
        k = kv[:, None, :, :]
        if win_cache is not None:
            k_ret, _ = win_cache.update_and_fetch(k, k)
            window_kv = k_ret.squeeze(1)
        else:
            window_kv = kv

        # Apply blocked window attention
        n_local = window_kv.shape[1]
        out = blocked_window_attention(
            q,
            window_kv,
            scale=self.scale,
            window=self.window,
            q_start=offset,
            kv_start=offset + S - n_local,
            sinks=self._sink_for(q.dtype),
            n_local=n_local,
            block=block,
        )

        # Fused: inverse-RoPE + transpose/flatten
        from mlx_lm.models.deepseek_v4 import _attn_inv_rope_flatten
        o = _attn_inv_rope_flatten(
            out, offset, rd, self.rope.freqs, self.n_heads * self.head_dim
        )

        # Output projection
        o = self._grouped_output_projection(o)
        return o

    dsv4.V4Attention.__call__ = v4_call
    _PATCHED = True
    return True
