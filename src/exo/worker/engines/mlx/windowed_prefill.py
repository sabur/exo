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
    """Legacy compatibility hook; native MLX-LM V4 attention owns prefill."""
    return False
