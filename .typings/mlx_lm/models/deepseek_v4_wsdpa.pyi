"""Type stubs for mlx_lm.models.deepseek_v4_wsdpa."""

from typing import Optional

import mlx.core as mx

def wsdpa_prefill_route_active(*, topk: bool = False) -> bool: ...
def wsdpa_prefill(
    q: mx.array,
    kv: mx.array,
    pooled: Optional[mx.array],
    sinks: mx.array,
    scale: float,
    offset: int,
    window: int,
    ratio: int,
) -> Optional[mx.array]: ...
def wsdpa_topk_prefill(
    q: mx.array,
    kv: mx.array,
    pooled: mx.array,
    topk: mx.array,
    sinks: mx.array,
    scale: float,
    offset: int,
    window: int,
    ratio: int,
) -> Optional[mx.array]: ...
