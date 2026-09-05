import gc
import os
from copy import deepcopy
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
import psutil
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.models.deepseek_v4 import (
    DeepseekV4Cache,
)
from mlx_lm.models.deepseek_v4 import (
    _CompressorBranch as CompressorBranch,  # type: ignore
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.memory import Memory
from exo.worker.engines.mlx.constants import CACHE_GROUP_SIZE, KV_CACHE_BITS
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import MediaRegion


# Fraction of device memory above which LRU eviction kicks in.
# Smaller machines need more aggressive eviction.
def _default_memory_threshold() -> float:
    total_gb = Memory.from_bytes(psutil.virtual_memory().total).in_gb
    if total_gb >= 128:
        return 0.85
    if total_gb >= 64:
        return 0.80
    if total_gb >= 32:
        return 0.75
    return 0.70


_MEMORY_THRESHOLD = float(
    os.environ.get("EXO_MEMORY_THRESHOLD", _default_memory_threshold())
)


def _read_non_negative_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return value


_V4_PREFIX_CACHE_MAX_ENTRIES = _read_non_negative_int_env(
    "EXO_DEEPSEEK_V4_PREFIX_CACHE_MAX_ENTRIES", 8
)

# Retain fixed logarithmic anchors plus the two tail-safe rollback points and
# exact pre-generation state. The anchor count grows only logarithmically.
_V4_PREFIX_CACHE_FIRST_LANDMARK_TOKENS = 10_000
_V4_PREFIX_CACHE_TAIL_SNAPSHOT_COUNT = 3


class CacheSnapshot:
    """Snapshot of states at a known token position."""

    def __init__(
        self,
        states: list[
            RotatingKVCache | ArraysCache | CacheList | DeepseekV4Cache | None
        ],
        token_count: int,
    ):
        self.states = states
        self.token_count = token_count

    @property
    def nbytes(self) -> int:
        return sum(_cache_state_nbytes(state) for state in self.states)


def _cache_state_nbytes(state: object | None) -> int:
    if state is None:
        return 0
    if isinstance(state, ArraysCache):
        return sum(
            int(entry.nbytes)
            for entry in state.cache  # type: ignore[reportUnknownMemberType]
            if isinstance(entry, mx.array)
        )
    if isinstance(state, CacheList):
        return sum(_cache_state_nbytes(entry) for entry in state)
    return int(getattr(state, "nbytes", 0))


def _detached_copy(a: mx.array) -> mx.array:
    """Create an independent copy of an mlx array without numpy round-trip.
    
    Uses astype(dtype) which forces a copy while preserving the dtype.
    This keeps data on GPU and avoids expensive CPU transfer.
    
    Args:
        a: Input mlx array
    
    Returns:
        Independent copy of the array (same dtype, different memory)
    """
    # astype(dtype) forces a copy without changing dtype
    # This breaks shared_ptr chains while staying on GPU
    return a.astype(a.dtype)


def copy_rotating_kv_cache(cache: RotatingKVCache) -> RotatingKVCache | None:
    """
    Deepcopy copies the metadata associated with an mx array.
    Specifically, it shares a shared_ptr to the underlying data and
    the mlx graph inputs of the array. This causes a memory leak for rotating
    kv cache. By creating an np array, no metadata is stored so the old cache
    can be cleaned up nicely.
    """
    if cache.keys is None or cache.values is None:
        return None
    n = min(cache.max_size, cache.keys.shape[2])
    k_slice = _detached_copy(cache.keys[..., -n:, :])
    v_slice = _detached_copy(cache.values[..., -n:, :])
    mx.eval(k_slice, v_slice)
    snap = RotatingKVCache.__new__(RotatingKVCache)
    snap.keys = k_slice
    snap.values = v_slice
    snap.offset = cache.offset
    snap._idx = n
    snap.keep = cache.keep
    snap.max_size = cache.max_size
    return snap


def _copy_arrays_cache(ac: ArraysCache) -> ArraysCache:
    entries: list[mx.array | None] = []
    for entry in ac.cache:  # type: ignore[reportUnknownMemberType]
        if entry is None:
            entries.append(None)
            continue
        assert isinstance(entry, mx.array)
        entries.append(_detached_copy(entry))
    copy = ArraysCache(len(entries))
    copy.cache = entries  # type: ignore[reportUnknownMemberType]
    return copy


def _copy_cache_list(cl: CacheList) -> CacheList:
    inners: list[object] = list(cl)  # type: ignore[reportUnknownArgumentType]
    copied: list[object] = []
    for inner in inners:
        if isinstance(inner, RotatingKVCache):
            snap = copy_rotating_kv_cache(inner)
            copied.append(snap if snap is not None else deepcopy(inner))
        elif isinstance(inner, ArraysCache):
            copied.append(_copy_arrays_cache(inner))
        else:
            copied.append(deepcopy(inner))
    return CacheList(*copied)


def _detached_copy_or_none(a: mx.array | None) -> mx.array | None:
    if a is None:
        return None
    out = _detached_copy(a)
    mx.eval(out)
    return out


def _copy_compressor_branch(b: CompressorBranch) -> CompressorBranch:
    out = CompressorBranch.__new__(CompressorBranch)
    out.buffer_kv = _detached_copy_or_none(b.buffer_kv)
    out.buffer_gate = _detached_copy_or_none(b.buffer_gate)
    out.prev_kv = _detached_copy_or_none(b.prev_kv)
    out.prev_gate = _detached_copy_or_none(b.prev_gate)
    out.pool = _detached_copy_or_none(b.pool)
    out.buffer_lengths = deepcopy(b.buffer_lengths)
    out.pool_lengths = deepcopy(b.pool_lengths)
    out.buffer_count = deepcopy(b.buffer_count)
    out._new_pool_lengths = deepcopy(b._new_pool_lengths)
    return out


def _copy_v4_cache(c: DeepseekV4Cache) -> DeepseekV4Cache:
    snap = DeepseekV4Cache.__new__(DeepseekV4Cache)

    local: RotatingKVCache = c.local
    local_snap = copy_rotating_kv_cache(local)
    if local_snap is None:
        local_snap = RotatingKVCache.__new__(RotatingKVCache)
        local_snap.keys = None
        local_snap.values = None
        local_snap.offset = local.offset
        local_snap._idx = 0
        local_snap.keep = local.keep
        local_snap.max_size = local.max_size
    snap.local = local_snap

    snap._branches = {
        key: _copy_compressor_branch(branch) for key, branch in c._branches.items()
    }
    snap._pending_lengths = deepcopy(c._pending_lengths)
    return snap


def copy_snapshot_entry(
    entry: ArraysCache | RotatingKVCache | CacheList | DeepseekV4Cache | None,
) -> ArraysCache | RotatingKVCache | CacheList | DeepseekV4Cache | None:
    match entry:
        case None:
            return None
        case RotatingKVCache():
            snap = copy_rotating_kv_cache(entry)
            return snap if snap is not None else deepcopy(entry)
        case ArraysCache():
            return _copy_arrays_cache(entry)
        case CacheList():
            return _copy_cache_list(entry)
        case DeepseekV4Cache():
            return _copy_v4_cache(entry)


def snapshot_ssm_states(cache: KVCacheType) -> CacheSnapshot:
    states: list[
        RotatingKVCache | ArraysCache | CacheList | DeepseekV4Cache | None
    ] = []
    for c in cache:
        if isinstance(c, ArraysCache):
            states.append(_copy_arrays_cache(c))
        elif isinstance(c, RotatingKVCache):
            states.append(copy_rotating_kv_cache(c))
        elif isinstance(c, CacheList) and not bool(c.is_trimmable()):  # type: ignore[reportUnknownMemberType]
            states.append(_copy_cache_list(c))
        elif isinstance(c, DeepseekV4Cache):
            states.append(_copy_v4_cache(c))
        else:
            states.append(None)
    token_count = cache_length(cache)
    return CacheSnapshot(states=states, token_count=token_count)


def _find_nearest_snapshot(
    snapshots: list[CacheSnapshot],
    target_token_count: int,
) -> CacheSnapshot | None:
    best: CacheSnapshot | None = None
    for snap in snapshots:
        if snap.token_count <= target_token_count and (
            best is None or snap.token_count > best.token_count
        ):
            best = snap
    return best


def is_non_trimmable_cache_entry(c: object) -> bool:
    """A cache entry is non-trimmable if `trim(n)` can't roll back its full
    state — meaning the prefill +2 rollback must snapshot+restore it instead.
    """
    if isinstance(c, (ArraysCache, RotatingKVCache)):
        return True
    if isinstance(c, CacheList):
        return not bool(c.is_trimmable())  # type: ignore[reportUnknownMemberType]
    return isinstance(c, DeepseekV4Cache)


def has_non_kv_caches(cache: KVCacheType) -> bool:
    """Check whether any cache entry requires snapshot-based restoration."""
    return any(is_non_trimmable_cache_entry(c) for c in cache)


def has_deepseek_v4_cache(cache: KVCacheType) -> bool:
    return any(isinstance(c, DeepseekV4Cache) for c in cache)


def v4_snapshot_landmark_targets(final_token_count: int) -> tuple[int, ...]:
    targets: list[int] = []
    target = _V4_PREFIX_CACHE_FIRST_LANDMARK_TOKENS
    while target < final_token_count:
        targets.append(target)
        target *= 2
    return tuple(targets)


def _bounded_v4_snapshots(
    snapshots: list[CacheSnapshot] | None,
) -> list[CacheSnapshot] | None:
    if not snapshots:
        return None
    snapshots_by_position = {
        snapshot.token_count: snapshot for snapshot in snapshots
    }
    ordered = sorted(
        snapshots_by_position.values(),
        key=lambda snapshot: snapshot.token_count,
    )
    selected: dict[int, CacheSnapshot] = {}
    final_token_count = ordered[-1].token_count
    for target in v4_snapshot_landmark_targets(final_token_count):
        snapshot = _find_nearest_snapshot(ordered, target)
        if snapshot is not None:
            selected[snapshot.token_count] = snapshot
    for snapshot in ordered[-_V4_PREFIX_CACHE_TAIL_SNAPSHOT_COUNT:]:
        selected[snapshot.token_count] = snapshot
    return sorted(selected.values(), key=lambda snapshot: snapshot.token_count)


def _log_v4_snapshot_retention(snapshots: list[CacheSnapshot]) -> None:
    label = "snapshot" if len(snapshots) == 1 else "snapshots"
    retained_mib = sum(snapshot.nbytes for snapshot in snapshots) / (1 << 20)
    logger.info(
        "DeepSeek V4 prefix cache retained "
        f"{len(snapshots)} {label} ({retained_mib:.1f} MiB) "
        f"at tokens={[snapshot.token_count for snapshot in snapshots]}"
    )


class KVPrefixCache:
    def __init__(self, group: mx.distributed.Group | None):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._snapshots: list[list[CacheSnapshot] | None] = []
        self._media_regions: list[list["MediaRegion"]] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self.prefill_tps: list[float] = []
        self._access_counter: int = 0
        self._group = group
        
        # Structured cache separation: semantic boundaries
        # Each segment is cached independently with a hash for change detection
        # Segments: [system, context, conversation...]
        self.segmented_caches: dict[str, tuple[mx.array, KVCacheType, str]] = {}  # segment_id -> (tokens, cache, content_hash)

    def clear(self):
        """Clear all cached prompts and caches."""
        self.prompts.clear()
        self.caches.clear()
        self._snapshots.clear()
        self._media_regions.clear()
        self._last_used.clear()
        self.prefill_tps.clear()
        # Keep segmented caches - they persist across turns unless invalidated

    def set_segment_cache(
        self,
        segment_id: str,
        tokens: mx.array,
        cache: KVCacheType,
        content_hash: str,
    ):
        """Set a segmented cache with content hash for change detection."""
        self.segmented_caches[segment_id] = (tokens, deepcopy(cache), content_hash)
        logger.info(f"Segment cache '{segment_id}' set: {len(tokens)} tokens, hash={content_hash[:8]}")

    def get_segmented_cache(
        self,
        model: Model,
        prompt_tokens: mx.array,
        segment_boundaries: list[tuple[str, int, int, str]],  # (segment_id, start, end, content_hash)
        media_regions: list["MediaRegion"] | None = None,
    ) -> tuple[KVCacheType, mx.array, float]:
        """Get KV cache using semantic boundaries with change detection.
        
        Args:
            model: The model
            prompt_tokens: Full prompt tokens
            segment_boundaries: List of (segment_id, start_token, end_token, content_hash)
            media_regions: Optional media regions for validation
            
        Returns:
            Tuple of (cache, remaining_tokens, hit_rate) where:
            - cache: KV cache with matching segments already prefilled
            - remaining_tokens: tokens that need prefilling (changed segments)
            - hit_rate: percentage of prompt that was cached
        """
        total_len = len(prompt_tokens)
        cache = make_kv_cache(model)
        cached_len = 0
        
        # Process each segment independently
        for segment_id, start, end, content_hash in segment_boundaries:
            segment_len = end - start
            if segment_len <= 0:
                continue
            
            # Check if we have a cached version with matching hash
            if segment_id in self.segmented_caches:
                cached_tokens, cached_cache, cached_hash = self.segmented_caches[segment_id]
                
                if cached_hash == content_hash and len(cached_tokens) == segment_len:
                    # Hash match - reuse this segment's cache
                    logger.info(f"Segment '{segment_id}' cache hit: {segment_len} tokens")
                    
                    # Merge this segment's KV into the working cache
                    for i, (src, dst) in enumerate(zip(cached_cache, cache)):
                        # Handle DeepseekV4Cache specially
                        if isinstance(src, DeepseekV4Cache) and isinstance(dst, DeepseekV4Cache):
                            # Merge local RotatingKVCache
                            src_local = src.local
                            dst_local = dst.local
                            
                            if src_local.keys is not None and src_local.keys.shape[2] == segment_len:
                                if dst_local.keys is None:
                                    dst_local.keys = _detached_copy(src_local.keys)
                                    dst_local.values = _detached_copy(src_local.values)
                                else:
                                    dst_local.keys = mx.concatenate([dst_local.keys, _detached_copy(src_local.keys)], axis=2)
                                    dst_local.values = mx.concatenate([dst_local.values, _detached_copy(src_local.values)], axis=2)
                                
                                dst_local.offset = cached_len + segment_len
                                dst_local._idx = cached_len + segment_len
                            
                            # Merge compressor branches
                            for branch_key, src_branch in src._branches.items():
                                if branch_key not in dst._branches:
                                    continue
                                dst_branch = dst._branches[branch_key]
                                
                                # Merge buffer_kv
                                if src_branch.buffer_kv is not None and src_branch.buffer_kv.shape[2] == segment_len:
                                    if dst_branch.buffer_kv is None:
                                        dst_branch.buffer_kv = _detached_copy(src_branch.buffer_kv)
                                        dst_branch.buffer_gate = _detached_copy(src_branch.buffer_gate)
                                    else:
                                        dst_branch.buffer_kv = mx.concatenate([dst_branch.buffer_kv, _detached_copy(src_branch.buffer_kv)], axis=2)
                                        dst_branch.buffer_gate = mx.concatenate([dst_branch.buffer_gate, _detached_copy(src_branch.buffer_gate)], axis=2)
                                
                                # Merge prev_kv
                                if src_branch.prev_kv is not None and src_branch.prev_kv.shape[2] == segment_len:
                                    if dst_branch.prev_kv is None:
                                        dst_branch.prev_kv = _detached_copy(src_branch.prev_kv)
                                        dst_branch.prev_gate = _detached_copy(src_branch.prev_gate)
                                    else:
                                        dst_branch.prev_kv = mx.concatenate([dst_branch.prev_kv, _detached_copy(src_branch.prev_kv)], axis=2)
                                        dst_branch.prev_gate = mx.concatenate([dst_branch.prev_gate, _detached_copy(src_branch.prev_gate)], axis=2)
                                
                                # Merge pool
                                if src_branch.pool is not None:
                                    if dst_branch.pool is None:
                                        dst_branch.pool = _detached_copy(src_branch.pool)
                                    else:
                                        dst_branch.pool = mx.concatenate([dst_branch.pool, _detached_copy(src_branch.pool)], axis=2)
                                
                                # Copy lengths
                                dst_branch.buffer_lengths = deepcopy(src_branch.buffer_lengths)
                                dst_branch.pool_lengths = deepcopy(src_branch.pool_lengths)
                                dst_branch.buffer_count = deepcopy(src_branch.buffer_count)
                        
                        elif hasattr(dst, 'keys') and hasattr(src, 'keys'):
                            # Standard RotatingKVCache / ArraysCache handling
                            src_keys = src.keys
                            src_vals = src.values
                            
                            if src_keys is not None and src_keys.shape[2] == segment_len:
                                if dst.keys is None:
                                    dst.keys = _detached_copy(src_keys)
                                    dst.values = _detached_copy(src_vals)
                                else:
                                    # Append to existing cache
                                    dst.keys = mx.concatenate([dst.keys, _detached_copy(src_keys)], axis=2)
                                    dst.values = mx.concatenate([dst.values, _detached_copy(src_vals)], axis=2)
                                
                                dst.offset = cached_len + segment_len
                                if hasattr(src, '_idx'):
                                    dst._idx = cached_len + segment_len
                    
                    cached_len += segment_len
                    continue
            
            # No cache hit - this segment needs prefilling
            logger.info(f"Segment '{segment_id}' cache miss: {segment_len} tokens (hash changed or new)")
        
        # Remaining tokens need prefilling
        remaining = prompt_tokens[cached_len:] if cached_len < total_len else mx.array([], dtype=prompt_tokens.dtype)
        hit_rate = (cached_len / total_len * 100) if total_len > 0 else 0
        
        logger.info(f"Segmented cache: {cached_len}/{total_len} tokens cached ({hit_rate:.1f}%)")
        
        return cache, remaining, hit_rate

    def add_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        ssm_snapshots: list[CacheSnapshot] | None = None,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Add a new cache entry. Evicts LRU entries if memory is high."""
        is_v4 = has_deepseek_v4_cache(cache)
        if is_v4 and _V4_PREFIX_CACHE_MAX_ENTRIES == 0:
            logger.info("DeepSeek V4 prefix cache persistence is disabled")
            return

        stored_cache = deepcopy(cache)
        stored_snapshots = (
            _bounded_v4_snapshots(ssm_snapshots) if is_v4 else ssm_snapshots
        )
        if is_v4:
            self._evict_v4_entries_for_add()
        self._evict_if_needed()

        access_counter = self._access_counter + 1
        start_length = len(self.prompts)
        try:
            self.prompts.append(prompt_tokens)
            self.caches.append(stored_cache)
            self._snapshots.append(stored_snapshots)
            self._media_regions.append(media_regions or [])
            self.prefill_tps.append(prefill_tps)
            self._last_used.append(access_counter)
        except Exception:
            for collection in (
                self.prompts,
                self.caches,
                self._snapshots,
                self._media_regions,
                self.prefill_tps,
                self._last_used,
            ):
                del collection[start_length:]
            raise

        self._access_counter = access_counter
        logger.info(
            f"KV cache added (index {start_length}): "
            f"{len(prompt_tokens)} tokens, {len(self.prompts)} entries"
        )
        if is_v4 and stored_snapshots:
            _log_v4_snapshot_retention(stored_snapshots)
            self._log_total_v4_snapshot_retention()

    def update_kv_cache(
        self,
        index: int,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None,
        restore_pos: int,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Update an existing cache entry in-place."""
        old_snapshots = self._snapshots[index]
        is_v4 = has_deepseek_v4_cache(cache)
        if is_v4 and _V4_PREFIX_CACHE_MAX_ENTRIES == 0:
            self._evict_entry(index, "DeepSeek V4 prefix persistence disabled")
            gc.collect()
            mx.clear_cache()
            return

        if is_v4:
            eligible = [
                snapshot
                for snapshot in old_snapshots or []
                if snapshot.token_count <= restore_pos
            ]
            eligible.extend(snapshots or [])
            stored_snapshots = _bounded_v4_snapshots(eligible)
        else:
            merged: list[CacheSnapshot] = []
            if old_snapshots:
                merged = [s for s in old_snapshots if s.token_count <= restore_pos]
            if snapshots:
                merged.extend(snapshots)
            stored_snapshots = merged or None

        stored_cache = deepcopy(cache)
        stored_media_regions = media_regions or []
        access_counter = self._access_counter + 1
        self.prompts[index] = prompt_tokens
        self.caches[index] = stored_cache
        self._snapshots[index] = stored_snapshots
        self._media_regions[index] = stored_media_regions
        self.prefill_tps[index] = prefill_tps
        self._access_counter = access_counter
        self._last_used[index] = access_counter
        logger.info(f"KV cache updated (index {index}): {len(prompt_tokens)} tokens")
        if is_v4 and stored_snapshots:
            _log_v4_snapshot_retention(stored_snapshots)
            self._log_total_v4_snapshot_retention()

    def _get_snapshot(
        self, entry_index: int, target_token_count: int
    ) -> tuple[int, CacheSnapshot | None]:
        if not has_non_kv_caches(self.caches[entry_index]):
            return target_token_count, None

        snapshots = self._snapshots[entry_index]
        if not snapshots:
            return 0, None

        snap = _find_nearest_snapshot(snapshots, target_token_count)
        if snap is not None:
            return snap.token_count, snap

        return 0, None

    def get_kv_cache(
        self,
        model: Model,
        prompt_tokens: mx.array,
        media_regions: list["MediaRegion"] | None = None,
    ) -> tuple[KVCacheType, mx.array, int | None, bool]:
        """Get KV cache for prompt, returning remaining tokens to prefill.

        Returns:
            Tuple of (cache, remaining_tokens, matched_index, is_exact) where:
            - cache: KV cache to use for generation
            - remaining_tokens: tokens that still need prefilling
            - matched_index: index of the matched entry (None if no match)
            - is_exact: True if the full prompt matched the cached entry

        For models with SSM layers (which are ArraysCache in mlx), the cache is trimmed to the
        nearest SSM snapshot position at or before the match point for correctness.
        Same for rotating KV Cache.

        Media region validation: if the token-level prefix match extends into
        a cached media region whose content_hash differs from the query's, the
        match is truncated to the start of that region.
        """
        max_length = len(prompt_tokens)
        query_regions = media_regions or []

        best_index: int | None = None
        best_raw_length = 0
        best_length = 0
        best_restore_pos = 0
        best_restore_snap: CacheSnapshot | None = None
        best_cached_length = 0
        best_is_exact = False
        best_score: tuple[int, int, int] | None = None
        candidate_details: list[str] = []

        # Rank candidates by the state we can actually restore, not merely by
        # the raw token prefix. V4 snapshots may make a shorter raw match more
        # useful than a longer match whose nearest checkpoint is much earlier.
        for i, cached_prompt in enumerate(self.prompts):
            raw_length = get_prefix_length(prompt_tokens, cached_prompt)
            validated_length = raw_length
            if validated_length > 0:
                validated_length = self._validate_media_match(
                    validated_length,
                    self._media_regions[i],
                    query_regions,
                )
            candidate_is_exact = validated_length >= max_length - 1
            if validated_length <= 0 and not candidate_is_exact:
                candidate_details.append(
                    f"{i}:tokens={len(cached_prompt)},raw={raw_length},"
                    "validated=0,restore=0,reason=no-prefix"
                )
                continue

            candidate_cache = self.caches[i]
            candidate_cached_length = cache_length(candidate_cache)
            candidate_has_ssm = has_non_kv_caches(candidate_cache)
            if candidate_has_ssm:
                target = (
                    min(validated_length, max_length - 1)
                    if candidate_is_exact
                    else validated_length
                )
            else:
                desired = (max_length - 1) if candidate_is_exact else validated_length
                target = min(candidate_cached_length, desired)

            restore_pos, restore_snap = self._get_snapshot(i, target)
            snapshots = self._snapshots[i] or []
            snapshot_positions = [snapshot.token_count for snapshot in snapshots]
            snapshot_mib = sum(snapshot.nbytes for snapshot in snapshots) / (1 << 20)
            stored_cache_mib = sum(
                _cache_state_nbytes(state) for state in candidate_cache
            ) / (1 << 20)
            usable = restore_snap is not None or not candidate_has_ssm
            candidate_details.append(
                f"{i}:tokens={len(cached_prompt)},raw={raw_length},"
                f"validated={validated_length},restore={restore_pos},"
                f"cached={candidate_cached_length},exact={candidate_is_exact},"
                f"last_used={self._last_used[i]},"
                f"cache={stored_cache_mib:.1f}MiB,"
                f"snapshots={snapshot_positions},"
                f"snapshot_bytes={snapshot_mib:.1f}MiB,"
                f"usable={usable}"
            )
            if not usable:
                continue

            score = (
                restore_pos,
                validated_length,
                self._last_used[i],
            )
            if best_score is None or score > best_score:
                best_index = i
                best_raw_length = raw_length
                best_length = validated_length
                best_restore_pos = restore_pos
                best_restore_snap = restore_snap
                best_cached_length = candidate_cached_length
                best_is_exact = candidate_is_exact
                best_score = score

            # No later entry can restore more than the complete query prefix.
            if restore_pos >= max_length - 1:
                break

        if candidate_details:
            logger.info("KV cache candidates: " + " | ".join(candidate_details))

        if best_index is None:
            if self.prompts:
                logger.info(
                    "KV cache miss: no restorable token prefix across "
                    f"{len(self.prompts)} entries"
                )
            return make_kv_cache(model), prompt_tokens, None, False

        logger.info(
            "KV cache selected: "
            f"entry={best_index}, raw={best_raw_length}/{max_length}, "
            f"validated={best_length}, restore={best_restore_pos}, "
            f"cached={best_cached_length}, exact={best_is_exact}"
        )

        prompt_cache = deepcopy(self.caches[best_index])
        tokens_to_trim = best_cached_length - best_restore_pos
        if tokens_to_trim > 0:
            trim_cache(prompt_cache, tokens_to_trim, best_restore_snap)
            # Reset cache offset to match trimmed length
            for c in prompt_cache:
                if isinstance(c, (ArraysCache, RotatingKVCache)):
                    continue
                if isinstance(c, DeepseekV4Cache):
                    continue
                if hasattr(c, "offset"):
                    c.offset = best_restore_pos

        self._access_counter += 1
        self._last_used[best_index] = self._access_counter
        remaining = prompt_tokens[best_restore_pos:]

        return prompt_cache, remaining, best_index, best_is_exact

    @staticmethod
    def _validate_media_match(
        match_length: int,
        cached_regions: list["MediaRegion"],
        query_regions: list["MediaRegion"],
    ) -> int:
        if not cached_regions:
            return match_length

        query_by_start: dict[int, "MediaRegion"] = {
            r.start_pos: r for r in query_regions
        }

        for cached_r in cached_regions:
            if cached_r.start_pos >= match_length:
                break
            query_r = query_by_start.get(cached_r.start_pos)
            if query_r is None:
                continue
            if query_r.content_hash != cached_r.content_hash:
                logger.info(
                    f"Media region mismatch at pos {cached_r.start_pos}: "
                    f"cached={cached_r.content_hash[:12]}... "
                    f"query={query_r.content_hash[:12]}... — "
                    f"truncating match from {match_length} to {cached_r.start_pos}"
                )
                match_length = cached_r.start_pos
                break

        return match_length

    def _evict_if_needed(self):
        """Evict least recently used entries while memory usage is high."""
        if len(self.caches) == 0:
            return

        evicted_any = False
        # Evict LRU entries until below threshold
        while (
            len(self.caches) > 0
            and self.get_memory_used_percentage() > _MEMORY_THRESHOLD
        ):
            lru_index = self._last_used.index(min(self._last_used))
            self._evict_entry(lru_index, "memory pressure")
            evicted_any = True

        if evicted_any:
            gc.collect()
            mx.clear_cache()

    def _evict_v4_entries_for_add(self) -> None:
        evicted_any = False
        while (
            sum(has_deepseek_v4_cache(cache) for cache in self.caches)
            >= _V4_PREFIX_CACHE_MAX_ENTRIES
        ):
            candidates = [
                index
                for index, cache in enumerate(self.caches)
                if has_deepseek_v4_cache(cache)
            ]
            lru_index = min(candidates, key=lambda index: self._last_used[index])
            self._evict_entry(
                lru_index,
                f"DeepSeek V4 entry cap {_V4_PREFIX_CACHE_MAX_ENTRIES}",
            )
            evicted_any = True

        if evicted_any:
            gc.collect()
            mx.clear_cache()

    def _evict_entry(self, index: int, reason: str) -> None:
        evicted_tokens = len(self.prompts[index])
        self.prompts.pop(index)
        self.caches.pop(index)
        self._snapshots.pop(index)
        self._media_regions.pop(index)
        self._last_used.pop(index)
        self.prefill_tps.pop(index)
        logger.info(
            f"KV cache evicted LRU entry index {index} "
            f"({evicted_tokens} tokens): {reason}"
        )

    def _log_total_v4_snapshot_retention(self) -> None:
        total_bytes = sum(
            snapshot.nbytes
            for cache, snapshots in zip(
                self.caches, self._snapshots, strict=True
            )
            if has_deepseek_v4_cache(cache)
            for snapshot in snapshots or []
        )
        v4_entry_count = sum(has_deepseek_v4_cache(cache) for cache in self.caches)
        logger.info(
            "DeepSeek V4 prefix cache total retained snapshots: "
            f"{total_bytes / (1 << 20):.1f} MiB across "
            f"{v4_entry_count} entries"
        )

    def get_memory_used_percentage(self) -> float:
        local_pressure: float = get_memory_used_percentage()

        if self._group is None:
            return local_pressure

        all_pressure = mx.distributed.all_gather(
            mx.array([local_pressure], dtype=mx.float32),
            group=self._group,
        )
        # .item() evals.
        max_pressure = float(mx.max(all_pressure).item())
        return max_pressure


def trim_cache(
    cache: KVCacheType,
    num_tokens: int,
    snapshot: CacheSnapshot | None = None,
) -> None:
    for i, c in enumerate(cache):
        non_trimmable = is_non_trimmable_cache_entry(c)
        if non_trimmable:
            if snapshot is not None and snapshot.states[i] is not None:
                restored = copy_snapshot_entry(snapshot.states[i])
                if restored is not None:
                    cache[i] = restored  # type: ignore
            elif isinstance(c, (ArraysCache, RotatingKVCache)):
                c.state = [None] * len(c.state)
                if isinstance(c, RotatingKVCache):
                    c.offset = 0
                    c._idx = 0
            elif isinstance(c, DeepseekV4Cache):
                cache[i] = DeepseekV4Cache(c.local.max_size)  # type: ignore
            else:
                # CacheList without a snapshot — zero each inner cache's state
                for inner in c:  # type: ignore[reportUnknownVariableType]
                    if isinstance(inner, (ArraysCache, RotatingKVCache)):
                        inner.state = [None] * len(inner.state)
                        if isinstance(inner, RotatingKVCache):
                            inner.offset = 0
                            inner._idx = 0
        else:
            c.trim(num_tokens)


def encode_prompt(tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
    """Encode a prompt string to token array.

    For chat-templated prompts (which have their own structure markers like
    <|im_user|>, <|im_middle|>, etc.), we should NOT add BOS/EOS tokens as
    that would corrupt the prompt structure.
    """
    # Chat templates define their own structure - don't add BOS/EOS
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return mx.array(prompt_tokens)


def _entry_length(
    c: KVCache
    | RotatingKVCache
    | QuantizedKVCache
    | ArraysCache
    | CacheList
    | DeepseekV4Cache,
) -> int:
    # Use .offset attribute which KVCache types have (len() not implemented in older QuantizedKVCache).
    if hasattr(c, "offset"):
        return c.offset
    # For CacheList
    if hasattr(c, "size"):
        return int(c.size())  # type: ignore
    return 0


def cache_length(cache: KVCacheType) -> int:
    """Get the number of tokens in a KV cache."""
    return max((_entry_length(c) for c in cache), default=0)


def get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Find the length of the common prefix between two token arrays."""
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]))
    if n == 0:
        return 0

    equal = mx.equal(prompt[:n], cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)  # stays 1 until first mismatch, then 0 forever
    return int(mx.sum(prefix_mask).item())


def get_available_memory() -> Memory:
    mem: int = psutil.virtual_memory().available
    return Memory.from_bytes(mem)


def get_memory_used_percentage() -> float:
    mem = psutil.virtual_memory()
    # percent is 0-100
    return float(mem.percent / 100)


def make_kv_cache(
    model: Model, max_kv_size: int | None = None, keep: int = 0
) -> KVCacheType:
    assert hasattr(model, "layers")

    if hasattr(model, "make_cache"):
        logger.info("Using MLX LM's make cache")
        return model.make_cache()  # type: ignore

    if max_kv_size is None:
        if KV_CACHE_BITS is None:
            logger.info("Using default KV cache")
            return [KVCache() for _ in model.layers]
        else:
            logger.info("Using quantized KV cache")
            return [
                QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS)
                for _ in model.layers
            ]
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]
