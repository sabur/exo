# type: ignore
import time
from typing import cast
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache
from mlx_lm.sample_utils import make_sampler

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    cache_length,
    encode_prompt,
    get_prefix_length,
    make_kv_cache,
    snapshot_ssm_states,
    trim_cache,
)
from exo.worker.engines.mlx.generator.generate import mlx_generate, prefill
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.tests.unittests.test_mlx.conftest import (
    DEFAULT_GPT_OSS_CONFIG,
    DEFAULT_GPT_OSS_MODEL_ID,
)


def _check_model_exists() -> bool:
    return DEFAULT_GPT_OSS_CONFIG.model_path.exists()


def _make_v4_cache(offset: int, pool_rows: int) -> DeepseekV4Cache:
    cache = DeepseekV4Cache(sliding_window=8)
    keys = mx.arange(32, dtype=mx.float32).reshape(1, 1, 8, 4) + offset
    cache.local.keys = keys
    cache.local.values = keys + 0.5
    cache.local.offset = offset
    cache.local._idx = 8

    for key, fill in (("compressor", 1.0), ("indexer", 2.0)):
        branch = cache._branches[key]
        branch.pool = mx.full((1, pool_rows, 4), fill * offset)
        branch.pool_lengths = None
    return cache


class TestGetPrefixLength:
    def test_identical_arrays(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3, 4, 5])
        assert get_prefix_length(a, b) == 5

    def test_no_common_prefix(self):
        a = mx.array([1, 2, 3])
        b = mx.array([4, 5, 6])
        assert get_prefix_length(a, b) == 0

    def test_partial_prefix(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3, 7, 8])
        assert get_prefix_length(a, b) == 3

    def test_prompt_longer_than_cached(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 3])
        assert get_prefix_length(a, b) == 3

    def test_cached_longer_than_prompt(self):
        a = mx.array([1, 2, 3])
        b = mx.array([1, 2, 3, 4, 5])
        assert get_prefix_length(a, b) == 3

    def test_single_token_match(self):
        a = mx.array([1, 2, 3])
        b = mx.array([1, 5, 6])
        assert get_prefix_length(a, b) == 1

    def test_empty_prompt(self):
        a = mx.array([]).astype(mx.int32)
        b = mx.array([1, 2, 3])
        assert get_prefix_length(a, b) == 0

    def test_empty_cached(self):
        a = mx.array([1, 2, 3])
        b = mx.array([]).astype(mx.int32)
        assert get_prefix_length(a, b) == 0

    def test_both_empty(self):
        a = mx.array([]).astype(mx.int32)
        b = mx.array([]).astype(mx.int32)
        assert get_prefix_length(a, b) == 0


class TestKVPrefix:
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a minimal mock tokenizer for tests that don't need real tokenization."""
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        return tokenizer

    def test_starts_empty(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        assert len(cache.prompts) == 0
        assert len(cache.caches) == 0

    def test_clear_empties_cache(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        cache.prompts.append(mx.array([1, 2, 3]))
        cache.caches.append([KVCache()])
        cache.clear()
        assert len(cache.prompts) == 0
        assert len(cache.caches) == 0

    def test_clear_on_empty_cache(self, mock_tokenizer):
        cache = KVPrefixCache(None)
        cache.clear()
        assert len(cache.prompts) == 0

    def test_trim_cache_restores_deepseek_v4_snapshot(self):
        snapshot_cache = _make_v4_cache(offset=12, pool_rows=3)
        live_cache = _make_v4_cache(offset=100, pool_rows=25)
        snapshot = CacheSnapshot(states=[snapshot_cache], token_count=12)
        cache = [live_cache]

        trim_cache(cache, num_tokens=88, snapshot=snapshot)

        restored = cast(DeepseekV4Cache, cache[0])
        assert restored is not live_cache
        assert restored.offset == 12
        assert restored.local._idx == 8
        assert mx.array_equal(restored.local.keys, snapshot_cache.local.keys)
        assert mx.array_equal(
            restored._branches["compressor"].pool,
            snapshot_cache._branches["compressor"].pool,
        )
        assert mx.array_equal(
            restored._branches["indexer"].pool,
            snapshot_cache._branches["indexer"].pool,
        )

    def test_partial_v4_prefix_hit_restores_snapshot_before_suffix(self):
        cached_prompt = mx.arange(100, dtype=mx.int32)
        query = mx.concatenate(
            [
                cached_prompt[:12],
                mx.arange(1000, 1101, dtype=mx.int32),
            ]
        )
        snapshot = CacheSnapshot(
            states=[_make_v4_cache(offset=12, pool_rows=3)],
            token_count=12,
        )
        prefix_cache = KVPrefixCache(None)
        prefix_cache.prompts = [cached_prompt]
        prefix_cache.caches = [[_make_v4_cache(offset=100, pool_rows=25)]]
        prefix_cache._snapshots = [[snapshot]]
        prefix_cache._media_regions = [[]]
        prefix_cache._last_used = [0]
        prefix_cache.prefill_tps = [0.0]

        restored, remaining, matched_index, is_exact = prefix_cache.get_kv_cache(
            MagicMock(), query
        )

        assert matched_index == 0
        assert not is_exact
        assert cache_length(restored) == 12
        assert mx.array_equal(remaining, query[12:])
        restored_v4 = cast(DeepseekV4Cache, restored[0])
        assert restored_v4._branches["compressor"].pool.shape[1] == 3

    def test_final_v4_snapshot_supports_append_only_hit(self):
        cached_prompt = mx.arange(100, dtype=mx.int32)
        query = mx.concatenate(
            [cached_prompt, mx.arange(1000, 1010, dtype=mx.int32)]
        )
        snapshot = CacheSnapshot(
            states=[_make_v4_cache(offset=98, pool_rows=24)],
            token_count=98,
        )
        prefix_cache = KVPrefixCache(None)
        prefix_cache.prompts = [cached_prompt]
        prefix_cache.caches = [[_make_v4_cache(offset=98, pool_rows=24)]]
        prefix_cache._snapshots = [[snapshot]]
        prefix_cache._media_regions = [[]]
        prefix_cache._last_used = [0]
        prefix_cache.prefill_tps = [0.0]

        restored, remaining, matched_index, is_exact = prefix_cache.get_kv_cache(
            MagicMock(), query
        )

        assert matched_index == 0
        assert not is_exact
        assert cache_length(restored) == 98
        assert mx.array_equal(remaining, query[98:])

    def test_v4_mutation_before_final_snapshot_falls_back_to_fresh_prefill(self):
        cached_prompt = mx.arange(100, dtype=mx.int32)
        query = mx.concatenate(
            [cached_prompt[:12], mx.arange(1000, 1010, dtype=mx.int32)]
        )
        snapshot = CacheSnapshot(
            states=[_make_v4_cache(offset=98, pool_rows=24)],
            token_count=98,
        )
        prefix_cache = KVPrefixCache(None)
        prefix_cache.prompts = [cached_prompt]
        prefix_cache.caches = [[_make_v4_cache(offset=98, pool_rows=24)]]
        prefix_cache._snapshots = [[snapshot]]
        prefix_cache._media_regions = [[]]
        prefix_cache._last_used = [0]
        prefix_cache.prefill_tps = [0.0]
        model = MagicMock(spec=[])
        model.layers = []

        restored, remaining, matched_index, is_exact = prefix_cache.get_kv_cache(
            model, query
        )

        assert restored == []
        assert mx.array_equal(remaining, query)
        assert matched_index is None
        assert not is_exact

    def test_v4_add_evicts_previous_entry_at_cap_one(self):
        prefix_cache = KVPrefixCache(None)

        first_prompt = mx.arange(12, dtype=mx.int32)
        first_cache = [_make_v4_cache(offset=12, pool_rows=3)]
        first_snapshot = CacheSnapshot(states=first_cache, token_count=12)

        second_prompt = mx.arange(20, dtype=mx.int32)
        second_cache = [_make_v4_cache(offset=20, pool_rows=5)]
        second_snapshot = CacheSnapshot(states=second_cache, token_count=20)

        with patch(
            "exo.worker.engines.mlx.cache._V4_PREFIX_CACHE_MAX_ENTRIES", 1
        ):
            prefix_cache.add_kv_cache(first_prompt, first_cache, [first_snapshot])
            prefix_cache.add_kv_cache(
                second_prompt, second_cache, [second_snapshot]
            )

        assert len(prefix_cache.prompts) == 1
        assert mx.array_equal(prefix_cache.prompts[0], second_prompt)
        assert prefix_cache._snapshots[0] is not None
        assert [s.token_count for s in prefix_cache._snapshots[0]] == [20]

    def test_v4_add_retains_five_entries_and_evicts_the_lru(self):
        prefix_cache = KVPrefixCache(None)

        with patch(
            "exo.worker.engines.mlx.cache._V4_PREFIX_CACHE_MAX_ENTRIES", 5
        ):
            for token_count in range(12, 72, 12):
                prompt = mx.arange(token_count, dtype=mx.int32)
                cache = [
                    _make_v4_cache(
                        offset=token_count,
                        pool_rows=token_count // 4,
                    )
                ]
                snapshot = CacheSnapshot(
                    states=cache,
                    token_count=token_count,
                )
                prefix_cache.add_kv_cache(prompt, cache, [snapshot])

            assert len(prefix_cache.prompts) == 5

            prompt = mx.arange(72, dtype=mx.int32)
            cache = [_make_v4_cache(offset=72, pool_rows=18)]
            snapshot = CacheSnapshot(states=cache, token_count=72)
            prefix_cache.add_kv_cache(prompt, cache, [snapshot])

        assert len(prefix_cache.prompts) == 5
        assert [len(prompt) for prompt in prefix_cache.prompts] == [
            24,
            36,
            48,
            60,
            72,
        ]
        assert all(snapshots is not None for snapshots in prefix_cache._snapshots)
        assert [
            snapshots[0].token_count
            for snapshots in prefix_cache._snapshots
            if snapshots is not None
        ] == [24, 36, 48, 60, 72]

    def test_v4_update_retains_only_latest_snapshot(self):
        prefix_cache = KVPrefixCache(None)
        prefix_cache.prompts = [mx.arange(12, dtype=mx.int32)]
        prefix_cache.caches = [[_make_v4_cache(offset=12, pool_rows=3)]]
        prefix_cache._snapshots = [
            [
                CacheSnapshot(
                    states=[_make_v4_cache(offset=4, pool_rows=1)],
                    token_count=4,
                ),
                CacheSnapshot(
                    states=[_make_v4_cache(offset=8, pool_rows=2)],
                    token_count=8,
                ),
            ]
        ]
        prefix_cache._media_regions = [[]]
        prefix_cache._last_used = [1]
        prefix_cache.prefill_tps = [0.0]

        updated_cache = [_make_v4_cache(offset=20, pool_rows=5)]
        prefix_cache.update_kv_cache(
            0,
            mx.arange(20, dtype=mx.int32),
            updated_cache,
            [
                CacheSnapshot(
                    states=[_make_v4_cache(offset=16, pool_rows=4)],
                    token_count=16,
                ),
                CacheSnapshot(states=updated_cache, token_count=20),
            ],
            restore_pos=8,
        )

        assert prefix_cache._snapshots[0] is not None
        assert [s.token_count for s in prefix_cache._snapshots[0]] == [20]

    def test_v4_persistence_can_be_disabled(self):
        prefix_cache = KVPrefixCache(None)
        prompt = mx.arange(12, dtype=mx.int32)
        cache = [_make_v4_cache(offset=12, pool_rows=3)]
        snapshot = CacheSnapshot(states=cache, token_count=12)

        with patch(
            "exo.worker.engines.mlx.cache._V4_PREFIX_CACHE_MAX_ENTRIES", 0
        ):
            prefix_cache.add_kv_cache(prompt, cache, [snapshot])

        assert prefix_cache.prompts == []
        assert prefix_cache.caches == []
        assert prefix_cache._snapshots == []

    def test_add_failure_keeps_parallel_collections_aligned(self):
        prefix_cache = KVPrefixCache(None)

        with patch(
            "exo.worker.engines.mlx.cache.deepcopy",
            side_effect=RuntimeError("copy failed"),
        ):
            with pytest.raises(RuntimeError, match="copy failed"):
                prefix_cache.add_kv_cache(
                    mx.arange(4, dtype=mx.int32),
                    [KVCache()],
                )

        lengths = {
            len(prefix_cache.prompts),
            len(prefix_cache.caches),
            len(prefix_cache._snapshots),
            len(prefix_cache._media_regions),
            len(prefix_cache._last_used),
            len(prefix_cache.prefill_tps),
        }
        assert lengths == {0}

    def test_v4_replacement_copy_failure_preserves_existing_entry(self):
        prefix_cache = KVPrefixCache(None)
        first_prompt = mx.arange(12, dtype=mx.int32)
        first_cache = [_make_v4_cache(offset=12, pool_rows=3)]
        first_snapshot = CacheSnapshot(states=first_cache, token_count=12)
        prefix_cache.add_kv_cache(first_prompt, first_cache, [first_snapshot])

        with patch(
            "exo.worker.engines.mlx.cache.deepcopy",
            side_effect=RuntimeError("copy failed"),
        ):
            with pytest.raises(RuntimeError, match="copy failed"):
                prefix_cache.add_kv_cache(
                    mx.arange(20, dtype=mx.int32),
                    [_make_v4_cache(offset=20, pool_rows=5)],
                    [
                        CacheSnapshot(
                            states=[_make_v4_cache(offset=20, pool_rows=5)],
                            token_count=20,
                        )
                    ],
                )

        assert len(prefix_cache.prompts) == 1
        assert mx.array_equal(prefix_cache.prompts[0], first_prompt)
        assert prefix_cache._snapshots[0] is not None
        assert [s.token_count for s in prefix_cache._snapshots[0]] == [12]

    def test_v4_prefill_captures_only_pre_generation_snapshot(self):
        prompt_tokens = mx.arange(8193, dtype=mx.int32)
        cache = [_make_v4_cache(offset=0, pool_rows=0)]
        model = MagicMock()
        model.layers = []

        def fake_stream_generate(*, prompt, prompt_cache, prompt_progress_callback, **_):
            total = len(prompt)
            prompt_progress_callback(0, total)
            prompt_cache[0] = _make_v4_cache(offset=4096, pool_rows=1024)
            prompt_progress_callback(4096, total)
            prompt_cache[0] = _make_v4_cache(
                offset=total - 1,
                pool_rows=(total - 1) // 4,
            )
            prompt_progress_callback(total - 1, total)
            prompt_cache[0] = _make_v4_cache(
                offset=total + 1,
                pool_rows=(total + 1) // 4,
            )
            prompt_progress_callback(total, total)
            yield object()

        with (
            patch(
                "exo.worker.engines.mlx.generator.generate.stream_generate",
                new=fake_stream_generate,
            ),
            patch(
                "exo.worker.engines.mlx.generator.generate.snapshot_ssm_states",
                wraps=snapshot_ssm_states,
            ) as snapshot_mock,
        ):
            _, _, snapshots = prefill(
                model,
                MagicMock(),
                MagicMock(),
                prompt_tokens,
                cache,
                group=None,
                on_prefill_progress=None,
                distributed_prompt_progress_callback=None,
            )

        assert snapshot_mock.call_count == 1
        assert len(snapshots) == 1
        assert snapshots[0].token_count == len(prompt_tokens) - 1
        assert cache_length(cache) == len(prompt_tokens) - 1


def _load_gpt_oss() -> tuple[Model, object]:
    from mlx_lm.utils import load_model

    from exo.worker.engines.mlx.utils_mlx import load_tokenizer_for_model_id

    model_path = DEFAULT_GPT_OSS_CONFIG.model_path
    model_id = ModelId(DEFAULT_GPT_OSS_MODEL_ID)

    model, _ = load_model(model_path, lazy=False)
    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    return cast(Model, model), tokenizer


@pytest.mark.slow
@pytest.mark.skipif(
    not _check_model_exists(),
    reason=f"GPT-OSS model not found at {DEFAULT_GPT_OSS_CONFIG.model_path}",
)
class TestKVPrefixCacheWithModel:
    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        model, tokenizer = _load_gpt_oss()
        return model, tokenizer

    def test_prefill_populates_cache(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hello!!")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        # Cache should now hold the prompt tokens minus one
        assert cache_length(cache) == len(tokens) - 1
        # Snapshots should be available for models with non-KV caches
        assert len(snapshots) > 0

    def test_add_and_get_exact_match(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Test exact")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        assert len(kv_prefix_cache.prompts) == 1
        stored_length = cache_length(kv_prefix_cache.caches[0])
        assert stored_length > 0

        # Retrieve with same prompt: exact match
        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, tokens
        )
        assert matched_index == 0

        # Exact match returns last token(s) — for models with SSM/rotating caches,
        # snapshot availability constrains how far back we can trim, so remaining
        # may be 1 or 2 tokens depending on the model.
        assert len(remaining_tokens) >= 1
        assert mx.array_equal(remaining_tokens, tokens[-len(remaining_tokens) :])

    def test_add_and_get_prefix_match(self, model_and_tokenizer):
        """get_kv_cache with a longer prompt sharing prefix should return partial match."""
        model, tokenizer = model_and_tokenizer

        short_task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hi")],
            max_output_tokens=1,
        )
        short_prompt = apply_chat_template(tokenizer, short_task)
        short_tokens = encode_prompt(tokenizer, short_prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            short_tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(short_tokens, cache, snapshots)

        # Query with longer prompt that shares the chat template prefix
        long_task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hi there, how are you?")],
            max_output_tokens=1,
        )
        long_prompt = apply_chat_template(tokenizer, long_task)
        long_tokens = encode_prompt(tokenizer, long_prompt)

        # The prompts share a prefix (chat template preamble + "Hi")
        expected_prefix = get_prefix_length(long_tokens, short_tokens)
        assert expected_prefix > 0, (
            "Prompts should share a prefix from the chat template"
        )

        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, long_tokens
        )
        assert matched_index == 0

        # remaining_tokens covers from snapshot restore position to end
        assert len(remaining_tokens) >= len(long_tokens) - expected_prefix

    def test_stored_cache_not_mutated_after_get_and_generation(
        self, model_and_tokenizer
    ):
        """Getting a cache and then mutating it (as generation does) must not corrupt stored cache."""
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Mutation test")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        stored_length = cache_length(kv_prefix_cache.caches[0])

        # Get cache and mutate it (simulating what generation does)
        result_cache, _, matched_index, _ = kv_prefix_cache.get_kv_cache(model, tokens)
        assert matched_index == 0

        # Simulate generation: feed many additional tokens through the cache
        head_dim = result_cache[0].keys.shape[-1]
        num_heads = result_cache[0].keys.shape[1]
        extra_keys = mx.random.normal((1, num_heads, 50, head_dim))
        extra_values = mx.random.normal((1, num_heads, 50, head_dim))
        for layer_cache in result_cache:
            layer_cache.update_and_fetch(extra_keys, extra_values)
        mx.eval([c.keys for c in result_cache])

        # Stored cache must be unchanged
        assert cache_length(kv_prefix_cache.caches[0]) == stored_length

    def test_stored_cache_survives_repeated_get_mutate_cycles(
        self, model_and_tokenizer
    ):
        """Multiple get+mutate cycles (like repeated user requests) must not corrupt cache."""
        model, tokenizer = model_and_tokenizer

        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Repeat test")],
            max_output_tokens=1,
        )
        prompt = apply_chat_template(tokenizer, task)
        tokens = encode_prompt(tokenizer, prompt)
        cache = make_kv_cache(model)

        _, _, snapshots = prefill(
            model,
            tokenizer,
            make_sampler(0.0),
            tokens,
            cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

        kv_prefix_cache = KVPrefixCache(None)
        kv_prefix_cache.add_kv_cache(tokens, cache, snapshots)

        stored_length = cache_length(kv_prefix_cache.caches[0])

        for i in range(3):
            result_cache, _, _, _ = kv_prefix_cache.get_kv_cache(model, tokens)

            head_dim = result_cache[0].keys.shape[-1]
            num_heads = result_cache[0].keys.shape[1]
            extra = mx.random.normal((1, num_heads, 30, head_dim))
            for layer_cache in result_cache:
                layer_cache.update_and_fetch(extra, extra)
            mx.eval([c.keys for c in result_cache])

            assert cache_length(kv_prefix_cache.caches[0]) == stored_length, (
                f"Failed on loop {i}"
            )

    def test_mlx_generate_populates_cache(self, model_and_tokenizer):
        """mlx_generate should save the post-prefill cache (before the decode loop)."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)
        prompt_tokens = encode_prompt(tokenizer, prompt)

        # Consume the entire generator so the cache-saving code after yield runs
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        assert len(kv_prefix_cache.prompts) == 1
        assert len(kv_prefix_cache.caches) == 1
        # add_kv_cache is called before the decode loop and stores a deepcopy of
        # the cache as it is just after prefill + trim(2). Generation tokens are
        # never written into the stored entry.
        assert cache_length(kv_prefix_cache.caches[0]) == len(prompt_tokens) - 2

    def test_mlx_generate_second_call_gets_prefix_hit(self, model_and_tokenizer):
        """Second mlx_generate call with same prompt should get a prefix hit from stored cache."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Reuse test")],
            max_output_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)
        prompt_tokens = encode_prompt(tokenizer, prompt)

        # First generation populates cache
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        assert len(kv_prefix_cache.prompts) == 1

        # Second call should find a prefix match (the stored cache contains
        # prompt + generated tokens, which shares the prompt prefix)
        result_cache, remaining_tokens, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, prompt_tokens
        )
        # The stored cache is longer than the prompt (it includes generated tokens),
        # so this is a prefix match where our prompt is fully contained
        assert matched_index == 0
        # Exact match: remaining_tokens is just the last token and the one before
        assert len(remaining_tokens) == 2
        assert mx.array_equal(remaining_tokens, prompt_tokens[-2:])

    def test_mlx_generate_long_prompt_updates_cache_in_place(self, model_and_tokenizer):
        """With a prompt > 1000 tokens, second generation should update the cache entry in-place."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)

        # Build a long user message (> 1000 tokens) to exceed _MIN_PREFIX_HIT_TO_UPDATE
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = tokenizer.encode(base_text)
        repeats = (1200 // len(base_tokens)) + 2
        long_content = base_text * repeats

        task1 = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content=long_content)],
            max_output_tokens=5,
        )
        prompt1 = apply_chat_template(tokenizer, task1)
        prompt1_tokens = encode_prompt(tokenizer, prompt1)
        assert len(prompt1_tokens) > 1000, (
            "Prompt must exceed _MIN_PREFIX_HIT_TO_UPDATE"
        )

        # First generation populates the cache (must prefill all tokens)
        t0 = time.perf_counter()
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task1,
            prompt=prompt1,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass
        first_gen_time = time.perf_counter() - t0

        assert len(kv_prefix_cache.prompts) == 1
        first_cache_length = cache_length(kv_prefix_cache.caches[0])

        # Second generation: same long prompt + extra content (simulating multi-turn)
        task2 = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[
                InputMessage(role="user", content=long_content),
                InputMessage(role="assistant", content="Sure, I can help."),
                InputMessage(role="user", content="Tell me more."),
            ],
            max_output_tokens=5,
        )
        prompt2 = apply_chat_template(tokenizer, task2)
        prompt2_tokens = encode_prompt(tokenizer, prompt2)

        # Verify the prompts share a long prefix
        prefix_len = get_prefix_length(prompt2_tokens, prompt1_tokens)
        assert prefix_len > 1000, "Prompts must share > 1000 token prefix"

        # Second generation should reuse the cached prefix (only prefill new tokens)
        t0 = time.perf_counter()
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task2,
            prompt=prompt2,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass
        second_gen_time = time.perf_counter() - t0

        # Second generation should be significantly faster due to prefix cache hit - hopefully not flaky
        assert second_gen_time < first_gen_time * 0.5, (
            f"Expected prefix cache speedup: "
            f"first={first_gen_time:.2f}s, second={second_gen_time:.2f}s"
        )

        # With prefix_hit > 1000, should update in-place (not add a second entry)
        assert len(kv_prefix_cache.prompts) == 1
        # Updated cache should be longer (prompt2 + generated > prompt1 + generated)
        updated_cache_length = cache_length(kv_prefix_cache.caches[0])
        assert updated_cache_length > first_cache_length

    def test_mlx_generate_stored_cache_not_mutated(self, model_and_tokenizer):
        """After mlx_generate saves a cache, a second generation must not corrupt the stored copy."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)
        task = TextGenerationTaskParams(
            model=DEFAULT_GPT_OSS_MODEL_ID,
            input=[InputMessage(role="user", content="Immutable test")],
            max_output_tokens=5,
        )
        prompt = apply_chat_template(tokenizer, task)

        # First generation populates cache
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        firstcache_length = cache_length(kv_prefix_cache.caches[0])

        # Second generation gets the cache and mutates it during generation
        for _response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=None,
        ):
            pass

        # The first stored cache must not have been mutated by the second generation
        assert cache_length(kv_prefix_cache.caches[0]) == firstcache_length

    def test_evicts_lru_entry_under_memory_pressure(self, model_and_tokenizer):
        """Under memory pressure, adding a new cache entry evicts the least recently used one."""
        model, tokenizer = model_and_tokenizer

        kv_prefix_cache = KVPrefixCache(None)

        # Add three cache entries with different prompts
        prompts = ["First entry", "Second entry", "Third entry"]
        for i, content in enumerate(prompts):
            task = TextGenerationTaskParams(
                model=DEFAULT_GPT_OSS_MODEL_ID,
                input=[InputMessage(role="user", content=content)],
                max_output_tokens=1,
            )
            prompt = apply_chat_template(tokenizer, task)
            tokens = encode_prompt(tokenizer, prompt)
            cache = make_kv_cache(model)
            prefill(
                model,
                tokenizer,
                make_sampler(0.0),
                tokens,
                cache,
                group=None,
                on_prefill_progress=None,
                distributed_prompt_progress_callback=None,
            )
            kv_prefix_cache.add_kv_cache(tokens, cache)
            # Stagger _last_used so LRU order is deterministic
            kv_prefix_cache._last_used[i] = float(i)

        assert len(kv_prefix_cache.prompts) == 3

        # Access the third entry to make it most recently used
        kv_prefix_cache._last_used[2] = 100.0
        # Entry 0 (_last_used=0.0) is LRU, entry 1 (_last_used=1.0) is next

        # Simulate memory pressure: return usage above _MEMORY_THRESHOLD (0.9)
        with patch(
            "exo.worker.engines.mlx.cache.get_memory_used_percentage",
            return_value=0.95,
        ):
            # Trigger eviction by adding a new entry
            task = TextGenerationTaskParams(
                model=DEFAULT_GPT_OSS_MODEL_ID,
                input=[InputMessage(role="user", content="New entry")],
                max_output_tokens=1,
            )
            prompt = apply_chat_template(tokenizer, task)
            tokens = encode_prompt(tokenizer, prompt)
            cache = make_kv_cache(model)
            prefill(
                model,
                tokenizer,
                make_sampler(0.0),
                tokens,
                cache,
                group=None,
                on_prefill_progress=None,
                distributed_prompt_progress_callback=None,
            )
            kv_prefix_cache.add_kv_cache(tokens, cache)

        # LRU entries should have been evicted (entries 0, 1, 2 in order of _last_used)
        # Since fake_active stays above threshold after each eviction (we don't change it),
        # all old entries get evicted, leaving only the newly added one
        assert len(kv_prefix_cache.prompts) == 1
        # The surviving entry should be the newly added one
        assert get_prefix_length(kv_prefix_cache.prompts[0], tokens) == len(tokens)
