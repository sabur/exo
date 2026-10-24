from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.models.glm5_next import LanguageModel, TextConfig
from mlx_vlm.models.glm5_next.language import Glm5NextLinearAttention, Glm5NextMLP

from exo.worker.engines.mlx import auto_parallel, glm5_next
from exo.worker.engines.mlx.cache import cache_length
from exo.worker.engines.mlx.types import KVCacheType


@dataclass
class _FakeOutput:
    logits: mx.array


class _FakeLoadedModel(nn.Module):
    def __init__(self, language_model: object) -> None:
        super().__init__()
        self.language_model = language_model


class _FakeTextModel:
    def __init__(self) -> None:
        self.layers: list[nn.Module] = []
        self.embed_tokens = object()


class _FakeLanguageModel:
    def __init__(self) -> None:
        self.model = _FakeTextModel()
        self.layers: list[nn.Module] = []

    def make_cache(self) -> list[object]:
        return ["cache"]

    def __call__(
        self,
        inputs: mx.array,
        cache: KVCacheType | None = None,
        inputs_embeds: mx.array | None = None,
    ) -> _FakeOutput:
        del cache, inputs_embeds
        return _FakeOutput(logits=inputs.astype(mx.float32))


class _FakeGroup:
    def size(self) -> int:
        return 2

    def rank(self) -> int:
        return 0


def _glm_sharding_strategy() -> auto_parallel.Glm5NextShardingStrategy:
    def identity(module: nn.Module) -> nn.Module:
        return module

    def no_op(_module: nn.Module) -> None:
        return None

    return auto_parallel.Glm5NextShardingStrategy(
        cast(mx.distributed.Group, _FakeGroup()),
        identity,  # type: ignore[arg-type]
        identity,  # type: ignore[arg-type]
        no_op,
        no_op,
    )


def _small_glm5_config() -> TextConfig:
    return TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_nextn_predict_layers=0,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_rope_head_dim=0,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_topk=8,
        index_head_dim=8,
        index_n_heads=4,
        index_kpool=4,
        linear_head_dim=8,
        linear_num_heads=4,
        linear_conv_kernel_dim=4,
        hc_mult=2,
        first_k_dense_replace=3,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
        indexer_types=["full", "full", "full", "full"],
    )


def test_glm5_next_adapter_exposes_exo_model_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    language_model = _FakeLanguageModel()

    def load_model(_path: Path, *, lazy: bool) -> nn.Module:
        del lazy
        return _FakeLoadedModel(language_model=language_model)

    monkeypatch.setattr(glm5_next, "LanguageModel", _FakeLanguageModel)
    monkeypatch.setattr(glm5_next.vlm_utils, "load_model", load_model)

    model = glm5_next.load_glm5_next_model(Path("model"), lazy=True)
    inputs = mx.array([[1, 2]])

    assert model.model_type == "glm5_next"
    assert model.make_cache() == ["cache"]
    assert mx.array_equal(model(inputs), inputs.astype(mx.float32)).item()


def test_reduced_glm5_next_runs_prompt_and_decode_with_hybrid_cache() -> None:
    model = glm5_next.Glm5NextModel(LanguageModel(_small_glm5_config()))
    cache = model.make_cache()
    prompt = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)

    prompt_logits = model(prompt, cache=cache)
    mx.eval(prompt_logits)

    assert prompt_logits.shape == (1, 6, 128)
    assert cache_length(cache) == 6

    decode_logits = model(mx.array([[7]], dtype=mx.int32), cache=cache)
    mx.eval(decode_logits)

    assert decode_logits.shape == (1, 1, 128)
    assert cache_length(cache) == 7
    assert bool(mx.all(mx.isfinite(decode_logits)).item())


def test_glm5_tensor_sharding_splits_quantized_gate_up_by_logical_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_glm5_config()
    mlp = Glm5NextMLP(config)
    nn.quantize(mlp, group_size=32, bits=4)
    captured_segments: list[list[int]] = []

    def capture_shard(
        module: nn.Module,
        _sharding: str,
        *,
        segments: list[int],
        group: mx.distributed.Group,
    ) -> nn.Module:
        del group
        captured_segments.append(segments)
        return module

    monkeypatch.setattr(auto_parallel, "shard_linear", capture_shard)

    assert mlp.down_proj.weight.shape[-1] != config.intermediate_size
    _glm_sharding_strategy()._shard_gate_up(mlp)

    assert captured_segments == [[config.intermediate_size]]


def test_glm5_tensor_sharding_preserves_quantized_fbg_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_glm5_config()
    attention = Glm5NextLinearAttention(config, layer_idx=0)
    nn.quantize(attention, group_size=32, bits=4)
    original = attention.fbg_a_proj
    assert isinstance(original, nn.QuantizedLinear)
    original_weight = original.weight
    original_scales = original.scales
    original_biases = original.biases

    def preserve_module(
        module: nn.Module,
        _sharding: str,
        *,
        segments: list[int],
        group: mx.distributed.Group,
    ) -> nn.Module:
        del segments, group
        return module

    monkeypatch.setattr(auto_parallel, "shard_linear", preserve_module)
    _glm_sharding_strategy()._shard_linear_attention(attention)

    assert isinstance(attention.fbg_a_proj, nn.QuantizedLinear)
    expected_indices = mx.concatenate(
        [
            mx.arange(config.linear_head_dim),
            mx.arange(
                config.linear_head_dim,
                config.linear_head_dim + config.linear_num_heads // 2,
            ),
            mx.arange(
                config.linear_head_dim + config.linear_num_heads,
                2 * config.linear_head_dim + config.linear_num_heads,
            ),
        ]
    )
    assert mx.array_equal(
        attention.fbg_a_proj.weight, original_weight[expected_indices]
    ).item()
    assert mx.array_equal(
        attention.fbg_a_proj.scales, original_scales[expected_indices]
    ).item()
    assert mx.array_equal(
        attention.fbg_a_proj.biases, original_biases[expected_indices]
    ).item()
