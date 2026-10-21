from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.models.glm5_next import LanguageModel, TextConfig

from exo.worker.engines.mlx import glm5_next
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
