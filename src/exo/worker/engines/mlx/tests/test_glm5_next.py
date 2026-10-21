from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.worker.engines.mlx import glm5_next


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
        cache: list[object] | None = None,
        inputs_embeds: mx.array | None = None,
    ) -> _FakeOutput:
        del cache, inputs_embeds
        return _FakeOutput(logits=inputs.astype(mx.float32))


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
