from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from exo.worker.engines.mlx import glm5_next


class _FakeLanguageModel:
    def __init__(self) -> None:
        self.model = SimpleNamespace(layers=[], embed_tokens=object())
        self.layers: list[Any] = []

    def make_cache(self) -> list[Any]:
        return ["cache"]

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        inputs_embeds: mx.array | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        del cache, inputs_embeds
        return SimpleNamespace(logits=inputs.astype(mx.float32))


def test_glm5_next_adapter_exposes_exo_model_contract(
    monkeypatch: Any,
) -> None:
    language_model = _FakeLanguageModel()
    monkeypatch.setattr(glm5_next, "LanguageModel", _FakeLanguageModel)
    monkeypatch.setattr(
        glm5_next,
        "load_vlm_model",
        lambda _path, lazy: SimpleNamespace(language_model=language_model),
    )

    model = glm5_next.load_glm5_next_model(Path("model"), lazy=True)
    inputs = mx.array([[1, 2]])

    assert model.model_type == "glm5_next"
    assert model.make_cache() == ["cache"]
    assert mx.array_equal(model(inputs), inputs.astype(mx.float32)).item()
