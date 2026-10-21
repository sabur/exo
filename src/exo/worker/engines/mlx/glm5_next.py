from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.glm5_next import LanguageModel
from mlx_vlm.models.glm5_next.language import Glm5NextTextModel

from exo.worker.engines.mlx.types import KVCacheType


class _LanguageModelOutput(Protocol):
    logits: mx.array


class _VlmUtils(Protocol):
    load_model: Callable[..., nn.Module]


vlm_utils = cast(_VlmUtils, cast(object, import_module("mlx_vlm.utils")))


class Glm5NextModel(nn.Module):
    """Expose MLX-VLM's GLM-5-Next language model through Exo's MLX contract."""

    def __init__(self, language_model: LanguageModel):
        super().__init__()
        self.language_model = language_model
        self.model_type = "glm5_next"

    @property
    def model(self) -> Glm5NextTextModel:
        return self.language_model.model

    @property
    def layers(self) -> list[nn.Module]:
        return cast(list[nn.Module], self.language_model.layers)

    def make_cache(self) -> KVCacheType:
        return cast(KVCacheType, self.language_model.make_cache())

    def __call__(
        self,
        inputs: mx.array,
        cache: KVCacheType | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        output = cast(
            _LanguageModelOutput,
            self.language_model(
                inputs,
                cache=cache,
                inputs_embeds=input_embeddings,
            ),
        )
        return output.logits


def load_glm5_next_model(model_path: Path, *, lazy: bool) -> Glm5NextModel:
    model = vlm_utils.load_model(model_path, lazy=lazy)
    language_model = getattr(model, "language_model", None)
    if not isinstance(language_model, LanguageModel):
        raise TypeError(
            "Expected mlx_vlm.models.glm5_next.LanguageModel, "
            f"got {type(language_model).__name__}"
        )
    return Glm5NextModel(language_model)
