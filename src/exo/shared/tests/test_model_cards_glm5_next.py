from pathlib import Path

import pytest
import tomlkit

from exo.shared.constants import RESOURCES_DIR
from exo.shared.models.model_cards import ConfigData, ModelCard


@pytest.mark.parametrize(
    ("filename", "model_id", "quantization", "storage_size"),
    [
        (
            "mlx-community--GLM-5.3-Flash-4bit.toml",
            "mlx-community/GLM-5.3-Flash-4bit",
            "4bit",
            203_992_076_296,
        ),
        (
            "pipenetwork--GLM-5.3-Flash-MLX-4bit.toml",
            "pipenetwork/GLM-5.3-Flash-MLX-4bit",
            "4bit",
            177_548_978_628,
        ),
        (
            "pipenetwork--GLM-5.3-Flash-MLX-6bit.toml",
            "pipenetwork/GLM-5.3-Flash-MLX-6bit",
            "6bit",
            255_837_890_513,
        ),
        (
            "pipenetwork--GLM-5.3-Flash-MLX-8bit.toml",
            "pipenetwork/GLM-5.3-Flash-MLX-8bit",
            "8bit",
            334_126_804_629,
        ),
    ],
)
def test_glm5_next_builtin_model_cards(
    filename: str,
    model_id: str,
    quantization: str,
    storage_size: int,
) -> None:
    card_path = Path(RESOURCES_DIR) / "inference_model_cards" / filename
    card = ModelCard.model_validate(tomlkit.loads(card_path.read_text()))

    assert str(card.model_id) == model_id
    assert card.n_layers == 45
    assert card.hidden_size == 4096
    assert card.num_key_value_heads == 64
    assert card.supports_tensor
    assert card.quantization == quantization
    assert card.context_length == 1_048_576
    assert card.storage_size.in_bytes == storage_size
    assert card.capabilities == ["text", "thinking", "vision"]
    assert card.vision is not None
    assert card.vision.image_token_id == 154854
    assert card.vision.model_type == "glm5_next"
    assert card.vision.weights_repo == model_id


def test_glm5_next_config_supports_tensor_and_vision() -> None:
    config = ConfigData.model_validate(
        {
            "architectures": ["Glm5NextForConditionalGeneration"],
            "model_type": "glm5_next",
            "text_config": {
                "hidden_size": 4096,
                "max_position_embeddings": 1_048_576,
                "num_hidden_layers": 45,
                "num_key_value_heads": 64,
            },
            "vision_config": {
                "model_type": "glm5_next_vision",
            },
            "image_token_id": 154854,
        },
        context={"model_id": "mlx-community/GLM-5.3-Flash-4bit"},
    )

    assert config.layer_count == 45
    assert config.hidden_size == 4096
    assert config.max_position_embeddings == 1_048_576
    assert config.supports_tensor
    assert config.vision is not None
    assert config.vision.image_token_id == 154854
    assert config.vision.model_type == "glm5_next"
    assert config.vision.weights_repo == "mlx-community/GLM-5.3-Flash-4bit"
