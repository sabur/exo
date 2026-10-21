from exo.shared.models.model_cards import ConfigData


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
