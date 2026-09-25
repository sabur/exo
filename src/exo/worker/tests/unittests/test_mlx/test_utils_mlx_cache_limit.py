from unittest.mock import patch

import pytest

from exo.worker.engines.mlx.utils_mlx import _configure_mlx_cache_limit


def test_configure_mlx_cache_limit_uses_runtime_default_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_MLX_CACHE_LIMIT_GB", raising=False)

    with patch("exo.worker.engines.mlx.utils_mlx.mx.set_cache_limit") as set_limit:
        _configure_mlx_cache_limit()

    set_limit.assert_not_called()


def test_configure_mlx_cache_limit_applies_explicit_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_MLX_CACHE_LIMIT_GB", "32")

    with patch("exo.worker.engines.mlx.utils_mlx.mx.set_cache_limit") as set_limit:
        _configure_mlx_cache_limit()

    set_limit.assert_called_once_with(32 * (1 << 30))


def test_configure_mlx_cache_limit_rejects_negative_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_MLX_CACHE_LIMIT_GB", "-1")

    with pytest.raises(
        ValueError,
        match="EXO_MLX_CACHE_LIMIT_GB must be greater than or equal to zero",
    ):
        _configure_mlx_cache_limit()
