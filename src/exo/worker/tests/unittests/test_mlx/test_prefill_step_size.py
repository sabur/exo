import pytest

from exo.worker.engines.mlx.generator.generate import _parse_prefill_step_size


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 4096),
        ("4096", 4096),
        ("2048", 2048),
        ("1024", 1024),
    ],
)
def test_parse_prefill_step_size_accepts_benchmark_values(
    value: str | None,
    expected: int,
) -> None:
    assert _parse_prefill_step_size(value) == expected


@pytest.mark.parametrize("value", ["", "abc", "0", "-4", "3", "1025"])
def test_parse_prefill_step_size_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="EXO_PREFILL_STEP_SIZE"):
        _parse_prefill_step_size(value)
