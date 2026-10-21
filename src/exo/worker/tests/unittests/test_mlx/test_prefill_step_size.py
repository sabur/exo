import pytest

from exo.worker.engines.mlx.constants import parse_prefill_step_size
from exo.worker.engines.mlx.generator import batch_generate, generate


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 2048),
        ("4096", 4096),
        ("2048", 2048),
        ("1024", 1024),
    ],
)
def test_parse_prefill_step_size_accepts_benchmark_values(
    value: str | None,
    expected: int,
) -> None:
    assert parse_prefill_step_size(value) == expected


@pytest.mark.parametrize("value", ["", "abc", "0", "-4", "3", "1025"])
def test_parse_prefill_step_size_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="EXO_PREFILL_STEP_SIZE"):
        parse_prefill_step_size(value)


def test_batch_and_sequential_generators_share_prefill_step_size() -> None:
    assert batch_generate.PREFILL_STEP_SIZE == generate.PREFILL_STEP_SIZE
