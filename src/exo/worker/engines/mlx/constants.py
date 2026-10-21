import os

# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = None

DEFAULT_TOP_LOGPROBS: int = 5
DEFAULT_PREFILL_STEP_SIZE: int = 2048


def parse_prefill_step_size(value: str | None) -> int:
    if value is None:
        return DEFAULT_PREFILL_STEP_SIZE

    try:
        step_size = int(value)
    except ValueError as exc:
        raise ValueError("EXO_PREFILL_STEP_SIZE must be an integer") from exc

    if step_size < 4 or step_size % 4 != 0:
        raise ValueError(
            "EXO_PREFILL_STEP_SIZE must be at least 4 and divisible by 4"
        )
    return step_size


PREFILL_STEP_SIZE: int = parse_prefill_step_size(
    os.environ.get("EXO_PREFILL_STEP_SIZE")
)

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
