from __future__ import annotations

from collections.abc import Iterator


def walk_forward_slices(n_samples: int, train_size: int, step: int, test_size: int) -> Iterator[tuple[slice, slice]]:
    """Yield expanding-window train/test slices for time series validation."""
    if min(n_samples, train_size, step, test_size) <= 0:
        raise ValueError("All parameters must be positive")
    train_end = train_size
    while train_end + test_size <= n_samples:
        yield slice(0, train_end), slice(train_end, train_end + test_size)
        train_end += step
