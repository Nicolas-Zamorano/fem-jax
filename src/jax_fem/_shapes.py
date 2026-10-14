"""
Internal shape checking used by dataclass validation.
"""

from __future__ import annotations

import jax


def check_shape(
    array: jax.Array, expected: tuple[int | None, ...], name: str
) -> None:
    """
    Check the shape of an array.

    Parameters
    ----------
    array : jax.Array
        The array whose shape to check.
    expected : tuple[int | None, ...]
        The expected shape. ``None`` entries accept any size along that
        axis.
    name : str
        A label for ``array`` used in the raised error message.

    Raises
    ------
    ValueError
        If ``array.shape`` does not match ``expected``.
    """
    if len(array.shape) != len(expected) or any(
        expected_size is not None and actual_size != expected_size
        for actual_size, expected_size in zip(array.shape, expected, strict=True)
    ):
        raise ValueError(
            f"{name} must have shape {expected} (None = any size), got "
            f"{array.shape}."
        )
