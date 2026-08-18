"""Internal shape-checking helper used by dataclasses' ``__post_init__``.

Not part of the public API.
"""

from __future__ import annotations

import jax


def check_shape(
    array: jax.Array, expected: tuple[int | None, ...], name: str
) -> None:
    """Raise ``ValueError`` unless ``array.shape`` matches ``expected``.

    Parameters
    ----------
    array : jax.Array
        The array whose shape to check.
    expected : tuple[int | None, ...]
        The expected shape. ``None`` entries accept any size along that
        axis.
    name : str
        A label for ``array`` used in the raised error message.

    Returns
    -------
    None
    """
    if len(array.shape) != len(expected) or any(
        expected_size is not None and actual_size != expected_size
        for actual_size, expected_size in zip(array.shape, expected, strict=True)
    ):
        raise ValueError(
            f"{name} must have shape {expected} (None = any size), got "
            f"{array.shape}."
        )
