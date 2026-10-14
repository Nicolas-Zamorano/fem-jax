"""
Cell-local integration of pointwise values.
"""

import jax
import jax.numpy as jnp


def integrate_cellwise(
    pointwise_values: jax.Array, physical_weights: jax.Array
) -> jax.Array:
    """
    Integrate pointwise values over each cell.

    Parameters
    ----------
    pointwise_values : jax.Array
        Pointwise values, shape ``(..., K, Q, a, b)``: a bilinear form gives
        ``(K, Q, N_test, N_trial)``, a linear form ``(K, Q, N_test, 1)``, and
        a functional ``(K, Q, 1, 1)``.
    physical_weights : jax.Array
        Physical quadrature weights, shape ``(..., K, Q, 1, 1)``.

    Returns
    -------
    local_values : jax.Array
        Integrated local contributions, shape ``(..., K, 1, a, b)``.

    Notes
    -----
    The quadrature axis is kept with size one, so the result has the same
    layout as every other cellwise array. Facet quantities of shape
    ``(F, Q, a, b)`` are integrated the same way with facet weights.
    """
    return jnp.sum(physical_weights * pointwise_values, axis=-3, keepdims=True)
