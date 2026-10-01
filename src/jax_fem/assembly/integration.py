"""
Cell-local integration of pointwise form values.
"""

import jax
import jax.numpy as jnp


def integrate_cellwise(
    pointwise_values: jax.Array, physical_weights: jax.Array
) -> jax.Array:
    """
    Integrate pointwise form values over each cell's quadrature points.

    Parameters
    ----------
    pointwise_values : jax.Array
        Evaluated form integrand, shape ``(..., Q, N_test, N_trial)``: a
        bilinear form gives ``(K, Q, N_test, N_trial)``, a linear form
        ``(K, Q, N_test, 1)``, and a functional ``(K, Q, 1, 1)``.
    physical_weights : jax.Array
        Physical quadrature weights, shape ``(..., Q, 1, 1)``.

    Returns
    -------
    cellwise_values : jax.Array
        Integrated local contributions, shape ``(..., N_test, N_trial)``.
    """
    return jnp.sum(physical_weights * pointwise_values, axis=-3)
