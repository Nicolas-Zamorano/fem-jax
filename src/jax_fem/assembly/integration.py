"""
Cell-local integration of pointwise form values.
"""

import jax
import jax.numpy as jnp


def integrate_cellwise(
    pointwise_values: jax.Array, physical_weights: jax.Array
) -> jax.Array:
    """Integrate pointwise form values over each cell's quadrature points.

    Parameters
    ----------
    pointwise_values : jax.Array
        Evaluated form integrand, shape ``(K, Q, rows, columns)``: a
        bilinear form gives ``(K, Q, N_phi, N_phi)``, a linear form gives
        ``(K, Q, N_phi, 1)``, a functional gives ``(K, Q, 1, 1)``.
    physical_weights : jax.Array
        Physical quadrature weights, shape ``(K, Q, 1, 1)``.

    Returns
    -------
    cell_wise_value : jax.Array
        Shape ``(K, rows, columns)``: bilinear form ``(K, N_phi, N_phi)``,
        linear form ``(K, N_phi, 1)``, functional ``(K, 1, 1)``.
    """
    return jnp.sum(
        physical_weights * pointwise_values,
        axis=1,
    )
