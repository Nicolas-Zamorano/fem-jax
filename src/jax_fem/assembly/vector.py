"""
Global vector assembly.
"""

import jax
import jax.numpy as jnp

from jax_fem.assembly.integration import integrate_cellwise
from jax_fem.forms.protocols import LinearForm
from jax_fem.problem.elliptic import EllipticCoefficientValues
from jax_fem.space.cell_basis import CellBasis


def assemble_linear_form(
    form: LinearForm,
    basis: CellBasis,
    coefficients: EllipticCoefficientValues,
) -> jax.Array:
    """
    Assemble a linear form into a global dense vector.

    Parameters
    ----------
    form : LinearForm
        The pointwise form to assemble, returning shape ``(K, Q, N_test, 1)``.
    basis : CellBasis
        Evaluated test-space geometry and basis.
    coefficients : EllipticCoefficientValues
        Evaluated elliptic coefficients at ``basis.physical_points``.

    Returns
    -------
    global_vector : jax.Array
        Assembled load vector, shape ``(N_dofs, 1)``.

    Notes
    -----
    The local ``(K, N_test, 1)`` vectors are accumulated into the global
    vector through ``space.cells_to_dofs`` with one flat scatter-add rather
    than a per-cell scatter.
    """
    pointwise = form(basis, coefficients)
    local_vectors = integrate_cellwise(pointwise, basis.physical_weights)

    flat_indices = basis.space.cells_to_dofs.ravel()
    flat_values = local_vectors.ravel()

    global_vector = jnp.zeros((basis.space.number_of_dofs,), dtype=flat_values.dtype)
    global_vector = global_vector.at[flat_indices].add(flat_values)

    return jnp.expand_dims(global_vector, axis=-1)
