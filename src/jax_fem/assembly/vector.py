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

    Evaluates the pointwise form, integrates it over each cell
    (Section 14.1), and accumulates the local ``(K, N_phi, 1)`` vectors
    into a global ``(N_dofs, 1)`` vector using ``space.cells_to_dofs``, via
    one flat scatter-add rather than a per-cell scatter.

    Parameters
    ----------
    form:
        The linear form to assemble.
    basis:
        Evaluated geometry and basis (Section 10.2).
    coefficients:
        Evaluated elliptic coefficients at ``basis.physical_points``.

    Returns
    -------
    jax.Array
        Shape ``(N_dofs, 1)``.
    """
    pointwise = form(basis, coefficients)  # (K, Q, N_phi, 1)
    local_vectors = integrate_cellwise(pointwise, basis.physical_weights)

    cells_to_dofs = basis.space.cells_to_dofs  # (K, N_phi)
    number_of_dofs = basis.space.number_of_dofs

    flat_indices = cells_to_dofs.reshape(-1)
    flat_values = local_vectors.reshape(-1)

    global_vector = jnp.zeros((number_of_dofs,), dtype=flat_values.dtype)
    global_vector = global_vector.at[flat_indices].add(flat_values)
    return global_vector[:, None]
