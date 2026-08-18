"""
Global matrix assembly.
"""

from typing import TypeAlias

import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse

from jax_fem.assembly.integration import integrate_cellwise
from jax_fem.forms.protocols import BilinearForm
from jax_fem.problem.elliptic import EllipticCoefficientValues
from jax_fem.space.cell_basis import CellBasis

#: The library's sparse global matrix representation
GlobalMatrix: TypeAlias = jax_sparse.BCOO


def assemble_bilinear_form(
    form: BilinearForm,
    basis: CellBasis,
    coefficients: EllipticCoefficientValues,
) -> GlobalMatrix:
    """
    Assemble a bilinear form into a global sparse matrix.

    Evaluates the pointwise form, integrates it over each cell, and scatters
    the local ``(K, N_phi, N_phi)`` matrices into a global ``(N_dofs, N_dofs)``
    sparse matrix using ``space.cells_to_dofs`` for the row and column indices,
    via one flat construction rather than a per-cell scatter.

    Local contributions from cells sharing a DOF land on the same ``(row, col)``
    index pair and are **not** eagerly deduplicated: the returned matrix may store
    several entries at the same index. JAX's ``BCOO`` sums duplicate-index entries
    on ``todense()``, matrix products, and other consuming operations, so this is
    transparent to callers; call ``.sum_duplicates()`` explicitly for a compacted
    representation.

    Parameters
    ----------
    form : BilinearForm
        The bilinear form to assemble.
    basis : CellBasis
        Evaluated geometry and basis.
    coefficients : EllipticCoefficientValues
        Evaluated elliptic coefficients at ``basis.physical_points``.

    Returns
    -------
    GlobalMatrix
        Shape ``(N_dofs, N_dofs)``.
    """
    pointwise = form(basis, coefficients)  # (K, Q, N_phi, N_phi)
    local_matrices = integrate_cellwise(pointwise, basis.physical_weights)

    cells_to_dofs = basis.space.cells_to_dofs  # (K, N_phi)
    number_of_dofs = basis.space.number_of_dofs

    row_indices = jnp.broadcast_to(cells_to_dofs[:, :, None], local_matrices.shape)
    col_indices = jnp.broadcast_to(cells_to_dofs[:, None, :], local_matrices.shape)
    flat_indices = jnp.stack(
        (row_indices.reshape(-1), col_indices.reshape(-1)), axis=-1
    )
    flat_data = local_matrices.reshape(-1)

    return jax_sparse.BCOO(
        (flat_data, flat_indices), shape=(number_of_dofs, number_of_dofs)
    )
