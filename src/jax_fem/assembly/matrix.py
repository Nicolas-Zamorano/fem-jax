"""
Global matrix assembly.
"""

from collections.abc import Callable
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse

from jax_fem.assembly.integration import integrate_cellwise
from jax_fem.forms.protocols import BilinearForm
from jax_fem.problem.elliptic import EllipticCoefficientValues
from jax_fem.space.cell_basis import CellBasis

#: The library's sparse global matrix representation
GlobalMatrix: TypeAlias = jax_sparse.BCOO


def assemble_bilinear_form(
    form: BilinearForm | Callable[[CellBasis, CellBasis], jax.Array],
    basis_test: CellBasis,
    basis_trial: CellBasis | None = None,
    coefficients: EllipticCoefficientValues | None = None,
) -> GlobalMatrix:
    """
    Assemble a bilinear form into a global sparse matrix.

    Parameters
    ----------
    form : BilinearForm | Callable[[CellBasis, CellBasis], jax.Array]
        The pointwise form to assemble, returning shape
        ``(K, Q, N_test, N_trial)``. Called as ``form(basis_test,
        coefficients)`` when ``basis_trial`` is ``None``, and as
        ``form(basis_test, basis_trial)`` otherwise.
    basis_test : CellBasis
        Evaluated test-space geometry and basis, indexing the matrix rows.
    basis_trial : CellBasis | None
        Evaluated trial-space geometry and basis, indexing the matrix
        columns. Must share the cells and quadrature points of
        ``basis_test``. If ``None``, the trial space is the test space.
    coefficients : EllipticCoefficientValues | None
        Evaluated coefficients at ``basis_test.physical_points``. Required
        if ``basis_trial`` is ``None``, and must be ``None`` otherwise.

    Returns
    -------
    global_matrix : GlobalMatrix
        Assembled sparse matrix, shape ``(N_dofs_test, N_dofs_trial)``.

    Notes
    -----
    With ``basis_trial`` omitted, this assembles a single-space square
    matrix (e.g. stiffness or mass). With ``basis_trial`` given, it
    assembles a two-space rectangular matrix (e.g. the mixed divergence
    form's P0 test space against its RT0 trial space).

    Local contributions from cells sharing a DOF land on the same
    ``(row, col)`` index pair and are **not** eagerly deduplicated: the
    returned matrix may store several entries at the same index. JAX's
    ``BCOO`` sums duplicate-index entries on ``todense()``, matrix products,
    and other consuming operations, so this is transparent to callers; call
    ``.sum_duplicates()`` explicitly for a compacted representation.
    """
    if basis_trial is None:
        if coefficients is None:
            raise ValueError(
                "coefficients is required when basis_trial is None (single-space form)."
            )
        basis_trial = basis_test
        pointwise = form(basis_test, coefficients)
    else:
        if coefficients is not None:
            raise ValueError(
                "coefficients must be None when basis_trial is given: a two-space "
                "form is called as form(basis_test, basis_trial)."
            )
        pointwise = form(basis_test, basis_trial)

    local_matrices = integrate_cellwise(pointwise, basis_test.physical_weights)

    row_indices = jnp.broadcast_to(
        jnp.expand_dims(basis_test.space.cells_to_dofs, axis=-1),
        local_matrices.shape,
    )
    col_indices = jnp.broadcast_to(
        jnp.expand_dims(basis_trial.space.cells_to_dofs, axis=-2),
        local_matrices.shape,
    )
    flat_indices = jnp.stack((row_indices.ravel(), col_indices.ravel()), axis=-1)
    flat_data = local_matrices.ravel()

    return jax_sparse.BCOO(
        (flat_data, flat_indices),
        shape=(basis_test.space.number_of_dofs, basis_trial.space.number_of_dofs),
    )
