"""
Block saddle-point system assembly for the mixed Poisson formulation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse

from jax_fem.assembly.matrix import GlobalMatrix, assemble_bilinear_form
from jax_fem.assembly.vector import assemble_linear_form
from jax_fem.forms.elliptic import elliptic_linear_form
from jax_fem.forms.mixed import elliptic_mixed_div_form, elliptic_mixed_mass_form
from jax_fem.problem.elliptic import EllipticProblem, evaluate_elliptic_coefficients
from jax_fem.space.cell_basis import CellBasis

#: Alias for assemble_bilinear_form when assembling mixed/two-space forms.
assemble_mixed_bilinear_form = assemble_bilinear_form


def assemble_mixed_system(
    basis_sigma: CellBasis,
    basis_u: CellBasis,
    problem: EllipticProblem,
) -> tuple[GlobalMatrix, jax.Array]:
    r"""
    Assemble the mixed Poisson saddle-point block system.

    Parameters
    ----------
    basis_sigma : CellBasis
        Evaluated H(div) geometry and basis (``mapping ==
        "contravariant_piola"``, e.g. RT0).
    basis_u : CellBasis
        Evaluated scalar geometry and basis (``mapping == "identity"``,
        e.g. P0). Must share the cells and quadrature points of
        ``basis_sigma``.
    problem : EllipticProblem
        The elliptic problem; only ``.diffusion`` and ``.source`` are used.

    Returns
    -------
    block_matrix : GlobalMatrix
        Saddle-point matrix, shape ``(N_sigma + N_u, N_sigma + N_u)``.
    rhs : jax.Array
        Right-hand side ``[0; F]``, shape ``(N_sigma + N_u, 1)``.

    Notes
    -----
    Assembles

    .. math::
        \begin{bmatrix} M & -B^T \\ B & 0 \end{bmatrix}
        \begin{bmatrix} \sigma \\ u \end{bmatrix} =
        \begin{bmatrix} G \\ F \end{bmatrix}

    where ``M`` is ``elliptic_mixed_mass_form`` (single-space) and ``B`` is
    ``elliptic_mixed_div_form`` (P0 test space, RT0 trial space), both
    assembled with ``assemble_bilinear_form``. The blocks are combined
    directly at the sparse ``(data, indices)`` level, offsetting ``B``'s
    row/column indices into the combined numbering.

    Only homogeneous Dirichlet data (``g = 0``) is supported, so ``G = 0``
    identically and is omitted: the general natural boundary term
    ``-int_{Gamma_D} g (tau . n) ds`` needs boundary-facet integration of
    the H(div) trace, which this library does not yet provide.
    ``problem.dirichlet_conditions`` is not consulted, and advection and
    reaction are not part of the mixed Poisson formulation, even though
    ``problem.advection`` and ``problem.reaction`` must still be valid
    callables to satisfy ``EllipticProblem``.
    """
    n_sigma = basis_sigma.space.number_of_dofs
    n_u = basis_u.space.number_of_dofs

    coefficients_sigma = evaluate_elliptic_coefficients(
        problem, basis_sigma.physical_points
    )
    mass_matrix = assemble_bilinear_form(
        elliptic_mixed_mass_form, basis_sigma, coefficients=coefficients_sigma
    )
    div_matrix = assemble_bilinear_form(elliptic_mixed_div_form, basis_u, basis_sigma)

    m_rows, m_cols = mass_matrix.indices[..., 0], mass_matrix.indices[..., 1]
    b_rows, b_cols = div_matrix.indices[..., 0], div_matrix.indices[..., 1]

    # Top-right block -B^T: sigma-numbered rows, u-numbered columns (offset
    # into the combined numbering).
    bt_rows = b_cols
    bt_cols = b_rows + n_sigma
    # Bottom-left block B: u-numbered rows (offset), sigma-numbered columns.
    b_block_rows = b_rows + n_sigma
    b_block_cols = b_cols

    all_rows = jnp.concatenate((m_rows, bt_rows, b_block_rows), axis=-1)
    all_cols = jnp.concatenate((m_cols, bt_cols, b_block_cols), axis=-1)
    all_data = jnp.concatenate(
        (mass_matrix.data, -div_matrix.data, div_matrix.data), axis=-1
    )

    total_dofs = n_sigma + n_u
    block_matrix = jax_sparse.BCOO(
        (all_data, jnp.stack((all_rows, all_cols), axis=-1)),
        shape=(total_dofs, total_dofs),
    )

    coefficients_u = evaluate_elliptic_coefficients(problem, basis_u.physical_points)
    load_vector = assemble_linear_form(elliptic_linear_form, basis_u, coefficients_u)
    rhs = jnp.concatenate((jnp.zeros((n_sigma, 1)), load_vector), axis=-2)

    return block_matrix, rhs
