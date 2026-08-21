"""
Mixed-space (rectangular) and block saddle-point system assembly.

See Section 5.3 of the implementation plan (``.context/implementation_plan.md``).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse

from jax_fem.assembly.integration import integrate_cellwise
from jax_fem.assembly.matrix import GlobalMatrix, assemble_bilinear_form
from jax_fem.assembly.vector import assemble_linear_form
from jax_fem.forms.elliptic import elliptic_linear_form
from jax_fem.forms.mixed import elliptic_mixed_div_form, elliptic_mixed_mass_form
from jax_fem.problem.elliptic import EllipticProblem, evaluate_elliptic_coefficients
from jax_fem.space.cell_basis import CellBasis


def assemble_mixed_bilinear_form(
    form: Callable[[CellBasis, CellBasis], jax.Array],
    basis_test: CellBasis,
    basis_trial: CellBasis,
) -> GlobalMatrix:
    """
    Assemble a two-space mixed bilinear form into a global sparse matrix.

    Generalizes ``assemble_bilinear_form`` (Section 14.2) to a form whose
    test and trial spaces genuinely differ (e.g.
    ``elliptic_mixed_div_form``'s P0 test space and RT0 trial space): rows
    are indexed by ``basis_test.space.cells_to_dofs``, columns by
    ``basis_trial.space.cells_to_dofs``, independently, so the result need
    not be square. Duplicate-index entries are not eagerly deduplicated, the
    same as ``assemble_bilinear_form``.

    Parameters
    ----------
    form : Callable[[CellBasis, CellBasis], jax.Array]
        The mixed form to assemble, returning shape
        ``(K, Q, N_phi_test, N_phi_trial)``.
    basis_test : CellBasis
        Evaluated test-space geometry and basis.
    basis_trial : CellBasis
        Evaluated trial-space geometry and basis. Must share the same cells
        and quadrature points as ``basis_test`` (built from the same
        ``QuadratureRule``).

    Returns
    -------
    GlobalMatrix
        Shape ``(N_dofs_test, N_dofs_trial)``.
    """
    pointwise = form(basis_test, basis_trial)  # (K, Q, N_test, N_trial)
    local_matrices = integrate_cellwise(pointwise, basis_test.physical_weights)

    cells_to_dofs_test = basis_test.space.cells_to_dofs  # (K, N_test)
    cells_to_dofs_trial = basis_trial.space.cells_to_dofs  # (K, N_trial)
    number_of_dofs_test = basis_test.space.number_of_dofs
    number_of_dofs_trial = basis_trial.space.number_of_dofs

    row_indices = jnp.broadcast_to(
        cells_to_dofs_test[:, :, None], local_matrices.shape
    )
    col_indices = jnp.broadcast_to(
        cells_to_dofs_trial[:, None, :], local_matrices.shape
    )
    flat_indices = jnp.stack(
        (row_indices.reshape(-1), col_indices.reshape(-1)), axis=-1
    )
    flat_data = local_matrices.reshape(-1)

    return jax_sparse.BCOO(
        (flat_data, flat_indices),
        shape=(number_of_dofs_test, number_of_dofs_trial),
    )


def assemble_mixed_system(
    basis_sigma: CellBasis,
    basis_u: CellBasis,
    problem: EllipticProblem,
) -> tuple[GlobalMatrix, jax.Array]:
    r"""
    Assemble the mixed Poisson saddle-point block system.

    .. math::

        \begin{bmatrix} M & -B^T \\ B & 0 \end{bmatrix}
        \begin{bmatrix} \sigma \\ u \end{bmatrix} =
        \begin{bmatrix} G \\ F \end{bmatrix}

    ``M`` (``elliptic_mixed_mass_form``, single-space, assembled with the
    existing ``assemble_bilinear_form``) and ``B``
    (``elliptic_mixed_div_form``, assembled with
    ``assemble_mixed_bilinear_form``) are combined directly at the sparse
    ``(data, indices)`` level -- offsetting ``B``'s row/column indices into
    the combined numbering -- rather than via a generic block-sparse
    concatenation utility.

    **Scope**: only homogeneous Dirichlet data (``g = 0``) is supported, so
    ``G = 0`` identically and is omitted from the assembled system -- the
    general natural boundary term ``-int_{Gamma_D} g (tau . n) ds`` needs
    boundary-facet integration of the H(div) trace, a capability this
    library does not yet have (a future extension, mirroring Section 4.1's
    deferred Neumann term). ``problem.dirichlet_conditions`` is not
    consulted at all; only ``problem.diffusion`` and ``problem.source`` are
    used (advection/reaction are not part of the mixed Poisson formulation,
    Section 5.1, even though ``problem.advection``/``.reaction`` must still
    be valid callables to satisfy ``EllipticProblem``).

    Parameters
    ----------
    basis_sigma : CellBasis
        Evaluated H(div) geometry and basis (``mapping ==
        "contravariant_piola"``, e.g. RT0).
    basis_u : CellBasis
        Evaluated scalar geometry and basis (``mapping == "identity"``,
        e.g. P0). Must share the same cells and quadrature points as
        ``basis_sigma``.
    problem : EllipticProblem
        The elliptic problem; only ``.diffusion`` and ``.source`` are used.

    Returns
    -------
    block_matrix : GlobalMatrix
        Shape ``(N_sigma + N_u, N_sigma + N_u)``.
    rhs : jax.Array
        ``[0; F]``, shape ``(N_sigma + N_u, 1)``.
    """
    space_sigma = basis_sigma.space
    space_u = basis_u.space
    n_sigma = space_sigma.number_of_dofs
    n_u = space_u.number_of_dofs

    coefficients_sigma = evaluate_elliptic_coefficients(
        problem, basis_sigma.physical_points
    )
    mass_matrix = assemble_bilinear_form(
        elliptic_mixed_mass_form, basis_sigma, coefficients_sigma
    )  # (n_sigma, n_sigma), M
    div_matrix = assemble_mixed_bilinear_form(
        elliptic_mixed_div_form, basis_u, basis_sigma
    )  # (n_u, n_sigma), B

    m_rows, m_cols = mass_matrix.indices[:, 0], mass_matrix.indices[:, 1]
    m_data = mass_matrix.data

    b_rows, b_cols = div_matrix.indices[:, 0], div_matrix.indices[:, 1]
    b_data = div_matrix.data

    # Top-right block -B^T: sigma-numbered rows, u-numbered columns (offset
    # into the combined numbering).
    bt_rows = b_cols
    bt_cols = b_rows + n_sigma
    bt_data = -b_data

    # Bottom-left block B: u-numbered rows (offset), sigma-numbered columns.
    b_block_rows = b_rows + n_sigma
    b_block_cols = b_cols

    all_rows = jnp.concatenate((m_rows, bt_rows, b_block_rows))
    all_cols = jnp.concatenate((m_cols, bt_cols, b_block_cols))
    all_data = jnp.concatenate((m_data, bt_data, b_data))

    total_dofs = n_sigma + n_u
    block_matrix = jax_sparse.BCOO(
        (all_data, jnp.stack((all_rows, all_cols), axis=-1)),
        shape=(total_dofs, total_dofs),
    )

    coefficients_u = evaluate_elliptic_coefficients(problem, basis_u.physical_points)
    load_vector = assemble_linear_form(
        elliptic_linear_form, basis_u, coefficients_u
    )  # (n_u, 1), F

    rhs = jnp.concatenate((jnp.zeros((n_sigma, 1)), load_vector), axis=0)

    return block_matrix, rhs
