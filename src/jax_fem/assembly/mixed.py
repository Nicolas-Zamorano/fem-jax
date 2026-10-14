"""
Block saddle-point system for the mixed Poisson formulation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_fem.assembly.global_assembly import (
    GlobalMatrix,
    assemble_matrix,
    assemble_vector,
    combine_matrix_blocks,
)
from jax_fem.assembly.integration import integrate_cellwise
from jax_fem.forms.elliptic import elliptic_linear_form
from jax_fem.forms.mixed import elliptic_mixed_div_form, elliptic_mixed_mass_form
from jax_fem.problem.elliptic import EllipticProblem, evaluate_elliptic_coefficients
from jax_fem.space.cell_basis import CellBasis


def assemble_mixed_system(
    basis_sigma: CellBasis,
    basis_u: CellBasis,
    problem: EllipticProblem,
) -> tuple[GlobalMatrix, jax.Array]:
    r"""
    Assemble the mixed Poisson saddle-point system.

    Parameters
    ----------
    basis_sigma : CellBasis
        Evaluated H(div) basis (``mapping == "contravariant_piola"``, e.g.
        RT0).
    basis_u : CellBasis
        Evaluated scalar basis (``mapping == "identity"``, e.g. P0), on the
        cells and quadrature points of ``basis_sigma``.
    problem : EllipticProblem
        The elliptic problem; only ``diffusion`` and ``source`` are used.

    Returns
    -------
    block_matrix : GlobalMatrix
        Saddle-point matrix, shape ``(dofs_sigma + dofs_u, dofs_sigma + dofs_u)``.
    rhs : jax.Array
        Right-hand side ``[0; F]``, shape ``(dofs_sigma + dofs_u, 1)``.

    Notes
    -----
    Assembles

    .. math::
        \begin{bmatrix} M & -B^T \\ B & 0 \end{bmatrix}
        \begin{bmatrix} \sigma \\ u \end{bmatrix} =
        \begin{bmatrix} 0 \\ F \end{bmatrix}

    with ``M`` from ``elliptic_mixed_mass_form`` and ``B`` from
    ``elliptic_mixed_div_form`` (scalar test space, H(div) trial space).

    Only homogeneous Dirichlet data is supported: the natural boundary term
    ``-int_{Gamma_D} g (tau . n) ds`` would need boundary-facet integration of
    the H(div) trace, which the library does not provide.
    ``problem.dirichlet_conditions`` is not consulted, and advection and
    reaction are not part of the mixed Poisson formulation.
    """
    coefficients_sigma = evaluate_elliptic_coefficients(
        problem, basis_sigma.physical_points
    )
    mass_matrix = assemble_matrix(
        integrate_cellwise(
            elliptic_mixed_mass_form(basis_sigma, basis_sigma, coefficients_sigma),
            basis_sigma.physical_weights,
        ),
        basis_sigma.space,
        basis_sigma.space,
    )
    divergence_matrix = assemble_matrix(
        integrate_cellwise(
            elliptic_mixed_div_form(basis_u, basis_sigma, None),
            basis_u.physical_weights,
        ),
        basis_u.space,
        basis_sigma.space,
    )
    block_matrix = combine_matrix_blocks(
        ((mass_matrix, -divergence_matrix.T), (divergence_matrix, None))
    )

    coefficients_u = evaluate_elliptic_coefficients(problem, basis_u.physical_points)
    load_vector = assemble_vector(
        integrate_cellwise(
            elliptic_linear_form(basis_u, coefficients_u), basis_u.physical_weights
        ),
        basis_u.space,
    )
    flux_rhs = jnp.zeros(
        (basis_sigma.space.number_of_dofs, 1), dtype=load_vector.dtype
    )
    rhs = jnp.concatenate((flux_rhs, load_vector), axis=-2)

    return block_matrix, rhs
