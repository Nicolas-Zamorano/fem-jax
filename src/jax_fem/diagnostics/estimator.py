"""
Classical residual-based a posteriori error estimator.

See Section 4 of the implementation plan (``.context/implementation_plan.md``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_fem.assembly.integration import integrate_cellwise
from jax_fem.function.evaluation import evaluate_finite_element_function
from jax_fem.function.finite_element_function import FiniteElementFunction
from jax_fem.mesh.triangle import cell_diameters
from jax_fem.problem.elliptic import EllipticProblem
from jax_fem.space.cell_basis import CellBasis
from jax_fem.space.facet_basis import FacetBasis


def compute_residual_error_estimator(
    solution: FiniteElementFunction,
    basis: CellBasis,
    facet_basis: FacetBasis,
    problem: EllipticProblem,
) -> tuple[jax.Array, jax.Array]:
    r"""
    Classical residual-based a posteriori error estimator.

    .. math::

        \eta_K^2 = h_K^2 \|R_K\|^2_{L^2(K)}
            + \frac{1}{2} \sum_{e \in \partial K \setminus \partial\Omega}
              h_e \|J_e\|^2_{L^2(e)}

    where the interior residual is ``R_K = f - beta . grad(u_h) - c u_h``
    (assuming ``div(A grad(u_h)) == 0``, exact for a constant-in-cell
    reference-gradient element -- e.g. P1 -- with spatially constant ``A``:
    then ``grad(u_h)`` is itself piecewise constant and its divergence
    vanishes identically) and the normal flux jump on interior facet ``e =
    dK+ ^ dK-`` is ``J_e = (A grad(u_h)|+ - A grad(u_h)|-) . n_e``, ``n_e``
    pointing from the plus side to the minus side (``FacetBasis``'
    convention). No Neumann boundary term is included: only interior-facet
    jumps contribute (deferred until the library has Neumann boundary
    conditions).

    Requires ``basis.gradients`` to be quadrature-axis-collapsed (i.e. a
    constant-reference-gradient element, e.g. ``LagrangeTriangleP1``);
    raises for a quadrature-point-varying-gradient element (e.g.
    ``LagrangeTriangleP2``), whose second derivatives (needed for a general
    ``div(A grad(u_h))``) this library does not currently tabulate.

    Parameters
    ----------
    solution : FiniteElementFunction
        The finite element solution to estimate the error of.
    basis : CellBasis
        The cell basis (2D quadrature), matching ``solution.space``.
    facet_basis : FacetBasis
        The facet basis (1D quadrature over interior facets), matching
        ``solution.space``.
    problem : EllipticProblem
        The elliptic problem being estimated.

    Returns
    -------
    cell_indicators : jax.Array
        Per-cell indicators ``eta_K``, shape ``(K,)``.
    global_estimator : jax.Array
        The global estimator ``eta = sqrt(sum_K eta_K^2)``, shape ``()``.
    """
    if basis.gradients.shape[1] != 1:
        raise NotImplementedError(
            "compute_residual_error_estimator currently only supports a "
            "constant-reference-gradient element (e.g. LagrangeTriangleP1). "
            "The interior residual term assumes div(A grad(u_h)) == 0, "
            "exact only when grad(u_h) is itself piecewise constant; a "
            "varying-gradient element (e.g. LagrangeTriangleP2) would need "
            "second-derivative (Hessian) tabulation this library does not "
            "yet provide."
        )

    number_of_cells = basis.space.cells_to_dofs.shape[0]

    interior_residual_integral = _interior_residual_squared_integral(
        solution, basis, problem
    )  # (K,)
    diameters = cell_diameters(basis.space.mesh)  # (K,)
    interior_term = diameters**2 * interior_residual_integral

    facet_term = _facet_jump_term(solution, facet_basis, problem, number_of_cells)

    cell_indicators_squared = interior_term + facet_term
    cell_indicators = jnp.sqrt(cell_indicators_squared)
    global_estimator = jnp.sqrt(jnp.sum(cell_indicators_squared))
    return cell_indicators, global_estimator


def _interior_residual_squared_integral(
    solution: FiniteElementFunction, basis: CellBasis, problem: EllipticProblem
) -> jax.Array:
    """``int_K R_K^2 dx`` for each cell ``K``, shape ``(K,)``."""
    fields = evaluate_finite_element_function(solution, basis)
    source = problem.source(basis.physical_points)  # (K, Q, 1, 1)
    advection = problem.advection(basis.physical_points)  # (K, Q, 1, d)
    reaction = problem.reaction(basis.physical_points)  # (K, Q, 1, 1)

    advection_term = advection @ fields.gradients.mT  # (K, Q, 1, 1)
    reaction_term = reaction * fields.values  # (K, Q, 1, 1)
    residual = source - advection_term - reaction_term  # (K, Q, 1, 1)

    integrated = integrate_cellwise(residual * residual, basis.physical_weights)
    return integrated[:, 0, 0]  # (K,)


def _facet_jump_term(
    solution: FiniteElementFunction,
    facet_basis: FacetBasis,
    problem: EllipticProblem,
    number_of_cells: int,
) -> jax.Array:
    """Distribute ``0.5 * h_e * ||J_e||^2_{L2(e)}`` to each facet's two cells.

    Returns the per-cell accumulated facet term, shape ``(K,)``.
    """
    space = facet_basis.space
    local_plus = solution.dof_values[space.cells_to_dofs[facet_basis.cells_plus]]
    local_minus = solution.dof_values[space.cells_to_dofs[facet_basis.cells_minus]]

    grad_plus = local_plus[:, None, :, :].mT @ facet_basis.gradients_plus  # (F,Q,1,d)
    grad_minus = local_minus[:, None, :, :].mT @ facet_basis.gradients_minus

    diffusion = problem.diffusion(facet_basis.physical_points)  # (F, Q, d, d)
    diffusion_transpose = jnp.swapaxes(diffusion, -1, -2)
    # (A grad(u))^T = grad(u)_row @ A^T, as a row vector; A^T == A whenever
    # the diffusion tensor is symmetric (every manufactured problem here),
    # but this is correct regardless.
    flux_plus = grad_plus @ diffusion_transpose  # (F, Q, 1, d)
    flux_minus = grad_minus @ diffusion_transpose
    flux_jump = flux_plus - flux_minus

    normal_jump = flux_jump @ facet_basis.normals.mT  # (F, Q, 1, 1)

    jump_squared_integral = integrate_cellwise(
        normal_jump * normal_jump, facet_basis.physical_weights
    )[:, 0, 0]  # (F,): int_e J_e^2 ds

    facet_lengths = facet_basis.facet_lengths[:, 0, 0, 0]  # (F,)
    contribution = 0.5 * facet_lengths * jump_squared_integral  # (F,)

    segment_ids = jnp.concatenate((facet_basis.cells_plus, facet_basis.cells_minus))
    contributions = jnp.concatenate((contribution, contribution))
    return jax.ops.segment_sum(
        contributions, segment_ids, num_segments=number_of_cells
    )
