"""Uniform-refinement convergence study for the residual error estimator.

See Section 4.4 of the implementation plan (``.context/implementation_plan.md``):
on the L-shaped re-entrant-corner singular problem, the classical residual
estimator should converge at the same sub-optimal rate ``O(h^(2/3))`` as the
true H1 error (both limited by the domain's genuine 270-degree corner
singularity, not by the discretization), with a bounded effectivity index
``eta / |u - u_h|_H1``.

Uses ``create_gmsh_mesh_hierarchy`` for an exact 1-to-4 refinement sequence
(Section 2.2), rather than independently-built meshes: the estimator's
comparison across levels is only meaningful if ``h`` genuinely halves each
step, which post-hoc-constructed meshes at a target size only approximate.
"""

import math

import jax.numpy as jnp
import pytest

from jax_fem.assembly import assemble_bilinear_form, assemble_linear_form
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.diagnostics import compute_h1_seminorm_error, compute_residual_error_estimator
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import create_gmsh_mesh_hierarchy, l_shaped_gmsh_geometry
from jax_fem.problem import create_poisson_l_shaped_singular_problem, evaluate_elliptic_coefficients
from jax_fem.reference_cell import ReferenceInterval, ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space
from jax_fem.space.facet_basis import create_facet_basis

pytestmark = [pytest.mark.convergence, pytest.mark.end_to_end, pytest.mark.slow]

_TARGET_RATE = 2.0 / 3.0
# Generous tolerance: uniform refinement on a re-entrant-corner singularity
# converges to O(h^(2/3)) slowly (the measured rate is still visibly
# pre-asymptotic -- ~0.56 -> ~0.64 -- across these 4 levels, monotonically
# approaching 2/3 from below), so this checks the *trend*, not a tight
# asymptotic match; a much finer mesh (or AMR, Section 4.2's stretch item)
# would be needed to see the rate converge tighter than this.
_RATE_TOLERANCE = 0.3


def _estimate_at_each_level(levels: int, base_h: float):
    meshes = create_gmsh_mesh_hierarchy(l_shaped_gmsh_geometry, base_h, levels=levels)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())

    results = []
    for mesh in meshes:
        space = create_finite_element_space(mesh, element)
        quadrature = space.element.reference_cell.create_quadrature(4)
        basis = create_cell_basis(space, quadrature)
        problem = create_poisson_l_shaped_singular_problem(mesh)

        coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
        matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
        vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)
        dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
        condensed = condense_dirichlet_system(
            matrix, vector, dirichlet_dofs, dirichlet_values
        )
        free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
        full_dof_values = expand_condensed_solution(free_dof_values, condensed)
        solution = FiniteElementFunction(space=space, dof_values=full_dof_values)

        facet_quadrature = ReferenceInterval().create_quadrature(4)
        facet_basis = create_facet_basis(basis, facet_quadrature)
        _, eta = compute_residual_error_estimator(solution, basis, facet_basis, problem)
        h1_error = compute_h1_seminorm_error(solution, basis, problem)

        cells = mesh.cells_to_vertices.shape[0]
        results.append((cells, float(eta), float(h1_error)))
    return results


def test_residual_estimator_convergence_on_l_shaped_singular_problem() -> None:
    results = _estimate_at_each_level(levels=3, base_h=0.35)

    etas = [r[1] for r in results]
    h1_errors = [r[2] for r in results]

    # Monotonic decrease under refinement, for both the estimator and the
    # true error it estimates.
    assert all(a > b for a, b in zip(etas, etas[1:], strict=False))
    assert all(a > b for a, b in zip(h1_errors, h1_errors[1:], strict=False))

    # Bounded effectivity index at every level: eta stays the same order of
    # magnitude as the true error it estimates, neither collapsing to 0 nor
    # blowing up, across refinement.
    effectivity_indices = [eta / h1 for eta, h1 in zip(etas, h1_errors, strict=False)]
    assert all(1.0 < index < 6.0 for index in effectivity_indices)

    # Rate check: cells quadruple each level (exact 1-to-4 refinement), so
    # h halves; use sqrt(cells_fine / cells_coarse) as the h_coarse/h_fine
    # ratio.
    for (cells_coarse, eta_coarse, h1_coarse), (cells_fine, eta_fine, h1_fine) in zip(
        results, results[1:], strict=False
    ):
        h_ratio = math.sqrt(cells_fine / cells_coarse)
        eta_rate = math.log(eta_coarse / eta_fine) / math.log(h_ratio)
        h1_rate = math.log(h1_coarse / h1_fine) / math.log(h_ratio)
        assert eta_rate == pytest.approx(_TARGET_RATE, abs=_RATE_TOLERANCE)
        assert h1_rate == pytest.approx(_TARGET_RATE, abs=_RATE_TOLERANCE)

    # The finest-level (most-refined, closest-to-asymptotic) rate transition
    # should be the closest to the target -- checked separately with a
    # tighter bound to pin down the convergent trend, not just "somewhere in
    # a wide band" at every level.
    finest_eta_rate = math.log(etas[-2] / etas[-1]) / math.log(
        math.sqrt(results[-1][0] / results[-2][0])
    )
    assert finest_eta_rate == pytest.approx(_TARGET_RATE, abs=0.15)
