"""Closed-loop AMR study on the L-shaped re-entrant-corner singular problem.

A full solve -> estimate -> adapt loop: the residual estimator
(``diagnostics.compute_residual_error_estimator``) drives longest-edge
bisection (``mesh.adapt_mesh_from_error_estimator``).

Unlike ``test_residual_estimator_convergence.py``, which checks a tight
uniform-refinement rate, this only checks the robust signal available from a
few AMR iterations on small, pre-asymptotic meshes: the mesh grows every
iteration and the final estimator is smaller than the first. The estimator is
not required to decrease monotonically, because Dorfler marking on a very
coarse mesh can increase it transiently. Confirming the optimal
``O(N_dofs^(-1/2))`` rate needs more iterations than a fast test allows.
"""

import jax.numpy as jnp
import pytest

from jax_fem.assembly import assemble_matrix, assemble_vector, integrate_cellwise
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.diagnostics import compute_residual_error_estimator
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import adapt_mesh_from_error_estimator, create_gmsh_l_shaped_mesh
from jax_fem.problem import (
    create_poisson_l_shaped_singular_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceInterval, ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space
from jax_fem.space.facet_basis import create_facet_basis

pytestmark = [pytest.mark.convergence, pytest.mark.end_to_end, pytest.mark.slow]

_ELEMENT = LagrangeTriangleP1(reference_cell=ReferenceTriangle())


def _solve_and_estimate(mesh):
    space = create_finite_element_space(mesh, _ELEMENT)
    quadrature = space.element.reference_cell.create_quadrature(4)
    basis = create_cell_basis(space, quadrature)
    problem = create_poisson_l_shaped_singular_problem(mesh)

    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    local_matrices = integrate_cellwise(
        elliptic_bilinear_form(basis, basis, coefficients), basis.physical_weights
    )
    matrix = assemble_matrix(local_matrices, basis.space, basis.space)
    local_vectors = integrate_cellwise(
        elliptic_linear_form(basis, coefficients), basis.physical_weights
    )
    vector = assemble_vector(local_vectors, basis.space)
    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )
    free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    full_dof_values = expand_condensed_solution(free_dof_values, condensed)
    solution = FiniteElementFunction(space=space, dof_values=full_dof_values)

    facet_quadrature = ReferenceInterval().create_quadrature(4)
    facet_basis = create_facet_basis(basis, facet_quadrature)
    cell_indicators, eta = compute_residual_error_estimator(
        solution, basis, facet_basis, problem
    )
    return space.number_of_dofs, float(eta), cell_indicators


def test_amr_loop_grows_the_mesh_and_reduces_the_global_estimator() -> None:
    mesh = create_gmsh_l_shaped_mesh(0.5)
    cell_counts = []
    dof_counts = []
    etas = []

    for _ in range(4):
        cell_counts.append(mesh.cells_to_vertices.shape[0])
        n_dofs, eta, cell_indicators = _solve_and_estimate(mesh)
        dof_counts.append(n_dofs)
        etas.append(eta)
        assert eta > 0.0 and jnp.isfinite(eta)
        mesh = adapt_mesh_from_error_estimator(mesh, cell_indicators, theta=0.4)

    # The mesh (and DOF count) strictly grows every iteration: the loop is
    # actually doing something, not stalling or degenerating.
    assert all(a < b for a, b in zip(cell_counts, cell_counts[1:], strict=False))
    assert all(a < b for a, b in zip(dof_counts, dof_counts[1:], strict=False))

    # Overall trend across the whole run: the estimator ends up smaller than
    # it started, even though individual steps need not be monotonic on
    # such coarse, pre-asymptotic starting meshes.
    assert etas[-1] < etas[0]
