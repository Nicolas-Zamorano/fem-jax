"""Energy-norm identity test.

See Section 13 (energy-norm assumptions) of the FEM testing plan, ELL-05.
"""

import jax.numpy as jnp
import pytest

from jax_fem.assembly import assemble_bilinear_form
from jax_fem.diagnostics import compute_energy_error
from jax_fem.forms import elliptic_bilinear_form
from jax_fem.function import FiniteElementFunction
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    evaluate_elliptic_coefficients,
)
from jax_fem.space import (
    create_cell_basis,
    create_finite_element_space,
    find_boundary_dofs,
)


@pytest.mark.unit
@pytest.mark.integration
def test_energy_identity_matches_bilinear_form_and_hard_coded_value(
    four_triangle_center_vertex_mesh, p1_element
) -> None:
    """ELL-05: ``a(v_h, v_h) == ||v_h||_E^2 == 25/6`` for the center hat function.

    ``A = I``, ``beta = (x, y)`` so ``div(beta) = 2`` and the symmetrized
    reaction is ``c - div(beta)/2 = 2 - 1 = 1``. For the center hat
    function, ``int |grad v_h|^2 = 4`` and ``int v_h^2 = 1/6``, giving
    ``||v_h||_E^2 = 4 + 1/6 = 25/6`` (the testing plan's hard-coded value).

    Computed two independent ways: ``U^T A_h U`` through the assembled
    bilinear-form matrix, and the energy functional through
    ``compute_energy_error`` (with a zero exact solution, so the "error"
    equals ``v_h`` itself) -- an entirely different code path (Section
    13.3's pointwise energy density + quadrature integration, rather than
    global sparse assembly).
    """
    mesh = four_triangle_center_vertex_mesh
    space = create_finite_element_space(mesh, p1_element)
    quadrature = space.element.reference_cell.create_quadrature(4)
    basis = create_cell_basis(space, quadrature)

    # The center hat function: 1 at the interior (center) vertex, 0 at
    # every boundary vertex.
    boundary_dofs = find_boundary_dofs(space, tuple(mesh.boundary_tag_names.values()))
    dof_values = jnp.ones((space.number_of_dofs, 1)).at[boundary_dofs].set(0.0)
    v_h = FiniteElementFunction(space=space, dof_values=dof_values)

    def advection(points: jnp.ndarray) -> jnp.ndarray:
        return points  # beta(x, y) = (x, y): points already has shape (*b, 1, 2).

    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=advection,
        reaction=constant_scalar_coefficient(2.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(),
        exact_solution=constant_scalar_coefficient(0.0),
        exact_gradient=constant_vector_coefficient(jnp.zeros(2)),
        advection_divergence=constant_scalar_coefficient(2.0),
    )

    energy_norm_squared_via_functional = compute_energy_error(v_h, basis, problem) ** 2

    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    energy_norm_squared_via_bilinear_form = float(
        (v_h.dof_values.T @ (matrix @ v_h.dof_values))[0, 0]
    )

    assert energy_norm_squared_via_functional == pytest.approx(25.0 / 6.0, rel=1e-12)
    assert energy_norm_squared_via_bilinear_form == pytest.approx(
        25.0 / 6.0, rel=1e-12
    )
    assert energy_norm_squared_via_functional == pytest.approx(
        energy_norm_squared_via_bilinear_form, rel=1e-12
    )
    assert energy_norm_squared_via_functional >= 0.0
