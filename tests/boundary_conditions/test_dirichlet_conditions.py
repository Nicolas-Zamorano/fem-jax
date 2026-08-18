"""Tests for Dirichlet condensation and solution reconstruction.

See Section 15 of the architecture specification; BC-01 through BC-03 of
the FEM testing plan.
"""

import jax.numpy as jnp
import pytest

from jax_fem.assembly import assemble_bilinear_form, assemble_linear_form
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.mesh import create_triangle_mesh_from_arrays
from jax_fem.problem import (
    DirichletCondition,
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    create_full_boundary_dirichlet_condition,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import (
    create_cell_basis,
    create_finite_element_space,
    find_boundary_dofs,
)

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))


def _tagged_unit_square_mesh():
    return create_triangle_mesh_from_arrays(
        _UNIT_SQUARE_VERTICES,
        _UNIT_SQUARE_CELLS,
        boundary_data={
            "bottom": lambda x: jnp.isclose(x[..., 0, 1], 0.0),
            "right": lambda x: jnp.isclose(x[..., 0, 0], 1.0),
            "top": lambda x: jnp.isclose(x[..., 0, 1], 1.0),
            "left": lambda x: jnp.isclose(x[..., 0, 0], 0.0),
        },
    )


def _make_space():
    mesh = _tagged_unit_square_mesh()
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    return mesh, create_finite_element_space(mesh, element)


@pytest.mark.unit
def test_dirichlet_condition_convenience_constructor() -> None:
    mesh = _tagged_unit_square_mesh()
    condition = create_full_boundary_dirichlet_condition(
        mesh, constant_scalar_coefficient(0.0)
    )
    assert set(condition.boundary_tags) == set(mesh.boundary_tag_names.values())


@pytest.mark.unit
@pytest.mark.integration
def test_boundary_dof_classification_is_disjoint_and_covering(
    four_triangle_center_vertex_mesh, p1_element
) -> None:
    """BC-01: boundary and free DOFs are disjoint and together cover all DOFs.

    On the four-triangle center-vertex mesh, the 4 boundary vertices are
    constrained and the 1 interior (center) vertex is free.
    """
    mesh = four_triangle_center_vertex_mesh
    space = create_finite_element_space(mesh, p1_element)
    boundary_dofs = find_boundary_dofs(space, tuple(mesh.boundary_tag_names.values()))
    all_dofs = set(range(space.number_of_dofs))
    free_dofs = all_dofs - set(boundary_dofs.tolist())

    assert set(boundary_dofs.tolist()).isdisjoint(free_dofs)
    assert set(boundary_dofs.tolist()) | free_dofs == all_dofs
    assert len(boundary_dofs) == 4
    assert len(free_dofs) == 1


@pytest.mark.unit
def test_evaluate_dirichlet_dof_values_single_condition() -> None:
    mesh, space = _make_space()
    condition = DirichletCondition(
        boundary_tags=tuple(mesh.boundary_tag_names.values()),
        value=constant_scalar_coefficient(7.0),
    )
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(condition,),
    )
    dofs, values = evaluate_dirichlet_dof_values(problem, space)
    # No interior vertex in this 2-triangle mesh: all 4 dofs are boundary.
    assert jnp.array_equal(dofs, jnp.arange(4))
    assert jnp.allclose(values, 7.0)


@pytest.mark.unit
@pytest.mark.integration
def test_boundary_values_match_nonlinear_function_at_dof_coordinates() -> None:
    """BC-03: every constrained coefficient equals g evaluated at its DOF coordinate.

    Uses a boundary function nonlinear in space (Section BC-03's suggested
    ``g(x,y) = 1 + x + 2y + x^2``): Dirichlet evaluation is exact
    regardless of whether g is representable by the P1 space, since it is
    a pointwise evaluation at DOF coordinates, not an interpolation claim.
    """
    mesh, space = _make_space()

    def g(points: jnp.ndarray) -> jnp.ndarray:
        x, y = points[..., 0], points[..., 1]
        return (1.0 + x + 2.0 * y + x**2)[..., None]

    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(mesh, g),
        ),
    )
    dofs, values = evaluate_dirichlet_dof_values(problem, space)
    coordinates = space.dof_coordinates[dofs]
    expected = g(coordinates)[:, :, 0]
    assert jnp.allclose(values, expected)


@pytest.mark.unit
def test_evaluate_dirichlet_dof_values_first_condition_wins() -> None:
    mesh, space = _make_space()
    bottom_tag = mesh.boundary_tag_names["bottom"]
    left_tag = mesh.boundary_tag_names["left"]
    # Vertex 0 = (0, 0) is shared by "bottom" and "left".
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(
            DirichletCondition(
                boundary_tags=(bottom_tag,), value=constant_scalar_coefficient(1.0)
            ),
            DirichletCondition(
                boundary_tags=(left_tag,), value=constant_scalar_coefficient(2.0)
            ),
        ),
    )
    dofs, values = evaluate_dirichlet_dof_values(problem, space)
    dof_to_value = dict(zip(dofs.tolist(), values[:, 0].tolist(), strict=True))
    assert dof_to_value[0] == pytest.approx(1.0)  # bottom (listed first) wins
    assert dof_to_value[3] == pytest.approx(2.0)  # left-only vertex (0, 1)
    assert dof_to_value[1] == pytest.approx(1.0)  # bottom-only vertex (1, 0)


def _make_laplacian_matrix_and_vector():
    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(2)
    basis = create_cell_basis(space, quadrature)
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(2)),
        advection=constant_vector_coefficient(jnp.zeros(2)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(1.0),
        dirichlet_conditions=(),
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    matrix = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    vector = assemble_linear_form(elliptic_linear_form, basis, coefficients)
    return matrix, vector


@pytest.mark.unit
def test_condense_dirichlet_system_matches_hand_derivation() -> None:
    """BC-02: nonhomogeneous elimination, with a nonzero coupling block K_IB."""
    matrix, vector = _make_laplacian_matrix_and_vector()
    dirichlet_dofs = jnp.array([0, 1])
    dirichlet_values = jnp.array([[2.0], [3.0]])

    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )

    assert jnp.array_equal(condensed.free_dofs, jnp.array([2, 3]))
    assert condensed.number_of_dofs == 4

    # K = [[1,-.5,0,-.5],[-.5,1,-.5,0],[0,-.5,1,-.5],[-.5,0,-.5,1]], b=[1/3,1/6,1/3,1/6]
    # A_ff = K[2:,2:] = [[1,-.5],[-.5,1]]; A_fc = K[2:,:2] = [[0,-.5],[-.5,0]].
    # A_fc @ [2,3] = [-1.5, -1.0]; b_f - A_fc@g_c = [1/3+1.5, 1/6+1.0].
    expected_matrix = jnp.array([[1.0, -0.5], [-0.5, 1.0]])
    expected_vector = jnp.array([[1.0 / 3.0 + 1.5], [1.0 / 6.0 + 1.0]])
    assert jnp.allclose(condensed.matrix.todense(), expected_matrix)
    assert jnp.allclose(condensed.vector, expected_vector)


@pytest.mark.unit
def test_expand_condensed_solution_round_trip() -> None:
    matrix, vector = _make_laplacian_matrix_and_vector()
    dirichlet_dofs = jnp.array([0, 1])
    dirichlet_values = jnp.array([[2.0], [3.0]])
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )

    free_dof_values = jnp.array([[10.0], [20.0]])
    full = expand_condensed_solution(free_dof_values, condensed)

    assert full.shape == (4, 1)
    assert jnp.allclose(full, jnp.array([[2.0], [3.0], [10.0], [20.0]]))

@pytest.mark.unit
@pytest.mark.end_to_end
def test_solved_solution_recovers_prescribed_dirichlet_values() -> None:
    """BC-03, algebraic variant: constrained DOFs equal g after a full solve."""
    matrix, vector = _make_laplacian_matrix_and_vector()
    dirichlet_dofs = jnp.array([0, 1, 3])
    dirichlet_values = jnp.array([[0.0], [0.0], [0.0]])
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )

    free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    full = expand_condensed_solution(free_dof_values, condensed)

    assert jnp.allclose(full[dirichlet_dofs], dirichlet_values)
