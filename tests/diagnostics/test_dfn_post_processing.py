"""Tests for trace fluxes and the per-fracture and per-trace plots.

On the X-crossing of ``z = 0`` (fracture A) and ``y = 0`` (fracture B), the
head ``u = a |y| + b |z|`` is linear on each half-fracture, continuous on the
trace (the x-axis), and has outflows ``2 k_A a`` from A and ``2 k_B b`` from
B. With ``k_A a + k_B b = 0`` the trace balances the fluxes, so ``u`` solves
the network problem without sources, and P1 and P2 reproduce it exactly
because the mesh is aligned with the kinks.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fem.assembly import assemble_matrix, assemble_vector, integrate_cellwise
from jax_fem.constraints import (
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.diagnostics import evaluate_trace_fluxes
from jax_fem.element import LagrangeTriangleP1, LagrangeTriangleP2
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.function import FiniteElementFunction, interpolate_function
from jax_fem.mesh import (
    create_dfn_mesh_from_arrays,
    create_dfn_two_fractures_mesh,
    create_triangle_mesh_from_arrays,
)
from jax_fem.problem import (
    EllipticProblem,
    constant_scalar_coefficient,
    constant_tensor_coefficient,
    constant_vector_coefficient,
    create_dfn_two_fractures_problem,
    create_full_boundary_dirichlet_condition,
    evaluate_elliptic_coefficients,
    evaluate_fracture_transmissivity,
)
from jax_fem.reference_cell import ReferenceInterval, ReferenceTriangle
from jax_fem.space import (
    create_cell_basis,
    create_embedded_cell_basis,
    create_finite_element_space,
)

_ELEMENTS = {"P1": LagrangeTriangleP1, "P2": LagrangeTriangleP2}
_TRANSMISSIVITIES = jnp.array([1.0, 10.0])
_KINK_A = 1.0
_KINK_B = -_TRANSMISSIVITIES[0] * _KINK_A / _TRANSMISSIVITIES[1]
_FACET_POINTS = ReferenceInterval().create_quadrature(3).points


def _kinked_head(points):
    y, z = points[..., 1], points[..., 2]
    return jnp.expand_dims(_KINK_A * jnp.abs(y) + _KINK_B * jnp.abs(z), axis=-1)


def _solve(mesh, problem, element_name: str, transmissivities):
    element = _ELEMENTS[element_name](reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    basis = create_embedded_cell_basis(
        space, element.reference_cell.create_quadrature(4)
    )
    transmissivity = evaluate_fracture_transmissivity(mesh, transmissivities)
    coefficients = evaluate_elliptic_coefficients(
        problem, basis.physical_points
    )._replace(diffusion=transmissivity)
    matrix = assemble_matrix(
        integrate_cellwise(
            elliptic_bilinear_form(basis, basis, coefficients), basis.physical_weights
        ),
        space,
        space,
    )
    vector = assemble_vector(
        integrate_cellwise(
            elliptic_linear_form(basis, coefficients), basis.physical_weights
        ),
        space,
    )
    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    condensed = condense_dirichlet_system(
        matrix, vector, dirichlet_dofs, dirichlet_values
    )
    free_dof_values = jnp.linalg.solve(condensed.matrix.todense(), condensed.vector)
    solution = FiniteElementFunction(
        space=space, dof_values=expand_condensed_solution(free_dof_values, condensed)
    )
    return solution, basis, transmissivity


def _kinked_solution(mesh, element_name: str):
    problem = EllipticProblem(
        diffusion=constant_tensor_coefficient(jnp.eye(3)),
        advection=constant_vector_coefficient(jnp.zeros(3)),
        reaction=constant_scalar_coefficient(0.0),
        source=constant_scalar_coefficient(0.0),
        dirichlet_conditions=(
            create_full_boundary_dirichlet_condition(mesh, _kinked_head),
        ),
    )
    return _solve(mesh, problem, element_name, _TRANSMISSIVITIES)


@pytest.mark.integration
@pytest.mark.parametrize("element_name", list(_ELEMENTS))
def test_trace_fluxes_of_kinked_head_are_exact(
    element_name: str, x_crossing_dfn_mesh
) -> None:
    mesh = x_crossing_dfn_mesh
    solution, basis, transmissivity = _kinked_solution(mesh, element_name)
    fluxes = evaluate_trace_fluxes(solution, basis, transmissivity, _FACET_POINTS)
    expected = 2.0 * _TRANSMISSIVITIES * jnp.array([_KINK_A, _KINK_B])

    assert np.allclose(
        solution.dof_values,
        interpolate_function(_kinked_head, solution.space).dof_values,
        atol=1e-12,
    )
    assert fluxes.outflows.shape == (4, 2, 1, 2)
    assert np.allclose(fluxes.outflows[..., 0, :], expected, atol=1e-11)
    assert np.allclose(fluxes.physical_points[..., 1:], 0.0)
    assert np.allclose(
        fluxes.arc_lengths[..., 0, 0], fluxes.physical_points[..., 0, 0] + 1.0
    )


@pytest.mark.integration
def test_trace_fluxes_approximate_two_fractures_benchmark() -> None:
    mesh = create_dfn_two_fractures_mesh(4)
    solution, basis, transmissivity = _solve(
        mesh, create_dfn_two_fractures_problem(mesh), "P2", jnp.ones(2)
    )
    fluxes = evaluate_trace_fluxes(solution, basis, transmissivity, _FACET_POINTS)
    y = fluxes.physical_points[..., 0, 1]
    exact = jnp.stack((2.0 * y * (1.0 - y), -2.0 * y * (1.0 - y)), axis=-1)

    assert np.max(np.abs(fluxes.outflows[..., 0, :] - exact)) < 2.5e-2
    assert np.allclose(fluxes.arc_lengths[..., 0, 0], y)


@pytest.mark.unit
def test_trace_fluxes_on_network_without_traces() -> None:
    vertices = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    mesh = create_dfn_mesh_from_arrays(vertices, [(0, 1, 2)], [1])
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    basis = create_embedded_cell_basis(
        space, element.reference_cell.create_quadrature(1)
    )
    solution = FiniteElementFunction(space=space, dof_values=jnp.ones((3, 1)))
    fluxes = evaluate_trace_fluxes(
        solution,
        basis,
        evaluate_fracture_transmissivity(mesh, jnp.ones(1)),
        _FACET_POINTS,
    )
    assert fluxes.outflows.shape == (0, 2, 1, 2)


@pytest.mark.unit
def test_trace_fluxes_reject_invalid_input(x_crossing_dfn_mesh) -> None:
    mesh = x_crossing_dfn_mesh
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    basis = create_embedded_cell_basis(
        space, element.reference_cell.create_quadrature(1)
    )
    solution = FiniteElementFunction(
        space=space, dof_values=jnp.zeros((space.number_of_dofs, 1))
    )
    transmissivity = evaluate_fracture_transmissivity(mesh, _TRANSMISSIVITIES)

    with pytest.raises(ValueError, match="transmissivity"):
        evaluate_trace_fluxes(solution, basis, jnp.eye(3), _FACET_POINTS)
    with pytest.raises(ValueError, match="facet_points"):
        evaluate_trace_fluxes(solution, basis, transmissivity, jnp.zeros((2,)))
    other_space = create_finite_element_space(mesh, element)
    with pytest.raises(ValueError, match="same FiniteElementSpace"):
        evaluate_trace_fluxes(
            FiniteElementFunction(space=other_space, dof_values=solution.dof_values),
            basis,
            transmissivity,
            _FACET_POINTS,
        )

    planar_mesh = create_triangle_mesh_from_arrays(
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], [(0, 1, 2)]
    )
    planar_space = create_finite_element_space(planar_mesh, element)
    planar_basis = create_cell_basis(
        planar_space, element.reference_cell.create_quadrature(1)
    )
    with pytest.raises(TypeError, match="DfnMesh"):
        evaluate_trace_fluxes(
            FiniteElementFunction(space=planar_space, dof_values=jnp.zeros((3, 1))),
            planar_basis,
            transmissivity,
            _FACET_POINTS,
        )


@pytest.mark.integration
@pytest.mark.parametrize("element_name", list(_ELEMENTS))
def test_plot_fracture_solutions(element_name: str, x_crossing_dfn_mesh) -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    from jax_fem.diagnostics import plot_fracture_solutions

    solution, _, _ = _kinked_solution(x_crossing_dfn_mesh, element_name)
    figures = plot_fracture_solutions(solution)

    assert list(figures) == ["fracture_1", "fracture_2"]
    for figure in figures.values():
        axis = figure.axes[0]
        assert len(figure.axes) == 2
        assert len(axis.lines) == 1
        assert np.allclose(axis.collections[0].get_clim(), (-0.1, 1.0))
    plt.close("all")


@pytest.mark.integration
def test_plot_trace_fluxes(x_crossing_dfn_mesh) -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    from jax_fem.diagnostics import plot_trace_fluxes

    solution, basis, transmissivity = _kinked_solution(x_crossing_dfn_mesh, "P1")
    figures = plot_trace_fluxes(solution, basis, transmissivity)

    assert list(figures) == ["trace_0"]
    axis = figures["trace_0"].axes[0]
    labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert labels == [
        "fracture 1 (total 4.000e+00)",
        "fracture 2 (total -4.000e+00)",
        "sum",
    ]
    first_fracture = axis.lines[0]
    assert np.allclose(first_fracture.get_xdata(), np.repeat(np.arange(5) / 2, 2)[1:-1])
    assert np.allclose(first_fracture.get_ydata(), 2.0)
    plt.close("all")


@pytest.mark.unit
def test_plot_fracture_solutions_rejects_planar_meshes() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    from jax_fem.diagnostics import plot_fracture_solutions

    mesh = create_triangle_mesh_from_arrays(
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], [(0, 1, 2)]
    )
    space = create_finite_element_space(
        mesh, LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    )
    with pytest.raises(TypeError, match="DfnMesh"):
        plot_fracture_solutions(
            FiniteElementFunction(space=space, dof_values=jnp.zeros((3, 1)))
        )
    plt.close("all")
