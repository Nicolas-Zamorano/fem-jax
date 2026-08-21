"""Tests for the plotting module, in particular P1/P2-generic 3D plotting.

See ``_sub_triangulation_for_plotting`` in
``src/jax_fem/diagnostics/plotting.py``: matplotlib's ``tripcolor``/
``plot_trisurf`` only understand flat 3-node triangles, so a P2 solution (6
DOFs/cell: 3 vertices + 3 edge midpoints) needs an exact linear
sub-triangulation through every DOF, not just the 3 corners, to plot without
either a shape mismatch or silently discarding the edge-midpoint DOFs.
"""

import jax.numpy as jnp
import pytest

pytest.importorskip("matplotlib")

from jax_fem.assembly import assemble_bilinear_form, assemble_linear_form  # noqa: E402
from jax_fem.constraints import (  # noqa: E402
    condense_dirichlet_system,
    evaluate_dirichlet_dof_values,
    expand_condensed_solution,
)
from jax_fem.diagnostics.plotting import (  # noqa: E402
    _sub_triangulation_for_plotting,
    plot_fem_error,
    plot_fem_error_3d,
)
from jax_fem.element import LagrangeTriangleP1, LagrangeTriangleP2  # noqa: E402
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form  # noqa: E402
from jax_fem.function import FiniteElementFunction  # noqa: E402
from jax_fem.mesh import create_triangle_mesh_from_arrays  # noqa: E402
from jax_fem.problem import (  # noqa: E402
    create_simple_elliptic_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle  # noqa: E402
from jax_fem.space import create_cell_basis, create_finite_element_space  # noqa: E402

_REFERENCE_TRIANGLE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
_SINGLE_CELL = ((0, 1, 2),)

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))


@pytest.mark.unit
def test_sub_triangulation_p1_is_the_identity() -> None:
    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)

    x, y, triangles = _sub_triangulation_for_plotting(space)
    assert jnp.array_equal(triangles, space.cells_to_dofs)
    assert jnp.array_equal(x, space.dof_coordinates[:, 0, 0])
    assert jnp.array_equal(y, space.dof_coordinates[:, 0, 1])


@pytest.mark.unit
def test_sub_triangulation_p2_matches_documented_layout() -> None:
    """The 4 sub-triangles per cell must exactly match: sub0=(v0,m2,m1),
    sub1=(v1,m0,m2), sub2=(v2,m1,m0), sub3=(m0,m1,m2), where local DOF
    3/4/5 (m0/m1/m2) are the midpoints of facets 0/1/2 (opposite v0/v1/v2).
    """
    mesh = create_triangle_mesh_from_arrays(
        _REFERENCE_TRIANGLE_VERTICES, _SINGLE_CELL
    )
    element = LagrangeTriangleP2(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)

    x, y, triangles = _sub_triangulation_for_plotting(space)
    assert triangles.shape == (4, 3)
    assert x.shape == (space.number_of_dofs,)

    v0, v1, v2, m0, m1, m2 = space.cells_to_dofs[0]
    expected = jnp.asarray(
        [
            [v0, m2, m1],
            [v1, m0, m2],
            [v2, m1, m0],
            [m0, m1, m2],
        ]
    )
    assert jnp.array_equal(triangles, expected)

    # Every DOF coordinate is covered: the 4 sub-triangles' vertex set is
    # exactly all 6 local DOFs (as global indices), no DOF dropped.
    assert set(triangles.reshape(-1).tolist()) == set(space.cells_to_dofs[0].tolist())


@pytest.mark.unit
def test_sub_triangulation_rejects_unsupported_element() -> None:
    import types

    # _sub_triangulation_for_plotting only reads element.number_of_local_dofs,
    # space.dof_coordinates, and space.cells_to_dofs: a minimal stand-in
    # covers that without needing a full FiniteElementSpace (whose
    # number_of_local_dofs is an init=False field on the real elements, not
    # something dataclasses.replace can override).
    fake_space = types.SimpleNamespace(
        element=types.SimpleNamespace(number_of_local_dofs=4),
        dof_coordinates=jnp.zeros((4, 1, 2)),
        cells_to_dofs=jnp.zeros((1, 4), dtype=jnp.int32),
    )
    with pytest.raises(NotImplementedError, match="3-DOF"):
        _sub_triangulation_for_plotting(fake_space)


def _solve(mesh, element):
    space = create_finite_element_space(mesh, element)
    quadrature = space.element.reference_cell.create_quadrature(5)
    basis = create_cell_basis(space, quadrature)
    problem = create_simple_elliptic_problem(mesh)
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
    return solution, basis, problem


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.parametrize(
    "element_factory", [LagrangeTriangleP1, LagrangeTriangleP2], ids=["p1", "p2"]
)
def test_plot_fem_error_runs_end_to_end(element_factory) -> None:
    import matplotlib.pyplot as plt

    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    element = element_factory(reference_cell=ReferenceTriangle())
    solution, basis, problem = _solve(mesh, element)

    figure = plot_fem_error(solution, basis, problem, mesh)
    try:
        assert figure is not None
    finally:
        plt.close(figure)


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.parametrize(
    "element_factory", [LagrangeTriangleP1, LagrangeTriangleP2], ids=["p1", "p2"]
)
def test_plot_fem_error_3d_runs_end_to_end(element_factory) -> None:
    """Regression test: previously raised for P2 (a shape mismatch between
    ``exact`` evaluated only at mesh vertices and ``solution.dof_values``
    covering all N_dofs, since N_dofs > N_vertices for P2).
    """
    import matplotlib.pyplot as plt

    mesh = create_triangle_mesh_from_arrays(_UNIT_SQUARE_VERTICES, _UNIT_SQUARE_CELLS)
    element = element_factory(reference_cell=ReferenceTriangle())
    solution, basis, problem = _solve(mesh, element)

    figure = plot_fem_error_3d(solution, basis, problem, mesh)
    try:
        assert figure is not None
    finally:
        plt.close(figure)
