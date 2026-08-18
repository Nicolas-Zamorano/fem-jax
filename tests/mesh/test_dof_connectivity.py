"""Tests for FiniteElementSpace and P1 global DOF numbering.

See Section 10.1 of the architecture specification; DOF-01, DOF-02, and
DOF-04 of the FEM testing plan. DOF-03/DOF-05 (edge-orientation
permutation and interior-DOF exclusivity for degree > 1) are future
extension tests, absent until a higher-order element exists.
"""

import dataclasses

import jax.numpy as jnp
import pytest

from jax_fem.element import LagrangeTriangleP1
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import create_triangle_mesh_from_arrays
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_finite_element_space, find_boundary_dofs

_UNIT_SQUARE_VERTICES = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_UNIT_SQUARE_CELLS = ((0, 1, 2), (0, 2, 3))


@pytest.mark.unit
def test_p1_dof_map_matches_vertices(two_triangle_unit_square_mesh, p1_element) -> None:
    space = create_finite_element_space(two_triangle_unit_square_mesh, p1_element)
    assert jnp.array_equal(
        space.cells_to_dofs, two_triangle_unit_square_mesh.cells_to_vertices
    )
    assert jnp.array_equal(
        space.dof_coordinates, two_triangle_unit_square_mesh.vertex_coordinates
    )
    assert space.number_of_dofs == 4


@pytest.mark.unit
def test_shapes(two_triangle_unit_square_mesh, p1_element) -> None:
    space = create_finite_element_space(two_triangle_unit_square_mesh, p1_element)
    assert space.cells_to_dofs.shape == (2, 3)
    assert space.dof_coordinates.shape == (4, 1, 2)


@pytest.mark.unit
def test_global_dof_count_matches_vertex_count(
    four_triangle_center_vertex_mesh, p1_element
) -> None:
    """DOF-01: for P1, N_dofs == N_vertices (the general formula's p=1 case)."""
    space = create_finite_element_space(four_triangle_center_vertex_mesh, p1_element)
    assert (
        space.number_of_dofs
        == four_triangle_center_vertex_mesh.vertex_coordinates.shape[0]
    )
    assert space.number_of_dofs == 5  # 4 boundary vertices + 1 interior (center)


@pytest.mark.unit
def test_unsupported_element_raises(two_triangle_unit_square_mesh) -> None:
    @dataclasses.dataclass(frozen=True, slots=True, eq=False)
    class _FakeQuadraticElement:
        reference_cell: object
        polynomial_degree: int = 2
        number_of_local_dofs: int = 6
        value_shape: tuple[int, ...] = (1,)

        def tabulate_basis_values(self, reference_points):  # noqa: D102
            raise NotImplementedError

        def tabulate_basis_gradients(self, reference_points):  # noqa: D102
            raise NotImplementedError

    fake_element = _FakeQuadraticElement(reference_cell=ReferenceTriangle())
    with pytest.raises(NotImplementedError):
        create_finite_element_space(two_triangle_unit_square_mesh, fake_element)


@pytest.mark.unit
def test_find_boundary_dofs_default_tag_covers_all_vertices(
    two_triangle_unit_square_mesh, p1_element
) -> None:
    space = create_finite_element_space(two_triangle_unit_square_mesh, p1_element)
    boundary_dofs = find_boundary_dofs(space, (0,))
    # No interior vertex exists in this 2-triangle mesh.
    assert jnp.array_equal(boundary_dofs, jnp.arange(4))


@pytest.mark.unit
def test_find_boundary_dofs_selects_only_requested_tags() -> None:
    mesh = create_triangle_mesh_from_arrays(
        _UNIT_SQUARE_VERTICES,
        _UNIT_SQUARE_CELLS,
        boundary_data={
            "bottom": lambda x: jnp.isclose(x[..., 0, 1], 0.0),
            "right": lambda x: jnp.isclose(x[..., 0, 0], 1.0),
            "top": lambda x: jnp.isclose(x[..., 0, 1], 1.0),
            "left": lambda x: jnp.isclose(x[..., 0, 0], 0.0),
        },
    )
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)

    bottom_tag = mesh.boundary_tag_names["bottom"]
    bottom_dofs = find_boundary_dofs(space, (bottom_tag,))
    # The bottom edge (0, 0)-(1, 0) connects vertices 0 and 1.
    assert jnp.array_equal(bottom_dofs, jnp.array([0, 1]))

    left_tag = mesh.boundary_tag_names["left"]
    right_tag = mesh.boundary_tag_names["right"]
    left_or_right_dofs = find_boundary_dofs(space, (left_tag, right_tag))
    # left = vertices {0, 3}, right = vertices {1, 2}.
    assert jnp.array_equal(left_or_right_dofs, jnp.array([0, 1, 2, 3]))


def _shared_facet_and_cells(mesh):
    interior_mask = jnp.all(mesh.facets_to_cells != -1, axis=1)
    (shared_facet,) = jnp.nonzero(interior_mask)[0]
    cell_0, cell_1 = mesh.facets_to_cells[shared_facet].tolist()
    return int(shared_facet), cell_0, cell_1


@pytest.mark.unit
@pytest.mark.integration
def test_shared_edge_dofs_match_across_reversed_orientation(
    reversed_orientation_two_triangle_mesh, p1_element
) -> None:
    """DOF-02: shared-edge global DOFs agree, reversed to match opposite orientation.

    ``K_0 = (v0, v1, v2)``, ``K_1 = (v3, v2, v1)``: the shared edge is
    traversed in opposite directions locally (fixture docstring), so the
    two cells' local-edge-to-global sequences must be exact reversals of
    each other, not merely the same set.
    """
    mesh = reversed_orientation_two_triangle_mesh
    space = create_finite_element_space(mesh, p1_element)

    shared_facet, cell_0, cell_1 = _shared_facet_and_cells(mesh)
    local_facet_0 = int(
        jnp.nonzero(mesh.cells_to_facets[cell_0] == shared_facet)[0][0]
    )
    local_facet_1 = int(
        jnp.nonzero(mesh.cells_to_facets[cell_1] == shared_facet)[0][0]
    )

    reference_facets_to_vertices = space.element.reference_cell.facets_to_vertices
    local_vertices_0 = reference_facets_to_vertices[local_facet_0]
    local_vertices_1 = reference_facets_to_vertices[local_facet_1]

    edge_dofs_0 = space.cells_to_dofs[cell_0][local_vertices_0].tolist()
    edge_dofs_1 = space.cells_to_dofs[cell_1][local_vertices_1].tolist()

    assert set(edge_dofs_0) == set(edge_dofs_1)
    assert edge_dofs_0 == list(reversed(edge_dofs_1))


@pytest.mark.unit
@pytest.mark.integration
def test_finite_element_function_is_continuous_across_shared_edge(
    reversed_orientation_two_triangle_mesh, p1_element
) -> None:
    """DOF-04: a discrete function's trace is single-valued across an interior edge.

    Tests conformity end to end: local-to-global numbering, edge
    orientation, local coefficient extraction, the inverse geometry map,
    and basis evaluation. Uses nonsymmetric global DOF values and several
    non-nodal points along the (open) shared edge, not only its two
    endpoints, so an incorrect reversal or wrong local-to-global map cannot
    pass accidentally. Only continuity of values is checked: a conforming
    H1 Lagrange function's normal derivative is generally discontinuous
    across cell interfaces.
    """
    mesh = reversed_orientation_two_triangle_mesh
    space = create_finite_element_space(mesh, p1_element)

    global_dof_indices = jnp.arange(space.number_of_dofs, dtype=jnp.float64)
    global_values = (
        0.13 * global_dof_indices**2
        - 0.71 * global_dof_indices
        + jnp.sin(global_dof_indices + 0.2)
    )[:, None]  # (N_dofs, 1); deliberately nonsymmetric in the DOF index.
    function = FiniteElementFunction(space=space, dof_values=global_values)

    shared_facet, cell_0, cell_1 = _shared_facet_and_cells(mesh)
    edge_vertices = mesh.facets_to_vertices[shared_facet]
    # Drop mesh.vertex_coordinates' row-vector middle axis: plain (2,) points
    # are simplest for this test's own independent geometry computation.
    vertex_a = mesh.vertex_coordinates[edge_vertices[0], 0]
    vertex_b = mesh.vertex_coordinates[edge_vertices[1], 0]

    parameters = jnp.array([0.07, 0.23, 0.41, 0.68, 0.91])[:, None]
    physical_points = (1.0 - parameters) * vertex_a + parameters * vertex_b  # (5, 2)

    def reference_points_for_cell(cell_index: int) -> jnp.ndarray:
        origin = mesh.vertex_coordinates[mesh.cells_to_vertices[cell_index, 0], 0]
        edge_1 = (
            mesh.vertex_coordinates[mesh.cells_to_vertices[cell_index, 1], 0] - origin
        )
        edge_2 = (
            mesh.vertex_coordinates[mesh.cells_to_vertices[cell_index, 2], 0] - origin
        )
        jacobian = jnp.stack((edge_1, edge_2), axis=-1)  # (2, 2)
        displacement = physical_points - origin  # (5, 2)
        reference_points = displacement @ jnp.linalg.inv(jacobian).mT  # (5, 2)
        return reference_points.reshape(1, -1, 1, 2)  # (1, 5, 1, 2)

    reference_points_0 = reference_points_for_cell(cell_0)
    reference_points_1 = reference_points_for_cell(cell_1)

    basis_values_0 = space.element.tabulate_basis_values(reference_points_0)
    basis_values_1 = space.element.tabulate_basis_values(reference_points_1)

    local_values_0 = function.dof_values[space.cells_to_dofs[cell_0]]  # (3, 1)
    local_values_1 = function.dof_values[space.cells_to_dofs[cell_1]]

    trace_0 = basis_values_0.mT @ local_values_0  # (1, 5, 1, 1)
    trace_1 = basis_values_1.mT @ local_values_1

    assert jnp.allclose(trace_0, trace_1, atol=1e-12)
