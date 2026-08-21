"""Tests for FiniteElementSpace and generic (P1 and P2) global DOF numbering.

See Section 10.1 of the architecture specification; DOF-01 through DOF-05 of
the FEM testing plan, and Section 3.2/3.3 of the implementation plan
(``.context/implementation_plan.md``) for the generic entity_dofs-driven
numbering exercised throughout.
"""

import dataclasses

import jax.numpy as jnp
import pytest

from jax_fem.element import LagrangeTriangleP1, RaviartThomasTriangleRT0
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
def test_malformed_element_missing_entity_dofs_raises(
    two_triangle_unit_square_mesh,
) -> None:
    """create_finite_element_space is generic over any entity_dofs-driven
    element (Section 3.2): it no longer isinstance-gates on a specific
    concrete class, but still rejects an element that fails to implement
    the FiniteElement protocol at all (here, missing entity_dofs/is_nodal).
    """

    @dataclasses.dataclass(frozen=True, slots=True, eq=False)
    class _FakeIncompleteElement:
        reference_cell: object
        polynomial_degree: int = 2
        number_of_local_dofs: int = 6
        value_shape: tuple[int, ...] = (1,)

        def tabulate_basis_values(self, reference_points):  # noqa: D102
            raise NotImplementedError

        def tabulate_basis_gradients(self, reference_points):  # noqa: D102
            raise NotImplementedError

    fake_element = _FakeIncompleteElement(reference_cell=ReferenceTriangle())
    with pytest.raises(TypeError, match="FiniteElement protocol"):
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

    assert jnp.allclose(trace_0, trace_1, atol=1e-14)


# ---------------------------------------------------------------------------
# P2: same DOF-* properties, generalized
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_p2_global_dof_count(four_triangle_center_vertex_mesh, p2_element) -> None:
    """DOF-01 for P2: N_dofs == N_vertices + N_facets."""
    mesh = four_triangle_center_vertex_mesh
    space = create_finite_element_space(mesh, p2_element)
    number_of_vertices = mesh.vertex_coordinates.shape[0]
    number_of_facets = mesh.facets_to_vertices.shape[0]
    assert space.number_of_dofs == number_of_vertices + number_of_facets
    assert space.entity_dof_offset == {0: 0, 1: number_of_vertices}


@pytest.mark.unit
def test_p2_dof_coordinates_are_vertices_then_facet_midpoints(
    two_triangle_unit_square_mesh, p2_element
) -> None:
    mesh = two_triangle_unit_square_mesh
    space = create_finite_element_space(mesh, p2_element)
    number_of_vertices = mesh.vertex_coordinates.shape[0]

    assert jnp.array_equal(
        space.dof_coordinates[:number_of_vertices], mesh.vertex_coordinates
    )
    facet_midpoints = jnp.mean(mesh.vertex_coordinates[mesh.facets_to_vertices], axis=1)
    assert jnp.allclose(
        space.dof_coordinates[number_of_vertices:], facet_midpoints, atol=1e-14
    )


@pytest.mark.unit
@pytest.mark.integration
def test_p2_shared_edge_dof_is_orientation_invariant(
    reversed_orientation_two_triangle_mesh, p2_element
) -> None:
    """DOF-03 for P2: an edge-midpoint DOF is orientation-invariant by
    construction (Section 3.3's note), so the shared edge's midpoint DOF is
    the *same single global index* in both cells -- stronger than P1's
    DOF-02 "reversed but same set" (P1 has 2 DOFs per edge, its endpoints,
    which do reverse; P2's single midpoint DOF has no direction to
    reverse).
    """
    mesh = reversed_orientation_two_triangle_mesh
    space = create_finite_element_space(mesh, p2_element)

    shared_facet, cell_0, cell_1 = _shared_facet_and_cells(mesh)
    local_facet_0 = int(
        jnp.nonzero(mesh.cells_to_facets[cell_0] == shared_facet)[0][0]
    )
    local_facet_1 = int(
        jnp.nonzero(mesh.cells_to_facets[cell_1] == shared_facet)[0][0]
    )

    # Local facet i's midpoint DOF is local DOF (3 + i) (LagrangeTriangleP2's
    # entity_dofs[1] == ((3,), (4,), (5,))).
    midpoint_dof_0 = int(space.cells_to_dofs[cell_0, 3 + local_facet_0])
    midpoint_dof_1 = int(space.cells_to_dofs[cell_1, 3 + local_facet_1])
    assert midpoint_dof_0 == midpoint_dof_1


@pytest.mark.unit
def test_p2_find_boundary_dofs_excludes_interior_edge_midpoints(
    four_triangle_center_vertex_mesh, p2_element
) -> None:
    """DOF-05 for P2: interior-facet DOFs are excluded from boundary DOFs,
    just as interior-vertex DOFs are.
    """
    mesh = four_triangle_center_vertex_mesh
    space = create_finite_element_space(mesh, p2_element)
    boundary_dofs = set(
        find_boundary_dofs(space, tuple(mesh.boundary_tag_names.values())).tolist()
    )

    interior_facets = jnp.nonzero(jnp.all(mesh.facets_to_cells != -1, axis=1))[0]
    interior_facet_dofs = (interior_facets + space.entity_dof_offset[1]).tolist()

    assert boundary_dofs.isdisjoint(interior_facet_dofs)
    assert set(interior_facet_dofs).issubset(set(range(space.number_of_dofs)) - boundary_dofs)


@pytest.mark.unit
@pytest.mark.integration
def test_p2_finite_element_function_is_continuous_across_shared_edge(
    reversed_orientation_two_triangle_mesh, p2_element
) -> None:
    """DOF-04 for P2: a discrete function's trace is single-valued across an
    interior edge, sampled at several non-nodal points -- a stronger check
    than for P1, since a P2 edge trace is a genuine quadratic curve (3 DOFs
    per edge: 2 endpoints + midpoint), not merely affine.
    """
    mesh = reversed_orientation_two_triangle_mesh
    space = create_finite_element_space(mesh, p2_element)

    global_dof_indices = jnp.arange(space.number_of_dofs, dtype=jnp.float64)
    global_values = (
        0.13 * global_dof_indices**2
        - 0.71 * global_dof_indices
        + jnp.sin(global_dof_indices + 0.2)
    )[:, None]
    function = FiniteElementFunction(space=space, dof_values=global_values)

    shared_facet, cell_0, cell_1 = _shared_facet_and_cells(mesh)
    edge_vertices = mesh.facets_to_vertices[shared_facet]
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

    local_values_0 = function.dof_values[space.cells_to_dofs[cell_0]]  # (6, 1)
    local_values_1 = function.dof_values[space.cells_to_dofs[cell_1]]

    trace_0 = basis_values_0.mT @ local_values_0  # (1, 5, 1, 1)
    trace_1 = basis_values_1.mT @ local_values_1

    assert jnp.allclose(trace_0, trace_1, atol=1e-13)


# ---------------------------------------------------------------------------
# RT0: facet-owned, orientation-dependent DOF numbering (Section 5.2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rt0_global_dof_count_and_offset(four_triangle_center_vertex_mesh) -> None:
    """DOF-01 for RT0: N_dofs == N_facets (edge-owned only, no vertex/cell DOFs)."""
    mesh = four_triangle_center_vertex_mesh
    element = RaviartThomasTriangleRT0(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    assert space.number_of_dofs == mesh.facets_to_vertices.shape[0]
    assert space.entity_dof_offset == {1: 0}


@pytest.mark.unit
def test_rt0_dof_coordinates_are_facet_midpoints(two_triangle_unit_square_mesh) -> None:
    mesh = two_triangle_unit_square_mesh
    element = RaviartThomasTriangleRT0(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    expected = jnp.mean(mesh.vertex_coordinates[mesh.facets_to_vertices], axis=1)
    assert jnp.allclose(space.dof_coordinates, expected, atol=1e-14)


@pytest.mark.unit
def test_rt0_local_dof_signs_are_plus_or_minus_one(
    four_triangle_center_vertex_mesh,
) -> None:
    mesh = four_triangle_center_vertex_mesh
    element = RaviartThomasTriangleRT0(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    assert space.local_dof_signs.shape == (mesh.cells_to_vertices.shape[0], 3)
    assert jnp.all((space.local_dof_signs == 1.0) | (space.local_dof_signs == -1.0))


@pytest.mark.unit
def test_rt0_local_dof_signs_for_a_single_canonically_ordered_cell() -> None:
    """A single cell (0, 1, 2) -- the mesh's only cell, so its own facet
    vertex pairs *are* the global canonical ordering by construction.
    Signs still aren't trivially all +1: ``ReferenceTriangle``'s own local
    facet convention lists facet 1 as ``(local vertex 2, local vertex 0)``
    -- descending -- regardless of global vertex numbering, so local facet
    1 gets -1 even though facets 0 (``(1, 2)``) and 2 (``(0, 1)``) get +1.
    """
    mesh = create_triangle_mesh_from_arrays(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((0, 1, 2),)
    )
    element = RaviartThomasTriangleRT0(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    assert jnp.array_equal(space.local_dof_signs, jnp.array([[1.0, -1.0, 1.0]]))


@pytest.mark.unit
@pytest.mark.integration
def test_rt0_shared_facet_gets_opposite_signs_under_reversed_orientation(
    reversed_orientation_two_triangle_mesh,
) -> None:
    """DOF-03-style, for RT0: the two cells sharing an edge traversed in
    opposite local directions (fixture docstring) must disagree in sign on
    that shared facet -- otherwise the Piola-mapped physical flux would be
    discontinuous where it must be continuous (verified end to end in
    ``tests/space/test_mixed_cell_basis.py``).
    """
    mesh = reversed_orientation_two_triangle_mesh
    element = RaviartThomasTriangleRT0(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)

    interior_mask = jnp.all(mesh.facets_to_cells != -1, axis=1)
    (shared_facet,) = jnp.nonzero(interior_mask)[0]
    cell_0, cell_1 = mesh.facets_to_cells[shared_facet].tolist()
    local_facet_0 = int(
        jnp.nonzero(mesh.cells_to_facets[cell_0] == shared_facet)[0][0]
    )
    local_facet_1 = int(
        jnp.nonzero(mesh.cells_to_facets[cell_1] == shared_facet)[0][0]
    )
    sign_0 = space.local_dof_signs[cell_0, local_facet_0]
    sign_1 = space.local_dof_signs[cell_1, local_facet_1]
    assert sign_0 == -sign_1
